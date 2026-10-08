#!/usr/local/bin/python3
# -*- coding: utf-8 -*-


import argparse
import os
DirPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.join(DirPath, "src"))
import json
from loguru import logger

from datasets import load_dataset
from tqdm import tqdm
from pipline import Pipline
from define_benchmark_prompt import define_longbench


def ir_parse_args(args=None):
    parser = argparse.ArgumentParser()
    # InfiniComp Args
    parser.add_argument("--model_name", type=str, default="Qwen2.5-0.5B")
    parser.add_argument("--model_path", type=str, default=r"D:\Code\models\Qwen2.5-1.5B-Instruct")
    parser.add_argument("--cache_granularity", type=str, default="sentence-level")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--topk_phrase", type=int, default=None,
                        help="number of top attention-score indices selected "
                             "per retrieval step (phrase granularity via "
                             "--phrase_token_num); default: fall back to "
                             "--compress_budget")
    parser.add_argument("--compress_budget", type=int, default=512,
                        help="upper bound of the compressed context length "
                             "in tokens (the final length is dynamic and "
                             "never exceeds it)")
    parser.add_argument("--compress_ratio", type=str, default=None)
    parser.add_argument("--phrase_token_num", type=int, default=15)
    parser.add_argument("--iter_number", default=1)
    parser.add_argument("--need_generate", type=bool, default=True)
    parser.add_argument("--exact_eager", type=bool, default=True,
                        help="True (default): all-eager capture with in-hook "
                             "head-sum, bitwise reproducible; False: fast "
                             "SDPA path (kernel drift, for speed/memory)")
    parser.add_argument("--disable_thinking", type=bool, default=True,
                        help="True (default): disable <think> output of "
                             "hybrid-thinking models (e.g. Qwen3) via "
                             "enable_thinking=False plus a decode-time ban "
                             "of think marker tokens, so direct answers fit "
                             "the LongBench dataset2maxlen budgets; no "
                             "effect on models without thinking support")
    parser.add_argument("--task_name", type=str, default="all")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = ir_parse_args()
    model_name = args.model_name
    task_name = args.task_name
    config = {"llm_name_or_path": args.model_path}
    config["static_recall"] = True
    config["chunk_size"] = int(args.chunk_size)
    if args.compress_ratio is None: 
        config["compress_ratio"] = None
        compress_output = args.compress_budget
    else:
        config["compress_ratio"] = float(args.compress_ratio)
        compress_output = args.compress_ratio
    config["compress_budget"] = int(args.compress_budget)
    config["topk_phrase"] = int(args.topk_phrase) if args.topk_phrase is not None else None

    config["phrase_token_num"] = int(args.phrase_token_num)
    config["iter_number"] = int(args.iter_number)
    config["need_generate"] = args.need_generate
    config["exact_eager"] = args.exact_eager
    config["disable_thinking"] = args.disable_thinking
    
    save_path = os.path.join(DirPath, "experiences", "process", args.model_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path) 

    config["record_retrieval_path"] = save_path
    config["record_result_path"] = save_path
    
    LogFile = "metric_LongBench_%s.log" % (args.model_name) 
    LogPath = os.path.join(DirPath, "logs", LogFile)
    logger.add(LogPath)

    ir = Pipline(config)
    ir.retri.build_prompt = define_longbench(ir.retri.build_prompt)
    logger.info("InfiniComp Config: " + str(ir.config))

    if task_name == "all":
        tasks_name = [
            "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", 
            "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", 
            "samsum",  "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
        ]
    elif task_name == "coding":
        tasks_name = ["lcc", "repobench-p"]
    else:
        tasks_name = [task_name]

    for task_name in tasks_name:
        logger.info("Metric LongBench Task:%s" % task_name)
        LongBenchDirPath = os.path.join(DirPath, "metrics", "LongBench")
        data_path = os.path.join(LongBenchDirPath, "data", "%s.jsonl" % task_name)
        datas = load_dataset("json", data_files=data_path)

        # test_datas = datas["train"]
        test_datas = datas["train"].shuffle(223)

        longbench_config_path = os.path.join(LongBenchDirPath, "config")
        dataset2prompt = json.load(open(os.path.join(longbench_config_path, 
                                                     "dataset2prompt.json"), "r", encoding="utf-8"))
        prompt_format = dataset2prompt[task_name]
        
        dataset2len = json.load(open(os.path.join(longbench_config_path, 
                                                  "dataset2maxlen.json"), "r", encoding="utf-8"))
        maxlen = dataset2len[task_name]
        out_path = os.path.join(LongBenchDirPath, "pred")

        # model_name + cache_granularity + phrase_token_num + topk + cache_budget/compress_ratio + chunk_size
        topk_output = config["topk_phrase"] if config["topk_phrase"] is not None else config["compress_budget"]
        out_dir = "%s_%s_%d_topk%d_%s_%d" % (
            model_name, config["cache_granularity"],
            config["phrase_token_num"], topk_output, compress_output, config["chunk_size"]
        )

        out_dir_path = os.path.join(out_path, out_dir)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)
        out_path = os.path.join(out_dir_path, "%s.jsonl" % task_name)
        
        with open(LogPath, "a") as f:
            for index, data in enumerate(tqdm(test_datas, desc="%s:" % task_name, file=f)):
                if index >=3 : break

                data["task_name"] = task_name    
                if task_name in ['lcc', 'repobench-p']:
                    is_coding_task = True
                else:
                    is_coding_task = False
                
                if task_name == "repobench-p":
                    data["context"] = data["context"] + data["input"]
                    data["input"] = ""
                
                pred, info = ir.generate(data_dict=data, prompt=prompt_format,
                    gen_name="%s_%s" % (task_name, out_dir), task_name=task_name,
                    sample_id=index+1, topk_phrase=config["topk_phrase"],
                    compress_budget=config["compress_budget"], compress_ratio=config["compress_ratio"],
                    phrase_token_num=config["phrase_token_num"], temperature=0.1,
                    iter_number=config["iter_number"], max_new_tokens=maxlen,
                    is_coding_task=is_coding_task
                )
                
                result = {"llm_name_or_path": info["llm_name"], 
                          "llm_dtype": info["llm_type"], 
                          "answers": data["answers"], 
                          "all_classes": data["all_classes"], 
                          "answer_in_context" : info["answer_in_context"], 
                          "answer_in_retrieval": info["answer_in_retrieval"], 
                          "origin_length": info["origin_context_length"],
                          "input_length": info["final_input_length"],
                          "final_input": info["final_input_text"], 
                          "all_cost": info["all_pipline_cost"]} 
                          # "run_time": info["run_current_time"]}
                
                if pred is not None:
                    #pred_label = "%s_tem%.1f_pred" % (model_name, float(args.temperature))
                    pred_label = "pred"
                    result[pred_label] = pred
                    result[pred_label+"_generate_cost"] = info["final_generate_cost"]
                
                if "think" in info.keys():
                    result["think"] = info["think"]

                with open(out_path, "a", encoding="utf-8") as f:
                    json.dump( result, f, ensure_ascii=False)
                    f.write('\n')

            if config["need_generate"]:
                sys.path.append(LongBenchDirPath)
                python_file = os.path.join(LongBenchDirPath, "eval.py")
                os.system("python %s --model %s" % (python_file, out_dir))
