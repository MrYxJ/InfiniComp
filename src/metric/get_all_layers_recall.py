#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   metric_LongBench.py
@Time    :   2025/01/06 14:28:57
@Author  :   Anonymous (double-blind review)
@Version :   0.1
@Contact:    Anonymous (double-blind review)
@Desc    :   Metric and Evalutate InfiniComp Pipline in LongBench Benchmark
'''

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
    parser.add_argument("--model_name", type=str, default="Qwen2-7B")
    parser.add_argument("--model_path", type=str, default="/data/coding/models/Qwen2-7B-Instruct")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--phrase_token_num", type=int, default=15)
    parser.add_argument("--topk_token", type=int, default=1000)   # 最终压缩后文本大致长度，可控。
    parser.add_argument("--topk_phrase", type=int, default=200) # 选择topk个phrase所在的完整句子组成最终文本，因此其长度是动态，不可控。
    parser.add_argument("--topk_method", type=str, default="token") 
    parser.add_argument("--cache_granularity", type=str, default="sentence-level")
    parser.add_argument("--iter_number", default=1)
    parser.add_argument("--question_use_last", default=False)
    parser.add_argument("--question_add_prompt", default=True)
    
    # LongBench Args 
    parser.add_argument("--benchmark_name", type=str, default="LongBench")
    return parser.parse_args(args)



if __name__ == '__main__':
    args = ir_parse_args()
    model_name = args.model_name
    task_name = "hotpotqa"
    
    if "Llama3-8B" in args.model_name:
        args.model_path = "/data/coding/models/Llama3-8B-Instruct"
    elif "MistralV0.2" in args.model_name:
        args.model_path = "/data/coding/models/Mistral-7B-Instruct-v0.2"
    elif "MistralV0.3" in args.model_name:
        args.model_path = "/data/coding/models/Mistral-7B-Instruct-v0.3"
    elif "Qwen2-7B" in args.model_name:
        args.model_path = "/data/coding/models/Qwen2-7B-Instruct/"
    elif "Qwen2-0.5B" in args.model_name:
        # args.model_path = "/data/coding/models/Qwen2-0.5B-Instruct/"
        args.model_path = "D:\Code\models\Qwen2.5-0.5B-Instruct"
    elif "Phi" in args.model_name:
        args.model_path = "/data/coding/models/Phi-3-mini-128k-Instruct"
    else:
        assert 1 > 0, "Please Check Model Name!!!"

    config = {"llm_name_or_path": args.model_path}
    config["static_recall"] = True
    config["topk_token"] = int(args.topk_token)
    config["topk_phrase"] = int(args.topk_phrase)
    config["topk_method"] = args.topk_method
    config["chunk_size"] = int(args.chunk_size)
    config["phrase_token_num"] = int(args.phrase_token_num)
    config["cache_granularity"] = args.cache_granularity
    config["iter_number"] = int(args.iter_number)

    config["question_add_prompt"] = True # 效果会更好
    config["set_sys_prompt"] = False     # 先不花时间再加system prompt
    config["need_generate"] = False       

    save_path = os.path.join(DirPath, "experiences", "process")
    # config["record_retrieval_path"] = save_path
    config["record_result_path"] = save_path
    # config["record_attention_path"] = path
    config["retri_split_pattern"] = "(\n\n)|([。！？.!?])" # 出现[...]内任一符号或者连续出现的两个"\n"即切分
    
    topk_value = config["topk_token"] if config["topk_method"] == "token" else config["topk_phrase"]
    LogFile = "metric_LongBench_%s_chunk%d_token%d_phrase%d.log" % \
            (args.model_name, config["chunk_size"], topk_value, config["phrase_token_num"])
    
    LogPath = os.path.join(DirPath, "logs", LogFile)
    logger.add(LogPath)

    ir = Pipline(config)
    ir.retri.build_prompt = define_longbench(ir.retri.build_prompt)
    logger.info("InfiniComp Config: " + str(ir.config))

    all_layers = len(ir.model.model.layers)

    logger.info("Model %s have %d layers" % (model_name, all_layers))
    LongBenchDirPath = os.path.join(DirPath, "metrics", "LongBench")
    data_path = os.path.join(LongBenchDirPath, "data", "%s.jsonl" % task_name)
    datas = load_dataset("json", data_files=data_path)

    # test_datas = datas["train"]
    test_datas = datas["train"].shuffle(2258)

    longbench_config_path = os.path.join(LongBenchDirPath, "config")
    dataset2prompt = json.load(open(os.path.join(longbench_config_path, 
                                                    "dataset2prompt.json"), "r", encoding="utf-8"))
    prompt_format = dataset2prompt[task_name]
    
    dataset2len = json.load(open(os.path.join(longbench_config_path, 
                                                "dataset2maxlen.json"), "r", encoding="utf-8"))
    # maxlen = dataset2len[task_name]
    maxlen = 512
    out_path = os.path.join(LongBenchDirPath, "pred")
    
    if ir.config["question_use_last"] is True:
        question_use = "question_last_token"
    elif ir.config["question_add_prompt"] is True:
        question_use = "question_add_prompt"
    else:
        question_use = "question_origin_text"

    for layer_index in range(all_layers):

        out_dir = "%s_%s_%d_%d_%d_%s_layer%d" % (
            model_name, config["cache_granularity"], config["phrase_token_num"], 
            topk_value, config["chunk_size"], question_use, layer_index)

        out_dir_path = os.path.join(out_path, out_dir)
        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)
        out_file_path = os.path.join(out_dir_path, "%s.jsonl" % task_name)
        
        with open(LogPath, "a") as f:
            for index, data in enumerate(tqdm(test_datas, desc="%s:" % task_name, file=f)):
                if len(data[ir.config["data_format_map"]["question"]].strip()) > 0:
                    if data[ir.config["data_format_map"]["question"]].strip()[-1] not in ir.retri_split_pattern+':：': 
                        data[ir.config["data_format_map"]["question"]] += "?"
                    
                data["task_name"] = task_name        
                

                pred, info = ir.generate2(data_dict=data, prompt=prompt_format,
                    gen_name="%s_%s" % (task_name, out_dir), task_name=task_name,
                    sample_id=index+1, topk_token=config["topk_token"], layer=layer_index,
                    temperature=float(0.0), max_new_tokens=maxlen
                )
                
                result = {"llm_name_or_path": info["llm_name"],
                        "llm_dtype": info["llm_type"],
                        "answers": data["answers"], 
                        "all_classes": data["all_classes"], 
                        "answer_in_context" : info["answer_in_context"],
                        "answer_in_retrieval": info["answer_in_retrieval"],
                        "origin_length": info["origin_token_length"],
                        "input_length": info["final_token_length"],
                        "final_input": info["final_input_text"],
                        "all_cost": info["all_pipline_cost"]}
                            # "run_time": info["run_current_time"]}
                
                if pred is not None:
                    pred_label = "%s_tem%.1f_pred" % (model_name, float(args.temperature))
                    result[pred_label] = pred
                    result[pred_label+"_generate_cost"] = info["final_generate_cost"]
                
                if "think" in info.keys():
                    result["think"] = info["think"]

                with open(out_file_path, "a", encoding="utf-8") as f:
                    json.dump( result, f, ensure_ascii=False)
                    f.write('\n')
            
            sys.path.append(LongBenchDirPath)
            python_file = os.path.join(LongBenchDirPath, "eval.py")
            os.system("python %s --model %s" % (python_file, out_dir))
