#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   metric_LongBenchV2.py
@Time    :   2025/01/14 15:28:57
@Author  :   Anonymous (double-blind review)
@Version :   0.1
@Contact:    Anonymous (double-blind review)
@Desc    :   Metric and Evalutate InfiniComp Pipline in LongBenchV2 Benchmark, 
             fix from https://github.com/THUDM/LongBench/blob/main/pred.py#L60
'''

import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DirPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.join(DirPath, "src"))
import json
import re

from datasets import load_dataset
from tqdm import tqdm
from pipline import Pipline
from define_benchmark_prompt import define_longbenchv2


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # InfiniComp args
    parser.add_argument("--model_name", type=str, default="qwen2.5-7B")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--phrase_token_num", type=int, default=15)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--generate_method", type=str, default="token_id")
    parser.add_argument("--compute_score_method", type=int, default=3)
    
    # LongeBench V2 args
    parser.add_argument("--benchmark_name", type=str, default="LongBenchV2")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--start_index", type=int, default=0)
    return parser.parse_args(args)


def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name

    if len(args.model_path) == 0: 
        if "Llama" in args.model_name:
            args.model_path = "/data/coding/models/Llama3-8B-Instruct"
        elif "Mistral" in args.model_name:
            args.model_path = "/data/coding/models/Mistral-7B-Instruct-v0.2"
        elif "Qwen2.5" in args.model_name:
            args.model_path = "/data/coding/models/Qwen2.5-7B-Instruct/"
        else:
            print("Error, no model path.......")


    config = {"llm_name_or_path": args.model_path}
    config["topk"] = int(args.topk)
    config["chunk_size"] = int(args.chunk_size)
    save_path = os.path.join(DirPath, "experiences", "process")
    config["record_retrieval_path"] = save_path
    config["record_result_path"] = save_path
    # config["record_attention_path"] = path
    config["use_question_weight"] = True
    config["compute_score_method"] = int(args.compute_score_method)
    config["phrase_token_num"] = int(args.phrase_token_num)
    config["generate_method"] = args.generate_method
    config["data_format_map"] = {"context": "context", "question":"question"}
    config["retri_split_pattern"] = "(\n\n)|([，。！？.?!,])"
    print("ir config: ", config)
    
    ir = Pipline(config)
    ir.retri.build_prompt = define_longbenchv2(ir.retri.build_prompt)
    
    LongBenchV2DirPath = os.path.join(DirPath, "metrics", "LongBenchV2")
    data_path = os.path.join(LongBenchV2DirPath, "data", "data.json")
    datas = load_dataset("json", data_files=data_path)
    test_datas = datas["train"]
    
    out_path = os.path.join(LongBenchV2DirPath, "results")
    out_dir = "%s_com%d_%d_%d_%d" % (model_name, config["compute_score_method"], 
                                     config["phrase_token_num"], config["topk"], config["chunk_size"])
    out_dir_path = os.path.join(out_path, out_dir)
    os.makedirs(out_dir_path, exist_ok=True)
    
    prompt_path = os.path.join(LongBenchV2DirPath, "prompts")
    template_rag = open(os.path.join(prompt_path, "0shot_rag.txt"), encoding="utf-8").read()
    template_no_context = open(os.path.join(prompt_path, "0shot_no_context.txt"), encoding="utf-8").read()
    template_0shot = open(os.path.join(prompt_path, "0shot.txt"), encoding="utf-8").read()
    template_0shot_cot = open(os.path.join(prompt_path, "0shot_cot.txt"), encoding="utf-8").read()
    template_0shot_cot_ans = open(os.path.join(prompt_path,"0shot_cot_ans.txt"), encoding='utf-8').read()
    
    if args.rag > 0:
        out_file = os.path.join(out_dir_path, "longbenchv2" + f"_rag_{str(args.rag)}.jsonl")
        prompt_format = template_rag
        task_name = "rag"
    elif args.no_context:
        out_file = os.path.join(out_dir_path, "longbenchv2" + "_no_context.jsonl")
        prompt_format = template_no_context
        task_name = "no_context"
    elif args.cot:
        out_file = os.path.join(out_dir_path, "longebenchv2" + "_cot.jsonl")
        prompt_format = template_0shot_cot
        task_name = "cot"
    else:
        out_file = os.path.join(out_dir_path, "longbenchv2" + ".jsonl")
        prompt_format = template_0shot
        task_name = "0shot"
    
    results = []
    for index, data in enumerate(tqdm(test_datas)):
        if index < 185: continue
        if data[ir.config["data_format_map"]["question"]].strip()[-1] not in ir.retri_split_pattern: 
            data[ir.config["data_format_map"]["question"]] += "?"
        data["task_name"] = task_name
        
        pred, info = ir.generate2(data_dict=data, prompt=prompt_format,
                                  gen_name="%s_%s_%s_com%d_pharse%d" % (args.benchmark_name,
                                  task_name, model_name, config["compute_score_method"], config["phrase_token_num"]),
                                  sample_id=index+1, topk=config["topk"],
                                  temperature=0.1, max_new_tokens=1024 if args.cot else 128)
        
        if args.cot:
            response = pred.strip()
            data["response_cot"] = response
            data["task_name"] = "cot_ans"
            prompt_format = template_0shot_cot_ans
            pred, info = ir.generate2(data_dict=data, prompt=prompt_format,
                                     gen_name="%s_%s_%s_com%d_pharse%d" % (args.benchmark_name,
                                     task_name, model_name, config["compute_score_method"], config["phrase_token_num"]),
                                     sample_id=index+1, topk=config["topk"],
                                     temperature=0.1, max_new_tokens=128)
            if pred == '':
                continue

        result = {"response": pred,
                  "pred": extract_answer(pred),
                  "answer": data["answer"], 
                # "retrieval_answer": answer_in_pred, 
                # "input_text": info["input_text"],
                  "origin_length": info["origin_token_length"],
                  "input_length": info["input_token_length"],
                  "length": data["length"],
                  "difficulty": data["difficulty"]}
     
        result["judge"] = result["pred"] == result["answer"]
        results.append(result)

        with open(out_file, "a", encoding="utf-8") as f:
            json.dump( result, f, ensure_ascii=False)
            f.write('\n')
        
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    for pred in results:
        acc = int(pred['judge'])
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    compute_result = {"model_name": out_dir,
                      "all_acc": round(100*(easy_acc+hard_acc)/len(results), 1),
                      "easy_acc": round(100*easy_acc/easy, 1),
                      "hard_acc": round(100*hard_acc/hard, 1),
                      "short_acc": round(100*short_acc/short, 1),
                      "medium_acc": round(100*medium_acc/medium, 1),
                      "long_acc": round(100*long_acc/long, 1)}

    with open(os.path.join(out_path, "compute_result.jsonl"), "a", encoding="utf-8") as f:
        json.dump(compute_result, f, ensure_ascii=False)
        f.write('\n')
    
