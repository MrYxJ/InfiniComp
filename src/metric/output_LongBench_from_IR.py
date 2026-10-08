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

import os
DirPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(os.path.join(DirPath, "src"))
import json
import time 
import torch

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM



def read_json(path, encoding='utf-8', line_read=True):
    if line_read:
        obj = []
        with open(path, 'r', encoding=encoding) as fp:
            for row in fp.readlines():
                item = json.loads(row)
                obj.append(item)
        return obj
    else:
        return json.load(open(path, encoding=encoding))


def write_json(path, datas, encoding='utf-8', mode='a'):
    if str(type(datas)) == "<class 'list'>" :
        with open(path, mode, encoding=encoding) as fp:
            for data in datas:
                if str(type(data)) == "<class 'pyspark.sql.types.Row'>":
                    data = data.asDict()
                line = json.dumps(data, ensure_ascii=False)
                fp.write(line+'\n')
    else:
        with open(path, mode, encoding=encoding) as fp:
            json.dump(datas, fp, ensure_ascii=False)
    # print("Write [%s] successfully!" % path)


if __name__ == '__main__':


    tasks_name = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news"]
    models = ["Qwen2-7B-Instruct", "Mistral-7B-Instruct-v0.3", "Llama3-8B-Instruct"]
    models = ["Qwen2-7B-Instruct"]
    
    for tem in ["0.1", "1.0"]:
        for model_name in models:
            print("Model:", model_name)
            if "Llama" in model_name:
                model_path = "/data/coding/models/Llama3-8B-Instruct"
            elif "Mistral" in model_name:
                model_path = "/data/coding/models/Mistral-7B-Instruct-v0.3"
            elif "Qwen" in model_name:
                # model_path = "/data/coding/models/Qwen2-7B-Instruct/"
                model_path = "D:\Code\models\Qwen2.5-0.5B-Instruct"

            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                         attn_implementation="eager",
                                                         device_map="auto", 
                                                         torch_dtype="auto")
                                                        #  attn_implementation="flash_attention_2")
                                                        # torch_dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            LongBenchDirPath = os.path.join(DirPath, "metrics", "LongBench")
            longbench_config_path = os.path.join(LongBenchDirPath, "config")
            pred_path = os.path.join(LongBenchDirPath, "pred")
            out_dir = "%s_phrase%d_topk%d_chunk%d_temp%s_bf16" % (model_name, 15, 500, 1024, tem)
            out_dir_path = os.path.join(pred_path, out_dir)
            if not os.path.exists(out_dir_path):
                os.makedirs(out_dir_path)

            for task_name in tasks_name:
                dataset2len = json.load(open(os.path.join(longbench_config_path, 
                                                          "dataset2maxlen.json"), "r", encoding="utf-8"))
                maxlen = dataset2len[task_name]

     
                read_dir = "%s_phrase%d_topk%d_chunk%d_InfiniComp" % (model_name, 15, 500, 1024)
                read_path = os.path.join(pred_path, read_dir, "%s.jsonl"%task_name)
                test_datas = read_json(read_path)
   
                out_path = os.path.join(out_dir_path, "%s.jsonl" % task_name)

                for index, data in enumerate(tqdm(test_datas)):
                    input_text = data["final_input_text"]
                    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

                    t1 = time.time()
                    with torch.no_grad():
                        generated_ids = model.generate(**model_inputs, temperature=float(tem), max_new_tokens=maxlen)
                    cost_time = time.time() - t1

                    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)]
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    data["pred"] = response
                    data["llm_pred_dtype"] = str(model.dtype)
                    with open(out_path, "a", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                        f.write('\n')

            sys.path.append(LongBenchDirPath)
            python_file = os.path.join(LongBenchDirPath, "eval.py")
            os.system("python %s --model %s" % (python_file, out_dir))
