#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InfiniComp x InfiniteBench evaluation.

Compression parameters (current ``Pipline.generate`` API):
  * ``--topk``: number of top attention-score indices selected per retrieval
    step; granularity via ``--phrase_token_num`` (1 = token, >1 = phrase).
    Falls back to ``--compress_budget`` when not set (legacy behavior).
  * ``--compress_budget``: upper bound of the compressed context length in
    tokens; the final length is dynamic and never exceeds it.

Adapted to the refactored ``Pipline.generate`` API:
  * ``--topk`` (old ``topk_token``) -> ``--compress_budget``
  * ``ir.generate2(...)`` -> ``ir.generate(...)``
  * result field ``{model_name}_prediction`` -> ``prediction``
    (``compute_scores.get_preds`` only accepts "prediction"/"pred";
    the old key made the scorer raise ValueError)
  * ``info["origin_input_length"]`` -> ``info["origin_context_length"]``
  * scorer invoked with ``sys.executable`` (keeps the conda env) instead
    of ``os.system("python ...")``
  * resume support: finished samples are skipped on rerun (keyed by
    ``sample_id`` in the preds jsonl), per-sample exceptions are logged
    and retried on the next run instead of aborting the loop.

Example (code_debug, Qwen2.5-0.5B-Instruct):
  python metric_InfiniteBench.py --model_name Qwen2.5-0.5B \
      --model_path D:/Code/models/Qwen2.5-0.5B-Instruct \
      --task_name code_debug --chunk_size 128 --topk 128 --compress_budget 128
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

DirPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
InfiniteBenchDirPath = os.path.join(DirPath, "metrics", "InfiniteBench")
sys.path.append(os.path.join(DirPath, "src"))
sys.path.append(os.path.join(InfiniteBenchDirPath, "src"))

from datasets import load_dataset
from loguru import logger

from pipline import Pipline   # from DirPath/src
from define_benchmark_prompt import define_infinitebench  # from DirPath/src/metric
from eval_utils import get_answer  # from InfiniteBenchDirPath/src


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def ir_parse_args(args=None):
    parser = argparse.ArgumentParser()

    # InfiniComp Args
    parser.add_argument("--model_name", type=str, default="Qwen2.5-0.5B")
    parser.add_argument("--model_path", type=str,
                        default=r"D:\Code\models\Qwen2.5-0.5B-Instruct")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=None,
                        help="number of top attention-score indices selected "
                             "per retrieval step (phrase granularity via "
                             "--phrase_token_num); default: fall back to "
                             "--compress_budget (old name: topk_token)")
    parser.add_argument("--compress_budget", type=int, default=1000,
                        help="upper bound of the compressed context length "
                             "in tokens (dynamic length, never exceeds it)")
    parser.add_argument("--compress_ratio", type=str, default=None)
    parser.add_argument("--phrase_token_num", type=int, default=15)
    parser.add_argument("--cache_granularity", type=str, default="sentence-level")
    parser.add_argument("--iter_number", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="generation cap; PLAIN_TEXT tasks (incl. "
                             "code_debug) also stop at the first newline")
    parser.add_argument("--need_generate", type=str2bool, default=True)
    parser.add_argument("--exact_eager", type=str2bool, default=True,
                        help="True (default): bitwise-reproducible all-eager "
                             "capture; False: fast SDPA path (kernel drift)")
    parser.add_argument("--disable_thinking", type=str2bool, default=True,
                        help="True (default): no <think> output for hybrid "
                             "thinking models; no-op for Qwen2.5")

    # InfiniteBench Args
    parser.add_argument("--benchmark_name", type=str, default="InfiniteBench")
    parser.add_argument("--task_name", type=str, default="code_debug",
                        help="task name, comma separated list, or a group: "
                             "retrieval / qa / MemoRAG")
    parser.add_argument("--end_index", type=int, default=2000,
                        help="number of samples to evaluate (2000 = all)")
    parser.add_argument("--no_resume", action="store_true",
                        help="start fresh instead of skipping finished "
                             "samples found in the preds file")
    return parser.parse_args(args)


task_templates = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",  # noqa
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",  # noqa
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}\n\n",  # noqa
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",  # noqa
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion:\n\n{question}\n\nAnswer:",  # noqa
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",  # noqa
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "math_calc": "Let us calculate the intermediate values of an expression.\n\nExpression: 1 + 3 + 4\nValues: [1, 4, 8]\n\nExpression: 8 - 3 + 2 - 4\nValues: [8, 5, 7, 3]\n\nExpression: {context}\nValues:",  # noqa
    "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",  # noqa
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",  # noqa
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely",  # noqa
}

TASK_GROUPS = {
    "retrieval": ["kv_retrieval", "number_string"],
    "qa": ["longbook_qa_eng", "longbook_qa_chn"],
    "MemoRAG": ["longbook_qa_eng", "longbook_sum_eng"],
}


if __name__ == '__main__':
    args = ir_parse_args()
    model_name = args.model_name

    config = {"llm_name_or_path": args.model_path}
    config["chunk_size"] = int(args.chunk_size)
    config["compress_budget"] = int(args.compress_budget)
    config["topk_phrase"] = int(args.topk) if args.topk is not None else None
    config["compress_ratio"] = (float(args.compress_ratio)
                                if args.compress_ratio is not None else None)
    config["phrase_token_num"] = int(args.phrase_token_num)
    config["cache_granularity"] = args.cache_granularity
    config["iter_number"] = int(args.iter_number)
    config["set_sys_prompt"] = False
    config["need_generate"] = args.need_generate
    config["question_add_prompt"] = True
    config["question_use_last"] = False
    config["static_recall"] = False
    # InfiniteBench protocol of the original script: sentence splitting also
    # cuts at commas (fine-grained units for source code).
    config["retri_split_pattern"] = "(\n\n)|([。！,？.?!])"
    config["exact_eager"] = args.exact_eager
    config["disable_thinking"] = args.disable_thinking

    save_path = os.path.join(DirPath, "experiences", "process", args.model_name)
    os.makedirs(save_path, exist_ok=True)
    config["record_result_path"] = save_path

    logs_dir = os.path.join(DirPath, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger.add(os.path.join(logs_dir, "metric_InfiniteBench_%s.log" % model_name))

    ir = Pipline(config)
    ir.retri.build_prompt = define_infinitebench(ir.retri.build_prompt)
    logger.info("InfiniComp x InfiniteBench config: " + str(ir.config))

    task_name = args.task_name
    if task_name in TASK_GROUPS:
        test_tasks = TASK_GROUPS[task_name]
    else:
        test_tasks = task_name.split(",")

    for task_name in test_tasks:
        logger.info("InfiniteBench Task: %s" % task_name)
        data_path = os.path.join(InfiniteBenchDirPath, "data", "%s.jsonl" % task_name)
        datas = load_dataset("json", data_files=data_path)
        # Same sampling protocol as the original script (shuffle seed 58);
        # the seed also keeps resume order stable across reruns.
        test_datas = datas["train"].shuffle(58)
        n_total = min(len(test_datas), args.end_index)

        prompt_format = task_templates[task_name]

        out_path = os.path.join(InfiniteBenchDirPath, "results")
        topk_output = (config["topk_phrase"] if config["topk_phrase"] is not None
                       else config["compress_budget"])
        out_dir = "%s_phrase%d_topk%d_budget%d_chunk%d" % (
            model_name, config["phrase_token_num"], topk_output,
            config["compress_budget"], config["chunk_size"])
        out_dir_path = os.path.join(out_path, out_dir)
        os.makedirs(out_dir_path, exist_ok=True)
        preds_path = os.path.join(out_dir_path, "preds_%s.jsonl" % task_name)

        # Resume: collect the sample ids already written to the preds file.
        done_ids = set()
        if not args.no_resume and os.path.exists(preds_path):
            with open(preds_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done_ids.add(json.loads(line)["sample_id"])
                    except Exception:
                        continue
        logger.info("Task %s: %d target samples, %d already done -> %d to run"
                    % (task_name, n_total, len(done_ids & set(range(1, n_total + 1))),
                       n_total - len(done_ids & set(range(1, n_total + 1)))))

        failed_ids = []
        t_task = time.time()
        n_done_this_run = 0
        for index, data in enumerate(test_datas):
            if index >= args.end_index:
                break
            sample_id = index + 1
            if sample_id in done_ids:
                continue
            if "code" in task_name:
                is_coding_task = True
            else:
                is_coding_task = False

            data["task_name"] = task_name
            t0 = time.time()
            try:
                pred, info = ir.generate(
                    data_dict=data, prompt=prompt_format,
                    gen_name="%s_%s_%s" % (args.benchmark_name, task_name, out_dir),
                    task_name=task_name, sample_id=sample_id,
                    topk_phrase=config["topk_phrase"],
                    compress_budget=config["compress_budget"],
                    phrase_token_num=config["phrase_token_num"],
                    temperature=args.temperature,
                    iter_number=config["iter_number"],
                    max_new_tokens=args.max_new_tokens,
                    is_coding_task=is_coding_task
                    )
            except Exception:
                failed_ids.append(sample_id)
                logger.exception("Task %s sample %d FAILED:" % (task_name, sample_id))
                continue

            result = {
                "sample_id": sample_id,
                "prediction": pred,
                "ground_truth": get_answer(data, task_name),
                "final_input": info["final_input_text"],
                "origin_length": info["origin_context_length"],
                "input_length": info["final_input_length"],
                "time": round(time.time() - t0, 1),
            }
            with open(preds_path, "a", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')

            n_done_this_run += 1
            elapsed = time.time() - t_task
            eta = elapsed / n_done_this_run * (n_total - len(done_ids) - n_done_this_run)
            logger.info(
                "Task %s sample %d/%d done in %.1fs (origin %d -> input %d tokens) "
                "| run avg %.1fs | ETA %.1f min | pred: %r"
                % (task_name, sample_id, n_total, result["time"],
                   result["origin_length"], result["input_length"],
                   elapsed / n_done_this_run, eta / 60, str(pred)[:60]))

        logger.info("Task %s finished: %d done this run, %d failed %s"
                    % (task_name, n_done_this_run, len(failed_ids),
                       (str(failed_ids) if failed_ids else "")))

        # Score with the official InfiniteBench scorer (same interpreter so
        # the conda env / dependencies are preserved).
        python_file = os.path.join(InfiniteBenchDirPath, "src", "compute_scores.py")
        cmd = [sys.executable, python_file, "--task", task_name,
               "--model_name", out_dir, "--output_dir", InfiniteBenchDirPath
               + os.sep + "results"]
        logger.info("Scoring: %s" % " ".join(cmd))
        subprocess.run(cmd, check=False)

    logger.info("All tasks done.")
