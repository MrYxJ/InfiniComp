#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InfiniComp pipeline orchestrator.

:class:`Pipline` wires together :class:`Chunks` (long-context chunking) and
:class:`Retrieval` (attention-based retrieval): it iterates over the chunks,
keeps the retrieved context from every step, and finally generates an answer
from the condensed context.

Typical usage::

    from pipline import Pipline

    pipe = Pipline(config={"llm_name_or_path": "..."})
    response, result_info = pipe.generate(data_dict, prompt, task_name="...")
"""

import copy
import json
import os
import re
import time
from typing import Dict, Optional

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from chunks import Chunks
from retrieval import Retrieval

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tasks whose few-shot prompts are fed to the model as plain text (without a
# chat template) and whose generation should stop at a newline.
PLAIN_TEXT_TASKS = ("trec", "triviaqa","samsum", "lcc", "repobench-p", "code_debug")


class Pipline:
    """Chunk a long context, retrieve the relevant parts with attention
    scores, and generate the final answer."""

    def __init__(self, config: Optional[Dict] = None, model=None,
                 tokenizer=None, excute_log_path: str = None) -> None:
        self.config = config if config is not None else {}
        self.logger = logger

        if excute_log_path is None:
            excute_time_str = str(time.ctime(time.time())).replace(":", "_")
            excute_dir_path = os.path.join(BASE_DIR, "excute_logs")
            os.makedirs(excute_dir_path, exist_ok=True)
            excute_log_path = os.path.join(excute_dir_path, "%s.log" % excute_time_str)
        logger.add(excute_log_path)
        self.logger_path = excute_log_path

        self.default_config = {
            "llm_name_or_path": r"D:\Code\models\Qwen2.5-0.5B-Instruct",
            "data_format_map": {"context": "context", "question": "input"},
            "retri_split_pattern": "(\n\n)|([。！？.!?])",
            # Number of top attention-score indices to select per retrieval
            # step. The selection granularity is governed by
            # ``phrase_token_num``: 1 = single-token scores, >1 = phrase
            # scores (sums of ``phrase_token_num`` consecutive tokens).
            # None (default) falls back to ``compress_budget`` (legacy) or
            # 300 when neither is set.
            "topk_phrase": None,
            # Upper bound of the compressed context length in tokens: the
            # sentences containing the top-k indices are admitted until this
            # many tokens are gathered, so the final length is dynamic and
            # never exceeds the budget. None (default) = unbounded.
            "compress_budget": None,
            "compress_ratio": None,
            "chunk_size": 1024,
            "record_attention_path": None,   # heatmap recording costs much time
            "record_retrieval_path": None,
            "record_result_path": None,
            "phrase_token_num": 8,
            "question_add_prompt": True,
            "question_use_last": False,
            "iter_number": 1,                # total number of retrieval iterations
            "cache_granularity": "sentence-level",  # "sentence-level" or "token-level"
            "static_recall": False,
            "set_sys_prompt": False,
            "need_generate": True,
            # Hybrid-thinking models (e.g. Qwen3): disable <think> output so
            # the generation is a direct answer that fits the small LongBench
            # ``dataset2maxlen`` budgets (32-128 tokens) -- otherwise the
            # budget is exhausted mid-thought and no answer is produced.
            # Implemented as enable_thinking=False in the chat template (the
            # official Qwen3 template then inserts an empty think block,
            # i.e. NoThinking mode) plus a decode-time ban of the think
            # marker tokens (cf. CompKV, arXiv 2609.26300). Models without
            # thinking support (e.g. Qwen2.5) are unaffected.
            "disable_thinking": True,
            # Attention-capture mode (see Retrieval._forward_with_layer_attention):
            #   True (default) -- exact path: all-eager forward, hook head-sums
            #       the scoring layer immediately (bitwise identical to the
            #       original code, verified 160/160 LongBench samples);
            #       use for paper reproducibility.
            #   False -- fast path: fused SDPA on non-target layers. Faster and
            #       avoids the eager N^2 transient entirely (needed for 7B /
            #       very long sequences), but kernel-level drift changes
            #       per-sample results (aggregate scores unaffected).
            "exact_eager": True,
        }

        # Fill in config params that the caller did not provide.
        for key, value in self.default_config.items():
            self.config[key] = self.config.get(key, value)

        self.logger.info("InfiniComp Excute Default Config: %s" % str(self.config))

        self.data_format_map = self.config.get("data_format_map")
        # The pattern of text segmentation (re.split) during retrieval,
        # which determines the granularity of each retrieval unit.
        self.retri_split_pattern = self.config.get("retri_split_pattern")
        
        self.compress_ratio = self.config.get("compress_ratio")
        self.compress_budget = self.config.get("compress_budget")
        self.topk_phrase = self.config.get("topk_phrase")
        self.chunk_size = self.config.get("chunk_size")
        self.iter_number = self.config.get("iter_number")
        self.phrase_token_num = self.config.get("phrase_token_num")

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("llm_name_or_path"))
        else:
            self.tokenizer = tokenizer

        if model is None:
            model_path = self.config.get("llm_name_or_path") or ""
            self.config["torch_dtype"] = (
                torch.bfloat16 if "Qwen" in model_path else torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.config["torch_dtype"],
                device_map="auto",
                # attn_implementation="flash_attention_2",
                attn_implementation="eager",
            )
        else:
            self.model = model
            self.config["torch_dtype"] = torch.bfloat16

        self.model.eval()
        self.chunks = Chunks(self.config, self.tokenizer, self.logger)
        self.retri = Retrieval(self.config, self.model, self.tokenizer, self.logger)

        self.record_attention = self.config.get("record_attention_path")
        self.record_retrieval = self.config.get("record_retrieval_path")
        self.record_result = self.config.get("record_result_path")

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def record_info(self, record_path: str, record_file: str, print_info: dict) -> None:
        """Append one JSON line to ``record_path / record_file``."""
        os.makedirs(record_path, exist_ok=True)
        file_path = os.path.join(record_path, record_file)
        with open(file_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(print_info, ensure_ascii=False) + "\n")

    def generate_init(self) -> None:
        """Reset the per-request state of the pipeline."""
        self.retri.PREFIX_PROMPT_TOKEN_NUM = 0
        self.retri.LAST_PROMPT_TEXT = None
        self.retri.START_TIME = time.time()
        self.retri.thoughts.malplot_all_init()
        self.retri.QUESTION_WEIGHT = None

    # ------------------------------------------------------------------ #
    # Main entry                                                         #
    # ------------------------------------------------------------------ #

    def generate(self, data_dict: Dict, prompt: str, gen_name: str = "",
                 task_name: str = "", sample_id: int = 0, chunk_size: int = None,
                 topk_phrase: int = None, compress_budget: int = None, compress_ratio: float = None,
                 phrase_token_num:int = None, layer: int = -1, temperature: float = 0,
                 max_new_tokens: int = 258, iter_number: int = None,
                 generate_token: bool = False, is_coding_task: bool = False):
        """Retrieve the relevant context and generate the answer.

        The long context is chunked, and each chunk is scored against the
        question by attention; the retrieved context grows iteratively and
        the final answer is generated from it.

        Args:
            data_dict: Sample containing the context/question (under the keys
                of ``config["data_format_map"]``). It is modified in place.
            prompt: Prompt template (formatted with ``data_dict``).
            gen_name: Name of the generator run (for record paths).
            task_name: Benchmark task name (drives the text splitting rules).
            sample_id: Sample index (for record paths).
            chunk_size: Override of the configured chunk size.
            topk_phrase: Override of the configured top-k: number of top
                attention-score indices selected per retrieval step
                (phrase granularity via ``phrase_token_num``).
            compress_budget: Override of the configured compressed-length
                upper bound (tokens); ``None`` means unbounded (dynamic
                length driven purely by ``topk``).
            compress_ratio: Alternative way to set the upper bound, as a
                fraction of the original context length.
            layer: Which layer's attention to use for scoring.
            temperature: Sampling temperature (greedy when 0).
            max_new_tokens: Maximum number of generated tokens.
            iter_number: Override of the configured iteration number.
            generate_token: Also score the freshly generated token.

        Returns:
            Tuple ``(response, result_info)``. ``response`` is ``None`` when
            ``config["need_generate"]`` is ``False``.
        """
        origin_text = copy.deepcopy(data_dict[self.data_format_map["context"]])
        origin_token_length = len(
            self.tokenizer.encode(origin_text, add_special_tokens=False))

        iter_chunk_size = chunk_size if chunk_size else self.chunk_size

        # topk: number of top attention-score indices selected per step.
        # Falls back to compress_budget for legacy callers that only set the
        # budget (the budget doubled as the top-k before this refactor).
        if topk_phrase is not None:
            iter_topk = topk_phrase
        elif self.topk_phrase is not None:
            iter_topk = self.topk_phrase
        elif compress_budget is not None:
            iter_topk = compress_budget
        elif self.compress_budget is not None:
            iter_topk = self.compress_budget
        else:
            iter_topk = 300

        # compress_budget: upper bound of the compressed context length in
        # tokens. None = unbounded (dynamic length driven purely by topk).
        if compress_ratio is not None:
            iter_compress_budget = int(origin_token_length * compress_ratio)
        elif compress_budget is not None:
            iter_compress_budget = compress_budget
        elif self.compress_ratio is not None:
            iter_compress_budget = int(origin_token_length * self.compress_ratio)
        else:
            iter_compress_budget = self.compress_budget

        init_topk, init_compress_budget = iter_topk, iter_compress_budget

        if phrase_token_num is not None:
            self.phrase_token_num = phrase_token_num
        else:
            self.phrase_token_num = self.config.get("phrase_token_num")

        if iter_number is not None:
            self.iter_number = iter_number
        else:
            self.iter_number = self.config.get("iter_number")

        if is_coding_task:
            retri_split_pattern = "(\n)"
        else:
            retri_split_pattern = self.config.get("retri_split_pattern")
            
        start_pipline_t = time.time()
        past_retri_context = ""
        with open(self.logger_path, "a", encoding="utf-8") as f:
            for iter_idx in range(self.iter_number):
                question_text = data_dict[self.data_format_map["question"]]
                question_token_length = len(self.tokenizer.encode(question_text))
                if (iter_compress_budget is not None
                        and question_token_length > iter_compress_budget):
                    # The question alone overflows the compressed-length
                    # budget: nothing can be retrieved for this iteration.
                    past_retri_context = ""
                else:
                    contexts = (data_dict[self.data_format_map["context"]]
                                if iter_idx == 0 else past_retri_context)

                    self.generate_init()
                    past_retri_context = ""
                    docs = self.chunks.chunk_docs(
                        contexts, chunk_size=iter_chunk_size,
                        question=None, task_name=task_name, 
                        split_sen_pattern=retri_split_pattern)
                    doc_num = len(docs)
                    tqdm_desc = "Iter %d: Length[%d] Chunk[%d] Topk[%d] Budget[%s]" % (
                        iter_idx + 1, len(self.tokenizer.encode(contexts)),
                        iter_chunk_size, iter_topk, str(iter_compress_budget))

                    for index, context in enumerate(
                            tqdm(docs, desc=tqdm_desc, colour="red",
                                  leave=False, file=f)):
                        doc_id = "%s_%d_%d" % (gen_name, sample_id, index)

                        # Merge the previously retrieved context.
                        data_dict[self.data_format_map["context"]] = past_retri_context + context

                        (output, select_indexs, past_retri_context,
                         process_info) = self.retri.retrieval_attention(
                            data_dict=data_dict, prompt=prompt, layer=layer,
                            doc_id=doc_id, phrase_token_num=self.phrase_token_num,
                            retri_split_pattern=retri_split_pattern,
                            topk=iter_topk, compress_budget=iter_compress_budget,
                            iter_number=iter_idx, task_name=task_name, generate_token=generate_token)

                        if self.record_retrieval:
                            save_path = os.path.join(
                                self.record_retrieval, "%s_%s" % (gen_name, sample_id))
                            self.record_info(save_path, "process.json", process_info)

                        if index < doc_num - 1:
                            del output
                            torch.cuda.empty_cache()

                # Shrink the retrieval strength for the next iteration.
                iter_topk = int(iter_topk * 0.8)
                if iter_compress_budget is not None:
                    iter_compress_budget = int(iter_compress_budget * 0.8)

            # NOTE: kept from the original implementation -- the final context
            # is stored under the literal "context" key.
            data_dict["context"] = past_retri_context
            input_text = self.retri.build_prompt(data_dict=data_dict, prompt=prompt)

            final_context_length = len(
                self.tokenizer.encode(past_retri_context, add_special_tokens=False))

        result_info = {
            "llm_name": self.config.get("llm_name_or_path"),
            "llm_type": str(self.config["torch_dtype"]),
            # "origin_text": origin_text,
            "origin_context_length": origin_token_length,
            "final_retrieval_context": past_retri_context,
            "final_context_length": final_context_length,
            "run_current_time": str(time.ctime(time.time())),
            "model_name_or_path": self.model.name_or_path,
            "chunk_size": self.chunk_size,
            "retrieval_topk": init_topk,
            "compress_budget": init_compress_budget,
            "phrase_token_num": str(self.phrase_token_num),
            "max_new_tokens": max_new_tokens,
            "iter_number": self.iter_number,
            "retri_split_pattern": retri_split_pattern
        }

        if "answers" in data_dict.keys():
            result_info["answers"] = data_dict["answers"]

        response = None
        final_end_t = time.time()
        if self.config["need_generate"]:
            if task_name in PLAIN_TEXT_TASKS:
                input_build_text = input_text.strip()
                if task_name == "triviaqa":
                    input_build_text += "\n"
            else:
                input_build_text = self.retri.build_message(input_text)
            
            model_inputs = self.tokenizer([input_build_text], return_tensors="pt").to(self.model.device)
            final_input_length = model_inputs["input_ids"].shape[1]

            result_info["final_input_text"] = input_build_text
            result_info["final_input_length"] = final_input_length
            final_generate_t = time.time()
            generate_kwargs = dict(
                num_beams=1, do_sample=False,
                temperature=temperature, max_new_tokens=max_new_tokens,
            )
            if task_name in PLAIN_TEXT_TASKS:
                # Stop at the first newline for these few-shot tasks.
                generate_kwargs["eos_token_id"] = [
                    self.tokenizer.eos_token_id,
                    self.tokenizer.encode("\n", add_special_tokens=False)[-1],
                    self.tokenizer.encode("\n\n", add_special_tokens=False)[-1],
                    self.tokenizer.encode("\n\n\n", add_special_tokens=False)[-1],
                ]
            if self.config.get("disable_thinking"):
                # Double insurance (cf. CompKV, arXiv 2609.26300): even with
                # enable_thinking=False some checkpoints still emit thinking
                # markers, so ban the <think> / </think> special tokens at
                # every decoding step. Self-detected: only markers that map
                # to a single special-token id are banned, hence models
                # without thinking support (e.g. Qwen2.5) are unaffected.
                bad_words = []
                for marker in ("<think>", "</think>"):
                    marker_ids = self.tokenizer.encode(
                        marker, add_special_tokens=False)
                    if len(marker_ids) == 1:
                        bad_words.append(marker_ids)
                if bad_words:
                    generate_kwargs["bad_words_ids"] = bad_words

            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, **generate_kwargs)

            final_end_t = time.time()
            result_info["final_generate_cost"] = final_end_t - final_generate_t

            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Defensive strip of any residual thinking block (e.g. from a
            # run with thinking enabled whose budget ended mid-thought).
            # A no-op for outputs without a "<think>" marker.
            if "<think>" in response:
                response = re.sub(
                    r"<think>.*?(</think>|$)", "", response,
                    flags=re.DOTALL).strip()
            result_info["output"] = response

        result_info["all_pipline_cost"] = final_end_t - start_pipline_t

        if self.config["static_recall"]:
            result_info["answer_in_context"] = False
            result_info["answer_in_retrieval"] = False
            for answer in data_dict["answers"]:
                if answer in past_retri_context:
                    result_info["answer_in_retrieval"] = True
                if answer in origin_text:
                    result_info["answer_in_context"] = True

        if self.config["question_use_last"] is True:
            result_info["question_use"] = "last_token"
        elif self.config["question_add_prompt"] is True:
            result_info["question_use"] = "question_add_prompt"
        else:
            result_info["question_use"] = "question_origin_text"

        if self.record_attention:
            save_name = "%s_%s/" % (gen_name, sample_id)
            self.retri.thoughts.malplot_all_attention(save_name=save_name, page_max_num=1)

        if self.record_result:
            save_path = os.path.join(self.record_result, "%s_%s" % (gen_name, sample_id))
            self.record_info(save_path, "result.json", result_info)

        return response, result_info


# Backward-compatible alias.
Pipeline = Pipline
