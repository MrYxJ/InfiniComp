#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NIH 效率基准：InfiniComp vs FullKV 的推理性能实验 harness。

实现《实验设计_效率实验_TTFT_显存_压缩耗时》文档的执行清单第 1 步：
  加载模型 -> 循环长度档 -> reset_peak_memory_stats -> 前向 ->
  记录 peak mem / 耗时 / OOM 捕获（try-except 记录为 OOM 而非崩溃）。

数据集协议与 src/metric/run_needle_in_haystack.py 的 needle/question 同步
（PaulGrahamEssays haystack + magic number needle，按 depth 插入）；
两方法共用同一任务模板 NIH_TASK_PROMPT（FullKV 同样套 chat 模板），
以便效率数字与既有 NIH 精度结果同口径、两方法输入结构对称。

记录指标（对齐文档表 1/表 2/表 3）：
  - peak_mem_allocated_gb / peak_mem_reserved_gb   torch.cuda.max_memory_{allocated,reserved}
  - compress_time_s                               InfiniComp 检索(压缩)阶段耗时；FullKV=0
  - final_prefill_ttft_s                          末次 generate 的 prefill->首 token 时间
  - ttft_s = compress_time_s + final_prefill_ttft_s   （文档定义：TTFT 含压缩阶段）
  - decode_time_s                                 首 token 之后的解码耗时
  - walltime_s = 压缩 + prefill + N-token 解码（SALT 协议：默认 128-token decode）
  - score                                         NIH rouge1*10（顺带验证正确性）
  - answer_hit                                    response 是否包含答案关键词（精确命中检查）
  - status: ok / oom / error / skipped_after_oom

方法注册表 METHOD_RUNNERS 便于后续逐步接入
SpecPrefill/CF、TopoCompress、SALT、LLMLingua / LLMLingua-2
（获取方式见知识库《代码调研_SelectiveContext_LLMLingua系列》等文档）。

用法示例：
  # GPU 服务器主实验（Mistral-7B，32K -> 2M，10 次重复）
  python src/metric/efficiency_benchmark_NIH.py \
      --methods InfiniComp,full --model_name Mistral \
      --lengths 32K,64K,128K,256K,512K,1M,2M --depths 50 \
      --repeats 10 --plot

  # 本机冒烟（0.5B 小长度）
  python src/metric/efficiency_benchmark_NIH.py \
      --methods InfiniComp,full --model_name Qwen2.5-0.5B-Instruct \
      --lengths 4K,8K --repeats 1 --max_new_tokens 128

  # OOM 捕获路径自测（人为限制显存让 FullKV 必然 OOM）
  python src/metric/efficiency_benchmark_NIH.py \
      --methods full --model_name Qwen2.5-0.5B-Instruct \
      --lengths 4K,8K,16K,32K --repeats 1 --mem-fraction 0.16

输出：<out_dir>/<model_version>_<methods>_<ts>/
  records.jsonl   每次测量的原始记录（崩溃安全，逐条追加）
  meta.json       环境 / 模型 / 配置
  summary.json    分方法分长度的统计量（mean/median/std/min/max/n）
  summary.md      论文表 1/表 2/表 3 风格的 Markdown 汇总
  efficiency_curves.png（--plot 时）
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import glob
import json
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone

DirPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NIHDirPath = os.path.join(DirPath, "metrics", "Needle-in-haystack")
sys.path.append(os.path.join(DirPath, "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from rouge_score import rouge_scorer

from pipline import Pipline as InfiniComp

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

# --------------------------------------------------------------------- #
# NIH 协议常量（needle/question 与 run_needle_in_haystack.py 当前值同步） #
# --------------------------------------------------------------------- #
# magic number 版 needle（用户 2026-10 由描述性 needle 简化为魔法数字）：
# 换 needle 时同步修改 NEEDLE / RETRIEVAL_QUESTION / ANSWER_KEYWORD。
NEEDLE = "\n\nThe special magic numbers for San Francisco is: 32000."
RETRIEVAL_QUESTION = ("What is the special magic number for San Francisco "
                      "mentioned in the provided text?")
# 精确命中健全性检查的关键词（response 中是否出现答案本身）
ANSWER_KEYWORD = "32000"
FINAL_CONTEXT_LENGTH_BUFFER = 200

# 两方法共用的 NIH 任务模板（FullKV/InfiniComp 完全一致，用户要求统一）：
# - 结构 = 指令在前、context 居中、question 在最后
#   （InfiniComp 的 question→context 注意力打分依赖 question 段位于 context 之后）
# - "one short complete sentence / exact phrasing" 引导整句作答：
#   精度打分是 rouge1(needle 全句)，只答 "32000" 会只有 ~2 分看起来像失败
# - FullKV 端与 InfiniComp 一样再套 chat 模板（user 角色 + assistant 起始），
#   替换掉原 NIH full 分支裸文本里残缺的字面 "<|im_start|>" 前缀，
#   使两方法输入结构完全对称、TTFT 对比公平
# 两方法统一模板（与 run_needle_in_haystack.py 的 NIH_TASK_PROMPT 保持同步，
# few-shot 格式锚版，0.5B 实测 2K-32K 全深度 rouge1 满分）。
# 注意：{question} 必须用 \n\n 双侧隔离、且不得带 "Question: " 之类前缀——
# retrieval.pre_process_sentence 靠逐字句匹配定位 question（split_sentence
# 以 \n\n 或 .!? 为切分点），前缀粘连会触发
# "Searching Question Index ... Error!" 断言；示例 Q 用 New York 与真实
# question 区分，不会误匹配。
NIH_TASK_PROMPT = (
    "There is an important info hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about the important information.\n\n"
    "{context}\n\n"
    "Answer the quiz question with one complete sentence copied word for word "
    "from the text. Do not repeat the question. Do not add any extra words.\n\n"
    "Example:\n"
    "Question: What is the special magic number for New York mentioned in the "
    "provided text?\n"
    "Answer: The special magic numbers for New York is: 12345.\n\n"
    "{question}\n\n"
    "Answer:"
)

# 模型路径候选（本机 Windows / GPU 服务器 Linux）
MODEL_ROOT_CANDIDATES = [r"D:\Code\models", "/data/coding/models"]
PROVIDER_BY_KEYWORD = [("Llama", "LLaMA3"), ("Mistral", "Mistral"), ("Qwen", "Qwen"),
                       ("GLM", "GLM")]
# 便捷别名 -> 规范模型目录名（与 run_needle_in_haystack 的模型选择一致）
MODEL_ALIAS = {
    "llama": "Llama3-8B-Instruct",
    "llama3": "Llama3-8B-Instruct",
    "mistral": "Mistral-7B-Instruct-v0.2",
    "qwen": "Qwen2.5-1.5B-Instruct",
    "qwen2.5": "Qwen2.5-1.5B-Instruct",
}

DEFAULT_LENGTHS = [32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024,
                   512 * 1024, 1024 * 1024, 2 * 1024 * 1024]

# 参与统计的数值字段
STAT_FIELDS = ["ttft_s", "compress_time_s", "final_prefill_ttft_s",
               "decode_time_s", "walltime_s",
               "peak_mem_allocated_gb", "peak_mem_reserved_gb", "score"]


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def parse_length_token(s):
    """'32K' / '1M' / '1048576' -> token 数。"""
    s = str(s).strip().upper()
    mult = 1
    if s.endswith("K"):
        mult, s = 1024, s[:-1]
    elif s.endswith("M"):
        mult, s = 1024 * 1024, s[:-1]
    return int(float(s) * mult)


# --------------------------------------------------------------------- #
# NIH 上下文构造（忠实复刻 run_needle_in_haystack 的 token 级逻辑）      #
# --------------------------------------------------------------------- #
class NIHContextBuilder:
    """按 NIH 协议构造 (context_length, depth) -> 带 needle 的长文本。

    与 run_needle_in_haystack.LLMNeedleHaystackTester.generate_context 等价的
    token 级实现，但语料只 tokenize 一次（原实现按 while 循环重复整段编码，
    在 2M 长度下代价过高）。文件遍历顺序取 sorted(glob) 保证可复现。
    """

    def __init__(self, tokenizer, provider, haystack_dir, max_context_length,
                 needle=NEEDLE, buffer=FINAL_CONTEXT_LENGTH_BUFFER):
        self.enc = tokenizer
        self.provider = provider
        self.haystack_dir = haystack_dir
        self.max_context_length = max_context_length
        self.needle = needle
        self.buffer = buffer
        self._corpus_tokens = None
        self._cache = {}
        if provider == "LLaMA3":
            self.period_tokens = [13]
        elif provider == "Mistral":
            self.period_tokens = [842, 28723]
        elif provider == "GLM":
            self.period_tokens = [918, 30930]
        elif provider == "LLaMA":
            self.period_tokens = [29889, 869]
        else:
            self.period_tokens = self.encode_text_to_tokens(".")

    def encode_text_to_tokens(self, text):
        return self.enc.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens):
        return self.enc.decode(tokens, skip_special_tokens=True)

    def _load_corpus_tokens(self):
        if self._corpus_tokens is not None:
            return self._corpus_tokens
        files = sorted(glob.glob(os.path.join(self.haystack_dir, "*.txt")))
        if not files:
            raise FileNotFoundError("no haystack .txt under %s" % self.haystack_dir)
        tokens = []
        while len(tokens) < self.max_context_length:
            progressed = False
            for fp in files:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    tokens.extend(self.encode_text_to_tokens(f.read()))
                progressed = True
                if len(tokens) >= self.max_context_length:
                    break
            if not progressed:
                raise RuntimeError("haystack corpus too small for %d tokens"
                                   % self.max_context_length)
        self._corpus_tokens = tokens[: self.max_context_length]
        log("haystack corpus loaded: %d tokens" % len(self._corpus_tokens))
        return self._corpus_tokens

    def get_context(self, context_length, depth_percent):
        key = (context_length, depth_percent)
        if key in self._cache:
            return self._cache[key]
        corpus = self._load_corpus_tokens()
        tokens_needle = self.encode_text_to_tokens(self.needle)
        budget = context_length - self.buffer
        tokens_context = corpus[: max(0, budget - len(tokens_needle))]

        if depth_percent == 100:
            tokens_new = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new = tokens_context[:insertion_point]
            # 回退到最近的句号，保证 needle 落在句子边界（与原实现一致）
            while tokens_new and tokens_new[-1] not in self.period_tokens:
                insertion_point -= 1
                tokens_new = tokens_context[:insertion_point]
            tokens_new = (tokens_new + tokens_needle
                          + tokens_context[insertion_point:])

        context = self.decode_tokens(tokens_new)
        self._cache[key] = context
        return context


# --------------------------------------------------------------------- #
# 首 token 计时 streamer                                                #
# --------------------------------------------------------------------- #
class FirstTokenTimer(BaseStreamer):
    """挂在 model.generate(streamer=...) 上测 TTFT。

    transformers 5.x 会在生成循环开始前先 put(input_ids)（prompt 本身），
    必须跳过该次调用；首个非 prompt 的 put 发生在第一次前向完成之后，
    即为首 token 时刻（内部先 torch.cuda.synchronize 再取 perf_counter）。
    """

    def __init__(self, prompt_token_count=0):
        self.prompt_token_count = int(prompt_token_count)
        self.first_token_time = None
        self._prompt_put_seen = False
        self.put_count = 0

    def put(self, value):
        self.put_count += 1
        n = 0
        try:
            n = int(value.numel())
        except Exception:
            n = 1 if value is not None else 0
        if self.first_token_time is None and n > 0:
            is_prompt_put = (self.prompt_token_count > 0
                             and not self._prompt_put_seen
                             and n >= self.prompt_token_count)
            if is_prompt_put:
                self._prompt_put_seen = True
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.first_token_time = time.perf_counter()

    def end(self):
        pass


class TimedGenerateHook:
    """给 Pipline 内部的 model.generate 注入计时 streamer。

    Pipline.generate 末段会调用 self.model.generate(**model_inputs, ...)，
    这里把实例属性 generate 替换为包装函数：创建 FirstTokenTimer、记录
    t_gen_start / t_gen_end，供压缩阶段 / prefill / decode 三段分解。
    """

    def __init__(self, model):
        self.model = model
        self.last = None
        self._installed = False

    def install(self):
        if self._installed:
            return
        orig = self.model.generate
        hook = self

        def wrapped(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args and torch.is_tensor(args[0]):
                input_ids = args[0]
            prompt_count = int(input_ids.shape[-1]) if input_ids is not None else 0
            timer = FirstTokenTimer(prompt_token_count=prompt_count)
            kwargs.setdefault("streamer", timer)
            info = {"timer": timer, "t_gen_start": time.perf_counter()}
            hook.last = info
            try:
                return orig(*args, **kwargs)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                info["t_gen_end"] = time.perf_counter()

        self.model.generate = wrapped
        self._installed = True


# --------------------------------------------------------------------- #
# 方法运行状态                                                          #
# --------------------------------------------------------------------- #
class MethodState:
    def __init__(self, method, args, tokenizer, builder, out_dir):
        self.method = method
        self.args = args
        self.tokenizer = tokenizer
        self.builder = builder
        self.out_dir = out_dir
        self.model = None
        self.pipe = None
        self.gen_hook = None
        self.attn_implementation = None
        self.mem_after_load_gb = None
        self.prompt_cache = {}     # FullKV: (length, depth) -> tokenized inputs
        self.last_n_chunks = None


# --------------------------------------------------------------------- #
# 方法 runner：FullKV                                                   #
# --------------------------------------------------------------------- #
def apply_chat_prompt(tok, text):
    """与 InfiniComp build_message 一致的 chat 模板包裹（user 角色 + assistant
    起始；enable_thinking=False 与 pipline 的 disable_thinking 双保险对齐，
    对 Qwen2.5 等无 thinking 支持的模板逐字节无影响）。"""
    messages = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except (TypeError, KeyError):
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def run_fullkv(state, context, question, max_new_tokens):
    args = state.args
    model, tok = state.model, state.tokenizer

    prompt_text = apply_chat_prompt(
        tok, NIH_TASK_PROMPT.format(context=context, question=question))
    key = ("fullkv", state.cur_length, state.cur_depth)
    if key not in state.prompt_cache:
        t0 = time.perf_counter()
        enc = tok(prompt_text, return_tensors="pt")
        state.prompt_cache[key] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "tokenize_time_s": time.perf_counter() - t0,
        }
    cached = state.prompt_cache[key]
    input_ids = cached["input_ids"].to(model.device)
    prompt_len = int(input_ids.shape[1])

    timer = FirstTokenTimer(prompt_token_count=prompt_len)
    eos_ids = [tok.eos_token_id]
    try:
        eos_ids.append(tok.encode("\n", add_special_tokens=False)[-1])
    except Exception:
        pass

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            # 单序列无 padding：不传 attention_mask，sdpa 才能走
            # is_causal=True 的 flash 快路径（显式全 1 掩码会回退到
            # materialize 路径，峰值显存暴涨且在部分驱动上触发 sysmem
            # 溢出，观测到 100x 减速）。
            output_attentions=False,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            eos_token_id=eos_ids,
            use_cache=True,
            streamer=timer,
        )
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    new_tokens = output[0][prompt_len:]
    response = tok.decode(new_tokens, skip_special_tokens=True).strip()

    first_tok = timer.first_token_time if timer.first_token_time is not None else t_end
    return {
        "compress_time_s": 0.0,
        "final_prefill_ttft_s": first_tok - t_start,
        "ttft_s": first_tok - t_start,
        "decode_time_s": t_end - first_tok,
        "walltime_s": t_end - t_start,
        "tokenize_time_s": cached["tokenize_time_s"],
        "prompt_token_count": prompt_len,
        "actual_new_tokens": int(new_tokens.shape[0]),
        "response": response[:300],
    }


# --------------------------------------------------------------------- #
# 方法 runner：InfiniComp                                              #
# --------------------------------------------------------------------- #
def run_infiniretri(state, context, question, max_new_tokens):
    args = state.args
    pipe, hook = state.pipe, state.gen_hook
    data_dict = {"context": context, "question": question}
    state.last_n_chunks = None

    gen_name = "eff_%s_len_%d_depth_%d" % (
        os.path.basename(str(args.model_path)), state.cur_length, state.cur_depth)

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    pred, info = pipe.generate(
        data_dict=data_dict,
        gen_name=gen_name,
        max_new_tokens=max_new_tokens,
        prompt=NIH_TASK_PROMPT,
        temperature=0.1,
        task_name="nih",
    )
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    g = hook.last or {}
    timer = g.get("timer")
    t_gen_start = g.get("t_gen_start", t_end)
    t_gen_end = g.get("t_gen_end", t_end)
    first_tok = (timer.first_token_time if timer is not None
                 and timer.first_token_time is not None else t_gen_end)

    compress_time = t_gen_start - t_start
    prefill_ttft = first_tok - t_gen_start
    decode_time = t_gen_end - first_tok
    response = (pred or "").strip()

    return {
        "compress_time_s": compress_time,
        "final_prefill_ttft_s": prefill_ttft,
        "ttft_s": first_tok - t_start,
        "decode_time_s": decode_time,
        "walltime_s": t_end - t_start,
        "tokenize_time_s": None,
        "prompt_token_count": info.get("final_input_length"),
        "actual_new_tokens": None,
        "response": response[:300],
        "origin_context_length": info.get("origin_context_length"),
        "compressed_context_length": info.get("final_context_length"),
        "n_chunks": state.last_n_chunks,
        "pipe_all_pipline_cost": info.get("all_pipline_cost"),
        "pipe_final_generate_cost": info.get("final_generate_cost"),
    }


# --------------------------------------------------------------------- #
# 方法注册表（后续在此接入 SpecPrefill/CF、TopoCompress、SALT、          #
# LLMLingua / LLMLingua-2 —— 每个方法实现 setup/runner/teardown 三件套） #
# --------------------------------------------------------------------- #
def setup_fullkv(state):
    args = state.args
    impl = args.fullkv_attn
    if impl == "auto":
        chain = ["flash_attention_2", "sdpa", "eager"]
    else:
        chain = [impl]
    model = None
    last_err = None
    for impl_try in chain:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=impl_try,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            state.attn_implementation = impl_try
            break
        except (ValueError, ImportError, ModuleNotFoundError) as e:
            last_err = e
            log("attn_implementation=%s unavailable, falling back (%s)"
                % (impl_try, type(e).__name__))
    if model is None:
        raise RuntimeError("failed to load FullKV model: %s" % last_err)
    # 标准 serving 设定：开 KV cache（NIH 精度脚本里 full 分支的
    # use_cache=False 是为省显存的非标准做法，效率基准须用标准配置）。
    model.config.use_cache = True
    if model.generation_config is not None:
        model.generation_config.use_cache = True
    model.eval()
    state.model = model
    if state.attn_implementation != args.fullkv_attn and args.fullkv_attn == "auto":
        log("FullKV using attn_implementation=%s" % state.attn_implementation)


def setup_infiniretri(state):
    args = state.args
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
    ).eval()
    state.attn_implementation = "eager"
    state.model = model

    config = {
        "compress_budget": args.ir_topk,
        "phrase_token_num": args.ir_phrase_token_num,
        "chunk_size": args.ir_chunk_size,
        "data_format_map": {"context": "context", "question": "question"},
        # 计时区段内禁止磁盘 IO（原 NIH 脚本的 process.json 记录会污染计时）
        "record_retrieval_path": None,
        "record_attention_path": None,
        "record_result_path": None,
        "exact_eager": bool(args.ir_exact_eager),
    }
    pipe = InfiniComp(
        config=config, model=model, tokenizer=state.tokenizer,
        excute_log_path=os.path.join(state.out_dir, "pipline_InfiniComp.log"))
    state.pipe = pipe

    hook = TimedGenerateHook(model)
    hook.install()
    state.gen_hook = hook

    # 统计 chunk 数（不改变行为）
    orig_chunk_docs = pipe.chunks.chunk_docs

    def counted_chunk_docs(*a, **k):
        docs = orig_chunk_docs(*a, **k)
        state.last_n_chunks = len(docs)
        return docs

    pipe.chunks.chunk_docs = counted_chunk_docs


def teardown_state(state):
    state.pipe = None
    state.gen_hook = None
    state.model = None
    state.prompt_cache = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


METHODS = {
    "full": {"setup": setup_fullkv, "runner": run_fullkv,
             "display": "FullKV", "needs_pipe": False},
    "InfiniComp": {"setup": setup_infiniretri, "runner": run_infiniretri,
                    "display": "InfiniComp", "needs_pipe": True},
}


# --------------------------------------------------------------------- #
# 计时 + 显存峰值 + OOM 捕获的外层包装                                   #
# --------------------------------------------------------------------- #
def is_oom_exception(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def timed_measurement(state, context, question, max_new_tokens):
    """一次测量：reset 峰值统计 -> 运行方法 -> 记录峰值与耗时。

    任何 OOM（torch.cuda.OutOfMemoryError 或带 out of memory 的 RuntimeError）
    都被捕获为 status=oom 的记录而非让进程崩溃。
    """
    runner = METHODS[state.method]["runner"]
    record = {
        "method": METHODS[state.method]["display"],
        "status": "error",
        "peak_mem_allocated_gb": None,
        "peak_mem_reserved_gb": None,
        "oom_error": None,
    }
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        m = runner(state, context, question, max_new_tokens)
        record.update(m)
        record["status"] = "ok"
        resp = m.get("response") or ""
        if resp:
            record["score"] = scorer.score(NEEDLE, resp)["rouge1"].fmeasure * 10
        else:
            record["score"] = 0.0
        # 精确命中健全性检查：答案关键词是否出现在 response 中
        record["answer_hit"] = bool(ANSWER_KEYWORD and ANSWER_KEYWORD in resp)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        if is_oom_exception(exc):
            record["status"] = "oom"
            record["oom_error"] = str(exc)[:500]
            log("OOM captured at len=%d depth=%d: %s"
                % (state.cur_length, state.cur_depth, str(exc)[:120]))
        else:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)[:500]
            record["traceback"] = traceback.format_exc()[-2000:]
            log("ERROR at len=%d depth=%d: %s: %s"
                % (state.cur_length, state.cur_depth,
                   type(exc).__name__, str(exc)[:200]))
    finally:
        if torch.cuda.is_available():
            try:
                record["peak_mem_allocated_gb"] = \
                    torch.cuda.max_memory_allocated() / 1024 ** 3
                record["peak_mem_reserved_gb"] = \
                    torch.cuda.max_memory_reserved() / 1024 ** 3
            except Exception:  # noqa: BLE001
                pass
            torch.cuda.empty_cache()
    return record


# --------------------------------------------------------------------- #
# 模型路径解析                                                          #
# --------------------------------------------------------------------- #
def resolve_model(args):
    if args.model_path and os.path.isdir(args.model_path):
        path = args.model_path
    else:
        name = MODEL_ALIAS.get(args.model_name.lower(), args.model_name)
        candidates = [args.model_name, name] + \
            [os.path.join(root, name) for root in MODEL_ROOT_CANDIDATES] + \
            [os.path.join(root, name + "-Instruct")
             for root in MODEL_ROOT_CANDIDATES]
        path = next((c for c in candidates if c and os.path.isdir(c)), None)
        if path is None:
            raise FileNotFoundError(
                "model not found; pass --model_path explicitly (tried: %s)"
                % ", ".join(str(c) for c in candidates))
    provider = None
    for keyword, prov in PROVIDER_BY_KEYWORD:
        if keyword.lower() in path.lower():
            provider = prov
            break
    if provider is None:
        raise ValueError("cannot infer model provider from path: %s" % path)
    version = args.model_version or os.path.basename(path.rstrip("/\\"))
    return path, provider, version


# --------------------------------------------------------------------- #
# 汇总输出                                                              #
# --------------------------------------------------------------------- #
def summarize(records, methods, lengths):
    stats = {}
    for method in methods:
        display = METHODS[method]["display"]
        stats[display] = {}
        for L in lengths:
            rs = [r for r in records
                  if r.get("method") == display
                  and r.get("status") == "ok"
                  and r.get("context_length") == L]
            if not rs:
                stats[display][L] = None
                continue
            entry = {"n_ok": len(rs)}
            for f in STAT_FIELDS:
                vals = [r[f] for r in rs if r.get(f) is not None]
                if not vals:
                    entry[f] = None
                    continue
                entry[f] = {
                    "mean": statistics.fmean(vals),
                    "median": statistics.median(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "min": min(vals),
                    "max": max(vals),
                    "n": len(vals),
                }
            stats[display][L] = entry
    return stats


def cell_text(stats_entry, field, prec=2):
    if stats_entry is None:
        return None
    f = stats_entry.get(field)
    if not f:
        return None
    if f["std"] and f["std"] > 0:
        return "%.*f±%.*f" % (prec, f["mean"], prec + 1, f["std"])
    return "%.*f" % (prec, f["mean"])


def render_table(title, stats, methods, lengths, field, unit, records,
                 scaling=False):
    """输出文档表 1/表 2 风格的 Markdown 表。"""
    lines = ["### %s%s" % (title, (" (%s)" % unit) if unit else ""), ""]
    header = ["Method"] + [fmt_len(L) for L in lengths]
    if scaling:
        header.append("Scaling (max/min len)")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for method in methods:
        display = METHODS[method]["display"]
        row = [display]
        ok_lens = [L for L in lengths if stats.get(display, {}).get(L)]
        for L in lengths:
            entry = stats.get(display, {}).get(L)
            if entry is not None:
                txt = cell_text(entry, field)
                if txt is None:
                    txt = "n/a"
                n_oom = sum(1 for r in records if r.get("method") == display
                            and r.get("context_length") == L
                            and r.get("status") == "oom")
                if n_oom:
                    txt += " (oom %d/%d)" % (n_oom, n_oom + entry["n_ok"])
            else:
                oom = any(r.get("method") == display and r.get("context_length") == L
                          and r.get("status") == "oom" for r in records)
                txt = "OOM" if oom else "—"
            row.append(txt)
        if scaling and len(ok_lens) >= 2:
            lo, hi = ok_lens[0], ok_lens[-1]
            e_lo = stats[display][lo].get(field)
            e_hi = stats[display][hi].get(field)
            if e_lo and e_hi and e_lo["median"]:
                ratio = e_hi["median"] / e_lo["median"]
                row.append("%.2fx" % ratio)
            else:
                row.append("n/a")
        elif scaling:
            row.append("n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def fmt_len(L):
    if L % (1024 * 1024) == 0:
        return "%dM" % (L // (1024 * 1024))
    if L % 1024 == 0:
        return "%dK" % (L // 1024)
    return str(L)


def write_summary(run_dir, args, meta, records, stats, methods, lengths):
    summary_md = []
    summary_md.append("# NIH Efficiency Benchmark Summary\n")
    summary_md.append("- run_id: `%s`" % meta["run_id"])
    summary_md.append("- model: `%s` (%s, %s)" %
                     (meta["model_path"], meta["provider"], meta["torch_dtype"]))
    summary_md.append("- gpu: %s (%.1f GB)" %
                     (meta.get("gpu_name"), meta.get("gpu_total_gb") or 0))
    summary_md.append("- attn: FullKV=%s, InfiniComp=eager" % meta.get("fullkv_attn_used", "n/a"))
    summary_md.append("- methods: %s | repeats: %d | max_new_tokens: %d | depths: %s"
                     % (", ".join(METHODS[m]["display"] for m in methods),
                        args.repeats, args.max_new_tokens, args.depths))
    summary_md.append("- torch %s | transformers %s | %s\n"
                      % (meta["torch_version"], meta["transformers_version"],
                         meta["timestamp"]))

    summary_md.append("**测量定义**：")
    summary_md.append("- `TTFT` = 方法开始到首个输出 token 的墙钟时间（InfiniComp 含检索/压缩阶段，"
                      "与效率实验设计文档定义一致）；FullKV 为纯前向（tokenization 另列）。")
    summary_md.append("- `walltime` = 压缩 + prefill + %d-token 解码（SALT 协议）。" % args.max_new_tokens)
    summary_md.append("- 峰值显存 = `torch.cuda.max_memory_allocated()`（含模型权重）。")
    summary_md.append("- OOM 通过 try-except 捕获记录，进程不崩溃；`—` 表示该长度之前已 OOM 被跳过。\n")

    summary_md.append(render_table("表 1：峰值显存 Peak GPU Memory", stats, methods,
                                   lengths, "peak_mem_allocated_gb", "GB",
                                   records, scaling=True))
    summary_md.append(render_table("表 2：TTFT（含压缩阶段）", stats, methods,
                                   lengths, "ttft_s", "s", records))
    summary_md.append(render_table("表 3a：压缩阶段耗时 Compression time", stats, methods,
                                   lengths, "compress_time_s", "s", records))
    summary_md.append(render_table("表 3b：末段 prefill→首 token", stats, methods,
                                   lengths, "final_prefill_ttft_s", "s", records))
    summary_md.append(render_table("表 4：端到端 walltime（含 %d-token 解码）" % args.max_new_tokens,
                                   stats, methods, lengths, "walltime_s", "s", records))
    summary_md.append(render_table("表 5：NIH 精度（rouge1×10，健全性检查）", stats, methods,
                                   lengths, "score", "", records))

    # OOM 边界注记
    notes = []
    for method in methods:
        display = METHODS[method]["display"]
        oom_lens = sorted({r["context_length"] for r in records
                           if r.get("method") == display and r.get("status") == "oom"})
        if oom_lens:
            notes.append("- %s 在 %s 首次 OOM（更长档已按 skip-after-oom 跳过）"
                         % (display, ", ".join(fmt_len(L) for L in oom_lens)))
    if notes:
        summary_md.append("### OOM 边界\n")
        summary_md.extend(notes)
        summary_md.append("")

    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_md))

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "args": vars(args),
                   "stats": {m: {str(k): v for k, v in d.items()}
                             for m, d in stats.items()}}, f,
                  ensure_ascii=False, indent=2)
    return "\n".join(summary_md)


def make_plots(run_dir, stats, methods, lengths, records, max_new_tokens):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log("plotting skipped (matplotlib unavailable: %s)" % exc)
        return
    panels = [
        ("peak_mem_allocated_gb", "Peak GPU memory (GB)"),
        ("ttft_s", "TTFT incl. compression (s)"),
        ("walltime_s", "Walltime incl. %d-token decode (s)" % max_new_tokens),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (field, ylabel) in zip(axes, panels):
        for method in methods:
            display = METHODS[method]["display"]
            xs, ys, es = [], [], []
            for L in lengths:
                entry = stats.get(display, {}).get(L)
                if not entry or not entry.get(field):
                    continue
                xs.append(L)
                ys.append(entry[field]["median"])
                es.append(entry[field]["std"])
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=display)
            # OOM 墙标记：在首个 OOM 长度画 X（y 取最后一个成功点）
            oom_lens = sorted({r["context_length"] for r in records
                               if r.get("method") == display
                               and r.get("status") == "oom"})
            if oom_lens:
                ax.scatter([oom_lens[0]], [ys[-1]], marker="x", s=90, color="red")
                ax.annotate("OOM", (oom_lens[0], ys[-1]),
                            textcoords="offset points", xytext=(6, 4),
                            color="red", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = os.path.join(run_dir, "efficiency_curves.png")
    fig.savefig(out, dpi=150)
    log("plot saved: %s" % out)


# --------------------------------------------------------------------- #
# 主流程                                                                #
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="InfiniComp vs FullKV NIH efficiency benchmark "
                    "(TTFT / peak memory / compression time / walltime / OOM capture)")
    parser.add_argument("--methods", type=str, default="InfiniComp,full",
                        help="逗号分隔：InfiniComp,full（后续可扩展）")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct",
                        help="模型名或本地目录名")
    parser.add_argument("--model_path", type=str, default=None,
                        help="显式模型路径（优先于 model_name 解析）")
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument("--lengths", type=str,
                        default=",".join(fmt_len(L) for L in DEFAULT_LENGTHS),
                        help="逗号分隔的长度档，支持 32K/1M/2M 写法")
    parser.add_argument("--depths", type=str, default="50",
                        help="逗号分隔的 needle 深度（效率扫描默认单深度 50）")
    parser.add_argument("--repeats", type=int, default=3,
                        help="每档重复次数（论文建议 ≥10 取中位数）")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="解码 token 数（SALT 协议 128）")
    parser.add_argument("--warmup_length", type=int, default=4096,
                        help="正式测量前的小长度热身（<=0 关闭）")
    parser.add_argument("--skip-after-oom", dest="skip_after_oom",
                        action="store_true", default=True)
    parser.add_argument("--no-skip-after-oom", dest="skip_after_oom",
                        action="store_false",
                        help="OOM 后仍尝试更长档（默认跳过并记 skipped_after_oom）")
    parser.add_argument("--mem-fraction", type=float, default=0.0,
                        help=">0 时 torch.cuda.set_per_process_memory_fraction，"
                             "用于小卡上快速复现 OOM（0=不限制）")
    parser.add_argument("--fullkv-attn", type=str, default="auto",
                        choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--ir-topk", type=int, default=300)
    parser.add_argument("--ir-phrase-token-num", type=int, default=8)
    parser.add_argument("--ir-chunk-size", type=int, default=128)
    parser.add_argument("--ir-exact-eager", type=int, default=1,
                        help="1=论文可复现精确路径；0=SDPA 快速路径（更快，逐样本结果有核级漂移）")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(DirPath, "experiences", "efficiency"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this benchmark requires a GPU")

    lengths = sorted(set(parse_length_token(x) for x in args.lengths.split(",")))
    depths = sorted(set(float(x) for x in args.depths.split(",")))
    methods = []
    for m in args.methods.split(","):
        m = m.strip()
        if not m:
            continue
        if m.lower() in ("fullkv", "full"):
            m = "full"
        if m not in METHODS:
            raise ValueError("unknown method %r; available: %s"
                             % (m, list(METHODS)))
        if m not in methods:
            methods.append(m)

    args.model_path, provider, version = resolve_model(args)
    log("model: %s (provider=%s, version=%s)" % (args.model_path, provider, version))
    log("lengths: %s | depths: %s | methods: %s | repeats: %d"
        % ([fmt_len(L) for L in lengths], depths,
           [METHODS[m]["display"] for m in methods], args.repeats))

    if args.mem_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
        log("per-process GPU memory fraction set to %.3f (%.2f GB)"
            % (args.mem_fraction,
               torch.cuda.get_device_properties(0).total_memory
               / 1024 ** 3 * args.mem_fraction))

    run_id = "%s_%s_%s" % (version, "-".join(METHODS[m]["display"] for m in methods),
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    records_path = os.path.join(run_dir, "records.jsonl")

    import transformers
    meta = {
        "run_id": run_id,
        "model_path": args.model_path,
        "provider": provider,
        "model_version": version,
        "torch_dtype": "bfloat16",
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
        "gpu_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lengths": lengths,
        "depths": depths,
    }

    def append_record(rec, length, depth):
        rec["run_id"] = run_id
        rec["model_version"] = version
        rec["context_length"] = int(length)
        rec["depth_percent"] = float(depth)
        rec["max_new_tokens_cfg"] = args.max_new_tokens
        rec["timestamp"] = datetime.now(timezone.utc).isoformat()
        if state is not None and rec.get("attn_implementation") is None:
            rec["attn_implementation"] = state.attn_implementation
        with open(records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        records.append(rec)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    builder = NIHContextBuilder(
        tokenizer=tokenizer, provider=provider,
        haystack_dir=os.path.join(NIHDirPath, "PualGrahamEssays"),
        max_context_length=max(lengths + ([args.warmup_length]
                                           if args.warmup_length > 0 else [])))

    records = []
    state = None
    try:
        for method in methods:
            log("=" * 70)
            log("method: %s" % METHODS[method]["display"])
            log("=" * 70)
            state = MethodState(method, args, tokenizer, builder, run_dir)
            METHODS[method]["setup"](state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                state.mem_after_load_gb = \
                    torch.cuda.max_memory_allocated() / 1024 ** 3
            log("model loaded, attn=%s, mem_after_load=%.2f GB"
                % (state.attn_implementation, state.mem_after_load_gb))
            if method == "full":
                meta["fullkv_attn_used"] = state.attn_implementation

            # 热身（不计入 records）
            if args.warmup_length > 0:
                try:
                    state.cur_length = min(args.warmup_length, lengths[0])
                    state.cur_depth = 50.0
                    warm_ctx = builder.get_context(state.cur_length, state.cur_depth)
                    timed_measurement(state, warm_ctx, RETRIEVAL_QUESTION,
                                      max_new_tokens=4)
                    log("warmup done at %s" % fmt_len(state.cur_length))
                except Exception as exc:  # noqa: BLE001
                    log("warmup failed (ignored): %s" % exc)

            oomed = False
            for length in lengths:
                if oomed and args.skip_after_oom:
                    for depth in depths:
                        for rep in range(args.repeats):
                            append_record({
                                "method": METHODS[method]["display"],
                                "status": "skipped_after_oom",
                                "peak_mem_allocated_gb": None,
                                "peak_mem_reserved_gb": None,
                                "oom_error": None,
                                "repeat": rep,
                            }, length, depth)
                    log("len %s skipped (after OOM)" % fmt_len(length))
                    continue
                for depth in depths:
                    context = builder.get_context(length, depth)
                    for rep in range(args.repeats):
                        state.cur_length = length
                        state.cur_depth = depth
                        rec = timed_measurement(state, context,
                                                RETRIEVAL_QUESTION,
                                                args.max_new_tokens)
                        rec["repeat"] = rep
                        rec["mem_after_model_load_gb"] = state.mem_after_load_gb
                        if method == "InfiniComp":
                            rec["ir_chunk_size"] = args.ir_chunk_size
                            rec["ir_topk"] = args.ir_topk
                            rec["ir_phrase_token_num"] = args.ir_phrase_token_num
                            rec["ir_exact_eager"] = bool(args.ir_exact_eager)
                        append_record(rec, length, depth)
                        if rec["status"] == "ok":
                            log("[OK]   %s len=%s depth=%d rep=%d ttft=%.2fs "
                                "compress=%.2fs wall=%.2fs peak=%.2fGB score=%.1f hit=%s"
                                % (METHODS[method]["display"], fmt_len(length), depth,
                                   rep, rec["ttft_s"] or -1,
                                   rec["compress_time_s"] or 0,
                                   rec["walltime_s"] or -1,
                                   rec["peak_mem_allocated_gb"] or -1,
                                   rec["score"] if rec["score"] is not None else -1,
                                   rec.get("answer_hit")))
                        elif rec["status"] == "oom":
                            oomed = True
                            log("[OOM]  %s len=%s depth=%d rep=%d"
                                % (METHODS[method]["display"], fmt_len(length),
                                   depth, rep))
                        if oomed and args.skip_after_oom:
                            break
                    if oomed and args.skip_after_oom:
                        break
            teardown_state(state)
            state = None
    finally:
        if state is not None:
            teardown_state(state)
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    stats = summarize(records, methods, lengths)
    summary_text = write_summary(run_dir, args, meta, records, stats,
                                 methods, lengths)
    print("\n" + summary_text)
    if args.plot:
        make_plots(run_dir, stats, methods, lengths, records, args.max_new_tokens)
    log("run finished: %s" % run_dir)


if __name__ == "__main__":
    main()
