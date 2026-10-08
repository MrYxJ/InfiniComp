#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attention-based retrieval module for InfiniComp (prompt compression).

:class:`Retrieval` runs the language model over the current prompt, inspects
the question->context attention scores, and selects the context tokens the
query attends to most strongly. The selected tokens are decoded back into the
retrieved text, which the pipeline merges into the next prompt (prompt
compression). Attention heatmaps can optionally be recorded.

Two granularities locate the question / context segments inside the token
stream (``config["cache_granularity"]``):
    * ``sentence-level`` -- the default; the prompt is split sentence by
      sentence and each sentence is re-encoded, which yields the exact token
      position of every sentence (``sentence_index_split``).
    * ``token-level``    -- the raw context / question substrings are searched
      inside the built prompt and re-encoded piece by piece.
"""

import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from text_split import FEW_SHOT_SPLIT_WORDS, split_sentence, split_text_unit
from thoughts import Thoughts

# Sentinel used to mark "index not found yet" during the search.
_INDEX_NOT_FOUND = 1 << 28


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value states from KV heads to query heads (GQA)."""
    if n_rep == 1:
        return hidden_states
    bsz, n_kv, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, n_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, n_kv * n_rep, slen, head_dim)


def _additive_causal_mask(attention_mask, q_len, kv_len, dtype, device):
    """The additive mask the eager path would receive: 0 where attended,
    ``finfo(dtype).min`` elsewhere.

    Adapted from the collaborator's ``last_layer_attention.py``. Handles the
    mask forms transformers may hand down: ``None`` (plain causal, no
    padding), boolean, or an already-additive float mask.
    """
    zero = torch.zeros((), dtype=dtype, device=device)
    neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
    if attention_mask is None:
        # Queries are the last q_len positions of the kv sequence (there is
        # no cache offset in InfiniComp's single-forward calls).
        q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
        allowed = torch.arange(kv_len, device=device)[None, :] <= q_pos[:, None]
        return torch.where(allowed, zero, neg)[None, None]
    mask = attention_mask[:, :, :, :kv_len]
    if mask.dtype == torch.bool:
        return torch.where(mask, zero, neg)
    return mask.to(dtype)


def _head_summed_eager_probs(module, query, key, attention_mask, scaling,
                             max_group_bytes):
    """Sum over heads of eager-style attention probabilities, computed a few
    heads at a time to bound memory.

    Adapted from the collaborator's ``last_layer_attention.py``
    (``head_summed_eager_probs``). Per head this is exactly transformers'
    ``eager_attention_forward``::

        w = (Q @ K^T) * scaling ; w = w + mask ; softmax(w, fp32) -> cast back

    The probabilities are summed over heads with fp32 accumulation, matching
    ``bf16_tensor.sum(axis=0)`` in :mod:`thoughts`.

    Returns:
        Tensor of shape ``(batch, 1, q_len, kv_len)``.
    """
    bsz, n_heads, q_len, _ = query.shape
    key_states = _repeat_kv(key, module.num_key_value_groups)
    kv_len = key_states.shape[-2]
    mask = _additive_causal_mask(attention_mask, q_len, kv_len,
                                 query.dtype, query.device)
    # Rough upper bound of live bytes per head: bf16 scores, scaled/masked
    # copies, fp32 softmax, bf16 probs.
    per_head_bytes = q_len * kv_len * 12
    group = int(max(1, min(n_heads,
                           max_group_bytes // max(per_head_bytes, 1))))
    acc = torch.zeros((bsz, q_len, kv_len), dtype=torch.float32,
                      device=query.device)
    for h0 in range(0, n_heads, group):
        w = torch.matmul(query[:, h0:h0 + group],
                         key_states[:, h0:h0 + group].transpose(2, 3)) * scaling
        w = w + mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        acc += w.sum(dim=1, dtype=torch.float32)
        del w
    del mask, key_states
    return acc.to(query.dtype).unsqueeze(1)


class Retrieval:
    """Run the model, inspect attention, and return the retrieved context."""

    def __init__(self, config: Dict, model=None, tokenizer=None, logger=None) -> None:
        model_path = config["llm_name_or_path"]

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = tokenizer

        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", attn_implementation="eager")
        else:
            self.model = model

        self.config = config
        self.logger = logger

        self.record_attention = config.get("record_attention_path", None)
        self.plt_attention_scope = config.get("plt_attention_scope", "question2context")

        # When the first document segment is fed in, record the number of
        # prompt tokens that come before the context segment in the input.
        self.PREFIX_PROMPT_TOKEN_NUM = 0
        self.START_TIME = time.time()
        self.use_question_weight = config.get("use_question_weight", False)  # experiment trick
        self.QUESTION_WEIGHT = None

        # Retrieval scoring / analysis settings.
        self.compute_score_method = config.get("compute_score_method", "token")

        # Attention-capture mode (see _forward_with_layer_attention):
        #   False (default) -- fast path: non-target layers run fused SDPA,
        #       only the scoring layer's head-summed probabilities are kept
        #       (kernel-level drift vs eager; aggregate scores unaffected);
        #   True            -- exact path: every layer stays eager and the
        #       scoring layer's weights are head-summed inside the hook
        #       (bitwise identical to the original code, retention N^2 not
        #       H*N^2; use for reproducibility).
        self.exact_eager = config.get("exact_eager", False)
        # Per-group memory cap for the grouped head-sum probability
        # computation on the fast path.
        self.attn_group_max_bytes = int(
            config.get("attn_group_max_bytes", 4 * 2 ** 30))

        self.thoughts = Thoughts(
            config={
                "plt_save": self.record_attention,
                "plt_attention_scope": self.plt_attention_scope
            },
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Text splitting (delegated to the shared text_split module)         #
    # ------------------------------------------------------------------ #

    def split_sentence(self, text: str, spliter: str) -> List[str]:
        """Split by punctuation (in ``spliter``) and keep the punctuation."""
        return split_sentence(text, spliter)

    def split_text_in_few_shot(self, text: str, splite_word: str) -> List[str]:
        return split_text_in_few_shot(text, splite_word)

    def split_text_unit(self, text: str, spliter: str, task_name: str) -> List[str]:
        return split_text_unit(text, spliter, task_name)

    # ------------------------------------------------------------------ #
    # Prompt building                                                    #
    # ------------------------------------------------------------------ #

    def build_prompt(self, **kwargs):
        """Build the model prompt from ``data_dict`` and a template.

        Note: ``data_dict`` values are stripped in place.
        """
        data = kwargs["data_dict"]
        for key in data.keys():
            data[key] = data[key].strip()

        template = kwargs["prompt"]
        return template.format(**data)

    def build_message(self, text: str) -> str:
        """Wrap ``text`` in a chat-style message (Qwen2 format)."""
        if self.config["set_sys_prompt"]:
            message = [
                {"role": "system",
                 "content": "You are dealing with a QA-type task. "
                            "Please focus the content most relevant to the Question you are asking."},
                {"role": "user", "content": text},
            ]
        else:
            message = [{"role": "user", "content": text}]

        template_kwargs = {}
        if self.config.get("disable_thinking"):
            # Hybrid-thinking models (e.g. Qwen3): the official chat
            # template answers ``enable_thinking=False`` by inserting an
            # empty "<think>\n\n</think>\n\n" block (NoThinking mode),
            # giving a direct-answer generation that fits the small
            # LongBench output budgets. Templates without thinking
            # support (e.g. Qwen2.5) simply ignore the variable.
            template_kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(
            conversation=message, tokenize=False, add_generation_prompt=True,
            **template_kwargs)

    # ------------------------------------------------------------------ #
    # Index verification                                                 #
    # ------------------------------------------------------------------ #

    def verify_text_index(self, verify_text: str, input_ids: List[int],
                          start_index: int, end_index: int, spliter: str,
                          task_name: str) -> bool:
        """Check that ``[start_index, end_index)`` of ``input_ids`` decodes
        back to ``verify_text``.

        Used by the pre-processing step to make sure the located
        question / context segment boundaries are correct.

        Returns:
            ``True`` if the located segment matches ``verify_text``.
        """
        verify_ids = []
        for text in self.split_text_unit(verify_text, spliter, task_name):
            verify_ids.extend(self.tokenizer.encode(text, add_special_tokens=False))

        decode_verify_text = self.tokenizer.decode(verify_ids)
        input_text = self.tokenizer.decode(input_ids[start_index:end_index])

        # Handle dirty data where the context starts with a "<text>\n\n"
        # wrapper that is not part of the raw context text.
        if input_text.startswith("<text>\n\n"):
            input_text = input_text.replace("<text>\n\n", "")
            if input_text == decode_verify_text[:len(input_text)]:
                return True

        if input_text.strip() == decode_verify_text.strip():
            return True

        if self.logger is not None:
            self.logger.info(
                "Index verify failed:\n  input_ids segment [%d, %d]: %s\n" %  (start_index, end_index, input_text) + "expected text: %s" % decode_verify_text.strip())
        return False

    # ------------------------------------------------------------------ #
    # Pre-processing (locate question/context segments in the token ids) #
    # ------------------------------------------------------------------ #

    def pre_process_sentence(self, data_dict, prompt, task_name, spliter):
        """Encode the built prompt and locate the question/context segments.

        The prompt is split sentence by sentence; each sentence is encoded
        separately and concatenated, which yields the exact token position of
        every sentence (``sentence_index_split``) and of the question /
        context segments -- essential for attending over the right positions.

        Notes:
            * The current implementation requires the splitter to cleanly
              separate the context, the question and the prompt template.
              Keep those three parts clearly separated by ``\\n\\n`` in the
              prompt template.
            * The question-first / context-first segment may collide with an
              identical sentence in the prompt template; the probability is
              very low in practice and is not handled here.
        """
        input_text = self.build_prompt(data_dict=data_dict, prompt=prompt)
        assert input_text is not None, \
            "You need to provide a prompt template so build_prompt returns text."

        if task_name in ("trec", "triviaqa","samsum", "lcc", "repobench-p", "code_debug"):
            build_text = input_text
        else:
            build_text = self.build_message(input_text)

        split_build_texts = self.split_text_unit(
            build_text, spliter=spliter, task_name=task_name)

        # Use the same joining convention as build_prompt so the segments can
        # be matched back against the built prompt.
        context = data_dict[self.config["data_format_map"]["context"]].strip() + "\n\n"
        
        # Some task prompt have not question text, such as : InfiniBench-CodeDebug
        if self.config["data_format_map"]["question"] in prompt:
            question = data_dict[self.config["data_format_map"]["question"]].strip()
        else:
            question = ""

        question = question + "\n\n" if len(question) > 0 else ""  # question may be ""

        input_ids, index_split = [], []
        question_start_index, question_end_index = _INDEX_NOT_FOUND, _INDEX_NOT_FOUND
        context_start_index, context_end_index = _INDEX_NOT_FOUND, _INDEX_NOT_FOUND
        start = 0

        if task_name in FEW_SHOT_SPLIT_WORDS:
            question_first = question
            question_last = question
            question_token_length = len(
                self.tokenizer.encode(question, add_special_tokens=False))
            context_splited = self.split_text_unit(
                context, spliter=spliter, task_name=task_name)
            context_first = context_splited[1]
            context_last = context_splited[-1]
            context_token_length = sum(
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in context_splited[1:])
        else:
            if question != "":
                question_splited = self.split_sentence(question, spliter=spliter)
                question_first, question_last = question_splited[0], question_splited[-1]
                question_token_length = sum(
                    len(self.tokenizer.encode(text, add_special_tokens=False))
                    for text in question_splited)

            context_splited = self.split_sentence(context, spliter=spliter)
            context_first, context_last = context_splited[0], context_splited[-1]
            context_token_length = sum(
                len(self.tokenizer.encode(text, add_special_tokens=False))
                for text in context_splited)

        # A chunk whose first sentence is a punctuation-only fragment (e.g.
        # the ".." of a reST citation marker in source-code docstrings) is
        # glued onto the preceding prompt-template sentence by the sentence
        # merger (punctuation pieces always merge into the current
        # sentence), so no standalone sentence ever equals ``context_first``
        # and the index search below would fail. Split such a glued
        # sentence back into (template head, punctuation fragment) to
        # restore the exact context token boundary. This is a no-op for
        # every input whose first sentence matches normally, so previously
        # working prompts are unaffected.
        if (context_first.strip()
                and not any(ch.isalnum() for ch in context_first)
                and not any(
                    text.strip() == context_first.strip()
                    or text.replace("<text>\n\n", "").strip()
                    == context_first.strip()
                    for text in split_build_texts)):
            for i, text in enumerate(split_build_texts):
                if (text.endswith(context_first)
                        and len(text) > len(context_first)):
                    split_build_texts[i:i + 1] = [
                        text[: len(text) - len(context_first)], context_first]
                    break

        for text in split_build_texts:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_num = len(token_ids)

            input_ids.extend(token_ids)

            if start < context_start_index and (
                text.strip() == context_first.strip()
                or text.replace("<text>\n\n", "").strip() == context_first.strip()
            ):
                if self.PREFIX_PROMPT_TOKEN_NUM == 0:
                    self.PREFIX_PROMPT_TOKEN_NUM = start
                assert self.PREFIX_PROMPT_TOKEN_NUM == start, \
                    "PREFIX_PROMPT_TOKEN_NUM: %d Error! start: %d" % (
                        self.PREFIX_PROMPT_TOKEN_NUM, start)
                context_start_index = start
                # Provisional end by token counting; refined below.
                context_end_index = context_start_index + context_token_length

            # Align the context/question end to the build-text sentence
            # boundary instead of relying on token-count additivity: the
            # trailing newline run inside the prompt (e.g. the repobench-p
            # adjacent ``{context}{input}`` template, where an empty input
            # leaves "\n\n\n\n" after the context while ``context`` above
            # assumes a single "\n\n") can merge with the final content
            # token differently under BPE, shifting the token count and
            # misaligning a count-based end. The last sentence whose
            # stripped text matches ``context_last`` / ``question_last``
            # marks the true end; the newline difference then stays inside
            # the matched sentence and is absorbed by the strip-compare in
            # ``verify_text_index``. When no sentence matches, the
            # provisional count-based value above is kept (previous
            # behavior).
            if (context_start_index != _INDEX_NOT_FOUND
                    and start >= context_start_index
                    and text.strip() == context_last.strip()):
                context_end_index = start + token_num

            if question != "":
                if (start < question_start_index
                        and (start < context_start_index or start >= context_end_index)
                        and text.strip() == question_first.strip()):
                    question_start_index = start
                    # Provisional end by token counting; refined below.
                    question_end_index = question_start_index + question_token_length
                if (question_start_index != _INDEX_NOT_FOUND
                        and start >= question_start_index
                        and text.strip() == question_last.strip()):
                    question_end_index = start + token_num

            index_split.append(list(range(start, start + token_num)))
            start += token_num

        assert self.verify_text_index(
            context, input_ids, context_start_index, context_end_index,
            spliter, task_name), (
            "Searching Context Index in Final Input's Texts(after building prompt) Error!\n"
            "Splited texts: " + str(split_build_texts) + "Context First:" + context_first + "\n" + "Context Last:" + context_last + "\n" + "index split :" + str(index_split))

        if question != "" :
            assert self.verify_text_index(
                question, input_ids, question_start_index, question_end_index,
                spliter, task_name), (
                "Searching Question Index in Final Input's Texts(after building prompt) Error!\n"
                "Splited texts: " + str(split_build_texts) + "Question First:" + question_first)
        else:
            # When the question is empty, use the tail of the prompt (the
            # tokens right before the output position).
            question_start_index = context_end_index
            question_end_index = start

        if self.config["question_add_prompt"]:
            question_end_index = start  # include the tokens after the question

        if self.config["question_use_last"]:
            question_end_index = start
            question_start_index = question_end_index - 1  # last token of the prompt

        process_info = {
            "current_input_text": build_text,
            "sentence_index_split": index_split,
            "question_start_index": question_start_index,
            "question_end_index": question_end_index,
            "context_start_index": context_start_index,
            "context_end_index": context_end_index,
        }

        return torch.tensor([input_ids], device=self.model.device), process_info

    def pre_process_token(self, data_dict, prompt):
        """Encode the prompt and locate segments by substring search.

        Instead of splitting sentences, the raw context / question substrings
        are searched inside the built chat prompt. This requires the prompt
        to contain the exact (stripped) context and question substrings.
        """
        input_text = self.build_prompt(data_dict=data_dict, prompt=prompt)
        assert input_text is not None, \
            "You need to provide a prompt template so build_prompt returns text."

        build_text = self.build_message(input_text)
        build_text_length = len(build_text)

        context = data_dict[self.config["data_format_map"]["context"]].strip() + "\n\n"
        question = data_dict[self.config["data_format_map"]["question"]].strip()
        question = question + "\n\n" if len(question) > 0 else ""
        context_text_length, question_text_length = len(context), len(question)

        context_text_start = build_text.find(context)
        assert build_text[context_text_start: context_text_start + context_text_length] == context, \
            "Find context text index error!"

        if question_text_length > 0:
            question_text_start = build_text.find(question)
            assert build_text[question_text_start: question_text_start + question_text_length] == question, \
                "Find question text index error!"

        input_ids = []
        if context_text_start > 0:
            prompt_prefix_ids = self.tokenizer.encode(
                build_text[0: context_text_start], add_special_tokens=False)
            input_ids.extend(prompt_prefix_ids)

            context_start_index = len(input_ids)
            context_ids = self.tokenizer.encode(context, add_special_tokens=False)
            input_ids.extend(context_ids)
            context_end_index = context_start_index + len(context_ids)

            if question_text_length > 0:
                if context_text_start + context_text_length < question_text_start:
                    prompt_mid_ids = self.tokenizer.encode(
                        build_text[context_text_start + context_text_length: question_text_start],
                        add_special_tokens=False)
                    input_ids.extend(prompt_mid_ids)

                question_start_index = len(input_ids)
                question_ids = self.tokenizer.encode(question, add_special_tokens=False)
                input_ids.extend(question_ids)
                question_end_index = question_start_index + len(question_ids)

            if question_text_length > 0 and question_text_start + question_text_length < build_text_length:
                prompt_last_ids = self.tokenizer.encode(
                    build_text[question_text_start + question_text_length:],
                    add_special_tokens=False)
                input_ids.extend(prompt_last_ids)
            elif question_text_length == 0 and context_text_start + context_text_length < build_text_length:
                prompt_last_ids = self.tokenizer.encode(
                    build_text[context_text_start + context_text_length:],
                    add_special_tokens=False)
                input_ids.extend(prompt_last_ids)
        else:
            raise ValueError(
                "Token granularity: cannot locate the context text inside "
                "the built prompt; make sure the prompt contains the exact "
                "context substring.")

        if question_text_length == 0 or self.config["question_use_last"]:
            question_end_index = len(input_ids)
            question_start_index = question_end_index - 1

        if self.config["question_add_prompt"]:
            question_end_index = len(input_ids)  # include the tokens after the question

        process_info = {
            "current_input_text": build_text,
            "question_start_index": question_start_index,
            "question_end_index": question_end_index,
            "context_start_index": context_start_index,
            "context_end_index": context_end_index,
        }

        return torch.tensor([input_ids], device=self.model.device), process_info

    # ------------------------------------------------------------------ #
    # Sentence selection                                                 #
    # ------------------------------------------------------------------ #

    def select_sentence(self, selected_token_index, index_split, compress_budget=None):
        """Return the token indices of every sentence that contains at least
        one selected token, stopping once ``compress_budget`` tokens are
        gathered (``None`` = no budget, pure top-k selection).

        Sentences are admitted in the score order of the selected token
        indices (top-k phrases), so the compressed length is dynamic: it is
        driven by the number of selected phrases and only bounded above by
        ``compress_budget``. The sentence order of the original text is
        preserved.

        Complexity: O(all input tokens).
        """
        sentence_num = len(index_split)
        token2sentence = {}
        for index_sen, split in enumerate(index_split):
            for index_id in split:
                token2sentence[index_id] = index_sen

        select_sentence_index = [0] * sentence_num
        select_token_length = 0
        for token_index in selected_token_index:
            if select_sentence_index[token2sentence[token_index]] == 0:
                select_sentence_index[token2sentence[token_index]] = 1
                select_token_length += len(index_split[token2sentence[token_index]])

            if (compress_budget is not None
                    and select_token_length >= compress_budget):
                break

        return [index_split[index] for index in range(sentence_num)
                if select_sentence_index[index] == 1]

    # ------------------------------------------------------------------ #
    # Main retrieval loop                                                #
    # ------------------------------------------------------------------ #

    def _call_model_for_retrieval(self, input_ids):
        """Forward pass for retrieval: no KV cache, smallest possible logits.

        ``use_cache=False`` skips the KV cache (no generation happens here)
        and the "logits to keep" argument avoids materializing the full
        ``(seq, vocab)`` logits tensor, which retrieval never consumes.
        transformers renamed that argument across versions; a TypeError means
        the call was rejected before the forward pass ran, so retrying the
        next spelling is safe.
        """
        for extra_kwargs in ({"logits_to_keep": 1},
                             {"num_logits_to_keep": 1}, {}):
            try:
                return self.model(
                    input_ids=input_ids, output_attentions=False,
                    use_cache=False, **extra_kwargs)
            except TypeError:
                continue
        raise RuntimeError(
            "unreachable: model call rejected every logits argument spelling")

    def _resolve_layer_index(self, layer: int) -> int:
        """Translate ``layer`` (negative allowed) into an absolute index."""
        layers = self.model.model.layers  # HF decoder stack (Qwen2-style)
        layer_index = layer if layer >= 0 else len(layers) + layer
        if not 0 <= layer_index < len(layers):
            raise ValueError(
                "layer %r out of range for a %d-layer model"
                % (layer, len(layers)))
        return layer_index

    def _forward_with_layer_attention(self, input_ids, layer: int):
        """Run one forward pass and capture one layer's attention weights.

        Naively enabling ``output_attentions=True`` on the whole model keeps
        *every* layer's ``(1, num_heads, seq, seq)`` tensor alive at once --
        roughly ``num_layers * num_heads * seq_len^2`` values, which dominates
        GPU memory on long inputs. The scoring only needs one layer, so both
        implementations below retain only that layer's attention:

        * fast (default, :meth:`_forward_fast_layer_attention`): non-target
          layers run fused SDPA (no seq^2 matrix anywhere) and only the
          target layer's head-summed probabilities are recomputed.
        * exact (``config["exact_eager"] = True``,
          :meth:`_forward_exact_layer_attention`): every layer stays eager
          and the target layer's weights are head-summed inside the hook --
          bitwise identical to the original ``output_attentions=True`` code
          (verified 160/160 LongBench samples). Use this for
          paper-reproducibility checks.

        Any failure of the fast path falls back to the exact one.

        Returns:
            ``(model_output, layer_attention)`` where ``layer_attention`` is
            the ``(1, 1, seq, seq)`` head-summed tensor on both paths;
            ``thoughts`` sums over the (singleton) head axis either way.
        """
        if self.exact_eager:
            return self._forward_exact_layer_attention(input_ids, layer)
        try:
            return self._forward_fast_layer_attention(input_ids, layer)
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning(
                    "Fast attention capture unavailable (%s: %s); falling "
                    "back to the exact eager+hook path."
                    % (type(exc).__name__, exc))
            return self._forward_exact_layer_attention(input_ids, layer)

    def _forward_exact_layer_attention(self, input_ids, layer: int):
        """Capture the target layer's head-summed attention (all-eager).

        The model stays on eager attention and the target layer's attention
        module is wrapped: it materializes its weights (eager attention
        computes them anyway) and we stash a reference, while the module
        reports "no attention weights" back to the parent model so nothing
        else is retained.

        The hook immediately reduces the full ``(1, num_heads, N, N)``
        weights to their head-sum ``(1, 1, N, N)`` with the *exact same
        torch op* ``thoughts`` applies later (``w[0].sum(axis=0)``), so the
        result is bitwise identical to the original ``output_attentions=
        True`` implementation (verified 160/160 LongBench samples) while the
        per-head tensor dies with the hook's frame instead of being kept
        alive until scoring -- the retention drops from ``H * N^2`` to
        ``N^2`` values, which matters for 7B-scale long inputs.

        Returns:
            ``(model_output, layer_attention)`` where ``layer_attention`` is
            the captured ``(1, 1, seq, seq)`` head-summed tensor of ``layer``.
        """
        layer_index = self._resolve_layer_index(layer)
        target = self.model.model.layers[layer_index]
        original_forward = target.self_attn.forward
        captured = {}

        def capture_forward(*args, **kwargs):
            # Force the attention module to return its weights, head-sum them
            # right here (same op thoughts uses: w[0].sum(axis=0) -- bitwise
            # equal to summing later), then restore the "no weights" return
            # shape so the parent model never collects them.
            kwargs["output_attentions"] = True
            attn_output, attn_weights, *rest = original_forward(*args, **kwargs)
            captured["weights"] = (
                attn_weights[0].sum(dim=0).unsqueeze(0).unsqueeze(0))
            return (attn_output, None, *rest)

        target.self_attn.forward = capture_forward
        try:
            output = self._call_model_for_retrieval(input_ids)
        finally:
            target.self_attn.forward = original_forward

        if captured.get("weights") is None:
            raise RuntimeError(
                "Failed to capture attention weights from layer %d; the "
                "model must use an attention implementation that can return "
                "attentions (e.g. attn_implementation='eager')." % layer_index)
        return output, captured["weights"]

    def _forward_fast_layer_attention(self, input_ids, layer: int):
        """Capture the target layer's head-summed probabilities (fast path).

        Every layer's attention OUTPUT is computed with PyTorch SDPA, so no
        layer ever materializes the ``(heads, seq, seq)`` matrix. On the
        target layer the eager-style probabilities are additionally recomputed
        a few heads at a time and only their SUM OVER HEADS is kept, as a
        ``(1, 1, seq, seq)`` tensor -- exactly what the scoring consumes.

        This mirrors the collaborator's ``last_layer_attention.py``, but the
        interception mounts on the model module's ``eager_attention_forward``
        global (referenced by the attention modules at call time in both
        transformers 4.5x and 5.x) instead of the global attention-function
        registry, and it is restored right after this call, so nothing
        outside this forward pass is affected.

        Trade-off vs. the exact path: non-target layers no longer run eager,
        so hidden states can drift from the pure-eager run at the bf16
        rounding level. The retrieved text is verified to match on short
        samples; set ``config["exact_eager"] = True`` for bitwise-exact
        reproduction.

        Returns:
            ``(model_output, layer_attention)`` where ``layer_attention`` is
            the captured ``(1, 1, seq, seq)`` head-summed tensor of ``layer``.
        """
        layers = self.model.model.layers
        layer_index = self._resolve_layer_index(layer)

        # Sliding-window layers need window-aware masks that neither the SDPA
        # shortcut nor the probability recomputation handles; bail out to the
        # exact path (the dispatcher catches this).
        if any(getattr(l.self_attn, "sliding_window", None) is not None
               for l in layers):
            raise RuntimeError(
                "fast capture does not support sliding-window layers")

        attn_module = sys.modules[type(layers[0].self_attn).__module__]
        original_eager = getattr(attn_module, "eager_attention_forward", None)
        if original_eager is None:
            raise RuntimeError(
                "%s has no eager_attention_forward to intercept"
                % attn_module.__name__)

        captured = {}
        group_max_bytes = self.attn_group_max_bytes

        def fast_eager_forward(module, query, key, value, attention_mask,
                               scaling=None, dropout=0.0, **kwargs):
            key_states = _repeat_kv(key, module.num_key_value_groups)
            value_states = _repeat_kv(value, module.num_key_value_groups)
            # InfiniComp always feeds a single unpadded sequence, so the
            # additive causal mask is redundant and we can take the is_causal
            # kernel path without materializing or expanding any mask. A real
            # (padded) batch would go through the explicit-mask branch.
            if attention_mask is None or query.shape[0] == 1:
                attn_output = F.scaled_dot_product_attention(
                    query, key_states, value_states,
                    dropout_p=dropout if module.training else 0.0,
                    scale=scaling,
                    is_causal=query.shape[2] > 1)
            else:
                attn_output = F.scaled_dot_product_attention(
                    query, key_states, value_states,
                    attn_mask=attention_mask,
                    dropout_p=dropout if module.training else 0.0,
                    scale=scaling,
                    is_causal=False)
            attn_output = attn_output.transpose(1, 2).contiguous()

            if getattr(module, "layer_idx", None) == layer_index:
                captured["weights"] = _head_summed_eager_probs(
                    module, query, key, attention_mask,
                    scaling if scaling is not None else module.scaling,
                    group_max_bytes)
            return attn_output, None

        attn_module.eager_attention_forward = fast_eager_forward
        try:
            output = self._call_model_for_retrieval(input_ids)
        finally:
            attn_module.eager_attention_forward = original_eager

        if captured.get("weights") is None:
            raise RuntimeError(
                "Fast path captured nothing from layer %d; the attention "
                "dispatch did not route through eager_attention_forward."
                % layer_index)
        return output, captured["weights"]

    def retrieval_attention(
            self, data_dict: Dict[str, str], prompt: str, phrase_token_num:int,
            topk: int, compress_budget: int, iter_number: int,
            retri_split_pattern = "(\n\n)|([。！？.!?])", layer=-1,
            generate_token=False, doc_id=None, task_name=None):
        """Run one retrieval step over one document chunk (prompt compression).

        Args:
            data_dict: Contains ``context`` / ``question`` under the keys of
                ``config["data_format_map"]``. The context already contains
                the previously retrieved text merged in by the pipeline.
            prompt: Prompt template.
            topk: Number of top attention-score indices to select (phrase
                granularity via ``phrase_token_num``).
            compress_budget: Upper bound of the compressed length in tokens;
                ``None`` means unbounded (dynamic length).
            iter_number: Current iteration index (for bookkeeping).
            layer: Which layer's attention is used for scoring. Only this
                layer's weights are captured; every other layer releases
                its attention immediately.
            generate_token: If ``True``, run one generation step and re-encode
                the produced token before scoring.
            doc_id: Identifier of the current chunk (for bookkeeping).
            task_name: Task name used for text splitting.

        Returns:
            ``(output, select_indexs, retrieved_text, record_info)``.
        """
        if self.config["cache_granularity"] == "sentence-level":
            input_ids, process_info = self.pre_process_sentence(
                data_dict=data_dict, prompt=prompt, 
                task_name=task_name, spliter=retri_split_pattern)
        elif self.config["cache_granularity"] == "token-level":
            input_ids, process_info = self.pre_process_token(
                data_dict=data_dict, prompt=prompt)
        else:
            raise ValueError(
                "Unknown cache_granularity: %r" % self.config["cache_granularity"])

        with torch.no_grad():
            if generate_token:
                # Only the generated sequence is consumed below; requesting
                # attentions from generate() would keep every layer's prefill
                # attention alive for nothing.
                output = self.model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True, max_new_tokens=1)
                response = self.tokenizer.batch_decode(
                    output.sequences, skip_special_tokens=False)[0]
                inputs = self.tokenizer([response], return_tensors="pt").to(self.model.device)
                input_ids = inputs.input_ids
                # Score the freshly generated token as the query.
                process_info["current_input_text"] = response
                process_info["question_start_index"] = process_info["question_end_index"]
                process_info["question_end_index"] = process_info["question_start_index"] + 1

            # Capture only the requested layer's attention; every other layer
            # releases its weights immediately (the naive output_attentions=
            # True keeps all layers' attention matrices alive at once).
            output, layer_attention = self._forward_with_layer_attention(
                input_ids, layer)

        # The tuple only holds the captured layer, so select index 0.
        focus_indexs, attention_weights, _q_w = self.thoughts.get_focus_index(
            process_info=process_info,
            output_attentions=(layer_attention,),
            layer=0,
            topk_value=topk,
            first_q_w=self.QUESTION_WEIGHT,
            phrase_token_num=phrase_token_num,
        )
        attention_weights = attention_weights.float().cpu().detach().numpy()
        del layer_attention  # release the (1, num_heads, seq, seq) GPU tensor

        # Map the selected focus tokens back to sentence boundaries and
        # decode them into the retrieved text.
        if self.config["cache_granularity"] == "sentence-level":
            select_split_indexs = self.select_sentence(
                focus_indexs, process_info["sentence_index_split"], compress_budget)
        else:  # token-level: the focus tokens themselves are the selection
            select_split_indexs = [focus_indexs]
        select_indexs = [index for sublist in select_split_indexs for index in sublist]

        input_token_ids = input_ids[0].cpu().tolist()
        assert len(input_token_ids) == attention_weights.shape[1], "%d != %d" % (
            attention_weights.shape[1], len(input_token_ids))

        select_split_tokens = [
            [input_token_ids[token_index] for token_index in sentence_index]
            for sentence_index in select_split_indexs
        ]
        select_tokens = [token for sublist in select_split_tokens for token in sublist]
        focus_text = self.tokenizer.decode(select_tokens)

        record_info = {
            "iter_number": iter_number,
            "doc_id": doc_id,
            "input_text": process_info["current_input_text"],
            "focus_text": focus_text,
            "context_start_index": process_info["context_start_index"],
            "context_end_index": process_info["context_end_index"],
            "question_start_index": process_info["question_start_index"],
            "question_end_index": process_info["question_end_index"],
            "original_num": input_ids.shape[1],
            "select_token_num": len(select_tokens),
            "topk": topk,
            "compress_budget": compress_budget,
            "phrase_token_num": phrase_token_num,
            "retri_split_pattern": retri_split_pattern,
            "cost_time": float(time.time() - self.START_TIME),
        }

        # Optionally record the attention heatmap.
        if self.record_attention:
            self._record_attention_heatmap(
                attention_weights=attention_weights,
                input_token_ids=input_token_ids,
                process_info=process_info,
                focus_indexs=focus_indexs,
                doc_id=doc_id,
            )

        return output, select_indexs, focus_text, record_info

    def _record_attention_heatmap(self, attention_weights, input_token_ids,
                                  process_info, focus_indexs, doc_id) -> None:
        """Slice the attention matrix according to ``plt_attention_scope``
        and accumulate it into :class:`Thoughts` for rendering."""
        malplot_index = [index - self.PREFIX_PROMPT_TOKEN_NUM for index in focus_indexs]

        if doc_id:
            dir_name = "_".join(doc_id.split("_")[:-1])
            record_dir = os.path.join(self.record_attention, dir_name)
        else:
            record_dir = self.record_attention
        os.makedirs(record_dir, exist_ok=True)

        if self.plt_attention_scope == "all2all":  # full attention matrix
            key_tokens = input_token_ids
            query_tokens = input_token_ids
            attention_matrix = np.sum(attention_weights, axis=0, keepdims=True)

        elif self.plt_attention_scope == "question2context":  # context as key, question as query
            key_tokens = input_token_ids[
                self.PREFIX_PROMPT_TOKEN_NUM: process_info["context_end_index"]]
            query_tokens = input_token_ids[
                process_info["question_start_index"]: process_info["question_end_index"]]
            attention_matrix = np.copy(attention_weights[
                process_info["question_start_index"]: process_info["question_end_index"],
                self.PREFIX_PROMPT_TOKEN_NUM: process_info["context_end_index"],
            ])

        elif self.plt_attention_scope == "question2context+question":  # context+question as key
            key_tokens = input_token_ids[
                self.PREFIX_PROMPT_TOKEN_NUM: process_info["question_end_index"]]
            query_tokens = input_token_ids[
                process_info["question_start_index"]: process_info["question_end_index"]]
            attention_matrix = np.copy(attention_weights[
                process_info["question_start_index"]: process_info["question_end_index"],
                self.PREFIX_PROMPT_TOKEN_NUM: process_info["question_end_index"],
            ])

        else:
            raise ValueError("Unknown plt_attention_scope: %r" % self.plt_attention_scope)

        key_labels = [self.tokenizer.decode(token_id) for token_id in key_tokens]
        query_labels = [self.tokenizer.decode(token_id) for token_id in query_tokens]

        assert len(key_labels) == len(attention_matrix[0]), "row token length must match"
        assert len(query_labels) == len(attention_matrix), "col token length must match"

        self.thoughts.add_malplot_attention(
            attention_weights=attention_matrix,
            x_labels=key_labels,
            y_labels=query_labels,
            focus_indexs=malplot_index,
        )

        # If the query spans multiple tokens, also render the row-summed
        # version so the per-token focus is visible.
        if len(query_labels) > 1:
            summed_matrix = np.sum(attention_matrix, axis=0, keepdims=True)
            self.thoughts.add_malplot_attention(
                attention_weights=summed_matrix,
                x_labels=key_labels,
                y_labels=[query_labels[0]],
                focus_indexs=malplot_index,
            )
