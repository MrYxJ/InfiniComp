#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chunking utilities for InfiniComp.

:class:`Chunks` splits long contexts into continuous, size-bounded text
segments that are later fed to the retrieval module. The sentence-level
splitting rules live in :mod:`text_split` so that :class:`Retrieval` (which
must locate the same segments back inside the token stream) stays perfectly
consistent with the chunker.
"""

from typing import Any, List

import tiktoken

from text_split import split_sentence, split_text_in_few_shot, split_text_unit


class Chunks:
    """Split long contexts into size-bounded chunks while keeping sentence
    boundaries and the original text formatting as much as possible."""

    def __init__(self, config: dict, tokenizer, logger) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logger
        # The pattern of text segmentation (re.split) used during chunking,
        # which determines the granularity of each retrieval unit.
        self.split_sen_pattern = config.get("retri_split_pattern")

    # ------------------------------------------------------------------ #
    # Thin wrappers over the shared splitting rules (kept for API compat). #
    # ------------------------------------------------------------------ #

    def split_sentence(self, text: str, spliter: str) -> List[str]:
        """Split by punctuation (in ``spliter``) and keep the punctuation."""
        return split_sentence(text, spliter)

    def split_text_in_few_shot(self, text: str, splite_word: str) -> List[str]:
        return split_text_in_few_shot(text, splite_word)

    def split_text_unit(self, text: str, spliter: str, task_name: str) -> List[str]:
        return split_text_unit(text, spliter, task_name)

    # ------------------------------------------------------------------ #
    # Chunking                                                           #
    # ------------------------------------------------------------------ #

    def split_into_chunks(self, text: str, chunk_size: int, spliter: str,
                          task_name: str) -> List[str]:
        """Split ``text`` into chunks whose token length never exceeds
        ``chunk_size``.

        Sentences are accumulated until the next sentence would overflow the
        limit; an overlong chunk is recursively re-split using single spaces
        as separators. As a final step the last two chunks are rebalanced so
        their lengths are roughly equal.

        Args:
            text: The original input context.
            chunk_size: Maximum token length of each chunk.
            spliter: Punctuation used to cut individual sentences.
            task_name: Task name used to select the splitting rule.

        Returns:
            A collection of continuous text segments.
        """
        sentences = self.split_text_unit(text, spliter, task_name)

        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence_length = self.get_prompt_length(sentence)
            if self.get_prompt_length(current_chunk) + sentence_length <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    if self.get_prompt_length(current_chunk) <= chunk_size:
                        chunks.append(current_chunk)
                    elif spliter != " ":  # avoid infinite recursion
                        chunks.extend(self.split_into_chunks(
                            current_chunk, chunk_size=chunk_size,
                            spliter=" ", task_name=task_name))
                current_chunk = sentence

        if current_chunk != "":
            if self.get_prompt_length(current_chunk) <= chunk_size:
                chunks.append(current_chunk)
            elif spliter != " ":  # avoid infinite recursion
                chunks.extend(self.split_into_chunks(
                    current_chunk, chunk_size=chunk_size,
                    spliter=" ", task_name=task_name))
            else:  # few-shot chunks that cannot be split any further
                chunks.append(current_chunk)

        # Rebalance the last two chunks so their lengths are about equal.
        # Only the popped pair participates: re-splitting the full input
        # ``text`` here (as the original revision did) would duplicate the
        # content of the preceding chunks and drop the middle of the document
        # whenever the input spans three or more chunks.
        if len(chunks) > 1 and self.get_prompt_length(chunks[-1]) < chunk_size // 2:
            last_chunk = chunks.pop()
            penultimate_chunk = chunks.pop()
            combined_text = penultimate_chunk + last_chunk

            new_sentences = self.split_text_unit(combined_text, spliter, task_name)

            new_penultimate_chunk = ""
            new_last_chunk = ""
            i, j = 0, len(new_sentences) - 1

            while i <= j and len(new_sentences) != 1:
                flag = False
                if self.get_prompt_length(new_penultimate_chunk + new_sentences[i]) <= chunk_size:
                    flag = True
                    new_penultimate_chunk += new_sentences[i]
                    if i == j:
                        break
                    i += 1
                if self.get_prompt_length(new_last_chunk + new_sentences[j]) <= chunk_size:
                    new_last_chunk = new_sentences[j] + new_last_chunk
                    j -= 1
                    flag = True
                if not flag:
                    break

            if i < j:
                # Allocate the remaining middle part word by word.
                remaining_sentences = new_sentences[i:j + 1]
                if remaining_sentences:
                    remaining_text = "".join(remaining_sentences)
                    words = remaining_text.split(" ")
                    end_index = len(words) - 1
                    for index, word in enumerate(words):
                        if self.get_prompt_length(" ".join([new_penultimate_chunk, word])) <= chunk_size:
                            new_penultimate_chunk = " ".join([new_penultimate_chunk, word])
                        else:
                            end_index = index
                            break
                    if end_index != len(words) - 1:
                        new_last_chunk = " ".join(words[end_index:]) + " " + new_last_chunk

            if len(new_sentences) == 1:
                chunks.append(penultimate_chunk)
                chunks.append(last_chunk)
            else:
                chunks.append(new_penultimate_chunk)
                chunks.append(new_last_chunk)

        return chunks

    def chunk_docs(
        self, doc: str, chunk_size: int, task_name: str,
        split_sen_pattern: str, separator: str = "\n", 
        chunk_overlap: int = 0, question: str = None) -> List[str]:
        """Split an entire long context into continuous fragments, each no
        longer than ``chunk_size`` tokens, with optional overlap.

        Args:
            doc: The original long context to be split.
            chunk_size: Maximum token length of each fragment.
            task_name: Task name used to select the splitting rule.
            separator: First-level split punctuation. Defaults to ``\\n``.
            chunk_overlap: Overlap in tokens between consecutive fragments.
            question: Optional question prepended to the prompt; its token
                length is subtracted from ``chunk_size``.

        Returns:
            A collection of continuous text segments obtained from ``doc``.
        """
        if question is not None:
            chunk_size = chunk_size - self.get_prompt_length(question)

        splits = self.split_text_unit(doc, split_sen_pattern, task_name)
        splits = [s for s in splits if s != ""]
        separator_len = self.get_prompt_length_no_special(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            now_len = self.get_prompt_length_no_special(d)
            if total + now_len + (separator_len if len(current_doc) > 0 else 0) > chunk_size:
                if total > chunk_size:
                    self.logger.info(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {chunk_size}")
                    if len(current_doc) == 1:  # a single chunk is too long
                        split_again = self.split_into_chunks(
                            current_doc[0], chunk_size, self.split_sen_pattern, task_name)
                        docs.extend(split_again)
                        current_doc = []
                        total = 0

                if len(current_doc) > 0:
                    docs.append("".join(current_doc))
                    # Keep popping until the overlap requirement is met and
                    # the current chunk no longer overflows.
                    while (
                        total > chunk_overlap
                        or (
                            total + now_len + (separator_len if len(current_doc) > 0 else 0) > chunk_size
                            and total > 0
                        )
                    ):
                        total -= self.get_prompt_length_no_special(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]

            current_doc.append(d)
            total += now_len + (separator_len if len(current_doc) > 1 else 0)

        # Handle the final piece: split it again if it is a single overlong chunk.
        if len(current_doc) == 1 and self.get_prompt_length_no_special(current_doc[-1]) > chunk_size:
            split_again = self.split_into_chunks(
                current_doc[0], chunk_size, self.split_sen_pattern, task_name)
            docs.extend(split_again)
        else:
            docs.append("".join(current_doc))

        return [d for d in docs if d.strip() != ""]

    # ------------------------------------------------------------------ #
    # Token counting                                                     #
    # ------------------------------------------------------------------ #

    def get_prompt_length(self, prompt, **kwargs: Any) -> int:
        if isinstance(prompt, list):
            prompt = self.join_docs(prompt)
        return len(self.tokenizer.encode(prompt, **kwargs))

    def get_prompt_length_no_special(self, prompt, **kwargs: Any) -> int:
        if isinstance(prompt, list):
            prompt = self.join_docs(prompt)
        if not isinstance(self.tokenizer, tiktoken.core.Encoding):
            return len(self.tokenizer.encode(prompt, add_special_tokens=False, **kwargs))
        return len(self.tokenizer.encode(prompt, disallowed_special="all", **kwargs))

    def join_docs(self, docs: List[str]) -> str:
        if isinstance(docs, str):
            return docs
        return "\n\n".join(docs)
