#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared text-splitting utilities for InfiniComp.

These helpers split raw text into sentence-level units. Both :class:`Chunks`
(which chunks long contexts) and :class:`Retrieval` (which must locate the
question / context segments back inside the token stream) rely on exactly the
same splitting rules, so the implementations live here to keep them perfectly
consistent with each other.
"""

import re
from typing import List

# Tasks whose raw input is a sequence of few-shot examples. They are split by
# a fixed marker word instead of punctuation so every example stays complete.
FEW_SHOT_SPLIT_WORDS = {
    "trec": "Question:",
    "triviaqa": "Passage:",
    "samsum": "Dialogue:",
}

# Symbols that are treated as sentence terminators when re-joining the pieces
# produced by :func:`re.split` (the punctuation itself is kept with the
# preceding sentence).
_SENTENCE_PUNCTUATION = ("\n\n", "。", "！", "？", ".", "?", "!")


def split_sentence(text: str, spliter: str) -> List[str]:
    """Split ``text`` by the punctuation captured in ``spliter`` and keep the
    punctuation attached to the preceding sentence.

    Args:
        text: Original input text.
        spliter: Regular-expression pattern of the split punctuation.

    Returns:
        A list of continuous sentences.
    """
    pieces = list(filter(None, re.split(spliter, text)))

    # Splitting by a single space: re-attach the space, trim the tail.
    if spliter == " ":
        sentences = [piece + " " for piece in pieces if piece != ""]
        sentences[-1] = sentences[-1].strip()
        return sentences

    sentences = []
    current = ""
    for piece in pieces:
        # Re-join the captured punctuation so the original formatting is
        # preserved as much as possible. A very short piece following a '.'
        # (e.g. initials such as "J." in a name) is merged back instead of
        # starting a new sentence.
        if (
            len(current.strip()) == 0
            or len(piece.strip()) == 0
            or piece.strip() in _SENTENCE_PUNCTUATION
            or (current[-1] == "." and len(piece.strip()) <= 8)
        ):
            current += piece
        else:
            sentences.append(current)
            current = piece
    if len(current.strip()) > 0:
        sentences.append(current)
    return sentences


def split_text_in_few_shot(text: str, split_word: str) -> List[str]:
    """Split a few-shot text by ``split_word``, keeping the marker at the
    head of every piece except the first."""
    split_texts = text.split(split_word)
    split_list = [split_texts[0]]
    for piece in split_texts[1:]:
        split_list.append(split_word + piece)
    return split_list


def split_text_unit(text: str, spliter: str, task_name: str) -> List[str]:
    """Split text into sentence units, dispatching on the task type.

    Few-shot tasks (e.g. ``trec`` / ``triviaqa`` / ``samsum``) are split by
    their marker word; all other tasks fall back to punctuation splitting.
    """
    if task_name in FEW_SHOT_SPLIT_WORDS:
        return split_text_in_few_shot(text, FEW_SHOT_SPLIT_WORDS[task_name])
    return split_sentence(text, spliter)
