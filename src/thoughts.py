#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attention-score analysis and visualization for InfiniComp.

:class:`Thoughts` aggregates the attention scores produced by the model during
retrieval, extracts the token positions the query attends to most strongly,
and optionally renders attention heatmaps (saved as PNG files) for
inspection.
"""

import time
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d


class Thoughts:
    """Attention-score analysis and heatmap visualization."""

    def __init__(self, config: Dict, logger) -> None:
        self.config = config
        self.logger = logger
        self.cmap = config.get("cmap", "Reds")
        self.plt_save = config.get("plt_save", None)
        self.mark_position = config.get("mark_position", False)
        self.malplot_all_init()

    # ------------------------------------------------------------------ #
    # State management                                                   #
    # ------------------------------------------------------------------ #

    def malplot_all_init(self) -> None:
        """Reset the accumulators used by :meth:`malplot_all_attention`."""
        self.all_attention_weights = []
        self.all_x_labels = []
        self.all_y_labels = []
        self.all_focus_indexs = []

    # ------------------------------------------------------------------ #
    # Score computation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def softmax(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax over ``axis``."""
        exp_matrix = np.exp(matrix - np.max(matrix, axis=axis, keepdims=True))
        return exp_matrix / np.sum(exp_matrix, axis=axis, keepdims=True)

    @staticmethod
    def same_avgpool1d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Average pooling that preserves the input length ("same" padding)."""
        padding_left = (kernel_size - 1) // 2
        padding_right = kernel_size - 1 - padding_left
        x = F.pad(x, (padding_left, padding_right))
        pool = nn.AvgPool1d(kernel_size, stride=1, padding=0, ceil_mode=True)
        return pool(x)

    def compute_phrase_scores_token_numpy(
            self, PREFIX_PROMPT_TOKEN_NUM: int, attentions,
            context_indexs, question_indexs, first_q_w, topk: int,
            phrase_token_num: int = 5):
        """Pool question->context attention scores (numpy implementation).

        Attention weights are summed over heads, smoothed over
        ``phrase_token_num`` neighboring key positions with a convolution
        kernel, then summed over query tokens. The top-``topk`` key tokens are
        returned as the focused token indices.
        """
        select_attention = attentions[-1]                     # (1, head_num, len, len)
        attention_weights = select_attention[0].sum(axis=0)   # (len, len)
        attention_weights = attention_weights.float().cpu().detach().numpy()

        context_col_select = np.r_[context_indexs]
        question_row_select = np.r_[question_indexs]
        question2context = np.copy(
            attention_weights[question_row_select, :][:, context_col_select])

        # Smooth each position with its ``phrase_token_num`` neighbors.
        kernel = np.ones((1, phrase_token_num))
        question2context = convolve2d(
            question2context, kernel, mode="same", boundary="fill", fillvalue=0)

        token_scores = np.sum(question2context, axis=0)       # (context len)
        topk_token_indices = np.argsort(token_scores[0:])[-topk:] + PREFIX_PROMPT_TOKEN_NUM
        return topk_token_indices.tolist(), attention_weights, None

    def compute_phrase_scores_token(
            self, PREFIX_PROMPT_TOKEN_NUM: int, attentions,
            context_indexs, question_indexs, layer: int = -1,
            topk: int = None, phrase_token_num: int = 5):
        """Pool question->context attention scores (torch implementation).

        Same idea as :meth:`compute_phrase_scores_token_numpy` but computed
        on the GPU with an average-pooling kernel instead of a convolution.
        """
        select_attention = attentions[layer]                  # (1, head_num, len, len)
        attention_scores = select_attention[0].sum(axis=0)    # (len, len)
        question2all = attention_scores[question_indexs, :]
        question2context = question2all[:, context_indexs]
        compute_scores = self.same_avgpool1d(
            question2context, kernel_size=phrase_token_num)
        assert compute_scores.shape == question2context.shape, \
            "after pooling, the dim must equal original."
        token_scores = compute_scores.sum(axis=0)             # (context len)

        topk = min(topk, token_scores.shape[0])
        topk_token_indices = token_scores.topk(topk).indices + PREFIX_PROMPT_TOKEN_NUM
        return topk_token_indices.tolist(), attention_scores, None

    def get_focus_index(
            self, process_info: Dict, output_attentions,
            topk_value: int = None, layer: int = -1, first_q_w=None,
            phrase_token_num: int = 5):
        """Compute the focused token indices from the attention scores.

        Args:
            process_info: Pre-processing metadata produced by ``Retrieval``,
                containing the question/context start and end token indices.
            output_attentions: Attention tensors (tuple-like); only element
                ``layer`` is used, so a single-layer tuple suffices.
            topk_value: Number of focused tokens to keep.
            layer: Which layer's attention to use.
            first_q_w: Optional precomputed question weights.
            phrase_token_num: Kernel size of the pooling/smoothing.

        Returns:
            Tuple of ``(focus_indexs, attention_weights, question_weights)``.
        """
        context_indexs = list(range(process_info["context_start_index"],
                                    process_info["context_end_index"]))
        question_indexs = list(range(process_info["question_start_index"],
                                     process_info["question_end_index"]))

        return self.compute_phrase_scores_token(
            PREFIX_PROMPT_TOKEN_NUM=process_info["context_start_index"],
            attentions=output_attentions,
            context_indexs=context_indexs,
            question_indexs=question_indexs,
            layer=layer,
            topk=topk_value,
            phrase_token_num=phrase_token_num,
        )

    # ------------------------------------------------------------------ #
    # Heatmap visualization                                              #
    # ------------------------------------------------------------------ #
    def malplot_attention(self, attention_weights: np.ndarray,
                          x_labels: List[str] = None, y_labels: List[str] = None,
                          focus_indexs: List[int] = None, save_name: str = None,
                          pic_size: int = 358) -> None:
        """Render a single attention heatmap and optionally save it."""
        row_size, col_size = attention_weights.shape[0], attention_weights.shape[1]
        aspect = col_size / row_size
        plt.figure(figsize=(pic_size, pic_size / aspect))

        if x_labels is None and y_labels is None:
            heatmap = sns.heatmap(attention_weights, fmt=".2f", cmap=self.cmap)
        else:
            heatmap = sns.heatmap(attention_weights, xticklabels=x_labels,
                                  yticklabels=y_labels, fmt=".2f", cmap=self.cmap)

        # Mark the position of each focused token index on the heatmap.
        if focus_indexs is not None:
            for focus_index in focus_indexs:
                # (left, bottom), width, height of the rectangle.
                rect = Rectangle((focus_index, 0), 1, row_size,
                                 fill=False, edgecolor="g", linewidth=1)
                heatmap.add_artist(rect)

        plt.title("Attention Weights Heatmap")
        plt.xlabel("Key Token")
        plt.ylabel("Query Token")

        if self.plt_save and save_name:
            save_path = self.plt_save + "/" + save_name
            heatmap.get_figure().savefig(save_path)
            self.logger.info("Save %s successfully!" % save_path)

    def add_malplot_attention(self, attention_weights: np.ndarray,
                              x_labels: List[str], y_labels: List[str],
                              focus_indexs: List[int]) -> None:
        """Accumulate one heatmap entry to be rendered all at once."""
        self.all_attention_weights.append(attention_weights)
        self.all_x_labels.append(x_labels)
        self.all_y_labels.append(y_labels)
        self.all_focus_indexs.append(focus_indexs)

    def malplot_all_attention(self, save_name: str, max_width: int = 598,
                              max_height: int = 598,
                              page_max_num: int = 598) -> None:
        """Render all accumulated heatmaps into paginated PNG files."""
        attention_num = len(self.all_attention_weights)
        page_num = (attention_num - 1) // page_max_num + 1

        for page in range(page_num):
            t1 = time.time()
            start_iter = page * page_max_num
            end_iter = min((page + 1) * page_max_num, attention_num)
            self.logger.info(
                "Starting to generate attention heatmap from %d to %d iteration......"
                % (start_iter, end_iter))

            data_matrices = self.all_attention_weights[start_iter:end_iter]
            max_cols = max(data.shape[1] for data in data_matrices)
            total_rows = sum(data.shape[0] for data in data_matrices)

            fig_width = min(max_cols, max_width)
            fig_height = min(total_rows, max_height)
            fig = plt.figure(figsize=(fig_width / 5, fig_height / 5))

            gs = gridspec.GridSpec(len(data_matrices), 1)
            for index, data in enumerate(data_matrices):
                ax = fig.add_subplot(gs[index])
                rows, cols = data.shape
                aspect = cols / rows
                ax.set_box_aspect(1 / aspect)
                heatmap = sns.heatmap(
                    data, ax=ax, cbar=False, square=True, cmap=self.cmap,
                    xticklabels=self.all_x_labels[start_iter + index],
                    yticklabels=self.all_y_labels[start_iter + index],
                )
                ax.set_title("Attention Weights Heatmap at %d Iteration"
                             % (start_iter + index + 1), fontsize=25)
                ax.set_xlabel("Key Token", fontsize=15)
                ax.set_ylabel("Query Token", fontsize=15)
                ax.tick_params(axis="both", which="major", labelsize=5)

                # Mark the position of each focused token index on the heatmap.
                if self.mark_position and self.all_focus_indexs is not None:
                    for focus_index in self.all_focus_indexs[start_iter + index]:
                        # (left, bottom), width, height of the rectangle.
                        rect = Rectangle((focus_index, 0), 1, rows,
                                         fill=False, edgecolor="g", linewidth=1)
                        heatmap.add_artist(rect)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if self.plt_save and save_name:
                save_path = (self.plt_save + "/" + save_name
                             + "attention_heatmap_page%d.png" % (page + 1))
                plt.savefig(save_path, bbox_inches="tight", dpi=258)
                self.logger.info("Cost %fs to geneate all attention heatmap save in %s"
                                 % (time.time() - t1, save_path))


# Backward-compatible alias (the original module exposed the lowercase name).
thoughts = Thoughts
