"""Ranking metrics for link prediction: ROC AUC and average precision."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

__all__ = ["average_precision", "roc_auc"]


def roc_auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """Returns the area under the ROC curve (ties count half).

    Args:
        positive_scores: Scores of the positive examples.
        negative_scores: Scores of the negative examples.

    Returns:
        The probability that a random positive outscores a random negative.
    """
    scores = np.concatenate([positive_scores, negative_scores])
    ranks = rankdata(scores)
    num_positive, num_negative = len(positive_scores), len(negative_scores)
    rank_sum = ranks[:num_positive].sum()
    return float(
        (rank_sum - num_positive * (num_positive + 1) / 2)
        / (num_positive * num_negative)
    )


def average_precision(
    positive_scores: np.ndarray, negative_scores: np.ndarray
) -> float:
    """Returns the average precision: mean precision at each positive's rank.

    Args:
        positive_scores: Scores of the positive examples.
        negative_scores: Scores of the negative examples.

    Returns:
        Average precision for ranking by descending score.
    """
    scores = np.concatenate([positive_scores, negative_scores])
    labels = np.concatenate(
        [np.ones(len(positive_scores)), np.zeros(len(negative_scores))]
    )
    order = np.argsort(-scores, kind="stable")
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())
