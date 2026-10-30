"""Per-bucket link-prediction scoring, for decomposing aggregate AUC by node type.

A test edge ``(i, j)`` counts toward the bucket of *both* endpoints, so each node's
held-out positives and sampled negatives pool within its bucket. Per-node AUC is far
too noisy at citation-graph degrees (Cora averages about 4), which is why the unit of
analysis is the bucket, not the node.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tnbbeta_vae.models.losses.ranking import average_precision, roc_auc

__all__ = ["bucketed_link_scores", "quantile_buckets"]


def quantile_buckets(
    values: npt.NDArray[np.float64], num_buckets: int
) -> npt.NDArray[np.int64]:
    """Assigns each value to one of ``num_buckets`` roughly equal-sized bins.

    Ties are broken by ranking, so heavy ties (e.g. many nodes of degree 1) still
    spread across buckets, but equal values may land in different buckets.

    Args:
        values: One value per node.
        num_buckets: Number of bins.

    Returns:
        Bucket index in ``[0, num_buckets)`` per node, ordered by value.
    """
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return (ranks * num_buckets // len(values)).astype(np.int64)


def bucketed_link_scores(
    embeddings: npt.NDArray[np.float64],
    positive: npt.NDArray[np.int64],
    negative: npt.NDArray[np.int64],
    buckets: npt.NDArray[np.int64],
) -> dict[int, dict[str, float]]:
    """Scores held-out edges by inner product, pooled within each endpoint's bucket.

    Args:
        embeddings: Node embeddings, shape ``(n, d)``.
        positive: Held-out edges, shape ``(2, n_pos)``.
        negative: Sampled non-edges, shape ``(2, n_neg)``.
        buckets: Bucket index per node, shape ``(n,)``; negative values mean
            "unbucketed" and are skipped.

    Returns:
        For each bucket with at least one positive and one negative: ``auc``, ``ap``
        and the incidence counts ``n_pos`` / ``n_neg``.
    """
    positive_scores = (embeddings[positive[0]] * embeddings[positive[1]]).sum(-1)
    negative_scores = (embeddings[negative[0]] * embeddings[negative[1]]).sum(-1)
    results: dict[int, dict[str, float]] = {}
    for bucket in sorted(int(b) for b in np.unique(buckets) if b >= 0):
        in_bucket = buckets == bucket
        positive_mask = in_bucket[positive[0]] | in_bucket[positive[1]]
        negative_mask = in_bucket[negative[0]] | in_bucket[negative[1]]
        # An edge with both endpoints in the bucket is one incidence, not two; edges
        # spanning two buckets appear once in each.
        if not positive_mask.any() or not negative_mask.any():
            continue
        pos, neg = positive_scores[positive_mask], negative_scores[negative_mask]
        results[bucket] = {
            "auc": roc_auc(pos, neg),
            "ap": average_precision(pos, neg),
            "n_pos": float(len(pos)),
            "n_neg": float(len(neg)),
        }
    return results
