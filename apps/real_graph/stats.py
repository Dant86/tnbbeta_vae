"""Exact paired tests and intervals of the real-graph revision (pre-registered).

The paired sign-flip test enumerates all ``2^n`` sign patterns of the per-seed
differences (``n = 10`` seeds: 1024 patterns), so its p-values are exact; the tests
check it against an independent enumeration.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from apps.eval.graph_search_stats import benjamini_hochberg, holm

__all__ = [
    "benjamini_hochberg",
    "bootstrap_mean_ci",
    "holm",
    "sign_flip_p",
    "within_block_permutation_p",
]

Array = npt.NDArray[np.float64]
_TOLERANCE = 1e-12
BOOTSTRAP_SEED = 20261104
BOOTSTRAP_RESAMPLES = 10_000


def sign_flip_p(
    differences: Array, *, margin: float = 0.0, alternative: str = "two-sided"
) -> float:
    """Exact paired sign-flip test of ``mean(differences - margin)`` against 0.

    Args:
        differences: Per-seed paired differences (one per seed).
        margin: Shift subtracted from every difference before testing.
        alternative: ``"two-sided"`` (statistic ``|mean|``) or ``"greater"`` (the mean
            is larger than under random signs).

    Returns:
        The fraction of the ``2^n`` sign patterns whose statistic is at least the
        observed one (the observed pattern included, so the minimum is ``2^-n``).
    """
    values = np.asarray(differences, dtype=np.float64) - margin
    count = values.shape[0]
    patterns = (np.arange(2**count)[:, None] >> np.arange(count)) & 1
    signs = patterns * 2.0 - 1.0
    flipped = (signs * values).mean(axis=1)
    observed = values.mean()
    if alternative == "two-sided":
        return float(np.mean(np.abs(flipped) >= abs(observed) - _TOLERANCE))
    if alternative == "greater":
        return float(np.mean(flipped >= observed - _TOLERANCE))
    raise ValueError(f"Unknown alternative {alternative!r}.")


def bootstrap_mean_ci(
    values: Array,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval of a mean over the rows of ``values``.

    Args:
        values: Array whose first axis is resampled (per-seed values).
        resamples: Number of bootstrap resamples.
        seed: Generator seed.
        level: Coverage.

    Returns:
        ``(low, high)`` of the mean.
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(values.shape[0], size=(resamples, values.shape[0]))
    means = values[index].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def within_block_permutation_p(
    outcomes: list[Array],
    covariates: list[Array],
    *,
    permutations: int = 100_000,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, list[float]]:
    """One-sided permutation test of the mean within-block Spearman correlation.

    Each block (a dataset) is an array ``(num_seeds, num_levels)`` of outcomes with a
    matching array of covariates (the realised density). The statistic is the mean over
    blocks of the Spearman correlation of the flattened outcomes with the covariates.
    The null permutes the outcomes across the levels within each seed row (so seed
    effects are preserved).

    Args:
        outcomes: One ``(num_seeds, num_levels)`` array per block.
        covariates: Matching covariate arrays.
        permutations: Number of Monte Carlo permutations.
        seed: Generator seed.

    Returns:
        The observed combined statistic, the one-sided p-value (statistic at least as
        large; the observed value counts as one permutation) and the per-block
        Spearman correlations.
    """
    from scipy.stats import rankdata  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    ranked = [rankdata(c.ravel()) for c in covariates]

    def spearman(rank_x: Array, y: Array) -> float:
        rank_y = rankdata(y.ravel())
        if np.std(rank_x) == 0 or np.std(rank_y) == 0:
            return 0.0
        return float(np.corrcoef(rank_x, rank_y)[0, 1])

    per_block = [spearman(r, y) for r, y in zip(ranked, outcomes, strict=True)]
    observed = float(np.mean(per_block))
    exceed = 1
    for _ in range(permutations):
        shuffled = []
        for y in outcomes:
            order = np.argsort(rng.random(y.shape), axis=1)
            shuffled.append(np.take_along_axis(y, order, axis=1))
        statistic = np.mean(
            [spearman(r, y) for r, y in zip(ranked, shuffled, strict=True)]
        )
        exceed += int(statistic >= observed - _TOLERANCE)
    return observed, exceed / (permutations + 1), per_block
