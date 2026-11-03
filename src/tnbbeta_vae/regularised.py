"""Small elastic-net and grouped cross-validation utilities (numpy only).

Used by the metric-first graph-search analysis. The elastic net minimises
``(1 / 2n) ||y - X b - b0||^2 + lambda (alpha ||b||_1 + (1 - alpha) / 2 ||b||_2^2)`` on
standardised columns by cyclic coordinate descent with warm starts along a decreasing
``lambda`` path. Standardisation uses the training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "ElasticNetFit",
    "block_permutation",
    "fit_elastic_net",
    "leave_family_out",
    "select_lambda",
]

Array = npt.NDArray[np.float64]


@dataclass
class ElasticNetFit:
    """A fitted elastic net on standardised columns.

    Attributes:
        mean: Training column means.
        scale: Training column standard deviations (1 where constant).
        coefficients: Coefficients on the standardised columns.
        intercept: Intercept (the training outcome mean).
        lam: The penalty used.
    """

    mean: Array
    scale: Array
    coefficients: Array
    intercept: float
    lam: float

    def predict(self, x: Array) -> Array:
        """Predicts the outcome for raw (unstandardised) covariates ``x``."""
        return self.intercept + ((x - self.mean) / self.scale) @ self.coefficients


def lambda_grid(x: Array, y: Array, alpha: float, count: int = 12) -> Array:
    """A log-spaced decreasing grid from the smallest all-zero penalty down by 1e-3."""
    z = _standardise(x)[0]
    top = float(np.max(np.abs(z.T @ (y - y.mean()))) / (len(y) * alpha))
    top = max(top, 1e-8)
    return top * np.logspace(0, -2.5, count)


def fit_elastic_net(
    x: Array, y: Array, lams: Array, *, alpha: float = 0.5, iterations: int = 200
) -> list[ElasticNetFit]:
    """Fits the whole penalty path (decreasing ``lams``) with warm starts.

    Args:
        x: Raw covariates, shape ``(n, p)``.
        y: Outcome, shape ``(n,)``.
        lams: Decreasing penalties.
        alpha: L1 share of the penalty.
        iterations: Maximum coordinate-descent sweeps per penalty.

    Returns:
        One :class:`ElasticNetFit` per penalty.
    """
    z, mean, scale = _standardise(x)
    n, p = z.shape
    gram = z.T @ z / n
    moment = z.T @ (y - y.mean()) / n
    diagonal = np.diag(gram).copy()
    beta = np.zeros(p)
    fits = []
    for lam in lams:
        for _ in range(iterations):
            largest = 0.0
            for j in range(p):
                old = beta[j]
                rho = moment[j] - gram[j] @ beta + diagonal[j] * old
                new = np.sign(rho) * max(abs(rho) - lam * alpha, 0.0)
                new /= diagonal[j] + lam * (1 - alpha)
                if new != old:
                    beta[j] = new
                    largest = max(largest, abs(new - old))
            if largest < 1e-6:
                break
        fits.append(
            ElasticNetFit(mean, scale, beta.copy(), float(y.mean()), float(lam))
        )
    return fits


def select_lambda(
    x: Array,
    y: Array,
    groups: npt.NDArray[Any],
    *,
    alpha: float = 0.5,
    rule: str = "one_se",
    count: int = 12,
) -> float:
    """Chooses the penalty by leave-one-group-out CV (``one_se`` or ``min`` MSE)."""
    lams = lambda_grid(x, y, alpha, count)
    labels = np.unique(groups)
    errors = np.zeros((labels.size, lams.size))
    for row, label in enumerate(labels):
        test = groups == label
        fits = fit_elastic_net(x[~test], y[~test], lams, alpha=alpha)
        for col, fit in enumerate(fits):
            errors[row, col] = float(np.mean((y[test] - fit.predict(x[test])) ** 2))
    mean = errors.mean(0)
    best = int(np.argmin(mean))
    if rule == "min" or labels.size < 2:
        return float(lams[best])
    standard_error = errors.std(0, ddof=1)[best] / np.sqrt(labels.size)
    eligible = np.flatnonzero(mean <= mean[best] + standard_error)
    return float(lams[eligible.min()])  # the largest penalty within one SE


def leave_family_out(
    x: Array,
    y: Array,
    groups: npt.NDArray[Any],
    *,
    alpha: float = 0.5,
) -> tuple[float, Array, list[ElasticNetFit]]:
    """Outer leave-one-group-out CV with the penalty chosen on the training groups.

    Returns:
        ``(r2, predictions, fits)``: ``1 - SSE_model / SSE_base`` where the baseline
        predicts the mean outcome of the training groups, the held-out predictions in
        row order, and the fit of each outer fold (in sorted group order).
    """
    labels = np.unique(groups)
    predictions = np.zeros_like(y)
    baseline = np.zeros_like(y)
    fits = []
    for label in labels:
        test = groups == label
        lam = select_lambda(x[~test], y[~test], groups[~test], alpha=alpha)
        fit = fit_elastic_net(x[~test], y[~test], np.array([lam]), alpha=alpha)[0]
        fits.append(fit)
        predictions[test] = fit.predict(x[test])
        baseline[test] = y[~test].mean()
    sse_model = float(((y - predictions) ** 2).sum())
    sse_base = float(((y - baseline) ** 2).sum())
    return 1.0 - sse_model / sse_base, predictions, fits


def block_permutation(
    groups: npt.NDArray[Any], rng: np.random.Generator
) -> npt.NDArray[Any]:
    """Returns row indices that reassign each equal-sized group's outcomes to another.

    Requires every group to have the same size. Groups are permuted as blocks and the
    order within each block is shuffled; applying the result to ``y`` keeps both the
    covariates' and the outcomes' group clustering.
    """
    labels = np.unique(groups)
    rows = [np.flatnonzero(groups == label) for label in labels]
    size = rows[0].size
    if any(r.size != size for r in rows):
        raise ValueError("block permutation needs equally sized groups")
    order = rng.permutation(len(labels))
    permuted = np.empty(groups.size, dtype=np.int64)
    for target, source in zip(range(len(labels)), order, strict=True):
        permuted[rows[target]] = rng.permutation(rows[source])
    return permuted


def _standardise(x: Array) -> tuple[Array, Array, Array]:
    mean = x.mean(0)
    scale = x.std(0)
    scale = np.where(scale > 0, scale, 1.0)
    return (x - mean) / scale, mean, scale
