"""The single-axis latitude-shell task at any ambient dimension, with the gentle map.

``x = h(a . z) + N(0, sigma^2 I)``, ``z ~ Uniform(S^(d-1))``, ``a = e_1`` and
``h(w) = [w, sin 1.5 w, cos 2 w]`` (the map of :mod:`tnbbeta_vae.data.ring_classes`).
Only the feature map differs from :class:`~tnbbeta_vae.data.ring_posterior.
RingPosteriorTask`, so its exact evidence, exact posterior and Monte Carlo KL all apply.
The latitude density is ``proportional to N(x; h(w), sigma^2) (1 - w^2)^((d-3)/2)``;
the quadrature runs in the polar angle ``t = arccos w`` (weight ``sin^(d-2) t``), where
the integrand is smooth even for ``d = 3, 5`` (no endpoint singularity).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from tnbbeta_vae.data.ring_posterior import (
    RingPosteriorTask,
    kl_uniform_to_posterior,
)

__all__ = [
    "DIMENSIONS",
    "SIGMA",
    "DimensionRingTask",
    "dimension_ring_task",
    "information_summary",
]

DIMENSIONS = (3, 5, 9, 17)
SIGMA = 0.2
_DEFAULT_GRID = 16001
_CHUNK = 500


@dataclass(frozen=True)
class DimensionRingTask(RingPosteriorTask):
    """:class:`RingPosteriorTask` with ``h(w) = [w, sin 1.5 w, cos 2 w]``."""

    def features(self, w: Tensor) -> Tensor:
        """Latitudes ``w`` to ``h(w)``; shape ``(*w.shape, 3)``."""
        return torch.stack([w, torch.sin(1.5 * w), torch.cos(2.0 * w)], dim=-1)


def dimension_ring_task(
    dim: int, sigma: float = SIGMA, grid_size: int = _DEFAULT_GRID
) -> DimensionRingTask:
    """The task at ambient dimension ``dim`` (axis ``e_1``).

    Args:
        dim: Ambient dimension, at least 3.
        sigma: Observation noise standard deviation.
        grid_size: Quadrature nodes over the polar angle.

    Returns:
        The task.

    Raises:
        ValueError: If ``dim < 3``.
    """
    if dim < 3:
        raise ValueError(f"dimension_ring_task needs dim >= 3; got {dim}.")
    axis = torch.zeros(dim)
    axis[0] = 1.0
    return DimensionRingTask(dim, sigma, axis, grid_size)


def information_summary(
    task: DimensionRingTask, num: int = 20_000, seed: int = 7
) -> dict[str, float]:
    """How informative ``x`` is about ``z``: ``I(z; x)`` and the uniform baseline KL.

    Args:
        task: The task.
        num: Monte Carlo draws of ``(x, z)``.
        seed: RNG seed.

    Returns:
        ``mutual_information`` (``E[log p(x|z) - log p(x)]``, nats),
        ``kl_uniform_to_posterior`` (``E_x KL(Uniform || p(. | x))``),
        ``posterior_w_std`` (mean over ``x`` of the std of ``w`` under the exact
        posterior)
        and ``prior_w_std`` (``1/sqrt(dim)``).
    """
    generator = torch.Generator().manual_seed(seed)
    x, z = task.sample(num, generator)
    x, z = x.double(), z.double()
    chunks = range(0, num, _CHUNK)
    mutual = torch.cat(
        [
            task.log_likelihood(x[i : i + _CHUNK], z[i : i + _CHUNK])
            - task.log_evidence(x[i : i + _CHUNK])
            for i in chunks
        ]
    ).mean()
    kl_uniform = torch.cat(
        [kl_uniform_to_posterior(task, x[i : i + _CHUNK]) for i in chunks]
    ).mean()
    theta, density = task.latitude_density(x[:2000])
    w = torch.cos(theta)
    mass = density * (math.pi / task.grid_size)
    mean_w = (mass * w).sum(-1)
    std_w = ((mass * w**2).sum(-1) - mean_w**2).clamp_min(0).sqrt().mean()
    return {
        "mutual_information": float(mutual),
        "kl_uniform_to_posterior": float(kl_uniform),
        "posterior_w_std": float(std_w),
        "prior_w_std": 1.0 / math.sqrt(task.dim),
    }
