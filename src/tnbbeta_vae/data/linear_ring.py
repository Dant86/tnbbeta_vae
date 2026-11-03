"""Sphere probabilistic PCA: ``y = a . z + noise`` with a scalar ``y``.

``z ~ Uniform(S^(d-1))`` and ``y = h(a . z) + N(0, sigma^2)`` for a fixed unit axis
``a`` (``h`` is the identity for the linear task). With the latitude ``w = a . z`` the
exact posterior has density, with respect to the sphere's surface measure, proportional
to ``N(y; h(w), sigma^2)``: a latitude ring about ``a`` (azimuth uniform on
``S^(d-2)``). The latitude marginal is ``N(y; h(w), sigma^2) (1 - w^2)^((d-3)/2)`` on
``[-1, 1]``; the quadrature here runs in the polar angle ``t = arccos w`` (weight
``sin^(d-2) t``), where the integrand is smooth even for ``d = 3, 5``. The same routines
evaluate a *model's* evidence and ring posterior from its own mean function ``h`` and
noise scale (a learned linear or 1-D decoder), because that decoder's posterior is also
a function of one latitude only.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor

from tnbbeta_vae.data.ring_posterior import latitude_grid

__all__ = [
    "DEFAULT_GRID",
    "information_nats",
    "kl_uniform_to_posterior",
    "latitude_std",
    "log_evidence",
    "sample_observations",
]

DEFAULT_GRID = 16001

MeanFn = Callable[[Tensor], Tensor]


def sample_observations(
    dim: int, sigma: float, num: int, generator: torch.Generator | None = None
) -> Tensor:
    """Draws ``num`` scalar observations ``y = z_1 + N(0, sigma^2)`` (axis ``e_1``).

    Args:
        dim: Ambient dimension of the sphere, at least 3.
        sigma: Observation noise standard deviation.
        num: Number of draws.
        generator: Optional RNG.

    Returns:
        ``y`` of shape ``(num,)``.

    Raises:
        ValueError: If ``dim < 3``.
    """
    if dim < 3:
        raise ValueError(f"needs dim >= 3; got {dim}.")
    z = torch.randn(num, dim, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    return z[:, 0] + sigma * torch.randn(num, generator=generator)


def log_evidence(
    y: Tensor,
    dim: int,
    sigma: float | Tensor,
    mean_fn: MeanFn | None = None,
    grid_size: int = DEFAULT_GRID,
) -> Tensor:
    """Exact ``log p(y) = log E_{z ~ Uniform}[N(y; h(a . z), sigma^2)]``.

    Args:
        y: Observations, shape ``(batch,)``.
        dim: Ambient dimension.
        sigma: Noise standard deviation (the truth, or a model's learned value).
        mean_fn: ``h``, mapping latitudes of shape ``(grid,)`` to means of the same
            shape (default: identity).
        grid_size: Midpoint nodes over the polar angle.

    Returns:
        Tensor of shape ``(batch,)``.
    """
    energy, log_uniform = _energy_and_prior(y, dim, sigma, mean_fn, grid_size)
    constant = -0.5 * math.log(2 * math.pi) - torch.log(torch.as_tensor(sigma))
    return constant + torch.logsumexp(-energy + log_uniform, dim=-1)


def latitude_std(
    y: Tensor,
    dim: int,
    sigma: float | Tensor,
    mean_fn: MeanFn | None = None,
    grid_size: int = DEFAULT_GRID,
) -> Tensor:
    """Standard deviation of the latitude ``w`` under the exact posterior ``p(. | y)``.

    Args:
        y: Observations, shape ``(batch,)``.
        dim: Ambient dimension.
        sigma: Noise standard deviation.
        mean_fn: ``h`` (default: identity).
        grid_size: Midpoint nodes over the polar angle.

    Returns:
        Tensor of shape ``(batch,)``.
    """
    energy, log_uniform = _energy_and_prior(y, dim, sigma, mean_fn, grid_size)
    log_mass = -energy + log_uniform
    mass = torch.exp(log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True))
    theta, _ = latitude_grid(grid_size, y.dtype)
    w = torch.cos(theta)
    mean = (mass * w).sum(-1)
    return ((mass * w**2).sum(-1) - mean**2).clamp_min(0).sqrt()


def information_nats(
    y: Tensor, dim: int, sigma: float, grid_size: int = DEFAULT_GRID
) -> float:
    """``I(z; y) = -E log p(y) - (1/2) log(2 pi e sigma^2)`` on the sample ``y``.

    Args:
        y: Observations drawn from the task, shape ``(num,)``.
        dim: Ambient dimension.
        sigma: True noise standard deviation.
        grid_size: Midpoint nodes over the polar angle.

    Returns:
        The mutual information in nats (the gap of a model whose latent is ignored).
    """
    entropy = -log_evidence(y, dim, sigma, None, grid_size).mean()
    return float(entropy - 0.5 * math.log(2 * math.pi * math.e * sigma**2))


def kl_uniform_to_posterior(
    y: Tensor,
    dim: int,
    sigma: float | Tensor,
    mean_fn: MeanFn | None = None,
    grid_size: int = DEFAULT_GRID,
) -> Tensor:
    """Exact ``KL(Uniform || p(. | y))`` per observation (the "ignore y" baseline).

    The ELBO gap of a model with the exact decoder whose posterior is the prior is
    ``E_y`` of this quantity (it is *not* ``I(z; y)``).

    Args:
        y: Observations, shape ``(batch,)``.
        dim: Ambient dimension.
        sigma: Noise standard deviation.
        mean_fn: ``h`` (default: identity).
        grid_size: Midpoint nodes over the polar angle.

    Returns:
        Tensor of shape ``(batch,)``.
    """
    energy, log_uniform = _energy_and_prior(y, dim, sigma, mean_fn, grid_size)
    expected_energy = (energy * log_uniform.exp()).sum(-1)
    return expected_energy + torch.logsumexp(-energy + log_uniform, dim=-1)


def _energy_and_prior(
    y: Tensor,
    dim: int,
    sigma: float | Tensor,
    mean_fn: MeanFn | None,
    grid_size: int,
) -> tuple[Tensor, Tensor]:
    """Energies ``(y - h(w))^2 / (2 sigma^2)`` and the latitude prior on the grid."""
    theta, delta = latitude_grid(grid_size, y.dtype)
    log_weight = (dim - 2) * torch.log(torch.sin(theta)) + math.log(delta)
    log_uniform = log_weight - torch.logsumexp(log_weight, dim=0)
    w = torch.cos(theta)
    mean = w if mean_fn is None else mean_fn(w)
    sigma_t = torch.as_tensor(sigma, dtype=y.dtype)
    energy = ((y[:, None] - mean[None]) / sigma_t).pow(2) / 2
    return energy, log_uniform
