"""Maps raw network outputs to latent posteriors, shared across encoder types."""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal

from tnbbeta_vae.distributions import (
    HypersphericalUniform,
    TNBBetaSpherical,
    VonMisesFisher,
)
from tnbbeta_vae.models.priors.tnbbeta_spherical import (
    FixedTNBBetaSphericalPrior,
    uniform_prior_params,
)

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "LatentFamily",
    "gaussian_posterior",
    "head_size",
    "posterior_centre",
    "posterior_from_raw",
    "standard_prior",
    "tnbbeta_posterior",
    "vmf_kappa",
    "vmf_kappa_inverse",
    "vmf_posterior",
]

LatentFamily = Literal["gaussian", "vmf", "tnbbeta"]

_PARAM_EPS = 1e-4
# Bounds for the posterior's p and q. A clamped value passes no gradient; 1e-6 keeps
# the bound off the fitted values (at 1e-4, p sat exactly on it at latent_dim=2) while
# staying above what float32 can resolve near 1 (about 1e-7).
_PQ_CLAMP = 1e-6
_LOG_VAR_BOUND = 10.0
# In float32 the vMF log-normalizer and KL lose all precision as kappa grows: the KL is
# still accurate to ~0.01 nats at 1e6, wrong by ~0.4 at 1e7, and at ~5e8 it collapses to
# log(2 pi) while the sample's cosine with its mean is exactly 1. Beyond this cap kappa
# is meaningless, so it is clamped (no reference-style run has come near it).
_MAX_KAPPA = 1e6


def tnbbeta_posterior(raw: Tensor, latent_dim: int) -> TNBBetaSpherical:
    """Builds a TNBBetaSpherical posterior from ``latent_dim + 3`` raw outputs.

    Args:
        raw: Raw outputs, shape ``(batch, latent_dim + 3)``: an unnormalized mean
            direction, then the raw p, q and epsilon.
        latent_dim: Ambient dimension of the sphere.

    Returns:
        A batch of posteriors: unit mean direction, p and q squashed to (0, 1)
        and clamped at 1e-6, and epsilon = softplus(raw) > 0.
    """
    raw_direction, raw_p, raw_q, raw_epsilon = raw.split([latent_dim, 1, 1, 1], dim=-1)
    mean_direction = _unit_direction(raw_direction)
    p = raw_p.squeeze(-1).sigmoid().clamp(_PQ_CLAMP, 1 - _PQ_CLAMP)
    q = raw_q.squeeze(-1).sigmoid().clamp(_PQ_CLAMP, 1 - _PQ_CLAMP)
    epsilon = nn.functional.softplus(raw_epsilon.squeeze(-1)) + _PARAM_EPS
    return TNBBetaSpherical(mean_direction, p, q, epsilon)


def vmf_posterior(raw_mean: Tensor, raw_kappa: Tensor) -> VonMisesFisher:
    """Builds a von Mises-Fisher posterior as in the reference S-VAE.

    Args:
        raw_mean: Unnormalized mean direction, shape ``(batch, latent_dim)``.
        raw_kappa: Raw concentration, shape ``(batch, 1)``.

    Returns:
        A batch of posteriors with unit mean direction and concentration
        ``vmf_kappa(raw_kappa)``.
    """
    return VonMisesFisher(_unit_direction(raw_mean), vmf_kappa(raw_kappa))


def vmf_kappa(raw: Tensor) -> Tensor:
    """Maps a raw network output to a concentration: ``softplus(raw) + 1``, at most 1e6.

    The mapping is the reference S-VAE's. It is linear in ``raw`` for large kappa, so a
    large concentration needs a proportionally large raw output. The cap is the float32
    limit described above.
    """
    return (nn.functional.softplus(raw) + 1).clamp(max=_MAX_KAPPA)


def vmf_kappa_inverse(kappa: float) -> float:
    """Returns the raw output that :func:`vmf_kappa` maps to ``kappa`` (in (1, 1e6))."""
    if not 1 < kappa < _MAX_KAPPA:
        raise ValueError(f"kappa must be in (1, {_MAX_KAPPA:g}); got {kappa}.")
    excess = kappa - 1
    # softplus^-1(x) = log(expm1(x)), which is x itself once exp(-x) is negligible.
    return excess if excess > 30 else math.log(math.expm1(excess))


def gaussian_posterior(raw: Tensor) -> Independent:
    """Builds a diagonal-Gaussian posterior from ``2 * latent_dim`` raw outputs.

    Args:
        raw: Raw outputs, shape ``(batch, 2 * latent_dim)``: the mean, then the
            log-variance (clamped to +-10).

    Returns:
        An ``Independent(Normal(mu, sigma), 1)`` batch.
    """
    mu, log_var = raw.chunk(2, dim=-1)
    sigma = (0.5 * log_var.clamp(-_LOG_VAR_BOUND, _LOG_VAR_BOUND)).exp()
    return Independent(Normal(mu, sigma), 1)


def head_size(family: LatentFamily, latent_dim: int) -> int:
    """Returns how many raw outputs a head needs for ``family`` at ``latent_dim``."""
    return {
        "gaussian": 2 * latent_dim,
        "vmf": latent_dim + 1,
        "tnbbeta": latent_dim + 3,
    }[family]


def posterior_from_raw(
    family: LatentFamily, raw: Tensor, latent_dim: int
) -> Distribution:
    """Builds a distribution of ``family`` from ``head_size`` raw outputs per row.

    Args:
        family: ``"gaussian"``, ``"vmf"`` or ``"tnbbeta"``.
        raw: Raw outputs, shape ``(..., head_size(family, latent_dim))``.
        latent_dim: Latent dimension (ambient dimension for the sphere families).

    Returns:
        A batch of distributions over ``latent_dim``-dimensional vectors.
    """
    if family == "gaussian":
        return gaussian_posterior(raw)
    if family == "vmf":
        return vmf_posterior(raw[..., :-1], raw[..., -1:])
    return tnbbeta_posterior(raw, latent_dim)


def posterior_centre(family: LatentFamily, distribution: Distribution) -> Tensor:
    """Returns a posterior's centre: the mean, or the mode direction on a sphere.

    For TNBBeta the centre is the mean direction, negated when p < 0.5, which undoes
    the (mu, p) ~ (-mu, 1 - p) alias.

    Args:
        family: The family ``distribution`` was built with.
        distribution: A batch built by :func:`posterior_from_raw`.

    Returns:
        Tensor of shape ``(..., latent_dim)``.
    """
    if family == "gaussian":
        return distribution.base_dist.loc  # pyright: ignore[reportAttributeAccessIssue]
    if family == "vmf":
        return distribution.loc  # pyright: ignore[reportAttributeAccessIssue]
    direction = distribution.mean_direction  # pyright: ignore[reportAttributeAccessIssue]
    flip = distribution.p > 0.5  # pyright: ignore[reportAttributeAccessIssue]
    return torch.where(flip[..., None], direction, -direction)


def standard_prior(
    family: LatentFamily, latent_dim: int, device: torch.device
) -> Distribution:
    """Returns the family's standard prior: N(0, I), or uniform on the sphere.

    Args:
        family: ``"gaussian"``, ``"vmf"`` or ``"tnbbeta"``.
        latent_dim: Latent dimension.
        device: Device for the distribution's parameters.

    Returns:
        ``N(0, I)`` for the Gaussian, ``HypersphericalUniform`` for vMF and the
        uniform-sphere TNBBetaSpherical (p = 0.5, q = 0, epsilon = (d - 1) / 2) for
        TNBBeta.
    """
    if family == "gaussian":
        zeros = torch.zeros(latent_dim, device=device)
        return Independent(Normal(zeros, torch.ones_like(zeros)), 1)
    if family == "vmf":
        return HypersphericalUniform(latent_dim - 1, device=device)
    return _tnbbeta_uniform_prior(latent_dim, device)()


@functools.cache
def _tnbbeta_uniform_prior(
    latent_dim: int, device: torch.device
) -> FixedTNBBetaSphericalPrior:
    prior = FixedTNBBetaSphericalPrior(latent_dim, *uniform_prior_params(latent_dim))
    return prior.to(device)


def _unit_direction(raw: Tensor) -> Tensor:
    """Normalizes ``raw`` rows to unit length, using e_1 for (near-)zero rows.

    A row can be exactly zero, e.g. a graph node with no features and no neighbours
    gets a zero GCN output. Dividing by its norm would give NaN (or a non-unit vector
    if the norm is clamped), so such rows fall back to the fixed pole e_1.
    """
    norm = raw.norm(dim=-1, keepdim=True)
    pole = torch.zeros_like(raw)
    pole[..., 0] = 1.0
    return torch.where(norm > _PARAM_EPS, raw / norm.clamp_min(_PARAM_EPS), pole)
