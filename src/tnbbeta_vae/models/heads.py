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


def tnbbeta_posterior(
    raw: Tensor,
    latent_dim: int,
    fixed_epsilon: float | None = None,
    fixed_mean_direction: bool = False,
) -> TNBBetaSpherical:
    """Builds a TNBBetaSpherical posterior from raw network outputs.

    Args:
        raw: Raw outputs, in order: an unnormalized mean direction (unless
            ``fixed_mean_direction``), then the raw p and q, then the raw epsilon
            (unless ``fixed_epsilon`` is set). Shape ``(batch, head_size)``, where
            ``head_size`` is ``latent_dim + 3`` minus ``latent_dim`` if
            ``fixed_mean_direction`` and minus ``1`` if ``fixed_epsilon`` is set,
            since a fixed parameter is nothing for the network to predict.
        latent_dim: Ambient dimension of the sphere.
        fixed_epsilon: If set, every posterior in the batch gets this constant
            epsilon instead of one predicted from ``raw`` -- an ablation testing
            whether removing epsilon's freedom pushes p and/or q to pick up
            whatever work epsilon was doing (see ``ConvTNBBetaSphericalVAEConfig``).
        fixed_mean_direction: If set, every posterior in the batch gets the fixed
            pole ``e_1`` as its mean direction instead of one predicted from
            ``raw`` -- an ablation testing whether p/q/epsilon become informative
            once direction, the channel that otherwise carries essentially all
            per-example signal, is no longer available to carry it instead.

    Returns:
        A batch of posteriors: unit mean direction (or the fixed pole), p and q
        squashed to (0, 1) and clamped at 1e-6, and epsilon = softplus(raw) > 0
        (or the fixed value).
    """
    direction_size = 0 if fixed_mean_direction else latent_dim
    epsilon_size = 0 if fixed_epsilon is not None else 1
    raw_direction, raw_p, raw_q, raw_epsilon = raw.split(
        [direction_size, 1, 1, epsilon_size], dim=-1
    )
    if fixed_mean_direction:
        mean_direction = _fixed_pole(latent_dim, raw_p.shape[:-1], raw.device)
    else:
        mean_direction = _unit_direction(raw_direction)
    if fixed_epsilon is None:
        epsilon = nn.functional.softplus(raw_epsilon.squeeze(-1)) + _PARAM_EPS
    else:
        epsilon = torch.full_like(raw_p.squeeze(-1), fixed_epsilon)
    p = raw_p.squeeze(-1).sigmoid().clamp(_PQ_CLAMP, 1 - _PQ_CLAMP)
    q = raw_q.squeeze(-1).sigmoid().clamp(_PQ_CLAMP, 1 - _PQ_CLAMP)
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


def _fixed_pole(
    latent_dim: int, batch_shape: torch.Size, device: torch.device
) -> Tensor:
    """Returns the fixed pole ``e_1``, broadcast to ``(*batch_shape, latent_dim)``."""
    pole = torch.zeros(*batch_shape, latent_dim, device=device)
    pole[..., 0] = 1.0
    return pole


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
