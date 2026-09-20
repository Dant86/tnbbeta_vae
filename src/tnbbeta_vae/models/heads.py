"""Maps raw network outputs to latent posteriors, shared across encoder types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn
from torch.distributions import Independent, Normal

from tnbbeta_vae.distributions import TNBBetaSpherical, VonMisesFisher

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["gaussian_posterior", "tnbbeta_posterior", "vmf_posterior"]

_PARAM_EPS = 1e-4
# Bounds for the posterior's p and q. A clamped value passes no gradient; 1e-6 keeps
# the bound off the fitted values (at 1e-4, p sat exactly on it at latent_dim=2) while
# staying above what float32 can resolve near 1 (about 1e-7).
_PQ_CLAMP = 1e-6
_LOG_VAR_BOUND = 10.0


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
    mean_direction = raw_direction / raw_direction.norm(dim=-1, keepdim=True).clamp_min(
        _PARAM_EPS
    )
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
        A batch of posteriors with unit mean and ``kappa = softplus(raw) + 1``.
    """
    mean = raw_mean / raw_mean.norm(dim=-1, keepdim=True)
    return VonMisesFisher(mean, nn.functional.softplus(raw_kappa) + 1)


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
