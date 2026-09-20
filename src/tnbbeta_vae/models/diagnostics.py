"""Cheap statistics of a `TNBBetaSpherical` posterior's parameters, for logging."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from tnbbeta_vae.distributions import TNBBetaSpherical

__all__ = ["tnbbeta_spherical_posterior_diagnostics"]


def tnbbeta_spherical_posterior_diagnostics(
    posterior: TNBBetaSpherical,
) -> dict[str, Tensor]:
    """Summarizes a batched posterior's p, q and epsilon.

    Logged every training step so a trend in the parameters (e.g. p or q
    pinned at its clamp, or q drifting toward 1) shows up in the run's
    metrics.

    Args:
        posterior: A batched `TNBBetaSpherical`, e.g. one example per row
            of a training batch.

    Returns:
        A dict of scalar tensors: min/mean/max of `p` and `q`, and mean
        `epsilon`.
    """
    return {
        "posterior_p_mean": posterior.p.mean(),
        "posterior_p_min": posterior.p.min(),
        "posterior_p_max": posterior.p.max(),
        "posterior_q_mean": posterior.q.mean(),
        "posterior_q_min": posterior.q.min(),
        "posterior_q_max": posterior.q.max(),
        "posterior_epsilon_mean": posterior.epsilon.mean(),
    }
