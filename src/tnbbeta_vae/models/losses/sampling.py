"""Sampling a posterior together with its log-density, exactly where that is possible.

``TNBBetaSpherical.log_prob(rsample())`` is inexact below ``epsilon`` of about
0.03: a unit vector cannot encode a distance to a pole smaller than its
floating-point resolution, so the density evaluated at the returned vector
under-states ``E_q[log q]`` and a Monte Carlo ELBO / KL built from it rewards
driving ``epsilon`` to the head floor. ``TNBBetaSpherical.rsample_with_log_prob``
evaluates the log-density from the sampled latitude logit instead and is exact.
Every Monte Carlo ``E_q[log q]`` in this package goes through the helpers below,
which use that method when the distribution has it and otherwise do exactly what
the call sites did before (``rsample`` / ``sample`` then ``log_prob``, same RNG
consumption).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributions import kl_divergence

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution

__all__ = ["sample_and_kl", "sample_with_log_prob"]


def sample_with_log_prob(
    dist: Distribution,
    sample_shape: torch.Size | None = None,
    *,
    reparameterize: bool = True,
    need_log_prob: bool = True,
) -> tuple[Tensor, Tensor | None]:
    """Draws ``z ~ dist`` and returns it with ``log q(z)``.

    Args:
        dist: The distribution. If it has ``rsample_with_log_prob`` (TNBBeta
            spherical) the log-density is the exact one from that method.
        sample_shape: Sample shape, or ``None`` to call ``rsample()`` /
            ``sample()`` without an argument (what the models did before).
        reparameterize: If False, draw without gradients through the sample
            (``dist.sample``). For a distribution with ``rsample_with_log_prob``
            the whole call then runs under ``torch.no_grad`` and the returned
            log-density is detached; use that only for evaluation.
        need_log_prob: If False, skip the log-density (second element is
            ``None``) and just draw the sample.

    Returns:
        ``(z, log_q)``.
    """
    exact = getattr(dist, "rsample_with_log_prob", None)
    if exact is not None and need_log_prob:
        shape = torch.Size() if sample_shape is None else sample_shape
        if reparameterize:
            return exact(shape)
        with torch.no_grad():
            return exact(shape)
    if reparameterize:
        z = dist.rsample() if sample_shape is None else dist.rsample(sample_shape)
    else:
        z = dist.sample() if sample_shape is None else dist.sample(sample_shape)
    return z, (dist.log_prob(z) if need_log_prob else None)


def sample_and_kl(
    posterior: Distribution,
    prior: Distribution,
    sample_shape: torch.Size | None = None,
) -> tuple[Tensor, Tensor]:
    """Draws ``z ~ posterior`` and returns it with ``KL(posterior || prior)``.

    The KL is closed-form where PyTorch has one for the pair, else the
    single-draw (per sample-shape element) estimate ``log q(z) - log p(z)``,
    with ``log q`` exact for TNBBeta (see :func:`sample_with_log_prob`).

    Args:
        posterior: ``q(z|x)``.
        prior: ``p(z)``.
        sample_shape: As in :func:`sample_with_log_prob`.

    Returns:
        ``(z, kl)``.
    """
    try:
        exact_kl = kl_divergence(posterior, prior)
    except NotImplementedError:
        z, log_q = sample_with_log_prob(posterior, sample_shape)
        assert log_q is not None
        return z, log_q - prior.log_prob(z)
    z, _ = sample_with_log_prob(posterior, sample_shape, need_log_prob=False)
    return z, exact_kl
