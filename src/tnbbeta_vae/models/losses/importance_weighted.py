"""Importance-weighted evaluation metrics, as in the S-VAE paper's Table 1.

For each image the marginal log-likelihood is estimated with importance
sampling using the posterior as the proposal (Burda et al., 2016)::

    LL ~= log (1/K) sum_k p(x|z_k) p(z_k) / q(z_k|x),   z_k ~ q(z|x)

alongside the reconstruction term ``RE = E_q[log p(x|z)]``, the KL, and the
ELBO ``RE - KL``. It needs only ``log_prob`` and ``sample``, so it works for any
posterior/prior pair, including TNBBetaSpherical.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch.distributions import kl_divergence

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["importance_weighted_metrics"]


@torch.no_grad()
def importance_weighted_metrics(
    model: Any, x: Tensor, *, num_samples: int = 500, chunk_size: int = 100
) -> dict[str, Tensor]:
    """Estimates per-image LL, ELBO, RE and KL for a batch.

    Args:
        model: A model with ``posterior_and_prior(x)`` and ``log_likelihood(x, z)``
            (see :class:`~tnbbeta_vae.models.conv_gaussian_vae.ConvGaussianVAE`).
        x: Images, shape ``(batch, channels, height, width)``.
        num_samples: Importance samples per image (the paper uses 500).
        chunk_size: Samples decoded at once, to bound memory.

    Returns:
        Per-image tensors of shape ``(batch,)``, in nats: ``"ll"`` (importance
        weighted log-likelihood), ``"elbo"`` (``re - kl``), ``"re"`` (mean
        ``log p(x|z)`` over the samples) and ``"kl"`` (exact where PyTorch has a
        closed form for the pair, else the sample mean of ``log q - log p``).
    """
    posterior, prior = model.posterior_and_prior(x)
    log_weights, log_likelihoods, log_ratios = [], [], []
    remaining = num_samples
    while remaining > 0:
        count = min(chunk_size, remaining)
        z = posterior.sample(torch.Size([count]))
        log_likelihood = model.log_likelihood(x, z)
        log_ratio = posterior.log_prob(z) - prior.log_prob(z)
        log_likelihoods.append(log_likelihood)
        log_ratios.append(log_ratio)
        log_weights.append(log_likelihood - log_ratio)
        remaining -= count

    weights = torch.cat(log_weights)
    ll = torch.logsumexp(weights, dim=0) - math.log(num_samples)
    re = torch.cat(log_likelihoods).mean(dim=0)
    try:
        kl = kl_divergence(posterior, prior)
    except NotImplementedError:
        kl = torch.cat(log_ratios).mean(dim=0)
    return {"ll": ll, "elbo": re - kl, "re": re, "kl": kl}
