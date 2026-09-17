"""A generic Monte Carlo ELBO, averaged over one or more reparameterized samples.

Uses::

    ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
         = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]

estimated by averaging ``num_samples`` reparameterized draws ``z ~
q(z|x)``. This works for *any* pair of ``torch.distributions.Distribution``
objects with ``has_rsample=True`` and a tractable ``log_prob`` --
deliberately not a closed-form Gaussian KL. TNBBetaSpherical has no known
closed-form KL in general: for two rotationally-symmetric distributions
with *different* mean directions, the KL's cross term needs the
distribution of one distribution's cosine-similarity coordinate under the
other's measure, which (like the analogous von Mises-Fisher case) doesn't
reduce to a 1-D integral in general dimension -- so there's no shortcut
around Monte Carlo estimation here.

Note that a single-sample KL estimate (``log q(z) - log p(z)`` for one
``z ~ q``) can be negative for a given draw even though the true KL is
always >= 0 (Gibbs' inequality) -- only its *expectation* is guaranteed
non-negative. Averaging over more samples (``num_samples``) reduces the
estimator's variance but does not change this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributions import Normal

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.distributions import Distribution

__all__ = ["monte_carlo_elbo"]


def monte_carlo_elbo(
    x: Tensor,
    posterior: Distribution,
    prior: Distribution,
    decoder: Callable[[Tensor], Tensor],
    likelihood_scale: float = 1.0,
    num_samples: int = 1,
) -> dict[str, Tensor]:
    """Computes a Monte Carlo ELBO for one batch, averaged over `num_samples` draws.

    The reconstruction likelihood is a fixed-scale Gaussian over pixels
    (equivalent to MSE up to an additive constant) -- the simplest choice
    that keeps this function decoder-agnostic; swap in a different
    likelihood by not using this function for that term.

    Args:
        x: Target images, shape ``(batch, *event_shape)``.
        posterior: ``q(z|x)``, e.g. a per-example ``TNBBetaSpherical``.
        prior: ``p(z)``, e.g. a :class:`FixedTNBBetaSphericalPrior`'s
            output.
        decoder: Maps a latent sample (shape ``(batch,
            *posterior.event_shape)``) to a reconstruction, same shape as
            ``x``.
        likelihood_scale: Fixed standard deviation of the Gaussian
            reconstruction likelihood.
        num_samples: Number of independent ``z ~ q(z|x)`` draws to average
            over. More samples lower the estimator's variance (including
            the KL term's) at the cost of that many extra decoder calls.

    Returns:
        A dict with per-example (shape ``(batch,)``) tensors, each
        averaged over ``num_samples`` draws: ``"elbo"``,
        ``"log_likelihood"``, and ``"kl"`` (the Monte Carlo KL estimate,
        ``log q(z|x) - log p(z)``).
    """
    log_likelihoods = []
    kls = []
    for _ in range(num_samples):
        z = posterior.rsample()
        reconstruction = decoder(z)

        log_likelihood = Normal(reconstruction, likelihood_scale).log_prob(x)
        log_likelihoods.append(log_likelihood.flatten(1).sum(-1))
        kls.append(posterior.log_prob(z) - prior.log_prob(z))

    log_likelihood = torch.stack(log_likelihoods).mean(0)
    kl = torch.stack(kls).mean(0)

    return {
        "elbo": log_likelihood - kl,
        "log_likelihood": log_likelihood,
        "kl": kl,
    }
