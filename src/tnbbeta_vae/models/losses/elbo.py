"""A generic, single-sample Monte Carlo ELBO.

Uses::

    ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
         = E_q(z|x)[log p(x|z) + log p(z) - log q(z|x)]

estimated with a single reparameterized sample ``z ~ q(z|x)``. This works
for *any* pair of ``torch.distributions.Distribution`` objects with
``has_rsample=True`` and a tractable ``log_prob`` -- deliberately not a
closed-form Gaussian KL, since TNBBetaSpherical has no known closed-form
KL divergence (unlike, say, two Gaussians or even two von Mises-Fisher
distributions in some cases).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.distributions import Normal

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution

__all__ = ["monte_carlo_elbo"]


def monte_carlo_elbo(
    x: Tensor,
    reconstruction: Tensor,
    z: Tensor,
    posterior: Distribution,
    prior: Distribution,
    likelihood_scale: float = 1.0,
) -> dict[str, Tensor]:
    """Computes a single-sample Monte Carlo ELBO for one batch.

    The reconstruction likelihood is a fixed-scale Gaussian over pixels
    (equivalent to MSE up to an additive constant) -- the simplest choice
    that keeps this function decoder-agnostic; swap in a different
    likelihood by not using this function for that term.

    Args:
        x: Target images, shape ``(batch, *event_shape)``.
        reconstruction: Decoder output (the Gaussian mean), same shape as
            ``x``.
        z: The reparameterized latent sample used to produce
            ``reconstruction``, shape ``(batch, *posterior.event_shape)``.
        posterior: ``q(z|x)``, e.g. a per-example ``TNBBetaSpherical``.
        prior: ``p(z)``, e.g. a :class:`FixedTNBBetaSphericalPrior`'s
            output.
        likelihood_scale: Fixed standard deviation of the Gaussian
            reconstruction likelihood.

    Returns:
        A dict with per-example (shape ``(batch,)``) tensors: ``"elbo"``,
        ``"log_likelihood"``, and ``"kl"`` (the Monte Carlo KL estimate,
        ``log q(z|x) - log p(z)``).
    """
    log_likelihood = Normal(reconstruction, likelihood_scale).log_prob(x)
    log_likelihood = log_likelihood.flatten(1).sum(-1)

    log_prior = prior.log_prob(z)
    log_posterior = posterior.log_prob(z)
    kl = log_posterior - log_prior

    return {
        "elbo": log_likelihood - kl,
        "log_likelihood": log_likelihood,
        "kl": kl,
    }
