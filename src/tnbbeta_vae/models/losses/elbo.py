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

Where a closed-form KL *does* exist (e.g. Gaussian vs. Gaussian), pass
``analytic_kl=True`` to use ``torch.distributions.kl_divergence`` instead:
it's exact, deterministic given the distributions, and always >= 0. The
reconstruction term is still a Monte Carlo average either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import binary_cross_entropy_with_logits

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.distributions import Distribution

__all__ = ["monte_carlo_elbo", "pixel_log_likelihood"]


def monte_carlo_elbo(
    x: Tensor,
    posterior: Distribution,
    prior: Distribution,
    decoder: Callable[[Tensor], Tensor],
    likelihood_scale: float | Tensor = 1.0,
    num_samples: int = 1,
    analytic_kl: bool = False,
    likelihood: Literal["gaussian", "bernoulli"] = "gaussian",
) -> dict[str, Tensor]:
    """Computes a Monte Carlo ELBO for one batch, averaged over `num_samples` draws.

    The reconstruction likelihood is a Gaussian over pixels with standard
    deviation ``likelihood_scale`` (equivalent to MSE up to an additive
    constant), or, with ``likelihood="bernoulli"``, a Bernoulli over pixels
    (for binarized images such as MNIST).

    Args:
        x: Target images, shape ``(batch, *event_shape)``.
        posterior: ``q(z|x)``, e.g. a per-example ``TNBBetaSpherical``.
        prior: ``p(z)``, e.g. a :class:`FixedTNBBetaSphericalPrior`'s
            output.
        decoder: Maps a latent sample (shape ``(batch,
            *posterior.event_shape)``) to a reconstruction, same shape as
            ``x``: the Gaussian mean, or, for ``likelihood="bernoulli"``, the
            pixel *logits* (before the sigmoid).
        likelihood_scale: Standard deviation of the Gaussian reconstruction
            likelihood. A tensor (e.g. a learned scale) receives gradients.
            Ignored for ``likelihood="bernoulli"``.
        num_samples: Number of independent ``z ~ q(z|x)`` draws to average
            over. More samples lower the estimator's variance (including
            the KL term's, when it's Monte Carlo) at the cost of that many
            extra decoder calls.
        analytic_kl: If True, compute the KL in closed form via
            ``torch.distributions.kl_divergence(posterior, prior)`` instead
            of estimating it from samples. Raises ``NotImplementedError``
            for distribution pairs with no registered closed form (e.g.
            any pair involving ``TNBBetaSpherical``).
        likelihood: ``"gaussian"`` or ``"bernoulli"``; see above.

    Returns:
        A dict with per-example (shape ``(batch,)``) tensors, each
        averaged over ``num_samples`` draws: ``"elbo"``,
        ``"log_likelihood"``, and ``"kl"`` (Monte Carlo ``log q(z|x) -
        log p(z)``, or exact if ``analytic_kl``).
    """
    exact_kl = kl_divergence(posterior, prior) if analytic_kl else None

    log_likelihoods = []
    kls = []
    for _ in range(num_samples):
        z = posterior.rsample()
        reconstruction = decoder(z)

        log_likelihoods.append(
            pixel_log_likelihood(x, reconstruction, likelihood, likelihood_scale)
        )
        if exact_kl is None:
            kls.append(posterior.log_prob(z) - prior.log_prob(z))

    log_likelihood = torch.stack(log_likelihoods).mean(0)
    kl = exact_kl if exact_kl is not None else torch.stack(kls).mean(0)

    return {
        "elbo": log_likelihood - kl,
        "log_likelihood": log_likelihood,
        "kl": kl,
    }


def pixel_log_likelihood(
    x: Tensor,
    reconstruction: Tensor,
    likelihood: Literal["gaussian", "bernoulli"],
    likelihood_scale: float | Tensor = 1.0,
) -> Tensor:
    """Returns ``log p(x | z)`` summed over the image dimensions.

    Args:
        x: Target images, shape ``(batch, channels, height, width)``.
        reconstruction: Decoder output for ``x``, with any leading sample
            dimensions in front of ``x``'s shape (e.g. ``(samples, batch, ...)``):
            the Gaussian mean, or the pixel logits for ``"bernoulli"``.
        likelihood: ``"gaussian"`` or ``"bernoulli"``.
        likelihood_scale: Standard deviation of the Gaussian likelihood;
            ignored for ``"bernoulli"``.

    Returns:
        Tensor of shape ``reconstruction.shape[:-3]``.
    """
    if likelihood == "bernoulli":
        per_pixel = -binary_cross_entropy_with_logits(
            reconstruction, x.expand_as(reconstruction), reduction="none"
        )
    else:
        per_pixel = Normal(reconstruction, likelihood_scale).log_prob(x)
    return per_pixel.sum(dim=(-3, -2, -1))
