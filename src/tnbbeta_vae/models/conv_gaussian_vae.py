"""A standard Gaussian VAE on the same conv encoder/decoder, as a baseline.

Exists to separate "is the problem the TNBBetaSpherical latent?" from "is it
the architecture/data/optimization/likelihood?": same
:class:`~tnbbeta_vae.models.architectures.conv.ConvEncoder`/``ConvDecoder``,
same reconstruction likelihood and ELBO machinery as
:class:`~tnbbeta_vae.models.conv_vae.ConvTNBBetaSphericalVAE`, but with a
diagonal-Gaussian posterior, a standard-normal prior, and a closed-form KL
(``torch.distributions.kl_divergence``) rather than a Monte Carlo one.
"""

from __future__ import annotations

from pydantic import BaseModel
import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

from tnbbeta_vae.models.architectures.conv import ConvDecoder, ConvEncoder
from tnbbeta_vae.models.diagnostics import gaussian_posterior_diagnostics
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.registry import register_model

__all__ = ["ConvGaussianVAE", "ConvGaussianVAEConfig"]

_LOG_VAR_BOUND = 10.0


class ConvGaussianVAEConfig(BaseModel):
    """Hyperparameters for :class:`ConvGaussianVAE`.

    Attributes:
        image_channels: Number of image channels (3 for RGB).
        image_size: Height/width of the (square) input image; must be
            divisible by 8.
        hidden_channels: Base conv channel width; must be divisible by 8.
        latent_dim: Dimensionality of the Gaussian latent.
        likelihood_scale: Fixed standard deviation of the Gaussian
            reconstruction likelihood.
            (The starting value when ``learn_likelihood_scale`` is set.)
        learn_likelihood_scale: If True, ``likelihood_scale`` becomes a learned
            shared scalar (parameterized by log sigma^2) instead of a fixed value.
        num_elbo_samples: Number of z ~ q(z|x) draws averaged for the
            reconstruction term (the KL is closed-form, so it needs none).
    """

    image_channels: int = 3
    image_size: int = 32
    hidden_channels: int = 32
    latent_dim: int = 8
    likelihood_scale: float = 1.0
    learn_likelihood_scale: bool = False
    num_elbo_samples: int = 1


@register_model("conv_gaussian_vae", config_cls=ConvGaussianVAEConfig)
class ConvGaussianVAE(nn.Module):
    """A conv VAE with a diagonal-Gaussian posterior and N(0, I) prior."""

    def __init__(self, config: ConvGaussianVAEConfig) -> None:
        """Initializes the model from ``config``.

        Args:
            config: Hyperparameters; see :class:`ConvGaussianVAEConfig`.
        """
        super().__init__()
        self.config = config

        self.encoder = ConvEncoder(
            config.image_channels, config.image_size, config.hidden_channels
        )
        self.posterior_head = nn.Linear(
            self.encoder.out_features, 2 * config.latent_dim
        )
        self.decoder = ConvDecoder(
            config.latent_dim,
            config.image_channels,
            config.image_size,
            config.hidden_channels,
        )
        self.learned_scale = (
            LearnedLikelihoodScale(config.likelihood_scale)
            if config.learn_likelihood_scale
            else None
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Independent, Tensor]:
        """Runs a full encode -> sample -> decode pass.

        Args:
            x: Input images, shape ``(batch, image_channels, image_size,
                image_size)``.

        Returns:
            A tuple ``(reconstruction, posterior, z)``: the decoded image
            mean, the ``q(z|x)`` distribution, and the reparameterized
            latent sample drawn from it.
        """
        mu, sigma = self._encode(x)
        posterior = Independent(Normal(mu, sigma), 1)
        z = posterior.rsample()
        return self.decoder(z), posterior, z

    def training_step(self, batch: Tensor) -> dict[str, Tensor]:
        """Computes the negative ELBO (closed-form KL) for one batch.

        Args:
            batch: Input images, shape ``(batch, image_channels,
                image_size, image_size)``.

        Returns:
            A dict with ``"loss"`` (the mean negative ELBO), ``"log_likelihood"``,
            ``"kl"``, and Gaussian collapse diagnostics (see
            :func:`tnbbeta_vae.models.diagnostics.gaussian_posterior_diagnostics`).
        """
        scale = self._likelihood_scale()
        mu, sigma = self._encode(batch)
        posterior = Independent(Normal(mu, sigma), 1)
        prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(sigma)), 1)
        elbo_terms = monte_carlo_elbo(
            batch,
            posterior,
            prior,
            self.decoder,
            scale,
            self.config.num_elbo_samples,
            analytic_kl=True,
        )
        return {
            "loss": -elbo_terms["elbo"].mean(),
            "log_likelihood": elbo_terms["log_likelihood"].mean(),
            "kl": elbo_terms["kl"].mean(),
            "likelihood_scale": torch.as_tensor(scale).detach(),
            **gaussian_posterior_diagnostics(mu, sigma),
        }

    @torch.no_grad()
    def generate(self, num_samples: int) -> Tensor:
        """Decodes ``num_samples`` draws from the N(0, I) prior.

        Args:
            num_samples: Number of images to generate.

        Returns:
            Images of shape ``(num_samples, image_channels, image_size,
            image_size)``.
        """
        weight = self.posterior_head.weight
        z = torch.randn(
            num_samples,
            self.config.latent_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        return self.decoder(z)

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Maps images to per-example posterior means and standard deviations."""
        mu, log_var = self.posterior_head(self.encoder(x)).chunk(2, dim=-1)
        log_var = log_var.clamp(-_LOG_VAR_BOUND, _LOG_VAR_BOUND)
        return mu, torch.exp(0.5 * log_var)

    def _likelihood_scale(self) -> float | Tensor:
        """Returns the learned scale if enabled, else the configured constant."""
        if self.learned_scale is None:
            return self.config.likelihood_scale
        return self.learned_scale()
