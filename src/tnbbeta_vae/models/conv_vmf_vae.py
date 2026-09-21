"""A convolutional VAE with a von Mises-Fisher latent posterior (S-VAE).

The standard hyperspherical baseline for
:class:`~tnbbeta_vae.models.conv_vae.ConvTNBBetaSphericalVAE`: same
encoder/decoder, uniform-on-the-sphere prior. The latent parameterization
follows the reference S-VAE example (Davidson et al., 2018): a normalized
mean direction and ``kappa = softplus(.) + 1``, with the analytic KL to the
uniform distribution. Comparing the two models separates "does the sphere
itself help?" from "does TNBbeta's extra flexibility help?".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
import torch
from torch import Tensor, nn

from tnbbeta_vae.distributions import HypersphericalUniform, VonMisesFisher
from tnbbeta_vae.models.architectures.conv import ConvDecoder, ConvEncoder
from tnbbeta_vae.models.heads import (
    KappaParameterization,
    vmf_kappa_inverse,
    vmf_posterior,
)
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo, pixel_log_likelihood
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.registry import register_model

__all__ = ["ConvVonMisesFisherVAE", "ConvVonMisesFisherVAEConfig"]


class ConvVonMisesFisherVAEConfig(BaseModel):
    """Hyperparameters for :class:`ConvVonMisesFisherVAE`.

    Attributes:
        image_channels: Number of image channels (3 for RGB).
        image_size: Height/width of the (square) input image; must be
            divisible by 8.
        hidden_channels: Base conv channel width.
        latent_dim: Ambient dimension of the latent sphere S^(latent_dim
            - 1).
        likelihood_scale: Starting value of the Gaussian reconstruction
            likelihood's standard deviation. It is learned (one scalar shared by
            all pixels, parameterized by log sigma^2), so this only sets where
            training starts.
        likelihood: ``"gaussian"`` (learned sigma) or ``"bernoulli"`` (binarized
            images; the decoder logits are the Bernoulli parameters and
            ``likelihood_scale`` is unused).
        num_elbo_samples: Number of z ~ q(z|x) draws to average per ELBO
            estimate (the KL is exact, so this only affects the
            likelihood term).
        initial_kappa: If set, initializes the concentration head's bias so every
            posterior starts near this kappa (> 1). ``None`` keeps the reference
            initialization (kappa near 1.7). The reference start is too noisy at high
            latent dimension: a sample's expected cosine with its mean direction is only
            about 0.04 at ``latent_dim=40``, the decoder learns to ignore z, and the
            model collapses to the mean image. A kappa around ``latent_dim`` avoids it.
        kappa_parameterization: ``"softplus"`` (reference) or ``"exp"``; see
            :func:`tnbbeta_vae.models.heads.vmf_kappa`.
    """

    image_channels: int = 3
    image_size: int = 32
    hidden_channels: int = 32
    latent_dim: int = 8
    likelihood_scale: float = 1.0
    likelihood: Literal["gaussian", "bernoulli"] = "gaussian"
    num_elbo_samples: int = 1
    initial_kappa: float | None = None
    kappa_parameterization: KappaParameterization = "softplus"


@register_model("conv_vmf_vae", config_cls=ConvVonMisesFisherVAEConfig)
class ConvVonMisesFisherVAE(nn.Module):
    """A conv encoder/decoder VAE with a von Mises-Fisher posterior.

    The prior is uniform on the sphere, so it needs no parameters; the KL to
    it is analytic.
    """

    def __init__(self, config: ConvVonMisesFisherVAEConfig) -> None:
        """Initializes the model from ``config``.

        Args:
            config: Hyperparameters; see :class:`ConvVonMisesFisherVAEConfig`.
        """
        super().__init__()
        self.config = config

        self.encoder = ConvEncoder(
            config.image_channels, config.image_size, config.hidden_channels
        )
        self.fc_mean = nn.Linear(self.encoder.out_features, config.latent_dim)
        self.fc_var = nn.Linear(self.encoder.out_features, 1)
        self.decoder = ConvDecoder(
            config.latent_dim,
            config.image_channels,
            config.image_size,
            config.hidden_channels,
        )
        self.learned_scale = LearnedLikelihoodScale(config.likelihood_scale)
        if config.initial_kappa is not None:
            raw = vmf_kappa_inverse(config.initial_kappa, config.kappa_parameterization)
            with torch.no_grad():
                self.fc_var.bias.fill_(raw)

    def prior(self) -> HypersphericalUniform:
        """Builds the uniform-on-the-sphere prior."""
        return HypersphericalUniform(
            self.config.latent_dim - 1, device=self.fc_mean.weight.device
        )

    def forward(self, x: Tensor) -> tuple[Tensor, VonMisesFisher, Tensor]:
        """Runs a full encode -> sample -> decode pass.

        Args:
            x: Input images, shape ``(batch, image_channels, image_size,
                image_size)``.

        Returns:
            A tuple ``(reconstruction, posterior, z)``.
        """
        posterior = self._encode(x)
        z = posterior.rsample()
        return self.decoder(z), posterior, z

    def training_step(self, batch: Tensor, kl_weight: float = 1.0) -> dict[str, Tensor]:
        """Computes the negative ELBO (analytic KL) for one batch.

        Args:
            batch: Input images, shape ``(batch, image_channels,
                image_size, image_size)``.
            kl_weight: Multiplier on the KL term in the returned loss (for KL
                warm-up). ``"kl"`` and ``"log_likelihood"`` are unweighted.

        Returns:
            A dict with ``"loss"``, ``"log_likelihood"``, ``"kl"``,
            ``"likelihood_scale"`` and ``"posterior_kappa_mean"``.
        """
        scale = self.learned_scale()
        posterior = self._encode(batch)
        elbo_terms = monte_carlo_elbo(
            batch,
            posterior,
            self.prior(),
            self._decode_for_likelihood,
            scale,
            self.config.num_elbo_samples,
            analytic_kl=True,
            likelihood=self.config.likelihood,
        )
        return {
            "loss": -(
                elbo_terms["log_likelihood"] - kl_weight * elbo_terms["kl"]
            ).mean(),
            "log_likelihood": elbo_terms["log_likelihood"].mean(),
            "kl": elbo_terms["kl"].mean(),
            "likelihood_scale": torch.as_tensor(scale).detach(),
            "posterior_kappa_mean": posterior.scale.mean().detach(),
        }

    def posterior_and_prior(
        self, x: Tensor
    ) -> tuple[VonMisesFisher, HypersphericalUniform]:
        """Returns ``q(z|x)`` and the uniform-sphere prior for a batch of images."""
        return self._encode(x), self.prior()

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """Returns ``log p(x|z)`` summed over pixels.

        Args:
            x: Images, shape ``(batch, channels, height, width)``.
            z: Latents, shape ``(*samples, batch, latent_dim)``.

        Returns:
            Tensor of shape ``(*samples, batch)``.
        """
        return pixel_log_likelihood(
            x,
            self._decode_for_likelihood(z),
            self.config.likelihood,
            self.learned_scale(),
        )

    @torch.no_grad()
    def generate(self, num_samples: int) -> Tensor:
        """Decodes ``num_samples`` draws from the uniform prior.

        Args:
            num_samples: Number of images to generate.

        Returns:
            Images of shape ``(num_samples, image_channels, image_size,
            image_size)``.
        """
        return self.decoder(self.prior().sample(torch.Size([num_samples])))

    def _encode(self, x: Tensor) -> VonMisesFisher:
        """Maps images to a per-example von Mises-Fisher posterior."""
        features = self.encoder(x)
        return vmf_posterior(
            self.fc_mean(features),
            self.fc_var(features),
            self.config.kappa_parameterization,
        )

    def _decode_for_likelihood(self, z: Tensor) -> Tensor:
        """Decodes to Gaussian means, or to logits for a Bernoulli likelihood."""
        if self.config.likelihood == "bernoulli":
            return self.decoder.logits(z)
        return self.decoder(z)
