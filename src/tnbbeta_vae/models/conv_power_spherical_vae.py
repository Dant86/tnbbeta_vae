"""A convolutional VAE with a Power Spherical latent posterior.

A fourth baseline alongside
:class:`~tnbbeta_vae.models.conv_vae.ConvTNBBetaSphericalVAE`,
:class:`~tnbbeta_vae.models.conv_gaussian_vae.ConvGaussianVAE` and
:class:`~tnbbeta_vae.models.conv_vmf_vae.ConvVonMisesFisherVAE`: same
encoder/decoder, uniform-on-the-sphere prior. The Power Spherical
distribution (De Cao & Aziz, 2020) shares von Mises-Fisher's rotational
symmetry and closed-form KL to the uniform prior, but is fully
reparameterizable without rejection sampling -- see
:mod:`tnbbeta_vae.distributions.power_spherical` for the exact
construction. Comparing this against :class:`ConvVonMisesFisherVAE`
isolates "does dropping vMF's rejection sampler change anything?" from the
sphere-vs-TNBBeta comparison the other two baselines are for.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
import torch
from torch import Tensor, nn

from tnbbeta_vae.distributions import HypersphericalUniform, PowerSpherical
from tnbbeta_vae.models.architectures.conv import ConvDecoder, ConvEncoder
from tnbbeta_vae.models.heads import (
    power_spherical_kappa_inverse,
    power_spherical_posterior,
)
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo, pixel_log_likelihood
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.registry import register_model

__all__ = ["ConvPowerSphericalVAE", "ConvPowerSphericalVAEConfig"]


class ConvPowerSphericalVAEConfig(BaseModel):
    """Hyperparameters for :class:`ConvPowerSphericalVAE`.

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
            posterior starts near this kappa (between 0 and 1e6). ``None`` keeps
            the default initialization (kappa near softplus(0) ~= 0.69). As with
            :class:`~tnbbeta_vae.models.conv_vmf_vae.ConvVonMisesFisherVAEConfig`,
            that start is too noisy at high latent dimension, and a kappa around
            ``latent_dim`` avoids it.
    """

    image_channels: int = 3
    image_size: int = 32
    hidden_channels: int = 32
    latent_dim: int = 8
    likelihood_scale: float = 1.0
    likelihood: Literal["gaussian", "bernoulli"] = "gaussian"
    num_elbo_samples: int = 1
    initial_kappa: float | None = None


@register_model("conv_power_spherical_vae", config_cls=ConvPowerSphericalVAEConfig)
class ConvPowerSphericalVAE(nn.Module):
    """A conv encoder/decoder VAE with a Power Spherical posterior.

    The prior is uniform on the sphere, so it needs no parameters; the KL to
    it is analytic (:meth:`~tnbbeta_vae.distributions.power_spherical.
    PowerSpherical.kl_to_uniform`, registered with
    ``torch.distributions.kl_divergence``).
    """

    def __init__(self, config: ConvPowerSphericalVAEConfig) -> None:
        """Initializes the model from ``config``.

        Args:
            config: Hyperparameters; see :class:`ConvPowerSphericalVAEConfig`.
        """
        super().__init__()
        self.config = config

        self.encoder = ConvEncoder(
            config.image_channels, config.image_size, config.hidden_channels
        )
        self.fc_mean = nn.Linear(self.encoder.out_features, config.latent_dim)
        self.fc_kappa = nn.Linear(self.encoder.out_features, 1)
        self.decoder = ConvDecoder(
            config.latent_dim,
            config.image_channels,
            config.image_size,
            config.hidden_channels,
        )
        self.learned_scale = LearnedLikelihoodScale(config.likelihood_scale)
        if config.initial_kappa is not None:
            raw = power_spherical_kappa_inverse(config.initial_kappa)
            with torch.no_grad():
                self.fc_kappa.bias.fill_(raw)

    def prior(self) -> HypersphericalUniform:
        """Builds the uniform-on-the-sphere prior."""
        return HypersphericalUniform(
            self.config.latent_dim - 1, device=self.fc_mean.weight.device
        )

    def forward(self, x: Tensor) -> tuple[Tensor, PowerSpherical, Tensor]:
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
            "posterior_kappa_mean": posterior.kappa.mean().detach(),
        }

    def posterior_and_prior(
        self, x: Tensor
    ) -> tuple[PowerSpherical, HypersphericalUniform]:
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

    def _encode(self, x: Tensor) -> PowerSpherical:
        """Maps images to a per-example Power Spherical posterior."""
        features = self.encoder(x)
        return power_spherical_posterior(
            self.fc_mean(features), self.fc_kappa(features)
        )

    def _decode_for_likelihood(self, z: Tensor) -> Tensor:
        """Decodes to Gaussian means, or to logits for a Bernoulli likelihood."""
        if self.config.likelihood == "bernoulli":
            return self.decoder.logits(z)
        return self.decoder(z)
