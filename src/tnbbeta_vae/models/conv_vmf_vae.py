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

from pydantic import BaseModel
import torch
from torch import Tensor, nn

from tnbbeta_vae.distributions import HypersphericalUniform, VonMisesFisher
from tnbbeta_vae.models.architectures.conv import ConvDecoder, ConvEncoder
from tnbbeta_vae.models.diagnostics import vmf_posterior_diagnostics
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo
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
            - 1); must be >= 3.
        likelihood_scale: Fixed standard deviation of the Gaussian
            reconstruction likelihood.
        num_elbo_samples: Number of z ~ q(z|x) draws to average per ELBO
            estimate (the KL is exact, so this only affects the
            likelihood term).
    """

    image_channels: int = 3
    image_size: int = 32
    hidden_channels: int = 32
    latent_dim: int = 8
    likelihood_scale: float = 1.0
    num_elbo_samples: int = 1


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

    def training_step(self, batch: Tensor) -> dict[str, Tensor]:
        """Computes the negative ELBO (analytic KL) for one batch.

        Args:
            batch: Input images, shape ``(batch, image_channels,
                image_size, image_size)``.

        Returns:
            A dict with ``"loss"``, ``"log_likelihood"``, ``"kl"``, and
            concentration diagnostics (see
            :func:`tnbbeta_vae.models.diagnostics.vmf_posterior_diagnostics`).
        """
        posterior = self._encode(batch)
        elbo_terms = monte_carlo_elbo(
            batch,
            posterior,
            self.prior(),
            self.decoder,
            self.config.likelihood_scale,
            self.config.num_elbo_samples,
            analytic_kl=True,
        )
        return {
            "loss": -elbo_terms["elbo"].mean(),
            "log_likelihood": elbo_terms["log_likelihood"].mean(),
            "kl": elbo_terms["kl"].mean(),
            **vmf_posterior_diagnostics(posterior),
        }

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
        z_mean = self.fc_mean(features)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        z_var = nn.functional.softplus(self.fc_var(features)) + 1
        return VonMisesFisher(z_mean, z_var)
