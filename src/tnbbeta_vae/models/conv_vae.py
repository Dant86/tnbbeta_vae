"""A simple convolutional VAE with a TNBBetaSpherical latent posterior/prior.

Wires together :mod:`tnbbeta_vae.models.architectures.conv`,
:mod:`tnbbeta_vae.models.priors.tnbbeta_spherical`, and
:mod:`tnbbeta_vae.models.losses.elbo` into a single registrable model, so
this is the first concrete, trainable instance of the project's actual
research target: a VAE whose latent space is the unit hypersphere under
TNBbeta, rather than the usual Gaussian.
"""

from __future__ import annotations

from pydantic import BaseModel
import torch
from torch import Tensor, nn

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models.architectures.conv import ConvDecoder, ConvEncoder
from tnbbeta_vae.models.diagnostics import tnbbeta_spherical_posterior_diagnostics
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.models.priors.tnbbeta_spherical import (
    FixedTNBBetaSphericalPrior,
    uniform_prior_params,
)
from tnbbeta_vae.registry import register_model

__all__ = ["ConvTNBBetaSphericalVAE", "ConvTNBBetaSphericalVAEConfig"]

_PARAM_EPS = 1e-4
# Bounds for the posterior's p and q. A clamped value passes no gradient; 1e-6 keeps
# the bound off the fitted values (at 1e-4, p sat exactly on it at latent_dim=2) while
# staying above what float32 can resolve near 1 (about 1e-7).
_PQ_CLAMP = 1e-6


class ConvTNBBetaSphericalVAEConfig(BaseModel):
    """Hyperparameters for :class:`ConvTNBBetaSphericalVAE`.

    Attributes:
        image_channels: Number of image channels (3 for RGB).
        image_size: Height/width of the (square) input image; must be
            divisible by 8. Defaults to CIFAR-10's 32.
        hidden_channels: Base conv channel width.
        latent_dim: Ambient dimension of the latent sphere S^(latent_dim
            - 1). Kept small by default so the latent space stays cheap to
            inspect/visualize while getting the pipeline working.
        likelihood_scale: Starting value of the Gaussian reconstruction
            likelihood's standard deviation. It is learned (one scalar shared by
            all pixels, parameterized by log sigma^2), so this only sets where
            training starts.
        num_elbo_samples: Number of z ~ q(z|x) draws to average per ELBO
            estimate. Higher values lower variance (there's no
            closed-form KL to fall back on here) at the cost of that many
            extra decoder calls per training step.
    """

    image_channels: int = 3
    image_size: int = 32
    hidden_channels: int = 32
    latent_dim: int = 8
    likelihood_scale: float = 1.0
    num_elbo_samples: int = 1


@register_model("conv_tnbbeta_spherical_vae", config_cls=ConvTNBBetaSphericalVAEConfig)
class ConvTNBBetaSphericalVAE(nn.Module):
    """A conv encoder/decoder VAE with a TNBBetaSpherical latent distribution."""

    def __init__(self, config: ConvTNBBetaSphericalVAEConfig) -> None:
        """Initializes the model from ``config``.

        Args:
            config: Hyperparameters; see :class:`ConvTNBBetaSphericalVAEConfig`.
        """
        super().__init__()
        self.config = config

        self.encoder = ConvEncoder(
            config.image_channels, config.image_size, config.hidden_channels
        )
        self.posterior_head = nn.Linear(
            self.encoder.out_features, config.latent_dim + 3
        )
        self.decoder = ConvDecoder(
            config.latent_dim,
            config.image_channels,
            config.image_size,
            config.hidden_channels,
        )
        self.learned_scale = LearnedLikelihoodScale(config.likelihood_scale)
        self.prior = FixedTNBBetaSphericalPrior(
            config.latent_dim, *uniform_prior_params(config.latent_dim)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, TNBBetaSpherical, Tensor]:
        """Runs a full encode -> sample -> decode pass.

        Args:
            x: Input images, shape ``(batch, image_channels, image_size,
                image_size)``.

        Returns:
            A tuple ``(reconstruction, posterior, z)``: the decoded image
            mean, the ``q(z|x)`` distribution, and the reparameterized
            latent sample drawn from it.
        """
        posterior = self._encode(x)
        z = posterior.rsample()
        reconstruction = self.decoder(z)
        return reconstruction, posterior, z

    def training_step(self, batch: Tensor) -> dict[str, Tensor]:
        """Computes the negative ELBO loss for one batch.

        Args:
            batch: Input images, shape ``(batch, image_channels,
                image_size, image_size)``.

        Returns:
            A dict with ``"loss"`` (the mean negative ELBO), plus
            ``"log_likelihood"``, ``"kl"``, and posterior-collapse
            diagnostics (see
            :func:`tnbbeta_vae.models.diagnostics.tnbbeta_spherical_posterior_diagnostics`)
            for logging.
        """
        scale = self.learned_scale()
        posterior = self._encode(batch)
        prior = self.prior()
        elbo_terms = monte_carlo_elbo(
            batch,
            posterior,
            prior,
            self.decoder,
            scale,
            self.config.num_elbo_samples,
        )
        return {
            "loss": -elbo_terms["elbo"].mean(),
            "log_likelihood": elbo_terms["log_likelihood"].mean(),
            "kl": elbo_terms["kl"].mean(),
            "likelihood_scale": torch.as_tensor(scale).detach(),
            **tnbbeta_spherical_posterior_diagnostics(posterior),
        }

    @torch.no_grad()
    def generate(self, num_samples: int) -> Tensor:
        """Decodes ``num_samples`` draws from the model's fixed prior.

        Args:
            num_samples: Number of images to generate.

        Returns:
            Images of shape ``(num_samples, image_channels, image_size,
            image_size)``.
        """
        return self.decoder(self.prior().sample(torch.Size([num_samples])))

    def _encode(self, x: Tensor) -> TNBBetaSpherical:
        """Maps images to a per-example TNBBetaSpherical posterior."""
        features = self.encoder(x)
        raw_direction, raw_p, raw_q, raw_epsilon = self.posterior_head(features).split(
            [self.config.latent_dim, 1, 1, 1], dim=-1
        )

        mean_direction = raw_direction / raw_direction.norm(
            dim=-1, keepdim=True
        ).clamp_min(_PARAM_EPS)
        p = torch.sigmoid(raw_p.squeeze(-1)).clamp(_PQ_CLAMP, 1 - _PQ_CLAMP)
        q = torch.sigmoid(raw_q.squeeze(-1)).clamp(_PQ_CLAMP, 1 - _PQ_CLAMP)
        epsilon = nn.functional.softplus(raw_epsilon.squeeze(-1)) + _PARAM_EPS

        return TNBBetaSpherical(mean_direction, p, q, epsilon)
