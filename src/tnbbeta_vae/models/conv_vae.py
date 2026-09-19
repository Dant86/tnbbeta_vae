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
from tnbbeta_vae.models.priors.tnbbeta_spherical import FixedTNBBetaSphericalPrior
from tnbbeta_vae.registry import register_model

__all__ = ["ConvTNBBetaSphericalVAE", "ConvTNBBetaSphericalVAEConfig"]

_PARAM_EPS = 1e-4


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
        prior_p: Fixed prior median, in (0, 1).
        prior_q: Fixed prior concentration, in (0, 1).
        prior_epsilon: Fixed prior boundary parameter, > 0.
        likelihood_scale: Fixed standard deviation of the Gaussian
            reconstruction likelihood.
            (The starting value when ``learn_likelihood_scale`` is set.)
        learn_likelihood_scale: If True, ``likelihood_scale`` becomes a learned
            shared scalar (parameterized by log sigma^2) instead of a fixed value.
        param_clamp: The posterior's p and q are clamped to ``[param_clamp,
            1 - param_clamp]``. A clamped value passes no gradient, so p or q
            sitting exactly on the boundary means the model wants more extreme
            values than allowed. The floor of what float32 can resolve near
            1 is about 1e-7.
        num_elbo_samples: Number of z ~ q(z|x) draws to average per ELBO
            estimate. Higher values lower variance (there's no
            closed-form KL to fall back on here) at the cost of that many
            extra decoder calls per training step.
    """

    image_channels: int = 3
    image_size: int = 32
    hidden_channels: int = 32
    latent_dim: int = 8
    prior_p: float = 0.9
    prior_q: float = 0.9
    prior_epsilon: float = 1.0
    likelihood_scale: float = 1.0
    learn_likelihood_scale: bool = False
    param_clamp: float = _PARAM_EPS
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
        self.learned_scale = (
            LearnedLikelihoodScale(config.likelihood_scale)
            if config.learn_likelihood_scale
            else None
        )
        self.prior = FixedTNBBetaSphericalPrior(
            config.latent_dim, config.prior_p, config.prior_q, config.prior_epsilon
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
        scale = self._likelihood_scale()
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
        bound = self.config.param_clamp
        p = torch.sigmoid(raw_p.squeeze(-1)).clamp(bound, 1 - bound)
        q = torch.sigmoid(raw_q.squeeze(-1)).clamp(bound, 1 - bound)
        epsilon = nn.functional.softplus(raw_epsilon.squeeze(-1)) + _PARAM_EPS

        return TNBBetaSpherical(mean_direction, p, q, epsilon)

    def _likelihood_scale(self) -> float | Tensor:
        """Returns the learned scale if enabled, else the configured constant."""
        if self.learned_scale is None:
            return self.config.likelihood_scale
        return self.learned_scale()
