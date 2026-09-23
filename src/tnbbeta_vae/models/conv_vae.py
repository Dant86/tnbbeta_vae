"""A simple convolutional VAE with a TNBBetaSpherical latent posterior/prior.

Wires together :mod:`tnbbeta_vae.models.architectures.conv`,
:mod:`tnbbeta_vae.models.priors.tnbbeta_spherical`, and
:mod:`tnbbeta_vae.models.losses.elbo` into a single registrable model, so
this is the first concrete, trainable instance of the project's actual
research target: a VAE whose latent space is the unit hypersphere under
TNBbeta, rather than the usual Gaussian.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
import torch
from torch import Tensor, nn

from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models.architectures.conv import ConvDecoder, ConvEncoder
from tnbbeta_vae.models.diagnostics import tnbbeta_spherical_posterior_diagnostics
from tnbbeta_vae.models.heads import tnbbeta_posterior
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo, pixel_log_likelihood
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.models.priors.tnbbeta_spherical import (
    FixedTNBBetaSphericalPrior,
    uniform_prior_params,
)
from tnbbeta_vae.registry import register_model

__all__ = ["ConvTNBBetaSphericalVAE", "ConvTNBBetaSphericalVAEConfig"]


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
        likelihood: ``"gaussian"`` (learned sigma) or ``"bernoulli"`` (binarized
            images; the decoder logits are the Bernoulli parameters and
            ``likelihood_scale`` is unused).
        num_elbo_samples: Number of z ~ q(z|x) draws to average per ELBO
            estimate. Higher values lower variance (there's no
            closed-form KL to fall back on here) at the cost of that many
            extra decoder calls per training step.
        fixed_epsilon: If set, epsilon is held at this constant for every
            example instead of being predicted by the encoder (which removes
            an entire degree of freedom from the posterior_head's output).
            An ablation: MNIST training saturates p near 1 with q near 0 (a
            cap at the mean direction, the same shape a von Mises-Fisher
            posterior always has), with epsilon alone doing the work of
            "how concentrated" -- fixing epsilon tests whether p and/or q
            pick up that role once epsilon cannot. ``None`` (default) keeps
            epsilon learned, unchanged from before this option existed.
        fixed_mean_direction: If set, every posterior's mean direction is the
            fixed pole ``e_1`` instead of one predicted by the encoder (which
            removes ``latent_dim`` degrees of freedom from the
            posterior_head's output). An ablation: direction alone appears to
            carry essentially all of the per-example signal TNBBeta's
            posterior needs to convey (see the fixed-epsilon ablation and the
            latitude/confidence-probe results); fixing it removes that escape
            hatch so any per-example information has to route entirely
            through p/q/epsilon instead, testing whether that extra
            expressivity is usable when it's the only channel available.
            ``False`` (default) keeps direction learned, unchanged from
            before this option existed.
    """

    image_channels: int = 3
    image_size: int = 32
    hidden_channels: int = 32
    latent_dim: int = 8
    likelihood_scale: float = 1.0
    likelihood: Literal["gaussian", "bernoulli"] = "gaussian"
    num_elbo_samples: int = 1
    fixed_epsilon: float | None = None
    fixed_mean_direction: bool = False


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
        head_size = (0 if config.fixed_mean_direction else config.latent_dim) + (
            2 if config.fixed_epsilon is not None else 3
        )
        self.posterior_head = nn.Linear(self.encoder.out_features, head_size)
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

    def training_step(self, batch: Tensor, kl_weight: float = 1.0) -> dict[str, Tensor]:
        """Computes the negative ELBO loss for one batch.

        Args:
            batch: Input images, shape ``(batch, image_channels,
                image_size, image_size)``.
            kl_weight: Multiplier on the KL term in the returned loss (for KL
                warm-up). ``"kl"`` and ``"log_likelihood"`` are unweighted.

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
            self._decode_for_likelihood,
            scale,
            self.config.num_elbo_samples,
            likelihood=self.config.likelihood,
        )
        return {
            "loss": -(
                elbo_terms["log_likelihood"] - kl_weight * elbo_terms["kl"]
            ).mean(),
            "log_likelihood": elbo_terms["log_likelihood"].mean(),
            "kl": elbo_terms["kl"].mean(),
            "likelihood_scale": torch.as_tensor(scale).detach(),
            **tnbbeta_spherical_posterior_diagnostics(posterior),
        }

    def posterior_and_prior(
        self, x: Tensor
    ) -> tuple[TNBBetaSpherical, TNBBetaSpherical]:
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
        raw = self.posterior_head(self.encoder(x))
        return tnbbeta_posterior(
            raw,
            self.config.latent_dim,
            fixed_epsilon=self.config.fixed_epsilon,
            fixed_mean_direction=self.config.fixed_mean_direction,
        )

    def _decode_for_likelihood(self, z: Tensor) -> Tensor:
        """Decodes to Gaussian means, or to logits for a Bernoulli likelihood."""
        if self.config.likelihood == "bernoulli":
            return self.decoder.logits(z)
        return self.decoder(z)
