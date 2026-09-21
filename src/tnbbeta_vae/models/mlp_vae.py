"""An MLP VAE for vector data with a Gaussian, vMF or TNBBeta latent.

Used for small synthetic experiments (e.g. recovering a circle from a noisy
embedding in R^100, S-VAE paper section 5.1). The encoder and decoder are plain
MLPs; the latent family only changes the posterior head, the prior and the KL
(exact for the Gaussian and vMF, Monte Carlo for TNBBeta).
"""

from __future__ import annotations

from pydantic import BaseModel
import torch
from torch import Tensor, nn
from torch.distributions import Distribution

from tnbbeta_vae.models.heads import (
    LatentFamily,
    head_size,
    posterior_from_raw,
    standard_prior,
)
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo, pixel_log_likelihood
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.registry import register_model

__all__ = ["MlpVAE", "MlpVAEConfig"]


class MlpVAEConfig(BaseModel):
    """Hyperparameters for :class:`MlpVAE`.

    Attributes:
        family: Latent family: ``"gaussian"`` (N(0, I) prior), ``"vmf"`` or
            ``"tnbbeta"`` (both with the uniform-sphere prior).
        input_dim: Dimension of the data vectors.
        hidden_dims: Encoder hidden sizes; the decoder mirrors them in reverse.
        latent_dim: Latent dimension (the sphere is ``S^(latent_dim - 1)``).
        likelihood_scale: Starting value of the learned Gaussian likelihood sigma.
        num_elbo_samples: ``z ~ q(z|x)`` draws averaged per ELBO estimate.
    """

    family: LatentFamily = "tnbbeta"
    input_dim: int = 100
    hidden_dims: list[int] = [256, 128]
    latent_dim: int = 2
    likelihood_scale: float = 0.1
    num_elbo_samples: int = 1


@register_model("mlp_vae", config_cls=MlpVAEConfig)
class MlpVAE(nn.Module):
    """An MLP encoder/decoder VAE on vectors with a selectable latent family."""

    def __init__(self, config: MlpVAEConfig) -> None:
        """Initializes the model from ``config``.

        Args:
            config: Hyperparameters; see :class:`MlpVAEConfig`.
        """
        super().__init__()
        self.config = config
        self.encoder = _mlp([config.input_dim, *config.hidden_dims])
        self.posterior_head = nn.Linear(
            config.hidden_dims[-1], head_size(config.family, config.latent_dim)
        )
        self.decoder = nn.Sequential(
            _mlp([config.latent_dim, *reversed(config.hidden_dims)]),
            nn.Linear(config.hidden_dims[0], config.input_dim),
        )
        self.learned_scale = LearnedLikelihoodScale(config.likelihood_scale)

    def forward(self, x: Tensor) -> tuple[Tensor, Distribution, Tensor]:
        """Runs a full encode -> sample -> decode pass.

        Args:
            x: Data, shape ``(batch, input_dim)``.

        Returns:
            A tuple ``(reconstruction, posterior, z)``.
        """
        posterior = self._encode(x)
        z = posterior.rsample()
        return self.decoder(z), posterior, z

    def training_step(self, batch: Tensor, kl_weight: float = 1.0) -> dict[str, Tensor]:
        """Computes the negative ELBO for one batch.

        Args:
            batch: Data, shape ``(batch, input_dim)``.
            kl_weight: Multiplier on the KL term in the returned loss.

        Returns:
            A dict with ``"loss"``, ``"log_likelihood"``, ``"kl"`` and
            ``"likelihood_scale"``.
        """
        scale = self.learned_scale()
        posterior, prior = self.posterior_and_prior(batch)
        terms = monte_carlo_elbo(
            batch,
            posterior,
            prior,
            self.decoder,
            scale,
            self.config.num_elbo_samples,
            analytic_kl=self.config.family != "tnbbeta",
        )
        return {
            "loss": -(terms["log_likelihood"] - kl_weight * terms["kl"]).mean(),
            "log_likelihood": terms["log_likelihood"].mean(),
            "kl": terms["kl"].mean(),
            "likelihood_scale": torch.as_tensor(scale).detach(),
        }

    def posterior_and_prior(self, x: Tensor) -> tuple[Distribution, Distribution]:
        """Returns ``q(z|x)`` and the prior for a batch of data."""
        return self._encode(x), self._prior()

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """Returns ``log p(x|z)`` summed over the data dimensions.

        Args:
            x: Data, shape ``(batch, input_dim)``.
            z: Latents, shape ``(*samples, batch, latent_dim)``.

        Returns:
            Tensor of shape ``(*samples, batch)``.
        """
        return pixel_log_likelihood(
            x, self.decoder(z), "gaussian", self.learned_scale()
        )

    @torch.no_grad()
    def generate(self, num_samples: int) -> Tensor:
        """Decodes ``num_samples`` draws from the prior.

        Args:
            num_samples: Number of vectors to generate.

        Returns:
            Tensor of shape ``(num_samples, input_dim)``.
        """
        return self.decoder(self._prior().sample(torch.Size([num_samples])))

    def _encode(self, x: Tensor) -> Distribution:
        raw = self.posterior_head(self.encoder(x))
        return posterior_from_raw(self.config.family, raw, self.config.latent_dim)

    def _prior(self) -> Distribution:
        device = self.posterior_head.weight.device
        return standard_prior(self.config.family, self.config.latent_dim, device)


def _mlp(sizes: list[int]) -> nn.Sequential:
    """ReLU MLP whose layers follow ``sizes`` (activation after every layer)."""
    layers: list[nn.Module] = []
    for width_in, width_out in zip(sizes[:-1], sizes[1:], strict=True):
        layers += [nn.Linear(width_in, width_out), nn.ReLU()]
    return nn.Sequential(*layers)
