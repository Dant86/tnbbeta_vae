"""A plain-MLP-encoder VAE trained on GraphVAE's exact pairwise/BCE objective.

The "(b)" ablation arm of the bimodality investigation documented in
CLAUDE.md: hold the architecture family (plain MLP encoder, no
message-passing/no graph at all) and data (synthetic, no real features
beyond what correlates with cluster identity, see
:mod:`tnbbeta_vae.data.cluster_mixture`) fixed, and swap ONLY the loss from
reconstruction to :class:`~tnbbeta_vae.models.graph_vae.GraphVAE`'s exact
pairwise dot-product + BCE objective, with no decoder at all. If the
posterior still goes bimodal here, that isolates "the pairwise/ranking loss
itself" as the driver, independent of the GCN architecture or any real
graph.

``log_temperature``/``temperature()`` intentionally duplicate
:class:`GraphVAE`'s own copies rather than sharing them: ``GraphVAE``'s must
stay exactly where they are so its existing checkpoints keep loading (see
CLAUDE.md), so this model keeps its own small, separate copy instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from pydantic import BaseModel
import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.nn.functional import binary_cross_entropy_with_logits

from tnbbeta_vae.models.architectures.mlp import mlp_stack
from tnbbeta_vae.models.heads import (
    LatentFamily,
    head_size,
    posterior_centre,
    posterior_from_raw,
    standard_prior,
)
from tnbbeta_vae.models.pairwise import pairwise_logits
from tnbbeta_vae.registry import register_model

__all__ = ["PairwiseBatch", "PairwiseMlpVAE", "PairwiseMlpVAEConfig"]

_INITIAL_TEMPERATURE = 5.0


@dataclass
class PairwiseBatch:
    """A batch of items for full-batch training.

    Attributes:
        features: Item features, shape ``(num_items, input_dim)``.
        positive_pairs: Same-cluster item-index pairs, shape ``(2,
            num_pairs)``.
    """

    features: Tensor
    positive_pairs: Tensor

    @property
    def num_items(self) -> int:
        """Returns the number of items."""
        return self.features.shape[0]

    def to(self, device: torch.device) -> PairwiseBatch:
        """Returns a copy on ``device``."""
        return PairwiseBatch(self.features.to(device), self.positive_pairs.to(device))


class PairwiseMlpVAEConfig(BaseModel):
    """Hyperparameters for :class:`PairwiseMlpVAE`.

    Attributes:
        family: ``"gaussian"`` (N(0, I) prior), or ``"vmf"``, ``"power_spherical"``
            or ``"tnbbeta"`` (all uniform-sphere prior).
        input_dim: Dimension of the item features.
        hidden_dims: Encoder hidden sizes.
        latent_dim: Latent dimension.
        fixed_temperature: For the sphere families, the inner product of unit
            vectors lies in [-1, 1], which caps a pair's logit. ``None``
            (default) learns a positive multiplier on it, starting at 5; a
            number fixes it (1.0 is the plain inner product). Ignored for the
            Gaussian, whose latent scale is free.
    """

    family: LatentFamily = "tnbbeta"
    input_dim: int = 20
    hidden_dims: list[int] = [64, 32]
    latent_dim: int = 16
    fixed_temperature: float | None = None


@register_model("pairwise_mlp_vae", config_cls=PairwiseMlpVAEConfig)
class PairwiseMlpVAE(nn.Module):
    """An MLP-encoder model trained on a pairwise dot-product + BCE loss only.

    No decoder anywhere in this class -- see the module docstring.
    """

    def __init__(self, config: PairwiseMlpVAEConfig) -> None:
        """Initializes the model from ``config``.

        Args:
            config: Hyperparameters; see :class:`PairwiseMlpVAEConfig`.
        """
        super().__init__()
        self.config = config
        self.encoder = mlp_stack([config.input_dim, *config.hidden_dims])
        self.posterior_head = nn.Linear(
            config.hidden_dims[-1], head_size(config.family, config.latent_dim)
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(_INITIAL_TEMPERATURE)),
            requires_grad=config.fixed_temperature is None,
        )

    def training_step(
        self, batch: PairwiseBatch, kl_weight: float = 1.0
    ) -> dict[str, Tensor]:
        """Computes the pairwise dot-product + BCE loss on one full batch.

        Mirrors :meth:`~tnbbeta_vae.models.graph_vae.GraphVAE.training_step`
        exactly, minus the graph: sample ``z ~ posterior.rsample()``, score
        the positive pairs plus freshly-sampled negatives via
        :func:`~tnbbeta_vae.models.pairwise.pairwise_logits`, and divide the
        KL by the number of items (the same VGAE-style normalization
        ``GraphVAE`` uses).

        Args:
            batch: The training batch.
            kl_weight: Multiplier on the KL term.

        Returns:
            A dict with ``"loss"`` (pair cross-entropy plus the KL divided by
            the number of items), ``"link_loss"`` and ``"kl"`` (mean per item).
        """
        posterior, prior = self.posterior_and_prior(batch)
        z = posterior.rsample()
        positives = batch.positive_pairs
        negatives = self._sample_negatives(batch)
        all_pairs = torch.cat([positives, negatives], dim=1)
        logits = pairwise_logits(z, all_pairs, self.config.family, self.temperature())
        targets = torch.cat(
            [torch.ones(positives.shape[1]), torch.zeros(negatives.shape[1])]
        ).to(logits.device)
        link_loss = binary_cross_entropy_with_logits(logits, targets)
        try:
            kl = torch.distributions.kl_divergence(posterior, prior)
        except NotImplementedError:
            kl = posterior.log_prob(z) - prior.log_prob(z)
        kl_mean = kl.mean()
        return {
            "loss": link_loss + kl_weight * kl_mean / batch.num_items,
            "link_loss": link_loss.detach(),
            "kl": kl_mean.detach(),
        }

    def posterior_and_prior(
        self, batch: PairwiseBatch
    ) -> tuple[Distribution, Distribution]:
        """Returns the per-item posteriors and the prior."""
        raw = self.posterior_head(self.encoder(batch.features))
        return self._posterior(raw), self._prior(raw)

    def temperature(self) -> Tensor:
        """Returns the multiplier applied to inner products of unit vectors."""
        if self.config.fixed_temperature is not None:
            return torch.tensor(
                self.config.fixed_temperature, device=self.log_temperature.device
            )
        return self.log_temperature.exp()

    @torch.no_grad()
    def embeddings(self, batch: PairwiseBatch) -> Tensor:
        """Returns each item's posterior centre: the mean, or the mode direction."""
        posterior, _ = self.posterior_and_prior(batch)
        return posterior_centre(self.config.family, posterior)

    def _posterior(self, raw: Tensor) -> Distribution:
        return posterior_from_raw(self.config.family, raw, self.config.latent_dim)

    def _prior(self, raw: Tensor) -> Distribution:
        return standard_prior(self.config.family, self.config.latent_dim, raw.device)

    def _sample_negatives(self, batch: PairwiseBatch) -> Tensor:
        """Draws one uniform random item pair per positive pair.

        Unfiltered -- matches
        :meth:`~tnbbeta_vae.models.graph_vae.GraphVAE._sample_negatives`'s
        convention of not excluding accidental same-cluster negatives.
        """
        count = batch.positive_pairs.shape[1]
        return torch.randint(
            batch.num_items, (2, count), device=batch.positive_pairs.device
        )
