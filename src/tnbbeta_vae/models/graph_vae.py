"""A variational graph auto-encoder with a Gaussian, vMF or TNBBeta latent.

Follows Kipf and Welling's VGAE as used in the S-VAE paper's link-prediction
experiment: a two-layer GCN encoder gives one posterior per node, and a link's
probability is a sigmoid of the inner product of its endpoints' latents. Training uses
one random node pair per training edge each step as a negative (as in the S-VAE
paper) and the VGAE's normalization, in which the KL term is divided by the number
of nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from pydantic import BaseModel
import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Independent, Normal
from torch.nn.functional import binary_cross_entropy_with_logits

from tnbbeta_vae.distributions import HypersphericalUniform
from tnbbeta_vae.models.heads import (
    gaussian_posterior,
    tnbbeta_posterior,
    vmf_posterior,
)
from tnbbeta_vae.models.priors.tnbbeta_spherical import (
    FixedTNBBetaSphericalPrior,
    uniform_prior_params,
)
from tnbbeta_vae.registry import register_model

__all__ = ["GraphBatch", "GraphVAE", "GraphVAEConfig"]

_INITIAL_TEMPERATURE = 5.0


@dataclass
class GraphBatch:
    """A whole graph for full-batch training.

    Attributes:
        features: Node features, shape ``(num_nodes, in_features)``.
        norm_adjacency: Sparse ``D^-1/2 (A + I) D^-1/2`` of the training graph.
        positive_edges: Training edges, shape ``(2, num_edges)``, each undirected
            edge once.
    """

    features: Tensor
    norm_adjacency: Tensor
    positive_edges: Tensor

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes."""
        return self.features.shape[0]

    def to(self, device: torch.device) -> GraphBatch:
        """Returns a copy on ``device``."""
        return GraphBatch(
            self.features.to(device),
            self.norm_adjacency.to(device),
            self.positive_edges.to(device),
        )


class GraphVAEConfig(BaseModel):
    """Hyperparameters for :class:`GraphVAE`.

    Attributes:
        family: ``"gaussian"`` (N(0, I) prior), ``"vmf"`` or ``"tnbbeta"`` (uniform
            sphere prior).
        in_features: Number of node features.
        hidden_dim: Width of the first GCN layer.
        latent_dim: Latent dimension.
        dropout: Dropout on the inputs of both GCN layers.
        fixed_temperature: For the sphere families, the inner product of unit vectors
            lies in [-1, 1], which caps a link's logit. ``None`` (default) learns a
            positive multiplier on it, starting at 5; a number fixes it (1.0 is the
            plain inner product). Ignored for the Gaussian, whose latent scale is free.
    """

    family: Literal["gaussian", "vmf", "tnbbeta"] = "tnbbeta"
    in_features: int = 1433
    hidden_dim: int = 32
    latent_dim: int = 16
    dropout: float = 0.0
    fixed_temperature: float | None = None


@register_model("graph_vae", config_cls=GraphVAEConfig)
class GraphVAE(nn.Module):
    """A GCN-encoder VAE for link prediction with a selectable latent family."""

    def __init__(self, config: GraphVAEConfig) -> None:
        """Initializes the model from ``config``.

        Args:
            config: Hyperparameters; see :class:`GraphVAEConfig`.
        """
        super().__init__()
        self.config = config
        head_sizes = {
            "gaussian": 2 * config.latent_dim,
            "vmf": config.latent_dim + 1,
            "tnbbeta": config.latent_dim + 3,
        }
        self.first = nn.Linear(config.in_features, config.hidden_dim, bias=False)
        self.second = nn.Linear(
            config.hidden_dim, head_sizes[config.family], bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(_INITIAL_TEMPERATURE)),
            requires_grad=config.fixed_temperature is None,
        )
        self.tnbbeta_prior = FixedTNBBetaSphericalPrior(
            config.latent_dim, *uniform_prior_params(config.latent_dim)
        )

    def training_step(
        self, batch: GraphBatch, kl_weight: float = 1.0
    ) -> dict[str, Tensor]:
        """Computes the VGAE loss on one full graph.

        Args:
            batch: The training graph.
            kl_weight: Multiplier on the KL term.

        Returns:
            A dict with ``"loss"`` (link cross-entropy plus the KL divided by the
            number of nodes), ``"link_loss"`` and ``"kl"`` (mean per node).
        """
        posterior, prior = self.posterior_and_prior(batch)
        z = posterior.rsample()
        positives = batch.positive_edges
        negatives = self._sample_negatives(batch)
        logits = self.link_logits(z, torch.cat([positives, negatives], dim=1))
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
            "loss": link_loss + kl_weight * kl_mean / batch.num_nodes,
            "link_loss": link_loss.detach(),
            "kl": kl_mean.detach(),
        }

    def posterior_and_prior(
        self, batch: GraphBatch
    ) -> tuple[Distribution, Distribution]:
        """Returns the per-node posteriors and the prior."""
        hidden = torch.relu(
            torch.sparse.mm(
                batch.norm_adjacency, self.first(self.dropout(batch.features))
            )
        )
        raw = torch.sparse.mm(batch.norm_adjacency, self.second(self.dropout(hidden)))
        return self._posterior(raw), self._prior(raw)

    def link_logits(self, z: Tensor, edges: Tensor) -> Tensor:
        """Returns the logit of each edge from its endpoints' latents.

        Args:
            z: Latents, shape ``(num_nodes, latent_dim)``.
            edges: Node pairs, shape ``(2, num_edges)``.

        Returns:
            Inner products, times the temperature for the sphere families.
        """
        inner = (z[edges[0]] * z[edges[1]]).sum(-1)
        if self.config.family == "gaussian":
            return inner
        return self.temperature() * inner

    def temperature(self) -> Tensor:
        """Returns the multiplier applied to inner products of unit vectors."""
        if self.config.fixed_temperature is not None:
            return torch.tensor(
                self.config.fixed_temperature, device=self.log_temperature.device
            )
        return self.log_temperature.exp()

    @torch.no_grad()
    def embeddings(self, batch: GraphBatch) -> Tensor:
        """Returns each node's posterior centre: the mean, or the mode direction.

        The TNBBeta mode direction is the mean direction, negated when p < 0.5 (which
        undoes the (mu, p) ~ (-mu, 1 - p) alias).
        """
        posterior, _ = self.posterior_and_prior(batch)
        if self.config.family == "gaussian":
            return posterior.base_dist.loc  # pyright: ignore[reportAttributeAccessIssue]
        if self.config.family == "vmf":
            return posterior.loc  # pyright: ignore[reportAttributeAccessIssue]
        direction = posterior.mean_direction  # pyright: ignore[reportAttributeAccessIssue]
        return torch.where((posterior.p > 0.5)[:, None], direction, -direction)  # pyright: ignore[reportAttributeAccessIssue]

    def _posterior(self, raw: Tensor) -> Distribution:
        if self.config.family == "gaussian":
            return gaussian_posterior(raw)
        if self.config.family == "vmf":
            return vmf_posterior(raw[..., :-1], raw[..., -1:])
        return tnbbeta_posterior(raw, self.config.latent_dim)

    def _prior(self, raw: Tensor) -> Distribution:
        if self.config.family == "gaussian":
            shape = (self.config.latent_dim,)
            return Independent(
                Normal(
                    torch.zeros(shape, device=raw.device),
                    torch.ones(shape, device=raw.device),
                ),
                1,
            )
        if self.config.family == "vmf":
            return HypersphericalUniform(self.config.latent_dim - 1, device=raw.device)
        return self.tnbbeta_prior()

    def _sample_negatives(self, batch: GraphBatch) -> Tensor:
        """Draws one uniform random node pair per positive edge."""
        count = batch.positive_edges.shape[1]
        return torch.randint(
            batch.num_nodes, (2, count), device=batch.positive_edges.device
        )
