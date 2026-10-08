"""A variational graph auto-encoder with a selectable latent family.

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

from pydantic import BaseModel
import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.nn.functional import binary_cross_entropy_with_logits

from tnbbeta_vae.models.heads import (
    LatentFamily,
    head_size,
    posterior_centre,
    posterior_from_raw,
    standard_prior,
)
from tnbbeta_vae.models.pairwise import pairwise_logits
from tnbbeta_vae.registry import register_model

__all__ = ["GraphBatch", "GraphVAE", "GraphVAEConfig"]

_INITIAL_TEMPERATURE = 5.0


@dataclass
class GraphBatch:
    """A whole graph for full-batch training.

    Attributes:
        features: Node features, shape ``(num_nodes, in_features)``. May be dense
            (torch.Tensor) or sparse (torch.sparse.Tensor, typically COO format).
            For sparse identity features on a featureless graph, use
            torch.sparse_coo_tensor.
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
        family: ``"gaussian"`` (N(0, I) prior), or ``"vmf"``, ``"power_spherical"``
            or ``"tnbbeta"`` (all uniform-sphere prior).
        in_features: Number of node features.
        hidden_dim: Width of the first GCN layer.
        latent_dim: Latent dimension.
        dropout: Dropout on the inputs of both GCN layers.
        fixed_temperature: For the sphere families, the inner product of unit vectors
            lies in [-1, 1], which caps a link's logit. ``None`` (default) learns a
            positive multiplier on it, starting at 5; a number fixes it (1.0 is the
            plain inner product). Ignored for the Gaussian, whose latent scale is free.
        feature_reconstruction_weight: Weight on an added node-feature reconstruction
            term, on top of the usual link-prediction loss. ``0.0`` (the default)
            means no feature decoder exists at all -- ``GraphVAE.__init__`` then
            builds exactly the parameters it always has, so every checkpoint trained
            before this field existed still loads with a strict
            ``load_state_dict``. A value ``> 0`` adds a linear decoder from the
            latent to per-feature Bernoulli logits (features are assumed binary,
            as Planetoid's bag-of-words features are) and adds
            ``feature_reconstruction_weight * <binary cross-entropy>`` to the
            training loss.
    """

    family: LatentFamily = "tnbbeta"
    in_features: int = 1433
    hidden_dim: int = 32
    latent_dim: int = 16
    dropout: float = 0.0
    fixed_temperature: float | None = None
    feature_reconstruction_weight: float = 0.0


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
        self.first = nn.Linear(config.in_features, config.hidden_dim, bias=False)
        self.second = nn.Linear(
            config.hidden_dim, head_size(config.family, config.latent_dim), bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(_INITIAL_TEMPERATURE)),
            requires_grad=config.fixed_temperature is None,
        )
        # Constructed only when requested: an unconditionally-built, zero-weighted
        # decoder would still add parameters, breaking strict load_state_dict on
        # every checkpoint trained before this field existed (see CLAUDE.md).
        self.feature_decoder: nn.Linear | None = None
        if config.feature_reconstruction_weight > 0:
            self.feature_decoder = nn.Linear(config.latent_dim, config.in_features)

    def training_step(
        self, batch: GraphBatch, kl_weight: float = 1.0
    ) -> dict[str, Tensor]:
        """Computes the VGAE loss on one full graph.

        Args:
            batch: The training graph.
            kl_weight: Multiplier on the KL term.

        Returns:
            A dict with ``"loss"`` (link cross-entropy plus the KL divided by the
            number of nodes, plus the weighted feature-reconstruction term if
            ``config.feature_reconstruction_weight > 0``), ``"link_loss"`` and
            ``"kl"`` (mean per node), and ``"feature_loss"`` (the raw, unweighted
            reconstruction cross-entropy, detached) when the feature decoder exists.
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
        loss = link_loss + kl_weight * kl_mean / batch.num_nodes
        result = {
            "loss": loss,
            "link_loss": link_loss.detach(),
            "kl": kl_mean.detach(),
        }
        if self.feature_decoder is not None:
            feature_targets = (
                batch.features.to_dense()
                if batch.features.is_sparse
                else batch.features
            )
            feature_loss = binary_cross_entropy_with_logits(
                self.feature_decoder(z), feature_targets
            )
            result["loss"] = (
                loss + self.config.feature_reconstruction_weight * feature_loss
            )
            result["feature_loss"] = feature_loss.detach()
        return result

    def posterior_and_prior(
        self, batch: GraphBatch
    ) -> tuple[Distribution, Distribution]:
        """Returns the per-node posteriors and the prior."""
        if batch.features.is_sparse:
            # For sparse features (typically identity matrices), apply the first layer
            # via sparse matrix multiplication to avoid densifying. This is necessary
            # for graphs with no real node features (e.g., com-DBLP: 317K would be
            # 402GB dense), where the standard substitute is an identity matrix.
            # torch.sparse.mm(A, B) computes A @ B^T, so we use self.first.weight.t()
            # and get features @ W^T directly (mathematically identical to
            # nn.Linear(features) but never densifies).
            first_out = torch.sparse.mm(batch.features, self.first.weight.t())
            # Dropout on sparse identity features is not well-defined: dropping a
            # feature on an identity matrix zeros a whole node's row, which is a
            # different operation than per-feature dropout semantics. Skip it entirely
            # for sparse features, a conservative choice that preserves correctness.
            hidden = torch.relu(torch.sparse.mm(batch.norm_adjacency, first_out))
        else:
            # Dense features: apply dropout and the first layer normally.
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
        return pairwise_logits(z, edges, self.config.family, self.temperature())

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
        return posterior_centre(self.config.family, posterior)

    def _posterior(self, raw: Tensor) -> Distribution:
        return posterior_from_raw(self.config.family, raw, self.config.latent_dim)

    def _prior(self, raw: Tensor) -> Distribution:
        return standard_prior(self.config.family, self.config.latent_dim, raw.device)

    def _sample_negatives(self, batch: GraphBatch) -> Tensor:
        """Draws one uniform random node pair per positive edge."""
        count = batch.positive_edges.shape[1]
        return torch.randint(
            batch.num_nodes, (2, count), device=batch.positive_edges.device
        )
