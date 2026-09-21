"""The stacked M1+M2 semi-supervised VAE of Kingma et al. (2014), as in the S-VAE paper.

M1 is a VAE ``x -> z1``. M2 sits on z1 with a class ``y`` and a second latent ``z2``:
``q(y|z1)`` (the classifier), ``q(z2|z1, y)`` and ``p(z1|y, z2)``. Both latents may
follow any family (Gaussian, vMF or TNBBeta), which gives the paper's N+N, S+S and S+N
models and the TNBBeta variants. Everything is trained end to end on one objective::

    labelled:    -L(x, y)  = -(A + B_y)               + alpha N * CE(q(y|z1), y)
    unlabelled:  -U(x)     = -(A + sum_y q(y|z1) B_y + H(q(y|z1)))
    A   = log p(x|z1) - log q(z1|x)
    B_y = E_{q(z2|z1,y)} [log p(z1|y,z2) + log p(y) + log p(z2) - log q(z2|z1,y)]

with one sample of z1 ~ q(z1|x) shared by the terms, averaged over the examples in a
batch. Every term uses only ``log_prob`` and ``rsample``, so TNBBeta needs no
closed-form KL. ``alpha * N`` follows Kingma et al.'s ``alpha = 0.1 N`` (the paper's
alpha in [0.1, 1] multiplies N).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

from pydantic import BaseModel
import torch
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, one_hot

from tnbbeta_vae.models.heads import (
    LatentFamily,
    head_size,
    posterior_centre,
    posterior_from_raw,
    standard_prior,
    vmf_kappa_inverse,
)
from tnbbeta_vae.registry import register_model

__all__ = ["M1M2Config", "M1M2VAE", "SemiBatch"]


@dataclass
class SemiBatch:
    """A step's data: all labelled examples plus a batch of unlabelled ones.

    Attributes:
        labeled_x: Labelled inputs, shape ``(n_labeled, input_dim)`` (may be empty).
        labels: Their classes, shape ``(n_labeled,)``.
        unlabeled_x: Unlabelled inputs, shape ``(n_unlabeled, input_dim)``.
    """

    labeled_x: Tensor
    labels: Tensor
    unlabeled_x: Tensor

    def __len__(self) -> int:
        """Returns the total number of examples."""
        return self.labeled_x.shape[0] + self.unlabeled_x.shape[0]

    def to(self, device: torch.device) -> SemiBatch:
        """Returns a copy on ``device``."""
        return SemiBatch(
            self.labeled_x.to(device),
            self.labels.to(device),
            self.unlabeled_x.to(device),
        )


class M1M2Config(BaseModel):
    """Hyperparameters for :class:`M1M2VAE`.

    Attributes:
        z1_family: Family of the M1 latent ``z1``.
        z2_family: Family of the M2 latent ``z2``.
        input_dim: Dimension of the (flattened, binarized) inputs.
        z1_dim: Dimension of ``z1``.
        z2_dim: Dimension of ``z2``.
        hidden_dim: Width of every network's single hidden layer.
        num_classes: Number of classes.
        alpha: Classification weight, multiplied by ``num_examples`` (the paper's
            alpha is in [0.1, 1]).
        num_examples: Total number of training examples ``N`` in ``alpha * N``.
        kappa_init: ``"reference"`` keeps the default initialization of every vMF
            concentration head (kappa near 1.7). ``"dimension"`` starts each vMF head
            at kappa equal to its latent dimension. The reference start is too noisy
            at high dimension and collapses the model (see
            ``ConvVonMisesFisherVAEConfig``).
    """

    z1_family: LatentFamily = "vmf"
    z2_family: LatentFamily = "gaussian"
    input_dim: int = 784
    z1_dim: int = 10
    z2_dim: int = 10
    hidden_dim: int = 500
    num_classes: int = 10
    alpha: float = 0.5
    num_examples: int = 50_000
    kappa_init: Literal["reference", "dimension"] = "reference"


@register_model("m1m2_vae", config_cls=M1M2Config)
class M1M2VAE(nn.Module):
    """The stacked M1+M2 model with pluggable latent families."""

    def __init__(self, config: M1M2Config) -> None:
        """Initializes the model from ``config``.

        Args:
            config: Hyperparameters; see :class:`M1M2Config`.
        """
        super().__init__()
        self.config = config
        hidden, classes = config.hidden_dim, config.num_classes
        self.encoder1 = _mlp(
            config.input_dim, hidden, head_size(config.z1_family, config.z1_dim)
        )
        self.decoder1 = _mlp(config.z1_dim, hidden, config.input_dim)
        self.classifier = _mlp(config.z1_dim, hidden, classes)
        self.encoder2 = _mlp(
            config.z1_dim + classes, hidden, head_size(config.z2_family, config.z2_dim)
        )
        self.decoder2 = _mlp(
            config.z2_dim + classes, hidden, head_size(config.z1_family, config.z1_dim)
        )
        if config.kappa_init == "dimension":
            self._start_kappas_at_dimension()

    def training_step(
        self, batch: SemiBatch, kl_weight: float = 1.0
    ) -> dict[str, Tensor]:
        """Computes the semi-supervised loss for one batch.

        Args:
            batch: Labelled and unlabelled examples; either part may be empty (a
                validation batch has no labelled examples).
            kl_weight: Unused; accepted so the model works with :class:`Trainer`.

        Returns:
            A dict with ``"loss"`` (the batch-averaged objective), ``"labeled_elbo"``,
            ``"unlabeled_elbo"`` (mean per example, or 0 if that part is empty),
            ``"classification_loss"`` and ``"accuracy"`` (of the labelled part).
        """
        del kl_weight
        zero = torch.zeros((), device=batch.unlabeled_x.device)
        total = -zero
        outputs = {
            "labeled_elbo": zero,
            "unlabeled_elbo": zero,
            "classification_loss": zero,
            "accuracy": zero,
        }
        if batch.labeled_x.shape[0] > 0:
            z1, base = self._sample_z1(batch.labeled_x)
            logits = self.classifier(z1)
            target = one_hot(batch.labels, self.config.num_classes).float()
            elbo = base + self._z2_terms(z1, target)
            classification = cross_entropy(logits, batch.labels, reduction="none")
            weight = self.config.alpha * self.config.num_examples
            total = total - elbo.sum() + weight * classification.sum()
            outputs["labeled_elbo"] = elbo.mean().detach()
            outputs["classification_loss"] = classification.mean().detach()
            outputs["accuracy"] = (
                (logits.argmax(-1) == batch.labels).float().mean().detach()
            )
        if batch.unlabeled_x.shape[0] > 0:
            z1, base = self._sample_z1(batch.unlabeled_x)
            probabilities = torch.softmax(self.classifier(z1), dim=-1)
            per_class = self._all_class_terms(z1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
            elbo = base + (probabilities * per_class).sum(-1) + entropy
            total = total - elbo.sum()
            outputs["unlabeled_elbo"] = elbo.mean().detach()
        return {"loss": total / len(batch), **outputs}

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Returns the predicted class of each input, from the centre of ``q(z1|x)``.

        Args:
            x: Inputs, shape ``(batch, input_dim)``.

        Returns:
            Integer classes, shape ``(batch,)``.
        """
        posterior = posterior_from_raw(
            self.config.z1_family, self.encoder1(x), self.config.z1_dim
        )
        centre = posterior_centre(self.config.z1_family, posterior)
        return self.classifier(centre).argmax(-1)

    def _start_kappas_at_dimension(self) -> None:
        """Sets every vMF head's kappa bias so it starts at its latent dimension."""
        heads = [
            (self.config.z1_family, self.encoder1, self.config.z1_dim),
            (self.config.z2_family, self.encoder2, self.config.z2_dim),
            (self.config.z1_family, self.decoder2, self.config.z1_dim),
        ]
        with torch.no_grad():
            for family, network, dimension in heads:
                if family == "vmf":
                    last: Any = network[-1]
                    last.bias[-1] = vmf_kappa_inverse(float(dimension))

    def _sample_z1(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Samples ``z1 ~ q(z1|x)``; returns it and ``A = log p(x|z1) - log q``."""
        posterior = posterior_from_raw(
            self.config.z1_family, self.encoder1(x), self.config.z1_dim
        )
        z1 = posterior.rsample()
        log_likelihood = -binary_cross_entropy_with_logits(
            self.decoder1(z1), x, reduction="none"
        ).sum(-1)
        return z1, log_likelihood - posterior.log_prob(z1)

    def _z2_terms(self, z1: Tensor, y: Tensor) -> Tensor:
        """Returns one-sample ``B_y`` per row; ``y`` is one-hot, aligned with ``z1``."""
        q2 = posterior_from_raw(
            self.config.z2_family,
            self.encoder2(torch.cat([z1, y], dim=-1)),
            self.config.z2_dim,
        )
        z2 = q2.rsample()
        p1 = posterior_from_raw(
            self.config.z1_family,
            self.decoder2(torch.cat([z2, y], dim=-1)),
            self.config.z1_dim,
        )
        prior2 = standard_prior(self.config.z2_family, self.config.z2_dim, z1.device)
        return (
            p1.log_prob(z1)
            + prior2.log_prob(z2)
            - q2.log_prob(z2)
            - math.log(self.config.num_classes)
        )

    def _all_class_terms(self, z1: Tensor) -> Tensor:
        """Returns ``B_y`` for every class, shape ``(batch, num_classes)``."""
        rows, classes = z1.shape[0], self.config.num_classes
        repeated = z1.repeat_interleave(classes, dim=0)
        labels = torch.arange(classes, device=z1.device).repeat(rows)
        terms = self._z2_terms(repeated, one_hot(labels, classes).float())
        return terms.view(rows, classes)


def _mlp(width_in: int, hidden: int, width_out: int) -> nn.Sequential:
    """One-hidden-layer ReLU MLP."""
    return nn.Sequential(
        nn.Linear(width_in, hidden), nn.ReLU(), nn.Linear(hidden, width_out)
    )
