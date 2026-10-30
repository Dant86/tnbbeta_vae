"""Posterior-predictive edge probabilities and calibration metrics for ``GraphVAE``.

Centre-based link-prediction AUC (``apps.link_prediction.main``) never sees a
posterior's shape. These helpers score what the *distribution* implies: the edge
probability averaged over posterior samples, its log loss and calibration, and (for
synthetic graphs with a known truth) its divergence from the true edge probabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tnbbeta_vae.models.graph_vae import GraphBatch, GraphVAE

__all__ = [
    "balanced_truth",
    "bernoulli_kl",
    "bernoulli_log_loss",
    "dense_edge_probabilities",
    "expected_calibration_error",
]


@torch.no_grad()
def dense_edge_probabilities(
    model: GraphVAE, batch: GraphBatch, *, num_samples: int | None
) -> torch.Tensor:
    """Returns every ordered pair's edge probability, shape ``(n, n)``.

    Args:
        model: A trained model (evaluation mode is set here).
        batch: The graph the encoder runs over (use the *training* adjacency).
        num_samples: Posterior samples to average the pair probability over, or ``None``
            to score the posterior centres only (the plug-in probability).

    Returns:
        ``mean_s sigmoid(logit(z_i^s, z_j^s))``, or ``sigmoid(logit(c_i, c_j))`` for
        centres ``c``. The diagonal is not meaningful.
    """
    model.eval()
    scale = (
        torch.ones(())
        if model.config.family == "gaussian"
        else model.temperature().detach().cpu()
    )
    if num_samples is None:
        centres = model.embeddings(batch).cpu()
        return torch.sigmoid(scale * centres @ centres.T)
    posterior, _ = model.posterior_and_prior(batch)
    samples = posterior.rsample(torch.Size([num_samples])).cpu()
    total = torch.zeros(samples.shape[1], samples.shape[1])
    for sample in samples:
        total += torch.sigmoid(scale * sample @ sample.T)
    return total / num_samples


def balanced_truth(probability: torch.Tensor) -> torch.Tensor:
    """Maps true edge probabilities to what 1:1 negative sampling teaches a model.

    Training pairs positives (drawn proportionally to the true probability) with equally
    many uniformly random pairs, so the Bayes-optimal classifier outputs
    ``P / (P + rho)`` with ``rho`` the mean pair probability, not ``P`` itself.

    Args:
        probability: True pair probabilities, shape ``(n, n)``, zero diagonal.

    Returns:
        ``P / (P + rho)``, the same shape.
    """
    num_nodes = probability.shape[0]
    rho = probability.sum() / (num_nodes * (num_nodes - 1))
    return probability / (probability + rho)


def bernoulli_log_loss(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Mean negative log likelihood of binary ``target`` under ``prediction``."""
    p = prediction.clamp(eps, 1 - eps)
    return -(target * p.log() + (1 - target) * (1 - p).log()).mean()


def bernoulli_kl(
    truth: torch.Tensor, prediction: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Mean KL(Bernoulli(truth) || Bernoulli(prediction)) over elements."""
    t = truth.clamp(eps, 1 - eps)
    p = prediction.clamp(eps, 1 - eps)
    return (t * (t / p).log() + (1 - t) * ((1 - t) / (1 - p)).log()).mean()


def expected_calibration_error(
    prediction: torch.Tensor, target: torch.Tensor, bins: int = 10
) -> torch.Tensor:
    """Equal-width-bin expected calibration error for binary ``target``."""
    edges = torch.linspace(0, 1, bins + 1)
    index = torch.bucketize(prediction.clamp(0, 1 - 1e-7), edges[1:-1])
    error = torch.zeros(())
    for b in range(bins):
        in_bin = index == b
        if in_bin.any():
            gap = (prediction[in_bin].mean() - target[in_bin].float().mean()).abs()
            error += in_bin.float().mean() * gap
    return error
