"""Tests for tnbbeta_vae.models.predictive."""

from __future__ import annotations

import pytest
import torch

from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.predictive import (
    balanced_truth,
    bernoulli_kl,
    bernoulli_log_loss,
    dense_edge_probabilities,
    expected_calibration_error,
)


def test_balanced_truth_equals_one_half_at_the_mean_density() -> None:
    probability = torch.full((5, 5), 0.2)
    probability.fill_diagonal_(0.0)

    # Every pair sits at the mean pair probability rho, so P / (P + rho) is 1/2.
    assert torch.allclose(balanced_truth(probability)[0, 1], torch.tensor(0.5))


def test_bernoulli_kl_is_zero_at_equality_and_positive_otherwise() -> None:
    p = torch.tensor([0.1, 0.5, 0.9])

    assert bernoulli_kl(p, p) == pytest.approx(0.0, abs=1e-6)
    assert bernoulli_kl(p, p.flip(0)) > 0.1


def test_log_loss_matches_the_closed_form() -> None:
    loss = bernoulli_log_loss(torch.tensor([0.8, 0.3]), torch.tensor([1.0, 0.0]))

    assert loss == pytest.approx(
        -(torch.tensor(0.8).log() + torch.tensor(0.7).log()) / 2
    )


def test_ece_is_zero_when_calibrated_and_large_when_confidently_wrong() -> None:
    target = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    calibrated = torch.where(target == 1, 1.0, 0.0)
    wrong = 1.0 - calibrated

    assert expected_calibration_error(calibrated, target) == pytest.approx(
        0.0, abs=1e-6
    )
    assert expected_calibration_error(wrong.clamp(0.01, 0.99), target) > 0.9


@pytest.mark.parametrize("family", ["gaussian", "vmf", "power_spherical", "tnbbeta"])
def test_dense_probabilities_are_valid_for_centres_and_samples(family: str) -> None:
    n = 12
    model = GraphVAE(GraphVAEConfig(family=family, in_features=n, latent_dim=4))  # type: ignore[arg-type]
    batch = GraphBatch(
        features=torch.eye(n),
        norm_adjacency=torch.eye(n).to_sparse(),
        positive_edges=torch.tensor([[0, 1], [1, 2]]),
    )

    centre = dense_edge_probabilities(model, batch, num_samples=None)
    sampled = dense_edge_probabilities(model, batch, num_samples=8)

    for probability in (centre, sampled):
        assert probability.shape == (n, n)
        assert torch.allclose(probability, probability.T, atol=1e-5)
        assert ((probability >= 0) & (probability <= 1)).all()
