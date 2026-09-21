"""Tests for tnbbeta_vae.models.semi_supervised (the M1+M2 model)."""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest
import torch
from torch.nn.functional import cross_entropy

from tnbbeta_vae.models import M1M2VAE, M1M2Config, SemiBatch
from tnbbeta_vae.registry import build_model, list_registered_models

_FAMILIES = ["gaussian", "vmf", "tnbbeta"]


def _model(z1: Any = "vmf", z2: Any = "gaussian", **kwargs: Any) -> M1M2VAE:
    config = M1M2Config(
        z1_family=z1,
        z2_family=z2,
        input_dim=20,
        z1_dim=4,
        z2_dim=3,
        hidden_dim=16,
        num_examples=1000,
        **kwargs,
    )
    return M1M2VAE(config)


def _batch(labeled: int = 6, unlabeled: int = 5, seed: int = 0) -> SemiBatch:
    generator = torch.Generator().manual_seed(seed)
    return SemiBatch(
        torch.bernoulli(torch.rand(labeled, 20, generator=generator)),
        torch.randint(10, (labeled,), generator=generator),
        torch.bernoulli(torch.rand(unlabeled, 20, generator=generator)),
    )


def test_registered_and_buildable() -> None:
    assert "m1m2_vae" in list_registered_models()
    assert isinstance(build_model("m1m2_vae", z1_dim=4, z2_dim=3), M1M2VAE)


@pytest.mark.parametrize(("z1", "z2"), list(itertools.product(_FAMILIES, _FAMILIES)))
def test_every_family_pair_trains_and_predicts(z1: Any, z2: Any) -> None:
    torch.manual_seed(0)
    model = _model(z1, z2)
    batch = _batch()

    out = model.training_step(batch)
    out["loss"].backward()

    assert torch.isfinite(out["loss"])
    for module in (
        model.encoder1,
        model.decoder1,
        model.classifier,
        model.encoder2,
        model.decoder2,
    ):
        assert all(p.grad is not None for p in module.parameters())
    predictions = model.predict(batch.unlabeled_x)
    assert predictions.shape == (5,) and predictions.dtype == torch.long
    assert ((predictions >= 0) & (predictions < 10)).all()


def test_loss_matches_the_formula_when_the_random_parts_are_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    model = _model(alpha=0.2)
    batch = _batch(labeled=4, unlabeled=3)
    per_class = torch.arange(10.0) * 0.3 - 1.0  # B_y for y = 0..9
    base = {4: torch.tensor([1.0, 2.0, 3.0, 4.0]), 3: torch.tensor([-1.0, -2.0, -3.0])}
    z1_fixed = torch.randn(7, 4)

    monkeypatch.setattr(model, "_z2_terms", lambda _z1, y: y @ per_class)
    monkeypatch.setattr(
        model, "_sample_z1", lambda x: (z1_fixed[: x.shape[0]], base[x.shape[0]])
    )

    out = model.training_step(batch)

    z_l, z_u = z1_fixed[:4], z1_fixed[:3]
    labeled_elbo = base[4] + per_class[batch.labels]
    classification = cross_entropy(
        model.classifier(z_l), batch.labels, reduction="none"
    )
    probabilities = torch.softmax(model.classifier(z_u), dim=-1)
    entropy = -(probabilities * probabilities.log()).sum(-1)
    unlabeled_elbo = base[3] + probabilities @ per_class + entropy
    expected = (
        -labeled_elbo.sum() - unlabeled_elbo.sum() + 0.2 * 1000 * classification.sum()
    ) / 7
    assert torch.allclose(out["loss"], expected, atol=1e-4)
    assert torch.allclose(out["labeled_elbo"], labeled_elbo.mean())
    assert torch.allclose(out["unlabeled_elbo"], unlabeled_elbo.mean(), atol=1e-4)
    assert out["classification_loss"].item() == pytest.approx(
        classification.mean().item()
    )


def test_either_part_of_a_batch_may_be_empty() -> None:
    torch.manual_seed(0)
    model = _model()
    full = _batch()
    only_unlabeled = SemiBatch(
        torch.zeros(0, 20), torch.zeros(0, dtype=torch.long), full.unlabeled_x
    )
    only_labeled = SemiBatch(full.labeled_x, full.labels, torch.zeros(0, 20))

    unlabeled = model.training_step(only_unlabeled)
    labeled = model.training_step(only_labeled)

    assert len(only_unlabeled) == 5 and len(only_labeled) == 6
    assert unlabeled["classification_loss"] == 0 and unlabeled["labeled_elbo"] == 0
    assert labeled["unlabeled_elbo"] == 0
    assert torch.isfinite(unlabeled["loss"]) and torch.isfinite(labeled["loss"])


def test_batch_moves_devices() -> None:
    moved = _batch().to(torch.device("cpu"))

    assert len(moved) == 11


@pytest.mark.parametrize("z1", _FAMILIES)
def test_learns_to_classify_separable_data_from_a_few_labels(z1: Any) -> None:
    torch.manual_seed(0)
    classes, dim = 10, 20
    patterns = torch.bernoulli(torch.full((classes, dim), 0.5))
    model = _model(z1, "gaussian", alpha=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def draw(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(classes, (n,))
        flips = torch.bernoulli(torch.full((n, dim), 0.05))
        return (patterns[labels] + flips) % 2, labels

    labeled_x, labels = draw(20)
    out: dict[str, torch.Tensor] = {}
    for _ in range(400):
        unlabeled_x, _ = draw(32)
        optimizer.zero_grad()
        out = model.training_step(SemiBatch(labeled_x, labels, unlabeled_x))
        out["loss"].backward()
        optimizer.step()

    # Two labels per class and a tiny network: expect far above chance (0.1), not
    # perfection, and a perfect fit of the labelled examples.
    test_x, test_labels = draw(500)
    accuracy = (model.predict(test_x) == test_labels).float().mean().item()
    assert accuracy > 0.5, accuracy
    assert out["accuracy"].item() > 0.9
    assert math.isfinite(accuracy)
