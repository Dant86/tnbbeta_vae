"""Tests for apps.synthetic.graph_families."""

from __future__ import annotations

import math

import pytest
import torch

from apps.synthetic.graph_families import PRESETS, train_and_evaluate
from tnbbeta_vae.data.block_graph import assortative_spec, block_graph
from tnbbeta_vae.models import GraphVAEConfig


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_samples_a_graph_with_edges_and_named_kinds(name: str) -> None:
    sample = PRESETS[name](0)

    assert sample.adjacency.nnz > 0
    assert sample.probability.shape == sample.kind.shape
    assert all(isinstance(kind, str) for kind in set(sample.kind.ravel()))


@pytest.mark.parametrize("family", ["vmf", "tnbbeta"])
def test_train_and_evaluate_reports_finite_metrics_for_centres_and_samples(
    family: str,
) -> None:
    sample = block_graph(assortative_spec(3, 25, p_in=0.4, p_out=0.02), seed=0)

    result = train_and_evaluate(
        sample,
        GraphVAEConfig(
            family=family,  # type: ignore[arg-type]
            in_features=75,
            latent_dim=5,
            fixed_kappa=5.0 if family == "vmf" else None,
        ),
        lr=0.01,
        epochs=15,
        seed=0,
        device=torch.device("cpu"),
        num_samples=4,
    )

    assert 0.5 < result["oracle_test_auc"] <= 1.0
    for name in ("centre", "sampled"):
        for metric in ("test_auc", "test_log_loss", "test_ece", "truth_kl"):
            assert math.isfinite(result[f"{name}_{metric}"])
        assert set(result[f"{name}_truth_kl_by_kind"]) == {"within", "between"}
    assert result["test_auc"] == result["centre_test_auc"]
