"""Tests for apps.synthetic.graph_families."""

from __future__ import annotations

import math

import pytest
import torch

from apps.synthetic.graph_families import (
    PRESETS,
    second_mode_mass,
    train_and_evaluate,
)
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
    assert set(result["posterior_by_block"]) == {"0", "1", "2"}
    assert 0.0 < result["posterior"]["r_bar"] <= 1.0
    assert ("frac_bimodal" in result["posterior"]) == (family == "tnbbeta")
    assert 0.0 <= result["posterior"]["second_mode_mass"] <= 0.5


class _FakePosterior:
    """Posterior stub drawing fixed-law samples about a fixed axis."""

    def __init__(self, direction: torch.Tensor, kind: str) -> None:
        self.mean_direction = direction
        self._kind = kind

    def rsample(self, shape: torch.Size) -> torch.Tensor:
        nodes, dim = self.mean_direction.shape
        if self._kind == "uniform":
            z = torch.randn(*shape, nodes, dim)
            return z / z.norm(dim=-1, keepdim=True)
        sign = torch.where(torch.rand(*shape, nodes, 1) < 0.5, -1.0, 1.0)
        z = sign * self.mean_direction + 0.01 * torch.randn(*shape, nodes, dim)
        return z / z.norm(dim=-1, keepdim=True)


def _axes(nodes: int, dim: int) -> torch.Tensor:
    axis = torch.randn(nodes, dim)
    return axis / axis.norm(dim=-1, keepdim=True)


def test_axis_abs_cos_of_a_uniform_posterior_matches_the_uniform_baseline() -> None:
    torch.manual_seed(0)
    dim = 16
    stats = second_mode_mass(_FakePosterior(_axes(300, dim), "uniform"), 64)

    assert stats["axis_abs_cos"] == pytest.approx(
        math.sqrt(2 / (math.pi * dim)), abs=0.03
    )
    assert stats["axis_abs_cos_node_spread"] < 0.05


def test_axis_abs_cos_of_a_tight_antipodal_mixture_is_one_with_half_second_mass() -> (
    None
):
    torch.manual_seed(0)
    stats = second_mode_mass(_FakePosterior(_axes(100, 16), "antipodal"), 64)

    assert stats["axis_abs_cos"] > 0.95
    assert stats["second_mode_mass"] == pytest.approx(0.5, abs=0.05)
