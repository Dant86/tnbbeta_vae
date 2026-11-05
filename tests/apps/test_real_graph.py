"""Tests of the real-graph revision: thinning, buckets, scoring, tests, a tiny run."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from apps.real_graph import run as real_run
from apps.real_graph.stats import (
    bootstrap_mean_ci,
    sign_flip_p,
    within_block_permutation_p,
)
from tnbbeta_vae.data.planetoid import Graph
from tnbbeta_vae.models.real_graph_eval import (
    BUCKET_LABELS,
    degree_buckets,
    graph_statistics,
    pair_metrics,
    thin_edges,
    vmf_sampler_name,
)


def _ring(n: int = 12, chords: int = 4) -> Any:
    rows = list(range(n))
    cols = [(i + 1) % n for i in range(n)]
    rows += list(range(chords))
    cols += [(i + 5) % n for i in range(chords)]
    one: Any = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return ((one + one.T) > 0).astype(np.float32).tocsr()


def test_thin_edges_keeps_fraction_symmetric_and_is_seeded() -> None:
    adjacency = _ring(40, 20)
    edges = int(adjacency.nnz // 2)
    thinned = thin_edges(adjacency, 0.5, seed=3)
    assert thinned.shape == adjacency.shape
    assert (thinned != thinned.T).nnz == 0
    assert thinned.diagonal().sum() == 0
    assert thinned.nnz // 2 == round(edges * 0.5)
    assert (thinned.multiply(adjacency) != thinned).nnz == 0  # a subset of the edges
    again = thin_edges(adjacency, 0.5, seed=3)
    assert (again != thinned).nnz == 0
    other = thin_edges(adjacency, 0.5, seed=4)
    assert (other != thinned).nnz > 0
    assert (thin_edges(adjacency, 1.0, seed=0) != adjacency).nnz == 0
    with pytest.raises(ValueError):
        thin_edges(adjacency, 0.0, seed=0)


def test_graph_statistics_on_a_triangle_plus_a_pendant_and_isolate() -> None:
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
    one: Any = sp.csr_matrix(
        (np.ones(4), ([a for a, _ in edges], [b for _, b in edges])), shape=(5, 5)
    )
    adjacency = (one + one.T).tocsr()
    stats = graph_statistics(adjacency)
    assert stats["num_edges"] == 4
    assert stats["mean_degree"] == pytest.approx(8 / 5)
    assert stats["isolated_fraction"] == pytest.approx(0.2)
    # one triangle, connected triples: nodes of degree 2, 2, 3, 1 -> 1 + 1 + 3 + 0
    assert stats["transitivity"] == pytest.approx(3 * 1 / 5)


def test_degree_buckets_use_the_lower_endpoint() -> None:
    degree = np.array([0.0, 1.0, 3.0, 5.0, 9.0, 30.0])
    pairs = np.array([[0, 1, 2, 3, 4, 5], [5, 5, 5, 5, 5, 5]])
    buckets = degree_buckets(degree, pairs)
    assert buckets.tolist() == [0, 0, 1, 2, 3, 4]
    assert len(BUCKET_LABELS) == 5
    flipped = degree_buckets(degree, pairs[::-1])
    assert flipped.tolist() == buckets.tolist()


def test_pair_metrics_perfect_and_uninformative() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    perfect = pair_metrics(
        np.array([0.9, 0.8, 0.1, 0.2]), np.array([0.9, 0.8, 0.1, 0.2]), labels
    )
    assert perfect["sampled_auc"] == 1.0
    assert perfect["ap"] == 1.0
    assert perfect["log_loss"] == pytest.approx(-np.log([0.9, 0.8, 0.9, 0.8]).mean())
    flat = pair_metrics(np.full(4, 0.5), np.full(4, 0.5), labels)
    assert flat["sampled_auc"] == 0.5
    assert flat["log_loss"] == pytest.approx(np.log(2))


@pytest.mark.parametrize("alternative", ["two-sided", "greater"])
def test_sign_flip_matches_independent_enumeration(alternative: str) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 1.0, size=8)
    margin = 0.1
    shifted = values - margin
    observed = shifted.mean()
    count = 0
    total = 0
    for signs in itertools.product([-1.0, 1.0], repeat=len(values)):
        statistic = float(np.mean(np.array(signs) * shifted))
        total += 1
        if alternative == "two-sided":
            count += abs(statistic) >= abs(observed) - 1e-12
        else:
            count += statistic >= observed - 1e-12
    assert sign_flip_p(values, margin=margin, alternative=alternative) == pytest.approx(
        count / total
    )


def test_sign_flip_extremes() -> None:
    assert sign_flip_p(np.ones(10)) == pytest.approx(2 / 1024)  # +-all-ones
    assert sign_flip_p(np.ones(10), alternative="greater") == pytest.approx(1 / 1024)
    assert sign_flip_p(np.zeros(10)) == 1.0


def test_bootstrap_interval_contains_the_mean_and_is_deterministic() -> None:
    values = np.arange(10.0)
    low, high = bootstrap_mean_ci(values, resamples=2000)
    assert low < values.mean() < high
    assert (low, high) == bootstrap_mean_ci(values, resamples=2000)


def test_within_block_permutation_detects_a_trend_and_not_noise() -> None:
    rng = np.random.default_rng(1)
    covariate = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (10, 1))
    trend = covariate + rng.normal(0, 0.3, size=covariate.shape)
    observed, p, _ = within_block_permutation_p(
        [trend, trend], [covariate, covariate], permutations=2000
    )
    assert observed > 0.8
    assert p < 0.01
    noise = rng.normal(size=covariate.shape)
    _, p_noise, _ = within_block_permutation_p([noise], [covariate], permutations=500)
    assert p_noise > 0.05


def test_run_id_is_stable_and_distinguishes_jobs() -> None:
    job = {
        "stage": "thin",
        "dataset": "cora",
        "features": "off",
        "family": "vmf",
        "keep": 0.5,
        "community": None,
        "pin": None,
        "seed": 3,
    }
    assert real_run.run_id(job) == "thin_cora_off_vmf_keep0.5_s3"
    assert real_run.run_id({**job, "keep": 0.7}) != real_run.run_id(job)


def test_model_config_for_conditions() -> None:
    vmf = real_run.model_config_for("vmf", 10, 16, 0.0)
    assert vmf.latent_dim == 17
    assert vmf.initial_kappa == 17.0
    assert vmf.fixed_kappa is None
    assert vmf.decoder_bias
    assert real_run.model_config_for("gaussian", 10, 16, 0.0).latent_dim == 16
    pinned = real_run.model_config_for("tnbbeta", 10, 16, 0.0, pin_fraction=0.0)
    assert pinned.fixed_epsilon == pytest.approx((17 - 1) / 2)
    with pytest.raises(ValueError):
        real_run.model_config_for("vmf", 10, 16, 0.0, pin_fraction=0.0)


def _tiny_graph(n: int = 60, seed: int = 0) -> Graph:
    rng = np.random.default_rng(seed)
    block = np.arange(n) % 3
    upper = np.triu(
        (rng.random((n, n)) < np.where(block[:, None] == block[None, :], 0.25, 0.02)),
        k=1,
    )
    adjacency: Any = sp.csr_matrix((upper | upper.T).astype(np.float32))
    features: Any = sp.csr_matrix(np.eye(3, dtype=np.float32)[block])
    return Graph(adjacency, features)


@pytest.mark.parametrize("family", ["tnbbeta", "vmf", "power_spherical", "gaussian"])
def test_tiny_end_to_end_run_records_sampler_and_metrics(
    family: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(real_run, "_load_graph", lambda dataset: _tiny_graph())
    monkeypatch.setitem(real_run.EPOCHS, "cora", 8)
    job = {
        "stage": "main",
        "dataset": "cora",
        "features": "on",
        "family": family,
        "keep": 0.8,
        "community": None,
        "pin": None,
        "lr": 0.03,
        "dropout": 0.0,
        "dim": 4,
        "seed": 1,
    }
    job["run_id"] = real_run.run_id(job)
    result = real_run.run_job(job, torch.device("cpu"))
    assert result["diverged"] == 0.0
    expected_sampler = "exact_envelope" if family == "vmf" else "n/a"
    assert result["vmf_sampler"] == expected_sampler
    for key in ("log_loss", "sampled_auc", "centre_auc", "ap", "ece"):
        assert np.isfinite(result["test"][key])
        assert np.isfinite(result["val"][key])
    arrays = result["_arrays"]
    assert arrays["pairs"]["sampled"].shape == arrays["pairs"]["labels"].shape
    assert set(arrays) == {"pairs", "nodes"}
    assert result["graph_stats"]["num_nodes"] == 60
    if family == "tnbbeta":
        assert "m" in arrays["nodes"]
        assert 0.0 <= result["shape"]["frac_m_negative"] <= 1.0


def test_vmf_sampler_name_reports_the_flag() -> None:
    class Fake:
        exact_envelope = False

    assert vmf_sampler_name(Fake()) == "taylor"
    Fake.exact_envelope = True
    assert vmf_sampler_name(Fake()) == "exact_envelope"
    assert vmf_sampler_name(object()) == "n/a"


def test_vmf_runs_use_the_exact_sampler_by_default() -> None:
    from tnbbeta_vae.distributions import VonMisesFisher

    loc = torch.nn.functional.normalize(torch.randn(3, 5), dim=-1)
    assert VonMisesFisher(loc, torch.ones(3, 1)).exact_envelope is True
