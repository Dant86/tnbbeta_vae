"""Tests for apps.eval.neighbor_heterogeneity_correlation (tiny fake graphs only).

No real cluster checkpoints exist locally for this script's actual datasets
(com-DBLP, MAG CS) -- see this module's own docstring and the dated write-up for the
exact cluster command. These tests mock the graph/community loaders and build tiny
checkpoints directly, exactly like ``apps/eval/graph_posterior_shape.py``'s tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import scipy.sparse as sp
import torch

from apps.eval import neighbor_heterogeneity_correlation as nhc
from tnbbeta_vae.data.snap_community import Graph as SnapGraph
from tnbbeta_vae.models import GraphVAE, GraphVAEConfig


def _fake_snap_graph(*_args: object, **_kwargs: object) -> SnapGraph:
    # 6 nodes, two triangles (0,1,2) and (3,4,5), a single bridging edge (2, 3).
    num_nodes = 6
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    rows = [a for a, b in edges] + [b for a, b in edges]
    cols = [b for a, b in edges] + [a for a, b in edges]
    data = [1.0] * len(rows)
    adjacency = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    features = sp.identity(num_nodes, format="csr", dtype="float32")
    node_id_map = {i: i for i in range(num_nodes)}
    return SnapGraph(adjacency, features, node_id_map)


def _fake_snap_communities(*_args: object, **_kwargs: object) -> dict[int, set[int]]:
    # Raw SNAP-ID space happens to equal internal IDs here (identity node_id_map).
    return {0: {0}, 1: {0}, 2: {0}, 3: {1}, 4: {1}, 5: {1}}


def _fake_mag_graph(*_args: object, **_kwargs: object) -> SnapGraph:
    return _fake_snap_graph()


def _fake_mag_communities(*_args: object, **_kwargs: object) -> dict[int, set[int]]:
    return _fake_snap_communities()


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(nhc, "load_snap_community", _fake_snap_graph)
    monkeypatch.setattr(nhc, "load_snap_communities", _fake_snap_communities)
    monkeypatch.setattr(nhc, "load_mag_coauthor", _fake_mag_graph)
    monkeypatch.setattr(nhc, "load_mag_communities", _fake_mag_communities)


def _write_checkpoint(tmp_path: Path, run_name: str, config: GraphVAEConfig) -> None:
    model = GraphVAE(config)
    run_dir = tmp_path / "ckpt" / run_name
    run_dir.mkdir(parents=True)
    torch.save(
        {
            "model_name": "graph_vae",
            "config": config.model_dump(),
            "model_state_dict": model.state_dict(),
        },
        run_dir / "final.pt",
    )


def test_writes_correlation_summary_for_tnbbeta(tmp_path: Path) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "tnbbeta"), in_features=6, hidden_dim=8, latent_dim=3
    )
    _write_checkpoint(tmp_path, "dblp_tnb_smoke", config)

    nhc.main(
        [
            "--run-name",
            "dblp_tnb_smoke",
            "--dataset",
            "dblp",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    output = tmp_path / "ckpt" / "dblp_tnb_smoke" / "neighbor_heterogeneity_final.json"
    results = json.loads(output.read_text())
    assert results["num_nodes"] == 6
    assert {"pearson_r", "spearman_r", "m_mean", "heterogeneity_mean"} <= set(results)
    assert -1.0 <= results["pearson_r"] <= 1.0
    assert -1.0 <= results["spearman_r"] <= 1.0

    html = tmp_path / "ckpt" / "dblp_tnb_smoke" / "neighbor_heterogeneity_final.html"
    assert html.exists()


def test_skips_correlation_for_a_non_tnbbeta_family(tmp_path: Path) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "vmf"), in_features=6, hidden_dim=8, latent_dim=3
    )
    _write_checkpoint(tmp_path, "dblp_vmf_smoke", config)

    nhc.main(
        [
            "--run-name",
            "dblp_vmf_smoke",
            "--dataset",
            "dblp",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    output = tmp_path / "ckpt" / "dblp_vmf_smoke" / "neighbor_heterogeneity_final.json"
    results = json.loads(output.read_text())
    assert "pearson_r" not in results
    assert "m_mean" not in results
    html = tmp_path / "ckpt" / "dblp_vmf_smoke" / "neighbor_heterogeneity_final.html"
    assert not html.exists()


def test_works_on_a_mag_dataset_too(tmp_path: Path) -> None:
    config = GraphVAEConfig(
        family=cast("Any", "tnbbeta"), in_features=6, hidden_dim=8, latent_dim=3
    )
    _write_checkpoint(tmp_path, "mag_cs_tnb_smoke", config)

    nhc.main(
        [
            "--run-name",
            "mag_cs_tnb_smoke",
            "--dataset",
            "mag_cs",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    output = (
        tmp_path / "ckpt" / "mag_cs_tnb_smoke" / "neighbor_heterogeneity_final.json"
    )
    results = json.loads(output.read_text())
    assert results["dataset"] == "mag_cs"
    assert "pearson_r" in results
