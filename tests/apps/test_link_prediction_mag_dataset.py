"""Tests for apps.link_prediction.main's MAG co-authorship dataset dispatch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from apps.link_prediction import main as link_prediction_main
from tnbbeta_vae.data.snap_community import Graph as SnapGraph
from tnbbeta_vae.paths import checkpoint_dir

_NUM_NODES = 20


def _fake_mag_graph(*_args: object, **_kwargs: object) -> SnapGraph:
    # split_edges floor()s its val/test fractions to an edge count, so a graph this
    # small needs enough edges that both floors are still >= 1 (a bare ring's 8 edges
    # round down to 0/0) -- but NOT a complete graph, which leaves zero non-edges for
    # split_edges' negative sampler to draw from (an infinite loop). A moderately
    # dense Erdos-Renyi-style graph has both.
    upper = sp.random(_NUM_NODES, _NUM_NODES, density=0.3, rng=0, format="coo")
    adjacency = ((upper + upper.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    rng = np.random.default_rng(0)
    features = sp.csr_matrix(
        rng.integers(0, 5, size=(_NUM_NODES, 5)).astype(np.float32)
    )
    node_id_map = {node: node for node in range(_NUM_NODES)}
    return SnapGraph(adjacency, features, node_id_map)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setattr(link_prediction_main, "load_mag_coauthor", _fake_mag_graph)


def test_dataset_mag_cs_is_a_valid_choice_dispatching_to_load_mag_coauthor(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    original = link_prediction_main.load_mag_coauthor

    def recording(root: Path, name: str) -> SnapGraph:
        calls.append(name)
        return original(root, name)

    link_prediction_main.load_mag_coauthor = recording  # type: ignore[assignment]

    link_prediction_main.main(
        [
            "--dataset",
            "mag_cs",
            "--family",
            "tnbbeta",
            "--lrs",
            "0.01",
            "--dropouts",
            "0",
            "--latent-dims",
            "3",
            "--epochs",
            "2",
            "--seeds",
            "0",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    assert calls == ["cs"]
    output = checkpoint_dir() / "link_prediction" / "mag_cs_tnbbeta.json"
    assert output.exists()


@pytest.mark.parametrize("dataset", ["mag_cs", "mag_eng", "mag_chem", "mag_med"])
def test_all_four_mag_subjects_are_valid_dataset_choices(
    dataset: str, tmp_path: Path
) -> None:
    link_prediction_main.main(
        [
            "--dataset",
            dataset,
            "--family",
            "gaussian",
            "--lrs",
            "0.01",
            "--dropouts",
            "0",
            "--latent-dims",
            "3",
            "--epochs",
            "1",
            "--seeds",
            "0",
            "--device",
            "cpu",
        ]  # fmt: skip
    )

    output = checkpoint_dir() / "link_prediction" / f"{dataset}_gaussian.json"
    assert output.exists()
