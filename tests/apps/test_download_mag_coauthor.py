"""Tests for apps/data/download_mag_coauthor.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from apps.data import download_mag_coauthor


def _fake_npz_bytes() -> dict[str, Any]:
    """A tiny but real-shaped SparseGraph payload: 6 nodes, 4 features, 3 classes."""
    rows = np.arange(6)
    cols = np.roll(rows, 1)
    adjacency = sp.coo_matrix((np.ones(6), (rows, cols)), shape=(6, 6))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    rng = np.random.default_rng(0)
    features = sp.csr_matrix(rng.integers(0, 5, size=(6, 4)).astype(np.float32))
    labels = sp.csr_matrix(
        np.array(
            [[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
            dtype=np.float32,
        )
    )
    return {
        "adj_matrix.data": adjacency.data,
        "adj_matrix.indices": adjacency.indices,
        "adj_matrix.indptr": adjacency.indptr,
        "adj_matrix.shape": np.array(adjacency.shape, dtype=np.int64),
        "attr_matrix.data": features.data,
        "attr_matrix.indices": features.indices,
        "attr_matrix.indptr": features.indptr,
        "attr_matrix.shape": np.array(features.shape, dtype=np.int64),
        "labels.data": labels.data,
        "labels.indices": labels.indices,
        "labels.indptr": labels.indptr,
        "labels.shape": np.array(labels.shape, dtype=np.int64),
        "node_names": np.array([f"n{i}" for i in range(6)]),
        "attr_names": np.array([f"kw{i}" for i in range(4)]),
        "edge_attr_matrix": None,
        "edge_attr_names": None,
        "class_names": np.array(["a", "b", "c"], dtype=object),
        "metadata": None,
        "type": "SparseGraph",
    }


def _fake_urlretrieve(url: str, target: Path) -> None:
    np.savez(target, **_fake_npz_bytes(), allow_pickle=True)
    # np.savez always appends .npz; urlretrieve's target already has that suffix, but
    # when Path.name already ends in .npz np.savez writes exactly to that path, so no
    # rename is needed -- asserted by the caller inspecting `target` directly.


def test_download_fetches_missing_files_verifies_and_skips_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path))
    calls: list[str] = []

    def recording(url: str, target: Path) -> None:
        calls.append(url)
        _fake_urlretrieve(url, target)

    monkeypatch.setattr(download_mag_coauthor, "urlretrieve", recording)

    download_mag_coauthor.main(["--datasets", "cs"])
    first = len(calls)
    download_mag_coauthor.main(["--datasets", "cs"])

    assert first == 1 and len(calls) == first
    assert calls[0] == f"{download_mag_coauthor._BASE_URL}/mag_cs.npz"
    out = capsys.readouterr().out
    assert "cs: 6 nodes, 6 edges, 4 features, 3 communities OK" in out
    assert (tmp_path / "mag_coauthor" / "mag_cs.npz").exists()


def test_download_defaults_to_all_four_mag_subjects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path))
    calls: list[str] = []

    def recording(url: str, target: Path) -> None:
        calls.append(url)
        _fake_urlretrieve(url, target)

    monkeypatch.setattr(download_mag_coauthor, "urlretrieve", recording)

    download_mag_coauthor.main([])

    assert len(calls) == 4
    for name in ("cs", "eng", "chem", "med"):
        assert (tmp_path / "mag_coauthor" / f"mag_{name}.npz").exists()
