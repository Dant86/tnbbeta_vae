"""Tests for apps/data/download_planetoid.py."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from apps.data import download_planetoid


def _fake_urlretrieve(url: str, target: Path) -> None:
    name = Path(url).name  # ind.cora.<suffix>
    suffix = name.removeprefix("ind.cora.")
    features = sp.csr_matrix(np.eye(6, 3, dtype=np.float32))
    values = {
        "x": features[:2],
        "allx": features[:4],
        "tx": features[4:],
        "graph": {node: [(node + 1) % 6] for node in range(6)},
    }
    if suffix == "test.index":
        Path(target).write_text("4\n5")
    else:
        with Path(target).open("wb") as file:
            pickle.dump(values[suffix], file)


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

    monkeypatch.setattr(download_planetoid, "urlretrieve", recording)

    download_planetoid.main(["--datasets", "cora"])
    first = len(calls)
    download_planetoid.main(["--datasets", "cora"])

    assert first == 5 and len(calls) == first
    assert calls[0].startswith(download_planetoid.PLANETOID_URL)
    assert "cora: 6 nodes, 6 edges, 3 features OK" in capsys.readouterr().out
    assert (tmp_path / "planetoid" / "ind.cora.graph").exists()
