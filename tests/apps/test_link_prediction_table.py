"""Tests for apps.link_prediction.table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apps.link_prediction import table


@pytest.fixture(autouse=True)
def _isolated_checkpoint_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))


def _write_family_json(
    tmp_path: Path,
    dataset: str,
    family: str,
    configurations: list[dict[str, Any]],
) -> None:
    directory = tmp_path / "ckpt" / "link_prediction"
    directory.mkdir(parents=True, exist_ok=True)
    selected = max(configurations, key=lambda c: float(c["val_auc"]))
    (directory / f"{dataset}_{family}.json").write_text(
        json.dumps({"selected": selected, "configurations": configurations})
    )


def _config(
    latent_dim: int, val_auc: float, test_auc: float, test_ap: float
) -> dict[str, Any]:
    return {
        "latent_dim": latent_dim,
        "val_auc": val_auc,
        "val_ap": val_auc,
        "test_auc": test_auc,
        "test_auc_std": 0.01,
        "test_ap": test_ap,
        "test_ap_std": 0.02,
    }


def test_main_prints_the_selected_row_per_dataset_and_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_family_json(tmp_path, "cora", "gaussian", [_config(16, 0.9, 0.85, 0.80)])

    table.main(["--datasets", "cora"])

    output = capsys.readouterr().out
    assert "85.0 ± 1.0" in output
    assert "80.0 ± 2.0" in output


def test_main_prints_dash_for_a_missing_family(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table.main(["--datasets", "cora"])

    output = capsys.readouterr().out
    assert "| cora | AUC | - | - | - | - |" in output


def test_by_dimension_picks_the_best_lr_dropout_per_latent_dim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Two configurations at latent_dim=16 (different val_auc) and one at 32.
    _write_family_json(
        tmp_path,
        "cora",
        "gaussian",
        [
            _config(16, val_auc=0.80, test_auc=0.70, test_ap=0.60),
            _config(16, val_auc=0.95, test_auc=0.90, test_ap=0.85),  # better, kept
            _config(32, val_auc=0.50, test_auc=0.55, test_ap=0.45),
        ],
    )

    table.main(["--datasets", "cora", "--by-dimension"])

    output = capsys.readouterr().out
    assert "dim=16: 90.0" in output
    assert "dim=32: 55.0" in output
    assert "dim=16: 70.0" not in output  # the worse latent_dim=16 config is dropped


def test_by_dimension_labels_each_cell_with_its_own_familys_latent_dim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Gaussian uses d=16 directly; a sphere family uses d+1=17 at the same manifold
    # dimension -- both should land in "row 0" (same list position) but keep their
    # own distinct latent_dim label.
    _write_family_json(tmp_path, "cora", "gaussian", [_config(16, 0.9, 0.80, 0.70)])
    _write_family_json(tmp_path, "cora", "vmf", [_config(17, 0.9, 0.82, 0.72)])

    table.main(["--datasets", "cora", "--by-dimension"])

    output = capsys.readouterr().out
    assert "dim=16: 80.0" in output
    assert "dim=17: 82.0" in output


def test_by_dimension_reports_no_runs_found_when_nothing_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table.main(["--datasets", "cora", "--by-dimension"])

    assert "(no runs found)" in capsys.readouterr().out
