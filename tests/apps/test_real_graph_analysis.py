"""End-to-end test of the real-graph analysis and figures on fabricated run records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from apps.real_graph import analyze, figures
from apps.real_graph import run as real_run

NODES = 120
PAIRS = 400


def _record(
    job: dict[str, Any], rng: np.random.Generator, edge: float
) -> dict[str, Any]:
    """A fabricated run record; TNBBeta gets a log-loss edge of ``edge`` nats."""
    base = {
        "vmf": 0.40,
        "power_spherical": 0.41,
        "gaussian": 0.42,
        "tnbbeta": 0.40 - edge,
    }
    log_loss = base[job["family"]] + rng.normal(0, 0.01)
    auc = 0.9 - (0.01 if job["family"] == "tnbbeta" else 0.0) + rng.normal(0, 0.005)
    metrics = {
        "log_loss": log_loss,
        "centre_log_loss": log_loss + 0.01,
        "sampled_auc": auc,
        "centre_auc": auc - 0.002,
        "ap": auc + 0.01,
        "ece": 0.05 + rng.normal(0, 0.005),
    }
    keep = job.get("keep") or 1.0
    return {
        "job": job,
        "graph_stats": {
            "num_nodes": 300.0 + job["seed"],
            "num_edges": 900.0 * keep,
            "mean_degree": 6.0 * keep + 0.01 * job["seed"],
            "degree_cv": 1.2,
            "transitivity": 0.1,
            "isolated_fraction": 0.01,
        },
        "best_epoch": 10,
        "diverged": 0.0,
        "val_log_loss": log_loss,
        "val": metrics,
        "test": metrics,
        "shape": {
            "second_mode_mass": 0.01,
            "frac_nodes_second_mode": 0.0,
            "m_mean": -1.0,
            "frac_m_negative": 0.8,
            "m_degree_spearman": 0.2,
            "second_mode_degree_spearman": 0.0,
            "p_mean": 0.9,
            "q_mean": 0.1,
            "epsilon_mean": 3.0,
        },
        "seconds": 1.0,
    }


def _populate(root: Path, rng: np.random.Generator) -> None:
    (root / "real_graph").mkdir(parents=True)
    tuned = {
        f"{d}|{f}|{fam}": {"lr": 0.01, "dropout": 0.0, "dim": 16, "val_log_loss": 0.4}
        for d in real_run.DATASETS
        for f in real_run.FEATURES
        for fam in real_run.FAMILIES
    }
    (root / "real_graph" / "tuned.json").write_text(json.dumps(tuned))
    degree = rng.integers(0, 30, size=NODES).astype(np.float64)
    for stage in ("main", "thin", "community", "pin"):
        for sub in ("runs", "pairs", "nodes"):
            (root / "real_graph" / stage / sub).mkdir(parents=True)
    jobs: list[dict[str, Any]] = []
    base = {
        "keep": None,
        "community": None,
        "pin": None,
        "lr": 0.01,
        "dropout": 0.0,
        "dim": 16,
    }
    for d in real_run.DATASETS:
        for f in real_run.FEATURES:
            for fam in real_run.FAMILIES:
                for s in real_run.FINAL_SEEDS:
                    jobs.append(
                        {
                            **base,
                            "stage": "main",
                            "dataset": d,
                            "features": f,
                            "family": fam,
                            "seed": s,
                        }
                    )
                    for keep in real_run.KEEP_LEVELS:
                        jobs.append(
                            {
                                **base,
                                "stage": "thin",
                                "dataset": d,
                                "features": f,
                                "family": fam,
                                "seed": s,
                                "keep": keep,
                            }
                        )
            for s in real_run.FINAL_SEEDS:
                for fraction in real_run.PIN_FRACTIONS:
                    jobs.append(
                        {
                            **base,
                            "stage": "pin",
                            "dataset": d,
                            "features": f,
                            "family": "tnbbeta",
                            "seed": s,
                            "pin": fraction,
                        }
                    )
            if d in real_run.COMMUNITY_DATASETS:
                for c in range(5):
                    for fam in real_run.FAMILIES:
                        for s in real_run.COMMUNITY_SEEDS:
                            jobs.append(
                                {
                                    **base,
                                    "stage": "community",
                                    "dataset": d,
                                    "features": f,
                                    "family": fam,
                                    "seed": s,
                                    "community": c,
                                }
                            )
    for job in jobs:
        job["run_id"] = real_run.run_id(job)
        edge = 0.02 + 0.01 * (job.get("keep") or 1.0)
        record = _record(job, rng, edge)
        folder = root / "real_graph" / job["stage"]
        (folder / "runs" / f"{job['run_id']}.json").write_text(json.dumps(record))
        if job["stage"] == "main":
            labels = np.tile([1.0, 0.0], PAIRS // 2)
            np.savez(
                folder / "pairs" / f"{job['run_id']}.npz",
                pairs=rng.integers(0, NODES, size=(2, PAIRS)),
                labels=labels,
                sampled=rng.uniform(0.05, 0.95, size=PAIRS).astype(np.float32),
                centre=rng.uniform(0.05, 0.95, size=PAIRS).astype(np.float32),
                spread=rng.uniform(0, 0.1, size=PAIRS).astype(np.float32),
                bucket=rng.integers(0, 5, size=PAIRS),
            )
        if job["stage"] in ("main", "pin"):
            nodes: dict[str, np.ndarray] = {
                "degree": degree,
                "second_mode_mass": rng.uniform(0, 0.02, NODES),
            }
            if job["family"] == "tnbbeta":
                nodes["m"] = rng.normal(-1, 1, NODES)
            np.savez(folder / "nodes" / f"{job['run_id']}.npz", **cast("Any", nodes))


@pytest.fixture
def fabricated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path))
    _populate(tmp_path, np.random.default_rng(0))
    return tmp_path


def test_analysis_and_figures_run_end_to_end(fabricated: Path, tmp_path: Path) -> None:
    analyze.main(["--workers", "2", "--permutations", "200"])
    results = json.loads((fabricated / "real_graph" / "analysis.json").read_text())
    cells = results["main"]["cells"]
    assert len(cells) == 8
    primary = cells["cora|on"]["comparisons"]["log_loss"]["primary"]
    assert primary["mean"] == pytest.approx(0.03, abs=0.01)
    assert primary["label"] in {
        "confirmed win",
        "significant win, below margin",
        "not significant",
    }
    assert 0 < primary["p"] <= 1
    auc = cells["cora|on"]["comparisons"]["sampled_auc"]["primary"]
    assert auc["mean"] < 0  # fabricated: TNBBeta loses AUROC
    assert len(results["buckets"]["cells"]["cora|on"]["buckets"]) == 5
    thin = results["thin"]
    assert thin["tests"]["on"]["log_loss"]["combined_spearman"] > 0
    assert set(results["shape"]["pin"]["cora|on"]) == {
        f"{f:g}" for f in real_run.PIN_FRACTIONS
    }
    assert results["community"]["pubmed|off"]["rows"]
    assert (
        (fabricated / "real_graph" / "analysis.md").read_text().startswith("## Stage 1")
    )
    written = figures.render_all(
        results, tmp_path / "figures", prefix="real_graph_test"
    )
    assert len(written) == 6
    assert all(path.exists() for path in written)


def test_paired_summary_secondary_margin() -> None:
    advantage = np.full(10, 0.03)
    out = analyze.paired_summary(advantage, margin=0.01, secondary_margin=0.05)
    assert out["p_win"] < 0.01  # 0.03 > 0.01 in every seed
    assert out["p_win2"] > 0.5  # 0.03 < 0.05 in every seed
    assert out["p_loss2"] == 1.0
