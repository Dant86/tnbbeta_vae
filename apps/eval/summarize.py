"""Prints a markdown table comparing evaluated runs.

Usage:
    uv run python -m apps.eval.summarize RUN_NAME [RUN_NAME ...]

Reads, from each run's checkpoint directory, whichever of these exist:
``eval_final_test.json`` and ``eval_final_train.json`` (``apps.eval.main``),
``fid_final.json`` (``apps.eval.fid``) and ``latent_probe_final_test.json``
(``apps.eval.export_latents``). Missing files show as ``-``. The gap columns
are train minus test PSNR (dB) and test minus train ELBO (nats per image).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from tnbbeta_vae.paths import checkpoint_dir

_COLUMNS = [
    ("epochs", "test", "epochs_completed", "{:d}"),
    ("sigma", "test", "likelihood_scale", "{:.4f}"),
    ("test ELBO", "test", "elbo", "{:.1f}"),
    ("test KL", "test", "kl", "{:.1f}"),
    ("test PSNR", "test", "psnr_db", "{:.2f}"),
    ("prior NN ratio", "test", "prior_nn_ratio", "{:.2f}"),
    ("FID prior", "fid", "fid_prior", "{:.2f}"),
    ("FID floor", "fid", "fid_real_floor", "{:.2f}"),
    ("kNN cosine", "probe", "direction_cosine", "{:.3f}"),
    ("linear", "probe", "linear_direction", "{:.3f}"),
]


def main(argv: list[str] | None = None) -> None:
    """Prints one table row per run name.

    Args:
        argv: Run names, defaulting to ``sys.argv[1:]``.
    """
    run_names = sys.argv[1:] if argv is None else argv
    if not run_names:
        raise SystemExit("Give at least one run name.")
    header = ["run", *(c[0] for c in _COLUMNS), "PSNR gap (train-test)", "ELBO gap"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name in run_names:
        print("| " + " | ".join(_row(name)) + " |")


def _row(run_name: str) -> list[str]:
    run_dir = checkpoint_dir() / run_name
    files = {
        "test": _load(run_dir / "eval_final_test.json"),
        "train": _load(run_dir / "eval_final_train.json"),
        "fid": _load(run_dir / "fid_final.json"),
        "probe": _load(run_dir / "latent_probe_final_test.json"),
    }
    cells = [run_name]
    for _, source, key, fmt in _COLUMNS:
        cells.append(_format(files[source], key, fmt))
    train, test = files["train"], files["test"]
    psnr_gap = _difference(train, test, "psnr_db")
    elbo_gap = _difference(test, train, "elbo")
    cells += [_format_value(psnr_gap, "{:.2f}"), _format_value(elbo_gap, "{:.1f}")]
    return cells


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _format(source: dict[str, Any], key: str, fmt: str) -> str:
    return _format_value(source.get(key), fmt)


def _format_value(value: float | None, fmt: str) -> str:
    return "-" if value is None else fmt.format(value)


def _difference(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    if key not in a or key not in b:
        return None
    return a[key] - b[key]


if __name__ == "__main__":
    main()
