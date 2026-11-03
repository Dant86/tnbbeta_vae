"""Computes the graph-side covariates (Set G) of the metric-first graph-search study.

Usage:
    uv run python -m apps.eval.graph_search_covariates --stage discovery --draws 32 \
        --workers 16

Regenerates every (config, draw) graph of a stage from its seed (CPU only) and writes
``graph_search/metric_first/covariates_<stage>.json``:
``{config_id: {draw: {covariate: value}}}``. The seed of draw ``k`` of the config at
position ``i`` is ``SEED_BASE[stage] + 1000 * i + k``, exactly as in the run manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

from apps.synthetic.graph_search import SEED_BASE, stage_dir
from tnbbeta_vae.data.graph_covariates import draw_covariates
from tnbbeta_vae.data.graph_space import GraphSpec
from tnbbeta_vae.paths import checkpoint_dir


def main(argv: list[str] | None = None) -> None:
    """Computes and writes the covariate table of one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(SEED_BASE))
    parser.add_argument("--draws", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    print(f"wrote {compute(args.stage, args.draws, args.workers)}")


def covariate_path(stage: str) -> Path:
    """Returns the covariate table path of ``stage``."""
    return (
        checkpoint_dir() / "graph_search" / "metric_first" / f"covariates_{stage}.json"
    )


def compute(stage: str, draws: int, workers: int) -> Path:
    """Computes covariates for every config and draw of ``stage`` and writes them."""
    configs = json.loads((stage_dir(stage) / "configs.json").read_text())
    tasks = []
    for index, config in enumerate(configs):
        for draw in range(draws):
            tasks.append((config, SEED_BASE[stage] + 1000 * index + draw, draw))
    tasks.sort(key=lambda task: -task[0]["num_nodes"])
    table: dict[str, dict[str, dict[str, float]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (config, _, draw), result in zip(
            tasks, pool.map(_one, tasks, chunksize=4), strict=True
        ):
            table.setdefault(config["config_id"], {})[str(draw)] = result
    path = covariate_path(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table))
    return path


def _one(task: tuple[dict[str, Any], int, int]) -> dict[str, float]:
    config, seed, _ = task
    spec = GraphSpec(**{k: v for k, v in config.items() if k != "config_id"})
    return draw_covariates(spec, seed)


if __name__ == "__main__":
    main()
