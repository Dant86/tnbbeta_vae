"""Head-bias sweep on the graph-search pilot configs (exploratory).

Compares a GCN posterior head with and without a bias (``GraphVAEConfig.head_bias``),
and for TNBBeta with the raw-p bias initialised to ln 3 (``p_init_offset``), at
several learning rates. Every condition of one (family, draw) sees the same graph,
edge split and seed. Training and scoring use ``run_once`` and
``graph_search.evaluate``, unchanged.

Usage:
    python -m apps.synthetic.head_bias_sweep OUT.jsonl FAMILIES LRS DRAWS CONDS \
        [SHARD NUM_SHARDS]

Comma-separated lists. Resumable (finished cells in OUT.jsonl are skipped). The device
comes from ``select_device`` (exit code 75 without a GPU).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch

from apps.link_prediction.main import run_once
from apps.synthetic import graph_search as gs
from tnbbeta_vae.data.graph_space import GraphSpec, sample_graph
from tnbbeta_vae.data.planetoid import Graph, split_edges
from tnbbeta_vae.models import GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.heads import P_RAW_INIT
from tnbbeta_vae.training import select_device

# Condition -> (graph_search base condition, GraphVAEConfig overrides). The ``_def``
# vMF/PS arms start kappa at softplus(0) + 1 (the reference S-VAE start); ``vmf_init``
# is the existing arm that starts it at the ambient dimension. TNBBeta has no kappa.
CONDITIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "tnb": ("tnbbeta", {}),
    "tnb_hb": ("tnbbeta", {"head_bias": True}),
    "tnb_hb_p3": ("tnbbeta", {"head_bias": True, "p_init_offset": P_RAW_INIT}),
    "vmf_init": ("vmf_init", {}),
    "vmf_def": ("vmf_default", {}),
    "vmf_def_hb": ("vmf_default", {"head_bias": True}),
    "ps_def": ("power_spherical_default", {}),
    "ps_def_hb": ("power_spherical_default", {"head_bias": True}),
}
_SEED_BASE = 9_000_000  # graph_search's pilot stage: base + 1000 * config index + draw


def main(argv: list[str] | None = None) -> None:
    """Runs this shard's cells; see the module docstring for the arguments."""
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0])
    families, lrs = args[1].split(","), [float(x) for x in args[2].split(",")]
    draws, names = [int(x) for x in args[3].split(",")], args[4].split(",")
    shard, num_shards = (int(args[5]), int(args[6])) if len(args) > 6 else (0, 1)
    torch.set_num_threads(1)

    pilot = gs.stage_dir("pilot")
    configs = json.loads((pilot / "configs.json").read_text())
    done: set[tuple[str, int, float, str]] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            done.add((row["family"], row["draw"], row["lr"], row["cond"]))
    cells = [
        (family, draw, lr, name)
        for family in families
        for draw in draws
        for lr in lrs
        for name in names
    ]
    cells = [
        c for i, c in enumerate(cells) if i % num_shards == shard and c not in done
    ]
    device = select_device(None)
    cache: dict[tuple[str, int], Any] = {}
    for family, draw, lr, name in cells:
        index = next(
            i for i, c in enumerate(configs) if c["config_id"] == f"{family}-00"
        )
        seed = _SEED_BASE + 1000 * index + draw
        if (family, draw) not in cache:
            spec = GraphSpec(
                **{k: v for k, v in configs[index].items() if k != "config_id"}
            )
            sample = sample_graph(spec, seed)
            cache[(family, draw)] = (sample, split_edges(sample.adjacency, seed=seed))
        sample, split = cache[(family, draw)]
        row = run_cell(name, family, draw, lr, seed, sample, split, device)
        with out.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            family,
            draw,
            lr,
            name,
            f"esc {row['escape_epoch']} rbarF {row['r_bar_final']:.2f} "
            f"rbarBest {row['r_bar_best']:.2f} ep {row['best_epoch']} "
            f"auc {row['sampled_auc']:.3f} ll {row['ll']:.3f} "
            f"recal {row['recal_ll']:.3f} {row['seconds']:.0f}s",
            flush=True,
        )


def run_cell(
    name: str,
    family: str,
    draw: int,
    lr: float,
    seed: int,
    sample: Any,
    split: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Trains one condition on one graph and returns its row of metrics."""
    base, overrides = CONDITIONS[name]
    config: GraphVAEConfig = gs.model_config_for(
        base, int(sample.features.get_shape()[1])
    ).model_copy(update=overrides)
    trajectory: list[dict[str, float]] = []
    state: dict[str, torch.Tensor] = {}
    start = time.time()
    trained = run_once(
        Graph(sample.adjacency, sample.features),
        split,
        config,
        lr=lr,
        epochs=gs.EPOCHS,
        seed=seed,
        device=device,
        best_state=state,
        trajectory=trajectory,
    )
    model = GraphVAE(config).to(device)
    model.load_state_dict(state)
    metrics = gs.evaluate(model, sample, split, seed=seed)
    logged = [t for t in trajectory if t["epoch"] % 10 == 0 or t["epoch"] < 0]
    escape = next((int(t["epoch"]) for t in logged if t["r_bar"] > 0.3), None)
    last = trajectory[-1]
    return {
        "family": family,
        "draw": draw,
        "seed": seed,
        "lr": lr,
        "cond": name,
        "r_bar_init": trajectory[0]["r_bar"],
        "escape_epoch": escape,
        "r_bar_final": last["r_bar"],
        "final_epoch_logged": int(last["epoch"]),
        "r_bar_best": metrics["post_r_bar"],
        "kappa_init": trajectory[0].get("kappa_mean"),
        "kappa_final": last.get("kappa_mean"),
        "best_epoch": trained["best_epoch"],
        "val_auc": trained["val_auc"],
        "diverged": trained.get("diverged", 0.0),
        "centre_auc": metrics["centre_auc"],
        "sampled_auc": metrics["sampled_auc"],
        "ll": metrics["sampled_log_loss"],
        "recal_ll": metrics["recal_log_loss"],
        "truth_kl": metrics["truth_kl"],
        "post": {k: v for k, v in metrics.items() if k.startswith("post_")},
        "traj": [
            {
                k: t.get(k)
                for k in ("epoch", "r_bar", "val_auc", "p_mean", "epsilon_mean")
                + ("kappa_mean",)
            }
            for t in logged
            if t["epoch"] % 50 == 0 or t["epoch"] < 0
        ],
        "seconds": time.time() - start,
        "logit_ln3": math.log(3),
    }


if __name__ == "__main__":
    main()
