"""Correlates GCN neighbor-heterogeneity against TNBBeta's per-node bimodality.

Usage:
    uv run python -m apps.eval.neighbor_heterogeneity_correlation --run-name NAME \
        --dataset dblp|amazon|mag_cs|mag_eng|mag_chem|mag_med [--checkpoint final] \
        [--device cpu]

The live mechanistic theory
(``writeup/results/graph_bimodality_investigation_summary_2026-10-08.md``, section
5) is that a GCN's neighbor-averaging pulls a node's embedding toward a compromise
position that scores poorly against a *heterogeneous* neighborhood -- and that
TNBBeta's bimodal capacity lets training escape that compromise where a unimodal
vMF/Power Spherical cap cannot. This script tests that directly on one real,
community-labeled graph: for every node, it computes
``tnbbeta_vae.data.neighbor_heterogeneity.neighbor_heterogeneity`` (the Shannon
entropy of its neighbors' community labels) and ``m = posterior.epsilon -
(latent_dim - 1) / 2`` (negative iff the node's posterior is in the proven-bimodal
regime -- see ``tnbbeta_vs_power_spherical_expressivity.md``'s Theorem 5.1/Corollary
5.3), then reports the Pearson and Spearman correlation between them. The theory
predicts a *negative* correlation: higher neighbor heterogeneity -> more negative
``m`` -> more bimodal.

``epsilon``/``m`` have no analogue outside TNBBeta, so for any other family this
prints a message and skips the correlation -- the same pattern
``apps/eval/svae_latitude.py`` and ``apps/eval/graph_posterior_shape.py`` already use
for "doesn't apply to this family," not a crash.

Dataset loading mirrors ``apps/eval/graph_posterior_shape.py``'s dispatch (reusing
its ``graph_to_batch``), restricted to the datasets that actually carry community
labels: SNAP community graphs (``dblp``, ``amazon``, via
``tnbbeta_vae.data.snap_community``) and MAG co-authorship networks (``mag_cs``,
``mag_eng``, ``mag_chem``, ``mag_med``, via ``tnbbeta_vae.data.mag_coauthor``) --
unlike Planetoid, neither of which this script's community source covers.

This needs REAL cluster checkpoints (com-DBLP, MAG CS) that don't exist locally; its
tests use mocked tiny graphs/checkpoints only (see
``tests/apps/test_neighbor_heterogeneity_correlation.py``), the same convention as
this project's other ``apps/eval`` scripts. See the dated write-up in
``writeup/results/`` for the exact cluster command.

Writes ``neighbor_heterogeneity_<checkpoint>.json`` (and, for TNBBeta,
``neighbor_heterogeneity_<checkpoint>.html``, a heterogeneity-vs-``m`` plotly
scatter) next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr
import torch

from apps.eval.graph_posterior_shape import graph_to_batch
from tnbbeta_vae.data.mag_coauthor import (
    MAG_COAUTHOR_DATASETS,
    load_mag_coauthor,
    load_mag_communities,
)
from tnbbeta_vae.data.neighbor_heterogeneity import neighbor_heterogeneity
from tnbbeta_vae.data.snap_community import (
    SNAP_COMMUNITY_DATASETS,
    load_snap_communities,
    load_snap_community,
    remap_communities,
)
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.plotting import TEMPLATE_NAME
from tnbbeta_vae.training import load_model_checkpoint

__all__ = ["main"]

_DATASETS = (
    *SNAP_COMMUNITY_DATASETS,
    *(f"mag_{name}" for name in MAG_COAUTHOR_DATASETS),
)
_TNBBETA_FAMILY = "tnbbeta"


def main(argv: list[str] | None = None) -> None:
    """Computes and writes the heterogeneity-vs-``m`` correlation for one run.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", required=True, choices=list(_DATASETS))
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)

    graph, communities = _load_graph_and_communities(args.dataset)
    batch = graph_to_batch(graph, device)

    model.eval()
    with torch.no_grad():
        posterior, _ = model.posterior_and_prior(batch)  # type: ignore[attr-defined]

    family = checkpoint["config"].get("family")
    results: dict[str, Any] = {
        "run_name": args.run_name,
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "model_name": checkpoint["model_name"],
        "num_nodes": batch.num_nodes,
    }

    if family != _TNBBETA_FAMILY or not hasattr(posterior, "epsilon"):
        print(
            f"{args.run_name}: family is {family!r}, not {_TNBBETA_FAMILY!r}; "
            "m has no analogue there, skipping the correlation."
        )
        _write_json(run_dir, args.checkpoint, results)
        return

    heterogeneity = neighbor_heterogeneity(graph.adjacency, communities)
    latent_dim = checkpoint["config"]["latent_dim"]
    epsilon = posterior.epsilon.detach().cpu().numpy()  # type: ignore[attr-defined]
    m = epsilon - (latent_dim - 1) / 2

    valid = np.isfinite(heterogeneity) & np.isfinite(m)
    num_valid = int(valid.sum())
    if num_valid > 1:
        pearson_r, pearson_p = cast(
            "tuple[float, float]", pearsonr(heterogeneity[valid], m[valid])
        )
        spearman_r, spearman_p = cast(
            "tuple[float, float]", spearmanr(heterogeneity[valid], m[valid])
        )
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    results.update(
        {
            "num_valid_nodes": num_valid,
            "m_mean": float(np.mean(m)),
            "heterogeneity_mean": float(np.nanmean(heterogeneity)),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }
    )
    _write_json(run_dir, args.checkpoint, results)

    figure = _scatter_figure(heterogeneity[valid], m[valid], args.run_name)
    html = run_dir / f"neighbor_heterogeneity_{args.checkpoint}.html"
    figure.write_html(html, include_plotlyjs=True)
    print(f"Wrote {html}")


def _load_graph_and_communities(dataset: str) -> tuple[Any, dict[int, set[int]]]:
    """Loads a dataset's graph and node-ID-remapped community memberships.

    Args:
        dataset: One of :data:`_DATASETS`.

    Returns:
        ``(graph, communities)``, with ``communities`` in ``graph``'s internal
        ``0..n-1`` node-ID space.
    """
    if dataset.startswith("mag_"):
        name = dataset.removeprefix("mag_")
        graph = load_mag_coauthor(data_dir() / "mag_coauthor", name)
        communities = load_mag_communities(data_dir() / "mag_coauthor", name)
    else:
        graph = load_snap_community(data_dir() / "snap_community", dataset)
        raw_communities = load_snap_communities(data_dir() / "snap_community", dataset)
        communities = remap_communities(graph.node_id_map, raw_communities)
    return graph, communities


def _scatter_figure(heterogeneity: np.ndarray, m: np.ndarray, title: str) -> go.Figure:
    """A heterogeneity-vs-``m`` scatter over the valid (non-NaN) nodes."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=heterogeneity,
            y=m,
            mode="markers",
            marker={"size": 4, "opacity": 0.5},
        )
    )
    figure.update_layout(
        template=TEMPLATE_NAME,
        title=f"{title}: neighbor heterogeneity vs m",
        xaxis={"title": "Neighbor heterogeneity (entropy, nats)"},
        yaxis={"title": "m = epsilon - (latent_dim - 1) / 2"},
    )
    return figure


def _write_json(run_dir: Any, checkpoint: str, results: dict[str, Any]) -> None:
    """Writes and prints the JSON summary at the usual checkpoint-relative path."""
    output = run_dir / f"neighbor_heterogeneity_{checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main(sys.argv[1:])
