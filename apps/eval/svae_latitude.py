"""TNBBeta's "cap vs ring" diagnostic: the distribution of p and q, MNIST test split.

Usage:
    uv run python -m apps.eval.svae_latitude --run-name NAME [--checkpoint final]

TNBBeta's median parameter p locates a posterior's latitude relative to its mean
direction: p near 0 or 1 is a tight cap centred at the mean direction, like a von
Mises-Fisher posterior; p away from the poles (nearer 0.5) is a ring at some angular
distance from it, a shape vMF cannot represent (its posterior is always a cap at the
mean direction). q separately controls how sharply that latitude is concentrated (q
near 1 can make it a thin ring or bimodal). Since vMF has no p or q, this diagnostic
only applies to ``conv_tnbbeta_spherical_vae`` runs; it exits without writing anything
for any other model.

Writes ``latitude_<checkpoint>.json`` (per-class and overall mean/std of p and q) and
``latitude_<checkpoint>.html`` (a p-vs-q scatter, one colour per class, plotly) next to
the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader

from tnbbeta_vae.data.mnist import load_mnist
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.plotting import TEMPLATE_NAME
from tnbbeta_vae.training import load_model_checkpoint

CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]  # fmt: skip
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]  # fmt: skip
_TNBBETA_MODEL = "conv_tnbbeta_spherical_vae"


def main(argv: list[str] | None = None) -> None:
    """Computes and plots the p/q diagnostic for one trained run.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)
    kind = checkpoint["model_name"]
    if kind != _TNBBETA_MODEL:
        print(
            f"{args.run_name}: model is {kind}, not {_TNBBETA_MODEL}; "
            "p and q have no analogue there, skipping."
        )
        return

    test_set = load_mnist(data_dir(), split="test")
    p, q = _latitude_parameters(model, test_set, args, device)
    labels = test_set.labels().numpy()

    results: dict[str, Any] = {
        "run_name": args.run_name,
        "p_mean": float(p.mean()),
        "p_std": float(p.std()),
        "q_mean": float(q.mean()),
        "q_std": float(q.std()),
        "per_class": {
            str(k): {
                "p_mean": float(p[labels == k].mean()),
                "p_std": float(p[labels == k].std()),
                "q_mean": float(q[labels == k].mean()),
                "q_std": float(q[labels == k].std()),
            }
            for k in range(10)
        },
    }
    output = run_dir / f"latitude_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")

    figure = _scatter_figure(p, q, labels, args.run_name, args.max_points, args.seed)
    html = run_dir / f"latitude_{args.checkpoint}.html"
    figure.write_html(html, include_plotlyjs=True)
    print(f"Wrote {html}")


@torch.no_grad()
def _latitude_parameters(
    model: Any, dataset: Any, args: argparse.Namespace, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    p_parts, q_parts = [], []
    for batch in loader:
        posterior, _ = model.posterior_and_prior(batch.to(device))
        p_parts.append(posterior.p.cpu())
        q_parts.append(posterior.q.cpu())
    return torch.cat(p_parts).numpy(), torch.cat(q_parts).numpy()


def _scatter_figure(
    p: np.ndarray,
    q: np.ndarray,
    labels: np.ndarray,
    title: str,
    max_points: int,
    seed: int,
) -> go.Figure:
    """A p-vs-q scatter, one trace per class, downsampled to ``max_points`` total."""
    keep = np.arange(len(labels))
    if len(keep) > max_points:
        keep = np.sort(
            np.random.default_rng(seed).choice(keep, max_points, replace=False)
        )
    p, q, labels = p[keep], q[keep], labels[keep]

    figure = go.Figure()
    for k, name in enumerate(CLASSES):
        selected = labels == k
        if not selected.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=p[selected],
                y=q[selected],
                mode="markers",
                name=name,
                marker={"size": 4, "color": COLORS[k], "opacity": 0.6},
            )
        )
    figure.update_layout(
        template=TEMPLATE_NAME,
        title=f"{title}: p (latitude) vs q (concentration), by class",
        xaxis={"title": "p", "range": [0, 1]},
        yaxis={"title": "q", "range": [0, 1]},
    )
    return figure


if __name__ == "__main__":
    main(sys.argv[1:])
