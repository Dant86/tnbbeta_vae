"""Interactive 3-D view of a TNBBeta latent at latent_dim=3 (apps/eval/export_latents.py).

Usage:
    uv run python notebooks/plot_sphere_3d.py \
        --npz ~/Downloads/d3/tnb_d3_clamp1e-6_seed0/latents_final_test.npz \
        --out-dir ~/Downloads/d3/html

Writes one self-contained ``<run>_sphere.html`` (the run name is the npz's parent
directory unless --tag is given). Open it in a browser and rotate.

* A faint unit sphere, and one point per image at its ``mode_direction`` (which
  undoes the antipodal alias), colored by CIFAR-10 class. Click a legend entry
  to hide/show a class, double-click to isolate it.
* Hover shows class, p, q, epsilon and KL.
* Points whose p is within --p-margin of 0.5 have an ill-defined mode direction
  (it flips between mu and -mu), so only their axis is meaningful. They are drawn
  as diamonds instead of circles.
* The dropdown switches the point color between class and p.
* Posterior samples ``z`` are an extra layer per class, hidden until you click
  "<class> (z)" in the legend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from tnbbeta_vae.plotting import TEMPLATE_NAME

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]  # fmt: skip
_HOVER = (
    "%{customdata[0]}<br>p=%{customdata[1]:.3f} q=%{customdata[2]:.3f}"
    "<br>epsilon=%{customdata[3]:.2f} KL=%{customdata[4]:.2f}%{customdata[5]}"
    "<extra></extra>"
)


def build_figure(
    data: dict[str, np.ndarray],
    title: str,
    max_points: int = 5000,
    p_margin: float = 0.1,
    seed: int = 0,
) -> go.Figure:
    """Builds the interactive sphere figure from an exported latents dict.

    Args:
        data: Arrays from ``latents_final_test.npz`` (``labels``, ``p``, ``q``,
            ``epsilon``, ``kl``, ``z`` and ``mode_direction`` or ``direction``).
        title: Figure title.
        max_points: Random subsample size (over all classes); larger inputs are
            downsampled to keep the HTML small.
        p_margin: Points with ``|p - 0.5| < p_margin`` are marked as axis-only.
        seed: Seed for the subsample.

    Returns:
        A plotly figure: one sphere mesh, then per class a point trace and a
        (initially hidden) posterior-sample trace.
    """
    mode = data["mode_direction"] if "mode_direction" in data else data["direction"]
    labels = data["labels"]
    keep = np.arange(len(labels))
    if len(keep) > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(keep, max_points, replace=False))
    mode, labels, z = mode[keep], labels[keep], data["z"][keep]
    p, q, eps, kl = (data[k][keep].astype(float) for k in ("p", "q", "epsilon", "kl"))
    ambiguous = np.abs(p - 0.5) < p_margin

    fig = go.Figure()
    fig.add_trace(_sphere_mesh())
    class_traces: list[int] = []
    for k, name in enumerate(CLASSES):
        sel = labels == k
        if not sel.any():
            continue
        note = np.where(ambiguous[sel], "<br>p near 0.5: axis only", "")
        custom = _hover_data(name, p[sel], q[sel], eps[sel], kl[sel], note)
        class_traces.append(len(fig.data))
        fig.add_trace(
            go.Scatter3d(
                x=mode[sel, 0], y=mode[sel, 1], z=mode[sel, 2], mode="markers",
                name=name, legendgroup=name, customdata=custom, hovertemplate=_HOVER,
                marker={
                    "size": 3, "color": COLORS[k], "opacity": 0.85,
                    "symbol": np.where(ambiguous[sel], "diamond", "circle"),
                    "colorscale": "Viridis", "cmin": 0, "cmax": 1,
                    "colorbar": {"title": "p", "len": 0.5},
                },
            )
        )  # fmt: skip
    for k, name in enumerate(CLASSES):
        sel = labels == k
        if not sel.any():
            continue
        custom = _hover_data(f"{name} (z)", p[sel], q[sel], eps[sel], kl[sel], np.full(sel.sum(), ""))
        fig.add_trace(
            go.Scatter3d(
                x=z[sel, 0], y=z[sel, 1], z=z[sel, 2], mode="markers",
                name=f"{name} (z)", legendgroup=f"{name} (z)", visible="legendonly",
                customdata=custom, hovertemplate=_HOVER,
                marker={"size": 2, "color": COLORS[k], "opacity": 0.5},
            )
        )  # fmt: skip

    by_class = [COLORS[k] for k in range(len(CLASSES)) if (labels == k).any()]
    by_p = [p[labels == k] for k in range(len(CLASSES)) if (labels == k).any()]
    show_scale = [i == 0 for i in range(len(class_traces))]
    fig.update_layout(
        template=TEMPLATE_NAME,
        title=f"{title} ({len(labels)} images, {int(ambiguous.sum())} with p near 0.5)",
        scene={
            "aspectmode": "cube",
            "xaxis": {"visible": False}, "yaxis": {"visible": False}, "zaxis": {"visible": False},
        },
        legend={"itemsizing": "constant"},
        updatemenus=[
            {
                "type": "dropdown", "x": 0.0, "y": 1.0, "xanchor": "left", "yanchor": "top",
                "buttons": [
                    {"label": "color: class", "method": "restyle",
                     "args": [{"marker.color": by_class, "marker.showscale": [False] * len(class_traces)}, class_traces]},
                    {"label": "color: p", "method": "restyle",
                     "args": [{"marker.color": by_p, "marker.showscale": show_scale}, class_traces]},
                ],
            }
        ],
    )  # fmt: skip
    return fig


def _hover_data(name: str, *columns: np.ndarray) -> np.ndarray:
    """Object array of (name, *columns) per row, so numbers keep their type."""
    return np.column_stack([np.full(len(columns[0]), name, dtype=object), *columns]).astype(object)


def _sphere_mesh(resolution: int = 40) -> go.Surface:
    theta = np.linspace(0, np.pi, resolution)
    phi = np.linspace(0, 2 * np.pi, 2 * resolution)
    return go.Surface(
        x=np.outer(np.sin(theta), np.cos(phi)),
        y=np.outer(np.sin(theta), np.sin(phi)),
        z=np.outer(np.cos(theta), np.ones_like(phi)),
        opacity=0.08, showscale=False, hoverinfo="skip", showlegend=False,
        colorscale=[[0, "#888888"], [1, "#888888"]],
    )  # fmt: skip


def main() -> None:
    """Loads an export and writes ``<tag>_sphere.html``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--p-margin", type=float, default=0.1)
    args = parser.parse_args()

    npz = np.load(args.npz)
    data = {key: npz[key] for key in npz.files if key != "model_name"}
    if data["z"].shape[1] != 3:
        raise SystemExit("This script is for latent_dim=3 TNBBeta exports.")
    if "p" not in data:
        raise SystemExit("Export has no TNBBeta parameters (p, q, epsilon).")
    tag = args.tag or args.npz.expanduser().parent.name
    fig = build_figure(data, tag, args.max_points, args.p_margin)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{tag}_sphere.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
