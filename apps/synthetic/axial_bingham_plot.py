"""Figures for the axial Bingham study (plotly, project ``alptex`` template, 3x scale).

* ``gap_vs_sigma``: mean test ELBO gap against the true ``log p(X)`` versus the noise
  scale, one line per arm, rows ``d = 2, 3, 4``, columns decoders (error bars: seed sd).
* ``sphere``: for ``d = 3`` and the fixed decoder, draws of one test posterior from each
  arm (median-gap seed of ``tnbbeta``) next to draws from the exact Bingham posterior,
  on an orthographic view with the data's top eigenvector ``v1`` vertical.

Usage::

    uv run python -m apps.synthetic.axial_bingham_plot \\
        --out-dir ring_recovery/axial_bingham --fig-dir writeup/results/figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from apps.synthetic.axial_bingham import (
    ARMS,
    DECODERS,
    DIMS,
    SIGMAS,
    TEST_SEED,
    AxialVAE,
    arm_p_init,
    run_path,
)
from tnbbeta_vae.data.axial_bingham import AxialBinghamTask, sphere_quadrature
from tnbbeta_vae.models.losses.sampling import sample_with_log_prob
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

TITLES = {
    "vmf": "vMF",
    "power_spherical": "PowerSpherical",
    "tnbbeta": "TNBBeta",
    "tnbbeta_init0": "TNBBeta (p init 0)",
    "vmf_sym": "sign-sym. vMF",
    "ps_sym": "sign-sym. PS",
}
COLORS = {
    "vmf": "#2a78d6",
    "power_spherical": "#eb6834",
    "tnbbeta": "#1baf7a",
    "tnbbeta_init0": "#6250d6",
    "vmf_sym": "#7fb2ea",
    "ps_sym": "#f1a98c",
}
DECODER_TITLES = {"fixed": "fixed true decoder", "linear": "linear", "mlp": "MLP"}
SCALE = 3
NUM_DRAWS = 1200


def main(argv: list[str] | None = None) -> None:
    """Writes both figures.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--date", default="2026-11-02")
    parser.add_argument("--sphere-sigma", type=float, default=0.25)
    args = parser.parse_args(argv)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    gap_path = args.fig_dir / f"axial_bingham_gap_vs_sigma_{args.date}.png"
    gap_vs_sigma(args.out_dir).write_image(gap_path, scale=SCALE)
    print(f"wrote {gap_path}")
    sphere_path = args.fig_dir / f"axial_bingham_sphere_d3_{args.date}.png"
    sphere_figure(args.out_dir, args.sphere_sigma).write_image(sphere_path, scale=SCALE)
    print(f"wrote {sphere_path}")


def gap_vs_sigma(out_dir: Path) -> go.Figure:
    """Mean ELBO gap versus sigma for every (d, decoder, arm)."""
    fig = make_subplots(
        rows=len(DIMS),
        cols=len(DECODERS),
        shared_xaxes=True,
        horizontal_spacing=0.07,
        vertical_spacing=0.07,
        column_titles=[DECODER_TITLES[d] for d in DECODERS],
        row_titles=[f"d = {d}" for d in DIMS],
    )
    for row, dim in enumerate(DIMS, start=1):
        for col, decoder in enumerate(DECODERS, start=1):
            for arm in ARMS:
                means, sds, sigmas = [], [], []
                for sigma in SIGMAS:
                    gaps = _gaps(out_dir, dim, sigma, decoder, arm)
                    if len(gaps):
                        sigmas.append(sigma)
                        means.append(float(np.mean(gaps)))
                        sds.append(
                            float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
                        )
                fig.add_trace(
                    go.Scatter(
                        x=sigmas,
                        y=means,
                        error_y={"type": "data", "array": sds, "thickness": 1},
                        mode="lines+markers",
                        name=TITLES[arm],
                        legendgroup=arm,
                        showlegend=row == 1 and col == 1,
                        line={"color": COLORS[arm], "width": 2},
                        marker={"size": 6},
                    ),
                    row=row,
                    col=col,
                )
            fig.add_hline(
                y=math.log(2),
                line={"color": "#888888", "dash": "dot", "width": 1},
                row=row,  # pyright: ignore[reportArgumentType]
                col=col,  # pyright: ignore[reportArgumentType]
            )
    fig.update_xaxes(
        type="log", tickvals=list(SIGMAS), ticktext=[f"{s:g}" for s in SIGMAS]
    )
    fig.update_yaxes(type="log", exponentformat="power")
    fig.update_xaxes(title_text="noise scale sigma", row=len(DIMS))
    fig.update_yaxes(title_text="ELBO gap (nats)", col=1)
    fig.update_layout(
        template=TEMPLATE_NAME,
        width=1000,
        height=900,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.08,
            "x": 0.5,
            "xanchor": "center",
        },
        margin={"l": 70, "r": 20, "t": 40, "b": 90},
        font={"family": FONT_FAMILY},
    )
    return fig


def sphere_figure(out_dir: Path, sigma: float) -> go.Figure:
    """Posterior draws on S^2: exact Bingham and each arm (fixed decoder, one test
    x)."""
    dim = 3
    task = AxialBinghamTask(dim, sigma)
    test_x, _ = task.sample(2000, torch.Generator().manual_seed(TEST_SEED))
    index = 0
    x = test_x[index : index + 1].double()
    axis = task.top_axis(x)[0].numpy()
    panels = [("exact Bingham", _bingham_draws(task, x))]
    for arm in ARMS:
        seed = _median_seed(out_dir, dim, sigma, "fixed", arm)
        if seed is None:
            continue
        panels.append((TITLES[arm], _model_draws(out_dir, dim, sigma, arm, seed, x)))
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        specs=[[{"type": "xy"}] * len(panels)],
        horizontal_spacing=0.01,
    )
    view = _view(axis)
    for col, (title, points) in enumerate(panels, start=1):
        _panel(fig, col, points, view, axis)
        fig.add_annotation(
            x=0.5,
            y=1.02,
            xref=f"x{col if col > 1 else ''} domain",
            yref=f"y{col if col > 1 else ''} domain",
            text=f"<b>{title}</b>",
            showarrow=False,
        )
    fig.update_layout(
        template=TEMPLATE_NAME,
        width=250 * len(panels),
        height=330,
        showlegend=False,
        margin={"l": 5, "r": 5, "t": 30, "b": 5},
        font={"family": FONT_FAMILY},
    )
    return fig


def _gaps(out_dir: Path, dim: int, sigma: float, decoder: str, arm: str) -> np.ndarray:
    values = []
    for seed in range(10):
        path = run_path(out_dir, dim, sigma, decoder, arm, seed).with_suffix(".json")
        if path.exists():
            record = json.loads(path.read_text())
            if not record["diverged"]:
                values.append(record["gap_elbo"])
    return np.array(values)


def _median_seed(
    out_dir: Path, dim: int, sigma: float, decoder: str, arm: str
) -> int | None:
    rows = []
    for seed in range(10):
        path = run_path(out_dir, dim, sigma, decoder, arm, seed).with_suffix(".json")
        if path.exists():
            record = json.loads(path.read_text())
            if not record["diverged"]:
                rows.append((record["gap_elbo"], seed))
    if not rows:
        return None
    rows.sort()
    return rows[(len(rows) - 1) // 2][1]


def _bingham_draws(task: AxialBinghamTask, x: torch.Tensor) -> np.ndarray:
    z, log_w = sphere_quadrature(3, 2)
    matrix = task.bingham_matrix(x)[0]
    logit = torch.einsum("ki,ij,kj->k", z, matrix, z) + log_w
    probs = torch.softmax(logit, 0)
    generator = torch.Generator().manual_seed(0)
    picks = torch.multinomial(probs, NUM_DRAWS, replacement=True, generator=generator)
    return z[picks].numpy()


@torch.no_grad()
def _model_draws(
    out_dir: Path, dim: int, sigma: float, arm: str, seed: int, x: torch.Tensor
) -> np.ndarray:
    model = AxialVAE(dim, sigma, "fixed", arm, p_init_offset=arm_p_init(arm))
    path = run_path(out_dir, dim, sigma, "fixed", arm, seed).with_suffix(".pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.double().eval()
    torch.manual_seed(0)
    z, _ = sample_with_log_prob(
        model.posterior(x), torch.Size([NUM_DRAWS]), reparameterize=False
    )
    return z[:, 0].numpy()


def _view(axis: np.ndarray, tilt: float = math.radians(25)) -> np.ndarray:
    """Rows ``(right, up, depth)`` of an orthographic view with ``axis`` near
    vertical."""
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    right = np.cross(axis, helper)
    right /= np.linalg.norm(right)
    toward = np.cross(right, axis)
    up = math.cos(tilt) * axis - math.sin(tilt) * toward
    depth = math.sin(tilt) * axis + math.cos(tilt) * toward
    return np.stack([right, up, depth])


def _panel(
    fig: go.Figure, col: int, points: np.ndarray, view: np.ndarray, axis: np.ndarray
) -> None:
    angle = np.linspace(0, 2 * math.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=np.cos(angle),
            y=np.sin(angle),
            mode="lines",
            line={"color": "#bbbbbb", "width": 1},
        ),
        row=1,
        col=col,
    )
    screen = points @ view.T
    near = screen[:, 2] >= 0
    for mask, opacity in ((~near, 0.15), (near, 0.55)):
        fig.add_trace(
            go.Scatter(
                x=screen[mask, 0],
                y=screen[mask, 1],
                mode="markers",
                marker={"size": 3, "color": "#1a1a1a", "opacity": opacity},
            ),
            row=1,
            col=col,
        )
    tip = view @ axis
    fig.add_trace(
        go.Scatter(
            x=[-tip[0], tip[0]],
            y=[-tip[1], tip[1]],
            mode="lines",
            line={"color": "#eb6834", "width": 2, "dash": "dot"},
        ),
        row=1,
        col=col,
    )
    suffix = "" if col == 1 else str(col)
    fig.update_xaxes(visible=False, range=[-1.05, 1.05], row=1, col=col)
    fig.update_yaxes(
        visible=False,
        range=[-1.05, 1.05],
        scaleanchor=f"x{suffix}",
        scaleratio=1,
        row=1,
        col=col,
    )


if __name__ == "__main__":
    main()
