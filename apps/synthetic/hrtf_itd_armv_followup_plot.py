"""Sphere figure of the ITD arm V follow-up: fitted posteriors at a few ITD values.

One figure per decoder for one split. Rows are vMF, PowerSpherical and TNBBeta;
columns are three ITD values. Each panel shows draws of ``q(z | x)`` (colour) at that
ITD and the true head-frame directions of held-out points within 15 us of it (purple
diamonds), on an orthographic view of ``S^2``; the signed interaural axis is the arrow
from the origin (lighter when it points away from the viewer; the far side is drawn
faintly). The latent frame of a VAE is arbitrary, so ``q``'s draws are rotated by the
minimal rotation taking the signed decoder axis to ``+y`` (a ring stays a ring; azimuth
about the axis carries no meaning). Plotly with the project's ``alptex`` template
(``tnbbeta_vae.plotting``) through kaleido.

Usage:
    uv run python -m apps.synthetic.hrtf_itd_armv_followup_plot \
        --out-dir hrtf_runs/armv_followup --fig-dir writeup/results/figures --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from apps.synthetic.hrtf_itd import family_spec
from apps.synthetic.hrtf_itd_armv_followup import DECODERS, FollowupModel, run_path
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

FAMILIES = (
    ("vmf", "vMF", "#2a78d6", "circle"),
    ("power_spherical", "PowerSpherical", "#eb6834", "square"),
    ("tnbbeta", "TNBBeta", "#1baf7a", "triangle-up"),
)
TRUTH_COLOR = "#6250d6"
INK = "#1a1a1a"
OUTLINE = "#c8c8c8"
ITDS = (-450.0, 0.0, 450.0)
WINDOW_US = 15.0
NUM_POINTS = 350
NUM_TRUE = 150
MARKER_SIZE = 5
VIEW = np.array([0.55, 0.62, 0.56])
Y = np.array([0.0, 1.0, 0.0])


def main(argv: list[str] | None = None) -> None:
    """Writes one figure per decoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--date", default="2026-11-01")
    args = parser.parse_args(argv)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for decoder in DECODERS:
        path = (
            args.fig_dir
            / f"hrtf_itd_armv_followup_{decoder}_s{args.seed}_{args.date}.png"
        )
        _figure(args.out_dir, decoder, args.seed, path)
        print(f"wrote {path}")


def _figure(out_dir: Path, decoder: str, seed: int, path: Path) -> None:
    runs = {f[0]: _load(out_dir, decoder, f[0], seed) for f in FAMILIES}
    truth = runs["tnbbeta"]["arrays"]
    fig = make_subplots(
        rows=len(FAMILIES),
        cols=len(ITDS),
        horizontal_spacing=0.01,
        vertical_spacing=0.03,
    )
    rng = np.random.default_rng(0)
    for col, itd in enumerate(ITDS, start=1):
        near = np.nonzero(np.abs(truth["itd_us"] - itd) < WINDOW_US)[0]
        near = rng.choice(near, min(len(near), NUM_TRUE), replace=False)
        for row, (family, title, color, symbol) in enumerate(FAMILIES, start=1):
            _sphere_axes(fig, row, col)
            _outline(fig, row, col)
            first = row == 1 and col == 1
            _points(
                fig,
                row,
                col,
                truth["z_true"][near],
                ("diamond", TRUTH_COLOR, 6),
                "True Held-Out Directions (within 15 us of the ITD)" if first else None,
                "truth",
            )
            draws = _draws(runs[family], itd)
            _points(
                fig,
                row,
                col,
                draws,
                (symbol, color, MARKER_SIZE),
                title if col == 1 else None,
                family,
            )
            _arrow(fig, row, col)
            if row == 1:
                _annotate(fig, row, col, f"<b>ITD = {itd:.0f} us</b>", 1.02, "bottom")
            if col == 1:
                fig.update_yaxes(title_text=f"<b>{title}</b>", row=row, col=col)
    entries = [
        ("True Held-Out Directions (within 15 us of the ITD)", "diamond", TRUTH_COLOR)
    ]
    entries += [(title, symbol, color) for _, title, color, symbol in FAMILIES]
    for name, symbol, color in entries:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"symbol": symbol, "size": 9, "color": color},
                name=name,
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        template=TEMPLATE_NAME,
        width=1500,
        height=1450,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 60, "r": 20, "t": 110, "b": 90},
        title={
            "text": (
                "SADIE II ITD VAE: Fitted Posteriors "
                f"({decoder.replace('_', ' ').title()} Decoder)"
                f"<br><sup>Split {seed}; arrow = signed interaural axis from the "
                "origin; "
                "faint = far side</sup>"
            ),
            "x": 0.5,
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": -0.01,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"family": FONT_FAMILY, "size": 15},
        },
    )
    fig.write_image(path, scale=3)


def _load(out_dir: Path, decoder: str, family: str, seed: int) -> dict[str, Any]:
    stem = run_path(out_dir, decoder, family, seed)
    summary = json.loads(stem.with_suffix(".json").read_text())
    model = FollowupModel(decoder, **family_spec(family))
    model.load_state_dict(torch.load(stem.with_suffix(".pt")))
    return {
        "summary": summary,
        "arrays": dict(np.load(stem.with_suffix(".npz"))),
        "model": model.double().eval(),
    }


@torch.no_grad()
def _draws(run: dict[str, Any], itd_us: float) -> np.ndarray:
    """Draws of q at an ITD, rotated so the signed decoder axis sits on ``+y``."""
    summary = run["summary"]
    x = torch.tensor([[(itd_us - summary["itd_mean_us"]) / summary["itd_sd_us"]]])
    posterior = run["model"].posterior(x.double())
    z = posterior.rsample(torch.Size([NUM_POINTS]))[:, 0].numpy()
    return z @ _rotation_to_y(np.asarray(summary["axis"])).T


def _rotation_to_y(axis: np.ndarray) -> np.ndarray:
    """The minimal rotation taking the unit vector ``axis`` to ``+y``."""
    v = np.cross(axis, Y)
    c = float(axis @ Y)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def _project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic coordinates (up = world z) and the front-hemisphere mask."""
    w = VIEW / np.linalg.norm(VIEW)
    e1 = np.cross(np.array([0.0, 0.0, 1.0]), w)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(w, e1)
    return np.stack([points @ e1, points @ e2], axis=-1), points @ w > 0


def _axis_ids(fig: go.Figure, row: int, col: int) -> tuple[str, str]:
    subplot: Any = fig.get_subplot(row, col)
    return (
        subplot.xaxis.plotly_name.replace("axis", ""),
        subplot.yaxis.plotly_name.replace("axis", ""),
    )


def _sphere_axes(fig: go.Figure, row: int, col: int) -> None:
    xid, _ = _axis_ids(fig, row, col)
    common = {
        "range": [-1.05, 1.05],
        "showgrid": False,
        "zeroline": False,
        "showline": False,
        "showticklabels": False,
        "ticks": "",
        "constrain": "domain",
        "row": row,
        "col": col,
    }
    fig.update_xaxes(**common)
    fig.update_yaxes(scaleanchor=xid, scaleratio=1, **common)


def _outline(fig: go.Figure, row: int, col: int) -> None:
    theta = np.linspace(0, 2 * np.pi, 400)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line={"color": OUTLINE, "width": 1.5},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def _points(
    fig: go.Figure,
    row: int,
    col: int,
    points: np.ndarray,
    style: tuple[str, str, int],
    name: str | None,
    group: str,
) -> None:
    symbol, color, size = style
    xy, front = _project(points)
    for mask, opacity, label in ((front, 0.7, name), (~front, 0.12, None)):
        fig.add_trace(
            go.Scatter(
                x=xy[mask, 0],
                y=xy[mask, 1],
                mode="markers",
                marker={
                    "symbol": symbol,
                    "size": size,
                    "color": color,
                    "opacity": opacity,
                    "line": {"width": 0},
                },
                name=label,
                legendgroup=group,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def _arrow(fig: go.Figure, row: int, col: int) -> None:
    xid, yid = _axis_ids(fig, row, col)
    xy, front = _project(Y[None])
    tip = xy[0]
    fig.add_annotation(
        x=float(tip[0]),
        y=float(tip[1]),
        ax=0.0,
        ay=0.0,
        xref=xid,
        yref=yid,
        axref=xid,
        ayref=yid,
        showarrow=True,
        arrowhead=2,
        arrowsize=1.1,
        arrowwidth=2.5,
        arrowcolor=INK if front[0] else "rgba(26,26,26,0.35)",
        text="",
    )


def _annotate(
    fig: go.Figure, row: int, col: int, text: str, y: float, anchor: str
) -> None:
    xid, yid = _axis_ids(fig, row, col)
    fig.add_annotation(
        text=text,
        xref=f"{xid} domain",
        yref=f"{yid} domain",
        x=0.5,
        y=y,
        showarrow=False,
        yanchor=anchor,
        font={"family": FONT_FAMILY, "size": 14, "color": INK},
    )


if __name__ == "__main__":
    main()
