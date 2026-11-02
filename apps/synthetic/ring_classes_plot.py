"""Sphere figures for the four-class ring study: data, exact and fitted posteriors.

For each design and seed, one figure. The leftmost sphere (spanning every row) shows
the true data distribution: latent samples ``z`` of the generative model coloured by
class, with the class axes marked. Each row is one class (design 1: latitude band;
design 2: class); its panels show one representative test ``x`` (the test ``x`` of that
class whose true latitude is nearest a fixed target): the exact posterior, then the
vMF, PowerSpherical and TNBBeta posteriors fitted by the same seed, on an orthographic
view of ``S^2`` with the class axis marked (star: ``+a``; open star: ``-a``; the far
side is drawn faintly). Rendered with plotly and the project's ``alptex`` template
(``tnbbeta_vae.plotting``) through kaleido.

Usage:
    uv run python -m apps.synthetic.ring_classes_plot \
        --out-dir ring_recovery/classes --fig-dir writeup/results/figures
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

from tnbbeta_vae.data.ring_classes import (
    BAND_EDGES,
    NUM_CLASSES,
    RingClassTask,
    ring_class_task,
)
from tnbbeta_vae.distributions import PowerSpherical, TNBBetaSpherical, VonMisesFisher
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
SYMBOLS = ("circle", "square", "triangle-up", "diamond")
INK = "#1a1a1a"
INK_SECONDARY = "#555555"
OUTLINE = "#c8c8c8"
TARGET_W = {1: (-0.75, -0.25, 0.25, 0.75), 2: (-0.6, -0.2, 0.2, 0.6)}
FAMILIES = (
    ("exact", "exact posterior"),
    ("vmf", "vMF"),
    ("power_spherical", "PowerSpherical"),
    ("tnbbeta", "TNBBeta"),
)
NUM_POINTS = 350
NUM_DATA_POINTS = 3000
VIEW_ANGLE_DEG = 50.0
DATA_VIEW_DESIGN2 = np.array([0.6, 0.5, 0.62])
MARKER_SIZE = 5


def main(argv: list[str] | None = None) -> None:
    """Writes one figure per (design, representative seed)."""
    parser = argparse.ArgumentParser(description="Class-sphere figures.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--date", default="2026-11-01")
    args = parser.parse_args(argv)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for design in (1, 2):
        for kind, seed in _representative_seeds(args.out_dir, design).items():
            path = (
                args.fig_dir
                / f"ring_classes_design{design}_{kind}_s{seed}_{args.date}.png"
            )
            _figure(args.out_dir, design, seed, kind, path)
            print(f"wrote {path}")


def _representative_seeds(out_dir: Path, design: int) -> dict[str, int]:
    """The best aligned TNBBeta seed and the worst non-aligned (else worst) one."""
    runs = []
    for path in sorted((out_dir / f"design{design}" / "tnbbeta").glob("s*.json")):
        runs.append(json.loads(path.read_text()))
    aligned = [r for r in runs if r["aligned"]]
    unaligned = [r for r in runs if not r["aligned"]]
    chosen: dict[str, int] = {}
    if aligned:
        chosen["aligned"] = min(aligned, key=lambda r: r["kl_mean"])["seed"]
    if unaligned:
        chosen["nonaligned"] = max(unaligned, key=lambda r: r["kl_mean"])["seed"]
    elif len(aligned) > 1:
        # No non-aligned seed exists: show the aligned seed with the highest KL instead.
        chosen["worst"] = max(aligned, key=lambda r: r["kl_mean"])["seed"]
    return chosen


def _figure(out_dir: Path, design: int, seed: int, kind: str, path: Path) -> None:
    task = ring_class_task(design)
    data = {
        arm: np.load(out_dir / f"design{design}" / arm / f"s{seed}.npz")
        for arm in ("vmf", "power_spherical", "tnbbeta")
    }
    tnb = data["tnbbeta"]
    summary = json.loads(
        (out_dir / f"design{design}" / "tnbbeta" / f"s{seed}.json").read_text()
    )
    specs = [[{"rowspan": NUM_CLASSES}, {}, {}, {}, {}]] + [
        [None, {}, {}, {}, {}] for _ in range(NUM_CLASSES - 1)
    ]
    fig = make_subplots(
        rows=NUM_CLASSES,
        cols=1 + len(FAMILIES),
        specs=specs,
        column_widths=[0.32] + [0.17] * len(FAMILIES),
        horizontal_spacing=0.012,
        vertical_spacing=0.03,
    )
    _data_panel(fig, task, design)
    rng = torch.Generator().manual_seed(0)
    for c in range(NUM_CLASSES):
        mask = np.nonzero(tnb["cls"] == c)[0]
        index = int(mask[np.argmin(np.abs(tnb["w"][mask] - TARGET_W[design][c]))])
        x = torch.tensor(tnb["x"][index : index + 1])
        cls = torch.tensor([c])
        axis = task.axis_of(cls)[0].numpy()
        view = _view(axis)
        for column, (family, title) in enumerate(FAMILIES):
            row, col = c + 1, column + 2
            if family == "exact":
                points = task.sample_posterior(x, cls, NUM_POINTS, rng)[:, 0].numpy()
                note = f"true latitude w = {tnb['w'][index]:.2f}"
            else:
                points = _fitted_samples(family, data[family], index, rng)
                note = (
                    f"KL {data[family]['kl'][index]:.2f}, "
                    f"TV {data[family]['tv'][index]:.2f}, "
                    f"|cos| {data[family]['abs_cos'][index]:.2f}"
                )
            _sphere_axes(fig, row, col)
            _outline(fig, row, col)
            _points(fig, row, col, points, view, c)
            _axis_markers(fig, row, col, axis[None], view)
            _note(fig, row, col, note)
            if c == 0:
                _title(fig, row, col, f"<b>{title}</b>")
            if column == 0:
                label = "band" if design == 1 else "class"
                fig.update_yaxes(title_text=f"<b>{label} {c + 1}</b>", row=row, col=col)
    verdict = {
        "aligned": "aligned, lowest-KL seed",
        "nonaligned": "not aligned",
        "worst": "aligned, highest-KL seed",
    }[kind]
    fig.update_layout(
        template=TEMPLATE_NAME,
        width=1800,
        height=1400,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 40, "r": 20, "t": 110, "b": 40},
        title={
            "text": (
                f"Design {design}: true data distribution, exact posterior and fitted "
                f"posteriors, seed {seed}<br><sup>TNBBeta {verdict}, mean exact KL "
                f"{summary['kl_mean']:.2f}; star = +a, open star = -a, faint = far side"
                "</sup>"
            ),
            "x": 0.5,
        },
        legend={
            "orientation": "v",
            "x": 0.0,
            "y": 0.27,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"family": FONT_FAMILY, "size": 13},
        },
    )
    fig.write_image(path, scale=1.5)


def _data_panel(fig: go.Figure, task: RingClassTask, design: int) -> None:
    """The leftmost sphere: the true data distribution of ``z``, coloured by class."""
    generator = torch.Generator().manual_seed(1)
    _, z, cls, _ = task.sample(NUM_DATA_POINTS, generator)
    z, cls = z.numpy(), cls.numpy()
    axes = task.axes(torch.float64).numpy()
    view = _view(axes[0]) if design == 1 else DATA_VIEW_DESIGN2
    _sphere_axes(fig, 1, 1)
    _outline(fig, 1, 1)
    if design == 1:
        _band_edges(fig, 1, 1, axes[0], view)
    for c in range(NUM_CLASSES):
        name = (
            f"band {c + 1}: w in [{BAND_EDGES[c]:g}, {BAND_EDGES[c + 1]:g}]"
            if design == 1
            else f"class {c + 1} (axis a{c + 1})"
        )
        _points(fig, 1, 1, z[cls == c], view, c, name=name, size=MARKER_SIZE + 1)
    labels = [f"a{c + 1}" for c in range(NUM_CLASSES)] if design == 2 else None
    _axis_markers(fig, 1, 1, axes if design == 2 else axes[:1], view, labels)
    heading = (
        "<b>true data distribution</b><br>z coloured by latitude band"
        if design == 1
        else "<b>true data distribution</b><br>z is uniform for every class"
    )
    _title(fig, 1, 1, heading)


def _axis_ids(fig: go.Figure, row: int, col: int) -> tuple[str, str]:
    subplot: Any = fig.get_subplot(row, col)
    xid = subplot.xaxis.plotly_name.replace("axis", "")
    yid = subplot.yaxis.plotly_name.replace("axis", "")
    return xid, yid


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
    view: np.ndarray,
    c: int,
    name: str | None = None,
    size: int = MARKER_SIZE,
) -> None:
    xy, front = _project(points, view)
    for mask, opacity, label in ((front, 0.65, name), (~front, 0.12, None)):
        fig.add_trace(
            go.Scatter(
                x=xy[mask, 0],
                y=xy[mask, 1],
                mode="markers",
                marker={
                    "symbol": SYMBOLS[c],
                    "size": size,
                    "color": COLORS[c],
                    "opacity": opacity,
                    "line": {"width": 0},
                },
                name=label,
                showlegend=label is not None,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def _band_edges(
    fig: go.Figure, row: int, col: int, axis: np.ndarray, view: np.ndarray
) -> None:
    """Thin circles at the design-1 band edges ``w = -0.5, 0, 0.5``."""
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    phi = np.linspace(0, 2 * np.pi, 300)
    for w in BAND_EDGES[1:-1]:
        radius = np.sqrt(1 - w**2)
        ring = radius * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
        xy, front = _project(ring + w * axis, view)
        xy = np.where(front[:, None], xy, np.nan)
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="lines",
                line={"color": INK_SECONDARY, "width": 1},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def _axis_markers(
    fig: go.Figure,
    row: int,
    col: int,
    axes: np.ndarray,
    view: np.ndarray,
    labels: list[str] | None = None,
) -> None:
    """Stars at ``+axes`` (filled) and ``-axes`` (open); ``labels`` name the filled."""
    for sign, symbol in ((1.0, "star"), (-1.0, "star-open")):
        xy, front = _project(sign * axes, view)
        for mask, opacity in ((front, 1.0), (~front, 0.3)):
            if not mask.any():
                continue
            text = (
                [labels[i] for i in np.nonzero(mask)[0]]
                if labels is not None and sign > 0
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=xy[mask, 0],
                    y=xy[mask, 1],
                    mode="markers+text" if text else "markers",
                    text=text,
                    textposition="top right",
                    textfont={"family": FONT_FAMILY, "size": 14, "color": INK},
                    marker={
                        "symbol": symbol,
                        "size": 15,
                        "color": INK,
                        "opacity": opacity,
                        "line": {"width": 1.5, "color": INK},
                    },
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
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
        font={"family": FONT_FAMILY, "size": 13, "color": INK},
    )


def _note(fig: go.Figure, row: int, col: int, text: str) -> None:
    _annotate(fig, row, col, text, -0.02, "top")


def _title(fig: go.Figure, row: int, col: int, text: str) -> None:
    _annotate(fig, row, col, text, 1.02, "bottom")


def _fitted_samples(
    family: str, arrays: np.lib.npyio.NpzFile, index: int, rng: torch.Generator
) -> np.ndarray:
    mu = torch.tensor(arrays["mu"][index : index + 1])
    if family == "vmf":
        kappa = torch.tensor(arrays["kappa"][index : index + 1, None])
        dist = VonMisesFisher(mu, kappa)
    elif family == "power_spherical":
        dist = PowerSpherical(mu, torch.tensor(arrays["kappa"][index : index + 1]))
    else:
        dist = TNBBetaSpherical(
            mu,
            torch.tensor(arrays["p"][index : index + 1]),
            torch.tensor(arrays["q"][index : index + 1]),
            torch.tensor(arrays["epsilon"][index : index + 1]),
        )
    del rng
    return dist.sample(torch.Size([NUM_POINTS]))[:, 0].numpy()


def _view(axis: np.ndarray) -> np.ndarray:
    """A view direction at 50 degrees from the axis (so both poles are off-centre)."""
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    angle = np.radians(VIEW_ANGLE_DEG)
    return np.cos(angle) * axis + np.sin(angle) * e1


def _project(points: np.ndarray, view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic coordinates (``up`` = the axis-most direction) and front mask."""
    w = view / np.linalg.norm(view)
    up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, w)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(w, e1)
    return np.stack([points @ e1, points @ e2], axis=-1), points @ w > 0


if __name__ == "__main__":
    main()
