"""Figures for the ITD-only HRTF study: latitude marginals and sphere views of q.

Reads one split's runs written by ``apps.synthetic.hrtf_itd train`` (checkpoints, per-x
arrays and summaries) and draws

* (a) ITD against the lateral coordinate ``w`` of the held-out directions, with the
  three families' fitted latitude marginals at a few ITD values, for each arm;
* (b) orthographic sphere views of q (vMF, PowerSpherical, TNBBeta) at a few ITDs, with
  the true directions of held-out points whose ITD is within 15 us. The interaural axis
  is marked (filled star = left ear, +y; open star = right ear). In arm V the latent
  frame is arbitrary, so q's draws are rotated by the minimal rotation taking the
  signed decoder axis to +y (a ring stays a ring; its azimuth about the axis carries no
  meaning).

Usage:
    uv run --with matplotlib python -m apps.synthetic.hrtf_itd_plot \
        --out-dir hrtf_runs --fig-dir writeup/results/figures --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

import matplotlib  # pyright: ignore[reportMissingImports]
import numpy as np
import torch

from apps.synthetic.hrtf_itd import HrtfModel, family_spec, run_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pyright: ignore[reportMissingImports]

COLORS = {
    "vmf": "#2a78d6",
    "power_spherical": "#eb6834",
    "tnbbeta": "#1baf7a",
    "truth": "#6250d6",
}
MARKERS = {"vmf": "o", "power_spherical": "s", "tnbbeta": "^", "truth": "D"}
TITLES = {"vmf": "vMF", "power_spherical": "PowerSpherical", "tnbbeta": "TNBBeta"}
FAMILIES = ("vmf", "power_spherical", "tnbbeta")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d5d4cf"
LATITUDE_ITDS = (-500.0, -150.0, 150.0, 500.0)
SPHERE_ITDS = (-450.0, 0.0, 450.0)
WINDOW_US = 15.0
NUM_POINTS = 300
LAT_DRAWS = 4000
VIEW = np.array([0.55, 0.62, 0.56])


def main(argv: list[str] | None = None) -> None:
    """Writes the latitude-marginal figure and one sphere figure per arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--estimator", default="xcorr")
    parser.add_argument("--date", default="2026-11-01")
    args = parser.parse_args(argv)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        (arm, family): _load(args.out_dir, args.estimator, arm, family, args.seed)
        for arm in ("V", "L")
        for family in FAMILIES
    }
    path = args.fig_dir / f"hrtf_itd_latitude_s{args.seed}_{args.date}.png"
    _latitude_figure(runs, args.seed, path)
    print(f"wrote {path}")
    for arm in ("V", "L"):
        path = args.fig_dir / f"hrtf_itd_spheres_arm{arm}_s{args.seed}_{args.date}.png"
        _sphere_figure(runs, arm, args.seed, path)
        print(f"wrote {path}")


def _load(out_dir: Path, estimator: str, arm: str, family: str, seed: int) -> dict:
    stem = run_path(out_dir, estimator, arm, family, seed)
    summary = json.loads(stem.with_suffix(".json").read_text())
    model = HrtfModel(arm, **family_spec(family))
    model.load_state_dict(torch.load(stem.with_suffix(".pt")))
    return {
        "summary": summary,
        "arrays": dict(np.load(stem.with_suffix(".npz"))),
        "model": model.double().eval(),
    }


@torch.no_grad()
def _draws(run: dict, itd_us: float, num: int, rotate: bool) -> np.ndarray:
    """``num`` draws of q at an ITD (head frame if ``rotate``), shape ``(num, 3)``."""
    summary = run["summary"]
    x = torch.tensor([[(itd_us - summary["itd_mean_us"]) / summary["itd_sd_us"]]])
    z = run["model"].posterior(x.double()).rsample(torch.Size([num]))[:, 0].numpy()
    if rotate:
        z = z @ _rotation_to_y(np.asarray(summary["axis"])).T
    return z


def _rotation_to_y(axis: np.ndarray) -> np.ndarray:
    """The minimal rotation taking the unit vector ``axis`` to ``+y``."""
    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(axis, target)
    c = float(axis @ target)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew @ skew / (1 + c)


def _latitude_figure(runs: dict, seed: int, path: Path) -> None:
    fig, axes = plt.subplots(
        2, 1 + len(LATITUDE_ITDS), figsize=(17.5, 7.2), facecolor=SURFACE
    )
    for row, arm in enumerate(("V", "L")):
        arrays = runs[(arm, "tnbbeta")]["arrays"]
        itd, w_true = arrays["itd_us"], arrays["w_true"]
        panel = axes[row, 0]
        sel = np.random.default_rng(0).choice(len(itd), 2500, replace=False)
        panel.scatter(itd[sel], w_true[sel], s=4, color=COLORS["truth"], alpha=0.35)
        for value in LATITUDE_ITDS:
            panel.axvline(value, color=GRID, linewidth=1.0)
        panel.set_xlabel("ITD (us)")
        frame = "u" if arm == "V" else "y"
        panel.set_ylabel(f"w = {frame} . z (held-out true)")
        panel.set_title(
            f"arm {arm}: held-out ITD against w" + (" (signed u)" if arm == "V" else "")
        )
        for column, value in enumerate(LATITUDE_ITDS, start=1):
            panel = axes[row, column]
            near = np.abs(itd - value) < WINDOW_US
            panel.hist(
                w_true[near],
                bins=40,
                range=(-1, 1),
                density=True,
                color=GRID,
                label=f"true w, |ITD - {value:.0f}| < {WINDOW_US:.0f} us",
            )
            for family in FAMILIES:
                run = runs[(arm, family)]
                axis = np.asarray(run["summary"]["axis"])
                z = _draws(run, value, LAT_DRAWS, rotate=False)
                density, edges = np.histogram(
                    z @ axis, bins=60, range=(-1, 1), density=True
                )
                panel.plot(
                    0.5 * (edges[1:] + edges[:-1]),
                    density,
                    color=COLORS[family],
                    marker=MARKERS[family],
                    markevery=6,
                    markersize=4,
                    linewidth=1.4,
                    label=TITLES[family],
                )
            panel.set_xlabel("w")
            panel.set_title(f"ITD = {value:.0f} us")
            if column == 1 and row == 0:
                panel.legend(fontsize=7, frameon=False)
        axes[row, 0].set_facecolor(SURFACE)
    for panel in axes.ravel():
        panel.set_facecolor(SURFACE)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
    fig.suptitle(
        f"SADIE II ITD task, split {seed}: fitted latitude marginals of q "
        "(density of w; arm V about the signed decoder axis u, arm L about +y)",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _sphere_figure(runs: dict, arm: str, seed: int, path: Path) -> None:
    fig, axes = plt.subplots(
        len(FAMILIES),
        len(SPHERE_ITDS),
        figsize=(3.6 * len(SPHERE_ITDS), 3.6 * len(FAMILIES) + 0.6),
        facecolor=SURFACE,
    )
    arrays = runs[(arm, "tnbbeta")]["arrays"]
    itd, z_true = arrays["itd_us"], arrays["z_true"]
    rng = np.random.default_rng(0)
    y_axis = np.array([0.0, 1.0, 0.0])
    for column, value in enumerate(SPHERE_ITDS):
        near = np.nonzero(np.abs(itd - value) < WINDOW_US)[0]
        near = rng.choice(near, min(len(near), 150), replace=False)
        for row, family in enumerate(FAMILIES):
            panel = axes[row, column]
            points = _draws(runs[(arm, family)], value, NUM_POINTS, rotate=arm == "V")
            _frame(panel)
            _scatter(panel, z_true[near], "truth", 22, 0.9)
            _scatter(panel, points, family, 8, 0.6)
            _axis_markers(panel, y_axis, VIEW)
            for sign, name in ((1.0, "L"), (-1.0, "R")):
                xy, front = _project((sign * y_axis)[None], VIEW)
                panel.annotate(
                    name,
                    (xy[0, 0], xy[0, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    color=INK if front[0] else INK_SECONDARY,
                )
            if row == 0:
                panel.set_title(f"ITD = {value:.0f} us", color=INK, fontsize=11)
            if column == 0:
                panel.set_ylabel(TITLES[family], color=INK, fontsize=12)
    note = (
        "q rotated so the signed decoder axis u sits on +y (ring azimuth about u is "
        "arbitrary in a VAE)"
        if arm == "V"
        else "q in the head frame"
    )
    title = (
        f"Arm {arm}, split {seed}: draws of q (colour) and true held-out directions "
        f"within {WINDOW_US:.0f} us (purple diamonds); {note}; stars = +/- y "
        "(L = left ear, R = right ear), faint = far side"
    )
    fig.suptitle("\n".join(textwrap.wrap(title, 110)), color=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _project(points: np.ndarray, view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic coordinates (``up`` = world z) and the front-hemisphere mask."""
    w = view / np.linalg.norm(view)
    up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, w)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(w, e1)
    return np.stack([points @ e1, points @ e2], axis=-1), points @ w > 0


def _frame(panel) -> None:  # noqa: ANN001
    theta = np.linspace(0, 2 * np.pi, 400)
    panel.plot(np.cos(theta), np.sin(theta), color=GRID, linewidth=1.2)
    panel.set_xlim(-1.05, 1.05)
    panel.set_ylim(-1.05, 1.05)
    panel.set_aspect("equal")
    panel.set_xticks([])
    panel.set_yticks([])
    panel.set_facecolor(SURFACE)
    for spine in panel.spines.values():
        spine.set_visible(False)


def _axis_markers(panel, axis: np.ndarray, view: np.ndarray) -> None:  # noqa: ANN001
    for sign, filled in ((1.0, True), (-1.0, False)):
        xy, front = _project((sign * axis)[None], view)
        panel.scatter(
            xy[0, 0],
            xy[0, 1],
            s=80,
            marker="*",
            facecolors=INK if filled else "none",
            edgecolors=INK,
            alpha=1.0 if front[0] else 0.35,
            zorder=5,
        )


def _scatter(panel, points: np.ndarray, kind: str, size: float, alpha: float) -> None:  # noqa: ANN001
    xy, front = _project(points, VIEW)
    for mask, a in ((front, alpha), (~front, alpha * 0.2)):
        panel.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=size,
            marker=MARKERS[kind],
            color=COLORS[kind],
            alpha=a,
            linewidths=0,
        )


if __name__ == "__main__":
    main()
