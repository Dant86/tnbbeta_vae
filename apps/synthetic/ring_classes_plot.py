"""Sphere figures for the four-class ring study: exact posterior next to fitted q's.

For each design, one representative test ``x`` per class (the test ``x`` of that
class whose true latitude is nearest a fixed target) is drawn as: the exact posterior,
then the vMF, Power Spherical and TNBBeta posteriors fitted by the same seed, on an
orthographic view of ``S^2`` with the class axis marked (star: ``+a``; open star:
``-a``; the far side of the sphere is drawn faintly). One figure per (design, seed): an
aligned TNBBeta seed and a non-aligned one when both exist.

Usage:
    uv run --with matplotlib python -m apps.synthetic.ring_classes_plot \
        --out-dir ring_recovery/classes --fig-dir writeup/results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib  # pyright: ignore[reportMissingImports]
import numpy as np
import torch

from tnbbeta_vae.data.ring_classes import NUM_CLASSES, ring_class_task
from tnbbeta_vae.distributions import PowerSpherical, TNBBetaSpherical, VonMisesFisher

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pyright: ignore[reportMissingImports]

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
MARKERS = ("o", "s", "^", "D")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d5d4cf"
TARGET_W = {1: (-0.75, -0.25, 0.25, 0.75), 2: (-0.6, -0.2, 0.2, 0.6)}
FAMILIES = (
    ("exact", "exact posterior"),
    ("vmf", "vMF"),
    ("power_spherical", "PowerSpherical"),
    ("tnbbeta", "TNBBeta"),
)
NUM_POINTS = 350
VIEW_ANGLE_DEG = 50.0


def main(argv: list[str] | None = None) -> None:
    """Writes one figure per (design, representative seed)."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    rng = torch.Generator().manual_seed(0)
    fig, axes = plt.subplots(
        NUM_CLASSES, len(FAMILIES), figsize=(13.0, 13.4), facecolor=SURFACE
    )
    for c in range(NUM_CLASSES):
        mask = np.nonzero(tnb["cls"] == c)[0]
        index = int(mask[np.argmin(np.abs(tnb["w"][mask] - TARGET_W[design][c]))])
        x = torch.tensor(tnb["x"][index : index + 1])
        cls = torch.tensor([c])
        axis = task.axis_of(cls)[0].numpy()
        view = _view(axis)
        for column, (family, title) in enumerate(FAMILIES):
            panel = axes[c, column]
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
            _frame(panel)
            _scatter(panel, points, view, c)
            _axis_markers(panel, axis, view)
            panel.set_xlabel(note, color=INK_SECONDARY, fontsize=9)
            if c == 0:
                panel.set_title(title, color=INK, fontsize=12)
            if column == 0:
                label = "band" if design == 1 else "class"
                panel.set_ylabel(f"{label} {c + 1}", color=INK, fontsize=12)
    verdict = {
        "aligned": "aligned, lowest-KL seed",
        "nonaligned": "not aligned",
        "worst": "aligned, highest-KL seed",
    }[kind]
    fig.suptitle(
        f"Design {design}: exact posterior vs fitted posteriors, seed {seed} "
        f"(TNBBeta {verdict}, mean exact KL {summary['kl_mean']:.2f}); "
        "star = +a, open star = -a, faint = far side",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


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


def _scatter(panel, points: np.ndarray, view: np.ndarray, c: int) -> None:  # noqa: ANN001
    xy, front = _project(points, view)
    for mask, alpha in ((front, 0.65), (~front, 0.12)):
        panel.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=8,
            marker=MARKERS[c],
            color=COLORS[c],
            alpha=alpha,
            linewidths=0,
        )


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


if __name__ == "__main__":
    main()
