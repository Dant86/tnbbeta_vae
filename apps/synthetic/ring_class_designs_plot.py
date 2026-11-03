"""Illustrates two candidate four-class designs for the ring-posterior task on S^2.

Both designs observe the latent only through its cosine with a class axis, so the exact
posterior given ``x`` is a latitude ring about that axis:

* **Design 1, shared axis.** ``z ~ Uniform(S^2)``; the class is the latitude band of
  ``w = a . z`` (four equal-mass bands, edges ``-1, -0.5, 0, 0.5, 1``);
  ``x = h(w) + noise``. The mixture of the four bands is exactly uniform, so the VAE's
  uniform prior is the true marginal.
* **Design 2, one axis per class.** ``z ~ Uniform(S^2)`` for every class; class ``c``
  has its own axis ``a_c`` (tetrahedron vertices) and observes
  ``x = h(a_c . z) + o_c + noise`` with a class offset ``o_c`` so the class is
  identifiable from ``x``. ``p(z | class)`` is uniform, so the latent samples of the
  four classes are indistinguishable; the classes differ in their posteriors given
  ``x``.

Each row shows ``z`` samples coloured by class from two views and, in the last column,
samples from the exact posterior ``p(z | x, class)`` for a few example ``x``.

Usage:
    uv run --with matplotlib python -m apps.synthetic.ring_class_designs_plot \
        --out writeup/results/figures/ring_class_designs_2026-11-01.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib  # pyright: ignore[reportMissingImports]
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pyright: ignore[reportMissingImports]

SIGMA = 0.2
NUM_Z = 3000
POSTERIOR_SAMPLES = 350
BAND_EDGES = (-1.0, -0.5, 0.0, 0.5, 1.0)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
MARKERS = ("o", "s", "^", "D")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d5d4cf"
TETRAHEDRON = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
) / np.sqrt(3.0)
# Class offsets in observation space (design 2) so the class is identifiable from x.
OFFSETS = 1.5 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], float)


def main(argv: list[str] | None = None) -> None:
    """Draws the two designs and writes the figure."""
    parser = argparse.ArgumentParser(
        description="Plots the two four-class ring designs on S^2."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    z = _uniform_sphere(NUM_Z, rng)
    axis = np.array([0.0, 0.0, 1.0])
    band = np.digitize(z @ axis, BAND_EDGES[1:-1])
    views = (np.array([0.8, 0.0, 0.25]), np.array([0.1, 0.1, 1.0]))
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 9.0), facecolor=SURFACE)

    # Design 1: shared axis, classes are latitude bands.
    for column, view in enumerate(views):
        panel = axes[0, column]
        _frame(
            panel,
            view,
            "z by class (band of a.z)" if column == 0 else "same, from above",
        )
        for c in range(4):
            _scatter(panel, z[band == c], view, c)
        _band_edges(panel, view, axis)
    panel = axes[0, 2]
    _frame(panel, views[0], "posterior given x (one x per class)")
    for c in range(4):
        w_star = 0.5 * (BAND_EDGES[c] + BAND_EDGES[c + 1])
        x = _h(np.array([w_star])) + SIGMA * rng.standard_normal((1, 3))
        _scatter(panel, _posterior_samples(x[0], axis, rng), views[0], c, small=True)
    _band_edges(panel, views[0], axis)
    axes[0, 0].set_ylabel(
        "Design 1: one shared axis,\nclasses = 4 equal-mass latitude bands",
        color=INK,
        fontsize=11,
    )

    # Design 2: four axes, uniform z for every class.
    views2 = (np.array([0.6, 0.5, 0.62]), -np.array([0.6, 0.5, 0.62]))
    class_of_z = rng.integers(0, 4, NUM_Z)
    for column, view in enumerate(views2[:2]):
        panel = axes[1, column]
        _frame(
            panel,
            view,
            "z by class (z uniform for every class)"
            if column == 0
            else "same, other side",
        )
        for c in range(4):
            _scatter(panel, z[class_of_z == c], view, c)
        _axis_markers(panel, view)
    panel = axes[1, 2]
    _frame(panel, views2[0], "posterior given x (one x per class, ring about a_c)")
    for c, w_star in enumerate((-0.6, -0.2, 0.2, 0.6)):
        x = _h(np.array([w_star])) + SIGMA * rng.standard_normal((1, 3))
        samples = _posterior_samples(x[0], TETRAHEDRON[c], rng)
        _scatter(panel, samples, views2[0], c, small=True)
    _axis_markers(panel, views2[0])
    axes[1, 0].set_ylabel(
        "Design 2: one axis per class (tetrahedron),\n"
        "p(z | class) uniform, class changes the observation map",
        color=INK,
        fontsize=11,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker=MARKERS[c],
            linestyle="",
            color=COLORS[c],
            markersize=8,
            label=f"class {c + 1}",
        )
        for c in range(4)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11)
    fig.suptitle(
        "Four-class ring designs on S^2 (z samples and exact posteriors, sigma = 0.2)",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"wrote {args.out}")
    per_class = args.out.with_name(args.out.stem + "_design2_per_class.png")
    _per_class_figure(per_class, rng)
    print(f"wrote {per_class}")
    print(
        f"design 1 two-mode check: {_two_mode_fraction(rng):.3f} of x have a "
        "second mode > 5%"
    )


def _per_class_figure(path: Path, rng: np.random.Generator) -> None:
    """Design 2, one panel per class, viewed along that class's axis ``a_c``.

    Each panel shows exact posterior samples for three ``x`` (latitudes -0.6, 0, 0.6 of
    ``a_c . z``), which are concentric rings about the star at the centre (``a_c``).
    """
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.8), facecolor=SURFACE)
    for c in range(4):
        panel = axes[c]
        _frame(panel, TETRAHEDRON[c], f"class {c + 1}: rings about a{c + 1}")
        for w_star in (-0.6, 0.0, 0.6):
            x = _h(np.array([w_star])) + SIGMA * rng.standard_normal((1, 3))
            samples = _posterior_samples(x[0], TETRAHEDRON[c], rng)
            _scatter(panel, samples, TETRAHEDRON[c], c, small=True)
        _axis_markers(panel, TETRAHEDRON[c])
    fig.suptitle(
        "Design 2: exact posteriors given x, each class viewed along its own axis "
        "(front hemisphere only; the w = -0.6 ring is on the far side)",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=170, facecolor=SURFACE)


def _h(w: np.ndarray) -> np.ndarray:
    """The observation map shared by both designs, ``(...,) -> (..., 3)``."""
    return np.stack([w, np.sin(1.5 * w), np.cos(2.0 * w)], axis=-1)


def _uniform_sphere(num: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.standard_normal((num, 3))
    return z / np.linalg.norm(z, axis=-1, keepdims=True)


def _posterior_samples(
    x: np.ndarray, axis: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Samples ``z ~ p(z | x)`` for ``x = h(axis . z) + noise``; uniform prior on S^2.

    The latitude ``w = axis . z`` has density proportional to ``exp(-||x - h(w)||^2 /
    (2 sigma^2))`` on ``[-1, 1]`` (uniform prior on S^2 is uniform in ``w``) and the
    azimuth about ``axis`` is uniform.
    """
    grid = np.linspace(-1.0, 1.0, 4001)
    energy = ((x[None, :] - _h(grid)) ** 2).sum(-1) / (2 * SIGMA**2)
    weights = np.exp(-(energy - energy.min()))
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    w = np.interp(rng.random(POSTERIOR_SAMPLES), cdf, grid)
    phi = rng.uniform(0, 2 * np.pi, POSTERIOR_SAMPLES)
    radius = np.sqrt(np.clip(1 - w**2, 0, None))
    local = np.stack([radius * np.cos(phi), radius * np.sin(phi), w], axis=-1)
    e1, e2 = _orthonormal_frame(axis)
    return local[:, 0:1] * e1 + local[:, 1:2] * e2 + local[:, 2:3] * axis


def _orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def _project(points: np.ndarray, view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic coordinates and front-hemisphere mask for a view direction."""
    w = view / np.linalg.norm(view)
    up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(up, w)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(w, e1)
    return np.stack([points @ e1, points @ e2], axis=-1), points @ w > 0


def _frame(panel, view: np.ndarray, title: str) -> None:  # noqa: ANN001
    """Draws the sphere outline and sets the panel's frame."""
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
    panel.set_title(title, color=INK_SECONDARY, fontsize=10)
    del view


def _scatter(
    panel, points: np.ndarray, view: np.ndarray, c: int, small: bool = False
) -> None:  # noqa: ANN001
    xy, front = _project(points, view)
    panel.scatter(
        xy[front, 0],
        xy[front, 1],
        s=7 if small else 9,
        marker=MARKERS[c],
        color=COLORS[c],
        alpha=0.65,
        linewidths=0,
    )


def _band_edges(panel, view: np.ndarray, axis: np.ndarray) -> None:  # noqa: ANN001
    """Thin grey circles at the band edges ``w = -0.5, 0, 0.5`` of design 1."""
    phi = np.linspace(0, 2 * np.pi, 300)
    e1, e2 = _orthonormal_frame(axis)
    for w in BAND_EDGES[1:-1]:
        r = np.sqrt(1 - w**2)
        circle = r * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2) + w * axis
        xy, front = _project(circle, view)
        xy = np.where(front[:, None], xy, np.nan)
        panel.plot(xy[:, 0], xy[:, 1], color=INK_SECONDARY, linewidth=0.7, alpha=0.8)


def _axis_markers(panel, view: np.ndarray) -> None:  # noqa: ANN001
    """Marks the four class axes (tetrahedron vertices) visible in this view."""
    xy, front = _project(TETRAHEDRON, view)
    for c in range(4):
        if front[c]:
            panel.scatter(
                xy[c, 0],
                xy[c, 1],
                s=70,
                marker="*",
                color=INK,
                zorder=5,
            )
            panel.annotate(
                f"a{c + 1}",
                xy[c],
                xytext=(5, 5),
                textcoords="offset points",
                color=INK,
                fontsize=9,
            )


def _two_mode_fraction(rng: np.random.Generator) -> float:
    """Fraction of x (smoother ``h``) whose latitude posterior has a 2nd mode."""
    grid = np.linspace(-1.0, 1.0, 2001)
    features = _h(grid)
    count = 0
    num = 2000
    for _ in range(num):
        w_true = rng.uniform(-1, 1)
        x = _h(np.array([w_true]))[0] + SIGMA * rng.standard_normal(3)
        energy = ((x[None, :] - features) ** 2).sum(-1) / (2 * SIGMA**2)
        density = np.exp(-(energy - energy.min()))
        minima = (
            np.nonzero((density[1:-1] < density[:-2]) & (density[1:-1] <= density[2:]))[
                0
            ]
            + 1
        )
        bounds = [0, *minima.tolist(), len(density)]
        masses = sorted(
            (density[a:b].sum() for a, b in zip(bounds[:-1], bounds[1:], strict=True)),
            reverse=True,
        )
        total = sum(masses)
        if len(masses) > 1 and masses[1] / total > 0.05:
            count += 1
    return count / num


if __name__ == "__main__":
    main()
