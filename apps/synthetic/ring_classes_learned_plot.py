"""Sphere figures for the learned-decoder four-class ring study.

For each setting (D1, D2g, D2l) and decoder (mlp, single_index) one figure: the
posterior means (``heads.posterior_centre``, a unit vector) of 1500 test ``x`` coloured
by class, on an orthographic view of ``S^2`` rotated so the learned decoder's estimated
axis ``u*`` is drawn as an arrow from the origin (lighter when it points away from the
viewer; the far side is drawn faintly), one column per family. The seed is the
median-gap ``tnbbeta`` seed of that setting and decoder. In D2g the decoder has one
axis per class, so all four ``u*_c`` are drawn. A second figure type (``--draws``)
shows one posterior DRAW per test ``x`` instead of the mean, as a check against the
means understating ring-shaped posteriors. Rendered with plotly and the project's
``alptex`` template (``tnbbeta_vae.plotting``) through kaleido.

Usage:
    uv run python -m apps.synthetic.ring_classes_learned_plot \
        --out-dir ring_recovery/classes_learned --fig-dir writeup/results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from apps.synthetic.ring_classes_learned import ARMS, DECODERS, SETTINGS
from apps.synthetic.ring_classes_plot import (
    COLORS,
    SYMBOLS,
    _axis_arrows,
    _outline,
    _points,
    _sphere_axes,
    _title,
    _view,
)
from tnbbeta_vae.data.ring_classes import NUM_CLASSES
from tnbbeta_vae.distributions import PowerSpherical, TNBBetaSpherical, VonMisesFisher
from tnbbeta_vae.plotting import FONT_FAMILY, TEMPLATE_NAME

TITLES = {
    "vmf": "vMF",
    "power_spherical": "PowerSpherical",
    "tnbbeta": "TNBBeta",
    "tnbbeta_init0": "TNBBeta (p init 0)",
}
NUM_POINTS = 1500
TITLE = "Synthetic Experiment: Learned-Decoder Latents by Class"


def main(argv: list[str] | None = None) -> None:
    """Writes one figure per (setting, decoder) plus the draws check for D1 / mlp."""
    parser = argparse.ArgumentParser(
        description="Learned-decoder class-sphere figures."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fig-dir", type=Path, required=True)
    parser.add_argument("--date", default="2026-11-01")
    args = parser.parse_args(argv)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for setting in SETTINGS:
        for decoder in DECODERS:
            seed = median_gap_seed(args.out_dir, setting, decoder)
            if seed is None:
                continue
            stem = f"ring_classes_learned_{setting}_{decoder}_s{seed}_{args.date}"
            _figure(args.out_dir, setting, decoder, seed, args.fig_dir / f"{stem}.png")
            print(f"wrote {args.fig_dir / stem}.png")
            if (setting, decoder) == ("D1", "mlp"):
                draws = (
                    args.fig_dir / f"{stem.replace('learned_', 'learned_draws_')}.png"
                )
                _figure(args.out_dir, setting, decoder, seed, draws, draws_mode=True)
                print(f"wrote {draws}")


def median_gap_seed(out_dir: Path, setting: str, decoder: str) -> int | None:
    """The seed of the median-gap (lower median) non-diverged ``tnbbeta`` run."""
    records = []
    for path in sorted((out_dir / setting / decoder / "tnbbeta").glob("s*.json")):
        record = json.loads(path.read_text())
        if not record["diverged"]:
            records.append(record)
    if not records:
        return None
    records.sort(key=lambda r: (r["gap_elbo"], r["seed"]))
    return int(records[(len(records) - 1) // 2]["seed"])


def _figure(
    out_dir: Path,
    setting: str,
    decoder: str,
    seed: int,
    path: Path,
    draws_mode: bool = False,
) -> None:
    fig = make_subplots(
        rows=1,
        cols=len(ARMS),
        horizontal_spacing=0.01,
    )
    rng = torch.Generator().manual_seed(0)
    for column, arm in enumerate(ARMS):
        arrays = np.load(out_dir / setting / decoder / arm / f"s{seed}.npz")
        record = json.loads(
            (out_dir / setting / decoder / arm / f"s{seed}.json").read_text()
        )
        cls = arrays["cls"]
        index = np.random.default_rng(0).permutation(len(cls))[:NUM_POINTS]
        points = (
            _posterior_draws(arm, arrays, index, rng)
            if draws_mode
            else arrays["centre"][index]
        )
        axes = arrays["u_star"]
        view = _view(axes[0])
        row, col = 1, column + 1
        _sphere_axes(fig, row, col)
        _outline(fig, row, col)
        for c in range(NUM_CLASSES):
            _points(
                fig,
                row,
                col,
                points[cls[index] == c],
                view,
                c,
            )
        labels = [f"u{c + 1}" for c in range(len(axes))] if len(axes) > 1 else ["u*"]
        _axis_arrows(fig, row, col, axes, view, labels)
        _title(
            fig,
            row,
            col,
            f"<b>{TITLES[arm]}</b><br>gap {record['gap_elbo']:.3f} nats",
        )
    for c in range(NUM_CLASSES):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"symbol": SYMBOLS[c], "size": 9, "color": COLORS[c]},
                name=f"Class {c + 1}",
            ),
            row=1,
            col=1,
        )
    kind = "one posterior draw per test x" if draws_mode else "posterior means"
    fig.update_layout(
        template=TEMPLATE_NAME,
        width=1800,
        height=620,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 20, "r": 20, "t": 120, "b": 90},
        title={
            "text": (
                f"{TITLE}<br><sup>Setting {setting}, {decoder} decoder, seed {seed} "
                f"(median-gap TNBBeta seed); {kind} of {NUM_POINTS} test x; arrow = "
                "learned decoder axis u* from the origin; faint = far side</sup>"
            ),
            "x": 0.5,
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": 0.0,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"family": FONT_FAMILY, "size": 15},
        },
    )
    fig.write_image(path, scale=3)


def _posterior_draws(
    arm: str, arrays: np.lib.npyio.NpzFile, index: np.ndarray, rng: torch.Generator
) -> np.ndarray:
    """One draw ``z ~ q(. | x)`` for each selected test ``x``, shape ``(n, 3)``."""
    del rng
    centre = torch.tensor(arrays["centre"][index])
    if arm == "vmf":
        kappa = torch.tensor(arrays["kappa"][index, None])
        return VonMisesFisher(centre, kappa).sample().numpy()
    if arm == "power_spherical":
        kappa = torch.tensor(arrays["kappa"][index])
        return PowerSpherical(centre, kappa).sample().numpy()
    p = torch.tensor(arrays["p"][index])
    mean_direction = torch.where((p > 0.5)[:, None], centre, -centre)
    dist = TNBBetaSpherical(
        mean_direction,
        p,
        torch.tensor(arrays["q"][index]),
        torch.tensor(arrays["epsilon"][index]),
    )
    return dist.sample().numpy()


if __name__ == "__main__":
    main()
