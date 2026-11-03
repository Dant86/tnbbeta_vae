"""Figure explaining the two ITD estimators on one SADIE II subject.

Top row: the left- and right-ear impulse responses (3 kHz low-passed) for three source
directions in the horizontal plane, with each estimator's result. Bottom row: ITD
against the lateral coordinate ``w = a . z`` over all 2818 directions, for both
estimators.

Usage:
    uv run --with h5py --with matplotlib python -m apps.eval.hrtf_itd_figure --out PATH
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py  # pyright: ignore[reportMissingImports]
import matplotlib  # pyright: ignore[reportMissingImports]
import numpy as np

from apps.eval.hrtf_itd_gate import ONSET_DB, SOFA_NAME, _lowpass, _onset, _xcorr_lag
from tnbbeta_vae.paths import data_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pyright: ignore[reportMissingImports]

LEFT, RIGHT = "#2a78d6", "#eb6834"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def main(argv: list[str] | None = None) -> None:
    """Draws the figure for subject H3."""
    parser = argparse.ArgumentParser(description="ITD estimator illustration.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subject", default="H3")
    args = parser.parse_args(argv)
    path = (
        data_dir()
        / "hrtf/sadie2"
        / args.subject
        / f"{args.subject}_HRIR_SOFA"
        / SOFA_NAME.format(s=args.subject)
    )
    with h5py.File(path, "r") as handle:
        irs = np.asarray(handle["Data.IR"])
        rate = float(np.asarray(handle["Data.SamplingRate"]).ravel()[0])
        pos = np.asarray(handle["SourcePosition"])
    filtered = _lowpass(irs, rate)
    onset = (_onset(filtered[:, 1], rate) - _onset(filtered[:, 0], rate)) * 1e6
    xcorr = _xcorr_lag(filtered[:, 0], filtered[:, 1], rate) * 1e6
    azimuth, elevation = np.radians(pos[:, 0]), np.radians(pos[:, 1])
    lateral = np.cos(elevation) * np.sin(azimuth)  # w = a . z with a = +y (left)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), facecolor=SURFACE)
    time = np.arange(irs.shape[-1]) / rate * 1e3
    for column, (az, label) in enumerate(((90, "left"), (0, "front"), (270, "right"))):
        i = int(np.argmin(np.abs(pos[:, 0] - az) + np.abs(pos[:, 1])))
        panel = axes[0, column]
        for ear, colour, name in ((0, LEFT, "left ear"), (1, RIGHT, "right ear")):
            panel.plot(time, irs[i, ear], color=colour, alpha=0.25, linewidth=0.8)
            panel.plot(time, filtered[i, ear], color=colour, linewidth=1.6, label=name)
            level = np.abs(filtered[i, ear]).max() * 10 ** (ONSET_DB / 20)
            panel.axhline(level, color=colour, linestyle=":", linewidth=0.8)
            panel.axhline(-level, color=colour, linestyle=":", linewidth=0.8)
        panel.set_xlim(0, 3.5)
        panel.set_title(
            f"source at the {label} (az {pos[i, 0]:.0f}, el {pos[i, 1]:.0f})\n"
            f"ITD onset {onset[i]:.0f} us | cross-corr {xcorr[i]:.0f} us",
            color=INK,
            fontsize=10,
        )
        panel.set_xlabel("time (ms)", color=INK2)
        panel.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0, 0].set_ylabel(
        "HRIR (faint = raw, bold = 3 kHz low-pass;\ndotted = -30 dB threshold)",
        color=INK2,
        fontsize=9,
    )
    for column, (itd, name) in enumerate(
        ((onset, "onset (-30 dB)"), (xcorr, "cross-correlation"))
    ):
        panel = axes[1, column]
        panel.scatter(lateral, itd, s=4, color=INK2, alpha=0.5, linewidths=0)
        panel.set_xlabel(
            "lateral coordinate w = a . z   (1 = left, -1 = right)", color=INK2
        )
        panel.set_title(f"ITD by {name}, all 2818 directions", color=INK, fontsize=10)
        panel.set_ylim(-1000, 1000)
    axes[1, 0].set_ylabel("ITD (us), positive = left", color=INK2)
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0,
        0.5,
        "Reading: a clean ITD is a smooth,\nmonotone function of w (one cone of\n"
        "confusion per ITD value). The\nthickness of the band at fixed w\n"
        "sets how thick the ring is.",
        color=INK2,
        fontsize=10,
        va="center",
    )
    fig.suptitle(
        f"SADIE II {args.subject}: how the ITD is read off a head-related impulse "
        "response",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")
    # Peak position versus the -30 dB onset, per ear, for the three directions.
    for az in (90, 0, 270):
        i = int(np.argmin(np.abs(pos[:, 0] - az) + np.abs(pos[:, 1])))
        for ear, name in ((0, "left"), (1, "right")):
            peak_ms = time[np.abs(filtered[i, ear]).argmax()]
            on_ms = _onset(filtered[i : i + 1, ear], rate)[0] * 1e3
            print(
                f"az {az:3d} {name:5s}: peak {peak_ms:.3f} ms, "
                f"-30 dB onset {on_ms:.3f} ms"
            )


if __name__ == "__main__":
    main()
