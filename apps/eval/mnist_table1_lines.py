"""Line-chart version of the MNIST Table 1 comparison (LL, L[q], RE, KL vs. latent
dimension), replacing a dense table with 4 line-plus-shaded-std-band panels.

Usage:
    uv run python -m apps.eval.mnist_table1_lines [--output PATH]

Reuses ``apps.eval.dblp_dose_response_plot``'s ``add_band_and_line``/``to_rgba``
helpers -- the same shaded-band drawing already used for the com-DBLP dose-response
figure -- rather than reimplementing it a second time.

The data is a hardcoded transcription of the published results: Gaussian (N-VAE)
and vMF (S-VAE) are from week 1's Table 1
(``writeup/weekly_tex_summaries/week_1/sections/3_results.tex``); TNBBeta and
Power Spherical are from week 2's update to the same table
(``writeup/weekly_tex_summaries/week_2/sections/2_results.tex``), which replaced
week 1's Gaussian column with Power Spherical at the same (model, dimension) grid --
this script is what puts all four models back in one place. This is visualizing an
already-recorded, already-published result, not a re-run, so the numbers are
transcribed rather than reloaded from any checkpoint.

vMF is dropped at d=40: that cell is the documented S-VAE instability
(``LL=-144.28 +- 31.06`` vs. TNBBeta/Power Spherical's ``~-89.1 +- 0.1``-ish), and
plotting it on the same linear axis as the other three models would flatten every
other point in every panel into visual noise. It's called out in the figure's
caption instead (by the paper's own prose, not this script) rather than drawn.
Gaussian has no such instability at d=40 and is plotted at every dimension.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apps.eval.dblp_dose_response_plot import add_band_and_line
from tnbbeta_vae.plotting import TEMPLATE_NAME

__all__ = ["build_figure", "main"]

_DIMS = [2, 5, 10, 20, 40]
# Arrow direction: higher is better for LL/L[q]/RE; lower is better for KL, since
# L[q] = RE - KL exactly (verifiable against the table), so a smaller KL directly
# raises the ELBO for the same reconstruction term.
_METRICS = [
    ("ll", "LL (↑)"),
    ("elbo", "L[q] (↑)"),
    ("re", "RE (↑)"),
    ("kl", "KL (↓)"),
]

# model -> dim -> metric -> (mean, std). vMF has no d=40 entry (see module docstring).
_DATA: dict[str, dict[int, dict[str, tuple[float, float]]]] = {
    "gaussian": {
        2: {
            "ll": (-132.64, 0.30),
            "elbo": (-136.81, 0.28),
            "re": (-129.19, 0.35),
            "kl": (7.62, 0.10),
        },
        5: {
            "ll": (-105.38, 0.11),
            "elbo": (-109.53, 0.09),
            "re": (-96.12, 0.12),
            "kl": (13.41, 0.17),
        },
        10: {
            "ll": (-88.75, 0.12),
            "elbo": (-92.91, 0.14),
            "re": (-72.38, 0.10),
            "kl": (20.53, 0.12),
        },
        20: {
            "ll": (-83.67, 0.09),
            "elbo": (-87.64, 0.11),
            "re": (-61.87, 0.14),
            "kl": (25.77, 0.07),
        },
        40: {
            "ll": (-83.24, 0.05),
            "elbo": (-87.40, 0.07),
            "re": (-60.77, 0.11),
            "kl": (26.63, 0.14),
        },
    },
    "vmf": {
        2: {
            "ll": (-130.86, 0.43),
            "elbo": (-134.42, 0.42),
            "re": (-126.89, 0.47),
            "kl": (7.53, 0.08),
        },
        5: {
            "ll": (-105.27, 0.14),
            "elbo": (-109.32, 0.08),
            "re": (-95.56, 0.12),
            "kl": (13.76, 0.06),
        },
        10: {
            "ll": (-89.38, 0.02),
            "elbo": (-93.60, 0.06),
            "re": (-71.84, 0.08),
            "kl": (21.77, 0.05),
        },
        20: {
            "ll": (-86.84, 0.04),
            "elbo": (-92.10, 0.05),
            "re": (-62.63, 0.21),
            "kl": (29.47, 0.18),
        },
        # d=40 intentionally omitted: S-VAE instability, see module docstring.
    },
    "tnbbeta": {
        2: {
            "ll": (-130.11, 0.57),
            "elbo": (-134.21, 0.46),
            "re": (-126.43, 0.54),
            "kl": (7.78, 0.11),
        },
        5: {
            "ll": (-104.89, 0.22),
            "elbo": (-109.15, 0.22),
            "re": (-95.36, 0.26),
            "kl": (13.79, 0.13),
        },
        10: {
            "ll": (-89.29, 0.10),
            "elbo": (-93.65, 0.14),
            "re": (-71.96, 0.25),
            "kl": (21.69, 0.14),
        },
        20: {
            "ll": (-86.90, 0.23),
            "elbo": (-92.12, 0.15),
            "re": (-62.33, 0.41),
            "kl": (29.78, 0.53),
        },
        40: {
            "ll": (-89.09, 0.15),
            "elbo": (-96.38, 0.14),
            "re": (-62.53, 0.42),
            "kl": (33.86, 0.36),
        },
    },
    "power_spherical": {
        2: {
            "ll": (-131.02, 0.39),
            "elbo": (-134.56, 0.28),
            "re": (-127.03, 0.40),
            "kl": (7.53, 0.13),
        },
        5: {
            "ll": (-105.37, 0.17),
            "elbo": (-109.38, 0.22),
            "re": (-95.67, 0.22),
            "kl": (13.71, 0.08),
        },
        10: {
            "ll": (-89.42, 0.10),
            "elbo": (-93.69, 0.09),
            "re": (-71.97, 0.10),
            "kl": (21.71, 0.03),
        },
        20: {
            "ll": (-86.95, 0.08),
            "elbo": (-92.17, 0.07),
            "re": (-62.72, 0.18),
            "kl": (29.45, 0.15),
        },
        40: {
            "ll": (-89.04, 0.11),
            "elbo": (-96.37, 0.13),
            "re": (-62.17, 0.23),
            "kl": (34.20, 0.23),
        },
    },
}
# (data key, legend label, color, dashed, marker symbol). Dashed/muted marks the
# Gaussian baseline as visually distinct from the three hyperspherical models being
# compared; distinct marker symbols keep vMF and TNBBeta tell-apart-able even where
# their lines nearly coincide (d=2..20, where the two are statistically
# indistinguishable -- see the module docstring).
_MODELS = [
    ("gaussian", "Gaussian (N-VAE)", "#7f7f7f", True, "circle"),
    ("vmf", "vMF (S-VAE)", "#1f77b4", False, "square"),
    ("tnbbeta", "TNBBeta", "#d62728", False, "diamond"),
    ("power_spherical", "Power Spherical", "#2ca02c", False, "triangle-up"),
]


def main(argv: list[str] | None = None) -> None:
    """Builds the 4-panel line chart and writes it as a PNG.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="mnist_table1_lines.png")
    args = parser.parse_args(argv)

    figure = build_figure(_DATA)
    figure.write_image(args.output, width=1200, height=900, scale=2)
    print(f"Wrote {args.output}")


def build_figure(
    data: Mapping[str, Mapping[int, Mapping[str, tuple[float, float]]]],
    *,
    models: list[tuple[str, str, str, bool, str]] = _MODELS,
    dims: list[int] = _DIMS,
) -> go.Figure:
    """Builds the 2x2 (one panel per metric) line-plus-shaded-std-band figure.

    Args:
        data: ``model -> dim -> metric -> (mean, std)``, as module-level ``_DATA``
            (a model may omit a dimension, e.g. vMF's dropped ``d=40``). Every entry
            in ``data`` must have a value for all 4 of ``_METRICS``, the fixed 2x2
            grid this function always draws.
        models: ``(data key, legend label, color, dashed, marker symbol)`` tuples,
            defaulting to module-level ``_MODELS``; overridable for tests.
        dims: Dimensions to plot (as category labels), defaulting to module-level
            ``_DIMS``; overridable for tests.

    Returns:
        The completed figure, ready for ``write_image``.
    """
    dim_labels = [str(d) for d in dims]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[label for _, label in _METRICS],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for metric_index, (metric_key, _) in enumerate(_METRICS):
        row, col = metric_index // 2 + 1, metric_index % 2 + 1
        for model_key, display_name, color, dashed, marker_symbol in models:
            per_dim = data[model_key]
            present_dims = [d for d in dims if d in per_dim]
            x = [str(d) for d in present_dims]
            mean = np.array([per_dim[d][metric_key][0] for d in present_dims])
            std = np.array([per_dim[d][metric_key][1] for d in present_dims])
            add_band_and_line(
                figure,
                row=row,
                col=col,
                x=x,
                mean=mean,
                std=std,
                color=color,
                name=display_name,
                show_legend=metric_index == 0,
                dash=dashed,
                marker_symbol=marker_symbol,
            )

    figure.update_xaxes(
        type="category", categoryorder="array", categoryarray=dim_labels, title_text="d"
    )
    figure.update_layout(
        template=TEMPLATE_NAME,
        title_text="MNIST Table 1 (mean ± 1 std, 5 seeds)",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
        margin={"l": 50, "r": 20, "t": 60, "b": 70},
    )
    return figure


if __name__ == "__main__":
    main(sys.argv[1:])
