"""TNBBeta p x q ablation grid: S^2 heatmaps with univariate densities below.

Usage:
    uv run python -m apps.distributions.pq_ablation [--output PATH] \\
        [--resolution 60] [--curve-points 400]

Ported from a matplotlib prototype in the sibling ``hypergen`` project, rebuilt
with plotly (this project's plotting library, styled via the shared
``tnbbeta_vae.plotting`` theme) and this project's own validated
``TNBBetaSpherical``/``TNBBetaUnivariate`` rather than reimplementing the
spherical density: the sphere heatmap uses ``TNBBetaSpherical.log_prob``
directly (including its Jacobian and surface-area normalization terms), not
just the univariate log-density evaluated at a cosine-similarity-derived
latitude, which is not a properly normalized spherical density on its own.

For each (p, q) in a grid (epsilon and the mean direction held fixed), plots:

* a 3D S^2 heatmap of ``TNBBetaSpherical``'s log-density (each sphere is
  colored on its own scale -- density magnitude varies by orders of magnitude
  across the grid, so a single shared color scale would wash out most cells);
* the corresponding ``TNBBetaUnivariate`` density curve on (0, 1) directly
  below it.

See ``p_epsilon_ablation`` for the sibling sweep (p x epsilon, q fixed).
"""

from __future__ import annotations

import argparse
import sys

from apps.distributions.ablation_grid import DEFAULT_MEAN_DIRECTION, build_grid_figure

_EPSILON = 1.0
_P_VALUES = [0.05, 0.2, 0.5, 0.8, 0.95]
_Q_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def main(argv: list[str] | None = None) -> None:
    """Builds the p x q ablation grid and writes it as a PNG.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="pq_ablation.png")
    parser.add_argument("--resolution", type=int, default=60)
    parser.add_argument("--curve-points", type=int, default=400)
    args = parser.parse_args(argv)

    figure = build_grid_figure(
        row_values=_P_VALUES,
        row_label="p",
        column_values=_Q_VALUES,
        column_label="q",
        params_for_cell=lambda p, q: (p, q, _EPSILON),
        mean_direction=DEFAULT_MEAN_DIRECTION,
        title=(
            f"TNBBeta p x q ablation on S^2 "
            f"(epsilon={_EPSILON:g}, mu={DEFAULT_MEAN_DIRECTION.tolist()})"
        ),
        resolution=args.resolution,
        curve_points=args.curve_points,
    )
    figure.write_image(
        args.output,
        width=210 * len(_Q_VALUES),
        height=230 * len(_P_VALUES),
        scale=2,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
