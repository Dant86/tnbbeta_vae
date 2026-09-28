"""TNBBeta p x epsilon ablation grid: S^2 heatmaps with univariate densities below.

Usage:
    uv run python -m apps.distributions.p_epsilon_ablation [--output PATH] \\
        [--resolution 60] [--curve-points 400]

Sibling sweep to ``pq_ablation`` (p x q, epsilon fixed): here q is fixed at 0
(the collapsed special case seen throughout this project's trained posteriors --
see ``writeup/results/tnbbeta_vmf_geometric_convergence_2026-09-23.md``) and
epsilon sweeps across all three of its qualitatively distinct regimes
(Proposition 3.1 in ``TNBBetaUnivariate``'s docstring): epsilon < 1 (density
diverges at both poles), epsilon = 1 (flat baseline; also the exact
uniform-sphere special case at p=0.5), epsilon > 1 (unimodal, vanishing at the
poles, increasingly concentrated as epsilon grows).
"""

from __future__ import annotations

import argparse
import sys

from apps.distributions.ablation_grid import DEFAULT_MEAN_DIRECTION, build_grid_figure

_Q = 0.0
_P_VALUES = [0.05, 0.2, 0.5, 0.8, 0.95]
_EPSILON_VALUES = [0.5, 0.8, 1.0, 1.5, 3.0, 10.0]


def main(argv: list[str] | None = None) -> None:
    """Builds the p x epsilon ablation grid and writes it as a PNG.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="p_epsilon_ablation.png")
    parser.add_argument("--resolution", type=int, default=60)
    parser.add_argument("--curve-points", type=int, default=400)
    args = parser.parse_args(argv)

    figure = build_grid_figure(
        row_values=_P_VALUES,
        row_label="p",
        column_values=_EPSILON_VALUES,
        column_label="epsilon",
        params_for_cell=lambda p, epsilon: (p, _Q, epsilon),
        mean_direction=DEFAULT_MEAN_DIRECTION,
        title=(
            f"TNBBeta p x epsilon ablation on S^2 "
            f"(q={_Q:g}, mu={DEFAULT_MEAN_DIRECTION.tolist()})"
        ),
        resolution=args.resolution,
        curve_points=args.curve_points,
    )
    figure.write_image(
        args.output,
        width=210 * len(_EPSILON_VALUES),
        height=230 * len(_P_VALUES),
        scale=2,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
