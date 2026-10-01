"""Summarizes com-DBLP bridge-node posterior shape statistics across seeds.

Usage:
    uv run python -m apps.eval.dblp_posterior_shape_summary [--results-dir dblp_results]

Reads the 15 ``bridge_diagnostic_final.json`` files (3 families x 5 seeds) produced by
``apps.eval.dblp_bridge_diagnostic``'s ``posterior_stats`` field and aggregates, per
family and statistic, bridge vs. non-bridge nodes' mean +- std across seeds, plus a
Welch's t-test (``equal_var=False``, matching ``apps/eval/svae_table.py``'s convention)
of whether bridge and non-bridge differ for that statistic.

vMF and Power Spherical only have ``entropy_mean`` and ``r_bar`` (closed-form entropy,
no (p, q, epsilon)); TNBBeta's ``entropy_mean`` is always NaN (no closed form) but it
additionally has ``p_mean``, ``q_mean``, ``epsilon_mean``, ``m_mean`` and
``frac_bimodal`` -- see ``dblp_bridge_diagnostic.py``'s ``_posterior_stats`` and
``writeup/weekly_markdown_summaries/week_2/tnbbeta_vs_power_spherical_expressivity.md``'s
Theorem 5.1/Corollary 5.3 for what ``m`` and ``frac_bimodal`` mean.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import stats

__all__ = ["aggregate", "load_results"]

_FAMILIES = ["tnbbeta", "power_spherical", "vmf"]
_FAMILY_TITLES = {
    "tnbbeta": "TNBBeta",
    "power_spherical": "Power Spherical",
    "vmf": "vMF",
}
# Every family reports these two; TNBBeta additionally reports the rest (see module
# docstring). A statistic entirely absent for a family (e.g. (p, q, epsilon) for vMF)
# is simply skipped for that family's table, not reported as NaN.
_STATISTICS = [
    "entropy_mean",
    "r_bar",
    "p_mean",
    "q_mean",
    "epsilon_mean",
    "m_mean",
    "frac_bimodal",
]

# seeds[family][statistic]["bridge" | "non_bridge"] -> list of per-seed values.
Seeds = dict[str, dict[str, dict[str, list[float]]]]


def main(argv: list[str] | None = None) -> None:
    """Prints the bridge-vs-non-bridge posterior shape summary for every family.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("dblp_results"))
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args(argv)

    seeds = load_results(args.results_dir)
    for family in _FAMILIES:
        rows = aggregate(seeds[family], alpha=args.alpha)
        if not rows:
            continue
        print(f"\n## {_FAMILY_TITLES[family]}\n")
        print("| Statistic | Bridge | Non-bridge | p (Welch) |")
        print("|---|---|---|---|")
        for row in rows:
            print(
                f"| {row['statistic']} | {row['bridge']} "
                f"| {row['non_bridge']} | {row['p_value']} |"
            )


def load_results(results_dir: Path) -> Seeds:
    """Loads per-(family, statistic, bridge/non-bridge) seed lists.

    Args:
        results_dir: Directory holding ``dblp_<family>_d16_seed<seed>/
            bridge_diagnostic_final.json`` for each family/seed.

    Returns:
        ``seeds[family][statistic]["bridge" | "non_bridge"]``, each a list of one value
        per seed found on disk for that family.
    """
    seeds: Seeds = {
        family: {statistic: defaultdict(list) for statistic in _STATISTICS}
        for family in _FAMILIES
    }
    for family in _FAMILIES:
        pattern = str(
            results_dir / f"dblp_{family}_d16_seed*" / "bridge_diagnostic_final.json"
        )
        for path in sorted(glob.glob(pattern)):
            with Path(path).open() as handle:
                record = json.load(handle)
            for group, key in (
                ("bridge", "bridge_nodes"),
                ("non_bridge", "non_bridge_nodes"),
            ):
                posterior_stats = record[key]["posterior_stats"]
                for statistic in _STATISTICS:
                    if statistic in posterior_stats:
                        seeds[family][statistic][group].append(
                            posterior_stats[statistic]
                        )
    return seeds


def aggregate(
    family_seeds: dict[str, dict[str, list[float]]], *, alpha: float
) -> list[dict[str, Any]]:
    """Builds one display row per statistic this family actually reports.

    A statistic is skipped entirely (not emitted as a NaN row) if no seed reported it
    (e.g. (p, q, epsilon) for vMF/Power Spherical) or if every value is NaN (TNBBeta's
    ``entropy_mean``, which has no closed form).

    Args:
        family_seeds: ``seeds[family]`` from :func:`load_results`.
        alpha: Significance threshold for flagging the Welch t-test's p-value.

    Returns:
        One dict per statistic with ``statistic``, formatted ``bridge``/``non_bridge``
        ("mean +- std"), and a formatted ``p_value`` (bolded if ``< alpha``, "n/a" if
        either group is degenerate for the test).
    """
    rows: list[dict[str, Any]] = []
    for statistic in _STATISTICS:
        bridge = np.array(family_seeds[statistic]["bridge"], dtype=float)
        non_bridge = np.array(family_seeds[statistic]["non_bridge"], dtype=float)
        if bridge.size == 0 or non_bridge.size == 0:
            continue
        if np.isnan(bridge).all() and np.isnan(non_bridge).all():
            continue

        p_value_display = "n/a"
        if (
            bridge.size > 1
            and non_bridge.size > 1
            and np.std(bridge) + np.std(non_bridge) > 0
        ):
            test: Any = stats.ttest_ind(bridge, non_bridge, equal_var=False)
            p_value_display = (
                f"**{test.pvalue:.4f}**"
                if test.pvalue < alpha
                else f"{test.pvalue:.4f}"
            )

        rows.append(
            {
                "statistic": statistic,
                "bridge": f"{bridge.mean():.4f} +/- {bridge.std():.4f}",
                "non_bridge": f"{non_bridge.mean():.4f} +/- {non_bridge.std():.4f}",
                "p_value": p_value_display,
            }
        )
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
