import numpy as np

from apps.eval.graph_search_kappa import (
    best_kappa,
    crossfit_differences,
    recal_tests,
    value_table,
)
from apps.synthetic.graph_search import KAPPA_GRID


def _runs(tnb_gain: float, draws: int = 16) -> list[dict]:
    rng = np.random.default_rng(0)
    runs = []
    for draw in range(draws):
        for condition in ["tnbbeta", "vmf_init", "power_spherical_init"] + [
            f"{f}_k{k:g}" for f in ("vmf", "power_spherical") for k in KAPPA_GRID
        ]:
            loss = 0.6 + 0.001 * rng.standard_normal()
            if condition == "tnbbeta":
                loss -= tnb_gain
            if condition.endswith("_k32"):
                loss -= 0.01
            runs.append(
                {
                    "config_id": "ring-00",
                    "draw": draw,
                    "condition": condition,
                    "metrics": {
                        "sampled_log_loss": loss,
                        "recal_log_loss": loss,
                        "recal_truth_kl": loss,
                        "post_r_bar": 0.5,
                    },
                }
            )
    return runs


def test_best_kappa_picks_the_grid_point_with_the_lowest_loss() -> None:
    table = value_table(_runs(0.0))

    assert best_kappa(table, "ring-00", "vmf", "sampled_log_loss", list(range(16))) == (
        "vmf_k32"
    )


def test_crossfit_uses_every_draw_and_reports_gain() -> None:
    table = value_table(_runs(0.05))
    diffs, chosen = crossfit_differences(
        table, "ring-00", "vmf", "sampled_log_loss", list(range(16))
    )

    assert len(diffs) == 16 and chosen == ["vmf_k32", "vmf_k32"]
    assert np.mean(diffs) > 0.03


def test_recal_tests_detect_a_consistent_advantage() -> None:
    rows = recal_tests(
        value_table(_runs(0.05)), ["ring-00"], {"ring-00": list(range(16))}
    )

    assert {r["metric"] for r in rows} == {"recal_log_loss", "recal_truth_kl"}
    assert all(r["p"] < 0.01 for r in rows)
