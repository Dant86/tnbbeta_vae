"""Exploratory: does the truth-KL advantage track mean degree, n or edge count?

Reads the fresh-stage graph-search results (200 configs, 10 families) and asks whether
the registered H1 (within-family Spearman of ``S_c = E_c / KL0`` against log nominal
degree) survives controlling for the node count, since at fixed ``n`` a higher degree
also means more training edges.

Usage (login node, light; needs the fresh-stage runs and their caches):
    uv run python -m apps.eval.edges_vs_degree_check
"""

import json
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

from apps.eval.scale_free_density_figure import pooled_points
from tnbbeta_vae.paths import checkpoint_dir

Array = npt.NDArray[np.float64]
BOOTSTRAPS = 4000
PERMUTATIONS = 3000
SEED = 0


def main() -> None:
    """Prints the correlations, bootstrap intervals and one permutation p-value."""
    rng = np.random.default_rng(SEED)
    points = pooled_points()
    configs = {
        c["config_id"]: c
        for c in json.loads(
            (checkpoint_dir() / "graph_search" / "configs_fresh.json").read_text()
        )
    }
    family = np.array([p["family"] for p in points])
    nodes = np.array([configs[p["config"]]["num_nodes"] for p in points], dtype=float)
    nominal = np.array([p["nominal_degree"] for p in points])
    realised = np.array([p["degree"] for p in points])
    advantage = np.array([p["E"] for p in points])
    skill = advantage / np.array([p["kl0"] for p in points])
    data = {
        "skill": skill,
        "nats": advantage,
        "log_degree": np.log(nominal),
        "log_n": np.log(nodes),
        "log_edges": np.log(nodes * realised / 2),
    }

    names = [
        "rho(S, log degree)",
        "rho(S, log n)",
        "rho(S, log edges)",
        "partial rho(S, log degree | log n)",
        "partial rho(S, log n | log degree)",
        "beta log degree (S ~ ld + ln, ranks)",
        "beta log n",
        "rho(E nats, log degree)",
        "rho(E nats, log edges)",
    ]
    everyone = np.arange(len(skill))
    observed = _statistics(data, family, everyone)
    groups = [np.flatnonzero(family == g) for g in np.unique(family)]
    boot = np.array(
        [
            _statistics(
                data,
                family,
                np.concatenate([rng.choice(g, len(g)) for g in groups]),
            )
            for _ in range(BOOTSTRAPS)
        ]
    )
    print(f"configs {len(skill)}, families {len(groups)}")
    for label, a, b in (
        ("log degree vs log n", "log_degree", "log_n"),
        ("log degree vs log edges", "log_degree", "log_edges"),
        ("log n vs log edges", "log_n", "log_edges"),
    ):
        print(f"within-family corr({label}) = {_corr(data[a], data[b], family):+.3f}")
    for name, value, draws in zip(names, observed, boot.T, strict=True):
        low, high = np.percentile(draws, [2.5, 97.5])
        print(f"{name:42s} {value:+.3f}  95% boot [{low:+.3f}, {high:+.3f}]")
    count = 0
    for _ in range(PERMUTATIONS):
        shuffled = skill.copy()
        for g in groups:
            shuffled[g] = skill[rng.permutation(g)]
        count += (
            _partial(shuffled, data["log_degree"], data["log_n"], family) >= observed[3]
        )
    p_value = (1 + count) / (1 + PERMUTATIONS)
    print(
        f"one-sided permutation p, partial degree | n: {p_value:.4f}"
        " (unadjusted permutation, not Freedman-Lane)"
    )


def _statistics(
    data: dict[str, Array], family: Array, index: npt.NDArray[np.int_]
) -> list[float]:
    f = family[index]
    s, e = data["skill"][index], data["nats"][index]
    ld, ln, le = (data[k][index] for k in ("log_degree", "log_n", "log_edges"))
    beta = _coefficients(s, [ld, ln], f)
    return [
        _corr(s, ld, f),
        _corr(s, ln, f),
        _corr(s, le, f),
        _partial(s, ld, ln, f),
        _partial(s, ln, ld, f),
        float(beta[0]),
        float(beta[1]),
        _corr(e, ld, f),
        _corr(e, le, f),
    ]


def _within_family_ranks(x: Array, family: Array) -> Array:
    ranks = rankdata(x).astype(float)
    for g in np.unique(family):
        ranks[family == g] -= ranks[family == g].mean()
    return ranks


def _corr(a: Array, b: Array, family: Array) -> float:
    ra, rb = _within_family_ranks(a, family), _within_family_ranks(b, family)
    return float(np.corrcoef(ra, rb)[0, 1])


def _partial(y: Array, x: Array, z: Array, family: Array) -> float:
    ry, rx, rz = (_within_family_ranks(v, family) for v in (y, x, z))

    def residual(a: Array) -> Array:
        return a - np.polyval(np.polyfit(rz, a, 1), rz)

    return float(np.corrcoef(residual(ry), residual(rx))[0, 1])


def _coefficients(y: Array, xs: list[Array], family: Array) -> Any:
    design = np.stack([_within_family_ranks(x, family) for x in xs], axis=1)
    design /= design.std(0)
    ry = _within_family_ranks(y, family)
    ry /= ry.std()
    return np.linalg.lstsq(design, ry, rcond=None)[0]


if __name__ == "__main__":
    main()
