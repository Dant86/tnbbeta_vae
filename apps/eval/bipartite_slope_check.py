"""Descriptive slope check: bipartite TNBBeta advantage vs degree covariates (fresh)."""
# ruff: noqa: E501  (long markdown report strings)

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from apps.eval.graph_search_metric_first import (
    PRIMARY,
    load_covariates,
    load_table,
    oriented_difference,
)

PERMS = 20_000
METRIC_LIST = ("truth_kl", "ece", "sampled_log_loss", "sampled_auc", "centre_auc")
COVS = (
    ("realised_degree", "mean degree (realised)"),
    ("expected_degree_cv", "degree CV"),
    ("clustering", "clustering"),
)


def main() -> None:
    """Prints markdown of the analysis."""
    table = load_table("fresh")
    cov = load_covariates("fresh")
    configs = sorted(c for c in table if c.rsplit("-", 1)[0] == "bipartite")
    x = {k: np.array([cov[c][k] for c in configs]) for k, _ in COVS}
    x["log_nominal_degree"] = np.array([cov[c]["log_nominal_degree"] for c in configs])
    x["hetero_level"] = np.array([cov[c]["hetero_level"] for c in configs])
    print(
        f"configs: {len(configs)}; draws: {table[configs[0]]['tnbbeta']['truth_kl'].size}"
    )
    for k in (
        "realised_degree",
        "log_nominal_degree",
        "expected_degree_cv",
        "clustering",
    ):
        print(f"range {k}: {x[k].min():.3f} .. {x[k].max():.3f}")
    print(
        "hetero_level counts:",
        {h: int((x["hetero_level"] == h).sum()) for h in (0.0, 1.0, 2.0)},
    )
    rng = np.random.default_rng(20261103)
    a, b = x["realised_degree"], x["expected_degree_cv"]
    print(
        f"\ncorr(degree, CV) within bipartite: spearman {spearmanr(a, b)[0]:+.3f} (p={spearmanr(a, b)[1]:.3f}), kendall {kendalltau(a, b)[0]:+.3f}"
    )
    print(f"corr(degree, clustering): spearman {spearmanr(a, x['clustering'])[0]:+.3f}")
    print(f"corr(CV, clustering): spearman {spearmanr(b, x['clustering'])[0]:+.3f}")
    adv: dict[str, np.ndarray] = {}
    for m in METRIC_LIST:
        v = []
        for c in configs:
            d = np.stack([oriented_difference(table, c, comp, m) for comp in PRIMARY])
            keep = np.all(np.isfinite(d), axis=0)
            v.append(np.median(d[:, keep], axis=1).min())  # less favourable comparator
        adv[m] = np.array(v)
    for m in METRIC_LIST:
        y = adv[m]
        print(
            f"\n### {m} (higher = TNBBeta better; median {np.median(y):+.4f}, n>0 {int((y > 0).sum())}/20)"
        )
        print(
            "| covariate | Spearman | perm p (2-sided) | Kendall | Kendall p |\n|---|---|---|---|---|"
        )
        for k, label in COVS:
            r = float(spearmanr(x[k], y)[0])  # pyright: ignore
            t, tp = kendalltau(x[k], y)
            perm = np.array(
                [spearmanr(x[k], rng.permutation(y))[0] for _ in range(PERMS // 10)]
            )
            p = (1 + np.sum(np.abs(perm) >= abs(r))) / (1 + perm.size)
            print(f"| {label} | {r:+.3f} | {p:.3f} | {t:+.3f} | {tp:.3f} |")
        pa = _partial(rankdata(a), rankdata(y), rankdata(b))
        pb = _partial(rankdata(b), rankdata(y), rankdata(a))
        print(
            f"partial Spearman(adv, degree | CV) = {pa:+.3f}; (adv, CV | degree) = {pb:+.3f}"
        )
        # permutation p for partials: permute y
        for name, u, w in (("degree|CV", a, b), ("CV|degree", b, a)):
            obs = _partial(rankdata(u), rankdata(y), rankdata(w))
            null = np.array(
                [
                    _partial(rankdata(u), rankdata(rng.permutation(y)), rankdata(w))
                    for _ in range(2000)
                ]
            )
            print(
                f"  perm p partial {name}: {(1 + np.sum(np.abs(null) >= abs(obs))) / (1 + null.size):.3f}"
            )
    print("\n### scatter (sorted by degree)")
    head = (
        "| config | het | degree | CV | clustering | " + " | ".join(METRIC_LIST) + " |"
    )
    print(head + "\n|" + "---|" * (5 + len(METRIC_LIST)))
    for i in np.argsort(a):
        c = configs[i]
        print(
            f"| {c} | {int(x['hetero_level'][i])} | {a[i]:.1f} | {b[i]:.2f} | {x['clustering'][i]:.3f} | "
            + " | ".join(f"{adv[m][i]:+.4f}" for m in METRIC_LIST)
            + " |"
        )


def _partial(u: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    def resid(v: np.ndarray) -> np.ndarray:
        design = np.stack([np.ones_like(z), z], 1)
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    return float(np.corrcoef(resid(u), resid(y))[0, 1])


if __name__ == "__main__":
    main()
