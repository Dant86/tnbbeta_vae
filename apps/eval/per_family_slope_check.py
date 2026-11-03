"""Descriptive per-family slope check: TNBBeta advantage vs degree, CV, clustering (fresh)."""
# ruff: noqa: E501  (long markdown report strings)

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from apps.eval.bipartite_slope_check import _partial
from apps.eval.graph_search_metric_first import (
    PRIMARY,
    family_of,
    load_covariates,
    load_table,
    oriented_difference,
)

PERMS = 10_000
BOOT = 2_000
SEED = 20261103
METRIC_LIST = ("truth_kl", "ece", "sampled_log_loss", "sampled_auc", "centre_auc")
COVS = (
    ("realised_degree", "degree"),
    ("expected_degree_cv", "CV"),
    ("clustering", "clust"),
)
LOW_DEG, HIGH_DEG, HIGH_CV = 11.0, 20.0, 0.5


def main() -> None:
    """Prints markdown tables of the per-family analysis."""
    table, cov = load_table("fresh"), load_covariates("fresh")
    fams = sorted({family_of(c) for c in table})
    rng = np.random.default_rng(SEED)
    res = {f: _family(table, cov, f, rng) for f in fams}
    print(
        "## A. truth-KL advantage vs covariates (Spearman / Kendall; perm p uncorrected)\n"
    )
    print(
        "| family | median adv | n>0 | rho deg (p) [boot 95%] | tau deg | rho CV (p) | tau CV | rho clust (p) | tau clust | partial deg\\|CV (p) | partial CV\\|deg (p) | rho deg drop top-2 |"
    )
    print("|" + "---|" * 12)
    for f in fams:
        r = res[f]
        s = r["m"]["truth_kl"]
        print(
            f"| {f} | {r['med']['truth_kl']:+.4f} | {r['npos']['truth_kl']}/20 | {s['rho'][0]:+.3f} ({s['p'][0]:.3f}) [{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}] | {s['tau'][0]:+.3f} | {s['rho'][1]:+.3f} ({s['p'][1]:.3f}) | {s['tau'][1]:+.3f} | {s['rho'][2]:+.3f} ({s['p'][2]:.3f}) | {s['tau'][2]:+.3f} | {s['pd'][0]:+.3f} ({s['pd'][1]:.3f}) | {s['pc'][0]:+.3f} ({s['pc'][1]:.3f}) | {s['drop2']:+.3f} |"
        )
    print(
        "\n## B. exploratory multiplicity guide over the 10 truth-KL-vs-degree tests (two-sided perm p)\n"
    )
    ps = np.array([res[f]["m"]["truth_kl"]["p"][0] for f in fams])
    holm, bh = _holm(ps), _bh(ps)
    print("| family | rho | p | Holm adj | BH adj |\n|---|---|---|---|---|")
    for i, f in enumerate(fams):
        print(
            f"| {f} | {res[f]['m']['truth_kl']['rho'][0]:+.3f} | {ps[i]:.4f} | {holm[i]:.4f} | {bh[i]:.4f} |"
        )
    for m in METRIC_LIST[1:]:
        print(f"\n## C. {m}: advantage vs covariates (higher = TNBBeta better)\n")
        print(
            "| family | median adv | n>0 | rho deg (p) [boot 95%] | rho CV (p) | rho clust (p) | partial deg\\|CV (p) | partial CV\\|deg (p) | rho deg drop top-2 |"
        )
        print("|" + "---|" * 9)
        for f in fams:
            r, s = res[f], res[f]["m"][m]
            print(
                f"| {f} | {r['med'][m]:+.4f} | {r['npos'][m]}/20 | {s['rho'][0]:+.3f} ({s['p'][0]:.3f}) [{s['ci'][0]:+.2f},{s['ci'][1]:+.2f}] | {s['rho'][1]:+.3f} ({s['p'][1]:.3f}) | {s['rho'][2]:+.3f} ({s['p'][2]:.3f}) | {s['pd'][0]:+.3f} ({s['pd'][1]:.3f}) | {s['pc'][0]:+.3f} ({s['pc'][1]:.3f}) | {s['drop2']:+.3f} |"
            )
    print("\n## D. design ranges and corners\n")
    print(
        f"low degree < {LOW_DEG:g}, high degree > {HIGH_DEG:g}, high CV > {HIGH_CV:g}; bands = groups of sorted CV split at gaps > 0.1\n"
    )
    print(
        "| family | degree range | CV range | CV bands | het levels | corr(deg,CV) | low-deg&highCV n (mean adv) | high-deg&highCV n (mean adv) | low-deg&lowCV n | high-deg&lowCV n | frac TNB>0 low-half / high-half deg | frac AUC adv<0 (sampled/centre) |"
    )
    print("|" + "---|" * 12)
    for f in fams:
        d = res[f]["design"]
        print(
            f"| {f} | {d['deg'][0]:.1f}-{d['deg'][1]:.1f} | {d['cv'][0]:.2f}-{d['cv'][1]:.2f} | {d['bands']} | {d['levels']} | {d['corr']:+.2f} | {d['ll'][0]} ({_f(d['ll'][1])}) | {d['hh'][0]} ({_f(d['hh'][1])}) | {d['lo_lo']} | {d['hi_lo']} | {d['halves'][0]:.2f} / {d['halves'][1]:.2f} | {d['auc_neg'][0]:.2f} / {d['auc_neg'][1]:.2f} |"
        )
    print(
        "\n## E. per-config scatter (family, config, het, degree, CV, clustering, truth_kl, ece, sampled_auc)\n"
    )
    for f in fams:
        r = res[f]
        for i in np.argsort(r["x"]["realised_degree"]):
            print(
                f"{r['configs'][i]},{int(r['x']['hetero_level'][i])},{r['x']['realised_degree'][i]:.1f},{r['x']['expected_degree_cv'][i]:.2f},{r['x']['clustering'][i]:.3f},{r['adv']['truth_kl'][i]:+.4f},{r['adv']['ece'][i]:+.4f},{r['adv']['sampled_auc'][i]:+.4f}"
            )


def _family(table, cov, fam, rng):
    configs = sorted(c for c in table if family_of(c) == fam)
    x = {
        k: np.array([cov[c][k] for c in configs])
        for k in ("realised_degree", "expected_degree_cv", "clustering", "hetero_level")
    }
    adv = {}
    for m in METRIC_LIST:
        v = []
        for c in configs:
            d = np.stack([oriented_difference(table, c, comp, m) for comp in PRIMARY])
            keep = np.all(np.isfinite(d), axis=0)
            v.append(np.median(d[:, keep], axis=1).min())
        adv[m] = np.array(v)
    deg, cv = x["realised_degree"], x["expected_degree_cv"]
    out: dict[str, Any] = {
        "configs": configs,
        "x": x,
        "adv": adv,
        "m": {},
        "med": {},
        "npos": {},
    }
    for m in METRIC_LIST:
        y = adv[m]
        out["med"][m], out["npos"][m] = float(np.median(y)), int((y > 0).sum())
        s: dict[str, Any] = {"rho": [], "p": [], "tau": []}
        for k, _ in COVS:
            s["rho"].append(_rho(x[k], y))
            s["tau"].append(float(kendalltau(x[k], y)[0]))  # pyright: ignore
            s["p"].append(_perm_p(x[k], y, rng))
        idx = rng.integers(0, len(y), (BOOT, len(y)))
        boots = np.array([_rho(deg[i], y[i]) for i in idx])
        boots = boots[np.isfinite(boots)]
        s["ci"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        s["pd"] = _partial_p(deg, y, cv, rng)
        s["pc"] = _partial_p(cv, y, deg, rng)
        s["drop2"] = _drop2(deg, y)
        out["m"][m] = s
    y = adv["truth_kl"]
    cvs = np.sort(cv)
    bands = 1 + int((np.diff(cvs) > 0.1).sum())
    ld, hd, hc = deg < LOW_DEG, deg > HIGH_DEG, cv > HIGH_CV
    med = np.median(deg)
    out["design"] = {
        "deg": (deg.min(), deg.max()),
        "cv": (cv.min(), cv.max()),
        "bands": bands,
        "levels": int(len(set(x["hetero_level"]))),
        "corr": (
            float(spearmanr(deg, cv)[0]) if cv.std() > 0 else float("nan")  # pyright: ignore
        ),
        "ll": (
            int((ld & hc).sum()),
            float(y[ld & hc].mean()) if (ld & hc).any() else float("nan"),
        ),
        "hh": (
            int((hd & hc).sum()),
            float(y[hd & hc].mean()) if (hd & hc).any() else float("nan"),
        ),
        "lo_lo": int((ld & ~hc).sum()),
        "hi_lo": int((hd & ~hc).sum()),
        "halves": (float((y[deg <= med] > 0).mean()), float((y[deg > med] > 0).mean())),
        "auc_neg": (
            float((adv["sampled_auc"] < 0).mean()),
            float((adv["centre_auc"] < 0).mean()),
        ),
    }
    return out


def _rho(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _perm_p(a, y, rng):
    obs = _rho(a, y)
    if not np.isfinite(obs):
        return float("nan")
    ra = rankdata(a) - rankdata(a).mean()
    ry = rankdata(y) - rankdata(y).mean()
    perms = np.argsort(rng.random((PERMS, len(y))), axis=1)
    null = (ry[perms] @ ra) / np.sqrt((ra**2).sum() * (ry**2).sum())
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + PERMS))


def _partial_p(u, y, z, rng):
    if np.std(z) == 0 or np.std(u) == 0:
        return (float("nan"), float("nan"))
    ru, rz = rankdata(u), rankdata(z)
    obs = _partial(ru, rankdata(y), rz)
    null = np.array(
        [_partial(ru, rankdata(rng.permutation(y)), rz) for _ in range(2000)]
    )
    return (obs, float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + null.size)))


def _drop2(a, y):
    keep = list(range(len(y)))
    for _ in range(2):
        base = _rho(a[keep], y[keep])
        shifts = [
            abs(
                _rho(a[[j for j in keep if j != i]], y[[j for j in keep if j != i]])
                - base
            )
            for i in keep
        ]
        keep.pop(int(np.argmax(shifts)))
    return _rho(a[keep], y[keep])


def _holm(p):
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1)
    return out


def _bh(p):
    o = np.argsort(p)
    n = len(p)
    adj = np.minimum.accumulate((n / (np.arange(n) + 1) * p[o])[::-1])[::-1]
    out = np.empty_like(p)
    out[o] = np.minimum(adj, 1)
    return out


def _f(v):
    return "n/a" if not np.isfinite(v) else f"{v:+.3f}"


if __name__ == "__main__":
    main()
