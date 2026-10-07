"""Scratch check #2: can the (A,B,C) latents be exactly SPLIT (thinned)?

Test 1 (already confirmed in tnbbeta_epsilon_additivity_check.py) showed
epsilon is additive: summing independent (A,B,C) triples at (p,q,eps1) and
(p,q,eps2) and redrawing Y from the combined Beta reproduces
TNBbeta(p,q,eps1+eps2) exactly. That's a "merge" operator -- useful for a
*generative* (noise -> data) direction, since merging concentrates mass
around p (TNBbeta(p,q,eps) sharpens toward a point mass at p as eps grows
with p,q fixed; eps=1,q=0 is the Beta(1,1)=Uniform reference point).

This script tests the REVERSE operator -- an exact split/thinning of a
combined-epsilon (A,B,C) state into two independent pieces at eps1,eps2 --
which is what a *forward* (data -> noise) corruption process would need.

Claimed algorithm (standard NB/Gamma-Poisson splitting facts, chained
through the C -> A,B dependency):
    1. C1 | C1+C2=c ~ BetaBinomial(c; eps1, eps2),  C2 = c - C1
    2. A1 | A1+A2=a ~ BetaBinomial(a; eps1+C1, eps2+C2),  A2 = a - A1
    3. B1 | B1+B2=b ~ BetaBinomial(b; eps1+C1, eps2+C2),  B2 = b - B1

If correct, (A1,B1,C1) should be marginally distributed exactly as a
*direct* Theorem-4.1 draw at (p, q, eps1) -- independent of the fact that
it was produced by splitting a deeper eps1+eps2 sample.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/Users/vedantpathak/Developer/projects/packages/tnbbeta_vae/src")

import torch  # noqa: E402

from tnbbeta_vae.distributions.tnbbeta_univariate import TNBBetaUnivariate  # noqa: E402

RNG = np.random.default_rng(1)
N = 200_000


def sample_augmented_abc(p: float, q: float, eps: float, n: int, rng: np.random.Generator):
    c = rng.negative_binomial(eps, 1 - q, size=n).astype(np.float64)
    a = rng.negative_binomial(eps + c, 1 - p)
    b = rng.negative_binomial(eps + c, p)
    return a.astype(np.float64), b.astype(np.float64), c


def split_betabinom(total: np.ndarray, a_shape: np.ndarray, b_shape: np.ndarray, rng):
    """Draws X1 ~ BetaBinomial(total; a_shape, b_shape) elementwise; X2 = total - X1."""
    # scipy's betabinom.rvs supports array params; np's generator doesn't have
    # beta-binomial built in, so go via scipy with our rng as the random_state.
    x1 = stats.betabinom.rvs(total.astype(np.int64), a_shape, b_shape, random_state=rng)
    return x1.astype(np.float64), (total - x1).astype(np.float64)


def ks_report(name: str, x: np.ndarray, y: np.ndarray) -> None:
    stat, pval = stats.ks_2samp(x, y)
    verdict = "MATCH" if pval > 0.01 and stat < 0.01 else "MISMATCH"
    print(f"{name:55s} KS stat={stat:.5f}  p={pval:.4f}  [{verdict}]")


def main() -> None:
    for p, q, eps1, eps2 in [
        (0.3, 0.5, 1.0, 1.0),
        (0.3, 0.5, 0.5, 2.0),
        (0.7, 0.2, 1.0, 3.0),
        (0.5, 0.8, 2.0, 1.0),
    ]:
        eps_sum = eps1 + eps2
        a, b, c = sample_augmented_abc(p, q, eps_sum, N, RNG)

        c1, c2 = split_betabinom(c, np.full_like(c, eps1), np.full_like(c, eps2), RNG)
        a1, a2 = split_betabinom(a, eps1 + c1, eps2 + c2, RNG)
        b1, b2 = split_betabinom(b, eps1 + c1, eps2 + c2, RNG)

        assert np.allclose(a1 + a2, a) and np.allclose(b1 + b2, b) and np.allclose(c1 + c2, c)

        y1_split = RNG.beta(eps1 + c1 + a1, eps1 + c1 + b1)
        y1_direct = TNBBetaUnivariate(
            torch.tensor(p), torch.tensor(q), torch.tensor(eps1)
        ).rsample((N,)).numpy()

        y2_split = RNG.beta(eps2 + c2 + a2, eps2 + c2 + b2)
        y2_direct = TNBBetaUnivariate(
            torch.tensor(p), torch.tensor(q), torch.tensor(eps2)
        ).rsample((N,)).numpy()

        print(f"-- p={p} q={q} eps1={eps1} eps2={eps2} (split from sum={eps_sum}) --")
        ks_report("  Y1 (split piece) vs. TNBbeta(p,q,eps1) closed-form", y1_split, y1_direct)
        ks_report("  Y2 (split piece) vs. TNBbeta(p,q,eps2) closed-form", y2_split, y2_direct)


if __name__ == "__main__":
    main()
