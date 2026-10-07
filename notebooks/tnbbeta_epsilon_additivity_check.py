"""Scratch check: is TNBbeta "additive in epsilon" via its NB augmentation?

Theorem 4.1 of Lederman & Schein (2026) builds Y ~ TNBbeta(p, q, eps) from:
    C ~ NB(eps, 1-q)
    A | C=c ~ NB(eps+c, 1-p)
    B | C=c ~ NB(eps+c, p)
    Y | A=a, B=b, C=c ~ Beta(eps+c+a, eps+c+b)

Hypothesis under test: drawing two INDEPENDENT augmented triples at the same
(p, q) but different eps1, eps2, summing them componentwise (C=C1+C2 etc.),
and redrawing Y from Beta(eps1+eps2+C+A, eps1+eps2+C+B) reproduces
TNBbeta(p, q, eps1+eps2) exactly -- i.e. epsilon behaves like an additive
"pseudo-count"/time parameter under a well-defined noise-injection operation,
the discrete/compound-Poisson analogue of how two independent Gaussian noise
additions compose into one.

This is the load-bearing claim for "is there a closed-form forward noising
chain for TNBbeta the way the OU process gives one for Gaussian diffusion".
If it fails, the naive componentwise sum is not the right composition
operation and we need the Beta-Binomial split correction instead.

Also runs a basic sanity check that the augmented sampler reproduces
TNBBetaUnivariate's own (independently-derived, paper-sourced) rsample,
so a failure of the main test isn't just a bug in the augmentation code.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/Users/vedantpathak/Developer/projects/packages/tnbbeta_vae/src")

import torch  # noqa: E402

from tnbbeta_vae.distributions.tnbbeta_univariate import TNBBetaUnivariate  # noqa: E402

RNG = np.random.default_rng(0)
N = 200_000


def sample_augmented(p: float, q: float, eps: float, n: int, rng: np.random.Generator):
    """Draws n samples via the Theorem 4.1 construction. Returns (y, a, b, c)."""
    c = rng.negative_binomial(eps, 1 - q, size=n).astype(np.float64)
    a = rng.negative_binomial(eps + c, 1 - p)
    b = rng.negative_binomial(eps + c, p)
    y = rng.beta(eps + c + a, eps + c + b)
    return y, a, b, c


def ks_report(name: str, x: np.ndarray, y: np.ndarray) -> None:
    stat, pval = stats.ks_2samp(x, y)
    verdict = "MATCH" if pval > 0.01 and stat < 0.01 else "MISMATCH"
    print(f"{name:45s} KS stat={stat:.5f}  p={pval:.4f}  [{verdict}]")


def main() -> None:
    print("=== Sanity check: augmented sampler vs. TNBBetaUnivariate.rsample ===")
    for p, q, eps in [(0.3, 0.5, 1.0), (0.7, 0.2, 0.3), (0.5, 0.8, 2.5), (0.6, 0.0, 1.0)]:
        y_aug, _, _, _ = sample_augmented(p, q, eps, N, RNG)
        y_ref = TNBBetaUnivariate(torch.tensor(p), torch.tensor(q), torch.tensor(eps)).rsample(
            (N,)
        ).numpy()
        ks_report(f"p={p} q={q} eps={eps}", y_aug, y_ref)

    print()
    print("=== Main test: epsilon-additivity under componentwise (A,B,C) summation ===")
    for p, q, eps1, eps2 in [
        (0.3, 0.5, 1.0, 1.0),
        (0.3, 0.5, 0.5, 2.0),
        (0.7, 0.2, 1.0, 3.0),
        (0.5, 0.8, 1.0, 1.0),
        (0.5, 0.0, 1.0, 1.0),  # q=0 special case
    ]:
        eps_sum = eps1 + eps2
        _, a1, b1, c1 = sample_augmented(p, q, eps1, N, RNG)
        _, a2, b2, c2 = sample_augmented(p, q, eps2, N, RNG)
        a, b, c = a1 + a2, b1 + b2, c1 + c2
        y_combined = RNG.beta(eps_sum + c + a, eps_sum + c + b)

        y_direct, _, _, _ = sample_augmented(p, q, eps_sum, N, RNG)
        y_closed = TNBBetaUnivariate(
            torch.tensor(p), torch.tensor(q), torch.tensor(eps_sum)
        ).rsample((N,)).numpy()

        print(f"-- p={p} q={q} eps1={eps1} eps2={eps2} (sum={eps_sum}) --")
        ks_report("  combined vs. direct-augmented(eps_sum)", y_combined, y_direct)
        ks_report("  combined vs. closed-form TNBbeta(eps_sum)", y_combined, y_closed)


if __name__ == "__main__":
    main()
