"""Check the proposed joint (A,B,C) forward-noising construction.

    C_t | C_0 ~ Leisen-Mena-Palma Mancilla-Rossini (2019, arXiv:1812.07271)
                closed-form kernel targeting NB(eps, q_target)
    A_t | C_t ~ NB(eps + C_t, 1-p)      (fresh, Theorem 4.1 recipe)
    B_t | C_t ~ NB(eps + C_t, p)
    Y_t       ~ Beta(eps+C_t+A_t, eps+C_t+B_t)

Their kernel (one-shot, closed-form, no path simulation needed):
    Theta_t = (1 - q_target) / (exp(c*t) - q_target)
    Y_bin ~ Binomial(C_0, Theta_t)
    Z     ~ NB(eps + Y_bin, q_target * (1 - Theta_t))
    C_t = Y_bin + Z

Checks:
  1. t=0 recovers the data's own TNBbeta(p, q_data, eps) exactly (Theta_0=1 => no-op).
  2. t->large converges to TNBbeta(p, q_target, eps) REGARDLESS of q_data (the
     "forgets its starting point" property that makes this usable as a forward
     noising process in the first place).
  3. Sanity: the implied concentration moves monotonically between those two
     endpoints as t increases (spot-checked via the sample variance of Y_t).
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/Users/vedantpathak/Developer/projects/packages/tnbbeta_vae/src")

import torch  # noqa: E402

from tnbbeta_vae.distributions.tnbbeta_univariate import TNBBetaUnivariate  # noqa: E402

RNG = np.random.default_rng(2)
N = 200_000


def sample_c0(p, q_data, eps, n, rng):
    return rng.negative_binomial(eps, 1 - q_data, size=n).astype(np.float64)


def leisen_step(c0: np.ndarray, t: float, c: float, eps: float, q_target: float, rng):
    """One-shot closed-form draw of C_t | C_0 (their eq. 4/10's stochastic form).

    Parameterization note: Leisen et al.'s NB(r, q_L) uses q_L^x(1-q_L)^r --
    the same convention TNBbeta's own "C ~ NB(eps, 1-q)" uses (verified against
    TNBBetaUnivariate directly in tnbbeta_epsilon_additivity_check.py), so
    their q_L *is* our q_tnb directly, no complement. numpy's
    negative_binomial(n, p) wants p = P(success) raised to the size power,
    i.e. p = 1 - q_L -- that complement belongs in the numpy call, not in the
    quantity passed as "q_L" itself.
    """
    theta_t = (1 - q_target) / (np.exp(c * t) - q_target)
    y_bin = rng.binomial(c0.astype(np.int64), theta_t)
    z = rng.negative_binomial(eps + y_bin, 1 - q_target * (1 - theta_t))
    return (y_bin + z).astype(np.float64)


def sample_y_given_c(c_t, p, eps, n, rng):
    a = rng.negative_binomial(eps + c_t, 1 - p)
    b = rng.negative_binomial(eps + c_t, p)
    return rng.beta(eps + c_t + a, eps + c_t + b)


def ks_report(name, x, y):
    stat, pval = stats.ks_2samp(x, y)
    verdict = "MATCH" if pval > 0.01 and stat < 0.01 else "MISMATCH"
    print(f"{name:60s} KS stat={stat:.5f}  p={pval:.4f}  [{verdict}]")


def main() -> None:
    p, eps = 0.65, 1.5
    q_data = 0.7
    q_target = 0.05
    speed = 1.0  # their "c" rate parameter

    c0 = sample_c0(p, q_data, eps, N, RNG)

    print(f"Setup: p={p} eps={eps} q_data={q_data} -> q_target={q_target}, speed={speed}")
    print()

    print("=== t=0: should recover TNBbeta(p, q_data, eps) exactly ===")
    c_t0 = leisen_step(c0, 0.0, speed, eps, q_target, RNG)
    y_t0 = sample_y_given_c(c_t0, p, eps, N, RNG)
    y_data_closed = TNBBetaUnivariate(
        torch.tensor(p), torch.tensor(q_data), torch.tensor(eps)
    ).rsample((N,)).numpy()
    ks_report("Y_0 vs. closed-form TNBbeta(p,q_data,eps)", y_t0, y_data_closed)

    print()
    print("=== t -> large: should converge to TNBbeta(p, q_target, eps), any q_data ===")
    y_target_closed = TNBBetaUnivariate(
        torch.tensor(p), torch.tensor(q_target), torch.tensor(eps)
    ).rsample((N,)).numpy()
    for t in [0.5, 1.0, 2.0, 4.0, 8.0]:
        c_t = leisen_step(c0, t, speed, eps, q_target, RNG)
        y_t = sample_y_given_c(c_t, p, eps, N, RNG)
        var = y_t.var()
        ks_report(f"t={t:4.1f}  Y_t vs. TNBbeta(p,q_target,eps)  [var={var:.5f}]", y_t, y_target_closed)

    print()
    print("=== Robustness: does convergence hold starting from a DIFFERENT q_data? ===")
    for q_data_alt in [0.0, 0.3, 0.95]:
        c0_alt = sample_c0(p, q_data_alt, eps, N, RNG)
        c_t = leisen_step(c0_alt, 8.0, speed, eps, q_target, RNG)
        y_t = sample_y_given_c(c_t, p, eps, N, RNG)
        ks_report(f"q_data={q_data_alt:4.2f}  Y_8 vs. TNBbeta(p,q_target,eps)", y_t, y_target_closed)


if __name__ == "__main__":
    main()
