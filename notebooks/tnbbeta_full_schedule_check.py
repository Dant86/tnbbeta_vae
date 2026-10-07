"""Check (1) the exact p-decoupling identity, and (2) the full combined
p + q + eps forward-noising recipe converging to true Uniform(0,1).

Identity:  TNBbeta(p,q,eps) =_d sigmoid(logit(p) + logit(TNBbeta(0.5,q,eps)))

Full one-shot recipe for Y_t | (p_data,q_data,eps_data), queried at time t
(each t independent -- no path simulation, matches how DSM only needs
marginals, not trajectories):

  eps_t      = 1 + (eps_data - 1) * exp(-t)              # -> 1
  C_data     ~ NB(eps_data, 1-q_data)
  C_0'       = BetaBinomial-split(C_data; eps_t, eps_data-eps_t)[0]  # one-shot eps resize, same q_data
  C_t        = Leisen-step(C_0', kernel-time=t, size=eps_t, target q_target)  # -> NB(eps_t, q_target)
  A_t        ~ NB(eps_t + C_t, 0.5),  B_t ~ NB(eps_t + C_t, 0.5)     # p pinned at 0.5 -> this is "U_t"
  U_t        ~ Beta(eps_t+C_t+A_t, eps_t+C_t+B_t)
  psi_t      = logit(p_data) * exp(-t)                    # -> 0
  Y_t        = sigmoid(psi_t + logit(U_t))

At t=0: eps_t=eps_data, split is a no-op, Leisen step is identity, psi_0=logit(p_data)
        => should recover TNBbeta(p_data,q_data,eps_data) exactly.
At t->large: eps_t->1, C_t->NB(1,q_target), psi_t->0
        => should converge to TNBbeta(0.5,q_target,1) ~= Uniform(0,1), any data params.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "/Users/vedantpathak/Developer/projects/packages/tnbbeta_vae/src")

import torch  # noqa: E402

from tnbbeta_vae.distributions.tnbbeta_univariate import TNBBetaUnivariate  # noqa: E402

RNG = np.random.default_rng(3)
N = 200_000


def logit(x):
    return np.log(x) - np.log1p(-x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ks_report(name, x, y):
    stat, pval = stats.ks_2samp(x, y)
    verdict = "MATCH" if pval > 0.01 and stat < 0.01 else "MISMATCH"
    print(f"{name:65s} KS stat={stat:.5f}  p={pval:.4f}  [{verdict}]")


def split_betabinom(total, a_shape, b_shape, rng):
    x1 = stats.betabinom.rvs(total.astype(np.int64), a_shape, b_shape, random_state=rng)
    return x1.astype(np.float64), (total - x1).astype(np.float64)


def resize_eps(c_data, eps_data, eps_t, q_data, n, rng):
    """One-shot resize of the data's C to a new epsilon, same q_data throughout.

    eps_t < eps_data: split off the eps_t-sized piece (Beta-Binomial thinning).
    eps_t > eps_data: merge in an independent fresh NB(eps_t-eps_data, 1-q_data) piece
    (the fresh piece must share q_data for the merge identity to hold).
    """
    if abs(eps_t - eps_data) < 1e-9:
        return c_data
    if eps_t < eps_data:
        c0_prime, _ = split_betabinom(
            c_data, np.full(n, eps_t), np.full(n, eps_data - eps_t), rng
        )
        return c0_prime
    extra = rng.negative_binomial(eps_t - eps_data, 1 - q_data, size=n).astype(np.float64)
    return c_data + extra


def leisen_step(c0, t, speed, eps, q_target, rng):
    theta_t = (1 - q_target) / (np.exp(speed * t) - q_target)
    y_bin = rng.binomial(c0.astype(np.int64), theta_t)
    z = rng.negative_binomial(eps + y_bin, 1 - q_target * (1 - theta_t))
    return (y_bin + z).astype(np.float64)


def part1_identity_check():
    print("=== Part 1: logit-decoupling identity ===")
    for p, q, eps in [(0.2, 0.5, 1.5), (0.9, 0.1, 0.8), (0.4, 0.0, 2.0)]:
        u = TNBBetaUnivariate(torch.tensor(0.5), torch.tensor(q), torch.tensor(eps)).rsample(
            (N,)
        ).numpy()
        y_decomposed = sigmoid(logit(p) + logit(u))
        y_closed = TNBBetaUnivariate(
            torch.tensor(p), torch.tensor(q), torch.tensor(eps)
        ).rsample((N,)).numpy()
        ks_report(f"p={p} q={q} eps={eps}: sigmoid(logit(p)+logit(U)) vs TNBbeta(p,q,eps)",
                   y_decomposed, y_closed)
    print()


def draw_y_t(p_data, q_data, eps_data, t, q_target, speed, n, rng):
    eps_t = 1 + (eps_data - 1) * np.exp(-t)
    c_data = rng.negative_binomial(eps_data, 1 - q_data, size=n).astype(np.float64)
    c0_prime = resize_eps(c_data, eps_data, eps_t, q_data, n, rng)
    c_t = leisen_step(c0_prime, t, speed, eps_t, q_target, rng)
    a_t = rng.negative_binomial(eps_t + c_t, 0.5)
    b_t = rng.negative_binomial(eps_t + c_t, 0.5)
    u_t = rng.beta(eps_t + c_t + a_t, eps_t + c_t + b_t)
    psi_t = logit(np.array(p_data)) * np.exp(-t)
    return sigmoid(psi_t + logit(u_t))


def part2_full_schedule_check():
    print("=== Part 2: full p+q+eps schedule ===")
    p_data, q_data, eps_data = 0.85, 0.6, 2.5
    q_target, speed = 0.05, 1.0

    print(f"data: p={p_data} q={q_data} eps={eps_data}  ->  target: p=0.5 q={q_target} eps=1")
    print()

    print("-- t=0: should recover the data's own TNBbeta exactly --")
    y0 = draw_y_t(p_data, q_data, eps_data, 0.0, q_target, speed, N, RNG)
    y0_closed = TNBBetaUnivariate(
        torch.tensor(p_data), torch.tensor(q_data), torch.tensor(eps_data)
    ).rsample((N,)).numpy()
    ks_report("Y_0 vs TNBbeta(p_data,q_data,eps_data)", y0, y0_closed)

    print()
    print("-- t -> large: should converge to TNBbeta(0.5,q_target,1) ~ near-Uniform(0,1) --")
    y_ref_closed = TNBBetaUnivariate(
        torch.tensor(0.5), torch.tensor(q_target), torch.tensor(1.0)
    ).rsample((N,)).numpy()
    uniform_ref = RNG.uniform(0, 1, N)
    for t in [1.0, 3.0, 6.0, 10.0]:
        y_t = draw_y_t(p_data, q_data, eps_data, t, q_target, speed, N, RNG)
        ks_report(f"t={t:5.1f}  Y_t vs TNBbeta(0.5,q_target,1)", y_t, y_ref_closed)
    y_final = draw_y_t(p_data, q_data, eps_data, 10.0, q_target, speed, N, RNG)
    ks_report("t=10.0  Y_t vs actual Uniform(0,1)  (checks q_target~0 approx is OK)",
               y_final, uniform_ref)

    print()
    print("-- robustness: different data params, same large-t target --")
    for p_alt, q_alt, eps_alt in [(0.1, 0.0, 0.5), (0.5, 0.9, 5.0)]:
        y_t = draw_y_t(p_alt, q_alt, eps_alt, 10.0, q_target, speed, N, RNG)
        ks_report(f"p={p_alt} q={q_alt} eps={eps_alt}  Y_10 vs TNBbeta(0.5,q_target,1)",
                   y_t, y_ref_closed)


if __name__ == "__main__":
    part1_identity_check()
    part2_full_schedule_check()
