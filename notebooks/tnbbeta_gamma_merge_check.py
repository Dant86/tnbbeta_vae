"""Check: does the eps-merge semigroup survive in a fully continuous,
differentiable form via the standard Beta(a,b) = Gamma(a)/(Gamma(a)+Gamma(b))
representation -- bypassing the discrete NB/Beta-Binomial augmentation
entirely?

Two things to confirm:
  1. Correctness: merging via summed Gamma draws reproduces TNBbeta(p,q,eps_sum)
     (same check as before, but through the continuous route).
  2. Differentiability: gradients actually flow from a loss on the merged
     sample back to eps1, eps2, p, q (and are finite, non-zero).
"""

from __future__ import annotations

import sys

import torch
from scipy import stats

sys.path.insert(0, "/Users/vedantpathak/Developer/projects/packages/tnbbeta_vae/src")

from tnbbeta_vae.distributions.tnbbeta_univariate import TNBBetaUnivariate  # noqa: E402

torch.manual_seed(0)
N = 200_000


def transform_pq(z: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """The fixed, eps-independent T_{p,q} from TNBBetaUnivariate.rsample."""
    s = 2 * z - 1
    u = (1 + s * torch.sqrt((1 - q) / (1 - q * s**2))) / 2
    return p * u / ((1 - p) * (1 - u) + p * u)


def gamma_ratio_beta(eps: torch.Tensor, n: int) -> torch.Tensor:
    """z ~ Beta(eps, eps) via two independent Gamma(eps, 1) draws. Differentiable in eps."""
    x1 = torch.distributions.Gamma(eps.expand(n), torch.ones(n)).rsample()
    x2 = torch.distributions.Gamma(eps.expand(n), torch.ones(n)).rsample()
    return x1 / (x1 + x2)


def main() -> None:
    print("=== Correctness: Gamma-ratio merge vs. closed-form TNBbeta(eps_sum) ===")
    for p_val, q_val, eps1_val, eps2_val in [
        (0.3, 0.5, 1.0, 1.0),
        (0.7, 0.2, 1.0, 3.0),
        (0.5, 0.8, 0.5, 2.0),
    ]:
        p, q = torch.tensor(p_val), torch.tensor(q_val)
        eps1, eps2 = torch.tensor(eps1_val), torch.tensor(eps2_val)

        z1 = gamma_ratio_beta(eps1, N)
        z2 = gamma_ratio_beta(eps2, N)
        # Merge at the z-level: independent Gamma pairs summed, same identity as before.
        x1a, x1b = torch.distributions.Gamma(eps1.expand(N), torch.ones(N)).rsample(), \
            torch.distributions.Gamma(eps1.expand(N), torch.ones(N)).rsample()
        x2a, x2b = torch.distributions.Gamma(eps2.expand(N), torch.ones(N)).rsample(), \
            torch.distributions.Gamma(eps2.expand(N), torch.ones(N)).rsample()
        z_merged = (x1a + x2a) / (x1a + x2a + x1b + x2b)

        y_merged = transform_pq(z_merged, p, q).detach().numpy()
        y_closed = TNBBetaUnivariate(p, q, eps1 + eps2).rsample((N,)).detach().numpy()

        stat, pval = stats.ks_2samp(y_merged, y_closed)
        verdict = "MATCH" if pval > 0.01 and stat < 0.01 else "MISMATCH"
        print(
            f"p={p_val} q={q_val} eps1={eps1_val} eps2={eps2_val} (sum={eps1_val + eps2_val}): "
            f"KS stat={stat:.5f} p={pval:.4f} [{verdict}]"
        )

    print()
    print("=== Differentiability: gradients from a loss on the merged sample ===")
    p = torch.tensor(0.6, requires_grad=True)
    q = torch.tensor(0.3, requires_grad=True)
    eps1 = torch.tensor(1.2, requires_grad=True)
    eps2 = torch.tensor(2.5, requires_grad=True)

    x1a = torch.distributions.Gamma(eps1.expand(N), torch.ones(N)).rsample()
    x1b = torch.distributions.Gamma(eps1.expand(N), torch.ones(N)).rsample()
    x2a = torch.distributions.Gamma(eps2.expand(N), torch.ones(N)).rsample()
    x2b = torch.distributions.Gamma(eps2.expand(N), torch.ones(N)).rsample()
    z_merged = (x1a + x2a) / (x1a + x2a + x1b + x2b)
    y_merged = transform_pq(z_merged, p, q)

    loss = (y_merged - 0.5).pow(2).mean()
    loss.backward()

    for name, t in [("p", p), ("q", q), ("eps1", eps1), ("eps2", eps2)]:
        g = t.grad
        ok = g is not None and torch.isfinite(g) and g.abs().item() > 0
        print(f"  d(loss)/d({name}) = {g.item(): .6f}  [{'OK' if ok else 'FAILED'}]")


if __name__ == "__main__":
    main()
