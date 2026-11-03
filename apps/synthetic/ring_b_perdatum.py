"""Per-datum optimal posterior against a *learned* decoder (ring task, condition B).

For a stage-3 checkpoint with a learned decoder ``f`` and sigma, the model's own
posterior is ``p_model(z | x) ∝ N(x; f(z), sigma^2 I)``. This optimises, per test x,
each family's posterior against that decoder (Adam on the raw head vector, the fixed
decoder, no encoder), then reports ``log p_model(x) - ELBO``, i.e.
``KL(q || p_model(z | x))``: how well each family *can* match the decoder's posterior,
separating family capacity from the amortised encoder's optimisation.
``log p_model(x)`` is a uniform-prior Monte Carlo estimate that does not use ``q``.

Usage::

    uv run python -m apps.synthetic.ring_b_perdatum \
        --ckpt ring_recovery/l23_d3/stage3/ckpt/d3_learned_vmf_s0.pt --variant vmf \
        --out out.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.distributions import kl_divergence

from apps.synthetic.ring_condition_b_diagnostics import (
    DIM,
    load_model,
    log_marginal,
)
from apps.synthetic.ring_recovery import PRIMARY_SIGMA
from tnbbeta_vae.data.ring_posterior import ring_task, sphere_log_area
from tnbbeta_vae.distributions import HypersphericalUniform
from tnbbeta_vae.models import MlpVAE
from tnbbeta_vae.models.heads import head_size, posterior_from_raw
from tnbbeta_vae.models.losses.sampling import sample_with_log_prob

_FAMILIES = ("vmf", "power_spherical", "tnbbeta")
_EPS_FLOOR = 0.3


def main(argv: list[str] | None = None) -> None:
    """Optimises each family per datum against one checkpoint's decoder.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-x", type=int, default=300)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--refine-encoder",
        action="store_true",
        help="Restart 0 of the checkpoint's own family starts at the encoder's output "
        "(per-datum refinement of the amortised posterior); others stay random.",
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    task = ring_task(DIM, PRIMARY_SIGMA)
    test_x, _ = task.sample(2000, torch.Generator().manual_seed(424242))
    x = test_x[: args.num_x]
    model = load_model(args.ckpt, "learned", args.variant, task)
    for p in model.parameters():
        p.requires_grad_(False)
    reference = log_marginal(model, x, 400_000)
    result: dict[str, Any] = {
        "ckpt": str(args.ckpt),
        "log_marginal_model": float(reference.mean()),
    }
    with torch.no_grad():
        posterior, _ = model.posterior_and_prior(x)
        z, log_q = sample_with_log_prob(posterior, torch.Size([1000]))
        assert log_q is not None
        amortised = (model.log_likelihood(x, z) - log_q - sphere_log_area(DIM)).mean(0)
    result["amortised_gap"] = float((reference - amortised).mean())
    for family in _FAMILIES:
        torch.manual_seed(args.seed)
        raw = _optimise(model, family, x, args)
        with torch.no_grad():
            elbo = _elbo(model, family, x, raw, 1000)
        extra = {}
        keep = torch.ones(x.shape[0], dtype=torch.bool)
        if family == "tnbbeta":
            eps = posterior_from_raw("tnbbeta", raw, DIM).epsilon  # pyright: ignore[reportAttributeAccessIssue]
            keep = eps >= _EPS_FLOOR
            extra = {
                "eps_median": float(eps.median()),
                "frac_eps_below_floor": float((eps < _EPS_FLOOR).float().mean()),
            }
        gap = reference - elbo
        result[family] = {
            "gap_mean": float(gap.mean()),
            "gap_median": float(gap.median()),
            "gap_mean_excluding_eps_floor": float(gap[keep].mean()),
            **extra,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    print(json.dumps(result))


def _own_family(variant: str) -> str:
    return "tnbbeta" if variant.startswith("tnbbeta") else variant


def _elbo(model: MlpVAE, family: str, x: Tensor, raw: Tensor, num: int) -> Tensor:
    posterior = posterior_from_raw(cast("Any", family), raw, DIM)
    if family == "tnbbeta":
        z, log_q = sample_with_log_prob(posterior, torch.Size([num]))
        assert log_q is not None
        reconstruction = model.log_likelihood(x, z).mean(0)
        kl = (log_q + sphere_log_area(DIM)).mean(0)
    else:
        z = posterior.rsample(torch.Size([num]))
        reconstruction = model.log_likelihood(x, z).mean(0)
        kl = kl_divergence(posterior, HypersphericalUniform(DIM - 1))
    return reconstruction - kl


def _optimise(
    model: MlpVAE, family: str, x: Tensor, args: argparse.Namespace
) -> Tensor:
    num = x.shape[0]
    x_rep = x.repeat_interleave(args.restarts, dim=0)
    raw = torch.randn(num * args.restarts, head_size(cast("Any", family), DIM))
    if args.refine_encoder and family == _own_family(args.variant):
        with torch.no_grad():
            start = model.posterior_head(model.encoder(x))
        raw = raw.reshape(num, args.restarts, -1)
        raw[:, 0] = start
        raw = raw.reshape(num * args.restarts, -1)
    raw.requires_grad_(True)
    optimiser = torch.optim.Adam([raw], lr=args.lr)
    for _ in range(args.steps):
        optimiser.zero_grad()
        (-_elbo(model, family, x_rep, raw, args.num_samples)).sum().backward()
        optimiser.step()
    with torch.no_grad():
        elbo = _elbo(model, family, x_rep, raw, 64).reshape(num, args.restarts)
        if family == "tnbbeta":
            eps = posterior_from_raw("tnbbeta", raw, DIM).epsilon  # pyright: ignore[reportAttributeAccessIssue]
            elbo = elbo.masked_fill(
                (eps < _EPS_FLOOR).reshape(num, args.restarts), -math.inf
            )
        best = elbo.argmax(dim=1)
    return raw.detach().reshape(num, args.restarts, -1)[torch.arange(num), best]


if __name__ == "__main__":
    main()
