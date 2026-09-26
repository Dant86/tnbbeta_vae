"""Compares vMF's and TNBBeta's ``rsample`` wall-clock time, GPU or CPU.

Usage:
    uv run python -m apps.benchmark.sampling_speed \\
        [--dims 2 5 10 20 40] [--batch-sizes 64 1024 8192] \\
        [--concentrations weak moderate strong extreme] \\
        [--repeats 30] [--warmup 5] [--device cuda] [--output PATH]

Motivation: vMF's ``rsample`` (Wood 1994's rejection sampler, this project's port
of the reference S-VAE implementation) draws ``k=20`` candidate proposals per
round and repeats an unbounded number of rounds until every row in the batch has
been accepted at least once -- a data-dependent Python ``while`` loop that reads a
boolean mask back from the device every iteration (``bool_mask.sum() != 0``),
forcing a CPU-GPU synchronization point each round. ``TNBBetaSpherical.rsample``
is a single deterministic transform of one ``Beta(epsilon, epsilon)`` draw (see
its own docstring) with no rejection step and no data-dependent control flow at
all. This predicts vMF being slower in absolute terms, and -- more specifically
-- vMF's cost scaling with concentration (rejection rate depends on kappa and
dimension) while TNBBeta's stays flat regardless of concentration.

Each timed call is bracketed by ``torch.cuda.synchronize()`` on GPU (with
untimed warm-up calls first), so async kernel launches don't make either
family's cost look artificially free.

Writes a JSON list of per-(dim, batch size, concentration level) records, each
with both families' wall-clock stats (mean/std/min/max milliseconds over
``--repeats`` timed calls) and the ratio between them.

Caveat: this uses one shared concentration per call. A real training batch has
a different kappa/epsilon per example, and since vMF's while loop only
terminates once every row in the batch has been accepted, a single
high-concentration "straggler" example can stall the whole batch in a way this
benchmark -- deliberately controlled, one concentration at a time -- does not
capture. A plausible follow-up, not measured here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Any

import torch

from tnbbeta_vae.distributions import TNBBetaSpherical, VonMisesFisher
from tnbbeta_vae.paths import runs_dir

# TNBBeta's epsilon isn't on the same numerical scale as vMF's kappa (no exact
# translation exists, since sampling cost for TNBBeta doesn't depend on
# concentration at all by construction); these are independently reasonable
# "weak/moderate/strong/extreme" values for each family, not claimed equivalents.
_CONCENTRATIONS: dict[str, dict[str, float]] = {
    "weak": {"kappa": 1.0, "epsilon": 1.0},
    "moderate": {"kappa": 10.0, "epsilon": 10.0},
    "strong": {"kappa": 100.0, "epsilon": 100.0},
    "extreme": {"kappa": 1000.0, "epsilon": 1000.0},
}
# p away from 0.5 (a cap, not the uniform special case) but away from training's
# 1e-6 clamp too -- this is a sampling benchmark, not a trained posterior.
_P = 1.0 - 1e-3
_Q = 0.0


def main(argv: list[str] | None = None) -> None:
    """Runs the sweep and writes the results.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 5, 10, 20, 40])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 1024, 8192])
    parser.add_argument(
        "--concentrations",
        nargs="+",
        default=list(_CONCENTRATIONS),
        choices=list(_CONCENTRATIONS),
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)

    results = []
    for dim in args.dims:
        for batch_size in args.batch_sizes:
            for level in args.concentrations:
                record = _benchmark_one(
                    dim, batch_size, level, args.repeats, args.warmup, device
                )
                results.append(record)
                print(json.dumps(record))

    output = args.output or str(runs_dir() / "sampling_speed_benchmark.json")
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output}")


def _benchmark_one(
    dim: int,
    batch_size: int,
    level: str,
    repeats: int,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    """Times one (dim, batch size, concentration level) cell for both families."""
    params = _CONCENTRATIONS[level]
    vmf_stats = _time_rsample(
        _vmf(dim, batch_size, params["kappa"], device), repeats, warmup, device
    )
    tnbbeta_stats = _time_rsample(
        _tnbbeta(dim, batch_size, params["epsilon"], device), repeats, warmup, device
    )
    return {
        "dim": dim,
        "batch_size": batch_size,
        "concentration": level,
        "vmf_kappa": params["kappa"],
        "tnbbeta_epsilon": params["epsilon"],
        "vmf_ms": vmf_stats,
        "tnbbeta_ms": tnbbeta_stats,
        "vmf_over_tnbbeta": vmf_stats["mean"] / tnbbeta_stats["mean"],
    }


def _vmf(
    dim: int, batch_size: int, kappa: float, device: torch.device
) -> VonMisesFisher:
    """Builds a batch of vMF posteriors with a shared concentration."""
    loc = torch.randn(batch_size, dim, device=device)
    loc = loc / loc.norm(dim=-1, keepdim=True)
    scale = torch.full((batch_size, 1), kappa, device=device)
    return VonMisesFisher(loc, scale)


def _tnbbeta(
    dim: int, batch_size: int, epsilon: float, device: torch.device
) -> TNBBetaSpherical:
    """Builds a batch of TNBBetaSpherical posteriors with a shared epsilon."""
    mean_direction = torch.randn(batch_size, dim, device=device)
    mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True)
    p = torch.full((batch_size,), _P, device=device)
    q = torch.full((batch_size,), _Q, device=device)
    eps = torch.full((batch_size,), epsilon, device=device)
    return TNBBetaSpherical(mean_direction, p, q, eps)


def _time_rsample(
    dist: VonMisesFisher | TNBBetaSpherical,
    repeats: int,
    warmup: int,
    device: torch.device,
) -> dict[str, float]:
    """Times ``repeats`` calls to ``dist.rsample()``, after ``warmup`` untimed ones."""
    for _ in range(warmup):
        dist.rsample()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        dist.rsample()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    return {
        "mean": statistics.mean(times_ms),
        "std": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min": min(times_ms),
        "max": max(times_ms),
    }


if __name__ == "__main__":
    main(sys.argv[1:])
