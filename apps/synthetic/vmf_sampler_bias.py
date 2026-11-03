"""Grid test of the library vMF rejection sampler against exact moments.

Usage:
    uv run python -m apps.synthetic.vmf_sampler_bias run --dtype float32 --exact-envelope 0
    uv run python -m apps.synthetic.vmf_sampler_bias summarize

For each ambient dimension ``m`` and concentration ``kappa`` it draws ``N`` samples of
``VonMisesFisher`` (random mean direction, the whole ``rsample`` path) and compares
``w = <z, loc>`` with the exact law ``f(w) ~ exp(kappa w) (1 - w^2)^((m - 3) / 2)``:
mean ``A_m(kappa) = I_{m/2}(kappa) / I_{m/2-1}(kappa)``, second moment
``1 - (m - 1) A_m(kappa) / kappa``, and the Kolmogorov-Smirnov distance to the exact CDF.
Results go to ``$TNBBETA_CHECKPOINT_DIR/vmf_sampler_bias/<dtype>_exact<flag>.json``.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import ive
import torch

from tnbbeta_vae.distributions import VonMisesFisher
from tnbbeta_vae.paths import checkpoint_dir

DIMS = (3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 20, 25, 33, 50, 65)
KAPPAS = (
    1,
    2,
    4,
    8,
    10,
    11,
    12,
    14,
    16,
    20,
    25,
    30,
    40,
    50,
    60,
    80,
    100,
    150,
    200,
    300,
    500,
)
SAMPLES = 400_000
# Flag a cell when the mean or second moment is off by more than Z_FLAG standard errors
# (the smallest detectable mean bias is about 0.004 at N = 400000), or the KS distance
# exceeds KS_FLAG / sqrt(N) (a 0.1% test).
Z_FLAG = 4.0
KS_FLAG = 1.95


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``run`` or ``summarize``."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--dtype", choices=["float32", "float64"], required=True)
    run.add_argument("--exact-envelope", type=int, choices=[0, 1], required=True)
    run.add_argument("--samples", type=int, default=SAMPLES)
    run.add_argument("--seed", type=int, default=0)
    sub.add_parser("summarize")
    args = parser.parse_args(argv)
    if args.command == "run":
        run_grid(args.dtype, bool(args.exact_envelope), args.samples, args.seed)
    else:
        print(summarize())


def out_dir() -> Path:
    """Returns the output directory (not created)."""
    return checkpoint_dir() / "vmf_sampler_bias"


def exact_moments(m: int, kappa: float) -> tuple[float, float]:
    """Returns ``(E[w], E[w^2])`` of the vMF marginal ``w = <z, loc>``."""
    ratio = float(ive(m / 2, kappa) / ive(m / 2 - 1, kappa))
    return ratio, 1.0 - (m - 1) * ratio / kappa


def ks_distance(w: np.ndarray, m: int, kappa: float) -> float:
    """Kolmogorov-Smirnov distance of ``w`` to the exact marginal CDF."""
    grid = np.linspace(-1.0, 1.0, 400_001)
    log_density = kappa * (grid - 1) + (m - 3) / 2 * np.log(
        np.maximum(1 - grid**2, 1e-300)
    )
    density = np.exp(log_density)
    cdf = cumulative_trapezoid(density, grid, initial=0.0)
    cdf /= cdf[-1]
    w = np.sort(w)
    exact = np.interp(w, grid, cdf)
    n = len(w)
    upper = np.arange(1, n + 1) / n - exact
    lower = exact - np.arange(0, n) / n
    return float(max(upper.max(), lower.max()))


def run_grid(dtype_name: str, exact_envelope: bool, samples: int, seed: int) -> Path:
    """Runs every (m, kappa) cell and writes the result JSON."""
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(seed)
    rows = []
    for m in DIMS:
        direction = torch.randn(1, m, dtype=dtype)
        loc = direction / direction.norm(dim=-1, keepdim=True)
        for kappa in KAPPAS:
            dist = VonMisesFisher(
                loc,
                torch.full((1, 1), float(kappa), dtype=dtype),
                exact_envelope=exact_envelope,
            )
            z = dist.sample(torch.Size([samples]))
            w = (z * loc).sum(-1).double().numpy().ravel()
            mean_exact, second_exact = exact_moments(m, kappa)
            se_mean = float(w.std(ddof=1) / math.sqrt(samples))
            se_second = float((w**2).std(ddof=1) / math.sqrt(samples))
            rows.append(
                {
                    "m": m,
                    "kappa": kappa,
                    "mean": float(w.mean()),
                    "mean_exact": mean_exact,
                    "mean_se": se_mean,
                    "second": float((w**2).mean()),
                    "second_exact": second_exact,
                    "second_se": se_second,
                    "ks": ks_distance(w, m, float(kappa)),
                    "norm_err": float((z.double().norm(dim=-1) - 1).abs().max()),
                }
            )
        print(f"done m={m}", flush=True)
    directory = out_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dtype_name}_exact{int(exact_envelope)}.json"
    path.write_text(
        json.dumps(
            {
                "dtype": dtype_name,
                "exact_envelope": exact_envelope,
                "samples": samples,
                "seed": seed,
                "rows": rows,
            }
        )
    )
    return path


def flagged(row: dict[str, float], samples: int) -> bool:
    """Whether a cell is biased by the module's stated criteria."""
    z_mean = (row["mean"] - row["mean_exact"]) / row["mean_se"]
    z_second = (row["second"] - row["second_exact"]) / row["second_se"]
    return (
        abs(z_mean) > Z_FLAG
        or abs(z_second) > Z_FLAG
        or row["ks"] * math.sqrt(samples) > KS_FLAG
    )


def summarize() -> str:
    """Renders markdown tables (bias of the mean cosine and flagged map) per variant."""
    lines: list[str] = []
    for path in sorted(out_dir().glob("*.json")):
        data = json.loads(path.read_text())
        rows = data["rows"]
        n = data["samples"]
        by_cell = {(r["m"], r["kappa"]): r for r in rows}
        lines.append(
            f"### {data['dtype']}, exact_envelope={data['exact_envelope']} "
            f"(N = {n} per cell)\n"
        )
        bad = [r for r in rows if flagged(r, n)]
        lines.append(
            f"Flagged cells: {len(bad)} of {len(rows)}. "
            f"Max |mean bias| {max(abs(r['mean'] - r['mean_exact']) for r in rows):.4f}, "
            f"max |second-moment bias| {max(abs(r['second'] - r['second_exact']) for r in rows):.4f}, "
            f"max KS {max(r['ks'] for r in rows):.4f} (1e-3 critical {KS_FLAG / math.sqrt(n):.4f}), "
            f"max |norm - 1| {max(r['norm_err'] for r in rows):.1e}.\n"
        )
        lines.append(
            "Mean-cosine bias x 1000 (sample - exact); `*` = flagged (rows m, columns kappa):\n"
        )
        lines.append("| m \\ kappa | " + " | ".join(str(k) for k in KAPPAS) + " |")
        lines.append("|---|" + "---|" * len(KAPPAS))
        for m in DIMS:
            cells = []
            for kappa in KAPPAS:
                r = by_cell[(m, kappa)]
                text = f"{1000 * (r['mean'] - r['mean_exact']):+.1f}"
                cells.append(text + ("*" if flagged(r, n) else ""))
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
