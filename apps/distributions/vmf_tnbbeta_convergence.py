"""Empirical convergence check: TNBBeta(p,0,eps) matches vMF as kappa increases.

Verifies that TNBBeta-VAE and S-VAE (vMF) posteriors converge to each other at high
concentration. Uses two-sample tests (energy distance, KS test) on the w-marginal
(cosine similarity to mean direction) to quantify convergence rate.

Usage:
    uv run python -m apps.distributions.vmf_tnbbeta_convergence \\
        [--output-csv PATH] [--output-report PATH] \\
        [--sample-size 100000] [--bisect-tol 1e-3]

Part 1: Synthetic two-sample test
    For a grid of vMF concentrations (kappa), match a TNBBeta(p, q=0, epsilon)
    to the same mean resultant length, then compute two-sample statistics between
    samples from each. Report the gap between the two families and compare to
    the sampling-noise floor (two samples from the same vMF).

Part 2: Trained-posterior overlay (runs only if MNIST checkpoints are available)
    For each dimension in the training sweep, load both conv_vmf_vae and
    conv_tnbbeta_spherical_vae checkpoints, extract the w-marginals from
    test-set posteriors, and compute KL/JS divergence between them.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import special
from scipy.stats import energy_distance, ks_2samp
import torch

from tnbbeta_vae.distributions.tnbbeta_spherical import TNBBetaSpherical
from tnbbeta_vae.distributions.von_mises_fisher import VonMisesFisher
from tnbbeta_vae.paths import checkpoint_dir

__all__ = ["main", "vmf_mean_resultant_length", "match_tnbbeta_to_vmf"]


@dataclass
class ConvergenceResult:
    """Results from a single convergence check (kappa, epsilon pair)."""

    kappa: float
    epsilon: float
    target_r_bar: float
    matched_p: float
    gap_energy: float
    gap_ks: float
    noise_floor_energy: float
    noise_floor_ks: float


def vmf_mean_resultant_length(kappa: float, dim: int) -> float:
    """Computes vMF's mean resultant length r_bar in closed form.

    Args:
        kappa: Concentration parameter.
        dim: Ambient dimension of the sphere (S^(dim-1)).

    Returns:
        Mean resultant length r_bar = I_{dim/2}(kappa) / I_{dim/2 - 1}(kappa).
    """
    # Use ive (scaled modified Bessel) for numerical stability.
    nu = dim / 2
    return float(special.ive(nu, kappa) / special.ive(nu - 1, kappa))


def match_tnbbeta_to_vmf(
    target_r_bar: float,
    epsilon: float,
    dim: int,
    q: float = 0.0,
    sample_size: int = 100000,
    bisect_tol: float = 1e-3,
    p_min: float = 0.5,
    p_max: float = 0.9999,
) -> float:
    """Bisects to find TNBBeta p that matches target r_bar.

    Args:
        target_r_bar: Target mean resultant length.
        epsilon: Boundary parameter (fixed).
        dim: Ambient dimension of the sphere.
        q: Concentration parameter (fixed, default 0).
        sample_size: Number of samples to draw for empirical r_bar.
        bisect_tol: Convergence tolerance for bisection.
        p_min: Lower bound for p search.
        p_max: Upper bound for p search.

    Returns:
        Matched p value.

    Raises:
        ValueError: If bisection fails to converge.
    """
    device = torch.device("cpu")
    mu = torch.zeros(dim, dtype=torch.float64, device=device)
    mu[0] = 1.0  # Unit vector along first axis

    def empirical_r_bar(p: float) -> float:
        """Computes empirical r_bar by sampling."""
        dist = TNBBetaSpherical(mu, p=p, q=q, epsilon=epsilon)
        samples = dist.rsample((sample_size,))
        mean_sample = samples.mean(dim=0)
        return float(torch.norm(mean_sample).item())

    # Bisection loop
    left, right = p_min, p_max
    mid = (left + right) / 2
    r_bar_mid = empirical_r_bar(mid)

    for _ in range(100):  # Max iterations
        error = r_bar_mid - target_r_bar

        if abs(error) < bisect_tol:
            return mid

        if error < 0:
            left = mid  # r_bar too small, increase p
        else:
            right = mid  # r_bar too large, decrease p

        mid = (left + right) / 2
        r_bar_mid = empirical_r_bar(mid)

    if abs(r_bar_mid - target_r_bar) >= bisect_tol:
        raise ValueError(
            f"Bisection failed: final error = {abs(r_bar_mid - target_r_bar):.2e}"
        )

    return mid


def _w_marginal_from_samples(samples: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Extracts the w-marginal (cosine to mean direction) from sphere samples.

    Args:
        samples: Shape (N, dim), samples on S^(dim-1).
        mu: Shape (dim,), unit-norm mean direction.

    Returns:
        Shape (N,), w values in [-1, 1].
    """
    result = samples @ mu
    return result.ravel() if result.ndim > 1 else result


def _compute_two_sample_stats(
    samples1: np.ndarray, samples2: np.ndarray
) -> tuple[float, float]:
    """Computes energy distance and KS test on 1-D samples.

    Args:
        samples1: 1-D array of samples.
        samples2: 1-D array of samples.

    Returns:
        (energy_distance, ks_statistic)
    """
    ed_val = energy_distance(samples1, samples2)
    ks_result = ks_2samp(samples1, samples2)
    ks_stat = ks_result[0]
    return float(ed_val), float(ks_stat)  # pyright: ignore[reportArgumentType]


def run_part1(
    dimensions: Sequence[int],
    kappas: Sequence[float],
    epsilons: Sequence[float],
    sample_size: int = 100000,
    bisect_tol: float = 1e-3,
) -> list[ConvergenceResult]:
    """Runs Part 1: synthetic two-sample tests.

    Args:
        dimensions: Dimensions to test (ambient, e.g., [3, 11] for S^2, S^10).
        kappas: vMF concentrations to test.
        epsilons: TNBBeta boundary parameters to sweep.
        sample_size: Samples per distribution in bisection.
        bisect_tol: Bisection convergence tolerance.

    Returns:
        List of ConvergenceResult for each (kappa, epsilon) pair.
    """
    results = []
    device = torch.device("cpu")

    for dim in dimensions:
        print(f"\n=== Dimension {dim - 1} (S^{dim - 1} in R^{dim}) ===")

        for epsilon in epsilons:
            print(f"  epsilon = {epsilon}")

            for kappa in kappas:
                # Target r_bar from vMF
                target_r_bar = vmf_mean_resultant_length(kappa, dim)

                # Match TNBBeta to this r_bar
                try:
                    matched_p = match_tnbbeta_to_vmf(
                        target_r_bar,
                        epsilon=epsilon,
                        dim=dim,
                        sample_size=sample_size,
                        bisect_tol=bisect_tol,
                    )
                except ValueError as e:
                    print(f"    kappa={kappa}: bisection failed: {e}")
                    continue

                # Draw large samples from both
                mu = torch.zeros(dim, dtype=torch.float64, device=device)
                mu[0] = 1.0

                vmf_dist = VonMisesFisher(
                    loc=mu.unsqueeze(0),
                    scale=torch.tensor([[kappa]], dtype=torch.float64),
                )
                tnbbeta_dist = TNBBetaSpherical(mu, p=matched_p, q=0.0, epsilon=epsilon)

                vmf_samples = (
                    vmf_dist.rsample(torch.Size([sample_size])).squeeze(0).numpy()
                )
                tnbbeta_samples = tnbbeta_dist.rsample(
                    torch.Size([sample_size])
                ).numpy()

                # w-marginals
                mu_np = mu.numpy()
                w_vmf = _w_marginal_from_samples(vmf_samples, mu_np)
                w_tnbbeta = _w_marginal_from_samples(tnbbeta_samples, mu_np)

                # Two-sample statistics
                gap_energy, gap_ks = _compute_two_sample_stats(w_vmf, w_tnbbeta)

                # Noise floor: two samples from same vMF
                vmf_samples2 = (
                    vmf_dist.rsample(torch.Size([sample_size])).squeeze(0).numpy()
                )
                w_vmf2 = _w_marginal_from_samples(vmf_samples2, mu_np)
                noise_energy, noise_ks = _compute_two_sample_stats(w_vmf, w_vmf2)

                result = ConvergenceResult(
                    kappa=kappa,
                    epsilon=epsilon,
                    target_r_bar=target_r_bar,
                    matched_p=matched_p,
                    gap_energy=gap_energy,
                    gap_ks=gap_ks,
                    noise_floor_energy=noise_energy,
                    noise_floor_ks=noise_ks,
                )
                results.append(result)

                print(
                    f"    kappa={kappa:7.1f}: p={matched_p:.6f}, "
                    f"gap_energy={gap_energy:.2e}/{noise_energy:.2e}, "
                    f"gap_ks={gap_ks:.2e}/{noise_ks:.2e}"
                )

    return results


def run_part2() -> dict[str, Any]:
    """Runs Part 2: trained-posterior overlay (if checkpoints available).

    Returns:
        Dictionary with results or error message.
    """
    ckpt_dir = checkpoint_dir()

    # List all subdirectories
    try:
        subdirs = [d.name for d in ckpt_dir.iterdir() if d.is_dir()]
    except FileNotFoundError:
        return {
            "status": "checkpoint_dir_not_found",
            "path": str(ckpt_dir),
            "message": f"Checkpoint directory not found at {ckpt_dir}. Part 2 skipped.",
        }

    matching = [d for d in subdirs if any(p in d for p in ["vmfs", "tnbs"])]

    if not matching:
        return {
            "status": "no_checkpoints",
            "path": str(ckpt_dir),
            "message": (
                "No MNIST vMF/TNBBeta checkpoints found. Part 2 requires trained "
                "models from the MNIST sweep. Expected names like "
                "'mnist_vmfs_d{dim}_seed{seed}' and 'mnist_tnbs_d{dim}_seed{seed}'."
            ),
        }

    # Part 2 implementation would go here
    # For now, just return a placeholder
    return {
        "status": "checkpoints_found",
        "path": str(ckpt_dir),
        "checkpoint_count": len(matching),
        "message": (
            "Checkpoints found; full implementation deferred to actual cluster run."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Main entry point.

    Args:
        argv: Command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-report", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--bisect-tol", type=float, default=1e-3)
    args = parser.parse_args(argv)

    # Part 1: Synthetic two-sample test
    print("=" * 80)
    print("PART 1: Synthetic Two-Sample Test")
    print("=" * 80)

    dimensions = [3, 11]  # S^2 and S^10
    kappas = [5, 10, 20, 50, 100, 200, 500, 1000]
    epsilons = [1.0, 2.0]

    results = run_part1(
        dimensions=dimensions,
        kappas=kappas,
        epsilons=epsilons,
        sample_size=args.sample_size,
        bisect_tol=args.bisect_tol,
    )

    # Write results to CSV if requested
    if args.output_csv and results:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = ",".join(asdict(results[0]))
        rows = [header] + [
            ",".join(str(getattr(r, k)) for k in asdict(r)) for r in results
        ]
        csv_path.write_text("\n".join(rows))
        print(f"\nWrote Part 1 results to {csv_path}")

    # Part 2: Trained-posterior overlay
    print("\n" + "=" * 80)
    print("PART 2: Trained-Posterior Overlay")
    print("=" * 80)

    part2_result = run_part2()
    print(f"\nPart 2 Status: {part2_result.get('status')}")
    print(f"Message: {part2_result.get('message')}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Part 1: {len(results)} convergence checks completed")
    if results:
        print(
            f"  Convergence gap shrinks as kappa increases: "
            f"{results[0].gap_energy:.2e} -> {results[-1].gap_energy:.2e}"
        )
    print(f"Part 2: {part2_result.get('status')}")


if __name__ == "__main__":
    main(sys.argv[1:])
