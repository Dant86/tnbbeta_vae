"""Checkpoint diagnostics for the ring task's learned-decoder condition (B), d = 3.

Reads ``ring_recovery.py stage3`` checkpoints
(``<stage3-dir>/ckpt/d3_<cond>_<variant>_s<k>.pt``) and, per checkpoint, rebuilds the
model and evaluates on the fixed test set (seed 424242):

* the ELBO split into reconstruction ``E_q[log p(x|z)]`` and ``KL(q || Uniform)``
  (Monte Carlo for every family; analytic too for vMF / Power Spherical), at
  ``--eval-samples`` draws;
* the model's own log marginal likelihood ``log p_model(x)`` by uniform-prior MC
  over ``--num-prior-z`` latent points (no use of ``q``), so ``log p(x) - ELBO`` splits
  into *decoder misfit* ``log p(x) - log p_model(x)`` plus *inference gap*
  ``log p_model(x) - ELBO = KL(q || p_model(z|x))``;
* for the learned decoder: how far its image is from the true curve ``g([-1, 1])``,
  whether the latent enters through one linear coordinate (``a' . z``), and the
  learned sigma.

Usage::

    uv run python -m apps.synthetic.ring_condition_b_diagnostics \
        --stage3-dir ring_recovery/l23_d3/stage3 --out diagnostics.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import re
from typing import Any, cast

import torch
from torch import Tensor
from torch.distributions import kl_divergence

from apps.synthetic.ring_recovery import (
    PRIMARY_SIGMA,
)
from apps.synthetic.ring_recovery import (
    _stage3_fixed_epsilon as stage3_fixed_epsilon,
)
from tnbbeta_vae.data.ring_posterior import (
    RingPosteriorTask,
    ring_task,
    sphere_log_area,
)
from tnbbeta_vae.distributions import HypersphericalUniform
from tnbbeta_vae.models import MlpVAE, MlpVAEConfig
from tnbbeta_vae.models.losses.sampling import sample_with_log_prob

DIM = 3
_CKPT_PATTERN = re.compile(r"d3_(true|learned)_(.+)_s(\d+)\.pt")


def main(argv: list[str] | None = None) -> None:
    """Runs the diagnostics over every checkpoint and writes one JSON list.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=["learned", "true"])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--num-prior-z", type=int, default=400_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    jobs = []
    for path in sorted((args.stage3_dir / "ckpt").glob("*.pt")):
        match = _CKPT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        condition, variant, seed = match.group(1), match.group(2), int(match.group(3))
        if condition not in args.conditions:
            continue
        if args.variants is not None and variant not in args.variants:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue
        jobs.append(
            (path, condition, variant, seed, args.eval_samples, args.num_prior_z)
        )
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_diagnose_star, jobs))
    args.out.write_text(json.dumps(results))
    print(f"wrote {len(results)} rows to {args.out}")


def reference_values(task: RingPosteriorTask, x: Tensor) -> dict[str, float]:
    """Exact-posterior reconstruction term and ``KL(p_true || Uniform)`` (test means).

    Args:
        task: The ring task.
        x: Test observations, shape ``(n, 3)``.

    Returns:
        ``{"log_evidence", "recon_true", "kl_true_to_prior"}``; the ELBO of the exact
        posterior equals ``log p(x)``, so ``kl = recon - log p(x)``.
    """
    x64 = x.double()
    theta, density = task.latitude_density(x64)
    delta = math.pi / task.grid_size
    w = torch.cos(theta)
    features = task.features(w)  # (grid, 3)
    sq = ((x64[:, None, :] - features[None]) / task.sigma).pow(2).sum(-1)
    log_lik = -0.5 * sq - 3 * math.log(task.sigma) - 1.5 * math.log(2 * math.pi)
    recon = (density * log_lik * delta).sum(-1)
    evidence = task.log_evidence(x64)
    return {
        "log_evidence": float(evidence.mean()),
        "recon_true": float(recon.mean()),
        "kl_true_to_prior": float((recon - evidence).mean()),
    }


def load_model(
    path: Path, condition: str, variant: str, task: RingPosteriorTask
) -> MlpVAE:
    """Rebuilds a stage-3 model (``d = 3``) and loads its checkpoint.

    Args:
        path: Checkpoint ``.pt`` (a ``state_dict``).
        condition: ``"true"`` or ``"learned"``.
        variant: Stage-3 variant name.
        task: The ring task.

    Returns:
        The model in eval mode, on CPU.
    """
    family = "tnbbeta" if variant.startswith("tnbbeta") else variant
    model = MlpVAE(
        MlpVAEConfig(
            family=cast("Any", family),
            input_dim=3,
            hidden_dims=[128, 128],
            latent_dim=DIM,
            likelihood_scale=PRIMARY_SIGMA,
            fixed_epsilon=stage3_fixed_epsilon(variant, task),
        )
    )
    if condition == "true":
        model.decoder = cast("Any", _TrueDecoder(task))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def _diagnose_star(job: tuple[Path, str, str, int, int, int]) -> dict[str, Any]:
    return diagnose(*job)


@torch.no_grad()
def diagnose(
    path: Path,
    condition: str,
    variant: str,
    seed: int,
    eval_samples: int,
    num_prior_z: int,
) -> dict[str, Any]:
    """Diagnostics for one checkpoint; see the module docstring.

    Args:
        path: Checkpoint ``.pt`` (a ``state_dict``).
        condition: ``"true"`` or ``"learned"``.
        variant: Stage-3 variant name.
        seed: Training seed.
        eval_samples: ``z ~ q`` draws per test x.
        num_prior_z: Uniform latent points for ``log p_model(x)``.

    Returns:
        A flat dict of scalars.
    """
    torch.set_num_threads(1)
    task = ring_task(DIM, PRIMARY_SIGMA)
    test_x, _ = task.sample(2000, torch.Generator().manual_seed(424242))
    family = "tnbbeta" if variant.startswith("tnbbeta") else variant
    model = load_model(path, condition, variant, task)
    sigma = float(model.learned_scale())
    log_evidence = task.log_evidence(test_x.double()).float()
    out: dict[str, Any] = {
        "condition": condition,
        "variant": variant,
        "seed": seed,
        "learned_sigma": sigma,
    }
    torch.manual_seed(7)
    recon, kl_mc, kl_an = [], [], []
    for start in range(0, test_x.shape[0], 100):
        xs = test_x[start : start + 100]
        posterior, _ = model.posterior_and_prior(xs)
        z, log_q = sample_with_log_prob(posterior, torch.Size([eval_samples]))
        assert log_q is not None
        recon.append(model.log_likelihood(xs, z).mean(0))
        kl_mc.append((log_q + sphere_log_area(DIM)).mean(0))
        if family != "tnbbeta":
            kl_an.append(kl_divergence(posterior, HypersphericalUniform(DIM - 1)))
    recon_all, kl_all = torch.cat(recon), torch.cat(kl_mc)
    elbo = recon_all - kl_all
    out.update(
        recon=float(recon_all.mean()),
        kl_prior_mc=float(kl_all.mean()),
        kl_prior_analytic=float(torch.cat(kl_an).mean()) if kl_an else None,
        elbo=float(elbo.mean()),
        elbo_gap=float((log_evidence - elbo).mean()),
        min_per_x_gap=float((log_evidence - elbo).min()),
    )
    log_marg = log_marginal(model, test_x, num_prior_z)
    out.update(
        log_marginal_model=float(log_marg.mean()),
        decoder_misfit=float((log_evidence - log_marg).mean()),
        inference_gap=float((log_marg - elbo).mean()),
        frac_elbo_above_model_marginal=float((elbo > log_marg + 0.05).float().mean()),
    )
    if condition == "learned":
        out.update(_decoder_structure(model, task))
    return out


def log_marginal(model: MlpVAE, x: Tensor, num_z: int) -> Tensor:
    """``log mean_n N(x; f(z_n), sigma^2 I)`` with ``z_n`` uniform on the sphere."""
    generator = torch.Generator().manual_seed(99)
    z = torch.randn(num_z, DIM, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    f = model.decoder(z)
    sigma = float(model.learned_scale())
    constant = -1.5 * math.log(2 * math.pi * sigma**2)
    rows = []
    for start in range(0, x.shape[0], 50):
        xs = x[start : start + 50]
        sq = torch.cdist(xs, f).pow(2)
        rows.append(torch.logsumexp(-sq / (2 * sigma**2), dim=-1) - math.log(num_z))
    return constant + torch.cat(rows)


def _decoder_structure(model: MlpVAE, task: RingPosteriorTask) -> dict[str, float]:
    """Distance of ``f(S^2)`` to the true curve; single-coordinate dependence."""
    generator = torch.Generator().manual_seed(5)
    z = torch.randn(40_000, DIM, generator=generator)
    z = z / z.norm(dim=-1, keepdim=True)
    f = model.decoder(z)
    w_grid = torch.linspace(-1, 1, 20_001)
    curve = task.features(w_grid)
    dists = []
    nearest = []
    for chunk in f.split(2_000):
        d = torch.cdist(chunk, curve)
        best, index = d.min(dim=-1)
        dists.append(best)
        nearest.append(w_grid[index])
    distance = torch.cat(dists)
    w_star = torch.cat(nearest)
    # Pushforward of the uniform sphere through w*(z) vs uniform on [-1, 1].
    hist = torch.histc(w_star, bins=20, min=-1, max=1) / w_star.numel()
    tv_uniform = 0.5 * float((hist - 1 / 20).abs().sum())
    # Single linear coordinate: regress w* on z, then binned eta^2 on the fitted axis.
    design = torch.cat([z, torch.ones(z.shape[0], 1)], dim=-1)
    coef = torch.linalg.lstsq(design, w_star[:, None]).solution[:, 0]
    axis = coef[:DIM] / coef[:DIM].norm()
    coordinate = z @ axis
    r2_linear = float(
        1
        - (w_star - design @ coef).pow(2).sum() / (w_star - w_star.mean()).pow(2).sum()
    )
    bins = torch.bucketize(coordinate, torch.linspace(-1, 1, 41)[1:-1])
    within = torch.zeros(())
    for b in range(40):
        group = w_star[bins == b]
        if group.numel() > 1:
            within += (group - group.mean()).pow(2).sum()
    eta2 = float(1 - within / (w_star - w_star.mean()).pow(2).sum())
    return {
        "dec_rms_dist_to_true_curve": float(distance.pow(2).mean().sqrt()),
        "dec_median_dist_to_true_curve": float(distance.median()),
        "dec_wstar_tv_to_uniform": tv_uniform,
        "dec_wstar_r2_linear_in_z": r2_linear,
        "dec_wstar_eta2_on_best_axis": eta2,
        "dec_abs_cos_best_axis_vs_true_axis": float((axis @ task.axis).abs()),
        "dec_output_sd_total": float(f.var(dim=0).sum().sqrt()),
    }


class _TrueDecoder(torch.nn.Module):
    """The task's true decoder ``z -> g(a . z)`` (no parameters)."""

    def __init__(self, task: RingPosteriorTask) -> None:
        super().__init__()
        self.task = task

    def forward(self, z: Tensor) -> Tensor:
        return self.task.decode(z)


if __name__ == "__main__":
    main()
