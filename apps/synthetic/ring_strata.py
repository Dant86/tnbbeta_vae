"""Stratifies the d = 3 ring task's test posteriors and per-x stage-3 KL by shape.

The ring task's noisy observations do not all have a single-ring exact posterior: the
periodic features ``sin 3w`` and ``cos 5w`` can leave a second latitude mode, and an x
with latitude near a pole has a cap, not a ring. This script classifies each of the 2000
stage-3 test observations (seed 424242) as ``single ring``, ``two-mode`` (second
mode with more than 5% mass) or ``polar cap`` (more than half the mass within 20 degrees
of a pole), prints the posterior statistics and, for the finished true-decoder
checkpoints in ``ring_recovery/l23_d3/stage3/ckpt``, the mean exact KL per family and
stratum.

Usage:
    uv run python -m apps.synthetic.ring_strata
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor

from tnbbeta_vae.data.ring_posterior import RingPosteriorTask, exact_kl, ring_task
from tnbbeta_vae.models import MlpVAE, MlpVAEConfig

DIM = 3
SIGMA = 0.2
TEST_SEED = 424242
NUM_TEST = 2000
SECOND_MODE_MASS = 0.05
POLAR_DEGREES = 20.0
CKPT = Path("ring_recovery/l23_d3/stage3/ckpt")


def main() -> None:
    """Prints the posterior statistics and the per-stratum KL table."""
    task = ring_task(DIM, SIGMA)
    generator = torch.Generator().manual_seed(TEST_SEED)
    x, _ = task.sample(NUM_TEST, generator)
    strata, stats = _classify(task, x)
    for line in stats:
        print(line)
    print({name: int(mask.sum()) for name, mask in strata.items()})
    folds = {"vmf": range(10), "power_spherical": range(10), "tnbbeta": range(10)}
    per_seed: dict[str, torch.Tensor] = {}
    for family, seeds in folds.items():
        kls = []
        for seed in seeds:
            model = _load(family, seed)
            with torch.no_grad():
                posterior = model.posterior_and_prior(x)[0]
                kls.append(exact_kl(posterior, task, x, num_samples=500)[0])
        per_seed[family] = torch.stack(kls)
        table = {
            name: round(float(per_seed[family][:, mask].mean()), 2)
            for name, mask in strata.items()
        }
        print(family, f"mean KL {float(per_seed[family].mean()):.2f}", table)
    kl = per_seed["tnbbeta"]
    for label, seeds in (("aligned", [2, 5, 6, 7, 9]), ("other", [0, 1, 3, 4, 8])):
        table = {
            name: round(float(kl[seeds][:, mask].mean()), 2)
            for name, mask in strata.items()
        }
        print(f"tnbbeta {label} seeds", table)


def _classify(
    task: RingPosteriorTask, x: Tensor
) -> tuple[dict[str, Tensor], list[str]]:
    """Returns the stratum masks and descriptive lines for the test observations."""
    theta, density = task.latitude_density(x.double())
    mass = density * (math.pi / task.grid_size)
    second = torch.zeros(len(x))
    for i in range(len(x)):
        d = density[i]
        minima = ((d[1:-1] < d[:-2]) & (d[1:-1] <= d[2:])).nonzero().flatten() + 1
        bounds = [0, *minima.tolist(), len(d)]
        segs = sorted(
            (
                float(mass[i, a:b].sum())
                for a, b in zip(bounds[:-1], bounds[1:], strict=True)
            ),
            reverse=True,
        )
        second[i] = segs[1] if len(segs) > 1 else 0.0
    radians = math.radians(POLAR_DEGREES)
    polar_mass = (mass * ((theta < radians) | (theta > math.pi - radians))).sum(-1)
    two = second > SECOND_MODE_MASS
    polar = polar_mass > 0.5
    strata = {
        "single ring": ~two & ~polar,
        "two-mode": two & ~polar,
        "polar cap": polar,
    }
    mean_t = (mass * theta).sum(-1)
    sd_t = (mass * (theta - mean_t[:, None]) ** 2).sum(-1).sqrt()
    quantiles = torch.tensor([0.05, 0.5, 0.95], dtype=sd_t.dtype)
    lines = [
        "posterior sd of latitude t (deg), 5/50/95%: "
        f"{[round(math.degrees(float(v)), 1) for v in sd_t.quantile(quantiles)]}",
        "second-mode mass > 1%/5%/20%: "
        f"{float((second > 0.01).double().mean()):.3f} / "
        f"{float(two.double().mean()):.3f} / "
        f"{float((second > 0.2).double().mean()):.3f}",
        f"polar share (>50% mass within {POLAR_DEGREES:g} deg of a pole): "
        f"{float(polar.double().mean()):.3f}",
    ]
    return strata, lines


def _load(family: str, seed: int) -> MlpVAE:
    model = MlpVAE(
        MlpVAEConfig(
            family=family,  # pyright: ignore[reportArgumentType]
            input_dim=3,
            hidden_dims=[128, 128],
            latent_dim=DIM,
            likelihood_scale=SIGMA,
            num_elbo_samples=4,
        )
    )
    state = torch.load(CKPT / f"d3_true_{family}_s{seed}.pt", map_location="cpu")
    state = {k: v for k, v in state.items() if not k.startswith("decoder.")}
    model.load_state_dict(state, strict=False)
    return model.eval()


if __name__ == "__main__":
    main()
