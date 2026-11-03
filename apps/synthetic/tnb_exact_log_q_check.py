"""Does TNBBeta still collapse epsilon to its floor once the ELBO uses the exact log q?

Verification for ``writeup/results/tnb_exact_log_q_in_elbo_2026-11-01.md``. Two modes,
both scored with the independent honest evaluator of
:mod:`apps.synthetic.tnb_low_epsilon_check` (exact ``E_q[log q]`` by quadrature, an
exact sampler for the cross term):

* ``stage2``: the stage-2 per-datum optimisation of ``ring_recovery`` for TNBBeta only
  (amortised-like and random-restart inits), whose training ELBO and restart-selection
  ELBO now use ``rsample_with_log_prob``.
* ``stage3``: a stage-3 checkpoint written by ``ring_recovery stage3`` (rebuilt with
  the same ``--epsilon-floor`` it was trained with).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from apps.synthetic import ring_recovery as rr
from apps.synthetic.tnb_low_epsilon_check import honest_kl
from tnbbeta_vae.data.ring_posterior import RingPosteriorTask, exact_kl, ring_task
from tnbbeta_vae.models import MlpVAE, MlpVAEConfig
from tnbbeta_vae.models.heads import posterior_from_raw

__all__ = ["main", "summarise"]

_FLOOR = 2e-4  # "at the head floor": epsilon within 1e-4 of the 1e-4 floor


def main(argv: list[str] | None = None) -> None:
    """Runs one mode and writes its JSON.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    s2 = sub.add_parser("stage2")
    s2.add_argument("--steps", type=int, default=2000)
    s2.add_argument("--num-x", type=int, default=300)
    s2.add_argument("--restarts", type=int, default=8)
    s2.add_argument("--out", type=Path, required=True)
    s3 = sub.add_parser("stage3")
    s3.add_argument("--ckpt", type=Path, required=True)
    s3.add_argument("--epsilon-floor", type=float, default=1e-4)
    s3.add_argument("--condition", choices=["true", "learned"], default="learned")
    s3.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    result = _stage2(args) if args.mode == "stage2" else _stage3(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def summarise(
    posterior: Any, task: RingPosteriorTask, x: Tensor, library_draws: int = 2000
) -> dict[str, float]:
    """Epsilon statistics plus the library and honest KL of a TNBBeta batch.

    Args:
        posterior: Batch ``TNBBetaSpherical`` over ``S^2`` for the rows of ``x``.
        task: The ring task (float32 axis).
        x: Observations, shape ``(batch, 3)``.
        library_draws: Draws per row for the library estimator.

    Returns:
        Epsilon, library-KL (exact ``log q`` path) and honest-KL summaries.
    """
    task64 = RingPosteriorTask(task.dim, task.sigma, task.axis.double())
    with torch.no_grad():
        eps = posterior.epsilon.double()
        torch.manual_seed(1)
        library, _ = exact_kl(posterior, task, x, library_draws)
        torch.manual_seed(2)
        honest = honest_kl(posterior, task64, x, 2000)
    return {
        "num_x": int(x.shape[0]),
        "eps_min": float(eps.min()),
        "eps_median": float(eps.median()),
        "frac_eps_at_floor": float((eps < _FLOOR).double().mean()),
        "frac_eps_below_0.03": float((eps < 0.03).double().mean()),
        "frac_eps_below_0.3": float((eps < 0.3).double().mean()),
        "library_kl_mean": float(library.mean()),
        "library_kl_median": float(library.median()),
        "library_kl_frac_below_1": float((library < 1).double().mean()),
        "library_kl_num_negative": int((library < 0).sum()),
        "honest_kl_mean": float(honest.mean()),
        "honest_kl_median": float(honest.median()),
        "honest_kl_frac_below_1": float((honest < 1).double().mean()),
        "honest_kl_num_negative": int((honest < 0).sum()),
    }


def _stage2(args: argparse.Namespace) -> dict[str, Any]:
    task = ring_task(3, rr.PRIMARY_SIGMA)
    generator = torch.Generator().manual_seed(777)
    x, _ = task.sample(args.num_x, generator)
    out: dict[str, Any] = {"steps": args.steps}
    for init, restarts, scale in (
        ("amortised_like", 1, 0.1),
        ("random_restarts", args.restarts, 2.0),
    ):
        torch.manual_seed(0)
        raw, _ = rr._optimise_per_datum("tnbbeta", task, x, restarts, scale, args.steps)  # noqa: SLF001
        out[init] = summarise(posterior_from_raw("tnbbeta", raw, 3), task, x)
    return out


def _stage3(args: argparse.Namespace) -> dict[str, Any]:
    task = ring_task(3, rr.PRIMARY_SIGMA)
    test_x, _ = task.sample(2000, torch.Generator().manual_seed(424242))
    model = MlpVAE(
        MlpVAEConfig(
            family="tnbbeta",
            input_dim=3,
            hidden_dims=[128, 128],
            latent_dim=3,
            likelihood_scale=rr.PRIMARY_SIGMA,
            num_elbo_samples=4,
            epsilon_floor=args.epsilon_floor,
        )
    )
    if args.condition == "true":
        model.decoder = rr._TrueDecoder(task)  # type: ignore[assignment]  # noqa: SLF001
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        posterior, _ = model.posterior_and_prior(test_x)
    result = summarise(posterior, task, test_x)
    result["epsilon_floor"] = args.epsilon_floor
    return result


if __name__ == "__main__":
    main()
