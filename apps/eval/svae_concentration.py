"""Quantifies whether a run's per-class posteriors are thin caps, MNIST test split.

Usage:
    uv run python -m apps.eval.svae_concentration --run-name NAME [--checkpoint final]

Two questions, in the order they need answering: first *shape* -- is each class a
single unimodal cap, or does it have a genuinely different (ring/bimodal) spread the
ring-exclusion check below would catch -- and only once that holds does a *magnitude*
comparison (how tight is the cap) mean anything. Applies to ``conv_vmf_vae`` and
``conv_tnbbeta_spherical_vae`` runs (vMF structurally cannot produce anything but a
cap, so it is included as the reference case); a no-op for any other model.

For each class, using every test image's posterior centre direction
(``VonMisesFisher.loc`` or TNBBeta's alias-resolved ``mode_direction``):

* ``far_side_pct``: percentage of a class's points more than 90 degrees from their own
  class's centroid direction -- i.e. on the far hemisphere from where most of the class
  sits. A thin cap has ~0% of these; a ring or antipodal-bimodal spread would not.
* ``antipodal_pct``: percentage within 0.9 cosine similarity of the exact antipode of
  their own centroid -- a stricter version of the same check.
* ``r_bar``: mean resultant length, ``||mean_i(z_i)||`` -- the standard directional-
  statistics concentration measure (1 = a point mass, 0 = uniform on the sphere),
  the same "how tight is the cap" quantity regardless of which family produced it.
* ``equivalent_kappa``: ``r_bar`` converted to the vMF concentration a fitted vMF with
  that same ``r_bar`` would report (Banerjee et al. 2005's approximation), so TNBBeta's
  whole (p, q, epsilon) apparatus lands in the same unit vMF's own kappa is already
  reported in.

Writes ``concentration_<checkpoint>.json`` (per-class and overall) next to the
checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tnbbeta_vae.data.mnist import load_mnist
from tnbbeta_vae.models.heads import LatentFamily, posterior_centre
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint

_FAMILY_BY_MODEL: dict[str, LatentFamily] = {
    "conv_vmf_vae": "vmf",
    "conv_tnbbeta_spherical_vae": "tnbbeta",
}
_FAR_SIDE_COSINE = 0.0
_ANTIPODAL_COSINE = -0.9
_NUM_CLASSES = 10


def main(argv: list[str] | None = None) -> None:
    """Computes and writes the ring-check and concentration diagnostic for one run.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)
    kind = checkpoint["model_name"]
    if kind not in _FAMILY_BY_MODEL:
        print(f"{args.run_name}: model is {kind}, which has no spherical latent to "
              "check; skipping.")  # fmt: skip
        return

    dim = checkpoint["config"]["latent_dim"]
    directions, labels = _centres(model, _FAMILY_BY_MODEL[kind], args, device)

    per_class = {
        str(k): _class_stats(directions[labels == k], dim) for k in range(_NUM_CLASSES)
    }
    overall = {
        name: float(np.mean([stats[name] for stats in per_class.values()]))
        for name in ("far_side_pct", "antipodal_pct", "r_bar", "equivalent_kappa")
    }
    results: dict[str, Any] = {
        "run_name": args.run_name,
        **overall,
        "per_class": per_class,
    }
    output = run_dir / f"concentration_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


@torch.no_grad()
def _centres(
    model: Any, family: LatentFamily, args: argparse.Namespace, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(centre directions, labels)`` for the whole MNIST test split."""
    test_set = load_mnist(data_dir(), split="test")
    loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers
    )
    parts = []
    for batch in loader:
        posterior, _ = model.posterior_and_prior(batch.to(device))
        parts.append(posterior_centre(family, posterior).cpu())
    return torch.cat(parts).numpy(), test_set.labels().numpy()


def _class_stats(vectors: np.ndarray, dim: int) -> dict[str, float]:
    """Returns one class's far-side/antipodal fractions, r_bar and equivalent kappa."""
    raw_mean = vectors.mean(axis=0)
    r_bar = float(np.linalg.norm(raw_mean))
    centroid = raw_mean / r_bar
    cosine = vectors @ centroid
    # A class with very few points (or, in the limit, exactly one) can have r_bar
    # exactly 1.0, where the kappa approximation's denominator is exactly zero; the
    # reported r_bar itself is left exact, only the kappa formula's input is capped.
    capped_r_bar = min(r_bar, 1.0 - 1e-9)
    return {
        "far_side_pct": float(100 * np.mean(cosine < _FAR_SIDE_COSINE)),
        "antipodal_pct": float(100 * np.mean(cosine < _ANTIPODAL_COSINE)),
        "r_bar": r_bar,
        "equivalent_kappa": (
            capped_r_bar * (dim - capped_r_bar**2) / (1 - capped_r_bar**2)
        ),
    }


if __name__ == "__main__":
    main(sys.argv[1:])
