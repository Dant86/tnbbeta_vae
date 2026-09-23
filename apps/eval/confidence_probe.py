"""Class-probe accuracy from a posterior's confidence scalar alone (not its direction).

Usage:
    uv run python -m apps.eval.confidence_probe --run-name NAME [--checkpoint final] \
        [--param p|epsilon] [--label-counts 100 600 1000] [--num-subsets 20] [--k 5]

The Gaussian's per-example mean standard deviation, vMF's kappa and TNBBeta's confidence
scalar are each a single scalar carrying no directional information. Running the same
k-NN probe as ``apps.eval.svae_knn`` on that scalar alone tests whether a family routes
class information through it, separately from the mean direction.

TNBBeta has two candidate confidence scalars, and which one is the fair comparison to
vMF's kappa depends on how training actually shapes the posterior: on MNIST it collapses
to the q=0, p->1 special case (a cap at the mean direction, same shape as vMF's), so p
saturates near 1 with almost no spread and cannot carry class information by
construction -- epsilon (the cap's thickness in that special case) is the parameter
actually analogous to kappa there. ``--param`` defaults to p (matching the original
per-family default: mean std for Gaussian, kappa for vMF, p for TNBBeta) for
backward compatibility with already-computed results; pass ``--param epsilon`` for
the fairer TNBBeta comparison. Passing ``--param epsilon`` for a non-TNBBeta run
(no such parameter) prints a message and exits without writing anything, so it is
safe to chain unconditionally across every model in a sweep.

Writes ``confidence_probe_<checkpoint>.json`` for the default parameter
(unchanged, so already-backfilled runs stay valid) or
``confidence_probe_<param>_<checkpoint>.json`` for an explicit non-default
``--param``, in the same shape as ``svae_knn_<checkpoint>.json`` (``acc_<N>`` and
``acc_<N>_std`` per label count), using plain Euclidean distance on the scalar.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from apps.eval.svae_knn import knn_accuracy
from tnbbeta_vae.data.mnist import load_mnist
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint

_DEFAULT_PARAM = {
    "conv_gaussian_vae": "mean_std",
    "conv_vmf_vae": "kappa",
    "conv_tnbbeta_spherical_vae": "p",
}
_EXTRACTORS: dict[tuple[str, str], Any] = {
    ("conv_gaussian_vae", "mean_std"): lambda p: p.base_dist.scale.mean(
        dim=-1, keepdim=True
    ),
    ("conv_vmf_vae", "kappa"): lambda p: p.scale,
    ("conv_tnbbeta_spherical_vae", "p"): lambda p: p.p.unsqueeze(-1),
    ("conv_tnbbeta_spherical_vae", "epsilon"): lambda p: p.epsilon.unsqueeze(-1),
}


def main(argv: list[str] | None = None) -> None:
    """Runs the confidence-only probe for one trained run.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument(
        "--param",
        default=None,
        help="Which posterior parameter to probe; defaults to the family's "
        "usual confidence scalar (see above). Only TNBBeta has more than one "
        "choice (p, epsilon).",
    )
    parser.add_argument("--label-counts", nargs="+", type=int, default=[100, 600, 1000])
    parser.add_argument("--num-subsets", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)
    kind = checkpoint["model_name"]
    default_param = _DEFAULT_PARAM[kind]
    param = args.param or default_param
    if (kind, param) not in _EXTRACTORS:
        print(
            f"{args.run_name}: model is {kind}, has no '{param}' confidence scalar; "
            "skipping."
        )
        return

    train_set = load_mnist(data_dir(), split="train")
    test_set = load_mnist(data_dir(), split="test")
    train_confidence = _confidences(model, kind, param, train_set, args, device)
    test_confidence = _confidences(model, kind, param, test_set, args, device)
    train_labels, test_labels = train_set.labels(), test_set.labels()

    rng = np.random.default_rng(args.seed)
    results: dict[str, Any] = {
        "run_name": args.run_name,
        "model_name": kind,
        "confidence": param,
        "k": args.k,
        "num_subsets": args.num_subsets,
    }
    for count in args.label_counts:
        accuracies = []
        for _ in range(args.num_subsets):
            chosen = torch.as_tensor(
                rng.choice(len(train_confidence), count, replace=False)
            )
            accuracies.append(
                knn_accuracy(
                    train_confidence[chosen],
                    train_labels[chosen],
                    test_confidence,
                    test_labels,
                    args.k,
                    "euclidean",
                )
            )
        results[f"acc_{count}"] = float(np.mean(accuracies))
        results[f"acc_{count}_std"] = float(np.std(accuracies, ddof=1))

    suffix = "" if param == default_param else f"_{param}"
    output = run_dir / f"confidence_probe{suffix}_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


@torch.no_grad()
def _confidences(
    model: Any,
    kind: str,
    param: str,
    dataset: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    extractor = _EXTRACTORS[(kind, param)]
    parts = []
    for batch in loader:
        posterior, _ = model.posterior_and_prior(batch.to(device))
        parts.append(extractor(posterior).cpu())
    return torch.cat(parts)


if __name__ == "__main__":
    main(sys.argv[1:])
