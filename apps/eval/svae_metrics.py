"""S-VAE Table 1 metrics (LL, ELBO, RE, KL) for one trained MNIST run.

Usage:
    uv run python -m apps.eval.svae_metrics --run-name NAME [--checkpoint final] \
        [--split test] [--num-samples 500]

Reads ``$TNBBETA_CHECKPOINT_DIR/<run-name>/<checkpoint>.pt`` and writes
``svae_metrics_<checkpoint>_<split>.json`` next to it. All values are means over
the split, in nats per image: ``ll`` is the importance-weighted log-likelihood
(``--num-samples`` samples per image, 500 in the paper), ``elbo`` is ``re - kl``,
``re`` is the expected log-likelihood ``E_q[log p(x|z)]`` and ``kl`` is
``KL(q(z|x) || p(z))``. The val and test splits are binarized once with a fixed
seed, so results are comparable across models.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

from tnbbeta_vae.data.mnist import load_mnist
from tnbbeta_vae.models.losses import importance_weighted_metrics
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint


def main(argv: list[str] | None = None) -> None:
    """Evaluates a checkpoint and writes the metrics next to it.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--sample-chunk", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
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
    loader = DataLoader(
        load_mnist(data_dir(), split=args.split),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    totals = {"ll": 0.0, "elbo": 0.0, "re": 0.0, "kl": 0.0}
    count = 0
    for batch in loader:
        batch = batch.to(device)
        metrics = importance_weighted_metrics(
            model, batch, num_samples=args.num_samples, chunk_size=args.sample_chunk
        )
        for name in totals:
            totals[name] += metrics[name].sum().item()
        count += batch.shape[0]

    results: dict[str, Any] = {
        "run_name": args.run_name,
        "model_name": checkpoint["model_name"],
        "latent_dim": checkpoint["config"]["latent_dim"],
        "epochs_completed": checkpoint["epochs_completed"],
        "split": args.split,
        "num_samples": args.num_samples,
        **{name: total / count for name, total in totals.items()},
    }
    output = run_dir / f"svae_metrics_{args.checkpoint}_{args.split}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main(sys.argv[1:])
