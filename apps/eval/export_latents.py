"""Exports every test image's posterior parameters and class label for analysis.

Usage:
    uv run python -m apps.eval.export_latents --run-name NAME [--checkpoint final] \
        [--split test]

Writes ``$TNBBETA_CHECKPOINT_DIR/<run-name>/latents_<checkpoint>_<split>.npz``
with, per image and in dataset order:

* ``labels``: CIFAR-10 class (0-9).
* ``direction``: the posterior's mean direction (TNBBeta) or mean (Gaussian).
* ``mode_direction``: TNBBeta only -- the direction the posterior is centered
  on. The parameterization has an antipodal alias ((mu, p) and (-mu, 1 - p)
  describe the same distribution), so this is ``direction`` if ``p > 0.5`` and
  ``-direction`` otherwise.
* ``p``, ``q``, ``epsilon``: TNBBeta posterior parameters (TNBBeta only).
* ``concentration``: the Gaussian's mean posterior std (per image; Gaussian only).
* ``z``: one posterior sample.
* ``kl``: per-image KL to the prior (exact for the Gaussian, a 16-sample
  Monte Carlo estimate for TNBBeta).

It also writes ``latent_probe_<checkpoint>_<split>.json`` with the accuracy of
a k-nearest-neighbour class probe on several features, as a rough measure of
how much class structure the posterior carries. Use the ``*_cosine`` entries
to compare models: raw Euclidean distance penalizes latents whose vector
lengths vary (e.g. Gaussian means), which unit-length sphere directions avoid.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch
from torch.distributions import Independent, Normal, kl_divergence
from torch.utils.data import DataLoader

from tnbbeta_vae.data.cifar10 import load_cifar10
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import load_model_checkpoint

_KL_SAMPLES = 16
_PROBE_NEIGHBOURS = 10


def main(argv: list[str] | None = None) -> None:
    """Exports latents and prints a class-probe summary.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--split", default="test", choices=["test", "train"])
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
    loaded, checkpoint = load_model_checkpoint(
        run_dir / f"{args.checkpoint}.pt", device
    )
    model: Any = loaded
    kind = checkpoint["model_name"]

    dataset = load_cifar10(data_dir(), train=args.split == "train")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    chunks: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in loader:
            for name, value in _encode_batch(model, kind, batch.to(device)).items():
                chunks.setdefault(name, []).append(value.cpu())
    arrays = {name: torch.cat(parts).numpy() for name, parts in chunks.items()}
    arrays["labels"] = dataset.labels().numpy()

    tag = f"{args.checkpoint}_{args.split}"
    payload: dict[str, Any] = {"model_name": kind, **arrays}
    np.savez_compressed(run_dir / f"latents_{tag}.npz", **payload)
    probe = _probe_accuracies(arrays)
    (run_dir / f"latent_probe_{tag}.json").write_text(json.dumps(probe, indent=2))
    print(json.dumps(probe, indent=2))
    print(f"Wrote {run_dir / f'latents_{tag}.npz'}")


def _encode_batch(
    model: Any, kind: str, batch: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Returns per-image posterior parameters, a sample and the KL for one batch."""
    if kind == "conv_gaussian_vae":
        mu, sigma = model._encode(batch)
        posterior = Independent(Normal(mu, sigma), 1)
        prior = Independent(Normal(torch.zeros_like(mu), torch.ones_like(sigma)), 1)
        return {
            "direction": mu,
            "concentration": sigma.mean(dim=-1),
            "z": posterior.sample(),
            "kl": kl_divergence(posterior, prior),
        }
    posterior = model._encode(batch)
    prior = model.prior()
    z = posterior.sample((_KL_SAMPLES,))
    kl = (posterior.log_prob(z) - prior.log_prob(z)).mean(dim=0)
    direction = posterior.mean_direction
    return {
        "direction": direction,
        "mode_direction": torch.where(
            (posterior.p > 0.5)[:, None], direction, -direction
        ),
        "p": posterior.p,
        "q": posterior.q,
        "epsilon": posterior.epsilon,
        "z": z[0],
        "kl": kl,
    }


def _probe_accuracies(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """k-NN class accuracy (train on the first half, test on the second half)."""
    labels = torch.as_tensor(arrays["labels"])
    features: dict[str, np.ndarray] = {
        "direction": arrays["direction"],
        "z_sample": arrays["z"],
        "direction_cosine": _normalized(arrays["direction"]),
        "z_cosine": _normalized(arrays["z"]),
    }
    if "p" in arrays:
        features["direction_and_p"] = np.concatenate(
            [arrays["direction"], arrays["p"][:, None]], axis=1
        )
    half = len(labels) // 2
    return {
        name: _knn_accuracy(torch.as_tensor(value), labels, half)
        for name, value in features.items()
    }


def _normalized(features: np.ndarray) -> np.ndarray:
    """Scales each row to unit length, so k-NN distance is cosine distance."""
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def _knn_accuracy(features: torch.Tensor, labels: torch.Tensor, split: int) -> float:
    train, test = features[:split], features[split:]
    neighbours = min(_PROBE_NEIGHBOURS, split)
    nearest = torch.cdist(test, train).topk(neighbours, largest=False).indices
    votes = torch.mode(labels[:split][nearest], dim=1).values
    return (votes == labels[split:]).float().mean().item()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
