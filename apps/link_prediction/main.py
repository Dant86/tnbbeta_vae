"""Link prediction on citation graphs (S-VAE paper, Table 4).

Usage:
    uv run python -m apps.link_prediction.main --dataset cora --family tnbbeta \
        [--lrs 0.01 0.005 0.001] [--dropouts 0 0.2 0.4] [--latent-dims 16 32 64] \
        [--epochs 200] [--seeds 0 1 2 3 4]

Trains a GCN variational graph auto-encoder (:class:`GraphVAE`) with a Gaussian, vMF or
TNBBeta latent. Every configuration in the grid is run for each seed on a fresh 85/5/10
edge split; within a run the epoch with the best validation AUC is selected and its test
AUC and average precision are recorded. The configuration with the best mean validation
AUC is the reported one. Writes ``$TNBBETA_CHECKPOINT_DIR/link_prediction/
<dataset>_<family>.json`` with the selected configuration's test mean and standard
deviation and a summary of every configuration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from typing import Any, cast

import numpy as np
import torch

from tnbbeta_vae.data.planetoid import (
    Graph,
    LinkSplit,
    load_planetoid,
    normalized_adjacency,
    split_edges,
)
from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.losses.ranking import average_precision, roc_auc
from tnbbeta_vae.paths import checkpoint_dir, data_dir

_DEFAULT_EPOCHS = {"cora": 200, "citeseer": 200, "pubmed": 400}


def main(argv: list[str] | None = None) -> None:
    """Runs the grid search for one dataset and latent family.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(_DEFAULT_EPOCHS))
    parser.add_argument(
        "--family", required=True, choices=["gaussian", "vmf", "tnbbeta"]
    )
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.01, 0.005, 0.001])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--fixed-temperature", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    epochs = args.epochs or _DEFAULT_EPOCHS[args.dataset]
    graph = load_planetoid(data_dir() / "planetoid", args.dataset)
    splits = {seed: split_edges(graph.adjacency, seed=seed) for seed in args.seeds}

    summaries: list[dict[str, Any]] = []
    for lr, dropout, latent_dim in itertools.product(
        args.lrs, args.dropouts, args.latent_dims
    ):
        runs = [
            run_once(
                graph,
                splits[seed],
                GraphVAEConfig(
                    family=cast("Any", args.family),
                    in_features=graph.features.shape[1],
                    latent_dim=latent_dim,
                    dropout=dropout,
                    fixed_temperature=args.fixed_temperature,
                ),
                lr=lr,
                epochs=epochs,
                seed=seed,
                device=device,
            )
            for seed in args.seeds
        ]
        summary = {
            "lr": lr,
            "dropout": dropout,
            "latent_dim": latent_dim,
            **summarize(runs),
        }
        summaries.append(summary)
        print(json.dumps(summary))

    selected = max(summaries, key=lambda summary: summary["val_auc"])
    result = {
        "dataset": args.dataset,
        "family": args.family,
        "epochs": epochs,
        "seeds": args.seeds,
        "selected": selected,
        "configurations": summaries,
    }
    output = checkpoint_dir() / "link_prediction" / f"{args.dataset}_{args.family}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Selected {json.dumps(selected)}\nWrote {output}")


def run_once(
    graph: Graph,
    split: LinkSplit,
    config: GraphVAEConfig,
    *,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """Trains one model and returns the metrics at its best-validation epoch.

    Args:
        graph: The full graph (features are used; its edges only through ``split``).
        split: Train/validation/test edges.
        config: Model hyperparameters.
        lr: Adam learning rate.
        epochs: Number of full-graph training steps.
        seed: Seed for the model initialization and negative sampling.
        device: Device to train on.

    Returns:
        ``val_auc``, ``val_ap``, ``test_auc``, ``test_ap`` and ``best_epoch``, plus
        ``diverged`` (1.0 if a non-finite loss or gradient stopped training early; the
        metrics are then those of the best epoch before it, or chance level if none).
    """
    torch.manual_seed(seed)
    upper = np.stack(np.nonzero(np.triu(split.train_adjacency.toarray(), k=1)))
    batch = GraphBatch(
        features=torch.as_tensor(graph.features.toarray(), dtype=torch.float32),
        norm_adjacency=normalized_adjacency(split.train_adjacency),
        positive_edges=torch.as_tensor(upper, dtype=torch.long),
    ).to(device)
    model = GraphVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best: dict[str, float] = {
        "val_auc": 0.5,
        "val_ap": 0.5,
        "test_auc": 0.5,
        "test_ap": 0.5,
        "best_epoch": -1.0,
    }
    best_val_auc = -1.0
    diverged = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.training_step(batch)["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
            diverged = 1.0
            break
        optimizer.step()

        model.eval()
        embeddings = model.embeddings(batch).cpu()
        val_auc, val_ap = score_edges(
            embeddings, split.val_positive, split.val_negative
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc, test_ap = score_edges(
                embeddings, split.test_positive, split.test_negative
            )
            best = {
                "val_auc": val_auc,
                "val_ap": val_ap,
                "test_auc": test_auc,
                "test_ap": test_ap,
                "best_epoch": float(epoch),
            }
    return {**best, "diverged": diverged}


def score_edges(
    embeddings: torch.Tensor, positive: np.ndarray, negative: np.ndarray
) -> tuple[float, float]:
    """Returns ``(auc, average_precision)`` of ranking edges by embedding inner product.

    Args:
        embeddings: Node embeddings, shape ``(num_nodes, d)``.
        positive: Held-out edges, shape ``(2, n)``.
        negative: Non-edges, shape ``(2, m)``.

    Returns:
        The ROC AUC and average precision of the ranking.
    """
    values = embeddings.numpy()
    positive_scores = (values[positive[0]] * values[positive[1]]).sum(-1)
    negative_scores = (values[negative[0]] * values[negative[1]]).sum(-1)
    return roc_auc(positive_scores, negative_scores), average_precision(
        positive_scores, negative_scores
    )


def summarize(runs: list[dict[str, float]]) -> dict[str, float]:
    """Means over seeds of every metric, the test metrics' deviations and the number of
    runs that diverged."""
    summary = {
        key: float(np.mean([run[key] for run in runs]))
        for key in ("val_auc", "val_ap", "test_auc", "test_ap")
    }
    for key in ("test_auc", "test_ap"):
        values = [run[key] for run in runs]
        summary[f"{key}_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
    summary["num_diverged"] = float(sum(run.get("diverged", 0.0) for run in runs))
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
