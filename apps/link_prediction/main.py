"""Link prediction on citation and co-authorship graphs.

Supports Planetoid citation graphs (S-VAE paper, Table 4) and SNAP community graphs
(co-authorship networks like com-DBLP with documented overlapping communities).

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
import scipy.sparse as sp
import torch

from tnbbeta_vae.data.planetoid import Graph as PlanetoidGraph
from tnbbeta_vae.data.planetoid import (
    LinkSplit,
    load_planetoid,
    normalized_adjacency,
    split_edges,
)
from tnbbeta_vae.data.snap_community import Graph as SnapGraph
from tnbbeta_vae.data.snap_community import load_snap_community
from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.losses.ranking import average_precision, roc_auc
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import select_device

_DEFAULT_EPOCHS = {"cora": 200, "citeseer": 200, "pubmed": 400, "dblp": 50}


def main(argv: list[str] | None = None) -> None:
    """Runs the grid search for one dataset and latent family.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(_DEFAULT_EPOCHS))
    parser.add_argument(
        "--family",
        required=True,
        choices=["gaussian", "vmf", "tnbbeta", "power_spherical"],
    )
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.01, 0.005, 0.001])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--fixed-temperature", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="If set, saves each seed's best-validation-epoch model as "
        "$TNBBETA_CHECKPOINT_DIR/<run-name>_seed<seed>/final.pt, loadable by "
        "apps.eval scripts (e.g. dblp_bridge_diagnostic.py --run-name "
        "<run-name>_seed<seed>). With more than one (lr, dropout, latent_dim) "
        "configuration, later configurations overwrite earlier ones' checkpoints "
        "for the same seed -- intended for a single-configuration call.",
    )
    args = parser.parse_args(argv)

    device = select_device(args.device)
    epochs = args.epochs or _DEFAULT_EPOCHS[args.dataset]
    num_configs = len(args.lrs) * len(args.dropouts) * len(args.latent_dims)
    if args.run_name is not None and num_configs > 1:
        print(
            f"Warning: --run-name with {num_configs} configurations -- each "
            "seed's checkpoint will be overwritten by the last configuration "
            "processed. Use a single (lr, dropout, latent-dim) per call if you "
            "need every configuration's checkpoint.",
            file=sys.stderr,
        )
    # Load from Planetoid or SNAP community dataset.
    if args.dataset == "dblp":
        graph = load_snap_community(data_dir() / "snap_community", "dblp")
    else:
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
                run_name=args.run_name,
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
    graph: PlanetoidGraph | SnapGraph,
    split: LinkSplit,
    config: GraphVAEConfig,
    *,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
    run_name: str | None = None,
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
        run_name: If given, saves the best-validation-epoch model's weights to
            ``$TNBBETA_CHECKPOINT_DIR/<run_name>_seed<seed>/final.pt``, in the same
            format :class:`~tnbbeta_vae.training.Trainer` writes (so
            :func:`~tnbbeta_vae.training.load_model_checkpoint` and the ``apps.eval``
            scripts can read it). Not saved if ``None`` (the default, preserving this
            function's original metrics-only behavior).

    Returns:
        ``val_auc``, ``val_ap``, ``test_auc``, ``test_ap`` and ``best_epoch``, plus
        ``diverged`` (1.0 if a non-finite loss or gradient stopped training early; the
        metrics are then those of the best epoch before it, or chance level if none).
    """
    torch.manual_seed(seed)
    # Sparse upper-triangle extraction, not .toarray() -- densifying the whole
    # adjacency (317,080 x 317,080 for com-DBLP) would try to allocate ~375GiB.
    upper_triangle: Any = sp.triu(split.train_adjacency, k=1).tocoo()
    upper = np.stack([upper_triangle.row, upper_triangle.col])
    # Convert features: if scipy.sparse (Planetoid), convert to dense torch.Tensor;
    # if already torch.sparse (SNAP), keep as-is (GraphVAE handles sparse features).
    if hasattr(graph.features, "tocoo"):
        # scipy.sparse matrix (Planetoid): convert to dense for backward compatibility.
        features_tensor = torch.as_tensor(graph.features.toarray(), dtype=torch.float32)
    elif isinstance(graph.features, torch.Tensor):
        # Already a torch tensor (torch.sparse or torch.dense).
        features_tensor = graph.features.to(dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported features type: {type(graph.features)}")
    batch = GraphBatch(
        features=features_tensor,
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
    best_state_dict: dict[str, torch.Tensor] | None = None
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
            if run_name is not None:
                best_state_dict = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

    if run_name is not None and best_state_dict is not None:
        _save_checkpoint(run_name, seed, config, best_state_dict)

    return {**best, "diverged": diverged}


def _save_checkpoint(
    run_name: str,
    seed: int,
    config: GraphVAEConfig,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Writes a best-validation-epoch checkpoint in :class:`Trainer`'s format.

    Args:
        run_name: Base run name; the checkpoint is written under
            ``<run_name>_seed<seed>``.
        seed: This run's seed, appended to ``run_name`` for the directory.
        config: The trained model's config.
        state_dict: The model's weights at its best validation epoch.
    """
    run_dir = checkpoint_dir() / f"{run_name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "graph_vae",
            "config": config.model_dump(),
            "model_state_dict": state_dict,
        },
        run_dir / "final.pt",
    )


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
