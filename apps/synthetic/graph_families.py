"""Link prediction on synthetic graph families with known edge probabilities.

Usage:
    uv run python -m apps.synthetic.graph_families --graph hub_spoke --family vmf \
        [--fixed-kappa K | --fixed-m M] [--latent-dim 16] [--epochs 500] \
        [--seeds 0 1 2 3 4] --results-name NAME

Which graph structures make a latent family's expressivity matter? For each seed this
samples a graph from a named preset (:data:`PRESETS`), holds out edges as usual, trains
a featureless ``GraphVAE`` and scores it three ways:

1. centre AUC (the standard link-prediction metric, blind to posterior shape);
2. held-out log loss and calibration of the *posterior-predictive* edge probability
   (mean of ``sigmoid(logit)`` over posterior samples) versus the plug-in centre
   probability;
3. divergence of the predicted pair probabilities from the *true* ones, over every pair
   that is not a training edge, overall and per pair kind (e.g. ``leaf-leaf-same-hub``).
   The truth is mapped through ``balanced_truth``, since 1:1 negative sampling trains
   the model to output ``P / (P + rho)``, not ``P``.

Writes ``$TNBBETA_CHECKPOINT_DIR/synthetic_graphs/<results-name>.json``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
import torch

from apps.link_prediction.main import run_once
from tnbbeta_vae.data.block_graph import (
    BlockGraph,
    assortative_spec,
    bipartite_spec,
    block_graph,
    hub_spoke_spec,
)
from tnbbeta_vae.data.planetoid import (
    Graph,
    LinkSplit,
    normalized_adjacency,
    split_edges,
)
from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.heads import epsilon_from_m
from tnbbeta_vae.models.losses.ranking import roc_auc
from tnbbeta_vae.models.predictive import (
    balanced_truth,
    bernoulli_kl,
    bernoulli_log_loss,
    dense_edge_probabilities,
    expected_calibration_error,
)
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import select_device

PRESETS: dict[str, Callable[[int], BlockGraph]] = {
    "assortative": lambda seed: block_graph(assortative_spec(), seed=seed),
    "assortative_dc": lambda seed: block_graph(
        assortative_spec(p_in=0.1), seed=seed, degree_exponent=2.5
    ),
    "bipartite": lambda seed: block_graph(bipartite_spec(), seed=seed),
    "hub_spoke": lambda seed: block_graph(hub_spoke_spec(), seed=seed),
}


def main(argv: list[str] | None = None) -> None:
    """Trains and evaluates one condition on one preset for each seed.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=sorted(PRESETS), required=True)
    parser.add_argument(
        "--family",
        choices=["gaussian", "vmf", "power_spherical", "tnbbeta"],
        required=True,
    )
    parser.add_argument("--fixed-m", type=float, default=None)
    parser.add_argument("--fixed-kappa", type=float, default=None)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--results-name", required=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = select_device(args.device)
    runs = []
    for seed in args.seeds:
        sample = PRESETS[args.graph](seed)
        config = GraphVAEConfig(
            family=cast("Any", args.family),
            in_features=sample.probability.shape[0],
            latent_dim=args.latent_dim,
            dropout=args.dropout,
            fixed_epsilon=(
                None
                if args.fixed_m is None
                else epsilon_from_m(args.fixed_m, args.latent_dim)
            ),
            fixed_kappa=args.fixed_kappa,
        )
        runs.append(
            {
                "seed": seed,
                **train_and_evaluate(
                    sample,
                    config,
                    lr=args.lr,
                    epochs=args.epochs,
                    seed=seed,
                    device=device,
                    num_samples=args.samples,
                ),
            }
        )
        print(
            json.dumps({k: v for k, v in runs[-1].items() if not isinstance(v, dict)})
        )
    output = checkpoint_dir() / "synthetic_graphs" / f"{args.results_name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"graph": args.graph, "args": vars(args), "runs": runs}, indent=2)
    )
    print(f"Wrote {output}")


def train_and_evaluate(
    sample: BlockGraph,
    config: GraphVAEConfig,
    *,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
    num_samples: int,
) -> dict[str, Any]:
    """Trains a featureless ``GraphVAE`` on ``sample`` and scores its best epoch.

    Args:
        sample: The sampled graph and its true pair probabilities.
        config: Model hyperparameters (``in_features`` must equal the node count).
        lr: Adam learning rate.
        epochs: Training steps.
        seed: Seed for the edge split and the model.
        device: Device to train on.
        num_samples: Posterior samples for the posterior-predictive metrics.

    Returns:
        See :func:`evaluate_graph_model`, plus ``best_epoch``.
    """
    num_nodes = sample.probability.shape[0]
    graph = Graph(sample.adjacency, sp.identity(num_nodes, format="csr"))
    split = split_edges(sample.adjacency, seed=seed)
    state: dict[str, torch.Tensor] = {}
    metrics = run_once(
        graph,
        split,
        config,
        lr=lr,
        epochs=epochs,
        seed=seed,
        device=device,
        best_state=state,
    )
    model = GraphVAE(config).to(device)
    model.load_state_dict(state)
    result = evaluate_graph_model(model, sample, split, num_samples=num_samples)
    return {**result, "best_epoch": metrics["best_epoch"]}


def evaluate_graph_model(
    model: GraphVAE, sample: BlockGraph, split: LinkSplit, *, num_samples: int
) -> dict[str, Any]:
    """Scores a trained model against held-out edges and the true pair probabilities.

    Args:
        model: A trained model.
        sample: The graph and its truth.
        split: The edge split the model was trained on.
        num_samples: Posterior samples for the posterior-predictive probabilities.

    Returns:
        ``test_auc`` (centres), ``oracle_test_auc`` (the true probabilities, an upper
        reference), ``test_log_loss`` / ``test_ece`` and ``truth_kl`` for each of
        ``centre`` and ``sampled`` predictions, and ``truth_kl_by_kind`` (the same KL
        per pair kind, over non-training pairs).
    """
    device = next(model.parameters()).device
    batch = GraphBatch(
        features=torch.eye(sample.probability.shape[0]),
        norm_adjacency=normalized_adjacency(split.train_adjacency),
        positive_edges=torch.as_tensor(
            np.stack(sp.triu(split.train_adjacency, k=1).nonzero()), dtype=torch.long
        ),
    ).to(device)
    truth = balanced_truth(torch.as_tensor(sample.probability, dtype=torch.float32))
    upper = np.triu(np.ones_like(sample.probability, dtype=bool), k=1)
    held = upper & (split.train_adjacency.toarray() == 0)
    held_mask = torch.as_tensor(held)
    positive, negative = split.test_positive, split.test_negative
    pairs = np.concatenate([positive, negative], axis=1)
    targets = torch.cat([torch.ones(positive.shape[1]), torch.zeros(negative.shape[1])])

    result: dict[str, Any] = {
        "oracle_test_auc": roc_auc(
            truth[positive[0], positive[1]].numpy(),
            truth[negative[0], negative[1]].numpy(),
        )
    }
    for name, count in (("centre", None), ("sampled", num_samples)):
        probability = dense_edge_probabilities(model, batch, num_samples=count)
        test_probability = probability[pairs[0], pairs[1]]
        result[f"{name}_test_auc"] = roc_auc(
            test_probability[: positive.shape[1]].numpy(),
            test_probability[positive.shape[1] :].numpy(),
        )
        result[f"{name}_test_log_loss"] = float(
            bernoulli_log_loss(test_probability, targets)
        )
        result[f"{name}_test_ece"] = float(
            expected_calibration_error(test_probability, targets)
        )
        result[f"{name}_truth_kl"] = float(
            bernoulli_kl(truth[held_mask], probability[held_mask])
        )
        result[f"{name}_truth_kl_by_kind"] = {
            kind: float(
                bernoulli_kl(
                    truth[held_mask & torch.as_tensor(sample.kind == kind)],
                    probability[held_mask & torch.as_tensor(sample.kind == kind)],
                )
            )
            for kind in sorted({str(k) for k in np.unique(sample.kind)})
            if (held & (sample.kind == kind)).any()
        }
    result["test_auc"] = result["centre_test_auc"]
    return result


if __name__ == "__main__":
    main()
