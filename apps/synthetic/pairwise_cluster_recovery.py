"""Does GraphVAE's own pairwise/BCE loss alone drive TNBBeta posterior bimodality?

Usage:
    uv run python -m apps.synthetic.pairwise_cluster_recovery [--out-dir DIR] \
        [--num-clusters 10] [--points-per-cluster 50] [--dim 20] \
        [--latent-dim 16] [--epochs 300] [--seed 0] \
        [--families gaussian vmf power_spherical tnbbeta]

This project's research has found TNBBetaSpherical's posterior goes genuinely
bimodal (m = epsilon - (latent_dim - 1) / 2 < 0) specifically on GraphVAE
link-prediction runs (Cora, Citeseer, Pubmed, com-DBLP), and specifically does
NOT on every reconstruction-style VAE task tried (MNIST, axial_mixture, DTD).
The live hypothesis under test here: it's GraphVAE's training objective
itself (pure pairwise dot-product + binary_cross_entropy_with_logits on
pairs, with NO decoder/reconstruction term at all) that drives this, not
graph structure, scale, or features specifically.

This script is the "(b)" ablation arm: it holds the architecture family
(:class:`~tnbbeta_vae.models.pairwise_mlp_vae.PairwiseMlpVAE`'s plain MLP
encoder, no message-passing/no graph at all) and data
(:mod:`tnbbeta_vae.data.cluster_mixture`'s synthetic clustered items, no real
features beyond what correlates with cluster identity) fixed, and trains on
ONLY :class:`~tnbbeta_vae.models.graph_vae.GraphVAE`'s exact pairwise
dot-product + BCE objective, with no decoder at all. If m < 0 shows up here
too, that isolates "the pairwise/ranking loss itself" as the driver,
independent of the GCN architecture or any real graph.

Mirrors ``apps/link_prediction/main.py``'s ``run_once`` manual full-batch
training loop (one full-batch step per epoch, not minibatches -- the
pairwise loss needs cross-item access every step, not a random subset) and
``apps/synthetic/axial_recovery.py``'s CLI/output conventions where they
apply. Writes ``pairwise_cluster_recovery.json`` into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

import torch

from tnbbeta_vae.data.cluster_mixture import cluster_mixture_data, positive_pairs
from tnbbeta_vae.models.losses.ranking import average_precision, roc_auc
from tnbbeta_vae.models.pairwise_mlp_vae import (
    PairwiseBatch,
    PairwiseMlpVAE,
    PairwiseMlpVAEConfig,
)
from tnbbeta_vae.models.posterior_stats import posterior_stats

_FAMILIES = ["gaussian", "vmf", "power_spherical", "tnbbeta"]


def main(argv: list[str] | None = None) -> None:
    """Runs the experiment and writes the JSON output.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("pairwise_cluster_recovery")
    )
    parser.add_argument("--num-clusters", type=int, default=10)
    parser.add_argument("--points-per-cluster", type=int, default=50)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--cluster-std", type=float, default=0.5)
    parser.add_argument("--center-scale", type=float, default=5.0)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help=(
            "Ambient latent dimension for every family. Default 16 matches the"
            " Planetoid/com-DBLP GraphVAE runs exactly, for direct m/epsilon"
            " comparability across this whole investigation."
        ),
    )
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=2000,
        help="Size of the positive-pair pool, split into train/test.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--families", nargs="+", default=list(_FAMILIES), choices=list(_FAMILIES)
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    features, labels = cluster_mixture_data(
        num_clusters=args.num_clusters,
        points_per_cluster=args.points_per_cluster,
        dim=args.dim,
        cluster_std=args.cluster_std,
        center_scale=args.center_scale,
        generator=generator,
    )
    num_items = features.shape[0]

    # Hold out a fraction of the positive-pair pool for testing (mirroring
    # tnbbeta_vae.data.planetoid.split_edges's spirit), with equally many
    # sampled negatives kept separate from training's freshly-resampled
    # negatives. Pairs are drawn with replacement, so a vanishingly small
    # number of duplicates could in principle land in both splits -- the
    # same small accepted noise as this project's unfiltered negative
    # sampling convention (GraphVAE._sample_negatives), not filtered out.
    pool = positive_pairs(labels, args.num_pairs, generator=generator)
    permutation = torch.randperm(args.num_pairs, generator=generator)
    num_test = max(1, int(args.num_pairs * args.test_fraction))
    test_positive = pool[:, permutation[:num_test]]
    train_positive = pool[:, permutation[num_test:]]
    test_negative = torch.randint(num_items, (2, num_test), generator=generator)

    results: dict[str, Any] = {}
    for family in args.families:
        torch.manual_seed(args.seed)
        model = PairwiseMlpVAE(
            PairwiseMlpVAEConfig(
                family=cast("Any", family),
                input_dim=args.dim,
                hidden_dims=args.hidden_dims,
                latent_dim=args.latent_dim,
            )
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        batch = PairwiseBatch(features=features, positive_pairs=train_positive)

        losses: list[float] = []
        diverged = False
        for _ in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = model.training_step(batch)["loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
                diverged = True
                break
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        embeddings = model.embeddings(batch)
        test_auc, test_ap = score_pairs(embeddings, test_positive, test_negative)
        posterior, _ = model.posterior_and_prior(batch)
        stats = posterior_stats(
            posterior,
            torch.ones(num_items, dtype=torch.bool),
            model.config.model_dump(),
        )
        results[family] = {
            "test_auc": test_auc,
            "test_ap": test_ap,
            "final_loss": losses[-1] if losses else float("nan"),
            "diverged": diverged,
            **stats,
        }
        print(family, json.dumps(results[family]))

    output = args.out_dir / "pairwise_cluster_recovery.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {output}")


def score_pairs(
    embeddings: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
) -> tuple[float, float]:
    """Returns ``(auc, average_precision)`` of ranking pairs by embedding inner product.

    Same scoring convention as ``apps/link_prediction/main.py``'s ``score_edges``.

    Args:
        embeddings: Item embeddings, shape ``(num_items, d)``.
        positive: Held-out positive pairs, shape ``(2, n)``.
        negative: Sampled negative pairs, shape ``(2, m)``.

    Returns:
        The ROC AUC and average precision of the ranking.
    """
    values = embeddings.detach().numpy()
    positive_scores = (values[positive[0]] * values[positive[1]]).sum(-1)
    negative_scores = (values[negative[0]] * values[negative[1]]).sum(-1)
    return roc_auc(positive_scores, negative_scores), average_precision(
        positive_scores, negative_scores
    )


if __name__ == "__main__":
    main(sys.argv[1:])
