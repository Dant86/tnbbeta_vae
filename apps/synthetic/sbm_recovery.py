"""Does GraphVAE's real GCN + a synthetic community-structured graph drive bimodality?

Usage:
    uv run python -m apps.synthetic.sbm_recovery [--out-dir DIR] \
        [--num-communities 10] [--nodes-per-community 50] \
        [--p-in 0.3] [--p-out 0.01] [--feature-noise-std NONE] \
        [--latent-dim 16] [--epochs 200] \
        [--graph-seed 0] [--seed 0] [--run-name sbm_recovery] \
        [--families gaussian vmf power_spherical tnbbeta]

This project's research has found ``TNBBetaSpherical``'s posterior goes genuinely
bimodal (``m = epsilon - (latent_dim - 1) / 2 < 0``) specifically on real
``GraphVAE`` link-prediction runs (Cora, Citeseer, Pubmed, com-DBLP -- all four,
every one combining a GCN encoder with a real graph), and specifically does NOT on
any reconstruction-style VAE task tried (MNIST, ``axial_mixture``, DTD -- none of
which have a GCN or a graph at all). Two prior ablations each independently ruled
out one candidate explanation while holding the other fixed:

- ``apps/synthetic/pairwise_cluster_recovery.py`` ("(b)"): ``GraphVAE``'s exact
  pairwise-dot-product+BCE loss, through a plain MLP encoder, no graph at all, on
  synthetic Gaussian-blob clusters. Result: stays unimodal -- rules out "the loss
  function alone" as sufficient.
- ``GraphVAE``'s feature-reconstruction-decoder ablation ("(a)"): added a real
  feature-reconstruction decoder onto the real GCN, on the real Cora graph. Result:
  ``m``/``epsilon``/``frac_bimodal`` barely moved -- rules out "absence of a
  reconstruction term" as sufficient on its own.

Both ablations left the GCN architecture and real-graph structure untested in
isolation: every bimodal result so far has both (GCN + a real graph); every
unimodal result so far has neither. This script is that isolation ("(c)"): it
routes (b)'s exact synthetic task (same item count and community count --
10 communities x 50 nodes = 500 by default, same lack of real features, same
no-decoder pairwise+BCE loss, via ``apps.link_prediction.main.run_once``) through
``GraphVAE``'s real, UNMODIFIED GCN architecture instead of a plain MLP, on a
synthetic stochastic-block-model graph (dense intra-community edges, sparse
inter-community edges) with identity features -- matching com-DBLP's featureless
setup exactly, at a much smaller, directly-comparable-to-(b) scale. If this ALSO
goes bimodal, that isolates "GCN message-passing + a graph with
community-correlated structure" as sufficient, independent of scale or anything
else specific to real citation/co-authorship networks.

``--no-aggregation`` is a further isolation on top of (c) ("aggregation OFF" vs.
(c)'s "aggregation ON"): it holds the exact same SBM graph, edges and loss fixed,
and removes only the GCN's neighbor-averaging, by passing
``apps.link_prediction.main.run_once``'s ``encoder_adjacency`` parameter a sparse
identity matrix. The encoder then sees each node's own (identity) feature row
with no neighbor mixing at all, while training still targets the real SBM edges
for the loss -- the sharpest remaining test of whether GCN aggregation itself
(not just a graph-correlated training signal in some looser sense) is what
(c) actually isolated.

Writes ``sbm_recovery.json`` into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

import scipy.sparse as sp
import torch

from apps.eval.graph_posterior_shape import graph_to_batch
from apps.link_prediction.main import run_once
from tnbbeta_vae.data.planetoid import split_edges
from tnbbeta_vae.data.stochastic_block_model import stochastic_block_model
from tnbbeta_vae.models import GraphVAEConfig
from tnbbeta_vae.models.posterior_stats import posterior_stats
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import load_model_checkpoint, select_device

_FAMILIES = ["gaussian", "vmf", "power_spherical", "tnbbeta"]


def main(argv: list[str] | None = None) -> None:
    """Runs the experiment for every requested family and writes the JSON output.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("sbm_recovery"))
    parser.add_argument(
        "--num-communities",
        type=int,
        default=10,
        help="Matches pairwise_cluster_recovery's (b) ablation's cluster count.",
    )
    parser.add_argument(
        "--nodes-per-community",
        type=int,
        default=50,
        help="Matches pairwise_cluster_recovery's (b) ablation's points-per-cluster.",
    )
    parser.add_argument("--p-in", type=float, default=0.3)
    parser.add_argument("--p-out", type=float, default=0.01)
    parser.add_argument(
        "--feature-noise-std",
        type=float,
        default=None,
        help=(
            "If set, overrides the default identity features with a one-hot"
            " community vector plus N(0, feature_noise_std**2) i.i.d. noise (see"
            " tnbbeta_vae.data.stochastic_block_model.stochastic_block_model's"
            " docstring) -- a tunable interpolation from 'noise completely swamps"
            " the signal' (large values) to 'exact community revealed' (near 0)."
            " Omitting this flag (the default, None) preserves today's identity-"
            " feature behavior exactly."
        ),
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help=(
            "Ambient latent dimension for every family. Default 16 matches every"
            " other comparison point in this investigation (Planetoid/com-DBLP"
            " GraphVAE runs, and pairwise_cluster_recovery's (b) ablation) exactly,"
            " for direct m/epsilon comparability."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--graph-seed",
        type=int,
        default=0,
        help=(
            "Seed for the SBM graph's edge sampling and its val/test edge split --"
            " generated once and shared across every --seed in a stability sweep, so"
            " a seed sweep varies only model init/negative sampling (via run_once's"
            " own seed), not the task (the graph and split) itself."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Model init/negative-sampling seed, passed straight to run_once.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="sbm_recovery",
        help=(
            "Base checkpoint name; each family's run is saved as"
            " $TNBBETA_CHECKPOINT_DIR/<run-name>_<family>_seed<seed>/final.pt (the"
            " '_seed<seed>' suffix is apps.link_prediction.main.run_once's own"
            " convention)."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--families", nargs="+", default=list(_FAMILIES), choices=list(_FAMILIES)
    )
    parser.add_argument(
        "--no-aggregation",
        action="store_true",
        help=(
            "If set, the GCN encoder aggregates over a sparse identity matrix"
            " (via run_once's encoder_adjacency) instead of the real graph's"
            " normalized adjacency -- each node's posterior then depends only on"
            " its own (identity) feature row, with no neighbor mixing at all. The"
            " loss is unaffected: training still targets the real SBM edges from"
            " `split`, exactly as without this flag. Everything else (graph scale,"
            " latent_dim, families, seeds) is identical to the default"
            " (aggregation-ON) run, so the two are directly comparable."
        ),
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    graph = stochastic_block_model(
        num_communities=args.num_communities,
        nodes_per_community=args.nodes_per_community,
        p_in=args.p_in,
        p_out=args.p_out,
        seed=args.graph_seed,
        feature_noise_std=args.feature_noise_std,
    )
    split = split_edges(
        graph.adjacency,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.graph_seed,
    )
    encoder_adjacency = (
        sp.identity(graph.adjacency.shape[0], format="csr", dtype="float32")
        if args.no_aggregation
        else None
    )

    results: dict[str, Any] = {}
    for family in args.families:
        run_name = f"{args.run_name}_{family}"
        config = GraphVAEConfig(
            family=cast("Any", family),
            in_features=graph.features.shape[1],
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
        )
        metrics = run_once(
            graph,
            split,
            config,
            lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            device=device,
            run_name=run_name,
            encoder_adjacency=encoder_adjacency,
        )
        stats = _posterior_stats_for_checkpoint(run_name, args.seed, graph, device)
        results[family] = {**metrics, **stats}
        print(family, json.dumps(results[family]))

    output = args.out_dir / "sbm_recovery.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {output}")


def _posterior_stats_for_checkpoint(
    run_name: str, seed: int, graph: Any, device: torch.device
) -> dict[str, float]:
    """Loads a ``run_once``-saved checkpoint and computes its whole-graph posterior
    stats.

    Args:
        run_name: The base run name passed to ``run_once`` (without ``_seed<seed>``).
        seed: The seed ``run_once`` was called with.
        graph: The same graph ``run_once`` was trained on.
        device: Device to load the checkpoint and run the forward pass on.

    Returns:
        ``tnbbeta_vae.models.posterior_stats.posterior_stats``'s output, or a single
        ``{"no_checkpoint": 1.0}`` if training diverged before any checkpoint was
        ever saved (``run_once`` only writes one once validation AUC improves past
        its chance-level default).
    """
    checkpoint_path = checkpoint_dir() / f"{run_name}_seed{seed}" / "final.pt"
    if not checkpoint_path.exists():
        return {"no_checkpoint": 1.0}

    model, checkpoint = load_model_checkpoint(checkpoint_path, device)
    batch = graph_to_batch(graph, device)
    model.eval()
    with torch.no_grad():
        posterior, _ = model.posterior_and_prior(batch)  # type: ignore[attr-defined]
    all_nodes = torch.ones(batch.num_nodes, dtype=torch.bool)
    return posterior_stats(posterior, all_nodes, checkpoint["config"])


if __name__ == "__main__":
    main(sys.argv[1:])
