"""Frozen-weights transfer test: does a trained checkpoint's posterior shape track
the graph it was TRAINED on, or the graph it's FED at eval time?

Usage:
    uv run python -m apps.eval.sbm_weight_transfer --run-name NAME \
        [--checkpoint final] [--device cpu] \
        [--num-communities 10] [--nodes-per-community 50] [--p-in 0.3] \
        --p-out P_OUT [--graph-seed 0] [--feature-noise-std NONE]

``writeup/results/sbm_pout_dose_response_2026-10-09.md`` found TNBBeta's aggregate
``m_mean`` grows steadily more negative (more bimodal) as a whole graph's ``p_out``
(hence average neighbor-community heterogeneity) rises, while the *per-node*
version of that story does not hold up
(``writeup/results/neighbor_cluster_visualization_2026-10-09.md`` and this
investigation's degree-correlation check). The live reframing this script tests:
``epsilon``/``m`` may be a *global regime* baked into a checkpoint's trained
weights, rather than something read adaptively from each forward pass's specific
input graph.

Given a trained checkpoint and an arbitrary stochastic-block-model graph
specification (reusing
``tnbbeta_vae.data.stochastic_block_model.stochastic_block_model`` directly, not a
new generator), this script builds the batch via
``apps.eval.graph_posterior_shape.graph_to_batch``, runs ONE frozen forward pass
(``model.posterior_and_prior(batch)``, no gradient updates, no training), and
reports ``tnbbeta_vae.models.posterior_stats.posterior_stats`` over every node --
exactly as ``apps/eval/graph_posterior_shape.py`` already does, no
reimplementation.

Running this once with a checkpoint against its OWN training graph (a sanity-check
baseline) and once against the OTHER condition's graph (the actual transfer test),
for each of two checkpoints trained at different ``p_out``, gives the four cells of
the 2x2 transfer grid this investigation needs: if a trained checkpoint's weights
bake in a global regime, its stats on the OTHER graph should stay close to its
stats on its OWN graph (barely shift despite the new graph's different
heterogeneity); if the model instead reads whatever graph it's given, its stats
should shift substantially toward what a model actually trained on that condition
would show. See the dated write-up in ``writeup/results/`` for the real numbers
and reading.

Writes ``weight_transfer_<checkpoint>_pout<p_out>_gseed<graph_seed>.json`` next to
the checkpoint -- the graph spec is encoded in the filename (not just the
checkpoint name), since this script is meant to be run against the SAME
checkpoint with multiple different graph specs.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import torch

from apps.eval.graph_posterior_shape import graph_to_batch
from tnbbeta_vae.data.stochastic_block_model import stochastic_block_model
from tnbbeta_vae.models.posterior_stats import posterior_stats
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import load_model_checkpoint

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Runs one frozen forward pass and writes the posterior-shape summary.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-communities", type=int, default=10)
    parser.add_argument("--nodes-per-community", type=int, default=50)
    parser.add_argument("--p-in", type=float, default=0.3)
    parser.add_argument("--p-out", type=float, required=True)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--feature-noise-std", type=float, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)

    graph = stochastic_block_model(
        num_communities=args.num_communities,
        nodes_per_community=args.nodes_per_community,
        p_in=args.p_in,
        p_out=args.p_out,
        seed=args.graph_seed,
        feature_noise_std=args.feature_noise_std,
    )
    batch = graph_to_batch(graph, device)

    model.eval()
    with torch.no_grad():
        posterior, _ = model.posterior_and_prior(batch)  # type: ignore[attr-defined]

    all_nodes = torch.ones(batch.num_nodes, dtype=torch.bool)
    stats = posterior_stats(posterior, all_nodes, checkpoint["config"])

    results: dict[str, Any] = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "model_name": checkpoint["model_name"],
        "graph_p_out": args.p_out,
        "graph_p_in": args.p_in,
        "graph_seed": args.graph_seed,
        "num_nodes": batch.num_nodes,
        "posterior_stats": stats,
    }

    output = run_dir / (
        f"weight_transfer_{args.checkpoint}_pout{args.p_out}_gseed{args.graph_seed}.json"
    )
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main(sys.argv[1:])
