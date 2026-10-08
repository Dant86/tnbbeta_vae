# GCN + synthetic community graph ablation: does that pair alone drive bimodality? 2026-10-08

- **Date:** 2026-10-08
- **Branch/commit:** `dtd-oriented-textures`, `55a6ce4` ("sbm_recovery: decouple
  the SBM graph's seed from the per-run training seed") -- `master` did not
  yet have this code at the time of this run.
- **Commands (main 4-family table + 3-seed TNBBeta stability table):**
  ```
  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/sbm_recovery_final --epochs 2000 \
      --graph-seed 0 --seed 0 --device cpu --run-name sbm_recovery

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/sbm_recovery_final_seed1 --epochs 2000 \
      --graph-seed 0 --seed 1 --device cpu --run-name sbm_recovery

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/sbm_recovery_final_seed2 --epochs 2000 \
      --graph-seed 0 --seed 2 --device cpu --run-name sbm_recovery
  ```
  (All three share `--graph-seed 0`, i.e. the identical SBM graph and edge
  split; only `--seed`, which `run_once` uses for model init and negative
  sampling, varies.)
- **Command (epsilon-drift-over-training check, ad hoc, not a committed
  script -- see "Methodology note" below for why and its full source):**
  ```
  PYTHONPATH=. uv run python /tmp/sbm_epsilon_drift_check.py \
      --seeds 0 1 2 --max-epochs 2000 --every 100
  ```

## Background

This project's research has found `TNBBetaSpherical`'s posterior goes
genuinely bimodal (`m = epsilon - (latent_dim - 1) / 2 < 0`) specifically on
real `GraphVAE` link-prediction runs (Cora, Citeseer, Pubmed, com-DBLP --
all four, every one combining a GCN encoder with a real graph), and
specifically NOT on any reconstruction-style VAE task tried (MNIST,
`axial_mixture`, DTD -- none of which has a GCN or a graph at all). Two
prior ablations each ruled out one candidate explanation while holding the
other fixed:

- **(b)** (`pairwise_mlp_cluster_recovery_2026-10-08.md`): `GraphVAE`'s
  exact pairwise-dot-product+BCE loss, through a plain MLP encoder, no
  graph at all, on synthetic Gaussian-blob clusters (10 clusters x 50
  points). Result: stays unimodal (`m_mean` strongly positive,
  `frac_bimodal = 0.0`), and if anything drifts *further* unimodal with more
  training -- rules out "the loss function alone" as sufficient.
- **(a)** (`graph_vae_feature_reconstruction_ablation_2026-10-08.md`): added
  a real feature-reconstruction decoder onto `GraphVAE`'s real GCN, on the
  real Cora graph. Result: `m`/`epsilon`/`frac_bimodal` barely moved from
  the no-decoder baseline (`m_mean = -2.21` vs. `-2.27`) -- rules out
  "absence of a reconstruction term" as sufficient on its own.

Both ablations left the GCN architecture and real-graph structure untested
in isolation: every bimodal result so far has both (GCN + a real graph);
every unimodal result so far has neither. This is that isolation ("(c)"):
(b)'s exact synthetic task (10 communities x 50 nodes = 500 items, no real
features, no decoder, `GraphVAE`'s pairwise+BCE loss via
`apps.link_prediction.main.run_once`, unchanged) routed through
`GraphVAE`'s real, **unmodified** GCN instead of a plain MLP, on a synthetic
stochastic-block-model (SBM) graph -- dense intra-community edges, sparse
inter-community edges, identity features matching com-DBLP's featureless
setup exactly, at a much smaller, directly-comparable-to-(b) scale.

## The generated graph: sanity check

Default scale (`num_communities=10`, `nodes_per_community=50`, `p_in=0.3`,
`p_out=0.01`, `graph_seed=0`):

| statistic | value |
|---|---|
| nodes | 500 |
| edges | 4758 |
| average degree | 19.03 (min 6, max 30) |
| isolated nodes | 0 |
| empirical within-community edge density | 0.2953 (requested `p_in=0.3`) |
| empirical across-community edge density | 0.0101 (requested `p_out=0.01`) |

The requested and empirical probabilities match within ordinary sampling
noise (also checked, more generally and across seeds, in
`tests/data/test_stochastic_block_model.py`), the graph has no isolated
nodes, and within-community pairs are connected roughly 29x more often
than across-community pairs -- a reasonably connected, non-degenerate graph
with real community-correlated structure, at (b)'s exact item/community
count for direct comparability.

## Methodology note: the epsilon-drift check is ad hoc, not a committed script

(b)'s write-up reported a full epsilon-vs-training-step curve (not just the
headline epoch's numbers) because that ablation showed `epsilon` still
drifting, unstably, for as long as training continued. The same honesty
standard applies here. Getting that curve for a `GraphVAE` run requires
snapshotting `posterior_stats` at many points *during* training, which
neither `apps.link_prediction.main.run_once` (only checkpoints the
best-validation-AUC epoch) nor `apps.synthetic.sbm_recovery` (calls
`run_once` once per family, by design, per the task's reuse requirement)
exposes. Rather than add instrumentation to the committed, already-tested
`run_once`/`sbm_recovery.py` pipeline for a one-off diagnostic, this check
used a short, uncommitted script (reusing `GraphVAE`, `posterior_stats`,
`stochastic_block_model`, `split_edges`, `normalized_adjacency` and
`apps.link_prediction.main.score_edges` -- no reimplementation of any of
their logic) that manually runs the same training loop and snapshots
`posterior_stats` every 100 epochs regardless of validation performance.
Its full source, for reproducibility:

```python
"""Ad hoc epsilon-drift check for apps.synthetic.sbm_recovery's TNBBeta family."""

from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.sparse as sp
import torch

from apps.link_prediction.main import score_edges
from tnbbeta_vae.data.planetoid import normalized_adjacency, split_edges
from tnbbeta_vae.data.stochastic_block_model import stochastic_block_model
from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.posterior_stats import posterior_stats

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--max-epochs", type=int, default=2000)
parser.add_argument("--every", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--latent-dim", type=int, default=16)
args = parser.parse_args()

graph = stochastic_block_model(
    num_communities=10, nodes_per_community=50, p_in=0.3, p_out=0.01, seed=0
)
results: dict[int, list[dict]] = {}
for seed in args.seeds:
    split = split_edges(graph.adjacency, seed=seed)
    upper = sp.triu(split.train_adjacency, k=1).tocoo()
    up = np.stack([upper.row, upper.col])
    batch = GraphBatch(
        features=graph.features.to(dtype=torch.float32),
        norm_adjacency=normalized_adjacency(split.train_adjacency),
        positive_edges=torch.as_tensor(up, dtype=torch.long),
    )
    torch.manual_seed(seed)
    config = GraphVAEConfig(
        family="tnbbeta", in_features=graph.features.shape[1], latent_dim=args.latent_dim
    )
    model = GraphVAE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    curve = []
    for epoch in range(args.max_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.training_step(batch)["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
            print(f"seed {seed}: diverged at epoch {epoch}")
            break
        optimizer.step()

        if epoch % args.every == 0 or epoch == args.max_epochs - 1:
            model.eval()
            with torch.no_grad():
                embeddings = model.embeddings(batch)
                val_auc, val_ap = score_edges(
                    embeddings, split.val_positive, split.val_negative
                )
                posterior, _ = model.posterior_and_prior(batch)
                all_nodes = torch.ones(batch.num_nodes, dtype=torch.bool)
                stats = posterior_stats(posterior, all_nodes, config.model_dump())
            curve.append(
                {
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "epsilon_mean": stats["epsilon_mean"],
                    "m_mean": stats["m_mean"],
                    "frac_bimodal": stats["frac_bimodal"],
                }
            )
            print(seed, json.dumps(curve[-1]))
    results[seed] = curve
```

One deliberate difference from the main 3-seed table below: this script
varies `split_edges(seed=seed)`'s edge split per seed (on top of model
init), while the committed `sbm_recovery.py`'s 3-seed table below holds
both the graph *and* the split fixed (`--graph-seed 0` throughout) and
varies only `--seed` (model init/negative sampling). That makes this
drift check, if anything, a *slightly* stronger robustness claim (it is
also robust to the specific held-out edges), not a weaker one -- but it is
a genuine methodological difference between the two tables below, flagged
here rather than left implicit.

## Result: the main 4-family table (`--graph-seed 0 --seed 0`, 2000 epochs)

| family | test_auc | test_ap | best_epoch | entropy_mean | r_bar |
|---|---|---|---|---|---|
| gaussian | 0.8423 | 0.8410 | 1755 | -4.588 | 1.586 |
| vmf | 0.8496 | 0.8554 | 1055 | -12.220 | 0.911 |
| power_spherical | 0.8500 | 0.8515 | 395 | -7.379 | 0.823 |
| tnbbeta | 0.8461 | 0.8481 | 259 | NaN (no closed form) | 0.918 |

TNBBeta-only diagnostic (seed 0): `p_mean = 0.0398`, `q_mean = 0.9189`,
`epsilon_mean = 2.953`, **`m_mean = -4.547`**, **`frac_bimodal = 1.0`**.

All four families solve this task reasonably well (test AUC ~0.84-0.85 --
visibly harder than (b)'s MLP/no-graph task at ~0.93-0.95, consistent with
a GCN on a sparser, smaller-degree graph than fully-separated Gaussian
blobs being a harder link-prediction problem) and no run diverged. Crucially,
**every node's TNBBeta posterior sits in the proven bimodal regime**
(`frac_bimodal = 1.0`) -- the opposite of ablation (b)'s `frac_bimodal = 0.0`
on the same item/community count, and qualitatively the same pattern as
every real `GraphVAE` checkpoint trained on an actual citation/co-authorship
graph.

## Result: TNBBeta stability across 3 seeds (same graph, varying only model init)

| seed | test_auc | test_ap | best_epoch | p_mean | q_mean | epsilon_mean | m_mean | frac_bimodal |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.8461 | 0.8481 | 259 | 0.0398 | 0.9189 | 2.953 | -4.547 | 1.0 |
| 1 | 0.8453 | 0.8519 | 1187 | 0.9666 | 0.8323 | 3.632 | -3.868 | 1.0 |
| 2 | 0.8510 | 0.8494 | 162 | 0.9585 | 0.9586 | 3.160 | -4.340 | 1.0 |

`frac_bimodal = 1.0` in every seed, and `m_mean` stays firmly negative
throughout (-3.87 to -4.55), with no sign flip anywhere in the range. (The
`p_mean` flip between seed 0 (~0.04) and seeds 1/2 (~0.96-0.97) is the
known `(mu, p) ~ (-mu, 1 - p)` alias documented in
`TNBBetaSpherical`/`posterior_centre` -- not evidence of a qualitatively
different posterior shape, since `q`, `epsilon` and `m` are the shape-only
quantities and all three land in the same regime across seeds.)

## Result: does epsilon stay stable over training, like (b) checked?

(b) found `epsilon` drifting upward *without bound* through 2000 epochs,
crossing from bimodal into unimodal territory for two of three seeds. Here,
with the real GCN on the synthetic SBM graph (seeds 0, 1, 2, same fixed
graph, `latent_dim=16`, threshold `epsilon = 7.5`):

| epoch | seed 0: eps (m, frac_bimodal) | seed 1: eps (m, frac_bimodal) | seed 2: eps (m, frac_bimodal) |
|---|---|---|---|
| 0 | 0.695 (-6.80, 1.00) | 0.695 (-6.80, 1.00) | 0.695 (-6.81, 1.00) |
| 200 | 3.073 (-4.43, 1.00) | 3.179 (-4.32, 1.00) | 2.965 (-4.53, 1.00) |
| 500 | 3.050 (-4.45, 1.00) | 3.052 (-4.45, 1.00) | 2.931 (-4.57, 1.00) |
| 1000 | 3.434 (-4.07, 1.00) | 3.445 (-4.05, 1.00) | 3.316 (-4.18, 1.00) |
| 1500 | 3.954 (-3.55, 1.00) | 3.916 (-3.58, 1.00) | 3.843 (-3.66, 1.00) |
| 2000 (final, 1999) | 4.487 (-3.01, 1.00) | 4.431 (-3.07, 1.00) | 4.351 (-3.15, 1.00) |

Validation AUC plateaus by roughly epoch 200-300 in all three seeds (~0.83-0.87)
and does not improve further; `epsilon` keeps climbing slowly for the entire
2000-epoch run in every seed, exactly the kind of unconstrained drift (b)
found. **Unlike (b), this drift never gets anywhere close to crossing the
bimodal/unimodal threshold (`epsilon = 7.5` at `latent_dim = 16`)** --
`frac_bimodal` is pinned at `1.0` at every single checkpoint, in every seed,
across the whole 2000-epoch run. Per-checkpoint rates (seed 0, computed
directly from the table above and the intermediate 100-epoch snapshots)
settle to roughly +0.08 to +0.14 per 100 epochs from epoch ~400 onward, with
no clear deceleration or acceleration trend visible over the remaining
1600 epochs (the rate is noisy but roughly steady, not monotonically
slowing the way "approaching an asymptote" would predict). Extrapolating
that steady rate linearly, reaching the threshold from epoch 2000's
`epsilon = 4.49` would take roughly another 2500-4000 epochs (epoch
~4500-6000 total) -- several times the 2000-epoch budget explored here, and
on strictly weaker evidence than a true asymptote would give, since the
rate is not visibly flattening. **This result is honestly reported as
"epsilon is still slowly drifting at 2000 epochs, exactly like (b), but the
drift here is far too slow and started from far too deep in the bimodal
regime to threaten `frac_bimodal = 1.0` within any training budget this
investigation has used elsewhere, and there is no evidence in this data
either that the drift is slowing toward an asymptote or that it will
accelerate the way (b)'s did" -- a slower, so-far-stable kind of drift than
(b)'s, not a provably bounded one.**

## Reading it

**This cleanly isolates the GCN + community-structured graph as sufficient
for sustained bimodality, independent of scale.** At the exact same item
count and community count where (b) (plain MLP, no graph) found
`frac_bimodal = 0.0` and a drift *toward* unimodal, routing the identical
task through `GraphVAE`'s real, unmodified GCN on a synthetic graph with
nothing but community-correlated edge structure (no real citation/
co-authorship-specific degree distribution, no tens-of-thousands of nodes)
gives `frac_bimodal = 1.0`, stable across three seeds and over the entire
2000-epoch training run -- qualitatively indistinguishable from the real
Cora result quoted in ablation (a) (`m_mean = -2.27`, `frac_bimodal = 0.952`
at `latent_dim = 16`, no feature reconstruction) and from the qualitative
"every node bimodal" pattern this investigation has found on real com-DBLP.
If anything, this synthetic graph's `m_mean` (-3.87 to -4.55) sits *more*
bimodal than Cora's (-2.27), not less.

**Combined with (a) and (b), the three ablations now triangulate the
mechanism precisely.** (b) ruled out "the pairwise/BCE loss alone" (no GCN,
no graph: unimodal). (a) ruled out "absence of a reconstruction term" (GCN
+ real graph + added decoder: still bimodal). This ablation (c) now
positively confirms "GCN message-passing + a graph with community-
correlated structure, together" as *sufficient* on its own (GCN + synthetic
graph, no real features, no real citation-network degree distribution,
1/634th of com-DBLP's node count: still bimodal). The remaining open
question is why -- what about message-passing over a graph with
community structure specifically produces this effect -- not whether it
happens; this ablation does not probe the mechanism further, only confirms
the GCN+graph combination is what matters, not scale or anything else
specific to a real dataset.

**Caveats:** one `(num_communities, nodes_per_community, p_in, p_out,
latent_dim)` configuration (matching (b)'s scale and this investigation's
standard `latent_dim=16`, not swept); one fixed graph realization
(`--graph-seed 0`) for the main 3-seed stability table (seed variation is
over model init/negative sampling only, not graph topology, except in the
separately-flagged epsilon-drift check, which also varies the edge split).
A sweep over `p_in`/`p_out`/graph density, or over multiple independently-
sampled SBM graphs, was out of scope here and could still show the effect
depends on how strongly community-correlated the graph is -- this ablation
establishes that *some* GCN+community-graph combination is sufficient, not
the full boundary of where the effect appears or disappears.
