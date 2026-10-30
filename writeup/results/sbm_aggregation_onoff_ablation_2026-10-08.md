# GCN aggregation ON vs. OFF, same graph and loss: the sharpest test of the aggregation theory, 2026-10-08

- **Date:** 2026-10-08
- **Branch/commit:** `dtd-oriented-textures`, `98c6ee4` ("sbm_recovery: add
  --no-aggregation, the 'aggregation OFF' arm of ablation (c)") -- the two commits
  immediately before this write-up on the same branch add `run_once`'s
  `encoder_adjacency` parameter (`123592a`) and `sbm_recovery.py`'s
  `--no-aggregation` flag (`98c6ee4`) that this result depends on. `master` does
  not yet have this code.
- **Commands (main 4-family table + 3-seed TNBBeta stability table, "aggregation
  OFF"):**
  ```
  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/sbm_recovery_noagg --epochs 2000 \
      --graph-seed 0 --seed 0 --device cpu --run-name sbm_recovery_noagg \
      --no-aggregation

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/sbm_recovery_noagg_seed1 --epochs 2000 \
      --graph-seed 0 --seed 1 --device cpu --run-name sbm_recovery_noagg \
      --no-aggregation --families tnbbeta

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/sbm_recovery_noagg_seed2 --epochs 2000 \
      --graph-seed 0 --seed 2 --device cpu --run-name sbm_recovery_noagg \
      --no-aggregation --families tnbbeta
  ```
  (All three share `--graph-seed 0`: the identical SBM graph and edge split as
  ablation (c)'s own tables, cited below unchanged, not re-run.)
- **Command (epsilon-drift-over-training check, ad hoc, not a committed script --
  reusing ablation (c)'s own methodology and justification for why this one check
  stays uncommitted; see that ablation's write-up for the reasoning, not repeated
  here):**
  ```
  PYTHONPATH=. uv run python /tmp/sbm_noagg_drift_check.py \
      --seeds 0 1 2 --max-epochs 2000 --every 100
  PYTHONPATH=. uv run python /tmp/sbm_noagg_drift_check.py \
      --seeds 0 --max-epochs 6000 --every 500
  ```

## Background

`writeup/results/sbm_gcn_recovery_ablation_2026-10-08.md` ("(c)", commit
`98c6ff4`) found that routing a synthetic, featureless, community-structured
stochastic-block-model (SBM) graph through `GraphVAE`'s real, unmodified GCN
encoder reproduces the bimodal-posterior pattern (`m = epsilon - (latent_dim -
1) / 2 < 0`) seen on every real citation/co-authorship `GraphVAE` checkpoint
this investigation has checked (Cora, Citeseer, Pubmed, com-DBLP): `m_mean =
-4.55`, `frac_bimodal = 1.00`, stable across 3 seeds and 2000 epochs of
training. Combined with ablation (b) (the same loss, no graph at all, on
Gaussian-blob clusters: unimodal) and ablation (a) (a feature decoder added to
the real GCN+Cora setup: still bimodal), this triangulated "GCN message-passing
aggregating over a community-correlated graph" as *sufficient* for bimodality.

The live mechanistic theory (`graph_bimodality_investigation_summary_2026-10-08.md`,
section 5): a GCN's neighbor-averaging pulls a node's embedding toward a
compromise position that may score poorly against *every* individual neighbor
if the neighborhood is angularly heterogeneous; a bimodal posterior can align
with different neighbor subsets on different stochastic draws, where a
unimodal cap is stuck with one mediocre compromise. This has not been tested
directly: (c) does not distinguish "GCN aggregation specifically" from "a
graph-correlated training signal exists in some looser sense," since (c)'s
GCN both aggregates over neighbors *and* is the only way its identity features
carry any information about graph structure at all.

This ablation holds (c)'s exact graph, exact edges (so the same loss), and
exact everything else fixed, and varies **only** whether the encoder
aggregates over neighbors: `--no-aggregation` passes `run_once`'s new
`encoder_adjacency` parameter a sparse identity matrix, so `GraphBatch.
norm_adjacency` is the identity (no neighbor mixing: `torch.sparse.mm(I, X) =
X`) instead of `normalized_adjacency(split.train_adjacency)`. Training still
targets the real SBM edges from `split` for the loss, unchanged.

## Result: the main 4-family table (`--graph-seed 0 --seed 0`, 2000 epochs, no aggregation)

| family | test_auc | test_ap | best_epoch | entropy_mean | r_bar |
|---|---|---|---|---|---|
| gaussian | 0.8169 | 0.8198 | 1736 | 0.970 | 0.849 |
| vmf | 0.7444 | 0.7084 | 1999 | 0.316 | 0.356 |
| power_spherical | 0.4754 | 0.4813 | 798 | 1.312 | 0.107 |
| tnbbeta | 0.4888 | 0.4886 | 161 | NaN (no closed form) | 0.098 |

TNBBeta-only diagnostic (seed 0): `p_mean = 0.5009`, `q_mean = 0.5461`,
`epsilon_mean = 0.7821`, **`m_mean = -6.718`**, **`frac_bimodal = 1.0`**.

**Critical fact, not present in (c):** TNBBeta's and Power Spherical's
`test_auc` (0.489, 0.475) are indistinguishable from chance (0.5); the task is
essentially unsolved for both. Gaussian and vMF still learn something useful
(0.82, 0.74), well below (c)'s ~0.85 for every family but clearly above
chance.

## Result: TNBBeta stability across 3 seeds (same graph, varying only model init)

| seed | test_auc | test_ap | best_epoch | p_mean | q_mean | epsilon_mean | m_mean | frac_bimodal |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.4888 | 0.4886 | 161 | 0.5009 | 0.5461 | 0.7821 | -6.718 | 1.0 |
| 1 | 0.5025 | 0.4922 | 602 | 0.4999 | 0.6297 | 0.9520 | -6.548 | 1.0 |
| 2 | 0.5095 | 0.5125 | 600 | 0.5015 | 0.6305 | 0.9574 | -6.543 | 1.0 |

`frac_bimodal = 1.0` in every seed and `test_auc` stays at chance in every
seed -- both the bimodality and the complete failure to solve the task are
stable findings, not a one-seed fluke.

## The aggregation-ON-vs-OFF comparison table (citing (c), not re-run)

| family | test_auc, ON (c) | test_auc, OFF (this run) |
|---|---|---|
| gaussian | 0.8423 | 0.8169 |
| vmf | 0.8496 | 0.7444 |
| power_spherical | 0.8500 | 0.4754 |
| tnbbeta | 0.8461 | 0.4888 |

| TNBBeta shape stat | ON (c), 3 seeds | OFF (this run), 3 seeds |
|---|---|---|
| `m_mean` range | -4.55 to -3.87 | -6.72 to -6.54 |
| `frac_bimodal` | 1.0 (all 3 seeds) | 1.0 (all 3 seeds) |

Taken at face value, turning aggregation off made the posterior **more**
bimodal, not less -- the opposite of what the theory in section 5 of the
summary predicts (removing the compromise-position pressure should move
`epsilon` up, `m` toward 0, same direction as ablation (b)'s no-graph MLP
control, which went unimodal). See "Reading it" below for why this headline
number is not a clean test of that theory.

## Result: does epsilon stay stable over training, like (b) and (c) checked?

Same ad hoc-script methodology as (c) (full source below, differing from (c)'s
own script only in building `GraphBatch.norm_adjacency` from a sparse identity
matrix instead of `normalized_adjacency(split.train_adjacency)`), seeds 0, 1,
2, 2000 epochs, `latent_dim = 16` (threshold `epsilon = 7.5`):

| epoch | seed 0: eps (m) | val_auc | seed 1: eps (m) | val_auc | seed 2: eps (m) | val_auc |
|---|---|---|---|---|---|---|
| 0 | 0.697 (-6.80) | 0.469 | 0.700 (-6.80) | 0.547 | 0.700 (-6.80) | 0.508 |
| 500 | 1.669 (-5.83) | 0.460 | 1.688 (-5.81) | 0.514 | 1.682 (-5.82) | 0.497 |
| 1000 | 1.868 (-5.63) | 0.518 | 1.865 (-5.63) | 0.514 | 1.862 (-5.64) | 0.480 |
| 1500 | 2.033 (-5.47) | 0.558 | 2.021 (-5.48) | 0.525 | 2.024 (-5.48) | 0.451 |
| 2000 (final, 1999) | 2.235 (-5.27) | 0.485 | 2.202 (-5.30) | 0.494 | 2.222 (-5.28) | 0.549 |

`val_auc` never clears a noisy band around 0.45-0.58 (chance is 0.5) at any
checkpoint, in any seed, across the whole 2000-epoch run -- there is no
training-dynamics evidence of the model ever finding a usable signal, only
noise around chance. `epsilon` still drifts upward the entire time, exactly
the kind of unconstrained drift (b) and (c) both found, and nowhere near the
`epsilon = 7.5` threshold within this budget.

**Extending seed 0 to 6000 epochs** (same script, `--every 500`) to check
whether this is a transient warm-up rather than a stable finding:

| epoch | epsilon | m | val_auc |
|---|---|---|---|
| 0 | 0.697 | -6.80 | 0.469 |
| 1000 | 1.881 | -5.62 | 0.534 |
| 2000 | 2.230 | -5.27 | 0.491 |
| 3000 | 2.680 | -4.82 | 0.579 |
| 4000 | 3.200 | -4.30 | 0.522 |
| 5000 | 3.783 | -3.72 | 0.490 |
| 5999 | 4.368 | -3.13 | 0.467 |

`val_auc` still never clears chance through 6000 epochs (0.47-0.58 throughout,
no trend) -- the failure to learn is not a slow start, it persists for 3x
(c)'s own budget. `epsilon` keeps climbing, and keeps climbing at a roughly
*steady or slightly increasing* per-500-epoch rate (+0.97, +0.21, +0.16, +0.19,
+0.21, +0.25, +0.27, +0.26, +0.26, +0.26, +0.28, +0.30 -- not decelerating
toward an asymptote the way a converging quantity would). By epoch 6000,
`epsilon = 4.37` is in the same range as (c)'s epoch-2000 value (`epsilon`
4.35-4.49 across its 3 seeds) -- this condition eventually reaches the same
`epsilon` ballpark as the aggregation-ON condition, just roughly 3x slower in
epochs. Extrapolating the epoch-5500-to-5999 rate (`+0.30`/499 epochs)
linearly, reaching `epsilon = 7.5` from epoch 6000's 4.37 would take roughly
another 5200 epochs (epoch ~11,000 total) -- far outside any budget this
investigation has used, and, as with (c)'s own extrapolation, on weaker
evidence than a genuine asymptote would give, since the rate is not clearly
slowing. **Honestly reported: at no point checked (0 to 6000 epochs) does
`epsilon` stabilize, and at no point does validation AUC exceed chance by any
visible margin -- both facts together, not separately.**

Ad hoc script's full source (identical reuse of `GraphVAE`, `posterior_stats`,
`stochastic_block_model`, `split_edges`, `normalized_adjacency` and
`apps.link_prediction.main.score_edges` as (c)'s own script; the only
substantive difference is `norm_adjacency=normalized_adjacency(identity)`
instead of `normalized_adjacency(split.train_adjacency)`):

```python
"""Ad hoc epsilon-drift check for the --no-aggregation ("aggregation OFF") arm.

Mirrors the methodology of ablation (c)'s own ad hoc drift-check script
(writeup/results/sbm_gcn_recovery_ablation_2026-10-08.md), but builds the
GraphBatch with a sparse identity norm_adjacency (no neighbor mixing) instead
of normalized_adjacency(split.train_adjacency), reusing GraphVAE,
posterior_stats, stochastic_block_model, split_edges, normalized_adjacency
and apps.link_prediction.main.score_edges -- no reimplementation of any of
their logic.
"""

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
num_nodes = graph.adjacency.shape[0]
identity = sp.identity(num_nodes, format="csr", dtype="float32")

results: dict[int, list[dict]] = {}
for seed in args.seeds:
    split = split_edges(graph.adjacency, seed=seed)
    upper = sp.triu(split.train_adjacency, k=1).tocoo()
    up = np.stack([upper.row, upper.col])
    batch = GraphBatch(
        features=graph.features.to(dtype=torch.float32),
        norm_adjacency=normalized_adjacency(identity),
        positive_edges=torch.as_tensor(up, dtype=torch.long),
    )
    torch.manual_seed(seed)
    config = GraphVAEConfig(
        family="tnbbeta",
        in_features=graph.features.shape[1],
        latent_dim=args.latent_dim,
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

## A check against the untrained model: is this just initialization?

Because `test_auc`/`val_auc` never clear chance, the natural next question is
whether 2000+ epochs of training moved the TNBBeta posterior at all, or
whether it is simply sitting near its random initialization the whole time. A
fresh, **untrained** model (same config, same seed-0 initialization, identity
`norm_adjacency`, forward pass only, no training step) gives:

```
p_mean=0.5004, q_mean=0.5000, epsilon_mean=0.6954, m_mean=-6.805, frac_bimodal=1.0
```

This is nearly identical to epoch 0 of the drift table above (`epsilon =
0.697`, `m = -6.80`) -- expected, since epoch 0 there is after exactly one
gradient step. More importantly, it means the entire 2000-6000-epoch
trajectory (`epsilon`: 0.70 to ~4.4, `m`: -6.80 to ~-3.1) is a *slow,
monotonic walk away from an already-bimodal initialization*, not a
sudden or dataset-driven snap into bimodality -- and the walk is happening
while validation AUC sits at chance the entire time, i.e., with no
detectable learning signal driving it. This strongly suggests the `epsilon`
drift here is driven by the KL term / Adam's optimization dynamics on this
parameterization in a low-gradient-information regime, not by anything
specific to the (near-absent) link-prediction signal.

## Reading it

**This does not cleanly confirm or refute "GCN aggregation specifically."** At
face value, removing aggregation while holding the graph, edges and loss fixed
left the posterior just as bimodal as (c) -- actually more so (`m_mean`
-6.5 to -6.7 vs. (c)'s -4.55 to -3.87) -- which is the *opposite* of what
section 5's theory predicts (unimodal, like ablation (b)). Taken alone, that
would be a real refutation.

But this condition **never solves the task**: TNBBeta's and Power Spherical's
`test_auc` (0.489, 0.475) are at chance, and training-curve monitoring out to
6000 epochs shows no departure from chance at any point -- not a slow start,
a persistent failure. That is a fundamentally different situation from
ablation (b)'s no-graph control, which *did* learn the task well (`test_auc`
0.93-0.95) and only then went unimodal. The reason is architectural, not
incidental: (b)'s synthetic Gaussian-blob features are continuous positions
that are independently informative of cluster membership with no graph
involved at all. Here, the SBM graph's features are a pure one-hot identity
matrix carrying zero information about community structure by construction
(by design, to match com-DBLP's featureless convention) -- the *only* way
those features become informative about which other nodes a node is like is
through the GCN's aggregation step itself. Turning aggregation off here does
not just remove "neighbor averaging that creates a compromise position" (the
theory's mechanism); it also removes the *entire channel* through which this
particular model could ever learn anything about graph structure, leaving a
free per-node lookup table (two linear layers applied to a one-hot row) that
apparently cannot find a useful optimum for TNBBeta/Power Spherical within an
budget 3x larger than (c)'s, while Gaussian and vMF partially do (`test_auc`
0.82, 0.74).

The untrained-model check above adds a second, independent piece of evidence
for the same conclusion: this condition's whole multi-thousand-epoch
trajectory is a slow walk away from an *already bimodal* random
initialization, with no validation-AUC evidence that anything is being
learned along the way. The honest conclusion is that this specific
"aggregation OFF" design -- identity features *and* identity
`encoder_adjacency` -- confounds "no aggregation" with "no informative
gradient signal of any kind," so its persistent (indeed deepened) bimodality
cannot be attributed to aggregation's absence in a *successfully trained*
model, which is what the theory in section 5 actually makes a claim about.
This ablation is better read as **inconclusive for the aggregation-specific
mechanism**, with a side finding that TNBBetaSpherical's default
initialization is itself solidly in the bimodal regime (`m = -6.80` at
init, for this `latent_dim = 16` configuration) and that very slow,
monotonic `epsilon` growth under Adam appears largely independent of whether
the training signal is informative at all (it happens here even at chance-level
AUC, and happened in (c) and (b) too) -- a dynamic this investigation has
now seen in three different training setups, worth treating as its own open
question going forward, separately from the aggregation theory.

**What would make this a clean test:** a no-aggregation control that *can*
still solve the task -- e.g., non-identity, informative node features
(continuous, graph-structure-correlated, but not requiring aggregation to
decode, analogous to ablation (b)'s blob positions) routed through the same
identity-`encoder_adjacency` path. That experiment is not what was run here
and is flagged as the natural next step, not attempted in this write-up.

**Caveats:** one graph realization (`--graph-seed 0`, identical to (c)'s);
TNBBeta-only for the 3-seed stability table (matching (c)'s own choice to
isolate TNBBeta there); the epsilon-drift check's 6000-epoch extension was run
for seed 0 only (not 1 and 2), on the premise that all three seeds already
tracked each other closely through 2000 epochs in both the main drift table
above and the separate 3-seed `run_once` table). The identity-feature
confound described above is not a subtle caveat -- it is the main reason
this result does not settle the question the task set out to test, and is
reported as such rather than downplayed.
