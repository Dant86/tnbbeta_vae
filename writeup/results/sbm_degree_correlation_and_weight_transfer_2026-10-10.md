# Is `epsilon`/`m` a global regime baked into the weights, or read per-node from the graph? 2026-10-10

- **Date:** 2026-10-10
- **Branch/commit:** `neighbor-heterogeneity-mechanism`.
  - Test (A) (degree correlation) reuses code already present at `cfffc38`
    (`ruff format: blank line after docstring in sbm_pout_dose_response's embedded
    script`) -- no new committed code, same as the precedent it follows (see below).
  - Test (B) (frozen-weights transfer) uses `19d7f1e`
    (`Add apps/eval/sbm_weight_transfer.py: frozen-weights transfer test`).
  `master` did not yet have either.
- **Checkpoints used (all reused from disk, no retraining):**
  `dose_response_pout_0.005_tnbbeta_seed0`, `dose_response_pout_0.01_tnbbeta_seed0`,
  `dose_response_pout_0.02_tnbbeta_seed0`, `dose_response_pout_0.05_tnbbeta_seed0` --
  the exact checkpoints `sbm_pout_dose_response_2026-10-09.md` trained and confirmed
  well-trained (`p_out=0.005` and `p_out=0.05` explicitly re-confirmed loadable and
  well-trained for this task; `p_out=0.01`/`p_out=0.02` reused on the same basis,
  since neither was flagged as confounded in the original sweep). `p_out=0.1` is
  reported for test (A) only, clearly labeled confounded (see below) -- it is NOT
  used as a "well-trained" anchor for test (B), per this task's explicit
  instruction.

## Background

`sbm_pout_dose_response_2026-10-09.md` found TNBBeta's aggregate `m_mean` grows
steadily more negative (more bimodal) as a whole graph's `p_out` rises,
but `neighbor_cluster_visualization_2026-10-09.md`'s per-node heterogeneity-vs-`m`
correlation came out the wrong sign, and its qualitative neighbor-cluster check found
no evidence of real per-node splitting. The live reframing this write-up tests:
`epsilon`/`m` may behave like a **global regime** the whole model settles into
(shaped by overall task difficulty across the graph), rather than something each
node's own forward pass reads adaptively from its specific neighborhood -- the same
way `GraphVAE.temperature()` is already one scalar shared by every node. Two
concrete tests:

- **(A)** Does a node's own **degree** (the obvious free per-node variable not yet
  checked) correlate with its `m` better than heterogeneity did, and is it
  confounded with heterogeneity?
- **(B)** Frozen, already-trained TNBBeta weights, forward-passed (no further
  training) on the OTHER condition's graph: does the posterior shape stay anchored
  to what the model was trained on, or shift toward what a model actually trained
  on that graph would show?

## Test (A): per-node degree correlation

**Approach chosen: uncommitted ad hoc script**, same convention as the original
heterogeneity-vs-`m` check in `sbm_pout_dose_response_2026-10-09.md` (and
`sbm_gcn_recovery_ablation_2026-10-08.md`'s epsilon-drift check before it). This is
a one-off diagnostic over already-trained checkpoints that reuses
`tnbbeta_vae.data.neighbor_heterogeneity.neighbor_heterogeneity`,
`labels_to_communities`, `tnbbeta_vae.data.stochastic_block_model`'s
`sbm_community_labels`/`stochastic_block_model`,
`apps.eval.graph_posterior_shape.graph_to_batch` and
`tnbbeta_vae.training.load_model_checkpoint` with zero reimplementation of any of
their logic -- degree itself is the one new ingredient, and it is the literal
one-liner `np.asarray(adjacency.sum(axis=1)).flatten()` the task anticipated, not
something that earns its own module. A small, reusable, generically-named script
would also have been reasonable here; this call went the ad hoc way because, like
its precedent, this is a single diagnostic pass over a small, fixed grid of
existing checkpoints, not something expected to be re-run with different
parameters later.

**Manual sanity check performed before trusting any real number** (per this task's
instruction to verify ad hoc analysis code, not skip it just because it's
uncommitted): a tiny hand-built 4-node graph (`node0`: degree 1, `node1`: degree 3,
`node2`/`node3`: degree 2) confirmed `adjacency.sum(axis=1)` recovers exactly those
degrees, and a synthetic perfectly-anti-correlated `m = -degree` on that graph gave
Pearson `r = -1.0` and Spearman `r = -1.0` as expected -- both the degree
extraction and the correlation sign convention behave as intended before being
pointed at real checkpoints.

Full source (not committed):

```python
"""Ad hoc: correlates per-node degree (and heterogeneity) against TNBBeta's m on the
SBM dose-response checkpoints. Not a committed script -- see the write-up for why.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

from apps.eval.graph_posterior_shape import graph_to_batch
from tnbbeta_vae.data.neighbor_heterogeneity import (
    labels_to_communities,
    neighbor_heterogeneity,
)
from tnbbeta_vae.data.stochastic_block_model import (
    sbm_community_labels,
    stochastic_block_model,
)
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import load_model_checkpoint

device = torch.device("cpu")
labels = sbm_community_labels(10, 50)
communities = labels_to_communities(labels)

results = {}
for p_out in [0.005, 0.01, 0.02, 0.05, 0.1]:
    graph = stochastic_block_model(
        num_communities=10, nodes_per_community=50, p_in=0.3, p_out=p_out, seed=0
    )
    heterogeneity = neighbor_heterogeneity(graph.adjacency, communities)
    degree = np.asarray(graph.adjacency.sum(axis=1)).flatten()

    run_name = f"dose_response_pout_{p_out}_tnbbeta_seed0"
    ckpt_path = checkpoint_dir() / run_name / "final.pt"
    model, checkpoint = load_model_checkpoint(ckpt_path, device)
    batch = graph_to_batch(graph, device)
    model.eval()
    with torch.no_grad():
        posterior, _ = model.posterior_and_prior(batch)
    latent_dim = checkpoint["config"]["latent_dim"]
    epsilon = posterior.epsilon.detach().cpu().numpy()
    m = epsilon - (latent_dim - 1) / 2

    valid = np.isfinite(heterogeneity) & np.isfinite(m) & np.isfinite(degree)
    deg_pearson_r, deg_pearson_p = pearsonr(degree[valid], m[valid])
    deg_spearman_r, deg_spearman_p = spearmanr(degree[valid], m[valid])
    dh_pearson_r, dh_pearson_p = pearsonr(degree[valid], heterogeneity[valid])
    dh_spearman_r, dh_spearman_p = spearmanr(degree[valid], heterogeneity[valid])

    results[p_out] = {
        "degree_mean": float(degree.mean()),
        "heterogeneity_mean": float(np.nanmean(heterogeneity)),
        "num_valid_nodes": int(valid.sum()),
        "m_mean": float(m.mean()),
        "degree_vs_m_pearson_r": float(deg_pearson_r),
        "degree_vs_m_pearson_p": float(deg_pearson_p),
        "degree_vs_m_spearman_r": float(deg_spearman_r),
        "degree_vs_m_spearman_p": float(deg_spearman_p),
        "degree_vs_heterogeneity_pearson_r": float(dh_pearson_r),
        "degree_vs_heterogeneity_pearson_p": float(dh_pearson_p),
        "degree_vs_heterogeneity_spearman_r": float(dh_spearman_r),
        "degree_vs_heterogeneity_spearman_p": float(dh_spearman_p),
    }
    print(p_out, json.dumps(results[p_out]))
print(json.dumps(results, indent=2))
```

### Results

| `p_out` | degree mean (std) | heterogeneity mean | `m` mean (std) | degree-vs-`m` Pearson `r` | degree-vs-`m` Spearman `r` | degree-vs-heterogeneity Pearson `r` | degree-vs-heterogeneity Spearman `r` |
|---|---|---|---|---|---|---|---|
| 0.005 | 16.70 (3.68) | 0.468 | -4.380 (0.306) | **+0.941** | +0.944 | +0.176 | +0.152 |
| 0.01 | 19.03 (4.00) | 0.825 | -4.586 (0.266) | **+0.984** | +0.984 | +0.189 | +0.177 |
| 0.02 | 23.32 (4.43) | 1.263 | -4.690 (0.257) | **+0.919** | +0.912 | +0.235 | +0.225 |
| 0.05 | 36.96 (5.85) | 1.874 | -5.052 (0.169) | **+0.954** | +0.949 | +0.301 | +0.245 |
| 0.1 (confounded) | 60.14 (7.48) | 2.151 | -6.805 (0.00016) | +0.687 | +0.667 | +0.251 | +0.220 |

All `n=500`, every `p` value effectively `0` (Pearson) or `<1e-65` (Spearman,
worst case) -- not noise at this sample size. `p_out=0.1`'s row is reported for
completeness only: `m`'s std there is `0.00016` (vs. `0.17`-`0.31` at every other
`p_out`), confirming the checkpoint has collapsed to a single near-constant value
(the near-random-initialization collapse already flagged in
`sbm_pout_dose_response_2026-10-09.md`), so its correlation number reflects
near-zero real variance, not a genuine weaker effect, and should not be compared
directly to the other four rows.

### Reading

**Degree correlates with `m` far more strongly than heterogeneity does, at every
well-trained `p_out`.** Pearson/Spearman `r` for degree-vs-`m` sits at `0.92`-`0.98`
across `p_out` in `{0.005, 0.01, 0.02, 0.05}`, dwarfing heterogeneity-vs-`m`'s
`0.13`-`0.27` from the prior write-up. This is not just heterogeneity in disguise:
degree-vs-heterogeneity itself is only modestly correlated (`r = 0.15`-`0.30`), so
degree carries most of its relationship with `m` independently of neighbor
heterogeneity, not as a proxy for it.

**The sign, though, is the same direction heterogeneity showed, not the mechanism's
predicted direction, and it is strong enough to be the dominant per-node effect in
this experiment.** The correlation is *positive*: higher-degree nodes have *less*
negative (less bimodal) `m`, and lower-degree nodes have *more* negative (more
bimodal) `m`. If anything, this is the opposite of what a "more neighbors to
reconcile demands more bimodal capacity" story would predict, and it is a much
bigger per-node effect than heterogeneity's small, also-wrong-signed correlation.
A plausible alternative reading: low-degree nodes get a noisier, less-determined
GCN-aggregated feature (fewer neighbors to average over), and the model's learned
response to that uncertainty is to push further into the already-dominant bimodal
corner, not to escape it for a more concentrated unimodal-like region -- i.e. "how
much the GCN had to average over" is a bigger per-node driver of `m` in this setup
than "how heterogeneous what it averaged over was." This is reported as a real,
measured pattern, not a confirmation of the heterogeneity mechanism this
investigation originally set out to test -- it answers "does *some* per-node
variable correlate strongly with `m`?" (yes, degree) separately from "does the
*theorized* mechanism (heterogeneity) explain it?" (no, not by itself, and not in
the predicted direction either).

## Test (B): frozen-weights transfer

**Commands** (training commands identical to `sbm_pout_dose_response_2026-10-09.md`,
already run and reused from disk; only the four new `sbm_weight_transfer` forward
passes are new):

```
# (training already done for sbm_pout_dose_response_2026-10-09.md; reused as-is)
uv run python -m apps.synthetic.sbm_recovery \
    --out-dir runs/neighbor_heterogeneity_dose_response/pout_0.005 \
    --p-out 0.005 --epochs 500 --device cpu \
    --run-name dose_response_pout_0.005
uv run python -m apps.synthetic.sbm_recovery \
    --out-dir runs/neighbor_heterogeneity_dose_response/pout_0.05 \
    --p-out 0.05 --epochs 500 --device cpu \
    --run-name dose_response_pout_0.05

# low-weights (trained at p_out=0.005) on its own training graph
uv run python -m apps.eval.sbm_weight_transfer \
    --run-name dose_response_pout_0.005_tnbbeta_seed0 --device cpu \
    --num-communities 10 --nodes-per-community 50 --p-in 0.3 \
    --p-out 0.005 --graph-seed 0

# low-weights on the OTHER (high-p_out) graph -- the transfer test
uv run python -m apps.eval.sbm_weight_transfer \
    --run-name dose_response_pout_0.005_tnbbeta_seed0 --device cpu \
    --num-communities 10 --nodes-per-community 50 --p-in 0.3 \
    --p-out 0.05 --graph-seed 0

# high-weights (trained at p_out=0.05) on its own training graph
uv run python -m apps.eval.sbm_weight_transfer \
    --run-name dose_response_pout_0.05_tnbbeta_seed0 --device cpu \
    --num-communities 10 --nodes-per-community 50 --p-in 0.3 \
    --p-out 0.05 --graph-seed 0

# high-weights on the OTHER (low-p_out) graph -- the transfer test
uv run python -m apps.eval.sbm_weight_transfer \
    --run-name dose_response_pout_0.05_tnbbeta_seed0 --device cpu \
    --num-communities 10 --nodes-per-community 50 --p-in 0.3 \
    --p-out 0.005 --graph-seed 0
```

### Results: the full 2x2 grid

| weights \ graph | low graph (`p_out=0.005`) | high graph (`p_out=0.05`) |
|---|---|---|
| **low weights** (trained `p_out=0.005`) | `m_mean=-4.380`, `epsilon_mean=3.120`, `epsilon_std=0.306`, `p_mean=0.959`, `q_mean=0.882`, `r_bar=0.914`, `frac_bimodal=1.0` **(own graph)** | `m_mean=-4.548`, `epsilon_mean=2.952`, `epsilon_std=0.212`, `p_mean=0.952`, `q_mean=0.871`, `r_bar=0.899`, `frac_bimodal=1.0` **(transfer)** |
| **high weights** (trained `p_out=0.05`) | `m_mean=-4.974`, `epsilon_mean=2.526`, `epsilon_std=0.290`, `p_mean=0.963`, `q_mean=0.922`, `r_bar=0.923`, `frac_bimodal=1.0` **(transfer)** | `m_mean=-5.052`, `epsilon_mean=2.448`, `epsilon_std=0.170`, `p_mean=0.959`, `q_mean=0.918`, `r_bar=0.914`, `frac_bimodal=1.0` **(own graph)** |

`frac_bimodal = 1.0` in all four cells -- every cell stays deep in the proven
bimodal regime regardless of which weights or which graph is used, same as the
original dose-response's whole grid.

### Reading

**Both checkpoints shift in the predicted direction under transfer, but only
partway -- most of each gap stays unclosed.** The two "own graph" values anchor the
comparison: a model actually trained at `p_out=0.005` shows `m_mean=-4.380`; a model
actually trained at `p_out=0.05` shows `m_mean=-5.052`. That's a `0.672` gap between
"what training on the low condition produces" and "what training on the high
condition produces."

- **Low weights, fed the high graph:** `m_mean` moves from `-4.380` to `-4.548` --
  a shift of `-0.168`, or **25% of the full `0.672` gap** toward the high-weights
  value. 75% of the gap stays unclosed: the low-weights model does not come close
  to looking like a model actually trained on the high-heterogeneity graph.
- **High weights, fed the low graph:** `m_mean` moves from `-5.052` to `-4.974` --
  a shift of `+0.078`, or **12% of the gap** toward the low-weights value. Here
  even less of the gap closes.

Neither cell is perfectly flat (a pure "100% baked into the weights, the input
graph has zero effect" story would predict both transfer shifts are ~0, which is
not what happened), but neither comes anywhere near matching what a model actually
trained on the new condition shows either. **The honest reading is a real, mixed
result that leans toward "mostly baked into the weights, with a modest,
real, direction-consistent sensitivity to the graph fed in"** -- 75-88% of each
gap stays anchored to the training condition, while 12-25% genuinely shifts toward
the new graph's regime. This is consistent with `m`/`epsilon` behaving more like a
regime the model settles into during training (closer to
`GraphVAE.temperature()`'s single shared scalar) than like a per-node quantity read
fresh from each forward pass's specific input -- but it is not an all-or-nothing
result, and is reported as such rather than rounded to either extreme.

## Tying test (A) and test (B) together

Both tests point the same way, though neither is a clean confirmation of the
original heterogeneity-driven, per-node-adaptive mechanism:

1. **Test (A)** shows a strong per-node correlate of `m` does exist (degree, not
   heterogeneity), but it is NOT read "fresh" from each node's current
   neighborhood composition in the way the original theory imagined --
   heterogeneity itself, the theory's actual candidate variable, remains a weak,
   wrong-signed per-node predictor even after controlling for degree being a
   mostly-separate signal.
2. **Test (B)** shows that frozen weights carry most (75-88%) of their
   training-time regime across to a structurally different graph, with only a
   modest genuine adaptive component -- consistent with `epsilon`/`m` being shaped
   substantially by what the whole training graph looked like (a "global regime"),
   not purely re-derived per node, per forward pass, from whatever graph is handed
   to the encoder.

Together: the per-node story this investigation originally tested (heterogeneous
neighbors -> a specific node's bimodal escape hatch) is not supported by either
test. What IS supported is a coarser picture -- the trained model settles into an
`epsilon`/`m` regime substantially determined by the graph it trained on, with
degree (not heterogeneity) the strongest per-node correlate of where individual
nodes land around that regime, and only a modest genuine shift when the same
weights see a different graph at eval time.

**Caveats:** single seed per `p_out` for both tests (consistent with this
investigation's existing per-point caveat); `p_out=0.1` excluded from test (B)
entirely (confounded, see `sbm_pout_dose_response_2026-10-09.md`) and included in
test (A) only for completeness, clearly flagged; one fixed
`(num_communities, nodes_per_community, p_in, latent_dim)` configuration, not
swept; the degree-vs-`m` correlation, while strong, is a correlation on one
synthetic graph family (stochastic block model), not a controlled intervention on
degree holding everything else fixed -- it does not by itself establish degree as
a *causal* driver of `m`, only as a very strong per-node correlate of it.
