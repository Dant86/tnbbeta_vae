# Does neighbor heterogeneity actually track TNBBeta bimodality? A real `p_out` dose-response, 2026-10-09

- **Date:** 2026-10-09
- **Branch/commit:** `neighbor-heterogeneity-mechanism`, `6094d63` at the time this
  sweep was run (`stochastic_block_model: add feature_noise_std, threaded into
  sbm_recovery.py`). `master` did not yet have this code.
- **Commands** (`--p-out` and `--epochs` are the only knobs varied across the five
  runs below; everything else is `apps/synthetic/sbm_recovery.py`'s existing CLI,
  already supporting `--p-out`, no new sweep-plumbing code needed):
  ```
  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/neighbor_heterogeneity_dose_response/pout_0.005 \
      --p-out 0.005 --epochs 500 --device cpu \
      --run-name dose_response_pout_0.005

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/neighbor_heterogeneity_dose_response/pout_0.01 \
      --p-out 0.01 --epochs 500 --device cpu \
      --run-name dose_response_pout_0.01

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/neighbor_heterogeneity_dose_response/pout_0.02 \
      --p-out 0.02 --epochs 500 --device cpu \
      --run-name dose_response_pout_0.02

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/neighbor_heterogeneity_dose_response/pout_0.05 \
      --p-out 0.05 --epochs 500 --device cpu \
      --run-name dose_response_pout_0.05

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/neighbor_heterogeneity_dose_response/pout_0.1 \
      --p-out 0.1 --epochs 500 --device cpu \
      --run-name dose_response_pout_0.1
  ```
  (Every run shares the defaults: `--num-communities 10 --nodes-per-community 50
  --p-in 0.3 --latent-dim 16 --hidden-dim 32 --graph-seed 0 --seed 0 --families
  gaussian vmf power_spherical tnbbeta`.) The additional heterogeneity-vs-`m`
  correlation (not something `sbm_recovery.py` itself computes) was produced by a
  short, uncommitted ad hoc script -- same convention as
  `sbm_gcn_recovery_ablation_2026-10-08.md`'s epsilon-drift check, for the same
  reason: it's a one-off diagnostic over already-trained checkpoints, reusing
  `tnbbeta_vae.data.neighbor_heterogeneity.neighbor_heterogeneity`,
  `labels_to_communities`, `stochastic_block_model`,
  `apps.eval.graph_posterior_shape.graph_to_batch` and
  `tnbbeta_vae.training.load_model_checkpoint` with no reimplementation of any of
  their logic, not a reason to add instrumentation to the committed, already-tested
  `sbm_recovery.py` pipeline. Its full source:

  ```python
  """Ad hoc: correlates per-node neighbor-heterogeneity against TNBBeta's m on the
  SBM dose-response checkpoints. Not a committed script -- see this write-up for
  why."""
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

      valid = np.isfinite(heterogeneity) & np.isfinite(m)
      pearson_r, pearson_p = pearsonr(heterogeneity[valid], m[valid])
      spearman_r, spearman_p = spearmanr(heterogeneity[valid], m[valid])

      results[p_out] = {
          "heterogeneity_mean": float(np.nanmean(heterogeneity)),
          "num_valid_nodes": int(valid.sum()),
          "m_mean": float(m.mean()),
          "pearson_r": float(pearson_r),
          "pearson_p": float(pearson_p),
          "spearman_r": float(spearman_r),
          "spearman_p": float(spearman_p),
      }
      print(p_out, json.dumps(results[p_out]))
  print(json.dumps(results, indent=2))
  ```

## Background

The live mechanistic theory
(`writeup/results/graph_bimodality_investigation_summary_2026-10-08.md`, section 5):
a GCN's neighbor-averaging pulls a node's embedding toward a compromise position
that scores poorly against a *heterogeneous* neighborhood; TNBBeta's bimodal
capacity should let training escape that compromise where vMF/Power Spherical's
unimodal cap cannot, so **more neighbor heterogeneity should predict more negative
`m`** (more bimodal) -- both when comparing graphs with more heterogeneity overall
(the dose-response tested here) and, more locally, when comparing nodes with more
heterogeneous neighborhoods within the same graph.

`stochastic_block_model`'s `p_out` directly controls ground-truth neighbor
heterogeneity by construction: holding `p_in=0.3` fixed, raising `p_out` raises the
fraction of each node's neighbors drawn from other communities, which raises the
Shannon entropy `tnbbeta_vae.data.neighbor_heterogeneity.neighbor_heterogeneity`
measures.

## Sanity check: the graph stays non-degenerate across the whole grid

(Same convention as the original `sbm_gcn_recovery_ablation_2026-10-08.md` sanity
check, now repeated at every `p_out`, `num_communities=10`,
`nodes_per_community=50`, `graph_seed=0`.)

| `p_out` | edges | avg degree (min, max) | isolated nodes | empirical within-density | empirical across-density |
|---|---|---|---|---|---|
| 0.005 | 4175 | 16.70 (6, 27) | 0 | 0.2953 (target 0.3) | 0.0050 (target 0.005) |
| 0.01 | 4758 | 19.03 (6, 30) | 0 | 0.2953 | 0.0101 (target 0.01) |
| 0.02 | 5829 | 23.32 (10, 37) | 0 | 0.2953 | 0.0197 (target 0.02) |
| 0.05 | 9239 | 36.96 (21, 55) | 0 | 0.2953 | 0.0500 (target 0.05) |
| 0.1 | 15035 | 60.14 (39, 83) | 0 | 0.2953 | 0.1015 (target 0.1) |

No isolated nodes, and empirical within/across densities match the requested
probabilities at every `p_out` -- every graph in the grid is a real, non-degenerate
measurement, not an artifact of a pathological corner case. (Within-density stays
fixed at ~0.295 since `p_in` never changes across this sweep.) The ground-truth
mean neighbor heterogeneity rises monotonically with `p_out`, as expected by
construction: **0.468 -> 0.825 -> 1.263 -> 1.874 -> 2.151** nats (`p_out` = 0.005,
0.01, 0.02, 0.05, 0.1).

## Result: AUC and TNBBeta shape, by `p_out`

| `p_out` | gaussian AUC | vMF AUC | Power Spherical AUC | TNBBeta AUC | TNBBeta `m_mean` | `frac_bimodal` |
|---|---|---|---|---|---|---|
| 0.005 | 0.8993 | 0.8969 | 0.8944 | 0.9017 | -4.380 | 1.0 |
| 0.01 | 0.8393 | 0.8501 | 0.8500 | 0.8496 | -4.586 | 1.0 |
| 0.02 | 0.7455 | 0.7789 | 0.7859 | 0.7821 | -4.690 | 1.0 |
| 0.05 | 0.6023 | 0.6526 | 0.6219 | 0.6475 | -5.052 | 1.0 |
| 0.1 | 0.4879 | 0.5268 | 0.5175 | 0.5216 | -6.805 | 1.0 |

TNBBeta-vs-baseline AUC gap:

| `p_out` | TNBBeta - vMF | TNBBeta - Power Spherical |
|---|---|---|
| 0.005 | +0.0048 | +0.0073 |
| 0.01 | -0.0005 | -0.0004 |
| 0.02 | +0.0032 | -0.0038 |
| 0.05 | -0.0051 | +0.0256 |
| 0.1 | -0.0052 | +0.0041 |

**Important caveat on `p_out=0.1`:** `power_spherical`'s and `tnbbeta`'s best
checkpoints there were saved at `best_epoch=0`, meaning validation AUC never
improved past random initialization in either family over the full 500-epoch run
-- the task has become essentially unlearnable at this `p_out` (AUC ~0.52,
barely above chance) for those two families, so `p_out=0.1`'s `m_mean`/`frac_bimodal`
and AUC-gap numbers reflect near-random-initialization geometry, not a trained
representation. They are reported for completeness, not as evidence either way.

**`m_mean` grows monotonically more negative as `p_out` (and ground-truth
heterogeneity) rises** -- -4.38 -> -4.59 -> -4.69 -> -5.05 -> -6.81 -- consistent
with the theory's aggregate (across-graph) prediction: a graph with more
neighbor heterogeneity overall produces a more bimodal TNBBeta posterior overall.
`frac_bimodal = 1.0` at every single `p_out`, so this entire grid stays deep in
the proven-bimodal regime regardless.

**The TNBBeta-vs-baseline AUC gap does *not* show a clean, monotonic trend.** It
stays small (within about ±0.005) against vMF throughout, and is mixed/noisy
against Power Spherical (-0.0038 at `p_out=0.02`, +0.0256 at `p_out=0.05`, the two
closest-together values in the grid). A 3-seed cross-check at `p_out=0.05` (same
`--graph-seed 0`, varying only `--seed 0/1/2`):

| seed | TNBBeta AUC | vMF AUC | Power Spherical AUC |
|---|---|---|---|
| 0 | 0.6475 | 0.6526 | 0.6219 |
| 1 | 0.6590 | 0.6551 | 0.5990 |
| 2 | 0.6086 | 0.6605 | 0.6117 |
| **mean** | **0.6384** | **0.6561** | **0.6109** |

Averaged over 3 seeds, TNBBeta is *behind* vMF by -0.0177 AUC and *ahead of* Power
Spherical by +0.0276 AUC -- a single-seed measurement at this scale is genuinely
noisy (one seed, #2, shows TNBBeta's checkpoint collapsing toward the uniform-prior
corner, `p_mean=0.501`, `q_mean=0.503`, vs. seeds 0/1's well-trained `p_mean~0.04-
0.07`, `q_mean~0.90-0.98`), and the TNBBeta-vs-vMF-specifically comparison does not
show a reliable advantage in this setting.

## Result: does the per-node heterogeneity-vs-`m` correlation match the theory's sign?

The theory predicts a **negative** within-graph correlation: nodes with more
heterogeneous neighborhoods should have more negative (more bimodal) `m`.

| `p_out` | heterogeneity mean | Pearson r | Pearson p | Spearman r | Spearman p |
|---|---|---|---|---|---|
| 0.005 | 0.468 | **+0.160** | 3.3e-4 | +0.133 | 2.9e-3 |
| 0.01 | 0.825 | **+0.199** | 7.5e-6 | +0.185 | 3.1e-5 |
| 0.02 | 1.263 | **+0.223** | 4.7e-7 | +0.219 | 7.6e-7 |
| 0.05 | 1.874 | **+0.265** | 1.6e-9 | +0.207 | 2.9e-6 |
| 0.1 | 2.151 | **+0.149** | 8.3e-4 | +0.139 | 1.8e-3 |

**This is the opposite sign from what the node-level mechanism predicts, at every
single `p_out`.** Every correlation is positive (and, with `n=500`, statistically
significant at every `p_out` -- small effect sizes, not noise), meaning that
*within* any one of these graphs, nodes whose neighbors are more community-mixed
have *slightly less negative* `m` than nodes whose neighbors are more homogeneous
-- the reverse of "heterogeneous neighbors -> more bimodal." This is reported
honestly rather than discarded: it does not match a naive node-level reading of
the mechanism, even though the *aggregate*, across-graph trend (rising `p_out` ->
rising average heterogeneity -> more negative `m_mean`) does.

## Reading it

**Mixed support, honestly split into two different claims the theory makes:**

1. **Aggregate (across-graph) claim -- supported.** As the whole graph's neighbor
   heterogeneity rises with `p_out`, TNBBeta's average bimodality (`m_mean`) does
   get monotonically more negative, exactly as predicted, across a 4.6x range of
   ground-truth heterogeneity (0.468 to 2.151 nats).
2. **Per-node (within-graph) claim -- not supported, in fact inverted.** The
   theory's finer-grained prediction -- that *within* a fixed graph, individual
   nodes with more heterogeneous neighbors should show more negative `m` than their
   more-homogeneous neighbors -- is contradicted at every single `p_out` tested
   here: the correlation is reliably positive, not negative, though small in
   magnitude (`r` between +0.13 and +0.27).
3. **The AUC-gap claim (TNBBeta's advantage growing with heterogeneity) -- not
   supported.** The gap stays small and sign-flipping against both baselines
   across the grid, and a 3-seed check at the one `p_out` with the largest single-
   seed gap (0.05) shows that gap is not robust to seed variation.

**Caveats:** single seed per `p_out` for the main table (the 3-seed check at
`p_out=0.05` is the only repeated point, and it already shows meaningful variance);
one fixed `(num_communities, nodes_per_community, p_in, latent_dim)` configuration,
not swept; `p_out=0.1`'s numbers are confounded by near-random-initialization
collapse in two of the four families. The aggregate trend in `m_mean` is real and
replicated across 5 distinct `p_out` values, but the sharper per-node and AUC-gap
predictions this investigation hoped to confirm directly are not borne out by this
particular synthetic dose-response -- a genuinely different, more qualified
picture than the real-dataset `featureless`-regime results this investigation's
other write-ups found.
