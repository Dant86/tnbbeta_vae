# Pairwise/BCE-only ablation on synthetic clusters: does GraphVAE's loss alone drive bimodality? 2026-10-08

- **Date:** 2026-10-08
- **Branch/commit:** `dtd-oriented-textures`, `1d8a28b` ("Add
  apps.synthetic.pairwise_cluster_recovery: the (b) ablation script") --
  `master` did not yet have this code at the time of this run.
- **Commands:**
  ```
  uv run python -m apps.synthetic.pairwise_cluster_recovery \
      --out-dir runs/pairwise_cluster_recovery_final --epochs 2000 --seed 0
  ```
  (plus two additional diagnostic-only runs at `--seed 1` and `--seed 2`,
  same other defaults, used only to check seed-to-seed stability of the
  TNBBeta diagnostic -- reported below.)

## Background

This project's research has found `TNBBetaSpherical`'s posterior goes
genuinely bimodal (`m = epsilon - (latent_dim - 1) / 2 < 0`) specifically on
`GraphVAE` link-prediction runs (Cora, Citeseer, Pubmed, com-DBLP -- all
four), and specifically does NOT on every reconstruction-style VAE task
tried (MNIST, `axial_mixture`, DTD). The hypothesis under test here: it's
`GraphVAE`'s training objective itself (pure pairwise dot-product +
`binary_cross_entropy_with_logits` on edges, no decoder/reconstruction term
at all) that drives this, independent of graph structure, scale, or real
features.

This script ("(b)" ablation arm) holds the architecture family (a plain MLP
encoder, no message-passing/no graph at all) and data (synthetic
`cluster_mixture`, no real features beyond what correlates with cluster
identity) fixed, and swaps ONLY the loss to `GraphVAE`'s exact pairwise
dot-product + BCE objective, with no decoder. Default scale: 10 clusters x
50 points/cluster = 500 items, `dim=20`, `latent_dim=16` (matching the
Planetoid/com-DBLP `GraphVAE` runs exactly, for direct `m`/`epsilon`
comparability).

## Result (seed 0, 2000 epochs)

| family | test_auc | test_ap | entropy_mean | r_bar |
|---|---|---|---|---|
| gaussian | 0.9478 | 0.9100 | -3.125 | 1.648 |
| vmf | 0.9343 | 0.8641 | -12.459 | 0.914 |
| power_spherical | 0.9456 | 0.8896 | -12.624 | 0.916 |
| tnbbeta | 0.9461 | 0.8851 | NaN (no closed form) | 0.909 |

TNBBeta-only diagnostic: `p_mean = 0.4073`, `q_mean = 0.1200`,
`epsilon_mean = 16.155`, **`m_mean = 8.655`**, **`frac_bimodal = 0.0`**.

All four families solve this task easily (AUC ~0.93-0.95, as expected for
well-separated synthetic clusters) and training never diverges. But
`m_mean` is strongly *positive*, not negative -- the posterior is not
bimodal by the proven criterion, at any node (`frac_bimodal = 0.0`,
identical to the pattern seen on every reconstruction-style VAE task, not
the real `GraphVAE` pattern this ablation was built to test for.

## The epsilon-drift finding (why "pick enough epochs to converge" is itself not simple here)

Inspecting training curves (same hyperparameters, `tnbbeta` family, seeds 0,
1, 2) shows the **link loss plateaus by ~epoch 200** (0.44-0.47, noisy
thereafter, never improving further), but **epsilon keeps climbing
monotonically well past that point, for as long as training continues** --
it never stabilizes in the 2000 epochs explored, in any of the three seeds:

| epoch | seed 0: eps_mean (m_mean, frac_bimodal) | seed 1: eps_mean (m_mean, frac_bimodal) | seed 2: eps_mean (m_mean, frac_bimodal) |
|---|---|---|---|
| 0 | 1.74 (-5.77, 1.00) | 1.21 (-6.29, 1.00) | 1.41 (-6.09, 1.00) |
| 200 | 9.03 (1.53, 0.14) | 9.17 (1.67, 0.29) | 5.20 (**-2.31**, 1.00) |
| 1000 | 12.76 (5.26, 0.00) | 11.66 (4.16, 0.20) | 7.55 (0.05, 0.47) |
| 2000 | 15.19 (7.69, 0.00) | 13.27 (5.77, 0.00) | 9.14 (1.64, 0.24) |

Every seed starts deep in the bimodal regime purely from initialization
(`epsilon` is initialized small, well below the `latent_dim=16` threshold of
7.5), then `epsilon` climbs steadily throughout training in every seed,
crossing the threshold and continuing upward. Seed 2 happens to sit
genuinely bimodal (`frac_bimodal = 1.0`, `m_mean = -2.31`) at epoch 200 --
if that one run had been stopped there (a plausible "epochs looked converged
by the loss" choice), it would have looked like a clean confirmation of the
hypothesis. By epoch 2000 the same seed has drifted to `m_mean = +1.64`,
`frac_bimodal = 0.24`, the opposite reading. Seeds 0 and 1 never cross back
into bimodal after their early transient.

This directly undercuts treating any single training-horizon's `m_mean` as
"the" answer: there is no epoch at which the TNBBeta diagnostic stabilizes
here, in contrast to the real `GraphVAE` checkpoints this investigation is
built around, which are already-converged, stable, and reported as robustly
bimodal across all four real datasets. The reported headline numbers above
(seed 0, 2000 epochs) are deep in the unimodal direction and were still
drifting further unimodal when the run stopped -- i.e. 2000 epochs
understates, if anything, how far from bimodal this ablation's converged
state would end up.

## Reading it

**The clean version of the hypothesis -- "the pairwise/ranking loss alone,
independent of graph structure, drives bimodality" -- is not supported by
this ablation.** At this synthetic scale, with a plain MLP encoder and no
graph, training TNBBeta on exactly `GraphVAE`'s loss does not produce a
stable bimodal posterior; if anything, the long-run trend (consistent across
all three seeds checked) is toward *more* unimodal (`epsilon` growing
without bound), the opposite direction. The task is easy enough (AUC ~0.94
across every family) that the pairwise/BCE loss has essentially no pressure
pushing epsilon down once directions are well-separated -- increasing
concentration only sharpens already-correct predictions, and the small
KL-divided-by-500-items term here is evidently too weak, at this `lr=0.01`
schedule, to hold epsilon back the way it apparently does (or is held back by
something else) on the real graphs.

**This argues for "something else about GraphVAE specifically" over "the
pairwise/ranking loss itself," at least as the sole explanation.** Candidates
this ablation does NOT control for and that remain live: the GCN
architecture itself (two-layer, spectral message-passing, vs. a plain MLP
here), the much larger item counts and much sparser degree structure of the
real graphs (500 items / 2000 positive pairs here vs. com-DBLP's 317K nodes
and real degree distributions), and the fact that a real citation/
co-authorship graph's local neighborhood structure is reused by every node's
own GCN receptive field in a way this independent-item MLP encoder cannot
replicate. This ablation isolates the loss function alone and finds it
insufficient by itself at this scale; it does not yet distinguish which of
those remaining candidates (if any) is doing the real work, and the
epsilon-drift finding above means any follow-up here needs to report a
drift curve, not a single epoch count, to be interpretable.

**Honest caveat on scale and run budget:** only three seeds and one
`(num_clusters, points_per_cluster, dim, latent_dim)` configuration were
explored under this time budget. The seed-to-seed variability already
visible here (seed 2's extended bimodal plateau vs. seeds 0/1's much
shorter one) suggests this ablation's outcome may be sensitive to
initialization and/or scale in ways the real multi-dataset `GraphVAE`
result was not -- that sensitivity is itself part of the answer, not just
noise to average away.
