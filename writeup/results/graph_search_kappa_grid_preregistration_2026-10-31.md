# Pre-registration: does a tuned fixed kappa, or recalibration, remove TNBBeta's wins?

Date: 2026-10-31. Written and committed before any `kappa_grid` run. Code: branch
`synthetic-graph-families` (PR #69); the commit that adds this file is the pre-registered
state. Follows `graph_search_results_2026-10-31.md` (46 confirmed pairs; exploratory
characterisation with two regimes).

## Question
The 46 confirmed pairs compare TNBBeta with vMF/PS whose kappa starts at d = 16 and is
trained. The exploratory characterisation suggests the wins are about *concentration*
(tight regime: comparators under-confident; diffuse regime: over-confident) and may
partly be *temperature*. Two things could erase them: a better kappa, or a one-parameter
recalibration. If either does, TNBBeta's edge is reachability of concentration, not
distribution shape.

## Design
- Configs: the 30 with a confirmed pair (ids below), on fresh graphs: seed base
  30,000,000, 32 draws each (paired by graph, split and seed across conditions).
- Conditions (the `--conditions` list below): `tnbbeta`, `vmf_init`, `power_spherical_init` (re-run on the new draws) and
  vMF / PS with kappa fixed at each of {4, 8, 16, 32, 64, 128} (`vmf_k<kappa>`,
  `power_spherical_k<kappa>`): 15 conditions, 14,400 runs. Learning rate: the pilot's
  tuned rate of the matching `_init` condition. All else as discovery (500 epochs, d=16,
  decoder bias on).
- New metrics on every run: `recal_log_loss` and `recal_truth_kl`. The mean sampled
  prediction is mapped by `p -> sigmoid(a + b * logit(p))`, with (a, b) fitted on that
  model's validation pairs (the same pairs used for early stopping, so the fit is
  slightly optimistic for every model alike), then scored on the test pairs / true
  probabilities.

## Confirmatory hypotheses
Margins and the one-sided paired Wilcoxon signed-rank test are those of the main
pre-registration. Each test is an intersection-union over vMF and PS.
- **H-grid** (per confirmed (config, metric) pair, 46 pairs, Holm over the 46): TNBBeta
  beats the best fixed-kappa vMF and the best fixed-kappa PS. The best kappa per
  (config, metric, family) is chosen on one half of the 32 draws (even or odd index) and
  scored on the other half, both ways, pooling the 32 paired differences. Null: TNBBeta
  is no better than one of them by the margin. A pair that survives is one whose win is
  **not** explained by kappa.
- **H-recal** (30 configs x {`recal_log_loss`, `recal_truth_kl`} = 60 tests, Holm over
  the 60): TNBBeta beats recalibrated `vmf_init` and recalibrated `power_spherical_init`.
  A test that survives is a win **not** explained by temperature. Calibration slope is
  not tested after recalibration (it is fitted to 1 by construction).
- Interpretation rule, fixed now: if fewer than a quarter of the H-grid pairs survive,
  the reachability reading is supported for the metrics that fail; pairs that survive
  stay unexplained. Failing to reject is **not** evidence that TNBBeta is no better:
  with 32 draws the power table of the main pre-registration applies.

## Descriptive only (not tested)
- Spearman of TNBBeta's trained `r_bar` against the `r_bar` of the best fixed-kappa vMF
  (by log loss) across configs: the concentration hypothesis predicts a positive value.
- Reverse-direction results (TNBBeta worse) and the Gaussian are not part of this study.
- Divergent TNBBeta runs are kept (as in the main study).

## Commands
```
CFG="assortative-00 assortative-03 assortative-06 assortative-07 assortative-10 \
bipartite-00 bipartite-01 bipartite-02 bipartite-03 bipartite-09 bipartite-11 \
core_periphery-05 core_periphery-07 core_periphery-10 degree_corrected-00 \
degree_corrected-03 degree_corrected-09 geometric-01 geometric-05 geometric-09 \
geometric-10 hierarchical-00 hierarchical-08 hierarchical-11 mixed_membership-06 \
ring-00 ring-01 ring-03 ring-11 scale_free-08"
CONDS="tnbbeta vmf_init power_spherical_init"
for k in 4 8 16 32 64 128; do CONDS="$CONDS vmf_k$k power_spherical_k$k"; done
uv run python -m apps.synthetic.graph_search manifest --stage kappa_grid \
    --configs discovery --config-ids $CFG --draws 32 --conditions $CONDS
STAGE=kappa_grid sbatch --array=0-23 scripts/slurm/graph_search.sbatch
uv run python -m apps.eval.graph_search_kappa report --output <report.md>
```
