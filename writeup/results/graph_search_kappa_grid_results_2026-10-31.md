# Results: tuned fixed kappa and recalibration vs TNBBeta's graph-search wins

Date: 2026-10-31. Pre-registration: `graph_search_kappa_grid_preregistration_2026-10-31.md`
(commit `0c33c09`). Run on branch `synthetic-graph-families` (PR #69), job 2285769 and its
no-GPU resubmissions; 14,400 of 14,400 runs, 0 failed. Commands are in the
pre-registration; report: `uv run python -m apps.eval.graph_search_kappa report`.

## Outcome (confirmatory)
- **H-grid: 8 of 46 confirmed pairs survive** against the cross-fitted best fixed-kappa
  vMF and PS (Holm). By metric: calibration slope 0 of 17, log loss 4 of 17, truth KL
  2 of 9, uncertainty AUROC 2 of 3. By the pre-registered rule (fewer than a quarter
  survive) the concentration-reachability reading is supported for the failing pairs.
- **H-recal: 15 of 60 survive** against recalibrated vMF/PS (log loss 7 of 30, truth KL
  8 of 30). On 23 of the 60 TNBBeta is worse on average than both recalibrated models.
- Descriptive: Spearman of TNBBeta's `r_bar` against the `r_bar` of the best fixed-kappa
  vMF is +0.72 over 30 configs.

## Reading
- Every calibration-slope win goes away once kappa is tuned: the comparators'
  calibration problem was their kappa, not their distribution.
- The surviving grid pairs are mostly `geometric` (log loss, KL, uncertainty), plus
  `core_periphery-05/07` and `bipartite-09`. Caution: for `geometric-05` and several
  other tight configs the best kappa was 128, the edge of the grid, so a tighter kappa
  might also close those gaps. The grid should be extended before calling any survivor
  unexplained.
- The bipartite, mixed-membership and scale-free wins (diffuse regime) are matched by
  kappa = 4, the loosest grid value; there too the grid's lower edge was chosen, so
  even looser kappa may match or beat TNBBeta.
- Non-rejection is not evidence of equal performance (32 draws; see the power table in
  the main pre-registration). Several mean differences are still positive for TNBBeta.
- Recalibration uses validation pairs also used for early stopping; the same for every
  model, but it makes the recalibrated numbers slightly optimistic.

## Full report
## kappa-grid stage: 14400 runs, 30 configs, 46 confirmed pairs

### Family grid: TNBBeta vs cross-fitted best kappa (Holm, alpha=0.05)
- still better: 8 of 46
  - cal_slope_dev: 0 of 17
  - sampled_log_loss: 4 of 17
  - truth_kl: 2 of 9
  - uncertainty_auroc: 2 of 3

| config | metric | p (Holm) | mean diff vMF-k / PS-k | chosen kappa |
|---|---|---|---|---|
| geometric-01 | truth_kl | 2.14e-08 | +0.0561 / +0.0698 | 32; 32,64 |
| geometric-01 | sampled_log_loss | 9.47e-06 | +0.0247 / +0.0265 | 32; 64 |
| bipartite-09 | sampled_log_loss | 1.10e-05 | +0.0218 / +0.0195 | 4,8; 8 |
| geometric-05 | truth_kl | 1.48e-05 | +0.0822 / +0.0802 | 128,64; 128 |
| geometric-05 | sampled_log_loss | 3.14e-05 | +0.0367 / +0.0356 | 128,64; 128 |
| core_periphery-05 | sampled_log_loss | 2.59e-04 | +0.0491 / +0.0494 | 16; 32 |
| geometric-05 | uncertainty_auroc | 3.60e-04 | +0.0477 / +0.0521 | 128,64; 128,64 |
| core_periphery-07 | uncertainty_auroc | 1.95e-03 | +0.0340 / +0.0412 | 128; 128 |
| degree_corrected-03 | sampled_log_loss | 2.00e-01 | +0.0151 / +0.0138 | 4; 4 |
| geometric-05 | cal_slope_dev | 4.44e-01 | +0.1354 / +0.2661 | 128; 128 |
| core_periphery-10 | sampled_log_loss | 6.14e-01 | +0.0292 / +0.0344 | 32,64; 128,64 |
| assortative-00 | cal_slope_dev | 1.00e+00 | -0.5011 / -0.3182 | 128; 128 |
| assortative-03 | cal_slope_dev | 1.00e+00 | -0.0575 / -0.0555 | 64; 128,64 |
| assortative-06 | sampled_log_loss | 1.00e+00 | +0.0099 / +0.0089 | 4; 4 |
| assortative-06 | truth_kl | 1.00e+00 | +0.0047 / +0.0022 | 4; 4 |
| assortative-07 | cal_slope_dev | 1.00e+00 | -0.2948 / -0.1464 | 128; 128 |
| assortative-10 | cal_slope_dev | 1.00e+00 | -0.0904 / -0.0705 | 128,64; 128 |
| bipartite-00 | sampled_log_loss | 1.00e+00 | -0.0000 / +0.0020 | 4; 4 |
| bipartite-00 | truth_kl | 1.00e+00 | +0.0006 / +0.0007 | 4; 4 |
| bipartite-01 | sampled_log_loss | 1.00e+00 | +0.0060 / +0.0069 | 4; 4,8 |
| bipartite-02 | sampled_log_loss | 1.00e+00 | +0.0011 / -0.0001 | 4; 4 |
| bipartite-03 | sampled_log_loss | 1.00e+00 | +0.0017 / +0.0010 | 4; 4 |
| bipartite-09 | truth_kl | 1.00e+00 | +0.0080 / +0.0089 | 4; 4 |
| bipartite-11 | sampled_log_loss | 1.00e+00 | -0.0000 / -0.0016 | 4; 4 |
| core_periphery-05 | truth_kl | 1.00e+00 | +0.0160 / +0.0119 | 8; 16 |
| core_periphery-07 | cal_slope_dev | 1.00e+00 | -0.0621 / -0.0471 | 128; 128 |
| core_periphery-07 | sampled_log_loss | 1.00e+00 | +0.0094 / +0.0332 | 64; 32,64 |
| degree_corrected-00 | cal_slope_dev | 1.00e+00 | -0.3449 / -0.2475 | 128; 128 |
| degree_corrected-03 | truth_kl | 1.00e+00 | +0.0093 / +0.0057 | 4; 4 |
| degree_corrected-09 | cal_slope_dev | 1.00e+00 | -0.1337 / -0.1469 | 64; 128 |
| geometric-01 | cal_slope_dev | 1.00e+00 | -0.3773 / -0.3225 | 128; 128 |
| geometric-01 | uncertainty_auroc | 1.00e+00 | +0.0169 / +0.0211 | 32; 128 |
| geometric-09 | cal_slope_dev | 1.00e+00 | +0.0591 / +0.1575 | 128; 128 |
| geometric-10 | cal_slope_dev | 1.00e+00 | -0.0533 / +0.1268 | 128; 128 |
| hierarchical-00 | cal_slope_dev | 1.00e+00 | -0.0114 / +0.0090 | 32,64; 128,64 |
| hierarchical-08 | cal_slope_dev | 1.00e+00 | +0.0009 / -0.0059 | 32,64; 64 |
| hierarchical-11 | cal_slope_dev | 1.00e+00 | -0.2743 / -0.1643 | 128; 128 |
| mixed_membership-06 | sampled_log_loss | 1.00e+00 | +0.0106 / +0.0089 | 4; 4 |
| mixed_membership-06 | truth_kl | 1.00e+00 | +0.0112 / +0.0071 | 4; 4 |
| ring-00 | cal_slope_dev | 1.00e+00 | +0.0302 / +0.1036 | 128; 128 |
| ring-01 | sampled_log_loss | 1.00e+00 | +0.0008 / -0.0004 | 4; 4 |
| ring-03 | cal_slope_dev | 1.00e+00 | +0.0062 / +0.1245 | 128; 128 |
| ring-11 | cal_slope_dev | 1.00e+00 | -0.6798 / -0.5974 | 128; 128 |
| ring-11 | sampled_log_loss | 1.00e+00 | -0.0296 / -0.0210 | 128; 128 |
| scale_free-08 | sampled_log_loss | 1.00e+00 | +0.0080 / +0.0083 | 4; 4 |
| scale_free-08 | truth_kl | 1.00e+00 | +0.0049 / +0.0050 | 4; 4 |

### Family recal: TNBBeta vs recalibrated vMF/PS (Holm over 60)
- still better: 15 of 60
  - recal_log_loss: 7 of 30
  - recal_truth_kl: 8 of 30
- configs-metrics where TNBBeta is worse on average than both: 23

### Concentration check (descriptive)
- configs: 30; Spearman of TNBBeta r_bar vs the r_bar of the best vMF kappa (by log loss): +0.72
## Post hoc: is TNBBeta's extra shape capacity used? (2026-11-01, exploratory)
Script: scratchpad `smm_matched.py` (reads the kappa_grid runs). For each config, TNBBeta's
mean `second_mode_mass` is compared with vMF / PS `second_mode_mass` interpolated, within
the same config, at the same mean `r_bar` along the fixed-kappa grid. Only configs whose
TNBBeta `r_bar` falls inside the comparator grid's range can be compared (18 for vMF, 15
for PS of 30).
- TNBBeta has more second-mode mass than a unimodal cap at the same `r_bar`: 0.066 vs
  0.026 (vMF), 0.097 vs 0.031 (PS); never lower by more than 0.02, higher by more than
  0.02 in 10 and 12 configs (Wilcoxon p = 8e-6, 6e-5; configs are not independent
  tests and this was not pre-registered).
- The excess is largest at mid concentration (`r_bar` 0.45-0.8): assortative-00 0.128 vs
  0.000, bipartite-01 0.129 vs 0.023. The configs with the largest excess are not
  among the 8 H-grid survivors.
- Six configs (bipartite, degree-corrected-00, ring-01) have `r_bar` = 0.10, more
  diffuse than any comparator on the grid (second-mode mass 0.43, near uniform's 0.5).
  The geometric and hierarchical-08 configs have `r_bar` 0.95-0.99, tighter than the
  kappa = 128 comparator, which explains why the best kappa sat at the grid edge there:
  TNBBeta's cap lies outside the grid at both ends.
- The H-grid survivors with a `geometric` config have second-mode mass 0.000: those wins
  are not bimodal.

## Post hoc: many TNBBeta posteriors never leave their initialisation (2026-11-01, exploratory)
Scripts: scratchpad `plateau.py` (reads the 7,680 tnbbeta and vmf_init discovery runs) and
`init_rbar.py` (untrained models on a `bipartite-00` draw, three seeds).
- The untrained TNBBeta head already gives `r_bar` 0.097, second-mode mass 0.43 and
  E|z.mu| 0.49 -- exactly the numbers the axis diagnostic found at `r_bar` = 0.10. Untrained
  vMF / PS (kappa = 16) give `r_bar` 0.614 / 0.523, E|z.mu| 0.61 / 0.52, matching the
  trained values there. So in those six configs no family's posterior moved.
- Discovery TNBBeta runs: `r_bar` is bimodal (spike at 0.095-0.11, spike at 0.9-1.0).
  1,002 of 3,840 runs (26%) sit at `r_bar` < 0.12, with median best epoch 6 (rest: 207) and
  mean validation AUC 0.607 (rest: 0.717). Share at that plateau by family:
  bipartite 0.61, degree_corrected 0.59, mixed_membership 0.35, ring 0.34, scale_free
  0.31, assortative 0.13, geometric 0.10, latent_kernel 0.09, hierarchical 0.08,
  core_periphery 0.02. Plateau fraction per config is 0 (40 configs), mixed (65), 1 (15).
- Median trained vMF_init `r_bar` is 0.624 against 0.614 untrained: vMF barely moves too.
- Mean `r_bar` rises with learning rate (0.43 at 0.003, 0.69 at 0.01, 0.78 at 0.03), and
  the pilot picked the lowest rate (0.003) for TNBBeta on bipartite, degree_corrected and
  ring, the families with the largest plateau shares. The pilot tuned on sampled log loss,
  which a stuck, diffuse posterior may favour (hypothesis, not tested).
- Consequence: the diffuse-regime wins are largely a comparison of initial posteriors
  (TNBBeta's is diffuse, kappa = 16's is not), consistent with kappa = 4 matching them.
