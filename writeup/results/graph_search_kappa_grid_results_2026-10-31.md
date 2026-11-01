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