# scale_free density figure: results, 2026-11-03

- **Date:** 2026-11-03. **Pre-registration:** `scale_free_density_figure_preregistration_2026-11-03.md` (frozen before any density run; Deviations appended there, all presentation or clarification).
- **Base commit:** `dd8ede08a749e580db13a43b5d92e4c31233af73` on `synthetic-graph-families`; the working tree has UNCOMMITTED work (this study: `apps/eval/scale_free_density_figure.py`, `tests/apps/test_scale_free_density_figure.py`, `scripts/slurm/scale_free_density_aux.sbatch`, additive `density` stage in `apps/synthetic/graph_search.py`, lint fixes in `apps/eval/per_family_slope_check.py` and `apps/eval/bipartite_slope_check.py`; plus other agents' and earlier uncommitted files, including `exact_envelope=True` as the default of `VonMisesFisher`, which these runs used).
- **Commands:**

  ```bash
  uv run python -m apps.eval.scale_free_density_figure configs
  uv run python -m apps.synthetic.graph_search manifest --stage density --draws 32 --conditions tnbbeta vmf_init power_spherical_init   # 288 runs
  TOTAL_TASKS=12 STAGE=density sbatch --array=0-11 scripts/slurm/graph_search_cpu.sbatch          # job 2304500, all 288 runs finished, no failures
  AUX=analyze sbatch scripts/slurm/scale_free_density_aux.sbatch                                   # 2304532 (first run, E_c H1 check +0.479), 2304533 (final)
  uv run python -m apps.eval.scale_free_density_figure plot --fig-dir writeup/results/figures      # login node (kaleido)
  AUX=pytest sbatch ...  # 2304513, 2304536, 2304560 (new + graph_search tests); AUX=full 2304538: 1181 passed, 1 skipped
  ```
  Figure: `writeup/results/figures/scale_free_density_figure_2026-11-03.png` (3x scale, `alptex`, legend at the bottom). Raw: `$TNBBETA_CHECKPOINT_DIR/graph_search/density/`, `.../metric_first/density_results.json`.
- Lint: ruff format clean; ruff check clean except the untracked scratch file `scratch_degree_correlation.py` (D205, not mine, not to be committed); pyright 0 errors on the whole repo.

## Headline (read this before the figure)

**The new 3-level experiment does NOT show a confirmed, monotone "denser => TNBBeta better" pattern.** Point estimates move the right way (truth-KL dz -0.34, +0.31, +0.82 for LOW, MED, HIGH; sampled log loss -0.41, +0.16, +0.81), and at HIGH density TNBBeta is better on truth KL and log loss, but after BH over the 15 tests **only one cell is significant: HIGH sampled log loss (better, q = 0.006)**. HIGH truth KL has q = 0.070; no cell is significantly worse; no AUROC cell is BH-significant (centre AUROC at LOW, dz -0.44, q = 0.097, is the nearest loss). The descriptive trend over the 96 draws (Spearman of per-draw truth-KL advantage vs realised degree +0.20, 95% bootstrap CI [-0.02, +0.41]) is positive but its interval touches 0, and within each level there is no relationship (-0.04 to +0.14). Pairing the MED and HIGH point estimates against the per-draw-min comparator shrinks them (truth KL HIGH +0.0101 vs +0.0589 raw), i.e. the HIGH advantage is largely against PowerSpherical (dz +0.82; vs vMF +0.50) and depends on which comparator is taken. At LOW density TNBBeta is worse on all three probability metrics in point estimate (mean-difference intervals exclude 0 for the bootstrap, but the Wilcoxon p-values are 0.38-0.96: heavy-tailed paired differences, a few draws dominate; the two tests disagree and neither is BH-significant). Honest summary: the figure illustrates a trend consistent with the pooled panel, but **the pooled panel (within-family Spearman +0.462 on S_c = E_c / KL0, reproduced exactly from the cached 200 fresh configs) remains the evidence; the 3-level bars are suggestive, with one significant cell.** The pooled panel used the library vMF sampler, the bars the exact one.

Confound note: the expected-degree CV falls with density (mean 1.28, 1.10, 0.95 for LOW, MED, HIGH; the pre-freeze check with 8 draws had suggested 1.03 to 0.88), a larger drift than assumed, because clipping trims the hubs; clustering rises (0.037, 0.075, 0.142) with density by construction. So "density" here is partly density plus lower CV and more clustering, as in the real configs. Oracle AUC is held at about 0.757 (seed-0 draw).

## Level covariates (mean over draws; realised from the sampled graphs)

| level | nominal degree | realised degree | expected-degree CV | clustering | oracle AUC (calibration draw) |
|---|---|---|---|---|---|
| low | 6 | 5.99 | 1.279 | 0.037 | 0.757 |
| med | 15 | 15.00 | 1.098 | 0.075 | 0.757 |
| high | 35 | 34.92 | 0.948 | 0.142 | 0.756 |

## Paired effects (positive = TNBBeta better; comparator = better level-mean of vmf_init / power_spherical_init)

| level | metric | comparator | draws (dropped) | mean diff [95% CI] | dz [95% CI] | Wilcoxon p | BH q | label | dz vmf / dz PS | per-draw-min mean (dz) |
|---|---|---|---|---|---|---|---|---|---|---|
| low | truth_kl | power_spherical_init | 32 (0) | -0.1114 [-0.2350, -0.0104] | -0.34 [-0.56, -0.06] | 0.3794 | 0.6470 | neither | +0.61 / -0.34 | -0.1126 (-0.34) |
| low | ece | power_spherical_init | 32 (0) | -0.0505 [-0.0980, -0.0071] | -0.38 [-0.65, -0.08] | 0.4105 | 0.6470 | neither | +1.03 / -0.38 | -0.0505 (-0.38) |
| low | sampled_log_loss | power_spherical_init | 32 (0) | -0.1605 [-0.3073, -0.0393] | -0.41 [-0.62, -0.19] | 0.9632 | 0.9632 | neither | +0.46 / -0.41 | -0.1605 (-0.41) |
| low | sampled_auc | vmf_init | 32 (0) | -0.0103 [-0.0363, +0.0149] | -0.14 [-0.51, +0.22] | 0.5483 | 0.6470 | neither | -0.14 / +0.13 | -0.0218 (-0.32) |
| low | centre_auc | vmf_init | 32 (0) | -0.0215 [-0.0379, -0.0040] | -0.44 [-0.91, -0.08] | 0.0194 | 0.0972 | neither | -0.44 / -0.15 | -0.0356 (-0.92) |
| med | truth_kl | power_spherical_init | 32 (0) | +0.0196 [-0.0024, +0.0400] | +0.31 [-0.03, +0.79] | 0.0595 | 0.1489 | neither | +1.30 / +0.31 | +0.0174 (+0.30) |
| med | ece | power_spherical_init | 32 (0) | -0.0099 [-0.0320, +0.0126] | -0.15 [-0.54, +0.19] | 0.4768 | 0.6470 | neither | +1.27 / -0.15 | -0.0116 (-0.18) |
| med | sampled_log_loss | power_spherical_init | 32 (0) | +0.0066 [-0.0082, +0.0200] | +0.16 [-0.17, +0.68] | 0.0280 | 0.1049 | neither | +1.32 / +0.16 | +0.0060 (+0.15) |
| med | sampled_auc | vmf_init | 32 (0) | +0.0246 [-0.0018, +0.0502] | +0.32 [-0.02, +0.81] | 0.0475 | 0.1426 | neither | +0.32 / +0.41 | +0.0106 (+0.15) |
| med | centre_auc | vmf_init | 32 (0) | +0.0004 [-0.0100, +0.0113] | +0.01 [-0.36, +0.36] | 0.9632 | 0.9632 | neither | +0.01 / +0.55 | -0.0058 (-0.21) |
| high | truth_kl | power_spherical_init | 32 (0) | +0.0589 [+0.0351, +0.0835] | +0.82 [+0.54, +1.18] | 0.0093 | 0.0700 | neither | +0.50 / +0.82 | +0.0101 (+0.25) |
| high | ece | power_spherical_init | 32 (0) | +0.0128 [-0.0191, +0.0440] | +0.14 [-0.23, +0.49] | 0.4429 | 0.6470 | neither | +0.21 / +0.14 | -0.0280 (-0.43) |
| high | sampled_log_loss | power_spherical_init | 32 (0) | +0.0302 [+0.0177, +0.0432] | +0.81 [+0.52, +1.16] | 0.0004 | 0.0060 | better | +0.30 / +0.81 | -0.0065 (-0.22) |
| high | sampled_auc | vmf_init | 32 (0) | +0.0157 [-0.0205, +0.0504] | +0.15 [-0.19, +0.60] | 0.1838 | 0.3938 | neither | +0.15 / +0.53 | +0.0039 (+0.04) |
| high | centre_auc | vmf_init | 32 (0) | -0.0003 [-0.0064, +0.0053] | -0.02 [-0.33, +0.40] | 0.5608 | 0.6470 | neither | -0.02 / +0.71 | -0.0018 (-0.11) |

## Panel summaries (auto-generated figure text)

- low: BH-sig.: probability metrics better 0/3, worse 0/3; AUROC better 0/2, worse 0/2
- med: BH-sig.: probability metrics better 0/3, worse 0/3; AUROC better 0/2, worse 0/2
- high: BH-sig.: probability metrics better 1/3, worse 0/3; AUROC better 0/2, worse 0/2

## Descriptive trends (NOT confirmatory; Spearman of per-draw advantage vs per-draw realised degree and vs level index, over all draws)

| metric | rho vs degree [95% CI] | rho vs level [95% CI] | within low | within med | within high |
|---|---|---|---|---|---|
| truth_kl | +0.20 [-0.02, +0.41] | +0.20 [-0.02, +0.42] | -0.00 | -0.04 | +0.14 |
| ece | +0.11 [-0.12, +0.32] | +0.11 [-0.11, +0.33] | -0.11 | +0.06 | -0.05 |
| sampled_log_loss | +0.19 [-0.05, +0.39] | +0.20 [-0.02, +0.40] | -0.10 | -0.03 | +0.03 |
| sampled_auc | +0.15 [-0.05, +0.34] | +0.16 [-0.05, +0.36] | +0.04 | -0.02 | -0.02 |
| centre_auc | +0.22 [+0.01, +0.41] | +0.25 [+0.05, +0.44] | -0.10 | -0.08 | -0.08 |

## Pooled panel check

Within-family Spearman of S_c = E_c / KL0 (the registered H1 outcome) vs log nominal degree over 200 fresh configs: +0.462 (registered: +0.462); with E_c in nats instead: +0.479.