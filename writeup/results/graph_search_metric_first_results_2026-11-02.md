# Metric-first re-analysis of the wide graph-type search: results, 2026-11-02

- **Date:** 2026-11-02.
- **Pre-registration:** `graph_search_metric_first_preregistration_2026-11-02.md` (frozen before any fresh training; Deviations appended there).
- **Base commit:** `6b1880f` on `synthetic-graph-families`; the code of this study is **uncommitted** (new: `src/tnbbeta_vae/models/search_metrics.py`, `src/tnbbeta_vae/data/graph_covariates.py`, `src/tnbbeta_vae/regularised.py`, `apps/eval/graph_search_covariates.py`, `apps/eval/graph_search_metric_first.py`, `apps/eval/graph_search_metric_first_plot.py`, `scripts/slurm/graph_search_covariates.sbatch`, `scripts/slurm/graph_search_cpu.sbatch`, their tests; additive edits to `apps/synthetic/graph_search.py`: `fresh` stage, `EXTENDED_STAGES`, `_extended`).
- **Commands:**

  ```bash
  uv run python -m apps.synthetic.graph_search configs --name fresh --per-family 20 --seed 13          # job 2297788
  uv run python -m apps.synthetic.graph_search manifest --stage fresh --draws 16 --conditions tnbbeta vmf_init power_spherical_init vmf_pin power_spherical_pin
  STAGE=discovery DRAWS=32 sbatch scripts/slurm/graph_search_covariates.sbatch                          # job 2297787
  STAGE=fresh DRAWS=16 sbatch scripts/slurm/graph_search_covariates.sbatch                              # job 2297794
  STAGE=fresh sbatch --array=0-23 scripts/slurm/graph_search.sbatch                                     # job 2297873 (GPU; tasks 0,1 ran, 2-23 failed on a node problem)
  TOTAL_TASKS=48 STAGE=fresh sbatch --array=0-47 scripts/slurm/graph_search_cpu.sbatch                  # job 2298489 (CPU, remaining runs)
  uv run python -m apps.eval.graph_search_metric_first cache --stage discovery                          # job 2297824
  uv run python -m apps.eval.graph_search_metric_first explore --workers 16                             # job 2297912, exploratory
  uv run python -m apps.eval.graph_search_metric_first cache --stage fresh && ... fresh                 # login node (Deviation 6)
  uv run python -m apps.eval.graph_search_metric_first_plot --fig-dir writeup/results/figures
  ```

  Cancelled (my own ids only): 2298193, 2298313 (pending GPU arrays superseded by the CPU array), 2297815 (a test job hung by a test bug, fixed), 2298870 (pending analysis job).
- **Fresh stage:** 200 configs (20 per family) x 16 draws x 5 conditions = 16,000 runs, all finished, no failures; tuned learning rates unchanged. 1,344 runs trained on GPU, 14,656 on CPU (Deviation 5). No TNBBeta run diverged in the fresh stage.

## Headline

**Confirmatory (fresh configs, pre-registered):**

1. **No graph-covariate rule transfers across graph types.** The pre-registered rule is degenerate: leave-family-out R^2 on discovery is exactly 0.000 (block-permutation p = 0.998), the elastic net kept no covariate, so `rho_fresh` is undefined and the decision rule fails at criterion (1). **We claim no pattern from the leave-family-out analysis.**
2. **H1 is confirmed:** within a graph family, TNBBeta's truth-KL advantage rises with mean degree (log nominal degree; within-family Spearman +0.462, one-sided within-family permutation p = 0.0001, 200 fresh configs). This is the earlier lead (20.5 vs 15.2) reproduced on fresh graphs; it is a within-family effect, not a between-family one (the leave-family-out model found nothing).
3. **Per family, on truth KL (`E_c` median, bootstrap over configs):** TNBBeta beats the less favourable of tuned vMF/PS in **bipartite only** (median +0.0134 nats, 95% [+0.0087, +0.0317]); it is **worse** in hierarchical, mixed-membership and latent-kernel (intervals entirely below 0); inconclusive elsewhere (scale-free +0.029 with an interval spanning 0).
4. **Config level (Holm over the 400 confirmatory tests):** truth KL 8 better / 11 worse of 200 configs, sampled log loss 10 better / 12 worse. Against the kappa-pinned pair: 19/13 and 26/18. Consistent with the 2026-10-31 search: no consistent TNBBeta advantage; sparse wins.

**Metric map (descriptive; BH, fresh, config counts better / worse of 200 over all families):** TNBBeta's advantage is on **probability-quality and calibration metrics**: ECE 31/8, truth_kl_w 28/22, truth_kl at eps 1e-4 23/17, calibration-slope deviation 23/14, Hits@50 on the sampled prediction 22/9, graph-statistic distances (clustering 21/2, triangles 20/5, degree W1 16/9), Hits@10 10/4, MRR 9/4. It **loses on ranking and latent recovery**: rank correlation 3/24, sampled AUC 8/21, centre AUC 2/17, latent Gram correlation 0/8 and Procrustes 0/2 (latent_kernel, geometric, ring), overlapping NMI 3/11, centre-scored Hits/MRR 1/7 and 0/6. The graph-statistic and ECE gains are concentrated in ring, degree-corrected, assortative and bipartite configs (see matrix). These are descriptive counts, not confirmatory.

**What the better-than-comparators configs share (descriptive, fresh, tiers from the truth-KL interval vs +-0.005 nats):** 47 win / 99 inconclusive / 54 loss configs; median realised mean degree 19.3 / 12.9 / 8.9, median degree dispersion (CV of expected degree) 0.68 / 0.58 / 0.21, median clustering 0.084 / 0.059 / 0.038; no difference in size or oracle AUC (log n 6.48 / 6.31 / 6.48; AUC 0.78 / 0.76 / 0.78). On the model side TNBBeta's posterior on win configs is more diffuse (median r_bar 0.46 / 0.70 / 0.78) with more second-mode mass (0.124 / 0.080 / 0.045); the discovery stage shows the same ordering. **These co-occur with wins; they are not shown to cause them.** In short: denser, more degree-heterogeneous, more clustered graphs, and bipartite (a representability control); sparse, homogeneous graphs (latent-kernel, hierarchical, mixed-membership) are where TNBBeta loses.

**Exploratory (discovery, 120 configs, stored metrics, BH):** reproduces the earlier search (truth KL 9 better / 28 worse; log loss 19/24); family means of the effect are positive only for bipartite and scale-free; within-family covariates carry signal (repeated-CV within-family R^2 = +0.174; E_c in nats leave-family-out R^2 = +0.113, no permutation p computed for that secondary).

**Post hoc supplement (not pre-registered):** a model using discovery family means plus a within-family elastic net, fitted on discovery, scored on fresh: R^2_fresh +0.252, Spearman +0.500 (family means alone: +0.077, +0.277). The largest within-family coefficients are log nominal degree (+0.83), oracle AUC (+0.66), informative features (+0.52), affinity effective rank (-0.25): consistent with H1, but this model uses family identity and is not a test of the pre-registered claim.

## Fresh stage, confirmatory evaluation

## Prediction rule on fresh configs (outcome S_c)

- R^2_fresh = +0.000 (baseline: mean discovery S_c = -0.716)
- rho_fresh (Spearman predicted vs observed) = +nan; lower 95% bound (5th percentile of 10000 within-family bootstrap resamples) = +nan
- within-fresh-family Spearman (family means removed) = +nan
- top coefficient `log_n`: fresh Spearman with S_c = -0.112 (sign does not match)
- H1 (log_nominal_degree, within family): statistic +0.462, one-sided permutation p = 0.0001

## Decision rule

1. discovery R^2_LOFO > 0, block p <= 0.05, > 25% of reliability: R^2_LOFO = +0.000, p = 0.9980 -> False
3. rho_fresh >= 0.30, lower bound > 0, R^2_fresh > 0: False
4. top-cluster sign matches: False

### Confirmatory family-level statement

| family | configs | median E_c (nats, truth_kl) | 95% interval | above 0 |
|---|---|---|---|---|
| assortative | 20 | +0.0066 | [-0.0252, +0.0471] | False |
| bipartite | 20 | +0.0134 | [+0.0087, +0.0317] | True |
| core_periphery | 20 | -0.0083 | [-0.0429, +0.0159] | False |
| hierarchical | 20 | -0.0151 | [-0.0690, -0.0056] | False |
| mixed_membership | 20 | -0.0170 | [-0.0660, -0.0041] | False |
| latent_kernel | 20 | -0.0314 | [-0.0601, -0.0087] | False |
| geometric | 20 | -0.0030 | [-0.0235, +0.0200] | False |
| ring | 20 | -0.0239 | [-0.0336, +0.0021] | False |
| scale_free | 20 | +0.0290 | [-0.0043, +0.0592] | False |
| degree_corrected | 20 | -0.0056 | [-0.0319, +0.0192] | False |

### Config x metric matrix on the fresh stage

(cells: better / worse / neither configs)

| family | truth_kl | sampled_log_loss | truth_kl_w | recal_truth_kl | truth_kl_eps1e4 | cal_slope_dev | ece | hard_hits10 | hard_hits50 | hardc_hits10 | hardc_hits50 | hard_mrr | hardc_mrr | gs_degree_w1 | gs_clustering_err | gs_triangle_err | latent_gram_rho | latent_nmi | latent_procrustes | centre_auc | sampled_auc | rank_corr | uncertainty_auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| assortative | 3/1/16 | 1/0/19 | 4/1/15 | 0/2/18 | 6/1/13 | 7/2/11 | 3/0/17 | 0/0/20 | 0/3/17 | 0/0/20 | 0/2/18 | 1/0/19 | 0/0/20 | 1/5/14 | 3/1/16 | 1/3/16 | - | 0/1/19 | - | 0/1/19 | 1/0/19 | 1/1/18 | 1/0/19 |
| bipartite | 0/0/20 | 4/0/16 | 13/0/7 | 0/0/20 | 2/0/18 | 0/1/19 | 12/0/8 | 5/0/15 | 12/0/8 | 0/3/17 | 4/1/15 | 3/0/17 | 0/2/18 | 3/0/17 | 0/1/19 | 1/0/19 | - | 3/2/15 | - | 2/4/14 | 1/2/17 | 2/4/14 | 0/0/20 |
| core_periphery | 0/0/20 | 2/0/18 | 1/0/19 | 3/0/17 | 4/0/16 | 2/0/18 | 1/0/19 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 1/0/19 | 0/0/20 | 1/0/19 | 2/0/18 | 2/0/18 | - | 0/0/20 | - | 0/0/20 | 1/0/19 | 0/0/20 | 1/0/19 |
| hierarchical | 0/0/20 | 0/1/19 | 1/2/17 | 1/1/18 | 1/1/18 | 0/0/20 | 0/3/17 | 0/0/20 | 0/1/19 | 0/0/20 | 0/0/20 | 1/0/19 | 0/0/20 | 1/1/18 | 0/0/20 | 0/0/20 | - | 0/0/20 | - | 0/2/18 | 1/2/17 | 0/3/17 | 1/0/19 |
| mixed_membership | 0/0/20 | 0/0/20 | 0/4/16 | 0/0/20 | 0/2/18 | 0/2/18 | 0/1/19 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 1/0/19 | 0/0/20 | 0/0/20 | - | 0/1/19 | - | 0/0/20 | 0/0/20 | 0/1/19 | 0/0/20 |
| latent_kernel | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 1/0/19 | 0/0/20 | 0/1/19 | 0/0/20 | - | 0/2/18 | 0/0/20 | 0/0/20 | 0/2/18 | 0/0/20 |
| geometric | 3/1/16 | 2/2/16 | 2/3/15 | 2/0/18 | 4/1/15 | 8/1/11 | 1/3/16 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/0/20 | 0/2/18 | 4/0/16 | 2/1/17 | 0/2/18 | - | - | 0/1/19 | 1/0/19 | 0/3/17 | 2/0/18 |
| ring | 0/4/16 | 0/4/16 | 3/7/10 | 0/1/19 | 1/6/13 | 4/1/15 | 7/0/13 | 2/2/16 | 5/2/13 | 1/3/16 | 2/3/15 | 2/2/16 | 0/3/17 | 6/0/14 | 8/0/12 | 8/0/12 | 0/6/14 | - | - | 0/4/16 | 0/7/13 | 0/4/16 | 0/1/19 |
| scale_free | 1/0/19 | 0/0/20 | 1/0/19 | 2/0/18 | 3/0/17 | 0/4/16 | 0/0/20 | 2/0/18 | 0/0/20 | 0/0/20 | 0/0/20 | 1/0/19 | 0/0/20 | 1/0/19 | 0/0/20 | 0/0/20 | - | - | - | 0/0/20 | 2/0/18 | 0/0/20 | 0/0/20 |
| degree_corrected | 1/5/14 | 1/5/14 | 3/5/12 | 0/4/16 | 2/6/12 | 2/3/15 | 7/1/12 | 1/2/17 | 5/3/12 | 0/1/19 | 2/3/15 | 0/2/18 | 0/1/19 | 1/1/18 | 4/0/16 | 6/0/14 | - | 0/7/13 | - | 0/5/15 | 1/10/9 | 0/6/14 | 0/1/19 |
| all | 8/11/181 | 10/12/178 | 28/22/150 | 8/8/184 | 23/17/160 | 23/14/163 | 31/8/161 | 10/4/186 | 22/9/169 | 1/7/192 | 8/9/183 | 9/4/187 | 0/6/194 | 16/9/175 | 21/2/177 | 20/5/175 | 0/8/52 | 3/11/106 | 0/2/18 | 2/17/181 | 8/21/171 | 3/24/173 | 5/2/193 |

Median standardised effect dz of the less favourable comparator:

| family | truth_kl | sampled_log_loss | truth_kl_w | recal_truth_kl | truth_kl_eps1e4 | cal_slope_dev | ece | hard_hits10 | hard_hits50 | hardc_hits10 | hardc_hits50 | hard_mrr | hardc_mrr | gs_degree_w1 | gs_clustering_err | gs_triangle_err | latent_gram_rho | latent_nmi | latent_procrustes | centre_auc | sampled_auc | rank_corr | uncertainty_auroc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| assortative | +0.21 | -0.06 | -0.19 | -0.14 | +0.21 | +0.51 | +0.65 | -0.46 | -0.70 | -0.08 | -0.32 | -0.45 | -0.04 | +0.07 | -0.19 | -0.09 | - | -0.24 | - | -0.30 | -0.21 | -0.28 | -0.29 |
| bipartite | +0.62 | +1.19 | +1.35 | +0.28 | +0.62 | -0.06 | +1.30 | +1.22 | +2.16 | -0.34 | +0.37 | +1.27 | -0.05 | +0.88 | -0.12 | -0.02 | - | +0.53 | - | -0.48 | -0.14 | -0.69 | -0.08 |
| core_periphery | -0.22 | -0.11 | -0.35 | +0.22 | -0.22 | -0.27 | -0.18 | +0.16 | -0.36 | -0.32 | -0.25 | +0.24 | -0.25 | +0.32 | +0.22 | +0.07 | - | -0.26 | - | -0.29 | +0.01 | -0.38 | -0.04 |
| hierarchical | -0.23 | -0.30 | -0.32 | -0.34 | -0.22 | +0.02 | -0.16 | -0.44 | -0.63 | -0.11 | -0.19 | -0.37 | -0.31 | -0.06 | +0.03 | +0.12 | - | -0.47 | - | -0.25 | -0.30 | -0.57 | +0.03 |
| mixed_membership | -0.46 | -0.43 | -0.47 | -0.15 | -0.46 | -0.82 | -0.40 | -0.35 | -0.45 | -0.08 | -0.17 | -0.22 | +0.02 | -0.02 | -0.14 | +0.16 | - | -0.14 | - | -0.11 | -0.11 | -0.23 | +0.01 |
| latent_kernel | -0.37 | -0.50 | -0.53 | -0.38 | -0.37 | -0.31 | -0.51 | -0.50 | -0.66 | -0.18 | -0.47 | -0.29 | -0.21 | -0.09 | -0.11 | -0.51 | -0.64 | - | -0.85 | -0.46 | -0.40 | -0.74 | -0.18 |
| geometric | -0.15 | -0.30 | -0.36 | -0.37 | -0.15 | +0.28 | -0.26 | -0.35 | -0.47 | -0.45 | -0.36 | -0.43 | -0.26 | -0.42 | +0.15 | -0.19 | -0.65 | - | - | -0.39 | -0.24 | -0.68 | +0.07 |
| ring | -0.46 | -0.06 | -0.13 | +0.12 | -0.46 | -0.46 | +0.60 | -0.18 | -0.30 | -0.27 | -0.12 | +0.03 | -0.21 | +0.08 | +1.33 | +0.75 | -1.22 | - | - | -0.51 | -0.62 | -0.53 | -0.41 |
| scale_free | +0.21 | -0.04 | -0.05 | +0.15 | +0.21 | -0.74 | +0.30 | +0.39 | +0.09 | -0.09 | -0.05 | +0.46 | -0.14 | +0.66 | -0.69 | -0.29 | - | - | - | -0.34 | +0.17 | -0.15 | -0.37 |
| degree_corrected | -0.14 | -0.18 | -0.20 | -0.54 | -0.14 | -0.55 | +0.71 | +0.00 | -0.74 | -0.21 | -0.32 | +0.16 | -0.16 | +0.33 | +0.07 | +0.36 | - | -0.46 | - | -0.64 | -1.01 | -0.90 | -0.54 |
| all | -0.16 | -0.21 | -0.27 | -0.16 | -0.16 | -0.26 | +0.13 | -0.04 | -0.40 | -0.18 | -0.19 | +0.02 | -0.16 | +0.19 | -0.05 | +0.07 | -0.78 | -0.23 | -0.85 | -0.37 | -0.29 | -0.53 | -0.18 |

### Pinned-kappa comparator set (descriptive)

(cells: better / worse / neither configs)

| family | truth_kl | sampled_log_loss |
|---|---|---|
| assortative | 2/0/18 | 3/1/16 |
| bipartite | 0/0/20 | 4/0/16 |
| core_periphery | 1/0/19 | 4/0/16 |
| hierarchical | 1/1/18 | 1/2/17 |
| mixed_membership | 1/2/17 | 1/2/17 |
| latent_kernel | 5/1/14 | 3/2/15 |
| geometric | 8/1/11 | 6/2/12 |
| ring | 0/3/17 | 3/4/13 |
| scale_free | 1/0/19 | 1/0/19 |
| degree_corrected | 0/5/15 | 0/5/15 |
| all | 19/13/168 | 26/18/156 |

Median standardised effect dz of the less favourable comparator:

| family | truth_kl | sampled_log_loss |
|---|---|---|
| assortative | +0.37 | +0.05 |
| bipartite | +0.58 | +1.16 |
| core_periphery | -0.07 | +0.13 |
| hierarchical | -0.30 | -0.32 |
| mixed_membership | -0.32 | -0.37 |
| latent_kernel | -0.31 | -0.48 |
| geometric | +0.81 | -0.16 |
| ring | -0.28 | +0.60 |
| scale_free | +0.36 | -0.02 |
| degree_corrected | -0.23 | -0.09 |
| all | -0.00 | -0.04 |

### Post hoc supplement

Fitted on discovery only; scored on fresh configs against the same baseline (mean discovery S_c). The pre-registered leave-family-out rule cannot use family identity; fresh configs share the ten families, so this is a different, easier question.

- discovery family means alone: R^2_fresh = +0.077; Spearman with observed S_c = +0.277
- family means + within-family elastic net (penalty 0.08619, 15 nonzero coefficients): R^2_fresh = +0.252; Spearman = +0.500
- nonzero within-family coefficients (per SD of the family-centred covariate): log_n +0.0107, log_nominal_degree +0.8313, log_groups -0.0997, hetero_level +0.1927, informative_features +0.5165, oracle_auc +0.6598, expected_degree_cv -0.1696, clustering +0.1558, assortativity -0.1358, affinity_erank -0.2529, logit_rank16 +0.0072, high_prob_fraction +0.2309, clip_fraction +0.0695, leaf_fraction +0.0720, giant_fraction -0.0462

Figures: `figures/metric_first_family_by_metric_2026-11-02.png` (fresh median dz and better/worse counts), `figures/metric_first_truth_kl_by_config_2026-11-02.png` (per-config truth-KL effect, discovery and fresh), `figures/metric_first_covariates_2026-11-02.png` (S_c against the three covariates with the largest rule weights; the rule is all-zero, so these are the first three by magnitude of zero, i.e. arbitrary: do not read them as selected), `figures/metric_first_fresh_prediction_2026-11-02.png` (degenerate: constant prediction).

## Exploratory analysis of the discovery configs

## Step 1: predictability (outcome S_c, elastic net, leave-one-family-out)

- R^2_LOFO = +0.000; block-permutation p = 0.9980 (2000 permutations); plain-permutation p (optimistic) = 0.9970 (1000)
- reliability ceiling (Spearman-Brown split-half) = 0.900; 25% of it = 0.225
- secondary: outcome E_c in nats R^2_LOFO = +0.113
- secondary: excluding bipartite R^2_LOFO = +0.000
- secondary: excluding 10 configs with TNBBeta divergence rate > 10% R^2_LOFO = +0.000
- secondary: within-family increment (repeated 4-fold, 20x) R^2 = +0.174

### Step 2 and 3

No covariate cluster was selected (full-fit coefficients all zero at the one-SE penalty 2.02 in every leave-family-out fit); no partial dependence is reported.

## Step 4: descriptive characterisation by tier (labels are descriptive only)

| family | win | inconclusive | loss |
|---|---|---|---|
| assortative | 3 | 7 | 2 |
| bipartite | 5 | 6 | 1 |
| core_periphery | 3 | 4 | 5 |
| hierarchical | 0 | 6 | 6 |
| mixed_membership | 1 | 5 | 6 |
| latent_kernel | 1 | 3 | 8 |
| geometric | 3 | 2 | 7 |
| ring | 2 | 5 | 5 |
| scale_free | 6 | 6 | 0 |
| degree_corrected | 2 | 3 | 7 |

| covariate | median (win) | median (inconclusive) | median (loss) |
|---|---|---|---|
| log_n | 6.397 | 6.310 | 6.620 |
| log_nominal_degree | 3.011 | 2.262 | 2.241 |
| log_groups | 0.347 | 1.099 | 1.609 |
| uses_groups | 0.500 | 1.000 | 1.000 |
| hetero_level | 0.000 | 1.000 | 0.000 |
| informative_features | 0.500 | 1.000 | 1.000 |
| oracle_auc | 0.766 | 0.779 | 0.788 |
| realised_degree | 20.345 | 9.630 | 9.406 |
| expected_degree_cv | 0.664 | 0.603 | 0.205 |
| clustering | 0.090 | 0.060 | 0.038 |
| assortativity | -0.027 | -0.013 | -0.006 |
| spectral_gap | 0.899 | 0.588 | 0.477 |
| affinity_erank | 34.971 | 67.274 | 113.823 |
| logit_rank16 | 1.000 | 1.000 | 0.999 |
| high_prob_fraction | 0.261 | 0.278 | 0.245 |
| mass_top1pct | 0.098 | 0.105 | 0.122 |
| clip_fraction | 0.000 | 0.000 | 0.000 |
| log_kl0 | -3.005 | -2.741 | -2.714 |
| leaf_fraction | 0.000 | 0.004 | 0.007 |
| giant_fraction | 1.000 | 0.999 | 0.999 |
| post_p_mean | 0.502 | 0.500 | 0.498 |
| post_q_mean | 0.794 | 0.837 | 0.849 |
| post_epsilon_mean | 2.999 | 2.872 | 3.186 |
| post_m_mean | -4.501 | -4.628 | -4.314 |
| post_p_std | 0.029 | 0.024 | 0.018 |
| post_epsilon_std | 0.655 | 0.527 | 0.536 |
| post_frac_bimodal | 0.996 | 0.998 | 1.000 |
| post_r_bar | 0.489 | 0.611 | 0.810 |
| second_mode_mass | 0.139 | 0.073 | 0.036 |
| frac_nodes_second_mode | 0.386 | 0.182 | 0.094 |
| _best_epoch | 162.062 | 177.188 | 157.188 |
| _diverged | 0.000 | 0.000 | 0.000 |
| vmf_init_r_bar | 0.623 | 0.674 | 0.654 |
| power_spherical_init_r_bar | 0.530 | 0.624 | 0.641 |

## Step 5: model-side description (co-occurrence only)

| model-side variable | Spearman with S_c | within-family (family means removed from ranks) |
|---|---|---|
| post_p_mean | +0.03 | -0.01 |
| post_q_mean | +0.07 | +0.13 |
| post_epsilon_mean | +0.16 | +0.29 |
| post_m_mean | +0.16 | +0.29 |
| post_p_std | +0.21 | +0.26 |
| post_epsilon_std | +0.21 | +0.27 |
| post_frac_bimodal | -0.25 | -0.28 |
| post_r_bar | -0.10 | +0.03 |
| second_mode_mass | -0.06 | -0.21 |
| frac_nodes_second_mode | -0.04 | -0.18 |
| _best_epoch | +0.15 | +0.17 |
| _diverged | +0.16 | +0.12 |
| vmf_init_r_bar | -0.10 | -0.16 |
| power_spherical_init_r_bar | -0.12 | -0.02 |

## Config x metric matrix (discovery, stored metrics, BH over all tests)

(cells: better / worse / neither configs)

| family | truth_kl | sampled_log_loss | cal_slope_dev | centre_auc | sampled_auc | rank_corr | uncertainty_auroc |
|---|---|---|---|---|---|---|---|
| assortative | 1/3/8 | 1/2/9 | 4/1/7 | 0/2/10 | 0/1/11 | 0/1/11 | 0/0/12 |
| bipartite | 2/2/8 | 6/0/6 | 0/1/11 | 0/2/10 | 3/1/8 | 0/4/8 | 0/0/12 |
| core_periphery | 1/2/9 | 3/2/7 | 2/0/10 | 0/0/12 | 2/0/10 | 0/1/11 | 1/0/11 |
| hierarchical | 0/2/10 | 0/2/10 | 3/0/9 | 0/1/11 | 0/2/10 | 0/2/10 | 0/0/12 |
| mixed_membership | 1/4/7 | 1/3/8 | 0/1/11 | 0/0/12 | 0/0/12 | 0/1/11 | 0/0/12 |
| latent_kernel | 0/0/12 | 0/1/11 | 0/1/11 | 0/2/10 | 0/0/12 | 0/3/9 | 0/0/12 |
| geometric | 2/3/7 | 2/4/6 | 4/3/5 | 0/3/9 | 2/3/7 | 0/4/8 | 2/0/10 |
| ring | 0/5/7 | 2/3/7 | 4/2/6 | 0/3/9 | 0/4/8 | 0/4/8 | 0/3/9 |
| scale_free | 1/0/11 | 3/0/9 | 0/4/8 | 0/1/11 | 2/1/9 | 0/0/12 | 0/1/11 |
| degree_corrected | 1/7/4 | 1/7/4 | 2/1/9 | 0/5/7 | 1/7/4 | 0/7/5 | 0/4/8 |
| all | 9/28/83 | 19/24/77 | 19/14/87 | 0/19/101 | 10/19/91 | 0/27/93 | 3/8/109 |

Median standardised effect dz of the less favourable comparator:

| family | truth_kl | sampled_log_loss | cal_slope_dev | centre_auc | sampled_auc | rank_corr | uncertainty_auroc |
|---|---|---|---|---|---|---|---|
| assortative | -0.11 | -0.26 | +0.02 | -0.31 | -0.44 | -0.35 | -0.20 |
| bipartite | +0.37 | +0.61 | +0.12 | -0.17 | +0.11 | -0.27 | +0.13 |
| core_periphery | -0.10 | -0.14 | -0.19 | -0.39 | -0.13 | -0.26 | -0.07 |
| hierarchical | -0.33 | -0.36 | +0.05 | -0.36 | -0.36 | -0.42 | -0.10 |
| mixed_membership | -0.35 | -0.47 | -0.35 | -0.26 | -0.47 | -0.23 | -0.29 |
| latent_kernel | -0.79 | -0.76 | -0.83 | -0.18 | -0.15 | -0.49 | -0.40 |
| geometric | -0.43 | -0.52 | -0.03 | -0.27 | -0.24 | -0.54 | +0.22 |
| ring | -0.12 | +0.08 | +0.17 | -0.36 | -0.25 | -0.34 | -0.23 |
| scale_free | +0.31 | +0.39 | -0.60 | -0.20 | +0.23 | -0.10 | -0.16 |
| degree_corrected | -1.48 | -1.03 | -0.32 | -0.68 | -1.27 | -1.56 | -0.46 |
| all | -0.24 | -0.28 | -0.20 | -0.27 | -0.26 | -0.37 | -0.19 |

## Risks and caveats realised

- **The leave-family-out claim failed**, so the "what do winning graphs share" answer is only descriptive and within-family (H1, tier medians, the post hoc supplement). With 10 families the between-family sample is 10.
- **vMF sampler caveat:** the library `VonMisesFisher` rejection sampler was found biased for kappa about 10-40 at manifold dimension >= 9 (dimension-scaling study, its Deviation D2). These runs use ambient dimension 16 with vMF kappa near 16, inside that regime, so vMF comparators may be handicapped; the effect on the `min` over comparators is unknown. Not corrected.
- **Device change** between discovery (GPU) and most fresh runs (CPU): not tested for a device effect.
- **Graph-statistic distances** use the implied probability `rho p / (1 - p)`, which amplifies errors where p is near 1; they are descriptive. `truth_kl` decile columns are mostly empty for block families (Deviation 4).
- **Intervals and counts** carry the usual caveats: K = 16 draws per config, per-config Holm tests are low-power (minimum exact Wilcoxon p = 1.5e-5 against a 1.25e-4 threshold), draws are replicates of one config.
- **Tuned comparators:** learning rates were tuned on `sampled_log_loss` per family on a small pilot; the pinned-kappa set is the only control for kappa, not for learning rate.
- **The truth-KL scale `S_c`** (headroom share) is negative for most families (median S_c in discovery -0.42 to -0.77 in several), i.e. TNBBeta's KL exceeds the better comparator's by a large fraction of the constant-predictor headroom on many configs.
