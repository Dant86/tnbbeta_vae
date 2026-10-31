# A wide search over synthetic graph types: where does TNBBeta beat tuned vMF and power-spherical? Results, 2026-10-31

- **Date:** 2026-10-31.
- **Code:** branch `synthetic-graph-families` (PR #69) at `1797492`; design, margins and tests are in
  `graph_search_preregistration_2026-10-31.md` (committed at `27d0e5d`, before any discovery run).
- **Commands** (all in the pre-registration, plus):

  ```bash
  uv run python -m apps.eval.graph_search_stats test --stage discovery           # job 2283814, 30,720 runs
  uv run python -m apps.synthetic.graph_search manifest --stage confirmation --configs discovery \
      --config-ids <the 34 configs with a BH rejection> --draws 32 --conditions tnbbeta vmf_init power_spherical_init
  STAGE=confirmation sbatch --array=0-23 scripts/slurm/graph_search.sbatch       # job 2284479, 3,264 runs
  uv run python -m apps.eval.graph_search_stats test --stage confirmation
  ```

## Summary

- **Discovery** (120 configs x 32 draws, primary comparators = tuned vMF and PS with kappa starting at `d`
  and trained): of 720 (config, metric) tests, **50 rejected at BH q <= 0.05** (36 at Holm), but also
  **120 where TNBBeta is significantly worse than both** comparators. No pooled per-family test is
  significant after Holm (smallest p = 0.19).
- **Confirmation** on fresh draws of exactly the 50 rejected pairs: **46 of 50 confirmed** (Holm, 0.05),
  with discovery effect sizes largely reproduced (no large shrinkage).
- **Where TNBBeta wins** is a specific kind of metric: 17 log loss, 17 calibration slope, 9 truth KL, 3
  uncertainty AUROC, and **none** on centre AUC or rank correlation (it is significantly *worse* on 19
  and 27 configs there). Its wins are about the probabilities, not the ranking.
- **Where** is not one graph type: the 46 confirmed pairs cover 30 configs in 9 of 10 families
  (geometric 10, bipartite 8, assortative 6, core-periphery 6, ring 5, degree-corrected 4, hierarchical 3,
  scale-free 2, mixed-membership 2), none in `latent_kernel`.
- Confirmed configs have a higher mean degree than the space overall (mean degree 20.5 for the log-loss
  wins against 15.2 over all 120 configs); no other single graph property separates them.

## Discovery: configs (of 12) with a BH rejection, better / worse, primary comparators

| family | centre_auc | sampled_log_loss | cal_slope_dev | uncertainty_auroc | truth_kl | rank_corr |
|---|---|---|---|---|---|---|
| assortative | 0/2 | 1/2 | 4/1 | 0/0 | 1/3 | 0/1 |
| bipartite | 0/2 | 6/0 | 0/1 | 0/0 | 2/2 | 0/4 |
| core_periphery | 0/0 | 3/2 | 2/0 | 1/0 | 1/2 | 0/1 |
| hierarchical | 0/1 | 0/2 | 3/0 | 0/0 | 0/2 | 0/2 |
| mixed_membership | 0/0 | 1/3 | 0/1 | 0/0 | 1/4 | 0/1 |
| latent_kernel | 0/2 | 0/1 | 0/1 | 0/0 | 0/0 | 0/3 |
| geometric | 0/3 | 2/4 | 4/3 | 2/0 | 2/3 | 0/4 |
| ring | 0/3 | 2/3 | 4/2 | 0/3 | 0/5 | 0/4 |
| scale_free | 0/1 | 3/0 | 0/4 | 0/1 | 1/0 | 0/0 |
| degree_corrected | 0/5 | 1/7 | 2/1 | 0/4 | 1/7 | 0/7 |
| **total** | **0/19** | **19/24** | **19/14** | **3/8** | **9/28** | **0/27** |

Secondary comparator sets (descriptive): against the kappa-pinned pair 125 better / 135 worse, against
the default-initialised pair 54 better / 82 worse. **45 of the 46 confirmed primary wins are also BH-better
against the pinned pair, and none is worse against it**, so these wins are not a failure of kappa
optimisation alone.

## Confirmation of the 50 discovery-rejected pairs (Holm, alpha = 0.05)

| config | metric | confirmed | p (Holm) | mean diff: discovery -> confirm |
|---|---|---|---|---|
| bipartite-03 | sampled_log_loss | yes | 1.16e-08 | vmf_init: +0.0376 -> +0.0359; power_spherical_init: +0.0175 -> +0.0205 |
| geometric-01 | sampled_log_loss | yes | 1.16e-08 | vmf_init: +0.0654 -> +0.0666; power_spherical_init: +0.1053 -> +0.1036 |
| geometric-05 | sampled_log_loss | yes | 1.16e-08 | vmf_init: +0.1419 -> +0.1385; power_spherical_init: +0.1900 -> +0.1904 |
| degree_corrected-00 | cal_slope_dev | yes | 1.16e-08 | vmf_init: +0.9648 -> +1.1549; power_spherical_init: +2.4035 -> +2.5128 |
| geometric-05 | cal_slope_dev | yes | 1.16e-08 | vmf_init: +2.3481 -> +2.3278; power_spherical_init: +3.6866 -> +3.8391 |
| geometric-05 | uncertainty_auroc | yes | 1.16e-08 | vmf_init: +0.1046 -> +0.1008; power_spherical_init: +0.1414 -> +0.1365 |
| bipartite-00 | truth_kl | yes | 1.16e-08 | vmf_init: +0.0368 -> +0.0371; power_spherical_init: +0.0155 -> +0.0217 |
| geometric-01 | truth_kl | yes | 1.16e-08 | vmf_init: +0.0932 -> +0.0983; power_spherical_init: +0.1359 -> +0.1417 |
| geometric-05 | truth_kl | yes | 1.16e-08 | vmf_init: +0.2011 -> +0.1869; power_spherical_init: +0.2513 -> +0.2349 |
| geometric-01 | cal_slope_dev | yes | 1.91e-08 | vmf_init: +1.2609 -> +1.2834; power_spherical_init: +2.4622 -> +2.3970 |
| bipartite-09 | sampled_log_loss | yes | 4.66e-08 | vmf_init: +0.0278 -> +0.0356; power_spherical_init: +0.0222 -> +0.0267 |
| geometric-01 | uncertainty_auroc | yes | 4.66e-08 | vmf_init: +0.0627 -> +0.0667; power_spherical_init: +0.0979 -> +0.0871 |
| assortative-03 | cal_slope_dev | yes | 8.85e-08 | vmf_init: +0.6480 -> +0.5576; power_spherical_init: +1.3928 -> +1.2344 |
| core_periphery-07 | sampled_log_loss | yes | 3.70e-07 | vmf_init: +0.0424 -> +0.0411; power_spherical_init: +0.0522 -> +0.0356 |
| geometric-10 | cal_slope_dev | yes | 3.70e-07 | vmf_init: +0.3613 -> +0.6303; power_spherical_init: +0.4798 -> +0.5125 |
| scale_free-08 | sampled_log_loss | yes | 8.96e-07 | vmf_init: +0.1132 -> +0.1276; power_spherical_init: +0.0787 -> +0.0776 |
| core_periphery-07 | cal_slope_dev | yes | 1.08e-06 | vmf_init: +0.6758 -> +0.6960; power_spherical_init: +0.7561 -> +0.5509 |
| geometric-09 | cal_slope_dev | yes | 1.08e-06 | vmf_init: +0.2158 -> +0.2093; power_spherical_init: +0.2363 -> +0.2921 |
| core_periphery-07 | uncertainty_auroc | yes | 1.26e-06 | vmf_init: +0.0919 -> +0.0933; power_spherical_init: +0.0569 -> +0.0696 |
| bipartite-02 | sampled_log_loss | yes | 2.22e-06 | vmf_init: +0.0326 -> +0.0272; power_spherical_init: +0.0181 -> +0.0143 |
| degree_corrected-03 | sampled_log_loss | yes | 7.47e-06 | vmf_init: +0.1674 -> +0.1872; power_spherical_init: +0.0948 -> +0.0892 |
| bipartite-09 | truth_kl | yes | 8.51e-06 | vmf_init: +0.0197 -> +0.0360; power_spherical_init: +0.0175 -> +0.0273 |
| degree_corrected-09 | cal_slope_dev | yes | 1.80e-05 | vmf_init: +0.3939 -> +0.4352; power_spherical_init: +0.6909 -> +0.6518 |
| scale_free-08 | truth_kl | yes | 2.02e-05 | vmf_init: +0.1359 -> +0.1548; power_spherical_init: +0.0978 -> +0.0949 |
| bipartite-00 | sampled_log_loss | yes | 2.25e-05 | vmf_init: +0.0425 -> +0.0396; power_spherical_init: +0.0198 -> +0.0188 |
| core_periphery-10 | sampled_log_loss | yes | 3.33e-05 | vmf_init: +0.0267 -> +0.0298; power_spherical_init: +0.0265 -> +0.0347 |
| bipartite-01 | sampled_log_loss | yes | 4.21e-05 | vmf_init: +0.0368 -> +0.0358; power_spherical_init: +0.0191 -> +0.0201 |
| ring-01 | sampled_log_loss | yes | 4.21e-05 | vmf_init: +0.0325 -> +0.0348; power_spherical_init: +0.0133 -> +0.0142 |
| mixed_membership-06 | truth_kl | yes | 5.77e-05 | vmf_init: +0.2391 -> +0.2417; power_spherical_init: +0.0965 -> +0.1051 |
| ring-03 | cal_slope_dev | yes | 7.13e-05 | vmf_init: +0.5627 -> +0.6625; power_spherical_init: +0.6209 -> +0.8353 |
| degree_corrected-03 | truth_kl | yes | 7.13e-05 | vmf_init: +0.2099 -> +0.2350; power_spherical_init: +0.1223 -> +0.1108 |
| ring-11 | cal_slope_dev | yes | 8.31e-05 | vmf_init: +0.8363 -> +0.6813; power_spherical_init: +1.9661 -> +1.6823 |
| assortative-07 | cal_slope_dev | yes | 1.28e-04 | vmf_init: +0.7948 -> +0.5783; power_spherical_init: +1.2408 -> +0.9109 |
| mixed_membership-06 | sampled_log_loss | yes | 2.42e-04 | vmf_init: +0.1858 -> +0.1882; power_spherical_init: +0.0700 -> +0.0760 |
| assortative-06 | truth_kl | yes | 3.52e-04 | vmf_init: +0.2150 -> +0.2095; power_spherical_init: +0.0773 -> +0.0991 |
| assortative-00 | cal_slope_dev | yes | 5.01e-04 | vmf_init: +1.2381 -> +0.8590; power_spherical_init: +2.0463 -> +1.7914 |
| bipartite-11 | sampled_log_loss | yes | 6.34e-04 | vmf_init: +0.0412 -> +0.0344; power_spherical_init: +0.0194 -> +0.0144 |
| core_periphery-05 | sampled_log_loss | yes | 9.57e-04 | vmf_init: +0.0475 -> +0.0501; power_spherical_init: +0.0365 -> +0.0328 |
| assortative-06 | sampled_log_loss | yes | 1.84e-03 | vmf_init: +0.1593 -> +0.1520; power_spherical_init: +0.0540 -> +0.0663 |
| hierarchical-08 | cal_slope_dev | yes | 2.38e-03 | vmf_init: +0.6917 -> +0.5864; power_spherical_init: +0.5494 -> +0.4904 |
| ring-11 | sampled_log_loss | yes | 3.03e-03 | vmf_init: +0.0449 -> +0.0539; power_spherical_init: +0.1051 -> +0.1086 |
| assortative-10 | cal_slope_dev | yes | 3.03e-03 | vmf_init: +1.0555 -> +0.8261; power_spherical_init: +1.5601 -> +1.4603 |
| core_periphery-05 | truth_kl | yes | 3.03e-03 | vmf_init: +0.0633 -> +0.0598; power_spherical_init: +0.0420 -> +0.0321 |
| hierarchical-00 | cal_slope_dev | yes | 3.71e-03 | vmf_init: +0.3891 -> +0.4455; power_spherical_init: +0.3230 -> +0.3808 |
| hierarchical-11 | cal_slope_dev | yes | 3.71e-03 | vmf_init: +0.7361 -> +0.8940; power_spherical_init: +1.4272 -> +0.6553 |
| ring-00 | cal_slope_dev | yes | 1.18e-02 | vmf_init: +0.2204 -> +0.2286; power_spherical_init: +0.2327 -> +0.3950 |
| scale_free-11 | sampled_log_loss | no | 5.32e-02 | vmf_init: +0.1209 -> +0.1170; power_spherical_init: +0.0471 -> +0.0415 |
| core_periphery-04 | cal_slope_dev | no | 6.81e-02 | vmf_init: +0.9013 -> +0.9199; power_spherical_init: +0.6735 -> +0.4592 |
| ring-04 | cal_slope_dev | no | 6.81e-02 | vmf_init: +0.2668 -> +0.2133; power_spherical_init: +0.5392 -> +0.3175 |
| scale_free-06 | sampled_log_loss | no | 4.74e-01 | vmf_init: +0.0971 -> +0.0491; power_spherical_init: +0.0287 -> -0.0003 |

Diagnostics outside the pre-registered tests follow; they are exploratory.

## Exploratory: divergences (not pre-registered)

146 of the 3,840 TNBBeta discovery runs diverged (non-finite loss or gradient); no vMF, PS or Gaussian run
did. They concentrate in a few configs: `hierarchical-00` in 32 of 32 draws, `scale_free-01` in 30 of 32,
`bipartite-05` in 16 of 32. Removing every draw with a TNBBeta divergence changes the primary-set counts
from 50 better / 120 worse to 43 better / 116 worse (714 tests), and gains nothing in either direction, so
the divergences do not drive the pattern. The divergences themselves are a TNBBeta stability problem tied
to specific graphs and are not explained here.

## Exploratory: what the confirmed wins have in common (not pre-registered; confirmation runs, 34 configs)

Posterior shape of TNBBeta by graph family (means over the confirmed configs' draws) and of vMF/PS:

| family | TNBBeta r_bar | second-mode mass | nodes with > 10% in second mode | m | vMF r_bar | PS r_bar |
|---|---|---|---|---|---|---|
| assortative | 0.66 | 0.100 | 0.27 | -4.01 | 0.67 | 0.60 |
| bipartite | 0.24 | 0.309 | 0.72 | -5.43 | 0.65 | 0.56 |
| core_periphery | 0.75 | 0.008 | 0.02 | -1.67 | 0.67 | 0.70 |
| degree_corrected | 0.42 | 0.200 | 0.49 | -5.21 | 0.65 | 0.58 |
| geometric | 0.95 | 0.000 | 0.00 | -3.30 | 0.73 | 0.68 |
| hierarchical | 0.86 | 0.023 | 0.06 | -3.22 | 0.68 | 0.72 |
| mixed_membership | 0.34 | 0.233 | 0.57 | -4.17 | 0.63 | 0.58 |
| ring | 0.73 | 0.094 | 0.23 | -4.69 | 0.73 | 0.67 |
| scale_free | 0.44 | 0.171 | 0.45 | -3.73 | 0.55 | 0.54 |

- Unlike on `latent_kernel` (second-mode mass 0.000 there), TNBBeta **does** put substantial mass in a
  second mode on the graphs where its posterior is diffuse (bipartite, mixed-membership, degree-corrected,
  scale-free), and none where it is tight (geometric, core-periphery, hierarchical).
- The confirmed wins appear in **both** regimes: geometric / core-periphery / `ring-11` wins have
  essentially no second mode (6 of the 17 confirmed log-loss configs have second-mode mass below 0.02),
  and bipartite / degree-corrected / mixed-membership wins have a lot. Within a config, second-mode mass
  does not track the log-loss advantage over vMF (within-config Spearman 0.04), so there is no sign that
  using the second mode is what produces the win.
- **What does stand out:** vMF and PS posteriors stay close to the same concentration on every graph
  (r_bar 0.55-0.73, even though kappa starts at `d` and is trained with a tuned learning rate), while
  TNBBeta spans 0.24-0.95. So a hypothesis that fits all of this is that **TNBBeta reaches the
  concentration a graph needs (tight on geometric, diffuse on bipartite) and vMF/PS do not move far
  from where they started**, giving better-calibrated probabilities but no better ranking. This is a
  hypothesis from post-hoc data, not a finding; it predicts that vMF/PS with kappa fixed at a
  per-graph good value should close the gap.

## Limits

- The 120 "TNBBeta worse" results are discovery-stage only (not confirmed) and outnumber the wins on
  ranking-type metrics; this search does not show TNBBeta is better in general.
- Per-config tests had low power for small effects (see the pre-registration's power table), so a
  non-rejection is not evidence of no difference; the confirmed pairs are the large ones.
- All graphs are undirected and unweighted; the decoder is a symmetric dot product.
- `cal_slope_dev` wins are large in magnitude (mean differences up to 3.8) because vMF/PS calibration
  slopes are far from 1; the slope is clipped to [0, 20] and the test is rank-based, so a few extreme
  runs do not drive it.
