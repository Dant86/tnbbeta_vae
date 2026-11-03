# Graph-search learning-rate grid and log-loss tuning check (exploratory)

Date: 2026-11-01. Base commit: `ec2677e` (working tree has uncommitted ring files; unrelated).
Exploratory and post hoc; nothing here changes the pre-registered analysis.

## Question

Why does TNBBeta get lr 0.003 (the grid floor) in bipartite / ring / degree_corrected,
where posteriors stay near initialisation? Is the tuning criterion or the log-loss
evaluation at fault?

## Facts read from code and pilot data

* Grid: `LEARNING_RATES = (0.003, 0.01, 0.03)` (`apps/synthetic/graph_search.py:70`).
* `tune` picks the lr with the lowest mean `sampled_log_loss` per (family, condition)
  (`apps/eval/graph_search_stats.py`, `TUNING_METRIC`). Checkpoints, however, are the
  best-validation-AUC epoch (`run_once`), so selection and tuning use different criteria.
* `model_config_for` turns `decoder_bias=True` on for every condition, so a missing
  logit offset is NOT the explanation.
* Pilot TNBBeta tuned lr: 0.003 for bipartite, ring, degree_corrected; 0.01 for
  assortative, core_periphery, mixed_membership, latent_kernel, geometric, scale_free;
  0.03 for hierarchical. vMF/PS are mostly 0.003.
* Pilot (8 runs/cell): at lr 0.003 TNBBeta has `post_r_bar` 0.1-0.5 against 0.6-0.8 for
  vMF (which starts from `initial_kappa`), and clearly lower AUC in e.g. assortative
  (0.630 vs 0.756 at lr 0.01), core_periphery (0.534 vs 0.642), scale_free (0.524 vs 0.642).

## Check: raw vs Platt-recalibrated log loss across lr

Command (scratch script `lr_recal.py`, local, 1 thread, nice 10; it calls
`graph_search.run_item` on pilot manifest items, draws 000 and 001 of the two pilot
configs per family, TNBBeta and vmf_init only; 4 runs per cell below):

    PYTHONPATH=. OMP_NUM_THREADS=1 nice -n 10 uv run python lr_recal.py ring
    PYTHONPATH=. OMP_NUM_THREADS=1 nice -n 10 uv run python lr_recal.py degree_corrected

| family | cond | lr | raw log loss | recal log loss | AUC |
|---|---|---|---|---|---|
| ring | tnbbeta | 0.003 | 0.610 | 0.608 | 0.652 |
| ring | tnbbeta | 0.01 | 0.629 | 0.605 | 0.706 |
| ring | tnbbeta | 0.03 | 0.646 | 0.602 | 0.709 |
| ring | vmf_init | 0.003 | 0.601 | 0.594 | 0.710 |
| ring | vmf_init | 0.01 | 0.744 | 0.682 | 0.699 |
| ring | vmf_init | 0.03 | 0.802 | 0.698 | 0.699 |
| degree_corrected | tnbbeta | 0.003 | 0.718 | 0.691 | 0.508 |
| degree_corrected | tnbbeta | 0.01 | 0.749 | 0.693 | 0.511 |
| degree_corrected | tnbbeta | 0.03 | 1.623 | 1.603 | 0.589 |
| degree_corrected | vmf_init | 0.003 | 0.750 | 0.688 | 0.575 |
| degree_corrected | vmf_init | 0.01 | 0.784 | 0.694 | 0.552 |
| degree_corrected | vmf_init | 0.03 | 0.782 | 0.698 | 0.554 |

## Reading

* No bug found in the evaluation or tuning code. The raw-log-loss criterion does mix
  calibration with shape: for ring TNBBeta, raw loss rises with lr (0.610 -> 0.646) while
  recalibrated loss falls slightly (0.608 -> 0.602) and AUC rises 0.652 -> 0.709. But the
  recalibrated differences are ~0.006 nats, so this is a weak effect on log loss; the
  AUC difference is the larger one.
* vMF ring: lr 0.003 is the optimum even after recalibration, so for vMF the grid pick is genuine.
* degree_corrected TNBBeta at lr 0.03 has log loss 1.6 even after recalibration:
  saturated, extremely confident predictions (the shared learned temperature and/or
  concentration running away), not a fixable offset.
* The asymmetry that looks most like a defect: vMF/PS start from `initial_kappa` (r_bar
  ~0.6 at init) while TNBBeta starts at p = q = 0.5, eps = softplus(0) = 0.69 (r_bar ~0.1,
  axis-symmetric). At lr 0.003 TNBBeta needs ~700 epochs to leave that point
  (`tnb_plateau_mechanism_2026-11-01.md`). The tuned-lr comparison therefore partly measures
  initialisation, not just family. HYPOTHESIS, not tested: a TNBBeta head init giving a
  matched initial r_bar would remove the plateau at lr 0.003.

## Not done

* No TNBBeta init change was run. Single seeds/draws; 2 draws x 2 configs per cell.
