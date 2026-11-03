# GraphVAE head bias: does it remove the TNBBeta plateau and speed up vMF/PS kappa? (2026-11-01)

Date: 2026-11-01. Base commit `ec2677e` (branch `synthetic-graph-families`), **uncommitted working
tree** (code under test: `GraphVAEConfig.head_bias` / `p_init_offset` in
`src/tnbbeta_vae/models/graph_vae.py`, `apps/synthetic/head_bias_sweep.py`,
`apps/eval/head_bias_report.py`). EXPLORATORY: no pre-registration, no multiplicity correction, no
tuned learning rates (two fixed lrs), 6 draws per cell, no p-values -- paired mean differences and win
counts only.

## What changed in code

* `GraphVAEConfig.head_bias: bool = False`: the posterior head `GraphVAE.second` gets a bias
  (`second.bias`, one entry per raw output) only when on. The bias is added **after** the
  adjacency aggregation, `raw = A_hat (W h) + b`, so each raw parameter's baseline is a free
  per-parameter offset, not scaled by the node's `A_hat` row sum (adding it inside `nn.Linear`
  would have been multiplied by that row sum). Default state_dict keys/shapes unchanged
  (`first.weight, log_temperature, second.weight`; old checkpoints load strictly).
* `GraphVAEConfig.p_init_offset: float = 0.0`: with `head_bias=True` and `family="tnbbeta"`, the raw-p
  bias entry is set to this through `heads.init_tnbbeta_p_bias` (p sits at index `latent_dim`; a
  `fixed_epsilon` removes only the last output so the index is unaffected, tested). `0.0` keeps
  the random `nn.Linear` bias. Ignored without `head_bias` and for other families.
* Tests added to `tests/models/test_graph_vae.py` (keys unchanged by default for all four families,
  `second.bias` present when on, a `head_bias=False` model loads old state dicts strictly, bias is
  added after aggregation and trains, `p_init_offset` touches only the p entry incl. with
  `fixed_epsilon`, ignored otherwise). `tests/models` + `tests/apps`: 508 passed, 1 skipped
  (SLURM CPU job 2289058).
* The sweep script wraps the unchanged `apps.link_prediction.main.run_once` and
  `graph_search.evaluate`; `graph_search.py` untouched (no CLI flag added there, as its
  `CONDITIONS` table is keyed to the pre-registered conditions).

## Commands

```
sbatch --array=0-15 scripts/slurm/head_bias_sweep.sbatch      # job 2289020, 16 tasks x 6 procs/GPU, 384 runs
uv run python -m apps.eval.head_bias_report $TNBBETA_CHECKPOINT_DIR/head_bias_sweep
```
Defaults inside the sbatch: families `assortative,ring,degree_corrected,bipartite` (pilot `*-00`
configs), lrs `0.003,0.01`, draws `0-5`, conditions below. Seed = `9_000_000 + 1000 * config index +
draw`, identical across conditions (same graph, split, seed). Raw rows:
`/net/spaces/scratch/vpathak/tnbbeta_checkpoints/head_bias_sweep/sweep_shard*.jsonl`.
Conditions (all with `decoder_bias=True`, 500 epochs, latent dim 16 as in `graph_search`):

| name | model | head bias | start |
|---|---|---|---|
| `tnb` | TNBBeta | no | default init |
| `tnb_hb` | TNBBeta | yes | random bias |
| `tnb_hb_p3` | TNBBeta | yes | raw-p bias = ln 3 |
| `vmf_init` | vMF | no | kappa started at 16 (`initial_kappa`), the existing reference arm |
| `vmf_def` / `vmf_def_hb` | vMF | no / yes | no `initial_kappa`: softplus(raw)+1, kappa about 1.7 |
| `ps_def` / `ps_def_hb` | PowerSpherical | no / yes | no `initial_kappa`: softplus(raw), kappa about 0.7 (the PS head has no +1) |

Metric definitions: escape epoch = first epoch with `epoch % 10 == 0` and `r_bar > 0.3` (`-1` = already
above 0.3 at initialisation, so "escape" is vacuous: `tnb_hb_p3` starts at mean r_bar 0.36 and `vmf_init`
at 0.61; for the others r_bar at init is 0.10-0.14). `r_bar final` is the last logged epoch (495; the
trajectory logs every 5 epochs, the run is 500), `r_bar best` is measured on the selected
(best-validation) checkpoint, `kappa final` is the epoch-495 mean kappa. AUC = test sampled AUC, `ll`
= sampled log loss, `recal ll` = validation-recalibrated log loss. Mean +- sd (ddof 1). Note that a
run whose best-validation epoch is early (e.g. epoch 0-5 on graphs the model never learns) has `r_bar
best` near its init value even if `r_bar final` is high; the two differ and both are shown.

## Answers (exploratory; 24 paired runs per lr = 4 families x 6 draws)

1. **Head bias removes most of the TNBBeta plateau at lr 0.003.** Escaped (r_bar > 0.3): `tnb` 6/24,
   `tnb_hb` 21/24, `tnb_hb_p3` 24/24 (vacuous, started at 0.36). By family at lr 0.003 `tnb` escapes
   0/6 on assortative, bipartite and degree_corrected and 6/6 on ring (which does not plateau); `tnb_hb`
   escapes 6/6, 3/6 and 6/6 there (median epoch 110-205). Paired `tnb_hb` vs `tnb`: r_bar at the selected
   epoch +0.43 (22/24 wins), sampled AUC +0.096 (17/24 wins; assortative 0.50 -> 0.81), recalibrated log
   loss -0.045 (16/24). On bipartite and degree_corrected the escape does not buy much AUC (0.51, 0.54):
   the posterior concentrates but those graphs are not learned by this decoder at 500 epochs
   (compare 0.50/0.60 for vMF). The log loss on degree_corrected with a bias is worse (0.82 vs 0.70) while
   its recalibrated loss is at chance level (0.693): over-confident, mis-calibrated, not informative.
2. **Head bias speeds up kappa learning for vMF/PS, partly.** Mean final kappa at lr 0.003: `vmf_def`
   25.8 -> `vmf_def_hb` 39.2 (paired +13.5, 20/24 higher); `ps_def` 36.5 -> `ps_def_hb` 47.0 (+10.5, 18/24).
   Kappa starts at 1.7 / 0.7, so without a bias vMF reaches kappa about 8, 1.8, 2.1 and 91 on assortative,
   bipartite, degree_corrected and ring (the first three barely move; ring is the easy graph) and with
   one 36, 12, 19 and 90. Escape fraction (r_bar > 0.3): vMF 11/24 -> 24/24, PS 6/24 -> 13/24. PS with a
   bias still does not move on bipartite (0/6) and degree_corrected (1/6) at lr 0.003. AUC +0.057 (vMF,
   17/24) and +0.072 (PS, 17/24). The bias-free `vmf_init` arm (kappa started at 16) is the better of
   the vMF arms at lr 0.003 on its own (AUC 0.671, recal ll 0.608 vs 0.648 / 0.621 for `vmf_def_hb`),
   consistent with the previous finding that the kappa start matters: a head bias reaches only part of
   that gap by itself.
3. **The ln 3 p-offset on top of the bias matters little here.** `tnb_hb_p3` vs `tnb_hb` at lr 0.003:
   AUC +0.008 (17/24 wins, sd 0.019), recal ll -0.002 (15/24), r_bar best +0.007 (12/24); its one
   visible effect is removing the escape delay (starts at r_bar 0.36 instead of 0.11) and bipartite
   escape 3/6 -> 6/6 (without AUC gain: 0.516 vs 0.510). Evidence for a benefit beyond the bias is
   weak.
4. **The lr 0.003 vs 0.01 gap shrinks for TNBBeta, less so for vMF/PS.** Mean AUC gap (lr 0.01 minus
   lr 0.003): `tnb` +0.085, `tnb_hb` +0.007, `tnb_hb_p3` -0.001; `vmf_def` +0.037 -> `vmf_def_hb` +0.014;
   `ps_def` +0.066 -> `ps_def_hb` +0.027; `vmf_init` -0.003 (also no gap). The r_bar gap is +0.42 -> +0.10
   -> +0.02 for TNBBeta. At lr 0.01 the bias helps little for TNBBeta (AUC +0.017, 12/24 wins) and
   vMF/PS get higher kappa (+15, +35) with no consistent change in recalibrated log loss (it is slightly
   worse on average, driven by degree_corrected, where the bias-free arms are better calibrated).
   Of note, at lr 0.01 the TNBBeta log loss on degree_corrected is above chance (1.0-1.2): bias or not, this lr
   overshoots there.

## Caveats

* 4 pilot configs, one calibrated config per family, 6 graph draws; sd across draws is large
  (e.g. best epoch sd about 170 of 500). Several "wins" are 17/24 with sd comparable to the mean:
  suggestive, not established. Pooled rows mix families that are and are not learnable.
* A bias gives every node the same additive offset in the raw parameters, so it can change the
  initial concentration/baseline of every node but not how a node's parameters vary; it may also
  alter the generalisation of the learnt `p, q, eps`. I did not separate "faster escape" from "different
  solution": best-validation epoch moves later with a bias for TNBBeta (171 -> 295), which is what
  one expects if the earlier models were stopped at an initialisation-level checkpoint.
* The hypothesis "the missing bias is a main cause" is supported for the plateau of TNBBeta (and
  for vMF/PS kappa) at lr 0.003 in the sense that the bias removes most of it; this does not show it
  is the only cause (bipartite/degree_corrected still do not learn for any family), and no lr or
  budget was tuned for the bias-free arms beyond the two lrs here.
* Not touched: `scratch_degree_correlation.py` still has ruff errors (left alone as asked).

## Full output of `apps.eval.head_bias_report`

384 runs

### lr 0.003, all families pooled

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 24 | 6/24 | 20 | 0.310 +- 0.376 | 0.299 +- 0.357 | - | 0.563 +- 0.126 | 0.661 +- 0.065 | 0.656 +- 0.067 | 170.958 +- 194.802 |
| tnb_hb | 24 | 21/24 | 120 | 0.846 +- 0.258 | 0.731 +- 0.307 | - | 0.659 +- 0.140 | 0.653 +- 0.134 | 0.612 +- 0.085 | 294.625 +- 164.981 |
| tnb_hb_p3 | 24 | 24/24 | -1 | 0.872 +- 0.140 | 0.739 +- 0.268 | - | 0.667 +- 0.135 | 0.645 +- 0.135 | 0.609 +- 0.086 | 301.667 +- 172.980 |
| vmf_init | 24 | 24/24 | -1 | 0.703 +- 0.122 | 0.669 +- 0.098 | 32.891 +- 28.059 | 0.671 +- 0.130 | 0.635 +- 0.082 | 0.608 +- 0.080 | 183.375 +- 181.130 |
| vmf_def | 24 | 11/24 | 50 | 0.412 +- 0.323 | 0.326 +- 0.301 | 25.785 +- 38.547 | 0.591 +- 0.127 | 0.656 +- 0.064 | 0.651 +- 0.068 | 139.833 +- 145.127 |
| vmf_def_hb | 24 | 24/24 | 280 | 0.722 +- 0.162 | 0.579 +- 0.256 | 39.244 +- 31.515 | 0.648 +- 0.132 | 0.630 +- 0.080 | 0.621 +- 0.082 | 345.875 +- 178.903 |
| ps_def | 24 | 6/24 | 65 | 0.313 +- 0.349 | 0.272 +- 0.301 | 36.502 +- 62.881 | 0.561 +- 0.123 | 0.665 +- 0.058 | 0.661 +- 0.060 | 201.583 +- 166.444 |
| ps_def_hb | 24 | 13/24 | 290 | 0.477 +- 0.357 | 0.398 +- 0.316 | 47.002 +- 59.463 | 0.633 +- 0.152 | 0.635 +- 0.069 | 0.622 +- 0.082 | 260.292 +- 206.278 |

### lr 0.003, assortative

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 6 | 0/6 | - | 0.097 +- 0.001 | 0.097 +- 0.001 | - | 0.496 +- 0.017 | 0.698 +- 0.003 | 0.694 +- 0.001 | 2.833 +- 1.722 |
| tnb_hb | 6 | 6/6 | 110 | 0.940 +- 0.009 | 0.792 +- 0.137 | - | 0.811 +- 0.010 | 0.529 +- 0.028 | 0.516 +- 0.021 | 243.667 +- 155.254 |
| tnb_hb_p3 | 6 | 6/6 | -1 | 0.936 +- 0.007 | 0.844 +- 0.105 | - | 0.812 +- 0.009 | 0.522 +- 0.019 | 0.515 +- 0.012 | 328.667 +- 168.772 |
| vmf_init | 6 | 6/6 | -1 | 0.639 +- 0.003 | 0.618 +- 0.009 | 17.292 +- 0.155 | 0.806 +- 0.009 | 0.565 +- 0.029 | 0.520 +- 0.018 | 114.333 +- 193.715 |
| vmf_def | 6 | 5/6 | 460 | 0.420 +- 0.099 | 0.199 +- 0.136 | 8.284 +- 2.679 | 0.587 +- 0.109 | 0.679 +- 0.027 | 0.668 +- 0.059 | 118.333 +- 190.595 |
| vmf_def_hb | 6 | 6/6 | 205 | 0.808 +- 0.003 | 0.513 +- 0.325 | 35.579 +- 0.565 | 0.708 +- 0.164 | 0.587 +- 0.100 | 0.570 +- 0.100 | 282.500 +- 212.720 |
| ps_def | 6 | 0/6 | - | 0.125 +- 0.009 | 0.108 +- 0.001 | 1.258 +- 0.254 | 0.504 +- 0.006 | 0.695 +- 0.001 | 0.693 +- 0.000 | 131.167 +- 133.395 |
| ps_def_hb | 6 | 6/6 | 315 | 0.733 +- 0.009 | 0.560 +- 0.228 | 41.097 +- 1.883 | 0.760 +- 0.127 | 0.582 +- 0.059 | 0.542 +- 0.076 | 350.000 +- 173.563 |

### lr 0.003, bipartite

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 6 | 0/6 | - | 0.097 +- 0.000 | 0.097 +- 0.001 | - | 0.493 +- 0.015 | 0.697 +- 0.003 | 0.694 +- 0.001 | 157.167 +- 216.039 |
| tnb_hb | 6 | 3/6 | 190 | 0.550 +- 0.408 | 0.462 +- 0.404 | - | 0.510 +- 0.020 | 0.711 +- 0.021 | 0.693 +- 0.001 | 299.333 +- 201.149 |
| tnb_hb_p3 | 6 | 6/6 | -1 | 0.656 +- 0.127 | 0.371 +- 0.241 | - | 0.516 +- 0.022 | 0.696 +- 0.001 | 0.693 +- 0.001 | 279.833 +- 193.531 |
| vmf_init | 6 | 6/6 | -1 | 0.620 +- 0.001 | 0.615 +- 0.002 | 16.319 +- 0.030 | 0.498 +- 0.010 | 0.728 +- 0.015 | 0.693 +- 0.000 | 79.000 +- 141.278 |
| vmf_def | 6 | 0/6 | - | 0.149 +- 0.001 | 0.144 +- 0.001 | 1.823 +- 0.017 | 0.499 +- 0.008 | 0.696 +- 0.002 | 0.693 +- 0.000 | 161.667 +- 94.796 |
| vmf_def_hb | 6 | 6/6 | 395 | 0.521 +- 0.085 | 0.447 +- 0.172 | 12.129 +- 4.190 | 0.517 +- 0.020 | 0.694 +- 0.003 | 0.692 +- 0.001 | 396.667 +- 194.320 |
| ps_def | 6 | 0/6 | - | 0.108 +- 0.001 | 0.108 +- 0.001 | 0.732 +- 0.014 | 0.494 +- 0.006 | 0.695 +- 0.000 | 0.693 +- 0.000 | 351.500 +- 96.430 |
| ps_def_hb | 6 | 0/6 | - | 0.111 +- 0.003 | 0.108 +- 0.001 | 0.821 +- 0.124 | 0.495 +- 0.006 | 0.697 +- 0.002 | 0.693 +- 0.000 | 129.167 +- 197.527 |

### lr 0.003, degree_corrected

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 6 | 0/6 | - | 0.100 +- 0.002 | 0.099 +- 0.001 | - | 0.488 +- 0.027 | 0.697 +- 0.005 | 0.694 +- 0.002 | 295.333 +- 196.932 |
| tnb_hb | 6 | 6/6 | 205 | 0.947 +- 0.007 | 0.753 +- 0.340 | - | 0.541 +- 0.058 | 0.825 +- 0.096 | 0.693 +- 0.011 | 336.500 +- 169.617 |
| tnb_hb_p3 | 6 | 6/6 | -1 | 0.946 +- 0.006 | 0.825 +- 0.192 | - | 0.561 +- 0.051 | 0.818 +- 0.120 | 0.691 +- 0.015 | 318.833 +- 177.494 |
| vmf_init | 6 | 6/6 | -1 | 0.644 +- 0.001 | 0.629 +- 0.012 | 17.551 +- 0.088 | 0.605 +- 0.027 | 0.698 +- 0.014 | 0.677 +- 0.009 | 301.833 +- 196.406 |
| vmf_def | 6 | 0/6 | - | 0.161 +- 0.002 | 0.144 +- 0.001 | 2.087 +- 0.051 | 0.501 +- 0.035 | 0.696 +- 0.005 | 0.694 +- 0.001 | 54.167 +- 70.895 |
| vmf_def_hb | 6 | 6/6 | 370 | 0.639 +- 0.024 | 0.516 +- 0.204 | 18.815 +- 2.006 | 0.588 +- 0.048 | 0.689 +- 0.011 | 0.678 +- 0.011 | 428.833 +- 110.530 |
| ps_def | 6 | 0/6 | - | 0.115 +- 0.001 | 0.109 +- 0.001 | 0.944 +- 0.023 | 0.479 +- 0.032 | 0.699 +- 0.006 | 0.697 +- 0.007 | 71.667 +- 133.377 |
| ps_def_hb | 6 | 1/6 | 490 | 0.160 +- 0.085 | 0.149 +- 0.093 | 2.292 +- 2.621 | 0.503 +- 0.066 | 0.695 +- 0.009 | 0.694 +- 0.007 | 283.333 +- 237.455 |

### lr 0.003, ring

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 6 | 6/6 | 20 | 0.948 +- 0.002 | 0.904 +- 0.055 | - | 0.774 +- 0.013 | 0.551 +- 0.015 | 0.543 +- 0.010 | 228.500 +- 180.050 |
| tnb_hb | 6 | 6/6 | 20 | 0.948 +- 0.001 | 0.918 +- 0.050 | - | 0.772 +- 0.012 | 0.548 +- 0.012 | 0.544 +- 0.009 | 299.000 +- 163.101 |
| tnb_hb_p3 | 6 | 6/6 | -1 | 0.948 +- 0.001 | 0.915 +- 0.046 | - | 0.778 +- 0.012 | 0.546 +- 0.020 | 0.539 +- 0.014 | 279.333 +- 194.424 |
| vmf_init | 6 | 6/6 | -1 | 0.910 +- 0.003 | 0.813 +- 0.101 | 80.403 +- 2.979 | 0.775 +- 0.018 | 0.550 +- 0.018 | 0.542 +- 0.015 | 238.333 +- 124.903 |
| vmf_def | 6 | 6/6 | 40 | 0.920 +- 0.001 | 0.815 +- 0.108 | 90.948 +- 1.437 | 0.776 +- 0.011 | 0.552 +- 0.025 | 0.548 +- 0.023 | 225.167 +- 166.956 |
| vmf_def_hb | 6 | 6/6 | 40 | 0.920 +- 0.001 | 0.842 +- 0.102 | 90.452 +- 1.013 | 0.780 +- 0.013 | 0.549 +- 0.021 | 0.542 +- 0.018 | 275.500 +- 172.871 |
| ps_def | 6 | 6/6 | 65 | 0.904 +- 0.003 | 0.764 +- 0.169 | 143.074 +- 4.006 | 0.767 +- 0.013 | 0.570 +- 0.032 | 0.562 +- 0.026 | 252.000 +- 162.263 |
| ps_def_hb | 6 | 6/6 | 70 | 0.905 +- 0.002 | 0.774 +- 0.153 | 143.798 +- 3.322 | 0.772 +- 0.011 | 0.568 +- 0.023 | 0.558 +- 0.018 | 278.667 +- 197.169 |

### lr 0.01, all families pooled

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 24 | 23/24 | 90 | 0.919 +- 0.176 | 0.722 +- 0.337 | - | 0.648 +- 0.140 | 0.712 +- 0.329 | 0.697 +- 0.321 | 230.833 +- 142.382 |
| tnb_hb | 24 | 24/24 | 45 | 0.922 +- 0.080 | 0.832 +- 0.236 | - | 0.665 +- 0.133 | 0.752 +- 0.363 | 0.702 +- 0.348 | 225.167 +- 164.702 |
| tnb_hb_p3 | 24 | 24/24 | -1 | 0.907 +- 0.089 | 0.762 +- 0.264 | - | 0.666 +- 0.131 | 0.713 +- 0.307 | 0.690 +- 0.299 | 207.542 +- 171.048 |
| vmf_init | 24 | 24/24 | -1 | 0.852 +- 0.075 | 0.757 +- 0.125 | 66.125 +- 42.604 | 0.669 +- 0.128 | 0.668 +- 0.165 | 0.610 +- 0.084 | 238.500 +- 181.158 |
| vmf_def | 24 | 24/24 | 190 | 0.899 +- 0.029 | 0.624 +- 0.332 | 83.229 +- 32.462 | 0.627 +- 0.130 | 0.676 +- 0.164 | 0.649 +- 0.111 | 269.542 +- 174.837 |
| vmf_def_hb | 24 | 24/24 | 105 | 0.918 +- 0.017 | 0.733 +- 0.219 | 98.697 +- 24.002 | 0.662 +- 0.124 | 0.741 +- 0.307 | 0.692 +- 0.276 | 265.958 +- 149.090 |
| ps_def | 24 | 18/24 | 180 | 0.687 +- 0.341 | 0.489 +- 0.360 | 104.292 +- 86.438 | 0.627 +- 0.132 | 0.666 +- 0.108 | 0.635 +- 0.074 | 280.833 +- 187.393 |
| ps_def_hb | 24 | 22/24 | 140 | 0.829 +- 0.225 | 0.719 +- 0.261 | 139.316 +- 69.059 | 0.660 +- 0.127 | 0.748 +- 0.288 | 0.683 +- 0.216 | 284.125 +- 164.692 |

### lr 0.01, assortative

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 6 | 6/6 | 80 | 0.978 +- 0.002 | 0.768 +- 0.335 | - | 0.756 +- 0.139 | 0.544 +- 0.078 | 0.538 +- 0.076 | 134.167 +- 90.592 |
| tnb_hb | 6 | 6/6 | 40 | 0.971 +- 0.005 | 0.873 +- 0.056 | - | 0.814 +- 0.009 | 0.517 +- 0.022 | 0.506 +- 0.017 | 126.000 +- 98.016 |
| tnb_hb_p3 | 6 | 6/6 | -1 | 0.972 +- 0.004 | 0.865 +- 0.083 | - | 0.812 +- 0.008 | 0.519 +- 0.017 | 0.516 +- 0.013 | 145.500 +- 82.270 |
| vmf_init | 6 | 6/6 | -1 | 0.837 +- 0.005 | 0.717 +- 0.088 | 42.391 +- 1.344 | 0.806 +- 0.014 | 0.540 +- 0.064 | 0.517 +- 0.037 | 233.000 +- 186.298 |
| vmf_def | 6 | 6/6 | 150 | 0.892 +- 0.001 | 0.511 +- 0.399 | 66.254 +- 0.959 | 0.683 +- 0.146 | 0.600 +- 0.099 | 0.600 +- 0.100 | 212.333 +- 203.500 |
| vmf_def_hb | 6 | 6/6 | 80 | 0.903 +- 0.001 | 0.570 +- 0.294 | 74.105 +- 0.685 | 0.762 +- 0.102 | 0.591 +- 0.070 | 0.560 +- 0.068 | 202.167 +- 200.077 |
| ps_def | 6 | 6/6 | 180 | 0.868 +- 0.002 | 0.432 +- 0.345 | 99.091 +- 1.820 | 0.667 +- 0.160 | 0.621 +- 0.083 | 0.602 +- 0.099 | 178.000 +- 126.726 |
| ps_def_hb | 6 | 6/6 | 115 | 0.888 +- 0.003 | 0.649 +- 0.280 | 118.908 +- 3.553 | 0.759 +- 0.127 | 0.566 +- 0.070 | 0.548 +- 0.074 | 185.000 +- 146.420 |

### lr 0.01, bipartite

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 6 | 5/6 | 340 | 0.795 +- 0.342 | 0.528 +- 0.340 | - | 0.513 +- 0.022 | 0.695 +- 0.003 | 0.694 +- 0.002 | 351.333 +- 76.605 |
| tnb_hb | 6 | 6/6 | 50 | 0.801 +- 0.079 | 0.565 +- 0.369 | - | 0.518 +- 0.029 | 0.706 +- 0.022 | 0.693 +- 0.002 | 303.167 +- 229.024 |
| tnb_hb_p3 | 6 | 6/6 | -1 | 0.763 +- 0.032 | 0.431 +- 0.347 | - | 0.517 +- 0.021 | 0.696 +- 0.004 | 0.693 +- 0.002 | 196.333 +- 211.971 |
| vmf_init | 6 | 6/6 | -1 | 0.751 +- 0.046 | 0.667 +- 0.084 | 27.087 +- 4.345 | 0.510 +- 0.013 | 0.722 +- 0.022 | 0.693 +- 0.001 | 217.833 +- 222.586 |
| vmf_def | 6 | 6/6 | 275 | 0.873 +- 0.004 | 0.544 +- 0.316 | 55.604 +- 2.124 | 0.503 +- 0.011 | 0.702 +- 0.009 | 0.694 +- 0.001 | 317.167 +- 111.915 |
| vmf_def_hb | 6 | 6/6 | 160 | 0.913 +- 0.008 | 0.629 +- 0.132 | 84.524 +- 7.129 | 0.519 +- 0.013 | 0.701 +- 0.006 | 0.694 +- 0.001 | 240.833 +- 46.002 |
| ps_def | 6 | 0/6 | - | 0.114 +- 0.006 | 0.113 +- 0.006 | 0.925 +- 0.215 | 0.495 +- 0.010 | 0.694 +- 0.000 | 0.693 +- 0.000 | 409.167 +- 111.704 |
| ps_def_hb | 6 | 4/6 | 275 | 0.601 +- 0.386 | 0.553 +- 0.352 | 58.959 +- 47.484 | 0.513 +- 0.010 | 0.702 +- 0.007 | 0.694 +- 0.002 | 332.667 +- 178.591 |

### lr 0.01, degree_corrected

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 6 | 6/6 | 175 | 0.950 +- 0.004 | 0.662 +- 0.436 | - | 0.551 +- 0.063 | 1.055 +- 0.534 | 1.010 +- 0.541 | 242.333 +- 165.740 |
| tnb_hb | 6 | 6/6 | 60 | 0.952 +- 0.004 | 0.948 +- 0.007 | - | 0.558 +- 0.027 | 1.225 +- 0.472 | 1.060 +- 0.572 | 270.500 +- 177.533 |
| tnb_hb_p3 | 6 | 6/6 | -1 | 0.956 +- 0.004 | 0.845 +- 0.124 | - | 0.567 +- 0.037 | 1.074 +- 0.451 | 0.997 +- 0.489 | 226.333 +- 201.078 |
| vmf_init | 6 | 6/6 | -1 | 0.873 +- 0.005 | 0.730 +- 0.120 | 59.590 +- 1.987 | 0.584 +- 0.020 | 0.851 +- 0.204 | 0.686 +- 0.010 | 223.500 +- 201.401 |
| vmf_def | 6 | 6/6 | 230 | 0.885 +- 0.006 | 0.537 +- 0.340 | 74.116 +- 1.950 | 0.555 +- 0.060 | 0.835 +- 0.248 | 0.747 +- 0.135 | 244.833 +- 206.545 |
| vmf_def_hb | 6 | 6/6 | 130 | 0.910 +- 0.004 | 0.808 +- 0.141 | 100.874 +- 1.513 | 0.588 +- 0.024 | 1.113 +- 0.437 | 0.972 +- 0.451 | 318.167 +- 142.019 |
| ps_def | 6 | 6/6 | 295 | 0.826 +- 0.013 | 0.538 +- 0.346 | 81.248 +- 7.168 | 0.574 +- 0.048 | 0.782 +- 0.124 | 0.685 +- 0.009 | 298.833 +- 238.407 |
| ps_def_hb | 6 | 6/6 | 200 | 0.887 +- 0.005 | 0.802 +- 0.171 | 142.857 +- 3.836 | 0.598 +- 0.011 | 1.156 +- 0.307 | 0.931 +- 0.308 | 409.167 +- 113.927 |

### lr 0.01, ring

| cond | n | escaped | median esc ep | r_bar final | r_bar best | kappa final | AUC | log loss | recal ll | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| tnb | 6 | 6/6 | 10 | 0.952 +- 0.022 | 0.929 +- 0.018 | - | 0.774 +- 0.013 | 0.553 +- 0.028 | 0.546 +- 0.017 | 195.500 +- 146.336 |
| tnb_hb | 6 | 6/6 | 10 | 0.962 +- 0.001 | 0.941 +- 0.012 | - | 0.771 +- 0.014 | 0.560 +- 0.017 | 0.548 +- 0.013 | 201.000 +- 94.539 |
| tnb_hb_p3 | 6 | 6/6 | -1 | 0.939 +- 0.035 | 0.907 +- 0.036 | - | 0.768 +- 0.020 | 0.564 +- 0.034 | 0.552 +- 0.021 | 262.000 +- 183.576 |
| vmf_init | 6 | 6/6 | -1 | 0.946 +- 0.001 | 0.912 +- 0.040 | 135.431 +- 2.229 | 0.775 +- 0.015 | 0.558 +- 0.026 | 0.544 +- 0.019 | 279.667 +- 152.636 |
| vmf_def | 6 | 6/6 | 20 | 0.946 +- 0.002 | 0.902 +- 0.085 | 136.943 +- 3.857 | 0.770 +- 0.011 | 0.566 +- 0.023 | 0.554 +- 0.024 | 303.833 +- 187.674 |
| vmf_def_hb | 6 | 6/6 | 20 | 0.946 +- 0.002 | 0.925 +- 0.026 | 135.285 +- 4.190 | 0.779 +- 0.020 | 0.557 +- 0.030 | 0.542 +- 0.021 | 302.667 +- 171.471 |
| ps_def | 6 | 6/6 | 30 | 0.940 +- 0.001 | 0.873 +- 0.072 | 235.903 +- 6.082 | 0.774 +- 0.013 | 0.567 +- 0.017 | 0.560 +- 0.022 | 237.333 +- 202.600 |
| ps_def_hb | 6 | 6/6 | 30 | 0.940 +- 0.002 | 0.873 +- 0.080 | 236.539 +- 7.057 | 0.770 +- 0.016 | 0.566 +- 0.024 | 0.561 +- 0.025 | 209.667 +- 135.727 |

### Paired comparisons (same family, draw, lr); win = treatment better

| lr | treatment vs control | metric | n | mean diff (sd) | wins | ties |
|---|---|---|---|---|---|---|
| 0.003 | tnb_hb vs tnb | sampled_auc | 24 | +0.096 (0.135) | 17 | 1 |
| 0.003 | tnb_hb vs tnb | r_bar_best | 24 | +0.432 (0.378) | 22 | 0 |
| 0.003 | tnb_hb vs tnb | recal_ll | 24 | -0.045 (0.079) | 16 | 1 |
| 0.003 | tnb_hb_p3 vs tnb_hb | sampled_auc | 24 | +0.008 (0.019) | 17 | 0 |
| 0.003 | tnb_hb_p3 vs tnb_hb | r_bar_best | 24 | +0.007 (0.261) | 12 | 0 |
| 0.003 | tnb_hb_p3 vs tnb_hb | recal_ll | 24 | -0.002 (0.009) | 15 | 1 |
| 0.003 | tnb_hb_p3 vs tnb | sampled_auc | 24 | +0.104 (0.131) | 21 | 0 |
| 0.003 | vmf_def_hb vs vmf_def | sampled_auc | 24 | +0.057 (0.093) | 17 | 0 |
| 0.003 | vmf_def_hb vs vmf_def | kappa_final | 24 | +13.458 (10.609) | 20 | 0 |
| 0.003 | vmf_def_hb vs vmf_def | recal_ll | 24 | -0.030 (0.060) | 17 | 1 |
| 0.003 | ps_def_hb vs ps_def | sampled_auc | 24 | +0.072 (0.128) | 17 | 0 |
| 0.003 | ps_def_hb vs ps_def | kappa_final | 24 | +10.500 (17.382) | 18 | 0 |
| 0.003 | ps_def_hb vs ps_def | recal_ll | 24 | -0.039 (0.075) | 10 | 8 |
| 0.003 | vmf_def vs vmf_init | kappa_final | 24 | -7.106 (10.889) | 6 | 0 |
| 0.01 | tnb_hb vs tnb | sampled_auc | 24 | +0.017 (0.078) | 12 | 0 |
| 0.01 | tnb_hb vs tnb | r_bar_best | 24 | +0.110 (0.344) | 17 | 0 |
| 0.01 | tnb_hb vs tnb | recal_ll | 24 | +0.005 (0.255) | 13 | 1 |
| 0.01 | tnb_hb_p3 vs tnb_hb | sampled_auc | 24 | +0.001 (0.027) | 10 | 0 |
| 0.01 | tnb_hb_p3 vs tnb_hb | r_bar_best | 24 | -0.070 (0.246) | 10 | 0 |
| 0.01 | tnb_hb_p3 vs tnb_hb | recal_ll | 24 | -0.012 (0.290) | 11 | 0 |
| 0.01 | tnb_hb_p3 vs tnb | sampled_auc | 24 | +0.018 (0.074) | 12 | 0 |
| 0.01 | vmf_def_hb vs vmf_def | sampled_auc | 24 | +0.035 (0.105) | 18 | 0 |
| 0.01 | vmf_def_hb vs vmf_def | kappa_final | 24 | +15.468 (13.912) | 22 | 0 |
| 0.01 | vmf_def_hb vs vmf_def | recal_ll | 24 | +0.043 (0.228) | 13 | 0 |
| 0.01 | ps_def_hb vs ps_def | sampled_auc | 24 | +0.033 (0.082) | 13 | 0 |
| 0.01 | ps_def_hb vs ps_def | kappa_final | 24 | +35.024 (34.598) | 20 | 0 |
| 0.01 | ps_def_hb vs ps_def | recal_ll | 24 | +0.048 (0.194) | 8 | 2 |
| 0.01 | vmf_def vs vmf_init | kappa_final | 24 | +17.105 (10.945) | 22 | 0 |

### Learning-rate gap (mean over all runs of lr 0.01 minus lr 0.003)

| cond | dAUC | d r_bar best | d recal ll |
|---|---|---|---|
| tnb | +0.085 | +0.422 | +0.041 |
| tnb_hb | +0.007 | +0.100 | +0.090 |
| tnb_hb_p3 | -0.001 | +0.023 | +0.080 |
| vmf_init | -0.003 | +0.088 | +0.002 |
| vmf_def | +0.037 | +0.298 | -0.002 |
| vmf_def_hb | +0.014 | +0.154 | +0.071 |
| ps_def | +0.066 | +0.217 | -0.026 |
| ps_def_hb | +0.027 | +0.321 | +0.061 |
