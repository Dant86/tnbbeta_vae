# Why TNBBeta posteriors sit near initialisation on the synthetic graphs: mechanism (2026-11-01)

Date: 2026-11-01. Base commit: `ec2677efa042f789c26470e44a9cb0a35c51c4c5` (branch
`synthetic-graph-families`), uncommitted working tree. Exploratory, post hoc, single graph draw
per case, one training seed each; descriptive only, no hypothesis tests.

## Question

`graph_search_kappa_grid_results_2026-10-31.md` (plateau section) found 26% of discovery TNBBeta
runs at `r_bar` ~ 0.097, the untrained value. Three hypotheses were to be checked:
(a) head gradients for p/q/eps are tiny because softplus/sigmoid are saturated; (b) the
Monte Carlo gradient w.r.t. eps is noise-dominated; (c) lr/epoch budget too short for the eps
scale needed.

## Method

Local CPU only (1 thread, `nice -n 10`, at most 2 processes), no SLURM. No existing TNBBeta
synthetic-graph trajectory logs exist (the 280 `--log-trajectory` files are real-graph
fixed-kappa / kdyn runs; only the 20 `kdyn_matched_tnbbeta_*` ones are TNBBeta, at lr 0.01),
so I re-ran the exact `graph_search.run_item` setup (same graph spec and seed from the
manifest, same `GraphVAE` config via `model_config_for`, same Adam / full-graph step, DIM=16)
with a per-epoch logger. Scratch script (not committed):
`/tmp/claude-14853/-home-vpathak-tnbbeta-vae/39a83aae-c9e5-4e62-8b02-10e6fc87ddf7/scratchpad/plateau_diag.py`.
It logs, per epoch: raw head outputs (pre-softplus/sigmoid), per-output row gradient norms of
`model.second.weight`, mean L1 norm of the hidden layer, and, at epochs {0,25,50,100,200,300,
400,600,1000,1500,1999}, a 48-draw probe (fresh rsample, negatives and dropout, no parameter
step) giving for each head output the gradient SNR `||mean|| / ||std||` and the mean
per-element Adam drift factor `|m| / sqrt(m^2 + s^2)`.

Commands (`PYTHONPATH=.`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, args: config seed lr epochs
condition out):

```
python plateau_diag.py bipartite-00 10012000 0.003 500  tnbbeta bip_lr003_e500.json
python plateau_diag.py bipartite-00 10012000 0.003 2000 tnbbeta bip_lr003_e2000.json
python plateau_diag.py bipartite-00 10012000 0.01  500  tnbbeta bip_lr01_e500.json
python plateau_diag.py bipartite-00 10012000 0.03  500  tnbbeta bip_lr03_e500.json
python plateau_diag.py bipartite-00 10012001 0.003 2000 tnbbeta bip1_e2000.json
python plateau_diag.py assortative-00 10000000 0.01 500 tnbbeta asr_d0_lr01.json   # draw 0
python plateau_diag.py assortative-00 10000001 0.01 500 tnbbeta asr_d1_lr01.json   # draw 1
python plateau_diag.py ring-01 10085000 0.003 500  tnbbeta ring_lr003.json
python plateau_diag.py ring-01 10085000 0.003 2000 tnbbeta ring_e2000.json
python plateau_diag.py bipartite-00 10012000 0.003 500 vmf_init bip_vmf_lr003.json
```

Seeds are the discovery-stage manifest seeds (`bipartite-00_d000` = 10012000, etc.). The
r_bar at epoch 0 and the plateau value reproduce the discovery runs (0.097).

## Results

### (a) Saturation: refuted as the cause

- At init every raw head output is ~0 (raw eps 0.001, raw p -0.003, raw q -0.006), so
  sigmoid' = 0.25 and softplus' = 0.50, the maxima. Across the whole plateau (500 epochs at
  lr 0.003) raw eps rises only to 0.29 and |raw p| stays < 0.1: derivative factors stay
  within 0.25 -> 0.24 and 0.50 -> 0.57. Nothing is saturated while stuck.
- Saturation does appear, but only after escape: p -> 0.02 (sigmoid' ~ 0.02) and eps ~ 3.3
  (softplus' ~ 0.96, not saturated, but the gradient w.r.t. eps falls to ~5e-5 once the
  posterior is a tight cap). The row gradient of eps drops from 1.7e-3 (init) to 4e-5 to 1e-4
  after escape. That is a consequence of arrival, not a barrier to leaving.
- Raw gradient magnitudes are small (mean row-gradient norm of `second.weight`: mu rows 3.8e-2,
  p 2.6e-4, q 6.2e-4, eps 1.7e-3 on bipartite-00 at init) but Adam is invariant to gradient
  scale; per-element eps gradients (~4e-4) are 4 orders of magnitude above Adam's 1e-8
  epsilon. Small magnitude does not slow Adam.

### (b) Noise-dominated eps gradient: refuted for eps and q; supported for p only

48-draw probe, mean over draws vs std over draws of the head-weight gradient (SNR as defined
above; Adam drift factor in brackets; 1.0 = sign-consistent, ~0.1 = noise):

| case / epoch | mu | p | q | eps |
|---|---|---|---|---|
| bipartite-00 lr .003, ep 0 | 2.1 (.78) | 0.12 (.11) | 8.0 (.99) | 8.0 (.99) |
| bipartite-00 lr .003, ep 100 | 0.18 (.14) | 0.12 (.10) | 3.7 (.96) | 3.7 (.96) |
| bipartite-00 lr .003, ep 400 | 0.14 (.11) | 0.19 (.18) | 2.1 (.89) | 2.4 (.91) |
| ring-01 lr .003, ep 0 | 2.1 (.76) | 0.16 (.14) | 8.0 (.99) | 8.0 (.99) |
| ring-01 lr .003, ep 400 | 0.16 (.13) | 0.65 (.53) | 2.1 (.90) | 2.4 (.92) |
| assortative-00 d1 lr .01, ep 0 | 3.6 (.80) | 0.10 (.08) | 11.2 (1.0) | 10.8 (1.0) |
| assortative-00 d1 lr .01, ep 50 | 0.33 (.21) | 1.31 (.56) | 5.1 (.88) | 4.8 (.88) |
| assortative-00 d1 lr .01, ep 100 | 1.30 (.58) | 8.0 (.89) | 6.0 (.88) | 4.3 (.85) |

- eps and q gradients are strongly sign-consistent from the start (SNR 2 to 11) over the
  entire plateau; raw eps rises monotonically in every run (no random walk). So eps is not
  noise-limited; the logged per-node SNR is low (0.2 to 0.6) but it averages out across nodes
  and hidden units in the shared head weights.
- p is different: at init its gradient is ~50x smaller than eps's (4e-5 vs 2.2e-3 mean norm)
  and noise-dominated (SNR 0.10 to 0.16, Adam drift factor 0.08 to 0.14). p stays within
  0.475 to 0.535 for all 500 epochs in the stuck runs.
- Why p matters: at p = q = 0.5 the distribution is axis-symmetric. I sampled the
  distribution (d = 16, 20k draws): at p = q = 0.5, E[z.mu] = 0.000 +/- 0.003 for every eps
  in 0.69..3.0, and E|z.mu| falls from 0.50 to 0.24 as eps rises, i.e. growing eps alone makes
  the posterior more uniform, not more concentrated. Only a p away from 0.5 (e.g. p = .65,
  q = .63: E[z.mu] = 0.22 to 0.28) gives a cap. So escape in `r_bar` requires p to break
  symmetry, and p is the noisy coordinate. Raising eps with p pinned near 0.5 moves the
  parameters but not `r_bar`: r_bar stays 0.096 to 0.12 while eps goes 0.69 -> 0.90.
- Control: vMF kappa (same graph, lr .003, kappa head row) has SNR 9 to 22 and Adam factor
  1.0 throughout, mean gradient 1.5e-4 -> 1.6e-3; its `r_bar` still barely moves
  (0.614 -> 0.645 in 500 epochs) because the kappa log-parameterisation rate, not noise, limits
  it (consistent with the existing note that vMF barely moves). This is the same mechanism as
  (c) below.

### (c) lr / epoch budget too short: supported, quantitatively

- Drift speed is predicted by `lr * ||h||_1` (Adam moves every head weight ~lr per step in a
  consistent direction, so the raw output shifts by about lr times the hidden L1 norm; the
  hidden activations are small, L1 ~ 0.10 to 0.14 at init, identity features). Observed raw-eps
  shift vs `lr * cumulative ||h||_1` on bipartite-00 lr .003: 0.055 vs 0.053 at epoch 100,
  0.109 vs 0.118 at 200, 0.292 vs 0.361 at 500; ring-01: 0.049 vs 0.047 (100), 0.100 vs
  0.107 (200). Before escape the prediction is within ~10%, then overshoots by 20% at 500.
- Escape time (first logged epoch with `r_bar` > 0.3; logged every 25 epochs) on bipartite-00
  draw 0: lr .003 -> 725, lr .01 -> 125, lr .03 -> 25. The study budget is 500 epochs and the
  pilot picked lr .003 for bipartite/degree_corrected/ring, so those runs end before escaping.
  With 2000 epochs at lr .003 they all escape: bipartite-00 draw 0 at 725, draw 1 at 825,
  ring-01 at 675 (final r_bar 0.945, 0.945..0.967, eps 3.6 to 3.8).
- Positive feedback: the hidden L1 norm grows with training (0.12 -> 0.31 over 500 epochs at
  lr .003, 0.14 -> 0.72 at 100 epochs at lr .01), so the speed accelerates and escape time scales
  slower than 1/lr (725 : 125 : 25 for lr ratio 1 : 3.3 : 10).
- Model selection can hide an escape: assortative-00 at lr .01 (both draws escape in my
  re-runs) draw 0 reaches r_bar .62 by epoch 100, draw 1 by epoch ~75. The discovery file for draw 0
  nevertheless reports r_bar .097 with best epoch 9 (val AUC .849), because `best_state` is
  the best-validation-AUC epoch, not the last. In my re-run, logged val AUC after escape is
  at most .823 (epochs >= 100), below the .849 the discovery run saw at epoch 9. Hypothesis
  (not verified epoch by epoch; I log val AUC every 25 epochs): some of the 0.097 values in
  the discovery files are early-epoch checkpoints chosen by validation AUC before an
  escape that the final-epoch posterior had made. If so, the plateau counts in the earlier note
  mix "never moved" with "moved, but the selected checkpoint predates it".

### Caveat: escaping is not obviously better

On bipartite-00 and ring-01 the validation AUC after escape is no better (bipartite-00:
.42 to .53 across the whole run, max .58; ring-01 final .546, max .599), because the
dot-product decoder cannot represent these graphs; link loss falls (0.93 -> 0.07 on bipartite)
but that is overfitting to training edges (KL 3 -> 23). So lr .003's choice by the pilot on
sampled log loss is not necessarily a bug in tuning; the "plateau" is TNBBeta's initial
posterior being a reasonable diffuse prior on graphs it cannot fit.

## Summary

| hypothesis | verdict | key evidence |
|---|---|---|
| (a) softplus/sigmoid saturation | refuted | raw outputs ~0 during plateau; sigmoid' .25, softplus' .5 at init; gradients shrink only after escape |
| (b) MC noise in eps gradient | refuted for eps (SNR 2 to 11); supported for p (SNR 0.1) | p controls symmetry breaking, `r_bar` insensitive to eps while p ~ 0.5 |
| (c) lr / epoch budget | supported | drift = lr * ||h||_1 within ~10%; escape at epoch 725 / 125 / 25 for lr .003 / .01 / .03; 2000-epoch runs at lr .003 all escape (675 to 825) |

## Not tested

- Whether a larger head-weight init or an eps/p-specific learning rate removes the plateau
  without hurting results (would need a model change; I did not touch `distributions/` or
  `heads.py`).
- Only one draw per config for the escape-time numbers; discovery stats show it varies by
  draw. The 26% plateau fractions in the earlier note were not re-derived.
- Gradient-noise probe is at the head-weight level (what Adam sees), not an analysis of
  the analytic expected gradient w.r.t. p at p = q = 0.5 (it may be exactly zero by symmetry;
  measured mean norm is 4e-5 vs 3e-4 draw norm, i.e. consistent with zero within noise).
