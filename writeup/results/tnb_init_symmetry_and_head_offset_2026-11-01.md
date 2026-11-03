# Is the symmetric start the cause of the TNBBeta plateau? Test and a head-offset recommendation (2026-11-01)

Date: 2026-11-01. Base commit: `ec2677e` (branch `synthetic-graph-families`; uncommitted working tree,
including the new untracked files listed below). Exploratory and post hoc: 3 graph draws per cell, one
training seed per draw (the manifest seed), mean +/- sd over the 3 draws (sd is `ddof=1`, n = 3, so
treat it as a rough indication only). No hypothesis tests. Nothing under `src/` was modified; the
offsets are applied by monkeypatch in the scripts below.

## Question

`tnb_plateau_mechanism_2026-11-01.md` found TNBBeta at its default head (raw outputs ~ 0, so
p = q = 0.5, eps = softplus(0) + 1e-4 = 0.693) needs ~700 epochs at lr 0.003 to leave `r_bar` ~ 0.1,
and that the `p` gradient is ~50x smaller than eps's and noise-dominated. Hypothesis (A): the
*axis-symmetric start* (p <-> 1-p flips w -> -w) is the cause. (B): what fixed, data-free
initialisation convention would be principled, as the vMF/PS reference code's `softplus(raw) + 1` is?

## Files and commands

New, untracked, exploratory (not part of the package):
`apps/synthetic/init_symmetry_sweep.py` (the sweep), `apps/synthetic/axial_init_shift.py` (axial cost
check), `scripts/slurm/init_symmetry_sweep.sbatch`, `scripts/slurm/axial_init_shift.sbatch`.
Scratch probes (local only, in the session scratchpad
`/tmp/claude-14853/-home-vpathak-tnbbeta-vae/39a83aae-c9e5-4e62-8b02-10e6fc87ddf7/scratchpad/`):
`init_analytics.py`, `mu_signal.py`, `agg.py`, `axial_agg.py`.

Mechanism of an offset: a constant added to the raw p / q / eps head output inside
`GraphVAE._posterior` before the usual sigmoid / softplus (exactly how `initial_kappa` shifts vMF's
raw), so the offset is a fixed parametrisation convention, not a trainable bias, and not a
function of data. `w10`/`w30` multiply the head weight init by 10/30 after construction (symmetry
broken through weights, offsets 0).

```
# analytics (local, 1 thread, nice 10)
PYTHONPATH=. OMP_NUM_THREADS=1 uv run python $S/init_analytics.py $S/init_analytics.json
PYTHONPATH=. OMP_NUM_THREADS=1 uv run python $S/mu_signal.py 5.0 $S/mu_signal.json
# main grid on SLURM: 192 cells = 3 families x 3 draws x 16 conditions, packed 6 workers/task, 8 tasks (job 2288629)
sbatch --array=0-7 scripts/slurm/init_symmetry_sweep.sbatch
# axial cost check (jobs 2288630, and 2288667 re-run of seed 1 after a NaN crash, see below)
sbatch --array=0-2 scripts/slurm/axial_init_shift.sbatch
# aggregation
uv run --no-sync python $S/agg.py $S/sweep_all.jsonl ; python3 $S/axial_agg.py $CKPT/init_symmetry_sweep/axial
```
Per-shard JSONL (with `r_bar`/val-AUC trajectories every 25 epochs) is in
`$TNBBETA_CHECKPOINT_DIR/init_symmetry_sweep/` (`sweep_shard*.jsonl`, `axial/`).

Setup: the pilot configs `bipartite-00` (800 nodes), `ring-00` (300 nodes, informative features),
`assortative-00` (600 nodes), pilot draws 000-002 (seeds 9002000+, 9014000+, 9000000+), DIM = 16,
500 epochs, decoder bias on, as in `graph_search.run_item`; learning rate 0.003 for all three and,
for assortative, additionally the pilot's tuned TNBBeta rate 0.01 (tuned rate is 0.003 for bipartite
and ring, so no separate run). Metrics are those of `graph_search.evaluate`. `r_bar` is the repo's
`posterior_stats` value (norm of the mean of 100 samples, averaged over nodes).
Note ring-00 is not the ring-01 used in the plateau note (there `r_bar` escaped at epoch ~675); here
the default already escapes by epoch ~15 (informative features give larger hidden activations).

## Part 1: analytic / Monte Carlo map (p, q, eps) -> posterior concentration

Fact found along the way (visible in the MC: every `Ew` is bit-identical across d): the marginal law of
w = z . mu under `TNBBetaSpherical` depends on (p, q, eps) only, **not on the ambient dimension d**.
So E[w], E|w| and the population `r_bar = |E[z]| = |E[w]|` are the same function of (p, q, eps) at every
d. d enters only through the uniform-sphere reference (eps = (d-1)/2 for p = 0.5, q ~ 0) and the KL.
200k draws, se of E[w] <= 0.0015:

| (p, q, eps) | E[w] (= population r_bar) | E abs w | 100-sample r_bar (as logged) |
|---|---|---|---|
| .5, .5, .693 (default) | -0.002 | 0.491 | 0.098 |
| .6, .5, .693 | 0.133 | 0.503 | 0.162 |
| .75, .5, .693 | 0.350 | 0.572 | 0.360 |
| .9, .5, .693 | 0.625 | 0.725 | 0.629 |
| .5, .5, 2.0 | 0.002 | 0.293 | 0.101 |
| .5, .5, 8.5 | -0.0001 | 0.139 | 0.099 |
| .5, .1, .693 | -0.003 | 0.561 | 0.097 |
| .5, .9, .693 | -0.001 | 0.312 | 0.098 |
| .75, .5, 2.0 | 0.448 | 0.488 | 0.454 |
| .75, .5, 8.5 | 0.488 | 0.489 | 0.497 |
| .75, .25, .693 | 0.325 | 0.597 | 0.336 |
| vMF kappa = softplus(0)+1 = 1.693 (reference default), d = 3 / 8 / 16 / 32 | 0.480 / 0.205 / 0.105 / 0.053 | | 0.487 / 0.223 / 0.144 / 0.111 |
| vMF kappa = d (repo `initial_kappa`), d = 3 / 8 / 16 / 32 | 0.672 / 0.638 / 0.609 / 0.601 | | 0.676 / 0.641 / 0.615 / 0.606 |

Readings:
* Population E[w] at the default is 0 (-0.002 +/- 0.001), and no change of eps alone (0.69 -> 8.5) or of q
  alone (0.1, 0.9) moves it (|E[w]| < 0.003). Only p does. The logged `r_bar` ~ 0.098 at the default is
  the sampling floor of the estimator (norm of the mean of 100 uniform unit vectors is 1/sqrt(100) = 0.1
  at any d), not a measured concentration.
* The reference vMF default (kappa = 1.69) is itself nearly uniform at d >= 16 (E[w] = 0.105 at d = 16,
  0.053 at d = 32), so "the reference code just uses +1 and default init" does not put vMF at a
  concentrated start in high d; the repo's `vmf_default` and `vmf_init` differ by a lot (r_bar 0.14 vs 0.61).
  The fair high-d comparison is `vmf_default` (r_bar 0.14), not `vmf_init`.
* Autograd d E[w] / d(raw) at d-independent fixed seed (20k draws): at the default d E[w]/d raw_p = 0.33
  (large: the *expected embedding* is very sensitive to p there) while d/d raw_eps = 5e-4 and d/d raw_q =
  3e-4. At p = .75 these are 0.29, 0.089 (eps), 0.030 (q); at p = .9, 0.20, 0.13, 0.040. So at the symmetric
  point the mean embedding's only lever is p, which is the coordinate the plateau note found
  gradient-starved and noisy.

### Gradient reaching mu from a positive pair

`mu_signal.py`: two nodes with mu_i, mu_j at 60 degrees, loss = softplus(-t z_i . z_j), t = 5 (the model's
initial temperature, no bias), 40000 paired draws; reports the mean *descent* component on mu_i along the
tangent direction toward mu_j (positive = pulls mu_i toward mu_j; se in brackets; the gradient
w.r.t. a unit-vector input also has a radial part, which is projected out):

| model | d=3 | d=8 | d=16 | d=32 |
|---|---|---|---|---|
| TNB default (.5,.5,.69) | -0.010 (.008) | -0.106 (.006) | **-0.169** (.004) | **-0.212** (.004) |
| TNB p=.6 | +0.031 | -0.083 | -0.147 | -0.186 |
| TNB p=.75 | +0.241 | +0.053 | -0.024 | -0.063 |
| TNB p=.9 | +0.597 | +0.302 | +0.203 | +0.154 |
| TNB eps=2 (p=.5) | -0.156 | -0.004 | -0.011 | -0.023 |
| vMF kappa=1.69 | +0.475 | +0.082 | +0.021 | +0.006 |
| vMF kappa=d | +0.801 | +0.630 | +0.536 | +0.497 |

This is stronger than "the signal is weak": at the symmetric start and d >= 8 the expected gradient on
mu points *away* from the neighbour. Plausible reading (not separately tested): with E[z] = 0 the loss
is dominated by its even part in z_i . z_j (a convex softplus), which is reduced by making the two axes
orthogonal; only a non-zero first moment (E[w] > 0, i.e. p away from .5) lets the odd part, which
rewards alignment, win. The threshold in p grows with d (p = .75 is positive at d = 3, 8 but not 16, 32;
p = .9 is positive at all four). This is one idealised two-node probe, not the full GCN loss.

## Part 2: the symmetry test (SLURM grid, 3 draws)

Conditions (targets given as the initial p, q, eps; raw offsets are logit(p), logit(q), softplus^-1(eps)):
`default`; `p0.6/p0.75/p0.9` (p only); `eps2`, `eps8.5` (eps only; 8.5 = (d-1)/2 + 1, d = 16); `q0.1`,
`q0.9`; `w10`/`w30` (head weight init x10/x30); `p0.58` (E[w] = 0.105, matches vMF at its reference
default for d = 16); `p0.75q0.25`, `p0.75eps2`, `p0.75eps8.5`; baselines `vmf_init` (kappa = d) and
`vmf_default` (softplus(0)+1). `r_bar0` is `r_bar` at the untrained model. "Escaped" = `r_bar` > 0.3 at
some logged epoch (every 5 epochs); for inits whose `r_bar0` is already above 0.3 this is trivially true
and is flagged "start>0.3". "r_bar final" is the LAST epoch, whereas AUC / log loss are from the
best-validation-AUC checkpoint (`run_once`), so the two can disagree (see caveat below). "log loss" is the
raw sampled log loss, "recal" the validation-Platt-recalibrated one.

### Assortative-00 (the case where TNBBeta lagged most), lr 0.003

| cond | r_bar0 | escaped | r_bar final | best ep | centre AUC | sampled AUC | log loss | recal |
|---|---|---|---|---|---|---|---|---|
| default | 0.097+-0.002 | 0/3 | 0.097+-0.000 | 2 | 0.764+-0.053 | 0.506+-0.013 | 0.696+-0.002 | 0.693+-0.000 |
| p0.58 | 0.141+-0.001 | 0/3 | 0.165+-0.011 | 2 | 0.798+-0.004 | 0.537+-0.008 | 0.691+-0.001 | 0.692+-0.002 |
| p0.6 | 0.162+-0.001 | 1/3 (450) | 0.478+-0.395 | 161 | 0.799+-0.004 | 0.627+-0.145 | 0.640+-0.085 | 0.633+-0.101 |
| **p0.75** | 0.361+-0.001 | start>0.3 | 0.967+-0.001 | 276 | 0.812+-0.010 | **0.810+-0.008** | **0.522+-0.022** | 0.513+-0.023 |
| p0.9 | 0.630+-0.001 | start>0.3 | 0.968+-0.001 | 157 | 0.805+-0.004 | 0.802+-0.010 | 0.565+-0.025 | 0.542+-0.022 |
| eps2 | 0.098+-0.002 | 3/3, med 365 | 0.929+-0.009 | 380 | 0.805+-0.007 | 0.712+-0.169 | 0.586+-0.098 | 0.572+-0.108 |
| eps8.5 | 0.098+-0.002 | 3/3, med 295 | 0.942+-0.004 | 283 | 0.806+-0.009 | 0.652+-0.161 | 0.625+-0.109 | 0.617+-0.104 |
| q0.1 | 0.096+-0.002 | 0/3 | 0.096+-0.001 | 2 | 0.763+-0.050 | 0.510+-0.006 | 0.696+-0.001 | 0.694+-0.001 |
| q0.9 | 0.098+-0.002 | 3/3, med 305 | 0.904+-0.056 | 383 | 0.807+-0.006 | 0.784+-0.046 | 0.565+-0.091 | 0.539+-0.065 |
| w10 | 0.097+-0.002 | 0/3 | 0.104+-0.012 | 2 | 0.792+-0.015 | 0.510+-0.006 | 0.696+-0.001 | 0.694+-0.001 |
| w30 | 0.097+-0.002 | 2/3, med 100 | 0.682+-0.500 | 79 | 0.798+-0.006 | 0.603+-0.166 | 0.640+-0.098 | 0.638+-0.096 |
| p0.75q0.25 | 0.337+-0.002 | start>0.3 | 0.966+-0.001 | 196 | 0.805+-0.007 | 0.779+-0.049 | 0.577+-0.066 | 0.551+-0.060 |
| p0.75eps2 | 0.454+-0.002 | start>0.3 | 0.960+-0.002 | 96 | 0.809+-0.005 | 0.799+-0.015 | 0.579+-0.066 | 0.539+-0.029 |
| p0.75eps8.5 | 0.496+-0.001 | start>0.3 | 0.953+-0.002 | 88 | 0.810+-0.006 | 0.804+-0.011 | 0.568+-0.063 | 0.528+-0.026 |
| vmf_init | 0.614+-0.001 | start>0.3 | 0.642+-0.001 | 66 | 0.807+-0.009 | 0.805+-0.009 | 0.565+-0.039 | 0.515+-0.018 |
| vmf_default | 0.144+-0.001 | 3/3, med 455 | 0.463+-0.045 | 168 | 0.808+-0.011 | 0.625+-0.153 | 0.672+-0.034 | 0.647+-0.080 |

Same conditions at the pilot's tuned lr 0.01, bipartite-00 and ring-00 at lr 0.003, and the full
set of numbers: see the appendix table at the end (copied unedited from `agg.py`).

### Answer to (A)

* **Breaking the symmetry through p alone removes the plateau, provided the offset is large enough.**
  On assortative at lr 0.003 (3/3 draws) default: sampled AUC 0.506 +- 0.013, raw log loss 0.696 (chance;
  the posterior never moved); p = 0.75: sampled AUC 0.810 +- 0.008, log loss 0.522 +- 0.022 (recal
  0.513), p = 0.9: 0.802 / 0.565. That equals what the default reaches only after a 3.3x higher lr
  (0.01: sampled AUC 0.811 +- 0.007, log loss 0.512 +- 0.016), and matches `vmf_init`
  (0.805 / 0.565 / recal 0.515). A small offset is not enough: p = 0.6 escaped in 1/3 draws (epoch
  ~450) and p = 0.58 in 0/3 (the offset matching vMF's reference-default E[w]; the posterior stayed
  within r_bar 0.14-0.17).
* **The "eps only" control did not behave as predicted.** The prediction was that eps alone does not
  help. At lr 0.003 on assortative, eps = 2 and 8.5 and q = 0.9 each escaped in 3/3 draws (median first
  epoch with r_bar > 0.3: 365, 295, 305; default: 0/3), although the population E[w] is 0 at their
  initialisation (Part 1). So eps/q alone do break the stall, more slowly, through some route I did not
  identify (the analytic map says eps/q change the d E[w]/d raw_p lever, 0.33 -> 0.43-0.49; I did not
  verify this is the mechanism). Symmetry is therefore not the *only* cause of the plateau, but it
  is the factor that, when removed directly, removes the plateau fastest and most reliably.
  q = 0.1 behaves like the default (0/3).
* Larger head weights are not a safe substitute: w10 did nothing on assortative (0/3), w30 escaped in
  2/3 with large variance, and on ring-00 w30 ruined training (log loss 0.945 +- 0.327, AUC 0.70
  vs 0.79 default).
* **Where the grid cannot say much.** bipartite-00: every condition including vMF is at AUC ~ 0.49-0.51
  (the dot-product decoder cannot represent it; plateau note). Initialisations that escape pay in raw
  log loss (p0.75: 0.724 +- 0.041; vmf_init 0.727; default 0.697; recalibrated all 0.693-0.694), i.e. the
  escaped posterior is over-confident on a graph with nothing to learn, the same as vMF at kappa = d.
  ring-00 at lr 0.003: the default escapes by epoch ~15 and every p/eps/q start is within noise
  (sampled AUC 0.78-0.79); only q0.9, w10, w30 are worse. So the plateau is not universal: it appeared
  on assortative-00 here and on bipartite-00 / ring-01 in the earlier note.

### Caveat that affects the lr 0.01 rows (and "best ep")

Checkpoints are the best *centre*-AUC validation epoch, and centre AUC cannot see the posterior's
shape. At lr 0.01 on assortative every run's final-epoch r_bar is 0.98, but several rows have large sd in
sampled AUC (e.g. eps2 0.60 +- 0.18, q0.9 0.60 +- 0.18, p0.58 0.72 +- 0.16): in the offending draws
the chosen checkpoint is an early epoch whose posterior had not yet moved (e.g. q0.9 draw 2: best
epoch 5, checkpoint `r_bar` 0.10, sampled AUC 0.487, centre AUC 0.818). The default at lr 0.01 got
lucky on all 3 draws (best epochs 109-183). This is a selection artefact, not training failure, and
it equally limits how much the lr 0.01 table can say about the inits. `p0.75` at lr 0.01 had it too (draw 0:
best epoch 2, sampled AUC 0.743).

## Part 3: cost on a task that wants antipodal bimodality (axial, cheap check)

`apps.synthetic.axial_recovery` (MLP VAE, TNBBeta only, 200 epochs, seeds 0-2), with a fixed raw-p
offset giving initial p in {default 0.5, 0.75, 0.9}, latent_dim 2 and 5. Per-seed values in brackets.
Bimodal needs `m_mean = eps_mean - (d-1)/2 < 0`.

| latent_dim | p start | seeds done | angle_error (centre) | test_ll | test_kl | p_mean | m_mean | centre_axis_resultant |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.75 | [0, 1, 2] | 0.352 (0.082, 0.749, 0.225) | 150.332 (148.093, 152.629, 150.273) | 6.943 (9.237, 4.713, 6.878) | 1.000 (1.000, 1.000, 1.000) | 136.762 (349.494, 1.241, 59.550) | 0.499 (0.807, 0.087, 0.604) |
| 2 | 0.9 | [0, 1, 2] | 0.709 (0.654, 0.757, 0.715) | 151.796 (151.554, 151.095, 152.740) | 5.381 (4.545, 7.135, 4.463) | 1.000 (1.000, 1.000, 1.000) | 2.839 (1.395, 5.809, 1.313) | 0.131 (0.199, 0.049, 0.145) |
| 2 | none | [0, 2] | 0.446 (0.624, 0.268) | 151.908 (151.730, 152.087) | 4.641 (4.906, 4.376) | 0.766 (1.000, 0.531) | 9.293 (3.282, 15.304) | 0.322 (0.104, 0.540) |
| 5 | 0.75 | [0, 1, 2] | 0.703 (0.689, 0.742, 0.678) | 150.627 (150.289, 150.751, 150.841) | 6.914 (6.791, 7.015, 6.935) | 0.988 (0.988, 0.989, 0.988) | 5.982 (5.940, 5.753, 6.253) | 0.072 (0.051, 0.078, 0.086) |
| 5 | 0.9 | [0, 1, 2] | 0.685 (0.723, 0.684, 0.649) | 150.389 (150.097, 150.767, 150.303) | 7.115 (7.043, 7.138, 7.165) | 0.990 (0.989, 0.990, 0.990) | 6.344 (6.261, 5.749, 7.022) | 0.149 (0.100, 0.172, 0.175) |
| 5 | none | [0, 1, 2] | 0.534 (0.713, 0.710, 0.178) | 150.930 (150.064, 149.927, 152.800) | 6.183 (6.785, 7.730, 4.034) | 0.170 (0.012, 0.008, 0.489) | 8.018 (5.587, 5.461, 13.006) | 0.368 (0.054, 0.050, 0.999) |

* The unmodified default crashed on seed 1, latent_dim 2 (a NaN in the reconstruction `Normal`, in
  the code as it stands at base commit with no offset), so that cell has 2 seeds. The p-offset runs
  did not crash there.
* No run in any condition reached `m < 0` (the best is m = 1.24, seed 1, p0.75, dim 2), so no start is
  shown to be *harmful* to reaching the bimodal regime, nor helpful: the default does not find it either
  (consistent with the 2026-10-07 axial note). `p_mean` goes to ~1 (or, when the head lands on the other side,
  ~0.01, the mirror image) irrespective of the start.
* Axis recovery (angle error; ~0.70-0.78 = chance, lower better) is erratic across seeds at n = 3:
  dim 2 default 0.62, 0.27 (seed 1 crashed); p0.75 0.08, 0.75, 0.23; p0.9 0.65, 0.76, 0.72; dim 5
  default 0.71, 0.71, 0.18; p0.75 0.69, 0.74, 0.68; p0.9 0.72, 0.68, 0.65. I cannot distinguish the
  starts from this; the signal is dominated by which seeds happen to find the axis. test_ll: default dim 2
  151.9 (n = 2), p0.75 150.3, p0.9 151.8; dim 5 150.9 / 150.6 / 150.4, differences within seed spread.
  Honest summary: no demonstrated cost, and no demonstrated benefit, of a p-biased start on this task.
  A p-biased start is a *start*, not a constraint (p moves; `p_mean` ends ~1 either way); it breaks
  the p <-> 1-p mirror at init, which is a gauge for the family but not for the likelihood.
  Not tested: SBM / GraphVAE bimodality (m < 0) with a biased start.

## Part 4 (B): recommendation

**Recommended convention (tested at d = 16 only for training; analytically d-independent in r_bar):**
leave q and eps at their defaults and add a fixed offset `P_OFFSET = ln 3` (p = 0.75 at init) to the
raw p output, i.e. `p = sigmoid(raw_p + ln 3)`. This is a parametrisation convention analogous to
the reference's `+1` in `softplus(raw) + 1`: a constant of the family, independent of dataset, labels,
lr and latent_dim; with `raw ~ 0` the init is p = .75, q = .5, eps = .693.

Properties (all from the MC above, identical at every d because the marginal of w is d-free):

| d | init r_bar, population / logged (p=.75, q=.5, eps=.693) | vMF at reference default, population / logged | vMF kappa=d |
|---|---|---|---|
| 3 | 0.350 / 0.360 | 0.480 / 0.487 | 0.672 / 0.676 |
| 8 | 0.350 / 0.360 | 0.205 / 0.223 | 0.638 / 0.641 |
| 16 | 0.350 / 0.360 | 0.105 / 0.144 | 0.609 / 0.615 |
| 32 | 0.350 / 0.360 | 0.053 / 0.111 | 0.601 / 0.606 |

* Not degenerate: r_bar 0.35 is neither the uniform floor nor a delta; E|w| = 0.57; KL to uniform is
  moderate (p = .6 gives KL 3.55 vs 3.34 at the default, d = 16, from the quadrature note).
* Does it need to depend on d? For r_bar, no. For the *gradient on mu* the two-node probe says the
  margin shrinks with d: p = .75 is positive for d = 3, 8 and slightly negative for d = 16, 32, while
  p = .9 is positive at all four (+0.60, +0.30, +0.20, +0.15). Empirically p = .75 worked at d = 16 in
  training (assortative, lr 0.003), which the probe alone would not have predicted, so the probe is
  not a sufficient guide. If d >> 16 is of interest, p = 0.9 (offset ln 9, r_bar = 0.63, matching vMF's
  kappa = d start) is the safer d-independent choice; its measured cost here was slightly worse raw
  log loss (assortative lr 0.003: 0.565 vs 0.522, n = 3; bipartite 0.795 vs 0.724). I would not make
  the offset a function of d without a d > 16 training test (none run).
* q and eps: leave at defaults. p.75 + eps2 / eps8.5 / q0.25 were no better than p.75 alone (sampled AUC
  0.799 / 0.804 / 0.779 vs 0.810, log loss 0.579 / 0.568 / 0.577 vs 0.522), and eps/q-only changes did
  not fix the mean-direction lever. Note default eps = 0.693 is far below the uniform value (d-1)/2 at
  d = 16 (m = -6.8), i.e. the default posterior is already deep in the `m < 0` regime numerically,
  although its second-mode mass need not be (see the CLAUDE.md note on `second_mode_mass`).
* Cost: (i) changes the init distribution and so every future result relative to checkpoints trained
  with the current head (old `state_dict`s load but mean a different p at the same weights); (ii) p-start
  was not shown harmful on axial, but the evidence is weak (above); (iii) escaped posteriors
  over-confident on graphs with nothing to learn (bipartite raw log loss), the same as `vmf_init`.

Concrete patch (NOT applied; sketch against `src/tnbbeta_vae/models/heads.py`):

```diff
--- a/src/tnbbeta_vae/models/heads.py
+++ b/src/tnbbeta_vae/models/heads.py
@@
 _PARAM_EPS = 1e-4
+# Offset added to the raw p output so a zero head starts at p = 0.75, not the
+# axis-symmetric p = 0.5 (where E[z . mu] = 0 and the mean direction gets no
+# useful gradient; see writeup/results/tnb_init_symmetry_and_head_offset_2026-11-01.md).
+# A fixed convention, like vmf_kappa's "+ 1": independent of data and latent_dim.
+_P_RAW_OFFSET = math.log(3.0)
@@ def tnbbeta_posterior(
-    p = raw_p.squeeze(-1).sigmoid().clamp(_PQ_CLAMP, 1 - _PQ_CLAMP)
+    p = (raw_p.squeeze(-1) + _P_RAW_OFFSET).sigmoid().clamp(_PQ_CLAMP, 1 - _PQ_CLAMP)
```
Also needed if applied: the uniform prior and `posterior_centre` are unaffected (they use p, not raw);
`apps/eval` code that reads raw p directly (a grep of `apps/` and `src/` outside `heads.py` for `raw_p`/raw splits found none) should be checked, and
the docstring ("p and q squashed to (0, 1)") updated; existing checkpoints would need the offset
removed (use `raw_p - ln 3` in their loader) to reproduce old posteriors. Tests: a zero raw head gives p = .75.
The sweep reproduces this patch bit for bit through monkeypatching (`init_symmetry_sweep.py`, condition `p0.75`).

## What this does not show

* One graph family where the plateau bites at lr 0.003 (assortative-00); bipartite is unlearnable
  and ring-00 does not plateau. Other families, other graphs, d != 16, real graphs: untested.
* 3 draws per cell, sd on n = 3. Differences of a few hundredths in log loss (p.75 vs p.9, p.75 vs
  vmf_init) are within noise; only "default vs any large-enough p offset" (0/3 vs 3/3, AUC 0.51 vs 0.81)
  and "p0.58-0.6 does not suffice" are clear-cut.
* "Escape epoch" for p >= 0.75 is by construction -1; I did not measure how long a p-biased start
  *remains* in a concentrated state versus drifts. A second hypothesis, that part of the plateau is
  the optimiser rate on the shared head weights (plateau note, (c)), is not refuted: the default at lr 0.01
  also leaves the plateau without any offset.
* The eps-only / q0.9 escapes contradict the simplest "symmetry only" story and are unexplained.
* The mean-direction gradient probe is a two-node idealisation at temperature 5, no decoder bias, one
  fixed 60-degree geometry.
* The tuned-lr comparison in the pilot is not re-derived: this note holds the lr fixed and varies init.
  It does not show that, with the offset, TNBBeta's tuned lr would still be 0.003 or that it beats
  tuned vMF/PS.
* Checkpoint selection by centre AUC confounds several lr 0.01 rows (caveat above).

## Appendix: full aggregated table (mean +- sd over 3 draws; bipartite, ring, assortative at both rates)

| family | lr | cond | n | r_bar0 | escaped (>0.3 by ep500) / median first ep | r_bar final | best ep | centre AUC | sampled AUC | log loss | recal log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| assortative | 0.003 | default | 3 | 0.097+-0.002 | 0/3 | 0.097+-0.000 | 2 | 0.764+-0.053 | 0.506+-0.013 | 0.696+-0.002 | 0.693+-0.000 |
| assortative | 0.003 | p0.58 | 3 | 0.141+-0.001 | 0/3 | 0.165+-0.011 | 2 | 0.798+-0.004 | 0.537+-0.008 | 0.691+-0.001 | 0.692+-0.002 |
| assortative | 0.003 | p0.75q0.25 | 3 | 0.337+-0.002 | 3/3 (start>0.3: 3) | 0.966+-0.001 | 196 | 0.805+-0.007 | 0.779+-0.049 | 0.577+-0.066 | 0.551+-0.060 |
| assortative | 0.003 | p0.75eps2 | 3 | 0.454+-0.002 | 3/3 (start>0.3: 3) | 0.960+-0.002 | 96 | 0.809+-0.005 | 0.799+-0.015 | 0.579+-0.066 | 0.539+-0.029 |
| assortative | 0.003 | p0.75eps8.5 | 3 | 0.496+-0.001 | 3/3 (start>0.3: 3) | 0.953+-0.002 | 88 | 0.810+-0.006 | 0.804+-0.011 | 0.568+-0.063 | 0.528+-0.026 |
| assortative | 0.003 | vmf_init | 3 | 0.614+-0.001 | 3/3 (start>0.3: 3) | 0.642+-0.001 | 66 | 0.807+-0.009 | 0.805+-0.009 | 0.565+-0.039 | 0.515+-0.018 |
| assortative | 0.003 | vmf_default | 3 | 0.144+-0.001 | 3/3, med 455 | 0.463+-0.045 | 168 | 0.808+-0.011 | 0.625+-0.153 | 0.672+-0.034 | 0.647+-0.080 |
| assortative | 0.003 | p0.6 | 3 | 0.162+-0.001 | 1/3, med 450 | 0.478+-0.395 | 161 | 0.799+-0.004 | 0.627+-0.145 | 0.640+-0.085 | 0.633+-0.101 |
| assortative | 0.003 | p0.75 | 3 | 0.361+-0.001 | 3/3 (start>0.3: 3) | 0.967+-0.001 | 276 | 0.812+-0.010 | 0.810+-0.008 | 0.522+-0.022 | 0.513+-0.023 |
| assortative | 0.003 | p0.9 | 3 | 0.630+-0.001 | 3/3 (start>0.3: 3) | 0.968+-0.001 | 157 | 0.805+-0.004 | 0.802+-0.010 | 0.565+-0.025 | 0.542+-0.022 |
| assortative | 0.003 | eps2 | 3 | 0.098+-0.002 | 3/3, med 365 | 0.929+-0.009 | 380 | 0.805+-0.007 | 0.712+-0.169 | 0.586+-0.098 | 0.572+-0.108 |
| assortative | 0.003 | eps8.5 | 3 | 0.098+-0.002 | 3/3, med 295 | 0.942+-0.004 | 283 | 0.806+-0.009 | 0.652+-0.161 | 0.625+-0.109 | 0.617+-0.104 |
| assortative | 0.003 | q0.1 | 3 | 0.096+-0.002 | 0/3 | 0.096+-0.001 | 2 | 0.763+-0.050 | 0.510+-0.006 | 0.696+-0.001 | 0.694+-0.001 |
| assortative | 0.003 | q0.9 | 3 | 0.098+-0.002 | 3/3, med 305 | 0.904+-0.056 | 383 | 0.807+-0.006 | 0.784+-0.046 | 0.565+-0.091 | 0.539+-0.065 |
| assortative | 0.003 | w10 | 3 | 0.097+-0.002 | 0/3 | 0.104+-0.012 | 2 | 0.792+-0.015 | 0.510+-0.006 | 0.696+-0.001 | 0.694+-0.001 |
| assortative | 0.003 | w30 | 3 | 0.097+-0.002 | 2/3, med 100 | 0.682+-0.500 | 79 | 0.798+-0.006 | 0.603+-0.166 | 0.640+-0.098 | 0.638+-0.096 |
| assortative | 0.01 | default | 3 | 0.097+-0.001 | 3/3, med 80 | 0.979+-0.000 | 132 | 0.809+-0.010 | 0.811+-0.007 | 0.512+-0.016 | 0.505+-0.006 |
| assortative | 0.01 | p0.58 | 3 | 0.141+-0.001 | 3/3, med 55 | 0.981+-0.001 | 69 | 0.804+-0.009 | 0.720+-0.156 | 0.564+-0.110 | 0.562+-0.111 |
| assortative | 0.01 | p0.75q0.25 | 3 | 0.337+-0.002 | 3/3 (start>0.3: 3) | 0.980+-0.001 | 93 | 0.807+-0.010 | 0.777+-0.053 | 0.579+-0.072 | 0.553+-0.064 |
| assortative | 0.01 | p0.75eps2 | 3 | 0.454+-0.002 | 3/3 (start>0.3: 3) | 0.981+-0.000 | 22 | 0.807+-0.010 | 0.799+-0.003 | 0.579+-0.048 | 0.533+-0.020 |
| assortative | 0.01 | p0.75eps8.5 | 3 | 0.496+-0.001 | 3/3 (start>0.3: 3) | 0.981+-0.001 | 24 | 0.806+-0.012 | 0.802+-0.005 | 0.566+-0.040 | 0.524+-0.012 |
| assortative | 0.01 | vmf_init | 3 | 0.614+-0.001 | 3/3 (start>0.3: 3) | 0.840+-0.003 | 364 | 0.809+-0.013 | 0.811+-0.015 | 0.505+-0.014 | 0.501+-0.013 |
| assortative | 0.01 | vmf_default | 3 | 0.144+-0.001 | 3/3, med 150 | 0.893+-0.000 | 177 | 0.811+-0.003 | 0.634+-0.156 | 0.634+-0.098 | 0.634+-0.099 |
| assortative | 0.01 | p0.6 | 3 | 0.161+-0.001 | 3/3, med 50 | 0.981+-0.001 | 29 | 0.804+-0.007 | 0.637+-0.145 | 0.631+-0.099 | 0.629+-0.105 |
| assortative | 0.01 | p0.75 | 3 | 0.361+-0.002 | 3/3 (start>0.3: 3) | 0.980+-0.000 | 115 | 0.807+-0.009 | 0.785+-0.037 | 0.575+-0.062 | 0.549+-0.051 |
| assortative | 0.01 | p0.9 | 3 | 0.630+-0.001 | 3/3 (start>0.3: 3) | 0.977+-0.001 | 65 | 0.806+-0.007 | 0.806+-0.009 | 0.520+-0.037 | 0.510+-0.019 |
| assortative | 0.01 | eps2 | 3 | 0.098+-0.001 | 3/3, med 90 | 0.978+-0.002 | 83 | 0.805+-0.003 | 0.602+-0.183 | 0.630+-0.114 | 0.629+-0.113 |
| assortative | 0.01 | eps8.5 | 3 | 0.098+-0.001 | 3/3, med 75 | 0.978+-0.001 | 179 | 0.810+-0.009 | 0.811+-0.008 | 0.527+-0.033 | 0.517+-0.019 |
| assortative | 0.01 | q0.1 | 3 | 0.096+-0.001 | 3/3, med 95 | 0.976+-0.003 | 139 | 0.808+-0.014 | 0.709+-0.177 | 0.568+-0.111 | 0.566+-0.110 |
| assortative | 0.01 | q0.9 | 3 | 0.098+-0.001 | 3/3, med 90 | 0.979+-0.000 | 56 | 0.809+-0.011 | 0.604+-0.182 | 0.630+-0.114 | 0.630+-0.113 |
| assortative | 0.01 | w10 | 3 | 0.097+-0.001 | 3/3, med 45 | 0.985+-0.001 | 32 | 0.806+-0.007 | 0.615+-0.166 | 0.635+-0.103 | 0.635+-0.106 |
| assortative | 0.01 | w30 | 3 | 0.097+-0.001 | 3/3, med 15 | 0.977+-0.011 | 137 | 0.806+-0.008 | 0.604+-0.169 | 0.650+-0.080 | 0.644+-0.087 |
| bipartite | 0.003 | default | 3 | 0.097+-0.000 | 0/3 | 0.097+-0.000 | 302 | 0.482+-0.012 | 0.488+-0.013 | 0.697+-0.004 | 0.693+-0.000 |
| bipartite | 0.003 | p0.58 | 3 | 0.140+-0.001 | 0/3 | 0.147+-0.004 | 389 | 0.478+-0.005 | 0.499+-0.007 | 0.695+-0.000 | 0.693+-0.000 |
| bipartite | 0.003 | p0.75q0.25 | 3 | 0.336+-0.001 | 3/3 (start>0.3: 3) | 0.958+-0.001 | 488 | 0.500+-0.010 | 0.499+-0.009 | 0.757+-0.008 | 0.694+-0.001 |
| bipartite | 0.003 | p0.75eps2 | 3 | 0.454+-0.001 | 3/3 (start>0.3: 3) | 0.964+-0.004 | 321 | 0.492+-0.016 | 0.492+-0.015 | 0.761+-0.048 | 0.694+-0.001 |
| bipartite | 0.003 | p0.75eps8.5 | 3 | 0.496+-0.000 | 3/3 (start>0.3: 3) | 0.964+-0.004 | 268 | 0.498+-0.012 | 0.497+-0.013 | 0.745+-0.042 | 0.693+-0.000 |
| bipartite | 0.003 | vmf_init | 3 | 0.614+-0.000 | 3/3 (start>0.3: 3) | 0.621+-0.001 | 128 | 0.494+-0.011 | 0.498+-0.011 | 0.727+-0.020 | 0.694+-0.001 |
| bipartite | 0.003 | vmf_default | 3 | 0.143+-0.000 | 0/3 | 0.149+-0.001 | 191 | 0.511+-0.009 | 0.495+-0.009 | 0.696+-0.002 | 0.693+-0.000 |
| bipartite | 0.003 | p0.6 | 3 | 0.161+-0.001 | 0/3 | 0.179+-0.008 | 438 | 0.488+-0.011 | 0.490+-0.008 | 0.695+-0.001 | 0.693+-0.000 |
| bipartite | 0.003 | p0.75 | 3 | 0.361+-0.001 | 3/3 (start>0.3: 3) | 0.962+-0.003 | 284 | 0.500+-0.007 | 0.506+-0.010 | 0.724+-0.041 | 0.694+-0.000 |
| bipartite | 0.003 | p0.9 | 3 | 0.630+-0.001 | 3/3 (start>0.3: 3) | 0.964+-0.007 | 418 | 0.497+-0.010 | 0.496+-0.010 | 0.795+-0.047 | 0.694+-0.000 |
| bipartite | 0.003 | eps2 | 3 | 0.099+-0.000 | 0/3 | 0.098+-0.001 | 99 | 0.477+-0.017 | 0.495+-0.007 | 0.697+-0.002 | 0.693+-0.001 |
| bipartite | 0.003 | eps8.5 | 3 | 0.099+-0.000 | 0/3 | 0.098+-0.001 | 0 | 0.484+-0.022 | 0.498+-0.010 | 0.697+-0.002 | 0.693+-0.000 |
| bipartite | 0.003 | q0.1 | 3 | 0.097+-0.000 | 0/3 | 0.096+-0.000 | 297 | 0.479+-0.005 | 0.499+-0.005 | 0.695+-0.001 | 0.693+-0.000 |
| bipartite | 0.003 | q0.9 | 3 | 0.098+-0.000 | 0/3 | 0.098+-0.000 | 186 | 0.502+-0.013 | 0.495+-0.017 | 0.696+-0.002 | 0.693+-0.000 |
| bipartite | 0.003 | w10 | 3 | 0.098+-0.000 | 0/3 | 0.097+-0.000 | 238 | 0.489+-0.007 | 0.499+-0.009 | 0.695+-0.001 | 0.694+-0.001 |
| bipartite | 0.003 | w30 | 3 | 0.098+-0.000 | 0/3 | 0.098+-0.001 | 338 | 0.477+-0.011 | 0.501+-0.012 | 0.695+-0.001 | 0.694+-0.001 |
| ring | 0.003 | default | 3 | 0.120+-0.002 | 3/3, med 15 | 0.947+-0.001 | 153 | 0.787+-0.008 | 0.784+-0.008 | 0.546+-0.015 | 0.540+-0.009 |
| ring | 0.003 | p0.58 | 3 | 0.141+-0.044 | 3/3, med 15 | 0.948+-0.002 | 92 | 0.785+-0.011 | 0.782+-0.010 | 0.546+-0.015 | 0.542+-0.007 |
| ring | 0.003 | p0.75q0.25 | 3 | 0.321+-0.082 | 3/3 (start>0.3: 2) | 0.948+-0.001 | 62 | 0.789+-0.013 | 0.788+-0.014 | 0.544+-0.013 | 0.536+-0.009 |
| ring | 0.003 | p0.75eps2 | 3 | 0.426+-0.073 | 3/3 (start>0.3: 3) | 0.948+-0.001 | 151 | 0.785+-0.003 | 0.785+-0.005 | 0.537+-0.017 | 0.536+-0.008 |
| ring | 0.003 | p0.75eps8.5 | 3 | 0.460+-0.066 | 3/3 (start>0.3: 3) | 0.948+-0.002 | 376 | 0.784+-0.006 | 0.783+-0.006 | 0.531+-0.008 | 0.529+-0.006 |
| ring | 0.003 | vmf_init | 3 | 0.612+-0.004 | 3/3 (start>0.3: 3) | 0.910+-0.004 | 163 | 0.792+-0.009 | 0.789+-0.009 | 0.546+-0.028 | 0.533+-0.017 |
| ring | 0.003 | vmf_default | 3 | 0.140+-0.004 | 3/3, med 40 | 0.920+-0.001 | 340 | 0.781+-0.005 | 0.781+-0.004 | 0.536+-0.006 | 0.533+-0.003 |
| ring | 0.003 | p0.6 | 3 | 0.157+-0.053 | 3/3, med 20 | 0.949+-0.002 | 95 | 0.785+-0.012 | 0.782+-0.010 | 0.546+-0.016 | 0.543+-0.006 |
| ring | 0.003 | p0.75 | 3 | 0.347+-0.086 | 3/3 (start>0.3: 2) | 0.948+-0.001 | 231 | 0.783+-0.007 | 0.782+-0.007 | 0.528+-0.009 | 0.528+-0.010 |
| ring | 0.003 | p0.9 | 3 | 0.620+-0.091 | 3/3 (start>0.3: 3) | 0.947+-0.000 | 218 | 0.782+-0.011 | 0.782+-0.011 | 0.539+-0.013 | 0.537+-0.012 |
| ring | 0.003 | eps2 | 3 | 0.137+-0.009 | 3/3, med 8 | 0.923+-0.042 | 240 | 0.785+-0.006 | 0.781+-0.011 | 0.544+-0.022 | 0.544+-0.018 |
| ring | 0.003 | eps8.5 | 3 | 0.144+-0.014 | 3/3, med 2 | 0.949+-0.002 | 174 | 0.789+-0.006 | 0.787+-0.007 | 0.536+-0.015 | 0.533+-0.007 |
| ring | 0.003 | q0.1 | 3 | 0.113+-0.003 | 3/3, med 15 | 0.947+-0.001 | 54 | 0.783+-0.006 | 0.781+-0.005 | 0.552+-0.012 | 0.545+-0.009 |
| ring | 0.003 | q0.9 | 3 | 0.133+-0.004 | 3/3, med 9 | 0.899+-0.050 | 216 | 0.753+-0.051 | 0.733+-0.083 | 0.580+-0.074 | 0.583+-0.078 |
| ring | 0.003 | w10 | 3 | 0.441+-0.149 | 3/3 (start>0.3: 2) | 0.845+-0.025 | 200 | 0.741+-0.016 | 0.737+-0.011 | 0.579+-0.007 | 0.579+-0.010 |
| ring | 0.003 | w30 | 3 | 0.821+-0.045 | 3/3 (start>0.3: 3) | 0.781+-0.040 | 157 | 0.698+-0.040 | 0.656+-0.074 | 0.945+-0.327 | 1.110+-0.479 |
