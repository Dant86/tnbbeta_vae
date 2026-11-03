# Why free vMF / power-spherical stay at the prior: a fixed-kappa sweep and training dynamics, 2026-10-30

- **Date:** 2026-10-30
- **Code:** branch `fixed-m-sweep` (PR #68) at `d966a18` (`Add fixed_kappa for
  vMF/power-spherical, per-epoch trajectory logging, and kappa sweep/dynamics scripts`),
  on top of `master` `9eb8071`. Follow-up to
  `fixed_m_sweeps_baselines_and_parameter_probes_2026-10-30.md`, whose "Not established"
  item this addresses.
- **Commands** (featureless graphs, 5 seeds each; every cell has its own `--results-name`):

  ```bash
  sbatch --array=0-19 scripts/slurm/fixed_kappa_sweep_graphs.sbatch   # 160 cells
  sbatch --array=0-14 scripts/slurm/kappa_dynamics_graphs.sbatch      # 120 cells
  uv run python -m apps.eval.kappa_table                              # the tables below
  ```

  (Task 0 of each array ran first as a canary, jobs 2279457 / 2279458; the rest were jobs
  2279463 / 2279468. All 33 tasks completed.)
- **What was added:** `GraphVAEConfig.fixed_kappa` / `--fixed-kappa` (vMF and power-spherical;
  the head shrinks by one output), and `--log-trajectory`, which writes
  `<results-name>_trajectory.json`: per epoch (every epoch below 10, then every 5th) kappa
  mean/median/min/max, `r_bar`, loss terms, gradient norm of the kappa head row vs. the
  direction rows, and validation AUC, plus the untrained initialization as epoch -1. The
  statistics are sampled under a forked RNG; a CPU test confirms the returned metrics are
  identical with and without logging.

## Question

At TNBBeta's matched dimension, free-kappa vMF and power-spherical posteriors on Cora and
MAG-CS are indistinguishable from the prior (`r_bar` at the uniform floor), score about
chance when decoded from samples, and trail TNBBeta by about 0.1 AUC. Is that because the
sphere families cannot represent what link prediction needs, or because free kappa is not
learned?

## Summary

1. **It is kappa learning, not the family.** With kappa *pinned* at `d` (ambient dimension),
   vMF and power-spherical match or beat free-epsilon TNBBeta and are within about 0.01 of
   the Gaussian on all four graphs (table 1). Any fixed kappa from `0.5 d` to `2 d` gives
   most of the gain; `5 d` is slightly worse.
2. **Free kappa barely moves in the training budget.** On Cora and MAG-CS the seed-mean
   kappa goes `1.69 -> 2.5` (vMF) and `0.69 -> 1.3-1.5` (power-spherical) over 200 epochs,
   `r_bar` stays at the floor, and the link loss ends at about 0.70 (chance is `ln 2 =
   0.693`; fixed kappa reaches about 0.63). Only Pubmed (400 epochs) escapes, to kappa
   about 27.
3. **TNBBeta also starts at the prior and also sits there for about 100 epochs.** Its
   `r_bar` is at the floor and `p`, `q`, `epsilon` stay within 0.5-0.55 / 0.7-0.8 of their
   initial values (0.5, 0.5, 0.69) for the first 50 epochs, and
   its untrained validation AUC is *lower* than vMF's (0.56 vs. 0.73 on Cora). It takes off
   between epochs 100 and 150 on every dataset (`q` 0.6 -> 0.85-0.94, `r_bar` 0.14 -> 0.68,
   validation AUC 0.65 -> 0.78 on Cora) and is still improving at epoch 195. So TNBBeta's
   edge over free vMF/PS at a 200-epoch budget looks like a difference in *how long the
   escape from the prior takes*, not a qualitative one. This is the reachability picture
   (kappa needs a raw output of order `d`; `q` needs a logit of about 3), but **not yet
   tested**: see "Not established".
4. **The "TNBBeta beats vMF/PS" gap in the earlier note was partly a configuration
   artifact.** vMF at its *own* selected dimension (ambient 9) scores 0.784 / 0.746 / 0.823 /
   0.893 on Cora / Citeseer / Pubmed / MAG-CS, versus 0.752 / 0.735 / 0.814 / 0.853 at
   TNBBeta's dimension. The gap to TNBBeta shrinks accordingly (Cora 0.06 instead of 0.09,
   MAG-CS 0.05 instead of 0.09), although it does not vanish.

## Table 1: test AUC, mean +- std over 5 seeds

`matched_*`: TNBBeta's selected dimension (Cora 65, Citeseer 17, Pubmed 17, MAG-CS 33
ambient; Gaussian `d - 1`), free parameters. `own_*`: the family's own selected Figure 5
configuration. `x<k>`: kappa pinned at `k * d`, at TNBBeta's dimension. **`x1` is the
a-priori setting; the other multiples were scanned, so a best-of-four number would be
optimistic** (selected on test AUC, no validation selection across multiples).

| | Cora | Citeseer | Pubmed | MAG-CS |
|---|---|---|---|---|
| TNBBeta free (matched) | 0.841 | 0.770 | 0.826 | 0.943 |
| Gaussian (matched) | 0.851 | 0.784 | 0.842 | 0.938 |
| vMF free (matched) | 0.752 | 0.735 | 0.814 | 0.853 |
| power-spherical free (matched) | 0.748 | 0.705 | 0.786 | 0.838 |
| vMF free (own `d`) | 0.784 | 0.746 | 0.823 | 0.893 |
| power-spherical free (own `d`) | 0.739 | 0.700 | 0.813 | 0.861 |
| vMF `x0.5` | 0.854 | 0.790 | 0.815 | 0.925 |
| **vMF `x1`** | **0.854** | **0.791** | **0.840** | **0.947** |
| vMF `x2` | 0.839 | 0.778 | 0.836 | 0.943 |
| vMF `x5` | 0.830 | 0.758 | 0.827 | 0.936 |
| power-spherical `x0.5` | 0.859 | 0.770 | 0.804 | 0.923 |
| **power-spherical `x1`** | **0.865** | **0.790** | **0.830** | **0.944** |
| power-spherical `x2` | 0.860 | 0.787 | 0.842 | 0.947 |
| power-spherical `x5` | 0.841 | 0.775 | 0.835 | 0.941 |

Standard deviations are 0.002-0.021 (full table from `apps.eval.kappa_table`). The matched
free rows differ from the earlier note's baselines by at most 0.004 (e.g. Cora TNBBeta
0.838 -> 0.841, vMF 0.756 -> 0.752) for the same seeds; I have not investigated the cause
(GPU nondeterminism is the likely one) and it is below the seed spread.

## Table 2: training dynamics of the free runs (seed means)

`init val AUC` is the validation AUC of the *untrained* model (epoch -1, centres).
`r_bar` is the norm of the mean of **100** samples here (uniform floor about 0.1; the
earlier note used 32 samples, floor 0.177), so it is not comparable with that note's column.
`grad ratio` is the gradient norm of the kappa head row over the mean gradient norm of the
direction rows.

| dataset | condition | init val AUC | kappa init | kappa e10 | kappa last | r_bar e0 / e10 / e50 | grad ratio e0 / e50 |
|---|---|---|---|---|---|---|---|
| Cora | vMF matched | 0.728 | 1.69 | 1.71 | 2.48 | 0.103 / 0.103 / 0.103 | 0.0035 / 0.049 |
| | PS matched | 0.728 | 0.69 | 0.71 | 1.34 | 0.10 / 0.10 / 0.10 | 0.0012 / 0.022 |
| | vMF own (d=9) | 0.638 | 1.69 | 1.71 | 4.37 | 0.205 / 0.207 / 0.218 | 0.0039 / 0.16 |
| | PS own (d=65) | 0.728 | 0.69 | 0.69 | 0.70 | 0.10 / 0.10 / 0.10 | 0.0012 / 0.0022 |
| MAG-CS | vMF matched | 0.810 | 1.69 | 1.71 | 2.68 | 0.111 / 0.112 / 0.113 | 0.0029 / 0.19 |
| | PS matched | 0.810 | 0.69 | 0.71 | 1.51 | 0.102 / 0.102 / 0.102 | 0.0014 / 0.098 |
| | vMF own (d=9) | 0.739 | 1.69 | 1.71 | 6.56 | 0.205 / 0.207 / 0.218 | 0.0018 / 0.50 |
| Citeseer | vMF matched | 0.649 | 1.69 | 1.71 | 3.28 | 0.139 / 0.139 / 0.144 | 0.0043 / 0.12 |
| | PS matched | 0.649 | 0.69 | 0.71 | 1.79 | 0.107 / 0.107 / 0.110 | 0.0021 / 0.045 |
| Pubmed (400 ep.) | vMF matched | 0.713 | 1.69 | 1.71 | 26.8 | 0.139 / 0.139 / 0.143 | 0.0044 / 0.22 |
| | PS matched | 0.713 | 0.69 | 0.71 | 26.7 | 0.107 / 0.107 / 0.109 | 0.0020 / 0.10 |
| Cora | TNBBeta matched | 0.562 | - | - | - | 0.098 / 0.098 / 0.099 | - |
| MAG-CS | TNBBeta matched | 0.705 | - | - | - | 0.098 / 0.098 / 0.107 | - |

TNBBeta trajectory, Cora seed mean (epoch: `p`, `q`, `epsilon`, `r_bar`, validation AUC):
-1: .50, .50, 0.69, 0.098, 0.562; 50: .50, .55, 0.79, 0.099, 0.596; 100: .48, .61, 0.94,
0.139, 0.648; 150: .44, .85, 2.03, 0.679, 0.781; 195: .41, .96, 3.08, 0.905, 0.845.
MAG-CS and Pubmed look the same shape, taking off between epochs 100 and 150.

## What this supports

- Free vMF and power-spherical do not fail for lack of expressiveness: pin kappa and they
  reach TNBBeta's level (or the Gaussian's). The link-prediction advantage the earlier note
  attributed to TNBBeta over vMF/PS at matched dimension is therefore *not* evidence that
  the TNBBeta family is better for this task; it is evidence that free kappa is hard to
  learn here and that TNBBeta's parameterization gets going sooner.
- The untrained GCN already gives a validation AUC of 0.64-0.83 from centres, which is
  most of what free vMF/PS end at (0.75-0.89). So their centre AUC reflects the encoder's
  structure, and their link loss is flat at chance: they are not really learning.
- The kappa head row gets about 250-1000x less gradient than the direction rows at
  initialization (ratio 0.001-0.004), rising to 0.02-0.5 once kappa has grown. That fits a
  low signal-to-noise start, but note Adam normalizes per-parameter step size, so
  gradient *magnitude* itself does not set the speed; what matters is how consistent the
  gradient's sign is. I have not measured that.

## Not established

- **Why TNBBeta takes off sooner.** The reachability picture fits (kappa needs raw about
  `d`, `q` and `p` saturate by a logit of about 3), but nothing here tests it. Three
  tests, in decreasing cost-effectiveness: (a) free-kappa vMF/PS with `initial kappa = d`
  (GraphVAE has no such knob yet; the MLP/conv models do); if it then learns kappa and
  matches `x1`, the problem is initialization. (b) Free vMF/PS for 5-10x the epochs, to see
  whether it takes off like TNBBeta and Pubmed do. (c) A larger learning rate on the kappa
  row only.
- **Whether TNBBeta is slow too.** Its validation AUC is still rising at epoch 195, so the
  200-epoch budget may under-train it as well. Longer runs would show whether the matched
  TNBBeta numbers are a budget artifact.
- **TNBBeta's `r_bar` cannot show concentration of a bimodal posterior.** Antipodal modes
  cancel in the mean of samples, so `r_bar` near 0.1 for TNBBeta early on does not prove it
  is diffuse. The idea that TNBBeta "starts concentrated" (small `epsilon`) is therefore
  untested: `r_bar` cannot confirm or refute it, and I have not computed an axial
  concentration measure (e.g. the norm of the mean of `z z^T`).
- Fixed-kappa results are at TNBBeta's selected `(lr, dropout, d)`, not tuned per
  condition, and the multiples were scanned on test AUC (see table 1).
- The matched-dimension TNBBeta-vs-vMF gap is still 0.05-0.06 AUC on Cora and MAG-CS at
  vMF's own dimension; some of that may again be the slow-kappa effect.

## Implication for the paper draft

The link-prediction comparison against vMF and power-spherical as currently framed
(learned kappa) measures optimization of kappa as much as family quality. A fixed-kappa
(`kappa = d`) vMF should be included as a baseline, or the claim narrowed to "learned
posteriors".
