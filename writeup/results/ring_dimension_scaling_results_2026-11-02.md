# Ring posterior advantage versus ambient dimension: results (2026-11-02)

Date: 2026-11-02. Base commit `6b1880f` (branch `synthetic-graph-families`) plus the UNCOMMITTED working tree (new files only: `src/tnbbeta_vae/data/ring_dimension.py`,
`apps/synthetic/ring_dimension_scaling.py`, `apps/synthetic/ring_dimension_scaling_plot.py`, `scripts/slurm/ring_dimension_scaling.sbatch`,
`scripts/slurm/ring_dimension_scaling_aux.sbatch`, `tests/data/test_ring_dimension.py`, `tests/apps/test_ring_dimension_scaling.py`; no existing file changed).
Pre-registration (FROZEN before any run; deviations D1 and D2 appended before the full runs): `ring_dimension_scaling_preregistration_2026-11-02.md`.

## Commands

    # 720 runs = {A,B} x d{3,5,9,17} x 7 arms x 10 seeds + C x d x 4 arms x 10 seeds; 24 array tasks x 6 workers sharing a GPU
    PROCS=6 sbatch --exclude=k003 --array=0-23 scripts/slurm/ring_dimension_scaling.sbatch        # job 2296725 (k003 had a broken CUDA install)
    AUX=info sbatch scripts/slurm/ring_dimension_scaling_aux.sbatch                               # job 2296563 -> ring_recovery/dimension_scaling/information.json
    uv run python -m apps.synthetic.ring_dimension_scaling summarize --out-dir ring_recovery/dimension_scaling
    uv run python -m apps.synthetic.ring_dimension_scaling_plot --out-dir ring_recovery/dimension_scaling --fig-dir writeup/results/figures
    AUX=pytest sbatch scripts/slurm/ring_dimension_scaling_aux.sbatch                             # full pytest as a CPU job, counts below
    # one cell: uv run python -m apps.synthetic.ring_dimension_scaling train --condition B --dim 9 --arm tnbbeta --seed 0 --out-dir ring_recovery/dimension_scaling

Other jobs I submitted: smoke tests 2296561 (CUDA-broken node, failed), 2296565, 2296628; pytest jobs 2296562 (1 failure, a test bug, fixed), 2296726 (full suite on the final code: 1067 passed, 1 skipped; only a cosmetic plot-axis edit came later) and 2298787 (cancelled while queued behind the QOS limit);
a tail job 2298413 (cancelled, shard 15 finished the cells itself) and a summarize job 2298784 (cancelled). All 720 runs finished, 0 divergences, none excluded.
Raw outputs `ring_recovery/dimension_scaling/<cond>/d<d>/<arm>/s<seed>.{json,npz,pt}` (gitignored). The tables below are the verbatim `summarize` output.

## Headline (against the pre-registered rules)

**Hypothesis H (advantage grows with d) is not supported as stated: the advantage is real at d >= 5 but non-monotone in d (it peaks at d = 5 or 9 and shrinks at d = 17), and at d = 3 with a learned decoder it is negative.**

Confirmatory family A (true frozen decoder; Holm over 8 tests): **8/8 confirmed** (p = 1/1024 on every one, TNBBeta better on 10/10 seeds, margins >= 1.28 nats).
Mean paired gap difference (comparator minus TNBBeta, nats) by d = 3, 5, 9, 17: vMF 1.77, 3.27, 3.07, 1.53; PowerSpherical 1.74, 3.08, 2.55, 1.28. TNBBeta's own gap
is 0.01-0.08 (it is near the exact posterior everywhere); the caps' gap is 1.3-3.3 nats.

Confirmatory family B (learned MLP decoder; Holm over 8 tests): **5/8 confirmed.**

| d | vs vMF (mean diff, seeds better, verdict) | vs PowerSpherical |
|---|---|---|
| 3 | -0.106, 0/10, p = 1, not confirmed (TNBBeta worse; reproduces the earlier S^2 study) | -0.118, 0/10, p = 1, not confirmed |
| 5 | 0.182, 10/10, confirmed | 0.038, 10/10, Holm rejects but margin 0.05 fails: NOT confirmed |
| 9 | 0.567, 10/10, confirmed | 0.487, 10/10, confirmed |
| 17 | 0.301, 10/10, confirmed | 0.258, 10/10, confirmed |

Descriptive trend (not in the Holm family): the paired difference rises with d in B (Spearman +0.775 pooled, mean per-seed OLS slope +0.024 to +0.026 nats per unit d) but it is
not linear: -0.11, 0.18, 0.57, 0.30 (vMF). In A it falls (Spearman -0.387, slope -0.06 to -0.07) because it peaks at d = 5. The per-seed-slope bootstrap intervals are very narrow only because
the seeds nearly coincide (per-seed gaps differ by ~0.01); a 4-point non-monotone curve is not summarised by a slope, so read the by-d means (figure) rather than the slope.

## What drives the pattern (mostly descriptive, post hoc)

1. **Information in x about z falls with d** (`information.json`): I(z; x) = 1.72, 1.51, 1.22, 0.93 nats and E KL(Uniform || exact posterior) = 23.6, 14.6, 8.0, 4.0 nats at d = 3, 5, 9, 17,
   while the exact posterior's std of w stays ~0.09-0.10 and the prior std of w shrinks (0.58 to 0.24). So the shell stays equally thin but x tells less about z at high d, and the
   *absolute* gap a cap pays shrinks with it. Relative to the available KL(Uniform || posterior), the vMF-minus-TNBBeta A difference is 7.5%, 22%, 38%, 38% at d = 3, 5, 9, 17
   (post hoc ratio, descriptive): in relative terms the penalty does grow with d and then plateaus, matching the heuristic's direction but not a clean monotone absolute effect.
2. **Condition A**: caps reach a gap of 1.8 (d=3), 3.1-3.3 (d=5), 2.6-3.1 (d=9), 1.3-1.5 (d=17); TNBBeta 0.01-0.08 at every d. The IWAE (K=64) gap of caps falls to ~0.01 at d >= 9 (weak information makes importance
   sampling easy), so at high d the ELBO deficit of caps is almost entirely bound looseness, not a worse model (same decoder in all arms by construction; the ELBO gap is KL(q||p), cross-checked against the exact Monte Carlo KL to < 0.01 nats).
3. **Condition B**: the library-default TNBBeta ends as a cap at d = 3 (median q at its floor 1e-6, p 0.98, ring statistic R = 0.95) and loses by 0.1 nat; at d = 5 it is intermediate (R = 0.55, q 2e-4); at d >= 9
   it is a genuine ring-like posterior (R = 0.06, q ~ 0.47, m = epsilon - (d-1)/2 about +17) paired with a single-index learned decoder (eigenvalue fraction 1.0) and learned sigma about 0.20 (true 0.2),
   whereas vMF/PS learn sigma 0.25-0.29 with a broad near-uniform posterior (kappa about 2.6-3.9). The free decoder does NOT fold the latent to rescue the caps at d >= 9: their gaps stay 0.28-0.59.
4. **R is not a ring detector for caps at high d**: at d >= 9 vMF/PS have R ~ 0.06-0.18 because kappa collapses to 2-4 (almost uniform), not because they form a ring; use R only with kappa/q.
5. **Controls**: untuned vMF / PS (library initial kappa) differ from the tuned ones by < 0.013 nats at every (condition, d) (seed-pair means), so kappa initialisation is not the story. `tnbbeta_epsinit` ~ `tnbbeta`
   (paired mean |diff| < 0.05 at every (condition, d)). `tnbbeta_init0` finds the ring at d = 3 on 7/10 B seeds (gap 0.18 vs cap 0.41) and is equal to `tnbbeta` at d >= 9 (descriptive; as in the earlier study).
6. **Condition C** (forced single-index decoder, descriptive): TNBBeta beats both caps at every d on 10/10 seeds; the difference shrinks monotonically 1.04, 0.85, 0.61, 0.31 (vMF) with d, the information effect.
7. **Surprises / checks**: (a) an IWAE gap below zero for vMF at d = 17 in a smoke test exposed a bias in the library vMF rejection sampler (kappa 10-40, m >= 9; see D2): the study uses an exact-b subclass; this bias would
   have inflated vMF evaluation noise and favoured neither arm predictably. (b) 21 of the 1440 per-run gap values (ELBO or IWAE) are slightly negative, all within -0.0012 nats and all TNBBeta variants at d >= 9 where the gap is ~0 (Monte Carlo noise of the 256-draw ELBO and 64-sample IWAE estimates and 16001-node quadrature; checked, no bound violation beyond noise); (c) vMF is worse than PS in every cell (0/10 seeds) on the ELBO.

## What this does and does not show

* Shows, for this task (one h, sigma 0.2, single axis, x in R^3), optimiser, 300 epochs, 10 seeds: with the true decoder TNBBeta's ring-capable posterior wins by >1.2 nats at every d in {3,5,9,17}; with a learned
  free decoder it wins from d = 5 up (confirmed for vMF at 5, 9, 17 and for PS at 9, 17), loses at d = 3, and the advantage is largest at d = 9, not at d = 17.
* Does not show the predicted monotone growth of the *absolute* advantage with d; the absolute advantage is bounded by how informative x is, which falls with d here (x low-dimensional, fixed noise). A design in which the
  information about the latitude stays constant as d grows is not covered and would be the clean test of the heuristic.
* Seeds are nearly identical in outcome (sd of gaps ~0.001-0.02), so the 10/10 results are strong for this task and training recipe, not evidence about other tasks. Latitude information, optimiser dynamics
  (the d = 3 B cap solution) and the initialisation of TNBBeta (library default) all matter.
* The vMF numbers use the exact-sampler subclass; library-sampler vMF at d >= 9 with kappa near 10-40 is biased (see D2), which also bears on earlier high-d vMF results in this repo (not re-examined here).

## Deviations

* D1 (before runs): quadrature grid 16001 nodes (see the pre-registration).
* D2 (before runs): the vMF posterior uses `ExactVonMisesFisher` (exact envelope parameter b); library `VonMisesFisher` unchanged (see the pre-registration).
* Confirmatory PowerSpherical arm uses initial kappa = d (pre-registration design note); untuned PS is the descriptive control.
* Operationalisations as pre-registered: m = epsilon - (d-1)/2 with the median over test x; ring statistic about `e_1` (A), learned axis (C) or the active-subspace axis (B); the OLS-slope bootstrap resamples seeds.
* Cluster events: node k003 has no working CUDA (one smoke job failed there; the array excluded it). Shard 15 (four slow `A d=5 vmf`/`vmf_untuned` cells at 45-140 min each) finished last after ~5.5 h; the cells ran to completion without resubmission.
* The "Spearman p-values" in the trend table pool 40 seed-by-d points that are not independent; they are descriptive only.

## Figures (`writeup/results/figures/`)

* `ring_dimension_gap_vs_d_2026-11-02.png` (gap versus d per family, A and B, per-seed points)
* `ring_dimension_paired_difference_vs_d_2026-11-02.png` (vMF/PS minus TNBBeta versus d)

## Full tables (verbatim `summarize` output)


## Information in x about z (descriptive, per d)

| d | I(z; x) (nats) | E KL(Uniform to exact posterior) | posterior std of w | prior std of w |
|---|---|---|---|---|
| 3 | 1.716 | 23.564 | 0.091 | 0.577 |
| 5 | 1.511 | 14.594 | 0.095 | 0.447 |
| 9 | 1.222 | 7.969 | 0.097 | 0.333 |
| 17 | 0.926 | 3.971 | 0.097 | 0.243 |

## Per-arm test ELBO gap (nats; lower is better)

| cond | d | arm | seeds (valid) | gap mean +- sd | median | min / max | IWAE gap mean +- sd |
|---|---|---|---|---|---|---|---|
| A | 3 | vmf | 10/10 | 1.8523 +- 0.0023 | 1.8522 | 1.8502 / 1.8579 | 1.1847 +- 0.0081 |
| A | 3 | power_spherical | 10/10 | 1.8256 +- 0.0038 | 1.8254 | 1.8216 / 1.8350 | 1.1568 +- 0.0115 |
| A | 3 | tnbbeta | 10/10 | 0.0833 +- 0.0406 | 0.0659 | 0.0520 / 0.1694 | 0.0085 +- 0.0034 |
| A | 3 | tnbbeta_init0 | 10/10 | 0.0453 +- 0.0129 | 0.0401 | 0.0367 / 0.0737 | 0.0074 +- 0.0019 |
| A | 3 | tnbbeta_epsinit | 10/10 | 0.0705 +- 0.0161 | 0.0727 | 0.0488 / 0.0919 | 0.0080 +- 0.0026 |
| A | 3 | vmf_untuned | 10/10 | 1.8540 +- 0.0036 | 1.8528 | 1.8509 / 1.8620 | 1.1913 +- 0.0073 |
| A | 3 | power_spherical_untuned | 10/10 | 1.8264 +- 0.0054 | 1.8247 | 1.8217 / 1.8410 | 1.1533 +- 0.0180 |
| A | 5 | vmf | 10/10 | 3.3030 +- 0.0086 | 3.3040 | 3.2852 / 3.3121 | 1.0557 +- 0.0212 |
| A | 5 | power_spherical | 10/10 | 3.1131 +- 0.0048 | 3.1141 | 3.1049 / 3.1177 | 1.1262 +- 0.0429 |
| A | 5 | tnbbeta | 10/10 | 0.0291 +- 0.0066 | 0.0290 | 0.0186 / 0.0416 | 0.0017 +- 0.0009 |
| A | 5 | tnbbeta_init0 | 10/10 | 0.0425 +- 0.0305 | 0.0319 | 0.0144 / 0.1106 | 0.0017 +- 0.0021 |
| A | 5 | tnbbeta_epsinit | 10/10 | 0.0730 +- 0.0528 | 0.0557 | 0.0220 / 0.1999 | 0.0072 +- 0.0060 |
| A | 5 | vmf_untuned | 10/10 | 3.3004 +- 0.0080 | 3.2995 | 3.2887 / 3.3158 | 1.0496 +- 0.0165 |
| A | 5 | power_spherical_untuned | 10/10 | 3.1111 +- 0.0053 | 3.1097 | 3.1054 / 3.1210 | 1.1458 +- 0.0261 |
| A | 9 | vmf | 10/10 | 3.1055 +- 0.0203 | 3.1060 | 3.0775 / 3.1421 | 0.0195 +- 0.0091 |
| A | 9 | power_spherical | 10/10 | 2.5937 +- 0.0042 | 2.5940 | 2.5840 / 2.5983 | 0.0099 +- 0.0017 |
| A | 9 | tnbbeta | 10/10 | 0.0390 +- 0.0450 | 0.0267 | 0.0140 / 0.1636 | 0.0027 +- 0.0067 |
| A | 9 | tnbbeta_init0 | 10/10 | 0.0280 +- 0.0230 | 0.0189 | 0.0110 / 0.0854 | 0.0013 +- 0.0026 |
| A | 9 | tnbbeta_epsinit | 10/10 | 0.0250 +- 0.0130 | 0.0240 | 0.0086 / 0.0445 | 0.0010 +- 0.0017 |
| A | 9 | vmf_untuned | 10/10 | 3.0932 +- 0.0098 | 3.0933 | 3.0776 / 3.1062 | 0.0309 +- 0.0118 |
| A | 9 | power_spherical_untuned | 10/10 | 2.5956 +- 0.0094 | 2.5931 | 2.5900 / 2.6211 | 0.0109 +- 0.0017 |
| A | 17 | vmf | 10/10 | 1.5415 +- 0.0058 | 1.5410 | 1.5352 / 1.5530 | 0.0058 +- 0.0009 |
| A | 17 | power_spherical | 10/10 | 1.2965 +- 0.0033 | 1.2965 | 1.2902 / 1.3007 | 0.0084 +- 0.0009 |
| A | 17 | tnbbeta | 10/10 | 0.0147 +- 0.0081 | 0.0130 | 0.0066 / 0.0305 | 0.0004 +- 0.0006 |
| A | 17 | tnbbeta_init0 | 10/10 | 0.0151 +- 0.0083 | 0.0117 | 0.0057 / 0.0339 | 0.0003 +- 0.0006 |
| A | 17 | tnbbeta_epsinit | 10/10 | 0.0124 +- 0.0080 | 0.0092 | 0.0048 / 0.0278 | 0.0002 +- 0.0005 |
| A | 17 | vmf_untuned | 10/10 | 1.5375 +- 0.0037 | 1.5368 | 1.5326 / 1.5434 | 0.0050 +- 0.0015 |
| A | 17 | power_spherical_untuned | 10/10 | 1.2986 +- 0.0034 | 1.2986 | 1.2925 / 1.3031 | 0.0085 +- 0.0010 |
| B | 3 | vmf | 10/10 | 0.4141 +- 0.0066 | 0.4127 | 0.4053 / 0.4252 | 0.1529 +- 0.0146 |
| B | 3 | power_spherical | 10/10 | 0.4022 +- 0.0062 | 0.4018 | 0.3938 / 0.4105 | 0.1329 +- 0.0135 |
| B | 3 | tnbbeta | 10/10 | 0.5198 +- 0.0110 | 0.5179 | 0.5030 / 0.5379 | 0.1826 +- 0.0439 |
| B | 3 | tnbbeta_init0 | 10/10 | 0.1844 +- 0.2300 | 0.0440 | 0.0378 / 0.5355 | 0.0671 +- 0.0852 |
| B | 3 | tnbbeta_epsinit | 10/10 | 0.5235 +- 0.0113 | 0.5227 | 0.5089 / 0.5432 | 0.2152 +- 0.0151 |
| B | 3 | vmf_untuned | 10/10 | 0.4152 +- 0.0048 | 0.4154 | 0.4074 / 0.4242 | 0.1495 +- 0.0149 |
| B | 3 | power_spherical_untuned | 10/10 | 0.4057 +- 0.0101 | 0.4053 | 0.3927 / 0.4260 | 0.1205 +- 0.0174 |
| B | 5 | vmf | 10/10 | 0.7426 +- 0.0063 | 0.7435 | 0.7344 / 0.7497 | 0.4459 +- 0.0091 |
| B | 5 | power_spherical | 10/10 | 0.5990 +- 0.0020 | 0.5986 | 0.5957 / 0.6027 | 0.1860 +- 0.0104 |
| B | 5 | tnbbeta | 10/10 | 0.5608 +- 0.0042 | 0.5602 | 0.5541 / 0.5673 | 0.1489 +- 0.0080 |
| B | 5 | tnbbeta_init0 | 10/10 | 0.4677 +- 0.2000 | 0.5631 | 0.0707 / 0.5647 | 0.1257 +- 0.0601 |
| B | 5 | tnbbeta_epsinit | 10/10 | 0.5605 +- 0.0040 | 0.5600 | 0.5549 / 0.5667 | 0.1499 +- 0.0080 |
| B | 5 | vmf_untuned | 10/10 | 0.7421 +- 0.0059 | 0.7404 | 0.7342 / 0.7503 | 0.4487 +- 0.0106 |
| B | 5 | power_spherical_untuned | 10/10 | 0.5979 +- 0.0018 | 0.5979 | 0.5950 / 0.6017 | 0.1865 +- 0.0080 |
| B | 9 | vmf | 10/10 | 0.5912 +- 0.0029 | 0.5905 | 0.5868 / 0.5976 | 0.3919 +- 0.0068 |
| B | 9 | power_spherical | 10/10 | 0.5117 +- 0.0024 | 0.5125 | 0.5079 / 0.5142 | 0.2842 +- 0.0080 |
| B | 9 | tnbbeta | 10/10 | 0.0244 +- 0.0092 | 0.0217 | 0.0156 / 0.0438 | 0.0052 +- 0.0026 |
| B | 9 | tnbbeta_init0 | 10/10 | 0.0314 +- 0.0172 | 0.0276 | 0.0084 / 0.0585 | 0.0067 +- 0.0036 |
| B | 9 | tnbbeta_epsinit | 10/10 | 0.0362 +- 0.0379 | 0.0228 | 0.0096 / 0.1372 | 0.0064 +- 0.0071 |
| B | 9 | vmf_untuned | 10/10 | 0.5918 +- 0.0023 | 0.5920 | 0.5881 / 0.5966 | 0.3898 +- 0.0089 |
| B | 9 | power_spherical_untuned | 10/10 | 0.5120 +- 0.0022 | 0.5115 | 0.5091 / 0.5153 | 0.2838 +- 0.0048 |
| B | 17 | vmf | 10/10 | 0.3240 +- 0.0037 | 0.3247 | 0.3181 / 0.3309 | 0.2134 +- 0.0079 |
| B | 17 | power_spherical | 10/10 | 0.2815 +- 0.0037 | 0.2819 | 0.2763 / 0.2878 | 0.1637 +- 0.0038 |
| B | 17 | tnbbeta | 10/10 | 0.0235 +- 0.0071 | 0.0248 | 0.0107 / 0.0328 | 0.0052 +- 0.0041 |
| B | 17 | tnbbeta_init0 | 10/10 | 0.0204 +- 0.0075 | 0.0184 | 0.0118 / 0.0324 | 0.0038 +- 0.0020 |
| B | 17 | tnbbeta_epsinit | 10/10 | 0.0195 +- 0.0106 | 0.0188 | 0.0076 / 0.0419 | 0.0037 +- 0.0031 |
| B | 17 | vmf_untuned | 10/10 | 0.3242 +- 0.0022 | 0.3235 | 0.3220 / 0.3278 | 0.2141 +- 0.0040 |
| B | 17 | power_spherical_untuned | 10/10 | 0.2812 +- 0.0032 | 0.2818 | 0.2771 / 0.2869 | 0.1633 +- 0.0043 |
| C | 3 | vmf | 10/10 | 1.1049 +- 0.0033 | 1.1044 | 1.1009 / 1.1113 | 0.7222 +- 0.0219 |
| C | 3 | power_spherical | 10/10 | 0.9842 +- 0.0031 | 0.9843 | 0.9798 / 0.9889 | 0.6646 +- 0.0167 |
| C | 3 | tnbbeta | 10/10 | 0.0628 +- 0.0291 | 0.0514 | 0.0376 / 0.1339 | 0.0193 +- 0.0070 |
| C | 3 | tnbbeta_init0 | 10/10 | 0.0733 +- 0.0699 | 0.0458 | 0.0347 / 0.2638 | 0.0263 +- 0.0321 |
| C | 5 | vmf | 10/10 | 0.8833 +- 0.0042 | 0.8827 | 0.8783 / 0.8917 | 0.6714 +- 0.0120 |
| C | 5 | power_spherical | 10/10 | 0.7597 +- 0.0044 | 0.7575 | 0.7550 / 0.7680 | 0.5229 +- 0.0128 |
| C | 5 | tnbbeta | 10/10 | 0.0352 +- 0.0177 | 0.0348 | 0.0129 / 0.0606 | 0.0071 +- 0.0033 |
| C | 5 | tnbbeta_init0 | 10/10 | 0.0673 +- 0.0627 | 0.0466 | 0.0111 / 0.1835 | 0.0185 +- 0.0220 |
| C | 9 | vmf | 10/10 | 0.6420 +- 0.0037 | 0.6421 | 0.6349 / 0.6483 | 0.4806 +- 0.0063 |
| C | 9 | power_spherical | 10/10 | 0.5597 +- 0.0023 | 0.5593 | 0.5572 / 0.5644 | 0.3957 +- 0.0086 |
| C | 9 | tnbbeta | 10/10 | 0.0299 +- 0.0247 | 0.0165 | 0.0110 / 0.0808 | 0.0057 +- 0.0044 |
| C | 9 | tnbbeta_init0 | 10/10 | 0.0271 +- 0.0191 | 0.0190 | 0.0055 / 0.0742 | 0.0051 +- 0.0032 |
| C | 17 | vmf | 10/10 | 0.3282 +- 0.0032 | 0.3286 | 0.3220 / 0.3335 | 0.2276 +- 0.0065 |
| C | 17 | power_spherical | 10/10 | 0.2829 +- 0.0030 | 0.2834 | 0.2783 / 0.2873 | 0.1859 +- 0.0052 |
| C | 17 | tnbbeta | 10/10 | 0.0152 +- 0.0075 | 0.0135 | 0.0034 / 0.0294 | 0.0025 +- 0.0023 |
| C | 17 | tnbbeta_init0 | 10/10 | 0.0108 +- 0.0056 | 0.0106 | 0.0023 / 0.0188 | 0.0024 +- 0.0025 |

## Per-seed ELBO gap, condition A


A / d=3

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 1.8507 | 1.8254 | 0.1295 | 0.0367 | 0.0919 | 1.8620 | 1.8240 |
| 1 | 1.8502 | 1.8224 | 0.0531 | 0.0380 | 0.0918 | 1.8509 | 1.8217 |
| 2 | 1.8506 | 1.8350 | 0.0520 | 0.0414 | 0.0539 | 1.8516 | 1.8410 |
| 3 | 1.8527 | 1.8254 | 0.0670 | 0.0407 | 0.0640 | 1.8530 | 1.8282 |
| 4 | 1.8521 | 1.8216 | 0.0648 | 0.0399 | 0.0488 | 1.8518 | 1.8248 |
| 5 | 1.8528 | 1.8268 | 0.0550 | 0.0649 | 0.0497 | 1.8535 | 1.8243 |
| 6 | 1.8533 | 1.8227 | 0.1166 | 0.0386 | 0.0812 | 1.8509 | 1.8253 |
| 7 | 1.8522 | 1.8254 | 0.0566 | 0.0737 | 0.0732 | 1.8526 | 1.8239 |
| 8 | 1.8502 | 1.8245 | 0.0691 | 0.0404 | 0.0722 | 1.8554 | 1.8245 |
| 9 | 1.8579 | 1.8271 | 0.1694 | 0.0382 | 0.0786 | 1.8578 | 1.8266 |

A / d=5

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 3.3104 | 3.1174 | 0.0289 | 0.1106 | 0.0435 | 3.2994 | 3.1210 |
| 1 | 3.2852 | 3.1157 | 0.0307 | 0.0326 | 0.1999 | 3.2974 | 3.1111 |
| 2 | 3.3016 | 3.1121 | 0.0237 | 0.0491 | 0.0436 | 3.3068 | 3.1202 |
| 3 | 3.3052 | 3.1049 | 0.0231 | 0.0144 | 0.0220 | 3.2887 | 3.1083 |
| 4 | 3.3109 | 3.1177 | 0.0186 | 0.0181 | 0.0274 | 3.2905 | 3.1082 |
| 5 | 3.2930 | 3.1117 | 0.0416 | 0.0206 | 0.1089 | 3.3051 | 3.1111 |
| 6 | 3.2999 | 3.1145 | 0.0292 | 0.0312 | 0.0995 | 3.2969 | 3.1111 |
| 7 | 3.3092 | 3.1050 | 0.0266 | 0.0800 | 0.0573 | 3.3043 | 3.1070 |
| 8 | 3.3028 | 3.1137 | 0.0324 | 0.0302 | 0.0540 | 3.2996 | 3.1054 |
| 9 | 3.3121 | 3.1177 | 0.0359 | 0.0381 | 0.0735 | 3.3158 | 3.1078 |

A / d=9

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 3.0900 | 2.5928 | 0.0188 | 0.0128 | 0.0445 | 3.0857 | 2.5900 |
| 1 | 3.0775 | 2.5968 | 0.0314 | 0.0854 | 0.0112 | 3.1004 | 2.5988 |
| 2 | 3.0824 | 2.5920 | 0.0481 | 0.0163 | 0.0412 | 3.0776 | 2.5931 |
| 3 | 3.1194 | 2.5921 | 0.0273 | 0.0213 | 0.0211 | 3.1062 | 2.5901 |
| 4 | 3.1050 | 2.5983 | 0.0261 | 0.0158 | 0.0086 | 3.0868 | 2.5950 |
| 5 | 3.1070 | 2.5915 | 0.0157 | 0.0258 | 0.0346 | 3.1000 | 2.5939 |
| 6 | 3.1421 | 2.5952 | 0.0300 | 0.0498 | 0.0269 | 3.0877 | 2.6211 |
| 7 | 3.1077 | 2.5973 | 0.0140 | 0.0165 | 0.0336 | 3.1044 | 2.5906 |
| 8 | 3.1280 | 2.5969 | 0.1636 | 0.0253 | 0.0143 | 3.0846 | 2.5931 |
| 9 | 3.0956 | 2.5840 | 0.0148 | 0.0110 | 0.0139 | 3.0989 | 2.5902 |

A / d=17

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 1.5352 | 1.3001 | 0.0305 | 0.0203 | 0.0072 | 1.5405 | 1.2989 |
| 1 | 1.5363 | 1.2955 | 0.0153 | 0.0109 | 0.0048 | 1.5425 | 1.2962 |
| 2 | 1.5388 | 1.2902 | 0.0070 | 0.0095 | 0.0239 | 1.5384 | 1.2925 |
| 3 | 1.5437 | 1.2969 | 0.0067 | 0.0218 | 0.0131 | 1.5348 | 1.2984 |
| 4 | 1.5530 | 1.3000 | 0.0106 | 0.0163 | 0.0278 | 1.5347 | 1.2963 |
| 5 | 1.5433 | 1.2934 | 0.0209 | 0.0339 | 0.0062 | 1.5373 | 1.3020 |
| 6 | 1.5370 | 1.2949 | 0.0193 | 0.0125 | 0.0166 | 1.5342 | 1.3020 |
| 7 | 1.5478 | 1.2971 | 0.0210 | 0.0106 | 0.0056 | 1.5434 | 1.3005 |
| 8 | 1.5366 | 1.3007 | 0.0088 | 0.0057 | 0.0104 | 1.5326 | 1.2961 |
| 9 | 1.5433 | 1.2961 | 0.0066 | 0.0096 | 0.0080 | 1.5364 | 1.3031 |

## Per-seed ELBO gap, condition B


B / d=3

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.4233 | 0.4097 | 0.5379 | 0.0452 | 0.5432 | 0.4190 | 0.4064 |
| 1 | 0.4071 | 0.3997 | 0.5340 | 0.5036 | 0.5089 | 0.4138 | 0.3971 |
| 2 | 0.4136 | 0.4024 | 0.5255 | 0.0451 | 0.5206 | 0.4101 | 0.4066 |
| 3 | 0.4252 | 0.4105 | 0.5211 | 0.0385 | 0.5248 | 0.4242 | 0.4260 |
| 4 | 0.4119 | 0.3973 | 0.5030 | 0.5135 | 0.5338 | 0.4185 | 0.4032 |
| 5 | 0.4053 | 0.4012 | 0.5255 | 0.0408 | 0.5187 | 0.4074 | 0.4159 |
| 6 | 0.4175 | 0.4101 | 0.5141 | 0.0428 | 0.5338 | 0.4143 | 0.4043 |
| 7 | 0.4109 | 0.3938 | 0.5112 | 0.0378 | 0.5135 | 0.4164 | 0.4101 |
| 8 | 0.4098 | 0.3944 | 0.5148 | 0.5355 | 0.5099 | 0.4165 | 0.3927 |
| 9 | 0.4169 | 0.4032 | 0.5107 | 0.0408 | 0.5279 | 0.4121 | 0.3944 |

B / d=5

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.7473 | 0.5972 | 0.5626 | 0.5579 | 0.5640 | 0.7396 | 0.5950 |
| 1 | 0.7497 | 0.6006 | 0.5596 | 0.1066 | 0.5637 | 0.7412 | 0.5968 |
| 2 | 0.7448 | 0.6005 | 0.5669 | 0.5647 | 0.5645 | 0.7503 | 0.6017 |
| 3 | 0.7372 | 0.5983 | 0.5541 | 0.5577 | 0.5565 | 0.7466 | 0.5970 |
| 4 | 0.7491 | 0.5988 | 0.5609 | 0.5630 | 0.5583 | 0.7395 | 0.5984 |
| 5 | 0.7354 | 0.5957 | 0.5568 | 0.0707 | 0.5568 | 0.7342 | 0.5976 |
| 6 | 0.7365 | 0.5979 | 0.5673 | 0.5644 | 0.5667 | 0.7378 | 0.5981 |
| 7 | 0.7344 | 0.6001 | 0.5593 | 0.5633 | 0.5606 | 0.7469 | 0.5989 |
| 8 | 0.7422 | 0.5980 | 0.5581 | 0.5636 | 0.5549 | 0.7353 | 0.5969 |
| 9 | 0.7494 | 0.6027 | 0.5628 | 0.5647 | 0.5593 | 0.7501 | 0.5987 |

B / d=9

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.5898 | 0.5133 | 0.0237 | 0.0254 | 0.0349 | 0.5909 | 0.5135 |
| 1 | 0.5901 | 0.5133 | 0.0158 | 0.0370 | 0.0472 | 0.5920 | 0.5112 |
| 2 | 0.5906 | 0.5081 | 0.0308 | 0.0538 | 0.0435 | 0.5907 | 0.5100 |
| 3 | 0.5868 | 0.5117 | 0.0438 | 0.0585 | 0.0211 | 0.5881 | 0.5119 |
| 4 | 0.5923 | 0.5079 | 0.0337 | 0.0203 | 0.0233 | 0.5894 | 0.5108 |
| 5 | 0.5904 | 0.5098 | 0.0195 | 0.0297 | 0.0223 | 0.5966 | 0.5095 |
| 6 | 0.5893 | 0.5142 | 0.0196 | 0.0197 | 0.1372 | 0.5920 | 0.5138 |
| 7 | 0.5926 | 0.5142 | 0.0165 | 0.0084 | 0.0120 | 0.5923 | 0.5145 |
| 8 | 0.5928 | 0.5135 | 0.0156 | 0.0475 | 0.0108 | 0.5932 | 0.5153 |
| 9 | 0.5976 | 0.5109 | 0.0249 | 0.0137 | 0.0096 | 0.5932 | 0.5091 |

B / d=17

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.3259 | 0.2844 | 0.0250 | 0.0155 | 0.0419 | 0.3278 | 0.2839 |
| 1 | 0.3272 | 0.2843 | 0.0273 | 0.0160 | 0.0227 | 0.3275 | 0.2817 |
| 2 | 0.3309 | 0.2817 | 0.0263 | 0.0320 | 0.0253 | 0.3245 | 0.2820 |
| 3 | 0.3249 | 0.2878 | 0.0246 | 0.0244 | 0.0149 | 0.3260 | 0.2869 |
| 4 | 0.3200 | 0.2832 | 0.0328 | 0.0118 | 0.0116 | 0.3220 | 0.2833 |
| 5 | 0.3246 | 0.2798 | 0.0236 | 0.0207 | 0.0081 | 0.3232 | 0.2790 |
| 6 | 0.3248 | 0.2820 | 0.0244 | 0.0135 | 0.0276 | 0.3223 | 0.2824 |
| 7 | 0.3181 | 0.2763 | 0.0293 | 0.0324 | 0.0130 | 0.3224 | 0.2771 |
| 8 | 0.3207 | 0.2767 | 0.0107 | 0.0236 | 0.0227 | 0.3239 | 0.2771 |
| 9 | 0.3232 | 0.2786 | 0.0113 | 0.0145 | 0.0076 | 0.3227 | 0.2782 |

## Per-seed ELBO gap, condition C


C / d=3

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 1.1014 | 0.9798 | 0.0539 | 0.0885 |
| 1 | 1.1046 | 0.9889 | 0.0856 | 0.0367 |
| 2 | 1.1031 | 0.9848 | 0.0479 | 0.2638 |
| 3 | 1.1043 | 0.9860 | 0.0489 | 0.0442 |
| 4 | 1.1027 | 0.9813 | 0.0420 | 0.0347 |
| 5 | 1.1061 | 0.9812 | 0.0447 | 0.0491 |
| 6 | 1.1054 | 0.9817 | 0.0747 | 0.0387 |
| 7 | 1.1009 | 0.9874 | 0.1339 | 0.0886 |
| 8 | 1.1092 | 0.9869 | 0.0376 | 0.0411 |
| 9 | 1.1113 | 0.9838 | 0.0590 | 0.0474 |

C / d=5

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.8783 | 0.7550 | 0.0134 | 0.0441 |
| 1 | 0.8851 | 0.7641 | 0.0505 | 0.0690 |
| 2 | 0.8841 | 0.7597 | 0.0301 | 0.0305 |
| 3 | 0.8882 | 0.7570 | 0.0319 | 0.0492 |
| 4 | 0.8917 | 0.7571 | 0.0377 | 0.0238 |
| 5 | 0.8813 | 0.7570 | 0.0606 | 0.1813 |
| 6 | 0.8789 | 0.7563 | 0.0458 | 0.0111 |
| 7 | 0.8837 | 0.7680 | 0.0547 | 0.1835 |
| 8 | 0.8806 | 0.7647 | 0.0141 | 0.0493 |
| 9 | 0.8816 | 0.7579 | 0.0129 | 0.0313 |

C / d=9

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.6420 | 0.5589 | 0.0147 | 0.0742 |
| 1 | 0.6483 | 0.5581 | 0.0178 | 0.0172 |
| 2 | 0.6404 | 0.5572 | 0.0110 | 0.0186 |
| 3 | 0.6430 | 0.5605 | 0.0219 | 0.0192 |
| 4 | 0.6393 | 0.5578 | 0.0658 | 0.0329 |
| 5 | 0.6429 | 0.5623 | 0.0419 | 0.0372 |
| 6 | 0.6349 | 0.5596 | 0.0148 | 0.0055 |
| 7 | 0.6423 | 0.5602 | 0.0153 | 0.0154 |
| 8 | 0.6409 | 0.5644 | 0.0150 | 0.0187 |
| 9 | 0.6460 | 0.5575 | 0.0808 | 0.0317 |

C / d=17

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.3265 | 0.2841 | 0.0175 | 0.0083 |
| 1 | 0.3335 | 0.2848 | 0.0074 | 0.0057 |
| 2 | 0.3266 | 0.2798 | 0.0130 | 0.0188 |
| 3 | 0.3298 | 0.2858 | 0.0135 | 0.0097 |
| 4 | 0.3286 | 0.2873 | 0.0034 | 0.0023 |
| 5 | 0.3286 | 0.2824 | 0.0119 | 0.0170 |
| 6 | 0.3309 | 0.2847 | 0.0134 | 0.0152 |
| 7 | 0.3220 | 0.2790 | 0.0217 | 0.0114 |
| 8 | 0.3253 | 0.2826 | 0.0206 | 0.0148 |
| 9 | 0.3299 | 0.2783 | 0.0294 | 0.0044 |

## Confirmatory tests, condition A (Holm 0.05, 8 tests)

| d | comparator | n | mean paired diff (comparator - tnbbeta) | seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.05 | confirmed |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | power_spherical | 10 | 1.7423 | 10/10 | 0.00098 | 0.00098 | 0.00714 | True | True | True |
| 3 | vmf | 10 | 1.7690 | 10/10 | 0.00098 | 0.00098 | 0.00625 | True | True | True |
| 5 | power_spherical | 10 | 3.0840 | 10/10 | 0.00098 | 0.00098 | 0.01000 | True | True | True |
| 5 | vmf | 10 | 3.2740 | 10/10 | 0.00098 | 0.00098 | 0.00833 | True | True | True |
| 9 | power_spherical | 10 | 2.5547 | 10/10 | 0.00098 | 0.00098 | 0.01667 | True | True | True |
| 9 | vmf | 10 | 3.0665 | 10/10 | 0.00098 | 0.00098 | 0.01250 | True | True | True |
| 17 | power_spherical | 10 | 1.2818 | 10/10 | 0.00098 | 0.00098 | 0.05000 | True | True | True |
| 17 | vmf | 10 | 1.5268 | 10/10 | 0.00098 | 0.00098 | 0.02500 | True | True | True |

Confirmed advantages: 8 of 8.

## Confirmatory tests, condition B (Holm 0.05, 8 tests)

| d | comparator | n | mean paired diff (comparator - tnbbeta) | seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.05 | confirmed |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | power_spherical | 10 | -0.1175 | 0/10 | 1.00000 | 1.00000 | 0.05000 | False | False | False |
| 3 | vmf | 10 | -0.1056 | 0/10 | 1.00000 | 1.00000 | 0.02500 | False | False | False |
| 5 | power_spherical | 10 | 0.0381 | 10/10 | 0.00098 | 0.00098 | 0.00714 | True | False | False |
| 5 | vmf | 10 | 0.1817 | 10/10 | 0.00098 | 0.00098 | 0.00625 | True | True | True |
| 9 | power_spherical | 10 | 0.4873 | 10/10 | 0.00098 | 0.00098 | 0.01000 | True | True | True |
| 9 | vmf | 10 | 0.5668 | 10/10 | 0.00098 | 0.00098 | 0.00833 | True | True | True |
| 17 | power_spherical | 10 | 0.2580 | 10/10 | 0.00098 | 0.00098 | 0.01667 | True | True | True |
| 17 | vmf | 10 | 0.3005 | 10/10 | 0.00098 | 0.00098 | 0.01250 | True | True | True |

Confirmed advantages: 5 of 8.

## Descriptive trend of the paired difference with d

| cond | comparator | metric | mean diff by d (3, 5, 9, 17) | per-seed OLS slope (per unit d): mean [95% seed-bootstrap] | Spearman(d, diff), pooled | descriptive |
|---|---|---|---|---|---|---|
| A | vmf | gap_elbo | 1.769, 3.274, 3.067, 1.527 | -0.0581 [-0.0589, -0.0568] | -0.387 (p 0.0135) | yes |
| A | vmf | gap_iwae | 1.176, 1.054, 0.017, 0.005 | -0.0879 [-0.0882, -0.0876] | -0.955 (p 1.18e-21) | yes |
| A | power_spherical | gap_elbo | 1.742, 3.084, 2.555, 1.282 | -0.0713 [-0.0723, -0.0700] | -0.387 (p 0.0135) | yes |
| A | power_spherical | gap_iwae | 1.148, 1.125, 0.007, 0.008 | -0.0885 [-0.0894, -0.0876] | -0.847 (p 5.94e-12) | yes |
| B | vmf | gap_elbo | -0.106, 0.182, 0.567, 0.301 | 0.0242 [0.0237, 0.0247] | 0.775 (p 4.41e-09) | yes |
| B | vmf | gap_iwae | -0.030, 0.297, 0.387, 0.208 | 0.0095 [0.0080, 0.0109] | 0.387 (p 0.0135) | yes |
| B | power_spherical | gap_elbo | -0.118, 0.038, 0.487, 0.258 | 0.0256 [0.0252, 0.0260] | 0.775 (p 4.41e-09) | yes |
| B | power_spherical | gap_iwae | -0.050, 0.037, 0.279, 0.158 | 0.0142 [0.0128, 0.0156] | 0.773 (p 5.09e-09) | yes |
| C | vmf | gap_elbo | 1.042, 0.848, 0.612, 0.313 | -0.0499 [-0.0510, -0.0487] | -0.969 (p 1.47e-24) | yes |
| C | vmf | gap_iwae | 0.703, 0.664, 0.475, 0.225 | -0.0351 [-0.0358, -0.0345] | -0.961 (p 9e-23) | yes |
| C | power_spherical | gap_elbo | 0.921, 0.725, 0.530, 0.268 | -0.0440 [-0.0451, -0.0430] | -0.969 (p 1.47e-24) | yes |
| C | power_spherical | gap_iwae | 0.645, 0.516, 0.390, 0.183 | -0.0313 [-0.0318, -0.0308] | -0.969 (p 1.47e-24) | yes |

## Per-seed IWAE (K=64) gap, condition A


A / d=3

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 1.1929 | 1.1489 | 0.0120 | 0.0069 | 0.0095 | 1.1989 | 1.1624 |
| 1 | 1.1788 | 1.1739 | 0.0068 | 0.0106 | 0.0130 | 1.1772 | 1.1707 |
| 2 | 1.1756 | 1.1559 | 0.0069 | 0.0082 | 0.0111 | 1.1955 | 1.1503 |
| 3 | 1.1804 | 1.1494 | 0.0074 | 0.0069 | 0.0076 | 1.1827 | 1.1528 |
| 4 | 1.1958 | 1.1603 | 0.0081 | 0.0038 | 0.0049 | 1.2003 | 1.1690 |
| 5 | 1.1817 | 1.1364 | 0.0076 | 0.0088 | 0.0049 | 1.1890 | 1.1258 |
| 6 | 1.1840 | 1.1491 | 0.0085 | 0.0083 | 0.0083 | 1.1942 | 1.1633 |
| 7 | 1.1851 | 1.1733 | 0.0074 | 0.0056 | 0.0059 | 1.1922 | 1.1746 |
| 8 | 1.1750 | 1.1600 | 0.0041 | 0.0081 | 0.0074 | 1.1869 | 1.1258 |
| 9 | 1.1973 | 1.1607 | 0.0165 | 0.0069 | 0.0080 | 1.1958 | 1.1382 |

A / d=5

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 1.0504 | 1.1435 | 0.0020 | 0.0062 | 0.0033 | 1.0463 | 1.1176 |
| 1 | 1.0507 | 1.1405 | 0.0020 | -0.0001 | 0.0221 | 1.0866 | 1.1701 |
| 2 | 1.0863 | 1.1166 | 0.0023 | 0.0024 | 0.0050 | 1.0315 | 1.1035 |
| 3 | 1.0570 | 1.1269 | 0.0024 | -0.0003 | 0.0029 | 1.0634 | 1.1549 |
| 4 | 1.0173 | 1.0522 | -0.0005 | 0.0011 | 0.0017 | 1.0538 | 1.1236 |
| 5 | 1.0827 | 1.1693 | 0.0015 | 0.0012 | 0.0095 | 1.0340 | 1.1814 |
| 6 | 1.0541 | 1.0528 | 0.0021 | 0.0032 | 0.0097 | 1.0444 | 1.1756 |
| 7 | 1.0317 | 1.1767 | 0.0025 | 0.0021 | 0.0032 | 1.0455 | 1.1538 |
| 8 | 1.0548 | 1.1529 | 0.0006 | -0.0009 | 0.0054 | 1.0350 | 1.1452 |
| 9 | 1.0726 | 1.1305 | 0.0016 | 0.0027 | 0.0093 | 1.0551 | 1.1326 |

A / d=9

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.0091 | 0.0095 | 0.0004 | -0.0004 | 0.0052 | 0.0242 | 0.0111 |
| 1 | 0.0149 | 0.0117 | 0.0001 | 0.0084 | 0.0005 | 0.0329 | 0.0120 |
| 2 | 0.0347 | 0.0075 | 0.0005 | 0.0007 | 0.0008 | 0.0335 | 0.0093 |
| 3 | 0.0190 | 0.0123 | 0.0018 | -0.0006 | -0.0002 | 0.0267 | 0.0124 |
| 4 | 0.0137 | 0.0096 | 0.0002 | 0.0009 | 0.0007 | 0.0329 | 0.0117 |
| 5 | 0.0132 | 0.0071 | 0.0007 | 0.0009 | 0.0011 | 0.0103 | 0.0099 |
| 6 | 0.0086 | 0.0105 | 0.0005 | 0.0020 | -0.0011 | 0.0310 | 0.0084 |
| 7 | 0.0279 | 0.0103 | 0.0003 | 0.0007 | 0.0016 | 0.0218 | 0.0130 |
| 8 | 0.0263 | 0.0087 | 0.0218 | 0.0006 | 0.0003 | 0.0409 | 0.0085 |
| 9 | 0.0280 | 0.0114 | 0.0009 | 0.0000 | 0.0008 | 0.0548 | 0.0127 |

A / d=17

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.0050 | 0.0081 | 0.0015 | 0.0013 | 0.0004 | 0.0056 | 0.0082 |
| 1 | 0.0049 | 0.0076 | 0.0005 | 0.0002 | -0.0000 | 0.0039 | 0.0083 |
| 2 | 0.0054 | 0.0095 | 0.0005 | -0.0001 | 0.0004 | 0.0034 | 0.0075 |
| 3 | 0.0064 | 0.0081 | -0.0004 | 0.0006 | 0.0003 | 0.0058 | 0.0094 |
| 4 | 0.0075 | 0.0067 | 0.0002 | -0.0004 | 0.0006 | 0.0044 | 0.0080 |
| 5 | 0.0063 | 0.0094 | 0.0004 | 0.0004 | 0.0001 | 0.0072 | 0.0071 |
| 6 | 0.0053 | 0.0084 | 0.0008 | 0.0012 | 0.0014 | 0.0067 | 0.0091 |
| 7 | 0.0061 | 0.0081 | 0.0007 | -0.0004 | 0.0002 | 0.0049 | 0.0090 |
| 8 | 0.0061 | 0.0084 | -0.0003 | 0.0000 | -0.0001 | 0.0026 | 0.0080 |
| 9 | 0.0049 | 0.0095 | 0.0001 | 0.0000 | -0.0008 | 0.0057 | 0.0106 |

## Per-seed IWAE (K=64) gap, condition B


B / d=3

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.1714 | 0.1519 | 0.2131 | 0.0166 | 0.2081 | 0.1713 | 0.1463 |
| 1 | 0.1440 | 0.1194 | 0.2113 | 0.2204 | 0.2112 | 0.1401 | 0.1255 |
| 2 | 0.1378 | 0.1187 | 0.1601 | 0.0167 | 0.2237 | 0.1388 | 0.1251 |
| 3 | 0.1805 | 0.1435 | 0.2195 | 0.0165 | 0.2134 | 0.1778 | 0.0980 |
| 4 | 0.1471 | 0.1285 | 0.1022 | 0.1951 | 0.2050 | 0.1411 | 0.1221 |
| 5 | 0.1417 | 0.1089 | 0.2190 | 0.0143 | 0.2144 | 0.1484 | 0.0934 |
| 6 | 0.1578 | 0.1384 | 0.1517 | 0.0162 | 0.2433 | 0.1406 | 0.1325 |
| 7 | 0.1611 | 0.1422 | 0.1983 | 0.0119 | 0.2200 | 0.1555 | 0.1002 |
| 8 | 0.1503 | 0.1366 | 0.2236 | 0.1483 | 0.2267 | 0.1506 | 0.1321 |
| 9 | 0.1372 | 0.1408 | 0.1276 | 0.0148 | 0.1858 | 0.1313 | 0.1293 |

B / d=5

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.4319 | 0.1883 | 0.1446 | 0.1420 | 0.1508 | 0.4514 | 0.1856 |
| 1 | 0.4482 | 0.2012 | 0.1583 | 0.0190 | 0.1683 | 0.4592 | 0.1907 |
| 2 | 0.4572 | 0.1778 | 0.1315 | 0.1560 | 0.1381 | 0.4462 | 0.1774 |
| 3 | 0.4521 | 0.1806 | 0.1462 | 0.1537 | 0.1488 | 0.4386 | 0.1882 |
| 4 | 0.4561 | 0.1823 | 0.1534 | 0.1493 | 0.1480 | 0.4508 | 0.1928 |
| 5 | 0.4481 | 0.1841 | 0.1548 | 0.0065 | 0.1457 | 0.4497 | 0.1766 |
| 6 | 0.4382 | 0.1685 | 0.1479 | 0.1553 | 0.1439 | 0.4321 | 0.1791 |
| 7 | 0.4534 | 0.1917 | 0.1580 | 0.1572 | 0.1522 | 0.4698 | 0.1990 |
| 8 | 0.4362 | 0.2027 | 0.1504 | 0.1710 | 0.1553 | 0.4477 | 0.1956 |
| 9 | 0.4376 | 0.1830 | 0.1439 | 0.1474 | 0.1483 | 0.4410 | 0.1798 |

B / d=9

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.3866 | 0.2740 | 0.0032 | 0.0062 | 0.0045 | 0.3860 | 0.2775 |
| 1 | 0.3919 | 0.2884 | 0.0044 | 0.0097 | 0.0088 | 0.3923 | 0.2871 |
| 2 | 0.3964 | 0.2825 | 0.0035 | 0.0119 | 0.0048 | 0.3972 | 0.2833 |
| 3 | 0.3833 | 0.2902 | 0.0117 | 0.0104 | 0.0024 | 0.3814 | 0.2817 |
| 4 | 0.3883 | 0.2708 | 0.0074 | 0.0038 | 0.0046 | 0.3718 | 0.2786 |
| 5 | 0.3847 | 0.2836 | 0.0038 | 0.0035 | 0.0033 | 0.3957 | 0.2800 |
| 6 | 0.4007 | 0.2986 | 0.0043 | 0.0048 | 0.0259 | 0.3853 | 0.2911 |
| 7 | 0.3916 | 0.2813 | 0.0043 | 0.0021 | 0.0019 | 0.3982 | 0.2866 |
| 8 | 0.4040 | 0.2879 | 0.0050 | 0.0106 | 0.0048 | 0.4006 | 0.2908 |
| 9 | 0.3916 | 0.2844 | 0.0042 | 0.0038 | 0.0029 | 0.3890 | 0.2818 |

B / d=17

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_epsinit | vmf_untuned | power_spherical_untuned |
|---|---|---|---|---|---|---|---|
| 0 | 0.2162 | 0.1667 | 0.0068 | 0.0075 | 0.0082 | 0.2121 | 0.1691 |
| 1 | 0.2267 | 0.1681 | 0.0047 | 0.0019 | 0.0014 | 0.2213 | 0.1667 |
| 2 | 0.2234 | 0.1685 | 0.0097 | 0.0043 | 0.0039 | 0.2170 | 0.1645 |
| 3 | 0.2024 | 0.1616 | 0.0094 | 0.0066 | 0.0055 | 0.2084 | 0.1586 |
| 4 | 0.2097 | 0.1575 | 0.0113 | 0.0035 | 0.0018 | 0.2127 | 0.1574 |
| 5 | 0.2179 | 0.1675 | 0.0007 | 0.0029 | 0.0010 | 0.2173 | 0.1686 |
| 6 | 0.2059 | 0.1617 | 0.0048 | 0.0052 | 0.0095 | 0.2092 | 0.1602 |
| 7 | 0.2051 | 0.1625 | 0.0034 | 0.0022 | 0.0016 | 0.2154 | 0.1664 |
| 8 | 0.2137 | 0.1602 | -0.0008 | 0.0024 | 0.0028 | 0.2161 | 0.1617 |
| 9 | 0.2135 | 0.1622 | 0.0022 | 0.0017 | 0.0009 | 0.2114 | 0.1596 |

## Per-seed IWAE (K=64) gap, condition C


C / d=3

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.7353 | 0.6799 | 0.0179 | 0.0240 |
| 1 | 0.7111 | 0.6713 | 0.0341 | 0.0099 |
| 2 | 0.7172 | 0.6475 | 0.0193 | 0.1151 |
| 3 | 0.7126 | 0.6538 | 0.0201 | 0.0182 |
| 4 | 0.7102 | 0.6511 | 0.0122 | 0.0093 |
| 5 | 0.7079 | 0.6659 | 0.0116 | 0.0134 |
| 6 | 0.7016 | 0.6829 | 0.0256 | 0.0134 |
| 7 | 0.7491 | 0.6761 | 0.0235 | 0.0348 |
| 8 | 0.7078 | 0.6355 | 0.0134 | 0.0137 |
| 9 | 0.7689 | 0.6822 | 0.0152 | 0.0116 |

C / d=5

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.6698 | 0.5289 | 0.0024 | 0.0071 |
| 1 | 0.6737 | 0.5358 | 0.0059 | 0.0107 |
| 2 | 0.6839 | 0.5297 | 0.0112 | 0.0111 |
| 3 | 0.6638 | 0.5121 | 0.0068 | 0.0101 |
| 4 | 0.6701 | 0.5127 | 0.0104 | 0.0069 |
| 5 | 0.6730 | 0.5264 | 0.0121 | 0.0748 |
| 6 | 0.6512 | 0.5107 | 0.0073 | 0.0031 |
| 7 | 0.6897 | 0.5458 | 0.0071 | 0.0369 |
| 8 | 0.6820 | 0.5217 | 0.0037 | 0.0187 |
| 9 | 0.6567 | 0.5050 | 0.0042 | 0.0055 |

C / d=9

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.4760 | 0.3965 | 0.0013 | 0.0124 |
| 1 | 0.4838 | 0.3952 | 0.0038 | 0.0030 |
| 2 | 0.4726 | 0.3927 | 0.0015 | 0.0022 |
| 3 | 0.4836 | 0.3989 | 0.0030 | 0.0031 |
| 4 | 0.4728 | 0.3799 | 0.0131 | 0.0064 |
| 5 | 0.4768 | 0.3982 | 0.0109 | 0.0063 |
| 6 | 0.4849 | 0.4060 | 0.0043 | 0.0014 |
| 7 | 0.4885 | 0.4039 | 0.0027 | 0.0043 |
| 8 | 0.4896 | 0.4027 | 0.0047 | 0.0060 |
| 9 | 0.4772 | 0.3831 | 0.0117 | 0.0062 |

C / d=17

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.2274 | 0.1901 | 0.0053 | 0.0060 |
| 1 | 0.2387 | 0.1917 | 0.0030 | 0.0010 |
| 2 | 0.2288 | 0.1905 | 0.0031 | 0.0026 |
| 3 | 0.2273 | 0.1839 | 0.0050 | 0.0031 |
| 4 | 0.2162 | 0.1749 | 0.0003 | -0.0004 |
| 5 | 0.2303 | 0.1902 | 0.0021 | 0.0020 |
| 6 | 0.2208 | 0.1837 | 0.0054 | 0.0066 |
| 7 | 0.2358 | 0.1824 | 0.0012 | 0.0002 |
| 8 | 0.2256 | 0.1885 | -0.0007 | 0.0030 |
| 9 | 0.2250 | 0.1832 | -0.0002 | -0.0003 |

## Descriptive arm comparisons (mean paired diff = A - B)

| cond | d | comparison | n | mean gap (A) | mean gap (B) | mean paired diff (A - B) | seeds A better | IWAE paired diff (A - B) |
|---|---|---|---|---|---|---|---|---|
| A | 3 | tnbbeta_init0 vs tnbbeta | 10 | 0.0453 | 0.0833 | -0.0381 | 8/10 | -0.0011 |
| A | 3 | tnbbeta_epsinit vs tnbbeta | 10 | 0.0705 | 0.0833 | -0.0128 | 6/10 | -0.0005 |
| A | 3 | tnbbeta_epsinit vs vmf | 10 | 0.0705 | 1.8523 | -1.7817 | 10/10 | -1.1766 |
| A | 3 | tnbbeta_init0 vs vmf | 10 | 0.0453 | 1.8523 | -1.8070 | 10/10 | -1.1772 |
| A | 3 | vmf_untuned vs vmf | 10 | 1.8540 | 1.8523 | 0.0017 | 3/10 | 0.0066 |
| A | 3 | power_spherical_untuned vs power_spherical | 10 | 1.8264 | 1.8256 | 0.0008 | 5/10 | -0.0035 |
| A | 3 | vmf vs power_spherical | 10 | 1.8523 | 1.8256 | 0.0266 | 0/10 | 0.0279 |
| A | 5 | tnbbeta_init0 vs tnbbeta | 10 | 0.0425 | 0.0291 | 0.0134 | 4/10 | 0.0001 |
| A | 5 | tnbbeta_epsinit vs tnbbeta | 10 | 0.0730 | 0.0291 | 0.0439 | 1/10 | 0.0056 |
| A | 5 | tnbbeta_epsinit vs vmf | 10 | 0.0730 | 3.3030 | -3.2301 | 10/10 | -1.0485 |
| A | 5 | tnbbeta_init0 vs vmf | 10 | 0.0425 | 3.3030 | -3.2606 | 10/10 | -1.0540 |
| A | 5 | vmf_untuned vs vmf | 10 | 3.3004 | 3.3030 | -0.0026 | 6/10 | -0.0062 |
| A | 5 | power_spherical_untuned vs power_spherical | 10 | 3.1111 | 3.1131 | -0.0019 | 6/10 | 0.0196 |
| A | 5 | vmf vs power_spherical | 10 | 3.3030 | 3.1131 | 0.1900 | 0/10 | -0.0705 |
| A | 9 | tnbbeta_init0 vs tnbbeta | 10 | 0.0280 | 0.0390 | -0.0110 | 6/10 | -0.0014 |
| A | 9 | tnbbeta_epsinit vs tnbbeta | 10 | 0.0250 | 0.0390 | -0.0140 | 7/10 | -0.0017 |
| A | 9 | tnbbeta_epsinit vs vmf | 10 | 0.0250 | 3.1055 | -3.0805 | 10/10 | -0.0186 |
| A | 9 | tnbbeta_init0 vs vmf | 10 | 0.0280 | 3.1055 | -3.0775 | 10/10 | -0.0182 |
| A | 9 | vmf_untuned vs vmf | 10 | 3.0932 | 3.1055 | -0.0122 | 8/10 | 0.0114 |
| A | 9 | power_spherical_untuned vs power_spherical | 10 | 2.5956 | 2.5937 | 0.0019 | 5/10 | 0.0010 |
| A | 9 | vmf vs power_spherical | 10 | 3.1055 | 2.5937 | 0.5118 | 0/10 | 0.0097 |
| A | 17 | tnbbeta_init0 vs tnbbeta | 10 | 0.0151 | 0.0147 | 0.0005 | 5/10 | -0.0001 |
| A | 17 | tnbbeta_epsinit vs tnbbeta | 10 | 0.0124 | 0.0147 | -0.0023 | 5/10 | -0.0002 |
| A | 17 | tnbbeta_epsinit vs vmf | 10 | 0.0124 | 1.5415 | -1.5291 | 10/10 | -0.0056 |
| A | 17 | tnbbeta_init0 vs vmf | 10 | 0.0151 | 1.5415 | -1.5264 | 10/10 | -0.0055 |
| A | 17 | vmf_untuned vs vmf | 10 | 1.5375 | 1.5415 | -0.0040 | 8/10 | -0.0008 |
| A | 17 | power_spherical_untuned vs power_spherical | 10 | 1.2986 | 1.2965 | 0.0021 | 3/10 | 0.0001 |
| A | 17 | vmf vs power_spherical | 10 | 1.5415 | 1.2965 | 0.2450 | 0/10 | -0.0026 |
| B | 3 | tnbbeta_init0 vs tnbbeta | 10 | 0.1844 | 0.5198 | -0.3354 | 8/10 | -0.1156 |
| B | 3 | tnbbeta_epsinit vs tnbbeta | 10 | 0.5235 | 0.5198 | 0.0037 | 4/10 | 0.0325 |
| B | 3 | tnbbeta_epsinit vs vmf | 10 | 0.5235 | 0.4141 | 0.1094 | 0/10 | 0.0623 |
| B | 3 | tnbbeta_init0 vs vmf | 10 | 0.1844 | 0.4141 | -0.2298 | 7/10 | -0.0858 |
| B | 3 | vmf_untuned vs vmf | 10 | 0.4152 | 0.4141 | 0.0011 | 5/10 | -0.0033 |
| B | 3 | power_spherical_untuned vs power_spherical | 10 | 0.4057 | 0.4022 | 0.0034 | 5/10 | -0.0124 |
| B | 3 | vmf vs power_spherical | 10 | 0.4141 | 0.4022 | 0.0119 | 0/10 | 0.0200 |
| B | 5 | tnbbeta_init0 vs tnbbeta | 10 | 0.4677 | 0.5608 | -0.0932 | 5/10 | -0.0232 |
| B | 5 | tnbbeta_epsinit vs tnbbeta | 10 | 0.5605 | 0.5608 | -0.0003 | 6/10 | 0.0010 |
| B | 5 | tnbbeta_epsinit vs vmf | 10 | 0.5605 | 0.7426 | -0.1821 | 10/10 | -0.2960 |
| B | 5 | tnbbeta_init0 vs vmf | 10 | 0.4677 | 0.7426 | -0.2749 | 10/10 | -0.3202 |
| B | 5 | vmf_untuned vs vmf | 10 | 0.7421 | 0.7426 | -0.0005 | 5/10 | 0.0028 |
| B | 5 | power_spherical_untuned vs power_spherical | 10 | 0.5979 | 0.5990 | -0.0010 | 7/10 | 0.0005 |
| B | 5 | vmf vs power_spherical | 10 | 0.7426 | 0.5990 | 0.1436 | 0/10 | 0.2599 |
| B | 9 | tnbbeta_init0 vs tnbbeta | 10 | 0.0314 | 0.0244 | 0.0070 | 3/10 | 0.0015 |
| B | 9 | tnbbeta_epsinit vs tnbbeta | 10 | 0.0362 | 0.0244 | 0.0118 | 5/10 | 0.0012 |
| B | 9 | tnbbeta_epsinit vs vmf | 10 | 0.0362 | 0.5912 | -0.5550 | 10/10 | -0.3855 |
| B | 9 | tnbbeta_init0 vs vmf | 10 | 0.0314 | 0.5912 | -0.5598 | 10/10 | -0.3852 |
| B | 9 | vmf_untuned vs vmf | 10 | 0.5918 | 0.5912 | 0.0006 | 3/10 | -0.0021 |
| B | 9 | power_spherical_untuned vs power_spherical | 10 | 0.5120 | 0.5117 | 0.0003 | 4/10 | -0.0003 |
| B | 9 | vmf vs power_spherical | 10 | 0.5912 | 0.5117 | 0.0795 | 0/10 | 0.1077 |
| B | 17 | tnbbeta_init0 vs tnbbeta | 10 | 0.0204 | 0.0235 | -0.0031 | 6/10 | -0.0014 |
| B | 17 | tnbbeta_epsinit vs tnbbeta | 10 | 0.0195 | 0.0235 | -0.0040 | 7/10 | -0.0015 |
| B | 17 | tnbbeta_epsinit vs vmf | 10 | 0.0195 | 0.3240 | -0.3045 | 10/10 | -0.2098 |
| B | 17 | tnbbeta_init0 vs vmf | 10 | 0.0204 | 0.3240 | -0.3036 | 10/10 | -0.2096 |
| B | 17 | vmf_untuned vs vmf | 10 | 0.3242 | 0.3240 | 0.0002 | 4/10 | 0.0006 |
| B | 17 | power_spherical_untuned vs power_spherical | 10 | 0.2812 | 0.2815 | -0.0003 | 5/10 | -0.0004 |
| B | 17 | vmf vs power_spherical | 10 | 0.3240 | 0.2815 | 0.0426 | 0/10 | 0.0498 |
| C | 3 | tnbbeta_init0 vs tnbbeta | 10 | 0.0733 | 0.0628 | 0.0105 | 6/10 | 0.0071 |
| C | 3 | tnbbeta_init0 vs vmf | 10 | 0.0733 | 1.1049 | -1.0316 | 10/10 | -0.6958 |
| C | 3 | vmf vs power_spherical | 10 | 1.1049 | 0.9842 | 0.1207 | 0/10 | 0.0575 |
| C | 5 | tnbbeta_init0 vs tnbbeta | 10 | 0.0673 | 0.0352 | 0.0321 | 2/10 | 0.0114 |
| C | 5 | tnbbeta_init0 vs vmf | 10 | 0.0673 | 0.8833 | -0.8160 | 10/10 | -0.6529 |
| C | 5 | vmf vs power_spherical | 10 | 0.8833 | 0.7597 | 0.1237 | 0/10 | 0.1485 |
| C | 9 | tnbbeta_init0 vs tnbbeta | 10 | 0.0271 | 0.0299 | -0.0028 | 6/10 | -0.0006 |
| C | 9 | tnbbeta_init0 vs vmf | 10 | 0.0271 | 0.6420 | -0.6149 | 10/10 | -0.4754 |
| C | 9 | vmf vs power_spherical | 10 | 0.6420 | 0.5597 | 0.0823 | 0/10 | 0.0849 |
| C | 17 | tnbbeta_init0 vs tnbbeta | 10 | 0.0108 | 0.0152 | -0.0044 | 7/10 | -0.0001 |
| C | 17 | tnbbeta_init0 vs vmf | 10 | 0.0108 | 0.3282 | -0.3174 | 10/10 | -0.2252 |
| C | 17 | vmf vs power_spherical | 10 | 0.3282 | 0.2829 | 0.0453 | 0/10 | 0.0417 |

## Ring statistic, decoder axis and fitted parameters

| cond | d | arm | diverged | sigma (mean +- sd; true 0.2) | ring R (median over x; seed median [min-max]) | single-index frac | abs cos(axis, e_1) | p | q | epsilon | m = eps - (d-1)/2 | kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 3 | vmf | 0/10 | 0.2000 +- 0.0000 | 0.982 [0.982-0.983] | 1.000 | 1.000 | - | - | - | - | 36.3 |
| A | 3 | power_spherical | 0/10 | 0.2000 +- 0.0000 | 0.982 [0.982-0.983] | 1.000 | 1.000 | - | - | - | - | 71.1 |
| A | 3 | tnbbeta | 0/10 | 0.2000 +- 0.0000 | 0.052 [0.050-0.053] | 1.000 | 1.000 | 0.741 | 0.67 | 9.08 | 8.08 | - |
| A | 3 | tnbbeta_init0 | 0/10 | 0.2000 +- 0.0000 | 0.052 [0.051-0.053] | 1.000 | 1.000 | 0.498 | 0.589 | 13.1 | 12.1 | - |
| A | 3 | tnbbeta_epsinit | 0/10 | 0.2000 +- 0.0000 | 0.052 [0.050-0.053] | 1.000 | 1.000 | 0.742 | 0.702 | 8.8 | 7.8 | - |
| A | 3 | vmf_untuned | 0/10 | 0.2000 +- 0.0000 | 0.982 [0.982-0.983] | 1.000 | 1.000 | - | - | - | - | 36.7 |
| A | 3 | power_spherical_untuned | 0/10 | 0.2000 +- 0.0000 | 0.982 [0.982-0.983] | 1.000 | 1.000 | - | - | - | - | 70.7 |
| A | 5 | vmf | 0/10 | 0.2000 +- 0.0000 | 0.877 [0.870-0.886] | 1.000 | 1.000 | - | - | - | - | 14.2 |
| A | 5 | power_spherical | 0/10 | 0.2000 +- 0.0000 | 0.888 [0.871-0.892] | 1.000 | 1.000 | - | - | - | - | 27.6 |
| A | 5 | tnbbeta | 0/10 | 0.2000 +- 0.0000 | 0.057 [0.056-0.058] | 1.000 | 1.000 | 0.672 | 0.684 | 12.6 | 10.6 | - |
| A | 5 | tnbbeta_init0 | 0/10 | 0.2000 +- 0.0000 | 0.057 [0.056-0.058] | 1.000 | 1.000 | 0.498 | 0.671 | 12.6 | 10.6 | - |
| A | 5 | tnbbeta_epsinit | 0/10 | 0.2000 +- 0.0000 | 0.057 [0.056-0.058] | 1.000 | 1.000 | 0.666 | 0.64 | 13.2 | 11.2 | - |
| A | 5 | vmf_untuned | 0/10 | 0.2000 +- 0.0000 | 0.876 [0.870-0.884] | 1.000 | 1.000 | - | - | - | - | 14.2 |
| A | 5 | power_spherical_untuned | 0/10 | 0.2000 +- 0.0000 | 0.886 [0.880-0.895] | 1.000 | 1.000 | - | - | - | - | 27.2 |
| A | 9 | vmf | 0/10 | 0.2000 +- 0.0000 | 0.070 [0.069-0.073] | 1.000 | 1.000 | - | - | - | - | 2.36 |
| A | 9 | power_spherical | 0/10 | 0.2000 +- 0.0000 | 0.061 [0.061-0.063] | 1.000 | 1.000 | - | - | - | - | 2.81 |
| A | 9 | tnbbeta | 0/10 | 0.2000 +- 0.0000 | 0.060 [0.059-0.061] | 1.000 | 1.000 | 0.619 | 0.63 | 16.6 | 12.6 | - |
| A | 9 | tnbbeta_init0 | 0/10 | 0.2000 +- 0.0000 | 0.061 [0.060-0.061] | 1.000 | 1.000 | 0.531 | 0.604 | 17.6 | 13.6 | - |
| A | 9 | tnbbeta_epsinit | 0/10 | 0.2000 +- 0.0000 | 0.060 [0.060-0.061] | 1.000 | 1.000 | 0.618 | 0.62 | 17.2 | 13.2 | - |
| A | 9 | vmf_untuned | 0/10 | 0.2000 +- 0.0000 | 0.070 [0.068-0.072] | 1.000 | 1.000 | - | - | - | - | 2.63 |
| A | 9 | power_spherical_untuned | 0/10 | 0.2000 +- 0.0000 | 0.061 [0.061-0.062] | 1.000 | 1.000 | - | - | - | - | 2.76 |
| A | 17 | vmf | 0/10 | 0.2000 +- 0.0000 | 0.065 [0.064-0.066] | 1.000 | 1.000 | - | - | - | - | 2.58 |
| A | 17 | power_spherical | 0/10 | 0.2000 +- 0.0000 | 0.062 [0.062-0.063] | 1.000 | 1.000 | - | - | - | - | 3.19 |
| A | 17 | tnbbeta | 0/10 | 0.2000 +- 0.0000 | 0.061 [0.061-0.062] | 1.000 | 1.000 | 0.572 | 0.545 | 21.8 | 13.8 | - |
| A | 17 | tnbbeta_init0 | 0/10 | 0.2000 +- 0.0000 | 0.061 [0.061-0.062] | 1.000 | 1.000 | 0.428 | 0.538 | 22.8 | 14.8 | - |
| A | 17 | tnbbeta_epsinit | 0/10 | 0.2000 +- 0.0000 | 0.061 [0.061-0.062] | 1.000 | 1.000 | 0.576 | 0.538 | 22.2 | 14.2 | - |
| A | 17 | vmf_untuned | 0/10 | 0.2000 +- 0.0000 | 0.064 [0.064-0.065] | 1.000 | 1.000 | - | - | - | - | 2.56 |
| A | 17 | power_spherical_untuned | 0/10 | 0.2000 +- 0.0000 | 0.062 [0.062-0.063] | 1.000 | 1.000 | - | - | - | - | 3.18 |
| B | 3 | vmf | 0/10 | 0.1815 +- 0.0012 | 0.940 [0.936-0.942] | 0.669 | 0.654 | - | - | - | - | 14.1 |
| B | 3 | power_spherical | 0/10 | 0.1876 +- 0.0034 | 0.932 [0.925-0.935] | 0.658 | 0.514 | - | - | - | - | 22.8 |
| B | 3 | tnbbeta | 0/10 | 0.1738 +- 0.0257 | 0.954 [0.900-0.966] | 0.706 | 0.464 | 0.982 | 1e-06 | 2.96 | 1.96 | - |
| B | 3 | tnbbeta_init0 | 0/10 | 0.1894 +- 0.0211 | 0.054 [0.052-0.971] | 0.767 | 0.296 | 0.498 | 0.517 | 13.4 | 12.4 | - |
| B | 3 | tnbbeta_epsinit | 0/10 | 0.1547 +- 0.0083 | 0.962 [0.950-0.970] | 0.736 | 0.355 | 0.986 | 1e-06 | 2.82 | 1.82 | - |
| B | 3 | vmf_untuned | 0/10 | 0.1824 +- 0.0011 | 0.937 [0.935-0.940] | 0.689 | 0.414 | - | - | - | - | 13.7 |
| B | 3 | power_spherical_untuned | 0/10 | 0.1938 +- 0.0097 | 0.928 [0.910-0.932] | 0.666 | 0.454 | - | - | - | - | 22.2 |
| B | 5 | vmf | 0/10 | 0.3185 +- 0.0013 | 0.286 [0.280-0.297] | 0.723 | 0.438 | - | - | - | - | 2.63 |
| B | 5 | power_spherical | 0/10 | 0.2474 +- 0.0013 | 0.505 [0.494-0.515] | 0.603 | 0.326 | - | - | - | - | 6.54 |
| B | 5 | tnbbeta | 0/10 | 0.2353 +- 0.0010 | 0.554 [0.537-0.557] | 0.529 | 0.292 | 0.828 | 0.000187 | 6.52 | 4.52 | - |
| B | 5 | tnbbeta_init0 | 0/10 | 0.2296 +- 0.0122 | 0.548 [0.057-0.568] | 0.531 | 0.555 | 0.607 | 0.000365 | 6.55 | 4.55 | - |
| B | 5 | tnbbeta_epsinit | 0/10 | 0.2352 +- 0.0015 | 0.547 [0.525-0.563] | 0.529 | 0.275 | 0.829 | 0.000373 | 6.58 | 4.58 | - |
| B | 5 | vmf_untuned | 0/10 | 0.3186 +- 0.0014 | 0.290 [0.284-0.301] | 0.726 | 0.362 | - | - | - | - | 2.64 |
| B | 5 | power_spherical_untuned | 0/10 | 0.2476 +- 0.0012 | 0.507 [0.494-0.514] | 0.601 | 0.444 | - | - | - | - | 6.54 |
| B | 9 | vmf | 0/10 | 0.2929 +- 0.0010 | 0.141 [0.139-0.151] | 0.803 | 0.215 | - | - | - | - | 2.66 |
| B | 9 | power_spherical | 0/10 | 0.2652 +- 0.0006 | 0.182 [0.173-0.186] | 0.766 | 0.261 | - | - | - | - | 3.91 |
| B | 9 | tnbbeta | 0/10 | 0.2023 +- 0.0012 | 0.060 [0.059-0.061] | 1.000 | 0.270 | 0.616 | 0.47 | 21.3 | 17.3 | - |
| B | 9 | tnbbeta_init0 | 0/10 | 0.2032 +- 0.0020 | 0.060 [0.059-0.061] | 1.000 | 0.303 | 0.512 | 0.401 | 24.5 | 20.5 | - |
| B | 9 | tnbbeta_epsinit | 0/10 | 0.2026 +- 0.0022 | 0.060 [0.059-0.061] | 1.000 | 0.367 | 0.617 | 0.497 | 20.8 | 16.8 | - |
| B | 9 | vmf_untuned | 0/10 | 0.2931 +- 0.0008 | 0.144 [0.131-0.146] | 0.805 | 0.222 | - | - | - | - | 2.61 |
| B | 9 | power_spherical_untuned | 0/10 | 0.2652 +- 0.0007 | 0.177 [0.170-0.183] | 0.768 | 0.246 | - | - | - | - | 3.86 |
| B | 17 | vmf | 0/10 | 0.2648 +- 0.0006 | 0.076 [0.075-0.078] | 0.933 | 0.151 | - | - | - | - | 2.64 |
| B | 17 | power_spherical | 0/10 | 0.2508 +- 0.0005 | 0.084 [0.083-0.090] | 0.887 | 0.134 | - | - | - | - | 3.53 |
| B | 17 | tnbbeta | 0/10 | 0.2026 +- 0.0017 | 0.062 [0.061-0.062] | 1.000 | 0.157 | 0.577 | 0.467 | 25.3 | 17.3 | - |
| B | 17 | tnbbeta_init0 | 0/10 | 0.2025 +- 0.0014 | 0.062 [0.061-0.062] | 1.000 | 0.160 | 0.459 | 0.359 | 28.5 | 20.5 | - |
| B | 17 | tnbbeta_epsinit | 0/10 | 0.2020 +- 0.0012 | 0.062 [0.061-0.062] | 1.000 | 0.191 | 0.577 | 0.457 | 25.1 | 17.1 | - |
| B | 17 | vmf_untuned | 0/10 | 0.2648 +- 0.0006 | 0.076 [0.075-0.078] | 0.938 | 0.189 | - | - | - | - | 2.66 |
| B | 17 | power_spherical_untuned | 0/10 | 0.2509 +- 0.0004 | 0.084 [0.082-0.087] | 0.890 | 0.155 | - | - | - | - | 3.54 |
| C | 3 | vmf | 0/10 | 0.3636 +- 0.0044 | 0.068 [0.064-0.074] | 1.000 | 0.793 | - | - | - | - | 2.92 |
| C | 3 | power_spherical | 0/10 | 0.3544 +- 0.0014 | 0.054 [0.052-0.055] | 1.000 | 0.804 | - | - | - | - | 3.54 |
| C | 3 | tnbbeta | 0/10 | 0.2004 +- 0.0014 | 0.053 [0.052-0.054] | 1.000 | 0.803 | 0.748 | 0.569 | 10.5 | 9.52 | - |
| C | 3 | tnbbeta_init0 | 0/10 | 0.2009 +- 0.0024 | 0.052 [0.050-0.053] | 1.000 | 0.782 | 0.501 | 0.585 | 11.6 | 10.6 | - |
| C | 5 | vmf | 0/10 | 0.3731 +- 0.0012 | 0.064 [0.063-0.067] | 1.000 | 0.479 | - | - | - | - | 1.94 |
| C | 5 | power_spherical | 0/10 | 0.3213 +- 0.0010 | 0.059 [0.058-0.060] | 1.000 | 0.509 | - | - | - | - | 3.06 |
| C | 5 | tnbbeta | 0/10 | 0.2020 +- 0.0016 | 0.057 [0.056-0.058] | 1.000 | 0.565 | 0.672 | 0.592 | 14.6 | 12.6 | - |
| C | 5 | tnbbeta_init0 | 0/10 | 0.2049 +- 0.0041 | 0.057 [0.056-0.059] | 1.000 | 0.550 | 0.366 | 0.567 | 15.2 | 13.2 | - |
| C | 9 | vmf | 0/10 | 0.3107 +- 0.0008 | 0.064 [0.063-0.065] | 1.000 | 0.334 | - | - | - | - | 2.17 |
| C | 9 | power_spherical | 0/10 | 0.2863 +- 0.0009 | 0.061 [0.060-0.061] | 1.000 | 0.334 | - | - | - | - | 3.02 |
| C | 9 | tnbbeta | 0/10 | 0.2026 +- 0.0019 | 0.060 [0.059-0.061] | 1.000 | 0.349 | 0.618 | 0.483 | 21.2 | 17.2 | - |
| C | 9 | tnbbeta_init0 | 0/10 | 0.2024 +- 0.0013 | 0.060 [0.059-0.060] | 1.000 | 0.333 | 0.469 | 0.409 | 24.3 | 20.3 | - |
| C | 17 | vmf | 0/10 | 0.2680 +- 0.0006 | 0.063 [0.062-0.063] | 1.000 | 0.157 | - | - | - | - | 2.57 |
| C | 17 | power_spherical | 0/10 | 0.2569 +- 0.0006 | 0.062 [0.062-0.062] | 1.000 | 0.159 | - | - | - | - | 3.23 |
| C | 17 | tnbbeta | 0/10 | 0.2020 +- 0.0012 | 0.062 [0.061-0.062] | 1.000 | 0.163 | 0.575 | 0.502 | 24.1 | 16.1 | - |
| C | 17 | tnbbeta_init0 | 0/10 | 0.2012 +- 0.0009 | 0.062 [0.061-0.062] | 1.000 | 0.170 | 0.5 | 0.469 | 25.4 | 17.4 | - |

## Condition A: exact KL(q || p) cross-check

| d | arm | ELBO gap mean (seeds) | exact KL(q||p) mean (seeds) | max abs diff over seeds |
|---|---|---|---|---|
| 3 | vmf | 1.8523 | 1.8536 | 0.0032 |
| 3 | power_spherical | 1.8256 | 1.8290 | 0.0055 |
| 3 | tnbbeta | 0.0833 | 0.0832 | 0.0012 |
| 3 | tnbbeta_init0 | 0.0453 | 0.0454 | 0.0008 |
| 3 | tnbbeta_epsinit | 0.0705 | 0.0705 | 0.0011 |
| 3 | vmf_untuned | 1.8540 | 1.8549 | 0.0036 |
| 3 | power_spherical_untuned | 1.8264 | 1.8307 | 0.0085 |
| 5 | vmf | 3.3030 | 3.3058 | 0.0061 |
| 5 | power_spherical | 3.1131 | 3.1119 | 0.0068 |
| 5 | tnbbeta | 0.0291 | 0.0291 | 0.0009 |
| 5 | tnbbeta_init0 | 0.0425 | 0.0429 | 0.0012 |
| 5 | tnbbeta_epsinit | 0.0730 | 0.0726 | 0.0018 |
| 5 | vmf_untuned | 3.3004 | 3.3027 | 0.0055 |
| 5 | power_spherical_untuned | 3.1111 | 3.1118 | 0.0043 |
| 9 | vmf | 3.1055 | 3.1067 | 0.0065 |
| 9 | power_spherical | 2.5937 | 2.5974 | 0.0075 |
| 9 | tnbbeta | 0.0390 | 0.0388 | 0.0010 |
| 9 | tnbbeta_init0 | 0.0280 | 0.0280 | 0.0012 |
| 9 | tnbbeta_epsinit | 0.0250 | 0.0252 | 0.0006 |
| 9 | vmf_untuned | 3.0932 | 3.0969 | 0.0111 |
| 9 | power_spherical_untuned | 2.5956 | 2.5998 | 0.0084 |
| 17 | vmf | 1.5415 | 1.5443 | 0.0078 |
| 17 | power_spherical | 1.2965 | 1.2939 | 0.0055 |
| 17 | tnbbeta | 0.0147 | 0.0147 | 0.0005 |
| 17 | tnbbeta_init0 | 0.0151 | 0.0152 | 0.0007 |
| 17 | tnbbeta_epsinit | 0.0124 | 0.0125 | 0.0006 |
| 17 | vmf_untuned | 1.5375 | 1.5400 | 0.0052 |
| 17 | power_spherical_untuned | 1.2986 | 1.2938 | 0.0078 |
