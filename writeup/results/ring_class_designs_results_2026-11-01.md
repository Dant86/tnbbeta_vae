# Four-class ring designs on S^2: results

Date: 2026-11-01. Pre-registration: `ring_class_designs_preregistration_2026-11-01.md`
(including its appended "Deviations" section, written before the full runs finished);
illustration of the designs: `ring_class_designs_illustration_2026-11-01.md`.

Code version: base commit `6e2e00d` (HEAD of `synthetic-graph-families`) plus the
**uncommitted working tree** (small-epsilon distribution fix, exact-log-q ELBO paths,
`p_init_offset`, `epsilon_floor`), plus the new, also uncommitted, files
`src/tnbbeta_vae/data/ring_classes.py`, `tests/data/test_ring_classes.py`,
`apps/synthetic/ring_classes.py`, `apps/synthetic/ring_classes_plot.py`,
`scripts/slurm/ring_classes.sbatch`. No existing file was edited for this study (apart from
appending the Deviations section to the pre-registration). Commit the tree before citing
these numbers.

## Commands

Training, 90 runs (SLURM job `2289940`, array 0-22, 4 worker processes per task on one GPU,
about 9-22 minutes per run, all 90 cells finished, no failures):

    sbatch --array=0-22 scripts/slurm/ring_classes.sbatch
    # = for design 1 in {vmf,power_spherical,tnbbeta,tnbbeta_init0,tnbbeta_fixed_mu},
    #   design 2 in {vmf,power_spherical,tnbbeta,tnbbeta_init0}, seed 0..9:
    uv run python -m apps.synthetic.ring_classes train --design D --arm A --seed S \
        --out-dir ring_recovery/classes

Analysis (login node, seconds) and figures:

    uv run python -m apps.synthetic.ring_classes summarize --out-dir ring_recovery/classes
    uv run --with matplotlib python -m apps.synthetic.ring_classes_plot \
        --out-dir ring_recovery/classes --fig-dir writeup/results/figures

Per-run outputs: `ring_recovery/classes/design<d>/<arm>/s<seed>.{json,npz,pt}` (summary,
per-x arrays, checkpoint). Tests: `tests/data/test_ring_classes.py` (11 tests, all pass);
the full suite (SLURM CPU job `2289963`) had 947 passed, 1 skipped. `ruff format`, `ruff check`
and `pyright` are clean on the new files.

Training settings (all from the pre-registration): `d = 3`, `sigma = 0.2`, MLP `[in,128,128]`
encoder, frozen true decoder, Uniform(S^2) prior, 4 ELBO samples, Adam 1e-3, batch 256, 300
epochs, 10% KL warm-up, 20,000 fresh training x (generator `100 + seed`), 2000 test x (seed
424242; design 2 balanced 500/class; design 1 band counts 477/530/510/483), exact KL with 1000
q draws in float64. The mean standard error of a per-x KL estimate is 0.036 for the caps and
0.007-0.012 for the TNBBeta arms.

## Headline against the pre-registered decision rules

| design | Holm (4 tests, alpha 0.05) | margin >= 0.25 nat | seeds with >= half of x within TV 0.25 (tnbbeta) | aligned seeds (tnbbeta) | "uses its ring capacity" |
|---|---|---|---|---|---|
| 1 (shared axis) | both survive | yes (1.79 vs vMF, 1.77 vs PS) | 10/10 | 10/10 | **met** |
| 2 (axis per class) | both survive | yes (1.72 vs vMF, 1.70 vs PS) | 10/10 | 10/10 | **met** |

Mean exact test KL (nats, mean +- sd over 10 seeds): design 1: vMF 1.856 +- 0.003,
PowerSpherical 1.830 +- 0.004, TNBBeta 0.065 +- 0.013; design 2: vMF 1.845 +- 0.001,
PowerSpherical 1.824 +- 0.004, TNBBeta 0.127 +- 0.065. TNBBeta is better than both caps on
all 10 seeds in both designs, so every one of the four Wilcoxon tests is at its smallest
attainable one-sided p, 2^-10 = 0.00098 (scipy and a direct enumeration agree), below every
Holm threshold (0.0125, 0.01667, 0.025, 0.05). Per-class (design 2) and per-band (design 1)
KL is below 0.16 for TNBBeta in every class and above 1.2 for both caps in every class
(design 1 caps: 1.2-1.3 in the polar bands 1 and 4, 2.4 in the equatorial bands 2 and 3; design 2 caps: 1.77-1.88 in every class).

The earlier single-axis pattern (TNBBeta aligned on 5 of 10 seeds only) did **not** recur: all
10 seeds of every TNBBeta arm in both designs have median |cos(mu, axis)| of 1.000 for every
class (design 2: the single network aligns four different axes, as a function of the class
input). `tnbbeta_init0` (plain random p bias) aligns as well, and `tnbbeta_fixed_mu` (direction
pinned to the true axis) is only 0.025 nat better than free `tnbbeta` in design 1, so
alignment is not what limits the free model here. What limits it is a thin tail: 0.5% (design
1) and 1.6% (design 2) of TNBBeta test-x have KL above 0.5 (per-x maximum 22 and 36 nats),
which drives the seed-to-seed variation in design 2 (seed KLs 0.05 to 0.25).

## Full output of `summarize`


## Design 1

| arm | seeds | mean test KL +- sd over seeds | median seed KL | seeds with >=half x within TV 0.25 | aligned seeds (median abs cos >= 0.99) | median abs cos (seed range) | KL(Uniform) ref |
|---|---|---|---|---|---|---|---|
| vmf | 10 | 1.856 +- 0.003 | 1.854 | 10/10 | 0/10 | 0.501 (0.494-0.512) | 23.34 |
| power_spherical | 10 | 1.830 +- 0.004 | 1.830 | 10/10 | 0/10 | 0.502 (0.496-0.515) | 23.34 |
| tnbbeta | 10 | 0.065 +- 0.013 | 0.062 | 10/10 | 10/10 | 1.000 (1.000-1.000) | 23.34 |
| tnbbeta_init0 | 10 | 0.057 +- 0.036 | 0.040 | 10/10 | 10/10 | 1.000 (1.000-1.000) | 23.34 |
| tnbbeta_fixed_mu | 10 | 0.040 +- 0.004 | 0.038 | 10/10 | 10/10 | 1.000 (1.000-1.000) | 23.34 |

Per-seed (design 1): KL / frac TV<=0.25 / median abs cos

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | tnbbeta_fixed_mu |
|---|---|---|---|---|---|
| 0 | 1.864 / 0.78 / 0.494 | 1.828 / 0.77 / 0.504 | 0.057 / 0.99 / 1.000 | 0.037 / 1.00 / 1.000 | 0.038 / 1.00 / 1.000 |
| 1 | 1.853 / 0.78 / 0.503 | 1.828 / 0.77 / 0.500 | 0.045 / 0.99 / 1.000 | 0.039 / 1.00 / 1.000 | 0.038 / 1.00 / 1.000 |
| 2 | 1.852 / 0.79 / 0.502 | 1.841 / 0.73 / 0.509 | 0.060 / 0.96 / 1.000 | 0.050 / 1.00 / 1.000 | 0.039 / 1.00 / 1.000 |
| 3 | 1.854 / 0.76 / 0.500 | 1.831 / 0.77 / 0.497 | 0.066 / 0.98 / 1.000 | 0.040 / 1.00 / 1.000 | 0.040 / 1.00 / 1.000 |
| 4 | 1.854 / 0.79 / 0.501 | 1.828 / 0.77 / 0.505 | 0.053 / 0.99 / 1.000 | 0.040 / 1.00 / 1.000 | 0.038 / 1.00 / 1.000 |
| 5 | 1.854 / 0.78 / 0.506 | 1.831 / 0.76 / 0.496 | 0.083 / 0.99 / 1.000 | 0.098 / 0.97 / 1.000 | 0.037 / 1.00 / 1.000 |
| 6 | 1.853 / 0.78 / 0.503 | 1.829 / 0.77 / 0.499 | 0.077 / 0.99 / 1.000 | 0.038 / 1.00 / 1.000 | 0.052 / 1.00 / 1.000 |
| 7 | 1.855 / 0.78 / 0.494 | 1.827 / 0.77 / 0.498 | 0.065 / 0.99 / 1.000 | 0.145 / 0.92 / 1.000 | 0.038 / 1.00 / 1.000 |
| 8 | 1.857 / 0.75 / 0.512 | 1.831 / 0.74 / 0.515 | 0.084 / 0.99 / 1.000 | 0.040 / 1.00 / 1.000 | 0.039 / 1.00 / 1.000 |
| 9 | 1.859 / 0.78 / 0.497 | 1.832 / 0.74 / 0.508 | 0.058 / 1.00 / 1.000 | 0.038 / 1.00 / 1.000 | 0.038 / 1.00 / 1.000 |

Mean exact KL by band (mean over seeds; test x of that band)

| arm | band 1 | band 2 | band 3 | band 4 |
|---|---|---|---|---|
| vmf | 1.275 | 2.405 | 2.402 | 1.249 |
| power_spherical | 1.221 | 2.406 | 2.407 | 1.192 |
| tnbbeta | 0.084 | 0.040 | 0.053 | 0.087 |
| tnbbeta_init0 | 0.083 | 0.015 | 0.018 | 0.118 |
| tnbbeta_fixed_mu | 0.078 | 0.006 | 0.005 | 0.076 |

Mean exact KL by band, split by alignment of the seed

| arm | seeds | band 1 | band 2 | band 3 | band 4 |
|---|---|---|---|---|---|
| vmf (not aligned) | 10 | 1.275 | 2.405 | 2.402 | 1.249 |
| power_spherical (not aligned) | 10 | 1.221 | 2.406 | 2.407 | 1.192 |
| tnbbeta (aligned) | 10 | 0.084 | 0.040 | 0.053 | 0.087 |
| tnbbeta_init0 (aligned) | 10 | 0.083 | 0.015 | 0.018 | 0.118 |
| tnbbeta_fixed_mu (aligned) | 10 | 0.078 | 0.006 | 0.005 | 0.076 |

Seeds with per-band median abs cos >= 0.99 (out of seeds run)

| arm | band 1 | band 2 | band 3 | band 4 |
|---|---|---|---|---|
| vmf | 9/10 | 0/10 | 0/10 | 10/10 |
| power_spherical | 10/10 | 0/10 | 0/10 | 10/10 |
| tnbbeta | 10/10 | 10/10 | 10/10 | 10/10 |
| tnbbeta_init0 | 10/10 | 10/10 | 10/10 | 10/10 |
| tnbbeta_fixed_mu | 10/10 | 10/10 | 10/10 | 10/10 |

Fraction of test x within TV 0.25, by band (mean over seeds)

| arm | band 1 | band 2 | band 3 | band 4 |
|---|---|---|---|---|
| vmf | 0.534 | 0.998 | 0.998 | 0.542 |
| power_spherical | 0.488 | 0.997 | 0.996 | 0.516 |
| tnbbeta | 0.987 | 0.984 | 0.977 | 0.997 |
| tnbbeta_init0 | 0.993 | 0.996 | 0.999 | 0.967 |
| tnbbeta_fixed_mu | 1.000 | 1.000 | 1.000 | 1.000 |

## Design 2

| arm | seeds | mean test KL +- sd over seeds | median seed KL | seeds with >=half x within TV 0.25 | aligned seeds (median abs cos >= 0.99) | median abs cos (seed range) | KL(Uniform) ref |
|---|---|---|---|---|---|---|---|
| vmf | 10 | 1.845 +- 0.001 | 1.846 | 10/10 | 0/10 | 0.521 (0.512-0.527) | 23.54 |
| power_spherical | 10 | 1.824 +- 0.004 | 1.823 | 10/10 | 0/10 | 0.524 (0.515-0.532) | 23.54 |
| tnbbeta | 10 | 0.127 +- 0.065 | 0.098 | 10/10 | 10/10 | 1.000 (1.000-1.000) | 23.54 |
| tnbbeta_init0 | 10 | 0.091 +- 0.065 | 0.048 | 10/10 | 10/10 | 1.000 (1.000-1.000) | 23.54 |

Per-seed (design 2): KL / frac TV<=0.25 / median abs cos

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 1.846 / 0.74 / 0.521 | 1.831 / 0.72 / 0.523 | 0.249 / 0.94 / 1.000 | 0.126 / 0.98 / 1.000 |
| 1 | 1.847 / 0.73 / 0.526 | 1.824 / 0.74 / 0.525 | 0.138 / 0.98 / 1.000 | 0.042 / 1.00 / 1.000 |
| 2 | 1.843 / 0.74 / 0.524 | 1.820 / 0.73 / 0.532 | 0.054 / 1.00 / 1.000 | 0.045 / 1.00 / 1.000 |
| 3 | 1.846 / 0.77 / 0.512 | 1.821 / 0.74 / 0.518 | 0.086 / 0.98 / 1.000 | 0.123 / 0.95 / 1.000 |
| 4 | 1.844 / 0.73 / 0.521 | 1.822 / 0.74 / 0.524 | 0.084 / 0.99 / 1.000 | 0.230 / 0.93 / 1.000 |
| 5 | 1.844 / 0.78 / 0.521 | 1.826 / 0.76 / 0.523 | 0.080 / 0.98 / 1.000 | 0.044 / 1.00 / 1.000 |
| 6 | 1.846 / 0.74 / 0.515 | 1.826 / 0.73 / 0.515 | 0.224 / 0.95 / 1.000 | 0.155 / 0.95 / 1.000 |
| 7 | 1.844 / 0.75 / 0.527 | 1.820 / 0.76 / 0.517 | 0.110 / 0.98 / 1.000 | 0.046 / 1.00 / 1.000 |
| 8 | 1.848 / 0.75 / 0.521 | 1.823 / 0.74 / 0.524 | 0.160 / 0.94 / 1.000 | 0.050 / 1.00 / 1.000 |
| 9 | 1.846 / 0.74 / 0.521 | 1.828 / 0.74 / 0.530 | 0.087 / 0.99 / 1.000 | 0.046 / 1.00 / 1.000 |

Mean exact KL by class (mean over seeds; test x of that class)

| arm | class 1 | class 2 | class 3 | class 4 |
|---|---|---|---|---|
| vmf | 1.878 | 1.792 | 1.852 | 1.859 |
| power_spherical | 1.851 | 1.772 | 1.835 | 1.838 |
| tnbbeta | 0.112 | 0.153 | 0.116 | 0.127 |
| tnbbeta_init0 | 0.119 | 0.066 | 0.095 | 0.082 |

Mean exact KL by class, split by alignment of the seed

| arm | seeds | class 1 | class 2 | class 3 | class 4 |
|---|---|---|---|---|---|
| vmf (not aligned) | 10 | 1.878 | 1.792 | 1.852 | 1.859 |
| power_spherical (not aligned) | 10 | 1.851 | 1.772 | 1.835 | 1.838 |
| tnbbeta (aligned) | 10 | 0.112 | 0.153 | 0.116 | 0.127 |
| tnbbeta_init0 (aligned) | 10 | 0.119 | 0.066 | 0.095 | 0.082 |

Seeds with per-class median abs cos >= 0.99 (out of seeds run)

| arm | class 1 | class 2 | class 3 | class 4 |
|---|---|---|---|---|
| vmf | 0/10 | 0/10 | 0/10 | 0/10 |
| power_spherical | 0/10 | 0/10 | 0/10 | 0/10 |
| tnbbeta | 10/10 | 10/10 | 10/10 | 10/10 |
| tnbbeta_init0 | 10/10 | 10/10 | 10/10 | 10/10 |

Fraction of test x within TV 0.25, by class (mean over seeds)

| arm | class 1 | class 2 | class 3 | class 4 |
|---|---|---|---|---|
| vmf | 0.752 | 0.737 | 0.751 | 0.752 |
| power_spherical | 0.731 | 0.748 | 0.744 | 0.736 |
| tnbbeta | 0.966 | 0.979 | 0.979 | 0.972 |
| tnbbeta_init0 | 0.985 | 0.985 | 0.979 | 0.975 |

## Confirmatory tests (one-sided paired Wilcoxon, Holm at 0.05)

| design | comparator | n | mean paired diff (comparator - tnbbeta) | seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.25 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | power_spherical | 10 | 1.766 | 10/10 | 0.00098 | 0.00098 | 0.01667 | True | True |
| 1 | vmf | 10 | 1.791 | 10/10 | 0.00098 | 0.00098 | 0.01250 | True | True |
| 2 | power_spherical | 10 | 1.697 | 10/10 | 0.00098 | 0.00098 | 0.05000 | True | True |
| 2 | vmf | 10 | 1.718 | 10/10 | 0.00098 | 0.00098 | 0.02500 | True | True |

Decision rule 'TNBBeta uses its ring capacity' (both Holm tests, margin, 
and >= 8 of 10 seeds with >= half of test x within TV 0.25):

| design | Holm (both) | margin (both) | ring-shape seeds | aligned seeds | rule met |
|---|---|---|---|---|---|
| 1 | True | True | 10/10 | 10/10 | True |
| 2 | True | True | 10/10 | 10/10 | True |

## Descriptive comparisons (no p-values)

| design | comparison | n | mean KL (A) | mean KL (B) | mean paired diff (A - B) | seeds A better |
|---|---|---|---|---|---|---|
| 1 | tnbbeta vs tnbbeta_init0 | 10 | 0.065 | 0.057 | 0.008 | 2/10 |
| 2 | tnbbeta vs tnbbeta_init0 | 10 | 0.127 | 0.091 | 0.037 | 2/10 |
| 1 | tnbbeta_fixed_mu vs tnbbeta | 10 | 0.040 | 0.065 | -0.025 | 10/10 |
| 1 | tnbbeta_fixed_mu vs vmf | 10 | 0.040 | 1.856 | -1.816 | 10/10 |
| 1 | tnbbeta_fixed_mu vs power_spherical | 10 | 0.040 | 1.830 | -1.791 | 10/10 |
| 1 | tnbbeta_init0 vs vmf | 10 | 0.057 | 1.856 | -1.799 | 10/10 |
| 2 | tnbbeta_init0 vs vmf | 10 | 0.091 | 1.845 | -1.755 | 10/10 |
| 1 | tnbbeta_init0 vs power_spherical | 10 | 0.057 | 1.830 | -1.774 | 10/10 |
| 2 | tnbbeta_init0 vs power_spherical | 10 | 0.091 | 1.824 | -1.734 | 10/10 |
| 1 | vmf vs power_spherical | 10 | 1.856 | 1.830 | 0.025 | 0/10 |
| 2 | vmf vs power_spherical | 10 | 1.845 | 1.824 | 0.021 | 0/10 |

Fitted parameters (median over seeds of the per-seed median over test x):

| design | arm | p | q | epsilon | m = eps - 1 | kappa |
|---|---|---|---|---|---|---|
| 1 | power_spherical | - | - | - | - | 70.8 |
| 1 | tnbbeta | 0.744 | 0.712 | 8.61 | 7.61 | - |
| 1 | tnbbeta_fixed_mu | 0.495 | 0.578 | 13.4 | 12.4 | - |
| 1 | tnbbeta_init0 | 0.499 | 0.586 | 13.4 | 12.4 | - |
| 1 | vmf | - | - | - | - | 36.8 |
| 2 | power_spherical | - | - | - | - | 70.9 |
| 2 | tnbbeta | 0.749 | 0.719 | 8.13 | 7.13 | - |
| 2 | tnbbeta_init0 | 0.5 | 0.606 | 11.7 | 10.7 | - |
| 2 | vmf | - | - | - | - | 36.1 |

## Figures

Exact posterior next to the fitted vMF, PowerSpherical and TNBBeta posteriors (same seed,
same test x), one representative test x per class, orthographic view of S^2, star = class axis
`+a`, open star = `-a`, the far side drawn faintly. Representative seeds: the TNBBeta seed with
the lowest KL, and, because no non-aligned seed exists, the aligned seed with the highest KL.
Each panel's caption gives that x's KL, TV and |cos|.

* Design 1, lowest-KL seed (1): `figures/ring_classes_design1_aligned_s1_2026-11-01.png`
* Design 1, highest-KL seed (8): `figures/ring_classes_design1_worst_s8_2026-11-01.png`
* Design 2, lowest-KL seed (2): `figures/ring_classes_design2_aligned_s2_2026-11-01.png`
* Design 2, highest-KL seed (0): `figures/ring_classes_design2_worst_s0_2026-11-01.png`
  (class 4's panel shows a TNBBeta failure on one x, KL 1.06, TV 0.49).

The figures show what the tables cannot: the caps sit on one arc of the ring (their centre
is on the ring, so |cos(mu, axis)| is about |w|, not 1), while TNBBeta draws the whole ring
about the class axis.

## What this does and does not show

Shows, on these two synthetic tasks with the true decoder frozen: TNBBeta (library default
head, no tuning) recovers a latitude-ring posterior that vMF and PowerSpherical cannot
represent, with about 1.7 nats lower exact KL per observation, on every seed, with all four
pre-registered confirmatory tests rejecting under Holm with a margin seven times the required
0.25 nat, and the same holds when one network must serve four class-dependent axes (design 2).
The ring-capacity rule is met in both designs. The five-of-ten alignment failure of the
single-axis study is absent: 50 of 50 TNBBeta-family runs aligned.

Does not show, and cautions:

* **Alignment is only meaningful for TNBBeta.** For vMF and PowerSpherical the median
  |cos| of about 0.5 (0/10 "aligned") is not a failure: the best cap centre lies *on* the ring
  (|cos| is about |w|), so the pre-registered alignment statistic says nothing about those arms.
* **The registered ring-shape rule (latitude-marginal TV) cannot tell a ring from a cap.**
  The latitude marginal about the axis of a cap on the ring is close to the ring's (mean TV
  0.23-0.24, and 72-79% of x within 0.25 on every seed, i.e. also 10/10 seeds
  "ring-shaped" for vMF/PS). The rule was registered for TNBBeta only and TNBBeta passes it with much margin
  (mean TV 0.09-0.11), but the azimuthal uniformity of the ring is evidenced by the KL and the
  figures, not by TV.
* **Easy setting by construction.** The decoder is the true, frozen one; the posterior is a
  single exact ring (no second mode, 0.000 of 20,000 x); d = 3 only; sigma = 0.2. Level 1
  (representation) and Level 2 (per-datum optimisation) were not isolated, and a learned decoder
  (which can choose a cap-friendly decoder), d > 3, a latent class and real data were not run.
  It does not show that TNBBeta is needed on any real task.
* **The earlier failure is not explained, only absent.** This study differs from the earlier
  one in `h` (smoother), sigma, d = 3, 300 instead of 100 epochs, and the library-default p
  initialisation, but `tnbbeta_init0` also aligns, so initialisation is not the whole story;
  the cause of the earlier 5-of-10 pattern is not identified here.
* **Descriptive arms, no p-values (as registered).** `tnbbeta` vs `tnbbeta_init0`: 0.065 vs 0.057
  (design 1) and 0.127 vs 0.091 (design 2); `init0` is better on 8 of 10 seeds in each design
  (mean differences 0.008 and 0.037 nat; with the large seed variance of design 2 this is not
  evidence of an effect). `tnbbeta_fixed_mu` vs `tnbbeta`, design 1: 0.040 vs 0.065, better on
  10/10 seeds, mostly in the middle bands (bands 2 and 3: 0.005-0.006 vs 0.04-0.05; the polar bands are
  0.077 vs 0.085) - a small gain, so direction alignment is not the bottleneck. vMF vs PowerSpherical:
  PowerSpherical is lower by 0.02-0.025 nat on 10/10 seeds in both designs.
* **Seeds are the unit of the tests.** Test x, data stream and shuffle seeds are shared across
  arms within a seed; the 10 seeds do not capture variation across test sets (one fixed
  2000-x test set per design). The design-1 test set is not class balanced (477-530 per band).
* **Parameter regime.** TNBBeta settles at m = epsilon - 1 of about 7 (free, p ~ 0.75, q ~ 0.7)
  or 10.7-12.4 (init0, fixed_mu, p ~ 0.5): a unimodal ring, far from the bimodal `m < 0`
  regime of the graph experiments, with two different (p, q, epsilon) solutions of similar KL.

## Deviations

See the pre-registration's "Deviations" section (written before the full runs finished): no
class offset in design 2; float64 CPU evaluation; TV from a separate 8000-draw 90-bin histogram;
`tnbbeta_fixed_mu` keeps the default p init; a study-local model wrapper; the second-mode check
on 20,000 x per design; 90 runs packed 4 per GPU task. Nothing in the generative models, arms,
training, metrics, tests or decision rules was changed after seeing results.
