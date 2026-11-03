# ITD-only HRTF task: results (SADIE II, 10 subject splits)

Date: 2026-11-01. Code: base commit `667cfe1` plus an uncommitted working tree (new files
`apps/data/prepare_sadie2_itd.py`, `src/tnbbeta_vae/data/hrtf.py`, `apps/synthetic/hrtf_itd.py`,
`apps/synthetic/hrtf_itd_plot.py`, `scripts/slurm/hrtf_itd.sbatch`, `tests/data/test_hrtf.py`,
`tests/apps/test_hrtf_itd.py`; `.gitignore` gains `hrtf_runs/`); `HEAD` had moved to `945f133` by the time
of writing, no library code under `src/tnbbeta_vae/distributions/` or `models/losses/` was touched. Design,
tests and decision rules: `hrtf_itd_ring_preregistration_2026-11-01.md` (frozen; deviations appended there,
and repeated below). Gate: `hrtf_itd_gate_2026-11-01.md`.

Commands:

    uv run --with h5py python -m apps.data.prepare_sadie2_itd
    sbatch --array=0-23 scripts/slurm/hrtf_itd.sbatch      # 160 runs: xcorr + onset, arms V/L, 4 families, 10 splits
    uv run python -m apps.synthetic.hrtf_itd summarize --out-dir hrtf_runs
    uv run --with matplotlib python -m apps.synthetic.hrtf_itd_plot --out-dir hrtf_runs --fig-dir writeup/results/figures --seed 0

(each cell is `python -m apps.synthetic.hrtf_itd train --estimator E --arm A --family F --seed S --out-dir hrtf_runs`;
about 55 min per arm-V run, a few minutes for arm L, 7 runs sharing one GPU.) The full `pytest` suite was run
as a SLURM CPU job (957 passed, 1 skipped) before the final cosmetic edits to the summary and plot scripts;
`ruff format`, `ruff check` and `pyright` are clean on every new file.

## Headline (primary estimator: cross-correlation)

Pre-registered rule "TNBBeta uses its ring capacity on this task" (all four Holm tests and the 0.25-nat
margin): **NOT met**. Arm L passes both tests with a large margin; arm V does not.

| arm | comparator | mean paired diff (comparator - tnbbeta) | range over splits | splits tnbbeta better | p | Holm threshold | Holm | margin 0.25 |
|---|---|---|---|---|---|---|---|---|
| L (NLL of true direction) | vmf | 1.406 | 1.343 to 1.453 | 10/10 | 0.00098 | 0.0125 | reject | yes |
| L | power_spherical | 1.319 | 1.251 to 1.367 | 10/10 | 0.00098 | 0.0167 | reject | yes |
| V (exact KL to p_model) | vmf | 0.016 | -0.008 to 0.032 | 9/10 | 0.00293 | 0.0250 | reject | **no** |
| V | power_spherical | -0.014 | -0.037 to 0.004 | 1/10 | 0.99707 | 0.0500 | **no** | **no** |

p-values are the exact one-sided Wilcoxon p-values and agree with the independent sign-pattern enumeration.
The 10 splits overlap in subjects (each subject is held out in roughly 1/6 of them), so the p-values are
optimistic; the ranges over splits are given for that reason.

Alignment (median over test x of |cos(mu, axis)| >= 0.99): arm V 10/10 splits for every family (axis = the
decoder's u); arm L 10/10 for every family (median |cos| vmf 0.994, power_spherical 0.996, tnbbeta 1.000,
tnbbeta_init0 1.000). Oracle baselines for scale (held-out NLL of the true direction, same splits): uniform
2.531, oracle ring 0.532 (0.506 to 0.572), oracle cap 1.860. Arm L: tnbbeta 0.460 and tnbbeta_init0 0.450 sit
*below* the oracle ring (the free-form estimator of the gate was also below it: the ITD carries
non-axisymmetric information), vMF 1.867 and PowerSpherical 1.779 are at the oracle-cap level.
Sensitivity (onset ITD): arm L still Holm-significant for both comparisons (p = 0.00098, 10/10 splits) but the
margin fails (0.165 and 0.090 nats); arm V vmf significant, PowerSpherical not; rule not met. The oracle ring is
1.853 against the cap 2.074 there (ring better by 0.22), i.e. the gate's near-failure shows up as a small effect.

## What the arm V numbers actually are (read before the table)

Arm V's primary metric, the KL of q to the model's own p_model(z|x), is small for every family (0.01 to 0.05 nats)
and is **dominated by posterior collapse, not by ring fitting**. The learned sigma and training loss per run (JSON
`sigma`, `final_train_loss`) show it: for vMF and PowerSpherical, sigma is 0.83 and 0.76 (xcorr) and the
negative ELBO is 1.455 and 1.419, the value of a decoder that ignores z (log sqrt(2 pi) + 1/2 at sigma 1 is 1.419);
their fitted kappa is about 1 (vMF) and 0.98 (PS), i.e. q is almost uniform on S^2. p_model is then nearly
uniform too, and the KL between two nearly uniform densities is small. TNBBeta used the latent in most splits
(sigma 0.10 to 0.27, loss 1.366 to 1.404, versus 1.42 for the collapsed runs) but collapsed in xcorr split 4
(sigma 0.78) and, for `tnbbeta_init0`, in splits 5 (1.00) and 9 (loss 1.435); with the onset estimator
PowerSpherical collapses in 10 of 10 splits (sigma about 1.0; its KL is 0 to three decimals), `tnbbeta` in 6 and
`tnbbeta_init0` in 4. So a
lower KL on arm V is not evidence of a better posterior: PowerSpherical's lower KL (xcorr, 0.013 against
0.027) is the collapsed model whose p_model is trivially matched by its q, while TNBBeta's ELBO is better
(descriptive: negative ELBO 1.38 against 1.42 for PS and 1.46 for vMF). The pre-registered primary metric and
test are reported as registered; they do not measure what the study intended, and the rule fails for this
reason as much as for any difference in capacity. A matched comparison (equal sigma, or a decoder with sigma
fixed small so no run can collapse) would be needed to compare q against a ring; that was not registered and is
not claimed.

Arm V also does not recover the head-frame interaural axis: |cos(u, +/- y)| ranges 0.006 to 0.998 across splits
and families (median 0.5 to 0.7), so the unsupervised single-index latent is **not** tied to y. This is expected: x is
scalar, so the data constrain only the marginal of x and any axis u with a suitable g reproduces it. The
label-based latitude check about the signed u therefore mostly measures that mismatch (coverage 0.08 to 0.38 at
nominal 0.50) and is meaningful only in splits where |u_y| is near 1 (xcorr: vMF split 3, PowerSpherical split 3,
TNBBeta splits 4 and 7, |u_y| 0.995 and 0.956). I did not break the check out per split.

## Arm L

Arm L is the supervised density-fitting arm. vMF and PowerSpherical collapse to a cap whose NLL is at the
oracle-cap level (1.87 and 1.78) because a single cap cannot represent a thin ring about +y; TNBBeta
reaches 0.46 (xcorr), about 1.3 to 1.4 nats lower in every one of 10 splits, with a latitude marginal that
covers 0.475 / 0.941 at nominal 0.50 / 0.95 (vMF 0.92 / 0.99, PS 0.91 / 0.99: much too wide). TNBBeta's fitted
m is about 0.27 (xcorr; p 0.73, q 0.994, epsilon 1.27), i.e. m > 0: a unimodal ring-like latitude (a thin cap
at epsilon about 1.27 and q at its ceiling is a ring about its mean direction, which here aligns with +y at
|cos| 1.000). The `tnbbeta` versus `tnbbeta_init0` difference is within noise in both arms (arm L 0.010 mean,
range -0.050 to 0.051; arm V 0.009).

## What this does and does not show

* On supervised density fitting of measured ITD -> direction (arm L), TNBBeta captures the lateral ring that vMF
  and PowerSpherical cannot, by about 1.4 nats, in all 10 splits, with the primary estimator. This holds up to
  the onset estimator (still 10/10, but 0.09 to 0.17 nats, below the 0.25 margin).
* The unsupervised VAE arm does not show a TNBBeta advantage by the registered metric: the metric is dominated by
  collapse of vMF/PS, and the VAE does not find the interaural axis. This does not show that a VAE could not use
  TNBBeta's ring capacity, only that this configuration (learned sigma, free u, single-index decoder, 300 epochs)
  does not provide the comparison.
* Subject-level replication is thin (18 subjects, splits share subjects): treat p-values as descriptive.
* No collapse-controlled arm V, no hyperparameter tuning for any family (all fixed in advance), and one dataset.
* The oracle ring is not a ceiling on this task (arm L TNBBeta is below it), so "TNBBeta matches the oracle
  ring" is not the right reading either.

## Figures (split 0)

* `figures/hrtf_itd_latitude_s0_2026-11-01.png`: ITD against w and the fitted latitude marginals at ITD = -500,
  -150, 150, 500 us, arm V (about the signed u) and arm L (about +y), with the true w of held-out points
  within 15 us.
* `figures/hrtf_itd_spheres_armL_s0_2026-11-01.png`, `figures/hrtf_itd_spheres_armV_s0_2026-11-01.png`: orthographic
  views of q (vMF, PowerSpherical, TNBBeta) at ITD = -450, 0, 450 us with true held-out directions within 15 us;
  stars mark +/- y (L, R ears). In arm V q is rotated so the signed decoder axis sits on +y (ring azimuth is
  arbitrary there).

## Deviations from the pre-registration

Appended to the pre-registration before training: directions per subject are 2818 (H3-H9) and 2114 (H10-H20), not
2818 for all (42 980 rows); area-weighted standardisation; initial sigma 0.5 and random u; 4001-node grid;
seeds for sampler (200 + split), model (split) and evaluation (31337); arm L alignment against +/- y only.
Further, found while analysing (no analysis choice changed): (a) the oracle baselines in the summary use the
gate's own functions on this split's training subjects and are the gate's 50-quantile-bin fits;
(b) arm V's `|u_y|` and the label-based latitude check use `u` with the sign picked on the training subjects as
registered; (c) 14 tasks of an unrelated SLURM job (`ring-learned`, job 2290776) were cancelled by mistake (I took them for a
duplicate resubmission of this study's array) and were resubmitted with the same command (job 2291018, array
0-13; finished cells skip, in-progress cells restart); this study's runs were unaffected.

# Full tables (output of `summarize`)

Note: the 10 split replicates overlap in subjects (each of 18 subjects is held out in about 1/6 of the splits and trains in the rest), so the Wilcoxon p-values are optimistic; the range over splits accompanies every effect.


## Estimator: xcorr (primary)


### Arm V (VAE; primary metric: exact KL(q || p_model(.|x)), nats)

| family | splits | mean +- sd over splits | median | range | aligned splits (median abs cos >= 0.99) | median abs cos (range) | |u_y| median (range) | p / q / eps / m medians |
|---|---|---|---|---|---|---|---|---|
| vmf | 10 | 0.042 +- 0.001 | 0.042 | 0.041 to 0.043 | 10/10 | 1.000 (1.000-1.000) | 0.695 (0.072-0.982) | - / - / - / - |
| power_spherical | 10 | 0.013 +- 0.001 | 0.013 | 0.012 to 0.014 | 10/10 | 1.000 (0.999-1.000) | 0.726 (0.063-0.974) | - / - / - / - |
| tnbbeta | 10 | 0.027 +- 0.012 | 0.026 | 0.010 to 0.051 | 10/10 | 1.000 (1.000-1.000) | 0.529 (0.006-0.995) | 0.604 / 0.657 / 11.1 / 10.1 |
| tnbbeta_init0 | 10 | 0.018 +- 0.008 | 0.017 | 0.002 to 0.036 | 10/10 | 1.000 (0.999-1.000) | 0.515 (0.038-0.998) | 0.499 / 0.725 / 22.2 / 21.2 |

Per split (arm V): metric / median abs cos

| split | test subjects | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|---|
| 0 | H5,H6,H13 | 0.043 / 1.000 | 0.014 / 1.000 | 0.022 / 1.000 | 0.020 / 1.000 |
| 1 | H4,H14,H15 | 0.043 / 1.000 | 0.012 / 0.999 | 0.024 / 1.000 | 0.015 / 1.000 |
| 2 | H9,H10,H19 | 0.043 / 1.000 | 0.012 / 1.000 | 0.029 / 1.000 | 0.018 / 1.000 |
| 3 | H15,H18,H20 | 0.043 / 1.000 | 0.013 / 1.000 | 0.039 / 1.000 | 0.018 / 1.000 |
| 4 | H3,H10,H12 | 0.042 / 1.000 | 0.014 / 1.000 | 0.010 / 1.000 | 0.015 / 1.000 |
| 5 | H6,H10,H20 | 0.042 / 1.000 | 0.012 / 1.000 | 0.031 / 1.000 | 0.002 / 0.999 |
| 6 | H5,H13,H19 | 0.042 / 1.000 | 0.012 / 1.000 | 0.027 / 1.000 | 0.022 / 1.000 |
| 7 | H6,H9,H17 | 0.042 / 1.000 | 0.013 / 1.000 | 0.016 / 1.000 | 0.017 / 1.000 |
| 8 | H3,H12,H19 | 0.043 / 1.000 | 0.014 / 1.000 | 0.051 / 1.000 | 0.017 / 1.000 |
| 9 | H6,H13,H16 | 0.041 / 1.000 | 0.012 / 1.000 | 0.016 / 1.000 | 0.036 / 1.000 |

Per split (arm V): |u_y| = |cos(u, +/- y)|

| split | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.825 | 0.793 | 0.583 | 0.646 |
| 1 | 0.648 | 0.716 | 0.363 | 0.406 |
| 2 | 0.106 | 0.100 | 0.139 | 0.121 |
| 3 | 0.982 | 0.974 | 0.529 | 0.537 |
| 4 | 0.742 | 0.736 | 0.995 | 0.998 |
| 5 | 0.553 | 0.551 | 0.012 | 0.038 |
| 6 | 0.791 | 0.815 | 0.006 | 0.153 |
| 7 | 0.072 | 0.063 | 0.956 | 0.933 |
| 8 | 0.753 | 0.752 | 0.530 | 0.494 |
| 9 | 0.622 | 0.630 | 0.660 | 0.702 |

Label-based latitude check, arm V (about the signed decoder axis u; mean over splits; nominal coverage 0.50 / 0.95)

| family | central 50% coverage | central 95% coverage | mean abs error of the marginal's median (w units) | median abs error |
|---|---|---|---|---|
| vmf | 0.375 | 0.862 | 0.6530 | 0.6356 |
| power_spherical | 0.358 | 0.748 | 0.6414 | 0.6153 |
| tnbbeta | 0.117 | 0.219 | 0.7118 | 0.6603 |
| tnbbeta_init0 | 0.081 | 0.183 | 0.7699 | 0.7115 |

### Arm L (supervised; primary metric: held-out -log q(z_true|x), nats)

| family | splits | mean +- sd over splits | median | range | aligned splits (median abs cos >= 0.99) | median abs cos (range) | p / q / eps / m medians |
|---|---|---|---|---|---|---|---|
| vmf | 10 | 1.867 +- 0.008 | 1.865 | 1.854 to 1.885 | 10/10 | 0.994 (0.993-0.995) | - / - / - / - |
| power_spherical | 10 | 1.779 +- 0.010 | 1.777 | 1.765 to 1.799 | 10/10 | 0.996 (0.995-0.996) | - / - / - / - |
| tnbbeta | 10 | 0.460 +- 0.036 | 0.454 | 0.412 to 0.519 | 10/10 | 1.000 (0.999-1.000) | 0.733 / 0.994 / 1.27 / 0.274 |
| tnbbeta_init0 | 10 | 0.450 +- 0.051 | 0.440 | 0.382 to 0.569 | 10/10 | 1.000 (0.999-1.000) | 0.499 / 0.994 / 1.24 / 0.238 |

Per split (arm L): metric / median abs cos

| split | test subjects | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|---|
| 0 | H5,H6,H13 | 1.865 / 0.995 | 1.776 / 0.996 | 0.509 / 1.000 | 0.478 / 1.000 |
| 1 | H4,H14,H15 | 1.862 / 0.994 | 1.770 / 0.996 | 0.519 / 1.000 | 0.569 / 1.000 |
| 2 | H9,H10,H19 | 1.865 / 0.995 | 1.778 / 0.996 | 0.468 / 0.999 | 0.457 / 0.999 |
| 3 | H15,H18,H20 | 1.873 / 0.995 | 1.790 / 0.996 | 0.451 / 1.000 | 0.431 / 1.000 |
| 4 | H3,H10,H12 | 1.864 / 0.993 | 1.775 / 0.996 | 0.456 / 1.000 | 0.425 / 1.000 |
| 5 | H6,H10,H20 | 1.866 / 0.994 | 1.781 / 0.996 | 0.489 / 1.000 | 0.476 / 1.000 |
| 6 | H5,H13,H19 | 1.865 / 0.994 | 1.779 / 0.996 | 0.412 / 1.000 | 0.406 / 1.000 |
| 7 | H6,H9,H17 | 1.854 / 0.994 | 1.765 / 0.996 | 0.416 / 0.999 | 0.436 / 0.999 |
| 8 | H3,H12,H19 | 1.885 / 0.994 | 1.799 / 0.996 | 0.433 / 1.000 | 0.382 / 1.000 |
| 9 | H6,H13,H16 | 1.866 / 0.993 | 1.777 / 0.995 | 0.450 / 1.000 | 0.444 / 1.000 |

Label-based latitude check, arm L (about +y; mean over splits; nominal coverage 0.50 / 0.95)

| family | central 50% coverage | central 95% coverage | mean abs error of the marginal's median (w units) | median abs error |
|---|---|---|---|---|
| vmf | 0.918 | 0.989 | 0.1215 | 0.1319 |
| power_spherical | 0.910 | 0.988 | 0.1392 | 0.1625 |
| tnbbeta | 0.475 | 0.941 | 0.0609 | 0.0461 |
| tnbbeta_init0 | 0.470 | 0.942 | 0.0607 | 0.0461 |

### Held-out NLL of the true direction under the gate baselines (nats, same splits; for the scale of arm L)

| baseline | mean over splits | range |
|---|---|---|
| uniform | 2.531 | 2.531 to 2.531 |
| oracle_ring | 0.532 | 0.506 to 0.572 |
| oracle_cap | 1.860 | 1.852 to 1.879 |
| arm L vmf | 1.867 | 1.854 to 1.885 |
| arm L power_spherical | 1.779 | 1.765 to 1.799 |
| arm L tnbbeta | 0.460 | 0.412 to 0.519 |
| arm L tnbbeta_init0 | 0.450 | 0.382 to 0.569 |

### Confirmatory tests (one-sided paired Wilcoxon, Holm at 0.05)

| arm | comparator | n | mean paired diff (comparator - tnbbeta) | range over splits | splits tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.25 |
|---|---|---|---|---|---|---|---|---|---|---|
| L | power_spherical | 10 | 1.319 | 1.251 to 1.367 | 10/10 | 0.00098 | 0.00098 | 0.01667 | True | True |
| L | vmf | 10 | 1.406 | 1.343 to 1.453 | 10/10 | 0.00098 | 0.00098 | 0.01250 | True | True |
| V | power_spherical | 10 | -0.014 | -0.037 to 0.004 | 1/10 | 0.99707 | 0.99707 | 0.05000 | False | False |
| V | vmf | 10 | 0.016 | -0.008 to 0.032 | 9/10 | 0.00293 | 0.00293 | 0.02500 | True | False |

Decision rule 'TNBBeta uses its ring capacity on this task' (all four tests survive Holm and the 0.25-nat margin holds): Holm all = False, margin all = False, rule met = False


### Descriptive comparisons (no p-values)

| arm | comparison | n | mean metric (A) | mean metric (B) | mean paired diff (A - B) | range of diff | splits A lower |
|---|---|---|---|---|---|---|---|
| V | tnbbeta vs tnbbeta_init0 | 10 | 0.027 | 0.018 | 0.009 | -0.020 to 0.034 | 3/10 |
| V | tnbbeta_init0 vs vmf | 10 | 0.018 | 0.042 | -0.024 | -0.040 to -0.006 | 10/10 |
| V | tnbbeta_init0 vs power_spherical | 10 | 0.018 | 0.013 | 0.005 | -0.010 to 0.023 | 1/10 |
| V | vmf vs power_spherical | 10 | 0.042 | 0.013 | 0.029 | 0.028 to 0.031 | 0/10 |
| L | tnbbeta vs tnbbeta_init0 | 10 | 0.460 | 0.450 | 0.010 | -0.050 to 0.051 | 2/10 |
| L | tnbbeta_init0 vs vmf | 10 | 0.450 | 1.867 | -1.416 | -1.502 to -1.293 | 10/10 |
| L | tnbbeta_init0 vs power_spherical | 10 | 0.450 | 1.779 | -1.329 | -1.417 to -1.201 | 10/10 |
| L | vmf vs power_spherical | 10 | 1.867 | 1.779 | 0.088 | 0.083 to 0.092 | 0/10 |

## Estimator: onset (sensitivity)


### Arm V (VAE; primary metric: exact KL(q || p_model(.|x)), nats)

| family | splits | mean +- sd over splits | median | range | aligned splits (median abs cos >= 0.99) | median abs cos (range) | |u_y| median (range) | p / q / eps / m medians |
|---|---|---|---|---|---|---|---|---|
| vmf | 10 | 0.051 +- 0.004 | 0.051 | 0.044 to 0.055 | 10/10 | 1.000 (0.999-1.000) | 0.737 (0.084-0.984) | - / - / - / - |
| power_spherical | 10 | 0.000 +- 0.000 | 0.000 | -0.000 to 0.000 | 0/10 | 0.689 (0.089-0.954) | 0.648 (0.062-0.962) | - / - / - / - |
| tnbbeta | 10 | 0.003 +- 0.003 | 0.001 | 0.000 to 0.008 | 6/10 | 0.999 (0.333-1.000) | 0.520 (0.106-0.996) | 0.504 / 7.5e-06 / 1.03 / 0.0307 |
| tnbbeta_init0 | 10 | 0.003 +- 0.003 | 0.004 | 0.000 to 0.008 | 8/10 | 1.000 (0.872-1.000) | 0.556 (0.042-0.998) | 0.502 / 0.000127 / 4.03 / 3.03 |

Per split (arm V): metric / median abs cos

| split | test subjects | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|---|
| 0 | H5,H6,H13 | 0.055 / 1.000 | -0.000 / 0.838 | 0.001 / 0.997 | 0.006 / 1.000 |
| 1 | H4,H14,H15 | 0.055 / 1.000 | 0.000 / 0.892 | 0.000 / 0.711 | 0.001 / 0.999 |
| 2 | H9,H10,H19 | 0.044 / 1.000 | 0.000 / 0.806 | 0.001 / 1.000 | 0.005 / 1.000 |
| 3 | H15,H18,H20 | 0.052 / 1.000 | 0.000 / 0.954 | 0.006 / 1.000 | 0.005 / 1.000 |
| 4 | H3,H10,H12 | 0.050 / 0.999 | 0.000 / 0.664 | 0.008 / 1.000 | 0.004 / 1.000 |
| 5 | H6,H10,H20 | 0.051 / 0.999 | 0.000 / 0.089 | 0.000 / 0.333 | 0.001 / 1.000 |
| 6 | H5,H13,H19 | 0.054 / 0.999 | 0.000 / 0.455 | 0.000 / 0.765 | 0.004 / 1.000 |
| 7 | H6,H9,H17 | 0.044 / 1.000 | -0.000 / 0.321 | 0.000 / 0.960 | 0.008 / 1.000 |
| 8 | H3,H12,H19 | 0.054 / 1.000 | 0.000 / 0.168 | 0.004 / 1.000 | 0.000 / 0.937 |
| 9 | H6,H13,H16 | 0.050 / 1.000 | 0.000 / 0.713 | 0.005 / 1.000 | 0.000 / 0.872 |

Per split (arm V): |u_y| = |cos(u, +/- y)|

| split | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.817 | 0.861 | 0.600 | 0.646 |
| 1 | 0.730 | 0.670 | 0.390 | 0.445 |
| 2 | 0.104 | 0.062 | 0.164 | 0.122 |
| 3 | 0.984 | 0.962 | 0.521 | 0.532 |
| 4 | 0.744 | 0.494 | 0.996 | 0.998 |
| 5 | 0.556 | 0.493 | 0.106 | 0.042 |
| 6 | 0.790 | 0.767 | 0.174 | 0.132 |
| 7 | 0.084 | 0.297 | 0.930 | 0.944 |
| 8 | 0.745 | 0.720 | 0.519 | 0.580 |
| 9 | 0.603 | 0.626 | 0.673 | 0.710 |

Label-based latitude check, arm V (about the signed decoder axis u; mean over splits; nominal coverage 0.50 / 0.95)

| family | central 50% coverage | central 95% coverage | mean abs error of the marginal's median (w units) | median abs error |
|---|---|---|---|---|
| vmf | 0.383 | 0.878 | 0.6325 | 0.5771 |
| power_spherical | 0.500 | 0.951 | 0.5002 | 0.4755 |
| tnbbeta | 0.386 | 0.797 | 0.5336 | 0.5276 |
| tnbbeta_init0 | 0.290 | 0.638 | 0.6128 | 0.5779 |

### Arm L (supervised; primary metric: held-out -log q(z_true|x), nats)

| family | splits | mean +- sd over splits | median | range | aligned splits (median abs cos >= 0.99) | median abs cos (range) | p / q / eps / m medians |
|---|---|---|---|---|---|---|---|
| vmf | 10 | 2.083 +- 0.027 | 2.077 | 2.032 to 2.123 | 3/10 | 0.985 (0.976-0.992) | - / - / - / - |
| power_spherical | 10 | 2.008 +- 0.029 | 2.004 | 1.964 to 2.051 | 4/10 | 0.989 (0.984-0.996) | - / - / - / - |
| tnbbeta | 10 | 1.919 +- 0.066 | 1.911 | 1.834 to 2.008 | 10/10 | 0.994 (0.993-0.996) | 0.724 / 0.0393 / 2.43 / 1.43 |
| tnbbeta_init0 | 10 | 1.917 +- 0.067 | 1.909 | 1.825 to 2.016 | 10/10 | 0.994 (0.993-0.996) | 0.495 / 0.0291 / 2.41 / 1.41 |

Per split (arm L): metric / median abs cos

| split | test subjects | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|---|
| 0 | H5,H6,H13 | 2.075 / 0.984 | 1.999 / 0.989 | 1.884 / 0.993 | 1.888 / 0.995 |
| 1 | H4,H14,H15 | 2.101 / 0.992 | 2.051 / 0.996 | 2.008 / 0.996 | 2.016 / 0.996 |
| 2 | H9,H10,H19 | 2.061 / 0.990 | 1.975 / 0.992 | 1.881 / 0.994 | 1.871 / 0.993 |
| 3 | H15,H18,H20 | 2.079 / 0.986 | 2.009 / 0.988 | 1.947 / 0.994 | 1.931 / 0.995 |
| 4 | H3,H10,H12 | 2.123 / 0.979 | 2.039 / 0.988 | 1.991 / 0.993 | 1.988 / 0.993 |
| 5 | H6,H10,H20 | 2.075 / 0.976 | 1.993 / 0.984 | 1.867 / 0.996 | 1.875 / 0.996 |
| 6 | H5,H13,H19 | 2.090 / 0.990 | 2.012 / 0.991 | 1.937 / 0.994 | 1.940 / 0.993 |
| 7 | H6,H9,H17 | 2.032 / 0.985 | 1.964 / 0.990 | 1.834 / 0.993 | 1.825 / 0.993 |
| 8 | H3,H12,H19 | 2.121 / 0.983 | 2.042 / 0.989 | 1.997 / 0.994 | 1.995 / 0.993 |
| 9 | H6,H13,H16 | 2.076 / 0.981 | 2.000 / 0.988 | 1.839 / 0.994 | 1.842 / 0.994 |

Label-based latitude check, arm L (about +y; mean over splits; nominal coverage 0.50 / 0.95)

| family | central 50% coverage | central 95% coverage | mean abs error of the marginal's median (w units) | median abs error |
|---|---|---|---|---|
| vmf | 0.658 | 0.990 | 0.2673 | 0.2351 |
| power_spherical | 0.629 | 0.983 | 0.2623 | 0.2256 |
| tnbbeta | 0.511 | 0.948 | 0.2442 | 0.2082 |
| tnbbeta_init0 | 0.511 | 0.949 | 0.2440 | 0.2077 |

### Held-out NLL of the true direction under the gate baselines (nats, same splits; for the scale of arm L)

| baseline | mean over splits | range |
|---|---|---|
| uniform | 2.531 | 2.531 to 2.531 |
| oracle_ring | 1.853 | 1.735 to 1.975 |
| oracle_cap | 2.074 | 2.029 to 2.113 |
| arm L vmf | 2.083 | 2.032 to 2.123 |
| arm L power_spherical | 2.008 | 1.964 to 2.051 |
| arm L tnbbeta | 1.919 | 1.834 to 2.008 |
| arm L tnbbeta_init0 | 1.917 | 1.825 to 2.016 |

### Confirmatory tests (one-sided paired Wilcoxon, Holm at 0.05)

| arm | comparator | n | mean paired diff (comparator - tnbbeta) | range over splits | splits tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.25 |
|---|---|---|---|---|---|---|---|---|---|---|
| L | power_spherical | 10 | 0.090 | 0.042 to 0.161 | 10/10 | 0.00098 | 0.00098 | 0.02500 | True | False |
| L | vmf | 10 | 0.165 | 0.093 to 0.237 | 10/10 | 0.00098 | 0.00098 | 0.01667 | True | False |
| V | power_spherical | 10 | -0.003 | -0.008 to -0.000 | 0/10 | 1.00000 | 1.00000 | 0.05000 | False | False |
| V | vmf | 10 | 0.048 | 0.042 to 0.055 | 10/10 | 0.00098 | 0.00098 | 0.01250 | True | False |

Decision rule 'TNBBeta uses its ring capacity on this task' (all four tests survive Holm and the 0.25-nat margin holds): Holm all = False, margin all = False, rule met = False


### Descriptive comparisons (no p-values)

| arm | comparison | n | mean metric (A) | mean metric (B) | mean paired diff (A - B) | range of diff | splits A lower |
|---|---|---|---|---|---|---|---|
| V | tnbbeta vs tnbbeta_init0 | 10 | 0.003 | 0.003 | -0.001 | -0.008 to 0.005 | 6/10 |
| V | tnbbeta_init0 vs vmf | 10 | 0.003 | 0.051 | -0.048 | -0.054 to -0.036 | 10/10 |
| V | tnbbeta_init0 vs power_spherical | 10 | 0.003 | 0.000 | 0.003 | 0.000 to 0.008 | 0/10 |
| V | vmf vs power_spherical | 10 | 0.051 | 0.000 | 0.051 | 0.044 to 0.055 | 0/10 |
| L | tnbbeta vs tnbbeta_init0 | 10 | 1.919 | 1.917 | 0.001 | -0.008 to 0.016 | 5/10 |
| L | tnbbeta_init0 vs vmf | 10 | 1.917 | 2.083 | -0.166 | -0.234 to -0.085 | 10/10 |
| L | tnbbeta_init0 vs power_spherical | 10 | 1.917 | 2.008 | -0.091 | -0.158 to -0.034 | 10/10 |
| L | vmf vs power_spherical | 10 | 2.083 | 2.008 | 0.075 | 0.050 to 0.086 | 0/10 |
