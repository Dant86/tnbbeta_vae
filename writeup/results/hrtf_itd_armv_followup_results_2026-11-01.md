# ITD-only HRTF, arm V follow-up (collapse-controlled): results

Date: 2026-11-01. Code: base commit `10d30b9` plus an uncommitted working tree (new files
`apps/synthetic/hrtf_itd_armv_followup.py`, `apps/synthetic/hrtf_itd_armv_followup_plot.py`,
`scripts/slurm/hrtf_itd_armv_followup.sbatch`, `tests/apps/test_hrtf_itd_armv_followup.py`; the registered
`apps/synthetic/hrtf_itd.py` and `data/hrtf.py` are imported, not modified; nothing under
`src/tnbbeta_vae/distributions/`, `models/` or `losses/` was touched). Design, tests and decision rules:
`hrtf_itd_armv_followup_preregistration_2026-11-01.md` (frozen; its "Deviations" section lists every choice
made after freezing, including one post-hoc addition, repeated below). Registered study this follows:
`hrtf_itd_ring_results_2026-11-01.md`.

Commands (SLURM job 2291217, 10 tasks x 8 workers, 80 cells, 36 to 70 min per task):

    sbatch --array=0-9 scripts/slurm/hrtf_itd_armv_followup.sbatch
    uv run python -m apps.synthetic.hrtf_itd_armv_followup summarize --out-dir hrtf_runs/armv_followup
    uv run python -m apps.synthetic.hrtf_itd_armv_followup_plot --out-dir hrtf_runs/armv_followup --fig-dir writeup/results/figures --seed 0

(each cell is `python -m apps.synthetic.hrtf_itd_armv_followup train --decoder D --family F --seed S --out-dir hrtf_runs/armv_followup`;
runs, JSON, per-point npz and checkpoints under `hrtf_runs/armv_followup/<decoder>/<family>/s<split>.*`, gitignored.)
Tests: 11 new tests pass; the full suite was run as a SLURM CPU job (991 passed, 1 skipped, job 2292032, after the last test
was added); `ruff format`, `ruff check`, `pyright` clean on every new file.

## Headline

Provenance caveat (from the pre-registration): this is a second look at the same data, designed after the first
result, and the 10 splits share subjects, so p-values are optimistic; effects and ranges over splits are given.

**The registered rule "TNBBeta improves a collapse-controlled ITD VAE" is met**: all four one-sided paired Wilcoxon
tests have p = 0.00098 (the smallest attainable, 10/10 splits with TNBBeta higher), all survive Holm at 0.05, and
every mean paired difference exceeds the 0.02-nat margin.

| decoder | comparator | mean paired diff of the IWAE bound (tnbbeta - comparator, nats) | range over splits | splits tnbbeta higher | p | Holm threshold | Holm | margin 0.02 |
|---|---|---|---|---|---|---|---|---|
| learned_axis | vmf | 0.0557 | 0.0346 to 0.0730 | 10/10 | 0.00098 | 0.0125 | reject | yes |
| learned_axis | power_spherical | 0.0479 | 0.0260 to 0.0624 | 10/10 | 0.00098 | 0.0167 | reject | yes |
| fixed_axis | vmf | 0.0439 | 0.0076 to 0.0662 | 10/10 | 0.00098 | 0.0250 | reject | yes |
| fixed_axis | power_spherical | 0.0362 | 0.0002 to 0.0582 | 10/10 | 0.00098 | 0.0500 | reject | yes |

scipy and sign-pattern-enumerated p-values agree. Smallest single-split gap: 0.0002 nats (fixed_axis vs
PowerSpherical), so the margin is a statement about means, not every split.

## What this does and does not show (read before the headline)

**Collapse, the trainability side.** Despite the 100-epoch warm-up and the initial sigma 0.1, **every vMF and every
PowerSpherical run collapsed (20/20, sigma 0.75 to 0.83, in both decoders)**, exactly as in the registered arm V.
TNBBeta did not collapse with the learned axis (0/10; `tnbbeta_init0` 0/10) and collapsed in 1/10 (split 9, sigma 0.78)
with the fixed axis (`tnbbeta_init0`: 2/10, splits 0 and 6, sigma 0.99). Collapsed runs have bound about -1.41 to -1.42, the
value of a decoder that ignores z; uncollapsed TNBBeta reaches about -1.36.

**Capacity, the restricted comparison.** The restricted comparison (pairs where both runs have sigma <= 0.5) is
**empty: n = 0 in all four tests**, because there is no uncollapsed vMF or PowerSpherical run. It therefore cannot be
used to say whether TNBBeta's bound advantage survives without collapse. The confirmatory result is a statement
that TNBBeta trains where, in this setup, vMF/PS do not (they do not use the latent at all, and the whole 0.036 to 0.056 nat
gap is the gap between a collapsed decoder and a working one); it is **not** evidence that TNBBeta's ring or shape
capacity gives a better bound than a vMF/PS that has not collapsed. A comparison with matched, uncollapsed caps (for
example vMF/PS with the sigma pinned small or with a free-bits floor) was not registered and is not claimed. Within the
fixed_axis decoder, the two TNBBeta runs that collapsed are the ones closest to the cap baselines (split 9 bound -1.416).

**Shape of q (descriptive).** The registered-style latitude check (about the sign-oriented +y) is poor for many runs
(TNBBeta fixed_axis 50% coverage 0.36 mean, range 0.04 to 0.67; 95%: 0.67), but this is largely an orientation artefact: the
sign of the learned g mirrors the latent latitude (median |cos(mu, y)| is 1.000 for all families: the direction is on the
interaural axis, and corr(median latitude of q, true y . z) is +/-0.91 to +/-0.97 in the eight runs inspected (TNBBeta and vMF, splits 0, 1, 5, 9)). The post-hoc,
label-free orientation (not registered; see Deviations item 8) gives, for fixed_axis, TNBBeta 50%/95% coverage
0.596/0.928 (mean abs error of the median 0.081), tnbbeta_init0 0.512/0.897, and vMF and PowerSpherical 0.82/0.99
(errors 0.18 and 0.12). Over-coverage for the collapsed caps (0.82 against a nominal 0.50) is the signature of a
near-uniform, too-wide posterior; TNBBeta's 0.60 is also wider than nominal but much closer. This shows TNBBeta's q
carries the lateral information with the best-calibrated latitude among the four; because vMF/PS are collapsed it does
not show that a ring is needed. In learned_axis the axis u is only sometimes the interaural axis (|cos(u, +/- y)| from 0.03
to 1.00 per run), so its latitude check is not about y. Figures (split 0): see below.

**Other caveats.** `tnbbeta` against `tnbbeta_init0` is indistinguishable (mean difference 0.002 and 0.0003 nats, wins 5/10
and 4/10). learned_axis against fixed_axis: no difference for vMF/PS (about 1e-4, both collapsed), TNBBeta 0.012 higher
with the learned axis (7/10 splits; the fixed-axis collapse in split 9 and larger sigma drive part of it). The 10 splits
share subjects (each held out in about 1/6 of the splits), so effect sizes with ranges are the safer reading.

Figures (split 0; rows vMF/PowerSpherical/TNBBeta, columns ITD -450/0/450 us; q rotated so the signed decoder axis is +y,
arrow = signed axis, faint = far side):
`figures/hrtf_itd_armv_followup_learned_axis_s0_2026-11-01.png`, `figures/hrtf_itd_armv_followup_fixed_axis_s0_2026-11-01.png`.

## Deviations (full text in the pre-registration note)

1 signed axis for the latitude check (training-label correlation of the axis, as in the registered study); 2 diverged runs
count as -inf (none occurred: 0 diverged runs); 3 bound proposal/prior and the test negative ELBO definition; 4 Wilcoxon
details; 5 seeds/pairing as registered; 6 descriptive diagnostics as registered; 7 compute layout (10 tasks x 8 workers);
8 **post-hoc, unregistered**: the label-free ITD-oriented latitude table, added after seeing that the sign convention of
item 1 leaves the latent mirrored; it enters no test or verdict.

## Full output of `summarize`

Note: the 10 split replicates overlap in subjects (each of 18 subjects is held out in about 1/6 of the splits and trains in the rest), so the Wilcoxon p-values are optimistic; the range over splits accompanies every effect.


## Decoder learned_axis: test IWAE bound (K = 64, nats; higher better)

| family | splits | collapsed (sigma > 0.5) | bound mean +- sd | median | range | negative ELBO mean | KL(q || p_model) mean | sigma median (range) |
|---|---|---|---|---|---|---|---|---|
| vmf | 10 | 10/10 | -1.4154 +- 0.0198 | -1.4145 | -1.4554 to -1.3817 | 1.4575 | 0.0424 | 0.827 (0.823-0.830) |
| power_spherical | 10 | 10/10 | -1.4076 +- 0.0190 | -1.4072 | -1.4449 to -1.3742 | 1.4200 | 0.0127 | 0.758 (0.751-0.760) |
| tnbbeta | 10 | 0/10 | -1.3597 +- 0.0196 | -1.3571 | -1.3946 to -1.3215 | 1.3927 | 0.0362 | 0.169 (0.101-0.255) |
| tnbbeta_init0 | 10 | 0/10 | -1.3618 +- 0.0280 | -1.3583 | -1.4085 to -1.3207 | 1.3907 | 0.0323 | 0.117 (0.086-0.283) |

## Decoder fixed_axis: test IWAE bound (K = 64, nats; higher better)

| family | splits | collapsed (sigma > 0.5) | bound mean +- sd | median | range | negative ELBO mean | KL(q || p_model) mean | sigma median (range) |
|---|---|---|---|---|---|---|---|---|
| vmf | 10 | 10/10 | -1.4153 +- 0.0197 | -1.4144 | -1.4547 to -1.3818 | 1.4574 | 0.0423 | 0.828 (0.824-0.830) |
| power_spherical | 10 | 10/10 | -1.4076 +- 0.0190 | -1.4078 | -1.4450 to -1.3754 | 1.4200 | 0.0130 | 0.757 (0.750-0.774) |
| tnbbeta | 10 | 1/10 | -1.3714 +- 0.0257 | -1.3658 | -1.4160 to -1.3287 | 1.4068 | 0.0386 | 0.175 (0.091-0.776) |
| tnbbeta_init0 | 10 | 2/10 | -1.3717 +- 0.0366 | -1.3688 | -1.4177 to -1.3211 | 1.4061 | 0.0340 | 0.141 (0.078-0.997) |

Per split, decoder learned_axis: IWAE bound / negative ELBO / KL(q || p_model) / sigma (c = collapsed)

| split | test subjects | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|---|
| 0 | H5,H6,H13 | -1.4078 / 1.4511 / 0.0429 / 0.827 c | -1.4003 / 1.4130 / 0.0132 / 0.754 c | -1.3496 / 1.3757 / 0.0277 / 0.182 | -1.3437 / 1.3652 / 0.0224 / 0.110 |
| 1 | H4,H14,H15 | -1.3817 / 1.4245 / 0.0430 / 0.829 c | -1.3742 / 1.3855 / 0.0124 / 0.753 c | -1.3215 / 1.3356 / 0.0160 / 0.123 | -1.3207 / 1.3378 / 0.0187 / 0.110 |
| 2 | H9,H10,H19 | -1.4123 / 1.4550 / 0.0430 / 0.829 c | -1.4052 / 1.4155 / 0.0125 / 0.759 c | -1.3572 / 1.3792 / 0.0239 / 0.219 | -1.3429 / 1.3602 / 0.0187 / 0.109 |
| 3 | H15,H18,H20 | -1.4292 / 1.4704 / 0.0428 / 0.827 c | -1.4206 / 1.4346 / 0.0132 / 0.751 c | -1.3946 / 1.4411 / 0.0483 / 0.101 | -1.4085 / 1.4458 / 0.0455 / 0.086 |
| 4 | H3,H10,H12 | -1.4203 / 1.4623 / 0.0423 / 0.825 c | -1.4121 / 1.4252 / 0.0135 / 0.758 c | -1.3657 / 1.3933 / 0.0332 / 0.210 | -1.3760 / 1.4220 / 0.0509 / 0.283 |
| 5 | H6,H10,H20 | -1.3942 / 1.4360 / 0.0420 / 0.827 c | -1.3875 / 1.3994 / 0.0117 / 0.760 c | -1.3557 / 1.4497 / 0.1026 / 0.206 | -1.3353 / 1.3602 / 0.0290 / 0.163 |
| 6 | H5,H13,H19 | -1.4135 / 1.4548 / 0.0416 / 0.823 c | -1.4054 / 1.4180 / 0.0120 / 0.758 c | -1.3628 / 1.4108 / 0.0530 / 0.255 | -1.3468 / 1.3635 / 0.0185 / 0.116 |
| 7 | H6,H9,H17 | -1.4156 / 1.4586 / 0.0426 / 0.830 c | -1.4091 / 1.4208 / 0.0125 / 0.760 c | -1.3502 / 1.3713 / 0.0227 / 0.156 | -1.3910 / 1.4333 / 0.0465 / 0.186 |
| 8 | H3,H12,H19 | -1.4554 / 1.4976 / 0.0425 / 0.830 c | -1.4449 / 1.4598 / 0.0142 / 0.759 c | -1.3825 / 1.3986 / 0.0177 / 0.119 | -1.3829 / 1.3992 / 0.0179 / 0.119 |
| 9 | H6,H13,H16 | -1.4237 / 1.4650 / 0.0416 / 0.828 c | -1.4167 / 1.4279 / 0.0117 / 0.751 c | -1.3571 / 1.3718 / 0.0172 / 0.113 | -1.3699 / 1.4198 / 0.0549 / 0.208 |

Per split, decoder fixed_axis: IWAE bound / negative ELBO / KL(q || p_model) / sigma (c = collapsed)

| split | test subjects | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|---|
| 0 | H5,H6,H13 | -1.4077 / 1.4512 / 0.0429 / 0.826 c | -1.3995 / 1.4120 / 0.0136 / 0.753 c | -1.3776 / 1.4240 / 0.0507 / 0.288 | -1.4145 / 1.4168 / 0.0024 / 0.994 c |
| 1 | H4,H14,H15 | -1.3818 / 1.4246 / 0.0431 / 0.828 c | -1.3754 / 1.3877 / 0.0123 / 0.755 c | -1.3665 / 1.4253 / 0.0647 / 0.382 | -1.3211 / 1.3358 / 0.0163 / 0.106 |
| 2 | H9,H10,H19 | -1.4120 / 1.4545 / 0.0428 / 0.829 c | -1.4044 / 1.4160 / 0.0120 / 0.758 c | -1.3463 / 1.3633 / 0.0192 / 0.147 | -1.3400 / 1.3533 / 0.0150 / 0.102 |
| 3 | H15,H18,H20 | -1.4293 / 1.4704 / 0.0427 / 0.827 c | -1.4216 / 1.4348 / 0.0144 / 0.756 c | -1.4019 / 1.4314 / 0.0314 / 0.091 | -1.4126 / 1.5694 / 0.1341 / 0.078 |
| 4 | H3,H10,H12 | -1.4204 / 1.4621 / 0.0421 / 0.824 c | -1.4118 / 1.4244 / 0.0134 / 0.755 c | -1.3595 / 1.3856 / 0.0303 / 0.153 | -1.3543 / 1.3809 / 0.0286 / 0.161 |
| 5 | H6,H10,H20 | -1.3943 / 1.4360 / 0.0420 / 0.827 c | -1.3869 / 1.3985 / 0.0118 / 0.759 c | -1.3287 / 1.3484 / 0.0208 / 0.110 | -1.3305 / 1.3448 / 0.0164 / 0.101 |
| 6 | H5,H13,H19 | -1.4134 / 1.4546 / 0.0415 / 0.824 c | -1.4050 / 1.4174 / 0.0134 / 0.759 c | -1.3651 / 1.4084 / 0.0447 / 0.300 | -1.4177 / 1.4185 / 0.0008 / 0.997 c |
| 7 | H6,H9,H17 | -1.4155 / 1.4582 / 0.0422 / 0.829 c | -1.4107 / 1.4236 / 0.0121 / 0.774 c | -1.3639 / 1.4451 / 0.0905 / 0.188 | -1.3527 / 1.3868 / 0.0361 / 0.155 |
| 8 | H3,H12,H19 | -1.4547 / 1.4970 / 0.0425 / 0.830 c | -1.4450 / 1.4581 / 0.0146 / 0.762 c | -1.3886 / 1.4111 / 0.0249 / 0.163 | -1.3832 / 1.4021 / 0.0207 / 0.126 |
| 9 | H6,H13,H16 | -1.4236 / 1.4650 / 0.0416 / 0.828 c | -1.4162 / 1.4272 / 0.0120 / 0.750 c | -1.4160 / 1.4250 / 0.0090 / 0.776 c | -1.3905 / 1.4529 / 0.0695 / 0.310 |

## Confirmatory tests (one-sided paired Wilcoxon on the IWAE bound, Holm)

| decoder | comparator | n | mean paired diff (tnbbeta - comparator) | range over splits | splits tnbbeta higher | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed_axis | power_spherical | 10 | 0.0362 | 0.0002 to 0.0582 | 10/10 | 0.00098 | 0.00098 | 0.05000 | True | True |
| fixed_axis | vmf | 10 | 0.0439 | 0.0076 to 0.0662 | 10/10 | 0.00098 | 0.00098 | 0.02500 | True | True |
| learned_axis | power_spherical | 10 | 0.0479 | 0.0260 to 0.0624 | 10/10 | 0.00098 | 0.00098 | 0.01667 | True | True |
| learned_axis | vmf | 10 | 0.0557 | 0.0346 to 0.0730 | 10/10 | 0.00098 | 0.00098 | 0.01250 | True | True |

Decision rule 'TNBBeta improves a collapse-controlled ITD VAE' (all four tests survive Holm and the 0.02-nat margin holds): Holm all = True, margin all = True, rule met = True


## Collapse accounting (sigma > 0.5 at the end of training)

| decoder | family | collapsed / runs | collapsed splits | diverged runs |
|---|---|---|---|---|
| learned_axis | vmf | 10/10 | 0,1,2,3,4,5,6,7,8,9 | 0 |
| learned_axis | power_spherical | 10/10 | 0,1,2,3,4,5,6,7,8,9 | 0 |
| learned_axis | tnbbeta | 0/10 | - | 0 |
| learned_axis | tnbbeta_init0 | 0/10 | - | 0 |
| fixed_axis | vmf | 10/10 | 0,1,2,3,4,5,6,7,8,9 | 0 |
| fixed_axis | power_spherical | 10/10 | 0,1,2,3,4,5,6,7,8,9 | 0 |
| fixed_axis | tnbbeta | 1/10 | 9 | 0 |
| fixed_axis | tnbbeta_init0 | 2/10 | 0,6 | 0 |

## Restricted comparison (both runs uncollapsed; outside the Holm family)

| decoder | comparator | n | mean paired diff (tnbbeta - comparator) | range over splits | splits tnbbeta higher | p (scipy) | p (enumerated) | splits used |
|---|---|---|---|---|---|---|---|---|
| (no pairs) |

## Descriptive comparisons (bound, no p-values)

| comparison | group | n | mean bound (A) | mean bound (B) | mean paired diff (A - B) | range of diff | splits A higher |
|---|---|---|---|---|---|---|---|
| tnbbeta vs tnbbeta_init0 | learned_axis | 10 | -1.3597 | -1.3618 | 0.0021 | -0.0204 to 0.0408 | 5/10 |
| tnbbeta vs tnbbeta_init0 | fixed_axis | 10 | -1.3714 | -1.3717 | 0.0003 | -0.0455 to 0.0526 | 4/10 |
| learned_axis vs fixed_axis | vmf | 10 | -1.4154 | -1.4153 | -0.0001 | -0.0007 to 0.0001 | 4/10 |
| learned_axis vs fixed_axis | power_spherical | 10 | -1.4076 | -1.4076 | 0.0001 | -0.0008 to 0.0016 | 4/10 |
| learned_axis vs fixed_axis | tnbbeta | 10 | -1.3597 | -1.3714 | 0.0117 | -0.0271 to 0.0589 | 7/10 |
| learned_axis vs fixed_axis | tnbbeta_init0 | 10 | -1.3618 | -1.3717 | 0.0100 | -0.0383 to 0.0710 | 6/10 |

## Label-based latitude check, decoder fixed_axis (q's latitude marginal about the sign-oriented +y against the true y . z; mean over splits; nominal coverage 0.50 / 0.95)

| family | central 50% coverage (range) | central 95% coverage (range) | median abs error of the marginal's median (area-weighted mean) | median |cos(mu, y)| over splits (range) |
|---|---|---|---|---|
| vmf | 0.232 (0.080-0.816) | 0.808 (0.756-0.990) | 0.8143 | 1.000 (1.000-1.000) |
| power_spherical | 0.341 (0.215-0.821) | 0.658 (0.564-0.991) | 0.7595 | 1.000 (1.000-1.000) |
| tnbbeta | 0.360 (0.036-0.668) | 0.671 (0.068-0.968) | 0.4354 | 1.000 (1.000-1.000) |
| tnbbeta_init0 | 0.421 (0.034-0.628) | 0.747 (0.065-0.964) | 0.3299 | 1.000 (0.999-1.000) |

Per split, fixed_axis: coverage 50% / 95% / median abs error / |cos(mu, y)|

| split | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.09 / 0.76 / 0.979 / 1.000 | 0.21 / 0.57 / 0.931 / 1.000 | 0.49 / 0.97 / 0.113 / 1.000 | 0.54 / 0.96 / 0.429 / 1.000 |
| 1 | 0.09 / 0.77 / 0.963 / 1.000 | 0.22 / 0.58 / 0.917 / 1.000 | 0.08 / 0.25 / 0.982 / 1.000 | 0.43 / 0.84 / 0.063 / 1.000 |
| 2 | 0.08 / 0.77 / 0.973 / 1.000 | 0.22 / 0.58 / 0.919 / 1.000 | 0.58 / 0.93 / 0.061 / 1.000 | 0.44 / 0.87 / 0.061 / 1.000 |
| 3 | 0.09 / 0.76 / 0.968 / 1.000 | 0.24 / 0.58 / 0.905 / 1.000 | 0.43 / 0.82 / 0.063 / 1.000 | 0.39 / 0.77 / 0.063 / 1.000 |
| 4 | 0.09 / 0.76 / 0.972 / 1.000 | 0.22 / 0.57 / 0.925 / 1.000 | 0.58 / 0.92 / 0.064 / 1.000 | 0.63 / 0.93 / 0.062 / 1.000 |
| 5 | 0.82 / 0.99 / 0.180 / 1.000 | 0.82 / 0.99 / 0.124 / 1.000 | 0.04 / 0.07 / 0.999 / 1.000 | 0.03 / 0.06 / 0.996 / 1.000 |
| 6 | 0.81 / 0.99 / 0.186 / 1.000 | 0.82 / 0.99 / 0.125 / 1.000 | 0.07 / 0.20 / 1.022 / 1.000 | 0.52 / 0.96 / 0.482 / 0.999 |
| 7 | 0.09 / 0.77 / 0.967 / 1.000 | 0.23 / 0.60 / 0.895 / 1.000 | 0.46 / 0.95 / 0.086 / 1.000 | 0.61 / 0.94 / 0.062 / 1.000 |
| 8 | 0.08 / 0.76 / 0.978 / 1.000 | 0.22 / 0.56 / 0.926 / 1.000 | 0.67 / 0.94 / 0.060 / 1.000 | 0.56 / 0.92 / 0.061 / 1.000 |
| 9 | 0.09 / 0.76 / 0.977 / 1.000 | 0.22 / 0.57 / 0.928 / 1.000 | 0.20 / 0.66 / 0.907 / 1.000 | 0.06 / 0.21 / 1.020 / 1.000 |

## Decoder learned_axis: |cos(u, +/- y)| and |cos(mu, u)|, per split

| split | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.836 / 1.000 | 0.811 / 1.000 | 0.699 / 1.000 | 0.646 / 1.000 |
| 1 | 0.703 / 1.000 | 0.707 / 1.000 | 0.290 / 1.000 | 0.412 / 1.000 |
| 2 | 0.076 / 1.000 | 0.139 / 0.999 | 0.137 / 1.000 | 0.123 / 1.000 |
| 3 | 0.979 / 1.000 | 0.980 / 1.000 | 0.557 / 1.000 | 0.549 / 1.000 |
| 4 | 0.735 / 0.999 | 0.738 / 1.000 | 1.000 / 1.000 | 0.983 / 1.000 |
| 5 | 0.550 / 0.999 | 0.536 / 1.000 | 0.028 / 1.000 | 0.057 / 1.000 |
| 6 | 0.755 / 0.999 | 0.791 / 1.000 | 0.173 / 1.000 | 0.088 / 1.000 |
| 7 | 0.043 / 1.000 | 0.068 / 1.000 | 0.953 / 1.000 | 0.936 / 1.000 |
| 8 | 0.737 / 1.000 | 0.713 / 1.000 | 0.548 / 1.000 | 0.497 / 1.000 |
| 9 | 0.607 / 1.000 | 0.622 / 1.000 | 0.690 / 1.000 | 0.619 / 1.000 |

Learned-axis latitude check (about the signed u; mean over splits): family: coverage 50% / 95%

- vmf: 0.368 / 0.862
- power_spherical: 0.357 / 0.752
- tnbbeta: 0.109 / 0.216
- tnbbeta_init0: 0.083 / 0.199

## POST-HOC (not registered): latitude check with q oriented by the ITD (label-free; see the Deviations of the pre-registration)

| decoder | family | latent mirrored (splits) | central 50% coverage (mean, range) | central 95% coverage (mean, range) | mean abs error of the median |
|---|---|---|---|---|---|
| learned_axis | vmf | 6/10 | 0.518 (0.342-0.811) | 0.971 (0.922-0.990) | 0.3985 |
| learned_axis | power_spherical | 6/10 | 0.477 (0.305-0.820) | 0.950 (0.825-0.987) | 0.3852 |
| learned_axis | tnbbeta | 7/10 | 0.129 (0.039-0.685) | 0.279 (0.119-0.946) | 0.4614 |
| learned_axis | tnbbeta_init0 | 6/10 | 0.102 (0.033-0.491) | 0.260 (0.105-0.962) | 0.4762 |
| fixed_axis | vmf | 8/10 | 0.818 (0.813-0.827) | 0.990 (0.988-0.991) | 0.1822 |
| fixed_axis | power_spherical | 8/10 | 0.822 (0.812-0.828) | 0.991 (0.989-0.992) | 0.1221 |
| fixed_axis | tnbbeta | 4/10 | 0.596 (0.431-0.820) | 0.928 (0.821-0.973) | 0.0811 |
| fixed_axis | tnbbeta_init0 | 2/10 | 0.512 (0.389-0.628) | 0.897 (0.766-0.970) | 0.1460 |
