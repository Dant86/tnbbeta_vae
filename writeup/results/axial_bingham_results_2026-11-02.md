# Axial Bingham study: results

Date: 2026-11-02/03. Pre-registration: `axial_bingham_preregistration_2026-11-02.md` (frozen; Deviations 1-2 and
Note 3 appended there). Base: commit `6b1880f` plus the uncommitted tree (this study's new files are
`src/tnbbeta_vae/data/axial_bingham.py`, `src/tnbbeta_vae/distributions/sign_symmetrised.py`,
`apps/synthetic/axial_bingham{,_checks,_plot}.py`, `scripts/slurm/axial_bingham{,_aux}.sbatch`, and tests).
Figures: `figures/axial_bingham_gap_vs_sigma_2026-11-02.png`, `figures/axial_bingham_sphere_d3_2026-11-02.png`.

## Commands

* Pre-freeze checks (CPU, SLURM jobs 2298077 threshold, 2298078 normaliser, 2298080 anisotropy, 2298190
  representability): `AUX=<name> sbatch scripts/slurm/axial_bingham_aux.sbatch` ->
  `python -m apps.synthetic.axial_bingham_checks <name> --out-dir ring_recovery/axial_bingham/checks`.
  Post-hoc free-axis representability: job 2299061 (`AUX=representability_general`).
* Training grid (CPU, 22-task array, 8 worker processes per task): `sbatch --array=0-21
  scripts/slurm/axial_bingham.sbatch` = job **2298694** (an earlier launch, job 2298324, was cancelled after the
  non-finite-minibatch crash of Deviation 1 and its two finished runs deleted). Each run:
  `python -m apps.synthetic.axial_bingham train --dim D --sigma S --decoder DEC --arm ARM --seed K --device cpu
  --out-dir ring_recovery/axial_bingham`.
* Analysis: `AUX=summarize` (job 2301910) = `python -m apps.synthetic.axial_bingham summarize --out-dir
  ring_recovery/axial_bingham` (full markdown output appended below, unedited). Figures: `AUX=plot` (job 2301997).
  Debug jobs that were cancelled before running: 2298607-2298610. Tests: full suite `AUX=pytest` (job 2301912):
  1154 passed, 1 skipped.

## Headline

1. **Confirmatory family: 34 of 48 cells confirmed (Holm and 0.05-nat margin) in the registered direction (TNBBeta
   better than vMF / PS); 4 further cells go the other way by at least the margin; none of the 48 secondary tests
   against the sign-symmetrised baselines is confirmed, 15 go the other way by at least the margin.**
   Every scipy p equals its sign-pattern enumeration (no WARNING lines). Smallest attainable p (2^-10 = 0.00098)
   is reached in all rejected cells.
2. **d = 2: TNBBeta beats vMF and PS in all 24 cells except two (PS, sigma = 1, learned decoders: 0.005-0.007 nats,
   Holm-significant but below the margin).** Mean ELBO gaps, fixed decoder: vMF 0.84 / 0.71 / 0.64 / 0.38 and PS
   0.81 / 0.71 / 0.61 / 0.28 against TNBBeta 0.087 / 0.033 / 0.011 / 0.006 at sigma 0.1 / 0.25 / 0.5 / 1.0. The ln 2
   prediction P1 holds exactly (sign-unaware caps sit at 0.71-0.84 at sigma <= 0.5). BUT TNBBeta is statistically
   indistinguishable from the sign-symmetrised vMF/PS (all 24 secondary d=2 tests: no confirmation, differences
   <= 0.04 nats) and **its mechanism is not bimodality in TNBBeta's sense**: its fitted epsilon is far above the
   modality threshold (m = +36 at sigma 0.1, +9 at 0.25, +1 at 0.5, +0.11 at 1.0; fraction of x with m < 0 is
   0 to 0.08), with p = 0.5, mean direction perpendicular to the spike. On S^1 the latitude `mu . z = 0` IS the
   antipodal pair, so a latitude-concentrated unimodal TNBBeta is exactly two caps (KL about 0 at every kappa in the
   free-axis check; the mass about its own axis is balanced, 0.48). This is a d = 2 coincidence that does not
   extend to the sphere's higher dimensions.
3. **d = 3 and d = 4 fixed (true) decoder: the clean test.** TNBBeta is NOT better than a plain cap at sigma 0.1 and
   0.25 (d=3 vs vMF: -0.054 and -0.071 nats; vs PS: -0.087 and -0.088; d=4 descriptive: 1.44 / 1.03 vs vMF 1.35 /
   0.98), exactly the representability prediction P3 (its posterior is unimodal there: mass balance 0.000,
   m = +1.6 / +1.1 at d=3). It is better at sigma 0.5 (d=3: 0.63 vs vMF 0.78, PS 0.79) and sigma 1.0 (0.18 vs 0.41,
   0.35), confirmed for both comparators, and uses `m < 0` there (d=3: m = -0.51 at 0.5, -0.41 at 1.0; mass balance
   0.09 at 0.5, 0.48 at 1.0; p = 0.95 at 0.5, 0.5 at 1.0). **The sign-symmetrised caps do far better than TNBBeta in
   every d=3 and d=4 fixed cell** (d=3: 0.29 / 0.11 / 0.12 / 0.068 for vMF-sym against TNBBeta 1.06 / 0.89 / 0.63 /
   0.18; d=4: 0.66 / 0.28 / 0.26 / 0.12 against 1.44 / 1.03 / 0.76 / 0.21). So on this task, at d >= 3, an antipodal
   shape helps, TNBBeta's family captures only a minority of it, and a trivial antipodal mixture of the baselines
   captures nearly all of it. The exact axially-symmetric lower bound (0.005-0.18 nats) shows the axial symmetry
   of TNBBeta is not the limitation.
4. **Prediction scorecard (as registered).** P1 (ln 2 for sign-unaware caps at low noise): confirmed (d=2: 0.71-0.84;
   d=3: 0.82-1.01; d=4: 0.98-1.45). P2 (symmetrised caps near the axial bound): confirmed at d=3, 4 (0.07-0.29 vs
   bound 0.005-0.18 plus amortisation), d=2 exact. P3: d>=3 fixed decoder confirmed qualitatively (disadvantage at
   sigma 0.1/0.25, advantage at 0.5; the sigma=1.0 advantage, 0.16-0.23, is larger than the "under 0.05"
   prediction, so that part FAILED); d=2 prediction of a disadvantage at sigma 0.1/0.25 FAILED, for the reason in
   Deviation 2. P4 (topology): mass-balance tables show vMF/PS one-sided (0.000) at every d at sigma <= 0.5, and
   their d=2 gaps (0.71-0.84) sit at the one-sided bound, not above it; no evidence of an extra topology penalty
   was separable.
5. **Learned decoders (B, C) at d >= 3 are dominated by posterior collapse, not by shape.** For d=3 sigma <= 0.5 and
   d=4 sigma <= 0.5, vMF/PS/TNBBeta (and often the symmetrised arms) end at a no-latent-structure solution: the
   learned sigma is far above the truth (d=3 linear TNBBeta: 0.285, 0.369, 0.570 at sigma 0.1, 0.25, 0.5;
   d=4: 0.236, 0.331, 0.545), `KL(q || p_model)` about 0 and the decoder ignores `z`; the shared "floor" gaps
   (d=3 sigma 0.5: 0.125 for PS, TNBBeta and symmetrised arms alike; d=4 sigma 0.5: 0.099) are the cost of not
   using `z`. The 8 "confirmed" d=3 learned-decoder cells against vMF (0.14-0.16 nats) are an artefact of that:
   vMF's kappa floor (kappa >= 1) cannot represent the uniform posterior of a collapsed model (its floor is 0.27
   vs 0.125), while TNBBeta and PS tie exactly (differences -0.0001 to 0.0075, no Holm rejections with margin) and
   the one arm that sometimes escapes collapse is sign-symmetrised vMF (d=3 sigma 0.1: gap 0.27 vs 5.2 for
   everything else; d=4 sigma 0.1: 0.63). These are optimisation findings with a learned scalar sigma starting at
   1.0, not evidence about antipodal bimodality; the registered learned-decoder tests are reported as run but should
   not be read as "TNBBeta helps". At d=2 the learned decoders train fine (learned sigma within 3% of truth) and
   the same picture as the fixed decoder holds (TNBBeta and sign-sym tie at 0.01-0.1).
6. **Anomaly investigated (not discarded).** d=3, sigma 0.1 and 0.25, fixed decoder, `tnbbeta_init0`: 4 of 10 seeds at
   sigma 0.1 (seeds 0, 2, 7, 9) end at gap about 43.7 (minimum per-x gap 19 nats) with p = 0.5, q = 0.85, epsilon
   about 10.4: a thin ring about the head's axis, the d=2 equatorial solution transplanted to S^2, where a
   great circle is not a Bingham mode. The default p-init (log 3) never lands there (all 10 seeds 0.96-1.27). No
   run has a negative MEAN gap (smallest 0.0012). Per-x gaps against the TRUE evidence are negative in 1428 runs,
   down to -6.2 nats (e.g. d=2, sigma 0.25, learned-decoder PS): investigated, not discarded. In all 720
   fixed-decoder runs the smallest per-x gap is -0.033 (Monte Carlo error of a 256-draw ELBO, as the estimator is
   unbiased), and in all learned-decoder runs `KL(q || p_model)` per x is at least -0.035, so the ELBOs are valid
   bounds on the MODEL's own evidence; the large negative gaps against the true `log p(X)` are x for which the
   learned model (with its own learned sigma and decoder) assigns higher likelihood than the true model, which
   is possible per datum, and the mean over test x stays positive.
7. **Quadrature and sampler checks.** True-model sphere-quadrature error vs closed-form Bingham: at most 1.1e-11 nats
   (d=3, sigma 0.1); learned-model 2x-refinement differences at most 1.05e-3 nats (d=3 sigma 0.1), 2.5e-4 (d=2),
   far below every reported difference. The d=4 exact evidence (1-D Bessel integral) agrees with an independent 3-D
   Hopf quadrature to 7.4e-12 (pre-freeze table). vMF rejection-sampler bias (reported by the dimension-scaling
   agent for m >= 9): checked here at m = 2, 3, 4, kappa = 5 to 300 (400k draws each, first and second latitude
   moments vs quadrature, largest |z| = 2.5 of 42): no bias detected, and the vMF arms in this study (m <= 4)
   are outside the affected regime.
8. **Free-axis representability (post-hoc, job 2299061)**: with `mu` free, TNBBeta represents the rank-one
   Bingham exactly on S^1 (KL <= 0.005 at all kappa from 1 to 128, `psi` = 90 degrees, p = 0.5, but m > 0 for
   kappa >= 2), and for d = 3, 4 the best fit is the aligned-axis one (psi = 0 or pi): KL 0.014 / 0.070 / 0.34 / 0.69
   at kappa 1 / 2 / 4 / 8 (d=3), then 0.76-0.82 (worse than vMF's 0.69) for kappa >= 16; d=4: 0.007 / 0.039 / 0.235 /
   0.665, then 0.73-0.76. So the aligned-axis table of the pre-registration stands for d >= 3.

Interpretation. The Bingham/spiked-matrix task does give a posterior that is antipodal by symmetry, with the decoder
unable to absorb it (fixed or sign-invariant), and the answer to "is TNBBeta's structural capacity needed"
is: an antipodal family IS needed (sign-unaware caps lose >= ln 2, and the d=3, 4 fixed-decoder gap to the
sign-symmetrised caps is 0.1-0.8 nats), but TNBBeta provides it only partially: exactly (through a non-bimodal
mechanism) at d = 2, and at d >= 3 only at broad posteriors (sigma >= 0.5), where it recovers roughly half of the
gain a trivially sign-symmetrised cap achieves, and at low noise not at all. Confirmatory "TNBBeta beats vMF/PS"
verdicts must therefore be read together with the sign-symmetrised baselines, which no TNBBeta cell beats.

## Deviations

Deviations 1 and 2 (non-finite minibatch skipping; representability restriction) are recorded in the
pre-registration; no others. Not run: the optional 2-component vMF mixture arm (as registered).

---

# Full `summarize` output (job 2301910, unedited)


## Per-arm test ELBO gap against the true log p(X) (nats; lower is better)

| d | sigma | decoder | arm | seeds (valid) | gap mean +- sd | median seed | min / max | IWAE gap mean |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.1 | fixed | vmf | 10/10 | 0.8395 +- 0.1344 | 0.7618 | 0.7126 / 1.1274 | 0.7729 |
| 2 | 0.1 | fixed | power_spherical | 10/10 | 0.8060 +- 0.1120 | 0.7698 | 0.7123 / 1.0504 | 0.7301 |
| 2 | 0.1 | fixed | tnbbeta | 10/10 | 0.0872 +- 0.0694 | 0.0721 | 0.0247 / 0.2747 | 0.0328 |
| 2 | 0.1 | fixed | tnbbeta_init0 | 10/10 | 0.1271 +- 0.1408 | 0.0961 | 0.0130 / 0.4805 | 0.0534 |
| 2 | 0.1 | fixed | vmf_sym | 10/10 | 0.1252 +- 0.0650 | 0.1157 | 0.0364 / 0.2435 | 0.0638 |
| 2 | 0.1 | fixed | ps_sym | 10/10 | 0.1008 +- 0.1059 | 0.0674 | 0.0151 / 0.3779 | 0.0407 |
| 2 | 0.1 | linear | vmf | 10/10 | 2.9676 +- 0.0007 | 2.9674 | 2.9667 / 2.9691 | 2.7665 |
| 2 | 0.1 | linear | power_spherical | 10/10 | 1.3795 +- 0.9661 | 0.8209 | 0.7101 / 2.7781 | 1.3436 |
| 2 | 0.1 | linear | tnbbeta | 10/10 | 0.1014 +- 0.0715 | 0.0861 | 0.0261 / 0.2312 | 0.0498 |
| 2 | 0.1 | linear | tnbbeta_init0 | 10/10 | 0.1138 +- 0.0674 | 0.0992 | 0.0443 / 0.2428 | 0.0582 |
| 2 | 0.1 | linear | vmf_sym | 10/10 | 0.1241 +- 0.1093 | 0.0979 | 0.0244 / 0.3951 | 0.0630 |
| 2 | 0.1 | linear | ps_sym | 10/10 | 0.1397 +- 0.1596 | 0.1011 | 0.0212 / 0.5094 | 0.0711 |
| 2 | 0.1 | mlp | vmf | 10/10 | 2.9679 +- 0.0005 | 2.9680 | 2.9671 / 2.9688 | 2.7667 |
| 2 | 0.1 | mlp | power_spherical | 10/10 | 2.3753 +- 0.8496 | 2.7781 | 0.7530 / 2.7795 | 2.3688 |
| 2 | 0.1 | mlp | tnbbeta | 10/10 | 0.1646 +- 0.2217 | 0.0706 | 0.0369 / 0.7493 | 0.0871 |
| 2 | 0.1 | mlp | tnbbeta_init0 | 10/10 | 0.0634 +- 0.0228 | 0.0600 | 0.0275 / 0.1041 | 0.0283 |
| 2 | 0.1 | mlp | vmf_sym | 10/10 | 0.1570 +- 0.1762 | 0.0982 | 0.0213 / 0.6035 | 0.0845 |
| 2 | 0.1 | mlp | ps_sym | 10/10 | 0.0896 +- 0.0773 | 0.0662 | 0.0328 / 0.2957 | 0.0502 |
| 2 | 0.25 | fixed | vmf | 10/10 | 0.7110 +- 0.0170 | 0.7076 | 0.6972 / 0.7536 | 0.6908 |
| 2 | 0.25 | fixed | power_spherical | 10/10 | 0.7118 +- 0.0153 | 0.7054 | 0.6953 / 0.7463 | 0.6940 |
| 2 | 0.25 | fixed | tnbbeta | 10/10 | 0.0329 +- 0.0335 | 0.0195 | 0.0118 / 0.1196 | 0.0053 |
| 2 | 0.25 | fixed | tnbbeta_init0 | 10/10 | 0.0265 +- 0.0133 | 0.0248 | 0.0062 / 0.0552 | 0.0066 |
| 2 | 0.25 | fixed | vmf_sym | 10/10 | 0.0216 +- 0.0108 | 0.0179 | 0.0051 / 0.0398 | 0.0074 |
| 2 | 0.25 | fixed | ps_sym | 10/10 | 0.0580 +- 0.1286 | 0.0201 | 0.0046 / 0.4234 | 0.0173 |
| 2 | 0.25 | linear | vmf | 10/10 | 0.9240 +- 0.0006 | 0.9241 | 0.9234 / 0.9254 | 0.7259 |
| 2 | 0.25 | linear | power_spherical | 10/10 | 0.7217 +- 0.0148 | 0.7299 | 0.6975 / 0.7390 | 0.7100 |
| 2 | 0.25 | linear | tnbbeta | 10/10 | 0.0189 +- 0.0088 | 0.0204 | 0.0059 / 0.0319 | 0.0033 |
| 2 | 0.25 | linear | tnbbeta_init0 | 10/10 | 0.0301 +- 0.0234 | 0.0246 | 0.0083 / 0.0919 | 0.0069 |
| 2 | 0.25 | linear | vmf_sym | 10/10 | 0.0205 +- 0.0230 | 0.0144 | 0.0076 / 0.0843 | 0.0059 |
| 2 | 0.25 | linear | ps_sym | 10/10 | 0.0207 +- 0.0155 | 0.0134 | 0.0090 / 0.0493 | 0.0045 |
| 2 | 0.25 | mlp | vmf | 10/10 | 0.9246 +- 0.0005 | 0.9247 | 0.9238 / 0.9252 | 0.7262 |
| 2 | 0.25 | mlp | power_spherical | 10/10 | 0.7293 +- 0.0050 | 0.7307 | 0.7153 / 0.7327 | 0.7258 |
| 2 | 0.25 | mlp | tnbbeta | 10/10 | 0.0283 +- 0.0163 | 0.0200 | 0.0141 / 0.0655 | 0.0053 |
| 2 | 0.25 | mlp | tnbbeta_init0 | 10/10 | 0.0183 +- 0.0092 | 0.0159 | 0.0072 / 0.0307 | 0.0049 |
| 2 | 0.25 | mlp | vmf_sym | 10/10 | 0.0261 +- 0.0182 | 0.0219 | 0.0048 / 0.0610 | 0.0094 |
| 2 | 0.25 | mlp | ps_sym | 10/10 | 0.0221 +- 0.0082 | 0.0206 | 0.0131 / 0.0404 | 0.0064 |
| 2 | 0.5 | fixed | vmf | 10/10 | 0.6414 +- 0.0026 | 0.6416 | 0.6368 / 0.6457 | 0.4010 |
| 2 | 0.5 | fixed | power_spherical | 10/10 | 0.6053 +- 0.0030 | 0.6043 | 0.6009 / 0.6116 | 0.4873 |
| 2 | 0.5 | fixed | tnbbeta | 10/10 | 0.0114 +- 0.0027 | 0.0113 | 0.0081 / 0.0170 | 0.0006 |
| 2 | 0.5 | fixed | tnbbeta_init0 | 10/10 | 0.0133 +- 0.0076 | 0.0108 | 0.0052 / 0.0285 | 0.0014 |
| 2 | 0.5 | fixed | vmf_sym | 10/10 | 0.0108 +- 0.0056 | 0.0085 | 0.0068 / 0.0240 | 0.0012 |
| 2 | 0.5 | fixed | ps_sym | 10/10 | 0.0134 +- 0.0023 | 0.0126 | 0.0113 / 0.0188 | 0.0013 |
| 2 | 0.5 | linear | vmf | 10/10 | 0.3157 +- 0.0005 | 0.3157 | 0.3151 / 0.3169 | 0.1191 |
| 2 | 0.5 | linear | power_spherical | 10/10 | 0.1171 +- 0.0005 | 0.1171 | 0.1165 / 0.1181 | 0.1171 |
| 2 | 0.5 | linear | tnbbeta | 10/10 | 0.0095 +- 0.0022 | 0.0090 | 0.0073 / 0.0125 | 0.0014 |
| 2 | 0.5 | linear | tnbbeta_init0 | 10/10 | 0.0105 +- 0.0030 | 0.0094 | 0.0072 / 0.0166 | 0.0012 |
| 2 | 0.5 | linear | vmf_sym | 10/10 | 0.0103 +- 0.0028 | 0.0101 | 0.0072 / 0.0161 | 0.0018 |
| 2 | 0.5 | linear | ps_sym | 10/10 | 0.0141 +- 0.0023 | 0.0138 | 0.0096 / 0.0175 | 0.0015 |
| 2 | 0.5 | mlp | vmf | 10/10 | 0.3162 +- 0.0006 | 0.3164 | 0.3151 / 0.3169 | 0.1195 |
| 2 | 0.5 | mlp | power_spherical | 10/10 | 0.1174 +- 0.0006 | 0.1175 | 0.1164 / 0.1181 | 0.1174 |
| 2 | 0.5 | mlp | tnbbeta | 10/10 | 0.0124 +- 0.0040 | 0.0115 | 0.0059 / 0.0203 | 0.0027 |
| 2 | 0.5 | mlp | tnbbeta_init0 | 10/10 | 0.0131 +- 0.0042 | 0.0119 | 0.0061 / 0.0213 | 0.0024 |
| 2 | 0.5 | mlp | vmf_sym | 10/10 | 0.0115 +- 0.0025 | 0.0111 | 0.0075 / 0.0160 | 0.0021 |
| 2 | 0.5 | mlp | ps_sym | 10/10 | 0.0157 +- 0.0047 | 0.0142 | 0.0117 / 0.0265 | 0.0026 |
| 2 | 1 | fixed | vmf | 10/10 | 0.3790 +- 0.0014 | 0.3788 | 0.3766 / 0.3814 | 0.0228 |
| 2 | 1 | fixed | power_spherical | 10/10 | 0.2785 +- 0.0014 | 0.2791 | 0.2748 / 0.2799 | 0.0064 |
| 2 | 1 | fixed | tnbbeta | 10/10 | 0.0056 +- 0.0011 | 0.0053 | 0.0045 / 0.0082 | 0.0003 |
| 2 | 1 | fixed | tnbbeta_init0 | 10/10 | 0.0056 +- 0.0008 | 0.0054 | 0.0048 / 0.0072 | 0.0003 |
| 2 | 1 | fixed | vmf_sym | 10/10 | 0.0060 +- 0.0012 | 0.0058 | 0.0046 / 0.0087 | 0.0003 |
| 2 | 1 | fixed | ps_sym | 10/10 | 0.0101 +- 0.0013 | 0.0101 | 0.0085 / 0.0132 | 0.0003 |
| 2 | 1 | linear | vmf | 10/10 | 0.2130 +- 0.0005 | 0.2129 | 0.2124 / 0.2138 | 0.0165 |
| 2 | 1 | linear | power_spherical | 10/10 | 0.0121 +- 0.0005 | 0.0120 | 0.0114 / 0.0130 | 0.0120 |
| 2 | 1 | linear | tnbbeta | 10/10 | 0.0055 +- 0.0009 | 0.0055 | 0.0041 / 0.0071 | 0.0006 |
| 2 | 1 | linear | tnbbeta_init0 | 10/10 | 0.0054 +- 0.0006 | 0.0054 | 0.0044 / 0.0064 | 0.0009 |
| 2 | 1 | linear | vmf_sym | 10/10 | 0.0065 +- 0.0009 | 0.0064 | 0.0053 / 0.0081 | 0.0010 |
| 2 | 1 | linear | ps_sym | 10/10 | 0.0089 +- 0.0014 | 0.0084 | 0.0076 / 0.0118 | 0.0035 |
| 2 | 1 | mlp | vmf | 10/10 | 0.2133 +- 0.0007 | 0.2131 | 0.2124 / 0.2146 | 0.0165 |
| 2 | 1 | mlp | power_spherical | 10/10 | 0.0121 +- 0.0006 | 0.0120 | 0.0113 / 0.0129 | 0.0121 |
| 2 | 1 | mlp | tnbbeta | 10/10 | 0.0068 +- 0.0021 | 0.0064 | 0.0045 / 0.0115 | 0.0026 |
| 2 | 1 | mlp | tnbbeta_init0 | 10/10 | 0.0062 +- 0.0007 | 0.0063 | 0.0051 / 0.0077 | 0.0015 |
| 2 | 1 | mlp | vmf_sym | 10/10 | 0.0071 +- 0.0024 | 0.0064 | 0.0048 / 0.0127 | 0.0010 |
| 2 | 1 | mlp | ps_sym | 10/10 | 0.0094 +- 0.0012 | 0.0092 | 0.0077 / 0.0116 | 0.0045 |
| 3 | 0.1 | fixed | vmf | 10/10 | 1.0099 +- 0.0958 | 1.0040 | 0.8858 / 1.1665 | 0.8207 |
| 3 | 0.1 | fixed | power_spherical | 10/10 | 0.9765 +- 0.1329 | 0.9442 | 0.8397 / 1.3235 | 0.7975 |
| 3 | 0.1 | fixed | tnbbeta | 10/10 | 1.0637 +- 0.0960 | 1.0297 | 0.9607 / 1.2677 | 0.7921 |
| 3 | 0.1 | fixed | tnbbeta_init0 | 10/10 | 18.1553 +- 22.0008 | 1.1997 | 0.9710 / 43.8011 | 0.5963 |
| 3 | 0.1 | fixed | vmf_sym | 10/10 | 0.2859 +- 0.1354 | 0.2386 | 0.1639 / 0.5872 | 0.0999 |
| 3 | 0.1 | fixed | ps_sym | 10/10 | 0.3241 +- 0.1352 | 0.2824 | 0.1720 / 0.5647 | 0.1187 |
| 3 | 0.1 | linear | vmf | 10/10 | 5.3489 +- 0.0012 | 5.3488 | 5.3472 / 5.3510 | 5.2027 |
| 3 | 0.1 | linear | power_spherical | 10/10 | 5.2119 +- 0.0010 | 5.2116 | 5.2107 / 5.2142 | 5.2114 |
| 3 | 0.1 | linear | tnbbeta | 10/10 | 5.1899 +- 0.0033 | 5.1884 | 5.1856 / 5.1954 | 5.1257 |
| 3 | 0.1 | linear | tnbbeta_init0 | 10/10 | 5.1833 +- 0.0229 | 5.1904 | 5.1185 / 5.1966 | 5.1171 |
| 3 | 0.1 | linear | vmf_sym | 10/10 | 0.2715 +- 0.0833 | 0.2941 | 0.1331 / 0.3856 | 0.1103 |
| 3 | 0.1 | linear | ps_sym | 10/10 | 5.2089 +- 0.0046 | 5.2107 | 5.1999 / 5.2136 | 5.1992 |
| 3 | 0.1 | mlp | vmf | 10/10 | 5.3489 +- 0.0010 | 5.3489 | 5.3476 / 5.3509 | 5.2036 |
| 3 | 0.1 | mlp | power_spherical | 10/10 | 5.2112 +- 0.0008 | 5.2113 | 5.2099 / 5.2129 | 5.2112 |
| 3 | 0.1 | mlp | tnbbeta | 10/10 | 5.2113 +- 0.0008 | 5.2113 | 5.2102 / 5.2130 | 5.2112 |
| 3 | 0.1 | mlp | tnbbeta_init0 | 10/10 | 5.2112 +- 0.0008 | 5.2113 | 5.2099 / 5.2129 | 5.2112 |
| 3 | 0.1 | mlp | vmf_sym | 10/10 | 0.3717 +- 0.1798 | 0.3857 | 0.1550 / 0.7601 | 0.1622 |
| 3 | 0.1 | mlp | ps_sym | 10/10 | 5.2112 +- 0.0008 | 5.2113 | 5.2098 / 5.2129 | 5.2112 |
| 3 | 0.25 | fixed | vmf | 10/10 | 0.8167 +- 0.0247 | 0.8082 | 0.7878 / 0.8497 | 0.7188 |
| 3 | 0.25 | fixed | power_spherical | 10/10 | 0.7994 +- 0.0176 | 0.7997 | 0.7678 / 0.8268 | 0.7118 |
| 3 | 0.25 | fixed | tnbbeta | 10/10 | 0.8872 +- 0.0225 | 0.8757 | 0.8645 / 0.9303 | 0.7081 |
| 3 | 0.25 | fixed | tnbbeta_init0 | 10/10 | 3.0099 +- 2.2321 | 3.0272 | 0.8634 / 5.1636 | 0.3745 |
| 3 | 0.25 | fixed | vmf_sym | 10/10 | 0.1122 +- 0.0183 | 0.1067 | 0.0898 / 0.1416 | 0.0173 |
| 3 | 0.25 | fixed | ps_sym | 10/10 | 0.1202 +- 0.0231 | 0.1146 | 0.0945 / 0.1611 | 0.0199 |
| 3 | 0.25 | linear | vmf | 10/10 | 1.2194 +- 0.0010 | 1.2194 | 1.2181 / 1.2208 | 1.0750 |
| 3 | 0.25 | linear | power_spherical | 10/10 | 1.0779 +- 0.0007 | 1.0780 | 1.0769 / 1.0791 | 1.0776 |
| 3 | 0.25 | linear | tnbbeta | 10/10 | 1.0768 +- 0.0025 | 1.0779 | 1.0726 / 1.0793 | 1.0723 |
| 3 | 0.25 | linear | tnbbeta_init0 | 10/10 | 1.0742 +- 0.0016 | 1.0744 | 1.0721 / 1.0764 | 1.0604 |
| 3 | 0.25 | linear | vmf_sym | 10/10 | 0.1190 +- 0.0214 | 0.1244 | 0.0813 / 0.1452 | 0.0230 |
| 3 | 0.25 | linear | ps_sym | 10/10 | 0.1149 +- 0.0149 | 0.1102 | 0.0982 / 0.1414 | 0.0283 |
| 3 | 0.25 | mlp | vmf | 10/10 | 1.2199 +- 0.0007 | 1.2197 | 1.2190 / 1.2216 | 1.0755 |
| 3 | 0.25 | mlp | power_spherical | 10/10 | 1.0774 +- 0.0006 | 1.0773 | 1.0767 / 1.0785 | 1.0774 |
| 3 | 0.25 | mlp | tnbbeta | 10/10 | 1.0699 +- 0.0065 | 1.0673 | 1.0618 / 1.0777 | 1.0609 |
| 3 | 0.25 | mlp | tnbbeta_init0 | 10/10 | 1.0747 +- 0.0047 | 1.0769 | 1.0663 / 1.0785 | 1.0709 |
| 3 | 0.25 | mlp | vmf_sym | 10/10 | 0.2208 +- 0.3017 | 0.1283 | 0.1074 / 1.0785 | 0.1325 |
| 3 | 0.25 | mlp | ps_sym | 10/10 | 0.7958 +- 0.4535 | 1.0771 | 0.1269 / 1.0779 | 0.7650 |
| 3 | 0.5 | fixed | vmf | 10/10 | 0.7782 +- 0.0040 | 0.7787 | 0.7719 / 0.7860 | 0.5404 |
| 3 | 0.5 | fixed | power_spherical | 10/10 | 0.7880 +- 0.0062 | 0.7851 | 0.7812 / 0.7976 | 0.5835 |
| 3 | 0.5 | fixed | tnbbeta | 10/10 | 0.6269 +- 0.0062 | 0.6274 | 0.6137 / 0.6339 | 0.0646 |
| 3 | 0.5 | fixed | tnbbeta_init0 | 10/10 | 0.7090 +- 0.1039 | 0.6366 | 0.6210 / 0.8310 | 0.0399 |
| 3 | 0.5 | fixed | vmf_sym | 10/10 | 0.1239 +- 0.0050 | 0.1235 | 0.1154 / 0.1321 | 0.0070 |
| 3 | 0.5 | fixed | ps_sym | 10/10 | 0.1375 +- 0.0101 | 0.1356 | 0.1255 / 0.1535 | 0.0131 |
| 3 | 0.5 | linear | vmf | 10/10 | 0.2697 +- 0.0007 | 0.2697 | 0.2690 / 0.2709 | 0.1251 |
| 3 | 0.5 | linear | power_spherical | 10/10 | 0.1248 +- 0.0007 | 0.1247 | 0.1242 / 0.1261 | 0.1247 |
| 3 | 0.5 | linear | tnbbeta | 10/10 | 0.1250 +- 0.0007 | 0.1248 | 0.1242 / 0.1263 | 0.1247 |
| 3 | 0.5 | linear | tnbbeta_init0 | 10/10 | 0.1249 +- 0.0007 | 0.1248 | 0.1242 / 0.1261 | 0.1247 |
| 3 | 0.5 | linear | vmf_sym | 10/10 | 0.1267 +- 0.0041 | 0.1268 | 0.1174 / 0.1344 | 0.1022 |
| 3 | 0.5 | linear | ps_sym | 10/10 | 0.1254 +- 0.0018 | 0.1246 | 0.1243 / 0.1298 | 0.1026 |
| 3 | 0.5 | mlp | vmf | 10/10 | 0.2697 +- 0.0007 | 0.2696 | 0.2688 / 0.2712 | 0.1250 |
| 3 | 0.5 | mlp | power_spherical | 10/10 | 0.1246 +- 0.0006 | 0.1243 | 0.1240 / 0.1256 | 0.1246 |
| 3 | 0.5 | mlp | tnbbeta | 10/10 | 0.1245 +- 0.0004 | 0.1244 | 0.1241 / 0.1255 | 0.1242 |
| 3 | 0.5 | mlp | tnbbeta_init0 | 10/10 | 0.1244 +- 0.0008 | 0.1244 | 0.1231 / 0.1256 | 0.1239 |
| 3 | 0.5 | mlp | vmf_sym | 10/10 | 0.1246 +- 0.0040 | 0.1266 | 0.1161 / 0.1285 | 0.0801 |
| 3 | 0.5 | mlp | ps_sym | 10/10 | 0.1259 +- 0.0023 | 0.1247 | 0.1241 / 0.1308 | 0.0920 |
| 3 | 1 | fixed | vmf | 10/10 | 0.4130 +- 0.0020 | 0.4121 | 0.4108 / 0.4161 | 0.0312 |
| 3 | 1 | fixed | power_spherical | 10/10 | 0.3476 +- 0.0074 | 0.3445 | 0.3406 / 0.3585 | 0.0067 |
| 3 | 1 | fixed | tnbbeta | 10/10 | 0.1835 +- 0.0016 | 0.1838 | 0.1801 / 0.1856 | 0.0018 |
| 3 | 1 | fixed | tnbbeta_init0 | 10/10 | 0.1648 +- 0.0165 | 0.1549 | 0.1491 / 0.1859 | 0.0013 |
| 3 | 1 | fixed | vmf_sym | 10/10 | 0.0676 +- 0.0025 | 0.0676 | 0.0637 / 0.0732 | 0.0024 |
| 3 | 1 | fixed | ps_sym | 10/10 | 0.0768 +- 0.0019 | 0.0763 | 0.0737 / 0.0798 | 0.0019 |
| 3 | 1 | linear | vmf | 10/10 | 0.1536 +- 0.0006 | 0.1535 | 0.1525 / 0.1546 | 0.0089 |
| 3 | 1 | linear | power_spherical | 10/10 | 0.0078 +- 0.0006 | 0.0076 | 0.0068 / 0.0088 | 0.0077 |
| 3 | 1 | linear | tnbbeta | 10/10 | 0.0079 +- 0.0006 | 0.0077 | 0.0069 / 0.0090 | 0.0077 |
| 3 | 1 | linear | tnbbeta_init0 | 10/10 | 0.0079 +- 0.0006 | 0.0077 | 0.0068 / 0.0088 | 0.0078 |
| 3 | 1 | linear | vmf_sym | 10/10 | 0.0111 +- 0.0006 | 0.0109 | 0.0101 / 0.0121 | 0.0075 |
| 3 | 1 | linear | ps_sym | 10/10 | 0.0079 +- 0.0006 | 0.0077 | 0.0069 / 0.0088 | 0.0077 |
| 3 | 1 | mlp | vmf | 10/10 | 0.1536 +- 0.0006 | 0.1535 | 0.1528 / 0.1548 | 0.0089 |
| 3 | 1 | mlp | power_spherical | 10/10 | 0.0077 +- 0.0006 | 0.0075 | 0.0067 / 0.0086 | 0.0077 |
| 3 | 1 | mlp | tnbbeta | 10/10 | 0.0078 +- 0.0005 | 0.0077 | 0.0069 / 0.0087 | 0.0077 |
| 3 | 1 | mlp | tnbbeta_init0 | 10/10 | 0.0078 +- 0.0006 | 0.0076 | 0.0067 / 0.0086 | 0.0077 |
| 3 | 1 | mlp | vmf_sym | 10/10 | 0.0112 +- 0.0006 | 0.0111 | 0.0104 / 0.0121 | 0.0074 |
| 3 | 1 | mlp | ps_sym | 10/10 | 0.0078 +- 0.0006 | 0.0076 | 0.0068 / 0.0086 | 0.0077 |
| 4 | 0.1 | fixed | vmf | 10/10 | 1.3501 +- 0.0693 | 1.3285 | 1.2761 / 1.5163 | 0.9338 |
| 4 | 0.1 | fixed | power_spherical | 10/10 | 1.4491 +- 0.1639 | 1.4364 | 1.2158 / 1.7426 | 0.9851 |
| 4 | 0.1 | fixed | tnbbeta | 10/10 | 1.4365 +- 0.1037 | 1.4353 | 1.3217 / 1.6861 | 0.9454 |
| 4 | 0.1 | fixed | tnbbeta_init0 | 10/10 | 1.3808 +- 0.1248 | 1.3996 | 1.1853 / 1.5803 | 0.9214 |
| 4 | 0.1 | fixed | vmf_sym | 10/10 | 0.6643 +- 0.1433 | 0.6319 | 0.5036 / 0.9384 | 0.2555 |
| 4 | 0.1 | fixed | ps_sym | 10/10 | 0.7645 +- 0.2416 | 0.7110 | 0.4726 / 1.2425 | 0.3143 |
| 4 | 0.1 | linear | vmf | 10/10 | 7.9208 +- 0.0012 | 7.9207 | 7.9196 / 7.9236 | 7.8100 |
| 4 | 0.1 | linear | power_spherical | 10/10 | 7.8124 +- 0.0011 | 7.8124 | 7.8113 / 7.8145 | 7.8107 |
| 4 | 0.1 | linear | tnbbeta | 10/10 | 7.7888 +- 0.0069 | 7.7895 | 7.7779 / 7.7975 | 7.7276 |
| 4 | 0.1 | linear | tnbbeta_init0 | 10/10 | 7.7907 +- 0.0038 | 7.7907 | 7.7848 / 7.7977 | 7.7306 |
| 4 | 0.1 | linear | vmf_sym | 10/10 | 0.6309 +- 0.2346 | 0.5966 | 0.4155 / 1.1101 | 0.2829 |
| 4 | 0.1 | linear | ps_sym | 10/10 | 7.8013 +- 0.0053 | 7.8020 | 7.7911 / 7.8078 | 7.7529 |
| 4 | 0.1 | mlp | vmf | 10/10 | 7.9211 +- 0.0008 | 7.9212 | 7.9198 / 7.9223 | 7.8111 |
| 4 | 0.1 | mlp | power_spherical | 10/10 | 7.8102 +- 0.0007 | 7.8102 | 7.8092 / 7.8117 | 7.8102 |
| 4 | 0.1 | mlp | tnbbeta | 10/10 | 7.8104 +- 0.0008 | 7.8103 | 7.8094 / 7.8122 | 7.8103 |
| 4 | 0.1 | mlp | tnbbeta_init0 | 10/10 | 7.8103 +- 0.0008 | 7.8104 | 7.8093 / 7.8117 | 7.8102 |
| 4 | 0.1 | mlp | vmf_sym | 10/10 | 7.0832 +- 2.2972 | 7.8092 | 0.5452 / 7.8114 | 7.0479 |
| 4 | 0.1 | mlp | ps_sym | 10/10 | 7.8102 +- 0.0008 | 7.8103 | 7.8093 / 7.8117 | 7.8102 |
| 4 | 0.25 | fixed | vmf | 10/10 | 0.9807 +- 0.0313 | 0.9786 | 0.9225 / 1.0324 | 0.7521 |
| 4 | 0.25 | fixed | power_spherical | 10/10 | 0.9866 +- 0.0253 | 0.9855 | 0.9531 / 1.0282 | 0.7639 |
| 4 | 0.25 | fixed | tnbbeta | 10/10 | 1.0269 +- 0.0257 | 1.0322 | 0.9885 / 1.0595 | 0.7414 |
| 4 | 0.25 | fixed | tnbbeta_init0 | 10/10 | 1.0148 +- 0.0261 | 1.0215 | 0.9654 / 1.0428 | 0.7406 |
| 4 | 0.25 | fixed | vmf_sym | 10/10 | 0.2787 +- 0.0237 | 0.2800 | 0.2499 / 0.3124 | 0.0540 |
| 4 | 0.25 | fixed | ps_sym | 10/10 | 0.2855 +- 0.0242 | 0.2772 | 0.2647 / 0.3449 | 0.0638 |
| 4 | 0.25 | linear | vmf | 10/10 | 1.3581 +- 0.0007 | 1.3581 | 1.3572 / 1.3598 | 1.2482 |
| 4 | 0.25 | linear | power_spherical | 10/10 | 1.2457 +- 0.0006 | 1.2456 | 1.2450 / 1.2470 | 1.2448 |
| 4 | 0.25 | linear | tnbbeta | 10/10 | 1.2457 +- 0.0006 | 1.2456 | 1.2450 / 1.2469 | 1.2425 |
| 4 | 0.25 | linear | tnbbeta_init0 | 10/10 | 1.2439 +- 0.0016 | 1.2436 | 1.2420 / 1.2471 | 1.2315 |
| 4 | 0.25 | linear | vmf_sym | 10/10 | 1.1600 +- 0.2751 | 1.2470 | 0.3770 / 1.2480 | 1.1292 |
| 4 | 0.25 | linear | ps_sym | 10/10 | 1.2454 +- 0.0007 | 1.2454 | 1.2444 / 1.2465 | 1.2434 |
| 4 | 0.25 | mlp | vmf | 10/10 | 1.3571 +- 0.0005 | 1.3569 | 1.3566 / 1.3580 | 1.2479 |
| 4 | 0.25 | mlp | power_spherical | 10/10 | 1.2441 +- 0.0004 | 1.2440 | 1.2434 / 1.2446 | 1.2441 |
| 4 | 0.25 | mlp | tnbbeta | 10/10 | 1.2442 +- 0.0005 | 1.2442 | 1.2436 / 1.2453 | 1.2441 |
| 4 | 0.25 | mlp | tnbbeta_init0 | 10/10 | 1.2442 +- 0.0004 | 1.2441 | 1.2436 / 1.2448 | 1.2441 |
| 4 | 0.25 | mlp | vmf_sym | 10/10 | 1.2461 +- 0.0006 | 1.2459 | 1.2454 / 1.2470 | 1.2428 |
| 4 | 0.25 | mlp | ps_sym | 10/10 | 1.2441 +- 0.0004 | 1.2440 | 1.2435 / 1.2447 | 1.2440 |
| 4 | 0.5 | fixed | vmf | 10/10 | 0.9439 +- 0.0089 | 0.9434 | 0.9294 / 0.9566 | 0.4997 |
| 4 | 0.5 | fixed | power_spherical | 10/10 | 0.9399 +- 0.0064 | 0.9406 | 0.9302 / 0.9494 | 0.6230 |
| 4 | 0.5 | fixed | tnbbeta | 10/10 | 0.7562 +- 0.0088 | 0.7544 | 0.7454 / 0.7668 | 0.0824 |
| 4 | 0.5 | fixed | tnbbeta_init0 | 10/10 | 0.7536 +- 0.0057 | 0.7538 | 0.7451 / 0.7632 | 0.0842 |
| 4 | 0.5 | fixed | vmf_sym | 10/10 | 0.2582 +- 0.0059 | 0.2568 | 0.2484 / 0.2712 | 0.0196 |
| 4 | 0.5 | fixed | ps_sym | 10/10 | 0.2721 +- 0.0047 | 0.2716 | 0.2625 / 0.2783 | 0.0270 |
| 4 | 0.5 | linear | vmf | 10/10 | 0.2128 +- 0.0005 | 0.2128 | 0.2123 / 0.2139 | 0.1033 |
| 4 | 0.5 | linear | power_spherical | 10/10 | 0.0994 +- 0.0005 | 0.0993 | 0.0987 / 0.1002 | 0.0989 |
| 4 | 0.5 | linear | tnbbeta | 10/10 | 0.0994 +- 0.0005 | 0.0995 | 0.0988 / 0.1003 | 0.0988 |
| 4 | 0.5 | linear | tnbbeta_init0 | 10/10 | 0.0994 +- 0.0004 | 0.0994 | 0.0989 / 0.1003 | 0.0988 |
| 4 | 0.5 | linear | vmf_sym | 10/10 | 0.1019 +- 0.0005 | 0.1019 | 0.1013 / 0.1030 | 0.0984 |
| 4 | 0.5 | linear | ps_sym | 10/10 | 0.0994 +- 0.0005 | 0.0993 | 0.0987 / 0.1004 | 0.0989 |
| 4 | 0.5 | mlp | vmf | 10/10 | 0.2123 +- 0.0004 | 0.2123 | 0.2117 / 0.2128 | 0.1031 |
| 4 | 0.5 | mlp | power_spherical | 10/10 | 0.0983 +- 0.0004 | 0.0983 | 0.0978 / 0.0991 | 0.0983 |
| 4 | 0.5 | mlp | tnbbeta | 10/10 | 0.0985 +- 0.0004 | 0.0984 | 0.0978 / 0.0994 | 0.0984 |
| 4 | 0.5 | mlp | tnbbeta_init0 | 10/10 | 0.0984 +- 0.0004 | 0.0984 | 0.0979 / 0.0992 | 0.0980 |
| 4 | 0.5 | mlp | vmf_sym | 10/10 | 0.1012 +- 0.0004 | 0.1012 | 0.1006 / 0.1020 | 0.0981 |
| 4 | 0.5 | mlp | ps_sym | 10/10 | 0.0984 +- 0.0004 | 0.0983 | 0.0979 / 0.0991 | 0.0983 |
| 4 | 1 | fixed | vmf | 10/10 | 0.4352 +- 0.0016 | 0.4355 | 0.4331 / 0.4374 | 0.0182 |
| 4 | 1 | fixed | power_spherical | 10/10 | 0.3850 +- 0.0035 | 0.3855 | 0.3804 / 0.3890 | 0.0079 |
| 4 | 1 | fixed | tnbbeta | 10/10 | 0.2115 +- 0.0016 | 0.2115 | 0.2091 / 0.2142 | 0.0025 |
| 4 | 1 | fixed | tnbbeta_init0 | 10/10 | 0.2104 +- 0.0013 | 0.2101 | 0.2090 / 0.2130 | 0.0021 |
| 4 | 1 | fixed | vmf_sym | 10/10 | 0.1238 +- 0.0017 | 0.1233 | 0.1211 / 0.1268 | 0.0022 |
| 4 | 1 | fixed | ps_sym | 10/10 | 0.1337 +- 0.0014 | 0.1339 | 0.1310 / 0.1357 | 0.0036 |
| 4 | 1 | linear | vmf | 10/10 | 0.1161 +- 0.0005 | 0.1159 | 0.1151 / 0.1169 | 0.0068 |
| 4 | 1 | linear | power_spherical | 10/10 | 0.0022 +- 0.0005 | 0.0020 | 0.0015 / 0.0033 | 0.0019 |
| 4 | 1 | linear | tnbbeta | 10/10 | 0.0023 +- 0.0005 | 0.0021 | 0.0016 / 0.0034 | 0.0019 |
| 4 | 1 | linear | tnbbeta_init0 | 10/10 | 0.0023 +- 0.0005 | 0.0021 | 0.0016 / 0.0035 | 0.0020 |
| 4 | 1 | linear | vmf_sym | 10/10 | 0.0051 +- 0.0005 | 0.0050 | 0.0046 / 0.0060 | 0.0020 |
| 4 | 1 | linear | ps_sym | 10/10 | 0.0022 +- 0.0005 | 0.0020 | 0.0015 / 0.0033 | 0.0020 |
| 4 | 1 | mlp | vmf | 10/10 | 0.1158 +- 0.0005 | 0.1157 | 0.1151 / 0.1167 | 0.0066 |
| 4 | 1 | mlp | power_spherical | 10/10 | 0.0018 +- 0.0005 | 0.0016 | 0.0012 / 0.0028 | 0.0018 |
| 4 | 1 | mlp | tnbbeta | 10/10 | 0.0020 +- 0.0005 | 0.0018 | 0.0013 / 0.0029 | 0.0018 |
| 4 | 1 | mlp | tnbbeta_init0 | 10/10 | 0.0019 +- 0.0005 | 0.0017 | 0.0013 / 0.0028 | 0.0018 |
| 4 | 1 | mlp | vmf_sym | 10/10 | 0.0049 +- 0.0005 | 0.0047 | 0.0042 / 0.0059 | 0.0019 |
| 4 | 1 | mlp | ps_sym | 10/10 | 0.0018 +- 0.0005 | 0.0016 | 0.0012 / 0.0028 | 0.0018 |

## Per-seed ELBO gap


d=2, sigma=0.1, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.9287 | 0.8179 | 0.2747 | 0.1317 | 0.1136 | 0.0151 |
| 1 | 0.7509 | 0.7469 | 0.0553 | 0.0604 | 0.1148 | 0.0351 |
| 2 | 0.7628 | 0.7160 | 0.0427 | 0.1854 | 0.2286 | 0.0893 |
| 3 | 0.7489 | 0.7123 | 0.0736 | 0.1623 | 0.1303 | 0.0649 |
| 4 | 1.1274 | 0.9561 | 0.0708 | 0.0130 | 0.1185 | 0.0378 |
| 5 | 0.9534 | 1.0504 | 0.0904 | 0.4805 | 0.0364 | 0.0546 |
| 6 | 0.7608 | 0.7635 | 0.0735 | 0.0315 | 0.0752 | 0.1645 |
| 7 | 0.7126 | 0.8006 | 0.0661 | 0.0216 | 0.2435 | 0.0985 |
| 8 | 0.7408 | 0.7198 | 0.0247 | 0.1583 | 0.0745 | 0.0698 |
| 9 | 0.9086 | 0.7761 | 0.1001 | 0.0265 | 0.1166 | 0.3779 |

d=2, sigma=0.1, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 2.9671 | 0.8468 | 0.2312 | 0.0773 | 0.0553 | 0.0224 |
| 1 | 2.9667 | 2.7776 | 0.0753 | 0.0974 | 0.3951 | 0.1165 |
| 2 | 2.9676 | 0.7385 | 0.0303 | 0.1123 | 0.0826 | 0.0212 |
| 3 | 2.9675 | 2.7781 | 0.0385 | 0.1156 | 0.1132 | 0.0930 |
| 4 | 2.9676 | 0.7101 | 0.0261 | 0.0641 | 0.1357 | 0.0359 |
| 5 | 2.9683 | 0.7203 | 0.1185 | 0.2244 | 0.0710 | 0.1091 |
| 6 | 2.9691 | 0.8582 | 0.0929 | 0.1009 | 0.1935 | 0.0242 |
| 7 | 2.9672 | 2.7779 | 0.2148 | 0.2428 | 0.1434 | 0.3320 |
| 8 | 2.9674 | 0.7951 | 0.1068 | 0.0443 | 0.0244 | 0.1335 |
| 9 | 2.9672 | 0.7929 | 0.0794 | 0.0591 | 0.0273 | 0.5094 |

d=2, sigma=0.1, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 2.9675 | 2.7781 | 0.7493 | 0.0786 | 0.0213 | 0.0600 |
| 1 | 2.9680 | 2.7781 | 0.0369 | 0.0585 | 0.0958 | 0.0834 |
| 2 | 2.9681 | 2.7785 | 0.0391 | 0.0275 | 0.0648 | 0.0497 |
| 3 | 2.9682 | 0.7530 | 0.1603 | 0.0499 | 0.1218 | 0.0808 |
| 4 | 2.9679 | 2.7784 | 0.0665 | 0.0794 | 0.0307 | 0.0328 |
| 5 | 2.9671 | 2.7777 | 0.0511 | 0.0822 | 0.2490 | 0.2957 |
| 6 | 2.9688 | 2.7795 | 0.0902 | 0.0388 | 0.0410 | 0.0561 |
| 7 | 2.9680 | 0.7737 | 0.3135 | 0.1041 | 0.2418 | 0.0725 |
| 8 | 2.9684 | 2.7781 | 0.0645 | 0.0614 | 0.6035 | 0.1270 |
| 9 | 2.9673 | 2.7780 | 0.0747 | 0.0538 | 0.1006 | 0.0381 |

d=2, sigma=0.25, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.7042 | 0.7060 | 0.0237 | 0.0062 | 0.0115 | 0.0248 |
| 1 | 0.7115 | 0.7278 | 0.0581 | 0.0273 | 0.0162 | 0.0106 |
| 2 | 0.7006 | 0.7135 | 0.0339 | 0.0256 | 0.0170 | 0.0247 |
| 3 | 0.7536 | 0.7019 | 0.0133 | 0.0241 | 0.0398 | 0.0235 |
| 4 | 0.7201 | 0.7040 | 0.0118 | 0.0174 | 0.0051 | 0.0159 |
| 5 | 0.6972 | 0.7175 | 0.0202 | 0.0212 | 0.0297 | 0.0046 |
| 6 | 0.6987 | 0.7463 | 0.0188 | 0.0191 | 0.0287 | 0.4234 |
| 7 | 0.7151 | 0.7009 | 0.1196 | 0.0394 | 0.0161 | 0.0166 |
| 8 | 0.6977 | 0.7048 | 0.0160 | 0.0296 | 0.0334 | 0.0098 |
| 9 | 0.7111 | 0.6953 | 0.0136 | 0.0552 | 0.0188 | 0.0261 |

d=2, sigma=0.25, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.9240 | 0.7301 | 0.0123 | 0.0181 | 0.0084 | 0.0231 |
| 1 | 0.9235 | 0.7297 | 0.0319 | 0.0194 | 0.0232 | 0.0112 |
| 2 | 0.9240 | 0.7047 | 0.0101 | 0.0158 | 0.0135 | 0.0486 |
| 3 | 0.9242 | 0.7306 | 0.0255 | 0.0298 | 0.0158 | 0.0105 |
| 4 | 0.9242 | 0.6975 | 0.0180 | 0.0083 | 0.0107 | 0.0103 |
| 5 | 0.9234 | 0.7363 | 0.0229 | 0.0299 | 0.0179 | 0.0157 |
| 6 | 0.9254 | 0.7100 | 0.0285 | 0.0378 | 0.0079 | 0.0493 |
| 7 | 0.9242 | 0.7301 | 0.0059 | 0.0186 | 0.0153 | 0.0110 |
| 8 | 0.9235 | 0.7093 | 0.0237 | 0.0919 | 0.0076 | 0.0090 |
| 9 | 0.9242 | 0.7390 | 0.0107 | 0.0319 | 0.0843 | 0.0180 |

d=2, sigma=0.25, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.9247 | 0.7310 | 0.0182 | 0.0192 | 0.0149 | 0.0189 |
| 1 | 0.9250 | 0.7310 | 0.0388 | 0.0296 | 0.0275 | 0.0222 |
| 2 | 0.9252 | 0.7307 | 0.0211 | 0.0248 | 0.0163 | 0.0404 |
| 3 | 0.9248 | 0.7327 | 0.0307 | 0.0125 | 0.0610 | 0.0276 |
| 4 | 0.9245 | 0.7304 | 0.0154 | 0.0307 | 0.0373 | 0.0284 |
| 5 | 0.9241 | 0.7302 | 0.0141 | 0.0072 | 0.0048 | 0.0131 |
| 6 | 0.9252 | 0.7315 | 0.0655 | 0.0127 | 0.0420 | 0.0153 |
| 7 | 0.9243 | 0.7153 | 0.0185 | 0.0108 | 0.0380 | 0.0185 |
| 8 | 0.9247 | 0.7307 | 0.0414 | 0.0278 | 0.0130 | 0.0145 |
| 9 | 0.9238 | 0.7299 | 0.0189 | 0.0077 | 0.0062 | 0.0225 |

d=2, sigma=0.5, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.6457 | 0.6116 | 0.0111 | 0.0083 | 0.0084 | 0.0133 |
| 1 | 0.6411 | 0.6069 | 0.0081 | 0.0052 | 0.0176 | 0.0113 |
| 2 | 0.6400 | 0.6040 | 0.0116 | 0.0105 | 0.0086 | 0.0126 |
| 3 | 0.6437 | 0.6043 | 0.0144 | 0.0195 | 0.0091 | 0.0124 |
| 4 | 0.6389 | 0.6032 | 0.0096 | 0.0125 | 0.0078 | 0.0114 |
| 5 | 0.6421 | 0.6009 | 0.0096 | 0.0225 | 0.0070 | 0.0135 |
| 6 | 0.6368 | 0.6087 | 0.0116 | 0.0285 | 0.0068 | 0.0157 |
| 7 | 0.6402 | 0.6043 | 0.0086 | 0.0078 | 0.0116 | 0.0188 |
| 8 | 0.6421 | 0.6051 | 0.0123 | 0.0071 | 0.0075 | 0.0126 |
| 9 | 0.6438 | 0.6043 | 0.0170 | 0.0112 | 0.0240 | 0.0126 |

d=2, sigma=0.5, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.3157 | 0.1172 | 0.0073 | 0.0097 | 0.0074 | 0.0134 |
| 1 | 0.3151 | 0.1165 | 0.0120 | 0.0166 | 0.0161 | 0.0126 |
| 2 | 0.3156 | 0.1170 | 0.0077 | 0.0093 | 0.0085 | 0.0175 |
| 3 | 0.3162 | 0.1176 | 0.0074 | 0.0072 | 0.0111 | 0.0136 |
| 4 | 0.3155 | 0.1169 | 0.0103 | 0.0127 | 0.0090 | 0.0166 |
| 5 | 0.3152 | 0.1166 | 0.0122 | 0.0083 | 0.0113 | 0.0096 |
| 6 | 0.3169 | 0.1181 | 0.0106 | 0.0142 | 0.0077 | 0.0139 |
| 7 | 0.3158 | 0.1173 | 0.0073 | 0.0095 | 0.0116 | 0.0130 |
| 8 | 0.3158 | 0.1168 | 0.0076 | 0.0081 | 0.0072 | 0.0165 |
| 9 | 0.3156 | 0.1172 | 0.0125 | 0.0092 | 0.0128 | 0.0143 |

d=2, sigma=0.5, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.3164 | 0.1178 | 0.0123 | 0.0121 | 0.0117 | 0.0134 |
| 1 | 0.3169 | 0.1181 | 0.0148 | 0.0175 | 0.0154 | 0.0265 |
| 2 | 0.3167 | 0.1175 | 0.0163 | 0.0109 | 0.0104 | 0.0208 |
| 3 | 0.3162 | 0.1175 | 0.0203 | 0.0151 | 0.0113 | 0.0150 |
| 4 | 0.3156 | 0.1170 | 0.0100 | 0.0105 | 0.0110 | 0.0117 |
| 5 | 0.3151 | 0.1166 | 0.0059 | 0.0061 | 0.0112 | 0.0128 |
| 6 | 0.3168 | 0.1181 | 0.0132 | 0.0141 | 0.0093 | 0.0124 |
| 7 | 0.3166 | 0.1175 | 0.0102 | 0.0213 | 0.0075 | 0.0121 |
| 8 | 0.3163 | 0.1175 | 0.0105 | 0.0117 | 0.0110 | 0.0166 |
| 9 | 0.3153 | 0.1164 | 0.0107 | 0.0112 | 0.0160 | 0.0161 |

d=2, sigma=1, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.3787 | 0.2782 | 0.0082 | 0.0049 | 0.0058 | 0.0103 |
| 1 | 0.3800 | 0.2792 | 0.0055 | 0.0052 | 0.0058 | 0.0085 |
| 2 | 0.3808 | 0.2777 | 0.0046 | 0.0064 | 0.0087 | 0.0104 |
| 3 | 0.3780 | 0.2786 | 0.0053 | 0.0056 | 0.0075 | 0.0086 |
| 4 | 0.3790 | 0.2792 | 0.0050 | 0.0072 | 0.0052 | 0.0104 |
| 5 | 0.3814 | 0.2792 | 0.0053 | 0.0059 | 0.0057 | 0.0108 |
| 6 | 0.3777 | 0.2791 | 0.0045 | 0.0048 | 0.0059 | 0.0094 |
| 7 | 0.3766 | 0.2799 | 0.0050 | 0.0050 | 0.0046 | 0.0100 |
| 8 | 0.3789 | 0.2748 | 0.0056 | 0.0051 | 0.0048 | 0.0093 |
| 9 | 0.3787 | 0.2792 | 0.0069 | 0.0061 | 0.0064 | 0.0132 |

d=2, sigma=1, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.2134 | 0.0123 | 0.0045 | 0.0055 | 0.0058 | 0.0111 |
| 1 | 0.2124 | 0.0114 | 0.0058 | 0.0058 | 0.0081 | 0.0081 |
| 2 | 0.2126 | 0.0118 | 0.0054 | 0.0051 | 0.0073 | 0.0079 |
| 3 | 0.2131 | 0.0123 | 0.0064 | 0.0052 | 0.0062 | 0.0082 |
| 4 | 0.2126 | 0.0117 | 0.0052 | 0.0064 | 0.0062 | 0.0085 |
| 5 | 0.2125 | 0.0118 | 0.0041 | 0.0056 | 0.0053 | 0.0076 |
| 6 | 0.2138 | 0.0130 | 0.0071 | 0.0061 | 0.0068 | 0.0089 |
| 7 | 0.2133 | 0.0124 | 0.0059 | 0.0053 | 0.0067 | 0.0118 |
| 8 | 0.2128 | 0.0118 | 0.0055 | 0.0047 | 0.0074 | 0.0088 |
| 9 | 0.2130 | 0.0121 | 0.0052 | 0.0044 | 0.0057 | 0.0081 |

d=2, sigma=1, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.2146 | 0.0129 | 0.0052 | 0.0062 | 0.0054 | 0.0092 |
| 1 | 0.2132 | 0.0121 | 0.0070 | 0.0057 | 0.0084 | 0.0116 |
| 2 | 0.2132 | 0.0120 | 0.0064 | 0.0077 | 0.0127 | 0.0089 |
| 3 | 0.2137 | 0.0125 | 0.0095 | 0.0064 | 0.0091 | 0.0086 |
| 4 | 0.2127 | 0.0118 | 0.0058 | 0.0058 | 0.0060 | 0.0100 |
| 5 | 0.2124 | 0.0113 | 0.0045 | 0.0051 | 0.0054 | 0.0079 |
| 6 | 0.2141 | 0.0129 | 0.0068 | 0.0064 | 0.0072 | 0.0091 |
| 7 | 0.2131 | 0.0121 | 0.0115 | 0.0063 | 0.0068 | 0.0107 |
| 8 | 0.2131 | 0.0119 | 0.0065 | 0.0060 | 0.0048 | 0.0102 |
| 9 | 0.2128 | 0.0114 | 0.0052 | 0.0065 | 0.0053 | 0.0077 |

d=3, sigma=0.1, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 1.1528 | 0.8894 | 1.0268 | 43.6205 | 0.5872 | 0.1720 |
| 1 | 0.9204 | 0.9406 | 1.0327 | 0.9710 | 0.1745 | 0.2968 |
| 2 | 0.9760 | 0.9361 | 1.1957 | 43.6886 | 0.1639 | 0.2628 |
| 3 | 0.8858 | 1.3235 | 0.9945 | 1.2293 | 0.4417 | 0.3400 |
| 4 | 0.9596 | 1.0487 | 1.0106 | 1.0836 | 0.2307 | 0.2204 |
| 5 | 0.9216 | 0.9479 | 1.0097 | 1.1700 | 0.2465 | 0.2680 |
| 6 | 1.1665 | 0.9285 | 1.0845 | 1.1103 | 0.3476 | 0.1940 |
| 7 | 1.0328 | 0.8397 | 0.9607 | 43.8011 | 0.1969 | 0.5327 |
| 8 | 1.0510 | 0.9579 | 1.0538 | 1.1179 | 0.2117 | 0.5647 |
| 9 | 1.0320 | 0.9529 | 1.2677 | 43.7609 | 0.2583 | 0.3898 |

d=3, sigma=0.1, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 5.3506 | 5.2126 | 5.1944 | 5.1966 | 0.2862 | 5.2008 |
| 1 | 5.3486 | 5.2114 | 5.1878 | 5.1875 | 0.2278 | 5.2107 |
| 2 | 5.3510 | 5.2142 | 5.1954 | 5.1920 | 0.1331 | 5.2136 |
| 3 | 5.3477 | 5.2113 | 5.1874 | 5.1185 | 0.3020 | 5.2102 |
| 4 | 5.3482 | 5.2114 | 5.1856 | 5.1863 | 0.1769 | 5.2107 |
| 5 | 5.3492 | 5.2118 | 5.1885 | 5.1907 | 0.1989 | 5.2111 |
| 6 | 5.3472 | 5.2107 | 5.1909 | 5.1907 | 0.3856 | 5.2096 |
| 7 | 5.3492 | 5.2123 | 5.1882 | 5.1890 | 0.3145 | 5.2115 |
| 8 | 5.3490 | 5.2122 | 5.1927 | 5.1917 | 0.3324 | 5.1999 |
| 9 | 5.3483 | 5.2114 | 5.1879 | 5.1900 | 0.3576 | 5.2107 |

d=3, sigma=0.1, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 5.3496 | 5.2113 | 5.2114 | 5.2114 | 0.4124 | 5.2114 |
| 1 | 5.3481 | 5.2105 | 5.2107 | 5.2106 | 0.7601 | 5.2106 |
| 2 | 5.3509 | 5.2129 | 5.2130 | 5.2129 | 0.5114 | 5.2129 |
| 3 | 5.3481 | 5.2106 | 5.2107 | 5.2106 | 0.4264 | 5.2106 |
| 4 | 5.3482 | 5.2108 | 5.2108 | 5.2109 | 0.1960 | 5.2108 |
| 5 | 5.3488 | 5.2112 | 5.2113 | 5.2112 | 0.2383 | 5.2113 |
| 6 | 5.3476 | 5.2099 | 5.2102 | 5.2099 | 0.3591 | 5.2098 |
| 7 | 5.3490 | 5.2118 | 5.2118 | 5.2118 | 0.1550 | 5.2117 |
| 8 | 5.3494 | 5.2116 | 5.2117 | 5.2116 | 0.2382 | 5.2116 |
| 9 | 5.3492 | 5.2113 | 5.2113 | 5.2113 | 0.4202 | 5.2116 |

d=3, sigma=0.25, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.8488 | 0.7862 | 0.8760 | 0.8785 | 0.1281 | 0.1508 |
| 1 | 0.8086 | 0.7966 | 0.8691 | 0.8890 | 0.0898 | 0.1097 |
| 2 | 0.8071 | 0.8188 | 0.9303 | 5.0968 | 0.0936 | 0.0945 |
| 3 | 0.7878 | 0.8028 | 0.8754 | 0.8634 | 0.1416 | 0.1611 |
| 4 | 0.7892 | 0.7878 | 0.9183 | 5.1130 | 0.0977 | 0.1000 |
| 5 | 0.8490 | 0.8048 | 0.8645 | 0.9577 | 0.1327 | 0.1194 |
| 6 | 0.8234 | 0.8133 | 0.8743 | 0.8743 | 0.1032 | 0.0973 |
| 7 | 0.7956 | 0.7892 | 0.8709 | 5.1481 | 0.1255 | 0.1059 |
| 8 | 0.8497 | 0.7678 | 0.8960 | 5.1144 | 0.0997 | 0.1303 |
| 9 | 0.8077 | 0.8268 | 0.8975 | 5.1636 | 0.1101 | 0.1332 |

d=3, sigma=0.25, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 1.2189 | 1.0776 | 1.0776 | 1.0743 | 0.0987 | 0.1356 |
| 1 | 1.2193 | 1.0781 | 1.0782 | 1.0724 | 0.1200 | 0.1047 |
| 2 | 1.2206 | 1.0787 | 1.0789 | 1.0764 | 0.0954 | 0.1126 |
| 3 | 1.2199 | 1.0781 | 1.0735 | 1.0757 | 0.1131 | 0.1078 |
| 4 | 1.2183 | 1.0772 | 1.0736 | 1.0738 | 0.1409 | 0.1195 |
| 5 | 1.2195 | 1.0780 | 1.0782 | 1.0750 | 0.1345 | 0.0982 |
| 6 | 1.2181 | 1.0769 | 1.0726 | 1.0721 | 0.0813 | 0.1013 |
| 7 | 1.2208 | 1.0791 | 1.0793 | 1.0762 | 0.1452 | 0.1414 |
| 8 | 1.2200 | 1.0783 | 1.0783 | 1.0744 | 0.1288 | 0.1241 |
| 9 | 1.2183 | 1.0772 | 1.0774 | 1.0721 | 0.1326 | 0.1035 |

d=3, sigma=0.25, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 1.2196 | 1.0770 | 1.0661 | 1.0663 | 0.1267 | 1.0770 |
| 1 | 1.2197 | 1.0771 | 1.0636 | 1.0781 | 0.1116 | 1.0771 |
| 2 | 1.2216 | 1.0779 | 1.0685 | 1.0693 | 1.0785 | 1.0779 |
| 3 | 1.2201 | 1.0778 | 1.0618 | 1.0780 | 0.1473 | 0.1610 |
| 4 | 1.2190 | 1.0767 | 1.0769 | 1.0768 | 0.1304 | 1.0768 |
| 5 | 1.2198 | 1.0776 | 1.0777 | 1.0776 | 0.1074 | 1.0777 |
| 6 | 1.2194 | 1.0767 | 1.0769 | 1.0768 | 0.1299 | 0.1269 |
| 7 | 1.2207 | 1.0785 | 1.0651 | 1.0785 | 0.1094 | 0.1284 |
| 8 | 1.2199 | 1.0777 | 1.0652 | 1.0685 | 0.1198 | 1.0777 |
| 9 | 1.2195 | 1.0771 | 1.0772 | 1.0771 | 0.1473 | 1.0773 |

d=3, sigma=0.5, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.7753 | 0.7832 | 0.6297 | 0.8261 | 0.1321 | 0.1282 |
| 1 | 0.7776 | 0.7941 | 0.6252 | 0.6213 | 0.1283 | 0.1535 |
| 2 | 0.7795 | 0.7836 | 0.6323 | 0.6210 | 0.1178 | 0.1468 |
| 3 | 0.7808 | 0.7945 | 0.6329 | 0.6286 | 0.1228 | 0.1481 |
| 4 | 0.7790 | 0.7812 | 0.6339 | 0.8310 | 0.1221 | 0.1347 |
| 5 | 0.7860 | 0.7867 | 0.6250 | 0.6284 | 0.1266 | 0.1307 |
| 6 | 0.7736 | 0.7936 | 0.6215 | 0.6407 | 0.1243 | 0.1255 |
| 7 | 0.7719 | 0.7976 | 0.6250 | 0.8308 | 0.1154 | 0.1365 |
| 8 | 0.7803 | 0.7828 | 0.6137 | 0.6324 | 0.1223 | 0.1267 |
| 9 | 0.7785 | 0.7828 | 0.6297 | 0.8303 | 0.1276 | 0.1448 |

d=3, sigma=0.5, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.2697 | 0.1245 | 0.1247 | 0.1247 | 0.1268 | 0.1244 |
| 1 | 0.2697 | 0.1248 | 0.1250 | 0.1250 | 0.1274 | 0.1248 |
| 2 | 0.2693 | 0.1243 | 0.1244 | 0.1244 | 0.1266 | 0.1244 |
| 3 | 0.2706 | 0.1258 | 0.1258 | 0.1258 | 0.1278 | 0.1259 |
| 4 | 0.2690 | 0.1242 | 0.1242 | 0.1243 | 0.1264 | 0.1243 |
| 5 | 0.2696 | 0.1253 | 0.1252 | 0.1253 | 0.1174 | 0.1272 |
| 6 | 0.2690 | 0.1243 | 0.1245 | 0.1244 | 0.1267 | 0.1243 |
| 7 | 0.2709 | 0.1261 | 0.1263 | 0.1261 | 0.1344 | 0.1298 |
| 8 | 0.2699 | 0.1249 | 0.1251 | 0.1250 | 0.1273 | 0.1250 |
| 9 | 0.2691 | 0.1242 | 0.1244 | 0.1242 | 0.1266 | 0.1243 |

d=3, sigma=0.5, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.2690 | 0.1243 | 0.1244 | 0.1245 | 0.1161 | 0.1244 |
| 1 | 0.2698 | 0.1244 | 0.1245 | 0.1244 | 0.1270 | 0.1308 |
| 2 | 0.2695 | 0.1240 | 0.1241 | 0.1241 | 0.1270 | 0.1275 |
| 3 | 0.2704 | 0.1255 | 0.1255 | 0.1255 | 0.1233 | 0.1285 |
| 4 | 0.2688 | 0.1241 | 0.1241 | 0.1231 | 0.1265 | 0.1241 |
| 5 | 0.2700 | 0.1250 | 0.1250 | 0.1250 | 0.1276 | 0.1250 |
| 6 | 0.2695 | 0.1243 | 0.1245 | 0.1243 | 0.1202 | 0.1243 |
| 7 | 0.2712 | 0.1256 | 0.1244 | 0.1256 | 0.1285 | 0.1256 |
| 8 | 0.2697 | 0.1245 | 0.1246 | 0.1235 | 0.1227 | 0.1245 |
| 9 | 0.2691 | 0.1242 | 0.1242 | 0.1242 | 0.1267 | 0.1242 |

d=3, sigma=1, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.4151 | 0.3462 | 0.1844 | 0.1491 | 0.0686 | 0.0764 |
| 1 | 0.4156 | 0.3579 | 0.1834 | 0.1819 | 0.0680 | 0.0737 |
| 2 | 0.4161 | 0.3585 | 0.1825 | 0.1507 | 0.0659 | 0.0784 |
| 3 | 0.4137 | 0.3432 | 0.1842 | 0.1859 | 0.0637 | 0.0757 |
| 4 | 0.4116 | 0.3577 | 0.1840 | 0.1517 | 0.0732 | 0.0763 |
| 5 | 0.4119 | 0.3452 | 0.1856 | 0.1838 | 0.0663 | 0.0798 |
| 6 | 0.4108 | 0.3406 | 0.1801 | 0.1575 | 0.0659 | 0.0758 |
| 7 | 0.4120 | 0.3406 | 0.1822 | 0.1523 | 0.0684 | 0.0795 |
| 8 | 0.4122 | 0.3437 | 0.1855 | 0.1834 | 0.0685 | 0.0754 |
| 9 | 0.4109 | 0.3426 | 0.1836 | 0.1515 | 0.0672 | 0.0775 |

d=3, sigma=1, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.1535 | 0.0076 | 0.0077 | 0.0077 | 0.0108 | 0.0077 |
| 1 | 0.1536 | 0.0076 | 0.0078 | 0.0077 | 0.0109 | 0.0077 |
| 2 | 0.1525 | 0.0068 | 0.0069 | 0.0068 | 0.0101 | 0.0069 |
| 3 | 0.1543 | 0.0088 | 0.0090 | 0.0088 | 0.0120 | 0.0088 |
| 4 | 0.1533 | 0.0075 | 0.0075 | 0.0075 | 0.0107 | 0.0075 |
| 5 | 0.1539 | 0.0083 | 0.0084 | 0.0083 | 0.0116 | 0.0084 |
| 6 | 0.1533 | 0.0077 | 0.0077 | 0.0078 | 0.0108 | 0.0077 |
| 7 | 0.1546 | 0.0087 | 0.0088 | 0.0087 | 0.0121 | 0.0087 |
| 8 | 0.1533 | 0.0076 | 0.0078 | 0.0077 | 0.0113 | 0.0077 |
| 9 | 0.1534 | 0.0075 | 0.0076 | 0.0075 | 0.0108 | 0.0077 |

d=3, sigma=1, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.1533 | 0.0075 | 0.0076 | 0.0075 | 0.0109 | 0.0076 |
| 1 | 0.1535 | 0.0075 | 0.0077 | 0.0076 | 0.0109 | 0.0075 |
| 2 | 0.1528 | 0.0067 | 0.0069 | 0.0067 | 0.0104 | 0.0068 |
| 3 | 0.1545 | 0.0086 | 0.0087 | 0.0086 | 0.0121 | 0.0086 |
| 4 | 0.1530 | 0.0074 | 0.0075 | 0.0075 | 0.0106 | 0.0074 |
| 5 | 0.1538 | 0.0083 | 0.0083 | 0.0083 | 0.0117 | 0.0083 |
| 6 | 0.1536 | 0.0077 | 0.0078 | 0.0077 | 0.0113 | 0.0078 |
| 7 | 0.1548 | 0.0084 | 0.0085 | 0.0085 | 0.0120 | 0.0085 |
| 8 | 0.1535 | 0.0075 | 0.0078 | 0.0075 | 0.0112 | 0.0075 |
| 9 | 0.1534 | 0.0075 | 0.0076 | 0.0075 | 0.0108 | 0.0076 |

d=4, sigma=0.1, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 1.3435 | 1.5840 | 1.4272 | 1.4363 | 0.6843 | 0.8547 |
| 1 | 1.3866 | 1.4585 | 1.4531 | 1.2155 | 0.5036 | 0.5547 |
| 2 | 1.3097 | 1.4144 | 1.4668 | 1.3039 | 0.9384 | 0.6712 |
| 3 | 1.3044 | 1.3601 | 1.6861 | 1.4242 | 0.5814 | 1.2425 |
| 4 | 1.3913 | 1.7426 | 1.3698 | 1.5203 | 0.8039 | 1.0877 |
| 5 | 1.3134 | 1.2359 | 1.3303 | 1.4308 | 0.8029 | 0.7818 |
| 6 | 1.5163 | 1.5116 | 1.3217 | 1.3749 | 0.6109 | 0.7509 |
| 7 | 1.2761 | 1.2158 | 1.4792 | 1.3363 | 0.6530 | 0.5932 |
| 8 | 1.3538 | 1.5885 | 1.3871 | 1.5803 | 0.5399 | 0.4726 |
| 9 | 1.3054 | 1.3793 | 1.4434 | 1.1853 | 0.5245 | 0.6360 |

d=4, sigma=0.1, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 7.9210 | 7.8128 | 7.7854 | 7.7868 | 0.5885 | 7.7956 |
| 1 | 7.9209 | 7.8124 | 7.7859 | 7.7848 | 0.4569 | 7.8026 |
| 2 | 7.9236 | 7.8145 | 7.7975 | 7.7977 | 0.4155 | 7.8078 |
| 3 | 7.9203 | 7.8113 | 7.7779 | 7.7882 | 0.4534 | 7.8064 |
| 4 | 7.9212 | 7.8126 | 7.7949 | 7.7914 | 1.1101 | 7.8022 |
| 5 | 7.9206 | 7.8124 | 7.7886 | 7.7934 | 0.9780 | 7.8018 |
| 6 | 7.9218 | 7.8136 | 7.7971 | 7.7941 | 0.6346 | 7.8075 |
| 7 | 7.9196 | 7.8113 | 7.7905 | 7.7908 | 0.6224 | 7.7993 |
| 8 | 7.9198 | 7.8115 | 7.7916 | 7.7906 | 0.6047 | 7.7985 |
| 9 | 7.9196 | 7.8115 | 7.7790 | 7.7889 | 0.4452 | 7.7911 |

d=4, sigma=0.1, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 7.9214 | 7.8103 | 7.8105 | 7.8104 | 7.8102 | 7.8104 |
| 1 | 7.9213 | 7.8103 | 7.8104 | 7.8103 | 7.8093 | 7.8103 |
| 2 | 7.9221 | 7.8109 | 7.8109 | 7.8109 | 7.8114 | 7.8109 |
| 3 | 7.9198 | 7.8092 | 7.8095 | 7.8093 | 7.8087 | 7.8093 |
| 4 | 7.9216 | 7.8102 | 7.8103 | 7.8104 | 7.8107 | 7.8102 |
| 5 | 7.9211 | 7.8107 | 7.8108 | 7.8108 | 7.8100 | 7.8107 |
| 6 | 7.9223 | 7.8117 | 7.8122 | 7.8117 | 0.5452 | 7.8117 |
| 7 | 7.9203 | 7.8094 | 7.8094 | 7.8094 | 7.8086 | 7.8094 |
| 8 | 7.9208 | 7.8097 | 7.8100 | 7.8098 | 7.8091 | 7.8098 |
| 9 | 7.9203 | 7.8096 | 7.8103 | 7.8096 | 7.8085 | 7.8096 |

d=4, sigma=0.25, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.9786 | 1.0043 | 1.0026 | 1.0176 | 0.3120 | 0.2885 |
| 1 | 1.0205 | 0.9531 | 1.0241 | 0.9714 | 0.2875 | 0.2707 |
| 2 | 0.9783 | 1.0138 | 0.9885 | 1.0254 | 0.2903 | 0.2742 |
| 3 | 0.9484 | 0.9642 | 1.0403 | 1.0428 | 0.2511 | 0.2710 |
| 4 | 0.9225 | 1.0038 | 1.0595 | 1.0356 | 0.2701 | 0.2663 |
| 5 | 0.9820 | 0.9808 | 1.0519 | 1.0274 | 0.2904 | 0.3449 |
| 6 | 0.9895 | 0.9632 | 1.0483 | 1.0147 | 0.3124 | 0.2801 |
| 7 | 1.0324 | 1.0282 | 1.0185 | 1.0152 | 0.2724 | 0.2922 |
| 8 | 0.9787 | 0.9644 | 1.0433 | 0.9654 | 0.2510 | 0.2647 |
| 9 | 0.9760 | 0.9902 | 0.9920 | 1.0326 | 0.2499 | 0.3026 |

d=4, sigma=0.25, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 1.3582 | 1.2458 | 1.2461 | 1.2427 | 1.2471 | 1.2457 |
| 1 | 1.3577 | 1.2453 | 1.2451 | 1.2441 | 1.2469 | 1.2456 |
| 2 | 1.3598 | 1.2470 | 1.2469 | 1.2471 | 1.2480 | 1.2465 |
| 3 | 1.3572 | 1.2450 | 1.2451 | 1.2420 | 1.2461 | 1.2444 |
| 4 | 1.3574 | 1.2452 | 1.2453 | 1.2428 | 1.2465 | 1.2447 |
| 5 | 1.3585 | 1.2460 | 1.2450 | 1.2431 | 1.2475 | 1.2457 |
| 6 | 1.3584 | 1.2462 | 1.2463 | 1.2459 | 1.2471 | 1.2462 |
| 7 | 1.3576 | 1.2453 | 1.2457 | 1.2425 | 0.3770 | 1.2452 |
| 8 | 1.3581 | 1.2454 | 1.2455 | 1.2440 | 1.2466 | 1.2448 |
| 9 | 1.3581 | 1.2461 | 1.2460 | 1.2446 | 1.2473 | 1.2452 |

d=4, sigma=0.25, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 1.3569 | 1.2441 | 1.2442 | 1.2442 | 1.2462 | 1.2441 |
| 1 | 1.3567 | 1.2439 | 1.2440 | 1.2439 | 1.2456 | 1.2439 |
| 2 | 1.3580 | 1.2446 | 1.2453 | 1.2448 | 1.2469 | 1.2446 |
| 3 | 1.3567 | 1.2437 | 1.2438 | 1.2439 | 1.2454 | 1.2438 |
| 4 | 1.3566 | 1.2434 | 1.2436 | 1.2436 | 1.2457 | 1.2435 |
| 5 | 1.3578 | 1.2446 | 1.2446 | 1.2446 | 1.2470 | 1.2447 |
| 6 | 1.3574 | 1.2446 | 1.2446 | 1.2448 | 1.2465 | 1.2447 |
| 7 | 1.3571 | 1.2439 | 1.2442 | 1.2440 | 1.2459 | 1.2440 |
| 8 | 1.3569 | 1.2437 | 1.2437 | 1.2438 | 1.2454 | 1.2437 |
| 9 | 1.3569 | 1.2440 | 1.2443 | 1.2441 | 1.2460 | 1.2440 |

d=4, sigma=0.5, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.9505 | 0.9456 | 0.7656 | 0.7543 | 0.2564 | 0.2779 |
| 1 | 0.9294 | 0.9494 | 0.7454 | 0.7539 | 0.2562 | 0.2715 |
| 2 | 0.9457 | 0.9302 | 0.7668 | 0.7451 | 0.2586 | 0.2699 |
| 3 | 0.9370 | 0.9348 | 0.7488 | 0.7632 | 0.2572 | 0.2717 |
| 4 | 0.9387 | 0.9435 | 0.7533 | 0.7519 | 0.2564 | 0.2783 |
| 5 | 0.9358 | 0.9327 | 0.7555 | 0.7504 | 0.2610 | 0.2741 |
| 6 | 0.9501 | 0.9359 | 0.7496 | 0.7537 | 0.2547 | 0.2693 |
| 7 | 0.9411 | 0.9459 | 0.7659 | 0.7465 | 0.2712 | 0.2694 |
| 8 | 0.9546 | 0.9395 | 0.7462 | 0.7560 | 0.2484 | 0.2625 |
| 9 | 0.9566 | 0.9416 | 0.7651 | 0.7610 | 0.2614 | 0.2758 |

d=4, sigma=0.5, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.2131 | 0.0994 | 0.0994 | 0.0995 | 0.1019 | 0.0993 |
| 1 | 0.2125 | 0.0991 | 0.0991 | 0.0992 | 0.1017 | 0.0991 |
| 2 | 0.2139 | 0.1002 | 0.1003 | 0.1003 | 0.1030 | 0.1004 |
| 3 | 0.2127 | 0.0991 | 0.0991 | 0.0992 | 0.1015 | 0.0991 |
| 4 | 0.2123 | 0.0989 | 0.0989 | 0.0990 | 0.1014 | 0.0990 |
| 5 | 0.2129 | 0.0995 | 0.0997 | 0.0996 | 0.1019 | 0.0999 |
| 6 | 0.2130 | 0.0995 | 0.0995 | 0.0995 | 0.1019 | 0.0993 |
| 7 | 0.2132 | 0.0999 | 0.0999 | 0.0998 | 0.1023 | 0.0998 |
| 8 | 0.2123 | 0.0987 | 0.0988 | 0.0989 | 0.1013 | 0.0987 |
| 9 | 0.2125 | 0.0993 | 0.0995 | 0.0994 | 0.1020 | 0.0994 |

d=4, sigma=0.5, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.2124 | 0.0987 | 0.0987 | 0.0988 | 0.1016 | 0.0987 |
| 1 | 0.2119 | 0.0982 | 0.0983 | 0.0982 | 0.1010 | 0.0982 |
| 2 | 0.2128 | 0.0991 | 0.0994 | 0.0992 | 0.1020 | 0.0991 |
| 3 | 0.2122 | 0.0985 | 0.0986 | 0.0986 | 0.1011 | 0.0985 |
| 4 | 0.2118 | 0.0979 | 0.0981 | 0.0981 | 0.1006 | 0.0980 |
| 5 | 0.2125 | 0.0983 | 0.0984 | 0.0984 | 0.1015 | 0.0982 |
| 6 | 0.2124 | 0.0983 | 0.0984 | 0.0983 | 0.1013 | 0.0983 |
| 7 | 0.2128 | 0.0984 | 0.0986 | 0.0984 | 0.1015 | 0.0985 |
| 8 | 0.2117 | 0.0978 | 0.0978 | 0.0979 | 0.1008 | 0.0979 |
| 9 | 0.2120 | 0.0982 | 0.0984 | 0.0979 | 0.1008 | 0.0983 |

d=4, sigma=1, fixed

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.4366 | 0.3877 | 0.2114 | 0.2130 | 0.1232 | 0.1310 |
| 1 | 0.4374 | 0.3804 | 0.2142 | 0.2095 | 0.1252 | 0.1346 |
| 2 | 0.4335 | 0.3863 | 0.2121 | 0.2108 | 0.1226 | 0.1323 |
| 3 | 0.4369 | 0.3890 | 0.2111 | 0.2090 | 0.1255 | 0.1341 |
| 4 | 0.4348 | 0.3809 | 0.2116 | 0.2096 | 0.1211 | 0.1341 |
| 5 | 0.4341 | 0.3811 | 0.2093 | 0.2100 | 0.1233 | 0.1331 |
| 6 | 0.4333 | 0.3829 | 0.2120 | 0.2107 | 0.1227 | 0.1353 |
| 7 | 0.4362 | 0.3848 | 0.2113 | 0.2121 | 0.1268 | 0.1332 |
| 8 | 0.4331 | 0.3882 | 0.2091 | 0.2094 | 0.1229 | 0.1337 |
| 9 | 0.4363 | 0.3889 | 0.2132 | 0.2102 | 0.1249 | 0.1357 |

d=4, sigma=1, linear

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.1165 | 0.0026 | 0.0027 | 0.0026 | 0.0056 | 0.0026 |
| 1 | 0.1159 | 0.0020 | 0.0021 | 0.0020 | 0.0049 | 0.0020 |
| 2 | 0.1169 | 0.0033 | 0.0034 | 0.0035 | 0.0060 | 0.0033 |
| 3 | 0.1167 | 0.0027 | 0.0028 | 0.0027 | 0.0055 | 0.0027 |
| 4 | 0.1159 | 0.0020 | 0.0021 | 0.0022 | 0.0050 | 0.0020 |
| 5 | 0.1157 | 0.0018 | 0.0018 | 0.0018 | 0.0046 | 0.0019 |
| 6 | 0.1158 | 0.0019 | 0.0021 | 0.0020 | 0.0046 | 0.0020 |
| 7 | 0.1162 | 0.0025 | 0.0026 | 0.0025 | 0.0052 | 0.0024 |
| 8 | 0.1151 | 0.0015 | 0.0016 | 0.0016 | 0.0046 | 0.0015 |
| 9 | 0.1158 | 0.0020 | 0.0020 | 0.0021 | 0.0047 | 0.0019 |

d=4, sigma=1, mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 | vmf_sym | ps_sym |
|---|---|---|---|---|---|---|
| 0 | 0.1164 | 0.0024 | 0.0025 | 0.0026 | 0.0055 | 0.0025 |
| 1 | 0.1155 | 0.0016 | 0.0017 | 0.0017 | 0.0045 | 0.0016 |
| 2 | 0.1167 | 0.0028 | 0.0029 | 0.0028 | 0.0059 | 0.0028 |
| 3 | 0.1163 | 0.0023 | 0.0024 | 0.0024 | 0.0054 | 0.0023 |
| 4 | 0.1155 | 0.0016 | 0.0020 | 0.0017 | 0.0046 | 0.0016 |
| 5 | 0.1153 | 0.0012 | 0.0013 | 0.0013 | 0.0042 | 0.0012 |
| 6 | 0.1158 | 0.0015 | 0.0016 | 0.0018 | 0.0047 | 0.0015 |
| 7 | 0.1160 | 0.0017 | 0.0018 | 0.0020 | 0.0050 | 0.0018 |
| 8 | 0.1151 | 0.0013 | 0.0017 | 0.0014 | 0.0046 | 0.0013 |
| 9 | 0.1156 | 0.0015 | 0.0016 | 0.0016 | 0.0046 | 0.0015 |

## Confirmatory tests (one-sided paired Wilcoxon, Holm at 0.05)

| d | sigma | decoder | comparator | n | mean paired diff (comparator - tnbbeta) | seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.05 | confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.1 | fixed | power_spherical | 10 | 0.7188 | 10/10 | 0.00098 | 0.00098 | 0.00106 | True | True | True |
| 2 | 0.1 | fixed | vmf | 10 | 0.7523 | 10/10 | 0.00098 | 0.00098 | 0.00104 | True | True | True |
| 2 | 0.1 | linear | power_spherical | 10 | 1.2781 | 10/10 | 0.00098 | 0.00098 | 0.00111 | True | True | True |
| 2 | 0.1 | linear | vmf | 10 | 2.8662 | 10/10 | 0.00098 | 0.00098 | 0.00109 | True | True | True |
| 2 | 0.1 | mlp | power_spherical | 10 | 2.2107 | 10/10 | 0.00098 | 0.00098 | 0.00116 | True | True | True |
| 2 | 0.1 | mlp | vmf | 10 | 2.8033 | 10/10 | 0.00098 | 0.00098 | 0.00114 | True | True | True |
| 2 | 0.25 | fixed | power_spherical | 10 | 0.6789 | 10/10 | 0.00098 | 0.00098 | 0.00122 | True | True | True |
| 2 | 0.25 | fixed | vmf | 10 | 0.6781 | 10/10 | 0.00098 | 0.00098 | 0.00119 | True | True | True |
| 2 | 0.25 | linear | power_spherical | 10 | 0.7028 | 10/10 | 0.00098 | 0.00098 | 0.00128 | True | True | True |
| 2 | 0.25 | linear | vmf | 10 | 0.9051 | 10/10 | 0.00098 | 0.00098 | 0.00125 | True | True | True |
| 2 | 0.25 | mlp | power_spherical | 10 | 0.7011 | 10/10 | 0.00098 | 0.00098 | 0.00135 | True | True | True |
| 2 | 0.25 | mlp | vmf | 10 | 0.8964 | 10/10 | 0.00098 | 0.00098 | 0.00132 | True | True | True |
| 2 | 0.5 | fixed | power_spherical | 10 | 0.5939 | 10/10 | 0.00098 | 0.00098 | 0.00143 | True | True | True |
| 2 | 0.5 | fixed | vmf | 10 | 0.6301 | 10/10 | 0.00098 | 0.00098 | 0.00139 | True | True | True |
| 2 | 0.5 | linear | power_spherical | 10 | 0.1076 | 10/10 | 0.00098 | 0.00098 | 0.00152 | True | True | True |
| 2 | 0.5 | linear | vmf | 10 | 0.3062 | 10/10 | 0.00098 | 0.00098 | 0.00147 | True | True | True |
| 2 | 0.5 | mlp | power_spherical | 10 | 0.1050 | 10/10 | 0.00098 | 0.00098 | 0.00161 | True | True | True |
| 2 | 0.5 | mlp | vmf | 10 | 0.3038 | 10/10 | 0.00098 | 0.00098 | 0.00156 | True | True | True |
| 2 | 1 | fixed | power_spherical | 10 | 0.2729 | 10/10 | 0.00098 | 0.00098 | 0.00172 | True | True | True |
| 2 | 1 | fixed | vmf | 10 | 0.3734 | 10/10 | 0.00098 | 0.00098 | 0.00167 | True | True | True |
| 2 | 1 | linear | power_spherical | 10 | 0.0065 | 10/10 | 0.00098 | 0.00098 | 0.00185 | True | False | False |
| 2 | 1 | linear | vmf | 10 | 0.2075 | 10/10 | 0.00098 | 0.00098 | 0.00179 | True | True | True |
| 2 | 1 | mlp | power_spherical | 10 | 0.0053 | 10/10 | 0.00098 | 0.00098 | 0.00200 | True | False | False |
| 2 | 1 | mlp | vmf | 10 | 0.2064 | 10/10 | 0.00098 | 0.00098 | 0.00192 | True | True | True |
| 3 | 0.1 | fixed | power_spherical | 10 | -0.0871 | 2/10 | 0.95801 | 0.95801 | 0.00714 | False | False | False |
| 3 | 0.1 | fixed | vmf | 10 | -0.0538 | 3/10 | 0.90332 | 0.90332 | 0.00556 | False | False | False |
| 3 | 0.1 | linear | power_spherical | 10 | 0.0220 | 10/10 | 0.00098 | 0.00098 | 0.00217 | True | False | False |
| 3 | 0.1 | linear | vmf | 10 | 0.1590 | 10/10 | 0.00098 | 0.00098 | 0.00208 | True | True | True |
| 3 | 0.1 | mlp | power_spherical | 10 | -0.0001 | 0/10 | 1.00000 | 1.00000 | 0.01250 | False | False | False |
| 3 | 0.1 | mlp | vmf | 10 | 0.1376 | 10/10 | 0.00098 | 0.00098 | 0.00227 | True | True | True |
| 3 | 0.25 | fixed | power_spherical | 10 | -0.0878 | 0/10 | 1.00000 | 1.00000 | 0.02500 | False | False | False |
| 3 | 0.25 | fixed | vmf | 10 | -0.0705 | 0/10 | 1.00000 | 1.00000 | 0.01667 | False | False | False |
| 3 | 0.25 | linear | power_spherical | 10 | 0.0011 | 4/10 | 0.50000 | 0.50000 | 0.00500 | False | False | False |
| 3 | 0.25 | linear | vmf | 10 | 0.1426 | 10/10 | 0.00098 | 0.00098 | 0.00238 | True | True | True |
| 3 | 0.25 | mlp | power_spherical | 10 | 0.0075 | 6/10 | 0.04199 | 0.04199 | 0.00455 | False | False | False |
| 3 | 0.25 | mlp | vmf | 10 | 0.1500 | 10/10 | 0.00098 | 0.00098 | 0.00250 | True | True | True |
| 3 | 0.5 | fixed | power_spherical | 10 | 0.1611 | 10/10 | 0.00098 | 0.00098 | 0.00278 | True | True | True |
| 3 | 0.5 | fixed | vmf | 10 | 0.1514 | 10/10 | 0.00098 | 0.00098 | 0.00263 | True | True | True |
| 3 | 0.5 | linear | power_spherical | 10 | -0.0001 | 1/10 | 0.99805 | 0.99805 | 0.00833 | False | False | False |
| 3 | 0.5 | linear | vmf | 10 | 0.1448 | 10/10 | 0.00098 | 0.00098 | 0.00294 | True | True | True |
| 3 | 0.5 | mlp | power_spherical | 10 | 0.0001 | 3/10 | 0.93457 | 0.93457 | 0.00625 | False | False | False |
| 3 | 0.5 | mlp | vmf | 10 | 0.1452 | 10/10 | 0.00098 | 0.00098 | 0.00313 | True | True | True |
| 3 | 1 | fixed | power_spherical | 10 | 0.1641 | 10/10 | 0.00098 | 0.00098 | 0.00357 | True | True | True |
| 3 | 1 | fixed | vmf | 10 | 0.2294 | 10/10 | 0.00098 | 0.00098 | 0.00333 | True | True | True |
| 3 | 1 | linear | power_spherical | 10 | -0.0001 | 1/10 | 0.99902 | 0.99902 | 0.01000 | False | False | False |
| 3 | 1 | linear | vmf | 10 | 0.1457 | 10/10 | 0.00098 | 0.00098 | 0.00385 | True | True | True |
| 3 | 1 | mlp | power_spherical | 10 | -0.0001 | 0/10 | 1.00000 | 1.00000 | 0.05000 | False | False | False |
| 3 | 1 | mlp | vmf | 10 | 0.1458 | 10/10 | 0.00098 | 0.00098 | 0.00417 | True | True | True |

Confirmed (TNBBeta better, Holm and margin): 34 of 48; comparator better by at least the margin (descriptive): 4.

## Secondary tests vs the sign-symmetrised baselines (own Holm family)

| d | sigma | decoder | comparator | n | mean paired diff (comparator - tnbbeta) | seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.05 | confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.1 | fixed | ps_sym | 10 | 0.0136 | 5/10 | 0.27832 | 0.27832 | 0.00147 | False | False | False |
| 2 | 0.1 | fixed | vmf_sym | 10 | 0.0380 | 8/10 | 0.08008 | 0.08008 | 0.00132 | False | False | False |
| 2 | 0.1 | linear | ps_sym | 10 | 0.0383 | 6/10 | 0.21582 | 0.21582 | 0.00143 | False | False | False |
| 2 | 0.1 | linear | vmf_sym | 10 | 0.0228 | 5/10 | 0.31250 | 0.31250 | 0.00161 | False | False | False |
| 2 | 0.1 | mlp | ps_sym | 10 | -0.0750 | 4/10 | 0.75391 | 0.75391 | 0.00263 | False | False | False |
| 2 | 0.1 | mlp | vmf_sym | 10 | -0.0076 | 5/10 | 0.57715 | 0.57715 | 0.00185 | False | False | False |
| 2 | 0.25 | fixed | ps_sym | 10 | 0.0251 | 5/10 | 0.65234 | 0.65234 | 0.00208 | False | False | False |
| 2 | 0.25 | fixed | vmf_sym | 10 | -0.0112 | 5/10 | 0.68750 | 0.68750 | 0.00227 | False | False | False |
| 2 | 0.25 | linear | ps_sym | 10 | 0.0017 | 5/10 | 0.50000 | 0.50000 | 0.00179 | False | False | False |
| 2 | 0.25 | linear | vmf_sym | 10 | 0.0015 | 3/10 | 0.86230 | 0.86230 | 0.00294 | False | False | False |
| 2 | 0.25 | mlp | ps_sym | 10 | -0.0061 | 4/10 | 0.75391 | 0.75391 | 0.00278 | False | False | False |
| 2 | 0.25 | mlp | vmf_sym | 10 | -0.0021 | 3/10 | 0.68750 | 0.68750 | 0.00238 | False | False | False |
| 2 | 0.5 | fixed | ps_sym | 10 | 0.0021 | 8/10 | 0.08008 | 0.08008 | 0.00135 | False | False | False |
| 2 | 0.5 | fixed | vmf_sym | 10 | -0.0005 | 3/10 | 0.65234 | 0.65234 | 0.00217 | False | False | False |
| 2 | 0.5 | linear | ps_sym | 10 | 0.0046 | 9/10 | 0.00488 | 0.00488 | 0.00116 | False | False | False |
| 2 | 0.5 | linear | vmf_sym | 10 | 0.0008 | 6/10 | 0.27832 | 0.27832 | 0.00152 | False | False | False |
| 2 | 0.5 | mlp | ps_sym | 10 | 0.0033 | 8/10 | 0.01855 | 0.01855 | 0.00125 | False | False | False |
| 2 | 0.5 | mlp | vmf_sym | 10 | -0.0009 | 5/10 | 0.72168 | 0.72168 | 0.00250 | False | False | False |
| 2 | 1 | fixed | ps_sym | 10 | 0.0045 | 10/10 | 0.00098 | 0.00098 | 0.00104 | True | False | False |
| 2 | 1 | fixed | vmf_sym | 10 | 0.0005 | 6/10 | 0.34766 | 0.34766 | 0.00172 | False | False | False |
| 2 | 1 | linear | ps_sym | 10 | 0.0034 | 10/10 | 0.00098 | 0.00098 | 0.00106 | True | False | False |
| 2 | 1 | linear | vmf_sym | 10 | 0.0010 | 8/10 | 0.00488 | 0.00488 | 0.00119 | False | False | False |
| 2 | 1 | mlp | ps_sym | 10 | 0.0026 | 8/10 | 0.00488 | 0.00488 | 0.00122 | False | False | False |
| 2 | 1 | mlp | vmf_sym | 10 | 0.0003 | 7/10 | 0.27832 | 0.27832 | 0.00156 | False | False | False |
| 3 | 0.1 | fixed | ps_sym | 10 | -0.7395 | 0/10 | 1.00000 | 1.00000 | 0.00455 | False | False | False |
| 3 | 0.1 | fixed | vmf_sym | 10 | -0.7778 | 0/10 | 1.00000 | 1.00000 | 0.00417 | False | False | False |
| 3 | 0.1 | linear | ps_sym | 10 | 0.0190 | 10/10 | 0.00098 | 0.00098 | 0.00109 | True | False | False |
| 3 | 0.1 | linear | vmf_sym | 10 | -4.9184 | 0/10 | 1.00000 | 1.00000 | 0.00500 | False | False | False |
| 3 | 0.1 | mlp | ps_sym | 10 | -0.0001 | 2/10 | 0.95801 | 0.95801 | 0.00333 | False | False | False |
| 3 | 0.1 | mlp | vmf_sym | 10 | -4.8396 | 0/10 | 1.00000 | 1.00000 | 0.00556 | False | False | False |
| 3 | 0.25 | fixed | ps_sym | 10 | -0.7670 | 0/10 | 1.00000 | 1.00000 | 0.00714 | False | False | False |
| 3 | 0.25 | fixed | vmf_sym | 10 | -0.7750 | 0/10 | 1.00000 | 1.00000 | 0.00625 | False | False | False |
| 3 | 0.25 | linear | ps_sym | 10 | -0.9619 | 0/10 | 1.00000 | 1.00000 | 0.01000 | False | False | False |
| 3 | 0.25 | linear | vmf_sym | 10 | -0.9577 | 0/10 | 1.00000 | 1.00000 | 0.00833 | False | False | False |
| 3 | 0.25 | mlp | ps_sym | 10 | -0.2741 | 5/10 | 0.61523 | 0.61523 | 0.00192 | False | False | False |
| 3 | 0.25 | mlp | vmf_sym | 10 | -0.8491 | 1/10 | 0.99902 | 0.99902 | 0.00385 | False | False | False |
| 3 | 0.5 | fixed | ps_sym | 10 | -0.4894 | 0/10 | 1.00000 | 1.00000 | 0.01667 | False | False | False |
| 3 | 0.5 | fixed | vmf_sym | 10 | -0.5030 | 0/10 | 1.00000 | 1.00000 | 0.01250 | False | False | False |
| 3 | 0.5 | linear | ps_sym | 10 | 0.0005 | 5/10 | 0.61523 | 0.61523 | 0.00200 | False | False | False |
| 3 | 0.5 | linear | vmf_sym | 10 | 0.0018 | 9/10 | 0.03223 | 0.03223 | 0.00128 | False | False | False |
| 3 | 0.5 | mlp | ps_sym | 10 | 0.0014 | 5/10 | 0.18750 | 0.18750 | 0.00139 | False | False | False |
| 3 | 0.5 | mlp | vmf_sym | 10 | 0.0000 | 6/10 | 0.31250 | 0.31250 | 0.00167 | False | False | False |
| 3 | 1 | fixed | ps_sym | 10 | -0.1067 | 0/10 | 1.00000 | 1.00000 | 0.05000 | False | False | False |
| 3 | 1 | fixed | vmf_sym | 10 | -0.1159 | 0/10 | 1.00000 | 1.00000 | 0.02500 | False | False | False |
| 3 | 1 | linear | ps_sym | 10 | -0.0000 | 3/10 | 0.94727 | 0.94727 | 0.00313 | False | False | False |
| 3 | 1 | linear | vmf_sym | 10 | 0.0032 | 10/10 | 0.00098 | 0.00098 | 0.00111 | True | False | False |
| 3 | 1 | mlp | ps_sym | 10 | -0.0001 | 3/10 | 0.98633 | 0.98633 | 0.00357 | False | False | False |
| 3 | 1 | mlp | vmf_sym | 10 | 0.0033 | 10/10 | 0.00098 | 0.00098 | 0.00114 | True | False | False |

Confirmed (TNBBeta better, Holm and margin): 0 of 48; comparator better by at least the margin (descriptive): 15.

## Exact KL(q || p_model) where the model evidence is exact (d = 2, 3)

| d | sigma | decoder | arm | mean KL to the model's own posterior (seed mean +- sd) | seeds |
|---|---|---|---|---|---|
| 2 | 0.1 | fixed | vmf | 0.8395 +- 0.1344 | 10 |
| 2 | 0.1 | fixed | power_spherical | 0.8060 +- 0.1120 | 10 |
| 2 | 0.1 | fixed | tnbbeta | 0.0872 +- 0.0694 | 10 |
| 2 | 0.1 | fixed | tnbbeta_init0 | 0.1271 +- 0.1408 | 10 |
| 2 | 0.1 | fixed | vmf_sym | 0.1252 +- 0.0650 | 10 |
| 2 | 0.1 | fixed | ps_sym | 0.1008 +- 0.1059 | 10 |
| 2 | 0.1 | linear | vmf | 0.2085 +- 0.0009 | 10 |
| 2 | 0.1 | linear | power_spherical | 0.5439 +- 0.3781 | 10 |
| 2 | 0.1 | linear | tnbbeta | 0.0980 +- 0.0715 | 10 |
| 2 | 0.1 | linear | tnbbeta_init0 | 0.1060 +- 0.0669 | 10 |
| 2 | 0.1 | linear | vmf_sym | 0.1167 +- 0.1072 | 10 |
| 2 | 0.1 | linear | ps_sym | 0.1344 +- 0.1571 | 10 |
| 2 | 0.1 | mlp | vmf | 0.2086 +- 0.0013 | 10 |
| 2 | 0.1 | mlp | power_spherical | 0.1503 +- 0.3169 | 10 |
| 2 | 0.1 | mlp | tnbbeta | 0.1412 +- 0.1964 | 10 |
| 2 | 0.1 | mlp | tnbbeta_init0 | 0.0509 +- 0.0227 | 10 |
| 2 | 0.1 | mlp | vmf_sym | 0.1395 +- 0.1691 | 10 |
| 2 | 0.1 | mlp | ps_sym | 0.0729 +- 0.0782 | 10 |
| 2 | 0.25 | fixed | vmf | 0.7110 +- 0.0170 | 10 |
| 2 | 0.25 | fixed | power_spherical | 0.7118 +- 0.0153 | 10 |
| 2 | 0.25 | fixed | tnbbeta | 0.0329 +- 0.0335 | 10 |
| 2 | 0.25 | fixed | tnbbeta_init0 | 0.0265 +- 0.0133 | 10 |
| 2 | 0.25 | fixed | vmf_sym | 0.0216 +- 0.0108 | 10 |
| 2 | 0.25 | fixed | ps_sym | 0.0580 +- 0.1286 | 10 |
| 2 | 0.25 | linear | vmf | 0.2049 +- 0.0009 | 10 |
| 2 | 0.25 | linear | power_spherical | 0.4295 +- 0.3699 | 10 |
| 2 | 0.25 | linear | tnbbeta | 0.0187 +- 0.0087 | 10 |
| 2 | 0.25 | linear | tnbbeta_init0 | 0.0292 +- 0.0233 | 10 |
| 2 | 0.25 | linear | vmf_sym | 0.0193 +- 0.0204 | 10 |
| 2 | 0.25 | linear | ps_sym | 0.0202 +- 0.0151 | 10 |
| 2 | 0.25 | mlp | vmf | 0.2052 +- 0.0008 | 10 |
| 2 | 0.25 | mlp | power_spherical | 0.1443 +- 0.3042 | 10 |
| 2 | 0.25 | mlp | tnbbeta | 0.0249 +- 0.0149 | 10 |
| 2 | 0.25 | mlp | tnbbeta_init0 | 0.0150 +- 0.0074 | 10 |
| 2 | 0.25 | mlp | vmf_sym | 0.0225 +- 0.0180 | 10 |
| 2 | 0.25 | mlp | ps_sym | 0.0196 +- 0.0074 | 10 |
| 2 | 0.5 | fixed | vmf | 0.6414 +- 0.0026 | 10 |
| 2 | 0.5 | fixed | power_spherical | 0.6053 +- 0.0030 | 10 |
| 2 | 0.5 | fixed | tnbbeta | 0.0114 +- 0.0027 | 10 |
| 2 | 0.5 | fixed | tnbbeta_init0 | 0.0133 +- 0.0076 | 10 |
| 2 | 0.5 | fixed | vmf_sym | 0.0108 +- 0.0056 | 10 |
| 2 | 0.5 | fixed | ps_sym | 0.0134 +- 0.0023 | 10 |
| 2 | 0.5 | linear | vmf | 0.2026 +- 0.0002 | 10 |
| 2 | 0.5 | linear | power_spherical | 0.0000 +- 0.0000 | 10 |
| 2 | 0.5 | linear | tnbbeta | 0.0091 +- 0.0024 | 10 |
| 2 | 0.5 | linear | tnbbeta_init0 | 0.0101 +- 0.0029 | 10 |
| 2 | 0.5 | linear | vmf_sym | 0.0094 +- 0.0029 | 10 |
| 2 | 0.5 | linear | ps_sym | 0.0138 +- 0.0024 | 10 |
| 2 | 0.5 | mlp | vmf | 0.2026 +- 0.0003 | 10 |
| 2 | 0.5 | mlp | power_spherical | 0.0000 +- 0.0000 | 10 |
| 2 | 0.5 | mlp | tnbbeta | 0.0112 +- 0.0033 | 10 |
| 2 | 0.5 | mlp | tnbbeta_init0 | 0.0121 +- 0.0044 | 10 |
| 2 | 0.5 | mlp | vmf_sym | 0.0099 +- 0.0027 | 10 |
| 2 | 0.5 | mlp | ps_sym | 0.0145 +- 0.0045 | 10 |
| 2 | 1 | fixed | vmf | 0.3790 +- 0.0014 | 10 |
| 2 | 1 | fixed | power_spherical | 0.2785 +- 0.0014 | 10 |
| 2 | 1 | fixed | tnbbeta | 0.0056 +- 0.0011 | 10 |
| 2 | 1 | fixed | tnbbeta_init0 | 0.0056 +- 0.0008 | 10 |
| 2 | 1 | fixed | vmf_sym | 0.0060 +- 0.0012 | 10 |
| 2 | 1 | fixed | ps_sym | 0.0101 +- 0.0013 | 10 |
| 2 | 1 | linear | vmf | 0.2019 +- 0.0001 | 10 |
| 2 | 1 | linear | power_spherical | 0.0000 +- 0.0000 | 10 |
| 2 | 1 | linear | tnbbeta | 0.0048 +- 0.0008 | 10 |
| 2 | 1 | linear | tnbbeta_init0 | 0.0047 +- 0.0007 | 10 |
| 2 | 1 | linear | vmf_sym | 0.0059 +- 0.0009 | 10 |
| 2 | 1 | linear | ps_sym | 0.0053 +- 0.0022 | 10 |
| 2 | 1 | mlp | vmf | 0.2022 +- 0.0003 | 10 |
| 2 | 1 | mlp | power_spherical | 0.0000 +- 0.0000 | 10 |
| 2 | 1 | mlp | tnbbeta | 0.0042 +- 0.0017 | 10 |
| 2 | 1 | mlp | tnbbeta_init0 | 0.0049 +- 0.0007 | 10 |
| 2 | 1 | mlp | vmf_sym | 0.0063 +- 0.0019 | 10 |
| 2 | 1 | mlp | ps_sym | 0.0049 +- 0.0020 | 10 |
| 3 | 0.1 | fixed | vmf | 1.0099 +- 0.0958 | 10 |
| 3 | 0.1 | fixed | power_spherical | 0.9765 +- 0.1329 | 10 |
| 3 | 0.1 | fixed | tnbbeta | 1.0637 +- 0.0960 | 10 |
| 3 | 0.1 | fixed | tnbbeta_init0 | 18.1553 +- 22.0008 | 10 |
| 3 | 0.1 | fixed | vmf_sym | 0.2859 +- 0.1354 | 10 |
| 3 | 0.1 | fixed | ps_sym | 0.3241 +- 0.1352 | 10 |
| 3 | 0.1 | linear | vmf | 0.1488 +- 0.0010 | 10 |
| 3 | 0.1 | linear | power_spherical | 0.0006 +- 0.0002 | 10 |
| 3 | 0.1 | linear | tnbbeta | 0.0651 +- 0.0047 | 10 |
| 3 | 0.1 | linear | tnbbeta_init0 | 0.0670 +- 0.0067 | 10 |
| 3 | 0.1 | linear | vmf_sym | 0.2552 +- 0.0780 | 10 |
| 3 | 0.1 | linear | ps_sym | 0.0097 +- 0.0171 | 10 |
| 3 | 0.1 | mlp | vmf | 0.1478 +- 0.0004 | 10 |
| 3 | 0.1 | mlp | power_spherical | 0.0000 +- 0.0000 | 10 |
| 3 | 0.1 | mlp | tnbbeta | 0.0001 +- 0.0001 | 10 |
| 3 | 0.1 | mlp | tnbbeta_init0 | 0.0000 +- 0.0000 | 10 |
| 3 | 0.1 | mlp | vmf_sym | 0.3261 +- 0.1496 | 10 |
| 3 | 0.1 | mlp | ps_sym | 0.0001 +- 0.0001 | 10 |
| 3 | 0.25 | fixed | vmf | 0.8167 +- 0.0247 | 10 |
| 3 | 0.25 | fixed | power_spherical | 0.7994 +- 0.0176 | 10 |
| 3 | 0.25 | fixed | tnbbeta | 0.8872 +- 0.0225 | 10 |
| 3 | 0.25 | fixed | tnbbeta_init0 | 3.0099 +- 2.2321 | 10 |
| 3 | 0.25 | fixed | vmf_sym | 0.1122 +- 0.0183 | 10 |
| 3 | 0.25 | fixed | ps_sym | 0.1202 +- 0.0231 | 10 |
| 3 | 0.25 | linear | vmf | 0.1463 +- 0.0005 | 10 |
| 3 | 0.25 | linear | power_spherical | 0.0003 +- 0.0001 | 10 |
| 3 | 0.25 | linear | tnbbeta | 0.0046 +- 0.0068 | 10 |
| 3 | 0.25 | linear | tnbbeta_init0 | 0.0139 +- 0.0012 | 10 |
| 3 | 0.25 | linear | vmf_sym | 0.1156 +- 0.0200 | 10 |
| 3 | 0.25 | linear | ps_sym | 0.1112 +- 0.0156 | 10 |
| 3 | 0.25 | mlp | vmf | 0.1462 +- 0.0002 | 10 |
| 3 | 0.25 | mlp | power_spherical | 0.0000 +- 0.0000 | 10 |
| 3 | 0.25 | mlp | tnbbeta | 0.0089 +- 0.0076 | 10 |
| 3 | 0.25 | mlp | tnbbeta_init0 | 0.0039 +- 0.0062 | 10 |
| 3 | 0.25 | mlp | vmf_sym | 0.1060 +- 0.0377 | 10 |
| 3 | 0.25 | mlp | ps_sym | 0.0389 +- 0.0629 | 10 |
| 3 | 0.5 | fixed | vmf | 0.7782 +- 0.0040 | 10 |
| 3 | 0.5 | fixed | power_spherical | 0.7880 +- 0.0062 | 10 |
| 3 | 0.5 | fixed | tnbbeta | 0.6269 +- 0.0062 | 10 |
| 3 | 0.5 | fixed | tnbbeta_init0 | 0.7090 +- 0.1039 | 10 |
| 3 | 0.5 | fixed | vmf_sym | 0.1239 +- 0.0050 | 10 |
| 3 | 0.5 | fixed | ps_sym | 0.1375 +- 0.0101 | 10 |
| 3 | 0.5 | linear | vmf | 0.1462 +- 0.0002 | 10 |
| 3 | 0.5 | linear | power_spherical | 0.0002 +- 0.0000 | 10 |
| 3 | 0.5 | linear | tnbbeta | 0.0003 +- 0.0001 | 10 |
| 3 | 0.5 | linear | tnbbeta_init0 | 0.0002 +- 0.0001 | 10 |
| 3 | 0.5 | linear | vmf_sym | 0.0254 +- 0.0458 | 10 |
| 3 | 0.5 | linear | ps_sym | 0.0243 +- 0.0507 | 10 |
| 3 | 0.5 | mlp | vmf | 0.1461 +- 0.0002 | 10 |
| 3 | 0.5 | mlp | power_spherical | 0.0000 +- 0.0000 | 10 |
| 3 | 0.5 | mlp | tnbbeta | 0.0003 +- 0.0007 | 10 |
| 3 | 0.5 | mlp | tnbbeta_init0 | 0.0005 +- 0.0009 | 10 |
| 3 | 0.5 | mlp | vmf_sym | 0.0460 +- 0.0547 | 10 |
| 3 | 0.5 | mlp | ps_sym | 0.0358 +- 0.0576 | 10 |
| 3 | 1 | fixed | vmf | 0.4130 +- 0.0020 | 10 |
| 3 | 1 | fixed | power_spherical | 0.3476 +- 0.0074 | 10 |
| 3 | 1 | fixed | tnbbeta | 0.1835 +- 0.0016 | 10 |
| 3 | 1 | fixed | tnbbeta_init0 | 0.1648 +- 0.0165 | 10 |
| 3 | 1 | fixed | vmf_sym | 0.0676 +- 0.0025 | 10 |
| 3 | 1 | fixed | ps_sym | 0.0768 +- 0.0019 | 10 |
| 3 | 1 | linear | vmf | 0.1461 +- 0.0001 | 10 |
| 3 | 1 | linear | power_spherical | 0.0001 +- 0.0000 | 10 |
| 3 | 1 | linear | tnbbeta | 0.0002 +- 0.0001 | 10 |
| 3 | 1 | linear | tnbbeta_init0 | 0.0001 +- 0.0000 | 10 |
| 3 | 1 | linear | vmf_sym | 0.0036 +- 0.0001 | 10 |
| 3 | 1 | linear | ps_sym | 0.0002 +- 0.0000 | 10 |
| 3 | 1 | mlp | vmf | 0.1461 +- 0.0001 | 10 |
| 3 | 1 | mlp | power_spherical | 0.0000 +- 0.0000 | 10 |
| 3 | 1 | mlp | tnbbeta | 0.0001 +- 0.0001 | 10 |
| 3 | 1 | mlp | tnbbeta_init0 | 0.0001 +- 0.0000 | 10 |
| 3 | 1 | mlp | vmf_sym | 0.0037 +- 0.0001 | 10 |
| 3 | 1 | mlp | ps_sym | 0.0001 +- 0.0000 | 10 |

## Model deficit log p_true(X) - log p_model(X) (d = 2, 3, learned)

| d | sigma | decoder | arm | mean deficit (seed mean +- sd) | seeds |
|---|---|---|---|---|---|
| 2 | 0.1 | linear | vmf | 2.7591 +- 0.0010 | 10 |
| 2 | 0.1 | linear | power_spherical | 0.8356 +- 1.3402 | 10 |
| 2 | 0.1 | linear | tnbbeta | 0.0034 +- 0.0033 | 10 |
| 2 | 0.1 | linear | tnbbeta_init0 | 0.0078 +- 0.0058 | 10 |
| 2 | 0.1 | linear | vmf_sym | 0.0075 +- 0.0081 | 10 |
| 2 | 0.1 | linear | ps_sym | 0.0053 +- 0.0066 | 10 |
| 2 | 0.1 | mlp | vmf | 2.7593 +- 0.0014 | 10 |
| 2 | 0.1 | mlp | power_spherical | 2.2250 +- 1.1664 | 10 |
| 2 | 0.1 | mlp | tnbbeta | 0.0234 +- 0.0266 | 10 |
| 2 | 0.1 | mlp | tnbbeta_init0 | 0.0125 +- 0.0063 | 10 |
| 2 | 0.1 | mlp | vmf_sym | 0.0176 +- 0.0082 | 10 |
| 2 | 0.1 | mlp | ps_sym | 0.0167 +- 0.0087 | 10 |
| 2 | 0.25 | linear | vmf | 0.7192 +- 0.0009 | 10 |
| 2 | 0.25 | linear | power_spherical | 0.2922 +- 0.3769 | 10 |
| 2 | 0.25 | linear | tnbbeta | 0.0002 +- 0.0009 | 10 |
| 2 | 0.25 | linear | tnbbeta_init0 | 0.0010 +- 0.0029 | 10 |
| 2 | 0.25 | linear | vmf_sym | 0.0012 +- 0.0028 | 10 |
| 2 | 0.25 | linear | ps_sym | 0.0005 +- 0.0014 | 10 |
| 2 | 0.25 | mlp | vmf | 0.7195 +- 0.0009 | 10 |
| 2 | 0.25 | mlp | power_spherical | 0.5850 +- 0.3070 | 10 |
| 2 | 0.25 | mlp | tnbbeta | 0.0033 +- 0.0024 | 10 |
| 2 | 0.25 | mlp | tnbbeta_init0 | 0.0033 +- 0.0032 | 10 |
| 2 | 0.25 | mlp | vmf_sym | 0.0036 +- 0.0031 | 10 |
| 2 | 0.25 | mlp | ps_sym | 0.0025 +- 0.0028 | 10 |
| 2 | 0.5 | linear | vmf | 0.1131 +- 0.0006 | 10 |
| 2 | 0.5 | linear | power_spherical | 0.1171 +- 0.0005 | 10 |
| 2 | 0.5 | linear | tnbbeta | 0.0004 +- 0.0007 | 10 |
| 2 | 0.5 | linear | tnbbeta_init0 | 0.0003 +- 0.0007 | 10 |
| 2 | 0.5 | linear | vmf_sym | 0.0009 +- 0.0006 | 10 |
| 2 | 0.5 | linear | ps_sym | 0.0003 +- 0.0005 | 10 |
| 2 | 0.5 | mlp | vmf | 0.1136 +- 0.0008 | 10 |
| 2 | 0.5 | mlp | power_spherical | 0.1174 +- 0.0006 | 10 |
| 2 | 0.5 | mlp | tnbbeta | 0.0012 +- 0.0012 | 10 |
| 2 | 0.5 | mlp | tnbbeta_init0 | 0.0010 +- 0.0013 | 10 |
| 2 | 0.5 | mlp | vmf_sym | 0.0015 +- 0.0011 | 10 |
| 2 | 0.5 | mlp | ps_sym | 0.0012 +- 0.0008 | 10 |
| 2 | 1 | linear | vmf | 0.0110 +- 0.0005 | 10 |
| 2 | 1 | linear | power_spherical | 0.0120 +- 0.0005 | 10 |
| 2 | 1 | linear | tnbbeta | 0.0007 +- 0.0005 | 10 |
| 2 | 1 | linear | tnbbeta_init0 | 0.0007 +- 0.0004 | 10 |
| 2 | 1 | linear | vmf_sym | 0.0007 +- 0.0004 | 10 |
| 2 | 1 | linear | ps_sym | 0.0036 +- 0.0028 | 10 |
| 2 | 1 | mlp | vmf | 0.0111 +- 0.0006 | 10 |
| 2 | 1 | mlp | power_spherical | 0.0121 +- 0.0006 | 10 |
| 2 | 1 | mlp | tnbbeta | 0.0026 +- 0.0036 | 10 |
| 2 | 1 | mlp | tnbbeta_init0 | 0.0013 +- 0.0007 | 10 |
| 2 | 1 | mlp | vmf_sym | 0.0008 +- 0.0008 | 10 |
| 2 | 1 | mlp | ps_sym | 0.0045 +- 0.0020 | 10 |
| 3 | 0.1 | linear | vmf | 5.2001 +- 0.0015 | 10 |
| 3 | 0.1 | linear | power_spherical | 5.2114 +- 0.0010 | 10 |
| 3 | 0.1 | linear | tnbbeta | 5.1248 +- 0.0071 | 10 |
| 3 | 0.1 | linear | tnbbeta_init0 | 5.1163 +- 0.0198 | 10 |
| 3 | 0.1 | linear | vmf_sym | 0.0163 +- 0.0061 | 10 |
| 3 | 0.1 | linear | ps_sym | 5.1992 +- 0.0216 | 10 |
| 3 | 0.1 | mlp | vmf | 5.2011 +- 0.0012 | 10 |
| 3 | 0.1 | mlp | power_spherical | 5.2112 +- 0.0008 | 10 |
| 3 | 0.1 | mlp | tnbbeta | 5.2112 +- 0.0008 | 10 |
| 3 | 0.1 | mlp | tnbbeta_init0 | 5.2112 +- 0.0008 | 10 |
| 3 | 0.1 | mlp | vmf_sym | 0.0457 +- 0.0333 | 10 |
| 3 | 0.1 | mlp | ps_sym | 5.2112 +- 0.0008 | 10 |
| 3 | 0.25 | linear | vmf | 1.0731 +- 0.0010 | 10 |
| 3 | 0.25 | linear | power_spherical | 1.0776 +- 0.0008 | 10 |
| 3 | 0.25 | linear | tnbbeta | 1.0721 +- 0.0093 | 10 |
| 3 | 0.25 | linear | tnbbeta_init0 | 1.0604 +- 0.0022 | 10 |
| 3 | 0.25 | linear | vmf_sym | 0.0035 +- 0.0027 | 10 |
| 3 | 0.25 | linear | ps_sym | 0.0037 +- 0.0023 | 10 |
| 3 | 0.25 | mlp | vmf | 1.0737 +- 0.0007 | 10 |
| 3 | 0.25 | mlp | power_spherical | 1.0774 +- 0.0006 | 10 |
| 3 | 0.25 | mlp | tnbbeta | 1.0610 +- 0.0140 | 10 |
| 3 | 0.25 | mlp | tnbbeta_init0 | 1.0708 +- 0.0109 | 10 |
| 3 | 0.25 | mlp | vmf_sym | 0.1149 +- 0.3372 | 10 |
| 3 | 0.25 | mlp | ps_sym | 0.7569 +- 0.5158 | 10 |
| 3 | 0.5 | linear | vmf | 0.1235 +- 0.0006 | 10 |
| 3 | 0.5 | linear | power_spherical | 0.1247 +- 0.0007 | 10 |
| 3 | 0.5 | linear | tnbbeta | 0.1247 +- 0.0007 | 10 |
| 3 | 0.5 | linear | tnbbeta_init0 | 0.1247 +- 0.0007 | 10 |
| 3 | 0.5 | linear | vmf_sym | 0.1013 +- 0.0463 | 10 |
| 3 | 0.5 | linear | ps_sym | 0.1012 +- 0.0491 | 10 |
| 3 | 0.5 | mlp | vmf | 0.1236 +- 0.0006 | 10 |
| 3 | 0.5 | mlp | power_spherical | 0.1246 +- 0.0006 | 10 |
| 3 | 0.5 | mlp | tnbbeta | 0.1242 +- 0.0009 | 10 |
| 3 | 0.5 | mlp | tnbbeta_init0 | 0.1239 +- 0.0016 | 10 |
| 3 | 0.5 | mlp | vmf_sym | 0.0785 +- 0.0582 | 10 |
| 3 | 0.5 | mlp | ps_sym | 0.0901 +- 0.0555 | 10 |
| 3 | 1 | linear | vmf | 0.0075 +- 0.0006 | 10 |
| 3 | 1 | linear | power_spherical | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | linear | tnbbeta | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | linear | tnbbeta_init0 | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | linear | vmf_sym | 0.0075 +- 0.0006 | 10 |
| 3 | 1 | linear | ps_sym | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | mlp | vmf | 0.0075 +- 0.0006 | 10 |
| 3 | 1 | mlp | power_spherical | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | mlp | tnbbeta | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | mlp | tnbbeta_init0 | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | mlp | vmf_sym | 0.0075 +- 0.0006 | 10 |
| 3 | 1 | mlp | ps_sym | 0.0077 +- 0.0006 | 10 |

## Importance-weighted (K = 64) gap

| d | sigma | decoder | arm | mean IWAE gap (seed mean +- sd) | seeds |
|---|---|---|---|---|---|
| 2 | 0.1 | fixed | vmf | 0.7729 +- 0.0869 | 10 |
| 2 | 0.1 | fixed | power_spherical | 0.7301 +- 0.0460 | 10 |
| 2 | 0.1 | fixed | tnbbeta | 0.0328 +- 0.0489 | 10 |
| 2 | 0.1 | fixed | tnbbeta_init0 | 0.0534 +- 0.0734 | 10 |
| 2 | 0.1 | fixed | vmf_sym | 0.0638 +- 0.0325 | 10 |
| 2 | 0.1 | fixed | ps_sym | 0.0407 +- 0.0641 | 10 |
| 2 | 0.1 | linear | vmf | 2.7665 +- 0.0009 | 10 |
| 2 | 0.1 | linear | power_spherical | 1.3436 +- 0.9901 | 10 |
| 2 | 0.1 | linear | tnbbeta | 0.0498 +- 0.0412 | 10 |
| 2 | 0.1 | linear | tnbbeta_init0 | 0.0582 +- 0.0471 | 10 |
| 2 | 0.1 | linear | vmf_sym | 0.0630 +- 0.0600 | 10 |
| 2 | 0.1 | linear | ps_sym | 0.0711 +- 0.1011 | 10 |
| 2 | 0.1 | mlp | vmf | 2.7667 +- 0.0013 | 10 |
| 2 | 0.1 | mlp | power_spherical | 2.3688 +- 0.8632 | 10 |
| 2 | 0.1 | mlp | tnbbeta | 0.0871 +- 0.1386 | 10 |
| 2 | 0.1 | mlp | tnbbeta_init0 | 0.0283 +- 0.0169 | 10 |
| 2 | 0.1 | mlp | vmf_sym | 0.0845 +- 0.0967 | 10 |
| 2 | 0.1 | mlp | ps_sym | 0.0502 +- 0.0452 | 10 |
| 2 | 0.25 | fixed | vmf | 0.6908 +- 0.0034 | 10 |
| 2 | 0.25 | fixed | power_spherical | 0.6940 +- 0.0042 | 10 |
| 2 | 0.25 | fixed | tnbbeta | 0.0053 +- 0.0058 | 10 |
| 2 | 0.25 | fixed | tnbbeta_init0 | 0.0066 +- 0.0047 | 10 |
| 2 | 0.25 | fixed | vmf_sym | 0.0074 +- 0.0052 | 10 |
| 2 | 0.25 | fixed | ps_sym | 0.0173 +- 0.0439 | 10 |
| 2 | 0.25 | linear | vmf | 0.7259 +- 0.0008 | 10 |
| 2 | 0.25 | linear | power_spherical | 0.7100 +- 0.0184 | 10 |
| 2 | 0.25 | linear | tnbbeta | 0.0033 +- 0.0032 | 10 |
| 2 | 0.25 | linear | tnbbeta_init0 | 0.0069 +- 0.0083 | 10 |
| 2 | 0.25 | linear | vmf_sym | 0.0059 +- 0.0097 | 10 |
| 2 | 0.25 | linear | ps_sym | 0.0045 +- 0.0063 | 10 |
| 2 | 0.25 | mlp | vmf | 0.7262 +- 0.0008 | 10 |
| 2 | 0.25 | mlp | power_spherical | 0.7258 +- 0.0113 | 10 |
| 2 | 0.25 | mlp | tnbbeta | 0.0053 +- 0.0035 | 10 |
| 2 | 0.25 | mlp | tnbbeta_init0 | 0.0049 +- 0.0045 | 10 |
| 2 | 0.25 | mlp | vmf_sym | 0.0094 +- 0.0083 | 10 |
| 2 | 0.25 | mlp | ps_sym | 0.0064 +- 0.0057 | 10 |
| 2 | 0.5 | fixed | vmf | 0.4010 +- 0.0073 | 10 |
| 2 | 0.5 | fixed | power_spherical | 0.4873 +- 0.0190 | 10 |
| 2 | 0.5 | fixed | tnbbeta | 0.0006 +- 0.0007 | 10 |
| 2 | 0.5 | fixed | tnbbeta_init0 | 0.0014 +- 0.0010 | 10 |
| 2 | 0.5 | fixed | vmf_sym | 0.0012 +- 0.0013 | 10 |
| 2 | 0.5 | fixed | ps_sym | 0.0013 +- 0.0006 | 10 |
| 2 | 0.5 | linear | vmf | 0.1191 +- 0.0006 | 10 |
| 2 | 0.5 | linear | power_spherical | 0.1171 +- 0.0005 | 10 |
| 2 | 0.5 | linear | tnbbeta | 0.0014 +- 0.0007 | 10 |
| 2 | 0.5 | linear | tnbbeta_init0 | 0.0012 +- 0.0009 | 10 |
| 2 | 0.5 | linear | vmf_sym | 0.0018 +- 0.0008 | 10 |
| 2 | 0.5 | linear | ps_sym | 0.0015 +- 0.0005 | 10 |
| 2 | 0.5 | mlp | vmf | 0.1195 +- 0.0008 | 10 |
| 2 | 0.5 | mlp | power_spherical | 0.1174 +- 0.0006 | 10 |
| 2 | 0.5 | mlp | tnbbeta | 0.0027 +- 0.0020 | 10 |
| 2 | 0.5 | mlp | tnbbeta_init0 | 0.0024 +- 0.0014 | 10 |
| 2 | 0.5 | mlp | vmf_sym | 0.0021 +- 0.0011 | 10 |
| 2 | 0.5 | mlp | ps_sym | 0.0026 +- 0.0012 | 10 |
| 2 | 1 | fixed | vmf | 0.0228 +- 0.0039 | 10 |
| 2 | 1 | fixed | power_spherical | 0.0064 +- 0.0082 | 10 |
| 2 | 1 | fixed | tnbbeta | 0.0003 +- 0.0006 | 10 |
| 2 | 1 | fixed | tnbbeta_init0 | 0.0003 +- 0.0004 | 10 |
| 2 | 1 | fixed | vmf_sym | 0.0003 +- 0.0004 | 10 |
| 2 | 1 | fixed | ps_sym | 0.0003 +- 0.0004 | 10 |
| 2 | 1 | linear | vmf | 0.0165 +- 0.0005 | 10 |
| 2 | 1 | linear | power_spherical | 0.0120 +- 0.0005 | 10 |
| 2 | 1 | linear | tnbbeta | 0.0006 +- 0.0006 | 10 |
| 2 | 1 | linear | tnbbeta_init0 | 0.0009 +- 0.0005 | 10 |
| 2 | 1 | linear | vmf_sym | 0.0010 +- 0.0006 | 10 |
| 2 | 1 | linear | ps_sym | 0.0035 +- 0.0029 | 10 |
| 2 | 1 | mlp | vmf | 0.0165 +- 0.0006 | 10 |
| 2 | 1 | mlp | power_spherical | 0.0121 +- 0.0006 | 10 |
| 2 | 1 | mlp | tnbbeta | 0.0026 +- 0.0037 | 10 |
| 2 | 1 | mlp | tnbbeta_init0 | 0.0015 +- 0.0008 | 10 |
| 2 | 1 | mlp | vmf_sym | 0.0010 +- 0.0006 | 10 |
| 2 | 1 | mlp | ps_sym | 0.0045 +- 0.0021 | 10 |
| 3 | 0.1 | fixed | vmf | 0.8207 +- 0.0647 | 10 |
| 3 | 0.1 | fixed | power_spherical | 0.7975 +- 0.0764 | 10 |
| 3 | 0.1 | fixed | tnbbeta | 0.7921 +- 0.0512 | 10 |
| 3 | 0.1 | fixed | tnbbeta_init0 | 0.5963 +- 0.3146 | 10 |
| 3 | 0.1 | fixed | vmf_sym | 0.0999 +- 0.0838 | 10 |
| 3 | 0.1 | fixed | ps_sym | 0.1187 +- 0.0764 | 10 |
| 3 | 0.1 | linear | vmf | 5.2027 +- 0.0015 | 10 |
| 3 | 0.1 | linear | power_spherical | 5.2114 +- 0.0011 | 10 |
| 3 | 0.1 | linear | tnbbeta | 5.1257 +- 0.0071 | 10 |
| 3 | 0.1 | linear | tnbbeta_init0 | 5.1171 +- 0.0191 | 10 |
| 3 | 0.1 | linear | vmf_sym | 0.1103 +- 0.0515 | 10 |
| 3 | 0.1 | linear | ps_sym | 5.1992 +- 0.0216 | 10 |
| 3 | 0.1 | mlp | vmf | 5.2036 +- 0.0012 | 10 |
| 3 | 0.1 | mlp | power_spherical | 5.2112 +- 0.0008 | 10 |
| 3 | 0.1 | mlp | tnbbeta | 5.2112 +- 0.0008 | 10 |
| 3 | 0.1 | mlp | tnbbeta_init0 | 5.2112 +- 0.0008 | 10 |
| 3 | 0.1 | mlp | vmf_sym | 0.1622 +- 0.1023 | 10 |
| 3 | 0.1 | mlp | ps_sym | 5.2112 +- 0.0008 | 10 |
| 3 | 0.25 | fixed | vmf | 0.7188 +- 0.0135 | 10 |
| 3 | 0.25 | fixed | power_spherical | 0.7118 +- 0.0073 | 10 |
| 3 | 0.25 | fixed | tnbbeta | 0.7081 +- 0.0111 | 10 |
| 3 | 0.25 | fixed | tnbbeta_init0 | 0.3745 +- 0.3462 | 10 |
| 3 | 0.25 | fixed | vmf_sym | 0.0173 +- 0.0088 | 10 |
| 3 | 0.25 | fixed | ps_sym | 0.0199 +- 0.0090 | 10 |
| 3 | 0.25 | linear | vmf | 1.0750 +- 0.0009 | 10 |
| 3 | 0.25 | linear | power_spherical | 1.0776 +- 0.0008 | 10 |
| 3 | 0.25 | linear | tnbbeta | 1.0723 +- 0.0091 | 10 |
| 3 | 0.25 | linear | tnbbeta_init0 | 1.0604 +- 0.0025 | 10 |
| 3 | 0.25 | linear | vmf_sym | 0.0230 +- 0.0079 | 10 |
| 3 | 0.25 | linear | ps_sym | 0.0283 +- 0.0100 | 10 |
| 3 | 0.25 | mlp | vmf | 1.0755 +- 0.0007 | 10 |
| 3 | 0.25 | mlp | power_spherical | 1.0774 +- 0.0006 | 10 |
| 3 | 0.25 | mlp | tnbbeta | 1.0609 +- 0.0141 | 10 |
| 3 | 0.25 | mlp | tnbbeta_init0 | 1.0709 +- 0.0108 | 10 |
| 3 | 0.25 | mlp | vmf_sym | 0.1325 +- 0.3311 | 10 |
| 3 | 0.25 | mlp | ps_sym | 0.7650 +- 0.5029 | 10 |
| 3 | 0.5 | fixed | vmf | 0.5404 +- 0.0063 | 10 |
| 3 | 0.5 | fixed | power_spherical | 0.5835 +- 0.0083 | 10 |
| 3 | 0.5 | fixed | tnbbeta | 0.0646 +- 0.0087 | 10 |
| 3 | 0.5 | fixed | tnbbeta_init0 | 0.0399 +- 0.0250 | 10 |
| 3 | 0.5 | fixed | vmf_sym | 0.0070 +- 0.0029 | 10 |
| 3 | 0.5 | fixed | ps_sym | 0.0131 +- 0.0042 | 10 |
| 3 | 0.5 | linear | vmf | 0.1251 +- 0.0007 | 10 |
| 3 | 0.5 | linear | power_spherical | 0.1247 +- 0.0007 | 10 |
| 3 | 0.5 | linear | tnbbeta | 0.1247 +- 0.0007 | 10 |
| 3 | 0.5 | linear | tnbbeta_init0 | 0.1247 +- 0.0007 | 10 |
| 3 | 0.5 | linear | vmf_sym | 0.1022 +- 0.0446 | 10 |
| 3 | 0.5 | linear | ps_sym | 0.1026 +- 0.0460 | 10 |
| 3 | 0.5 | mlp | vmf | 0.1250 +- 0.0006 | 10 |
| 3 | 0.5 | mlp | power_spherical | 0.1246 +- 0.0006 | 10 |
| 3 | 0.5 | mlp | tnbbeta | 0.1242 +- 0.0010 | 10 |
| 3 | 0.5 | mlp | tnbbeta_init0 | 0.1239 +- 0.0017 | 10 |
| 3 | 0.5 | mlp | vmf_sym | 0.0801 +- 0.0563 | 10 |
| 3 | 0.5 | mlp | ps_sym | 0.0920 +- 0.0524 | 10 |
| 3 | 1 | fixed | vmf | 0.0312 +- 0.0040 | 10 |
| 3 | 1 | fixed | power_spherical | 0.0067 +- 0.0019 | 10 |
| 3 | 1 | fixed | tnbbeta | 0.0018 +- 0.0010 | 10 |
| 3 | 1 | fixed | tnbbeta_init0 | 0.0013 +- 0.0017 | 10 |
| 3 | 1 | fixed | vmf_sym | 0.0024 +- 0.0015 | 10 |
| 3 | 1 | fixed | ps_sym | 0.0019 +- 0.0012 | 10 |
| 3 | 1 | linear | vmf | 0.0089 +- 0.0007 | 10 |
| 3 | 1 | linear | power_spherical | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | linear | tnbbeta | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | linear | tnbbeta_init0 | 0.0078 +- 0.0006 | 10 |
| 3 | 1 | linear | vmf_sym | 0.0075 +- 0.0006 | 10 |
| 3 | 1 | linear | ps_sym | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | mlp | vmf | 0.0089 +- 0.0006 | 10 |
| 3 | 1 | mlp | power_spherical | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | mlp | tnbbeta | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | mlp | tnbbeta_init0 | 0.0077 +- 0.0006 | 10 |
| 3 | 1 | mlp | vmf_sym | 0.0074 +- 0.0006 | 10 |
| 3 | 1 | mlp | ps_sym | 0.0077 +- 0.0006 | 10 |
| 4 | 0.1 | fixed | vmf | 0.9338 +- 0.0570 | 10 |
| 4 | 0.1 | fixed | power_spherical | 0.9851 +- 0.1026 | 10 |
| 4 | 0.1 | fixed | tnbbeta | 0.9454 +- 0.0558 | 10 |
| 4 | 0.1 | fixed | tnbbeta_init0 | 0.9214 +- 0.0885 | 10 |
| 4 | 0.1 | fixed | vmf_sym | 0.2555 +- 0.1031 | 10 |
| 4 | 0.1 | fixed | ps_sym | 0.3143 +- 0.1483 | 10 |
| 4 | 0.1 | linear | vmf | 7.8100 +- 0.0014 | 10 |
| 4 | 0.1 | linear | power_spherical | 7.8107 +- 0.0012 | 10 |
| 4 | 0.1 | linear | tnbbeta | 7.7276 +- 0.0111 | 10 |
| 4 | 0.1 | linear | tnbbeta_init0 | 7.7306 +- 0.0050 | 10 |
| 4 | 0.1 | linear | vmf_sym | 0.2829 +- 0.1513 | 10 |
| 4 | 0.1 | linear | ps_sym | 7.7529 +- 0.0094 | 10 |
| 4 | 0.1 | mlp | vmf | 7.8111 +- 0.0009 | 10 |
| 4 | 0.1 | mlp | power_spherical | 7.8102 +- 0.0007 | 10 |
| 4 | 0.1 | mlp | tnbbeta | 7.8103 +- 0.0007 | 10 |
| 4 | 0.1 | mlp | tnbbeta_init0 | 7.8102 +- 0.0008 | 10 |
| 4 | 0.1 | mlp | vmf_sym | 7.0479 +- 2.3949 | 10 |
| 4 | 0.1 | mlp | ps_sym | 7.8102 +- 0.0008 | 10 |
| 4 | 0.25 | fixed | vmf | 0.7521 +- 0.0170 | 10 |
| 4 | 0.25 | fixed | power_spherical | 0.7639 +- 0.0111 | 10 |
| 4 | 0.25 | fixed | tnbbeta | 0.7414 +- 0.0141 | 10 |
| 4 | 0.25 | fixed | tnbbeta_init0 | 0.7406 +- 0.0131 | 10 |
| 4 | 0.25 | fixed | vmf_sym | 0.0540 +- 0.0135 | 10 |
| 4 | 0.25 | fixed | ps_sym | 0.0638 +- 0.0126 | 10 |
| 4 | 0.25 | linear | vmf | 1.2482 +- 0.0008 | 10 |
| 4 | 0.25 | linear | power_spherical | 1.2448 +- 0.0006 | 10 |
| 4 | 0.25 | linear | tnbbeta | 1.2425 +- 0.0047 | 10 |
| 4 | 0.25 | linear | tnbbeta_init0 | 1.2315 +- 0.0026 | 10 |
| 4 | 0.25 | linear | vmf_sym | 1.1292 +- 0.3601 | 10 |
| 4 | 0.25 | linear | ps_sym | 1.2434 +- 0.0006 | 10 |
| 4 | 0.25 | mlp | vmf | 1.2479 +- 0.0006 | 10 |
| 4 | 0.25 | mlp | power_spherical | 1.2441 +- 0.0004 | 10 |
| 4 | 0.25 | mlp | tnbbeta | 1.2441 +- 0.0005 | 10 |
| 4 | 0.25 | mlp | tnbbeta_init0 | 1.2441 +- 0.0004 | 10 |
| 4 | 0.25 | mlp | vmf_sym | 1.2428 +- 0.0007 | 10 |
| 4 | 0.25 | mlp | ps_sym | 1.2440 +- 0.0004 | 10 |
| 4 | 0.5 | fixed | vmf | 0.4997 +- 0.0087 | 10 |
| 4 | 0.5 | fixed | power_spherical | 0.6230 +- 0.0109 | 10 |
| 4 | 0.5 | fixed | tnbbeta | 0.0824 +- 0.0084 | 10 |
| 4 | 0.5 | fixed | tnbbeta_init0 | 0.0842 +- 0.0111 | 10 |
| 4 | 0.5 | fixed | vmf_sym | 0.0196 +- 0.0036 | 10 |
| 4 | 0.5 | fixed | ps_sym | 0.0270 +- 0.0057 | 10 |
| 4 | 0.5 | linear | vmf | 0.1033 +- 0.0006 | 10 |
| 4 | 0.5 | linear | power_spherical | 0.0989 +- 0.0005 | 10 |
| 4 | 0.5 | linear | tnbbeta | 0.0988 +- 0.0004 | 10 |
| 4 | 0.5 | linear | tnbbeta_init0 | 0.0988 +- 0.0005 | 10 |
| 4 | 0.5 | linear | vmf_sym | 0.0984 +- 0.0006 | 10 |
| 4 | 0.5 | linear | ps_sym | 0.0989 +- 0.0004 | 10 |
| 4 | 0.5 | mlp | vmf | 0.1031 +- 0.0004 | 10 |
| 4 | 0.5 | mlp | power_spherical | 0.0983 +- 0.0004 | 10 |
| 4 | 0.5 | mlp | tnbbeta | 0.0984 +- 0.0004 | 10 |
| 4 | 0.5 | mlp | tnbbeta_init0 | 0.0980 +- 0.0008 | 10 |
| 4 | 0.5 | mlp | vmf_sym | 0.0981 +- 0.0004 | 10 |
| 4 | 0.5 | mlp | ps_sym | 0.0983 +- 0.0004 | 10 |
| 4 | 1 | fixed | vmf | 0.0182 +- 0.0038 | 10 |
| 4 | 1 | fixed | power_spherical | 0.0079 +- 0.0024 | 10 |
| 4 | 1 | fixed | tnbbeta | 0.0025 +- 0.0017 | 10 |
| 4 | 1 | fixed | tnbbeta_init0 | 0.0021 +- 0.0010 | 10 |
| 4 | 1 | fixed | vmf_sym | 0.0022 +- 0.0012 | 10 |
| 4 | 1 | fixed | ps_sym | 0.0036 +- 0.0013 | 10 |
| 4 | 1 | linear | vmf | 0.0068 +- 0.0005 | 10 |
| 4 | 1 | linear | power_spherical | 0.0019 +- 0.0005 | 10 |
| 4 | 1 | linear | tnbbeta | 0.0019 +- 0.0005 | 10 |
| 4 | 1 | linear | tnbbeta_init0 | 0.0020 +- 0.0006 | 10 |
| 4 | 1 | linear | vmf_sym | 0.0020 +- 0.0006 | 10 |
| 4 | 1 | linear | ps_sym | 0.0020 +- 0.0006 | 10 |
| 4 | 1 | mlp | vmf | 0.0066 +- 0.0006 | 10 |
| 4 | 1 | mlp | power_spherical | 0.0018 +- 0.0005 | 10 |
| 4 | 1 | mlp | tnbbeta | 0.0018 +- 0.0005 | 10 |
| 4 | 1 | mlp | tnbbeta_init0 | 0.0018 +- 0.0005 | 10 |
| 4 | 1 | mlp | vmf_sym | 0.0019 +- 0.0005 | 10 |
| 4 | 1 | mlp | ps_sym | 0.0018 +- 0.0005 | 10 |

## Descriptive paired differences and the ln 2 prediction

| d | sigma | decoder | comparison | n | mean gap (A) | mean gap (B) | mean paired diff (A - B) | seeds A better | IWAE paired diff (A - B) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.1 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 0.0872 | 0.1271 | -0.0399 | 5/10 | -0.0206 |
| 2 | 0.1 | fixed | tnbbeta vs vmf | 10 | 0.0872 | 0.8395 | -0.7523 | 10/10 | -0.7401 |
| 2 | 0.1 | fixed | tnbbeta vs power_spherical | 10 | 0.0872 | 0.8060 | -0.7188 | 10/10 | -0.6973 |
| 2 | 0.1 | fixed | tnbbeta vs vmf_sym | 10 | 0.0872 | 0.1252 | -0.0380 | 8/10 | -0.0309 |
| 2 | 0.1 | fixed | tnbbeta vs ps_sym | 10 | 0.0872 | 0.1008 | -0.0136 | 5/10 | -0.0078 |
| 2 | 0.1 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 0.1271 | 0.1252 | 0.0019 | 6/10 | -0.0104 |
| 2 | 0.1 | fixed | vmf vs vmf_sym | 10 | 0.8395 | 0.1252 | 0.7143 | 0/10 | 0.7091 |
| 2 | 0.1 | fixed | power_spherical vs ps_sym | 10 | 0.8060 | 0.1008 | 0.7052 | 0/10 | 0.6895 |
| 2 | 0.1 | fixed | vmf vs power_spherical | 10 | 0.8395 | 0.8060 | 0.0335 | 3/10 | 0.0428 |
| 2 | 0.1 | linear | tnbbeta vs tnbbeta_init0 | 10 | 0.1014 | 0.1138 | -0.0124 | 7/10 | -0.0084 |
| 2 | 0.1 | linear | tnbbeta vs vmf | 10 | 0.1014 | 2.9676 | -2.8662 | 10/10 | -2.7167 |
| 2 | 0.1 | linear | tnbbeta vs power_spherical | 10 | 0.1014 | 1.3795 | -1.2781 | 10/10 | -1.2937 |
| 2 | 0.1 | linear | tnbbeta vs vmf_sym | 10 | 0.1014 | 0.1241 | -0.0228 | 5/10 | -0.0132 |
| 2 | 0.1 | linear | tnbbeta vs ps_sym | 10 | 0.1014 | 0.1397 | -0.0383 | 6/10 | -0.0213 |
| 2 | 0.1 | linear | tnbbeta_init0 vs vmf_sym | 10 | 0.1138 | 0.1241 | -0.0103 | 3/10 | -0.0048 |
| 2 | 0.1 | linear | vmf vs vmf_sym | 10 | 2.9676 | 0.1241 | 2.8434 | 0/10 | 2.7035 |
| 2 | 0.1 | linear | power_spherical vs ps_sym | 10 | 1.3795 | 0.1397 | 1.2398 | 0/10 | 1.2724 |
| 2 | 0.1 | linear | vmf vs power_spherical | 10 | 2.9676 | 1.3795 | 1.5880 | 0/10 | 1.4230 |
| 2 | 0.1 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.1646 | 0.0634 | 0.1012 | 3/10 | 0.0588 |
| 2 | 0.1 | mlp | tnbbeta vs vmf | 10 | 0.1646 | 2.9679 | -2.8033 | 10/10 | -2.6797 |
| 2 | 0.1 | mlp | tnbbeta vs power_spherical | 10 | 0.1646 | 2.3753 | -2.2107 | 10/10 | -2.2818 |
| 2 | 0.1 | mlp | tnbbeta vs vmf_sym | 10 | 0.1646 | 0.1570 | 0.0076 | 5/10 | 0.0026 |
| 2 | 0.1 | mlp | tnbbeta vs ps_sym | 10 | 0.1646 | 0.0896 | 0.0750 | 4/10 | 0.0369 |
| 2 | 0.1 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 0.0634 | 0.1570 | -0.0936 | 8/10 | -0.0562 |
| 2 | 0.1 | mlp | vmf vs vmf_sym | 10 | 2.9679 | 0.1570 | 2.8109 | 0/10 | 2.6823 |
| 2 | 0.1 | mlp | power_spherical vs ps_sym | 10 | 2.3753 | 0.0896 | 2.2857 | 0/10 | 2.3187 |
| 2 | 0.1 | mlp | vmf vs power_spherical | 10 | 2.9679 | 2.3753 | 0.5926 | 0/10 | 0.3979 |
| 2 | 0.25 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 0.0329 | 0.0265 | 0.0064 | 6/10 | -0.0013 |
| 2 | 0.25 | fixed | tnbbeta vs vmf | 10 | 0.0329 | 0.7110 | -0.6781 | 10/10 | -0.6855 |
| 2 | 0.25 | fixed | tnbbeta vs power_spherical | 10 | 0.0329 | 0.7118 | -0.6789 | 10/10 | -0.6887 |
| 2 | 0.25 | fixed | tnbbeta vs vmf_sym | 10 | 0.0329 | 0.0216 | 0.0112 | 5/10 | -0.0021 |
| 2 | 0.25 | fixed | tnbbeta vs ps_sym | 10 | 0.0329 | 0.0580 | -0.0251 | 5/10 | -0.0120 |
| 2 | 0.25 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 0.0265 | 0.0216 | 0.0049 | 5/10 | -0.0009 |
| 2 | 0.25 | fixed | vmf vs vmf_sym | 10 | 0.7110 | 0.0216 | 0.6893 | 0/10 | 0.6834 |
| 2 | 0.25 | fixed | power_spherical vs ps_sym | 10 | 0.7118 | 0.0580 | 0.6538 | 0/10 | 0.6767 |
| 2 | 0.25 | fixed | vmf vs power_spherical | 10 | 0.7110 | 0.7118 | -0.0008 | 6/10 | -0.0032 |
| 2 | 0.25 | linear | tnbbeta vs tnbbeta_init0 | 10 | 0.0189 | 0.0301 | -0.0112 | 8/10 | -0.0037 |
| 2 | 0.25 | linear | tnbbeta vs vmf | 10 | 0.0189 | 0.9240 | -0.9051 | 10/10 | -0.7226 |
| 2 | 0.25 | linear | tnbbeta vs power_spherical | 10 | 0.0189 | 0.7217 | -0.7028 | 10/10 | -0.7067 |
| 2 | 0.25 | linear | tnbbeta vs vmf_sym | 10 | 0.0189 | 0.0205 | -0.0015 | 3/10 | -0.0026 |
| 2 | 0.25 | linear | tnbbeta vs ps_sym | 10 | 0.0189 | 0.0207 | -0.0017 | 5/10 | -0.0012 |
| 2 | 0.25 | linear | tnbbeta_init0 vs vmf_sym | 10 | 0.0301 | 0.0205 | 0.0097 | 3/10 | 0.0010 |
| 2 | 0.25 | linear | vmf vs vmf_sym | 10 | 0.9240 | 0.0205 | 0.9036 | 0/10 | 0.7200 |
| 2 | 0.25 | linear | power_spherical vs ps_sym | 10 | 0.7217 | 0.0207 | 0.7011 | 0/10 | 0.7055 |
| 2 | 0.25 | linear | vmf vs power_spherical | 10 | 0.9240 | 0.7217 | 0.2023 | 0/10 | 0.0159 |
| 2 | 0.25 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.0283 | 0.0183 | 0.0100 | 3/10 | 0.0004 |
| 2 | 0.25 | mlp | tnbbeta vs vmf | 10 | 0.0283 | 0.9246 | -0.8964 | 10/10 | -0.7209 |
| 2 | 0.25 | mlp | tnbbeta vs power_spherical | 10 | 0.0283 | 0.7293 | -0.7011 | 10/10 | -0.7205 |
| 2 | 0.25 | mlp | tnbbeta vs vmf_sym | 10 | 0.0283 | 0.0261 | 0.0021 | 3/10 | -0.0041 |
| 2 | 0.25 | mlp | tnbbeta vs ps_sym | 10 | 0.0283 | 0.0221 | 0.0061 | 4/10 | -0.0012 |
| 2 | 0.25 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 0.0183 | 0.0261 | -0.0078 | 4/10 | -0.0045 |
| 2 | 0.25 | mlp | vmf vs vmf_sym | 10 | 0.9246 | 0.0261 | 0.8985 | 0/10 | 0.7168 |
| 2 | 0.25 | mlp | power_spherical vs ps_sym | 10 | 0.7293 | 0.0221 | 0.7072 | 0/10 | 0.7193 |
| 2 | 0.25 | mlp | vmf vs power_spherical | 10 | 0.9246 | 0.7293 | 0.1953 | 0/10 | 0.0004 |
| 2 | 0.5 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 0.0114 | 0.0133 | -0.0019 | 4/10 | -0.0007 |
| 2 | 0.5 | fixed | tnbbeta vs vmf | 10 | 0.0114 | 0.6414 | -0.6301 | 10/10 | -0.4004 |
| 2 | 0.5 | fixed | tnbbeta vs power_spherical | 10 | 0.0114 | 0.6053 | -0.5939 | 10/10 | -0.4867 |
| 2 | 0.5 | fixed | tnbbeta vs vmf_sym | 10 | 0.0114 | 0.0108 | 0.0005 | 3/10 | -0.0006 |
| 2 | 0.5 | fixed | tnbbeta vs ps_sym | 10 | 0.0114 | 0.0134 | -0.0021 | 8/10 | -0.0007 |
| 2 | 0.5 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 0.0133 | 0.0108 | 0.0025 | 5/10 | 0.0002 |
| 2 | 0.5 | fixed | vmf vs vmf_sym | 10 | 0.6414 | 0.0108 | 0.6306 | 0/10 | 0.3999 |
| 2 | 0.5 | fixed | power_spherical vs ps_sym | 10 | 0.6053 | 0.0134 | 0.5919 | 0/10 | 0.4860 |
| 2 | 0.5 | fixed | vmf vs power_spherical | 10 | 0.6414 | 0.6053 | 0.0361 | 0/10 | -0.0863 |
| 2 | 0.5 | linear | tnbbeta vs tnbbeta_init0 | 10 | 0.0095 | 0.0105 | -0.0010 | 7/10 | 0.0001 |
| 2 | 0.5 | linear | tnbbeta vs vmf | 10 | 0.0095 | 0.3157 | -0.3062 | 10/10 | -0.1177 |
| 2 | 0.5 | linear | tnbbeta vs power_spherical | 10 | 0.0095 | 0.1171 | -0.1076 | 10/10 | -0.1157 |
| 2 | 0.5 | linear | tnbbeta vs vmf_sym | 10 | 0.0095 | 0.0103 | -0.0008 | 6/10 | -0.0004 |
| 2 | 0.5 | linear | tnbbeta vs ps_sym | 10 | 0.0095 | 0.0141 | -0.0046 | 9/10 | -0.0001 |
| 2 | 0.5 | linear | tnbbeta_init0 vs vmf_sym | 10 | 0.0105 | 0.0103 | 0.0002 | 4/10 | -0.0005 |
| 2 | 0.5 | linear | vmf vs vmf_sym | 10 | 0.3157 | 0.0103 | 0.3055 | 0/10 | 0.1173 |
| 2 | 0.5 | linear | power_spherical vs ps_sym | 10 | 0.1171 | 0.0141 | 0.1030 | 0/10 | 0.1156 |
| 2 | 0.5 | linear | vmf vs power_spherical | 10 | 0.3157 | 0.1171 | 0.1986 | 0/10 | 0.0020 |
| 2 | 0.5 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.0124 | 0.0131 | -0.0007 | 7/10 | 0.0003 |
| 2 | 0.5 | mlp | tnbbeta vs vmf | 10 | 0.0124 | 0.3162 | -0.3038 | 10/10 | -0.1168 |
| 2 | 0.5 | mlp | tnbbeta vs power_spherical | 10 | 0.0124 | 0.1174 | -0.1050 | 10/10 | -0.1147 |
| 2 | 0.5 | mlp | tnbbeta vs vmf_sym | 10 | 0.0124 | 0.0115 | 0.0009 | 5/10 | 0.0006 |
| 2 | 0.5 | mlp | tnbbeta vs ps_sym | 10 | 0.0124 | 0.0157 | -0.0033 | 8/10 | 0.0001 |
| 2 | 0.5 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 0.0131 | 0.0115 | 0.0016 | 3/10 | 0.0003 |
| 2 | 0.5 | mlp | vmf vs vmf_sym | 10 | 0.3162 | 0.0115 | 0.3047 | 0/10 | 0.1173 |
| 2 | 0.5 | mlp | power_spherical vs ps_sym | 10 | 0.1174 | 0.0157 | 0.1016 | 0/10 | 0.1148 |
| 2 | 0.5 | mlp | vmf vs power_spherical | 10 | 0.3162 | 0.1174 | 0.1988 | 0/10 | 0.0021 |
| 2 | 1 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 0.0056 | 0.0056 | -0.0001 | 6/10 | -0.0000 |
| 2 | 1 | fixed | tnbbeta vs vmf | 10 | 0.0056 | 0.3790 | -0.3734 | 10/10 | -0.0225 |
| 2 | 1 | fixed | tnbbeta vs power_spherical | 10 | 0.0056 | 0.2785 | -0.2729 | 10/10 | -0.0061 |
| 2 | 1 | fixed | tnbbeta vs vmf_sym | 10 | 0.0056 | 0.0060 | -0.0005 | 6/10 | -0.0000 |
| 2 | 1 | fixed | tnbbeta vs ps_sym | 10 | 0.0056 | 0.0101 | -0.0045 | 10/10 | -0.0000 |
| 2 | 1 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 0.0056 | 0.0060 | -0.0004 | 6/10 | -0.0000 |
| 2 | 1 | fixed | vmf vs vmf_sym | 10 | 0.3790 | 0.0060 | 0.3729 | 0/10 | 0.0224 |
| 2 | 1 | fixed | power_spherical vs ps_sym | 10 | 0.2785 | 0.0101 | 0.2684 | 0/10 | 0.0060 |
| 2 | 1 | fixed | vmf vs power_spherical | 10 | 0.3790 | 0.2785 | 0.1005 | 0/10 | 0.0164 |
| 2 | 1 | linear | tnbbeta vs tnbbeta_init0 | 10 | 0.0055 | 0.0054 | 0.0001 | 3/10 | -0.0003 |
| 2 | 1 | linear | tnbbeta vs vmf | 10 | 0.0055 | 0.2130 | -0.2075 | 10/10 | -0.0159 |
| 2 | 1 | linear | tnbbeta vs power_spherical | 10 | 0.0055 | 0.0121 | -0.0065 | 10/10 | -0.0114 |
| 2 | 1 | linear | tnbbeta vs vmf_sym | 10 | 0.0055 | 0.0065 | -0.0010 | 8/10 | -0.0004 |
| 2 | 1 | linear | tnbbeta vs ps_sym | 10 | 0.0055 | 0.0089 | -0.0034 | 10/10 | -0.0030 |
| 2 | 1 | linear | tnbbeta_init0 vs vmf_sym | 10 | 0.0054 | 0.0065 | -0.0011 | 8/10 | -0.0001 |
| 2 | 1 | linear | vmf vs vmf_sym | 10 | 0.2130 | 0.0065 | 0.2064 | 0/10 | 0.0155 |
| 2 | 1 | linear | power_spherical vs ps_sym | 10 | 0.0121 | 0.0089 | 0.0032 | 0/10 | 0.0085 |
| 2 | 1 | linear | vmf vs power_spherical | 10 | 0.2130 | 0.0121 | 0.2009 | 0/10 | 0.0045 |
| 2 | 1 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.0068 | 0.0062 | 0.0006 | 4/10 | 0.0011 |
| 2 | 1 | mlp | tnbbeta vs vmf | 10 | 0.0068 | 0.2133 | -0.2064 | 10/10 | -0.0139 |
| 2 | 1 | mlp | tnbbeta vs power_spherical | 10 | 0.0068 | 0.0121 | -0.0053 | 10/10 | -0.0095 |
| 2 | 1 | mlp | tnbbeta vs vmf_sym | 10 | 0.0068 | 0.0071 | -0.0003 | 7/10 | 0.0016 |
| 2 | 1 | mlp | tnbbeta vs ps_sym | 10 | 0.0068 | 0.0094 | -0.0026 | 8/10 | -0.0019 |
| 2 | 1 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 0.0062 | 0.0071 | -0.0009 | 7/10 | 0.0006 |
| 2 | 1 | mlp | vmf vs vmf_sym | 10 | 0.2133 | 0.0071 | 0.2062 | 0/10 | 0.0155 |
| 2 | 1 | mlp | power_spherical vs ps_sym | 10 | 0.0121 | 0.0094 | 0.0027 | 0/10 | 0.0075 |
| 2 | 1 | mlp | vmf vs power_spherical | 10 | 0.2133 | 0.0121 | 0.2012 | 0/10 | 0.0044 |
| 3 | 0.1 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 1.0637 | 18.1553 | -17.0917 | 9/10 | 0.1959 |
| 3 | 0.1 | fixed | tnbbeta vs vmf | 10 | 1.0637 | 1.0099 | 0.0538 | 3/10 | -0.0285 |
| 3 | 0.1 | fixed | tnbbeta vs power_spherical | 10 | 1.0637 | 0.9765 | 0.0871 | 2/10 | -0.0054 |
| 3 | 0.1 | fixed | tnbbeta vs vmf_sym | 10 | 1.0637 | 0.2859 | 0.7778 | 0/10 | 0.6922 |
| 3 | 0.1 | fixed | tnbbeta vs ps_sym | 10 | 1.0637 | 0.3241 | 0.7395 | 0/10 | 0.6734 |
| 3 | 0.1 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 18.1553 | 0.2859 | 17.8694 | 0/10 | 0.4964 |
| 3 | 0.1 | fixed | vmf vs vmf_sym | 10 | 1.0099 | 0.2859 | 0.7240 | 0/10 | 0.7208 |
| 3 | 0.1 | fixed | power_spherical vs ps_sym | 10 | 0.9765 | 0.3241 | 0.6524 | 0/10 | 0.6788 |
| 3 | 0.1 | fixed | vmf vs power_spherical | 10 | 1.0099 | 0.9765 | 0.0333 | 4/10 | 0.0231 |
| 3 | 0.1 | linear | tnbbeta vs tnbbeta_init0 | 10 | 5.1899 | 5.1833 | 0.0066 | 5/10 | 0.0085 |
| 3 | 0.1 | linear | tnbbeta vs vmf | 10 | 5.1899 | 5.3489 | -0.1590 | 10/10 | -0.0770 |
| 3 | 0.1 | linear | tnbbeta vs power_spherical | 10 | 5.1899 | 5.2119 | -0.0220 | 10/10 | -0.0857 |
| 3 | 0.1 | linear | tnbbeta vs vmf_sym | 10 | 5.1899 | 0.2715 | 4.9184 | 0/10 | 5.0154 |
| 3 | 0.1 | linear | tnbbeta vs ps_sym | 10 | 5.1899 | 5.2089 | -0.0190 | 10/10 | -0.0735 |
| 3 | 0.1 | linear | tnbbeta_init0 vs vmf_sym | 10 | 5.1833 | 0.2715 | 4.9118 | 0/10 | 5.0068 |
| 3 | 0.1 | linear | vmf vs vmf_sym | 10 | 5.3489 | 0.2715 | 5.0774 | 0/10 | 5.0924 |
| 3 | 0.1 | linear | power_spherical vs ps_sym | 10 | 5.2119 | 5.2089 | 0.0030 | 0/10 | 0.0122 |
| 3 | 0.1 | linear | vmf vs power_spherical | 10 | 5.3489 | 5.2119 | 0.1370 | 0/10 | -0.0087 |
| 3 | 0.1 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 5.2113 | 5.2112 | 0.0001 | 2/10 | -0.0000 |
| 3 | 0.1 | mlp | tnbbeta vs vmf | 10 | 5.2113 | 5.3489 | -0.1376 | 10/10 | 0.0075 |
| 3 | 0.1 | mlp | tnbbeta vs power_spherical | 10 | 5.2113 | 5.2112 | 0.0001 | 0/10 | -0.0000 |
| 3 | 0.1 | mlp | tnbbeta vs vmf_sym | 10 | 5.2113 | 0.3717 | 4.8396 | 0/10 | 5.0490 |
| 3 | 0.1 | mlp | tnbbeta vs ps_sym | 10 | 5.2113 | 5.2112 | 0.0001 | 2/10 | -0.0000 |
| 3 | 0.1 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 5.2112 | 0.3717 | 4.8395 | 0/10 | 5.0490 |
| 3 | 0.1 | mlp | vmf vs vmf_sym | 10 | 5.3489 | 0.3717 | 4.9772 | 0/10 | 5.0414 |
| 3 | 0.1 | mlp | power_spherical vs ps_sym | 10 | 5.2112 | 5.2112 | -0.0000 | 8/10 | -0.0000 |
| 3 | 0.1 | mlp | vmf vs power_spherical | 10 | 5.3489 | 5.2112 | 0.1377 | 0/10 | -0.0075 |
| 3 | 0.25 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 0.8872 | 3.0099 | -2.1227 | 8/10 | 0.3337 |
| 3 | 0.25 | fixed | tnbbeta vs vmf | 10 | 0.8872 | 0.8167 | 0.0705 | 0/10 | -0.0107 |
| 3 | 0.25 | fixed | tnbbeta vs power_spherical | 10 | 0.8872 | 0.7994 | 0.0878 | 0/10 | -0.0037 |
| 3 | 0.25 | fixed | tnbbeta vs vmf_sym | 10 | 0.8872 | 0.1122 | 0.7750 | 0/10 | 0.6908 |
| 3 | 0.25 | fixed | tnbbeta vs ps_sym | 10 | 0.8872 | 0.1202 | 0.7670 | 0/10 | 0.6882 |
| 3 | 0.25 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 3.0099 | 0.1122 | 2.8977 | 0/10 | 0.3572 |
| 3 | 0.25 | fixed | vmf vs vmf_sym | 10 | 0.8167 | 0.1122 | 0.7045 | 0/10 | 0.7015 |
| 3 | 0.25 | fixed | power_spherical vs ps_sym | 10 | 0.7994 | 0.1202 | 0.6792 | 0/10 | 0.6919 |
| 3 | 0.25 | fixed | vmf vs power_spherical | 10 | 0.8167 | 0.7994 | 0.0173 | 3/10 | 0.0070 |
| 3 | 0.25 | linear | tnbbeta vs tnbbeta_init0 | 10 | 1.0768 | 1.0742 | 0.0025 | 2/10 | 0.0119 |
| 3 | 0.25 | linear | tnbbeta vs vmf | 10 | 1.0768 | 1.2194 | -0.1426 | 10/10 | -0.0027 |
| 3 | 0.25 | linear | tnbbeta vs power_spherical | 10 | 1.0768 | 1.0779 | -0.0011 | 4/10 | -0.0053 |
| 3 | 0.25 | linear | tnbbeta vs vmf_sym | 10 | 1.0768 | 0.1190 | 0.9577 | 0/10 | 1.0493 |
| 3 | 0.25 | linear | tnbbeta vs ps_sym | 10 | 1.0768 | 0.1149 | 0.9619 | 0/10 | 1.0440 |
| 3 | 0.25 | linear | tnbbeta_init0 vs vmf_sym | 10 | 1.0742 | 0.1190 | 0.9552 | 0/10 | 1.0374 |
| 3 | 0.25 | linear | vmf vs vmf_sym | 10 | 1.2194 | 0.1190 | 1.1003 | 0/10 | 1.0520 |
| 3 | 0.25 | linear | power_spherical vs ps_sym | 10 | 1.0779 | 0.1149 | 0.9630 | 0/10 | 1.0493 |
| 3 | 0.25 | linear | vmf vs power_spherical | 10 | 1.2194 | 1.0779 | 0.1415 | 0/10 | -0.0026 |
| 3 | 0.25 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 1.0699 | 1.0747 | -0.0048 | 6/10 | -0.0099 |
| 3 | 0.25 | mlp | tnbbeta vs vmf | 10 | 1.0699 | 1.2199 | -0.1500 | 10/10 | -0.0145 |
| 3 | 0.25 | mlp | tnbbeta vs power_spherical | 10 | 1.0699 | 1.0774 | -0.0075 | 6/10 | -0.0165 |
| 3 | 0.25 | mlp | tnbbeta vs vmf_sym | 10 | 1.0699 | 0.2208 | 0.8491 | 1/10 | 0.9285 |
| 3 | 0.25 | mlp | tnbbeta vs ps_sym | 10 | 1.0699 | 0.7958 | 0.2741 | 5/10 | 0.2959 |
| 3 | 0.25 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 1.0747 | 0.2208 | 0.8539 | 1/10 | 0.9384 |
| 3 | 0.25 | mlp | vmf vs vmf_sym | 10 | 1.2199 | 0.2208 | 0.9991 | 0/10 | 0.9430 |
| 3 | 0.25 | mlp | power_spherical vs ps_sym | 10 | 1.0774 | 0.7958 | 0.2816 | 5/10 | 0.3124 |
| 3 | 0.25 | mlp | vmf vs power_spherical | 10 | 1.2199 | 1.0774 | 0.1425 | 0/10 | -0.0019 |
| 3 | 0.5 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 0.6269 | 0.7090 | -0.0822 | 7/10 | 0.0247 |
| 3 | 0.5 | fixed | tnbbeta vs vmf | 10 | 0.6269 | 0.7782 | -0.1514 | 10/10 | -0.4758 |
| 3 | 0.5 | fixed | tnbbeta vs power_spherical | 10 | 0.6269 | 0.7880 | -0.1611 | 10/10 | -0.5189 |
| 3 | 0.5 | fixed | tnbbeta vs vmf_sym | 10 | 0.6269 | 0.1239 | 0.5030 | 0/10 | 0.0576 |
| 3 | 0.5 | fixed | tnbbeta vs ps_sym | 10 | 0.6269 | 0.1375 | 0.4894 | 0/10 | 0.0515 |
| 3 | 0.5 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 0.7090 | 0.1239 | 0.5851 | 0/10 | 0.0329 |
| 3 | 0.5 | fixed | vmf vs vmf_sym | 10 | 0.7782 | 0.1239 | 0.6543 | 0/10 | 0.5334 |
| 3 | 0.5 | fixed | power_spherical vs ps_sym | 10 | 0.7880 | 0.1375 | 0.6505 | 0/10 | 0.5704 |
| 3 | 0.5 | fixed | vmf vs power_spherical | 10 | 0.7782 | 0.7880 | -0.0098 | 10/10 | -0.0431 |
| 3 | 0.5 | linear | tnbbeta vs tnbbeta_init0 | 10 | 0.1250 | 0.1249 | 0.0000 | 4/10 | 0.0000 |
| 3 | 0.5 | linear | tnbbeta vs vmf | 10 | 0.1250 | 0.2697 | -0.1448 | 10/10 | -0.0004 |
| 3 | 0.5 | linear | tnbbeta vs power_spherical | 10 | 0.1250 | 0.1248 | 0.0001 | 1/10 | 0.0000 |
| 3 | 0.5 | linear | tnbbeta vs vmf_sym | 10 | 0.1250 | 0.1267 | -0.0018 | 9/10 | 0.0225 |
| 3 | 0.5 | linear | tnbbeta vs ps_sym | 10 | 0.1250 | 0.1254 | -0.0005 | 5/10 | 0.0221 |
| 3 | 0.5 | linear | tnbbeta_init0 vs vmf_sym | 10 | 0.1249 | 0.1267 | -0.0018 | 9/10 | 0.0225 |
| 3 | 0.5 | linear | vmf vs vmf_sym | 10 | 0.2697 | 0.1267 | 0.1430 | 0/10 | 0.0229 |
| 3 | 0.5 | linear | power_spherical vs ps_sym | 10 | 0.1248 | 0.1254 | -0.0006 | 7/10 | 0.0221 |
| 3 | 0.5 | linear | vmf vs power_spherical | 10 | 0.2697 | 0.1248 | 0.1449 | 0/10 | 0.0004 |
| 3 | 0.5 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.1245 | 0.1244 | 0.0001 | 5/10 | 0.0003 |
| 3 | 0.5 | mlp | tnbbeta vs vmf | 10 | 0.1245 | 0.2697 | -0.1452 | 10/10 | -0.0008 |
| 3 | 0.5 | mlp | tnbbeta vs power_spherical | 10 | 0.1245 | 0.1246 | -0.0001 | 3/10 | -0.0004 |
| 3 | 0.5 | mlp | tnbbeta vs vmf_sym | 10 | 0.1245 | 0.1246 | -0.0000 | 6/10 | 0.0441 |
| 3 | 0.5 | mlp | tnbbeta vs ps_sym | 10 | 0.1245 | 0.1259 | -0.0014 | 5/10 | 0.0322 |
| 3 | 0.5 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 0.1244 | 0.1246 | -0.0001 | 6/10 | 0.0438 |
| 3 | 0.5 | mlp | vmf vs vmf_sym | 10 | 0.2697 | 0.1246 | 0.1451 | 0/10 | 0.0449 |
| 3 | 0.5 | mlp | power_spherical vs ps_sym | 10 | 0.1246 | 0.1259 | -0.0013 | 8/10 | 0.0326 |
| 3 | 0.5 | mlp | vmf vs power_spherical | 10 | 0.2697 | 0.1246 | 0.1451 | 0/10 | 0.0004 |
| 3 | 1 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 0.1835 | 0.1648 | 0.0187 | 1/10 | 0.0006 |
| 3 | 1 | fixed | tnbbeta vs vmf | 10 | 0.1835 | 0.4130 | -0.2294 | 10/10 | -0.0294 |
| 3 | 1 | fixed | tnbbeta vs power_spherical | 10 | 0.1835 | 0.3476 | -0.1641 | 10/10 | -0.0048 |
| 3 | 1 | fixed | tnbbeta vs vmf_sym | 10 | 0.1835 | 0.0676 | 0.1159 | 0/10 | -0.0005 |
| 3 | 1 | fixed | tnbbeta vs ps_sym | 10 | 0.1835 | 0.0768 | 0.1067 | 0/10 | -0.0001 |
| 3 | 1 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 0.1648 | 0.0676 | 0.0972 | 0/10 | -0.0011 |
| 3 | 1 | fixed | vmf vs vmf_sym | 10 | 0.4130 | 0.0676 | 0.3454 | 0/10 | 0.0289 |
| 3 | 1 | fixed | power_spherical vs ps_sym | 10 | 0.3476 | 0.0768 | 0.2708 | 0/10 | 0.0048 |
| 3 | 1 | fixed | vmf vs power_spherical | 10 | 0.4130 | 0.3476 | 0.0653 | 0/10 | 0.0246 |
| 3 | 1 | linear | tnbbeta vs tnbbeta_init0 | 10 | 0.0079 | 0.0079 | 0.0001 | 2/10 | -0.0000 |
| 3 | 1 | linear | tnbbeta vs vmf | 10 | 0.0079 | 0.1536 | -0.1457 | 10/10 | -0.0012 |
| 3 | 1 | linear | tnbbeta vs power_spherical | 10 | 0.0079 | 0.0078 | 0.0001 | 1/10 | 0.0000 |
| 3 | 1 | linear | tnbbeta vs vmf_sym | 10 | 0.0079 | 0.0111 | -0.0032 | 10/10 | 0.0002 |
| 3 | 1 | linear | tnbbeta vs ps_sym | 10 | 0.0079 | 0.0079 | 0.0000 | 3/10 | 0.0000 |
| 3 | 1 | linear | tnbbeta_init0 vs vmf_sym | 10 | 0.0079 | 0.0111 | -0.0032 | 10/10 | 0.0003 |
| 3 | 1 | linear | vmf vs vmf_sym | 10 | 0.1536 | 0.0111 | 0.1425 | 0/10 | 0.0014 |
| 3 | 1 | linear | power_spherical vs ps_sym | 10 | 0.0078 | 0.0079 | -0.0001 | 9/10 | -0.0000 |
| 3 | 1 | linear | vmf vs power_spherical | 10 | 0.1536 | 0.0078 | 0.1458 | 0/10 | 0.0012 |
| 3 | 1 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.0078 | 0.0078 | 0.0001 | 2/10 | -0.0000 |
| 3 | 1 | mlp | tnbbeta vs vmf | 10 | 0.0078 | 0.1536 | -0.1458 | 10/10 | -0.0012 |
| 3 | 1 | mlp | tnbbeta vs power_spherical | 10 | 0.0078 | 0.0077 | 0.0001 | 0/10 | -0.0000 |
| 3 | 1 | mlp | tnbbeta vs vmf_sym | 10 | 0.0078 | 0.0112 | -0.0033 | 10/10 | 0.0003 |
| 3 | 1 | mlp | tnbbeta vs ps_sym | 10 | 0.0078 | 0.0078 | 0.0001 | 3/10 | -0.0000 |
| 3 | 1 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 0.0078 | 0.0112 | -0.0034 | 10/10 | 0.0003 |
| 3 | 1 | mlp | vmf vs vmf_sym | 10 | 0.1536 | 0.0112 | 0.1424 | 0/10 | 0.0015 |
| 3 | 1 | mlp | power_spherical vs ps_sym | 10 | 0.0077 | 0.0078 | -0.0000 | 9/10 | -0.0000 |
| 3 | 1 | mlp | vmf vs power_spherical | 10 | 0.1536 | 0.0077 | 0.1459 | 0/10 | 0.0012 |
| 4 | 0.1 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 1.4365 | 1.3808 | 0.0557 | 5/10 | 0.0240 |
| 4 | 0.1 | fixed | tnbbeta vs vmf | 10 | 1.4365 | 1.3501 | 0.0864 | 2/10 | 0.0116 |
| 4 | 0.1 | fixed | tnbbeta vs power_spherical | 10 | 1.4365 | 1.4491 | -0.0126 | 5/10 | -0.0396 |
| 4 | 0.1 | fixed | tnbbeta vs vmf_sym | 10 | 1.4365 | 0.6643 | 0.7722 | 0/10 | 0.6900 |
| 4 | 0.1 | fixed | tnbbeta vs ps_sym | 10 | 1.4365 | 0.7645 | 0.6720 | 0/10 | 0.6312 |
| 4 | 0.1 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 1.3808 | 0.6643 | 0.7165 | 0/10 | 0.6659 |
| 4 | 0.1 | fixed | vmf vs vmf_sym | 10 | 1.3501 | 0.6643 | 0.6858 | 0/10 | 0.6783 |
| 4 | 0.1 | fixed | power_spherical vs ps_sym | 10 | 1.4491 | 0.7645 | 0.6845 | 0/10 | 0.6708 |
| 4 | 0.1 | fixed | vmf vs power_spherical | 10 | 1.3501 | 1.4491 | -0.0990 | 7/10 | -0.0513 |
| 4 | 0.1 | linear | tnbbeta vs tnbbeta_init0 | 10 | 7.7888 | 7.7907 | -0.0018 | 6/10 | -0.0030 |
| 4 | 0.1 | linear | tnbbeta vs vmf | 10 | 7.7888 | 7.9208 | -0.1320 | 10/10 | -0.0825 |
| 4 | 0.1 | linear | tnbbeta vs power_spherical | 10 | 7.7888 | 7.8124 | -0.0236 | 10/10 | -0.0832 |
| 4 | 0.1 | linear | tnbbeta vs vmf_sym | 10 | 7.7888 | 0.6309 | 7.1579 | 0/10 | 7.4447 |
| 4 | 0.1 | linear | tnbbeta vs ps_sym | 10 | 7.7888 | 7.8013 | -0.0125 | 10/10 | -0.0254 |
| 4 | 0.1 | linear | tnbbeta_init0 vs vmf_sym | 10 | 7.7907 | 0.6309 | 7.1597 | 0/10 | 7.4477 |
| 4 | 0.1 | linear | vmf vs vmf_sym | 10 | 7.9208 | 0.6309 | 7.2899 | 0/10 | 7.5271 |
| 4 | 0.1 | linear | power_spherical vs ps_sym | 10 | 7.8124 | 7.8013 | 0.0111 | 0/10 | 0.0578 |
| 4 | 0.1 | linear | vmf vs power_spherical | 10 | 7.9208 | 7.8124 | 0.1084 | 0/10 | -0.0007 |
| 4 | 0.1 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 7.8104 | 7.8103 | 0.0002 | 3/10 | 0.0001 |
| 4 | 0.1 | mlp | tnbbeta vs vmf | 10 | 7.8104 | 7.9211 | -0.1107 | 10/10 | -0.0008 |
| 4 | 0.1 | mlp | tnbbeta vs power_spherical | 10 | 7.8104 | 7.8102 | 0.0002 | 0/10 | 0.0001 |
| 4 | 0.1 | mlp | tnbbeta vs vmf_sym | 10 | 7.8104 | 7.0832 | 0.7273 | 2/10 | 0.7623 |
| 4 | 0.1 | mlp | tnbbeta vs ps_sym | 10 | 7.8104 | 7.8102 | 0.0002 | 0/10 | 0.0001 |
| 4 | 0.1 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 7.8103 | 7.0832 | 0.7271 | 2/10 | 0.7623 |
| 4 | 0.1 | mlp | vmf vs vmf_sym | 10 | 7.9211 | 7.0832 | 0.8379 | 0/10 | 0.7632 |
| 4 | 0.1 | mlp | power_spherical vs ps_sym | 10 | 7.8102 | 7.8102 | -0.0000 | 10/10 | 0.0000 |
| 4 | 0.1 | mlp | vmf vs power_spherical | 10 | 7.9211 | 7.8102 | 0.1109 | 0/10 | 0.0009 |
| 4 | 0.25 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 1.0269 | 1.0148 | 0.0121 | 4/10 | 0.0008 |
| 4 | 0.25 | fixed | tnbbeta vs vmf | 10 | 1.0269 | 0.9807 | 0.0462 | 1/10 | -0.0108 |
| 4 | 0.25 | fixed | tnbbeta vs power_spherical | 10 | 1.0269 | 0.9866 | 0.0403 | 3/10 | -0.0226 |
| 4 | 0.25 | fixed | tnbbeta vs vmf_sym | 10 | 1.0269 | 0.2787 | 0.7482 | 0/10 | 0.6873 |
| 4 | 0.25 | fixed | tnbbeta vs ps_sym | 10 | 1.0269 | 0.2855 | 0.7414 | 0/10 | 0.6776 |
| 4 | 0.25 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 1.0148 | 0.2787 | 0.7361 | 0/10 | 0.6865 |
| 4 | 0.25 | fixed | vmf vs vmf_sym | 10 | 0.9807 | 0.2787 | 0.7020 | 0/10 | 0.6981 |
| 4 | 0.25 | fixed | power_spherical vs ps_sym | 10 | 0.9866 | 0.2855 | 0.7011 | 0/10 | 0.7002 |
| 4 | 0.25 | fixed | vmf vs power_spherical | 10 | 0.9807 | 0.9866 | -0.0059 | 5/10 | -0.0118 |
| 4 | 0.25 | linear | tnbbeta vs tnbbeta_init0 | 10 | 1.2457 | 1.2439 | 0.0018 | 1/10 | 0.0111 |
| 4 | 0.25 | linear | tnbbeta vs vmf | 10 | 1.2457 | 1.3581 | -0.1124 | 10/10 | -0.0057 |
| 4 | 0.25 | linear | tnbbeta vs power_spherical | 10 | 1.2457 | 1.2457 | -0.0000 | 4/10 | -0.0023 |
| 4 | 0.25 | linear | tnbbeta vs vmf_sym | 10 | 1.2457 | 1.1600 | 0.0857 | 9/10 | 0.1133 |
| 4 | 0.25 | linear | tnbbeta vs ps_sym | 10 | 1.2457 | 1.2454 | 0.0003 | 2/10 | -0.0009 |
| 4 | 0.25 | linear | tnbbeta_init0 vs vmf_sym | 10 | 1.2439 | 1.1600 | 0.0839 | 9/10 | 0.1022 |
| 4 | 0.25 | linear | vmf vs vmf_sym | 10 | 1.3581 | 1.1600 | 0.1981 | 0/10 | 0.1190 |
| 4 | 0.25 | linear | power_spherical vs ps_sym | 10 | 1.2457 | 1.2454 | 0.0003 | 1/10 | 0.0014 |
| 4 | 0.25 | linear | vmf vs power_spherical | 10 | 1.3581 | 1.2457 | 0.1124 | 0/10 | 0.0034 |
| 4 | 0.25 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 1.2442 | 1.2442 | 0.0001 | 3/10 | 0.0000 |
| 4 | 0.25 | mlp | tnbbeta vs vmf | 10 | 1.2442 | 1.3571 | -0.1128 | 10/10 | -0.0038 |
| 4 | 0.25 | mlp | tnbbeta vs power_spherical | 10 | 1.2442 | 1.2441 | 0.0002 | 0/10 | 0.0000 |
| 4 | 0.25 | mlp | tnbbeta vs vmf_sym | 10 | 1.2442 | 1.2461 | -0.0018 | 10/10 | 0.0013 |
| 4 | 0.25 | mlp | tnbbeta vs ps_sym | 10 | 1.2442 | 1.2441 | 0.0001 | 3/10 | 0.0000 |
| 4 | 0.25 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 1.2442 | 1.2461 | -0.0019 | 10/10 | 0.0013 |
| 4 | 0.25 | mlp | vmf vs vmf_sym | 10 | 1.3571 | 1.2461 | 0.1110 | 0/10 | 0.0052 |
| 4 | 0.25 | mlp | power_spherical vs ps_sym | 10 | 1.2441 | 1.2441 | -0.0000 | 9/10 | 0.0000 |
| 4 | 0.25 | mlp | vmf vs power_spherical | 10 | 1.3571 | 1.2441 | 0.1130 | 0/10 | 0.0038 |
| 4 | 0.5 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 0.7562 | 0.7536 | 0.0026 | 4/10 | -0.0018 |
| 4 | 0.5 | fixed | tnbbeta vs vmf | 10 | 0.7562 | 0.9439 | -0.1877 | 10/10 | -0.4173 |
| 4 | 0.5 | fixed | tnbbeta vs power_spherical | 10 | 0.7562 | 0.9399 | -0.1837 | 10/10 | -0.5406 |
| 4 | 0.5 | fixed | tnbbeta vs vmf_sym | 10 | 0.7562 | 0.2582 | 0.4981 | 0/10 | 0.0628 |
| 4 | 0.5 | fixed | tnbbeta vs ps_sym | 10 | 0.7562 | 0.2721 | 0.4842 | 0/10 | 0.0555 |
| 4 | 0.5 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 0.7536 | 0.2582 | 0.4954 | 0/10 | 0.0646 |
| 4 | 0.5 | fixed | vmf vs vmf_sym | 10 | 0.9439 | 0.2582 | 0.6858 | 0/10 | 0.4801 |
| 4 | 0.5 | fixed | power_spherical vs ps_sym | 10 | 0.9399 | 0.2721 | 0.6679 | 0/10 | 0.5961 |
| 4 | 0.5 | fixed | vmf vs power_spherical | 10 | 0.9439 | 0.9399 | 0.0040 | 3/10 | -0.1233 |
| 4 | 0.5 | linear | tnbbeta vs tnbbeta_init0 | 10 | 0.0994 | 0.0994 | 0.0000 | 5/10 | -0.0000 |
| 4 | 0.5 | linear | tnbbeta vs vmf | 10 | 0.0994 | 0.2128 | -0.1134 | 10/10 | -0.0045 |
| 4 | 0.5 | linear | tnbbeta vs power_spherical | 10 | 0.0994 | 0.0994 | 0.0001 | 1/10 | -0.0001 |
| 4 | 0.5 | linear | tnbbeta vs vmf_sym | 10 | 0.0994 | 0.1019 | -0.0025 | 10/10 | 0.0004 |
| 4 | 0.5 | linear | tnbbeta vs ps_sym | 10 | 0.0994 | 0.0994 | 0.0000 | 4/10 | -0.0000 |
| 4 | 0.5 | linear | tnbbeta_init0 vs vmf_sym | 10 | 0.0994 | 0.1019 | -0.0025 | 10/10 | 0.0004 |
| 4 | 0.5 | linear | vmf vs vmf_sym | 10 | 0.2128 | 0.1019 | 0.1110 | 0/10 | 0.0049 |
| 4 | 0.5 | linear | power_spherical vs ps_sym | 10 | 0.0994 | 0.0994 | -0.0000 | 5/10 | 0.0000 |
| 4 | 0.5 | linear | vmf vs power_spherical | 10 | 0.2128 | 0.0994 | 0.1135 | 0/10 | 0.0044 |
| 4 | 0.5 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.0985 | 0.0984 | 0.0001 | 4/10 | 0.0003 |
| 4 | 0.5 | mlp | tnbbeta vs vmf | 10 | 0.0985 | 0.2123 | -0.1138 | 10/10 | -0.0047 |
| 4 | 0.5 | mlp | tnbbeta vs power_spherical | 10 | 0.0985 | 0.0983 | 0.0001 | 1/10 | 0.0000 |
| 4 | 0.5 | mlp | tnbbeta vs vmf_sym | 10 | 0.0985 | 0.1012 | -0.0027 | 10/10 | 0.0002 |
| 4 | 0.5 | mlp | tnbbeta vs ps_sym | 10 | 0.0985 | 0.0984 | 0.0001 | 2/10 | 0.0001 |
| 4 | 0.5 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 0.0984 | 0.1012 | -0.0028 | 10/10 | -0.0001 |
| 4 | 0.5 | mlp | vmf vs vmf_sym | 10 | 0.2123 | 0.1012 | 0.1111 | 0/10 | 0.0049 |
| 4 | 0.5 | mlp | power_spherical vs ps_sym | 10 | 0.0983 | 0.0984 | -0.0000 | 9/10 | 0.0000 |
| 4 | 0.5 | mlp | vmf vs power_spherical | 10 | 0.2123 | 0.0983 | 0.1139 | 0/10 | 0.0047 |
| 4 | 1 | fixed | tnbbeta vs tnbbeta_init0 | 10 | 0.2115 | 0.2104 | 0.0011 | 4/10 | 0.0004 |
| 4 | 1 | fixed | tnbbeta vs vmf | 10 | 0.2115 | 0.4352 | -0.2237 | 10/10 | -0.0157 |
| 4 | 1 | fixed | tnbbeta vs power_spherical | 10 | 0.2115 | 0.3850 | -0.1735 | 10/10 | -0.0054 |
| 4 | 1 | fixed | tnbbeta vs vmf_sym | 10 | 0.2115 | 0.1238 | 0.0877 | 0/10 | 0.0003 |
| 4 | 1 | fixed | tnbbeta vs ps_sym | 10 | 0.2115 | 0.1337 | 0.0778 | 0/10 | -0.0011 |
| 4 | 1 | fixed | tnbbeta_init0 vs vmf_sym | 10 | 0.2104 | 0.1238 | 0.0866 | 0/10 | -0.0002 |
| 4 | 1 | fixed | vmf vs vmf_sym | 10 | 0.4352 | 0.1238 | 0.3114 | 0/10 | 0.0160 |
| 4 | 1 | fixed | power_spherical vs ps_sym | 10 | 0.3850 | 0.1337 | 0.2513 | 0/10 | 0.0043 |
| 4 | 1 | fixed | vmf vs power_spherical | 10 | 0.4352 | 0.3850 | 0.0502 | 0/10 | 0.0103 |
| 4 | 1 | linear | tnbbeta vs tnbbeta_init0 | 10 | 0.0023 | 0.0023 | 0.0000 | 4/10 | -0.0001 |
| 4 | 1 | linear | tnbbeta vs vmf | 10 | 0.0023 | 0.1161 | -0.1137 | 10/10 | -0.0049 |
| 4 | 1 | linear | tnbbeta vs power_spherical | 10 | 0.0023 | 0.0022 | 0.0001 | 0/10 | -0.0000 |
| 4 | 1 | linear | tnbbeta vs vmf_sym | 10 | 0.0023 | 0.0051 | -0.0028 | 10/10 | -0.0001 |
| 4 | 1 | linear | tnbbeta vs ps_sym | 10 | 0.0023 | 0.0022 | 0.0001 | 1/10 | -0.0000 |
| 4 | 1 | linear | tnbbeta_init0 vs vmf_sym | 10 | 0.0023 | 0.0051 | -0.0028 | 10/10 | -0.0000 |
| 4 | 1 | linear | vmf vs vmf_sym | 10 | 0.1161 | 0.0051 | 0.1110 | 0/10 | 0.0048 |
| 4 | 1 | linear | power_spherical vs ps_sym | 10 | 0.0022 | 0.0022 | -0.0000 | 5/10 | -0.0000 |
| 4 | 1 | linear | vmf vs power_spherical | 10 | 0.1161 | 0.0022 | 0.1138 | 0/10 | 0.0048 |
| 4 | 1 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.0020 | 0.0019 | 0.0000 | 6/10 | -0.0000 |
| 4 | 1 | mlp | tnbbeta vs vmf | 10 | 0.0020 | 0.1158 | -0.1139 | 10/10 | -0.0048 |
| 4 | 1 | mlp | tnbbeta vs power_spherical | 10 | 0.0020 | 0.0018 | 0.0001 | 0/10 | 0.0000 |
| 4 | 1 | mlp | tnbbeta vs vmf_sym | 10 | 0.0020 | 0.0049 | -0.0029 | 10/10 | -0.0001 |
| 4 | 1 | mlp | tnbbeta vs ps_sym | 10 | 0.0020 | 0.0018 | 0.0001 | 0/10 | -0.0000 |
| 4 | 1 | mlp | tnbbeta_init0 vs vmf_sym | 10 | 0.0019 | 0.0049 | -0.0030 | 10/10 | -0.0000 |
| 4 | 1 | mlp | vmf vs vmf_sym | 10 | 0.1158 | 0.0049 | 0.1109 | 0/10 | 0.0048 |
| 4 | 1 | mlp | power_spherical vs ps_sym | 10 | 0.0018 | 0.0018 | -0.0000 | 8/10 | -0.0000 |
| 4 | 1 | mlp | vmf vs power_spherical | 10 | 0.1158 | 0.0018 | 0.1140 | 0/10 | 0.0048 |

ln 2 = 0.6931.

## Antipodal mass balance (median over x, seed median; 0.5 = balanced)

| d | sigma | decoder | arm | min hemisphere share about q's own axis | about the true X_sym axis (fixed decoder) | median abs cos(q axis, true axis) |
|---|---|---|---|---|---|---|
| 2 | 0.1 | fixed | vmf | 0.000 | 0.000 | 1.000 |
| 2 | 0.1 | fixed | power_spherical | 0.000 | 0.000 | 1.000 |
| 2 | 0.1 | fixed | tnbbeta | 0.480 | 0.480 | 1.000 |
| 2 | 0.1 | fixed | tnbbeta_init0 | 0.480 | 0.480 | 1.000 |
| 2 | 0.1 | fixed | vmf_sym | 0.480 | 0.480 | 1.000 |
| 2 | 0.1 | fixed | ps_sym | 0.480 | 0.480 | 1.000 |
| 2 | 0.1 | linear | vmf | 0.227 | - | - |
| 2 | 0.1 | linear | power_spherical | 0.000 | - | - |
| 2 | 0.1 | linear | tnbbeta | 0.480 | - | - |
| 2 | 0.1 | linear | tnbbeta_init0 | 0.480 | - | - |
| 2 | 0.1 | linear | vmf_sym | 0.480 | - | - |
| 2 | 0.1 | linear | ps_sym | 0.480 | - | - |
| 2 | 0.1 | mlp | vmf | 0.227 | - | - |
| 2 | 0.1 | mlp | power_spherical | 0.480 | - | - |
| 2 | 0.1 | mlp | tnbbeta | 0.480 | - | - |
| 2 | 0.1 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 2 | 0.1 | mlp | vmf_sym | 0.480 | - | - |
| 2 | 0.1 | mlp | ps_sym | 0.480 | - | - |
| 2 | 0.25 | fixed | vmf | 0.000 | 0.000 | 1.000 |
| 2 | 0.25 | fixed | power_spherical | 0.000 | 0.000 | 1.000 |
| 2 | 0.25 | fixed | tnbbeta | 0.480 | 0.480 | 1.000 |
| 2 | 0.25 | fixed | tnbbeta_init0 | 0.480 | 0.480 | 1.000 |
| 2 | 0.25 | fixed | vmf_sym | 0.480 | 0.480 | 1.000 |
| 2 | 0.25 | fixed | ps_sym | 0.480 | 0.480 | 1.000 |
| 2 | 0.25 | linear | vmf | 0.227 | - | - |
| 2 | 0.25 | linear | power_spherical | 0.000 | - | - |
| 2 | 0.25 | linear | tnbbeta | 0.480 | - | - |
| 2 | 0.25 | linear | tnbbeta_init0 | 0.480 | - | - |
| 2 | 0.25 | linear | vmf_sym | 0.480 | - | - |
| 2 | 0.25 | linear | ps_sym | 0.480 | - | - |
| 2 | 0.25 | mlp | vmf | 0.227 | - | - |
| 2 | 0.25 | mlp | power_spherical | 0.480 | - | - |
| 2 | 0.25 | mlp | tnbbeta | 0.480 | - | - |
| 2 | 0.25 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 2 | 0.25 | mlp | vmf_sym | 0.480 | - | - |
| 2 | 0.25 | mlp | ps_sym | 0.480 | - | - |
| 2 | 0.5 | fixed | vmf | 0.000 | 0.000 | 0.999 |
| 2 | 0.5 | fixed | power_spherical | 0.000 | 0.000 | 1.000 |
| 2 | 0.5 | fixed | tnbbeta | 0.480 | 0.480 | 1.000 |
| 2 | 0.5 | fixed | tnbbeta_init0 | 0.480 | 0.480 | 1.000 |
| 2 | 0.5 | fixed | vmf_sym | 0.480 | 0.480 | 1.000 |
| 2 | 0.5 | fixed | ps_sym | 0.480 | 0.480 | 1.000 |
| 2 | 0.5 | linear | vmf | 0.227 | - | - |
| 2 | 0.5 | linear | power_spherical | 0.480 | - | - |
| 2 | 0.5 | linear | tnbbeta | 0.480 | - | - |
| 2 | 0.5 | linear | tnbbeta_init0 | 0.480 | - | - |
| 2 | 0.5 | linear | vmf_sym | 0.480 | - | - |
| 2 | 0.5 | linear | ps_sym | 0.480 | - | - |
| 2 | 0.5 | mlp | vmf | 0.227 | - | - |
| 2 | 0.5 | mlp | power_spherical | 0.480 | - | - |
| 2 | 0.5 | mlp | tnbbeta | 0.480 | - | - |
| 2 | 0.5 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 2 | 0.5 | mlp | vmf_sym | 0.479 | - | - |
| 2 | 0.5 | mlp | ps_sym | 0.480 | - | - |
| 2 | 1 | fixed | vmf | 0.223 | 0.219 | 0.989 |
| 2 | 1 | fixed | power_spherical | 0.480 | 0.480 | 0.707 |
| 2 | 1 | fixed | tnbbeta | 0.480 | 0.480 | 0.999 |
| 2 | 1 | fixed | tnbbeta_init0 | 0.480 | 0.480 | 0.999 |
| 2 | 1 | fixed | vmf_sym | 0.479 | 0.479 | 0.999 |
| 2 | 1 | fixed | ps_sym | 0.480 | 0.480 | 0.999 |
| 2 | 1 | linear | vmf | 0.227 | - | - |
| 2 | 1 | linear | power_spherical | 0.480 | - | - |
| 2 | 1 | linear | tnbbeta | 0.480 | - | - |
| 2 | 1 | linear | tnbbeta_init0 | 0.480 | - | - |
| 2 | 1 | linear | vmf_sym | 0.479 | - | - |
| 2 | 1 | linear | ps_sym | 0.480 | - | - |
| 2 | 1 | mlp | vmf | 0.227 | - | - |
| 2 | 1 | mlp | power_spherical | 0.480 | - | - |
| 2 | 1 | mlp | tnbbeta | 0.480 | - | - |
| 2 | 1 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 2 | 1 | mlp | vmf_sym | 0.480 | - | - |
| 2 | 1 | mlp | ps_sym | 0.480 | - | - |
| 3 | 0.1 | fixed | vmf | 0.000 | 0.000 | 1.000 |
| 3 | 0.1 | fixed | power_spherical | 0.000 | 0.000 | 1.000 |
| 3 | 0.1 | fixed | tnbbeta | 0.000 | 0.000 | 1.000 |
| 3 | 0.1 | fixed | tnbbeta_init0 | 0.000 | 0.000 | 1.000 |
| 3 | 0.1 | fixed | vmf_sym | 0.477 | 0.477 | 1.000 |
| 3 | 0.1 | fixed | ps_sym | 0.480 | 0.480 | 1.000 |
| 3 | 0.1 | linear | vmf | 0.283 | - | - |
| 3 | 0.1 | linear | power_spherical | 0.480 | - | - |
| 3 | 0.1 | linear | tnbbeta | 0.477 | - | - |
| 3 | 0.1 | linear | tnbbeta_init0 | 0.480 | - | - |
| 3 | 0.1 | linear | vmf_sym | 0.477 | - | - |
| 3 | 0.1 | linear | ps_sym | 0.480 | - | - |
| 3 | 0.1 | mlp | vmf | 0.283 | - | - |
| 3 | 0.1 | mlp | power_spherical | 0.480 | - | - |
| 3 | 0.1 | mlp | tnbbeta | 0.480 | - | - |
| 3 | 0.1 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 3 | 0.1 | mlp | vmf_sym | 0.477 | - | - |
| 3 | 0.1 | mlp | ps_sym | 0.480 | - | - |
| 3 | 0.25 | fixed | vmf | 0.000 | 0.000 | 0.999 |
| 3 | 0.25 | fixed | power_spherical | 0.000 | 0.000 | 0.999 |
| 3 | 0.25 | fixed | tnbbeta | 0.000 | 0.000 | 0.999 |
| 3 | 0.25 | fixed | tnbbeta_init0 | 0.238 | 0.239 | 0.850 |
| 3 | 0.25 | fixed | vmf_sym | 0.477 | 0.477 | 0.999 |
| 3 | 0.25 | fixed | ps_sym | 0.480 | 0.480 | 0.999 |
| 3 | 0.25 | linear | vmf | 0.283 | - | - |
| 3 | 0.25 | linear | power_spherical | 0.480 | - | - |
| 3 | 0.25 | linear | tnbbeta | 0.480 | - | - |
| 3 | 0.25 | linear | tnbbeta_init0 | 0.479 | - | - |
| 3 | 0.25 | linear | vmf_sym | 0.477 | - | - |
| 3 | 0.25 | linear | ps_sym | 0.480 | - | - |
| 3 | 0.25 | mlp | vmf | 0.283 | - | - |
| 3 | 0.25 | mlp | power_spherical | 0.480 | - | - |
| 3 | 0.25 | mlp | tnbbeta | 0.479 | - | - |
| 3 | 0.25 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 3 | 0.25 | mlp | vmf_sym | 0.477 | - | - |
| 3 | 0.25 | mlp | ps_sym | 0.480 | - | - |
| 3 | 0.5 | fixed | vmf | 0.000 | 0.000 | 0.998 |
| 3 | 0.5 | fixed | power_spherical | 0.000 | 0.000 | 0.998 |
| 3 | 0.5 | fixed | tnbbeta | 0.094 | 0.094 | 0.998 |
| 3 | 0.5 | fixed | tnbbeta_init0 | 0.105 | 0.105 | 0.998 |
| 3 | 0.5 | fixed | vmf_sym | 0.477 | 0.480 | 0.999 |
| 3 | 0.5 | fixed | ps_sym | 0.480 | 0.480 | 0.999 |
| 3 | 0.5 | linear | vmf | 0.283 | - | - |
| 3 | 0.5 | linear | power_spherical | 0.480 | - | - |
| 3 | 0.5 | linear | tnbbeta | 0.480 | - | - |
| 3 | 0.5 | linear | tnbbeta_init0 | 0.480 | - | - |
| 3 | 0.5 | linear | vmf_sym | 0.480 | - | - |
| 3 | 0.5 | linear | ps_sym | 0.480 | - | - |
| 3 | 0.5 | mlp | vmf | 0.283 | - | - |
| 3 | 0.5 | mlp | power_spherical | 0.480 | - | - |
| 3 | 0.5 | mlp | tnbbeta | 0.480 | - | - |
| 3 | 0.5 | mlp | tnbbeta_init0 | 0.477 | - | - |
| 3 | 0.5 | mlp | vmf_sym | 0.480 | - | - |
| 3 | 0.5 | mlp | ps_sym | 0.480 | - | - |
| 3 | 1 | fixed | vmf | 0.277 | 0.270 | 0.941 |
| 3 | 1 | fixed | power_spherical | 0.473 | 0.480 | 0.580 |
| 3 | 1 | fixed | tnbbeta | 0.477 | 0.477 | 0.991 |
| 3 | 1 | fixed | tnbbeta_init0 | 0.479 | 0.480 | 0.712 |
| 3 | 1 | fixed | vmf_sym | 0.480 | 0.480 | 0.996 |
| 3 | 1 | fixed | ps_sym | 0.480 | 0.480 | 0.996 |
| 3 | 1 | linear | vmf | 0.283 | - | - |
| 3 | 1 | linear | power_spherical | 0.480 | - | - |
| 3 | 1 | linear | tnbbeta | 0.480 | - | - |
| 3 | 1 | linear | tnbbeta_init0 | 0.480 | - | - |
| 3 | 1 | linear | vmf_sym | 0.480 | - | - |
| 3 | 1 | linear | ps_sym | 0.480 | - | - |
| 3 | 1 | mlp | vmf | 0.283 | - | - |
| 3 | 1 | mlp | power_spherical | 0.480 | - | - |
| 3 | 1 | mlp | tnbbeta | 0.480 | - | - |
| 3 | 1 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 3 | 1 | mlp | vmf_sym | 0.480 | - | - |
| 3 | 1 | mlp | ps_sym | 0.480 | - | - |
| 4 | 0.1 | fixed | vmf | 0.000 | 0.000 | 0.999 |
| 4 | 0.1 | fixed | power_spherical | 0.000 | 0.000 | 0.999 |
| 4 | 0.1 | fixed | tnbbeta | 0.000 | 0.000 | 0.999 |
| 4 | 0.1 | fixed | tnbbeta_init0 | 0.000 | 0.000 | 0.999 |
| 4 | 0.1 | fixed | vmf_sym | 0.480 | 0.480 | 0.999 |
| 4 | 0.1 | fixed | ps_sym | 0.480 | 0.480 | 0.999 |
| 4 | 0.1 | linear | vmf | 0.328 | - | - |
| 4 | 0.1 | linear | power_spherical | 0.479 | - | - |
| 4 | 0.1 | linear | tnbbeta | 0.477 | - | - |
| 4 | 0.1 | linear | tnbbeta_init0 | 0.479 | - | - |
| 4 | 0.1 | linear | vmf_sym | 0.480 | - | - |
| 4 | 0.1 | linear | ps_sym | 0.480 | - | - |
| 4 | 0.1 | mlp | vmf | 0.328 | - | - |
| 4 | 0.1 | mlp | power_spherical | 0.479 | - | - |
| 4 | 0.1 | mlp | tnbbeta | 0.480 | - | - |
| 4 | 0.1 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 4 | 0.1 | mlp | vmf_sym | 0.480 | - | - |
| 4 | 0.1 | mlp | ps_sym | 0.480 | - | - |
| 4 | 0.25 | fixed | vmf | 0.000 | 0.000 | 0.998 |
| 4 | 0.25 | fixed | power_spherical | 0.000 | 0.000 | 0.998 |
| 4 | 0.25 | fixed | tnbbeta | 0.000 | 0.000 | 0.998 |
| 4 | 0.25 | fixed | tnbbeta_init0 | 0.000 | 0.000 | 0.998 |
| 4 | 0.25 | fixed | vmf_sym | 0.480 | 0.480 | 0.998 |
| 4 | 0.25 | fixed | ps_sym | 0.480 | 0.480 | 0.998 |
| 4 | 0.25 | linear | vmf | 0.328 | - | - |
| 4 | 0.25 | linear | power_spherical | 0.479 | - | - |
| 4 | 0.25 | linear | tnbbeta | 0.477 | - | - |
| 4 | 0.25 | linear | tnbbeta_init0 | 0.480 | - | - |
| 4 | 0.25 | linear | vmf_sym | 0.480 | - | - |
| 4 | 0.25 | linear | ps_sym | 0.480 | - | - |
| 4 | 0.25 | mlp | vmf | 0.328 | - | - |
| 4 | 0.25 | mlp | power_spherical | 0.479 | - | - |
| 4 | 0.25 | mlp | tnbbeta | 0.480 | - | - |
| 4 | 0.25 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 4 | 0.25 | mlp | vmf_sym | 0.480 | - | - |
| 4 | 0.25 | mlp | ps_sym | 0.480 | - | - |
| 4 | 0.5 | fixed | vmf | 0.004 | 0.004 | 0.995 |
| 4 | 0.5 | fixed | power_spherical | 0.000 | 0.000 | 0.996 |
| 4 | 0.5 | fixed | tnbbeta | 0.117 | 0.125 | 0.995 |
| 4 | 0.5 | fixed | tnbbeta_init0 | 0.121 | 0.127 | 0.995 |
| 4 | 0.5 | fixed | vmf_sym | 0.480 | 0.480 | 0.997 |
| 4 | 0.5 | fixed | ps_sym | 0.480 | 0.480 | 0.997 |
| 4 | 0.5 | linear | vmf | 0.328 | - | - |
| 4 | 0.5 | linear | power_spherical | 0.479 | - | - |
| 4 | 0.5 | linear | tnbbeta | 0.479 | - | - |
| 4 | 0.5 | linear | tnbbeta_init0 | 0.480 | - | - |
| 4 | 0.5 | linear | vmf_sym | 0.480 | - | - |
| 4 | 0.5 | linear | ps_sym | 0.479 | - | - |
| 4 | 0.5 | mlp | vmf | 0.328 | - | - |
| 4 | 0.5 | mlp | power_spherical | 0.479 | - | - |
| 4 | 0.5 | mlp | tnbbeta | 0.477 | - | - |
| 4 | 0.5 | mlp | tnbbeta_init0 | 0.479 | - | - |
| 4 | 0.5 | mlp | vmf_sym | 0.480 | - | - |
| 4 | 0.5 | mlp | ps_sym | 0.480 | - | - |
| 4 | 1 | fixed | vmf | 0.324 | 0.305 | 0.829 |
| 4 | 1 | fixed | power_spherical | 0.477 | 0.480 | 0.419 |
| 4 | 1 | fixed | tnbbeta | 0.477 | 0.477 | 0.983 |
| 4 | 1 | fixed | tnbbeta_init0 | 0.477 | 0.477 | 0.984 |
| 4 | 1 | fixed | vmf_sym | 0.477 | 0.479 | 0.991 |
| 4 | 1 | fixed | ps_sym | 0.480 | 0.480 | 0.990 |
| 4 | 1 | linear | vmf | 0.328 | - | - |
| 4 | 1 | linear | power_spherical | 0.479 | - | - |
| 4 | 1 | linear | tnbbeta | 0.477 | - | - |
| 4 | 1 | linear | tnbbeta_init0 | 0.480 | - | - |
| 4 | 1 | linear | vmf_sym | 0.480 | - | - |
| 4 | 1 | linear | ps_sym | 0.480 | - | - |
| 4 | 1 | mlp | vmf | 0.328 | - | - |
| 4 | 1 | mlp | power_spherical | 0.479 | - | - |
| 4 | 1 | mlp | tnbbeta | 0.480 | - | - |
| 4 | 1 | mlp | tnbbeta_init0 | 0.480 | - | - |
| 4 | 1 | mlp | vmf_sym | 0.480 | - | - |
| 4 | 1 | mlp | ps_sym | 0.480 | - | - |

## Learned sigma, divergences and fitted parameters

| d | sigma | decoder | arm | diverged | learned sigma (mean +- sd) | p | q | epsilon | m = eps - (d-1)/2 | frac m<0 | kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.1 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 197 |
| 2 | 0.1 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 394 |
| 2 | 0.1 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.499 | 0.643 | 36.9 | 36.4 | 0 | - |
| 2 | 0.1 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.501 | 0.444 | 59.3 | 58.8 | 0 | - |
| 2 | 0.1 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 199 |
| 2 | 0.1 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 392 |
| 2 | 0.1 | linear | vmf | 0/10 | 0.3657 +- 0.0005 | - | - | - | - | - | 1 |
| 2 | 0.1 | linear | power_spherical | 0/10 | 0.1824 +- 0.1278 | - | - | - | - | - | 365 |
| 2 | 0.1 | linear | tnbbeta | 0/10 | 0.1030 +- 0.0009 | 0.501 | 0.672 | 31.6 | 31.1 | 0 | - |
| 2 | 0.1 | linear | tnbbeta_init0 | 0/10 | 0.1052 +- 0.0024 | 0.5 | 0.571 | 42 | 41.5 | 0 | - |
| 2 | 0.1 | linear | vmf_sym | 0/10 | 0.1052 +- 0.0031 | - | - | - | - | - | 180 |
| 2 | 0.1 | linear | ps_sym | 0/10 | 0.1043 +- 0.0027 | - | - | - | - | - | 365 |
| 2 | 0.1 | mlp | vmf | 0/10 | 0.3656 +- 0.0005 | - | - | - | - | - | 1 |
| 2 | 0.1 | mlp | power_spherical | 0/10 | 0.3145 +- 0.1119 | - | - | - | - | - | 5.71e-09 |
| 2 | 0.1 | mlp | tnbbeta | 0/10 | 0.1037 +- 0.0016 | 0.5 | 0.641 | 33.8 | 33.3 | 0 | - |
| 2 | 0.1 | mlp | tnbbeta_init0 | 0/10 | 0.1033 +- 0.0020 | 0.501 | 0.608 | 34.3 | 33.8 | 0 | - |
| 2 | 0.1 | mlp | vmf_sym | 0/10 | 0.1039 +- 0.0030 | - | - | - | - | - | 196 |
| 2 | 0.1 | mlp | ps_sym | 0/10 | 0.1027 +- 0.0020 | - | - | - | - | - | 403 |
| 2 | 0.25 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 31 |
| 2 | 0.25 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 61.9 |
| 2 | 0.25 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.502 | 0.428 | 9.45 | 8.95 | 0.0005 | - |
| 2 | 0.25 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.499 | 0.343 | 10.7 | 10.2 | 0.0005 | - |
| 2 | 0.25 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 31.8 |
| 2 | 0.25 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 62 |
| 2 | 0.25 | linear | vmf | 0/10 | 0.4314 +- 0.0006 | - | - | - | - | - | 1 |
| 2 | 0.25 | linear | power_spherical | 0/10 | 0.3247 +- 0.0935 | - | - | - | - | - | 58.8 |
| 2 | 0.25 | linear | tnbbeta | 0/10 | 0.2525 +- 0.0015 | 0.502 | 0.513 | 7.66 | 7.16 | 0.001 | - |
| 2 | 0.25 | linear | tnbbeta_init0 | 0/10 | 0.2539 +- 0.0043 | 0.501 | 0.491 | 8.04 | 7.54 | 0.0005 | - |
| 2 | 0.25 | linear | vmf_sym | 0/10 | 0.2523 +- 0.0020 | - | - | - | - | - | 31.4 |
| 2 | 0.25 | linear | ps_sym | 0/10 | 0.2518 +- 0.0008 | - | - | - | - | - | 60.9 |
| 2 | 0.25 | mlp | vmf | 0/10 | 0.4313 +- 0.0006 | - | - | - | - | - | 1 |
| 2 | 0.25 | mlp | power_spherical | 0/10 | 0.3972 +- 0.0758 | - | - | - | - | - | 3.26e-09 |
| 2 | 0.25 | mlp | tnbbeta | 0/10 | 0.2533 +- 0.0015 | 0.502 | 0.492 | 8.22 | 7.72 | 0.00025 | - |
| 2 | 0.25 | mlp | tnbbeta_init0 | 0/10 | 0.2523 +- 0.0016 | 0.503 | 0.461 | 8.64 | 8.14 | 0 | - |
| 2 | 0.25 | mlp | vmf_sym | 0/10 | 0.2523 +- 0.0013 | - | - | - | - | - | 30.5 |
| 2 | 0.25 | mlp | ps_sym | 0/10 | 0.2529 +- 0.0016 | - | - | - | - | - | 60 |
| 2 | 0.5 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 5.76 |
| 2 | 0.5 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 15.1 |
| 2 | 0.5 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.498 | 0.703 | 1.55 | 1.05 | 0.0287 | - |
| 2 | 0.5 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.5 | 0.7 | 1.56 | 1.06 | 0.0375 | - |
| 2 | 0.5 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 8.12 |
| 2 | 0.5 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 14.6 |
| 2 | 0.5 | linear | vmf | 0/10 | 0.6107 +- 0.0011 | - | - | - | - | - | 1 |
| 2 | 0.5 | linear | power_spherical | 0/10 | 0.6126 +- 0.0011 | - | - | - | - | - | 1.34e-18 |
| 2 | 0.5 | linear | tnbbeta | 0/10 | 0.5019 +- 0.0018 | 0.501 | 0.706 | 1.51 | 1.01 | 0.0355 | - |
| 2 | 0.5 | linear | tnbbeta_init0 | 0/10 | 0.5020 +- 0.0019 | 0.502 | 0.703 | 1.52 | 1.02 | 0.0442 | - |
| 2 | 0.5 | linear | vmf_sym | 0/10 | 0.4987 +- 0.0020 | - | - | - | - | - | 8.13 |
| 2 | 0.5 | linear | ps_sym | 0/10 | 0.5003 +- 0.0018 | - | - | - | - | - | 14.3 |
| 2 | 0.5 | mlp | vmf | 0/10 | 0.6108 +- 0.0010 | - | - | - | - | - | 1 |
| 2 | 0.5 | mlp | power_spherical | 0/10 | 0.6126 +- 0.0011 | - | - | - | - | - | 1.37e-17 |
| 2 | 0.5 | mlp | tnbbeta | 0/10 | 0.5023 +- 0.0023 | 0.502 | 0.712 | 1.48 | 0.979 | 0.0355 | - |
| 2 | 0.5 | mlp | tnbbeta_init0 | 0/10 | 0.5025 +- 0.0019 | 0.504 | 0.704 | 1.5 | 1 | 0.0355 | - |
| 2 | 0.5 | mlp | vmf_sym | 0/10 | 0.4994 +- 0.0023 | - | - | - | - | - | 8.14 |
| 2 | 0.5 | mlp | ps_sym | 0/10 | 0.5009 +- 0.0023 | - | - | - | - | - | 14.5 |
| 2 | 1 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 1 |
| 2 | 1 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 3.85e-34 |
| 2 | 1 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.498 | 0.712 | 0.611 | 0.111 | 0.0775 | - |
| 2 | 1 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.498 | 0.71 | 0.621 | 0.121 | 0.066 | - |
| 2 | 1 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 2.76 |
| 2 | 1 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 4.05 |
| 2 | 1 | linear | vmf | 0/10 | 1.0583 +- 0.0025 | - | - | - | - | - | 1 |
| 2 | 1 | linear | power_spherical | 0/10 | 1.0609 +- 0.0025 | - | - | - | - | - | 1.89e-14 |
| 2 | 1 | linear | tnbbeta | 0/10 | 1.0122 +- 0.0046 | 0.501 | 0.683 | 0.591 | 0.0911 | 0.0582 | - |
| 2 | 1 | linear | tnbbeta_init0 | 0/10 | 1.0112 +- 0.0048 | 0.499 | 0.685 | 0.59 | 0.0896 | 0.0897 | - |
| 2 | 1 | linear | vmf_sym | 0/10 | 0.9998 +- 0.0052 | - | - | - | - | - | 2.78 |
| 2 | 1 | linear | ps_sym | 0/10 | 1.0296 +- 0.0094 | - | - | - | - | - | 3.01 |
| 2 | 1 | mlp | vmf | 0/10 | 1.0584 +- 0.0024 | - | - | - | - | - | 1 |
| 2 | 1 | mlp | power_spherical | 0/10 | 1.0610 +- 0.0024 | - | - | - | - | - | 5.92e-15 |
| 2 | 1 | mlp | tnbbeta | 0/10 | 1.0192 +- 0.0155 | 0.497 | 0.681 | 0.587 | 0.0865 | 0.081 | - |
| 2 | 1 | mlp | tnbbeta_init0 | 0/10 | 1.0145 +- 0.0070 | 0.5 | 0.687 | 0.579 | 0.0794 | 0.0865 | - |
| 2 | 1 | mlp | vmf_sym | 0/10 | 1.0001 +- 0.0059 | - | - | - | - | - | 2.65 |
| 2 | 1 | mlp | ps_sym | 0/10 | 1.0318 +- 0.0093 | - | - | - | - | - | 2.79 |
| 3 | 0.1 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 197 |
| 3 | 0.1 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 390 |
| 3 | 0.1 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.998 | 1e-06 | 2.62 | 1.62 | 0 | - |
| 3 | 0.1 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.5 | 1e-06 | 2.64 | 1.64 | 0 | - |
| 3 | 0.1 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 198 |
| 3 | 0.1 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 392 |
| 3 | 0.1 | linear | vmf | 0/10 | 0.2896 +- 0.0003 | - | - | - | - | - | 1 |
| 3 | 0.1 | linear | power_spherical | 0/10 | 0.2901 +- 0.0003 | - | - | - | - | - | 7.89e-09 |
| 3 | 0.1 | linear | tnbbeta | 0/10 | 0.2850 +- 0.0003 | 0.499 | 0.07 | 1.39 | 0.39 | 0.363 | - |
| 3 | 0.1 | linear | tnbbeta_init0 | 0/10 | 0.2838 +- 0.0033 | 0.5 | 1e-06 | 1.46 | 0.457 | 0.361 | - |
| 3 | 0.1 | linear | vmf_sym | 0/10 | 0.1046 +- 0.0010 | - | - | - | - | - | 181 |
| 3 | 0.1 | linear | ps_sym | 0/10 | 0.2894 +- 0.0013 | - | - | - | - | - | 0.511 |
| 3 | 0.1 | mlp | vmf | 0/10 | 0.2896 +- 0.0003 | - | - | - | - | - | 1 |
| 3 | 0.1 | mlp | power_spherical | 0/10 | 0.2901 +- 0.0003 | - | - | - | - | - | 4.38e-10 |
| 3 | 0.1 | mlp | tnbbeta | 0/10 | 0.2901 +- 0.0003 | 0.501 | 0.00925 | 0.996 | -0.00364 | 0.824 | - |
| 3 | 0.1 | mlp | tnbbeta_init0 | 0/10 | 0.2901 +- 0.0003 | 0.5 | 0.0183 | 0.99 | -0.00988 | 0.996 | - |
| 3 | 0.1 | mlp | vmf_sym | 0/10 | 0.1050 +- 0.0014 | - | - | - | - | - | 184 |
| 3 | 0.1 | mlp | ps_sym | 0/10 | 0.2901 +- 0.0003 | - | - | - | - | - | 0.991 |
| 3 | 0.25 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 32.5 |
| 3 | 0.25 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 63.4 |
| 3 | 0.25 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.99 | 1e-06 | 2.13 | 1.13 | 0 | - |
| 3 | 0.25 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.5 | 0.255 | 4.12 | 3.12 | 0 | - |
| 3 | 0.25 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 32.5 |
| 3 | 0.25 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 63 |
| 3 | 0.25 | linear | vmf | 0/10 | 0.3694 +- 0.0005 | - | - | - | - | - | 1 |
| 3 | 0.25 | linear | power_spherical | 0/10 | 0.3698 +- 0.0005 | - | - | - | - | - | 7.99e-10 |
| 3 | 0.25 | linear | tnbbeta | 0/10 | 0.3691 +- 0.0012 | 0.499 | 0.00943 | 0.997 | -0.00288 | 0.617 | - |
| 3 | 0.25 | linear | tnbbeta_init0 | 0/10 | 0.3676 +- 0.0005 | 0.5 | 0.0107 | 1.01 | 0.00868 | 0.487 | - |
| 3 | 0.25 | linear | vmf_sym | 0/10 | 0.2535 +- 0.0006 | - | - | - | - | - | 31.2 |
| 3 | 0.25 | linear | ps_sym | 0/10 | 0.2533 +- 0.0004 | - | - | - | - | - | 60.6 |
| 3 | 0.25 | mlp | vmf | 0/10 | 0.3694 +- 0.0005 | - | - | - | - | - | 1 |
| 3 | 0.25 | mlp | power_spherical | 0/10 | 0.3698 +- 0.0005 | - | - | - | - | - | 8.1e-12 |
| 3 | 0.25 | mlp | tnbbeta | 0/10 | 0.3679 +- 0.0016 | 0.503 | 1.05e-06 | 1.17 | 0.174 | 0.358 | - |
| 3 | 0.25 | mlp | tnbbeta_init0 | 0/10 | 0.3688 +- 0.0014 | 0.499 | 0.0158 | 0.996 | -0.00399 | 0.814 | - |
| 3 | 0.25 | mlp | vmf_sym | 0/10 | 0.2648 +- 0.0370 | - | - | - | - | - | 31.3 |
| 3 | 0.25 | mlp | ps_sym | 0/10 | 0.3348 +- 0.0565 | - | - | - | - | - | 1.01 |
| 3 | 0.5 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 8.18 |
| 3 | 0.5 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 15.2 |
| 3 | 0.5 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.947 | 0.703 | 0.488 | -0.512 | 0.998 | - |
| 3 | 0.5 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.5 | 0.694 | 0.498 | -0.502 | 0.998 | - |
| 3 | 0.5 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 8.35 |
| 3 | 0.5 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 15.2 |
| 3 | 0.5 | linear | vmf | 0/10 | 0.5693 +- 0.0009 | - | - | - | - | - | 1 |
| 3 | 0.5 | linear | power_spherical | 0/10 | 0.5697 +- 0.0009 | - | - | - | - | - | 1.5e-10 |
| 3 | 0.5 | linear | tnbbeta | 0/10 | 0.5697 +- 0.0009 | 0.501 | 0.0096 | 0.993 | -0.00721 | 0.873 | - |
| 3 | 0.5 | linear | tnbbeta_init0 | 0/10 | 0.5697 +- 0.0009 | 0.501 | 0.0169 | 1 | 0.000696 | 0.463 | - |
| 3 | 0.5 | linear | vmf_sym | 0/10 | 0.5592 +- 0.0216 | - | - | - | - | - | 1 |
| 3 | 0.5 | linear | ps_sym | 0/10 | 0.5587 +- 0.0237 | - | - | - | - | - | 0.991 |
| 3 | 0.5 | mlp | vmf | 0/10 | 0.5693 +- 0.0009 | - | - | - | - | - | 1 |
| 3 | 0.5 | mlp | power_spherical | 0/10 | 0.5697 +- 0.0009 | - | - | - | - | - | 1.65e-12 |
| 3 | 0.5 | mlp | tnbbeta | 0/10 | 0.5696 +- 0.0012 | 0.5 | 0.0112 | 0.995 | -0.00486 | 0.771 | - |
| 3 | 0.5 | mlp | tnbbeta_init0 | 0/10 | 0.5694 +- 0.0012 | 0.5 | 0.0173 | 0.998 | -0.00216 | 0.62 | - |
| 3 | 0.5 | mlp | vmf_sym | 0/10 | 0.5483 +- 0.0271 | - | - | - | - | - | 1 |
| 3 | 0.5 | mlp | ps_sym | 0/10 | 0.5532 +- 0.0263 | - | - | - | - | - | 1.02 |
| 3 | 1 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 1 |
| 3 | 1 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 0.113 |
| 3 | 1 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.498 | 1e-06 | 0.59 | -0.41 | 1 | - |
| 3 | 1 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.5 | 0.549 | 1.26 | 0.264 | 0.106 | - |
| 3 | 1 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 2.96 |
| 3 | 1 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 4.41 |
| 3 | 1 | linear | vmf | 0/10 | 1.0366 +- 0.0018 | - | - | - | - | - | 1 |
| 3 | 1 | linear | power_spherical | 0/10 | 1.0373 +- 0.0018 | - | - | - | - | - | 2.73e-14 |
| 3 | 1 | linear | tnbbeta | 0/10 | 1.0373 +- 0.0018 | 0.502 | 0.0122 | 0.991 | -0.009 | 0.957 | - |
| 3 | 1 | linear | tnbbeta_init0 | 0/10 | 1.0373 +- 0.0018 | 0.5 | 0.0165 | 0.994 | -0.00646 | 0.877 | - |
| 3 | 1 | linear | vmf_sym | 0/10 | 1.0366 +- 0.0018 | - | - | - | - | - | 1 |
| 3 | 1 | linear | ps_sym | 0/10 | 1.0373 +- 0.0018 | - | - | - | - | - | 0.999 |
| 3 | 1 | mlp | vmf | 0/10 | 1.0366 +- 0.0018 | - | - | - | - | - | 1 |
| 3 | 1 | mlp | power_spherical | 0/10 | 1.0373 +- 0.0018 | - | - | - | - | - | 1.6e-13 |
| 3 | 1 | mlp | tnbbeta | 0/10 | 1.0373 +- 0.0018 | 0.502 | 0.0136 | 0.992 | -0.00836 | 0.944 | - |
| 3 | 1 | mlp | tnbbeta_init0 | 0/10 | 1.0373 +- 0.0018 | 0.5 | 0.0152 | 0.995 | -0.00513 | 0.709 | - |
| 3 | 1 | mlp | vmf_sym | 0/10 | 1.0366 +- 0.0018 | - | - | - | - | - | 1 |
| 3 | 1 | mlp | ps_sym | 0/10 | 1.0373 +- 0.0018 | - | - | - | - | - | 0.971 |
| 4 | 0.1 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 194 |
| 4 | 0.1 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 385 |
| 4 | 0.1 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.997 | 1e-06 | 3.52 | 2.02 | 0 | - |
| 4 | 0.1 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.997 | 1e-06 | 3.52 | 2.02 | 0 | - |
| 4 | 0.1 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 195 |
| 4 | 0.1 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 385 |
| 4 | 0.1 | linear | vmf | 0/10 | 0.2384 +- 0.0003 | - | - | - | - | - | 1 |
| 4 | 0.1 | linear | power_spherical | 0/10 | 0.2386 +- 0.0003 | - | - | - | - | - | 3.25e-07 |
| 4 | 0.1 | linear | tnbbeta | 0/10 | 0.2363 +- 0.0004 | 0.501 | 1e-06 | 2.21 | 0.705 | 0.318 | - |
| 4 | 0.1 | linear | tnbbeta_init0 | 0/10 | 0.2364 +- 0.0003 | 0.5 | 1e-06 | 2.23 | 0.734 | 0.315 | - |
| 4 | 0.1 | linear | vmf_sym | 0/10 | 0.1047 +- 0.0007 | - | - | - | - | - | 176 |
| 4 | 0.1 | linear | ps_sym | 0/10 | 0.2368 +- 0.0003 | - | - | - | - | - | 0.47 |
| 4 | 0.1 | mlp | vmf | 0/10 | 0.2384 +- 0.0003 | - | - | - | - | - | 1 |
| 4 | 0.1 | mlp | power_spherical | 0/10 | 0.2386 +- 0.0003 | - | - | - | - | - | 4.24e-07 |
| 4 | 0.1 | mlp | tnbbeta | 0/10 | 0.2386 +- 0.0003 | 0.496 | 0.0128 | 1.5 | -0.0049 | 0.724 | - |
| 4 | 0.1 | mlp | tnbbeta_init0 | 0/10 | 0.2386 +- 0.0003 | 0.5 | 0.0205 | 1.49 | -0.0126 | 0.827 | - |
| 4 | 0.1 | mlp | vmf_sym | 0/10 | 0.2250 +- 0.0424 | - | - | - | - | - | 1 |
| 4 | 0.1 | mlp | ps_sym | 0/10 | 0.2386 +- 0.0003 | - | - | - | - | - | 0.988 |
| 4 | 0.25 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 31.4 |
| 4 | 0.25 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 62.8 |
| 4 | 0.25 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.983 | 1e-06 | 2.95 | 1.45 | 0 | - |
| 4 | 0.25 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.983 | 1e-06 | 2.95 | 1.45 | 0 | - |
| 4 | 0.25 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 32.6 |
| 4 | 0.25 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 62.7 |
| 4 | 0.25 | linear | vmf | 0/10 | 0.3308 +- 0.0005 | - | - | - | - | - | 1 |
| 4 | 0.25 | linear | power_spherical | 0/10 | 0.3309 +- 0.0005 | - | - | - | - | - | 5.12e-11 |
| 4 | 0.25 | linear | tnbbeta | 0/10 | 0.3307 +- 0.0007 | 0.499 | 0.0115 | 1.5 | -0.000526 | 0.514 | - |
| 4 | 0.25 | linear | tnbbeta_init0 | 0/10 | 0.3300 +- 0.0005 | 0.501 | 1e-06 | 1.66 | 0.163 | 0.395 | - |
| 4 | 0.25 | linear | vmf_sym | 0/10 | 0.3234 +- 0.0233 | - | - | - | - | - | 1 |
| 4 | 0.25 | linear | ps_sym | 0/10 | 0.3308 +- 0.0005 | - | - | - | - | - | 0.875 |
| 4 | 0.25 | mlp | vmf | 0/10 | 0.3308 +- 0.0005 | - | - | - | - | - | 1 |
| 4 | 0.25 | mlp | power_spherical | 0/10 | 0.3309 +- 0.0005 | - | - | - | - | - | 3.42e-08 |
| 4 | 0.25 | mlp | tnbbeta | 0/10 | 0.3309 +- 0.0005 | 0.499 | 0.0141 | 1.49 | -0.0137 | 0.863 | - |
| 4 | 0.25 | mlp | tnbbeta_init0 | 0/10 | 0.3309 +- 0.0005 | 0.5 | 0.022 | 1.47 | -0.0259 | 0.97 | - |
| 4 | 0.25 | mlp | vmf_sym | 0/10 | 0.3308 +- 0.0005 | - | - | - | - | - | 1 |
| 4 | 0.25 | mlp | ps_sym | 0/10 | 0.3309 +- 0.0005 | - | - | - | - | - | 1.01 |
| 4 | 0.5 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 6.55 |
| 4 | 0.5 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 15.2 |
| 4 | 0.5 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.902 | 0.462 | 0.657 | -0.843 | 1 | - |
| 4 | 0.5 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.87 | 0.429 | 0.666 | -0.834 | 1 | - |
| 4 | 0.5 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 8.84 |
| 4 | 0.5 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 15.2 |
| 4 | 0.5 | linear | vmf | 0/10 | 0.5451 +- 0.0008 | - | - | - | - | - | 1 |
| 4 | 0.5 | linear | power_spherical | 0/10 | 0.5452 +- 0.0008 | - | - | - | - | - | 5.6e-12 |
| 4 | 0.5 | linear | tnbbeta | 0/10 | 0.5452 +- 0.0008 | 0.5 | 0.0164 | 1.49 | -0.0134 | 0.882 | - |
| 4 | 0.5 | linear | tnbbeta_init0 | 0/10 | 0.5452 +- 0.0008 | 0.501 | 0.0224 | 1.47 | -0.0262 | 0.945 | - |
| 4 | 0.5 | linear | vmf_sym | 0/10 | 0.5451 +- 0.0008 | - | - | - | - | - | 1 |
| 4 | 0.5 | linear | ps_sym | 0/10 | 0.5452 +- 0.0008 | - | - | - | - | - | 0.968 |
| 4 | 0.5 | mlp | vmf | 0/10 | 0.5451 +- 0.0008 | - | - | - | - | - | 1 |
| 4 | 0.5 | mlp | power_spherical | 0/10 | 0.5452 +- 0.0008 | - | - | - | - | - | 1.12e-08 |
| 4 | 0.5 | mlp | tnbbeta | 0/10 | 0.5452 +- 0.0008 | 0.5 | 0.014 | 1.49 | -0.0094 | 0.799 | - |
| 4 | 0.5 | mlp | tnbbeta_init0 | 0/10 | 0.5451 +- 0.0008 | 0.5 | 0.0227 | 1.49 | -0.0139 | 0.839 | - |
| 4 | 0.5 | mlp | vmf_sym | 0/10 | 0.5451 +- 0.0008 | - | - | - | - | - | 1 |
| 4 | 0.5 | mlp | ps_sym | 0/10 | 0.5452 +- 0.0008 | - | - | - | - | - | 0.996 |
| 4 | 1 | fixed | vmf | 0/10 | (true, fixed) | - | - | - | - | - | 1 |
| 4 | 1 | fixed | power_spherical | 0/10 | (true, fixed) | - | - | - | - | - | 0.0584 |
| 4 | 1 | fixed | tnbbeta | 0/10 | (true, fixed) | 0.503 | 1e-06 | 0.847 | -0.653 | 1 | - |
| 4 | 1 | fixed | tnbbeta_init0 | 0/10 | (true, fixed) | 0.501 | 1e-06 | 0.837 | -0.663 | 1 | - |
| 4 | 1 | fixed | vmf_sym | 0/10 | (true, fixed) | - | - | - | - | - | 3.23 |
| 4 | 1 | fixed | ps_sym | 0/10 | (true, fixed) | - | - | - | - | - | 4.5 |
| 4 | 1 | linear | vmf | 0/10 | 1.0236 +- 0.0015 | - | - | - | - | - | 1 |
| 4 | 1 | linear | power_spherical | 0/10 | 1.0238 +- 0.0016 | - | - | - | - | - | 1.19e-17 |
| 4 | 1 | linear | tnbbeta | 0/10 | 1.0238 +- 0.0016 | 0.5 | 0.0211 | 1.48 | -0.0227 | 0.951 | - |
| 4 | 1 | linear | tnbbeta_init0 | 0/10 | 1.0238 +- 0.0016 | 0.498 | 0.0289 | 1.47 | -0.0256 | 0.945 | - |
| 4 | 1 | linear | vmf_sym | 0/10 | 1.0236 +- 0.0016 | - | - | - | - | - | 1 |
| 4 | 1 | linear | ps_sym | 0/10 | 1.0238 +- 0.0016 | - | - | - | - | - | 7.74e-09 |
| 4 | 1 | mlp | vmf | 0/10 | 1.0236 +- 0.0016 | - | - | - | - | - | 1 |
| 4 | 1 | mlp | power_spherical | 0/10 | 1.0238 +- 0.0016 | - | - | - | - | - | 1.87e-11 |
| 4 | 1 | mlp | tnbbeta | 0/10 | 1.0238 +- 0.0016 | 0.501 | 0.019 | 1.48 | -0.0186 | 0.895 | - |
| 4 | 1 | mlp | tnbbeta_init0 | 0/10 | 1.0238 +- 0.0016 | 0.501 | 0.0326 | 1.47 | -0.0318 | 0.96 | - |
| 4 | 1 | mlp | vmf_sym | 0/10 | 1.0235 +- 0.0016 | - | - | - | - | - | 1 |
| 4 | 1 | mlp | ps_sym | 0/10 | 1.0238 +- 0.0016 | - | - | - | - | - | 7.4e-12 |

## Quadrature checks (worst case over runs)

| d | sigma | max abs err of true-model quadrature vs Bingham | max abs refine diff |
|---|---|---|---|
| 2 | 0.1 | 3.82e-14 | 2.53e-04 |
| 2 | 0.25 | 6.22e-15 | 6.87e-05 |
| 2 | 0.5 | 3.55e-15 | 3.49e-05 |
| 2 | 1 | 1.78e-15 | 1.30e-05 |
| 3 | 0.1 | 1.14e-11 | 1.05e-03 |
| 3 | 0.25 | 2.77e-12 | 5.50e-05 |
| 3 | 0.5 | 1.03e-12 | 1.01e-05 |
| 3 | 1 | 3.69e-13 | 7.04e-07 |
