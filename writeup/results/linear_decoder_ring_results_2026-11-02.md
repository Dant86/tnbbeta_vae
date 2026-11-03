# Sphere probabilistic PCA with a linear learned decoder: results

Date: 2026-11-02 to 2026-11-03. Pre-registration (frozen before any run, one deviation block D1 appended before results):
`linear_decoder_ring_preregistration_2026-11-02.md`.
Base commit: `6b1880f` (branch `synthetic-graph-families`) plus the uncommitted working tree (new files only: `src/tnbbeta_vae/data/linear_ring.py`,
`apps/synthetic/linear_decoder_ring.py`, `apps/synthetic/linear_decoder_ring_plot.py`, `scripts/slurm/linear_decoder_ring.sbatch`,
`scripts/slurm/linear_decoder_ring_aux.sbatch`, tests `tests/data/test_linear_ring.py`, `tests/apps/test_linear_decoder_ring.py`,
`tests/apps/test_linear_decoder_ring_plot.py`).

## Commands

```
sbatch --array=0-15 scripts/slurm/linear_decoder_ring.sbatch        # job 2298172 (PROCS=6; 960 runs); resubmitted for 31 unfinished cells: job 2300777
AUX=summarize sbatch scripts/slurm/linear_decoder_ring_aux.sbatch   # job 2302346  (uv run python -m apps.synthetic.linear_decoder_ring summarize --out-dir ring_recovery/linear_decoder)
AUX=pytest    sbatch scripts/slurm/linear_decoder_ring_aux.sbatch   # job 2302348: 1154 passed, 1 skipped (full suite); AUX=quality job 2298189: ruff format/check and pyright clean on the new files
uv run python -m apps.synthetic.linear_decoder_ring_plot --out-dir ring_recovery/linear_decoder --fig-dir writeup/results/figures   # login node (kaleido's Chrome does not start on compute nodes)
```
Other job ids: 2298083 (first array, cancelled after two tasks started, to add the D1 metrics; no results kept), 2298076 (10-epoch smoke, numbers not used),
2298006/2298009/2298173 (own-test jobs). Raw outputs: `ring_recovery/linear_decoder/` (gitignored); the full generated analysis is reproduced at the end of this note.

All 960 runs finished, none diverged. The vMF arms use the exact-envelope sampler (`ExactVonMisesFisher`, a local copy of the subclass from the dimension-scaling
study) in both training and evaluation, so the library sampler bias at kappa about 10-40, m >= 9 does not apply (and the fitted vMF kappa is about 1-2 anyway).

## Headline

* **Confirmatory (decoder L, 24 tests, Holm 0.05, margin 0.05 nats): 19 of 24 confirmed.** All 24 reject by Holm (every one is 10/10 seeds, p = 1/1024, scipy
  and enumerated p agree). The five that fail do so on the margin only: sigma = 0.4 against PowerSpherical at all four d (paired differences 0.022, 0.034,
  0.023, 0.0095) and sigma = 0.4, d = 17 against vMF (0.024). Everywhere else TNBBeta's gap is smaller by 0.05 to 0.21 nats (vMF) and 0.07 to 0.15 (PS).
* Absolute gaps: TNBBeta 0.0011 to 0.07 nats (shrinking with d: sigma = 0.2: 0.063, 0.027, 0.013, 0.005 for d = 3, 5, 9, 17); vMF 0.025 to 0.28; PS 0.011 to 0.20.
* **Trend in d (descriptive):** the advantage over vMF shrinks with d (sigma = 0.2: 0.147, 0.162, 0.131, 0.106; sigma = 0.4: 0.088, 0.087, 0.050, 0.024), and
  against PS it is flat at sigma = 0.1 (0.129 to 0.149) and non-monotone at 0.2 (0.070, 0.096, 0.090, 0.077). So the hypothesis that the advantage grows with d is NOT
  supported for this metric; the likely reason is that the marginal of y becomes nearly Gaussian as d grows (sd of w is 1/sqrt(d)), so every arm's gap falls.
* **Mechanism (diagnostic, not a hypothesis test).** vMF and PS essentially *collapse to the prior*: fitted kappa about 1.0 to 1.9 (near-uniform posterior), and the
  decoder compensates with an inflated sigma_hat (sigma_hat/sigma = 5.3 / 4.8 at d = 3, sigma = 0.1, falling to 1.07 / 1.04 at sigma = 0.4, d = 17); the exact
  decomposition puts about 55-70% of their gap at sigma = 0.1 in model misfit (sigma_hat wrong) and the rest in KL(q || the model's own ring posterior), with the posterior latitude spread
  over-wide (std(u.z) 0.486 under q against 0.400 for the model's exact ring at d = 3, sigma = 0.1). TNBBeta instead recovers sigma_hat/sigma between 1.00 and 1.5 (mean
  1.2 at sigma = 0.1) with std(u.z) under q equal to the exact ring's to within about 1% (0.1196 vs 0.1188), epsilon large (m = epsilon - (d-1)/2 about 3-10 at sigma <= 0.2) and p about 0.55-0.75: it
  represents the latitude ring, which is what the linear decoder forces. Model misfit is 0.001-0.01 nats for TNBBeta.
* **The axis (pre-registered question 2) is not identified**, see Deviation D1: `|cos(u, a)|` is a symmetry coordinate (per-seed values equal the initial `|cos|` of
  the seed and are identical across arms; median 0.13 at d = 17, 0.2-0.27 at d = 3, with values up to 0.98 at d = 3, as expected for random init). What can be
  answered is encoder-decoder consistency: mean `|centre . u|` is 0.99-1.00 for PS and TNBBeta, and 0.89-0.97 for vMF. So all three place their centre on the learned axis;
  they differ in how wide a ring they put around it.
* **Ring statistic R is uninformative here**: it sits at the Monte Carlo floor (0.05-0.07, 256 draws) for every arm, because a collapsed uniform posterior and a true ring
  are both azimuthally symmetric. The informative ring diagnostics are the latitude spread ratio and the decomposition above.
* **Contrasts (sigma = 0.2, descriptive).** With the 1-D tanh decoder M or the free scalar decoder F, the caps also reach small gaps (0.001 to 0.08) and PowerSpherical is
  as good as or marginally better than TNBBeta at d >= 5 (paired differences A - B of tnbbeta against PS: M -0.001 at d = 5, 9, 17 in PS's favour; F about -0.0006 at d = 9, 17): *the advantage is specific to the strictly linear decoder*.
  At d >= 9 all four arms under M and F fit sigma_hat of about 1.57 sigma (the standard deviation of y itself) with posterior collapsed: the decoder simply ignores the latent,
  which costs almost nothing in marginal likelihood because the marginal of y is near-Gaussian. This re-confirms the earlier free-decoder result (no TNBBeta win), now with a scalar output,
  and shows that the gap-to-true-marginal metric cannot distinguish a collapsed model from a correct one once d is large. Bias variant Lb (sigma = 0.2): same as L (TNBBeta better by 0.07-0.16 over both comparators; learned bias in [-0.041, 0.021]).
* **Surprises investigated.** (i) No negative mean gaps; two M/F cells have a minimum seed of -0.0001 (PS, d = 17) and an IWAE gap -0.0001 (TNBBeta, d = 9, sigma = 0.4): within the Monte Carlo noise of the 256-draw estimators (the gap is zero in expectation at a perfect fit).
  (ii) tnbbeta vs tnbbeta_init0 (p-offset 0): paired differences below 0.006 nats, 2-5 of 10 seeds, no systematic difference.
  (iii) The collapse gap column (`E_y KL(Uniform || p(.|y))`, 0.2-31 nats) is NOT the collapsed models' gap (they are 0.01-0.28) because they re-tune sigma_hat; it applies only with the true decoder.
  (iv) TNBBeta's seed spread is larger than the caps' (sd 0.002-0.03 against 0.0003-0.002); the caps are almost deterministic because they all fall into the same collapsed solution.

## Caveats

* The primary metric is the ELBO gap to the *true marginal* log p(y); for the linear decoder `p_model(y)` does not depend on `u`, so the gap separates families only through
  KL(q || p_model) and sigma_hat (D1(c)). The caps' dominant loss is sigma_hat inflation (collapse), which is an optimisation outcome with kappa initialised at d and 300 epochs at Adam 1e-3
  (kappa trajectory not logged); a cap family trained with a pinned small sigma_hat or a longer schedule might behave differently. Not tested here.
* Single initialisation convention (sigma_hat starts at 0.2; vMF/PS kappa starts at d; TNBBeta at library default).
* The descriptive contrasts M, F, Lb are at sigma = 0.2 only, as pre-registered.

## Deviations

D1 (in the pre-registration): axis cosine non-identifiable, collapse-reference corrected, `abs_cos_centre_mean` and `collapse_gap` added before any run; the aligned-only sensitivity table was dropped.
D2 (execution): runs trained on GPU in float32 as registered; 6 worker processes per array task (the registered conventions), 16 tasks; 31 cells that did not finish inside the 6 h task limit were re-run by resubmitting the same array (finished cells skipped). The figure
files carry the date stamp 2026-11-02 in their names (script constant) though they were rendered on 2026-11-03. The pre-registered "pyright/ruff on what you add" and the full suite were run on SLURM; `tests/distributions/test_sign_symmetrised.py` failed once (job 2298192, an untracked file of another study in progress) and passed in the later full run (job 2302348).

## Figures

`writeup/results/figures/linear_ring_gap_vs_d_2026-11-02.png` (ELBO gap, log axis, per-seed points and means), `linear_ring_alignment_vs_d_2026-11-02.png` (|cos(u, a)|, symmetry check), `linear_ring_ringstat_vs_d_2026-11-02.png` (R, at its Monte Carlo floor).

## Full generated analysis (all tables, per-seed values)


## Per-cell test ELBO gap (nats; lower is better)

| decoder | d | sigma | arm | seeds (valid) | gap mean +- sd | median | min / max | IWAE gap mean | collapse gap | I(z;y) |
|---|---|---|---|---|---|---|---|---|---|---|
| L | 3 | 0.1 | vmf | 10/10 | 0.2793 +- 0.0004 | 0.2793 | 0.2787 / 0.2799 | 0.1606 | 31.2978 | 1.673 |
| L | 3 | 0.1 | power_spherical | 10/10 | 0.1990 +- 0.0015 | 0.1984 | 0.1977 / 0.2020 | 0.1381 | 31.2978 | 1.673 |
| L | 3 | 0.1 | tnbbeta | 10/10 | 0.0705 +- 0.0232 | 0.0602 | 0.0479 / 0.1082 | 0.0117 | 31.2978 | 1.673 |
| L | 3 | 0.1 | tnbbeta_init0 | 10/10 | 0.0704 +- 0.0292 | 0.0583 | 0.0417 / 0.1182 | 0.0158 | 31.2978 | 1.673 |
| L | 5 | 0.1 | vmf | 10/10 | 0.2375 +- 0.0015 | 0.2377 | 0.2350 / 0.2396 | 0.1399 | 18.5843 | 1.495 |
| L | 5 | 0.1 | power_spherical | 10/10 | 0.1695 +- 0.0010 | 0.1692 | 0.1687 / 0.1720 | 0.1067 | 18.5843 | 1.495 |
| L | 5 | 0.1 | tnbbeta | 10/10 | 0.0325 +- 0.0051 | 0.0324 | 0.0250 / 0.0416 | 0.0060 | 18.5843 | 1.495 |
| L | 5 | 0.1 | tnbbeta_init0 | 10/10 | 0.0460 +- 0.0226 | 0.0363 | 0.0274 / 0.1006 | 0.0119 | 18.5843 | 1.495 |
| L | 9 | 0.1 | vmf | 10/10 | 0.2043 +- 0.0011 | 0.2042 | 0.2027 / 0.2063 | 0.1114 | 10.0269 | 1.251 |
| L | 9 | 0.1 | power_spherical | 10/10 | 0.1589 +- 0.0008 | 0.1589 | 0.1578 / 0.1604 | 0.0904 | 10.0269 | 1.251 |
| L | 9 | 0.1 | tnbbeta | 10/10 | 0.0190 +- 0.0075 | 0.0161 | 0.0118 / 0.0355 | 0.0040 | 10.0269 | 1.251 |
| L | 9 | 0.1 | tnbbeta_init0 | 10/10 | 0.0149 +- 0.0076 | 0.0115 | 0.0096 / 0.0340 | 0.0029 | 10.0269 | 1.251 |
| L | 17 | 0.1 | vmf | 10/10 | 0.1954 +- 0.0010 | 0.1950 | 0.1945 / 0.1973 | 0.1063 | 4.8422 | 0.949 |
| L | 17 | 0.1 | power_spherical | 10/10 | 0.1627 +- 0.0008 | 0.1629 | 0.1614 / 0.1642 | 0.0914 | 4.8422 | 0.949 |
| L | 17 | 0.1 | tnbbeta | 10/10 | 0.0134 +- 0.0076 | 0.0105 | 0.0059 / 0.0296 | 0.0037 | 4.8422 | 0.949 |
| L | 17 | 0.1 | tnbbeta_init0 | 10/10 | 0.0168 +- 0.0091 | 0.0137 | 0.0074 / 0.0339 | 0.0038 | 4.8422 | 0.949 |
| L | 3 | 0.2 | vmf | 10/10 | 0.2098 +- 0.0004 | 0.2097 | 0.2093 / 0.2104 | 0.0961 | 7.1797 | 1.072 |
| L | 3 | 0.2 | power_spherical | 10/10 | 0.1330 +- 0.0024 | 0.1322 | 0.1310 / 0.1391 | 0.0767 | 7.1797 | 1.072 |
| L | 3 | 0.2 | tnbbeta | 10/10 | 0.0633 +- 0.0210 | 0.0558 | 0.0528 / 0.1218 | 0.0071 | 7.1797 | 1.072 |
| L | 3 | 0.2 | tnbbeta_init0 | 10/10 | 0.0574 +- 0.0109 | 0.0528 | 0.0501 / 0.0808 | 0.0080 | 7.1797 | 1.072 |
| L | 5 | 0.2 | vmf | 10/10 | 0.1891 +- 0.0015 | 0.1888 | 0.1868 / 0.1915 | 0.0996 | 4.1231 | 0.881 |
| L | 5 | 0.2 | power_spherical | 10/10 | 0.1235 +- 0.0005 | 0.1236 | 0.1228 / 0.1242 | 0.0684 | 4.1231 | 0.881 |
| L | 5 | 0.2 | tnbbeta | 10/10 | 0.0272 +- 0.0083 | 0.0235 | 0.0195 / 0.0454 | 0.0032 | 4.1231 | 0.881 |
| L | 5 | 0.2 | tnbbeta_init0 | 10/10 | 0.0232 +- 0.0033 | 0.0221 | 0.0198 / 0.0286 | 0.0033 | 4.1231 | 0.881 |
| L | 9 | 0.2 | vmf | 10/10 | 0.1441 +- 0.0012 | 0.1438 | 0.1427 / 0.1469 | 0.0652 | 2.1537 | 0.674 |
| L | 9 | 0.2 | power_spherical | 10/10 | 0.1029 +- 0.0007 | 0.1028 | 0.1020 / 0.1039 | 0.0474 | 2.1537 | 0.674 |
| L | 9 | 0.2 | tnbbeta | 10/10 | 0.0132 +- 0.0082 | 0.0105 | 0.0073 / 0.0358 | 0.0007 | 2.1537 | 0.674 |
| L | 9 | 0.2 | tnbbeta_init0 | 10/10 | 0.0112 +- 0.0048 | 0.0113 | 0.0055 / 0.0177 | 0.0009 | 2.1537 | 0.674 |
| L | 17 | 0.2 | vmf | 10/10 | 0.1103 +- 0.0011 | 0.1100 | 0.1094 / 0.1129 | 0.0439 | 1.0101 | 0.445 |
| L | 17 | 0.2 | power_spherical | 10/10 | 0.0823 +- 0.0012 | 0.0822 | 0.0809 / 0.0851 | 0.0330 | 1.0101 | 0.445 |
| L | 17 | 0.2 | tnbbeta | 10/10 | 0.0048 +- 0.0029 | 0.0036 | 0.0025 / 0.0109 | 0.0007 | 1.0101 | 0.445 |
| L | 17 | 0.2 | tnbbeta_init0 | 10/10 | 0.0051 +- 0.0044 | 0.0038 | 0.0023 / 0.0174 | 0.0005 | 1.0101 | 0.445 |
| L | 3 | 0.4 | vmf | 10/10 | 0.1183 +- 0.0004 | 0.1183 | 0.1177 / 0.1189 | 0.0269 | 1.5180 | 0.556 |
| L | 3 | 0.4 | power_spherical | 10/10 | 0.0525 +- 0.0010 | 0.0527 | 0.0511 / 0.0540 | 0.0171 | 1.5180 | 0.556 |
| L | 3 | 0.4 | tnbbeta | 10/10 | 0.0307 +- 0.0022 | 0.0298 | 0.0285 / 0.0356 | 0.0028 | 1.5180 | 0.556 |
| L | 3 | 0.4 | tnbbeta_init0 | 10/10 | 0.0334 +- 0.0070 | 0.0303 | 0.0284 / 0.0509 | 0.0048 | 1.5180 | 0.556 |
| L | 5 | 0.4 | vmf | 10/10 | 0.0977 +- 0.0012 | 0.0976 | 0.0962 / 0.0994 | 0.0310 | 0.8485 | 0.405 |
| L | 5 | 0.4 | power_spherical | 10/10 | 0.0443 +- 0.0009 | 0.0444 | 0.0426 / 0.0456 | 0.0127 | 0.8485 | 0.405 |
| L | 5 | 0.4 | tnbbeta | 10/10 | 0.0107 +- 0.0022 | 0.0100 | 0.0088 / 0.0157 | 0.0009 | 0.8485 | 0.405 |
| L | 5 | 0.4 | tnbbeta_init0 | 10/10 | 0.0108 +- 0.0040 | 0.0097 | 0.0083 / 0.0217 | 0.0010 | 0.8485 | 0.405 |
| L | 9 | 0.4 | vmf | 10/10 | 0.0532 +- 0.0006 | 0.0530 | 0.0525 / 0.0543 | 0.0102 | 0.4375 | 0.272 |
| L | 9 | 0.4 | power_spherical | 10/10 | 0.0257 +- 0.0009 | 0.0255 | 0.0244 / 0.0272 | 0.0041 | 0.4375 | 0.272 |
| L | 9 | 0.4 | tnbbeta | 10/10 | 0.0030 +- 0.0008 | 0.0030 | 0.0019 / 0.0041 | -0.0001 | 0.4375 | 0.272 |
| L | 9 | 0.4 | tnbbeta_init0 | 10/10 | 0.0030 +- 0.0011 | 0.0027 | 0.0018 / 0.0058 | 0.0001 | 0.4375 | 0.272 |
| L | 17 | 0.4 | vmf | 10/10 | 0.0251 +- 0.0004 | 0.0253 | 0.0244 / 0.0257 | 0.0019 | 0.2118 | 0.158 |
| L | 17 | 0.4 | power_spherical | 10/10 | 0.0111 +- 0.0003 | 0.0110 | 0.0108 / 0.0117 | 0.0012 | 0.2118 | 0.158 |
| L | 17 | 0.4 | tnbbeta | 10/10 | 0.0016 +- 0.0008 | 0.0014 | 0.0008 / 0.0034 | 0.0001 | 0.2118 | 0.158 |
| L | 17 | 0.4 | tnbbeta_init0 | 10/10 | 0.0011 +- 0.0005 | 0.0010 | 0.0006 / 0.0021 | 0.0001 | 0.2118 | 0.158 |
| Lb | 3 | 0.2 | vmf | 10/10 | 0.2096 +- 0.0004 | 0.2096 | 0.2088 / 0.2101 | 0.0962 | 7.1797 | 1.072 |
| Lb | 3 | 0.2 | power_spherical | 10/10 | 0.1318 +- 0.0006 | 0.1318 | 0.1307 / 0.1325 | 0.0767 | 7.1797 | 1.072 |
| Lb | 3 | 0.2 | tnbbeta | 10/10 | 0.0578 +- 0.0076 | 0.0546 | 0.0505 / 0.0756 | 0.0075 | 7.1797 | 1.072 |
| Lb | 3 | 0.2 | tnbbeta_init0 | 10/10 | 0.0555 +- 0.0080 | 0.0519 | 0.0498 / 0.0744 | 0.0081 | 7.1797 | 1.072 |
| Lb | 5 | 0.2 | vmf | 10/10 | 0.1891 +- 0.0020 | 0.1885 | 0.1869 / 0.1929 | 0.0994 | 4.1231 | 0.881 |
| Lb | 5 | 0.2 | power_spherical | 10/10 | 0.1241 +- 0.0011 | 0.1242 | 0.1227 / 0.1261 | 0.0684 | 4.1231 | 0.881 |
| Lb | 5 | 0.2 | tnbbeta | 10/10 | 0.0284 +- 0.0068 | 0.0261 | 0.0219 / 0.0413 | 0.0041 | 4.1231 | 0.881 |
| Lb | 5 | 0.2 | tnbbeta_init0 | 10/10 | 0.0250 +- 0.0055 | 0.0243 | 0.0187 / 0.0344 | 0.0036 | 4.1231 | 0.881 |
| Lb | 9 | 0.2 | vmf | 10/10 | 0.1444 +- 0.0013 | 0.1440 | 0.1430 / 0.1469 | 0.0655 | 2.1537 | 0.674 |
| Lb | 9 | 0.2 | power_spherical | 10/10 | 0.1037 +- 0.0011 | 0.1035 | 0.1020 / 0.1053 | 0.0477 | 2.1537 | 0.674 |
| Lb | 9 | 0.2 | tnbbeta | 10/10 | 0.0111 +- 0.0056 | 0.0099 | 0.0057 / 0.0253 | 0.0009 | 2.1537 | 0.674 |
| Lb | 9 | 0.2 | tnbbeta_init0 | 10/10 | 0.0104 +- 0.0036 | 0.0105 | 0.0057 / 0.0147 | 0.0010 | 2.1537 | 0.674 |
| Lb | 17 | 0.2 | vmf | 10/10 | 0.1104 +- 0.0007 | 0.1104 | 0.1092 / 0.1113 | 0.0437 | 1.0101 | 0.445 |
| Lb | 17 | 0.2 | power_spherical | 10/10 | 0.0825 +- 0.0015 | 0.0824 | 0.0807 / 0.0860 | 0.0331 | 1.0101 | 0.445 |
| Lb | 17 | 0.2 | tnbbeta | 10/10 | 0.0054 +- 0.0062 | 0.0037 | 0.0016 / 0.0224 | 0.0006 | 1.0101 | 0.445 |
| Lb | 17 | 0.2 | tnbbeta_init0 | 10/10 | 0.0057 +- 0.0055 | 0.0042 | 0.0017 / 0.0201 | 0.0006 | 1.0101 | 0.445 |
| M | 3 | 0.2 | vmf | 10/10 | 0.0755 +- 0.0006 | 0.0754 | 0.0749 / 0.0766 | 0.0317 | 7.1797 | 1.072 |
| M | 3 | 0.2 | power_spherical | 10/10 | 0.0367 +- 0.0007 | 0.0369 | 0.0354 / 0.0379 | 0.0259 | 7.1797 | 1.072 |
| M | 3 | 0.2 | tnbbeta | 10/10 | 0.0305 +- 0.0096 | 0.0344 | 0.0136 / 0.0414 | 0.0119 | 7.1797 | 1.072 |
| M | 3 | 0.2 | tnbbeta_init0 | 10/10 | 0.0325 +- 0.0099 | 0.0348 | 0.0157 / 0.0479 | 0.0157 | 7.1797 | 1.072 |
| M | 5 | 0.2 | vmf | 10/10 | 0.0448 +- 0.0003 | 0.0446 | 0.0445 / 0.0456 | 0.0153 | 4.1231 | 0.881 |
| M | 5 | 0.2 | power_spherical | 10/10 | 0.0155 +- 0.0001 | 0.0155 | 0.0154 / 0.0158 | 0.0155 | 4.1231 | 0.881 |
| M | 5 | 0.2 | tnbbeta | 10/10 | 0.0167 +- 0.0021 | 0.0159 | 0.0146 / 0.0202 | 0.0138 | 4.1231 | 0.881 |
| M | 5 | 0.2 | tnbbeta_init0 | 10/10 | 0.0152 +- 0.0057 | 0.0153 | 0.0089 / 0.0271 | 0.0070 | 4.1231 | 0.881 |
| M | 9 | 0.2 | vmf | 10/10 | 0.0208 +- 0.0007 | 0.0206 | 0.0200 / 0.0224 | 0.0040 | 2.1537 | 0.674 |
| M | 9 | 0.2 | power_spherical | 10/10 | 0.0041 +- 0.0003 | 0.0040 | 0.0038 / 0.0048 | 0.0041 | 2.1537 | 0.674 |
| M | 9 | 0.2 | tnbbeta | 10/10 | 0.0048 +- 0.0010 | 0.0045 | 0.0040 / 0.0072 | 0.0041 | 2.1537 | 0.674 |
| M | 9 | 0.2 | tnbbeta_init0 | 10/10 | 0.0051 +- 0.0005 | 0.0052 | 0.0043 / 0.0058 | 0.0039 | 2.1537 | 0.674 |
| M | 17 | 0.2 | vmf | 10/10 | 0.0097 +- 0.0012 | 0.0097 | 0.0085 / 0.0124 | 0.0014 | 1.0101 | 0.445 |
| M | 17 | 0.2 | power_spherical | 10/10 | 0.0013 +- 0.0014 | 0.0008 | -0.0001 / 0.0044 | 0.0013 | 1.0101 | 0.445 |
| M | 17 | 0.2 | tnbbeta | 10/10 | 0.0022 +- 0.0011 | 0.0021 | 0.0006 / 0.0047 | 0.0015 | 1.0101 | 0.445 |
| M | 17 | 0.2 | tnbbeta_init0 | 10/10 | 0.0020 +- 0.0012 | 0.0020 | 0.0002 / 0.0045 | 0.0015 | 1.0101 | 0.445 |
| F | 3 | 0.2 | vmf | 10/10 | 0.0751 +- 0.0006 | 0.0749 | 0.0745 / 0.0763 | 0.0317 | 7.1797 | 1.072 |
| F | 3 | 0.2 | power_spherical | 10/10 | 0.0368 +- 0.0008 | 0.0368 | 0.0355 / 0.0385 | 0.0262 | 7.1797 | 1.072 |
| F | 3 | 0.2 | tnbbeta | 10/10 | 0.0359 +- 0.0020 | 0.0356 | 0.0341 / 0.0408 | 0.0308 | 7.1797 | 1.072 |
| F | 3 | 0.2 | tnbbeta_init0 | 10/10 | 0.0338 +- 0.0086 | 0.0351 | 0.0132 / 0.0450 | 0.0183 | 7.1797 | 1.072 |
| F | 5 | 0.2 | vmf | 10/10 | 0.0450 +- 0.0006 | 0.0448 | 0.0446 / 0.0462 | 0.0155 | 4.1231 | 0.881 |
| F | 5 | 0.2 | power_spherical | 10/10 | 0.0154 +- 0.0000 | 0.0154 | 0.0154 / 0.0155 | 0.0154 | 4.1231 | 0.881 |
| F | 5 | 0.2 | tnbbeta | 10/10 | 0.0150 +- 0.0007 | 0.0150 | 0.0136 / 0.0162 | 0.0129 | 4.1231 | 0.881 |
| F | 5 | 0.2 | tnbbeta_init0 | 10/10 | 0.0139 +- 0.0021 | 0.0147 | 0.0092 / 0.0157 | 0.0102 | 4.1231 | 0.881 |
| F | 9 | 0.2 | vmf | 10/10 | 0.0207 +- 0.0004 | 0.0207 | 0.0201 / 0.0216 | 0.0038 | 2.1537 | 0.674 |
| F | 9 | 0.2 | power_spherical | 10/10 | 0.0040 +- 0.0001 | 0.0040 | 0.0038 / 0.0042 | 0.0040 | 2.1537 | 0.674 |
| F | 9 | 0.2 | tnbbeta | 10/10 | 0.0046 +- 0.0006 | 0.0044 | 0.0040 / 0.0057 | 0.0040 | 2.1537 | 0.674 |
| F | 9 | 0.2 | tnbbeta_init0 | 10/10 | 0.0044 +- 0.0003 | 0.0044 | 0.0039 / 0.0049 | 0.0038 | 2.1537 | 0.674 |
| F | 17 | 0.2 | vmf | 10/10 | 0.0098 +- 0.0017 | 0.0092 | 0.0084 / 0.0139 | 0.0014 | 1.0101 | 0.445 |
| F | 17 | 0.2 | power_spherical | 10/10 | 0.0010 +- 0.0008 | 0.0006 | 0.0001 / 0.0025 | 0.0010 | 1.0101 | 0.445 |
| F | 17 | 0.2 | tnbbeta | 10/10 | 0.0016 +- 0.0010 | 0.0011 | 0.0004 / 0.0036 | 0.0010 | 1.0101 | 0.445 |
| F | 17 | 0.2 | tnbbeta_init0 | 10/10 | 0.0013 +- 0.0008 | 0.0010 | 0.0005 / 0.0030 | 0.0010 | 1.0101 | 0.445 |

## Confirmatory tests, decoder L (one-sided paired Wilcoxon, Holm 0.05, 24)

| d | sigma | comparator | n | mean paired diff (comparator - tnbbeta) | seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.05 | confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.1 | power_spherical | 10 | 0.1285 | 10/10 | 0.00098 | 0.00098 | 0.00217 | True | True | True |
| 3 | 0.1 | vmf | 10 | 0.2088 | 10/10 | 0.00098 | 0.00098 | 0.00208 | True | True | True |
| 5 | 0.1 | power_spherical | 10 | 0.1369 | 10/10 | 0.00098 | 0.00098 | 0.00294 | True | True | True |
| 5 | 0.1 | vmf | 10 | 0.2049 | 10/10 | 0.00098 | 0.00098 | 0.00278 | True | True | True |
| 9 | 0.1 | power_spherical | 10 | 0.1399 | 10/10 | 0.00098 | 0.00098 | 0.00455 | True | True | True |
| 9 | 0.1 | vmf | 10 | 0.1853 | 10/10 | 0.00098 | 0.00098 | 0.00417 | True | True | True |
| 17 | 0.1 | power_spherical | 10 | 0.1493 | 10/10 | 0.00098 | 0.00098 | 0.01000 | True | True | True |
| 17 | 0.1 | vmf | 10 | 0.1820 | 10/10 | 0.00098 | 0.00098 | 0.00833 | True | True | True |
| 3 | 0.2 | power_spherical | 10 | 0.0697 | 10/10 | 0.00098 | 0.00098 | 0.00238 | True | True | True |
| 3 | 0.2 | vmf | 10 | 0.1465 | 10/10 | 0.00098 | 0.00098 | 0.00227 | True | True | True |
| 5 | 0.2 | power_spherical | 10 | 0.0963 | 10/10 | 0.00098 | 0.00098 | 0.00333 | True | True | True |
| 5 | 0.2 | vmf | 10 | 0.1619 | 10/10 | 0.00098 | 0.00098 | 0.00313 | True | True | True |
| 9 | 0.2 | power_spherical | 10 | 0.0897 | 10/10 | 0.00098 | 0.00098 | 0.00556 | True | True | True |
| 9 | 0.2 | vmf | 10 | 0.1309 | 10/10 | 0.00098 | 0.00098 | 0.00500 | True | True | True |
| 17 | 0.2 | power_spherical | 10 | 0.0774 | 10/10 | 0.00098 | 0.00098 | 0.01667 | True | True | True |
| 17 | 0.2 | vmf | 10 | 0.1055 | 10/10 | 0.00098 | 0.00098 | 0.01250 | True | True | True |
| 3 | 0.4 | power_spherical | 10 | 0.0218 | 10/10 | 0.00098 | 0.00098 | 0.00263 | True | False | False |
| 3 | 0.4 | vmf | 10 | 0.0876 | 10/10 | 0.00098 | 0.00098 | 0.00250 | True | True | True |
| 5 | 0.4 | power_spherical | 10 | 0.0335 | 10/10 | 0.00098 | 0.00098 | 0.00385 | True | False | False |
| 5 | 0.4 | vmf | 10 | 0.0870 | 10/10 | 0.00098 | 0.00098 | 0.00357 | True | True | True |
| 9 | 0.4 | power_spherical | 10 | 0.0226 | 10/10 | 0.00098 | 0.00098 | 0.00714 | True | False | False |
| 9 | 0.4 | vmf | 10 | 0.0502 | 10/10 | 0.00098 | 0.00098 | 0.00625 | True | True | True |
| 17 | 0.4 | power_spherical | 10 | 0.0095 | 10/10 | 0.00098 | 0.00098 | 0.05000 | True | False | False |
| 17 | 0.4 | vmf | 10 | 0.0236 | 10/10 | 0.00098 | 0.00098 | 0.02500 | True | False | False |

Confirmed (Holm and margin): 19 of 24.

| d | sigma | confirmed against both comparators |
|---|---|---|
| 3 | 0.1 | True |
| 3 | 0.2 | True |
| 3 | 0.4 | False |
| 5 | 0.1 | True |
| 5 | 0.2 | True |
| 5 | 0.4 | False |
| 9 | 0.1 | True |
| 9 | 0.2 | True |
| 9 | 0.4 | False |
| 17 | 0.1 | True |
| 17 | 0.2 | True |
| 17 | 0.4 | False |

## Descriptive trend in d (mean paired gap difference, comparator - tnbbeta)

| decoder | sigma | comparator | d=3 | d=5 | d=9 | d=17 | slope vs log2(d-1) |
|---|---|---|---|---|---|---|---|
| L | 0.1 | vmf | 0.2088 | 0.2049 | 0.1853 | 0.1820 | -0.0100 |
| L | 0.1 | power_spherical | 0.1285 | 0.1369 | 0.1399 | 0.1493 | 0.0065 |
| L | 0.2 | vmf | 0.1465 | 0.1619 | 0.1309 | 0.1055 | -0.0154 |
| L | 0.2 | power_spherical | 0.0697 | 0.0963 | 0.0897 | 0.0774 | 0.0017 |
| L | 0.4 | vmf | 0.0876 | 0.0870 | 0.0502 | 0.0236 | -0.0229 |
| L | 0.4 | power_spherical | 0.0218 | 0.0335 | 0.0226 | 0.0095 | -0.0048 |
| Lb | 0.2 | vmf | 0.1518 | 0.1607 | 0.1332 | 0.1050 | -0.0168 |
| Lb | 0.2 | power_spherical | 0.0739 | 0.0957 | 0.0925 | 0.0771 | 0.0006 |
| M | 0.2 | vmf | 0.0450 | 0.0281 | 0.0160 | 0.0075 | -0.0125 |
| M | 0.2 | power_spherical | 0.0062 | -0.0011 | -0.0007 | -0.0010 | -0.0021 |
| F | 0.2 | vmf | 0.0391 | 0.0300 | 0.0161 | 0.0083 | -0.0106 |
| F | 0.2 | power_spherical | 0.0008 | 0.0005 | -0.0006 | -0.0006 | -0.0005 |

## Per-seed ELBO gap


L, d = 3, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.2787 | 0.1988 | 0.0582 | 0.0417 |
| 1 | 0.2798 | 0.1999 | 0.1027 | 0.0421 |
| 2 | 0.2793 | 0.1984 | 0.0503 | 0.0527 |
| 3 | 0.2787 | 0.1979 | 0.0525 | 0.0892 |
| 4 | 0.2789 | 0.1980 | 0.0504 | 0.1182 |
| 5 | 0.2796 | 0.1984 | 0.1082 | 0.0577 |
| 6 | 0.2794 | 0.2011 | 0.0842 | 0.0923 |
| 7 | 0.2799 | 0.1979 | 0.0623 | 0.0589 |
| 8 | 0.2796 | 0.1977 | 0.0479 | 0.0419 |
| 9 | 0.2792 | 0.2020 | 0.0883 | 0.1091 |

L, d = 5, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.2365 | 0.1697 | 0.0314 | 0.0336 |
| 1 | 0.2394 | 0.1688 | 0.0300 | 0.0583 |
| 2 | 0.2396 | 0.1696 | 0.0287 | 0.0459 |
| 3 | 0.2373 | 0.1687 | 0.0334 | 0.0321 |
| 4 | 0.2381 | 0.1691 | 0.0250 | 0.0277 |
| 5 | 0.2376 | 0.1689 | 0.0338 | 0.0373 |
| 6 | 0.2350 | 0.1691 | 0.0275 | 0.1006 |
| 7 | 0.2358 | 0.1720 | 0.0416 | 0.0621 |
| 8 | 0.2378 | 0.1694 | 0.0381 | 0.0274 |
| 9 | 0.2378 | 0.1694 | 0.0358 | 0.0353 |

L, d = 9, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.2037 | 0.1594 | 0.0144 | 0.0183 |
| 1 | 0.2058 | 0.1598 | 0.0246 | 0.0107 |
| 2 | 0.2040 | 0.1590 | 0.0135 | 0.0111 |
| 3 | 0.2063 | 0.1586 | 0.0118 | 0.0340 |
| 4 | 0.2045 | 0.1590 | 0.0140 | 0.0103 |
| 5 | 0.2031 | 0.1582 | 0.0138 | 0.0096 |
| 6 | 0.2027 | 0.1580 | 0.0178 | 0.0117 |
| 7 | 0.2044 | 0.1604 | 0.0258 | 0.0199 |
| 8 | 0.2040 | 0.1588 | 0.0355 | 0.0115 |
| 9 | 0.2044 | 0.1578 | 0.0186 | 0.0115 |

L, d = 17, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1956 | 0.1614 | 0.0296 | 0.0075 |
| 1 | 0.1945 | 0.1623 | 0.0178 | 0.0104 |
| 2 | 0.1946 | 0.1630 | 0.0125 | 0.0228 |
| 3 | 0.1951 | 0.1642 | 0.0086 | 0.0106 |
| 4 | 0.1950 | 0.1634 | 0.0059 | 0.0074 |
| 5 | 0.1956 | 0.1632 | 0.0208 | 0.0158 |
| 6 | 0.1947 | 0.1618 | 0.0082 | 0.0339 |
| 7 | 0.1969 | 0.1629 | 0.0157 | 0.0207 |
| 8 | 0.1973 | 0.1622 | 0.0082 | 0.0277 |
| 9 | 0.1945 | 0.1629 | 0.0065 | 0.0116 |

L, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.2097 | 0.1312 | 0.0543 | 0.0512 |
| 1 | 0.2103 | 0.1391 | 0.0632 | 0.0511 |
| 2 | 0.2101 | 0.1337 | 0.0648 | 0.0501 |
| 3 | 0.2093 | 0.1322 | 0.0534 | 0.0552 |
| 4 | 0.2093 | 0.1310 | 0.0528 | 0.0505 |
| 5 | 0.2097 | 0.1321 | 0.0586 | 0.0808 |
| 6 | 0.2097 | 0.1325 | 0.1218 | 0.0558 |
| 7 | 0.2097 | 0.1315 | 0.0572 | 0.0543 |
| 8 | 0.2104 | 0.1345 | 0.0530 | 0.0508 |
| 9 | 0.2099 | 0.1322 | 0.0542 | 0.0740 |

L, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1887 | 0.1238 | 0.0358 | 0.0286 |
| 1 | 0.1888 | 0.1229 | 0.0454 | 0.0257 |
| 2 | 0.1878 | 0.1242 | 0.0321 | 0.0199 |
| 3 | 0.1887 | 0.1230 | 0.0220 | 0.0206 |
| 4 | 0.1915 | 0.1228 | 0.0212 | 0.0213 |
| 5 | 0.1880 | 0.1238 | 0.0212 | 0.0209 |
| 6 | 0.1868 | 0.1240 | 0.0277 | 0.0272 |
| 7 | 0.1890 | 0.1238 | 0.0195 | 0.0258 |
| 8 | 0.1912 | 0.1234 | 0.0251 | 0.0228 |
| 9 | 0.1906 | 0.1233 | 0.0217 | 0.0198 |

L, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1436 | 0.1039 | 0.0102 | 0.0154 |
| 1 | 0.1449 | 0.1035 | 0.0100 | 0.0055 |
| 2 | 0.1469 | 0.1025 | 0.0135 | 0.0143 |
| 3 | 0.1447 | 0.1020 | 0.0117 | 0.0072 |
| 4 | 0.1440 | 0.1026 | 0.0073 | 0.0165 |
| 5 | 0.1427 | 0.1033 | 0.0080 | 0.0060 |
| 6 | 0.1431 | 0.1021 | 0.0100 | 0.0096 |
| 7 | 0.1438 | 0.1031 | 0.0358 | 0.0130 |
| 8 | 0.1438 | 0.1036 | 0.0148 | 0.0063 |
| 9 | 0.1434 | 0.1023 | 0.0108 | 0.0177 |

L, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1094 | 0.0809 | 0.0092 | 0.0174 |
| 1 | 0.1102 | 0.0810 | 0.0025 | 0.0042 |
| 2 | 0.1113 | 0.0822 | 0.0109 | 0.0049 |
| 3 | 0.1129 | 0.0851 | 0.0028 | 0.0035 |
| 4 | 0.1102 | 0.0826 | 0.0036 | 0.0032 |
| 5 | 0.1098 | 0.0820 | 0.0035 | 0.0031 |
| 6 | 0.1098 | 0.0818 | 0.0044 | 0.0023 |
| 7 | 0.1102 | 0.0822 | 0.0029 | 0.0048 |
| 8 | 0.1094 | 0.0826 | 0.0056 | 0.0056 |
| 9 | 0.1099 | 0.0823 | 0.0029 | 0.0024 |

L, d = 3, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1183 | 0.0528 | 0.0356 | 0.0294 |
| 1 | 0.1183 | 0.0511 | 0.0330 | 0.0295 |
| 2 | 0.1187 | 0.0511 | 0.0290 | 0.0284 |
| 3 | 0.1180 | 0.0531 | 0.0313 | 0.0292 |
| 4 | 0.1179 | 0.0540 | 0.0300 | 0.0296 |
| 5 | 0.1182 | 0.0533 | 0.0293 | 0.0310 |
| 6 | 0.1177 | 0.0535 | 0.0319 | 0.0343 |
| 7 | 0.1185 | 0.0517 | 0.0285 | 0.0509 |
| 8 | 0.1187 | 0.0521 | 0.0295 | 0.0326 |
| 9 | 0.1189 | 0.0527 | 0.0294 | 0.0393 |

L, d = 5, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0974 | 0.0426 | 0.0157 | 0.0110 |
| 1 | 0.0994 | 0.0451 | 0.0100 | 0.0118 |
| 2 | 0.0970 | 0.0435 | 0.0132 | 0.0101 |
| 3 | 0.0978 | 0.0448 | 0.0091 | 0.0083 |
| 4 | 0.0986 | 0.0447 | 0.0114 | 0.0098 |
| 5 | 0.0963 | 0.0431 | 0.0089 | 0.0095 |
| 6 | 0.0984 | 0.0444 | 0.0107 | 0.0217 |
| 7 | 0.0962 | 0.0441 | 0.0096 | 0.0083 |
| 8 | 0.0969 | 0.0445 | 0.0099 | 0.0084 |
| 9 | 0.0994 | 0.0456 | 0.0088 | 0.0093 |

L, d = 9, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0529 | 0.0254 | 0.0038 | 0.0028 |
| 1 | 0.0526 | 0.0244 | 0.0032 | 0.0018 |
| 2 | 0.0530 | 0.0257 | 0.0023 | 0.0022 |
| 3 | 0.0528 | 0.0272 | 0.0022 | 0.0033 |
| 4 | 0.0543 | 0.0267 | 0.0041 | 0.0036 |
| 5 | 0.0531 | 0.0254 | 0.0033 | 0.0058 |
| 6 | 0.0525 | 0.0244 | 0.0019 | 0.0028 |
| 7 | 0.0540 | 0.0262 | 0.0041 | 0.0023 |
| 8 | 0.0536 | 0.0254 | 0.0028 | 0.0027 |
| 9 | 0.0532 | 0.0256 | 0.0023 | 0.0024 |

L, d = 17, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0255 | 0.0108 | 0.0015 | 0.0009 |
| 1 | 0.0253 | 0.0112 | 0.0009 | 0.0011 |
| 2 | 0.0253 | 0.0109 | 0.0013 | 0.0020 |
| 3 | 0.0257 | 0.0112 | 0.0010 | 0.0021 |
| 4 | 0.0254 | 0.0110 | 0.0022 | 0.0010 |
| 5 | 0.0246 | 0.0109 | 0.0008 | 0.0007 |
| 6 | 0.0250 | 0.0108 | 0.0034 | 0.0010 |
| 7 | 0.0249 | 0.0117 | 0.0016 | 0.0009 |
| 8 | 0.0244 | 0.0110 | 0.0018 | 0.0012 |
| 9 | 0.0253 | 0.0115 | 0.0012 | 0.0006 |

Lb, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.2088 | 0.1319 | 0.0544 | 0.0502 |
| 1 | 0.2095 | 0.1325 | 0.0505 | 0.0524 |
| 2 | 0.2101 | 0.1317 | 0.0625 | 0.0498 |
| 3 | 0.2094 | 0.1312 | 0.0533 | 0.0744 |
| 4 | 0.2094 | 0.1323 | 0.0521 | 0.0499 |
| 5 | 0.2100 | 0.1307 | 0.0615 | 0.0522 |
| 6 | 0.2098 | 0.1318 | 0.0548 | 0.0514 |
| 7 | 0.2096 | 0.1311 | 0.0756 | 0.0595 |
| 8 | 0.2097 | 0.1319 | 0.0617 | 0.0517 |
| 9 | 0.2098 | 0.1325 | 0.0521 | 0.0634 |

Lb, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1873 | 0.1237 | 0.0390 | 0.0344 |
| 1 | 0.1913 | 0.1227 | 0.0221 | 0.0295 |
| 2 | 0.1915 | 0.1233 | 0.0312 | 0.0324 |
| 3 | 0.1887 | 0.1250 | 0.0253 | 0.0244 |
| 4 | 0.1929 | 0.1227 | 0.0241 | 0.0242 |
| 5 | 0.1883 | 0.1238 | 0.0267 | 0.0198 |
| 6 | 0.1877 | 0.1249 | 0.0256 | 0.0187 |
| 7 | 0.1888 | 0.1246 | 0.0269 | 0.0187 |
| 8 | 0.1877 | 0.1261 | 0.0219 | 0.0230 |
| 9 | 0.1869 | 0.1247 | 0.0413 | 0.0253 |

Lb, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1462 | 0.1053 | 0.0121 | 0.0128 |
| 1 | 0.1443 | 0.1036 | 0.0253 | 0.0057 |
| 2 | 0.1435 | 0.1048 | 0.0107 | 0.0135 |
| 3 | 0.1442 | 0.1031 | 0.0091 | 0.0077 |
| 4 | 0.1469 | 0.1044 | 0.0074 | 0.0144 |
| 5 | 0.1435 | 0.1049 | 0.0109 | 0.0082 |
| 6 | 0.1449 | 0.1034 | 0.0078 | 0.0147 |
| 7 | 0.1430 | 0.1031 | 0.0078 | 0.0129 |
| 8 | 0.1435 | 0.1021 | 0.0057 | 0.0076 |
| 9 | 0.1438 | 0.1020 | 0.0148 | 0.0062 |

Lb, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1103 | 0.0826 | 0.0055 | 0.0056 |
| 1 | 0.1105 | 0.0811 | 0.0224 | 0.0017 |
| 2 | 0.1110 | 0.0833 | 0.0045 | 0.0085 |
| 3 | 0.1113 | 0.0860 | 0.0061 | 0.0050 |
| 4 | 0.1104 | 0.0825 | 0.0026 | 0.0017 |
| 5 | 0.1107 | 0.0816 | 0.0033 | 0.0201 |
| 6 | 0.1111 | 0.0807 | 0.0040 | 0.0038 |
| 7 | 0.1097 | 0.0823 | 0.0016 | 0.0034 |
| 8 | 0.1098 | 0.0828 | 0.0022 | 0.0045 |
| 9 | 0.1092 | 0.0819 | 0.0016 | 0.0024 |

M, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0750 | 0.0369 | 0.0241 | 0.0252 |
| 1 | 0.0765 | 0.0359 | 0.0228 | 0.0324 |
| 2 | 0.0754 | 0.0369 | 0.0326 | 0.0374 |
| 3 | 0.0753 | 0.0371 | 0.0391 | 0.0479 |
| 4 | 0.0749 | 0.0365 | 0.0361 | 0.0383 |
| 5 | 0.0766 | 0.0354 | 0.0136 | 0.0157 |
| 6 | 0.0755 | 0.0370 | 0.0373 | 0.0367 |
| 7 | 0.0749 | 0.0361 | 0.0414 | 0.0396 |
| 8 | 0.0755 | 0.0379 | 0.0378 | 0.0187 |
| 9 | 0.0754 | 0.0371 | 0.0198 | 0.0330 |

M, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0445 | 0.0154 | 0.0198 | 0.0192 |
| 1 | 0.0446 | 0.0154 | 0.0157 | 0.0159 |
| 2 | 0.0450 | 0.0155 | 0.0146 | 0.0149 |
| 3 | 0.0446 | 0.0154 | 0.0160 | 0.0091 |
| 4 | 0.0456 | 0.0158 | 0.0187 | 0.0116 |
| 5 | 0.0447 | 0.0154 | 0.0150 | 0.0196 |
| 6 | 0.0447 | 0.0155 | 0.0149 | 0.0156 |
| 7 | 0.0448 | 0.0157 | 0.0158 | 0.0271 |
| 8 | 0.0446 | 0.0154 | 0.0202 | 0.0089 |
| 9 | 0.0446 | 0.0155 | 0.0159 | 0.0103 |

M, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0205 | 0.0040 | 0.0047 | 0.0049 |
| 1 | 0.0205 | 0.0039 | 0.0040 | 0.0047 |
| 2 | 0.0208 | 0.0038 | 0.0043 | 0.0052 |
| 3 | 0.0211 | 0.0048 | 0.0055 | 0.0052 |
| 4 | 0.0206 | 0.0039 | 0.0049 | 0.0058 |
| 5 | 0.0200 | 0.0038 | 0.0042 | 0.0048 |
| 6 | 0.0210 | 0.0042 | 0.0042 | 0.0052 |
| 7 | 0.0224 | 0.0044 | 0.0072 | 0.0056 |
| 8 | 0.0203 | 0.0039 | 0.0042 | 0.0043 |
| 9 | 0.0205 | 0.0044 | 0.0047 | 0.0052 |

M, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0124 | 0.0044 | 0.0047 | 0.0045 |
| 1 | 0.0109 | 0.0014 | 0.0032 | 0.0030 |
| 2 | 0.0098 | 0.0015 | 0.0022 | 0.0019 |
| 3 | 0.0100 | 0.0031 | 0.0020 | 0.0023 |
| 4 | 0.0085 | -0.0001 | 0.0009 | 0.0002 |
| 5 | 0.0098 | 0.0011 | 0.0023 | 0.0021 |
| 6 | 0.0085 | 0.0002 | 0.0006 | 0.0007 |
| 7 | 0.0089 | 0.0004 | 0.0024 | 0.0015 |
| 8 | 0.0088 | 0.0005 | 0.0021 | 0.0011 |
| 9 | 0.0095 | 0.0003 | 0.0019 | 0.0028 |

F, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0745 | 0.0371 | 0.0347 | 0.0266 |
| 1 | 0.0750 | 0.0373 | 0.0353 | 0.0402 |
| 2 | 0.0754 | 0.0362 | 0.0344 | 0.0450 |
| 3 | 0.0749 | 0.0355 | 0.0367 | 0.0348 |
| 4 | 0.0747 | 0.0361 | 0.0347 | 0.0132 |
| 5 | 0.0763 | 0.0385 | 0.0365 | 0.0384 |
| 6 | 0.0750 | 0.0368 | 0.0408 | 0.0343 |
| 7 | 0.0745 | 0.0367 | 0.0341 | 0.0344 |
| 8 | 0.0757 | 0.0371 | 0.0359 | 0.0353 |
| 9 | 0.0748 | 0.0365 | 0.0364 | 0.0359 |

F, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0455 | 0.0154 | 0.0151 | 0.0119 |
| 1 | 0.0447 | 0.0154 | 0.0136 | 0.0154 |
| 2 | 0.0446 | 0.0155 | 0.0149 | 0.0149 |
| 3 | 0.0446 | 0.0154 | 0.0150 | 0.0157 |
| 4 | 0.0462 | 0.0155 | 0.0144 | 0.0145 |
| 5 | 0.0449 | 0.0154 | 0.0150 | 0.0156 |
| 6 | 0.0448 | 0.0155 | 0.0154 | 0.0126 |
| 7 | 0.0446 | 0.0155 | 0.0147 | 0.0146 |
| 8 | 0.0448 | 0.0154 | 0.0162 | 0.0092 |
| 9 | 0.0456 | 0.0154 | 0.0156 | 0.0148 |

F, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0205 | 0.0040 | 0.0050 | 0.0043 |
| 1 | 0.0209 | 0.0040 | 0.0041 | 0.0046 |
| 2 | 0.0208 | 0.0040 | 0.0040 | 0.0044 |
| 3 | 0.0209 | 0.0041 | 0.0049 | 0.0045 |
| 4 | 0.0205 | 0.0038 | 0.0057 | 0.0040 |
| 5 | 0.0201 | 0.0038 | 0.0042 | 0.0039 |
| 6 | 0.0205 | 0.0040 | 0.0046 | 0.0044 |
| 7 | 0.0216 | 0.0042 | 0.0050 | 0.0044 |
| 8 | 0.0206 | 0.0039 | 0.0041 | 0.0049 |
| 9 | 0.0207 | 0.0039 | 0.0041 | 0.0042 |

F, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0139 | 0.0022 | 0.0030 | 0.0024 |
| 1 | 0.0100 | 0.0006 | 0.0007 | 0.0011 |
| 2 | 0.0098 | 0.0015 | 0.0017 | 0.0019 |
| 3 | 0.0118 | 0.0025 | 0.0036 | 0.0030 |
| 4 | 0.0091 | 0.0006 | 0.0019 | 0.0010 |
| 5 | 0.0094 | 0.0008 | 0.0011 | 0.0010 |
| 6 | 0.0084 | 0.0005 | 0.0011 | 0.0009 |
| 7 | 0.0086 | 0.0003 | 0.0011 | 0.0006 |
| 8 | 0.0086 | 0.0004 | 0.0011 | 0.0005 |
| 9 | 0.0087 | 0.0001 | 0.0004 | 0.0008 |

## Per-seed |cos(u, a)| (symmetry check: not identified, see Deviations)


L, d = 3, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9101 | 0.8498 | 0.8773 | 0.8501 |
| 1 | 0.1329 | 0.1391 | 0.1349 | 0.1102 |
| 2 | 0.3361 | 0.0055 | 0.0122 | 0.0659 |
| 3 | 0.9649 | 0.9709 | 0.9794 | 0.9618 |
| 4 | 0.6907 | 0.7076 | 0.7153 | 0.7301 |
| 5 | 0.2102 | 0.2607 | 0.3066 | 0.2782 |
| 6 | 0.0494 | 0.0900 | 0.1917 | 0.1751 |
| 7 | 0.9466 | 0.9449 | 0.9519 | 0.9300 |
| 8 | 0.0176 | 0.0188 | 0.0044 | 0.0199 |
| 9 | 0.2060 | 0.0281 | 0.0583 | 0.0893 |

L, d = 5, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.5253 | 0.5243 | 0.5821 | 0.5570 |
| 1 | 0.1100 | 0.1209 | 0.1304 | 0.1518 |
| 2 | 0.0080 | 0.0127 | 0.0101 | 0.0011 |
| 3 | 0.9363 | 0.9468 | 0.9563 | 0.9230 |
| 4 | 0.5887 | 0.5909 | 0.6116 | 0.5959 |
| 5 | 0.2948 | 0.2512 | 0.2951 | 0.2925 |
| 6 | 0.0644 | 0.0065 | 0.0215 | 0.0246 |
| 7 | 0.9145 | 0.9181 | 0.9179 | 0.9192 |
| 8 | 0.0404 | 0.0368 | 0.0515 | 0.0492 |
| 9 | 0.0501 | 0.0196 | 0.0252 | 0.0099 |

L, d = 9, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.4282 | 0.4407 | 0.4559 | 0.4589 |
| 1 | 0.1013 | 0.1143 | 0.1327 | 0.1212 |
| 2 | 0.0303 | 0.0027 | 0.0155 | 0.0041 |
| 3 | 0.6250 | 0.6238 | 0.6780 | 0.6445 |
| 4 | 0.4551 | 0.4626 | 0.4684 | 0.4667 |
| 5 | 0.1744 | 0.1760 | 0.1824 | 0.1690 |
| 6 | 0.0132 | 0.0234 | 0.0327 | 0.0243 |
| 7 | 0.5319 | 0.5380 | 0.5507 | 0.5470 |
| 8 | 0.0081 | 0.0028 | 0.0179 | 0.0186 |
| 9 | 0.0196 | 0.0203 | 0.0206 | 0.0044 |

L, d = 17, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0225 | 0.0166 | 0.0130 | 0.0093 |
| 1 | 0.1054 | 0.0916 | 0.0991 | 0.0960 |
| 2 | 0.2412 | 0.2389 | 0.2355 | 0.2377 |
| 3 | 0.3633 | 0.3446 | 0.3571 | 0.3500 |
| 4 | 0.2306 | 0.2283 | 0.2430 | 0.2441 |
| 5 | 0.0792 | 0.0653 | 0.0816 | 0.0801 |
| 6 | 0.0019 | 0.0083 | 0.0037 | 0.0006 |
| 7 | 0.0209 | 0.0260 | 0.0246 | 0.0254 |
| 8 | 0.3680 | 0.3701 | 0.3665 | 0.3711 |
| 9 | 0.1553 | 0.1660 | 0.1647 | 0.1633 |

L, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.8814 | 0.8536 | 0.9042 | 0.8519 |
| 1 | 0.1261 | 0.1466 | 0.1607 | 0.1125 |
| 2 | 0.0330 | 0.0187 | 0.0443 | 0.0152 |
| 3 | 0.9899 | 0.9817 | 0.9691 | 0.9617 |
| 4 | 0.6887 | 0.7044 | 0.7204 | 0.6854 |
| 5 | 0.2221 | 0.2682 | 0.2689 | 0.3238 |
| 6 | 0.0751 | 0.1227 | 0.1376 | 0.2876 |
| 7 | 0.9492 | 0.9397 | 0.9400 | 0.9429 |
| 8 | 0.0122 | 0.0021 | 0.0052 | 0.0195 |
| 9 | 0.0838 | 0.0344 | 0.0638 | 0.0142 |

L, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.5080 | 0.5264 | 0.5493 | 0.5505 |
| 1 | 0.0985 | 0.1348 | 0.1252 | 0.1580 |
| 2 | 0.0227 | 0.0019 | 0.0162 | 0.0213 |
| 3 | 0.9508 | 0.9435 | 0.9328 | 0.9215 |
| 4 | 0.5707 | 0.5891 | 0.6005 | 0.5887 |
| 5 | 0.2684 | 0.2510 | 0.2970 | 0.2878 |
| 6 | 0.0494 | 0.0162 | 0.0163 | 0.0311 |
| 7 | 0.9210 | 0.9127 | 0.9125 | 0.9204 |
| 8 | 0.0537 | 0.0377 | 0.0411 | 0.0550 |
| 9 | 0.0390 | 0.0106 | 0.0260 | 0.0147 |

L, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.4968 | 0.4388 | 0.4384 | 0.4464 |
| 1 | 0.1213 | 0.1137 | 0.1274 | 0.1171 |
| 2 | 0.0089 | 0.0046 | 0.0044 | 0.0116 |
| 3 | 0.6469 | 0.6326 | 0.6670 | 0.6401 |
| 4 | 0.4705 | 0.4612 | 0.4712 | 0.4584 |
| 5 | 0.1771 | 0.1803 | 0.1845 | 0.1851 |
| 6 | 0.0307 | 0.0250 | 0.0251 | 0.0240 |
| 7 | 0.5335 | 0.5374 | 0.5446 | 0.5402 |
| 8 | 0.0025 | 0.0165 | 0.0079 | 0.0175 |
| 9 | 0.0291 | 0.0220 | 0.0196 | 0.0093 |

L, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0195 | 0.0195 | 0.0155 | 0.0173 |
| 1 | 0.0906 | 0.0934 | 0.0952 | 0.0877 |
| 2 | 0.2305 | 0.2364 | 0.2322 | 0.2328 |
| 3 | 0.3597 | 0.3464 | 0.3571 | 0.3580 |
| 4 | 0.2289 | 0.2314 | 0.2455 | 0.2354 |
| 5 | 0.0736 | 0.0715 | 0.0840 | 0.0826 |
| 6 | 0.0058 | 0.0098 | 0.0087 | 0.0056 |
| 7 | 0.0252 | 0.0304 | 0.0267 | 0.0259 |
| 8 | 0.3640 | 0.3660 | 0.3599 | 0.3701 |
| 9 | 0.1532 | 0.1549 | 0.1603 | 0.1551 |

L, d = 3, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.8865 | 0.8642 | 0.8753 | 0.8526 |
| 1 | 0.2055 | 0.1636 | 0.1466 | 0.1132 |
| 2 | 0.0875 | 0.0012 | 0.0112 | 0.0054 |
| 3 | 0.9695 | 0.9867 | 0.9661 | 0.9683 |
| 4 | 0.6989 | 0.6864 | 0.7446 | 0.6979 |
| 5 | 0.2520 | 0.2752 | 0.2771 | 0.3027 |
| 6 | 0.0573 | 0.1095 | 0.0410 | 0.2186 |
| 7 | 0.9417 | 0.9461 | 0.9473 | 0.9412 |
| 8 | 0.0091 | 0.0049 | 0.0019 | 0.0340 |
| 9 | 0.1758 | 0.0506 | 0.0453 | 0.1412 |

L, d = 5, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.5557 | 0.5287 | 0.5389 | 0.5449 |
| 1 | 0.1189 | 0.1247 | 0.1297 | 0.1537 |
| 2 | 0.0248 | 0.0054 | 0.0408 | 0.0031 |
| 3 | 0.9609 | 0.9415 | 0.9591 | 0.9415 |
| 4 | 0.5683 | 0.5847 | 0.5937 | 0.5705 |
| 5 | 0.2028 | 0.2340 | 0.2704 | 0.2606 |
| 6 | 0.0360 | 0.0145 | 0.0189 | 0.0367 |
| 7 | 0.9149 | 0.9187 | 0.9127 | 0.9232 |
| 8 | 0.0415 | 0.0300 | 0.0332 | 0.0280 |
| 9 | 0.0442 | 0.0204 | 0.0284 | 0.0103 |

L, d = 9, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.4507 | 0.4341 | 0.4219 | 0.4376 |
| 1 | 0.1301 | 0.1112 | 0.1287 | 0.1203 |
| 2 | 0.0146 | 0.0035 | 0.0054 | 0.0071 |
| 3 | 0.6479 | 0.6314 | 0.6656 | 0.6185 |
| 4 | 0.4764 | 0.4618 | 0.4602 | 0.4730 |
| 5 | 0.1856 | 0.1690 | 0.1709 | 0.1737 |
| 6 | 0.0246 | 0.0183 | 0.0257 | 0.0168 |
| 7 | 0.5306 | 0.5336 | 0.5423 | 0.5320 |
| 8 | 0.0246 | 0.0244 | 0.0134 | 0.0128 |
| 9 | 0.0175 | 0.0172 | 0.0127 | 0.0162 |

L, d = 17, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0119 | 0.0190 | 0.0202 | 0.0148 |
| 1 | 0.0892 | 0.0894 | 0.0911 | 0.0810 |
| 2 | 0.2245 | 0.2238 | 0.2311 | 0.2306 |
| 3 | 0.3473 | 0.3426 | 0.3423 | 0.3632 |
| 4 | 0.2348 | 0.2360 | 0.2375 | 0.2433 |
| 5 | 0.0792 | 0.0794 | 0.0756 | 0.0815 |
| 6 | 0.0088 | 0.0013 | 0.0126 | 0.0017 |
| 7 | 0.0207 | 0.0226 | 0.0216 | 0.0228 |
| 8 | 0.3649 | 0.3657 | 0.3630 | 0.3697 |
| 9 | 0.1496 | 0.1614 | 0.1480 | 0.1509 |

Lb, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.8520 | 0.8483 | 0.9057 | 0.8521 |
| 1 | 0.1707 | 0.1519 | 0.1302 | 0.1125 |
| 2 | 0.0320 | 0.0265 | 0.0060 | 0.0145 |
| 3 | 0.9814 | 0.9792 | 0.9599 | 0.9738 |
| 4 | 0.7029 | 0.6987 | 0.7369 | 0.6848 |
| 5 | 0.2556 | 0.2883 | 0.2891 | 0.2536 |
| 6 | 0.1666 | 0.1145 | 0.1745 | 0.1458 |
| 7 | 0.9494 | 0.9495 | 0.9538 | 0.9414 |
| 8 | 0.0235 | 0.0095 | 0.0317 | 0.0199 |
| 9 | 0.0630 | 0.0240 | 0.0290 | 0.0304 |

Lb, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.5214 | 0.5259 | 0.5404 | 0.5352 |
| 1 | 0.0774 | 0.1324 | 0.1315 | 0.1631 |
| 2 | 0.0534 | 0.0061 | 0.0048 | 0.0193 |
| 3 | 0.9346 | 0.9492 | 0.9436 | 0.9232 |
| 4 | 0.6094 | 0.5852 | 0.6059 | 0.5973 |
| 5 | 0.2588 | 0.2398 | 0.2879 | 0.2951 |
| 6 | 0.0244 | 0.0152 | 0.0302 | 0.0412 |
| 7 | 0.9119 | 0.9166 | 0.9103 | 0.9170 |
| 8 | 0.0280 | 0.0363 | 0.0403 | 0.0477 |
| 9 | 0.0488 | 0.0178 | 0.0467 | 0.0116 |

Lb, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.4210 | 0.4467 | 0.4373 | 0.4466 |
| 1 | 0.1190 | 0.1176 | 0.1292 | 0.1213 |
| 2 | 0.0172 | 0.0159 | 0.0075 | 0.0053 |
| 3 | 0.6792 | 0.6296 | 0.6742 | 0.6253 |
| 4 | 0.4740 | 0.4668 | 0.4674 | 0.4712 |
| 5 | 0.1707 | 0.1710 | 0.1806 | 0.1834 |
| 6 | 0.0377 | 0.0219 | 0.0276 | 0.0206 |
| 7 | 0.5339 | 0.5305 | 0.5491 | 0.5438 |
| 8 | 0.0073 | 0.0191 | 0.0136 | 0.0184 |
| 9 | 0.0095 | 0.0175 | 0.0285 | 0.0104 |

Lb, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0087 | 0.0235 | 0.0137 | 0.0131 |
| 1 | 0.1146 | 0.0973 | 0.0916 | 0.0867 |
| 2 | 0.2316 | 0.2395 | 0.2318 | 0.2327 |
| 3 | 0.3491 | 0.3480 | 0.3564 | 0.3503 |
| 4 | 0.2334 | 0.2305 | 0.2383 | 0.2365 |
| 5 | 0.0735 | 0.0728 | 0.0777 | 0.0858 |
| 6 | 0.0038 | 0.0066 | 0.0072 | 0.0051 |
| 7 | 0.0335 | 0.0273 | 0.0220 | 0.0241 |
| 8 | 0.3682 | 0.3685 | 0.3650 | 0.3708 |
| 9 | 0.1683 | 0.1634 | 0.1622 | 0.1526 |

M, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.8779 | 0.8544 | 0.8737 | 0.8621 |
| 1 | 0.1222 | 0.1143 | 0.1120 | 0.1075 |
| 2 | 0.0314 | 0.0028 | 0.0217 | 0.0282 |
| 3 | 0.9684 | 0.9591 | 0.9624 | 0.9596 |
| 4 | 0.7175 | 0.7167 | 0.7418 | 0.7322 |
| 5 | 0.2537 | 0.2523 | 0.2536 | 0.2340 |
| 6 | 0.1743 | 0.1718 | 0.2805 | 0.2397 |
| 7 | 0.9407 | 0.9248 | 0.9399 | 0.9307 |
| 8 | 0.0429 | 0.0292 | 0.0009 | 0.0185 |
| 9 | 0.0314 | 0.0256 | 0.0895 | 0.1340 |

M, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.5147 | 0.5330 | 0.5417 | 0.5591 |
| 1 | 0.1202 | 0.1658 | 0.1231 | 0.1356 |
| 2 | 0.0064 | 0.2296 | 0.0307 | 0.0174 |
| 3 | 0.9701 | 0.8849 | 0.9457 | 0.9634 |
| 4 | 0.5818 | 0.5636 | 0.6224 | 0.5877 |
| 5 | 0.2398 | 0.2901 | 0.3063 | 0.3172 |
| 6 | 0.0222 | 0.0013 | 0.0191 | 0.0181 |
| 7 | 0.9165 | 0.9028 | 0.9285 | 0.9285 |
| 8 | 0.0323 | 0.0081 | 0.0303 | 0.0282 |
| 9 | 0.0262 | 0.1214 | 0.0236 | 0.0213 |

M, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.4058 | 0.3734 | 0.4299 | 0.4546 |
| 1 | 0.1039 | 0.0982 | 0.1027 | 0.1033 |
| 2 | 0.0039 | 0.0324 | 0.0162 | 0.0005 |
| 3 | 0.6251 | 0.6321 | 0.6404 | 0.6274 |
| 4 | 0.4637 | 0.4875 | 0.4629 | 0.4646 |
| 5 | 0.1730 | 0.1697 | 0.1784 | 0.1546 |
| 6 | 0.0285 | 0.0071 | 0.0260 | 0.0266 |
| 7 | 0.5204 | 0.5236 | 0.5405 | 0.5634 |
| 8 | 0.0337 | 0.0069 | 0.0350 | 0.0271 |
| 9 | 0.0211 | 0.0171 | 0.0210 | 0.0290 |

M, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0136 | 0.0123 | 0.0110 | 0.0078 |
| 1 | 0.0926 | 0.0940 | 0.1002 | 0.0875 |
| 2 | 0.2250 | 0.2262 | 0.2310 | 0.2318 |
| 3 | 0.3506 | 0.3762 | 0.3389 | 0.3715 |
| 4 | 0.2429 | 0.2784 | 0.2420 | 0.2362 |
| 5 | 0.0772 | 0.1052 | 0.0689 | 0.0801 |
| 6 | 0.0004 | 0.0040 | 0.0091 | 0.0058 |
| 7 | 0.0294 | 0.0445 | 0.0296 | 0.0411 |
| 8 | 0.3587 | 0.3925 | 0.3672 | 0.3773 |
| 9 | 0.1573 | 0.1628 | 0.1452 | 0.1494 |

F, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.5448 | 0.7137 | 0.5319 | 0.4140 |
| 1 | 0.5066 | 0.4244 | 0.4617 | 0.9311 |
| 2 | 0.0591 | 0.2052 | 0.9127 | 0.0277 |
| 3 | 0.9936 | 0.9897 | 0.5791 | 0.9962 |
| 4 | 0.4386 | 0.0673 | 0.1720 | 0.0204 |
| 5 | 0.1460 | 0.2840 | 0.0554 | 0.9292 |
| 6 | 0.1046 | 0.1895 | 0.0065 | 0.5371 |
| 7 | 0.9699 | 0.8383 | 0.6880 | 0.0104 |
| 8 | 0.6398 | 0.5065 | 0.5673 | 0.6203 |
| 9 | 0.9924 | 0.9812 | 0.5616 | 0.4890 |

F, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.5085 | 0.0000 | 0.8615 | 0.3445 |
| 1 | 0.2960 | 0.0000 | 0.1629 | 0.4711 |
| 2 | 0.1214 | 0.0000 | 0.0927 | 0.1824 |
| 3 | 0.6787 | 0.0000 | 0.4426 | 0.6949 |
| 4 | 0.2906 | 0.0000 | 0.0731 | 0.7839 |
| 5 | 0.3340 | 0.0000 | 0.4027 | 0.2396 |
| 6 | 0.5901 | 0.0000 | 0.3737 | 0.7150 |
| 7 | 0.5508 | 0.0000 | 0.1610 | 0.6066 |
| 8 | 0.1356 | 0.0000 | 0.3302 | 0.7586 |
| 9 | 0.1390 | 0.0000 | 0.5614 | 0.2051 |

F, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.3273 | 0.0000 | 0.1114 | 0.4416 |
| 1 | 0.0330 | 0.0000 | 0.4181 | 0.1118 |
| 2 | 0.5629 | 0.2670 | 0.6876 | 0.0644 |
| 3 | 0.3226 | 0.0000 | 0.4561 | 0.5189 |
| 4 | 0.0351 | 0.0000 | 0.2901 | 0.2588 |
| 5 | 0.2397 | 0.0000 | 0.2920 | 0.0787 |
| 6 | 0.5857 | 0.1865 | 0.1890 | 0.1295 |
| 7 | 0.5606 | 0.0000 | 0.0821 | 0.0767 |
| 8 | 0.3648 | 0.0000 | 0.8068 | 0.3804 |
| 9 | 0.5832 | 0.5290 | 0.4072 | 0.2715 |

F, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.2973 | 0.0083 | 0.2338 | 0.1076 |
| 1 | 0.4068 | 0.0000 | 0.3544 | 0.2594 |
| 2 | 0.1916 | 0.4659 | 0.1478 | 0.0405 |
| 3 | 0.1173 | 0.3980 | 0.2260 | 0.0571 |
| 4 | 0.0981 | 0.3169 | 0.3360 | 0.3382 |
| 5 | 0.4153 | 0.0000 | 0.3930 | 0.2684 |
| 6 | 0.1432 | 0.0000 | 0.0466 | 0.2394 |
| 7 | 0.1735 | 0.0000 | 0.1363 | 0.2508 |
| 8 | 0.2050 | 0.0776 | 0.0596 | 0.3048 |
| 9 | 0.1526 | 0.2050 | 0.1839 | 0.2300 |

## Per-seed encoder-decoder consistency: mean |centre . u|


L, d = 3, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.8852 | 0.9990 | 0.9999 | 1.0000 |
| 1 | 0.9077 | 0.9989 | 0.9995 | 1.0000 |
| 2 | 0.8934 | 0.9993 | 1.0000 | 1.0000 |
| 3 | 0.8909 | 0.9996 | 0.9998 | 0.9999 |
| 4 | 0.8790 | 0.9983 | 1.0000 | 0.9999 |
| 5 | 0.8947 | 0.9990 | 0.9988 | 1.0000 |
| 6 | 0.9149 | 0.9987 | 0.9995 | 0.9997 |
| 7 | 0.9026 | 0.9981 | 1.0000 | 1.0000 |
| 8 | 0.8941 | 0.9986 | 1.0000 | 1.0000 |
| 9 | 0.9007 | 0.9979 | 0.9992 | 0.9997 |

L, d = 5, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9029 | 0.9980 | 0.9998 | 0.9999 |
| 1 | 0.9224 | 0.9981 | 0.9998 | 0.9995 |
| 2 | 0.9066 | 0.9994 | 1.0000 | 0.9997 |
| 3 | 0.9011 | 0.9966 | 1.0000 | 0.9998 |
| 4 | 0.8881 | 0.9963 | 1.0000 | 0.9997 |
| 5 | 0.8894 | 0.9973 | 0.9998 | 0.9999 |
| 6 | 0.9153 | 0.9981 | 0.9998 | 0.9997 |
| 7 | 0.9067 | 0.9983 | 0.9997 | 0.9997 |
| 8 | 0.9099 | 0.9989 | 0.9991 | 0.9995 |
| 9 | 0.8954 | 0.9983 | 0.9991 | 0.9994 |

L, d = 9, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9095 | 0.9931 | 0.9997 | 0.9998 |
| 1 | 0.9047 | 0.9929 | 0.9991 | 0.9998 |
| 2 | 0.8993 | 0.9946 | 0.9998 | 0.9997 |
| 3 | 0.9046 | 0.9916 | 0.9994 | 0.9997 |
| 4 | 0.9237 | 0.9939 | 0.9995 | 0.9995 |
| 5 | 0.9132 | 0.9938 | 0.9999 | 0.9997 |
| 6 | 0.9272 | 0.9936 | 0.9996 | 0.9999 |
| 7 | 0.9204 | 0.9913 | 0.9998 | 0.9997 |
| 8 | 0.9004 | 0.9947 | 0.9987 | 0.9998 |
| 9 | 0.9155 | 0.9911 | 0.9990 | 0.9990 |

L, d = 17, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9594 | 0.9890 | 0.9991 | 0.9998 |
| 1 | 0.9404 | 0.9888 | 0.9997 | 0.9995 |
| 2 | 0.9285 | 0.9839 | 0.9993 | 0.9997 |
| 3 | 0.9416 | 0.9867 | 0.9993 | 0.9996 |
| 4 | 0.9505 | 0.9900 | 0.9994 | 0.9996 |
| 5 | 0.9515 | 0.9899 | 0.9995 | 0.9991 |
| 6 | 0.9541 | 0.9889 | 0.9992 | 0.9984 |
| 7 | 0.9363 | 0.9908 | 0.9995 | 0.9990 |
| 8 | 0.9450 | 0.9858 | 0.9999 | 0.9994 |
| 9 | 0.9507 | 0.9892 | 0.9996 | 0.9996 |

L, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.8705 | 0.9985 | 1.0000 | 1.0000 |
| 1 | 0.8588 | 0.9961 | 0.9997 | 1.0000 |
| 2 | 0.9082 | 0.9985 | 1.0000 | 1.0000 |
| 3 | 0.9000 | 0.9997 | 0.9999 | 0.9999 |
| 4 | 0.8945 | 0.9993 | 0.9999 | 1.0000 |
| 5 | 0.9146 | 0.9991 | 0.9995 | 0.9980 |
| 6 | 0.8819 | 0.9963 | 0.9987 | 0.9999 |
| 7 | 0.9116 | 0.9981 | 1.0000 | 1.0000 |
| 8 | 0.8997 | 0.9971 | 1.0000 | 1.0000 |
| 9 | 0.9038 | 0.9993 | 0.9999 | 0.9974 |

L, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9046 | 0.9965 | 0.9999 | 0.9999 |
| 1 | 0.8990 | 0.9972 | 0.9982 | 0.9998 |
| 2 | 0.9015 | 0.9955 | 0.9998 | 0.9996 |
| 3 | 0.9195 | 0.9981 | 0.9998 | 0.9999 |
| 4 | 0.8851 | 0.9989 | 0.9999 | 0.9996 |
| 5 | 0.9271 | 0.9984 | 0.9992 | 0.9996 |
| 6 | 0.9160 | 0.9993 | 0.9988 | 0.9994 |
| 7 | 0.8827 | 0.9979 | 0.9995 | 0.9995 |
| 8 | 0.9275 | 0.9975 | 0.9992 | 1.0000 |
| 9 | 0.9028 | 0.9976 | 0.9996 | 0.9998 |

L, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9144 | 0.9923 | 0.9998 | 0.9993 |
| 1 | 0.9309 | 0.9934 | 0.9995 | 0.9999 |
| 2 | 0.9157 | 0.9935 | 0.9997 | 0.9996 |
| 3 | 0.9305 | 0.9951 | 0.9984 | 0.9997 |
| 4 | 0.9359 | 0.9962 | 0.9987 | 0.9996 |
| 5 | 0.9150 | 0.9973 | 0.9993 | 0.9996 |
| 6 | 0.9270 | 0.9957 | 0.9990 | 0.9986 |
| 7 | 0.9289 | 0.9925 | 0.9982 | 0.9997 |
| 8 | 0.9217 | 0.9868 | 0.9991 | 0.9997 |
| 9 | 0.9449 | 0.9909 | 0.9993 | 0.9996 |

L, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9481 | 0.9856 | 0.9968 | 0.9971 |
| 1 | 0.9546 | 0.9885 | 0.9990 | 0.9995 |
| 2 | 0.9480 | 0.9892 | 0.9992 | 0.9974 |
| 3 | 0.9593 | 0.9899 | 0.9987 | 0.9994 |
| 4 | 0.9578 | 0.9903 | 0.9996 | 0.9994 |
| 5 | 0.9496 | 0.9848 | 0.9994 | 0.9984 |
| 6 | 0.9579 | 0.9866 | 0.9991 | 0.9996 |
| 7 | 0.9577 | 0.9872 | 0.9991 | 0.9984 |
| 8 | 0.9471 | 0.9914 | 0.9993 | 0.9986 |
| 9 | 0.9443 | 0.9889 | 0.9988 | 0.9990 |

L, d = 3, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9077 | 0.9987 | 0.9976 | 1.0000 |
| 1 | 0.9133 | 0.9989 | 0.9995 | 1.0000 |
| 2 | 0.9193 | 0.9985 | 0.9992 | 1.0000 |
| 3 | 0.8995 | 0.9980 | 0.9999 | 0.9997 |
| 4 | 0.8753 | 0.9974 | 0.9994 | 0.9993 |
| 5 | 0.9198 | 0.9992 | 0.9996 | 0.9991 |
| 6 | 0.8870 | 0.9988 | 0.9997 | 0.9988 |
| 7 | 0.9001 | 0.9991 | 0.9999 | 0.9998 |
| 8 | 0.8907 | 0.9993 | 0.9993 | 0.9996 |
| 9 | 0.9213 | 0.9981 | 0.9997 | 0.9993 |

L, d = 5, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9063 | 0.9969 | 0.9990 | 0.9998 |
| 1 | 0.9264 | 0.9937 | 0.9988 | 0.9990 |
| 2 | 0.9087 | 0.9967 | 0.9989 | 0.9996 |
| 3 | 0.9035 | 0.9973 | 0.9994 | 0.9999 |
| 4 | 0.8988 | 0.9968 | 0.9995 | 0.9988 |
| 5 | 0.9173 | 0.9976 | 0.9997 | 0.9996 |
| 6 | 0.9269 | 0.9960 | 0.9999 | 0.9996 |
| 7 | 0.9369 | 0.9974 | 0.9998 | 0.9999 |
| 8 | 0.9200 | 0.9958 | 0.9986 | 1.0000 |
| 9 | 0.9249 | 0.9964 | 0.9992 | 0.9997 |

L, d = 9, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9513 | 0.9920 | 0.9995 | 0.9993 |
| 1 | 0.9545 | 0.9922 | 0.9986 | 0.9989 |
| 2 | 0.9468 | 0.9951 | 0.9991 | 0.9988 |
| 3 | 0.9460 | 0.9911 | 0.9990 | 0.9995 |
| 4 | 0.9463 | 0.9906 | 0.9987 | 0.9993 |
| 5 | 0.9441 | 0.9948 | 0.9992 | 0.9990 |
| 6 | 0.9509 | 0.9932 | 0.9983 | 0.9994 |
| 7 | 0.9550 | 0.9963 | 0.9990 | 0.9987 |
| 8 | 0.9590 | 0.9949 | 0.9997 | 0.9993 |
| 9 | 0.9319 | 0.9928 | 0.9982 | 0.9983 |

L, d = 17, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9698 | 0.9920 | 0.9984 | 0.9984 |
| 1 | 0.9750 | 0.9915 | 0.9982 | 0.9952 |
| 2 | 0.9729 | 0.9930 | 0.9972 | 0.9990 |
| 3 | 0.9727 | 0.9921 | 0.9971 | 0.9980 |
| 4 | 0.9687 | 0.9922 | 0.9966 | 0.9975 |
| 5 | 0.9664 | 0.9911 | 0.9984 | 0.9971 |
| 6 | 0.9706 | 0.9933 | 0.9971 | 0.9987 |
| 7 | 0.9745 | 0.9902 | 0.9977 | 0.9979 |
| 8 | 0.9648 | 0.9925 | 0.9976 | 0.9958 |
| 9 | 0.9726 | 0.9906 | 0.9963 | 0.9979 |

Lb, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9037 | 0.9990 | 0.9998 | 1.0000 |
| 1 | 0.9094 | 0.9982 | 1.0000 | 1.0000 |
| 2 | 0.9080 | 0.9979 | 1.0000 | 1.0000 |
| 3 | 0.9022 | 0.9991 | 0.9999 | 0.9998 |
| 4 | 0.8865 | 0.9994 | 1.0000 | 1.0000 |
| 5 | 0.9177 | 0.9967 | 1.0000 | 1.0000 |
| 6 | 0.8808 | 0.9975 | 0.9999 | 1.0000 |
| 7 | 0.9127 | 0.9987 | 0.9989 | 1.0000 |
| 8 | 0.8864 | 0.9980 | 0.9998 | 1.0000 |
| 9 | 0.9029 | 0.9995 | 1.0000 | 0.9999 |

Lb, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9119 | 0.9979 | 0.9993 | 0.9993 |
| 1 | 0.8999 | 0.9992 | 0.9997 | 0.9997 |
| 2 | 0.9088 | 0.9943 | 0.9999 | 0.9999 |
| 3 | 0.9172 | 0.9968 | 0.9993 | 0.9999 |
| 4 | 0.9199 | 0.9980 | 0.9997 | 0.9988 |
| 5 | 0.8912 | 0.9989 | 0.9999 | 0.9998 |
| 6 | 0.9189 | 0.9951 | 0.9999 | 0.9999 |
| 7 | 0.9121 | 0.9976 | 0.9997 | 1.0000 |
| 8 | 0.9098 | 0.9989 | 0.9994 | 1.0000 |
| 9 | 0.9129 | 0.9980 | 0.9991 | 0.9989 |

Lb, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9479 | 0.9909 | 0.9995 | 0.9992 |
| 1 | 0.9427 | 0.9880 | 0.9989 | 0.9996 |
| 2 | 0.9269 | 0.9902 | 0.9999 | 0.9998 |
| 3 | 0.9208 | 0.9925 | 0.9993 | 0.9996 |
| 4 | 0.9096 | 0.9975 | 0.9989 | 0.9994 |
| 5 | 0.9435 | 0.9905 | 0.9996 | 0.9995 |
| 6 | 0.9120 | 0.9894 | 0.9998 | 0.9998 |
| 7 | 0.9047 | 0.9933 | 0.9995 | 0.9999 |
| 8 | 0.9141 | 0.9949 | 0.9999 | 0.9997 |
| 9 | 0.9146 | 0.9955 | 0.9989 | 0.9995 |

Lb, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9527 | 0.9852 | 0.9991 | 0.9990 |
| 1 | 0.9583 | 0.9873 | 0.9986 | 0.9994 |
| 2 | 0.9518 | 0.9927 | 0.9983 | 0.9989 |
| 3 | 0.9448 | 0.9929 | 0.9982 | 0.9991 |
| 4 | 0.9622 | 0.9950 | 0.9988 | 0.9994 |
| 5 | 0.9711 | 0.9862 | 0.9993 | 0.9987 |
| 6 | 0.9565 | 0.9869 | 0.9993 | 0.9979 |
| 7 | 0.9580 | 0.9829 | 0.9995 | 0.9993 |
| 8 | 0.9514 | 0.9889 | 0.9990 | 0.9984 |
| 9 | 0.9603 | 0.9857 | 0.9993 | 0.9995 |

M, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9568 | 0.9965 | 1.0000 | 1.0000 |
| 1 | 0.9479 | 0.9995 | 1.0000 | 1.0000 |
| 2 | 0.9489 | 0.9996 | 1.0000 | 0.9973 |
| 3 | 0.9501 | 0.9978 | 1.0000 | 0.9993 |
| 4 | 0.9589 | 0.9986 | 0.9839 | 1.0000 |
| 5 | 0.9638 | 0.9991 | 1.0000 | 1.0000 |
| 6 | 0.9644 | 0.9973 | 0.9982 | 0.9978 |
| 7 | 0.9470 | 0.9980 | 0.9999 | 0.9990 |
| 8 | 0.9578 | 0.9977 | 0.9959 | 1.0000 |
| 9 | 0.9549 | 0.9976 | 1.0000 | 0.9999 |

M, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9763 | 0.7179 | 0.9980 | 0.9984 |
| 1 | 0.9838 | 0.8114 | 0.9644 | 0.9674 |
| 2 | 0.9841 | 0.3668 | 0.9978 | 0.9992 |
| 3 | 0.9714 | 0.9670 | 0.9895 | 1.0000 |
| 4 | 0.9791 | 0.6414 | 0.9995 | 1.0000 |
| 5 | 0.9789 | 0.9205 | 0.9968 | 0.9997 |
| 6 | 0.9739 | 0.8925 | 0.9986 | 1.0000 |
| 7 | 0.9724 | 0.7379 | 0.9980 | 1.0000 |
| 8 | 0.9766 | 0.1544 | 0.9977 | 1.0000 |
| 9 | 0.9754 | 0.2623 | 0.9988 | 1.0000 |

M, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9845 | 0.8742 | 0.9998 | 0.9068 |
| 1 | 0.9865 | 0.3701 | 0.8161 | 0.9998 |
| 2 | 0.9823 | 0.7330 | 0.8656 | 0.9999 |
| 3 | 0.9820 | 0.6450 | 0.8564 | 0.9680 |
| 4 | 0.9842 | 0.5486 | 0.9998 | 0.9998 |
| 5 | 0.9774 | 0.4462 | 0.8059 | 0.9999 |
| 6 | 0.9864 | 0.7060 | 0.9686 | 0.9999 |
| 7 | 0.9871 | 0.6314 | 0.9473 | 0.9940 |
| 8 | 0.9877 | 0.8827 | 0.8837 | 0.7057 |
| 9 | 0.9857 | 0.5527 | 0.8072 | 0.9897 |

M, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9862 | 0.2030 | 0.9395 | 0.9994 |
| 1 | 0.9847 | 0.1165 | 0.7132 | 0.9872 |
| 2 | 0.9840 | 0.5217 | 0.7430 | 0.9997 |
| 3 | 0.9812 | 0.3569 | 0.5675 | 0.9149 |
| 4 | 0.9832 | 0.5779 | 0.7822 | 0.9996 |
| 5 | 0.9823 | 0.1095 | 0.8856 | 0.7238 |
| 6 | 0.9891 | 0.4989 | 0.5985 | 0.9453 |
| 7 | 0.9821 | 0.4364 | 0.9313 | 0.8076 |
| 8 | 0.9836 | 0.3055 | 0.9128 | 0.9691 |
| 9 | 0.9839 | 0.3495 | 0.8760 | 0.9994 |

F, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9301 | 0.9987 | 0.9976 | 0.9999 |
| 1 | 0.9375 | 0.9983 | 0.9994 | 0.9941 |
| 2 | 0.9386 | 0.9986 | 0.9987 | 0.9999 |
| 3 | 0.9376 | 0.9993 | 0.9982 | 0.9963 |
| 4 | 0.9404 | 0.9952 | 0.9985 | 1.0000 |
| 5 | 0.9352 | 0.9970 | 0.9984 | 0.9965 |
| 6 | 0.9539 | 0.9959 | 0.9947 | 0.9995 |
| 7 | 0.9263 | 0.9947 | 0.9981 | 0.9999 |
| 8 | 0.9366 | 0.9977 | 0.9960 | 0.9977 |
| 9 | 0.9392 | 0.9988 | 0.9886 | 0.9990 |

F, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9694 | 0.4371 | 0.9945 | 0.9999 |
| 1 | 0.9648 | 0.0962 | 0.9997 | 0.9977 |
| 2 | 0.9635 | 0.4955 | 0.9976 | 0.9983 |
| 3 | 0.9707 | 0.2077 | 0.9959 | 0.9261 |
| 4 | 0.9659 | 0.1140 | 0.9989 | 0.9991 |
| 5 | 0.9630 | 0.4717 | 0.9984 | 0.2436 |
| 6 | 0.9723 | 0.7651 | 0.9978 | 0.9999 |
| 7 | 0.9632 | 0.7776 | 0.9971 | 0.9979 |
| 8 | 0.9644 | 0.3038 | 0.9991 | 1.0000 |
| 9 | 0.9649 | 0.4851 | 0.8063 | 0.9995 |

F, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9783 | 0.3576 | 0.8712 | 0.4498 |
| 1 | 0.9769 | 0.1840 | 0.9447 | 0.9340 |
| 2 | 0.9810 | 0.4383 | 0.4411 | 0.9999 |
| 3 | 0.9785 | 0.4658 | 0.9076 | 0.8110 |
| 4 | 0.9786 | 0.0684 | 0.9736 | 0.4947 |
| 5 | 0.9775 | 0.1677 | 0.9756 | 0.9535 |
| 6 | 0.9730 | 0.1367 | 0.8198 | 0.9128 |
| 7 | 0.9754 | 0.4131 | 0.2373 | 0.9149 |
| 8 | 0.9818 | 0.5569 | 0.8270 | 0.9998 |
| 9 | 0.9817 | 0.3784 | 0.9692 | 0.9451 |

F, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.9789 | 0.0965 | 0.5796 | 0.8414 |
| 1 | 0.9794 | 0.1021 | 0.7707 | 0.1884 |
| 2 | 0.9809 | 0.1494 | 0.4192 | 0.9130 |
| 3 | 0.9806 | 0.5326 | 0.6733 | 0.3812 |
| 4 | 0.9779 | 0.4507 | 0.9385 | 0.9215 |
| 5 | 0.9754 | 0.1712 | 0.8242 | 0.8036 |
| 6 | 0.9837 | 0.1626 | 0.7579 | 0.8063 |
| 7 | 0.9815 | 0.2906 | 0.6464 | 0.5772 |
| 8 | 0.9818 | 0.1955 | 0.6422 | 0.8330 |
| 9 | 0.9737 | 0.2905 | 0.7155 | 0.8549 |

## Per-seed IWAE (K=64) gap


L, d = 3, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1603 | 0.1380 | 0.0168 | 0.0097 |
| 1 | 0.1597 | 0.1392 | 0.0210 | 0.0093 |
| 2 | 0.1598 | 0.1376 | 0.0077 | 0.0055 |
| 3 | 0.1607 | 0.1374 | 0.0044 | 0.0178 |
| 4 | 0.1613 | 0.1384 | 0.0083 | 0.0335 |
| 5 | 0.1609 | 0.1379 | 0.0273 | 0.0101 |
| 6 | 0.1611 | 0.1382 | 0.0077 | 0.0115 |
| 7 | 0.1597 | 0.1379 | 0.0067 | 0.0055 |
| 8 | 0.1616 | 0.1374 | 0.0029 | 0.0093 |
| 9 | 0.1606 | 0.1392 | 0.0145 | 0.0454 |

L, d = 5, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1406 | 0.1074 | 0.0016 | 0.0095 |
| 1 | 0.1385 | 0.1066 | 0.0067 | 0.0261 |
| 2 | 0.1410 | 0.1071 | 0.0027 | 0.0026 |
| 3 | 0.1391 | 0.1067 | 0.0077 | 0.0101 |
| 4 | 0.1416 | 0.1058 | 0.0060 | 0.0057 |
| 5 | 0.1382 | 0.1067 | 0.0067 | 0.0136 |
| 6 | 0.1406 | 0.1072 | 0.0049 | 0.0156 |
| 7 | 0.1391 | 0.1068 | 0.0072 | 0.0212 |
| 8 | 0.1397 | 0.1064 | 0.0102 | 0.0042 |
| 9 | 0.1403 | 0.1064 | 0.0059 | 0.0103 |

L, d = 9, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1102 | 0.0902 | 0.0030 | 0.0059 |
| 1 | 0.1131 | 0.0901 | 0.0099 | 0.0017 |
| 2 | 0.1107 | 0.0910 | 0.0014 | 0.0016 |
| 3 | 0.1112 | 0.0902 | 0.0018 | 0.0074 |
| 4 | 0.1119 | 0.0911 | 0.0007 | 0.0017 |
| 5 | 0.1115 | 0.0905 | 0.0039 | 0.0015 |
| 6 | 0.1120 | 0.0905 | 0.0034 | 0.0014 |
| 7 | 0.1094 | 0.0902 | 0.0063 | 0.0028 |
| 8 | 0.1122 | 0.0903 | 0.0063 | 0.0033 |
| 9 | 0.1119 | 0.0904 | 0.0032 | 0.0019 |

L, d = 17, sigma = 0.1

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1054 | 0.0917 | 0.0054 | 0.0014 |
| 1 | 0.1059 | 0.0907 | 0.0060 | 0.0029 |
| 2 | 0.1071 | 0.0916 | 0.0019 | 0.0088 |
| 3 | 0.1067 | 0.0918 | 0.0037 | 0.0018 |
| 4 | 0.1053 | 0.0907 | 0.0027 | 0.0027 |
| 5 | 0.1062 | 0.0913 | 0.0055 | 0.0022 |
| 6 | 0.1073 | 0.0922 | 0.0026 | 0.0073 |
| 7 | 0.1057 | 0.0909 | 0.0047 | 0.0030 |
| 8 | 0.1067 | 0.0921 | 0.0015 | 0.0051 |
| 9 | 0.1070 | 0.0906 | 0.0022 | 0.0026 |

L, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0959 | 0.0759 | 0.0066 | 0.0036 |
| 1 | 0.0954 | 0.0793 | 0.0141 | 0.0069 |
| 2 | 0.0955 | 0.0763 | 0.0043 | 0.0068 |
| 3 | 0.0958 | 0.0767 | 0.0051 | 0.0051 |
| 4 | 0.0976 | 0.0758 | 0.0060 | 0.0066 |
| 5 | 0.0971 | 0.0791 | 0.0083 | 0.0141 |
| 6 | 0.0966 | 0.0759 | 0.0097 | 0.0068 |
| 7 | 0.0950 | 0.0765 | 0.0081 | 0.0093 |
| 8 | 0.0956 | 0.0758 | 0.0041 | 0.0124 |
| 9 | 0.0963 | 0.0755 | 0.0048 | 0.0080 |

L, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1005 | 0.0697 | 0.0023 | 0.0038 |
| 1 | 0.0987 | 0.0672 | 0.0067 | 0.0038 |
| 2 | 0.1001 | 0.0702 | 0.0036 | 0.0020 |
| 3 | 0.1000 | 0.0693 | 0.0026 | 0.0053 |
| 4 | 0.1012 | 0.0680 | 0.0032 | 0.0026 |
| 5 | 0.0995 | 0.0687 | 0.0025 | 0.0036 |
| 6 | 0.0986 | 0.0688 | 0.0035 | 0.0050 |
| 7 | 0.0996 | 0.0685 | 0.0022 | 0.0037 |
| 8 | 0.0999 | 0.0667 | 0.0039 | 0.0014 |
| 9 | 0.0978 | 0.0667 | 0.0012 | 0.0014 |

L, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0649 | 0.0471 | 0.0011 | 0.0020 |
| 1 | 0.0658 | 0.0477 | 0.0012 | 0.0003 |
| 2 | 0.0661 | 0.0483 | -0.0009 | 0.0005 |
| 3 | 0.0649 | 0.0473 | 0.0004 | -0.0002 |
| 4 | 0.0666 | 0.0481 | 0.0008 | 0.0029 |
| 5 | 0.0653 | 0.0476 | -0.0000 | -0.0005 |
| 6 | 0.0641 | 0.0475 | 0.0001 | 0.0001 |
| 7 | 0.0635 | 0.0462 | 0.0024 | 0.0008 |
| 8 | 0.0657 | 0.0465 | 0.0007 | 0.0007 |
| 9 | 0.0655 | 0.0475 | 0.0009 | 0.0027 |

L, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0430 | 0.0329 | 0.0009 | 0.0019 |
| 1 | 0.0430 | 0.0321 | 0.0001 | 0.0006 |
| 2 | 0.0438 | 0.0337 | 0.0012 | 0.0004 |
| 3 | 0.0441 | 0.0333 | 0.0001 | 0.0004 |
| 4 | 0.0436 | 0.0325 | 0.0005 | 0.0005 |
| 5 | 0.0442 | 0.0318 | 0.0006 | 0.0006 |
| 6 | 0.0438 | 0.0340 | 0.0005 | 0.0003 |
| 7 | 0.0448 | 0.0331 | 0.0008 | 0.0001 |
| 8 | 0.0438 | 0.0335 | 0.0017 | 0.0008 |
| 9 | 0.0446 | 0.0333 | 0.0004 | 0.0000 |

L, d = 3, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0262 | 0.0171 | 0.0022 | 0.0022 |
| 1 | 0.0265 | 0.0172 | 0.0030 | 0.0035 |
| 2 | 0.0272 | 0.0174 | 0.0041 | 0.0021 |
| 3 | 0.0278 | 0.0178 | 0.0024 | 0.0021 |
| 4 | 0.0284 | 0.0168 | 0.0025 | 0.0028 |
| 5 | 0.0270 | 0.0171 | 0.0024 | 0.0021 |
| 6 | 0.0268 | 0.0176 | 0.0026 | 0.0064 |
| 7 | 0.0269 | 0.0168 | 0.0040 | 0.0161 |
| 8 | 0.0255 | 0.0165 | 0.0020 | 0.0027 |
| 9 | 0.0265 | 0.0168 | 0.0030 | 0.0080 |

L, d = 5, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0318 | 0.0124 | 0.0023 | 0.0018 |
| 1 | 0.0315 | 0.0130 | 0.0004 | 0.0011 |
| 2 | 0.0320 | 0.0130 | 0.0022 | 0.0008 |
| 3 | 0.0319 | 0.0134 | 0.0005 | 0.0008 |
| 4 | 0.0327 | 0.0134 | 0.0009 | 0.0008 |
| 5 | 0.0303 | 0.0118 | 0.0003 | 0.0011 |
| 6 | 0.0302 | 0.0128 | 0.0008 | 0.0016 |
| 7 | 0.0297 | 0.0114 | 0.0007 | 0.0008 |
| 8 | 0.0304 | 0.0132 | 0.0006 | 0.0005 |
| 9 | 0.0296 | 0.0123 | -0.0000 | 0.0005 |

L, d = 9, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0095 | 0.0036 | 0.0001 | 0.0004 |
| 1 | 0.0103 | 0.0041 | 0.0000 | 0.0003 |
| 2 | 0.0101 | 0.0044 | -0.0002 | 0.0002 |
| 3 | 0.0097 | 0.0036 | -0.0001 | 0.0002 |
| 4 | 0.0119 | 0.0045 | 0.0001 | 0.0003 |
| 5 | 0.0103 | 0.0044 | 0.0001 | 0.0006 |
| 6 | 0.0094 | 0.0044 | -0.0001 | -0.0005 |
| 7 | 0.0096 | 0.0035 | -0.0000 | -0.0001 |
| 8 | 0.0110 | 0.0042 | -0.0002 | -0.0001 |
| 9 | 0.0106 | 0.0040 | -0.0002 | 0.0001 |

L, d = 17, sigma = 0.4

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0029 | 0.0014 | -0.0002 | -0.0001 |
| 1 | 0.0013 | 0.0009 | 0.0001 | 0.0001 |
| 2 | 0.0021 | 0.0012 | 0.0000 | 0.0000 |
| 3 | 0.0014 | 0.0012 | 0.0002 | 0.0000 |
| 4 | 0.0020 | 0.0011 | 0.0002 | 0.0001 |
| 5 | 0.0017 | 0.0011 | 0.0001 | 0.0001 |
| 6 | 0.0018 | 0.0013 | 0.0001 | 0.0002 |
| 7 | 0.0019 | 0.0015 | 0.0000 | 0.0003 |
| 8 | 0.0019 | 0.0016 | -0.0001 | 0.0002 |
| 9 | 0.0017 | 0.0013 | 0.0001 | 0.0002 |

Lb, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0950 | 0.0767 | 0.0058 | 0.0081 |
| 1 | 0.0966 | 0.0771 | 0.0038 | 0.0048 |
| 2 | 0.0956 | 0.0773 | 0.0033 | 0.0089 |
| 3 | 0.0957 | 0.0774 | 0.0076 | 0.0152 |
| 4 | 0.0977 | 0.0773 | 0.0079 | 0.0036 |
| 5 | 0.0968 | 0.0771 | 0.0124 | 0.0085 |
| 6 | 0.0968 | 0.0754 | 0.0071 | 0.0026 |
| 7 | 0.0951 | 0.0768 | 0.0160 | 0.0107 |
| 8 | 0.0966 | 0.0754 | 0.0051 | 0.0095 |
| 9 | 0.0958 | 0.0771 | 0.0062 | 0.0087 |

Lb, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0969 | 0.0686 | 0.0078 | 0.0048 |
| 1 | 0.1008 | 0.0680 | 0.0034 | 0.0073 |
| 2 | 0.1004 | 0.0685 | 0.0030 | 0.0005 |
| 3 | 0.0989 | 0.0685 | 0.0041 | 0.0045 |
| 4 | 0.1005 | 0.0674 | 0.0032 | 0.0049 |
| 5 | 0.1004 | 0.0683 | 0.0034 | 0.0027 |
| 6 | 0.0988 | 0.0682 | 0.0023 | 0.0030 |
| 7 | 0.0995 | 0.0690 | 0.0048 | 0.0026 |
| 8 | 0.0996 | 0.0689 | 0.0026 | 0.0027 |
| 9 | 0.0986 | 0.0682 | 0.0069 | 0.0028 |

Lb, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0653 | 0.0479 | 0.0012 | 0.0015 |
| 1 | 0.0651 | 0.0472 | 0.0041 | -0.0002 |
| 2 | 0.0667 | 0.0487 | 0.0002 | 0.0006 |
| 3 | 0.0644 | 0.0480 | 0.0001 | 0.0009 |
| 4 | 0.0674 | 0.0489 | 0.0003 | 0.0035 |
| 5 | 0.0656 | 0.0489 | 0.0012 | 0.0004 |
| 6 | 0.0665 | 0.0468 | -0.0002 | 0.0017 |
| 7 | 0.0645 | 0.0462 | -0.0003 | 0.0013 |
| 8 | 0.0655 | 0.0472 | 0.0003 | -0.0002 |
| 9 | 0.0645 | 0.0469 | 0.0020 | 0.0006 |

Lb, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0437 | 0.0341 | 0.0010 | 0.0017 |
| 1 | 0.0421 | 0.0318 | 0.0021 | -0.0003 |
| 2 | 0.0441 | 0.0344 | 0.0011 | 0.0028 |
| 3 | 0.0457 | 0.0342 | 0.0023 | 0.0012 |
| 4 | 0.0417 | 0.0324 | 0.0003 | -0.0004 |
| 5 | 0.0443 | 0.0319 | 0.0002 | 0.0024 |
| 6 | 0.0434 | 0.0338 | -0.0000 | -0.0004 |
| 7 | 0.0440 | 0.0326 | 0.0000 | -0.0003 |
| 8 | 0.0446 | 0.0333 | 0.0002 | -0.0004 |
| 9 | 0.0429 | 0.0328 | -0.0007 | -0.0008 |

M, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0320 | 0.0256 | 0.0028 | 0.0037 |
| 1 | 0.0313 | 0.0255 | 0.0007 | 0.0049 |
| 2 | 0.0313 | 0.0253 | 0.0016 | 0.0305 |
| 3 | 0.0321 | 0.0263 | 0.0053 | 0.0124 |
| 4 | 0.0317 | 0.0255 | 0.0358 | 0.0297 |
| 5 | 0.0320 | 0.0253 | 0.0015 | 0.0031 |
| 6 | 0.0308 | 0.0249 | 0.0358 | 0.0324 |
| 7 | 0.0318 | 0.0266 | 0.0043 | 0.0355 |
| 8 | 0.0313 | 0.0254 | 0.0303 | 0.0017 |
| 9 | 0.0325 | 0.0282 | 0.0012 | 0.0034 |

M, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0151 | 0.0154 | 0.0123 | 0.0127 |
| 1 | 0.0152 | 0.0154 | 0.0155 | 0.0155 |
| 2 | 0.0152 | 0.0155 | 0.0121 | 0.0133 |
| 3 | 0.0153 | 0.0154 | 0.0155 | 0.0033 |
| 4 | 0.0155 | 0.0158 | 0.0146 | 0.0028 |
| 5 | 0.0153 | 0.0154 | 0.0131 | 0.0104 |
| 6 | 0.0153 | 0.0155 | 0.0127 | 0.0031 |
| 7 | 0.0152 | 0.0157 | 0.0137 | 0.0042 |
| 8 | 0.0151 | 0.0154 | 0.0131 | 0.0019 |
| 9 | 0.0155 | 0.0155 | 0.0153 | 0.0023 |

M, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0033 | 0.0040 | 0.0028 | 0.0041 |
| 1 | 0.0032 | 0.0039 | 0.0039 | 0.0031 |
| 2 | 0.0032 | 0.0038 | 0.0041 | 0.0038 |
| 3 | 0.0052 | 0.0048 | 0.0051 | 0.0049 |
| 4 | 0.0038 | 0.0039 | 0.0023 | 0.0037 |
| 5 | 0.0038 | 0.0038 | 0.0038 | 0.0022 |
| 6 | 0.0036 | 0.0042 | 0.0040 | 0.0034 |
| 7 | 0.0058 | 0.0044 | 0.0061 | 0.0053 |
| 8 | 0.0042 | 0.0039 | 0.0040 | 0.0042 |
| 9 | 0.0041 | 0.0044 | 0.0045 | 0.0048 |

M, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0039 | 0.0043 | 0.0035 | 0.0038 |
| 1 | 0.0022 | 0.0014 | 0.0031 | 0.0026 |
| 2 | 0.0016 | 0.0015 | 0.0018 | 0.0012 |
| 3 | 0.0019 | 0.0031 | 0.0012 | 0.0016 |
| 4 | 0.0002 | -0.0001 | -0.0005 | -0.0007 |
| 5 | 0.0013 | 0.0011 | 0.0019 | 0.0020 |
| 6 | 0.0001 | 0.0002 | -0.0001 | 0.0001 |
| 7 | 0.0008 | 0.0004 | 0.0012 | 0.0011 |
| 8 | 0.0006 | 0.0005 | 0.0011 | 0.0009 |
| 9 | 0.0011 | 0.0003 | 0.0013 | 0.0023 |

F, d = 3, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0318 | 0.0252 | 0.0288 | 0.0004 |
| 1 | 0.0316 | 0.0263 | 0.0307 | 0.0322 |
| 2 | 0.0314 | 0.0255 | 0.0279 | 0.0004 |
| 3 | 0.0321 | 0.0259 | 0.0327 | 0.0292 |
| 4 | 0.0315 | 0.0254 | 0.0293 | -0.0004 |
| 5 | 0.0315 | 0.0269 | 0.0327 | 0.0306 |
| 6 | 0.0311 | 0.0263 | 0.0334 | 0.0279 |
| 7 | 0.0309 | 0.0269 | 0.0292 | -0.0018 |
| 8 | 0.0326 | 0.0251 | 0.0280 | 0.0289 |
| 9 | 0.0320 | 0.0286 | 0.0359 | 0.0357 |

F, d = 5, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0151 | 0.0154 | 0.0132 | 0.0032 |
| 1 | 0.0155 | 0.0154 | 0.0118 | 0.0147 |
| 2 | 0.0152 | 0.0155 | 0.0128 | 0.0118 |
| 3 | 0.0153 | 0.0154 | 0.0125 | 0.0153 |
| 4 | 0.0157 | 0.0155 | 0.0126 | 0.0123 |
| 5 | 0.0153 | 0.0154 | 0.0122 | 0.0155 |
| 6 | 0.0156 | 0.0155 | 0.0121 | 0.0040 |
| 7 | 0.0152 | 0.0155 | 0.0129 | 0.0130 |
| 8 | 0.0152 | 0.0154 | 0.0139 | 0.0008 |
| 9 | 0.0168 | 0.0154 | 0.0155 | 0.0111 |

F, d = 9, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0030 | 0.0040 | 0.0038 | 0.0040 |
| 1 | 0.0034 | 0.0040 | 0.0040 | 0.0039 |
| 2 | 0.0031 | 0.0040 | 0.0040 | 0.0032 |
| 3 | 0.0046 | 0.0041 | 0.0040 | 0.0041 |
| 4 | 0.0031 | 0.0038 | 0.0039 | 0.0038 |
| 5 | 0.0039 | 0.0038 | 0.0039 | 0.0038 |
| 6 | 0.0042 | 0.0040 | 0.0040 | 0.0040 |
| 7 | 0.0051 | 0.0042 | 0.0042 | 0.0042 |
| 8 | 0.0033 | 0.0039 | 0.0039 | 0.0031 |
| 9 | 0.0043 | 0.0039 | 0.0039 | 0.0038 |

F, d = 17, sigma = 0.2

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.0053 | 0.0022 | 0.0022 | 0.0021 |
| 1 | 0.0018 | 0.0006 | 0.0007 | 0.0007 |
| 2 | 0.0014 | 0.0015 | 0.0015 | 0.0016 |
| 3 | 0.0030 | 0.0025 | 0.0028 | 0.0025 |
| 4 | 0.0006 | 0.0006 | 0.0007 | 0.0006 |
| 5 | 0.0012 | 0.0008 | 0.0008 | 0.0008 |
| 6 | 0.0001 | 0.0005 | 0.0005 | 0.0005 |
| 7 | 0.0003 | 0.0003 | 0.0004 | 0.0004 |
| 8 | 0.0001 | 0.0004 | 0.0003 | 0.0004 |
| 9 | 0.0004 | 0.0001 | 0.0001 | 0.0001 |

## Exact decomposition: gap = model misfit + KL(q || model posterior)

| decoder | d | sigma | arm | gap | model misfit (log p_true - log p_model) | KL(q || p_model(.|y)) | I(z;y) |
|---|---|---|---|---|---|---|---|
| L | 3 | 0.1 | vmf | 0.2793 | 0.1601 | 0.1192 | 1.673 |
| L | 3 | 0.1 | power_spherical | 0.1990 | 0.1375 | 0.0615 | 1.673 |
| L | 3 | 0.1 | tnbbeta | 0.0705 | 0.0037 | 0.0668 | 1.673 |
| L | 3 | 0.1 | tnbbeta_init0 | 0.0704 | 0.0087 | 0.0617 | 1.673 |
| L | 5 | 0.1 | vmf | 0.2375 | 0.1386 | 0.0989 | 1.495 |
| L | 5 | 0.1 | power_spherical | 0.1695 | 0.1059 | 0.0635 | 1.495 |
| L | 5 | 0.1 | tnbbeta | 0.0325 | 0.0043 | 0.0282 | 1.495 |
| L | 5 | 0.1 | tnbbeta_init0 | 0.0460 | 0.0098 | 0.0363 | 1.495 |
| L | 9 | 0.1 | vmf | 0.2043 | 0.1107 | 0.0935 | 1.251 |
| L | 9 | 0.1 | power_spherical | 0.1589 | 0.0891 | 0.0698 | 1.251 |
| L | 9 | 0.1 | tnbbeta | 0.0190 | 0.0041 | 0.0149 | 1.251 |
| L | 9 | 0.1 | tnbbeta_init0 | 0.0149 | 0.0025 | 0.0123 | 1.251 |
| L | 17 | 0.1 | vmf | 0.1954 | 0.1069 | 0.0885 | 0.949 |
| L | 17 | 0.1 | power_spherical | 0.1627 | 0.0902 | 0.0725 | 0.949 |
| L | 17 | 0.1 | tnbbeta | 0.0134 | 0.0033 | 0.0101 | 0.949 |
| L | 17 | 0.1 | tnbbeta_init0 | 0.0168 | 0.0039 | 0.0130 | 0.949 |
| L | 3 | 0.2 | vmf | 0.2098 | 0.0956 | 0.1142 | 1.072 |
| L | 3 | 0.2 | power_spherical | 0.1330 | 0.0764 | 0.0566 | 1.072 |
| L | 3 | 0.2 | tnbbeta | 0.0633 | 0.0006 | 0.0627 | 1.072 |
| L | 3 | 0.2 | tnbbeta_init0 | 0.0574 | 0.0008 | 0.0566 | 1.072 |
| L | 5 | 0.2 | vmf | 0.1891 | 0.0983 | 0.0909 | 0.881 |
| L | 5 | 0.2 | power_spherical | 0.1235 | 0.0678 | 0.0557 | 0.881 |
| L | 5 | 0.2 | tnbbeta | 0.0272 | 0.0018 | 0.0253 | 0.881 |
| L | 5 | 0.2 | tnbbeta_init0 | 0.0232 | 0.0018 | 0.0215 | 0.881 |
| L | 9 | 0.2 | vmf | 0.1441 | 0.0651 | 0.0790 | 0.674 |
| L | 9 | 0.2 | power_spherical | 0.1029 | 0.0463 | 0.0566 | 0.674 |
| L | 9 | 0.2 | tnbbeta | 0.0132 | 0.0005 | 0.0127 | 0.674 |
| L | 9 | 0.2 | tnbbeta_init0 | 0.0112 | 0.0007 | 0.0105 | 0.674 |
| L | 17 | 0.2 | vmf | 0.1103 | 0.0440 | 0.0663 | 0.445 |
| L | 17 | 0.2 | power_spherical | 0.0823 | 0.0321 | 0.0502 | 0.445 |
| L | 17 | 0.2 | tnbbeta | 0.0048 | 0.0004 | 0.0045 | 0.445 |
| L | 17 | 0.2 | tnbbeta_init0 | 0.0051 | 0.0003 | 0.0048 | 0.445 |
| L | 3 | 0.4 | vmf | 0.1183 | 0.0262 | 0.0922 | 0.556 |
| L | 3 | 0.4 | power_spherical | 0.0525 | 0.0167 | 0.0358 | 0.556 |
| L | 3 | 0.4 | tnbbeta | 0.0307 | 0.0025 | 0.0282 | 0.556 |
| L | 3 | 0.4 | tnbbeta_init0 | 0.0334 | 0.0043 | 0.0291 | 0.556 |
| L | 5 | 0.4 | vmf | 0.0977 | 0.0302 | 0.0675 | 0.405 |
| L | 5 | 0.4 | power_spherical | 0.0443 | 0.0124 | 0.0319 | 0.405 |
| L | 5 | 0.4 | tnbbeta | 0.0107 | 0.0007 | 0.0100 | 0.405 |
| L | 5 | 0.4 | tnbbeta_init0 | 0.0108 | 0.0006 | 0.0103 | 0.405 |
| L | 9 | 0.4 | vmf | 0.0532 | 0.0099 | 0.0433 | 0.272 |
| L | 9 | 0.4 | power_spherical | 0.0257 | 0.0036 | 0.0221 | 0.272 |
| L | 9 | 0.4 | tnbbeta | 0.0030 | -0.0000 | 0.0030 | 0.272 |
| L | 9 | 0.4 | tnbbeta_init0 | 0.0030 | -0.0000 | 0.0030 | 0.272 |
| L | 17 | 0.4 | vmf | 0.0251 | 0.0021 | 0.0230 | 0.158 |
| L | 17 | 0.4 | power_spherical | 0.0111 | 0.0007 | 0.0104 | 0.158 |
| L | 17 | 0.4 | tnbbeta | 0.0016 | 0.0000 | 0.0015 | 0.158 |
| L | 17 | 0.4 | tnbbeta_init0 | 0.0011 | 0.0000 | 0.0011 | 0.158 |
| Lb | 3 | 0.2 | vmf | 0.2096 | 0.0957 | 0.1139 | 1.072 |
| Lb | 3 | 0.2 | power_spherical | 0.1318 | 0.0760 | 0.0557 | 1.072 |
| Lb | 3 | 0.2 | tnbbeta | 0.0578 | 0.0003 | 0.0576 | 1.072 |
| Lb | 3 | 0.2 | tnbbeta_init0 | 0.0555 | 0.0016 | 0.0539 | 1.072 |
| Lb | 5 | 0.2 | vmf | 0.1891 | 0.0982 | 0.0909 | 0.881 |
| Lb | 5 | 0.2 | power_spherical | 0.1241 | 0.0680 | 0.0562 | 0.881 |
| Lb | 5 | 0.2 | tnbbeta | 0.0284 | 0.0030 | 0.0254 | 0.881 |
| Lb | 5 | 0.2 | tnbbeta_init0 | 0.0250 | 0.0021 | 0.0229 | 0.881 |
| Lb | 9 | 0.2 | vmf | 0.1444 | 0.0653 | 0.0791 | 0.674 |
| Lb | 9 | 0.2 | power_spherical | 0.1037 | 0.0466 | 0.0571 | 0.674 |
| Lb | 9 | 0.2 | tnbbeta | 0.0111 | 0.0008 | 0.0103 | 0.674 |
| Lb | 9 | 0.2 | tnbbeta_init0 | 0.0104 | 0.0008 | 0.0096 | 0.674 |
| Lb | 17 | 0.2 | vmf | 0.1104 | 0.0442 | 0.0662 | 0.445 |
| Lb | 17 | 0.2 | power_spherical | 0.0825 | 0.0322 | 0.0503 | 0.445 |
| Lb | 17 | 0.2 | tnbbeta | 0.0054 | 0.0004 | 0.0050 | 0.445 |
| Lb | 17 | 0.2 | tnbbeta_init0 | 0.0057 | 0.0004 | 0.0053 | 0.445 |
| M | 3 | 0.2 | vmf | 0.0755 | 0.0310 | 0.0445 | 1.072 |
| M | 3 | 0.2 | power_spherical | 0.0367 | 0.0254 | 0.0112 | 1.072 |
| M | 3 | 0.2 | tnbbeta | 0.0305 | 0.0113 | 0.0191 | 1.072 |
| M | 3 | 0.2 | tnbbeta_init0 | 0.0325 | 0.0147 | 0.0178 | 1.072 |
| M | 5 | 0.2 | vmf | 0.0448 | 0.0149 | 0.0298 | 0.881 |
| M | 5 | 0.2 | power_spherical | 0.0155 | 0.0155 | 0.0000 | 0.881 |
| M | 5 | 0.2 | tnbbeta | 0.0167 | 0.0139 | 0.0028 | 0.881 |
| M | 5 | 0.2 | tnbbeta_init0 | 0.0152 | 0.0067 | 0.0086 | 0.881 |
| M | 9 | 0.2 | vmf | 0.0208 | 0.0042 | 0.0166 | 0.674 |
| M | 9 | 0.2 | power_spherical | 0.0041 | 0.0041 | 0.0000 | 0.674 |
| M | 9 | 0.2 | tnbbeta | 0.0048 | 0.0041 | 0.0007 | 0.674 |
| M | 9 | 0.2 | tnbbeta_init0 | 0.0051 | 0.0039 | 0.0012 | 0.674 |
| M | 17 | 0.2 | vmf | 0.0097 | 0.0012 | 0.0086 | 0.445 |
| M | 17 | 0.2 | power_spherical | 0.0013 | 0.0013 | 0.0000 | 0.445 |
| M | 17 | 0.2 | tnbbeta | 0.0022 | 0.0015 | 0.0007 | 0.445 |
| M | 17 | 0.2 | tnbbeta_init0 | 0.0020 | 0.0015 | 0.0005 | 0.445 |

## Diagnostics: sigma_hat, alignment, ring statistic, ring thickness

| decoder | d | sigma | arm | diverged | sigma_hat / sigma (mean +- sd) | |cos(u, a)| median [min-max] (symmetry check) | mean |centre . u| (seed mean) | axis eigen-fraction (F) | ring R (seed median [min-max]) | std(u.z) q / exact model (seed mean) |
|---|---|---|---|---|---|---|---|---|---|
| L | 3 | 0.1 | vmf | 0/10 | 5.285 +- 0.010 | 0.273 [0.018-0.965] | 0.896 | 1.000 | 0.067 [0.063-0.071] | 0.4856 / 0.4002 |
| L | 3 | 0.1 | power_spherical | 0/10 | 4.799 +- 0.019 | 0.200 [0.005-0.971] | 0.999 | 1.000 | 0.053 [0.052-0.055] | 0.4381 / 0.3748 |
| L | 3 | 0.1 | tnbbeta | 0/10 | 1.275 +- 0.314 | 0.249 [0.004-0.979] | 1.000 | 1.000 | 0.052 [0.051-0.053] | 0.1196 / 0.1188 |
| L | 3 | 0.1 | tnbbeta_init0 | 0/10 | 1.388 +- 0.550 | 0.227 [0.020-0.962] | 1.000 | 1.000 | 0.052 [0.051-0.053] | 0.1284 / 0.1280 |
| L | 5 | 0.1 | vmf | 0/10 | 4.839 +- 0.011 | 0.202 [0.008-0.936] | 0.904 | 1.000 | 0.066 [0.065-0.068] | 0.4200 / 0.3385 |
| L | 5 | 0.1 | power_spherical | 0/10 | 4.263 +- 0.008 | 0.186 [0.006-0.947] | 0.998 | 1.000 | 0.058 [0.057-0.059] | 0.3811 / 0.3162 |
| L | 5 | 0.1 | tnbbeta | 0/10 | 1.544 +- 0.106 | 0.213 [0.010-0.956] | 1.000 | 1.000 | 0.057 [0.056-0.058] | 0.1445 / 0.1441 |
| L | 5 | 0.1 | tnbbeta_init0 | 0/10 | 1.793 +- 0.338 | 0.222 [0.001-0.923] | 1.000 | 1.000 | 0.057 [0.056-0.058] | 0.1664 / 0.1639 |
| L | 9 | 0.1 | vmf | 0/10 | 3.731 +- 0.008 | 0.138 [0.008-0.625] | 0.912 | 1.000 | 0.065 [0.064-0.065] | 0.3214 / 0.2532 |
| L | 9 | 0.1 | power_spherical | 0/10 | 3.435 +- 0.007 | 0.145 [0.003-0.624] | 0.993 | 1.000 | 0.061 [0.060-0.061] | 0.3012 / 0.2432 |
| L | 9 | 0.1 | tnbbeta | 0/10 | 1.523 +- 0.145 | 0.158 [0.016-0.678] | 0.999 | 1.000 | 0.060 [0.059-0.061] | 0.1390 / 0.1379 |
| L | 9 | 0.1 | tnbbeta_init0 | 0/10 | 1.411 +- 0.138 | 0.145 [0.004-0.644] | 1.000 | 1.000 | 0.060 [0.059-0.060] | 0.1299 / 0.1294 |
| L | 17 | 0.1 | vmf | 0/10 | 2.805 +- 0.005 | 0.130 [0.002-0.368] | 0.946 | 1.000 | 0.063 [0.063-0.064] | 0.2378 / 0.1854 |
| L | 17 | 0.1 | power_spherical | 0/10 | 2.641 +- 0.005 | 0.129 [0.008-0.370] | 0.988 | 1.000 | 0.062 [0.061-0.062] | 0.2274 / 0.1804 |
| L | 17 | 0.1 | tnbbeta | 0/10 | 1.249 +- 0.078 | 0.132 [0.004-0.366] | 0.999 | 1.000 | 0.062 [0.061-0.062] | 0.1118 / 0.1110 |
| L | 17 | 0.1 | tnbbeta_init0 | 0/10 | 1.270 +- 0.101 | 0.130 [0.001-0.371] | 0.999 | 1.000 | 0.062 [0.061-0.062] | 0.1135 / 0.1124 |
| L | 3 | 0.2 | vmf | 0/10 | 2.689 +- 0.006 | 0.174 [0.012-0.990] | 0.894 | 1.000 | 0.068 [0.066-0.073] | 0.4848 / 0.4022 |
| L | 3 | 0.2 | power_spherical | 0/10 | 2.458 +- 0.015 | 0.207 [0.002-0.982] | 0.998 | 1.000 | 0.053 [0.052-0.054] | 0.4412 / 0.3786 |
| L | 3 | 0.2 | tnbbeta | 0/10 | 1.175 +- 0.108 | 0.215 [0.005-0.969] | 1.000 | 1.000 | 0.052 [0.051-0.053] | 0.2094 / 0.2068 |
| L | 3 | 0.2 | tnbbeta_init0 | 0/10 | 1.098 +- 0.145 | 0.306 [0.014-0.962] | 1.000 | 1.000 | 0.053 [0.051-0.054] | 0.1977 / 0.1943 |
| L | 5 | 0.2 | vmf | 0/10 | 2.488 +- 0.008 | 0.183 [0.023-0.951] | 0.907 | 1.000 | 0.067 [0.064-0.069] | 0.4190 / 0.3413 |
| L | 5 | 0.2 | power_spherical | 0/10 | 2.187 +- 0.006 | 0.193 [0.002-0.944] | 0.998 | 1.000 | 0.058 [0.057-0.058] | 0.3794 / 0.3187 |
| L | 5 | 0.2 | tnbbeta | 0/10 | 1.138 +- 0.048 | 0.211 [0.016-0.933] | 0.999 | 1.000 | 0.057 [0.056-0.058] | 0.2015 / 0.1996 |
| L | 5 | 0.2 | tnbbeta_init0 | 0/10 | 1.137 +- 0.043 | 0.223 [0.015-0.922] | 1.000 | 1.000 | 0.057 [0.056-0.058] | 0.2013 / 0.1995 |
| L | 9 | 0.2 | vmf | 0/10 | 1.955 +- 0.006 | 0.149 [0.003-0.647] | 0.926 | 1.000 | 0.064 [0.063-0.065] | 0.3200 / 0.2573 |
| L | 9 | 0.2 | power_spherical | 0/10 | 1.795 +- 0.004 | 0.147 [0.005-0.633] | 0.993 | 1.000 | 0.061 [0.060-0.061] | 0.2995 / 0.2472 |
| L | 9 | 0.2 | tnbbeta | 0/10 | 1.096 +- 0.036 | 0.156 [0.004-0.667] | 0.999 | 1.000 | 0.060 [0.059-0.060] | 0.1832 / 0.1823 |
| L | 9 | 0.2 | tnbbeta_init0 | 0/10 | 1.104 +- 0.051 | 0.151 [0.009-0.640] | 1.000 | 1.000 | 0.060 [0.059-0.061] | 0.1843 / 0.1832 |
| L | 17 | 0.2 | vmf | 0/10 | 1.526 +- 0.003 | 0.122 [0.006-0.364] | 0.952 | 1.000 | 0.063 [0.063-0.064] | 0.2373 / 0.1912 |
| L | 17 | 0.2 | power_spherical | 0/10 | 1.439 +- 0.004 | 0.124 [0.010-0.366] | 0.988 | 1.000 | 0.062 [0.061-0.062] | 0.2262 / 0.1866 |
| L | 17 | 0.2 | tnbbeta | 0/10 | 1.028 +- 0.017 | 0.128 [0.009-0.360] | 0.999 | 1.000 | 0.061 [0.061-0.062] | 0.1574 / 0.1567 |
| L | 17 | 0.2 | tnbbeta_init0 | 0/10 | 1.027 +- 0.015 | 0.121 [0.006-0.370] | 0.999 | 1.000 | 0.062 [0.061-0.062] | 0.1575 / 0.1566 |
| L | 3 | 0.4 | vmf | 0/10 | 1.455 +- 0.005 | 0.229 [0.009-0.970] | 0.903 | 1.000 | 0.065 [0.063-0.070] | 0.4821 / 0.4127 |
| L | 3 | 0.4 | power_spherical | 0/10 | 1.361 +- 0.006 | 0.219 [0.001-0.987] | 0.999 | 1.000 | 0.053 [0.052-0.054] | 0.4438 / 0.3954 |
| L | 3 | 0.4 | tnbbeta | 0/10 | 1.159 +- 0.015 | 0.212 [0.002-0.966] | 0.999 | 1.000 | 0.052 [0.052-0.054] | 0.3682 / 0.3529 |
| L | 3 | 0.4 | tnbbeta_init0 | 0/10 | 1.187 +- 0.064 | 0.261 [0.005-0.968] | 1.000 | 1.000 | 0.053 [0.051-0.055] | 0.3758 / 0.3589 |
| L | 5 | 0.4 | vmf | 0/10 | 1.399 +- 0.007 | 0.161 [0.025-0.961] | 0.917 | 1.000 | 0.065 [0.063-0.068] | 0.4169 / 0.3539 |
| L | 5 | 0.4 | power_spherical | 0/10 | 1.245 +- 0.005 | 0.179 [0.005-0.942] | 0.996 | 1.000 | 0.058 [0.057-0.058] | 0.3801 / 0.3344 |
| L | 5 | 0.4 | tnbbeta | 0/10 | 1.055 +- 0.016 | 0.200 [0.019-0.959] | 0.999 | 1.000 | 0.057 [0.057-0.059] | 0.3090 / 0.3042 |
| L | 5 | 0.4 | tnbbeta_init0 | 0/10 | 1.050 +- 0.013 | 0.207 [0.003-0.942] | 1.000 | 1.000 | 0.058 [0.057-0.058] | 0.3095 / 0.3034 |
| L | 9 | 0.4 | vmf | 0/10 | 1.191 +- 0.007 | 0.158 [0.015-0.648] | 0.949 | 1.000 | 0.063 [0.062-0.064] | 0.3196 / 0.2743 |
| L | 9 | 0.4 | power_spherical | 0/10 | 1.118 +- 0.007 | 0.140 [0.003-0.631] | 0.993 | 1.000 | 0.061 [0.060-0.061] | 0.3008 / 0.2678 |
| L | 9 | 0.4 | tnbbeta | 0/10 | 1.015 +- 0.011 | 0.150 [0.005-0.666] | 0.999 | 1.000 | 0.060 [0.059-0.061] | 0.2576 / 0.2569 |
| L | 9 | 0.4 | tnbbeta_init0 | 0/10 | 1.015 +- 0.013 | 0.147 [0.007-0.618] | 0.999 | 1.000 | 0.060 [0.059-0.060] | 0.2581 / 0.2569 |
| L | 17 | 0.4 | vmf | 0/10 | 1.065 +- 0.003 | 0.119 [0.009-0.365] | 0.971 | 1.000 | 0.062 [0.062-0.063] | 0.2378 / 0.2110 |
| L | 17 | 0.4 | power_spherical | 0/10 | 1.038 +- 0.003 | 0.125 [0.001-0.366] | 0.992 | 1.000 | 0.062 [0.061-0.062] | 0.2283 / 0.2095 |
| L | 17 | 0.4 | tnbbeta | 0/10 | 1.002 +- 0.005 | 0.120 [0.013-0.363] | 0.997 | 1.000 | 0.062 [0.061-0.062] | 0.2079 / 0.2074 |
| L | 17 | 0.4 | tnbbeta_init0 | 0/10 | 1.001 +- 0.005 | 0.116 [0.002-0.370] | 0.998 | 1.000 | 0.062 [0.061-0.062] | 0.2081 / 0.2074 |
| Lb | 3 | 0.2 | vmf | 0/10 | 2.690 +- 0.005 | 0.213 [0.024-0.981] | 0.901 | 1.000 | 0.067 [0.064-0.074] | 0.4842 / 0.4022 |
| Lb | 3 | 0.2 | power_spherical | 0/10 | 2.452 +- 0.005 | 0.220 [0.010-0.979] | 0.998 | 1.000 | 0.053 [0.053-0.054] | 0.4393 / 0.3780 |
| Lb | 3 | 0.2 | tnbbeta | 0/10 | 1.118 +- 0.086 | 0.232 [0.006-0.960] | 1.000 | 1.000 | 0.052 [0.051-0.053] | 0.2003 / 0.1977 |
| Lb | 3 | 0.2 | tnbbeta_init0 | 0/10 | 1.096 +- 0.165 | 0.200 [0.014-0.974] | 1.000 | 1.000 | 0.052 [0.051-0.054] | 0.1972 / 0.1940 |
| Lb | 5 | 0.2 | vmf | 0/10 | 2.487 +- 0.007 | 0.168 [0.024-0.935] | 0.910 | 1.000 | 0.066 [0.065-0.069] | 0.4189 / 0.3413 |
| Lb | 5 | 0.2 | power_spherical | 0/10 | 2.187 +- 0.002 | 0.186 [0.006-0.949] | 0.997 | 1.000 | 0.058 [0.057-0.059] | 0.3793 / 0.3188 |
| Lb | 5 | 0.2 | tnbbeta | 0/10 | 1.176 +- 0.060 | 0.210 [0.005-0.944] | 1.000 | 1.000 | 0.057 [0.056-0.058] | 0.2073 / 0.2051 |
| Lb | 5 | 0.2 | tnbbeta_init0 | 0/10 | 1.139 +- 0.066 | 0.229 [0.012-0.923] | 1.000 | 1.000 | 0.057 [0.056-0.058] | 0.2015 / 0.1997 |
| Lb | 9 | 0.2 | vmf | 0/10 | 1.956 +- 0.006 | 0.145 [0.007-0.679] | 0.924 | 1.000 | 0.064 [0.063-0.066] | 0.3201 / 0.2573 |
| Lb | 9 | 0.2 | power_spherical | 0/10 | 1.796 +- 0.006 | 0.144 [0.016-0.630] | 0.992 | 1.000 | 0.060 [0.060-0.061] | 0.2998 / 0.2473 |
| Lb | 9 | 0.2 | tnbbeta | 0/10 | 1.097 +- 0.057 | 0.155 [0.008-0.674] | 0.999 | 1.000 | 0.060 [0.059-0.060] | 0.1833 / 0.1823 |
| Lb | 9 | 0.2 | tnbbeta_init0 | 0/10 | 1.096 +- 0.043 | 0.152 [0.005-0.625] | 1.000 | 1.000 | 0.060 [0.059-0.061] | 0.1831 / 0.1823 |
| Lb | 17 | 0.2 | vmf | 0/10 | 1.527 +- 0.003 | 0.141 [0.004-0.368] | 0.957 | 1.000 | 0.063 [0.062-0.063] | 0.2375 / 0.1912 |
| Lb | 17 | 0.2 | power_spherical | 0/10 | 1.439 +- 0.003 | 0.130 [0.007-0.368] | 0.988 | 1.000 | 0.062 [0.061-0.062] | 0.2262 / 0.1866 |
| Lb | 17 | 0.2 | tnbbeta | 0/10 | 1.021 +- 0.007 | 0.127 [0.007-0.365] | 0.999 | 1.000 | 0.062 [0.061-0.062] | 0.1569 / 0.1561 |
| Lb | 17 | 0.2 | tnbbeta_init0 | 0/10 | 1.034 +- 0.033 | 0.120 [0.005-0.371] | 0.999 | 1.000 | 0.062 [0.061-0.062] | 0.1583 / 0.1573 |
| M | 3 | 0.2 | vmf | 0/10 | 2.536 +- 0.011 | 0.214 [0.031-0.968] | 0.955 | 1.000 | 0.060 [0.057-0.062] | 0.5038 / 0.4807 |
| M | 3 | 0.2 | power_spherical | 0/10 | 2.329 +- 0.012 | 0.212 [0.003-0.959] | 0.998 | 1.000 | 0.053 [0.051-0.053] | 0.4526 / 0.4336 |
| M | 3 | 0.2 | tnbbeta | 0/10 | 1.334 +- 1.087 | 0.267 [0.001-0.962] | 0.998 | 1.000 | 0.053 [0.052-0.054] | 0.2494 / 0.2462 |
| M | 3 | 0.2 | tnbbeta_init0 | 0/10 | 1.579 +- 0.987 | 0.237 [0.018-0.960] | 0.999 | 1.000 | 0.053 [0.052-0.053] | 0.2979 / 0.2919 |
| M | 5 | 0.2 | vmf | 0/10 | 2.262 +- 0.010 | 0.180 [0.006-0.970] | 0.977 | 1.000 | 0.060 [0.060-0.061] | 0.4267 / 0.4140 |
| M | 5 | 0.2 | power_spherical | 0/10 | 2.454 +- 0.012 | 0.260 [0.001-0.903] | 0.647 | 1.000 | 0.057 [0.057-0.058] | 0.4469 / 0.4472 |
| M | 5 | 0.2 | tnbbeta | 0/10 | 2.213 +- 0.165 | 0.215 [0.019-0.946] | 0.994 | 1.000 | 0.058 [0.056-0.058] | 0.4074 / 0.4023 |
| M | 5 | 0.2 | tnbbeta_init0 | 0/10 | 1.238 +- 0.754 | 0.226 [0.017-0.963] | 0.996 | 1.000 | 0.057 [0.056-0.058] | 0.2263 / 0.2232 |
| M | 9 | 0.2 | vmf | 0/10 | 1.837 +- 0.011 | 0.138 [0.004-0.625] | 0.984 | 1.000 | 0.061 [0.061-0.062] | 0.3260 / 0.3157 |
| M | 9 | 0.2 | power_spherical | 0/10 | 1.947 +- 0.011 | 0.134 [0.007-0.632] | 0.639 | 1.000 | 0.060 [0.060-0.061] | 0.3329 / 0.3333 |
| M | 9 | 0.2 | tnbbeta | 0/10 | 1.820 +- 0.267 | 0.141 [0.016-0.640] | 0.895 | 1.000 | 0.060 [0.060-0.061] | 0.3116 / 0.3116 |
| M | 9 | 0.2 | tnbbeta_init0 | 0/10 | 1.656 +- 0.291 | 0.129 [0.001-0.627] | 0.956 | 1.000 | 0.060 [0.060-0.060] | 0.2837 / 0.2837 |
| M | 17 | 0.2 | vmf | 0/10 | 1.511 +- 0.004 | 0.125 [0.000-0.359] | 0.984 | 1.000 | 0.061 [0.061-0.062] | 0.2404 / 0.2341 |
| M | 17 | 0.2 | power_spherical | 0/10 | 1.572 +- 0.004 | 0.134 [0.004-0.393] | 0.348 | 1.000 | 0.061 [0.061-0.062] | 0.2423 / 0.2425 |
| M | 17 | 0.2 | tnbbeta | 0/10 | 1.572 +- 0.004 | 0.123 [0.009-0.367] | 0.795 | 1.000 | 0.062 [0.061-0.062] | 0.2426 / 0.2425 |
| M | 17 | 0.2 | tnbbeta_init0 | 0/10 | 1.529 +- 0.055 | 0.118 [0.006-0.377] | 0.935 | 1.000 | 0.062 [0.061-0.062] | 0.2367 / 0.2362 |
| F | 3 | 0.2 | vmf | 0/10 | 2.524 +- 0.010 | 0.526 [0.059-0.994] | 0.938 | 0.991 | 0.065 [0.063-0.070] | 0.5033 / nan |
| F | 3 | 0.2 | power_spherical | 0/10 | 2.337 +- 0.018 | 0.465 [0.067-0.990] | 0.997 | 0.998 | 0.053 [0.052-0.054] | 0.4551 / nan |
| F | 3 | 0.2 | tnbbeta | 0/10 | 2.607 +- 0.174 | 0.547 [0.007-0.913] | 0.997 | 0.937 | 0.054 [0.053-0.055] | 0.4951 / nan |
| F | 3 | 0.2 | tnbbeta_init0 | 0/10 | 1.734 +- 1.139 | 0.513 [0.010-0.996] | 0.998 | 0.936 | 0.054 [0.052-0.055] | 0.3283 / nan |
| F | 5 | 0.2 | vmf | 0/10 | 2.261 +- 0.011 | 0.315 [0.121-0.679] | 0.966 | 0.997 | 0.062 [0.061-0.063] | 0.4269 / nan |
| F | 5 | 0.2 | power_spherical | 0/10 | 2.453 +- 0.011 | 0.000 [0.000-0.000] | 0.415 | 0.000 | 0.057 [0.056-0.057] | 0.4467 / nan |
| F | 5 | 0.2 | tnbbeta | 0/10 | 2.131 +- 0.114 | 0.352 [0.073-0.861] | 0.979 | 0.961 | 0.057 [0.057-0.059] | 0.3899 / nan |
| F | 5 | 0.2 | tnbbeta_init0 | 0/10 | 1.750 +- 0.726 | 0.539 [0.182-0.784] | 0.916 | 0.909 | 0.058 [0.057-0.058] | 0.3196 / nan |
| F | 9 | 0.2 | vmf | 0/10 | 1.836 +- 0.012 | 0.346 [0.033-0.586] | 0.978 | 0.998 | 0.062 [0.061-0.062] | 0.3261 / nan |
| F | 9 | 0.2 | power_spherical | 0/10 | 1.947 +- 0.011 | 0.000 [0.000-0.529] | 0.317 | 0.276 | 0.060 [0.060-0.060] | 0.3329 / nan |
| F | 9 | 0.2 | tnbbeta | 0/10 | 1.946 +- 0.011 | 0.350 [0.082-0.807] | 0.797 | 0.871 | 0.060 [0.059-0.061] | 0.3331 / nan |
| F | 9 | 0.2 | tnbbeta_init0 | 0/10 | 1.860 +- 0.182 | 0.194 [0.064-0.519] | 0.842 | 0.805 | 0.060 [0.059-0.060] | 0.3192 / nan |
| F | 17 | 0.2 | vmf | 0/10 | 1.513 +- 0.005 | 0.183 [0.098-0.415] | 0.979 | 0.998 | 0.062 [0.062-0.062] | 0.2404 / nan |
| F | 17 | 0.2 | power_spherical | 0/10 | 1.571 +- 0.004 | 0.043 [0.000-0.466] | 0.244 | 0.429 | 0.061 [0.061-0.061] | 0.2423 / nan |
| F | 17 | 0.2 | tnbbeta | 0/10 | 1.571 +- 0.004 | 0.205 [0.047-0.393] | 0.697 | 0.726 | 0.062 [0.061-0.062] | 0.2424 / nan |
| F | 17 | 0.2 | tnbbeta_init0 | 0/10 | 1.571 +- 0.004 | 0.245 [0.041-0.338] | 0.712 | 0.807 | 0.062 [0.061-0.062] | 0.2425 / nan |

## Fitted posterior parameters

| decoder | d | sigma | arm | p | q | epsilon | m = eps - (d-1)/2 | kappa |
|---|---|---|---|---|---|---|---|---|
| L | 3 | 0.1 | vmf | - | - | - | - | 1.15 |
| L | 3 | 0.1 | power_spherical | - | - | - | - | 1.29 |
| L | 3 | 0.1 | tnbbeta | 0.742 | 0.61 | 8.1 | 7.1 | - |
| L | 3 | 0.1 | tnbbeta_init0 | 0.591 | 0.56 | 7.29 | 6.29 | - |
| L | 5 | 0.1 | vmf | - | - | - | - | 1 |
| L | 5 | 0.1 | power_spherical | - | - | - | - | 1.26 |
| L | 5 | 0.1 | tnbbeta | 0.67 | 0.415 | 9.63 | 7.63 | - |
| L | 5 | 0.1 | tnbbeta_init0 | 0.37 | 0.29 | 9 | 7 | - |
| L | 9 | 0.1 | vmf | - | - | - | - | 1.08 |
| L | 9 | 0.1 | power_spherical | - | - | - | - | 1.42 |
| L | 9 | 0.1 | tnbbeta | 0.609 | 0.458 | 12 | 8.02 | - |
| L | 9 | 0.1 | tnbbeta_init0 | 0.406 | 0.464 | 14.2 | 10.2 | - |
| L | 17 | 0.1 | vmf | - | - | - | - | 1.41 |
| L | 17 | 0.1 | power_spherical | - | - | - | - | 1.74 |
| L | 17 | 0.1 | tnbbeta | 0.571 | 0.538 | 17.4 | 9.39 | - |
| L | 17 | 0.1 | tnbbeta_init0 | 0.441 | 0.497 | 19.4 | 11.4 | - |
| L | 3 | 0.2 | vmf | - | - | - | - | 1.1 |
| L | 3 | 0.2 | power_spherical | - | - | - | - | 1.2 |
| L | 3 | 0.2 | tnbbeta | 0.748 | 0.172 | 4.84 | 3.84 | - |
| L | 3 | 0.2 | tnbbeta_init0 | 0.506 | 0.18 | 5.45 | 4.45 | - |
| L | 5 | 0.2 | vmf | - | - | - | - | 1 |
| L | 5 | 0.2 | power_spherical | - | - | - | - | 1.25 |
| L | 5 | 0.2 | tnbbeta | 0.665 | 0.247 | 6.31 | 4.31 | - |
| L | 5 | 0.2 | tnbbeta_init0 | 0.336 | 0.216 | 6.54 | 4.54 | - |
| L | 9 | 0.2 | vmf | - | - | - | - | 1.16 |
| L | 9 | 0.2 | power_spherical | - | - | - | - | 1.51 |
| L | 9 | 0.2 | tnbbeta | 0.609 | 0.318 | 8.86 | 4.86 | - |
| L | 9 | 0.2 | tnbbeta_init0 | 0.527 | 0.254 | 9.46 | 5.46 | - |
| L | 17 | 0.2 | vmf | - | - | - | - | 1.54 |
| L | 17 | 0.2 | power_spherical | - | - | - | - | 1.89 |
| L | 17 | 0.2 | tnbbeta | 0.566 | 0.357 | 12.4 | 4.37 | - |
| L | 17 | 0.2 | tnbbeta_init0 | 0.441 | 0.338 | 12.5 | 4.5 | - |
| L | 3 | 0.4 | vmf | - | - | - | - | 1.06 |
| L | 3 | 0.4 | power_spherical | - | - | - | - | 1.11 |
| L | 3 | 0.4 | tnbbeta | 0.722 | 0.337 | 1.68 | 0.68 | - |
| L | 3 | 0.4 | tnbbeta_init0 | 0.5 | 0.344 | 1.64 | 0.642 | - |
| L | 5 | 0.4 | vmf | - | - | - | - | 1.01 |
| L | 5 | 0.4 | power_spherical | - | - | - | - | 1.14 |
| L | 5 | 0.4 | tnbbeta | 0.641 | 0.332 | 2.86 | 0.863 | - |
| L | 5 | 0.4 | tnbbeta_init0 | 0.372 | 0.32 | 2.87 | 0.866 | - |
| L | 9 | 0.4 | vmf | - | - | - | - | 1.13 |
| L | 9 | 0.4 | power_spherical | - | - | - | - | 1.42 |
| L | 9 | 0.4 | tnbbeta | 0.585 | 0.253 | 4.99 | 0.995 | - |
| L | 9 | 0.4 | tnbbeta_init0 | 0.419 | 0.258 | 5.01 | 1.01 | - |
| L | 17 | 0.4 | vmf | - | - | - | - | 1.35 |
| L | 17 | 0.4 | power_spherical | - | - | - | - | 1.56 |
| L | 17 | 0.4 | tnbbeta | 0.546 | 0.194 | 8.83 | 0.83 | - |
| L | 17 | 0.4 | tnbbeta_init0 | 0.454 | 0.215 | 8.65 | 0.649 | - |
| Lb | 3 | 0.2 | vmf | - | - | - | - | 1.11 |
| Lb | 3 | 0.2 | power_spherical | - | - | - | - | 1.24 |
| Lb | 3 | 0.2 | tnbbeta | 0.749 | 0.218 | 5.09 | 4.09 | - |
| Lb | 3 | 0.2 | tnbbeta_init0 | 0.5 | 0.189 | 6.35 | 5.35 | - |
| Lb | 5 | 0.2 | vmf | - | - | - | - | 1.01 |
| Lb | 5 | 0.2 | power_spherical | - | - | - | - | 1.25 |
| Lb | 5 | 0.2 | tnbbeta | 0.659 | 0.248 | 6.37 | 4.37 | - |
| Lb | 5 | 0.2 | tnbbeta_init0 | 0.339 | 0.237 | 6.5 | 4.5 | - |
| Lb | 9 | 0.2 | vmf | - | - | - | - | 1.15 |
| Lb | 9 | 0.2 | power_spherical | - | - | - | - | 1.47 |
| Lb | 9 | 0.2 | tnbbeta | 0.608 | 0.301 | 8.94 | 4.94 | - |
| Lb | 9 | 0.2 | tnbbeta_init0 | 0.471 | 0.289 | 9.23 | 5.23 | - |
| Lb | 17 | 0.2 | vmf | - | - | - | - | 1.46 |
| Lb | 17 | 0.2 | power_spherical | - | - | - | - | 1.88 |
| Lb | 17 | 0.2 | tnbbeta | 0.567 | 0.369 | 12.2 | 4.25 | - |
| Lb | 17 | 0.2 | tnbbeta_init0 | 0.436 | 0.324 | 12.6 | 4.64 | - |
| M | 3 | 0.2 | vmf | - | - | - | - | 1 |
| M | 3 | 0.2 | power_spherical | - | - | - | - | 1.09 |
| M | 3 | 0.2 | tnbbeta | 0.52 | 0.41 | 7.19 | 6.19 | - |
| M | 3 | 0.2 | tnbbeta_init0 | 0.499 | 0.248 | 4.01 | 3.01 | - |
| M | 5 | 0.2 | vmf | - | - | - | - | 1 |
| M | 5 | 0.2 | power_spherical | - | - | - | - | 2.85e-05 |
| M | 5 | 0.2 | tnbbeta | 0.612 | 0.347 | 1.85 | -0.15 | - |
| M | 5 | 0.2 | tnbbeta_init0 | 0.494 | 0.338 | 14.7 | 12.7 | - |
| M | 9 | 0.2 | vmf | - | - | - | - | 1 |
| M | 9 | 0.2 | power_spherical | - | - | - | - | 3.44e-06 |
| M | 9 | 0.2 | tnbbeta | 0.502 | 0.0291 | 3.93 | -0.0749 | - |
| M | 9 | 0.2 | tnbbeta_init0 | 0.501 | 0.136 | 4.8 | 0.795 | - |
| M | 17 | 0.2 | vmf | - | - | - | - | 1 |
| M | 17 | 0.2 | power_spherical | - | - | - | - | 3.96e-09 |
| M | 17 | 0.2 | tnbbeta | 0.5 | 0.0602 | 7.49 | -0.507 | - |
| M | 17 | 0.2 | tnbbeta_init0 | 0.496 | 0.112 | 7.32 | -0.683 | - |
| F | 3 | 0.2 | vmf | - | - | - | - | 1.01 |
| F | 3 | 0.2 | power_spherical | - | - | - | - | 1.06 |
| F | 3 | 0.2 | tnbbeta | 0.692 | 0.451 | 0.949 | -0.0509 | - |
| F | 3 | 0.2 | tnbbeta_init0 | 0.497 | 0.447 | 0.985 | -0.0155 | - |
| F | 5 | 0.2 | vmf | - | - | - | - | 1 |
| F | 5 | 0.2 | power_spherical | - | - | - | - | 4.67e-05 |
| F | 5 | 0.2 | tnbbeta | 0.615 | 0.402 | 1.78 | -0.222 | - |
| F | 5 | 0.2 | tnbbeta_init0 | 0.498 | 0.316 | 1.92 | -0.0765 | - |
| F | 9 | 0.2 | vmf | - | - | - | - | 1 |
| F | 9 | 0.2 | power_spherical | - | - | - | - | 3.44e-05 |
| F | 9 | 0.2 | tnbbeta | 0.5 | 0.0242 | 3.91 | -0.094 | - |
| F | 9 | 0.2 | tnbbeta_init0 | 0.499 | 0.0494 | 3.81 | -0.191 | - |
| F | 17 | 0.2 | vmf | - | - | - | - | 1 |
| F | 17 | 0.2 | power_spherical | - | - | - | - | 2.3e-05 |
| F | 17 | 0.2 | tnbbeta | 0.5 | 0.0702 | 7.5 | -0.5 | - |
| F | 17 | 0.2 | tnbbeta_init0 | 0.499 | 0.0987 | 7.26 | -0.736 | - |

## Descriptive contrasts (sigma = 0.2; paired by seed)

| comparison | d | n | mean gap (A) | mean gap (B) | mean paired diff (A - B) | seeds A better |
|---|---|---|---|---|---|---|
| L: tnbbeta vs vmf | 3 | 10 | 0.0633 | 0.2098 | -0.1465 | 10/10 |
| L: tnbbeta vs vmf | 5 | 10 | 0.0272 | 0.1891 | -0.1619 | 10/10 |
| L: tnbbeta vs vmf | 9 | 10 | 0.0132 | 0.1441 | -0.1309 | 10/10 |
| L: tnbbeta vs vmf | 17 | 10 | 0.0048 | 0.1103 | -0.1055 | 10/10 |
| L: tnbbeta vs power_spherical | 3 | 10 | 0.0633 | 0.1330 | -0.0697 | 10/10 |
| L: tnbbeta vs power_spherical | 5 | 10 | 0.0272 | 0.1235 | -0.0963 | 10/10 |
| L: tnbbeta vs power_spherical | 9 | 10 | 0.0132 | 0.1029 | -0.0897 | 10/10 |
| L: tnbbeta vs power_spherical | 17 | 10 | 0.0048 | 0.0823 | -0.0774 | 10/10 |
| L: tnbbeta vs tnbbeta_init0 | 3 | 10 | 0.0633 | 0.0574 | 0.0060 | 3/10 |
| L: tnbbeta vs tnbbeta_init0 | 5 | 10 | 0.0272 | 0.0232 | 0.0039 | 2/10 |
| L: tnbbeta vs tnbbeta_init0 | 9 | 10 | 0.0132 | 0.0112 | 0.0020 | 4/10 |
| L: tnbbeta vs tnbbeta_init0 | 17 | 10 | 0.0048 | 0.0051 | -0.0003 | 5/10 |
| L: vmf vs power_spherical | 3 | 10 | 0.2098 | 0.1330 | 0.0768 | 0/10 |
| L: vmf vs power_spherical | 5 | 10 | 0.1891 | 0.1235 | 0.0656 | 0/10 |
| L: vmf vs power_spherical | 9 | 10 | 0.1441 | 0.1029 | 0.0412 | 0/10 |
| L: vmf vs power_spherical | 17 | 10 | 0.1103 | 0.0823 | 0.0281 | 0/10 |
| Lb: tnbbeta vs vmf | 3 | 10 | 0.0578 | 0.2096 | -0.1518 | 10/10 |
| Lb: tnbbeta vs vmf | 5 | 10 | 0.0284 | 0.1891 | -0.1607 | 10/10 |
| Lb: tnbbeta vs vmf | 9 | 10 | 0.0111 | 0.1444 | -0.1332 | 10/10 |
| Lb: tnbbeta vs vmf | 17 | 10 | 0.0054 | 0.1104 | -0.1050 | 10/10 |
| Lb: tnbbeta vs power_spherical | 3 | 10 | 0.0578 | 0.1318 | -0.0739 | 10/10 |
| Lb: tnbbeta vs power_spherical | 5 | 10 | 0.0284 | 0.1241 | -0.0957 | 10/10 |
| Lb: tnbbeta vs power_spherical | 9 | 10 | 0.0111 | 0.1037 | -0.0925 | 10/10 |
| Lb: tnbbeta vs power_spherical | 17 | 10 | 0.0054 | 0.0825 | -0.0771 | 10/10 |
| Lb: tnbbeta vs tnbbeta_init0 | 3 | 10 | 0.0578 | 0.0555 | 0.0024 | 3/10 |
| Lb: tnbbeta vs tnbbeta_init0 | 5 | 10 | 0.0284 | 0.0250 | 0.0034 | 4/10 |
| Lb: tnbbeta vs tnbbeta_init0 | 9 | 10 | 0.0111 | 0.0104 | 0.0008 | 6/10 |
| Lb: tnbbeta vs tnbbeta_init0 | 17 | 10 | 0.0054 | 0.0057 | -0.0003 | 6/10 |
| Lb: vmf vs power_spherical | 3 | 10 | 0.2096 | 0.1318 | 0.0778 | 0/10 |
| Lb: vmf vs power_spherical | 5 | 10 | 0.1891 | 0.1241 | 0.0650 | 0/10 |
| Lb: vmf vs power_spherical | 9 | 10 | 0.1444 | 0.1037 | 0.0407 | 0/10 |
| Lb: vmf vs power_spherical | 17 | 10 | 0.1104 | 0.0825 | 0.0279 | 0/10 |
| M: tnbbeta vs vmf | 3 | 10 | 0.0305 | 0.0755 | -0.0450 | 10/10 |
| M: tnbbeta vs vmf | 5 | 10 | 0.0167 | 0.0448 | -0.0281 | 10/10 |
| M: tnbbeta vs vmf | 9 | 10 | 0.0048 | 0.0208 | -0.0160 | 10/10 |
| M: tnbbeta vs vmf | 17 | 10 | 0.0022 | 0.0097 | -0.0075 | 10/10 |
| M: tnbbeta vs power_spherical | 3 | 10 | 0.0305 | 0.0367 | -0.0062 | 7/10 |
| M: tnbbeta vs power_spherical | 5 | 10 | 0.0167 | 0.0155 | 0.0011 | 3/10 |
| M: tnbbeta vs power_spherical | 9 | 10 | 0.0048 | 0.0041 | 0.0007 | 0/10 |
| M: tnbbeta vs power_spherical | 17 | 10 | 0.0022 | 0.0013 | 0.0010 | 1/10 |
| M: tnbbeta vs tnbbeta_init0 | 3 | 10 | 0.0305 | 0.0325 | -0.0020 | 7/10 |
| M: tnbbeta vs tnbbeta_init0 | 5 | 10 | 0.0167 | 0.0152 | 0.0014 | 5/10 |
| M: tnbbeta vs tnbbeta_init0 | 9 | 10 | 0.0048 | 0.0051 | -0.0003 | 8/10 |
| M: tnbbeta vs tnbbeta_init0 | 17 | 10 | 0.0022 | 0.0020 | 0.0002 | 3/10 |
| M: vmf vs power_spherical | 3 | 10 | 0.0755 | 0.0367 | 0.0388 | 0/10 |
| M: vmf vs power_spherical | 5 | 10 | 0.0448 | 0.0155 | 0.0292 | 0/10 |
| M: vmf vs power_spherical | 9 | 10 | 0.0208 | 0.0041 | 0.0167 | 0/10 |
| M: vmf vs power_spherical | 17 | 10 | 0.0097 | 0.0013 | 0.0084 | 0/10 |
| F: tnbbeta vs vmf | 3 | 10 | 0.0359 | 0.0751 | -0.0391 | 10/10 |
| F: tnbbeta vs vmf | 5 | 10 | 0.0150 | 0.0450 | -0.0300 | 10/10 |
| F: tnbbeta vs vmf | 9 | 10 | 0.0046 | 0.0207 | -0.0161 | 10/10 |
| F: tnbbeta vs vmf | 17 | 10 | 0.0016 | 0.0098 | -0.0083 | 10/10 |
| F: tnbbeta vs power_spherical | 3 | 10 | 0.0359 | 0.0368 | -0.0008 | 8/10 |
| F: tnbbeta vs power_spherical | 5 | 10 | 0.0150 | 0.0154 | -0.0005 | 8/10 |
| F: tnbbeta vs power_spherical | 9 | 10 | 0.0046 | 0.0040 | 0.0006 | 0/10 |
| F: tnbbeta vs power_spherical | 17 | 10 | 0.0016 | 0.0010 | 0.0006 | 0/10 |
| F: tnbbeta vs tnbbeta_init0 | 3 | 10 | 0.0359 | 0.0338 | 0.0021 | 4/10 |
| F: tnbbeta vs tnbbeta_init0 | 5 | 10 | 0.0150 | 0.0139 | 0.0011 | 4/10 |
| F: tnbbeta vs tnbbeta_init0 | 9 | 10 | 0.0046 | 0.0044 | 0.0002 | 4/10 |
| F: tnbbeta vs tnbbeta_init0 | 17 | 10 | 0.0016 | 0.0013 | 0.0002 | 3/10 |
| F: vmf vs power_spherical | 3 | 10 | 0.0751 | 0.0368 | 0.0383 | 0/10 |
| F: vmf vs power_spherical | 5 | 10 | 0.0450 | 0.0154 | 0.0296 | 0/10 |
| F: vmf vs power_spherical | 9 | 10 | 0.0207 | 0.0040 | 0.0167 | 0/10 |
| F: vmf vs power_spherical | 17 | 10 | 0.0098 | 0.0010 | 0.0089 | 0/10 |
| vmf: Lb vs L | 3 | 10 | 0.2096 | 0.2098 | -0.0002 | 6/10 |
| vmf: Lb vs L | 5 | 10 | 0.1891 | 0.1891 | 0.0000 | 5/10 |
| vmf: Lb vs L | 9 | 10 | 0.1444 | 0.1441 | 0.0003 | 5/10 |
| vmf: Lb vs L | 17 | 10 | 0.1104 | 0.1103 | 0.0001 | 4/10 |
| power_spherical: Lb vs L | 3 | 10 | 0.1318 | 0.1330 | -0.0012 | 7/10 |
| power_spherical: Lb vs L | 5 | 10 | 0.1241 | 0.1235 | 0.0006 | 4/10 |
| power_spherical: Lb vs L | 9 | 10 | 0.1037 | 0.1029 | 0.0008 | 3/10 |
| power_spherical: Lb vs L | 17 | 10 | 0.0825 | 0.0823 | 0.0002 | 4/10 |
| tnbbeta: Lb vs L | 3 | 10 | 0.0578 | 0.0633 | -0.0055 | 6/10 |
| tnbbeta: Lb vs L | 5 | 10 | 0.0284 | 0.0272 | 0.0013 | 4/10 |
| tnbbeta: Lb vs L | 9 | 10 | 0.0111 | 0.0132 | -0.0020 | 5/10 |
| tnbbeta: Lb vs L | 17 | 10 | 0.0054 | 0.0048 | 0.0005 | 8/10 |
| tnbbeta_init0: Lb vs L | 3 | 10 | 0.0555 | 0.0574 | -0.0019 | 6/10 |
| tnbbeta_init0: Lb vs L | 5 | 10 | 0.0250 | 0.0232 | 0.0018 | 3/10 |
| tnbbeta_init0: Lb vs L | 9 | 10 | 0.0104 | 0.0112 | -0.0008 | 5/10 |
| tnbbeta_init0: Lb vs L | 17 | 10 | 0.0057 | 0.0051 | 0.0005 | 6/10 |
| vmf: M vs L | 3 | 10 | 0.0755 | 0.2098 | -0.1343 | 10/10 |
| vmf: M vs L | 5 | 10 | 0.0448 | 0.1891 | -0.1443 | 10/10 |
| vmf: M vs L | 9 | 10 | 0.0208 | 0.1441 | -0.1233 | 10/10 |
| vmf: M vs L | 17 | 10 | 0.0097 | 0.1103 | -0.1006 | 10/10 |
| power_spherical: M vs L | 3 | 10 | 0.0367 | 0.1330 | -0.0963 | 10/10 |
| power_spherical: M vs L | 5 | 10 | 0.0155 | 0.1235 | -0.1080 | 10/10 |
| power_spherical: M vs L | 9 | 10 | 0.0041 | 0.1029 | -0.0988 | 10/10 |
| power_spherical: M vs L | 17 | 10 | 0.0013 | 0.0823 | -0.0810 | 10/10 |
| tnbbeta: M vs L | 3 | 10 | 0.0305 | 0.0633 | -0.0329 | 10/10 |
| tnbbeta: M vs L | 5 | 10 | 0.0167 | 0.0272 | -0.0105 | 10/10 |
| tnbbeta: M vs L | 9 | 10 | 0.0048 | 0.0132 | -0.0084 | 10/10 |
| tnbbeta: M vs L | 17 | 10 | 0.0022 | 0.0048 | -0.0026 | 9/10 |
| tnbbeta_init0: M vs L | 3 | 10 | 0.0325 | 0.0574 | -0.0249 | 10/10 |
| tnbbeta_init0: M vs L | 5 | 10 | 0.0152 | 0.0232 | -0.0080 | 9/10 |
| tnbbeta_init0: M vs L | 9 | 10 | 0.0051 | 0.0112 | -0.0061 | 10/10 |
| tnbbeta_init0: M vs L | 17 | 10 | 0.0020 | 0.0051 | -0.0031 | 9/10 |
| vmf: F vs L | 3 | 10 | 0.0751 | 0.2098 | -0.1347 | 10/10 |
| vmf: F vs L | 5 | 10 | 0.0450 | 0.1891 | -0.1441 | 10/10 |
| vmf: F vs L | 9 | 10 | 0.0207 | 0.1441 | -0.1234 | 10/10 |
| vmf: F vs L | 17 | 10 | 0.0098 | 0.1103 | -0.1005 | 10/10 |
| power_spherical: F vs L | 3 | 10 | 0.0368 | 0.1330 | -0.0962 | 10/10 |
| power_spherical: F vs L | 5 | 10 | 0.0154 | 0.1235 | -0.1080 | 10/10 |
| power_spherical: F vs L | 9 | 10 | 0.0040 | 0.1029 | -0.0989 | 10/10 |
| power_spherical: F vs L | 17 | 10 | 0.0010 | 0.0823 | -0.0813 | 10/10 |
| tnbbeta: F vs L | 3 | 10 | 0.0359 | 0.0633 | -0.0274 | 10/10 |
| tnbbeta: F vs L | 5 | 10 | 0.0150 | 0.0272 | -0.0122 | 10/10 |
| tnbbeta: F vs L | 9 | 10 | 0.0046 | 0.0132 | -0.0086 | 10/10 |
| tnbbeta: F vs L | 17 | 10 | 0.0016 | 0.0048 | -0.0033 | 9/10 |
| tnbbeta_init0: F vs L | 3 | 10 | 0.0338 | 0.0574 | -0.0236 | 10/10 |
| tnbbeta_init0: F vs L | 5 | 10 | 0.0139 | 0.0232 | -0.0093 | 10/10 |
| tnbbeta_init0: F vs L | 9 | 10 | 0.0044 | 0.0112 | -0.0068 | 10/10 |
| tnbbeta_init0: F vs L | 17 | 10 | 0.0013 | 0.0051 | -0.0038 | 10/10 |
