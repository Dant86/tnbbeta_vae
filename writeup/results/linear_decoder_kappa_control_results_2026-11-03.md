# Linear-decoder ring: tuned-kappa control for vMF / PowerSpherical, results

Date: 2026-11-03. Pre-registration (frozen before any run; deviations D1, D2 appended): `linear_decoder_kappa_control_preregistration_2026-11-03.md`.
Parent study (reused arms `tnbbeta`, `vmf`, `power_spherical`, same seeds / data): `linear_decoder_ring_results_2026-11-02.md`.
Base commit: `dd8ede0` (branch `synthetic-graph-families`) plus the uncommitted working tree (new files: `apps/synthetic/linear_decoder_kappa_control.py`,
`apps/synthetic/linear_decoder_kappa_control_plot.py`, `scripts/slurm/linear_decoder_kappa_control.sbatch`, `scripts/slurm/linear_decoder_kappa_control_aux.sbatch`,
`tests/apps/test_linear_decoder_kappa_control.py`; the parent study's files, which this imports, are also uncommitted in the working tree).

## Commands

```
sbatch --array=0-11 scripts/slurm/linear_decoder_kappa_control.sbatch                         # job 2304377 (PROCS=8)
REVERSE=1 sbatch --array=0-7 ...; OFFSET=1020 sbatch --array=0-3 ...; OFFSET=300 --array=0-5; further REVERSE/OFFSET resubmissions    # jobs 2304488, 2304588, 2305552, 2305912, 2305913 (more capacity / time-limit resubmissions)
SKIP_ARMS='_pin(10|30|100|300) ' sbatch --array=0-7 ...                                       # job 2306720 (Deviation D2), then --array=0-1: job 2308588 (cancelled once all non-skipped cells were done)
AUX=summarize sbatch scripts/slurm/linear_decoder_kappa_control_aux.sbatch                    # job 2309112 (full analysis, reproduced below)
AUX=pytest sbatch ...   # job 2309113      AUX=quality sbatch ...   # job 2309114 (ruff format/check, pyright: clean)
uv run python -m apps.synthetic.linear_decoder_kappa_control_plot --out-dir ring_recovery/kappa_control --parent-dir ring_recovery/linear_decoder --fig-dir writeup/results/figures   # login node
```
Also: smoke/unit-test jobs 2304334, 2304355, 2304372 (tests), 2304356/2304373/2304378/2304397 (quality while fixing lint). Several array tasks hit GPU-less nodes (exit 75) and resubmitted themselves; the 6 h task limit killed
in-flight runs, which were redone on resubmission (finished cells are skipped). Raw outputs: `ring_recovery/kappa_control/` (gitignored). Wall time was dominated by a saturated GPU queue and a
request from the coordinator to throttle (no new tasks beyond what was queued) so another study's jobs could run.

## What ran

New arms (17) x 12 cells x 10 seeds: all 10 seeds finished for `{vmf,power_spherical}_{pin3, k5d, k20d, H}` and `tnbbeta_H` (no divergence anywhere). The pinned arms with kappa in {10, 30, 100, 300} were **stopped
early (D2)** and have 6-10 of 10 seeds per cell (counts in the table below); they are never the selected cap. Parent arms (`tnbbeta`, `vmf`, `power_spherical`) reused, 10/10.

## Headline

* **Primary question: yes, TNBBeta's advantage over the best-tuned cap survives in 18 of 24 tests (Holm 0.05 and the 0.05-nat margin); all 24 reject by Holm** (all n = 10, 10/10 seeds, p = 1/1024; scipy and enumerated p agree in all 24). The 6 that fail do so on the **margin only**, all at sigma = 0.4: PowerSpherical at d = 3, 5, 9, 17 (mean paired differences 0.021, 0.034, 0.022, 0.0095) and vMF at d = 9 (0.0500, a hair under) and d = 17 (0.023).
  Every test has n = 10 paired seeds, TNBBeta better in 10/10, p = 1/1024 (enumerated = scipy); the margin is the only binding criterion.
  In category terms: sigma = 0.1 and 0.2 (8 cells of d x sigma): advantage survives against both families; sigma = 0.4 (4 cells): "within margin" (small: at most 0.09 nats), vMF still confirmed at d = 3, 5.
  Size of the surviving advantage (best-tuned cap minus TNBBeta, nats): sigma = 0.1: PS 0.13-0.15, vMF 0.18-0.21; sigma = 0.2: PS 0.07-0.10, vMF 0.11-0.16; sigma = 0.4: PS 0.01-0.03, vMF 0.02-0.09.
  This is almost identical to the parent's untuned result (19 of 24; the same margin failures at sigma = 0.4), i.e. **tuning kappa did not close the gap**.
* **The parent's caveat is resolved: the collapse of the caps is not a kappa optimisation failure.** (i) *Trajectories (new logging)*: from initial kappa = 5d or 20d (25-340) the median kappa falls to about 1-2 within about 50 epochs and stays there for the whole 300 (or 600) epochs
  (`figures/kappa_control_trajectories_2026-11-03.png`); the optimiser finds, quickly, that a near-uniform posterior is better for the cap's loss. At d = 17 vMF from 20d goes to the kappa floor (1.0). The head-lr 1e-2, 600-epoch variant lands on the same kappa (e.g. d = 3, sigma = 0.1: vMF 1.2, PS 1.4; d = 17: vMF 1.4, PS 1.8) and the same gap as the standard run. (ii) *Pinned kappa*:
  every pinned value of the grid {3, 10, 30, 100, 300} is **much worse** than trainable kappa: gaps 0.1-0.55 at kappa = 3, rising monotonically with kappa to 3-22 nats at 300 (`figures/kappa_control_gap_vs_kappa_2026-11-03.png`). A single cap cannot represent a latitude ring, so concentrating it only pays the
  KL to the ring posterior in the wrong geometry. The best-tuned cap (selected per cell, in-sample, from 9 pool members) is always a **trainable** arm (initial `d`, `5d` or the fast variant `_H`), never a pinned one: kappa below the grid's smallest value (3) is where caps do best, as the optimiser had found. Best-pinned kappa is at the grid edge (3) in all 24 cells (flag in table S4: the grid does not reach the optimum; trainable arms cover that region).
  (iii) The caps' `sigma_hat` stays inflated (sigma_hat/sigma 5.3 for vMF at d = 3, sigma = 0.1 in every trainable variant), as in the parent: the fitted-sigma compensation for a collapsed posterior is what the optimum looks like, not a stuck optimiser.
* **Equal budget (S2) and pooled families (S3).** TNBBeta's best of {default, `_H`} against the best cap over all 17 cap arms: TNBBeta better in 10/10 seeds in all 12 cells, mean differences 0.0098-0.156 nats (same pattern; `tnbbeta_H` improves TNBBeta by 0.0001-0.009 nats, e.g. d = 17, sigma = 0.1: 0.0134 -> 0.0066). The fast head variant helps the caps by only 0.00-0.01 nats typically (`_H` is the selected cap in most cells but its gap is within about 0.01 of the initial-`d` arm).
* **Leave-one-seed-out tuning (S1)** gives the same 18 / 24 confirmed with the same failing cells (differences change by at most 0.006 nats), so the in-sample selection does not drive the result.

## Caveats

* Everything is conditional on the parent's setup (decoder L, 20k points, 300-600 epochs, 10 seeds, one test set shared by seeds, ELBO gap to the true marginal). The test ELBO gap separates families only through KL(q || model posterior) and `sigma_hat` (parent D1(c)).
* "Best-tuned" is within the 9-member pool per family and a pinned grid that starts at kappa = 3; no pinned kappa in (1, 3) was tried (trainable arms sit at 1-2 and are in the pool). A pinned-kappa cap at the trainable optimum (about 1.2-1.8) would be near the trainable arms by construction.
* Deviation D2: pinned arms with kappa >= 10 are partial (6-10 seeds); chosen with partial results in view, justified by gaps 5x-100x larger than the competitors in every cell; no analysis conclusion depends on them. The first array after D2 also included extra duplicate work from overlapping shards (harmless; the cells are deterministic).
* The sigma = 0.4 failures are margin failures, not evidence of no advantage: TNBBeta is better in 10/10 seeds there too, by 0.01-0.09 nats.
* This is a **negative result for the optimisation-artefact hypothesis**, not evidence about why TNBBeta wins (the ring statistic and decomposition analyses in the parent apply); no claim beyond decoder L is made.
* Parent-arm trajectories were not logged (the parent code did not), so the "init d" curve is missing from the trajectory figure; the 5d / 20d / `_H` variants bracket it.

## Deviations

D1, D2 as in the pre-registration. D3 (execution): the figure files carry stamp 2026-11-03; the sbatch script gained `REVERSE`, `OFFSET` and `SKIP_ARMS` environment options during the run (cell order only; no run settings changed). Unit tests: 8 new tests (`tests/apps/test_linear_decoder_kappa_control.py`), ruff format/check and pyright clean; full suite (job 2309113): 1181 passed, 1 skipped.

## Figures

`writeup/results/figures/kappa_control_gap_vs_kappa_2026-11-03.png`, `kappa_control_trajectories_2026-11-03.png`, `kappa_control_paired_difference_2026-11-03.png`.

## Full generated analysis (job 2309112)


## Per-arm test ELBO gap (nats)

| d | sigma | arm | valid | gap mean +- sd | min / max | IWAE gap | sigma_hat / sigma | kappa median | KL to model posterior |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.1 | tnbbeta | 10/10 | 0.0705 +- 0.0232 | 0.0479 / 0.1082 | 0.0117 | 1.275 | nan | 0.0668 |
| 3 | 0.1 | vmf | 10/10 | 0.2793 +- 0.0004 | 0.2787 / 0.2799 | 0.1606 | 5.285 | 1.15 | 0.1192 |
| 3 | 0.1 | power_spherical | 10/10 | 0.1990 +- 0.0015 | 0.1977 / 0.2020 | 0.1381 | 4.799 | 1.29 | 0.0615 |
| 3 | 0.1 | vmf_pin3 | 10/10 | 0.5425 +- 0.0011 | 0.5412 / 0.5445 | 0.1470 | 4.048 | 3 | 0.4389 |
| 3 | 0.1 | vmf_pin10 | 10/10 | 1.2017 +- 0.0010 | 1.2003 / 1.2035 | 0.6192 | 2.387 | 10 | 1.1670 |
| 3 | 0.1 | vmf_pin30 | 10/10 | 1.8075 +- 0.0013 | 1.8053 / 1.8097 | 1.2201 | 1.459 | 30 | 1.8027 |
| 3 | 0.1 | vmf_pin100 | 10/10 | 2.4475 +- 0.0020 | 2.4455 / 2.4517 | 1.8667 | 0.827 | 100 | 2.4434 |
| 3 | 0.1 | vmf_pin300 | 10/10 | 3.0516 +- 0.0071 | 3.0459 / 3.0680 | 2.4732 | 0.499 | 300 | 3.0059 |
| 3 | 0.1 | power_spherical_pin3 | 10/10 | 0.4202 +- 0.0005 | 0.4191 / 0.4208 | 0.1656 | 4.262 | 3 | 0.3070 |
| 3 | 0.1 | power_spherical_pin10 | 10/10 | 0.9046 +- 0.0006 | 0.9037 / 0.9057 | 0.4226 | 2.942 | 10 | 0.8481 |
| 3 | 0.1 | power_spherical_pin30 | 10/10 | 1.4662 +- 0.0009 | 1.4648 / 1.4678 | 0.9053 | 1.944 | 30 | 1.4473 |
| 3 | 0.1 | power_spherical_pin100 | 10/10 | 2.0906 +- 0.0012 | 2.0889 / 2.0927 | 1.5183 | 1.137 | 100 | 2.0908 |
| 3 | 0.1 | power_spherical_pin300 | 10/10 | 2.6703 +- 0.0030 | 2.6668 / 2.6746 | 2.1008 | 0.682 | 300 | 2.6563 |
| 3 | 0.1 | vmf_k5d | 10/10 | 0.2794 +- 0.0004 | 0.2788 / 0.2800 | 0.1606 | 5.287 | 1.15 | 0.1192 |
| 3 | 0.1 | vmf_k20d | 10/10 | 0.2846 +- 0.0071 | 0.2799 / 0.3041 | 0.1617 | 5.319 | 1.14 | 0.1229 |
| 3 | 0.1 | power_spherical_k5d | 10/10 | 0.1994 +- 0.0013 | 0.1976 / 0.2016 | 0.1383 | 4.804 | 1.25 | 0.0617 |
| 3 | 0.1 | power_spherical_k20d | 10/10 | 0.2031 +- 0.0050 | 0.1976 / 0.2157 | 0.1386 | 4.814 | 1.28 | 0.0649 |
| 3 | 0.1 | vmf_H | 10/10 | 0.2790 +- 0.0005 | 0.2782 / 0.2799 | 0.1607 | 5.284 | 1.17 | 0.1190 |
| 3 | 0.1 | power_spherical_H | 10/10 | 0.1994 +- 0.0020 | 0.1975 / 0.2047 | 0.1382 | 4.809 | 1.29 | 0.0614 |
| 3 | 0.1 | tnbbeta_H | 10/10 | 0.0617 +- 0.0233 | 0.0425 / 0.1232 | 0.0100 | 1.149 | nan | 0.0600 |
| 5 | 0.1 | tnbbeta | 10/10 | 0.0325 +- 0.0051 | 0.0250 / 0.0416 | 0.0060 | 1.544 | nan | 0.0282 |
| 5 | 0.1 | vmf | 10/10 | 0.2375 +- 0.0015 | 0.2350 / 0.2396 | 0.1399 | 4.839 | 1 | 0.0989 |
| 5 | 0.1 | power_spherical | 10/10 | 0.1695 +- 0.0010 | 0.1687 / 0.1720 | 0.1067 | 4.263 | 1.26 | 0.0635 |
| 5 | 0.1 | vmf_pin3 | 10/10 | 0.5164 +- 0.0007 | 0.5156 / 0.5175 | 0.1019 | 3.874 | 3 | 0.4307 |
| 5 | 0.1 | vmf_pin10 | 7/7 | 1.8268 +- 0.0006 | 1.8263 / 1.8279 | 0.6537 | 2.479 | 10 | 1.7995 |
| 5 | 0.1 | vmf_pin30 | 7/7 | 3.4123 +- 0.0009 | 3.4113 / 3.4140 | 2.0739 | 1.556 | 30 | 3.4080 |
| 5 | 0.1 | vmf_pin100 | 7/7 | 5.2032 +- 0.0010 | 5.2017 / 5.2043 | 3.8818 | 0.881 | 100 | 5.2025 |
| 5 | 0.1 | vmf_pin300 | 4/4 | 6.8590 +- 0.0094 | 6.8539 / 6.8732 | 5.5245 | 0.515 | 300 | 6.8478 |
| 5 | 0.1 | power_spherical_pin3 | 10/10 | 0.3828 +- 0.0004 | 0.3824 / 0.3834 | 0.1031 | 3.936 | 3 | 0.2940 |
| 5 | 0.1 | power_spherical_pin10 | 7/7 | 1.2215 +- 0.0016 | 1.2200 / 1.2242 | 0.3785 | 2.901 | 10 | 1.1791 |
| 5 | 0.1 | power_spherical_pin30 | 8/8 | 2.5333 +- 0.0007 | 2.5323 / 2.5343 | 1.3509 | 2.022 | 30 | 2.5192 |
| 5 | 0.1 | power_spherical_pin100 | 6/6 | 4.2135 +- 0.0035 | 4.2106 / 4.2200 | 2.9018 | 1.209 | 100 | 4.2131 |
| 5 | 0.1 | power_spherical_pin300 | 6/6 | 5.8281 +- 0.0014 | 5.8256 / 5.8299 | 4.4755 | 0.720 | 300 | 5.8249 |
| 5 | 0.1 | vmf_k5d | 10/10 | 0.2368 +- 0.0021 | 0.2336 / 0.2396 | 0.1400 | 4.843 | 1 | 0.0980 |
| 5 | 0.1 | vmf_k20d | 10/10 | 0.2371 +- 0.0022 | 0.2347 / 0.2403 | 0.1409 | 4.863 | 1 | 0.0972 |
| 5 | 0.1 | power_spherical_k5d | 10/10 | 0.1706 +- 0.0024 | 0.1680 / 0.1765 | 0.1066 | 4.267 | 1.26 | 0.0644 |
| 5 | 0.1 | power_spherical_k20d | 10/10 | 0.1703 +- 0.0013 | 0.1689 / 0.1725 | 0.1067 | 4.272 | 1.34 | 0.0639 |
| 5 | 0.1 | vmf_H | 10/10 | 0.2369 +- 0.0012 | 0.2341 / 0.2380 | 0.1394 | 4.831 | 1 | 0.0988 |
| 5 | 0.1 | power_spherical_H | 10/10 | 0.1692 +- 0.0009 | 0.1680 / 0.1712 | 0.1066 | 4.265 | 1.29 | 0.0632 |
| 5 | 0.1 | tnbbeta_H | 10/10 | 0.0294 +- 0.0210 | 0.0136 / 0.0652 | 0.0036 | 1.162 | nan | 0.0276 |
| 9 | 0.1 | tnbbeta | 10/10 | 0.0190 +- 0.0075 | 0.0118 / 0.0355 | 0.0040 | 1.523 | nan | 0.0149 |
| 9 | 0.1 | vmf | 10/10 | 0.2043 +- 0.0011 | 0.2027 / 0.2063 | 0.1114 | 3.731 | 1.08 | 0.0935 |
| 9 | 0.1 | power_spherical | 10/10 | 0.1589 +- 0.0008 | 0.1578 / 0.1604 | 0.0904 | 3.435 | 1.42 | 0.0698 |
| 9 | 0.1 | vmf_pin3 | 10/10 | 0.4109 +- 0.0006 | 0.4100 / 0.4122 | 0.0915 | 3.376 | 3 | 0.3259 |
| 9 | 0.1 | vmf_pin10 | 8/8 | 2.1279 +- 0.0009 | 2.1266 / 2.1290 | 0.4889 | 2.371 | 10 | 2.0997 |
| 9 | 0.1 | vmf_pin30 | 8/8 | 5.2603 +- 0.0020 | 5.2582 / 5.2643 | 2.9324 | 1.584 | 30 | 5.2556 |
| 9 | 0.1 | vmf_pin100 | 8/8 | 9.2412 +- 0.0010 | 9.2400 / 9.2427 | 6.6829 | 0.917 | 100 | 9.2411 |
| 9 | 0.1 | vmf_pin300 | 5/5 | 13.0240 +- 0.0015 | 13.0226 / 13.0261 | 10.4689 | 0.539 | 300 | 13.0214 |
| 9 | 0.1 | power_spherical_pin3 | 10/10 | 0.3184 +- 0.0004 | 0.3178 / 0.3189 | 0.0907 | 3.354 | 3 | 0.2349 |
| 9 | 0.1 | power_spherical_pin10 | 7/7 | 1.3261 +- 0.0009 | 1.3245 / 1.3270 | 0.3020 | 2.598 | 10 | 1.2875 |
| 9 | 0.1 | power_spherical_pin30 | 9/9 | 3.5818 +- 0.0008 | 3.5807 / 3.5833 | 1.6902 | 1.967 | 30 | 3.5681 |
| 9 | 0.1 | power_spherical_pin100 | 7/7 | 7.0709 +- 0.0011 | 7.0696 / 7.0727 | 4.7230 | 1.238 | 100 | 7.0702 |
| 9 | 0.1 | power_spherical_pin300 | 8/8 | 10.6856 +- 0.0026 | 10.6835 / 10.6911 | 8.1696 | 0.748 | 300 | 10.6847 |
| 9 | 0.1 | vmf_k5d | 10/10 | 0.2041 +- 0.0008 | 0.2025 / 0.2055 | 0.1123 | 3.742 | 1.09 | 0.0926 |
| 9 | 0.1 | vmf_k20d | 10/10 | 0.2066 +- 0.0018 | 0.2048 / 0.2107 | 0.1130 | 3.750 | 1.07 | 0.0944 |
| 9 | 0.1 | power_spherical_k5d | 10/10 | 0.1599 +- 0.0013 | 0.1587 / 0.1633 | 0.0904 | 3.438 | 1.38 | 0.0706 |
| 9 | 0.1 | power_spherical_k20d | 10/10 | 0.1614 +- 0.0021 | 0.1593 / 0.1658 | 0.0911 | 3.446 | 1.46 | 0.0715 |
| 9 | 0.1 | vmf_H | 10/10 | 0.2036 +- 0.0008 | 0.2023 / 0.2048 | 0.1113 | 3.735 | 1.11 | 0.0926 |
| 9 | 0.1 | power_spherical_H | 10/10 | 0.1587 +- 0.0008 | 0.1578 / 0.1599 | 0.0906 | 3.436 | 1.44 | 0.0695 |
| 9 | 0.1 | tnbbeta_H | 10/10 | 0.0192 +- 0.0136 | 0.0054 / 0.0442 | 0.0019 | 1.302 | nan | 0.0179 |
| 17 | 0.1 | tnbbeta | 10/10 | 0.0134 +- 0.0076 | 0.0059 / 0.0296 | 0.0037 | 1.249 | nan | 0.0101 |
| 17 | 0.1 | vmf | 10/10 | 0.1954 +- 0.0010 | 0.1945 / 0.1973 | 0.1063 | 2.805 | 1.41 | 0.0885 |
| 17 | 0.1 | power_spherical | 10/10 | 0.1627 +- 0.0008 | 0.1614 / 0.1642 | 0.0914 | 2.641 | 1.74 | 0.0725 |
| 17 | 0.1 | vmf_pin3 | 10/10 | 0.3122 +- 0.0005 | 0.3115 / 0.3131 | 0.1061 | 2.759 | 3 | 0.2101 |
| 17 | 0.1 | vmf_pin10 | 7/7 | 1.8651 +- 0.0013 | 1.8634 / 1.8668 | 0.2443 | 2.081 | 10 | 1.8233 |
| 17 | 0.1 | vmf_pin30 | 5/5 | 6.7482 +- 0.0010 | 6.7464 / 6.7489 | 3.2299 | 1.526 | 30 | 6.7371 |
| 17 | 0.1 | vmf_pin100 | 7/7 | 14.5498 +- 0.0014 | 14.5487 / 14.5521 | 10.3878 | 0.925 | 100 | 14.5501 |
| 17 | 0.1 | vmf_pin300 | 7/7 | 22.4432 +- 0.0024 | 22.4409 / 22.4469 | 18.1772 | 0.549 | 300 | 22.4418 |
| 17 | 0.1 | power_spherical_pin3 | 10/10 | 0.2635 +- 0.0005 | 0.2629 / 0.2644 | 0.1013 | 2.721 | 3 | 0.1652 |
| 17 | 0.1 | power_spherical_pin10 | 9/9 | 1.1758 +- 0.0004 | 1.1749 / 1.1763 | 0.1607 | 2.152 | 10 | 1.1287 |
| 17 | 0.1 | power_spherical_pin30 | 9/9 | 4.2421 +- 0.0012 | 4.2404 / 4.2435 | 1.6594 | 1.771 | 30 | 4.2198 |
| 17 | 0.1 | power_spherical_pin100 | 9/9 | 10.4006 +- 0.0009 | 10.3996 / 10.4016 | 6.6607 | 1.210 | 100 | 10.3982 |
| 17 | 0.1 | power_spherical_pin300 | 7/7 | 17.6242 +- 0.0019 | 17.6216 / 17.6279 | 13.4141 | 0.755 | 300 | 17.6242 |
| 17 | 0.1 | vmf_k5d | 10/10 | 0.1960 +- 0.0017 | 0.1944 / 0.1994 | 0.1072 | 2.809 | 1.36 | 0.0887 |
| 17 | 0.1 | vmf_k20d | 10/10 | 0.2417 +- 0.0104 | 0.2266 / 0.2516 | 0.1496 | 3.166 | 1 | 0.0949 |
| 17 | 0.1 | power_spherical_k5d | 10/10 | 0.1640 +- 0.0008 | 0.1631 / 0.1655 | 0.0913 | 2.645 | 1.71 | 0.0734 |
| 17 | 0.1 | power_spherical_k20d | 10/10 | 0.1852 +- 0.0376 | 0.1638 / 0.2602 | 0.1046 | 2.764 | 1.73 | 0.0818 |
| 17 | 0.1 | vmf_H | 10/10 | 0.1947 +- 0.0008 | 0.1939 / 0.1961 | 0.1063 | 2.802 | 1.37 | 0.0881 |
| 17 | 0.1 | power_spherical_H | 10/10 | 0.1626 +- 0.0009 | 0.1612 / 0.1637 | 0.0917 | 2.644 | 1.74 | 0.0721 |
| 17 | 0.1 | tnbbeta_H | 10/10 | 0.0066 +- 0.0037 | 0.0032 / 0.0133 | 0.0015 | 1.134 | nan | 0.0052 |
| 3 | 0.2 | tnbbeta | 10/10 | 0.0633 +- 0.0210 | 0.0528 / 0.1218 | 0.0071 | 1.175 | nan | 0.0627 |
| 3 | 0.2 | vmf | 10/10 | 0.2098 +- 0.0004 | 0.2093 / 0.2104 | 0.0961 | 2.689 | 1.1 | 0.1142 |
| 3 | 0.2 | power_spherical | 10/10 | 0.1330 +- 0.0024 | 0.1310 / 0.1391 | 0.0767 | 2.458 | 1.2 | 0.0566 |
| 3 | 0.2 | vmf_pin3 | 10/10 | 0.4810 +- 0.0006 | 0.4801 / 0.4820 | 0.0932 | 2.084 | 3 | 0.4334 |
| 3 | 0.2 | vmf_pin10 | 10/10 | 1.1440 +- 0.0009 | 1.1430 / 1.1455 | 0.5852 | 1.227 | 10 | 1.1430 |
| 3 | 0.2 | vmf_pin30 | 10/10 | 1.7733 +- 0.0013 | 1.7713 / 1.7752 | 1.2011 | 0.763 | 30 | 1.7585 |
| 3 | 0.2 | vmf_pin100 | 10/10 | 2.5035 +- 0.0015 | 2.5017 / 2.5059 | 1.9275 | 0.467 | 100 | 2.3944 |
| 3 | 0.2 | vmf_pin300 | 10/10 | 3.2855 +- 0.0055 | 3.2803 / 3.2980 | 2.6688 | 0.329 | 300 | 3.0142 |
| 3 | 0.2 | power_spherical_pin3 | 10/10 | 0.3625 +- 0.0004 | 0.3619 / 0.3632 | 0.1087 | 2.200 | 3 | 0.3063 |
| 3 | 0.2 | power_spherical_pin10 | 10/10 | 0.8450 +- 0.0004 | 0.8444 / 0.8458 | 0.3744 | 1.513 | 10 | 0.8333 |
| 3 | 0.2 | power_spherical_pin30 | 10/10 | 1.4132 +- 0.0009 | 1.4118 / 1.4146 | 0.8753 | 1.003 | 30 | 1.4133 |
| 3 | 0.2 | power_spherical_pin100 | 10/10 | 2.0844 +- 0.0011 | 2.0826 / 2.0863 | 1.5248 | 0.609 | 100 | 2.0397 |
| 3 | 0.2 | power_spherical_pin300 | 10/10 | 2.7804 +- 0.0018 | 2.7779 / 2.7834 | 2.2029 | 0.403 | 300 | 2.6169 |
| 3 | 0.2 | vmf_k5d | 10/10 | 0.2097 +- 0.0004 | 0.2092 / 0.2102 | 0.0962 | 2.693 | 1.13 | 0.1137 |
| 3 | 0.2 | vmf_k20d | 10/10 | 0.2144 +- 0.0057 | 0.2093 / 0.2291 | 0.0970 | 2.705 | 1.11 | 0.1174 |
| 3 | 0.2 | power_spherical_k5d | 10/10 | 0.1325 +- 0.0021 | 0.1299 / 0.1378 | 0.0767 | 2.456 | 1.23 | 0.0562 |
| 3 | 0.2 | power_spherical_k20d | 10/10 | 0.1360 +- 0.0028 | 0.1325 / 0.1399 | 0.0766 | 2.461 | 1.24 | 0.0594 |
| 3 | 0.2 | vmf_H | 10/10 | 0.2091 +- 0.0003 | 0.2087 / 0.2096 | 0.0962 | 2.689 | 1.14 | 0.1135 |
| 3 | 0.2 | power_spherical_H | 10/10 | 0.1322 +- 0.0013 | 0.1306 / 0.1342 | 0.0769 | 2.461 | 1.25 | 0.0555 |
| 3 | 0.2 | tnbbeta_H | 10/10 | 0.0547 +- 0.0038 | 0.0509 / 0.0641 | 0.0063 | 1.073 | nan | 0.0551 |
| 5 | 0.2 | tnbbeta | 10/10 | 0.0272 +- 0.0083 | 0.0195 / 0.0454 | 0.0032 | 1.138 | nan | 0.0253 |
| 5 | 0.2 | vmf | 10/10 | 0.1891 +- 0.0015 | 0.1868 / 0.1915 | 0.0996 | 2.488 | 1 | 0.0909 |
| 5 | 0.2 | power_spherical | 10/10 | 0.1235 +- 0.0005 | 0.1228 / 0.1242 | 0.0684 | 2.187 | 1.25 | 0.0557 |
| 5 | 0.2 | vmf_pin3 | 10/10 | 0.4679 +- 0.0004 | 0.4674 / 0.4688 | 0.0683 | 1.998 | 3 | 0.4171 |
| 5 | 0.2 | vmf_pin10 | 7/7 | 1.7534 +- 0.0011 | 1.7519 / 1.7549 | 0.6080 | 1.249 | 10 | 1.7488 |
| 5 | 0.2 | vmf_pin30 | 7/7 | 3.3365 +- 0.0007 | 3.3355 / 3.3375 | 2.0073 | 0.782 | 30 | 3.3331 |
| 5 | 0.2 | vmf_pin100 | 7/7 | 5.1468 +- 0.0008 | 5.1455 / 5.1480 | 3.8240 | 0.451 | 100 | 5.1075 |
| 5 | 0.2 | vmf_pin300 | 5/5 | 6.8565 +- 0.0059 | 6.8523 / 6.8666 | 5.5188 | 0.279 | 300 | 6.7418 |
| 5 | 0.2 | power_spherical_pin3 | 10/10 | 0.3392 +- 0.0005 | 0.3385 / 0.3400 | 0.0688 | 2.041 | 3 | 0.2847 |
| 5 | 0.2 | power_spherical_pin10 | 7/7 | 1.1562 +- 0.0015 | 1.1548 / 1.1586 | 0.3439 | 1.472 | 10 | 1.1420 |
| 5 | 0.2 | power_spherical_pin30 | 9/9 | 2.4560 +- 0.0009 | 2.4547 / 2.4571 | 1.2886 | 1.015 | 30 | 2.4559 |
| 5 | 0.2 | power_spherical_pin100 | 6/6 | 4.1414 +- 0.0010 | 4.1403 / 4.1431 | 2.8399 | 0.611 | 100 | 4.1268 |
| 5 | 0.2 | power_spherical_pin300 | 7/7 | 5.7870 +- 0.0027 | 5.7834 / 5.7920 | 4.4362 | 0.374 | 300 | 5.7248 |
| 5 | 0.2 | vmf_k5d | 10/10 | 0.1890 +- 0.0021 | 0.1858 / 0.1927 | 0.0997 | 2.489 | 1 | 0.0907 |
| 5 | 0.2 | vmf_k20d | 10/10 | 0.1938 +- 0.0078 | 0.1863 / 0.2140 | 0.1020 | 2.512 | 1 | 0.0929 |
| 5 | 0.2 | power_spherical_k5d | 10/10 | 0.1251 +- 0.0019 | 0.1234 / 0.1300 | 0.0683 | 2.188 | 1.26 | 0.0571 |
| 5 | 0.2 | power_spherical_k20d | 10/10 | 0.1259 +- 0.0026 | 0.1236 / 0.1325 | 0.0687 | 2.192 | 1.31 | 0.0576 |
| 5 | 0.2 | vmf_H | 10/10 | 0.1888 +- 0.0016 | 0.1870 / 0.1922 | 0.0990 | 2.487 | 1 | 0.0907 |
| 5 | 0.2 | power_spherical_H | 10/10 | 0.1238 +- 0.0008 | 0.1229 / 0.1251 | 0.0683 | 2.187 | 1.24 | 0.0560 |
| 5 | 0.2 | tnbbeta_H | 10/10 | 0.0241 +- 0.0092 | 0.0183 / 0.0495 | 0.0028 | 1.100 | nan | 0.0228 |
| 9 | 0.2 | tnbbeta | 10/10 | 0.0132 +- 0.0082 | 0.0073 / 0.0358 | 0.0007 | 1.096 | nan | 0.0127 |
| 9 | 0.2 | vmf | 10/10 | 0.1441 +- 0.0012 | 0.1427 / 0.1469 | 0.0652 | 1.955 | 1.16 | 0.0790 |
| 9 | 0.2 | power_spherical | 10/10 | 0.1029 +- 0.0007 | 0.1020 / 0.1039 | 0.0474 | 1.795 | 1.51 | 0.0566 |
| 9 | 0.2 | vmf_pin3 | 10/10 | 0.3458 +- 0.0008 | 0.3444 / 0.3467 | 0.0499 | 1.777 | 3 | 0.3014 |
| 9 | 0.2 | vmf_pin10 | 6/6 | 2.0163 +- 0.0009 | 2.0153 / 2.0178 | 0.4369 | 1.191 | 10 | 2.0139 |
| 9 | 0.2 | vmf_pin30 | 7/7 | 5.1319 +- 0.0013 | 5.1306 / 5.1343 | 2.8196 | 0.783 | 30 | 5.1262 |
| 9 | 0.2 | vmf_pin100 | 7/7 | 9.1155 +- 0.0008 | 9.1144 / 9.1165 | 6.5591 | 0.454 | 100 | 9.0804 |
| 9 | 0.2 | vmf_pin300 | 5/5 | 12.9076 +- 0.0027 | 12.9045 / 12.9107 | 10.3527 | 0.270 | 300 | 12.8367 |
| 9 | 0.2 | power_spherical_pin3 | 10/10 | 0.2589 +- 0.0003 | 0.2582 / 0.2593 | 0.0524 | 1.777 | 3 | 0.2146 |
| 9 | 0.2 | power_spherical_pin10 | 7/7 | 1.2272 +- 0.0007 | 1.2257 / 1.2279 | 0.2503 | 1.319 | 10 | 1.2198 |
| 9 | 0.2 | power_spherical_pin30 | 9/9 | 3.4576 +- 0.0007 | 3.4567 / 3.4589 | 1.5977 | 0.974 | 30 | 3.4574 |
| 9 | 0.2 | power_spherical_pin100 | 6/6 | 6.9430 +- 0.0016 | 6.9415 / 6.9457 | 4.5997 | 0.612 | 100 | 6.9257 |
| 9 | 0.2 | power_spherical_pin300 | 6/6 | 10.5624 +- 0.0032 | 10.5596 / 10.5671 | 8.0446 | 0.372 | 300 | 10.5143 |
| 9 | 0.2 | vmf_k5d | 10/10 | 0.1445 +- 0.0009 | 0.1431 / 0.1467 | 0.0657 | 1.958 | 1.14 | 0.0790 |
| 9 | 0.2 | vmf_k20d | 10/10 | 0.1467 +- 0.0029 | 0.1437 / 0.1515 | 0.0662 | 1.963 | 1.14 | 0.0807 |
| 9 | 0.2 | power_spherical_k5d | 10/10 | 0.1039 +- 0.0011 | 0.1023 / 0.1054 | 0.0473 | 1.796 | 1.48 | 0.0574 |
| 9 | 0.2 | power_spherical_k20d | 10/10 | 0.1063 +- 0.0045 | 0.1018 / 0.1168 | 0.0476 | 1.800 | 1.56 | 0.0594 |
| 9 | 0.2 | vmf_H | 10/10 | 0.1438 +- 0.0009 | 0.1424 / 0.1453 | 0.0655 | 1.954 | 1.17 | 0.0788 |
| 9 | 0.2 | power_spherical_H | 10/10 | 0.1023 +- 0.0005 | 0.1017 / 0.1030 | 0.0473 | 1.794 | 1.5 | 0.0560 |
| 9 | 0.2 | tnbbeta_H | 10/10 | 0.0093 +- 0.0028 | 0.0052 / 0.0123 | 0.0008 | 1.078 | nan | 0.0090 |
| 17 | 0.2 | tnbbeta | 10/10 | 0.0048 +- 0.0029 | 0.0025 / 0.0109 | 0.0007 | 1.028 | nan | 0.0045 |
| 17 | 0.2 | vmf | 10/10 | 0.1103 +- 0.0011 | 0.1094 / 0.1129 | 0.0439 | 1.526 | 1.54 | 0.0663 |
| 17 | 0.2 | power_spherical | 10/10 | 0.0823 +- 0.0012 | 0.0809 / 0.0851 | 0.0330 | 1.439 | 1.89 | 0.0502 |
| 17 | 0.2 | vmf_pin3 | 10/10 | 0.2239 +- 0.0003 | 0.2232 / 0.2244 | 0.0469 | 1.522 | 3 | 0.1805 |
| 17 | 0.2 | vmf_pin10 | 7/7 | 1.6950 +- 0.0011 | 1.6933 / 1.6960 | 0.1828 | 1.063 | 10 | 1.6938 |
| 17 | 0.2 | vmf_pin30 | 5/5 | 6.5401 +- 0.0007 | 6.5392 / 6.5408 | 3.0561 | 0.751 | 30 | 6.5309 |
| 17 | 0.2 | vmf_pin100 | 8/8 | 14.3425 +- 0.0013 | 14.3413 / 14.3452 | 10.1786 | 0.455 | 100 | 14.2944 |
| 17 | 0.2 | vmf_pin300 | 7/7 | 22.2374 +- 0.0031 | 22.2349 / 22.2423 | 17.9692 | 0.270 | 300 | 22.1554 |
| 17 | 0.2 | power_spherical_pin3 | 10/10 | 0.1808 +- 0.0003 | 0.1804 / 0.1813 | 0.0444 | 1.513 | 3 | 0.1388 |
| 17 | 0.2 | power_spherical_pin10 | 9/9 | 1.0271 +- 0.0006 | 1.0262 / 1.0284 | 0.1053 | 1.123 | 10 | 1.0237 |
| 17 | 0.2 | power_spherical_pin30 | 10/10 | 4.0357 +- 0.0011 | 4.0341 / 4.0374 | 1.5158 | 0.875 | 30 | 4.0339 |
| 17 | 0.2 | power_spherical_pin100 | 9/9 | 10.1917 +- 0.0007 | 10.1909 / 10.1928 | 6.4593 | 0.594 | 100 | 10.1653 |
| 17 | 0.2 | power_spherical_pin300 | 7/7 | 17.4179 +- 0.0033 | 17.4156 / 17.4251 | 13.2087 | 0.372 | 300 | 17.3547 |
| 17 | 0.2 | vmf_k5d | 10/10 | 0.1113 +- 0.0011 | 0.1104 / 0.1139 | 0.0441 | 1.529 | 1.44 | 0.0670 |
| 17 | 0.2 | vmf_k20d | 10/10 | 0.1512 +- 0.0077 | 0.1462 / 0.1724 | 0.0738 | 1.711 | 1 | 0.0775 |
| 17 | 0.2 | power_spherical_k5d | 10/10 | 0.0832 +- 0.0012 | 0.0813 / 0.0844 | 0.0330 | 1.441 | 1.78 | 0.0509 |
| 17 | 0.2 | power_spherical_k20d | 10/10 | 0.0857 +- 0.0032 | 0.0823 / 0.0932 | 0.0340 | 1.444 | 1.98 | 0.0530 |
| 17 | 0.2 | vmf_H | 10/10 | 0.1098 +- 0.0007 | 0.1091 / 0.1112 | 0.0434 | 1.525 | 1.48 | 0.0660 |
| 17 | 0.2 | power_spherical_H | 10/10 | 0.0823 +- 0.0007 | 0.0816 / 0.0841 | 0.0330 | 1.440 | 1.84 | 0.0501 |
| 17 | 0.2 | tnbbeta_H | 10/10 | 0.0059 +- 0.0056 | 0.0022 / 0.0209 | 0.0005 | 1.027 | nan | 0.0055 |
| 3 | 0.4 | tnbbeta | 10/10 | 0.0307 +- 0.0022 | 0.0285 / 0.0356 | 0.0028 | 1.159 | nan | 0.0282 |
| 3 | 0.4 | vmf | 10/10 | 0.1183 +- 0.0004 | 0.1177 / 0.1189 | 0.0269 | 1.455 | 1.06 | 0.0922 |
| 3 | 0.4 | power_spherical | 10/10 | 0.0525 +- 0.0010 | 0.0511 / 0.0540 | 0.0171 | 1.361 | 1.11 | 0.0358 |
| 3 | 0.4 | vmf_pin3 | 10/10 | 0.4103 +- 0.0005 | 0.4098 / 0.4112 | 0.0429 | 1.158 | 3 | 0.4079 |
| 3 | 0.4 | vmf_pin10 | 10/10 | 1.1158 +- 0.0008 | 1.1147 / 1.1177 | 0.5884 | 0.703 | 10 | 1.0766 |
| 3 | 0.4 | vmf_pin30 | 10/10 | 1.8756 +- 0.0011 | 1.8739 / 1.8773 | 1.2939 | 0.493 | 30 | 1.7094 |
| 3 | 0.4 | vmf_pin100 | 10/10 | 2.8513 +- 0.0017 | 2.8488 / 2.8540 | 2.1538 | 0.384 | 100 | 2.5133 |
| 3 | 0.4 | vmf_pin300 | 10/10 | 3.8571 +- 0.0032 | 3.8527 / 3.8612 | 3.0355 | 0.346 | 300 | 3.4155 |
| 3 | 0.4 | power_spherical_pin3 | 10/10 | 0.2963 +- 0.0005 | 0.2957 / 0.2971 | 0.0532 | 1.231 | 3 | 0.2899 |
| 3 | 0.4 | power_spherical_pin10 | 10/10 | 0.7900 +- 0.0004 | 0.7895 / 0.7911 | 0.3514 | 0.850 | 10 | 0.7803 |
| 3 | 0.4 | power_spherical_pin30 | 10/10 | 1.4277 +- 0.0005 | 1.4273 / 1.4291 | 0.9091 | 0.597 | 30 | 1.3432 |
| 3 | 0.4 | power_spherical_pin100 | 10/10 | 2.2859 +- 0.0019 | 2.2834 / 2.2904 | 1.6729 | 0.433 | 100 | 2.0413 |
| 3 | 0.4 | power_spherical_pin300 | 10/10 | 3.2186 +- 0.0021 | 3.2160 / 3.2219 | 2.4707 | 0.365 | 300 | 2.8337 |
| 3 | 0.4 | vmf_k5d | 10/10 | 0.1188 +- 0.0009 | 0.1178 / 0.1206 | 0.0273 | 1.457 | 1.06 | 0.0924 |
| 3 | 0.4 | vmf_k20d | 10/10 | 0.1236 +- 0.0075 | 0.1182 / 0.1427 | 0.0274 | 1.463 | 1.05 | 0.0966 |
| 3 | 0.4 | power_spherical_k5d | 10/10 | 0.0522 +- 0.0008 | 0.0509 / 0.0535 | 0.0169 | 1.362 | 1.12 | 0.0354 |
| 3 | 0.4 | power_spherical_k20d | 10/10 | 0.0565 +- 0.0022 | 0.0537 / 0.0594 | 0.0176 | 1.365 | 1.09 | 0.0395 |
| 3 | 0.4 | vmf_H | 10/10 | 0.1181 +- 0.0003 | 0.1177 / 0.1186 | 0.0273 | 1.457 | 1.05 | 0.0917 |
| 3 | 0.4 | power_spherical_H | 10/10 | 0.0520 +- 0.0009 | 0.0505 / 0.0534 | 0.0168 | 1.361 | 1.16 | 0.0353 |
| 3 | 0.4 | tnbbeta_H | 10/10 | 0.0305 +- 0.0018 | 0.0284 / 0.0345 | 0.0036 | 1.167 | nan | 0.0276 |
| 5 | 0.4 | tnbbeta | 10/10 | 0.0107 +- 0.0022 | 0.0088 / 0.0157 | 0.0009 | 1.055 | nan | 0.0100 |
| 5 | 0.4 | vmf | 10/10 | 0.0977 +- 0.0012 | 0.0962 / 0.0994 | 0.0310 | 1.399 | 1.01 | 0.0675 |
| 5 | 0.4 | power_spherical | 10/10 | 0.0443 +- 0.0009 | 0.0426 / 0.0456 | 0.0127 | 1.245 | 1.14 | 0.0319 |
| 5 | 0.4 | vmf_pin3 | 10/10 | 0.3808 +- 0.0005 | 0.3798 / 0.3813 | 0.0216 | 1.132 | 3 | 0.3770 |
| 5 | 0.4 | vmf_pin10 | 7/7 | 1.6273 +- 0.0005 | 1.6263 / 1.6278 | 0.5748 | 0.682 | 10 | 1.5925 |
| 5 | 0.4 | vmf_pin30 | 8/8 | 3.2518 +- 0.0005 | 3.2507 / 3.2524 | 1.9652 | 0.449 | 30 | 3.1021 |
| 5 | 0.4 | vmf_pin100 | 7/7 | 5.2396 +- 0.0007 | 5.2388 / 5.2406 | 3.8791 | 0.312 | 100 | 4.9039 |
| 5 | 0.4 | vmf_pin300 | 5/5 | 7.2226 +- 0.0013 | 7.2216 / 7.2249 | 5.7461 | 0.255 | 300 | 6.7278 |
| 5 | 0.4 | power_spherical_pin3 | 10/10 | 0.2648 +- 0.0003 | 0.2642 / 0.2654 | 0.0204 | 1.170 | 3 | 0.2586 |
| 5 | 0.4 | power_spherical_pin10 | 7/7 | 1.0415 +- 0.0013 | 1.0403 / 1.0434 | 0.3096 | 0.812 | 10 | 1.0307 |
| 5 | 0.4 | power_spherical_pin30 | 9/9 | 2.3347 +- 0.0008 | 2.3337 / 2.3361 | 1.2347 | 0.561 | 30 | 2.2569 |
| 5 | 0.4 | power_spherical_pin100 | 7/7 | 4.1192 +- 0.0016 | 4.1178 / 4.1223 | 2.8208 | 0.373 | 100 | 3.8882 |
| 5 | 0.4 | power_spherical_pin300 | 7/7 | 5.9720 +- 0.0007 | 5.9707 / 5.9727 | 4.5417 | 0.284 | 300 | 5.5671 |
| 5 | 0.4 | vmf_k5d | 10/10 | 0.0977 +- 0.0016 | 0.0952 / 0.1013 | 0.0312 | 1.400 | 1 | 0.0673 |
| 5 | 0.4 | vmf_k20d | 10/10 | 0.1040 +- 0.0090 | 0.0964 / 0.1275 | 0.0332 | 1.414 | 1 | 0.0716 |
| 5 | 0.4 | power_spherical_k5d | 10/10 | 0.0442 +- 0.0009 | 0.0428 / 0.0459 | 0.0126 | 1.245 | 1.16 | 0.0319 |
| 5 | 0.4 | power_spherical_k20d | 10/10 | 0.0450 +- 0.0011 | 0.0427 / 0.0463 | 0.0127 | 1.247 | 1.16 | 0.0325 |
| 5 | 0.4 | vmf_H | 10/10 | 0.0969 +- 0.0009 | 0.0960 / 0.0990 | 0.0309 | 1.398 | 1 | 0.0669 |
| 5 | 0.4 | power_spherical_H | 10/10 | 0.0445 +- 0.0010 | 0.0429 / 0.0466 | 0.0125 | 1.246 | 1.2 | 0.0319 |
| 5 | 0.4 | tnbbeta_H | 10/10 | 0.0098 +- 0.0017 | 0.0081 / 0.0134 | 0.0010 | 1.054 | nan | 0.0091 |
| 9 | 0.4 | tnbbeta | 10/10 | 0.0030 +- 0.0008 | 0.0019 / 0.0041 | -0.0001 | 1.015 | nan | 0.0030 |
| 9 | 0.4 | vmf | 10/10 | 0.0532 +- 0.0006 | 0.0525 / 0.0543 | 0.0102 | 1.191 | 1.13 | 0.0433 |
| 9 | 0.4 | power_spherical | 10/10 | 0.0257 +- 0.0009 | 0.0244 / 0.0272 | 0.0041 | 1.118 | 1.42 | 0.0221 |
| 9 | 0.4 | vmf_pin3 | 10/10 | 0.2482 +- 0.0002 | 0.2479 / 0.2485 | 0.0067 | 1.080 | 3 | 0.2468 |
| 9 | 0.4 | vmf_pin10 | 6/6 | 1.8346 +- 0.0006 | 1.8341 / 1.8356 | 0.4205 | 0.660 | 10 | 1.7737 |
| 9 | 0.4 | vmf_pin30 | 5/5 | 4.9240 +- 0.0005 | 4.9233 / 4.9245 | 2.7038 | 0.422 | 30 | 4.7006 |
| 9 | 0.4 | vmf_pin100 | 6/6 | 9.0083 +- 0.0008 | 9.0070 / 9.0094 | 6.4698 | 0.268 | 100 | 8.5314 |
| 9 | 0.4 | vmf_pin300 | 6/6 | 13.0258 +- 0.0018 | 13.0231 / 13.0282 | 10.4167 | 0.197 | 300 | 12.2999 |
| 9 | 0.4 | power_spherical_pin3 | 10/10 | 0.1747 +- 0.0001 | 0.1746 / 0.1749 | 0.0091 | 1.095 | 3 | 0.1725 |
| 9 | 0.4 | power_spherical_pin10 | 9/9 | 1.0779 +- 0.0007 | 1.0769 / 1.0788 | 0.2268 | 0.760 | 10 | 1.0491 |
| 9 | 0.4 | power_spherical_pin30 | 8/8 | 3.2463 +- 0.0006 | 3.2458 / 3.2475 | 1.5123 | 0.523 | 30 | 3.1112 |
| 9 | 0.4 | power_spherical_pin100 | 9/9 | 6.7647 +- 0.0011 | 6.7629 / 6.7668 | 4.4796 | 0.338 | 100 | 6.4301 |
| 9 | 0.4 | power_spherical_pin300 | 6/6 | 10.5244 +- 0.0030 | 10.5221 / 10.5299 | 8.0116 | 0.234 | 300 | 9.9482 |
| 9 | 0.4 | vmf_k5d | 10/10 | 0.0533 +- 0.0011 | 0.0522 / 0.0561 | 0.0103 | 1.192 | 1.13 | 0.0433 |
| 9 | 0.4 | vmf_k20d | 10/10 | 0.0548 +- 0.0011 | 0.0536 / 0.0571 | 0.0109 | 1.194 | 1.13 | 0.0445 |
| 9 | 0.4 | power_spherical_k5d | 10/10 | 0.0261 +- 0.0014 | 0.0239 / 0.0283 | 0.0040 | 1.118 | 1.38 | 0.0225 |
| 9 | 0.4 | power_spherical_k20d | 10/10 | 0.0274 +- 0.0014 | 0.0260 / 0.0309 | 0.0043 | 1.119 | 1.44 | 0.0237 |
| 9 | 0.4 | vmf_H | 10/10 | 0.0530 +- 0.0005 | 0.0524 / 0.0538 | 0.0097 | 1.190 | 1.14 | 0.0431 |
| 9 | 0.4 | power_spherical_H | 10/10 | 0.0249 +- 0.0006 | 0.0241 / 0.0258 | 0.0040 | 1.117 | 1.44 | 0.0214 |
| 9 | 0.4 | tnbbeta_H | 10/10 | 0.0027 +- 0.0012 | 0.0016 / 0.0047 | -0.0000 | 1.014 | nan | 0.0028 |
| 17 | 0.4 | tnbbeta | 10/10 | 0.0016 +- 0.0008 | 0.0008 / 0.0034 | 0.0001 | 1.002 | nan | 0.0015 |
| 17 | 0.4 | vmf | 10/10 | 0.0251 +- 0.0004 | 0.0244 / 0.0257 | 0.0019 | 1.065 | 1.35 | 0.0230 |
| 17 | 0.4 | power_spherical | 10/10 | 0.0111 +- 0.0003 | 0.0108 / 0.0117 | 0.0012 | 1.038 | 1.56 | 0.0104 |
| 17 | 0.4 | vmf_pin3 | 10/10 | 0.1304 +- 0.0001 | 0.1302 / 0.1307 | 0.0016 | 1.036 | 3 | 0.1298 |
| 17 | 0.4 | vmf_pin10 | 7/7 | 1.5050 +- 0.0006 | 1.5037 / 1.5058 | 0.2235 | 0.664 | 10 | 1.4186 |
| 17 | 0.4 | vmf_pin30 | 5/5 | 6.1866 +- 0.0003 | 6.1860 / 6.1869 | 2.9158 | 0.404 | 30 | 5.8218 |
| 17 | 0.4 | vmf_pin100 | 6/6 | 13.9723 +- 0.0024 | 13.9685 / 13.9749 | 9.8531 | 0.247 | 100 | 13.2537 |
| 17 | 0.4 | vmf_pin300 | 7/7 | 21.9432 +- 0.0032 | 21.9396 / 21.9486 | 17.6753 | 0.165 | 300 | 20.9202 |
| 17 | 0.4 | power_spherical_pin3 | 10/10 | 0.0985 +- 0.0002 | 0.0983 / 0.0988 | 0.0030 | 1.041 | 3 | 0.0977 |
| 17 | 0.4 | power_spherical_pin10 | 7/7 | 0.8881 +- 0.0004 | 0.8876 / 0.8888 | 0.1303 | 0.737 | 10 | 0.8389 |
| 17 | 0.4 | power_spherical_pin30 | 6/6 | 3.7306 +- 0.0010 | 3.7293 / 3.7315 | 1.4463 | 0.492 | 30 | 3.4914 |
| 17 | 0.4 | power_spherical_pin100 | 8/8 | 9.8138 +- 0.0018 | 9.8113 / 9.8162 | 6.1777 | 0.317 | 100 | 9.2777 |
| 17 | 0.4 | power_spherical_pin300 | 7/7 | 17.0666 +- 0.0051 | 17.0618 / 17.0765 | 12.8877 | 0.209 | 300 | 16.2229 |
| 17 | 0.4 | vmf_k5d | 10/10 | 0.0256 +- 0.0006 | 0.0246 / 0.0268 | 0.0023 | 1.066 | 1.39 | 0.0234 |
| 17 | 0.4 | vmf_k20d | 10/10 | 0.0281 +- 0.0042 | 0.0250 / 0.0396 | 0.0027 | 1.069 | 1.42 | 0.0257 |
| 17 | 0.4 | power_spherical_k5d | 10/10 | 0.0114 +- 0.0003 | 0.0110 / 0.0119 | 0.0012 | 1.038 | 1.62 | 0.0106 |
| 17 | 0.4 | power_spherical_k20d | 10/10 | 0.0141 +- 0.0021 | 0.0115 / 0.0185 | 0.0012 | 1.039 | 1.56 | 0.0134 |
| 17 | 0.4 | vmf_H | 10/10 | 0.0248 +- 0.0003 | 0.0243 / 0.0253 | 0.0021 | 1.066 | 1.39 | 0.0227 |
| 17 | 0.4 | power_spherical_H | 10/10 | 0.0112 +- 0.0005 | 0.0103 / 0.0117 | 0.0012 | 1.038 | 1.53 | 0.0105 |
| 17 | 0.4 | tnbbeta_H | 10/10 | 0.0014 +- 0.0007 | 0.0006 / 0.0031 | 0.0000 | 1.002 | nan | 0.0013 |

## Primary: best-tuned cap per family (in-sample) vs TNBBeta (one-sided paired Wilcoxon, Holm 0.05 over 24)

| d | sigma | family | best-tuned arm (in-sample) | n | mean diff (cap - tnbbeta) | seeds tnbbeta better | p (scipy) | p (enumerated) | Holm thr | Holm reject | margin | confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.1 | power_spherical | power_spherical | 10 | 0.1285 | 10/10 | 0.00098 | 0.00098 | 0.00217 | True | True | True |
| 3 | 0.1 | vmf | vmf_H | 10 | 0.2085 | 10/10 | 0.00098 | 0.00098 | 0.00208 | True | True | True |
| 5 | 0.1 | power_spherical | power_spherical_H | 10 | 0.1367 | 10/10 | 0.00098 | 0.00098 | 0.00294 | True | True | True |
| 5 | 0.1 | vmf | vmf_k5d | 10 | 0.2043 | 10/10 | 0.00098 | 0.00098 | 0.00278 | True | True | True |
| 9 | 0.1 | power_spherical | power_spherical_H | 10 | 0.1397 | 10/10 | 0.00098 | 0.00098 | 0.00455 | True | True | True |
| 9 | 0.1 | vmf | vmf_H | 10 | 0.1846 | 10/10 | 0.00098 | 0.00098 | 0.00417 | True | True | True |
| 17 | 0.1 | power_spherical | power_spherical_H | 10 | 0.1492 | 10/10 | 0.00098 | 0.00098 | 0.01000 | True | True | True |
| 17 | 0.1 | vmf | vmf_H | 10 | 0.1813 | 10/10 | 0.00098 | 0.00098 | 0.00833 | True | True | True |
| 3 | 0.2 | power_spherical | power_spherical_H | 10 | 0.0689 | 10/10 | 0.00098 | 0.00098 | 0.00238 | True | True | True |
| 3 | 0.2 | vmf | vmf_H | 10 | 0.1458 | 10/10 | 0.00098 | 0.00098 | 0.00227 | True | True | True |
| 5 | 0.2 | power_spherical | power_spherical | 10 | 0.0963 | 10/10 | 0.00098 | 0.00098 | 0.00333 | True | True | True |
| 5 | 0.2 | vmf | vmf_H | 10 | 0.1617 | 10/10 | 0.00098 | 0.00098 | 0.00313 | True | True | True |
| 9 | 0.2 | power_spherical | power_spherical_H | 10 | 0.0891 | 10/10 | 0.00098 | 0.00098 | 0.00556 | True | True | True |
| 9 | 0.2 | vmf | vmf_H | 10 | 0.1306 | 10/10 | 0.00098 | 0.00098 | 0.00500 | True | True | True |
| 17 | 0.2 | power_spherical | power_spherical | 10 | 0.0774 | 10/10 | 0.00098 | 0.00098 | 0.01667 | True | True | True |
| 17 | 0.2 | vmf | vmf_H | 10 | 0.1050 | 10/10 | 0.00098 | 0.00098 | 0.01250 | True | True | True |
| 3 | 0.4 | power_spherical | power_spherical_H | 10 | 0.0213 | 10/10 | 0.00098 | 0.00098 | 0.00263 | True | False | False |
| 3 | 0.4 | vmf | vmf_H | 10 | 0.0874 | 10/10 | 0.00098 | 0.00098 | 0.00250 | True | True | True |
| 5 | 0.4 | power_spherical | power_spherical_k5d | 10 | 0.0335 | 10/10 | 0.00098 | 0.00098 | 0.00385 | True | False | False |
| 5 | 0.4 | vmf | vmf_H | 10 | 0.0862 | 10/10 | 0.00098 | 0.00098 | 0.00357 | True | True | True |
| 9 | 0.4 | power_spherical | power_spherical_H | 10 | 0.0219 | 10/10 | 0.00098 | 0.00098 | 0.00714 | True | False | False |
| 9 | 0.4 | vmf | vmf_H | 10 | 0.0500 | 10/10 | 0.00098 | 0.00098 | 0.00625 | True | False | False |
| 17 | 0.4 | power_spherical | power_spherical | 10 | 0.0095 | 10/10 | 0.00098 | 0.00098 | 0.05000 | True | False | False |
| 17 | 0.4 | vmf | vmf_H | 10 | 0.0233 | 10/10 | 0.00098 | 0.00098 | 0.02500 | True | False | False |

Confirmed (Holm and margin): 18 of 24 tests.

| d | sigma | category (both families) |
|---|---|---|
| 3 | 0.1 | advantage survives |
| 3 | 0.2 | advantage survives |
| 3 | 0.4 | within margin / artefact (diff <= 0.05 for at least one family) |
| 5 | 0.1 | advantage survives |
| 5 | 0.2 | advantage survives |
| 5 | 0.4 | within margin / artefact (diff <= 0.05 for at least one family) |
| 9 | 0.1 | advantage survives |
| 9 | 0.2 | advantage survives |
| 9 | 0.4 | within margin / artefact (diff <= 0.05 for at least one family) |
| 17 | 0.1 | advantage survives |
| 17 | 0.2 | advantage survives |
| 17 | 0.4 | within margin / artefact (diff <= 0.05 for at least one family) |

## Sensitivity S1: leave-one-seed-out tuned cap vs TNBBeta (one-sided paired Wilcoxon, Holm 0.05 over 24)

| d | sigma | family | best-tuned arm (in-sample) | n | mean diff (cap - tnbbeta) | seeds tnbbeta better | p (scipy) | p (enumerated) | Holm thr | Holm reject | margin | confirmed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 0.1 | power_spherical | power_spherical | 10 | 0.1291 | 10/10 | 0.00098 | 0.00098 | 0.00217 | True | True | True |
| 3 | 0.1 | vmf | vmf_H | 10 | 0.2085 | 10/10 | 0.00098 | 0.00098 | 0.00208 | True | True | True |
| 5 | 0.1 | power_spherical | power_spherical_H | 10 | 0.1367 | 10/10 | 0.00098 | 0.00098 | 0.00294 | True | True | True |
| 5 | 0.1 | vmf | vmf_k5d | 10 | 0.2058 | 10/10 | 0.00098 | 0.00098 | 0.00278 | True | True | True |
| 9 | 0.1 | power_spherical | power_spherical_H | 10 | 0.1400 | 10/10 | 0.00098 | 0.00098 | 0.00455 | True | True | True |
| 9 | 0.1 | vmf | vmf_H | 10 | 0.1846 | 10/10 | 0.00098 | 0.00098 | 0.00417 | True | True | True |
| 17 | 0.1 | power_spherical | power_spherical_H | 10 | 0.1492 | 10/10 | 0.00098 | 0.00098 | 0.01000 | True | True | True |
| 17 | 0.1 | vmf | vmf_H | 10 | 0.1813 | 10/10 | 0.00098 | 0.00098 | 0.00833 | True | True | True |
| 3 | 0.2 | power_spherical | power_spherical_H | 10 | 0.0703 | 10/10 | 0.00098 | 0.00098 | 0.00238 | True | True | True |
| 3 | 0.2 | vmf | vmf_H | 10 | 0.1458 | 10/10 | 0.00098 | 0.00098 | 0.00227 | True | True | True |
| 5 | 0.2 | power_spherical | power_spherical | 10 | 0.0963 | 10/10 | 0.00098 | 0.00098 | 0.00333 | True | True | True |
| 5 | 0.2 | vmf | vmf_H | 10 | 0.1626 | 10/10 | 0.00098 | 0.00098 | 0.00313 | True | True | True |
| 9 | 0.2 | power_spherical | power_spherical_H | 10 | 0.0891 | 10/10 | 0.00098 | 0.00098 | 0.00556 | True | True | True |
| 9 | 0.2 | vmf | vmf_H | 10 | 0.1310 | 10/10 | 0.00098 | 0.00098 | 0.00500 | True | True | True |
| 17 | 0.2 | power_spherical | power_spherical | 10 | 0.0778 | 10/10 | 0.00098 | 0.00098 | 0.01667 | True | True | True |
| 17 | 0.2 | vmf | vmf_H | 10 | 0.1050 | 10/10 | 0.00098 | 0.00098 | 0.01250 | True | True | True |
| 3 | 0.4 | power_spherical | power_spherical_H | 10 | 0.0213 | 10/10 | 0.00098 | 0.00098 | 0.00263 | True | False | False |
| 3 | 0.4 | vmf | vmf_H | 10 | 0.0874 | 10/10 | 0.00098 | 0.00098 | 0.00250 | True | True | True |
| 5 | 0.4 | power_spherical | power_spherical_k5d | 10 | 0.0341 | 10/10 | 0.00098 | 0.00098 | 0.00385 | True | False | False |
| 5 | 0.4 | vmf | vmf_H | 10 | 0.0862 | 10/10 | 0.00098 | 0.00098 | 0.00357 | True | True | True |
| 9 | 0.4 | power_spherical | power_spherical_H | 10 | 0.0219 | 10/10 | 0.00098 | 0.00098 | 0.00714 | True | False | False |
| 9 | 0.4 | vmf | vmf_H | 10 | 0.0500 | 10/10 | 0.00098 | 0.00098 | 0.00625 | True | False | False |
| 17 | 0.4 | power_spherical | power_spherical | 10 | 0.0095 | 10/10 | 0.00098 | 0.00098 | 0.05000 | True | False | False |
| 17 | 0.4 | vmf | vmf_H | 10 | 0.0233 | 10/10 | 0.00098 | 0.00098 | 0.02500 | True | False | False |

Confirmed (Holm and margin): 18 of 24 tests.

| d | sigma | category (both families) |
|---|---|---|
| 3 | 0.1 | advantage survives |
| 3 | 0.2 | advantage survives |
| 3 | 0.4 | within margin / artefact (diff <= 0.05 for at least one family) |
| 5 | 0.1 | advantage survives |
| 5 | 0.2 | advantage survives |
| 5 | 0.4 | within margin / artefact (diff <= 0.05 for at least one family) |
| 9 | 0.1 | advantage survives |
| 9 | 0.2 | advantage survives |
| 9 | 0.4 | within margin / artefact (diff <= 0.05 for at least one family) |
| 17 | 0.1 | advantage survives |
| 17 | 0.2 | advantage survives |
| 17 | 0.4 | within margin / artefact (diff <= 0.05 for at least one family) |

## S2 / S3: per-cell best caps, equal budget and pooled families

| d | sigma | tnbbeta | tnbbeta_H | best TNBBeta (in-sample) | best cap overall (arm) | mean diff best cap - best TNBBeta | seeds TNBBeta better |
|---|---|---|---|---|---|---|---|
| 3 | 0.1 | 0.0705 | 0.0617 | tnbbeta_H | power_spherical | 0.1373 | 10/10 |
| 3 | 0.2 | 0.0633 | 0.0547 | tnbbeta_H | power_spherical_H | 0.0775 | 10/10 |
| 3 | 0.4 | 0.0307 | 0.0305 | tnbbeta_H | power_spherical_H | 0.0215 | 10/10 |
| 5 | 0.1 | 0.0325 | 0.0294 | tnbbeta_H | power_spherical_H | 0.1398 | 10/10 |
| 5 | 0.2 | 0.0272 | 0.0241 | tnbbeta_H | power_spherical | 0.0994 | 10/10 |
| 5 | 0.4 | 0.0107 | 0.0098 | tnbbeta_H | power_spherical_k5d | 0.0345 | 10/10 |
| 9 | 0.1 | 0.0190 | 0.0192 | tnbbeta | power_spherical_H | 0.1397 | 10/10 |
| 9 | 0.2 | 0.0132 | 0.0093 | tnbbeta_H | power_spherical_H | 0.0929 | 10/10 |
| 9 | 0.4 | 0.0030 | 0.0027 | tnbbeta_H | power_spherical_H | 0.0222 | 10/10 |
| 17 | 0.1 | 0.0134 | 0.0066 | tnbbeta_H | power_spherical_H | 0.1560 | 10/10 |
| 17 | 0.2 | 0.0048 | 0.0059 | tnbbeta | power_spherical | 0.0774 | 10/10 |
| 17 | 0.4 | 0.0016 | 0.0014 | tnbbeta_H | power_spherical | 0.0098 | 10/10 |

## S4: best pinned kappa, grid-edge flags, fitted kappa and sigma_hat

| d | sigma | family | best pinned kappa | edge? | gap at best pin | best pool arm |
|---|---|---|---|---|---|---|
| 3 | 0.1 | vmf | 3 | EDGE | 0.5425 | vmf_H |
| 3 | 0.1 | power_spherical | 3 | EDGE | 0.4202 | power_spherical |
| 3 | 0.2 | vmf | 3 | EDGE | 0.4810 | vmf_H |
| 3 | 0.2 | power_spherical | 3 | EDGE | 0.3625 | power_spherical_H |
| 3 | 0.4 | vmf | 3 | EDGE | 0.4103 | vmf_H |
| 3 | 0.4 | power_spherical | 3 | EDGE | 0.2963 | power_spherical_H |
| 5 | 0.1 | vmf | 3 | EDGE | 0.5164 | vmf_k5d |
| 5 | 0.1 | power_spherical | 3 | EDGE | 0.3828 | power_spherical_H |
| 5 | 0.2 | vmf | 3 | EDGE | 0.4679 | vmf_H |
| 5 | 0.2 | power_spherical | 3 | EDGE | 0.3392 | power_spherical |
| 5 | 0.4 | vmf | 3 | EDGE | 0.3808 | vmf_H |
| 5 | 0.4 | power_spherical | 3 | EDGE | 0.2648 | power_spherical_k5d |
| 9 | 0.1 | vmf | 3 | EDGE | 0.4109 | vmf_H |
| 9 | 0.1 | power_spherical | 3 | EDGE | 0.3184 | power_spherical_H |
| 9 | 0.2 | vmf | 3 | EDGE | 0.3458 | vmf_H |
| 9 | 0.2 | power_spherical | 3 | EDGE | 0.2589 | power_spherical_H |
| 9 | 0.4 | vmf | 3 | EDGE | 0.2482 | vmf_H |
| 9 | 0.4 | power_spherical | 3 | EDGE | 0.1747 | power_spherical_H |
| 17 | 0.1 | vmf | 3 | EDGE | 0.3122 | vmf_H |
| 17 | 0.1 | power_spherical | 3 | EDGE | 0.2635 | power_spherical_H |
| 17 | 0.2 | vmf | 3 | EDGE | 0.2239 | vmf_H |
| 17 | 0.2 | power_spherical | 3 | EDGE | 0.1808 | power_spherical |
| 17 | 0.4 | vmf | 3 | EDGE | 0.1304 | vmf_H |
| 17 | 0.4 | power_spherical | 3 | EDGE | 0.0985 | power_spherical |

## S4: kappa trajectories of trainable arms (median kappa at epochs)

| d | sigma | arm | ep 1 | ep 10 | ep 30 | ep 100 | ep 300 | final |
|---|---|---|---|---|---|---|---|---|
| 3 | 0.1 | vmf_k5d (kappa) | 16.6 | 30.8 | 2.02 | 1.2 | 1.25 | 1.25 |
| 3 | 0.1 | vmf_k20d (kappa) | 61.5 | 50.3 | 2.28 | 1.15 | 1.22 | 1.22 |
| 3 | 0.1 | power_spherical_k5d (kappa) | 16.6 | 45.7 | 1.95 | 1.36 | 1.35 | 1.35 |
| 3 | 0.1 | power_spherical_k20d (kappa) | 61.5 | 67.6 | 2.48 | 1.29 | 1.35 | 1.35 |
| 3 | 0.1 | vmf_H (kappa) | 18.5 | 69 | 42.4 | 1.22 | 1.23 | 1.22 |
| 3 | 0.1 | power_spherical_H (kappa) | 21.8 | 119 | 68.3 | 1.35 | 1.37 | 1.38 |
| 3 | 0.1 | tnbbeta_H (eps) | 3.97 | 2.73 | 5.17 | 6.53 | 7.9 | 10.8 |
| 3 | 0.2 | vmf_k5d (kappa) | 16.6 | 29.9 | 1.96 | 1.17 | 1.17 | 1.17 |
| 3 | 0.2 | vmf_k20d (kappa) | 61.5 | 50 | 2.09 | 1.15 | 1.15 | 1.15 |
| 3 | 0.2 | power_spherical_k5d (kappa) | 16.6 | 44.2 | 1.92 | 1.29 | 1.31 | 1.31 |
| 3 | 0.2 | power_spherical_k20d (kappa) | 61.5 | 66.9 | 2.47 | 1.27 | 1.26 | 1.26 |
| 3 | 0.2 | vmf_H (kappa) | 19.1 | 67.6 | 32.7 | 1.17 | 1.18 | 1.18 |
| 3 | 0.2 | power_spherical_H (kappa) | 21.2 | 117 | 50.7 | 1.31 | 1.33 | 1.33 |
| 3 | 0.2 | tnbbeta_H (eps) | 4.02 | 2.75 | 4.95 | 4.67 | 5.28 | 5.8 |
| 3 | 0.4 | vmf_k5d (kappa) | 16.6 | 25.7 | 1.67 | 1.1 | 1.09 | 1.09 |
| 3 | 0.4 | vmf_k20d (kappa) | 61.5 | 49.3 | 1.68 | 1.07 | 1.08 | 1.08 |
| 3 | 0.4 | power_spherical_k5d (kappa) | 16.7 | 39.3 | 1.82 | 1.12 | 1.16 | 1.16 |
| 3 | 0.4 | power_spherical_k20d (kappa) | 61.6 | 63.9 | 1.8 | 1.12 | 1.11 | 1.11 |
| 3 | 0.4 | vmf_H (kappa) | 19.3 | 58.6 | 10.3 | 1.08 | 1.1 | 1.09 |
| 3 | 0.4 | power_spherical_H (kappa) | 21.9 | 106 | 15.4 | 1.15 | 1.19 | 1.18 |
| 3 | 0.4 | tnbbeta_H (eps) | 3.84 | 2.7 | 4.77 | 1.74 | 1.72 | 1.66 |
| 5 | 0.1 | vmf_k5d (kappa) | 26.4 | 15.2 | 1.14 | 1.01 | 1 | 1 |
| 5 | 0.1 | vmf_k20d (kappa) | 101 | 16.1 | 1.23 | 1.01 | 1 | 1 |
| 5 | 0.1 | power_spherical_k5d (kappa) | 26.4 | 26.6 | 1.38 | 1.27 | 1.25 | 1.25 |
| 5 | 0.1 | power_spherical_k20d (kappa) | 101 | 51.8 | 1.51 | 1.21 | 1.29 | 1.29 |
| 5 | 0.1 | vmf_H (kappa) | 21.5 | 38 | 1.85 | 1.01 | 1 | 1 |
| 5 | 0.1 | power_spherical_H (kappa) | 23.9 | 70.6 | 2.6 | 1.28 | 1.26 | 1.25 |
| 5 | 0.1 | tnbbeta_H (eps) | 2.97 | 4.98 | 9.26 | 10.3 | 19.1 | 20.9 |
| 5 | 0.2 | vmf_k5d (kappa) | 26.3 | 14.5 | 1.17 | 1.01 | 1 | 1 |
| 5 | 0.2 | vmf_k20d (kappa) | 101 | 14.6 | 1.27 | 1.01 | 1 | 1 |
| 5 | 0.2 | power_spherical_k5d (kappa) | 26.4 | 27.1 | 1.41 | 1.31 | 1.25 | 1.25 |
| 5 | 0.2 | power_spherical_k20d (kappa) | 101 | 48.8 | 1.54 | 1.2 | 1.3 | 1.3 |
| 5 | 0.2 | vmf_H (kappa) | 21.4 | 36 | 1.87 | 1.01 | 1 | 1 |
| 5 | 0.2 | power_spherical_H (kappa) | 24.3 | 68.7 | 2.47 | 1.22 | 1.26 | 1.24 |
| 5 | 0.2 | tnbbeta_H (eps) | 3.22 | 4.35 | 7.89 | 7.56 | 7.7 | 7.38 |
| 5 | 0.4 | vmf_k5d (kappa) | 26.3 | 17.2 | 1.23 | 1.01 | 1 | 1 |
| 5 | 0.4 | vmf_k20d (kappa) | 101 | 12.2 | 1.33 | 1.01 | 1 | 1 |
| 5 | 0.4 | power_spherical_k5d (kappa) | 26.4 | 27.2 | 1.6 | 1.23 | 1.19 | 1.19 |
| 5 | 0.4 | power_spherical_k20d (kappa) | 101 | 52.9 | 1.67 | 1.06 | 1.2 | 1.2 |
| 5 | 0.4 | vmf_H (kappa) | 21.4 | 29.9 | 2.02 | 1.01 | 1.01 | 1 |
| 5 | 0.4 | power_spherical_H (kappa) | 25 | 57.9 | 2.94 | 1.22 | 1.19 | 1.24 |
| 5 | 0.4 | tnbbeta_H (eps) | 3.29 | 4.52 | 6.79 | 3.21 | 2.93 | 2.8 |
| 9 | 0.1 | vmf_k5d (kappa) | 46 | 2.33 | 1.17 | 1.08 | 1.07 | 1.07 |
| 9 | 0.1 | vmf_k20d (kappa) | 179 | 2.65 | 1.3 | 1.08 | 1.06 | 1.06 |
| 9 | 0.1 | power_spherical_k5d (kappa) | 46.1 | 3.05 | 1.4 | 1.36 | 1.39 | 1.39 |
| 9 | 0.1 | power_spherical_k20d (kappa) | 179 | 3.33 | 1.32 | 1.18 | 1.47 | 1.47 |
| 9 | 0.1 | vmf_H (kappa) | 24.3 | 7.94 | 1.81 | 1.09 | 1.1 | 1.09 |
| 9 | 0.1 | power_spherical_H (kappa) | 25 | 15.3 | 2.3 | 1.42 | 1.43 | 1.42 |
| 9 | 0.1 | tnbbeta_H (eps) | 2.81 | 8.41 | 16.7 | 12.9 | 16 | 18.5 |
| 9 | 0.2 | vmf_k5d (kappa) | 46 | 2.49 | 1.25 | 1.09 | 1.08 | 1.08 |
| 9 | 0.2 | vmf_k20d (kappa) | 179 | 3.17 | 1.37 | 1.1 | 1.09 | 1.09 |
| 9 | 0.2 | power_spherical_k5d (kappa) | 46.1 | 3.3 | 1.56 | 1.36 | 1.42 | 1.42 |
| 9 | 0.2 | power_spherical_k20d (kappa) | 179 | 3.96 | 1.58 | 1.33 | 1.56 | 1.56 |
| 9 | 0.2 | vmf_H (kappa) | 23.9 | 7.9 | 1.96 | 1.09 | 1.1 | 1.11 |
| 9 | 0.2 | power_spherical_H (kappa) | 25.4 | 14.3 | 2.56 | 1.44 | 1.42 | 1.47 |
| 9 | 0.2 | tnbbeta_H (eps) | 2.88 | 8.79 | 13.3 | 10.4 | 10.5 | 9.7 |
| 9 | 0.4 | vmf_k5d (kappa) | 46 | 3.48 | 1.39 | 1.09 | 1.1 | 1.1 |
| 9 | 0.4 | vmf_k20d (kappa) | 179 | 4.02 | 1.59 | 1.07 | 1.1 | 1.1 |
| 9 | 0.4 | power_spherical_k5d (kappa) | 46.1 | 4.72 | 1.68 | 1.29 | 1.31 | 1.31 |
| 9 | 0.4 | power_spherical_k20d (kappa) | 179 | 4.88 | 1.7 | 1.38 | 1.41 | 1.41 |
| 9 | 0.4 | vmf_H (kappa) | 24.7 | 6.1 | 2.5 | 1.09 | 1.11 | 1.11 |
| 9 | 0.4 | power_spherical_H (kappa) | 28.8 | 11.6 | 3.39 | 1.39 | 1.35 | 1.39 |
| 9 | 0.4 | tnbbeta_H (eps) | 2.96 | 7.54 | 8.98 | 5.79 | 5.23 | 4.92 |
| 17 | 0.1 | vmf_k5d (kappa) | 83.7 | 2.81 | 1.34 | 1.33 | 1.38 | 1.38 |
| 17 | 0.1 | vmf_k20d (kappa) | 339 | 1 | 1 | 1 | 1 | 1 |
| 17 | 0.1 | power_spherical_k5d (kappa) | 84 | 3.61 | 1.69 | 1.77 | 1.73 | 1.73 |
| 17 | 0.1 | power_spherical_k20d (kappa) | 339 | 3.79 | 1.71 | 1.5 | 1.76 | 1.76 |
| 17 | 0.1 | vmf_H (kappa) | 24.5 | 3.13 | 2.19 | 1.37 | 1.38 | 1.41 |
| 17 | 0.1 | power_spherical_H (kappa) | 25.1 | 4.07 | 2.82 | 1.82 | 1.8 | 1.76 |
| 17 | 0.1 | tnbbeta_H (eps) | 2.6 | 7.26 | 18 | 19.5 | 22.5 | 21.4 |
| 17 | 0.2 | vmf_k5d (kappa) | 83.6 | 3.37 | 1.54 | 1.44 | 1.45 | 1.45 |
| 17 | 0.2 | vmf_k20d (kappa) | 339 | 1 | 1 | 1 | 1 | 1 |
| 17 | 0.2 | power_spherical_k5d (kappa) | 84 | 4.41 | 1.78 | 1.83 | 1.84 | 1.84 |
| 17 | 0.2 | power_spherical_k20d (kappa) | 339 | 4.7 | 1.83 | 1.83 | 1.97 | 1.97 |
| 17 | 0.2 | vmf_H (kappa) | 24.6 | 3.68 | 2.59 | 1.47 | 1.52 | 1.5 |
| 17 | 0.2 | power_spherical_H (kappa) | 26 | 4.96 | 3.3 | 1.88 | 1.95 | 1.88 |
| 17 | 0.2 | tnbbeta_H (eps) | 2.66 | 7.27 | 17.5 | 13.8 | 13.2 | 12.4 |
| 17 | 0.4 | vmf_k5d (kappa) | 83.6 | 5.07 | 1.63 | 1.38 | 1.33 | 1.33 |
| 17 | 0.4 | vmf_k20d (kappa) | 338 | 6.09 | 1.99 | 1.3 | 1.35 | 1.35 |
| 17 | 0.4 | power_spherical_k5d (kappa) | 84.6 | 6.44 | 1.75 | 1.52 | 1.58 | 1.58 |
| 17 | 0.4 | power_spherical_k20d (kappa) | 339 | 7.15 | 2.22 | 1.44 | 1.54 | 1.54 |
| 17 | 0.4 | vmf_H (kappa) | 28.2 | 5.7 | 3.31 | 1.31 | 1.35 | 1.34 |
| 17 | 0.4 | power_spherical_H (kappa) | 32.7 | 7.6 | 4.32 | 1.57 | 1.55 | 1.55 |
| 17 | 0.4 | tnbbeta_H (eps) | 2.78 | 7.15 | 12.2 | 9.29 | 8.85 | 8.44 |
