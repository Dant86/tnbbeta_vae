# Ring-posterior task, Level 1 (representation): results

Date: 2026-11-01. Pre-registration: `ring_posterior_design_2026-11-01.md` (unchanged).

Code version: base commit `ec2677efa042f789c26470e44a9cb0a35c51c4c5` (branch `synthetic-graph-families`; note the design note
cites `ea01fde`, an ancestor) plus the UNCOMMITTED working tree files `apps/synthetic/ring_recovery.py`,
`src/tnbbeta_vae/data/ring_posterior.py`, `tests/apps/test_ring_recovery.py`, `tests/data/test_ring_posterior.py`.
No code changes were made to them in this session.

## Commands

Sanity checks (required by the design): `uv run pytest tests/data/test_ring_posterior.py tests/apps/test_ring_recovery.py -x -q --no-cov` -> 8 passed.

L1, one call per cell, 27 cells (d in {3,8,16} x latitude in {30,60,80} x sigma in {0.1,0.2,0.4}), seed default 0, run locally with
`nice -n 10`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, 4 concurrent processes (about 800 MB each, ~4-5 min per cell):

    uv run python -m apps.synthetic.ring_recovery stage1 --dim D --latitude T --sigma S --out-dir ring_recovery/stage1
    uv run python -m apps.synthetic.ring_recovery summarize --out-dir ring_recovery

Raw per-cell JSON: `ring_recovery/stage1/d{D}_t{T}_s{S}.json` (repo root, untracked).

## Result

**24 of 27 cells pass (X1 = 0.5 nats). The STOP rule (< 14 of 27) did NOT trigger.** L2/L3 were not run in this session
(out of scope of the request).

Columns: KL values are exact-KL (MC) in nats; "TNB" is the better of mu = a and mu free; gap = min(vMF, PS) - best TNBBeta.

| d | lat | sigma | truth mean/sd (deg) | KL uniform | TNB mu=a | TNB mu free | vMF | PS | gap | ring-shaped | band | TV | pole | m | ring-only | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 30 | 0.1 | 30.0/2.2 | 113.39 | 0.006 | 0.006 | 2.79 | 2.80 | 2.79 | True | 0.95 | 0.032 | 0.000 | 89.38 | True | True |
| 3 | 60 | 0.1 | 58.8/3.3 | 142.33 | 0.088 | 0.088 | 3.30 | 3.31 | 3.22 | True | 0.96 | 0.116 | 0.000 | 245.35 | True | True |
| 3 | 80 | 0.1 | 80.1/1.2 | 110.01 | 0.001 | 0.001 | 4.04 | 4.04 | 4.04 | True | 0.95 | 0.016 | 0.000 | 146.63 | True | True |
| 8 | 30 | 0.1 | 30.7/2.1 | 112.27 | 0.005 | 0.004 | 4.50 | 4.20 | 4.20 | True | 0.95 | 0.023 | 0.000 | 32.84 | True | True |
| 8 | 60 | 0.1 | 59.3/2.8 | 158.07 | 0.063 | 0.063 | 8.05 | 8.02 | 7.96 | True | 0.96 | 0.101 | 0.000 | 269.62 | True | True |
| 8 | 80 | 0.1 | 80.1/1.2 | 78.41 | 0.001 | 0.001 | 13.89 | 13.89 | 13.88 | True | 0.95 | 0.015 | 0.000 | 98.86 | True | True |
| 16 | 30 | 0.1 | 31.7/2.0 | 108.94 | 0.003 | 0.003 | 2.34 | 2.06 | 2.06 | True | 0.95 | 0.021 | 0.000 | 51.32 | True | True |
| 16 | 60 | 0.1 | 59.9/2.4 | 171.68 | 0.041 | 0.041 | 7.29 | 5.19 | 5.15 | True | 0.96 | 0.077 | 0.000 | 303.30 | True | True |
| 16 | 80 | 0.1 | 80.1/1.2 | 49.59 | 0.001 | 0.001 | 20.97 | 20.95 | 20.95 | True | 0.95 | 0.015 | 0.000 | 78.39 | True | True |
| 3 | 30 | 0.2 | 30.6/5.7 | 26.25 | 0.026 | 0.027 | 1.86 | 1.75 | 1.73 | True | 0.96 | 0.061 | 0.000 | 4.14 | True | True |
| 3 | 60 | 0.2 | 55.3/7.0 | 34.14 | 0.190 | 0.191 | 2.40 | 2.40 | 2.21 | True | 0.96 | 0.234 | 0.000 | 35.69 | True | True |
| 3 | 80 | 0.2 | 80.4/2.6 | 25.50 | 0.006 | 0.006 | 3.30 | 3.30 | 3.30 | True | 0.95 | 0.035 | 0.000 | 23.59 | True | True |
| 8 | 30 | 0.2 | 37.7/12.4 | 24.22 | 0.149 | 0.150 | 0.82 | 0.76 | 0.61 | True | 0.97 | 0.138 | 0.000 | -0.07 | False | True |
| 8 | 60 | 0.2 | 57.9/5.7 | 37.86 | 0.133 | 0.134 | 3.38 | 2.63 | 2.50 | True | 0.96 | 0.173 | 0.000 | 65.48 | True | True |
| 8 | 80 | 0.2 | 80.5/2.6 | 18.08 | 0.006 | 0.007 | 9.07 | 9.09 | 9.06 | True | 0.95 | 0.032 | 0.000 | 22.51 | True | True |
| 16 | 30 | 0.2 | 63.7/10.2 | 22.48 | 0.141 | 0.141 | 1.99 | 2.01 | 1.85 | True | 0.96 | 0.132 | 0.000 | 18.39 | True | True |
| 16 | 60 | 0.2 | 60.2/4.3 | 40.60 | 0.068 | 0.068 | 2.40 | 1.61 | 1.54 | True | 0.96 | 0.109 | 0.000 | 96.60 | True | True |
| 16 | 80 | 0.2 | 80.6/2.6 | 11.08 | 0.006 | 0.007 | 8.97 | 8.07 | 8.07 | True | 0.95 | 0.032 | 0.000 | 19.14 | True | True |
| 3 | 30 | 0.4 | 40.7/17.1 | 5.33 | 0.145 | 0.145 | 0.30 | 0.23 | 0.09 | True | 0.97 | 0.169 | 0.002 | 2.38 | True | False |
| 3 | 60 | 0.4 | 53.1/10.3 | 7.46 | 0.118 | 0.120 | 1.76 | 1.67 | 1.55 | True | 0.95 | 0.182 | 0.000 | 11.56 | True | True |
| 3 | 80 | 0.4 | 80.6/10.9 | 4.98 | 0.040 | 0.039 | 2.50 | 2.50 | 2.46 | True | 0.98 | 0.076 | 0.000 | 8.43 | True | True |
| 8 | 30 | 0.4 | 60.8/12.6 | 4.59 | 0.243 | 0.241 | 0.47 | 0.37 | 0.13 | True | 0.94 | 0.221 | 0.000 | -1.25 | False | False |
| 8 | 60 | 0.4 | 58.7/8.0 | 8.13 | 0.093 | 0.093 | 0.97 | 0.72 | 0.63 | True | 0.95 | 0.149 | 0.000 | 22.54 | True | True |
| 8 | 80 | 0.4 | 82.1/6.0 | 3.58 | 0.019 | 0.019 | 3.04 | 2.69 | 2.68 | True | 0.95 | 0.068 | 0.000 | 8.69 | True | True |
| 16 | 30 | 0.4 | 68.0/6.4 | 4.19 | 0.032 | 0.032 | 0.63 | 0.50 | 0.46 | True | 0.95 | 0.069 | 0.000 | -4.47 | False | False |
| 16 | 60 | 0.4 | 62.8/5.9 | 8.29 | 0.040 | 0.040 | 0.93 | 0.55 | 0.51 | True | 0.96 | 0.091 | 0.000 | 16.78 | True | True |
| 16 | 80 | 0.4 | 82.7/5.8 | 2.05 | 0.018 | 0.018 | 1.60 | 1.41 | 1.39 | True | 0.95 | 0.069 | 0.000 | 11.75 | True | True |

Pass counts per sigma: 0.1: 9/9; 0.2: 9/9; 0.4: 6/9. Per d: 3: 8/9; 8: 8/9; 16: 8/9.

## Notes

* The three failures are all sigma = 0.4 with latitude 30 (d = 3, 8, 16): the gap to the best cap is 0.09, 0.13, 0.46 nats (< 0.5), and
  the best cap KL is already small (0.23-0.50), because at the pre-registered x* = g(cos 30 deg) the posterior is broad and shifted
  (true mean latitude 41-68 deg, sd 6-17 deg, see "truth" column) so a cap is a decent fit. Near-passes: (d=16, 60, 0.4) gap 0.51,
  (d=8, 30, 0.2) gap 0.61.
* Three cells have best TNBBeta member with m < 0 (d=8 lat30 sigma0.2, m = -0.07; d=8 lat30 sigma0.4, m = -1.25; d=16 lat30 sigma0.4,
  m = -4.47), so "ring-only" is False there; pole mass is still <= 0.002 and band mass >= 0.94, so these members are ring-shaped by
  the X1 definition. Only the sigma=0.4/d=8 and d=16 cells of these also fail X1 on the gap.
* Optimiser repeatability (mu = a, quadrature KL): across the 27 cells, the spread among the best 5 starts is at most 1.1e-4 nats; the
  number of the 96 starts within 0.05 nats of the best ranges 27-90. Max MC standard error of any best TNBBeta KL is 1.2e-3. Quadrature
  and MC KL (mu = a best member) agree to within 0.003 nats in all 27 cells. The mu-free fit never beat mu = a by more than
  0.0017 nats (sometimes marginally worse, within MC noise), consistent with the axis being the optimal mu by symmetry.
* The "ring-shaped" requirement is satisfied in 27/27 cells, so the pass/fail was decided by the 0.5-nat gap alone.
* Unit-test sanity checks (posterior normalisation, KL(Uniform), KL(vMF) with independent normalisation, log p(x) identity) pass.
* L1 shows only that a TNBBeta member exists that represents the ring far better than any vMF/PS member (for thin rings). It says nothing
  about whether optimisation (L2) or amortised training (L3) finds it; on the earlier graph and axial tasks free TNBBeta did not.

## Deviations

None. No batch/sample sizes were reduced (peak ~800 MB per process, load average stayed under 6).
