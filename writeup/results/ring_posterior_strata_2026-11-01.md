# Ring task (d = 3): how many test posteriors are actually single rings, and where TNBBeta's KL sits

Date: 2026-11-01. Base commit: `ec2677e`; working tree uncommitted, with the small-epsilon
distribution fix and exact-log-q patches applied. Exploratory, post hoc; nothing here
changes the pre-registered analysis.

Command (login node, light, ~20 s; script `apps/synthetic/ring_strata.py`):

    OMP_NUM_THREADS=1 nice -n 10 uv run python -m apps.synthetic.ring_strata

It reads the finished true-decoder (condition A) stage-3 checkpoints in
`ring_recovery/l23_d3/stage3/ckpt` (original-init `tnbbeta`, `vmf`, `power_spherical`,
seeds 0-9) and the 2000 stage-3 test observations (generator seed 424242). Per-x KL is
`exact_kl` with 500 draws. "Aligned" TNBBeta seeds (2, 5, 6, 7, 9) are the ones with
|cos(mu, a)| >= 0.999 in `ring_posterior_l23_d3_2026-11-01.md`; "other" are 0, 1, 3, 4, 8.

## What the exact posteriors look like (2000 test x, sigma = 0.2)

* Latitude spread of the exact posterior (sd of the polar angle t): 2.4 / 5.3 / 13.3 degrees
  at the 5 / 50 / 95th percentile. Thin rings, as designed.
* **Not all posteriors are single rings.** The features `sin 3w`, `cos 5w` are periodic, so
  a second latitude mode appears: 27.2% of x have a second mode with > 1% mass, **18.9%
  with > 5%, 10.6% with > 20%** (max 0.5). For the x with second mode > 5% the median
  angular separation between the two biggest modes is about 24.5 degrees (10-90th percentile
  15-46). The exact posterior there is a mixture of two rings; no single TNBBeta, vMF or PS
  member can represent it.
* 5.4% of x have > 50% of their posterior mass within 20 degrees of a pole: a cap, not a
  ring.
* Strata used below: single ring 1523, two-mode 368, polar cap 109 (two-mode excludes polar).
* Level 1 used noiseless x* = g(cos t*), whose posterior is a single ring by construction,
  so L1 did not see this.

## Mean exact KL(q || p_true) by stratum (condition A, 10 seeds each)

| family | all | single ring | two-mode | polar cap |
|---|---|---|---|---|
| vMF | 2.70 | 2.85 | 2.75 | 0.50 |
| PowerSpherical | 2.78 | 2.89 | 2.85 | 1.05 |
| TNBBeta (original init), all seeds | 1.08 | 1.14 | 0.96 | 0.62 |
| TNBBeta, aligned seeds (2,5,6,7,9) | | **0.12** | 0.60 | 0.99 |
| TNBBeta, other seeds (0,1,3,4,8) | | 2.16 | 1.32 | 0.26 |

## Reading

* Where the exact posterior is a single ring and TNBBeta's mean direction is aligned with the
  axis, it recovers it almost exactly: KL 0.12 against 2.85 (vMF) and 2.89 (PS). A defect in
  the density or sampler would not allow this.
* Half the seeds (0, 1, 3, 4, 8) do not align mu with the axis (|cos| 0.63-0.72 in the
  earlier note) and stay at KL 2.16 on single-ring x, about the cap level. This, not
  representability, is the L3 condition-A shortfall. The earlier fixed-eps_ring variant
  (eps pinned near 123) aligned mu on every seed.
* The two-mode and polar x are not recoverable by a single ring. On the aligned seeds
  the 18% two-mode x contribute 44% of the total KL.
* Hypothesis, not tested: the 24% of x with TNBBeta KL > 3 in L2 is close to two-mode + polar
  (18.9% + 5.4%); stratifying the L2 per-x KL (needs the per-x values, not saved) would check it.

## Caveats

Strata use the latitude density on a 4001-point grid; "second mode" is a segment between
local minima with > 5% mass. Test x are float32 in the checkpoints' evaluation and float64 in
the stratification. One task, one axis, d = 3.
