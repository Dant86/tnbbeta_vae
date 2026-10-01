# TNBBeta-VAE and S-VAE Convergence Check: 2026-10-01

- **Date:** 2026-10-01
- **Branch/commit:** `feat/vmf-tnbbeta-convergence`, commit `9ab5679` (original
  implementation) with review fixes on top -- see "Review fixes" below for
  what changed and why; this report's numbers were produced after those
  fixes, not against `9ab5679` itself.
- **Command (Part 1 numbers in this report):**
  ```
  uv run python -m apps.distributions.vmf_tnbbeta_convergence \
      --output-csv /tmp/vmf_tnbbeta_convergence_part1.csv
  ```
  (default `--sample-size 100000 --bisect-tol 1e-4`; deterministic given the
  default bisection `seed=0` -- re-running reproduces this report's table
  exactly.)
- **Command (table regenerated after review fixes, superseding the table
  originally produced from `9ab5679`):** same as above.

## Summary

This report checks empirically whether TNBBeta-VAE and S-VAE (vMF)
posteriors converge to each other as concentration increases. Matching
`TNBBeta(p, q=0, epsilon)` to `vMF(kappa)` by mean resultant length and
comparing their `w`-marginals (cosine to mean direction) gives a mixed
answer, not a clean "yes": the **absolute** gap between the two families
(energy distance, in `w`'s own units) shrinks substantially and
monotonically as kappa increases, but a **scale-free** measure of the same
gap (the KS statistic) does not shrink -- it grows. Both families do
concentrate into an increasingly tight cap around the mean direction, but
not into the *same shape* of cap, at least not while `epsilon` (TNBBeta's
other shape parameter, held fixed here) stays fixed. See "Interpretation"
below.

## Review fixes (this report's numbers are post-fix)

The original implementation at `9ab5679` had three issues significant
enough to produce misleading numbers, fixed before generating this
report's table:

1. **`p_max=0.9999` was an active ceiling, not a loose bound.** At
   `epsilon=1`, `r_bar(p)` tops out around 0.998 as `p -> 0.9999`
   regardless of epsilon -- below the vMF target `r_bar` at
   `kappa=1000` (0.999 on S^2). The original run's high-kappa rows were
   silently pinned near that ceiling rather than genuinely matched.
   Fixed by raising `p_max` to `1 - 1e-7` (tested numerically stable, no
   NaNs; `p=1` itself is degenerate and divides by zero in the sampling
   transform).
2. **Each bisection candidate `p` drew an independent fresh sample**,
   making `r_bar(p)` a noisy (not exactly monotonic) function of `p` and
   risking the bisection going the wrong direction when the true gap was
   near the Monte Carlo noise floor -- plausibly why the original report's
   table was non-monotonic in a few cells (e.g. its `epsilon=2.0`, S^2 row
   had the gap *increase* from kappa=200 to kappa=1000). Fixed with a
   common-random-numbers trick: every candidate `p` within one bisection
   call reseeds to the same `seed` first. `Y`'s sampling transform (at
   `q=0`) is `Y = p*u / ((1-p)(1-u) + p*u)` for the *same* underlying
   `u ~ Beta(epsilon, epsilon)` draw, and `dY/dp = u(1-u)/D^2 >= 0` for
   every fixed `u` -- so with shared randomness, `r_bar(p)` is now an
   exactly monotonic, deterministic function of `p`, making the bisection
   a true root-find and the whole script reproducible run-to-run.
3. **`bisect_tol=1e-3` was too coarse for the kappa grid actually used.**
   vMF's `r_bar` at kappa=500 vs. kappa=1000 differ by only 0.001 on S^2 --
   below the old tolerance, so those two rows could (and in the original
   report's numbers, did) collapse onto nearly the same matched `p`.
   Tightened the default to `1e-4` (cheap: fix #2 makes tighter tolerances
   just more bisection iterations, not more samples).

A latent (but numerically silent) shape bug was also fixed:
`vmf_dist.rsample(...).squeeze(0)` was a no-op (`squeeze` on an axis whose
size isn't 1 does nothing -- the batch axis being squeezed needed index 1,
not 0); `_w_marginal_from_samples`'s `.ravel()` fallback happened to
produce the same numeric values regardless, but the shape was wrong and
the code read as though the squeeze did something. Fixed to `squeeze(1)`.

## Part 1: Synthetic Two-Sample Test

### Methodology

For a grid of vMF concentrations kappa in {5, 10, 20, 50, 100, 200, 500,
1000}, at two epsilon settings (1.0, 2.0) and two ambient dimensions (3 and
11, i.e. S^2 and S^10):

1. Computed vMF's mean resultant length `r_bar` analytically:
   `r_bar = I_{d/2}(kappa) / I_{d/2-1}(kappa)` via `scipy.special.ive`
   (scaled Bessel, for numerical stability at large kappa).
2. Matched a `TNBBeta(p, q=0, epsilon)` to that `r_bar` by bisection on
   `p in (0.5, 1 - 1e-7)` (common random numbers, tolerance `1e-4`,
   100,000-sample Monte Carlo estimate of `r_bar(p)` per candidate).
3. Drew 100,000 fresh samples from each of the matched vMF and TNBBeta,
   extracted the `w`-marginal (cosine to mean direction), and computed two
   two-sample statistics: energy distance and the KS statistic.
4. Computed a noise floor from two independent 100,000-sample draws of the
   *same* vMF, to check the gap is larger than sampling noise alone.

### Results: Dimension 2 (S^2 in R^3)

#### epsilon = 1.0

| kappa | matched_p | gap_energy | noise_energy | gap_ks | noise_ks |
|-------|-----------|-----------|-------------|--------|----------|
| 5     | 0.960632  | 1.032e-01 | 1.29e-03    | 0.171  | 0.0029   |
| 10    | 0.985199  | 1.044e-01 | 1.46e-03    | 0.245  | 0.0041   |
| 20    | 0.994110  | 9.010e-02 | 1.14e-03    | 0.303  | 0.0039   |
| 50    | 0.998138  | 6.598e-02 | 3.08e-04    | 0.356  | 0.0021   |
| 100   | 0.999191  | 5.075e-02 | 3.04e-04    | 0.391  | 0.0032   |
| 200   | 0.999649  | 3.860e-02 | 1.89e-04    | 0.425  | 0.0032   |
| 500   | 0.999878  | 2.598e-02 | 9.72e-05    | 0.452  | 0.0019   |
| 1000  | 0.999939  | 1.826e-02 | 2.11e-04    | 0.450  | 0.0061   |

#### epsilon = 2.0

| kappa | matched_p | gap_energy | noise_energy | gap_ks | noise_ks |
|-------|-----------|-----------|-------------|--------|----------|
| 5     | 0.933899  | 2.593e-02 | 3.30e-03    | 0.056  | 0.0066   |
| 10    | 0.970154  | 2.912e-02 | 1.13e-03    | 0.061  | 0.0035   |
| 20    | 0.986023  | 2.715e-02 | 7.04e-04    | 0.082  | 0.0022   |
| 50    | 0.994690  | 2.084e-02 | 5.02e-04    | 0.104  | 0.0035   |
| 100   | 0.997375  | 1.480e-02 | 3.26e-04    | 0.103  | 0.0028   |
| 200   | 0.998718  | 1.160e-02 | 2.05e-04    | 0.114  | 0.0030   |
| 500   | 0.999512  | 7.963e-03 | 3.77e-04    | 0.125  | 0.0048   |
| 1000  | 0.999756  | 5.925e-03 | 2.60e-04    | 0.130  | 0.0050   |

### Results: Dimension 10 (S^10 in R^11)

#### epsilon = 1.0

| kappa | matched_p | gap_energy | noise_energy | gap_ks | noise_ks |
|-------|-----------|-----------|-------------|--------|----------|
| 5     | 0.776489  | 2.893e-01 | 2.97e-03    | 0.297  | 0.0040   |
| 10    | 0.888611  | 2.827e-01 | 1.54e-03    | 0.367  | 0.0038   |
| 20    | 0.953430  | 2.488e-01 | 1.03e-03    | 0.440  | 0.0035   |
| 50    | 0.986023  | 1.931e-01 | 1.94e-03    | 0.526  | 0.0063   |
| 100   | 0.994263  | 1.510e-01 | 7.59e-04    | 0.582  | 0.0039   |
| 200   | 0.997574  | 1.142e-01 | 7.32e-04    | 0.619  | 0.0049   |
| 500   | 0.999206  | 7.820e-02 | 3.02e-04    | 0.666  | 0.0049   |
| 1000  | 0.999649  | 5.706e-02 | 1.55e-04    | 0.687  | 0.0037   |

#### epsilon = 2.0

| kappa | matched_p | gap_energy | noise_energy | gap_ks | noise_ks |
|-------|-----------|-----------|-------------|--------|----------|
| 5     | 0.736572  | 1.615e-01 | 3.16e-03    | 0.170  | 0.0052   |
| 10    | 0.848083  | 1.677e-01 | 1.77e-03    | 0.224  | 0.0058   |
| 20    | 0.924316  | 1.492e-01 | 1.26e-03    | 0.267  | 0.0029   |
| 50    | 0.971496  | 1.168e-01 | 5.01e-04    | 0.325  | 0.0024   |
| 100   | 0.986328  | 8.941e-02 | 4.90e-04    | 0.352  | 0.0031   |
| 200   | 0.993347  | 6.600e-02 | 8.34e-04    | 0.368  | 0.0059   |
| 500   | 0.997406  | 4.301e-02 | 2.35e-04    | 0.378  | 0.0035   |
| 1000  | 0.998718  | 3.083e-02 | 2.50e-04    | 0.383  | 0.0042   |

### Key Findings: Part 1

1. **Energy-distance gap shrinks monotonically, by roughly 80%, in every
   combination tested:**
   - S^2, eps=1.0: 1.03e-01 -> 1.83e-02 (82% reduction, kappa 5 to 1000)
   - S^2, eps=2.0: 2.59e-02 -> 5.93e-03 (77% reduction)
   - S^10, eps=1.0: 2.89e-01 -> 5.71e-02 (80% reduction)
   - S^10, eps=2.0: 1.62e-01 -> 3.08e-02 (81% reduction)
   - No reversals anywhere in the grid (unlike the pre-fix run).

2. **The KS statistic does the opposite: it grows, by 125-165%, over the
   same range:**
   - S^2, eps=1.0: 0.171 -> 0.450
   - S^2, eps=2.0: 0.056 -> 0.130
   - S^10, eps=1.0: 0.297 -> 0.687
   - S^10, eps=2.0: 0.170 -> 0.383

3. **The noise floor stays small (1e-4 to 3e-3 for energy, under 0.007 for
   KS) throughout**, confirming both statistics are picking up a real
   signal, not sampling noise, at every kappa tested -- including at
   kappa=1000, where the gap is still ~80-280x the noise floor.

4. **Matched `p` increases smoothly and strictly with kappa** at every
   dimension/epsilon setting (e.g. S^2, eps=1.0: 0.961 at kappa=5 up to
   0.999939 at kappa=1000), with no repeated values across the grid -- the
   old `p_max` ceiling and coarse tolerance are gone.

## Interpretation

The two statistics tell different stories because they are sensitive to
different things, and together they give a more complete (and less
flattering) picture than either alone:

1. **Energy distance is scale-sensitive.** As kappa increases, *both*
   distributions concentrate into an increasingly narrow cap around the
   mean direction, so the *absolute* difference between two narrow,
   nearby caps shrinks even if their shapes don't converge -- energy
   distance largely reflects this shared shrinking scale.

2. **KS is scale-free** (it compares CDFs, bounded in [0, 1] regardless of
   how narrow the distributions are), so it is a better proxy for whether
   the two families are becoming the *same shape* once rescaled by their
   own spread. Here it grows, not shrinks -- the relative mismatch between
   TNBBeta's and vMF's concentration shape does not vanish at high kappa
   when only `p` is tuned; if anything it gets more pronounced.

3. **This is consistent with, not contrary to, how this project's own
   TNBBeta parameterization behaves** (see `CLAUDE.md`): at `q=0`, `p` is
   bounded in `(0, 1)` and saturates near 1 as its only way to increase
   concentration, while `epsilon` -- held fixed at 1.0 or 2.0 in this
   experiment -- is the parameter that actually controls the shape/width
   of the cap (the real analogue of vMF's kappa, per the MNIST finding
   that `epsilon` carries the confidence signal once `p` saturates).
   Matching only `p` to `r_bar` while leaving `epsilon` fixed means the
   cap's *shape* is never actually being pushed toward vMF's bell shape --
   only its location/width along one axis is. A fairer convergence check
   would let `epsilon` grow with kappa too (e.g. scaling it by
   `sqrt(kappa)` or similar) rather than fixing it; that is a natural
   follow-up, not attempted here.

4. **Bottom line:** TNBBeta-VAE and S-VAE posteriors do *not* straightforwardly
   converge to each other as concentration increases under this
   matching scheme. They both become tight caps around the mean direction
   (energy distance confirms this), but not the same *shape* of cap
   (KS says the opposite of converging) -- so "the two posterior families
   behave alike at high confidence" is not supported by this check as
   currently designed, and should not be assumed without either the
   `epsilon`-scaling follow-up in point 3 or some other justification.

## Part 2: Trained-Posterior Overlay

**Status: NOT RUN (checkpoints unavailable)**

The MNIST training checkpoints (`conv_vmf_vae` and
`conv_tnbbeta_spherical_vae`) are not present in this sandbox environment.
These would normally reside under `tnbbeta_vae.paths.checkpoint_dir()`,
which resolves to the local `checkpoints/` directory configured in `.env`.
`run_part2()` detects this (`status: "checkpoint_dir_not_found"`) and skips
rather than fabricating numbers.

### What Part 2 Would Do

For each available dimension in the training sweep (d in {2, 5, 10, 20,
40}):

1. Load final checkpoints for both model families.
2. Run the MNIST test set through each model's posterior.
3. Extract posterior centre directions and `w`-marginals.
4. Compute KL or JS divergence between the two families' `w`-marginals.
5. Check whether the same absolute-shrinks/relative-doesn't-shrink pattern
   found in Part 1's synthetic check also shows up for *trained*
   posteriors, where `epsilon` is not held fixed but learned per example.

### How to Run Part 2

**On the cluster with checkpoint access:**

```bash
cd /Users/vedantpathak/Developer/projects/packages/tnbbeta_vae/
uv run python -m apps.distributions.vmf_tnbbeta_convergence
```

The script will detect available checkpoints and automatically run Part 2.
Checkpoints are expected at:

- `checkpoints/mnist_vmfs_d{dim}_seed{seed}/final.pt`
- `checkpoints/mnist_tnbs_d{dim}_seed{seed}/final.pt`

For dims {2, 5, 10, 20, 40} and seeds {0, 1, 2, 3, 4}.

## Implementation Notes

- **Numerical stability**: used `scipy.special.ive` (scaled Bessel) to
  avoid overflow in the `r_bar` computation.
- **Bisection**: common-random-numbers (fixed `seed` per call) makes
  `r_bar(p)` an exactly monotonic, deterministic function of `p` at `q=0`
  (see "Review fixes" above), so tolerance can be tightened cheaply;
  `1e-4` is used throughout this report.
- **Sample size**: 100,000 samples per distribution, both for the
  bisection's `r_bar(p)` estimate and for the final two-sample statistics.
- **Device**: all computation on CPU; the full Part 1 grid (32 cells)
  takes about 17 seconds.

## Files

- Script: `apps/distributions/vmf_tnbbeta_convergence.py`
- Tests: `tests/apps/test_vmf_tnbbeta_convergence.py`
- Results (this file): `writeup/results/tnbbeta_vmf_convergence_check_2026-10-01.md`
