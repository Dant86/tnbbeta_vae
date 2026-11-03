# Does TNBBeta's ring-posterior advantage scale with the ambient dimension? Pre-registration

Status: **FROZEN** (2026-11-02, before any run of the new code). Later changes are appended under "Deviations" only.
Base commit: `6b1880f` (branch `synthetic-graph-families`) plus the uncommitted working tree (new files only, listed under "Code").
Prior studies this builds on (not redone): `ring_class_designs_learned_decoder_results_2026-11-01.md` (S^2, learned free decoder: TNBBeta
did NOT beat vMF/PS; with the true or a forced single-index decoder it did), `ring_posterior_recovery_2026-11-01.md`.

## Question

With `x = h(a . z) + noise`, `z ~ Uniform(S^(d-1))`, the exact posterior is a thin latitude **shell** (a copy of S^(d-2) at one latitude
`w = a . z`). A cap-shaped posterior (vMF, PowerSpherical, TNBBeta with q at its floor) cannot represent the shell and must pay
roughly `(d-2) log(1/s)` nats of entropy (s = shell thickness relative to its radius). **Hypothesis H (not yet tested, may be false):**
the gap between ring-capable and cap-only families grows with the ambient dimension `d`. A free learned decoder may fold the latent to
compensate (as it did at d=3), so a null or negative result is legitimate and will be reported as such.

## Task

* Ambient dimension `d in {3, 5, 9, 17}` (sphere `S^(d-1)`; manifold dimensions 2, 4, 8, 16). `d` is the ambient dimension throughout, as for
  every sphere model in this repo (CLAUDE.md).
* Axis `a = e_1`; `z ~ Uniform(S^(d-1))`; `x in R^3`: `x = h(w) + N(0, 0.2^2 I)`, `w = a . z`, `h(w) = [w, sin 1.5 w, cos 2 w]` (the gentle map
  of the four-class study). `x` is deliberately low-dimensional so that it identifies only the latitude `w`.
* Exact posterior: `p(z | x) proportional to N(x; h(w), sigma^2 I)` on the sphere; the density of `w` is `proportional to N(x; h(w), sigma^2)
  (1-w^2)^((d-3)/2)` and the azimuth (the direction of the part of `z` orthogonal to `a`) is uniform on `S^(d-2)`.
* Exact `log p(x)`: 1-D quadrature in the polar angle `t = arccos w` (weight `sin^(d-2) t`) with 4001 midpoint nodes (the existing
  `RingPosteriorTask` convention; the Beta-type endpoint singularity of `(1-w^2)^((d-3)/2)` at `d = 3, 5` disappears in the `t` variable,
  where the integrand is smooth). The test suite checks normalisation of the latitude density, agreement of `log p(x)` with a
  40001-node rule (< 1e-7 nats), with Monte Carlo `E_U[N(x; h(a.z))]`, and (d = 3) agreement with the existing `RingClassTask` S^2 code.
  Exact `KL(q || p(. | x))` is Monte Carlo over `q` draws (2000/x, exact `log q` for TNBBeta) against the exact `log p(z | x)`.
* Data: 20,000 train `x` (generator seed `100 + seed`), 2000 test `x` (generator seed 424242, same for all families at a given `d`).

## Conditions

* **A (true frozen decoder; confirmatory family A).** Decoder `h(a . z)` and sigma = 0.2 fixed, only encoder + head trained. Here the ELBO
  gap equals `KL(q || p(. | x))` in expectation, so this is the "does the advantage grow with d when the decoder is correct" reference.
* **B (real VAE; confirmatory family B).** Free MLP decoder `[d, 128, 128, 3]` (ReLU), one learned sigma (`LearnedLikelihoodScale`, start 0.2).
* **C (descriptive, no tests).** Learned single-index decoder `x_hat = g(u . z)` (`u` a learned axis in R^d, `g` a tanh MLP 1-32-32-3, one learned
  sigma) as in `ring_classes_learned.py`.
* Encoder in every condition: MLP `[3, 128, 128]` (ReLU) then one linear posterior head sized for `latent_dim = d`.

## Families (arms)

Same initial encoder/decoder weights, data, and shuffling stream across arms for a seed (the head is built last, as in the earlier study).

| arm | status | description |
|---|---|---|
| `vmf` | confirmatory comparator | vMF head, **dimension-aware initial kappa = d** (raw kappa bias set so `kappa = d`; CLAUDE.md) |
| `power_spherical` | confirmatory comparator | PowerSpherical head, initial kappa = d (same convention; see Deviation-by-design note) |
| `tnbbeta` | confirmatory | TNBBeta, library defaults (`P_RAW_INIT` p bias, `epsilon_floor = 1e-4` as in the earlier study), exact `log q` |
| `tnbbeta_init0` | descriptive | as `tnbbeta`, raw p bias 0 (the arm that sometimes found the ring at d = 3) |
| `tnbbeta_epsinit` | descriptive | as `tnbbeta`, epsilon bias set so the initial `epsilon = (d-1)/2` (the uniform prior's value: the TNBBeta analogue of kappa = d) |
| `vmf_untuned` | descriptive | vMF, library default initialisation (the untuned control) |
| `power_spherical_untuned` | descriptive | PowerSpherical, library default initialisation |

Deviation-by-design note: the request named `power_spherical` without an initial kappa; the confirmatory PS arm gets the same
dimension-aware initial kappa (d) as vMF so the confirmatory comparators are both tuned and any win is not a tuning artefact; the untuned
PS arm is kept as a descriptive control. All arms are on all of A, B (7 arms each), and C runs only `vmf`, `power_spherical`, `tnbbeta`,
`tnbbeta_init0`.

## Training

Adam 1e-3, batch 256, 300 epochs, KL warm-up over the first 10% of epochs (linear), 4 reparameterised ELBO draws per example, analytic KL to the
uniform sphere for vMF / PS and Monte Carlo with exact `log q` for TNBBeta (`monte_carlo_elbo`, `sample_with_log_prob`). 10 seeds (0-9).
A run that produces a non-finite loss is "diverged"; diverged cells are reported, never silently dropped (the paired test then uses the
seeds valid in both arms and the count is reported).

## Metrics (per run, on the 2000 test `x`, float64, CPU)

1. **Primary: test ELBO gap** `mean_x[log p_ref(x) - ELBO(x)]`, ELBO from 256 posterior draws per `x` (decoder log-likelihood minus KL; analytic KL
   for vMF / PS, `mean(log q + log area(S^(d-1)))` for TNBBeta), `log p_ref` the exact quadrature evidence.
2. IWAE gap with K = 64 (`importance_weighted_metrics`, stub uniform prior).
3. Condition A: `KL(q || p(. | x))` by Monte Carlo (2000 draws) against the exact posterior (also reported as a cross-check of the ELBO gap).
4. Learned sigma versus 0.2 (B, C).
5. **Ring statistic**: for the 256 draws per `x`, project onto the orthogonal complement of the axis `u`, normalise, take the norm of the
   mean over draws (mean resultant length R; ~0 = uniform azimuth = ring/shell, ~1 = cap), median over `x`. `u` is `e_1` (A), the learned
   `u` (C), and for B the leading eigenvector of the average `E_z[J^T J]` of the decoder Jacobian (the "active-subspace" axis; for a single-index decoder this is
   exactly its axis); its eigenvalue fraction `lambda_1 / trace` is reported as a single-indexness measure. Also mean `|centre . u|`.
6. TNBBeta shape: median `p`, `q`, `epsilon`, `m = epsilon - (d-1)/2` over `x` (kappa median for vMF / PS).
7. Descriptive context per `d` (once, no training): the mutual information `I(z; x) = E[log p(x|z) - log p(x)]` (MC over 20,000 draws) and
   the exact `E KL(Uniform || p(. | x))` (a "no-information" baseline), to show how informative `x` is about the latitude at each `d`.

## Confirmatory tests

Two separate families (one per condition A and B) of 8 tests each: for each `d in {3,5,9,17}` and each comparator in {`vmf`, `power_spherical`},
a one-sided paired Wilcoxon signed-rank test over the 10 seeds on `diff = gap(comparator) - gap(tnbbeta)`, H1: `tnbbeta` has the smaller gap
(`diff > 0`). Holm correction at 0.05 within each family of 8. A comparison is a **confirmed advantage** only if it survives Holm AND the
mean paired difference is at least **0.05 nats**. Every p-value is verified by sign-pattern enumeration (2^n patterns) and must equal
scipy's exact value (a mismatch is reported). With n = 10 the smallest attainable one-sided p is 1/1024, below Holm's first threshold
0.05/8 = 0.00625, so rejection is possible only when TNBBeta wins on (nearly) all seeds. Pairing is by seed.

## Descriptive trend (not in the Holm family, no p-value claimed as confirmatory)

For each condition and comparator: the paired difference `gap(comparator) - gap(tnbbeta)` against `d`: mean +- sd per `d`; the per-seed
OLS slope of the difference on `d` (10 slopes: mean, sd, a seed-bootstrap 95% interval), and the Spearman correlation of the difference
with `d` pooled over the 40 (seed, d) points. Seeds are not data-paired across `d` (different `z` draws), so these are descriptive of the
trend only. A positive slope is the direction of H; the sign and size are reported whatever they are.

## Known risks and caveats (recorded before running)

* **The shell is not thin at large d relative to the prior.** The prior std of `w` is `1/sqrt(d)` (0.24 at d = 17) while the posterior std of
  `w` is about `0.2 / |h'(w)|` with `|h'| >= 1`; so at d = 17 `x` carries comparatively little information about `w` (reported by the
  mutual-information diagnostic), the posterior is a broad latitude band not a sharp shell, and a cap's entropy loss is then smaller than the
  `(d-2) log(1/s)` heuristic suggests. A null trend at large `d` can come from this and not from the families.
* **Kappa scaling.** A vMF / PS posterior needs `kappa` of order `d` to be informative at high `d`; initial kappa = d is used for the confirmatory
  arms and the untuned arms are descriptive controls. TNBBeta's `epsilon` threshold for bimodality moves as `(d-1)/2`, so the library-default
  initialisation (`epsilon` near 0.7 at init) is far from the prior's value at `d = 17`; `tnbbeta_epsinit` probes that, descriptively.
* **Free decoder folding** (B): as at d = 3 the MLP may use the latent in a way that makes a cap sufficient; the ELBO gap of TNBBeta was ~0.1
  nats worse at d = 3 and may stay worse.
* **Different estimators**: the TNBBeta ELBO uses Monte Carlo `log q`, the others an analytic KL (noise, not bias; 256 eval draws).
* **Sampler**: TNBBeta samples in logit space (`rsample_with_log_prob`), exact at small epsilon; vMF's rejection sampler at high `d`/large
  kappa is not changed here. float64 evaluation on CPU.
* Latent identifiability: B and C latents are identified only up to rotation; ring statistics use the learned axis.
* Single task (one `h`, one `sigma`, one `a`); no claim beyond it.

## Code (all new, additive)

`src/tnbbeta_vae/data/ring_dimension.py` (task, subclass of `RingPosteriorTask`), `apps/synthetic/ring_dimension_scaling.py`,
`apps/synthetic/ring_dimension_scaling_plot.py`, `scripts/slurm/ring_dimension_scaling.sbatch`, `tests/data/test_ring_dimension.py`,
`tests/apps/test_ring_dimension_scaling.py`. No existing file is changed. Outputs: `ring_recovery/dimension_scaling/<cond>/d<d>/<arm>/s<seed>.{json,npz,pt}`.

## Deviations

(none yet; appended below, with dates and reasons, if any)

### D1 (2026-11-02, before any run): quadrature grid
A test at d = 3 showed 4001 midpoint nodes in `t` differ from a 40001-node rule by up to 6e-7 nats (the midpoint rule's endpoint term at
d = 3), above the planned 1e-7 check. The task therefore uses **16001 nodes** (check: < 1e-7 vs 40001 nodes at every d). The d = 3
reproduction test against the existing `RingClassTask` uses the same 4001 nodes as that code. Nothing else changes.

### D2 (2026-11-02, before any full run): exact vMF sampler
Smoke tests showed an IWAE gap below zero for vMF at d = 17, which is impossible for a correct bound. Cause (checked by comparing the mean cosine
of 200,000 draws with the Bessel-ratio theory): the library's `VonMisesFisher` rejection sampler (a port of the S-VAE reference code) replaces the
envelope parameter `b` by the Taylor value `(m-1)/(4 kappa)` for kappa > 11 while `a`, `d` keep the exact-`b` formulas, so draws are biased for kappa of
about 10-40 at m >= 9 (m = 17, kappa = 17: E[cos] 0.608 vs theory 0.627; m = 33, kappa = 17: 0.351 vs 0.426). It is exact for m = 3 (closed form) and
for large kappa. The PowerSpherical sampler matches theory. The library is NOT changed; this study's vMF posterior is a local subclass
(`ExactVonMisesFisher`, `b = (m-1)/(c+2 kappa)`, exact) used for training and evaluation, so the vMF arms are not penalised by sampler bias.
The `vmf_untuned` and `vmf` arms both use it. (Any earlier library-sampler vMF result at m >= 9 with kappa in 10-40 inherits that bias.)
