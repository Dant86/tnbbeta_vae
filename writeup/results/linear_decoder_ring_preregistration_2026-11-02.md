# Sphere probabilistic PCA with a LINEAR decoder: does TNBBeta find the ring? Pre-registration

Status: **FROZEN** (2026-11-02, before any run of the new code). Later changes are appended under "Deviations" only.
Base commit: `6b1880f` (branch `synthetic-graph-families`) plus the uncommitted working tree (new files only, listed under "Code").
Prior studies this builds on (not redone): `ring_class_designs_learned_decoder_results_2026-11-01.md` (S^2, learned free MLP decoder: TNBBeta did
NOT beat vMF/PS; forced single-index tanh decoder: TNBBeta won on 10/10 seeds, ring-forcing by construction) and the dimension-scaling study
(`ring_dimension_scaling_preregistration_2026-11-02.md`, run by another agent, which already covers the *true frozen* decoder at several d;
that condition is NOT duplicated here).

## Question

The sphere analogue of probabilistic PCA: `z ~ Uniform(S^(d-1))`, `y = a . z + eta`, `eta ~ N(0, sigma^2)`, `y` a scalar, `a` a fixed unit vector
(`a = e_1`; WLOG because the encoder's first layer is rotation-symmetric in distribution only through `y`, which is one-dimensional, so the
choice of `a` is immaterial). With `w = a . z` the exact posterior is `p(z | y) proportional to N(y; w, sigma^2)` on the sphere: the density of `w` is
`proportional to N(y; w, sigma^2) (1 - w^2)^((d-3)/2)` on `[-1, 1]` and the azimuth (the direction of `z` orthogonal to `a`) is uniform on `S^(d-2)`:
a latitude ring about `a` of thickness about `sigma`. With a strictly **linear** learned decoder `y_hat = u . z` (`u` a learned unit vector) and a
learned scalar `sigma_hat`, the model cannot fold the latent the way MLP / tanh single-index decoders did: its *own* posterior
`p_model(z | y) proportional to N(y; u . z, sigma_hat^2)` is necessarily a latitude ring about the learned axis `u`. So the model class contains the
truth and the model's own posterior is a ring. Questions: (1) with this minimal learned decoder, does a VAE with a TNBBeta posterior reach a better
bound than vMF / PowerSpherical (PS)? (2) does it find the ring (azimuthal spread) and the axis (`u` close to `+-a`)? (3) how does any advantage
move with `d`? Null and negative results are legitimate and will be reported as such. No hypothesis is assumed true.

## Task

* `d in {3, 5, 9, 17}` ambient dimension (sphere `S^(d-1)`, manifold dimension 2, 4, 8, 16), as everywhere in this repo.
* `sigma in {0.1, 0.2, 0.4}`. Rationale: `w` has standard deviation `1/sqrt(d)` under the prior (0.58, 0.45, 0.33, 0.24 for the four `d`), so
  `sigma = 0.2` (the value of the earlier ring studies) is moderate at d = 3 and comparable to the prior spread at d = 17; `0.1` is a thin ring (high
  information, where a cap pays the most for its missing azimuth) and `0.4` a thick one (low information at large `d`: at `d = 17` the posterior is
  nearly the prior, and an uninformative-latent solution is nearly optimal). The grid is chosen a priori for this spread of information, not tuned.
  The exact information `I(z; y)` (nats) is reported per cell, because a run whose gap approaches it has collapsed to the prior.
* Data: 20,000 training `y` (generator seed `100 + seed`), 2000 test `y` (generator seed 424242, shared by all arms at a `(d, sigma)`).
* Exact evidence `log p_true(y) = log E_w N(y; w, sigma^2)` by 1-D quadrature in the polar angle `t = arccos w` (weight `sin^(d-2) t`), 16001
  midpoint nodes (the dimension-scaling study's grid; the endpoint singularity at `d = 3, 5` vanishes in `t`). The same routine with the learned
  `sigma_hat` gives `log p_model(y)` for the linear decoder (and, with the link function in place of the identity, for the 1-D decoder). The tests check
  normalisation, agreement with a 64001-node rule (< 1e-7 nats), Monte Carlo agreement and `d = 3` (where `w` is uniform) against a direct integral.
  The quadrature helpers `latitude_grid` and `sphere_log_area` are imported from `tnbbeta_vae.data.ring_posterior`; the dimension-scaling study's
  `RingPosteriorTask` is a 3-feature observation model and is not reused (a scalar-observation module is added; no file of that study is edited).

## Decoders and conditions

Encoder in every condition: MLP `[1, 128, 128]` (ReLU) then one linear posterior head sized for `latent_dim = d`. One learned scalar `sigma_hat`
(`LearnedLikelihoodScale`, log-variance parametrised), **starting at 0.2 in every cell** (the true value only at `sigma = 0.2`; deliberately not an
oracle at the other two, so the learned-`sigma` recovery is a real test).

| decoder | role | mean function |
|---|---|---|
| `L` linear | **confirmatory** | `y_hat = u . z`, `u = raw / ||raw||`, `raw ~ N(0, I_d)`, **no bias** (matches the DGP) |
| `Lb` linear + bias | descriptive, sigma = 0.2 only | `y_hat = u . z + b`, `b` learned scalar initialised at 0 |
| `M` 1-D MLP | descriptive, sigma = 0.2 only | `y_hat = g(u . z)`, `g` tanh MLP `[1, 32, 32, 1]` (the single-index decoder of the earlier study with scalar output) |
| `F` free MLP | descriptive, sigma = 0.2 only | `y_hat = f(z)`, ReLU MLP `[d, 128, 128, 1]` (a scalar-output re-test of the earlier free-decoder result) |

## Arms

Same initial encoder/decoder weights, data and shuffling stream across arms for a seed (the head is built last). Paired by seed.

| arm | status | description |
|---|---|---|
| `vmf` | confirmatory comparator | vMF head, dimension-aware **initial kappa = d**; the repo's exact-envelope vMF sampler (see Risks) |
| `power_spherical` | confirmatory comparator | PS head, initial kappa = d |
| `tnbbeta` | confirmatory | TNBBeta, library defaults (`P_RAW_INIT` p bias, `epsilon_floor = 1e-4`), exact `log q` |
| `tnbbeta_init0` | descriptive | as `tnbbeta`, raw p bias 0 |

## Training

Adam 1e-3, batch 256, 300 epochs, KL warm-up linear over the first 10% of epochs, 4 reparameterised ELBO draws per example, analytic KL to
`Uniform(S^(d-1))` for vMF / PS and Monte Carlo with exact `log q` for TNBBeta (`monte_carlo_elbo`); float32 on GPU, evaluation float64 on CPU.
10 seeds (0-9). A run with a non-finite loss is "diverged"; diverged cells are reported (never silently dropped) and the paired test uses the
seeds valid in both arms (stated; fewer than 5 pairs => no test).

## Grid

* `L`: `d` (4) x `sigma` (3) x 4 arms x 10 seeds = 480 runs (confirmatory).
* `Lb`, `M`, `F`: `d` (4) x `sigma = 0.2` x 4 arms x 10 seeds = 160 runs each (descriptive).
* Total 960 runs. Cells are queued in the order L, F, M, Lb, so a shortfall of compute truncates the descriptive contrasts first.

## Metrics (per run, 2000 test `y`, float64 CPU)

1. **Primary: test ELBO gap** `mean_y[log p_true(y) - ELBO(y)]`, ELBO from 256 posterior draws per `y` (log-likelihood minus KL; analytic KL
   for vMF / PS, `mean(log q + log area(S^(d-1)))` for TNBBeta). Also its standard error over `y`.
2. IWAE gap, K = 64.
3. **Exact decomposition** (decoders L, Lb, M, where `p_model(y)` has the 1-D quadrature): `gap = [log p_true - log p_model](y) + KL(q || p_model(. | y))`;
   both terms reported (`model_misfit`, `kl_to_model_posterior = log p_model(y) - ELBO(y)`, an identity). The second is the exact KL of `q` to the
   model's own ring posterior; the first is how far the learned decoder + `sigma_hat` is from the truth. For `F` only the ELBO / IWAE gaps exist.
4. `sigma_hat` versus the true `sigma`.
5. **Axis alignment** `|cos(u, a)| = |u_1|` (absolute value because the sign of `u` is unidentified: `(u, w) -> (-u, -w)` is an exact symmetry that
   the encoder absorbs). For `F` the axis is the leading eigenvector of the average gradient outer product `E_z[grad f grad f^T]` over uniform `z` and its eigenvalue
   fraction `lambda_1 / trace` is reported. Per seed. A run is called **aligned** if `|cos| >= 0.99` (descriptive threshold, fixed now).
6. **Ring statistic** `R`: the mean resultant length of the 256 draws of `q`, projected onto the orthogonal complement of the learned axis and
   normalised (about 0 = uniform azimuth = ring; about 1 = cap), median over test `y`. It is invariant to the sign of `u`. Also `std(u . z)` under `q`
   (draws) against the exact `std(u . z)` of `p_model(. | y)` (thickness of the ring in latitude; sign-invariant), averaged over `y`.
7. TNBBeta `p`, `q`, `epsilon` and `m = epsilon - (d-1)/2` (median over `y`); vMF / PS `kappa` (median).
8. `I(z; y) = -mean log p_true(y) - (1/2) log(2 pi e sigma^2)` exact on the test set (nats), the collapse reference.

## Tests (confirmatory; decoder L only)

For each `(d, sigma)` (12 cells) and each comparator (`vmf`, `power_spherical`): one-sided paired Wilcoxon signed-rank over the seeds valid in both arms,
H1: `tnbbeta` has the smaller ELBO gap. 24 tests, Holm step-down at 0.05 over all 24 (a single family), and a cell is **confirmed** only if Holm rejects
AND the mean paired gap difference (comparator - tnbbeta) is at least **0.05 nats**. Every p-value is also computed by enumerating all `2^n` sign
patterns (they must agree; a disagreement is reported). Power note: with `n = 10` the smallest one-sided p is `1/1024 = 0.00098`; Holm's first
threshold is `0.05/24 = 0.00208`, so a rejection needs essentially every seed (10/10, or 9/10 with the loser the smallest-magnitude difference). A
non-rejection is therefore *not* evidence of no effect; the per-seed values and win counts are reported alongside. Decoders `Lb`, `M`, `F`:
descriptive paired differences and win counts only, no p-values. **Descriptive trend in d:** for each `sigma` and comparator, the mean paired difference
at each `d`, and the least-squares slope of that mean against `log2(d - 1)` (manifold dimension); no inference on the trend.

## Pre-stated expectations and known risks (no outcome is assumed)

* **Thinness at large d.** At `d = 17`, `sigma = 0.1` the exact posterior is thin in `w` relative to its azimuthal volume (`S^15`); a cap family can
  mimic the thin latitude only at very large kappa, and the ELBO of a cap is dominated by azimuth entropy lost. TNBBeta can in principle match the
  ring (small `q`, small `m`) but must find it from the head's initial values by gradient descent; failure to find it is a legitimate result.
* **Kappa scaling.** vMF / PS start at `kappa = d` (CLAUDE.md) and may need `kappa` of order `d / sigma^2` for a thin posterior; Adam at 1e-3 on a linear
  raw-kappa output may not get there in 300 epochs (an optimisation limit, not a family limit; the kappa trajectory is not logged, only the final
  `kappa`). Reported.
* **vMF sampler.** The library's vMF rejection sampler is biased for kappa about 10-40 at `m >= 9` (documented in the dimension-scaling code); to avoid
  a biased comparator we use the same exact-envelope subclass (`ExactVonMisesFisher`, copied into this module's code rather than imported so that
  the other study's file can change independently).
* **Axis alignment can fail.** `u` starts random in `R^d` (`|cos|` about `1/sqrt(d)`), and the encoder and decoder must co-adapt; some seeds may stay
  unaligned or collapse to the prior. Alignment is reported per seed, and ELBO gaps are analysed for **all** seeds (an unaligned run is a bad run,
  not an excluded one). A descriptive sensitivity table restricted to aligned pairs is also given.
* **Collapse.** If a run's `u . z` carries no information (`sigma_hat` inflates, `q` = prior), its gap approaches `I(z; y)`.
* **Learned-sigma start.** 0.2 for all cells (off by a factor of 2 from the truth at `sigma = 0.1, 0.4`); it affects early optimisation of all arms equally.
* **Large `sigma` at large `d`:** `y` is nearly uninformative, gaps are small for every arm and differences may be below the 0.05-nat margin for
  a reason unrelated to family capacity.
* **Float32 training** of the TNBBeta `log q` at `d = 17` with a thin posterior; non-finite losses are counted as divergence.
* The sign-invariance handling (|cos|, `R`, latitude spread) is fixed above; no sign-dependent latitude metric is used.

## Code (all new files; no existing file is edited)

`src/tnbbeta_vae/data/linear_ring.py` (scalar-observation exact quadrature), `apps/synthetic/linear_decoder_ring.py` (train / evaluate / summarize),
`apps/synthetic/linear_decoder_ring_plot.py`, `scripts/slurm/linear_decoder_ring.sbatch`, tests in `tests/data/test_linear_ring.py` and
`tests/apps/test_linear_decoder_ring.py`, results note `writeup/results/linear_decoder_ring_results_2026-11-02.md`. Outputs under
`ring_recovery/linear_decoder/<decoder>/d<d>_s<sigma>/<arm>/s<seed>.{json,npz,pt}` (gitignored scratch).

## Deviations

**D1 (2026-11-02, before any result; found in a 10-epoch smoke run at d = 17, whose numbers are not used).**
(a) *The cosine of the learned axis with the true axis is not identified.* The latent `z` is never observed, and the model's evidence `p_model(y)` depends
on `u` only through the distribution of `u . z` under the uniform prior, which is the same for every unit `u`. Rotating the whole latent space maps
`(u, encoder)` to an exactly equally good solution, so `|cos(u, a)|` is a free symmetry coordinate (expected about `sqrt(2 / (pi d))`, as seen in the smoke run:
0.02 at d = 17) and "does it find the axis" in the sense of `a` cannot be answered by any estimator. Metric 5 is therefore reported only as a symmetry
check, the pre-registered "aligned" threshold and the "aligned-only sensitivity" table are dropped, and the meaningful notion of *finding the axis* is
**encoder-decoder consistency**: the mean over test `y` of `|centre . u|`, where `centre` is the posterior's mean direction (`posterior_centre`) and `u` the
learned decoder axis (fixed ring/cap geometry: a family that places its centre on `+-u` has found the ring's axis). Added to the per-run summary
(`abs_cos_centre_mean`) before any run. The ring statistic `R` and the latitude spread already use the learned axis and are unchanged.
(b) *The "collapse gap" reference was wrong.* The risk list said a collapsed run's gap approaches `I(z; y)`. The gap is measured against the marginal
`log p(y)`, and a model that ignores `z` can match a near-Gaussian marginal of `y` (large d) almost exactly; the ELBO gap of the exact decoder with
`q = Uniform` is `E_y KL(Uniform || p(. | y))`, not `I(z; y)`. Added (`collapse_gap`, exact quadrature) as the reference "ignore y" gap; `I(z; y)` is
still reported as information content only.
(c) *Consequence for the primary metric (a caveat, not a change).* Because `p_model(y)` is invariant to `u` for the linear decoder, a ring-shaped `q` and a
cap-shaped `q` are separated only by `KL(q || p_model(. | y))`, which is the term of the gap that depends on the posterior family; the other term
(`model_misfit`) depends on `sigma_hat` alone for decoder L. The decomposition table therefore carries the main information.
