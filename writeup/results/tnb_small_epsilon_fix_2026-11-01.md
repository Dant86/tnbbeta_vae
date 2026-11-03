# Fix: TNBBeta sampler and density at small epsilon (underflow, boundary clamp, float32 near q = 1 / the poles)

Date: 2026-11-01. Code: base commit `ec2677e` (branch `synthetic-graph-families`), plus this change (UNCOMMITTED in the
worktree `agent-ae3c0e945f8f08c30`; patch `tnb_small_epsilon_fix.patch`). Fixes the defects diagnosed in
`tnb_negative_kl_clamp_and_beta_underflow_2026-11-01.md` (that note lives in the main tree, untracked). The change to
`src/tnbbeta_vae/distributions/` was explicitly approved by the user.

## Commands

    uv run ruff format src/tnbbeta_vae/distributions tests/distributions
    uv run ruff check src/tnbbeta_vae/distributions tests/distributions
    uv run pyright src/tnbbeta_vae/distributions tests/distributions
    # tests and benchmark ran as SLURM CPU jobs (4 CPUs, 16 GB):
    sbatch pt.sbatch tests/distributions/test_tnbbeta_small_epsilon.py --no-cov   # 90 passed in 8 s
    sbatch pt.sbatch                                                              # full `uv run pytest`: 840 passed, 1 skipped in 77 s (job 2289144)
    sbatch bench.sbatch                                                           # old-vs-new timing and failing-parameter numbers

(`pt.sbatch` = `cd <worktree>; uv run pytest "$@" -q -p no:cacheprovider --durations=15`; `bench.py` imports the frozen old
implementations copied into the test file.)

## What changed

`src/tnbbeta_vae/distributions/tnbbeta_univariate.py`

* `rsample`: no `Beta(eps, eps)`. New private `_sample_logit(p, q, eps)` draws the logit of the Beta as
  `log G1 - log G2` with `log Gamma(eps) = log Gamma(1 + eps) + log(U) / eps` (`Gamma(1+eps).rsample()` keeps the pathwise
  gradient; `log U / eps` is differentiable in eps; `U` clamped to `finfo.tiny`). The tilt `s -> a` is then done on logits:
  `log(1 - s^2) = 2 log 2 - |T| - 2 softplus(-|T|)`, `1 - q s^2 = (1-q) + q (1 - s^2)` (no cancellation at q -> 1),
  `logit u = sign(T) (2 log1p|a| - log(1 - a^2))`, `t = logit p + logit u`. `rsample` returns `sigmoid(t)`.
  Same construction for every eps (no branch), so for eps >= 1 it is the same law as before.
* `log_prob`: the `[1e-6, 1 - 1e-6]` clamp is gone. New private `_log_density(log_y, log_1my, p, q, eps)` evaluates in log
  space with `1 - 4 q gamma = (1 - q) + q ((p(1-y) - (1-p)y)/denom)^2`. `value` is only moved off 0/1 by the dtype's own
  resolution (`[finfo.tiny, 1 - finfo.eps/2]`), because `y` in float32 cannot represent `1 - y` below 6e-8.
* `_similarity` is unchanged (still used by nothing but kept; not part of the public API).

`src/tnbbeta_vae/distributions/tnbbeta_spherical.py`

* `rsample`: latitude from `_sample_logit`; `w = tanh(t/2)`, radius `sech(t/2) = 2 e^{-|t|/2}/(1 + e^{-|t|})` (finite
  gradient at any |t|, no `sqrt(clamp(1 - w^2))`).
* `log_prob`: `y = ||z + mu||^2 / 4`, `1 - y = ||z - mu||^2 / 4` (chord form) instead of `w = mu . z` clamped to
  `1 - 1e-6`; the Jacobian is `-(d-3)/2 (log y + log(1-y) + 2 log 2)`. Points a few float32 ulp from a pole are resolved.
* NEW, additive: `rsample_with_log_prob(sample_shape) -> (z, log_q)`. It evaluates the log-density from the sampled
  latitude logit instead of from the returned vector (see "What is NOT fixed"). Existing API untouched.
* `_BOUNDARY_EPS` removed from both modules (a grep over `src/` and `apps/` shows nothing else uses it;
  `power_spherical.py` has its own copy). `_householder_reflect` and `_log_surface_area` are unchanged.

Nothing under `models/`, `apps/` or `CLAUDE.md` was edited. `models/` and `apps/` call only the public
`TNBBetaSpherical(...)` / `.rsample` / `.log_prob`; no one calls the removed private pieces.

No existing test was modified. The 58 existing tests in `tests/distributions` passed with the new implementation
(an earlier run on the login node, before the new test file existed).

## Tests added (`tests/distributions/test_tnbbeta_small_epsilon.py`, 21 test functions, 90 cases with parametrisation)

All quadratures are float64 in `t = logit(y)` on a `sinh`-spaced grid centred on `logit(p)` (resolution at scales 1e-5 and
up, reach `60/eps + 800`).

* (a) mass of `exp(log_prob)` is 1 +- 2e-5 for eps in {1e-4, 1e-3, 1e-2, 0.1, 0.3, 1, 10, 100} on a (p, q) grid with
  p in {1e-6, 0.015, 0.5, 0.9, 1-1e-6}, q in {1e-6, 3e-3, 0.5, 0.99, 1-1e-6}, for the univariate density and for the
  sphere in the latitude variable, d in {3, 8, 16}. The sphere quadrature uses the log-density in logit form
  (`_log_prob_from_logs`) because a unit vector cannot represent the poles at small eps; a separate test checks that the
  vector `log_prob` equals that logit-form density on actual samples.
* (b) KS of `_sample_logit` draws against the quadrature CDF (n = 20000, level about 0.1%, fixed seed) for the eps grid;
  KS of public `TNBBetaSpherical.rsample` vectors (random mean direction, d in {3, 8, 16}, eps in {0.3, 1, 10, 100}) and a
  uniformity check of the fibre direction. At eps = 1e-4: > 99% of draws within 1e-3 of {0, 1}, essentially none at
  `y = p`, and `P(y > 1/2)` equals the quadrature value (about 0.5), not the old 0.066 / 0.934.
* (c) MC `E_q[log q]` (sampler + density) equals quadrature across the grid; at p = 0.015, q = 3e-3, eps = 1e-4, d = 3 the
  exact value is +9987 (note quotes "about 9973"; the exact parameters of its x2 were slightly different) and the
  logit-space MC matches within MC error. Public-API MC KL(q || Uniform sphere) is >= 0 (within MC error) for d in
  {3, 8, 16}, eps in {1e-4 .. 10}.
* (d) pathwise gradients are finite (float32 and float64, eps 1e-6 .. 1e3, p/q at both clamps). For eps in {1, 10, 100} the
  mean per-sample gradient w.r.t. p, q, eps of `sin(3y) + y^2` agrees between a frozen copy of the old `Beta` sampler and
  the new one within 5 SE (n = 2e5); old-vs-new two-sample KS (eps in {1, 3, 30}) and `log_prob` equality to rtol 1e-6 on a
  (p, q, eps >= 1, y) grid pass.
* (e) no inf/NaN in float32/float64 for eps in {1e-6 .. 1e3}, p, q at their clamps (incl. 1 - 1e-6), samples and points
  at/next to the poles; float32 `log_prob` at q = 1 - 1e-6 matches float64 to < 0.05 nats (the old code was off by 1.4-8).

## Results at the failing parameters (d = 3, p = 0.015, q = 3e-3; 2e5 draws, float64; job `tnb-bench`)

| eps | exact `E_q[log q]` (quadrature) | old library MC | new public MC (`rsample`, `log_prob`) | new `rsample_with_log_prob` MC | old / new public KL to uniform | old frac y == p |
|---|---|---|---|---|---|---|
| 1e-4 | 9986.6 | -6.96 | 60.4 | 9958 (heavy tail, MC noise) | -4.43 / 62.9 | 0.867 |
| 1e-3 | 988.9 | 1.22 | 60.3 | 991.6 | 3.75 / 62.8 | 0.242 |
| 1e-2 | 91.3 | 4.98 | 43.6 | 91.1 | 7.51 / 46.1 | 0.000 |
| 0.1 | 4.43 | 1.93 | 4.41 | 4.42 | 4.46 / 6.94 | 0.000 |

Mass of the density that `log_prob` evaluates (p = 0.9, q = 0.5): old 0.0015 / 0.143 / 0.781 / 0.988 at eps = 1e-4 / 1e-2
/ 0.1 / 0.3, new 1.000000 at all of them.

## Timing (3000 nodes x d = 16 x 32 draws, 4 CPU threads, ms, mean of 5; `tnb-bench`)

| dtype, eps | rsample old / new | rsample + backward old / new | log_prob old / new |
|---|---|---|---|
| float32, 1 | 42.6 / 43.6 | 57.9 / 56.0 | 1.5 / 3.1 |
| float32, 5 | 25.4 / 27.7 | 41.0 / 45.1 | 1.5 / 3.1 |
| float64, 1 | 67.3 / 71.0 | 94.0 / 91.3 | 2.5 / 5.2 |
| float64, 5 | 51.0 / 54.6 | 71.7 / 79.6 | 2.6 / 5.4 |

Sampler within about 5-10% of the old one; `log_prob` is about 2x slower in absolute terms (about 2 ms) but is
negligible next to sampling.

## What is NOT fixed / caveats

* Public `log_prob(rsample())` on the sphere is limited by what a unit vector can encode: `1 - |w|` below about 1e-16
  (float64) is lost, so for eps below about 0.03 the vector path still under-states `E_q[log q]` (table: 60 vs 9987 at
  eps = 1e-4). It is large, finite and positive (KL no longer negative, no longer exploitable by driving eps to the floor
  in the same way), but not exact. Exact MC at any eps needs `rsample_with_log_prob`; `models/` (e.g. `monte_carlo_elbo`) does
  not call it yet, which is a separate change outside this task's scope.
* The head's epsilon floor (`_PARAM_EPS = 1e-4` in `models/heads.py`) is unchanged; the distribution is now correct at the
  floor, but a posterior that sits there is a degenerate pair of point masses at the poles.
* Previously finished results with logged epsilon below about 0.3 (stage-2 restarts, `d3_learned_tnbbeta_s4`, anything from
  `graph_search`/`graph_families` runs whose `posterior_stats` epsilon is small) were produced with the old sampler/density and
  are not corrected by this change; they must be re-run or re-evaluated with the exact quantities. For eps >= 1 the new and old
  implementations agree to MC / 1e-6 relative, so results at larger epsilon stand.
* Checkpoints load unchanged (no parameter or state_dict changes); the same seed gives different samples than before (the
  RNG consumption differs), so exact-seed reproductions of old runs will not be bit-identical even at large eps.
