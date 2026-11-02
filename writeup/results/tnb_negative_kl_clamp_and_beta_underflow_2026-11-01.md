# Negative Monte Carlo KL for TNBBeta: cause (Beta-sampler underflow + log_prob boundary clamp at small epsilon)

Date: 2026-11-01. Code: base commit `ec2677e` (branch `synthetic-graph-families`) plus the UNCOMMITTED working tree
(the L2/L3 code from `ring_posterior_l23_d3_2026-11-01.md`; new file `apps/synthetic/tnb_low_epsilon_check.py`).
Nothing under `src/tnbbeta_vae/distributions/` was edited. Resolves the "UNRESOLVED" caveat in
`ring_posterior_l23_d3_2026-11-01.md`.

## Commands

    # re-run of stage 2 (TNBBeta only, d=3, 2000 steps, 300 x, seed 0 / data seed 777), params captured
    # (scratch: s2_capture.py in the session scratchpad; it calls apps.synthetic.ring_recovery._optimise_per_datum
    #  exactly as run_stage2 does: torch.manual_seed(0); restarts 8 scale 2.0 | restarts 1 scale 0.1)
    # reproducer (no checkpoints):
    uv run python -m apps.synthetic.tnb_low_epsilon_check
    # stage-3 checkpoint re-evaluation (scratch reeval_s3.py): loads stage3/ckpt/*.pt, test set seed 424242, 2000 x

The re-run reproduces the recorded anomaly exactly: amortised-like 4/300 negative x (min -4.65), restarts 95/300 negative
(mean -0.738, min -7.82); recorded 4/300 and 96-100/300.

## Cause

Not float32, not the normaliser, not the sampler/density pair at ordinary parameters. All negative-KL x have epsilon at its
head floor (`softplus(raw) + _PARAM_EPS`, `_PARAM_EPS = 1e-4`): eps = 1.0e-4 ... 1.7e-4 (every one of the 4 and 95 negatives;
the p, q there are arbitrary, q median 0.005). In that regime two defects combine; both are independent of dtype
(float64 gives the same numbers, e.g. KL -5.64 vs -5.71 for x2).

1. **`TNBBetaUnivariate.rsample` is wrong for eps below about 3e-3** (the density it targets is fine). It draws
   `torch.distributions.Beta(eps, eps)`, which goes through Gamma draws; for tiny eps both Gamma draws underflow and torch's
   Dirichlet clamp makes the ratio exactly 0.5, so `s = 2z - 1 = 0`, `u = 1/2`, `y = p`. At eps = 1e-4, 86.8% of the draws sit at
   exactly the median `y = p` (24% at eps = 1e-3), where the true density is about `eps`-small; the true mass is two spikes at
   y = 0 and 1 (P(y > 1/2) = 0.5 for the density, 0.066 or 0.934 for the sampler depending on p). Evidence: `Beta(1e-4,1e-4).sample`:
   P(z == 0.5) = 0.8675 (float64) / 0.8678 (float32).
2. **`log_prob` clamps y to [1e-6, 1 - 1e-6] (and w to +-(1 - 1e-6))**, so the density it evaluates is a *sub-normalised*
   function: the clamped density integrates to 0.0015 at eps = 1e-4 (0.015 at 1e-3, 0.14 at 1e-2, 0.78 at 0.1, 0.99 at 0.3), while
   the true density integrates to 1.0000 (quadrature, float64). The true mass within 1e-6 of the boundary is about
   `(1e-6)^eps`, and there the true log-density is `log(eps / y)` (unbounded), so the clamp under-estimates `E_q[log q]` by about
   `P(y < 1e-6) * 1/eps` nats.

Combined effect at the fitted parameters (exact float64 quadrature in `t = logit(y)`; x2 of the restarts run, p = 0.015, q = 3e-3,
eps = 1e-4): `E_q[log q]` library MC -7.0 (sampler at the median, clamped log_prob), exact +9973. The honest KL(q || p_true) of these
fits is about 9.5e3 nats (a pair of point masses at the poles), but the MC estimate is -5.7. Because the estimator is
*optimistic* there, the optimiser is rewarded for driving eps to the floor (the `-eps log y`-type penalty vanishes in the MC
value), so the fits are not merely mis-evaluated, they are produced by exploiting the estimator, and the 64-sample restart-selection
ELBO (same `rsample`/`log_prob`) selects them over honest restarts.

A secondary, independent float32 effect (not the cause here): `log_prob` loses precision as q -> 1 because `1 - 4 q gamma` is
formed from `exp(log_gamma)`; with q = 1 - 1e-6 the float32 log-density error reaches 1.4-8 nats and E_q[log q] is biased
*upward* by 0.3-0.5 (p = 0.5, eps 8-120), and with p, q both at 1 - 1e-6 by 18 nats. Upward bias cannot make KL negative; it only
matters for q within about 1e-3 of 1 (float64 fixes it). Stage 2/3 q medians are <= 0.99, so it is not material there.

## Evidence (all float64 quadrature in t = logit(y) unless stated; scratch scripts quad_exact.py, honest.py, diag*.py, sweep_eps.py)

* Sampler vs density, d = 3 (`P(y > 1/2)` sampler/exact and `E[log f_Y]` library MC minus exact quadrature), p = 0.9, q = 0.5:
  eps 1e-4: 0.934/0.500, -9994; 1e-3: 0.623/0.501, -987; 3e-3: 0.512/0.504, -318; 1e-2: 0.513/0.514, -86 (sampler is fine
  here, the whole error is the clamp); 0.03: -21; 0.1: -2.2; 0.2: -0.26; 0.3: -0.04; 0.5: 0.00; 1, 3: 0.00. float32 and float64
  rows identical to 3 digits (`sweep_eps.out`).
* The integral of exp(log_prob) over the sphere is 1.0000 for the unclamped formula at every tested parameter; for the clamped one see
  the masses above.
* log p_true (`task.log_posterior`, normaliser, `log_evidence`) in float32 vs float64: max abs difference 2.8e-5 (log p),
  5e-7 (normaliser), 7e-7 (evidence) on the 300 x. p_true is not a contributor; x as float32 is fine.
* An exact sampler (log-Gamma representation: `Gamma(eps) = Gamma(1+eps) U^(1/eps)`, `logit z = logG1 - logG2`, then logit arithmetic,
  `w = tanh(t/2)`, radius `sech(t/2)`) matches the density: P(t > 0) 0.4996/0.4998 (eps 1e-4), 0.5626/0.5634 (0.05), 0.7959/0.7959 (0.5);
  `E[log f]` agrees (0.1% at 1e-4 from the heavy tail, 0.1-0.5% elsewhere).
* Validation of the honest evaluator against the library MC where the latter is valid: eps >= 2 (255 and 182 x): mean difference
  -0.003 / -0.002 nats (max 0.59 / 0.12, MC noise on large-KL x).

## What is affected

**Stage 2 (re-evaluated on the re-run, same seeds; honest KL = exact `E_q[log q]` minus MC `E_q[log p_true]`, exact sampler, 2000 draws):**

| run | x with eps < 0.3 | library KL mean / median / frac < 1 | honest KL mean / median / frac < 1 |
|---|---|---|---|
| amortised-like, 2000 steps | 30/300 (idx 11 15 24 28 37 41 53 58 71 88 92 94 104 109 112 117 132 135 142 143 150 181 185 187 216 222 231 246 267 278) | 3.16 / 0.187 / 0.713 | 957 / 0.201 / 0.700 |
| random restarts, 2000 steps | 102/300 | -0.74 / 0.090 / 0.877 | 3263 / 0.356 / 0.550 |

The 30 and 102 x each have honest KL about 9.5e3 (min 5.7e3), not -5 ... +15. For eps >= 0.3 the library values stand. So:
* the pre-registered X2 primary (amortised-like median): honest median 0.201 vs 0.187, still < 1.0 and still 3.7 below the best cap median
  (3.88): PASS unchanged, but "TNBBeta reaches KL < 1 on 71% of x" is 70.0% and the 10% of x at the eps floor are catastrophic (not 0.19).
* restarts: the median moves 0.090 -> 0.356 and the fraction below 1 nat from 0.877 to 0.550; the restart means in the L2 table are meaningless
  either way (honest mean 3263 is dominated by the floor x) and the best-of-8 selection (64-sample ELBO) picked exploit solutions; the
  all-restart raw vectors were not saved, so the honest selected-restart KL cannot be recomputed for the x where a valid restart existed, only
  re-run. The 10000-step stage-2 file was not re-run here; expect the same (frac m < 0 = 0.10 / 0.35 recorded).

**Stage 3 (re-evaluated from the saved checkpoints, no retraining; honest ELBO = decoder MC with the exact sampler, minus exact `E[log q]` and log A):**

| run | recorded ELBO gap | honest ELBO gap |
|---|---|---|
| learned tnbbeta seed 4 (eps = 1e-4 on all 2000 test x) | -6.69 | +9994.45 |
| learned tnbbeta seed 0 / 1 (eps >= 2) | 0.657 / 0.261 | 0.656 / 0.261 |
| true fixed_m_neg seed 0 (eps = 0.5) | 3.124 | 3.130 |
| true tnbbeta seed 9 (eps >= 2.7) | 0.181 | 0.180 |

Only `d3_learned_tnbbeta_s4` is invalid (it is the only stage-3 TNBBeta checkpoint with epsilon at the floor; all other checkpoints
have epsilon median >= 0.5: free 2.99-14.8, p_init 3.0-12.1, fixed_m_ring 123, fixed_m_neg 0.5). So condition B's "free TNBBeta gap
-0.251 +- 2.268" is an invalid mean: without seed 4 it is 0.46 (mean of 0.66 0.26 0.57 0.27 0.29 0.53 0.52 0.54
0.54); seed 4's honest gap is about 1e4, i.e. the seed collapsed, and *training* found that solution because the training ELBO has the same
estimator. fixed_m_neg (eps = 0.5) is affected only at the 0.005-0.01 nat level (clamp bias, table above). The few negative per-x KLs in the
true-decoder stage-3 runs (min -0.008, |value| < the SE, 17 of the 40 true-decoder TNBBeta runs) are ordinary MC noise on KL near 0 and not this defect; the note's
"no negative per-x KL in any true-decoder run" is not literally true but harmless.

L1 (float64 quadrature), vMF and PowerSpherical are unaffected.

## Does it bias training in `monte_carlo_elbo` and the graph experiments?

Yes, in the same regime and direction. `monte_carlo_elbo` for TNBBeta uses `rsample` + `log_prob` (`posterior.log_prob(z) - prior.log_prob(z)`), so
whenever a posterior has epsilon below about 0.3 (sampler wrong below about 3e-3; clamp bias `~ P(y < 1e-6) / eps` below about 0.3, 2 nats at
eps = 0.1, 0.26 at 0.2), the KL term is under-estimated (optimistic) and `eps` is pulled towards the floor 1e-4 by the optimiser. At
eps >= 1 the library values are exact to MC error (checked). The regime is *absolute eps*, not m: at larger latent_dim `m < 0`
(eps < (d-1)/2) only reaches the clamp if eps itself is below about 0.3. So graph runs with "m < 0" are affected only if their logged `epsilon` is
small. Check: any TNBBeta checkpoint/summary with `epsilon` median or a sizeable fraction below 0.3 (the logged `posterior_stats` epsilon) is
untrustworthy for KL/ELBO; I did not scan the graph results here. At d > 3 the spherical Jacobian `(1 - w^2)^-((d-3)/2)` is also evaluated at the
clamped w, which can only add to the same under-estimate.

## Proposed fix (NOT applied; distributions/ is validated math)

1. `TNBBetaUnivariate.rsample`: sample in logit space. `logit z = log G1 - log G2` with `log Gamma(eps) = log Gamma(1+eps) + log(U)/eps`
   (reparameterisable through `Gamma(1+eps)`'s `rsample`; the `log U / eps` term is differentiable in eps); `s = tanh(T/2)`,
   `logit u = sign(T) (2 log1p(|a|) - log(1 - a^2))` with `a = s sqrt((1-q)/(1-q s^2))` and `log(1 - a^2) = log(1 - s^2) - log1p(-q s^2)`,
   `log(1 - s^2) = -2 log cosh(T/2)` computed stably, `t = logit(p) + logit u`. See `sample_t` in `apps/synthetic/tnb_low_epsilon_check.py`
   (float64, matches the density). For the spherical lift use `w = tanh(t/2)`, `sqrt(1 - w^2) = sech(t/2)`.
2. `log_prob`: drop the y clamp and evaluate in logit/log space (`log y = -softplus(-t)`, `1 - 4 q gamma = (1-q) + q ((p(1-y)-(1-p)y)/denom)^2`,
   no `exp(log_gamma)` cancellation near q = 1); for the sphere recover `1 - |w|` from the chord `||z -+ mu||^2/2` instead of `mu . z` so points
   within 1e-7 of the pole are resolved even in float32 (a singular density needs this; at m < 0 it is integrable but unbounded).
3. Cheapest mitigation until then: raise the head's epsilon floor to about 0.3-0.5 (`_PARAM_EPS`, in `models/heads.py`; I did not touch it),
   which removes the exploit regime entirely (the clamp bias is below 0.04 nats from 0.3 on) at the cost of excluding spikes at the pole with
   eps < 0.3; and/or assert/log the fraction of posteriors with eps < 0.3 in every run.
4. Re-evaluation of finished results: float64 re-evaluation does NOT help (the defects are dtype-independent); use
   `apps.synthetic.tnb_low_epsilon_check.honest_kl` (exact `E[log q]` by quadrature, exact sampler) on the saved stage-3 checkpoints (done above for
   a sample; run it over all `d3_learned_tnbbeta*` for completeness, it takes minutes) and re-run stage-2 restarts with the eps floor raised (or
   the fix) and selection by an honest ELBO. Only `d3_learned_tnbbeta_s4` and the stage-2 rows above need to change; correct the B table
   and the L2 restart rows, and drop the "UNRESOLVED" caveat.
