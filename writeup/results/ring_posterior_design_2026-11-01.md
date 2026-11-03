# Ring-posterior ("single-beacon ranging") task: design and decision rules

Date: 2026-11-01. Base commit: `ea01fdedd2c12ec0d65c275d1c6927657f4231f0` (uncommitted
working tree on top of that). This note is written BEFORE any experiment is run and is
not edited afterwards except for an appended "Deviations" section. Results go in
`ring_posterior_recovery_2026-11-01.md`.

## Question

TNBBetaSpherical(mu, p, q, eps) can be a cap, bimodal at the antipodal pair +/-mu
(m = eps - (d-1)/2 < 0), or have an interior latitude ring mode. vMF and Power Spherical
are provably unimodal caps and the families are not nested (expressivity note, Cor 2.3,
Thm 5.1, section 6.2). On the graph tasks TNBBeta never needed its extra shapes. Here the
TRUE posterior is a latitude ring (genuine uncertainty, not a symmetry redundancy), so a cap
family cannot represent it. Three nested questions:

* L1 REPRESENT: is there a TNBBeta member much closer (exact KL) to the ring than any vMF/PS member?
* L2 PER-DATUM OPTIMISATION: does gradient ascent on the exact ELBO, with the true decoder and an
  amortised-encoder-like initialisation, find it within a fixed budget?
* L3 AMORTISED: does an MLP encoder trained by the ELBO find it?

## Task

* Ambient dimension d in {3, 8, 16}; sphere S^(d-1). Fixed unit axis `a` (a seeded random unit
  vector, seed `1000 + d`).
* z ~ Uniform(S^(d-1)); w = a . z; x = g(w) + N(0, sigma^2 I_3), with
  g(w) = [w, sin(3 w), cos(5 w)] (k = 3 features; the first feature identifies w, the others
  make the map nonlinear).
* Exact posterior (w.r.t. surface measure): p(z|x) = exp(-||x - g(w)||^2 / (2 sigma^2)) / Z'(x).
  `Z'(x) = A_{d-2} * integral_0^pi exp(-E(cos t)) sin^(d-2) t dt` (A_{d-2} = area of S^(d-2)),
  computed by a 4001-point midpoint rule in the latitude t (theta from `a`); equivalently
  `Z' = A_{d-1} * E_Uniform[exp(-E)]`. `log p(x) = -(k/2) log(2 pi sigma^2) + log E_Uniform[exp(-E)]`.
  A ring: the density depends on z only through w.
* sigma in {0.1, 0.2, 0.4}; sigma = 0.2 is the primary value for L2/L3. The posterior's latitude
  standard deviation is about sigma/(|g'(w)| sin t) radians, i.e. roughly 1-5 degrees at
  sigma = 0.1 and 0.2, and up to ~10 degrees at 0.4 (thin but not degenerate).
* L1 data: `x* = g(cos t*)` noiseless, ring latitude t* in {30, 60, 80} degrees (so the
  posterior ring sits near t*, shifted slightly by the nonlinearity; the actual central 95%
  latitude interval is computed and reported). Grid: 3 d x 3 t* x 3 sigma = 27 cells.
* L2/L3 data: the honest generative model above (z uniform), so the posterior latitudes are
  distributed as the data are. NOTE for d = 8, 16 uniform z concentrates at w ~ 0 (latitude
  90 +/- 20 / 14 degrees), so the typical ring is near the equator there; d = 3 has uniform w.

## Metrics

* Exact KL(q || p_true) = E_q[log q(z) - log p_true(z|x)], Monte Carlo over q-samples using each
  family's own `log_prob`; p_true is exact. Standard error reported (L1: 4e5 samples, final
  evaluation with fresh noise; L2/L3: 2000 per x). With the TNBBeta axis equal to `a`, the KL is
  also computed by 1-D quadrature (used inside the L1 optimiser; cross-checked against MC).
* ELBO gap = log p(x) - ELBO(x). With the true decoder this equals KL(q||p_true) exactly (used
  as an internal consistency check). With a learned decoder the latent can be rotated away from
  `a`, so KL(q||p_true) is not meaningful there; ELBO and the gap are the primary metrics.
* Latitude marginal of q about `a`: histogram of arccos(a.z) (90 bins) vs the true marginal;
  total variation (TV). "Band mass" = mass of q inside the true posterior's central 95% latitude
  interval (truth: 0.95). "Pole mass" = q-mass within 10 degrees of a or -a.
* TNBBeta: p, q, eps, m = eps - (d-1)/2, angle between mu and +/-a.
* "Ring-shaped" q: band mass >= 0.8 AND TV < 0.25. A "ring-only" TNBBeta member: m > 0 (no
  pole spikes) and pole mass < 0.01.

## Sanity checks required before any experiment (unit tests)

* True posterior integrates to 1 over the sphere (quadrature vs scipy `quad`, and a Monte Carlo
  check with uniform draws).
* KL(Uniform || p_true): the closed-form quadrature expression vs MC with the TNBBeta uniform
  member (p = 0.5, q = 0, eps = (d-1)/2) as q (also validates TNBBetaSpherical.log_prob).
* KL(vMF(a, kappa) || p_true): MC with `VonMisesFisher.log_prob` vs quadrature using the vMF
  latitude density normalised numerically (independent of the Bessel normaliser).
* log p(x) check: ELBO of the exact posterior, E_p_true[log N(x; g(w)) + log U - log p_true],
  equals log p(x).

## L1: representation (reverse KL, multi-start)

* TNBBeta, mu = a: quadrature KL, 96 starts per cell batched in float64: random starts
  (logit p ~ U(-4,4) in jittered form, q in {0, small, 0.5, 0.9, 0.99, 0.999}, log eps
  log-uniform on [0.05, 3e3]) and ring-like starts (p = (1 + cos t*)/2, q in {0, 0.9}, eps in
  {1, 5, 20, 100, 500}) and the certified trimodal pattern (0.85, 0.95, 0.3) re-scaled. Adam,
  then the best 5 are re-evaluated by MC.
* TNBBeta, mu free (check): Monte Carlo (reparameterised) Adam over (alpha, p, q, eps) where alpha
  is the angle between mu and a (azimuth irrelevant by the symmetry of p_true about a), started
  from the best mu = a solutions with alpha in {0.1, 0.5, 1.0, 2.0}, plus cap-on-the-ring starts.
* vMF and PowerSpherical, mu free (parameterised by alpha, kappa): grid over alpha (19 values in
  [0, pi]) x log kappa (25 values in [log 0.1, log 3e4]) with common random numbers, top 3
  refined by Nelder-Mead, final fresh evaluation. Also kappa -> 0 (uniform) is included.
* Reported per cell: best KL per family, best TNBBeta parameters, whether the best member is
  ring-only, its band mass / TV / pole mass; baseline KL(Uniform || p_true).
* **X1 = 0.5 nats.** A cell PASSES if best-TNBBeta KL <= min(best vMF, best PS) - 0.5 nats AND the
  best TNBBeta member is ring-shaped. Justification: the MC standard error is ~0.005 nats; the
  expected repeatability of the optimiser across starts is a few hundredths (measured and
  reported); 0.5 nats is a factor 1.65 in likelihood ratio per posterior sample (~0.7 bits),
  large compared with both, yet small compared with the several-nat gaps expected for a thin
  ring in d >= 8.
* **STOP rule:** if fewer than 14 of 27 cells pass, STOP after L1 and report.

## L2: per-datum optimisation, true decoder

* 300 held-out x (generative model, seed 777), sigma = 0.2, d in {3, 8, 16}.
* Objective: exact ELBO E_q[log N(x; g(a.z), sigma^2 I)] - KL(q || Uniform) per x, each x with
  its own parameters (raw head vector of the same form as the posterior heads in
  `models/heads.py`, so mu / p / q / eps / kappa maps are the amortised ones), Adam lr 0.05,
  8 reparameterised samples per step, fixed budgets of 2000 steps (primary) and 10000 steps
  (check). KL: analytic for vMF and PS (as in `MlpVAE`), Monte Carlo for TNBBeta.
* Init like a fresh encoder head: raw ~ 0.1 * N(0, 1). (TNBBeta's raw-0 member has p = q = 0.5,
  eps = softplus(0) = 0.69, i.e. m < 0 for d >= 3: this init is NOT near uniform and is
  reported as such.) Also 8 random restarts with raw ~ N(0, 2^2), best restart picked by ELBO
  (computed with 64 fresh samples, no use of the truth).
* Report: exact KL(q||p_true) (median, mean) per family, ELBO gap, fraction of x with
  TNBBeta better than both caps, the final TNBBeta eps and m.
* **X2 = 0.25 nats.** Level 2 PASSES (per d) if TNBBeta's median KL (amortised-like init, primary
  budget) <= min(median vMF, median PS) - 0.25 AND TNBBeta's median KL < 1.0 nat ("found").
  Failure here with the 10000-step check also failing is read as an optimisation obstruction.
  L3 is run regardless of the L2 outcome (it contains the controls that diagnose why), but L1
  failing stops everything.

## L3: amortised training

* Encoder: `mlp_stack([3, 128, 128])` + linear posterior head (`head_size`,
  `posterior_from_raw`; TNBBeta with `fixed_epsilon` through `tnbbeta_posterior`), reusing
  `MlpVAE`. Condition A (true decoder): decoder replaced by the true g(a.z) and a frozen
  sigma = 0.2 (no free parameters); condition B (learned decoder): the stock `MlpVAE`
  decoder (MLP [128,128]) with learned sigma.
* Training: 20000 fresh training x per seed, batch 256, Adam lr 1e-3, 100 epochs, linear KL
  warm-up over the first 10 epochs, 4 ELBO samples per x, sigma = 0.2. Test: 2000 fixed x.
  All families see the same data stream and seed; paired by seed. 10 seeds (0-9).
* Families: vMF, PowerSpherical, TNBBeta free, TNBBeta `fixed_epsilon` with m = -0.5 (the
  requested m < 0 control; forces pole spikes, which a ring does not want, so a negative result
  there is expected), and TNBBeta `fixed_epsilon` with eps_ring set from the data: eps_ring =
  (1/s^2 - 1)/2 with s^2 the median over training x of the true posterior variance of w (a
  label-free constant, computed from the truth only, to test whether TNBBeta fails only because
  eps must climb to ~10^2 from a softplus(0) start). Equal budget for all.
* Report: A: exact test KL(q||p_true) mean +/- sd over seeds; B: ELBO gap; both: ELBO,
  fitted latitude marginals at 3 representative x, band/pole mass, TNBBeta m, and whether free
  TNBBeta collapses to the unimodal cap (m > 0 and angle(mu, a) ~ 0 with a cap-shaped
  marginal) as on the axial task.
* **Test:** one-sided paired Wilcoxon signed-rank over the 10 seeds (H1: TNBBeta free better than
  vMF; and better than PS) on the primary metric (A: KL; B: ELBO gap). Primary family of tests =
  {vMF, PS} x {A, B} x {d = 3, 8, 16} = 12 tests, Holm-corrected at 0.05 (the smallest attainable
  one-sided p with 10 seeds is 2^-10 ~ 0.001, below 0.05/12 = 0.0042, so this is attainable).
  **X3 = 0.25 nats** margin on the mean paired difference as well. "TNBBeta uses its capacity" at
  L3 iff both tests survive Holm AND the margin holds AND the fitted marginals are ring-shaped.
  Fixed-eps variants and other comparisons are descriptive (no p-values asserted).

## Seeds, hardware, reproducibility

L1 uses `torch.manual_seed(cell index)`; L2 test-x seed 777, optimisation seed 0; L3 seeds 0-9. CPU
(64 cores locally), one torch thread per process.

## Known caveats recorded in advance

* `VonMisesFisher.rsample` draws w by rejection with non-differentiable proposals, so for m > 3 the
  gradient w.r.t. kappa is the S-VAE-style one without the correction term (biased); `_sample_w3`
  (d = 3) is exactly reparameterised. L1 does not use gradients for vMF/PS, but L2/L3 do, so vMF
  results at d = 8, 16 may partly reflect that estimator; PowerSpherical's reparameterisation is
  exact and serves as the clean comparison.

## Deviations

Appended 2026-11-01, after Level 1 (24/27 cells passed; see
`ring_posterior_recovery_2026-11-01.md`) and BEFORE any Level 2/3 result exists.

1. **Dimension.** L2/L3 are run first at d = 3 only (not d in {3, 8, 16}), so the posterior is a
   circle on S^2 and per-x exact-versus-fitted posteriors can be plotted on the sphere. At d = 3
   w is uniform, which also avoids the near-equator concentration noted above for d >= 8. L1
   showed the smallest gaps at d = 3, so a weak result there is not evidence about d >= 8.
2. **Holm family.** With d fixed at 3 the primary family is {vMF, PS} x {A, B} = 4 tests
   (Holm at 0.05, smallest attainable one-sided p with 10 seeds is 2^-10, below 0.05/4). X2 and X3
   thresholds are unchanged. d = 8, 16 become a follow-up with the original 12-test family.
3. **Class extension (next step, not yet run).** After the single-axis d = 3 result, a 4-class
   variant: four pre-selected axes (tetrahedron vertices), each class with its own feature map
   g_c(w) = [w, sin(k_c w), cos(k_c' w)] so that the class is identifiable from x. The exact
   posterior is p(z|x) proportional to sum_c pi_c exp(-E_c(z)); the per-x class-posterior weight is
   reported and ambiguous points (max weight < 0.99) are treated separately. Exact KL there is
   Monte Carlo against the mixture density (not axisymmetric overall). Seeds remain the unit of
   replication; classes are not independent replicates.
4. **Execution.** Trained on GPU via SLURM (`scripts/slurm/ring_recovery.sbatch`), model
   moved to CPU for the exact float64 evaluation. Checkpoints are saved so sphere plots can
   reload them. No change to any quantity or test defined above.
5. **Stage-3 evaluation samples.** Appended 2026-11-01 after the d = 3 L2/L3 runs: the stage-3 exact
   KL / ELBO-gap evaluation used 1000 q-samples per test x (`run_stage3`, `num_samples=1000`), not the
   2000 stated under Metrics (stage 2 used 2000). The test set (2000 x, seed 424242) and all tests/thresholds are
   unchanged.
6. **Exploratory variant `tnbbeta_p_init`.** Appended 2026-11-01, AFTER the d = 3 L3 default-init
   results were seen. A stage-3 variant `tnbbeta_p_init` (TNBBeta free, raw-p head bias initialised to
   ln 3 so p = 0.75 at init, `p_init_offset = ln 3`; otherwise identical to `tnbbeta`) was added, run for
   both conditions x 10 seeds at d = 3. It is exploratory and OUTSIDE the pre-registered Holm family
   (which stays {vMF, PS} x {A, B} against the original `tnbbeta`, whose runs keep the original init via
   `p_init_offset = 0.0`); only descriptive p-values are reported for it. See
   `tnb_p_init_offset_applied_2026-11-01.md`.
