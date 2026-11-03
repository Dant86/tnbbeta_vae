# Axial Bingham study: pre-registration

Date: 2026-11-02. Status: **FROZEN** before any run of the training grid; edited afterwards only by
appending a "Deviations" section. Base commit: `6b1880f` plus the uncommitted tree (other agents'
untracked files are unrelated). Results go in `axial_bingham_results_2026-11-02.md`. Code:
`src/tnbbeta_vae/data/axial_bingham.py`, `src/tnbbeta_vae/distributions/sign_symmetrised.py`,
`apps/synthetic/axial_bingham.py` (+ `_checks.py`, `_plot.py`), `scripts/slurm/axial_bingham.sbatch`,
`scripts/slurm/axial_bingham_aux.sbatch`. Builds on `axial_mixture_recovery_2026-10-07.md` (weak
S^1 attempt: free decoder, one seed, before the exact-log-q fix; TNBBeta stayed unimodal) and the
learned-decoder protocol of `ring_class_designs_learned_decoder_preregistration_2026-11-01.md`.

## Question

Is there data on which TNBBeta's structural capacity (`m < 0` antipodal bimodality) is genuinely
needed, with a decoder that cannot absorb it and without engineering the data-generating process
around TNBBeta? Candidate: a spiked-matrix model whose exact posterior is a Bingham distribution,
antipodal by symmetry. A null or negative answer is a legitimate result.

## Task

`z ~ Uniform(S^(d-1))`, `X = z z^T + sigma E`, `E_ij` iid `N(0,1)`, `X` is `d x d` and NOT symmetrised
(models see all `d^2` entries; the likelihood is Gaussian on all `d^2` entries). Since
`||z z^T||_F^2 = 1`, `log p(X | z) = z^T X_sym z / sigma^2 + c(X)`, so `p(z | X)` is Bingham with
matrix `A = X_sym / sigma^2`, exactly antipodal. The normalised Bingham constant
`E_U exp(z^T A z)` is, in the eigenbasis of `A`, a one-dimensional integral (closed form for d=2;
Gauss-Legendre with Bessel `I_0` for d=3,4; formulas in the module docstring), so the exact `log p(X)`
is available for d = 2, 3, 4.

## Pre-freeze checks (design constants; CPU jobs 2298077-2298080, 2298190)

Logs: `slurm_logs/axial-bingham-aux_<id>.out`; JSON in `ring_recovery/axial_bingham/checks/`
(`apps/synthetic/axial_bingham_checks.py`, commit/tree as above).

**Detectability threshold, our noise convention.** `X_sym` has off-diagonal noise std `sigma/sqrt(2)`,
diagonal std `sigma`; the bulk edge is `sigma sqrt(2d)` and the BBP transition for the unit spike is
`sigma sqrt(d/2) = 1`, i.e. **`sigma sqrt(d) = sqrt(2) = 1.414`** (not 1). Simulation (200 reps; 40 at
d=200): at d=200 the top eigenvalue and the overlap `(v1 . z)^2` track `1 + (sigma^2 d)/2` and
`1 - sigma^2 d / 2` below the threshold (`sigma sqrt(d)` = 0.5: eigenvalue 1.126 vs 1.125, overlap
0.873 vs 0.875; at 0.75: 1.276 vs 1.281, 0.712 vs 0.719) and the overlap is 0.124 at the threshold
and 0.043 at `sigma sqrt(d)` = 2. At d = 2, 3, 4 the transition is a smooth crossover (overlap at
`sigma sqrt(d)` = 1.414: 0.713, 0.612, 0.551). The finite-d object that matters here is the exact
posterior's axis concentration `E_post[(v1 . z)^2]` (1/d = uniform, 1 = point caps), median over X:

| sigma | d=2 | d=3 | d=4 |
|---|---|---|---|
| 0.1 | 0.995 | 0.990 | 0.985 |
| 0.2 | 0.981 | 0.960 | 0.940 |
| 0.5 | 0.884 | 0.767 | 0.662 |
| 0.8 | 0.768 | 0.599 | 0.492 |
| 1.0 | 0.714 | 0.542 | 0.437 |

The near-threshold `sigma` (`sqrt(2/d)`) is 1.0, 0.82, 0.71 for d = 2, 3, 4.

**sigma grid (all d): {0.1, 0.25, 0.5, 1.0}** = low noise (two narrow caps), moderate, broad (axis
concentration 0.68 to 0.88 at d=2..4 for sigma 0.5 vs 0.44 to 0.71 at 1.0), and near/above threshold
(d=3,4). The mean top-eigen gap `lam_1 - lam_2` of `A` (the rank-one-Bingham `kappa` analogue) is about
90-100 (sigma 0.1), 22-25 (0.2), 5-7 (0.4), 2-2.5 (0.8) (anisotropy table below); sigma 0.25 is about
14 and sigma 1.0 about 1.3-1.6 by interpolation.

**Normaliser accuracy** (float64; 200 test X per cell; `*_normaliser` job): the d=3,4 Gauss-Legendre
normaliser changes by at most 4.2e-11 nats between 2000 and 8000 nodes at every sigma; d=2 is a
closed form. The sphere quadrature used for the learned decoders' exact evidence (d=2: 720 angles; d=3:
200 Gauss-Legendre latitudes x 400 azimuths) agrees with its 2x refinement to at most 2.8e-14 (peaked
sigma = 0.1 included). d=4 is checked against an independent brute-force 3-D Hopf-coordinate quadrature
(300 x 256 x 256 nodes): at most 7.4e-12. Monte Carlo (2M draws, 20 X, sigma >= 0.4 only; peaked
integrands are not Monte-Carlo-checkable) agrees to 1.0e-3 at worst, within sampling error. Every
training run additionally records the quadrature's error on the TRUE model against the closed form
(`quad_true_model_maxerr`) and the learned model's evidence on a 2x grid (`quad_refine_maxdiff`).

**Anisotropy cost** (300 X per cell): the exact best axially symmetric `q` (any axis, any radial
profile, orbit-averaged log-density; `KL = log Z - log E_w exp(alpha w^2 + (1 - w^2)(tr A - alpha)/(d - 1))`
minimised over `alpha`), mean nats:

| d | sigma | best axial KL (any profile) | axis = top eigenvector | one-sided (hemisphere) = axial + ln 2 |
|---|---|---|---|---|
| 2 | any | 0 (every density on S^1 is axial) | 0 | 0.6931 |
| 3 | 0.1 | 0.0051 | 0.0051 | 0.6982 |
| 3 | 0.2 | 0.0233 | 0.0233 | 0.7164 |
| 3 | 0.4 | 0.0675 | 0.0739 | 0.7606 |
| 3 | 0.8 | 0.0421 | 0.0669 | 0.7352 |
| 4 | 0.1 | 0.0136 | 0.0136 | 0.7068 |
| 4 | 0.2 | 0.0608 | 0.0608 | 0.7539 |
| 4 | 0.4 | 0.1756 | 0.1793 | 0.8688 |
| 4 | 0.8 | 0.1130 | 0.1244 | 0.8062 |

An axially symmetric family with a perfect radial profile therefore loses at most 0.07 nats (d=3) and
0.18 nats (d=4) to the full Bingham in this range; the one-sided (hemisphere-restricted) value is
the axial value plus exactly ln 2.

**Representability of the rank-one Bingham `exp(kappa (mu . z)^2)`** by `TNBBetaSpherical` (and vMF/PS,
and their sign-symmetrised versions), best-fit `KL(q || Bingham)` by quadrature in the latitude
(20,000 nodes clustered at both poles), Nelder-Mead from 60 starts over (p, q, epsilon) (a first
pass with 12 starts gave the same numbers to the printed precision):

| d | kappa | TNBBeta free KL | fit p, q, eps, m | TNBBeta p=0.5 KL | vMF | PS | sign-sym vMF | sign-sym PS |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.5 | 0.00669 | 0.500, 0.000, 0.436, -0.064 | 0.00669 | 0.020 | 0.049 | 2.3e-05 | 3.8e-05 |
| 2 | 1 | 0.0282 | 0.500, 0.000, 0.389, -0.111 | 0.0282 | 0.065 | 0.107 | 0.0003 | 0.001 |
| 2 | 2 | 0.117 | 0.500, 0.000, 0.325, -0.175 | 0.117 | 0.238 | 0.305 | 0.0028 | 0.0085 |
| 2 | 4 | 0.415 | 0.899, 0.000, 0.279, -0.221 | 0.43 | 0.698 | 0.708 | 0.0077 | 0.015 |
| 2 | 8 | 0.714 | 0.990, 0.871, 0.270, -0.230 | 1.23 | 0.695 | 0.696 | 0.0019 | 0.0032 |
| 2 | 16 | 0.841 | 0.003, 0.000, 1.272, 0.772 | 2.77 | 0.693 | 0.694 | 0.00027 | 0.00054 |
| 2 | 32 | 0.872 | 0.998, 0.000, 1.472, 0.972 | 6.07 | 0.693 | 0.693 | 5.4e-05 | 0.00012 |
| 2 | 64 | 0.888 | 0.999, 0.000, 1.590, 1.090 | 13 | 0.693 | 0.693 | 1.2e-05 | 2.7e-05 |
| 2 | 128 | 0.897 | 1.000, 0.000, 1.663, 1.163 | 27.3 | 0.693 | 0.693 | 2.8e-06 | 6.6e-06 |
| 2 | 256 | 0.902 | 1.000, 0.000, 1.710, 1.210 | 56.1 | 0.693 | 0.693 | 4.8e-07 | 1.6e-06 |
| 3 | 0.5 | 0.00297 | 0.500, 0.000, 0.863, -0.137 | 0.00297 | 0.014 | 0.025 | 1.8e-05 | 3e-05 |
| 3 | 1 | 0.014 | 0.500, 0.000, 0.759, -0.241 | 0.014 | 0.049 | 0.066 | 0.00026 | 0.00083 |
| 3 | 2 | 0.0705 | 0.500, 0.000, 0.615, -0.385 | 0.0705 | 0.195 | 0.225 | 0.003 | 0.009 |
| 3 | 4 | 0.341 | 0.500, 0.000, 0.456, -0.544 | 0.341 | 0.689 | 0.722 | 0.015 | 0.031 |
| 3 | 8 | 0.689 | 0.974, 0.819, 0.527, -0.473 | 1.25 | 0.699 | 0.703 | 0.0059 | 0.0095 |
| 3 | 16 | 0.759 | 0.010, 0.000, 2.143, 1.143 | 3.23 | 0.694 | 0.695 | 0.00075 | 0.0015 |
| 3 | 32 | 0.777 | 0.005, 0.000, 2.403, 1.403 | 6.62 | 0.693 | 0.693 | 0.00015 | 0.00031 |
| 3 | 64 | 0.786 | 0.998, 0.000, 2.539, 1.539 | 13.2 | 0.693 | 0.693 | 3.3e-05 | 7.3e-05 |
| 3 | 128 | 0.791 | 0.999, 0.000, 2.612, 1.612 | 27.1 | 0.693 | 0.693 | 7.9e-06 | 1.8e-05 |
| 3 | 256 | 0.793 | 0.999, 0.000, 2.651, 1.651 | 55.6 | 0.693 | 0.693 | 1.7e-06 | 4.4e-06 |
| 4 | 0.5 | 0.00139 | 0.500, 0.000, 1.305, -0.195 | 0.00139 | 0.010 | 0.016 | 1.3e-05 | 2e-05 |
| 4 | 1 | 0.00693 | 0.500, 0.000, 1.148, -0.352 | 0.00693 | 0.036 | 0.045 | 0.00019 | 0.00057 |
| 4 | 2 | 0.0394 | 0.500, 0.000, 0.920, -0.580 | 0.0394 | 0.146 | 0.163 | 0.0025 | 0.0072 |
| 4 | 4 | 0.236 | 0.500, 0.000, 0.663, -0.837 | 0.236 | 0.627 | 0.719 | 0.018 | 0.041 |
| 4 | 8 | 0.665 | 0.955, 0.794, 0.705, -0.795 | 1.11 | 0.706 | 0.713 | 0.012 | 0.02 |
| 4 | 16 | 0.73 | 0.018, 0.000, 2.907, 1.407 | 3.3 | 0.695 | 0.696 | 0.0015 | 0.0029 |
| 4 | 32 | 0.745 | 0.991, 0.000, 3.276, 1.776 | 7.29 | 0.693 | 0.694 | 0.00028 | 0.0006 |
| 4 | 64 | 0.751 | 0.004, 0.000, 3.458, 1.958 | 13.9 | 0.693 | 0.693 | 6.3e-05 | 0.00014 |
| 4 | 128 | 0.755 | 0.998, 0.000, 3.550, 2.050 | 27.4 | 0.693 | 0.693 | 1.5e-05 | 3.3e-05 |
| 4 | 256 | 0.757 | 0.999, 0.000, 3.597, 2.097 | 55.6 | 0.693 | 0.693 | 3.5e-06 | 8.2e-06 |
Reading. (i) A bimodal (`m < 0`) TNBBeta fits the Bingham well only at low concentration:
`kappa <= 4` (KL 0.4 vs vMF 0.7 at d=2; 0.34 vs 0.69 at d=3; 0.24 vs 0.63 at d=4), with the optimiser
choosing `p = 0.5` or near it. (ii) From `kappa` about 8 on, TNBBeta's best fit is WORSE than vMF's
single cap (0.69): 0.71 (d=2), 0.69 (d=3), 0.67 (d=4) at kappa 8 (about equal), then 0.73-0.90 for
`kappa >= 16`; the optimiser abandons bimodality (`p` at 0 or 1, `q = 0`, `m > 0`: a unimodal cap with
heavier-than-vMF tails). The symmetric (`p = 0.5`) member cannot make narrow caps at all (KL 1.2 at
kappa 8, growing roughly like `kappa`): its density in `s = 1 - w^2` is `s^(eps-1) (1 - q s)^-(eps+1/2)`, a
power law in `s` with no exponential cut-off, whereas the Bingham falls like `exp(-kappa s)`. (iii)
Sign-symmetrised vMF and PS reproduce the Bingham to within 0.02 (vMF) / 0.04 (PS) nats at every
kappa. So "needing TNBBeta" is not what this task shows at high concentration; the useful regime for
TNBBeta is low-to-moderate concentration, and a sign-symmetrised cap is the better antipodal family
throughout.

## Predictions (stated before any training run)

* **P1 (the ln 2 bound).** For a sign-unaware family the exact KL to a posterior of two narrow
  antipodal caps is at least about ln 2 = 0.693 nats (the one-sided value above). Prediction: in
  the fixed-decoder condition at sigma = 0.1 and 0.25, vMF and PS have mean ELBO gap at least about
  0.69 at d=2 (the encoder's inability to represent a globally consistent axis sign, see below, can
  only add to it).
* **P2.** Sign-symmetrised vMF/PS close nearly all of the gap to the axial lower bound (a gap within
  a few hundredths of the table above plus amortisation error), so any advantage of TNBBeta over a
  sign-unaware cap that is not shared by the sign-symmetrised cap is not due to antipodal shape.
* **P3 (from the representability table).** TNBBeta's advantage over vMF/PS can appear only at low
  concentration: expected at sigma = 0.5 (axis gap about 4-5, `kappa`-analogue about 4: representability
  advantage about 0.3 at d=2,3), small or absent at sigma = 1.0 (advantage under 0.05, so below the
  margin), and a DISADVANTAGE at sigma = 0.1, 0.25 (TNBBeta fit KL 0.73-0.90 vs 0.69), i.e.
  the confirmatory direction H1 is expected to fail there and, if anything, reverse. The tests below
  are nevertheless run in the registered direction at every sigma.
* **P4 (topology, descriptive).** `X -> v1(X_sym)` has no continuous sign choice, so a sign-unaware
  posterior head is forced to flip somewhere in X-space, while any family that can place mass on both
  poles continuously can pass through the flip. This predicts that vMF/PS errors exceed the
  one-sided bound near the flip even at moderate sigma, a diagnostic for the antipodal mass balance
  tables below, not a tested hypothesis.

## Conditions (decoders)

* **A, `fixed`:** `X_hat = z z^T` with the true sigma known (amortised inference only). The exact
  ELBO gap against `log p(X)` equals the exact `KL(q || Bingham)`.
* **B, `linear`:** `X_hat = W vec_sym(z z^T) + b`, `W`, `b` learned, ONE learned scalar sigma
  (`LearnedLikelihoodScale`, initial value 1.0, not the truth). `vec_sym` = the `d(d+1)/2` unique
  entries of `z z^T` (off-diagonals times sqrt 2).
* **C, `mlp`:** MLP `[d(d+1)/2, 128, 128, d^2]` (ReLU) on `vec_sym(z z^T)`, learned sigma.

B and C are sign-invariant by construction, so the model's own posterior is exactly antipodal. For
B and C the latent frame is not identified (any rotation of the latent is absorbed by the decoder, and
`W` can be any map), so no metric compares the learned latent with the true `z`.

## Arms

`vmf`, `power_spherical` (both: dimension-aware initial kappa `kappa_0 = d`, ambient dimension, via the
head's kappa bias), `tnbbeta` (library default, raw-p bias `ln 3`), `tnbbeta_init0` (`p_init_offset = 0`;
descriptive), `vmf_sym`, `ps_sym` (sign-symmetrised: draw `z` from the family, multiply by a fair
random sign; `q(z) = 0.5 q0(z) + 0.5 q0(-z)`; exact log q, from the sampled latitude logit for
TNBBeta-style heads and from `log_prob` at `+-z0` for vMF/PS; Monte Carlo ELBO since no analytic KL).
The optional 2-component vMF mixture posterior is NOT run (sign-symmetrised vMF is its equal-weight,
shared-kappa, antipodal special case). Encoder: MLP `[d^2, 128, 128]` on the flattened `X` + a linear
head; prior Uniform(S^(d-1)); epsilon floor `1e-4`.

## Training and evaluation (as in the earlier studies)

Adam 1e-3, batch 256, 300 epochs, 10% linear KL warm-up, 4 ELBO samples (exact `log q` for TNBBeta and
the symmetrised arms; analytic KL to the uniform sphere for plain vMF/PS), 20,000 fresh training X per
seed (generator seed `100 + seed`, identical across arms and decoders), 10 seeds (0-9), paired by
seed (same data stream and encoder/decoder initial seed across arms; the head is built last).
Training on CPU (`--device cpu`): the networks are tiny and a 3-epoch benchmark ran at about
2 s/epoch single-threaded, faster per run than four runs sharing a GPU in the earlier ring studies.
Grid: d in {2,3,4} x sigma in {0.1, 0.25, 0.5, 1.0} x decoder in {fixed, linear, mlp} x 6 arms x
10 seeds = 2160 runs. Test set: 2000 fixed X per (d, sigma) (seed 424242).

## Metrics

* **Primary: test ELBO gap** `= mean_x [ log p_true(x) - ELBO(x) ]` (nats; lower is better; the ELBO
  from 256 posterior draws per test X with exact `log q`, float64 on CPU; the model's own decoder and
  sigma). `log p_true` = the TRUE model's exact evidence (Bingham). For B and C this gap includes the
  decoder's misfit.
* **Exact KL to the model's own posterior (B, C; d = 2, 3):** `log p_model(x) - ELBO(x)` with
  `p_model(x)` by sphere quadrature, and the model deficit `log p_true(x) - log p_model(x)`.
* **Check:** importance-weighted (K = 64) bound gap, descriptive.
* **Diagnostics:** TNBBeta `p`, `q`, `epsilon`, `m = epsilon - (d - 1)/2` (bimodal iff `m < 0`), fraction
  of x with `m < 0`; vMF/PS kappa; the antipodal mass balance: the smaller hemisphere share of the 256
  draws about the posterior's OWN axis (top eigenvector of its second-moment matrix; frame-free,
  valid for B and C; 0.5 = balanced) and, for the fixed decoder (latent frame = data frame), about the
  top eigenvector of `X_sym` and the median `|cos|` between the two axes; learned sigma vs true;
  quadrature self-checks; any negative per-seed gap is investigated, not discarded (the ELBO
  estimator is unbiased, so a negative mean gap beyond Monte Carlo error indicates a bug).

## Confirmatory tests

For d in {2, 3} (d = 4 is descriptive: no p-values), each decoder (A, B, C), each of the four
sigmas, and each of the comparators `vmf` and `power_spherical`: one-sided paired Wilcoxon signed-rank
test over the 10 seeds, H1: `tnbbeta` has the smaller ELBO gap. Holm at 0.05 over the family of
2 x 4 x 2 x 3 = **48** tests. A cell is **confirmed** iff its test survives Holm AND the mean paired
difference (comparator minus tnbbeta) is at least **0.05 nats**. Every p-value is cross-checked by
enumerating all `2^10` sign patterns (`_wilcoxon_greater_enumerated`; the two must agree to 1e-9).
Secondary (own 48-test Holm family, reported with p-values but descriptive in interpretation):
`tnbbeta` vs `vmf_sym` and vs `ps_sym` in the same cells. All other contrasts (`tnbbeta_init0`, vMF vs
PS, plain vs symmetrised, the IWAE gap, d = 4) are descriptive. Reverse differences of at least the
margin are reported as such. Divergent runs (non-finite) are excluded and counted.

Summary of results is produced by `uv run python -m apps.synthetic.axial_bingham summarize --out-dir
ring_recovery/axial_bingham` (markdown), figures by `apps.synthetic.axial_bingham_plot`.

## Deviations

(none yet; appended below, dated, if any.)

### Deviation 1 (2026-11-02, before any analysed result): non-finite minibatches

The first launch of the grid (job 2298324, cancelled by me after 38 minutes; its two finished
runs were deleted and all cells rerun) crashed in the training loop: in the fixed-decoder, d=2,
sigma=0.1 cells, about half of the Power Spherical, TNBBeta, TNBBeta-init0 and sign-symmetrised-PS
runs raised a `ValueError` from `torch.distributions.Normal` (a non-finite decoder input) on a single
minibatch late in training (reproduced: PS seed 0 at epoch 68, TNBBeta seed 4 at epoch 224; before the
failing batch the loss, gradient norm and head outputs were finite and ordinary, and 200k PS draws at
kappa up to 1e4 contained no non-finite sample; the cause is an unidentified rare sampler or gradient
NaN, not a diverging optimisation). vMF and sign-symmetrised vMF did not fail. Change, applied
identically to every arm: a training step whose loss or gradient is non-finite (or whose forward
raises `ValueError`) is skipped (no optimiser step) and counted (`skipped_steps`, `total_steps` in each
run's JSON); a run is flagged `diverged` (excluded from tests, counted in the tables) if its epoch loss
is non-finite or if it skipped more than 1% of its steps. At evaluation, non-finite posterior draws are
excluded from the per-x Monte Carlo averages and counted (`eval_nonfinite_draws`), and an IWAE chunk
that raises is recorded as NaN (`gap_iwae` is then a nan-mean; `eval_nonfinite_iwae_x` counts them);
`diverged` no longer depends on the IWAE value. No design constant, arm, metric or test changed.

### Deviation 2 (2026-11-02, observed while the grid ran): the representability table was too restrictive

The first completed cells (d=2, fixed decoder, sigma=0.1) show TNBBeta at mean ELBO gap about
0.09 against vMF at 0.84, which contradicts the registered reading of the representability table
(TNBBeta best-fit KL 0.9 at `kappa` about 100, prediction P3 of a disadvantage at sigma=0.1, 0.25 for
d=2). The cause is a restriction in the check, not in the table's arithmetic: it fixed the TNBBeta mean
direction `mu` ON the Bingham axis `a`. A TNBBeta about a `mu` perpendicular to `a` is concentrated on
the latitude `mu . z = 0`, which on S^1 is exactly the two antipodal points `+-a`, so d=2 contains
the antipodal pair of caps by construction (the great-circle ring it gives on S^2 and larger spheres is
not a Bingham density). The aligned-axis table above therefore does NOT bound what TNBBeta can
represent; the check was rerun with `mu` free (`representability_general`, job 2299061; Monte
Carlo KL with common random numbers and exact log q, Adam over (psi, p, q, epsilon) from 12 starts) and
its result is reported in the results note. Prediction P3 stays as registered and is judged as
written; the aligned-axis table and the free-axis table are both reported. No design constant,
arm, metric or test changed.

### Note 3 (2026-11-03): analysis run exactly as frozen

The grid completed (2160 of 2160 runs, none flagged diverged; 382 runs skipped 1-7 of 23,700 steps under
Deviation 1, none more than 1%; no non-finite evaluation draws). The analysis was run exactly as registered
(`summarize` job 2301910). Post-hoc diagnostics added to the results note, not part of any registered test:
the free-axis representability table (Deviation 2), a library vMF sampler-bias check at m = 2, 3, 4
(the dimension-scaling agent found the vMF rejection sampler's Taylor `b` biased at m >= 9 and
kappa about 10-40; here no first or second moment of the latitude differs from quadrature by more than
2.5 standard errors at kappa 5 to 300, so the vMF arms are outside the biased regime), and the
learned-decoder collapse description.
