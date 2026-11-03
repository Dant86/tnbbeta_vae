# Where the TNBBeta head-gradient noise comes from: link loss vs KL (2026-11-01)

Date: 2026-11-01. Base commit: `ec2677efa042f789c26470e44a9cb0a35c51c4c5` (branch
`synthetic-graph-families`), uncommitted working tree. Exploratory and post hoc: one graph draw and
one training seed per config, 48 fresh gradient draws per probe, no hypothesis tests. Follow-up to
`tnb_plateau_mechanism_2026-11-01.md`, which measured only the total head gradient.

## Question

The plateau note found the TNBBeta `p` head-gradient SNR is 0.10-0.16 at init, with a mean gradient
~50x smaller than eps's. How much of that noise is the link loss and how much is the
Monte Carlo KL(q || Uniform)? Does the exact (quadrature) KL gradient look different?

## Method

Local CPU only (1 thread, `nice -n 10`, at most 2 processes, no SLURM). Scratch scripts, not committed, all
in `/tmp/claude-14853/-home-vpathak-tnbbeta-vae/39a83aae-c9e5-4e62-8b02-10e6fc87ddf7/scratchpad/`:
`kl_quad.py` (quadrature KL), `kl_validate.py`, `kl_grad_exact.py`, `grad_decomp.py`, `noise_source.py`,
`ab_quad.py` (written, NOT run, see below). Raw outputs in the `kl/` subdirectory there.
Training follows the exact baseline (`graph_search.run_item` setup: lr 0.003, DIM=16, Adam, MC KL as in
`GraphVAE.training_step`); probes at epochs 0, 100, 400 take 48 draws (fresh rsample and negatives, no
parameter step, `torch.random.fork_rng`) of the gradient of `model.second.weight` for
(1) `link_loss`, (2) `kl_mc/N` (the model's MC estimate `log q(z) - log p(z)`, mean over nodes, divided by N),
(3) `kl_quad/N`, and the sums `total_mc = link + kl_mc/N`, `total_q = link + kl_quad/N`.

```
cd /home/vpathak/tnbbeta_vae; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
S=<scratchpad above>
nice -n 10 uv run python $S/kl_validate.py
nice -n 10 uv run python $S/kl_grad_exact.py
nice -n 10 uv run python $S/grad_decomp.py bipartite-00 10012000 0.003 401 $S/kl/decomp_bip.json
nice -n 10 uv run python $S/grad_decomp.py ring-01      10085000 0.003 401 $S/kl/decomp_ring.json
nice -n 10 uv run python $S/noise_source.py bipartite-00 10012000 0.003 401 $S/kl/noise_bip.json
```

Quadrature KL: the spherical TNBBeta density depends on z only through w = mu.z, and the uniform
sphere's on w is proportional to `(1-w^2)^((d-3)/2)`, so
`KL = integral f_w(w) [log f_sph(w) + log A_d] dw`, with `f_w` the paper's closed-form univariate density
(same formula as `TNBBetaUnivariate.log_prob`, rewritten in float64 with `logsigmoid`, no boundary clamp)
and `f_sph` the spherical density (`distributions/tnbbeta_spherical.py` formula). Substitution
`w = tanh t`, `t in [-30, 30]`, composite Gauss-Legendre (80 panels x 16 nodes); fully differentiable in
float64. `src/tnbbeta_vae/distributions/` and the model code are untouched.

## Results

### Quadrature validation (d = 16)

Density mass integrates to 1.000000 in all cases; doubling the range and panels (L=45, 160 panels) changes
KL by < 1e-5. Against 1e6-draw MC of `log q(z) - log u(z)` (model `log_prob`, float32):

| p, q, eps | quad KL | MC KL +- se |
|---|---|---|
| .5, .5, .693 (init) | 3.3412 | 3.3496 +- .0071 |
| .5, .5, 2.0 | 0.2756 | 0.2752 +- .0012 |
| .6, .5, .693 | 3.5471 | 3.5415 +- .0072 |
| .65, .63, 1.0 | 1.7803 | 1.7772 +- .0040 |
| .9, .3, 3.0 | 7.2117 | 7.2156 +- .0043 |
| .02, .002, 3.3 | 18.2491 | 18.2507 +- .0060 |
| .5, .001, 7.5 | 0.0000 | 0.0000 |
| .95, .9, .4 | 14.0214 | 13.9900 +- .0130 |

All within about 1.2 se except the last (2.4 se; float32 / 1e-6 boundary clamp in the model's `log_prob`
in a regime with strong boundary mass; not investigated).

### Exact KL gradient at p = q = 0.5 (eps = softplus(0)+1e-6 = 0.693)

Gradient w.r.t. the raw head outputs (what the head weights see, per node):

| raw (p, q, eps) | dKL/draw p | dKL/draw q | dKL/draw eps |
|---|---|---|---|
| (0, 0, 0) | 0 (< 1e-15) | -1.071 | -4.466 |
| (0, 0, 1) | 2e-16 | -0.526 | -1.174 |
| (0, 0, -1) | 0 | -1.910 | -16.32 |
| (+-0.05, 0, 0) | +-0.1256 | -1.070 | -4.465 |
| (0.3, 0, 0) | 0.751 | -1.060 | -4.436 |

**dKL/dp = 0 exactly at p = 0.5.** Confirmed three ways: `KL(p, q, eps) = KL(1-p, q, eps)` to
machine precision (p = .3 and .45), a central finite difference (-2e-13), and the autograd quadrature
gradient (0 to 1e-16, for any q, eps). The reason: KL is invariant to `(mu, p) -> (-mu, 1-p)` and
independent of `mu`, so it is even in `p - 1/2`, and the gradient grows linearly away from 1/2 (0.126 at
raw p = 0.05). The KL term supplies no restoring or driving force on p at the symmetric point; it only
pushes q and eps (negative gradient, i.e. KL falls as q and eps rise: 3.34 at init to 0.28 at eps = 2).
The model's MC estimator (20,000 single-node draws, p=q=.5, eps=.693) gives mean gradients
(p, q, eps) = (-0.015 +- 0.030, -1.062 +- 0.009, -4.43 +- 0.05), matching the exact (0, -1.071, -4.466),
so it is unbiased but has per-draw std (4.30, 1.25, 7.42): SNR for p of a *single node* draw is 0 +- 0.007.
The exact KL has no noise from sampling.

### Decomposition of the head-weight gradient (48 draws; norms of `second.weight` rows)

bipartite-00, seed 10012000 (mean norm / per-draw std norm / SNR; "klq" has zero draw noise, so its
mean only; dropout contributes nothing to the noise here: `klq`'s std is exactly 0):

| epoch | output | link: mean, std, SNR | KL_mc/N: mean, std, SNR | KL_quad/N mean | total: mean, std, SNR | var share link / KL / 2cov |
|---|---|---|---|---|---|---|
| 0 | p | 4.1e-5, 3.2e-4, 0.13 | 6.5e-7, 4.9e-6, 0.13 | 1.8e-7 | 4.2e-5, 3.2e-4, 0.13 | 1.00 / 0.000 / 0.000 |
| 0 | q | 7.3e-4, 9.5e-5, 7.6 | 2.5e-5, 1.5e-6, 16.6 | 2.5e-5 | 7.5e-4, 9.6e-5, 7.8 | 0.99 / 0.000 / 0.01 |
| 0 | eps | 2.1e-3, 2.8e-4, 7.4 | 1.1e-4, 9.0e-6, 11.7 | 1.1e-4 | 2.2e-3, 2.8e-4, 7.7 | 0.99 / 0.000 / 0.01 |
| 100 | p | 6.3e-5, 4.1e-4, 0.15 | 2.0e-6, 9.5e-6, 0.21 | 1.1e-6 | 6.1e-5, 4.1e-4, 0.15 | 1.02 / 0.000 / -0.02 |
| 100 | q | 2.8e-4, 9.5e-5, 3.0 | 4.1e-5, 2.8e-6, 14.5 | 4.1e-5 | 3.2e-4, 9.5e-5, 3.4 | 0.98 / 0.000 / 0.02 |
| 100 | eps | 7.7e-4, 2.6e-4, 3.0 | 1.6e-4, 1.5e-5, 10.7 | 1.6e-4 | 9.3e-4, 2.6e-4, 3.6 | 0.98 / 0.000 / 0.02 |
| 400 | p | 1.3e-4, 3.3e-4, 0.39 | 9.6e-6, 1.3e-5, 0.77 | 1.0e-5 | 1.2e-4, 3.2e-4, 0.37 | 1.04 / 0.000 / -0.04 |
| 400 | q | 9.3e-5, 6.8e-5, 1.4 | 5.4e-5, 4.2e-6, 12.9 | 5.4e-5 | 1.5e-4, 6.9e-5, 2.1 | 0.99 / 0.000 / 0.01 |
| 400 | eps | 2.2e-4, 1.6e-4, 1.3 | 1.7e-4, 1.8e-5, 9.6 | 1.7e-4 | 3.9e-4, 1.7e-4, 2.4 | 0.98 / 0.010 / 0.01 |

(mu rows for reference: link SNR 2.1 / 0.22 / 0.18 at epochs 0 / 100 / 400, total SNR identical; KL
gradient on mu is ~1e-8 pure noise with a mean of exactly 0 in quadrature, since KL does not depend on mu.)

ring-01, seed 10085000 (same layout, p / q / eps only):

| epoch | output | link: mean, std, SNR | KL_mc/N: mean, std, SNR | KL_quad/N mean | total SNR | var share link / KL / 2cov |
|---|---|---|---|---|---|---|
| 0 | p | 3.8e-5, 1.9e-4, 0.20 | 3.1e-7, 3.2e-6, 0.10 | 1.2e-7 | 0.20 | 1.00 / 0.000 / -0.004 |
| 0 | q | 4.5e-4, 5.7e-5, 7.8 | 1.7e-5, 8.6e-7, 19.6 | 1.7e-5 | 8.1 | 0.99 / 0.000 / 0.011 |
| 0 | eps | 1.3e-3, 1.7e-4, 7.7 | 7.0e-5, 5.1e-6, 13.8 | 7.0e-5 | 8.0 | 0.98 / 0.001 / 0.019 |
| 100 | p | 8.9e-5, 3.2e-4, 0.27 | 1.4e-6, 6.1e-6, 0.23 | 7.3e-7 | 0.27 | 1.02 / 0.000 / -0.016 |
| 400 | p | 1.4e-4, 2.2e-4, 0.61 | 7.1e-6, 7.8e-6, 0.91 | 5.7e-6 | 0.59 | 1.03 / 0.001 / -0.029 |
| 400 | q | 6.4e-5, 5.0e-5, 1.3 | 4.0e-5, 2.7e-6, 15.2 | 4.0e-5 | 2.1 | 0.99 / 0.003 / 0.007 |
| 400 | eps | 1.6e-4, 1.2e-4, 1.3 | 1.3e-4, 1.2e-5, 10.8 | 1.3e-4 | 2.3 | 0.97 / 0.010 / 0.017 |

The `var_share` columns are sums over elements of the 48-draw variance, relative to the total's; "2cov" is
`2 Cov(link, KL_mc)`.

**The KL term carries essentially none of the gradient variance** (<= 1% of the variance of any output at any
probed epoch; <= 0.1% for p) because its gradient is both small (it enters divided by N) and, for q and eps,
nearly noise-free (SNR 10-20). For p, the KL's mean gradient is 1-7% of the link term's mean (0.4% at init)
and the exact value at p=.5 is 0. Substituting the exact quadrature KL changes the total SNR by < 1% for p
(0.128 -> 0.128, 0.150 -> 0.150, 0.37 -> 0.36 on bipartite) and by 0.5-1% for q and eps (7.83 -> 7.88,
7.75 -> 7.80 at epoch 0). Quadrature does help eps and q at the margin (the MC KL adds ~1% noise variance
and the zero-noise KL mean gradient is the same): negligible.

### Source of the link-loss noise (bipartite-00, link term only, 48 draws)

`noise_source.py` re-draws only the rsample noise (z), only the negative pairs, or both
(explicit seeds per draw; same trained trajectory). Noise norm (SNR) for p / q / eps:

| epoch | both | z only (negatives fixed) | negatives only (z fixed) |
|---|---|---|---|
| 0 | 3.3e-4 (.15) / 8.3e-5 (8.6) / 2.4e-4 (8.4) | 3.2e-4 (.29) / 7.7e-5 (9.1) / 2.3e-4 (8.8) | 2.2e-4 (.89) / 4.1e-5 (17) / 1.2e-4 (17) |
| 100 | 4.4e-4 (.13) / 8.5e-5 (3.5) / 2.3e-4 (3.5) | 4.6e-4 (.13) / 8.2e-5 (3.5) / 2.3e-4 (3.4) | 2.2e-4 (1.8) / 4.9e-5 (5.7) / 1.5e-4 (5.3) |
| 400 | 3.3e-4 (.28) / 6.6e-5 (1.2) / 1.5e-4 (1.2) | 3.3e-4 (.25) / 6.8e-5 (1.2) / 1.6e-4 (1.2) | 1.9e-4 (2.0) / 4.4e-5 (1.8) / 1.1e-4 (1.8) |

Re-drawing z alone reproduces the full noise (93-100% of the variance of p, q, eps); re-drawing negatives
alone gives roughly 40-50% of p's variance and 25-30% of q's (not additive: each is measured conditional
on the other held fixed, and the two interact through the same gradient). The noise in the p gradient
is therefore predominantly **the latent sample z of the reparameterised link term**, not the KL or the
negative sampling.

### Interpretation (what is and is not established)

- The p noise std (2-4e-4) is similar to eps's (1.5-2.8e-4) in magnitude. What differs is the mean:
  p's mean gradient is 1/50 of eps's at init (4e-5 vs 2.1e-3). So "p is noisy" is better described as
  "p's expected gradient is small relative to a noise floor that all three outputs share".
  Not established: *why* the expected link-loss gradient w.r.t. p is small near p = 0.5 (the plausible
  reading, untested, is the same symmetry: at p = .5 the posterior is axis-symmetric and p enters only
  at first order through the odd-in-w part of the density).
- The KL cannot be blamed for the p plateau: its exact p-gradient is 0 at p = 0.5, its MC estimate adds <0.1%
  variance, and it does not change SNR measurably when replaced by quadrature.

### (4) A/B quadrature-KL vs MC-KL training: skipped

The KL term's share of p's gradient variance is ~0 (<= 0.1%) and swapping in the exact KL changes the
total SNR of p by < 1% at all probed epochs, so there is no mechanism by which quadrature KL would change
the epoch of escape (baseline ~725 at lr 0.003, bipartite-00 draw 0). `ab_quad.py` is written but was not
run; it would be the right harness if someone wants to check anyway.

## Caveats

One draw/seed per config; 48 draws give SNR estimates with ~10% relative error; probes at three epochs on
a single trajectory. The decomposition is at the head-weight level (what Adam sees), summed over rows of
the `second.weight` output, not per-node. Dropout is effectively off in these runs (the KL_quad term, a
deterministic function of the head output, had std 0 across draws). The quadrature check at (.95, .9, .4)
sits 2.4 se from the float32 MC estimate; that regime is not the one the probes visit.
