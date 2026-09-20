# Week 0 summary (through 2026-09-20)

Goal for the week: a working training run of the TNBBeta VAE on CIFAR-10, with the
Gaussian and von Mises-Fisher (vMF) VAEs as baselines. Status: **achieved**. TNBBeta
trains stably at every size we tried and matches the Gaussian on reconstruction. It does
not (yet) use the extra expressivity of the family, and we now have a fairly precise
picture of why.

## Headline results

1. **TNBBeta ~ Gaussian on reconstruction.** On CIFAR-10 (`hidden_channels=32`, 50
   epochs) PSNR agrees within about 0.1 dB at latent dimension 32 and 128.
2. **A learned likelihood scale settles the sigma question.** A single shared sigma
   (parameterized by log sigma^2) converges to about 0.105 (d=32) and 0.0687 (d=128), for
   both families and from starting values of 0.1 or 1.0. Theory says sigma^2 should equal
   the training MSE; it does, to about 1%.
3. **TNBBeta is more reliably trainable than the S-VAE vMF port.** At d=32, 2 of 3 vMF
   seeds stayed at the mean-image solution for all 50 epochs; none of the 13 TNBBeta runs
   at d=32 (fixed-sigma sweep, smoke run and learned-sigma runs) did. At d=128 the vMF port produces NaNs (float32 underflow of a Bessel function);
   TNBBeta is unaffected. When vMF does train it is indistinguishable from TNBBeta.
4. **The extra parameters are mostly idle.** q is essentially 0 in every fit, so the
   family behaves like its q = 0 special case (log-normal beta), with p (ring radius),
   epsilon (thickness) and the mean direction doing the work.
5. **At d=128 TNBBeta pays more rate but fills its prior better.** KL 304 vs 287 nats for
   equal PSNR (+5.8%), but prior samples are much closer to the data (nearest-neighbour
   ratio 1.12 vs 2.27).

## Method notes

- Models: shared conv encoder/decoder (GroupNorm). TNBBeta posterior on S^(d-1) via a
  Householder lift, uniform-sphere prior (p=0.5, q=0, epsilon=(d-1)/2); Gaussian VAE with
  closed-form KL; vMF VAE ported faithfully from the S-VAE reference (Wood rejection
  sampler, Bessel-function KL).
- ELBO with a Gaussian pixel likelihood. sigma acts as a KL weight: relative to summed
  squared error it is beta_eff = 2 sigma^2.
- CIFAR-10 on the UChicago DSI cluster: SLURM scripts with checkpoint/resume, a fail-fast
  check for GPU-less nodes with automatic resubmission, eval and latent-export scripts.
- Numbers below are 3-seed means unless stated. Everything is one dataset, at most 3
  seeds, a small decoder and 50 epochs.

## Findings in detail

### Synthetic-data phase (colored Gaussian blobs, known hue/position)

- With a fixed likelihood scale, the sphere posteriors learned hue at sigma <= 0.5 but
  stayed at the position-only solution at sigma >= 0.85 (TNBBeta: 1-2 of 3 seeds at 0.7,
  0 of 3 at 0.85 and 1.0; vMF learned at 0.7 in 3 of 3 seeds, failed at 1.0). The Gaussian
  got through sigma = 1.0. This is an optimization trap, not a bad optimum: a model
  trained at sigma = 0.5 scores 6.5 nats better under sigma = 1.0 than the model trained
  at 1.0, and warm-starting it under sigma = 1.0 keeps the hue-encoding solution.
- Not the cause: a Monte Carlo KL (a 64-sample KL changed nothing; a Gaussian with a
  1-sample MC KL still learned hue), the decoder's normalization layer, epsilon's
  initialization or fixing epsilon.

### CIFAR-10 with a fixed sigma

| setting | ELBO Gaussian | ELBO TNBBeta | KL G / T | PSNR G / T |
|---|---|---|---|---|
| d=32, sigma=0.1 | 2378.3 | 2361.4 | 93.5 / 91.2 | 19.36 / 19.32 |
| d=32, sigma=0.2 | 1598.4 | 1591.1 | 66.0 / 68.2 | 19.24 / 19.20 |
| d=128, sigma=0.1 | 3233.7 | 3213.1 | 230.2 / 249.9 | 22.91 / 22.90 |
| d=128, sigma=0.2 | 1748.4 | 1706.7 | 130.1 / 158.4 | 21.99 / 21.76 |

The Gaussian has the higher ELBO everywhere; seeds do not overlap. A 10-epoch smoke run
had shown TNBBeta 6 nats ahead, which did not survive 50 epochs.

### Learned sigma

| | model | final sigma | KL | PSNR |
|---|---|---|---|---|
| d=32 (start 0.1) | Gaussian / TNBBeta | 0.105 / 0.106 | 91.4 / 88.9 | 19.31 / 19.31 |
| d=32 (start 1.0) | Gaussian / TNBBeta | 0.104 / 0.106 | 88.3 / 89.0 | 19.39 / 19.29 |
| d=128 (start 0.1) | Gaussian / TNBBeta | 0.0687 / 0.0688 | 287.1 / 303.9 | 23.12 / 23.13 |

- Starting from sigma = 1.0 (the regime where fixed-sigma sphere runs stuck on the
  synthetic data) did not trap TNBBeta.
- Train vs test gap is the same for both models (about 0.27 dB, 6% in MSE; KL identical on
  train and test): no difference in memorization.
- Prior samples at d=128, sigma near 0.07: nearest-neighbour ratio 1.12 (TNBBeta) vs 2.27
  (Gaussian); the Gaussian's samples have harsh, saturated colors.
- Class structure is equal under a cosine-distance k-NN probe (about 0.41 for both at
  d=32 and d=128). A raw Euclidean probe made the Gaussian look worse (0.35, 0.27); that
  was an artifact of varying vector length.

### How TNBBeta uses its parameters (exported per-image posteriors)

- q is at or near 0 in every run (at most 0.007 at d=128).
- p is interior at d=32 and d=128 (the clamp was not binding), but at d=2 it sat exactly on
  the old 1e-4 clamp.
- The parameterization has an antipodal alias: (mu, p) and (-mu, 1-p) describe the same
  distribution. Different runs chose different branches (p near 0 at d=32, near 1 at
  d=128 and d=2). The export now provides `mode_direction`, which undoes it.
- epsilon is about d-1 (34 at d=32, 130 at d=128), twice the uniform prior's (d-1)/2.
- At d=128 the posterior is a thin shell about 5 degrees from the mean direction. In high
  dimension any rotationally symmetric distribution is a shell; TNBBeta decouples the
  shell's radius (p) from its thickness (epsilon, q), where vMF ties them through kappa.

### Low-dimensional layout studies

- **d=2** (both families): TNBBeta's posterior is a ring about 1 degree in radius (two
  points on the circle), set by the clamp, not learned. The Gaussian shows a semantic
  gradient (airplane and ship on one side, animals clustered) with heavy overlap.
- **d=3** (TNBBeta on S^2), same setting, two regimes:
  - *Cap regime* (seed 0, and the relaxed-clamp run): tight posterior (1 degree), KL 8.1.
    Coarse animals-vs-vehicles split on the sphere, no tight per-class clusters.
  - *Ring regime* (seed 1): q about 0.97, KL 4.46. All mean directions collapse to one
    axis (an antipodal pair of blobs); the code is the ring radius alone, with p spread
    over 0.025-0.943 across images. A one-dimensional scalar code inside a 3-D latent.
  - Relaxing the p/q clamp from 1e-4 to 1e-6 did not change the regime or the KL
    (8.28 vs 8.11).
  - Not yet checked: reconstruction quality (PSNR, learned sigma) of these d=3 runs.

### Theory

- KL(q || uniform sphere) depends only on the latitude marginal. Integrating the density
  numerically reproduces the measured KLs from the exported per-image parameters
  (8.11, 4.46 and 8.28 against 8.11, 4.46 and 8.21).
- At d=3 the uniform prior's latitude marginal is uniform on (0, 1). At the encoder's
  starting point the KL is only 0.015 nats and its gradient in q is +0.010, so the KL
  does not push the model toward the ring basin; that basin must come from the
  reconstruction term or the dynamics.
- At fixed distortion (matched to the cap solution, about 1.3 degrees), the minimum KL is
  nearly flat in q (8.074 at q=0, 8.098 at 0.3, 8.418 at 0.9). q > 0 never helps, but the
  preference for q = 0 is weak. It does not show that q = 0 makes optimization easier.

### Diagnosis of the earlier attempt (github.com/Dant86/hypergen)

The old repo used the same density and lift (checked algebraically), but it fixed
epsilon = 1 while the uniform prior in d=64 needs epsilon = 31.5. With epsilon = 1, KL
falls as q rises (16.9 nats at q=0 down to 0.34 at q=0.99), so the KL rewarded "collapse
to a spike"; the learned epsilon of about 61 was in fact reasonable. A Wasserstein-1
regularizer times 500 replaced the KL, and the loss was summed squared error, which
corresponds to sigma^2 = 0.5, a regime where sphere posteriors are weakly rewarded.

## Corrections made along the way

- An early "round cap" argument (a sphere posterior pays about d-1 nats per e-fold of
  precision) applies to vMF, not to TNBBeta, whose posterior is a ring set by p.
- Hypotheses that turned out wrong: normalization at the decoder bottleneck causing
  collapse; single-sample MC KL noise explaining the sphere's trap; KL pushing the model
  into the ring basin at initialization.
- A k-NN class-accuracy gap between the Gaussian and TNBBeta was a distance-metric
  artifact.

## Open questions

1. Is the ring regime worse for reconstruction than the cap regime, and how often does it
   occur? (Needs the d=3 evals and more seeds.)
2. Why does the model choose the ring at all? Not the KL at initialization; probably the
   decoder's early dynamics.
3. Does q ever help on other data or with a non-uniform prior? On CIFAR it is unused.
4. TNBBeta pays about 6% more rate at d=128 but fills the prior better. Is that a
   structural benefit of the sphere or of TNBBeta specifically?
5. Class layout on the sphere: plots need to use the axis (mu up to sign) for classes near
   p = 0.5, where the alias makes the mode direction ill-defined.

## Repository changes this week

- Added: CIFAR-10 pipeline and cluster scripts, vMF baseline, latent export and plotting,
  learned sigma, Trainer checkpoints, automatic GPU-node retry.
- Removed at the end of the week: the fixed-sigma option, and the configurable p/q clamp
  (now fixed at 1e-6).
