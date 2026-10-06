# TNBBeta-Spherical Diffusion Prior: Design Spec

- **Date:** 2026-10-05
- **Branch:** `research/sphere-diffusion-prior`
- **Status:** design, not yet implemented

## Motivation

`ConvTNBBetaSphericalVAE`'s `generate()` decodes samples from a fixed
`Uniform(sphere)` prior. Standard VAE failure mode: that prior rarely matches
the aggregate posterior `E_x[q(z|x)]` a trained encoder actually produces, so
prior samples land in low-density regions of the decoder's training
distribution and decode to blurry, "averaged" images. This spec covers
training a learned prior -- a score-matching/diffusion process native to the
`TNBBetaSpherical` family -- to sample from (an approximation of) the
aggregate posterior instead, and testing whether that produces visibly
sharper CIFAR-10 samples than the existing fixed-prior baseline.

This is exploratory research (see `notebooks/tnbbeta_*.py` for the math this
spec is built on), not a finished theoretical result -- several pieces are
explicitly approximations or open questions, flagged as such below.

## Background: what's established

All claims below are numerically verified (KS tests at N=200,000,
`notebooks/tnbbeta_*.py`) against `TNBBetaUnivariate`'s closed-form density,
except where marked otherwise. This section is a condensed reference, not a
re-derivation; see the scripts for the checks themselves.

1. **TNBbeta is exactly "mergeable/splittable" in `epsilon`**, holding
   `(p, q)` fixed: summing two independent Theorem-4.1 auxiliary-count
   triples at `eps1, eps2` (or Beta-Binomial-splitting one at `eps1+eps2`)
   reproduces `TNBbeta(p, q, eps1+eps2)` (resp. the two pieces) exactly. A
   fully continuous, differentiable realization exists via
   `Beta(eps,eps) = Gamma(eps,1)/(Gamma(eps,1)+Gamma(eps,1))` and Gamma's
   shape-additivity -- verified correct and gradient-bearing in PyTorch, but
   **not required for this design**, since forward-noising sampling doesn't
   need to be differentiable (see point 6).

2. **`q` has an exact closed-form forward-noising kernel**, from Leisen,
   Mena, Palma Mancilla & Rossini (2019, arXiv:1812.07271)'s reversible
   NB(r,𝐪) Markov chain: `Theta_t = (1-q_target)/(exp(c*t) - q_target)`
   gives a one-shot (no path simulation) sampler for `C_t | C_0` at any `t`,
   identity at `t=0`, and converges to `NB(eps, q_target)` as `t -> inf`,
   **regardless of the starting `q`**. `NB(r,𝐪)` requires `𝐪 < 1`, so this
   reaches `q_target` arbitrarily close to but never exactly `0`.

3. **`p` decouples exactly**: `logit(TNBbeta(p,q,eps)) = logit(p) +
   logit(TNBbeta(0.5,q,eps))`, an exact algebraic identity (not
   approximate). So `p -> 0.5` is a plain deterministic decay of
   `psi_t = logit(p_t)` to 0, independent of the `q`/`eps` machinery --
   point (1) and (2)'s mean-shrinkage counterpart, same role `sqrt(ᾱ_t)·x_0`
   plays in Euclidean diffusion.

4. **Combined schedule, verified end-to-end**
   (`notebooks/tnbbeta_full_schedule_check.py`): `eps_t = eps_target +
   (eps_data-eps_target)*exp(-t)` (one-shot Beta-Binomial resize from the
   data's `eps_data`), `q_t` via point 2's kernel, `psi_t =
   logit(p_data)*exp(-t)`. `t=0` recovers the data distribution exactly;
   `t >= 6` (at `speed=1`) converges to `TNBbeta(0.5, q_target,
   eps_target)` regardless of starting `(p_data, q_data, eps_data)`
   (tested with `eps_target=1`, both above and below it, i.e. both the
   merge and split directions).

   **Important correction caught while writing the implementation plan:**
   the verification scripts used `eps_target=1` throughout, which makes the
   *latitude* `Y` uniform on `(0,1)` in isolation -- but `eps=1` does
   **not** make the full spherical distribution uniform on `S^(dim-1)`.
   `TNBBetaSpherical`'s density carries a `(1-w^2)^((dim-3)/2)` Jacobian
   factor (see `tnbbeta_spherical.py`'s `log_prob`), so the latitude needs
   density `(1-w^2)^((dim-3)/2)` to cancel it, i.e. `eps = (dim-1)/2`
   (`uniform_prior_params(dim)` in `models/priors/tnbbeta_spherical.py`
   already encodes exactly this: `(p,q,eps) = (0.5, 0, (dim-1)/2)`). The
   merge/split/Leisen-kernel mechanism verified in points 1-4 is correct
   for *any* fixed target `eps`, including this one -- only the target
   *value* used when wiring this into the spherical model needs to be
   `eps_target = (latent_dim-1)/2`, not the `eps_target=1` the standalone
   verification scripts used. The implementation plan uses
   `uniform_prior_params(latent_dim)[2]` as `eps_target`, not a literal
   `1`.

5. **The spherical lift needs no separate treatment.** In
   `TNBBetaSpherical.rsample()`, the non-latitude direction is drawn
   fresh and `Uniform(S^(dim-2))` independent of `(p,q,eps)`, and
   `mean_direction` never needs to move -- a uniform latitude plus an
   independent uniform azimuthal direction, reflected through *any* fixed
   pole, is exactly rotationally symmetric, i.e. literal `Uniform(sphere)`.
   So diffusing the scalar latitude (points 1-4) while holding
   `mean_direction` fixed at the data point **is** the full spherical
   problem, not a reduction of it.

6. **Open gap, resolved by a scoped decision, not by more math:** we
   verified the forward process *samples* correctly, but never derived its
   closed-form *density*, so a literal denoising-score-matching loss isn't
   available. Decision (confirmed with the user): train via **direct
   denoising regression** instead -- predict `z_0` from `(z_t, t)`, plain
   regression loss -- the same simplification DDPM's practical ε-prediction
   loss makes relative to the full ELBO. This also means forward-noising
   sampling never needs to be differentiable (data is fixed per training
   pair; gradients only need to flow through the denoiser network itself),
   which was separately verified true for the `eps`/Gamma mechanism in
   point 1 but is moot either way under this decision.

7. **One-step generation is not a free consequence of a closed-form forward
   process.** Confirmed against DDPM's actual mechanics: the forward
   process being closed-form only yields exact *training targets*; the
   reverse direction still needs multiple small steps because a network's
   prediction from a mostly-noisy `z_t` is necessarily crude (close to a
   conditional-mean estimate), and iterative refinement is what recovers
   real sample diversity rather than regression-to-the-mean. This design
   uses multi-step reverse sampling (see "Generation" below), not a
   single-shot denoiser call.

8. **Joint VAE+diffusion training was considered and explicitly rejected**
   for this experiment, both on general grounds (Rombach et al. 2022,
   *Latent Diffusion Models*, found joint training "require[s] a difficult
   weighting between reconstruction and generative capabilities" and is
   outperformed by a frozen two-stage split) and on this project's specific
   grounds: the research question is whether a better-matched *prior*
   fixes blurry samples, and that claim is only attributable if the
   encoder/decoder are held fixed -- a jointly-trained encoder could
   reshape its own posterior to be easier for the diffusion prior to model,
   confounding the result.

## Architecture

Two registered models, both trainable via the existing
`apps/train/main.py` CLI unchanged (both receive plain CIFAR-10 image
batches):

```
Stage 1 (existing model, run if no checkpoint exists):
  conv_tnbbeta_spherical_vae  -->  checkpoints/<vae_run>/final.pt

Stage 2 (new model, this spec):
  tnbbeta_spherical_diffusion_prior
    - holds the Stage-1 VAE, frozen (LDM-style: eval(), train() neutralized,
      requires_grad=False on every parameter)
    - holds a small trainable MLP denoiser
    - training_step(images): same frozen-encode-on-the-fly pattern as
      CompVis latent-diffusion's LatentDiffusion.get_input (verified against
      their actual training code): z_0 = vae.posterior(images).rsample()
      .detach() each batch, not precomputed/cached.
    -->  checkpoints/<diffusion_run>/final.pt
```

### New library code: `src/tnbbeta_vae/diffusion/`

- `forward.py`: `resize_eps(c, eps_from, eps_to, q, ...)` (Beta-Binomial
  split / NB merge, point 1), `leisen_step(c0, t, speed, eps, q_target,
  ...)` (point 2), `draw_latitude_t(p_data, q_data, eps_data, t, ...)` (the
  full combined recipe, point 4) -- promoted and cleaned up from
  `notebooks/tnbbeta_full_schedule_check.py`, operating batched over a
  leading tensor dimension.
- `schedule.py`: a small `DiffusionSchedule` config object holding
  `q_target`, `speed`, and the `eps_t(t)` / `psi_t(t)` decay functions, so
  the schedule shape is a tunable rather than hardcoded in `forward.py`.

### `tnbbeta_spherical_diffusion_prior` model

`src/tnbbeta_vae/models/sphere_diffusion.py`, config fields:

| field | meaning | default |
|---|---|---|
| `vae_run_name` | checkpoint dir of the frozen Stage-1 VAE | required |
| `q_target` | forward process's target `q` (can't reach exactly 0) | `0.05` |
| `p_start` | near-point-mass starting `p` (point 3's `psi_t = logit(p_start)*exp(-t)`) | `0.999` |
| `eps_start` | near-point-mass starting `eps` for the diffusion's own schedule (point 6's design choice: fixed, shared, *not* read from the VAE's per-example posterior) | `50.0` |
| `q_start` | starting `q`, fixed at `0` by design (no ring structure at the point-mass start) -- not exposed as a tunable | `0.0` (fixed) |
| `t_max` | nominal diffusion-time horizon training samples `t` from (`Uniform(0, t_max)`); point 4 showed `speed=1` converges by `t~6`, so this needs headroom past that | `10.0` |
| `speed` | Leisen kernel rate constant `c` | `1.0` |
| `num_reverse_steps` | reverse sampler step count, spaced over `(0, t_max]` | `50` |
| `denoiser_hidden_dim`, `denoiser_depth` | MLP size | `256`, `4` |

- **Denoiser**: `[z_t (latent_dim) ; sinusoidal(t)] -> MLP -> latent_dim`,
  L2-normalized to the unit sphere. No convolutions -- `latent_dim` is a
  small vector (default 8), not spatial.
- **Noising a point `m` to time `t`** (shared by both methods below, call it
  `noise_to(m, t)`): draw latitude `w_t = forward.draw_latitude_t(p_start,
  q_start, eps_start, t, ...)` (point 4's recipe, a scalar per example);
  draw a fresh `v ~ Uniform(S^(dim-2))` independent of `w_t` (point 5);
  combine into the pole-frame point `(w_t, sqrt(1-w_t^2)*v)`; Householder-
  reflect onto `m` using `TNBBetaSpherical`'s existing
  `_householder_reflect` (exposed as a small public helper rather than
  re-implemented) -- the same two-step construction
  `TNBBetaSpherical.rsample()` already uses, just with `m` as the pole
  instead of a learned `mean_direction`.
- **`training_step(images)`**: frozen `vae.posterior(images).rsample()
  .detach()` -> `z_0`; `t ~ Uniform(0, t_max)`; `z_t = noise_to(z_0, t)`;
  denoiser predicts `ẑ_0` from `(z_t, t)`; loss `= 1 -
  cosine_similarity(ẑ_0, z_0)` (simple, scale-free; MSE is a documented
  alternative to try if cosine underperforms).
- **`generate(n)`**: `z ~ Uniform(sphere)` (reuse
  `FixedTNBBetaSphericalPrior(latent_dim, *uniform_prior_params(latent_dim))`,
  i.e. `p=0.5, q=0, eps=(latent_dim-1)/2` -- see the correction in point 4);
  for a decreasing
  step sequence `t_K=t_max > ... > t_1 > t_0=0`: `ẑ_0 = denoiser(z, t_k)`,
  then `z = noise_to(ẑ_0, t_{k-1})` (predict-then-renoise, point 7 -- an
  established pragmatic pattern, not a derived-optimal one); final `z`
  (from the `t_0=0` step, i.e. `ẑ_0` itself, no further noising) decoded
  through the frozen `vae.decoder`.

### Evaluation

- `apps/eval/fid.py` **unchanged** -- it already does exactly
  `load_model_checkpoint` + `model.generate(n)` + FID against the CIFAR-10
  test set, so pointing `--run-name` at the diffusion run gives `fid_prior`
  directly comparable to the plain VAE's own `fid_prior` number.
- A small new qualitative script, `apps/eval/sample_grid.py`: decode N
  samples from both the VAE's plain prior and the diffusion prior, save as
  side-by-side image grids for eyeballing (the "sharper than blurry VAE
  samples" check the user asked for isn't fully captured by FID alone).

### Testing

- `tests/diffusion/test_forward.py`: promote the `notebooks/` KS-test
  checks into real `pytest` coverage -- `t=0` identity, large-`t`
  convergence to the target regardless of starting `(p,q,eps)` (merge and
  split directions both), the `p`-decoupling identity. Statistical (KS
  threshold), matching the rigor already used in `notebooks/`.
- `tests/models/test_sphere_diffusion.py`: shape/dtype sanity on
  `training_step` and `generate`, a frozen-VAE-stays-frozen check (no
  `requires_grad` leaks, `.train()` on the parent model doesn't un-freeze
  it -- mirroring CompVis's `disabled_train` pattern), and a short
  overfit-a-handful-of-examples convergence check (loss decreases).

## Explicit non-goals (deferred, not forgotten)

- Deriving the true closed-form `Y_t` density/score for literal DSM
  (point 6) -- the regression-loss decision sidesteps it for now.
- One-step or few-step generation (point 7) -- multi-step reverse sampling
  only.
- Joint VAE+diffusion training (point 8) -- frozen VAE only.
- A composable schedule proof for `p`/`q` beyond what's in point 2/3 (e.g.
  a true closed-form reverse transition density) -- the predict-then-renoise
  sampler sidesteps needing one.

## Open parameters needing a first real run to tune

`eps_start`, `num_reverse_steps`, `t_max` (the schedule's nominal
diffusion-time horizon), denoiser size, and both models' epoch budgets are
reasonable-starting-point defaults above, not validated against this
dataset yet -- expect the first training run to motivate adjustments.
