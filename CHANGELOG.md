# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

### Added

- Project scaffolding: `uv`-managed package, ruff + pyright + pre-commit,
  GitHub Actions CI, branch protection.
- `tnbbeta_vae.distributions.TNBBetaUnivariate`: the univariate TNBbeta
  distribution (Lederman & Schein, 2026, arXiv:2606.11624), with
  closed-form `log_prob` and reparameterized (`rsample`) sampling via a
  deterministic transform of a `Beta(eps, eps)` draw.
- `tnbbeta_vae.distributions.TNBBetaSpherical`: TNBbeta lifted to the unit
  hypersphere via a Householder reflection (mirroring the von
  Mises-Fisher/Power Spherical construction), with closed-form `log_prob`
  and reparameterized `rsample`.
- `tnbbeta_vae.registry`: Pydantic-config-based model registry
  (`register_model`, `build_model`, `list_registered_models`).
- `tnbbeta_vae.training`: generic `Trainer` loop and local JSON/YAML
  `RunLogger` for train-run metadata.
- `tnbbeta_vae.models.conv_vae.ConvTNBBetaSphericalVAE`: the first
  trainable model -- a simple conv encoder/decoder with a
  `TNBBetaSpherical` latent posterior/prior, registered as
  `"conv_tnbbeta_spherical_vae"`.
- `tnbbeta_vae.models.losses.monte_carlo_elbo`: a generic Monte Carlo
  ELBO, averaged over `num_samples` reparameterized draws (works for any
  reparameterizable distribution with tractable `log_prob`, not just
  closed-form-KL families like Gaussian -- TNBBetaSpherical has no known
  closed form for KL between distributions with different mean
  directions, same as von Mises-Fisher).
- `tnbbeta_vae.models.priors.FixedTNBBetaSphericalPrior`: a fixed,
  non-learnable `TNBBetaSpherical` prior (mirrors the role of N(0, I) in
  a vanilla VAE).
- `tnbbeta_vae.models.diagnostics.tnbbeta_spherical_posterior_diagnostics`:
  per-batch posterior-collapse statistics (p/q min/mean/max, pairwise
  mean-direction cosine similarity), now logged by `ConvTNBBetaSphericalVAE
  .training_step` alongside loss/kl.
- `tnbbeta_vae.data.gaussian_blob_batch`: a synthetic dataset with known
  ground-truth generative factors (hue, position), for fast local
  experiments before touching real data.
- `tnbbeta_vae.models.priors.uniform_prior_params(dim)`: returns
  `(p, q, epsilon) = (0.5, 0, (dim-1)/2)`, the exact parameters making
  `TNBBetaSpherical` reduce to `Uniform(S^(dim-1))` -- verified by
  checking `log_prob` is constant across random points and independent
  of `mean_direction`. Unlike a concentrated prior, matching this one
  gives the encoder no cheap "collapse" target: a posterior that
  collapsed toward it would itself have to become uniform (near-random
  output regardless of `x`), a far worse reconstruction trade than
  collapsing toward a concentrated prior's single point.
