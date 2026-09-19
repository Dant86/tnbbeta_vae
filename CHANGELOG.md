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
- `ConvEncoder`/`ConvDecoder` now have `GroupNorm` after every
  conv/deconv/projection layer (except the final output). Motivated by a
  geodesic sensitivity sweep (see below) showing the decoder swinging
  output color wildly along *any* latent direction rather than some
  dedicated "color" subspace -- a poorly-conditioned, entangled mapping
  that left no safe direction for the encoder to route weakly-rewarded
  information through without disturbing already-encoded information.
  `hidden_channels` must now be divisible by 8.
- `tnbbeta_vae.models.diagnostics.sphere_geodesic_sweep` and
  `random_tangent_direction`: move a point along a true great-circle
  geodesic at a controlled angular distance. Perturbing a single
  Cartesian coordinate and renormalizing back onto the sphere is *not* a
  fair way to compare sensitivity across directions -- the actual
  angular distance moved for a fixed offset depends on how much that
  coordinate already overlaps with the base point.
- `tnbbeta_vae.models.conv_gaussian_vae.ConvGaussianVAE` (registered as
  `"conv_gaussian_vae"`): a standard diagonal-Gaussian VAE on the same
  `ConvEncoder`/`ConvDecoder`, N(0, I) prior, closed-form KL. A baseline for
  separating "is it the TNBBetaSpherical latent?" from "is it the
  architecture/data/optimization/likelihood?".
- `monte_carlo_elbo(..., analytic_kl=True)`: use
  `torch.distributions.kl_divergence` (exact, deterministic, always >= 0)
  instead of a Monte Carlo KL estimate where a closed form exists. Raises
  `NotImplementedError` for pairs without one (e.g. anything involving
  `TNBBetaSpherical`) rather than silently falling back.
- `gaussian_posterior_diagnostics`: Gaussian counterpart to the spherical
  collapse diagnostics (sigma stats, across-batch mu std, active units).

### Fixed

- Removed `ConvDecoder`'s `GroupNorm` on the initial dense
  latent -> feature-map projection (`project_norm`). Normalizing there
  measurably worsened collapse in a controlled experiment (posterior
  concentration and directional collapse both increased, reconstruction
  quality dropped) -- the narrowest point in the network, where a
  handful of true degrees of freedom expand into a much larger feature
  map, isn't a safe place for GroupNorm: its group statistics end up
  nearly identical regardless of the specific input, washing out the
  very z-dependence it needs to preserve. The conv/deconv layers' own
  GroupNorms (operating on an already spatially-expanded representation)
  are unaffected and stay.
