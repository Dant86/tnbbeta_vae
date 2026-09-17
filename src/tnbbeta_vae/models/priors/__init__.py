"""Latent priors, e.g. TNBBetaSpherical-based priors."""

from tnbbeta_vae.models.priors.tnbbeta_spherical import (
    FixedTNBBetaSphericalPrior,
    uniform_prior_params,
)

__all__ = ["FixedTNBBetaSphericalPrior", "uniform_prior_params"]
