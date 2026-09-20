"""VAE architectures, priors, and losses, indexed via `tnbbeta_vae.registry`.

Importing this package registers every model defined under it (each
model module's ``@register_model`` decorator runs on import), so
``tnbbeta_vae.registry.list_registered_models()`` is only complete after
``import tnbbeta_vae.models``.
"""

from tnbbeta_vae.models.conv_gaussian_vae import ConvGaussianVAE, ConvGaussianVAEConfig
from tnbbeta_vae.models.conv_vae import (
    ConvTNBBetaSphericalVAE,
    ConvTNBBetaSphericalVAEConfig,
)
from tnbbeta_vae.models.conv_vmf_vae import (
    ConvVonMisesFisherVAE,
    ConvVonMisesFisherVAEConfig,
)
from tnbbeta_vae.models.diagnostics import tnbbeta_spherical_posterior_diagnostics
from tnbbeta_vae.models.mlp_vae import MlpVAE, MlpVAEConfig

__all__ = [
    "ConvGaussianVAE",
    "ConvGaussianVAEConfig",
    "ConvTNBBetaSphericalVAE",
    "ConvTNBBetaSphericalVAEConfig",
    "ConvVonMisesFisherVAE",
    "ConvVonMisesFisherVAEConfig",
    "MlpVAE",
    "MlpVAEConfig",
    "tnbbeta_spherical_posterior_diagnostics",
]
