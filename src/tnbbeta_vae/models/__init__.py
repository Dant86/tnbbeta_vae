"""VAE architectures, priors, and losses, indexed via `tnbbeta_vae.registry`.

Importing this package registers every model defined under it (each
model module's ``@register_model`` decorator runs on import), so
``tnbbeta_vae.registry.list_registered_models()`` is only complete after
``import tnbbeta_vae.models``.
"""

from tnbbeta_vae.models.conv_vae import (
    ConvTNBBetaSphericalVAE,
    ConvTNBBetaSphericalVAEConfig,
)
from tnbbeta_vae.models.diagnostics import (
    random_tangent_direction,
    sphere_geodesic_sweep,
    tnbbeta_spherical_posterior_diagnostics,
)

__all__ = [
    "ConvTNBBetaSphericalVAE",
    "ConvTNBBetaSphericalVAEConfig",
    "random_tangent_direction",
    "sphere_geodesic_sweep",
    "tnbbeta_spherical_posterior_diagnostics",
]
