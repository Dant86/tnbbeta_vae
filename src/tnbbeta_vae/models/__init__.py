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

__all__ = ["ConvTNBBetaSphericalVAE", "ConvTNBBetaSphericalVAEConfig"]
