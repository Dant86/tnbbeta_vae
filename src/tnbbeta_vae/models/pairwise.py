"""Pairwise inner-product scoring, shared by every dot-product-plus-BCE model.

Extracted from :meth:`tnbbeta_vae.models.graph_vae.GraphVAE.link_logits`'s body
into a pure, stateless function, so :class:`~tnbbeta_vae.models.
pairwise_mlp_vae.PairwiseMlpVAE` (which has no graph, and thus no
``link_logits`` method of its own to copy) can reuse the identical formula
without duplicating it. ``GraphVAE`` itself keeps its own
``log_temperature``/``temperature()`` unchanged (see that module and
``CLAUDE.md``'s checkpoint-compatibility note) and simply calls this function
with the temperature it already computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["pairwise_logits"]


def pairwise_logits(
    z: Tensor, pairs: Tensor, family: str, temperature: Tensor
) -> Tensor:
    """Returns each pair's logit: an inner product, scaled unless Gaussian.

    Args:
        z: Latents, shape ``(num_items, latent_dim)``.
        pairs: Item-index pairs, shape ``(2, num_pairs)``.
        family: The latent family the inner product was computed for. For
            ``"gaussian"`` the raw inner product is returned unscaled, since its
            latent scale is already free; every other family's inner product of
            unit vectors lies in ``[-1, 1]``, which ``temperature`` rescales.
        temperature: Scalar multiplier applied for every non-Gaussian family.

    Returns:
        Each pair's logit, shape ``(num_pairs,)``.
    """
    inner = (z[pairs[0]] * z[pairs[1]]).sum(-1)
    if family == "gaussian":
        return inner
    return temperature * inner
