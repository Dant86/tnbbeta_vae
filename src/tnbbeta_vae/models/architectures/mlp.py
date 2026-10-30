"""A plain Linear+ReLU MLP stack, shared by every vector-input encoder/decoder.

Extracted from :mod:`tnbbeta_vae.models.mlp_vae`'s private ``_mlp`` helper so
:class:`~tnbbeta_vae.models.pairwise_mlp_vae.PairwiseMlpVAE` (which needs the
identical encoder-building logic but has no decoder to mirror it for) can
reuse it instead of copying it a third time.
"""

from __future__ import annotations

from torch import nn

__all__ = ["mlp_stack"]


def mlp_stack(sizes: list[int]) -> nn.Sequential:
    """Builds a ReLU MLP whose layer widths follow ``sizes``.

    Args:
        sizes: Layer widths; a ``Linear(sizes[i], sizes[i + 1])`` followed by a
            ``ReLU`` is added for each consecutive pair.

    Returns:
        An ``nn.Sequential`` of alternating ``Linear``/``ReLU`` layers, one pair
        per consecutive pair in ``sizes``.
    """
    layers: list[nn.Module] = []
    for width_in, width_out in zip(sizes[:-1], sizes[1:], strict=True):
        layers += [nn.Linear(width_in, width_out), nn.ReLU()]
    return nn.Sequential(*layers)
