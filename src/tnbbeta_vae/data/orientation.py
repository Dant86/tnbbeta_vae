"""Structure-tensor dominant line orientation and coherence for image batches.

General-purpose (not tied to any one dataset): DTD's oriented textures
(``tnbbeta_vae.data.dtd``) are its first user, via
``apps/data/curate_dtd_categories.py`` and ``apps/eval/dtd_orientation_probe.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn.functional import conv2d

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["structure_tensor_orientation"]

# Standard 3x3 Sobel kernels (horizontal/vertical gradient).
_SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
_SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]


def structure_tensor_orientation(
    images: Tensor, *, eps: float = 1e-8
) -> tuple[Tensor, Tensor]:
    """Computes each image's dominant line orientation (mod pi) and coherence.

    The structure tensor is the per-pixel gradient outer product, summed over
    the whole image, giving a 2x2 symmetric matrix
    ``[[Ixx, Ixy], [Ixy, Iyy]]`` per image. Its principal-axis angle is the
    dominant orientation; its anisotropy (normalized by its trace) is the
    coherence -- how strongly the image is oriented at all, independent of
    which direction.

    Args:
        images: Images, shape ``(batch, channels, height, width)``, values in
            ``[0, 1]``. Converted to grayscale by averaging channels when
            ``channels > 1``.
        eps: Added to the coherence denominator so a constant image (zero
            gradient everywhere) gives coherence ``0`` rather than ``nan``.

    Returns:
        A tuple ``(angle, coherence)``, each of shape ``(batch,)``. ``angle``
        is in ``(-pi/2, pi/2]`` -- the orientation mod ``pi``, since a line's
        direction has no head or tail. ``coherence`` is in ``[0, 1]``: near 1
        for a strongly, consistently oriented image (e.g. parallel stripes),
        near 0 for one with no dominant direction (e.g. noise).
    """
    gray = images.mean(dim=1, keepdim=True) if images.shape[1] > 1 else images

    sobel_x = torch.tensor(_SOBEL_X, dtype=gray.dtype, device=gray.device).view(
        1, 1, 3, 3
    )
    sobel_y = torch.tensor(_SOBEL_Y, dtype=gray.dtype, device=gray.device).view(
        1, 1, 3, 3
    )
    gx = conv2d(gray, sobel_x, padding=1)
    gy = conv2d(gray, sobel_y, padding=1)

    ixx = (gx * gx).sum(dim=(1, 2, 3))
    iyy = (gy * gy).sum(dim=(1, 2, 3))
    ixy = (gx * gy).sum(dim=(1, 2, 3))

    angle = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
    coherence = torch.sqrt((ixx - iyy) ** 2 + 4 * ixy**2) / (ixx + iyy + eps)
    return angle, coherence
