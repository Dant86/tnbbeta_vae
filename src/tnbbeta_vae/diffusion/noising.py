"""Forward-noising a point on the sphere toward TNBBetaSpherical's reference.

The latitude (``diffusion.forward``/``diffusion.schedule``) is the whole
problem: the non-latitude direction is always drawn fresh and uniform,
independent of the latitude's own parameters, and the mean direction never
needs to move -- a uniform latitude plus an independent uniform azimuthal
direction, reflected through *any* fixed pole, is exactly rotationally
symmetric (see TNBBetaSpherical's own module docstring, and
docs/superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md point
5). ``noise_to`` is therefore the same two-step construction
``TNBBetaSpherical.rsample()`` already uses, with the forward-noised
latitude from ``diffusion.schedule`` in place of a fresh TNBbeta draw.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tnbbeta_vae.diffusion.schedule import DiffusionSchedule
from tnbbeta_vae.distributions.tnbbeta_spherical import householder_reflect

__all__ = ["noise_to"]


def noise_to(
    mean_direction: Tensor,
    t: Tensor,
    schedule: DiffusionSchedule,
    p_start: float,
    q_start: float,
    eps_start: float,
) -> Tensor:
    """Forward-noises ``mean_direction`` to diffusion time ``t``.

    Args:
        mean_direction: Unit vectors to noise from, shape ``(..., dim)``.
        t: Diffusion time, >= 0, shape ``(...)`` matching ``mean_direction``'s
            batch dimensions.
        schedule: The forward process's target constants.
        p_start: Near-point-mass starting ``p`` (close to 1).
        q_start: Starting ``q`` (0 by design -- no ring structure at the
            point-mass start).
        eps_start: Near-point-mass starting ``eps`` (large).

    Returns:
        Unit-norm points on the sphere, same shape as ``mean_direction``.
    """
    w_t = schedule.draw_latitude(p_start, q_start, eps_start, t)
    dim = mean_direction.shape[-1]
    g = torch.randn(
        (*w_t.shape, dim - 1), dtype=mean_direction.dtype, device=mean_direction.device
    )
    v = g / g.norm(dim=-1, keepdim=True)
    radius = torch.sqrt((1 - w_t**2).clamp_min(0)).unsqueeze(-1)
    pole_frame = torch.cat([w_t.unsqueeze(-1), radius * v], dim=-1)
    return householder_reflect(pole_frame, mean_direction)
