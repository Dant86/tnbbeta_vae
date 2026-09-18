"""Diagnostics for `TNBBetaSpherical`-latent VAEs: collapse and sphere geometry.

In this parameterization, `p -> 0` combined with `q -> 1` is the
distribution's expression of the classic VAE "KL vanishing" / posterior
collapse failure: `q -> 1` collapses the latitude to a point mass at its
median (density diverges there, Proposition 3.1), and since
`mean_direction` is a free encoder output, the encoder can compensate for
`p -> 0` (which alone would put that point mass at the antipode of
`mean_direction`) by simply learning `mean_direction` pointing the other
way -- landing the collapsed point right where the fixed prior already
sits, independent of `x`.

These statistics are meant to be logged every training step (they're
cheap) so a collapse shows up as a trend over a run, not something
noticed only after the fact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

    from tnbbeta_vae.distributions import TNBBetaSpherical

__all__ = [
    "random_tangent_direction",
    "sphere_geodesic_sweep",
    "tnbbeta_spherical_posterior_diagnostics",
]


def tnbbeta_spherical_posterior_diagnostics(
    posterior: TNBBetaSpherical,
) -> dict[str, Tensor]:
    """Computes collapse-monitoring statistics for a batched posterior.

    Args:
        posterior: A batched `TNBBetaSpherical`, e.g. one example per row
            of a training batch.

    Returns:
        A dict of scalar tensors: min/mean/max of `p` and `q`,
        mean `epsilon`, and (when the batch has more than one example)
        `posterior_direction_pairwise_cosine_mean` -- the average cosine
        similarity between different examples' `mean_direction`s. That
        similarity trending toward 1 means the encoder is converging to
        (nearly) the same direction for every input, i.e. ignoring `x`.
    """
    diagnostics = {
        "posterior_p_mean": posterior.p.mean(),
        "posterior_p_min": posterior.p.min(),
        "posterior_p_max": posterior.p.max(),
        "posterior_q_mean": posterior.q.mean(),
        "posterior_q_min": posterior.q.min(),
        "posterior_q_max": posterior.q.max(),
        "posterior_epsilon_mean": posterior.epsilon.mean(),
    }

    mean_direction = posterior.mean_direction
    batch_size = mean_direction.shape[0]
    if batch_size > 1:
        cosine_similarity = mean_direction @ mean_direction.transpose(-1, -2)
        off_diagonal = ~torch.eye(
            batch_size, dtype=torch.bool, device=mean_direction.device
        )
        diagnostics["posterior_direction_pairwise_cosine_mean"] = cosine_similarity[
            off_diagonal
        ].mean()

    return diagnostics


def random_tangent_direction(base_point: Tensor) -> Tensor:
    """Draws a uniformly random unit tangent vector at ``base_point``.

    Args:
        base_point: A unit vector, shape ``(..., dim)``.

    Returns:
        A unit vector orthogonal to ``base_point``, same shape.
    """
    raw = torch.randn_like(base_point)
    tangent = raw - (raw * base_point).sum(-1, keepdim=True) * base_point
    return tangent / tangent.norm(dim=-1, keepdim=True)


def sphere_geodesic_sweep(
    base_point: Tensor, tangent_direction: Tensor, angles: Tensor
) -> Tensor:
    """Moves ``base_point`` along a great circle toward ``tangent_direction``.

    Perturbing a single Cartesian coordinate and renormalizing back onto
    the sphere is *not* a fair way to compare "sensitivity" across
    directions: the actual angular distance moved for a fixed offset
    depends on how much that coordinate already overlaps with
    ``base_point``. This instead moves by exactly ``angles`` radians along
    a true geodesic, for any direction, so sensitivity comparisons across
    different directions are apples-to-apples. Use
    :func:`random_tangent_direction` to get a valid ``tangent_direction``
    (must be a unit vector orthogonal to ``base_point``; not validated
    here).

    Args:
        base_point: A unit vector, shape ``(..., dim)``.
        tangent_direction: A unit vector orthogonal to ``base_point``,
            same shape.
        angles: Angular displacements in radians, shape ``(num_angles,)``.

    Returns:
        Points along the geodesic, shape ``(num_angles, ..., dim)``.
    """
    broadcast_shape = (-1, *([1] * base_point.dim()))
    cos = angles.cos().reshape(broadcast_shape)
    sin = angles.sin().reshape(broadcast_shape)
    return cos * base_point + sin * tangent_direction
