"""Closed-form forward-noising primitives for the TNBbeta latitude.

Three pieces, each verified against TNBBetaUnivariate's closed-form density
in notebooks/tnbbeta_*.py before being promoted here (see
docs/superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md for the
derivation):

1. ``resize_eps``: TNBbeta's Theorem-4.1 auxiliary count C is exactly
   mergeable/splittable in its ``eps`` (size) parameter, holding ``q``
   fixed -- a Beta-Binomial thinning (eps decreasing) or an independent
   negative-binomial merge (eps increasing).
2. ``leisen_step``: Leisen, Mena, Palma Mancilla & Rossini (2019,
   arXiv:1812.07271)'s closed-form reversible NB(r,q) Markov chain, used as
   a one-shot (no path simulation) forward-noising kernel for C that
   converges to NB(eps, q_target) as t -> infinity, regardless of the
   starting q.
3. ``draw_latitude``: combines both with the exact p-decoupling identity
   (p enters the latitude purely as an additive logit-space shift) into the
   full forward-noising recipe for the latitude ``w = 2y - 1``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["draw_latitude", "leisen_step", "resize_eps"]

_EPS = 1e-6
_LOGIT_CLAMP = 1e-6


def resize_eps(c: Tensor, eps_from: Tensor, eps_to: Tensor, q: Tensor) -> Tensor:
    """One-shot resize of an NB(eps_from, 1-q) count to NB(eps_to, 1-q), same q.

    ``eps_to < eps_from`` (per element): Beta-Binomial splits ``c`` and
    keeps the ``eps_to``-sized piece. ``eps_to > eps_from``: merges in an
    independent fresh ``NB(eps_to - eps_from, 1-q)`` piece. Mixed batches
    (some elements splitting, others merging) are supported.

    Args:
        c: Counts to resize, any shape.
        eps_from: Current size parameter, broadcastable to ``c``.
        eps_to: Target size parameter, broadcastable to ``c``.
        q: TNBbeta concentration parameter (shared, unaffected by the
            resize), broadcastable to ``c``.

    Returns:
        Resized counts, same shape as the broadcast of the inputs.
    """
    c, eps_from, eps_to, q = torch.broadcast_tensors(c, eps_from, eps_to, q)
    delta = eps_to - eps_from
    is_split = delta <= 0

    # Both branches are computed for every element (torch.where evaluates
    # eagerly), so each branch's own inputs are clamped positive even where
    # that branch's result will be discarded -- otherwise an invalid shape
    # parameter (e.g. a negative Beta shape) could raise or produce nan
    # before the where() ever gets to pick the valid branch.
    split_pi = torch.distributions.Beta(
        eps_to.clamp_min(_EPS), (-delta).clamp_min(_EPS)
    ).sample()
    split_result = torch.distributions.Binomial(total_count=c, probs=split_pi).sample()

    merge_piece = torch.distributions.NegativeBinomial(
        delta.clamp_min(_EPS), probs=q, validate_args=False
    ).sample()
    merge_result = c + merge_piece

    return torch.where(is_split, split_result, merge_result)


def leisen_step(
    c0: Tensor, t: Tensor, eps: Tensor, q_target: float, speed: float
) -> Tensor:
    """Closed-form, one-shot draw of C_t | C_0 (no intermediate-step simulation).

    Identity at ``t=0``; converges to ``NB(eps, q_target)`` as ``t ->
    infinity``, regardless of ``C_0``'s own distribution.

    Args:
        c0: Starting counts, any shape.
        t: Diffusion time, broadcastable to ``c0``, >= 0.
        eps: Size parameter (held fixed across this call), broadcastable to ``c0``.
        q_target: Target concentration in ``[0, 1)`` -- NB(r, q) requires q < 1,
            so this can be arbitrarily close to but never exactly 0.
        speed: Leisen et al.'s rate constant ``c`` (name avoided here to not
            collide with the count tensor ``c0``).

    Returns:
        C_t, same shape as the broadcast of the inputs.
    """
    c0, t, eps = torch.broadcast_tensors(c0, t, eps)
    theta_t = (1 - q_target) / (torch.exp(speed * t) - q_target)
    y_bin = torch.distributions.Binomial(total_count=c0, probs=theta_t).sample()
    z = torch.distributions.NegativeBinomial(
        eps + y_bin, probs=q_target * (1 - theta_t)
    ).sample()
    return y_bin + z


def draw_latitude(
    p_data: Tensor | float,
    q_data: Tensor | float,
    eps_data: Tensor | float,
    t: Tensor,
    eps_target: float,
    q_target: float,
    speed: float,
) -> Tensor:
    """Draws the forward-noised latitude w_t = 2*Y_t - 1, in (-1, 1).

    At ``t=0`` this recovers ``TNBbeta(p_data, q_data, eps_data)`` (as a
    latitude) exactly; as ``t -> infinity`` it converges to
    ``TNBbeta(0.5, q_target, eps_target)``, regardless of
    ``(p_data, q_data, eps_data)``.

    Args:
        p_data: The starting TNBbeta p parameter, scalar or tensor.
        q_data: The starting TNBbeta q parameter, scalar or tensor.
        eps_data: The starting TNBbeta eps parameter, scalar or tensor.
        t: Diffusion time, >= 0, any shape.
        eps_target: The schedule's asymptotic size parameter (for the
            spherical model this is ``(latent_dim - 1) / 2``, not an
            arbitrary constant -- see
            ``tnbbeta_vae.models.priors.tnbbeta_spherical.uniform_prior_params``).
        q_target: The schedule's asymptotic concentration, in ``[0, 1)``.
        speed: Leisen et al.'s rate constant.

    Returns:
        w_t, same shape as ``t``.
    """
    p_data, q_data, eps_data, t = torch.broadcast_tensors(
        torch.as_tensor(p_data, dtype=t.dtype),
        torch.as_tensor(q_data, dtype=t.dtype),
        torch.as_tensor(eps_data, dtype=t.dtype),
        t,
    )
    eps_t = eps_target + (eps_data - eps_target) * torch.exp(-t)
    c_data = torch.distributions.NegativeBinomial(
        eps_data, probs=q_data, validate_args=False
    ).sample()
    c0_prime = resize_eps(c_data, eps_data, eps_t, q_data)
    c_t = leisen_step(c0_prime, t, eps_t, q_target, speed)

    a_t = torch.distributions.NegativeBinomial(eps_t + c_t, probs=0.5).sample()
    b_t = torch.distributions.NegativeBinomial(eps_t + c_t, probs=0.5).sample()
    u_t = torch.distributions.Beta(eps_t + c_t + a_t, eps_t + c_t + b_t).rsample()

    psi_t = _logit(p_data) * torch.exp(-t)
    y_t = torch.sigmoid(psi_t + _logit(u_t))
    return 2 * y_t - 1


def _logit(x: Tensor) -> Tensor:
    """log(x / (1-x)), with x clamped away from the boundary for finiteness."""
    x = x.clamp(_LOGIT_CLAMP, 1 - _LOGIT_CLAMP)
    return torch.log(x) - torch.log1p(-x)
