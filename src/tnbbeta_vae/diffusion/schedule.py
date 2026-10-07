"""A tunable forward-noising schedule for the TNBbeta latitude."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from tnbbeta_vae.diffusion.forward import draw_latitude

__all__ = ["DiffusionSchedule"]


@dataclass(frozen=True)
class DiffusionSchedule:
    """The three constants that shape the forward-noising process.

    Attributes:
        eps_target: The schedule's asymptotic size parameter. For the
            spherical model this must be ``(latent_dim - 1) / 2`` (see
            ``tnbbeta_vae.models.priors.tnbbeta_spherical.uniform_prior_params``),
            not an arbitrary constant -- it is derived from ``latent_dim``
            by the caller, not defaulted here.
        q_target: The schedule's asymptotic concentration, in ``[0, 1)``.
        speed: Leisen et al.'s rate constant; larger values converge to the
            target faster as a function of ``t``.
    """

    eps_target: float
    q_target: float = 0.05
    speed: float = 1.0

    def draw_latitude(
        self,
        p_data: Tensor | float,
        q_data: Tensor | float,
        eps_data: Tensor | float,
        t: Tensor,
    ) -> Tensor:
        """Draw forward-noised latitude at time t; see forward.draw_latitude."""
        return draw_latitude(
            p_data, q_data, eps_data, t, self.eps_target, self.q_target, self.speed
        )
