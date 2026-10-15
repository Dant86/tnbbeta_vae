"""Power Spherical distribution (De Cao & Aziz, 2020, arXiv:2006.04437).

A hyperspherical distribution built, like von Mises-Fisher, from the
*tangent-normal decomposition* (Mardia & Jupp, 2009, Theorem 9.1.2, cited
as the source paper's Theorem 1): any unit vector ``x`` on ``S^(d-1)``
decomposes as ``x = mu*t + v*sqrt(1-t^2)``, with ``t = mu^T x in [-1, 1]``
and ``v`` a unit vector tangent to the sphere at ``mu``. Unlike von
Mises-Fisher, whose marginal over ``t`` has no closed-form inverse CDF and
so needs rejection sampling, the Power Spherical distribution is
*defined* by choosing a marginal over ``t`` that is an affine
reparameterization of a Beta distribution -- fully reparameterizable, with
a closed-form density and a closed-form KL to the uniform distribution.

Paper references (equation/theorem numbers as printed in the paper):

* Definition 2 (Eq. 6): the unnormalized density
  ``p_X(x; mu, kappa) ~ (1 + mu^T x)^kappa`` on ``S^(d-1)``, direction
  ``mu``, concentration ``kappa >= 0``.
* Theorem 12 (Eq. 33): the marginal ``t = mu^T x`` satisfies
  ``t = 2*z - 1`` with ``z ~ Beta(alpha, beta)``,
  ``alpha = (d-1)/2 + kappa``, ``beta = (d-1)/2``.
* Theorem 13 (Eq. 48): the normalized density is
  ``p_X(x; mu, kappa) = {2^(alpha+beta) * pi^beta * Gamma(alpha) /
  Gamma(alpha+beta)}^-1 * (1 + mu^T x)^kappa``.
* Algorithm 1: sampling draws ``z ~ Beta(alpha, beta)``, a direction
  ``v ~ Uniform(S^(d-2))``, sets ``t = 2*z - 1``, ``y = [t, sqrt(1-t^2)*v]``,
  and reflects ``y`` from the pole ``e_1`` to ``mu`` with a Householder
  reflection -- exactly the reflection :func:`~tnbbeta_vae.distributions.
  tnbbeta_spherical.householder_reflect` implements, so it's reused here
  (see the note on :func:`rsample` below) rather than duplicated.
* Theorem 15 (Eq. 62): the differential entropy is
  ``H(X) = log N_X(kappa,d) - kappa*(log(2) + psi(alpha) - psi(alpha+beta))``,
  where ``N_X(kappa,d)`` is the Theorem 13 normalizer and ``psi`` is the
  digamma function.
* Theorem 17 (Eq. 72): the KL divergence to ``Uniform(S^(d-1))`` is
  ``D_KL[P || Q] = -H(P) + H(Q)``, with ``H(Q) = log(A_(d-1))`` the log
  surface area of the sphere (Definition 7) -- the same quantity
  :func:`~tnbbeta_vae.distributions.tnbbeta_spherical._log_surface_area`
  computes, so it's reused here too.

At ``kappa = 0``, ``alpha == beta == (d-1)/2``, which makes the
unnormalized density constant (``(1 + mu^T x)^0 == 1``) -- exactly
``Uniform(S^(d-1))``, not merely close to it. This can be checked
algebraically via the Legendre duplication formula
(``Gamma(d-1) = 2^(d-2)/sqrt(pi) * Gamma((d-1)/2) * Gamma(d/2)``), which
shows the Theorem 13 normalizer at ``kappa=0`` equals the sphere's surface
area exactly; :mod:`tests.distributions.test_power_spherical` checks it
numerically too.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Beta, Distribution, constraints
from torch.distributions.kl import register_kl
from torch.distributions.utils import broadcast_all
from torch.types import _size

from tnbbeta_vae.distributions.hyperspherical_uniform import HypersphericalUniform
from tnbbeta_vae.distributions.tnbbeta_spherical import (
    _log_surface_area,
    _unit_sphere,
    householder_reflect,
)

__all__ = ["PowerSpherical"]

_BOUNDARY_EPS = 1e-6


class PowerSpherical(Distribution):
    """Power Spherical distribution on S^(dim-1) (De Cao & Aziz, 2020).

    Rotationally symmetric about ``mean_direction``, like von Mises-Fisher,
    but reparameterizable without rejection sampling and with a
    closed-form KL to the uniform distribution (:meth:`kl_to_uniform`).

    Attributes:
        mean_direction: Mean direction on S^(dim - 1), shape ``(..., dim)``.
            Normalized to unit norm in ``__init__`` if not already.
        kappa: Concentration, >= 0. ``kappa = 0`` is exactly
            ``Uniform(S^(dim - 1))``; larger ``kappa`` concentrates mass
            near ``mean_direction``.
    """

    arg_constraints = {  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        "mean_direction": constraints.real_vector,
        "kappa": constraints.nonnegative,
    }
    support = _unit_sphere  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    has_rsample = True

    def __init__(
        self,
        mean_direction: Tensor,
        kappa: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the distribution.

        Args:
            mean_direction: Mean direction, shape ``(..., dim)`` with
                ``dim >= 2``. Normalized to unit norm.
            kappa: Concentration, >= 0.
            validate_args: Whether to validate arguments and samples.

        Raises:
            ValueError: If ``mean_direction``'s last dimension is < 2.
        """
        if mean_direction.shape[-1] < 2:
            raise ValueError(
                "PowerSpherical requires dim >= 2 (i.e. at least S^1); got "
                f"mean_direction.shape[-1] = {mean_direction.shape[-1]}."
            )
        self.mean_direction = mean_direction / mean_direction.norm(dim=-1, keepdim=True)

        # Float kappa becomes a CPU tensor; keep it with the mean direction.
        (kappa_t,) = (t.to(self.mean_direction.device) for t in broadcast_all(kappa))
        batch_shape = torch.broadcast_shapes(
            self.mean_direction.shape[:-1], kappa_t.shape
        )
        self.kappa = kappa_t.expand(batch_shape)

        event_shape = self.mean_direction.shape[-1:]
        super().__init__(
            batch_shape=torch.Size(batch_shape),
            event_shape=torch.Size(event_shape),
            validate_args=validate_args,
        )

    @property
    def dim(self) -> int:
        """Ambient dimension of the sphere (mean_direction has `dim` components)."""
        return self.event_shape[0]

    def log_prob(self, value: Tensor) -> Tensor:
        """Computes the log-density at ``value`` (Theorem 13, Eq. 48).

        Args:
            value: Points on S^(dim - 1), shape ``(..., dim)``.

        Returns:
            Log-density, broadcast against the distribution's batch shape.
        """
        if self._validate_args:
            self._validate_sample(value)

        # Clamped away from -1 so kappa == 0 (where the (1 + dot)^kappa factor
        # must contribute exactly 0 to the log-density) never multiplies a
        # finite kappa gradient by -inf; see the kappa -> 0 discussion above.
        dot = (self.mean_direction * value).sum(-1).clamp(-1.0 + _BOUNDARY_EPS, 1.0)

        return self.kappa * torch.log1p(dot) - self._log_normalizer()

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:  # noqa: B008
        """Draws reparameterized samples via Algorithm 1 (Beta, then reflect).

        Args:
            sample_shape: Shape of the i.i.d. sample batch to draw.

        Returns:
            Unit-norm samples in S^(dim - 1), shape
            ``sample_shape + batch_shape + (dim,)``, differentiable w.r.t.
            ``mean_direction`` and ``kappa`` (via PyTorch's implicit
            reparameterization gradient for :class:`~torch.distributions.Beta`,
            as in Figurnov et al. 2018, cited by the paper for exactly
            this purpose -- no rejection sampling or correction term).
        """
        shape = self._extended_shape(sample_shape)
        batch_shape, dim = shape[:-1], shape[-1]

        kappa = self.kappa.expand(batch_shape)
        mean_direction = self.mean_direction.expand(shape)

        beta = torch.full_like(kappa, (dim - 1) / 2)
        alpha = beta + kappa
        z = Beta(alpha, beta).rsample()
        w = 2 * z - 1

        g = torch.randn((*batch_shape, dim - 1), dtype=w.dtype, device=w.device)
        v = g / g.norm(dim=-1, keepdim=True)

        radius = torch.sqrt((1 - w**2).clamp_min(0)).unsqueeze(-1)
        y = torch.cat([w.unsqueeze(-1), radius * v], dim=-1)

        return householder_reflect(y, mean_direction)

    def entropy(self) -> Tensor:
        """Differential entropy (Theorem 15, Eq. 62)."""
        alpha, beta = self._alpha_beta()
        return self._log_normalizer() - self.kappa * (
            math.log(2) + torch.digamma(alpha) - torch.digamma(alpha + beta)
        )

    def kl_to_uniform(self) -> Tensor:
        """Closed-form KL divergence to Uniform(S^(dim - 1)) (Theorem 17, Eq. 72).

        Returns:
            ``D_KL[PowerSpherical(mean_direction, kappa) || Uniform(S^(dim - 1))]``,
            broadcast against the distribution's batch shape.
        """
        return -self.entropy() + _log_surface_area(self.dim)

    def _alpha_beta(self) -> tuple[Tensor, Tensor]:
        """Returns the Beta marginal's shape parameters (Theorem 12)."""
        beta = torch.full_like(self.kappa, (self.dim - 1) / 2)
        return beta + self.kappa, beta

    def _log_normalizer(self) -> Tensor:
        """Returns ``log N_X(kappa, dim)``, the Theorem 13 normalizing constant."""
        alpha, beta = self._alpha_beta()
        return (
            (alpha + beta) * math.log(2)
            + beta * math.log(math.pi)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + beta)
        )


@register_kl(PowerSpherical, HypersphericalUniform)
def _kl_power_spherical_uniform(
    power_spherical: PowerSpherical, hyu: HypersphericalUniform
) -> Tensor:
    return -power_spherical.entropy() + hyu.entropy()
