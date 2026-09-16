"""Univariate triply-randomized negative binomial beta (TNBbeta) distribution.

Implements the distribution introduced in Lederman & Schein (2026),
"The Triply-Randomized Negative Binomial Beta for Robust Regression and
Conjugate Models of Bounded Support Data" (arXiv:2606.11624). The TNBbeta
generalizes the beta distribution with a median parameter ``p``, a
concentration parameter ``q``, and a boundary parameter ``epsilon``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Beta, Distribution, constraints
from torch.distributions.utils import broadcast_all
from torch.types import _size

__all__ = ["TNBBetaUnivariate"]

_BOUNDARY_EPS = 1e-6


class TNBBetaUnivariate(Distribution):
    """Triply-randomized negative binomial beta distribution, TNBbeta(p, q, epsilon).

    The density (Definition 3.1, Eq. 13-14 of the source paper) is::

        TNBbeta(y; p, q, eps) = beta(y; eps, eps)
            * [(1 - q) / (y * (1 - y)) * gamma(y, p)] ** eps
            * [1 - 4 * q * gamma(y, p)] ** -(eps + 1 / 2)

    where ``gamma(y, p) = y(1-y)p(1-p) / (p(1-y) + (1-p)y)^2`` is a
    similarity function between the log-odds of ``y`` and ``p``.

    Attributes:
        p: Median of the distribution, in (0, 1). ``median(Y) = p`` exactly
            (Theorem 3.1).
        q: Concentration parameter, in (0, 1). Larger ``q`` concentrates the
            density around ``p``.
        epsilon: Boundary/baseline-concentration parameter, > 0. Controls
            behavior at y=0 and y=1 (Proposition 3.1): densities diverge for
            epsilon < 1, are finite and positive for epsilon = 1, and vanish
            for epsilon > 1.

    Note:
        Sampling uses a fully deterministic, differentiable transform of a
        single ``Beta(eps, eps)`` draw -- not the (non-reparameterizable)
        auxiliary negative-binomial construction of Theorem 4.1. See
        :meth:`rsample` for the derivation. ``sample`` is inherited from
        :class:`~torch.distributions.Distribution`, which calls ``rsample``
        under ``torch.no_grad()``.
    """

    # torch.distributions subclasses always override these class-level
    # properties with plain dicts/constraints (see e.g. torch's own Beta);
    # pyright's stubs don't model that pattern.
    arg_constraints = {  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        "p": constraints.interval(0.0, 1.0),
        "q": constraints.interval(0.0, 1.0),
        "epsilon": constraints.positive,
    }
    support = constraints.interval(  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        0.0, 1.0
    )
    has_rsample = True

    def __init__(
        self,
        p: Tensor | float,
        q: Tensor | float,
        epsilon: Tensor | float,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the distribution.

        Args:
            p: Median parameter in (0, 1).
            q: Concentration parameter in (0, 1).
            epsilon: Boundary parameter, > 0.
            validate_args: Whether to validate the arguments and samples
                against the distribution's constraints.
        """
        self.p, self.q, self.epsilon = broadcast_all(p, q, epsilon)
        batch_shape = self.p.shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def median(self) -> Tensor:
        """Returns the median, which equals ``p`` exactly (Theorem 3.1)."""
        return self.p

    def _similarity(self, value: Tensor) -> Tensor:
        """Computes gamma(y, p), the log-odds similarity function (Eq. 14)."""
        y = value
        denom = self.p * (1 - y) + (1 - self.p) * y
        return (
            torch.log(y)
            + torch.log1p(-y)
            + torch.log(self.p)
            + torch.log1p(-self.p)
            - 2 * torch.log(denom)
        )

    def log_prob(self, value: Tensor) -> Tensor:
        """Computes the log-density at ``value``.

        The ``-log(y) - log(1-y)`` term below comes from combining the
        ``beta(y; eps, eps)`` factor with the ``[.. / (y(1-y))] ** eps``
        factor in Eq. 13 -- the ``eps`` powers of ``y(1-y)`` cancel,
        leaving a single factor independent of ``eps``.

        Args:
            value: Points in (0, 1) at which to evaluate the log-density.

        Returns:
            Log-density evaluated at ``value``, broadcast against the
            distribution's batch shape.
        """
        if self._validate_args:
            self._validate_sample(value)
        y = value.clamp(_BOUNDARY_EPS, 1.0 - _BOUNDARY_EPS)

        log_gamma = self._similarity(y)
        gamma = log_gamma.exp()
        log_beta_const = 2 * torch.lgamma(self.epsilon) - torch.lgamma(2 * self.epsilon)

        return (
            -torch.log(y)
            - torch.log1p(-y)
            - log_beta_const
            + self.epsilon * torch.log1p(-self.q)
            + self.epsilon * log_gamma
            - (self.epsilon + 0.5) * torch.log1p(-4 * self.q * gamma)
        )

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:  # noqa: B008
        """Draws reparameterized samples via a deterministic Beta(eps, eps) transform.

        Derivation: writing ``T = logit(Y) - logit(p)``, the TNBbeta density
        (Eq. 13) has the change-of-variables form (via ``S = tanh(T/2)``)

            ``f_S(s) ~ (1 - s^2)^(eps-1) * ((1-q) + q*s^2)^-(eps+1/2)``.

        A plain (shifted) ``Beta(eps, eps)`` draw has density
        ``(1 - s^2)^(eps-1)`` on ``(-1, 1)`` -- i.e. the ``q=0`` special
        case (this matches Corollary 3.1's ``TNBbeta(p, 0, eps) ~
        LNbeta(eps, eps, (1-p)/p)`` reduction). The map below carries that
        plain density onto the ``q``-tilted one exactly (verified directly
        by change-of-variables: it's a bijection on ``(-1, 1)`` with
        Jacobian ``sqrt(1-q) / (1 - q*s^2)^(3/2)``), so no accept-reject or
        implicit-CDF machinery is needed to handle ``q != 0``.

        Args:
            sample_shape: Shape of the i.i.d. sample batch to draw.

        Returns:
            Samples in (0, 1) with shape ``sample_shape + batch_shape``,
            differentiable w.r.t. ``p``, ``q``, and ``epsilon``.
        """
        shape = self._extended_shape(sample_shape)
        p = self.p.expand(shape)
        q = self.q.expand(shape)
        epsilon = self.epsilon.expand(shape)

        z = Beta(epsilon, epsilon).rsample()
        s = 2 * z - 1
        u = (1 + s * torch.sqrt((1 - q) / (1 - q * s**2))) / 2
        return p * u / ((1 - p) * (1 - u) + p * u)
