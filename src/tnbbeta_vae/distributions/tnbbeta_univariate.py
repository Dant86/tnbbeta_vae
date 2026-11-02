"""Univariate triply-randomized negative binomial beta (TNBbeta) distribution.

Implements the distribution introduced in Lederman & Schein (2026),
"The Triply-Randomized Negative Binomial Beta for Robust Regression and
Conjugate Models of Bounded Support Data" (arXiv:2606.11624). The TNBbeta
generalizes the beta distribution with a median parameter ``p``, a
concentration parameter ``q``, and a boundary parameter ``epsilon``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions.utils import broadcast_all
from torch.nn import functional
from torch.types import _size

__all__ = ["TNBBetaUnivariate"]


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

    def log_prob(self, value: Tensor) -> Tensor:
        """Computes the log-density at ``value``.

        The ``-log(y) - log(1-y)`` term below comes from combining the
        ``beta(y; eps, eps)`` factor with the ``[.. / (y(1-y))] ** eps``
        factor in Eq. 13 -- the ``eps`` powers of ``y(1-y)`` cancel,
        leaving a single factor independent of ``eps``.

        The density is evaluated exactly (in log space, with no clamp to
        ``[1e-6, 1 - 1e-6]``): a clamp would make it sub-normalised for small
        ``epsilon``, where a non-negligible share of the mass sits closer to
        the boundary than that. ``value`` is only moved off exactly 0 and 1
        by the floating-point resolution of its dtype (``y`` cannot represent
        ``1 - y`` below ``finfo.eps / 2``); callers that hold logits or
        chord distances (see :class:`TNBBetaSpherical`) avoid even that.

        Args:
            value: Points in (0, 1) at which to evaluate the log-density.

        Returns:
            Log-density evaluated at ``value``, broadcast against the
            distribution's batch shape.
        """
        if self._validate_args:
            self._validate_sample(value)
        info = torch.finfo(value.dtype)
        y = value.clamp(info.tiny, 1.0 - info.eps / 2)
        return _log_density(torch.log(y), torch.log1p(-y), self.p, self.q, self.epsilon)

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
        implicit-CDF machinery is needed to handle ``q != 0`` .

        The ``Beta(eps, eps)`` draw is never formed as a probability: for
        small ``eps`` both underlying Gamma draws underflow (below
        ``eps ~ 3e-3`` the ratio is exactly 1/2 and the sample would land at
        ``y = p`` instead of the boundary spikes). It is built in logit space
        as ``logit z = log G1 - log G2`` with
        ``log Gamma(eps) = log Gamma(1 + eps) + log(U) / eps``, and the whole
        transform is carried out on logits.

        Args:
            sample_shape: Shape of the i.i.d. sample batch to draw.

        Returns:
            Samples in (0, 1) with shape ``sample_shape + batch_shape``,
            differentiable w.r.t. ``p``, ``q``, and ``epsilon``. In float32
            a sample within ``6e-8`` of 0 or 1 rounds to it; use
            :class:`TNBBetaSpherical`, which works on logits, when that
            matters.
        """
        shape = self._extended_shape(sample_shape)
        t = _sample_logit(
            self.p.expand(shape), self.q.expand(shape), self.epsilon.expand(shape)
        )
        return torch.sigmoid(t)

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


def _log_density(
    log_y: Tensor, log_1my: Tensor, p: Tensor, q: Tensor, eps: Tensor
) -> Tensor:
    """Log-density of ``TNBBeta(p, q, eps)`` given ``log y`` and ``log(1 - y)``.

    Taking the logs of ``y`` and ``1 - y`` (rather than ``y``) lets callers
    resolve points arbitrarily close to 0 or 1, and
    ``1 - 4 q gamma = (1 - q) + q ((p(1-y) - (1-p)y) / denom)^2`` is formed
    without the cancellation of ``1 - 4 q exp(log gamma)`` near ``q = 1``.
    """
    tiny = torch.finfo(log_y.dtype).tiny
    log_p, log_1mp = torch.log(p), torch.log1p(-p)
    log_denom = torch.logaddexp(log_p + log_1my, log_1mp + log_y)
    log_gamma = log_y + log_1my + log_p + log_1mp - 2 * log_denom
    signed = torch.exp(log_p + log_1my - log_denom) - torch.exp(
        log_1mp + log_y - log_denom
    )
    one_minus = ((1 - q) + q * signed**2).clamp_min(tiny)
    log_beta_const = 2 * torch.lgamma(eps) - torch.lgamma(2 * eps)
    return (
        -log_y
        - log_1my
        - log_beta_const
        + eps * torch.log1p(-q)
        + eps * log_gamma
        - (eps + 0.5) * torch.log(one_minus)
    )


def _log_gamma_draw(eps: Tensor) -> Tensor:
    """Draws ``log Gamma(eps, 1)``, reparameterized, without underflow.

    Uses ``Gamma(eps) = Gamma(1 + eps) * U ** (1 / eps)``.
    """
    gamma = Gamma(1 + eps, torch.ones_like(eps)).rsample()
    uniform = torch.rand_like(eps).clamp_min(torch.finfo(eps.dtype).tiny)
    return torch.log(gamma) + torch.log(uniform) / eps


def _sample_logit(p: Tensor, q: Tensor, eps: Tensor) -> Tensor:
    """Draws ``t = logit(Y)``, ``Y ~ TNBBeta(p, q, eps)``, differentiably.

    ``p``, ``q`` and ``eps`` must share a shape. See
    :meth:`TNBBetaUnivariate.rsample` for the construction.
    """
    big_t = _log_gamma_draw(eps) - _log_gamma_draw(eps)  # logit of Beta(eps, eps)
    abs_t = big_t.abs()
    s = torch.tanh(big_t / 2)
    # log(1 - s^2) = -2 log cosh(T / 2), evaluated without overflow.
    log_1ms2 = 2 * math.log(2) - abs_t - 2 * functional.softplus(-abs_t)
    one_m_qs2 = (1 - q) + q * torch.exp(log_1ms2)  # 1 - q s^2
    a = s * torch.sqrt((1 - q) / one_m_qs2)
    log_1ma2 = log_1ms2 - torch.log(one_m_qs2)  # log(1 - a^2)
    logit_u = torch.sign(big_t) * (2 * torch.log1p(a.abs()) - log_1ma2)
    return torch.log(p) - torch.log1p(-p) + logit_u
