"""A sign-symmetrised wrapper: a fair antipodal mixture of a spherical family.

``SignSymmetrised(base)`` draws ``z ~ base`` and multiplies it by an independent fair
random sign, so ``q(z) = (q0(z) + q0(-z)) / 2`` is exactly antipodal. It gives a
sign-unaware family (von Mises-Fisher, Power Spherical) the one structural capability
TNBBeta has natively (mass on both of an axis's poles), so a comparison against it
separates the antipodal SHAPE from anything else the family does. There is no
closed-form KL; Monte Carlo ELBOs use :meth:`rsample_with_log_prob`, whose log-density
is exact (for TNBBeta from the sampled latitude logit, which stays exact where
``log_prob`` of the returned vector does not).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.distributions import Distribution, constraints
from torch.nn import functional

if TYPE_CHECKING:
    from torch import Tensor
    from torch.types import _size

__all__ = ["SignSymmetrised"]


class SignSymmetrised(Distribution):
    """``q(z) = (q0(z) + q0(-z)) / 2`` for a base spherical distribution ``q0``."""

    arg_constraints = {}  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    has_rsample = True

    def __init__(self, base: Distribution) -> None:
        """Wraps ``base``.

        Args:
            base: A batch of spherical distributions (vMF, Power Spherical or TNBBeta).
        """
        self.base = base
        super().__init__(
            batch_shape=base.batch_shape,
            event_shape=base.event_shape,
            validate_args=False,
        )

    @property
    def support(self) -> constraints.Constraint:  # pyright: ignore[reportIncompatibleVariableOverride]
        """The base's support (the unit sphere)."""
        return self.base.support  # pyright: ignore[reportReturnType]

    def log_prob(self, value: Tensor) -> Tensor:
        """``log((q0(z) + q0(-z)) / 2)`` evaluated at ``value``."""
        return torch.logaddexp(
            self.base.log_prob(value), self.base.log_prob(-value)
        ) - (math.log(2))

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:  # noqa: B008
        """Draws reparameterized samples (base draw times a fair random sign)."""
        return self.rsample_with_log_prob(sample_shape)[0]

    def rsample_with_log_prob(
        self,
        sample_shape: _size = torch.Size(),  # noqa: B008
    ) -> tuple[Tensor, Tensor]:
        """Draws samples with their exact log-density (see the module docstring)."""
        exact = getattr(self.base, "_rsample_with_logit", None)
        if exact is not None:
            z0, t = exact(sample_shape)
            # Latitude logit t of z0 about mu; -z0 has logit -t (log y and log(1 - y)
            # swap).
            log_y, log_1my = -functional.softplus(-t), -functional.softplus(t)
            log_forward = self.base._log_prob_from_logs(log_y, log_1my)  # pyright: ignore[reportAttributeAccessIssue]  # noqa: SLF001
            log_backward = self.base._log_prob_from_logs(log_1my, log_y)  # pyright: ignore[reportAttributeAccessIssue]  # noqa: SLF001
        else:
            z0 = self.base.rsample(sample_shape)
            log_forward = self.base.log_prob(z0)
            log_backward = self.base.log_prob(-z0)
        sign = torch.randint(0, 2, z0.shape[:-1], device=z0.device).to(z0.dtype) * 2 - 1
        log_q = torch.logaddexp(log_forward, log_backward) - math.log(2)
        return z0 * sign.unsqueeze(-1), log_q
