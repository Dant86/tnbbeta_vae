r"""Spherical extension of the TNBbeta distribution (research stub).

The source paper (Lederman & Schein, 2026, arXiv:2606.11624) only defines
TNBbeta on the unit interval (0, 1); it does not define a hyperspherical
analogue. Extending it to :math:`S^{d-1}` is the open research question this
package exists to answer, so this module is intentionally a design stub
rather than a finished implementation.

Candidate constructions to evaluate (see project discussion before
committing to one):

* **Radial-angular decomposition**: model the polar angle between a sample
  and a mean direction ``mu`` with a (rescaled) ``TNBBetaUnivariate`` via
  ``y = (1 + cos(theta)) / 2``, and the remaining ``d - 2`` angular
  coordinates uniformly -- analogous to how the von Mises-Fisher
  distribution is built from a 1-D exponential-family kernel.
* **Stereographic pushforward**: map a TNBbeta-distributed radius under
  stereographic projection from :math:`\\mathbb{R}^{d-1}` onto the sphere.
* **Marginal-conditional coupling**: use the univariate TNBbeta as the
  conditional density of the projection onto a great circle, matching the
  reverse-conjugacy structure the paper establishes for the univariate case.

Whichever construction is chosen needs, at minimum: a closed-form (or
tractable-normalizing-constant) log-density, and a sampling procedure --
ideally reparameterizable so it can serve as a VAE latent posterior, not
just a prior.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.types import _size

__all__ = ["TNBBetaSpherical"]


class TNBBetaSpherical(Distribution):
    """Placeholder for a TNBbeta-derived distribution on the unit hypersphere.

    Not yet implemented -- see the module docstring for candidate
    constructions. This class exists to fix the intended public API
    (``dim``, ``log_prob``, ``sample``) so downstream code (priors,
    architectures) can be written against it before the distribution itself
    is finalized.

    Attributes:
        mean_direction: Mean direction on :math:`S^{dim - 1}`, unit-norm.
        dim: Ambient dimension of the sphere (``mean_direction`` has
            ``dim`` components).
    """

    # See the note in tnbbeta_univariate.py on why this override needs
    # pyright suppression.
    arg_constraints = {  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        "mean_direction": constraints.real_vector
    }
    has_rsample = False

    def __init__(
        self,
        mean_direction: Tensor,
        validate_args: bool | None = None,
    ) -> None:
        """Initializes the distribution.

        Args:
            mean_direction: Unit-norm mean direction, shape ``(..., dim)``.
            validate_args: Whether to validate arguments and samples.

        Raises:
            NotImplementedError: Always -- see module docstring.
        """
        del validate_args
        self.mean_direction = mean_direction
        self.dim = mean_direction.shape[-1]
        raise NotImplementedError(
            "TNBBetaSpherical is a design stub; the hyperspherical "
            "extension of TNBbeta has not been finalized yet. See the "
            "module docstring for candidate constructions."
        )

    def log_prob(self, value: Tensor) -> Tensor:
        """Computes the log-density on the sphere.

        Args:
            value: Points on :math:`S^{dim - 1}`.

        Raises:
            NotImplementedError: Always -- not yet designed.
        """
        raise NotImplementedError

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:  # noqa: B008
        """Draws samples from the sphere.

        Args:
            sample_shape: Shape of the i.i.d. sample batch to draw.

        Raises:
            NotImplementedError: Always -- not yet designed.
        """
        del sample_shape
        raise NotImplementedError
