"""Uniform distribution on the unit hypersphere.

Ported from ``hyperspherical_vae/distributions/hyperspherical_uniform.py`` in
nicola-decao/s-vae-pytorch (Davidson et al., 2018; MIT License, Copyright
(c) 2018 Nicola De Cao).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints

__all__ = ["HypersphericalUniform"]


class HypersphericalUniform(Distribution):
    """Uniform distribution on S^dim, i.e. on unit vectors in R^(dim + 1).

    Note the S-VAE convention: ``dim`` is the sphere's intrinsic dimension,
    so a ``latent_dim``-dimensional latent uses ``dim = latent_dim - 1``.
    """

    support = constraints.real  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    arg_constraints = {}  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    has_rsample = False

    def __init__(
        self,
        dim: int,
        validate_args: bool | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initializes the distribution.

        Args:
            dim: Intrinsic dimension of the sphere (ambient dimension - 1).
            validate_args: Whether to validate arguments.
            device: Device for created tensors.
        """
        super().__init__(torch.Size([dim]), validate_args=validate_args)
        self._dim = dim
        self._device = torch.device(device)

    @property
    def dim(self) -> int:
        """Intrinsic dimension of the sphere."""
        return self._dim

    def sample(self, sample_shape: torch.Size | int = torch.Size()) -> Tensor:  # noqa: B008  # pyright: ignore[reportIncompatibleMethodOverride]
        """Draws uniform samples by normalizing standard Gaussians.

        Args:
            sample_shape: Shape of the sample batch.

        Returns:
            Unit vectors, shape ``sample_shape + (dim + 1,)``.
        """
        shape = (
            sample_shape
            if isinstance(sample_shape, torch.Size)
            else torch.Size([sample_shape])
        )
        output = torch.distributions.Normal(0, 1).sample(
            shape + torch.Size([self._dim + 1])
        )
        output = output.to(self._device)
        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self) -> Tensor:
        """Log surface area of the sphere."""
        return self._log_surface_area()

    def log_prob(self, value: Tensor) -> Tensor:
        """Constant log-density ``-log(surface area)``.

        Args:
            value: Points on the sphere, shape ``(..., dim + 1)``.

        Returns:
            Tensor of shape ``value.shape[:-1]``.
        """
        return -torch.ones(value.shape[:-1], device=self._device) * (
            self._log_surface_area()
        )

    def _log_surface_area(self) -> Tensor:
        lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]).to(self._device))
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - lgamma
