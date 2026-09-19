"""A learnable scale for the Gaussian pixel likelihood."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["LearnedLikelihoodScale"]


class LearnedLikelihoodScale(nn.Module):
    """A single learnable standard deviation shared by all pixels.

    Parameterized by ``log sigma^2`` so the scale stays positive without a
    constraint. With a shared ``sigma`` the ELBO's optimum is
    ``sigma^2 = mean squared reconstruction error``, so this replaces
    hand-picking the likelihood scale (a stand-in for the per-pixel,
    per-latent variance the original VAE decoder outputs).
    """

    def __init__(self, initial_scale: float) -> None:
        """Initializes the parameter.

        Args:
            initial_scale: Starting value of ``sigma`` (must be positive).
        """
        super().__init__()
        self.log_variance = nn.Parameter(torch.tensor(2 * math.log(initial_scale)))

    def forward(self) -> Tensor:
        """Returns the current ``sigma`` as a scalar tensor."""
        return torch.exp(0.5 * self.log_variance)
