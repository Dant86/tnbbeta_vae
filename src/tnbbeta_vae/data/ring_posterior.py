"""Single-beacon ranging: a task whose exact posterior on the sphere is a latitude ring.

A latent ``z ~ Uniform(S^(d-1))`` is observed only through its cosine with a fixed unit
axis ``a``, ``w = a . z``, passed through a smooth nonlinear map ``g`` and corrupted by
Gaussian noise::

    x = g(a . z) + N(0, sigma^2 I),    g(w) = [w, sin(3 w), cos(5 w)].

The observation fixes the latitude ``w`` but carries no information about the
longitude, so
the exact posterior (uniform prior) has density, with respect to the sphere's surface
measure,::

    p(z | x) = exp(-||x - g(a . z)||^2 / (2 sigma^2)) / Z'(x),

a function of ``w`` alone: a latitude ring about ``a``. It is *genuine* uncertainty,
not a symmetry redundancy. Its normalising constant is a 1-D integral over the
polar angle
``t = arccos(w)`` with the sphere's latitude weight ``sin^(d-2) t``, evaluated here by a
midpoint rule, so the exact log-posterior, the exact evidence ``log p(x)`` and the Monte
Carlo KL ``KL(q || p(. | x))`` for any ``q`` with a ``log_prob`` are all available.

Density conventions: every density below is with respect to the surface measure of
``S^(d-1)`` (the same convention as ``TNBBetaSpherical.log_prob`` and
``VonMisesFisher.log_prob``), so a latitude marginal in ``t`` is that density times
``A_(d-2) sin^(d-2) t`` where ``A_(d-2)`` is the area of ``S^(d-2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from tnbbeta_vae.models.losses.sampling import sample_with_log_prob

if TYPE_CHECKING:
    from torch.distributions import Distribution

__all__ = [
    "RingPosteriorTask",
    "exact_kl",
    "kl_uniform_to_posterior",
    "latitude_grid",
    "ring_task",
    "sphere_log_area",
]

NUM_FEATURES = 3
_DEFAULT_GRID = 4001


@dataclass(frozen=True)
class RingPosteriorTask:
    """The ranging task: ambient dimension, noise scale and the fixed axis.

    Attributes:
        dim: Ambient dimension ``d`` of the sphere ``S^(d-1)`` (``d >= 3``).
        sigma: Observation noise standard deviation.
        axis: The fixed unit axis ``a``, shape ``(dim,)``.
        grid_size: Number of midpoint-rule nodes over the polar angle.
    """

    dim: int
    sigma: float
    axis: Tensor
    grid_size: int = _DEFAULT_GRID

    def features(self, w: Tensor) -> Tensor:
        """Latitudes ``w = a . z`` to features ``g(w)``; shape ``(*w.shape, 3)``."""
        return torch.stack([w, torch.sin(3 * w), torch.cos(5 * w)], dim=-1)

    def decode(self, z: Tensor) -> Tensor:
        """The true decoder mean ``g(a . z)`` for ``z`` of shape ``(..., dim)``."""
        return self.features(z @ self.axis.to(z))

    def sample(
        self, num: int, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Draws ``num`` observations from the generative model.

        Args:
            num: Number of draws.
            generator: Optional RNG.

        Returns:
            ``(x, z)`` of shapes ``(num, 3)`` and ``(num, dim)``.
        """
        z = torch.randn(num, self.dim, generator=generator)
        z = z / z.norm(dim=-1, keepdim=True)
        noise = torch.randn(num, NUM_FEATURES, generator=generator)
        return self.decode(z) + self.sigma * noise, z

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """``log N(x; g(a . z), sigma^2 I)`` summed over the 3 features."""
        return (
            -0.5 * ((x - self.decode(z)) / self.sigma).pow(2)
            - math.log(self.sigma)
            - 0.5 * math.log(2 * math.pi)
        ).sum(-1)

    def log_posterior(self, x: Tensor, z: Tensor) -> Tensor:
        """Exact ``log p(z | x)`` (surface-measure density).

        Args:
            x: Observations, shape ``(batch, 3)``.
            z: Points on the sphere, shape ``(..., batch, dim)``.

        Returns:
            Tensor of shape ``z.shape[:-1]``.
        """
        return self.log_posterior_at_latitude(x, z @ self.axis.to(z.dtype))

    def log_posterior_at_latitude(self, x: Tensor, w: Tensor) -> Tensor:
        """Exact ``log p(z | x)`` as a function of the latitude ``w = a . z`` alone.

        Args:
            x: Observations, shape ``(batch, 3)`` (or broadcastable).
            w: Latitudes, shape ``(..., batch)`` (broadcastable against ``x``'s batch).

        Returns:
            Tensor of the shape ``w`` broadcast with the batch.
        """
        energy = ((x - self.features(w)) / self.sigma).pow(2).sum(-1) / 2
        return -energy - self._log_normaliser(x)

    def log_evidence(self, x: Tensor) -> Tensor:
        """Exact ``log p(x) = log E_{z ~ Uniform}[N(x; g(a . z), sigma^2 I)]``.

        Args:
            x: Observations, shape ``(batch, 3)``.

        Returns:
            Tensor of shape ``(batch,)``.
        """
        theta, delta = latitude_grid(self.grid_size, x.dtype)
        log_weight = (self.dim - 2) * torch.log(torch.sin(theta)) + math.log(delta)
        log_uniform = log_weight - torch.logsumexp(log_weight, dim=0)
        energy = self.energy_grid(x, theta)
        constant = -(NUM_FEATURES / 2) * math.log(2 * math.pi * self.sigma**2)
        return constant + torch.logsumexp(-energy + log_uniform, dim=-1)

    def latitude_density(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """The posterior's latitude marginal in the polar angle ``t`` about ``a``.

        Args:
            x: Observations, shape ``(batch, 3)``.

        Returns:
            ``(theta, density)``: the ``grid_size`` midpoint nodes ``theta`` of
            ``(0, pi)`` and the normalised density of ``t``, shape
            ``(batch, grid_size)``
            (it sums to 1 against ``pi / grid_size``).
        """
        theta, delta = latitude_grid(self.grid_size, x.dtype)
        log_weight = (self.dim - 2) * torch.log(torch.sin(theta))
        log_mass = -self.energy_grid(x, theta) + log_weight
        density = torch.exp(log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True))
        return theta, density / delta

    def latitude_cdf(self, x: Tensor, edges: Tensor) -> Tensor:
        """The posterior's latitude CDF at ``edges`` (angles in ``[0, pi]``).

        Args:
            x: Observations, shape ``(batch, 3)``.
            edges: Angles, shape ``(num_edges,)``.

        Returns:
            Tensor of shape ``(batch, num_edges)``, linearly interpolated between nodes.
        """
        theta, density = self.latitude_density(x)
        delta = math.pi / self.grid_size
        # Cumulative mass at the right edge of each cell; 0 at t = 0.
        cdf = torch.cat(
            [torch.zeros_like(density[:, :1]), torch.cumsum(density * delta, dim=-1)],
            dim=-1,
        )
        position = (edges.clamp(0, math.pi) / delta).to(x.dtype)
        index = position.floor().long().clamp(max=self.grid_size - 1)
        fraction = position - index.to(x.dtype)
        low = cdf[:, index]
        high = cdf[:, index + 1]
        return low + (high - low) * fraction

    def energy_grid(self, x: Tensor, theta: Tensor) -> Tensor:
        """``||x - g(cos t)||^2 / (2 sigma^2)`` on the grid, shape ``(batch, grid)``."""
        features = self.features(torch.cos(theta))
        difference = x[:, None, :] - features[None]
        return difference.pow(2).sum(-1) / (2 * self.sigma**2)

    def _log_normaliser(self, x: Tensor) -> Tensor:
        """``log Z'(x)`` for the surface-measure density, shape ``(batch,)``."""
        theta, delta = latitude_grid(self.grid_size, x.dtype)
        log_weight = (self.dim - 2) * torch.log(torch.sin(theta)) + math.log(delta)
        log_integral = torch.logsumexp(-self.energy_grid(x, theta) + log_weight, -1)
        return sphere_log_area(self.dim - 1) + log_integral


def ring_task(
    dim: int, sigma: float, grid_size: int = _DEFAULT_GRID
) -> RingPosteriorTask:
    """Builds the task with the project's fixed axis for ``dim`` (seed ``1000 + dim``).

    Args:
        dim: Ambient dimension, at least 3.
        sigma: Observation noise standard deviation.
        grid_size: Quadrature nodes over the polar angle.

    Returns:
        The task, with ``axis`` a seeded random unit vector.

    Raises:
        ValueError: If ``dim < 3`` (``S^(d-2)`` would be degenerate).
    """
    if dim < 3:
        raise ValueError(f"ring_task needs dim >= 3; got {dim}.")
    generator = torch.Generator().manual_seed(1000 + dim)
    axis = torch.randn(dim, generator=generator)
    return RingPosteriorTask(dim, sigma, axis / axis.norm(), grid_size)


def latitude_grid(
    num: int = _DEFAULT_GRID, dtype: torch.dtype = torch.float32
) -> tuple[Tensor, float]:
    """Midpoint nodes of ``(0, pi)`` and their spacing.

    Args:
        num: Number of nodes.
        dtype: Tensor dtype.

    Returns:
        ``(theta, delta)``: ``theta[j] = (j + 1/2) pi / num``, ``delta = pi / num``.
    """
    delta = math.pi / num
    return (torch.arange(num, dtype=dtype) + 0.5) * delta, delta


def sphere_log_area(ambient_dim: int) -> float:
    """Log surface area of ``S^(ambient_dim - 1)`` in ``R^ambient_dim``."""
    return (
        math.log(2)
        + (ambient_dim / 2) * math.log(math.pi)
        - math.lgamma(ambient_dim / 2)
    )


def kl_uniform_to_posterior(task: RingPosteriorTask, x: Tensor) -> Tensor:
    """Exact ``KL(Uniform || p(. | x))`` by quadrature (the "q = uniform" baseline).

    ``KL = E_U[E(w)] + log E_U[exp(-E(w))]`` with ``E = ||x - g(w)||^2 / (2 sigma^2)``.

    Args:
        task: The task.
        x: Observations, shape ``(batch, 3)``.

    Returns:
        Tensor of shape ``(batch,)``.
    """
    theta, delta = latitude_grid(task.grid_size, x.dtype)
    log_weight = (task.dim - 2) * torch.log(torch.sin(theta)) + math.log(delta)
    log_uniform = log_weight - torch.logsumexp(log_weight, dim=0)
    energy = task.energy_grid(x, theta)
    expected_energy = (energy * log_uniform.exp()).sum(-1)
    return expected_energy + torch.logsumexp(-energy + log_uniform, dim=-1)


@torch.no_grad()
def exact_kl(
    posterior: Distribution,
    task: RingPosteriorTask,
    x: Tensor,
    num_samples: int = 2000,
    max_draws_per_pass: int = 2000,
) -> tuple[Tensor, Tensor]:
    """Monte Carlo ``KL(q || p_true)`` per observation.

    Uses ``q.rsample`` and ``q.log_prob`` (exact ``log q`` for TNBBeta, via
    ``sample_with_log_prob``); ``p_true`` is exact.

    Args:
        posterior: A batch ``q(. | x_i)`` over ``task.dim``-vectors, batch shape
            ``(batch,)``.
        task: The task defining the exact posterior.
        x: Observations, shape ``(batch, 3)``.
        num_samples: ``z ~ q`` draws per observation.
        max_draws_per_pass: Upper bound on the draws per observation made in one pass
            (the samples are generated in several passes to bound memory).

    Returns:
        ``(kl, standard_error)``, each of shape ``(batch,)``.
    """
    batch = x.shape[0]
    pass_size = max(1, min(max_draws_per_pass, 4_000_000 // max(1, batch * task.dim)))
    total = torch.zeros(batch, dtype=x.dtype)
    total_sq = torch.zeros_like(total)
    remaining = num_samples
    while remaining > 0:
        count = min(pass_size, remaining)
        z, log_q = sample_with_log_prob(posterior, torch.Size([count]))
        assert log_q is not None
        integrand = log_q - task.log_posterior(x, z)
        total += integrand.sum(0)
        total_sq += integrand.pow(2).sum(0)
        remaining -= count
    mean = total / num_samples
    variance = (total_sq / num_samples - mean.pow(2)).clamp_min(0)
    return mean, (variance / num_samples).sqrt()
