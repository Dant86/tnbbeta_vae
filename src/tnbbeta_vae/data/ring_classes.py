"""Four-class ring task on S^2: two designs whose exact posteriors are latitude rings.

Both designs observe ``z ~ Uniform(S^2)`` through its latitude ``w = a . z`` about an
axis ``a``, passed through ``h(w) = [w, sin 1.5 w, cos 2 w]`` and corrupted by
``N(0, sigma^2 I_3)``::

    x = h(a . z) + sigma * eps.

The exact posterior given ``x`` (and the axis) has surface-measure density
proportional to ``exp(-||x - h(a . z)||^2 / (2 sigma^2))`` -- a latitude ring about
``a``
with a uniform azimuth. Its normaliser and the evidence ``log p(x)`` are 1-D midpoint
quadratures in the polar angle (``>= 4001`` nodes).

* **Design 1 (shared axis).** ``a = e_1`` (the pole ``heads._fixed_pole`` uses). The
  class is the latitude band of ``w`` (edges ``-1, -0.5, 0, 0.5, 1``, four equal-mass
  classes) and is a function of ``z``, not an input to the model.
* **Design 2 (one axis per class).** Axes are the tetrahedron vertices. The class
  ``c ~ Uniform{0..3}`` is drawn independently of ``z ~ Uniform(S^2)`` and selects the
  axis
  in the observation map, so ``p(z | x, c)`` is a ring about ``a_c``. The class is an
  input to the model.

Design and decision rules: ``writeup/results/ring_class_designs_preregistration_
2026-11-01.md``. Density conventions follow :mod:`tnbbeta_vae.data.ring_posterior`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from tnbbeta_vae.data.ring_posterior import latitude_grid, sphere_log_area
from tnbbeta_vae.models.losses.sampling import sample_with_log_prob

if TYPE_CHECKING:
    from torch.distributions import Distribution

__all__ = [
    "BAND_EDGES",
    "NUM_CLASSES",
    "OFFSET_DIRECTIONS",
    "OffsetRingClassTask",
    "RingClassTask",
    "TETRAHEDRON",
    "class_identifiability",
    "exact_class_kl",
    "offset_ring_class_task",
    "ring_class_task",
]

DIM = 3
NUM_FEATURES = 3
NUM_CLASSES = 4
SIGMA = 0.2
BAND_EDGES = (-1.0, -0.5, 0.0, 0.5, 1.0)
TETRAHEDRON = (
    (1.0, 1.0, 1.0),
    (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
)
_DEFAULT_GRID = 4001
BASE_OFFSET = 1.5
OFFSET_DIRECTIONS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (-1.0, -1.0, -1.0),
)


@dataclass(frozen=True)
class RingClassTask:
    """The four-class ring task in design 1 or design 2.

    Attributes:
        design: ``1`` (shared axis, class = latitude band) or ``2`` (axis per class).
        sigma: Observation noise standard deviation.
        grid_size: Number of midpoint-rule nodes over the polar angle.
    """

    design: int
    sigma: float = SIGMA
    grid_size: int = _DEFAULT_GRID

    def __post_init__(self) -> None:
        """Validates the design."""
        if self.design not in (1, 2):
            raise ValueError(f"design must be 1 or 2; got {self.design}.")

    @property
    def input_dim(self) -> int:
        """Encoder input size: ``x`` (design 1) or ``x`` plus the one-hot class (2)."""
        return NUM_FEATURES if self.design == 1 else NUM_FEATURES + NUM_CLASSES

    def features(self, w: Tensor) -> Tensor:
        """Latitudes ``w`` to ``h(w) = [w, sin 1.5 w, cos 2 w]``; shape ``(*w, 3)``."""
        return torch.stack([w, torch.sin(1.5 * w), torch.cos(2.0 * w)], dim=-1)

    def axes(self, dtype: torch.dtype = torch.float64) -> Tensor:
        """The class axes, shape ``(4, 3)``; design 1 has ``e_1`` for every class."""
        if self.design == 1:
            axes = torch.zeros(NUM_CLASSES, DIM, dtype=dtype)
            axes[:, 0] = 1.0
            return axes
        return torch.tensor(TETRAHEDRON, dtype=dtype) / math.sqrt(3.0)

    def axis_of(self, cls: Tensor, dtype: torch.dtype = torch.float64) -> Tensor:
        """The axis of each class index, shape ``(*cls.shape, 3)``."""
        return self.axes(dtype).to(cls.device)[cls]

    def encoder_input(self, x: Tensor, cls: Tensor) -> Tensor:
        """The model input: ``x`` (design 1) or ``[x, onehot(cls)]`` (design 2)."""
        if self.design == 1:
            return x
        return torch.cat(
            [x, torch.nn.functional.one_hot(cls, NUM_CLASSES).to(x.dtype)], dim=-1
        )

    def decode(self, z: Tensor, cls: Tensor) -> Tensor:
        """The true decoder mean ``h(a_c . z)``; ``z`` is ``(..., batch, 3)``."""
        axis = self.axis_of(cls, z.dtype)
        return self.features((z * axis).sum(-1))

    def band_of(self, w: Tensor) -> Tensor:
        """The design-1 class (latitude band index 0..3) of latitudes ``w``."""
        edges = torch.tensor(BAND_EDGES[1:-1], dtype=w.dtype, device=w.device)
        return torch.bucketize(w.contiguous(), edges)

    def sample(
        self,
        num: int,
        generator: torch.Generator | None = None,
        balanced: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Draws ``num`` observations from the generative model.

        Args:
            num: Number of draws.
            generator: Optional RNG. Design 2 draws the class first, then ``z``, then
                the noise; design 1 draws ``z`` then the noise.
            balanced: Design 2 only: classes ``0, 1, 2, 3, 0, ...`` instead of random
                (``num / 4`` per class when ``num`` is a multiple of 4).

        Returns:
            ``(x, z, cls, w)``: ``x`` ``(num, 3)``, ``z`` ``(num, 3)``, integer class
            ``(num,)`` and the true latitude ``w = a_c . z`` ``(num,)``.
        """
        cls = None
        if self.design == 2:
            if balanced:
                cls = torch.arange(num) % NUM_CLASSES
            else:
                cls = torch.randint(0, NUM_CLASSES, (num,), generator=generator)
        z = torch.randn(num, DIM, generator=generator)
        z = z / z.norm(dim=-1, keepdim=True)
        noise = torch.randn(num, NUM_FEATURES, generator=generator)
        if cls is None:
            w = z[:, 0]
            cls = self.band_of(w)
        else:
            w = (z * self.axis_of(cls, z.dtype)).sum(-1)
        return self.features(w) + self.sigma * noise, z, cls, w

    def log_likelihood(self, x: Tensor, z: Tensor, cls: Tensor) -> Tensor:
        """``log N(x; h(a_c . z), sigma^2 I)`` summed over the 3 features."""
        return (
            -0.5 * ((x - self.decode(z, cls)) / self.sigma).pow(2)
            - math.log(self.sigma)
            - 0.5 * math.log(2 * math.pi)
        ).sum(-1)

    def log_posterior(self, x: Tensor, z: Tensor, cls: Tensor) -> Tensor:
        """Exact ``log p(z | x, c)`` (surface-measure density).

        Args:
            x: Observations, shape ``(batch, 3)``.
            z: Points on the sphere, shape ``(..., batch, 3)``.
            cls: Class indices, shape ``(batch,)``.

        Returns:
            Tensor of shape ``z.shape[:-1]``.
        """
        w = (z * self.axis_of(cls, z.dtype)).sum(-1)
        return self.log_posterior_at_latitude(x, w)

    def log_posterior_at_latitude(self, x: Tensor, w: Tensor) -> Tensor:
        """Exact ``log p(z | x, c)`` as a function of the latitude ``w = a_c . z``."""
        energy = ((x - self.features(w)) / self.sigma).pow(2).sum(-1) / 2
        return -energy - self._log_normaliser(x)

    def log_evidence(self, x: Tensor) -> Tensor:
        """Exact ``log p(x | c) = log E_{z ~ Uniform}[N(x; h(a_c . z), sigma^2 I)]``.

        The latitude of a uniform ``z`` about any axis is uniform on ``[-1, 1]``, so the
        evidence does not depend on the class.
        """
        theta, delta = latitude_grid(self.grid_size, x.dtype)
        log_weight = (DIM - 2) * torch.log(torch.sin(theta)) + math.log(delta)
        log_uniform = log_weight - torch.logsumexp(log_weight, dim=0)
        constant = -(NUM_FEATURES / 2) * math.log(2 * math.pi * self.sigma**2)
        return constant + torch.logsumexp(-self.energy_grid(x, theta) + log_uniform, -1)

    def latitude_density(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """The posterior's latitude marginal in the polar angle ``t`` about the axis.

        Returns:
            ``(theta, density)``: the midpoint nodes and the normalised density of
            ``t``, shape ``(batch, grid_size)`` (sums to 1 against ``pi / grid_size``).
        """
        theta, delta = latitude_grid(self.grid_size, x.dtype)
        log_mass = -self.energy_grid(x, theta) + (DIM - 2) * torch.log(torch.sin(theta))
        density = torch.exp(log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True))
        return theta, density / delta

    def latitude_cdf(self, x: Tensor, edges: Tensor) -> Tensor:
        """The posterior's latitude CDF at angles ``edges`` in ``[0, pi]``.

        Returns:
            Tensor of shape ``(batch, num_edges)``, linearly interpolated between nodes.
        """
        _, density = self.latitude_density(x)
        delta = math.pi / self.grid_size
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
        """``||x - h(cos t)||^2 / (2 sigma^2)`` on the grid, shape ``(batch, grid)``."""
        difference = x[:, None, :] - self.features(torch.cos(theta))[None]
        return difference.pow(2).sum(-1) / (2 * self.sigma**2)

    def kl_uniform_to_posterior(self, x: Tensor) -> Tensor:
        """Exact ``KL(Uniform || p(. | x, c))`` by quadrature, shape ``(batch,)``.

        ``KL = E_U[E(w)] + log E_U[exp(-E(w))]`` with
        ``E = ||x - h(w)||^2 / (2 sigma^2)``.
        """
        theta, delta = latitude_grid(self.grid_size, x.dtype)
        log_weight = (DIM - 2) * torch.log(torch.sin(theta)) + math.log(delta)
        log_uniform = log_weight - torch.logsumexp(log_weight, dim=0)
        energy = self.energy_grid(x, theta)
        expected = (energy * log_uniform.exp()).sum(-1)
        return expected + torch.logsumexp(-energy + log_uniform, dim=-1)

    def second_mode_mass(self, x: Tensor, num_nodes: int = 2001) -> Tensor:
        """Mass of the second-largest latitude mode of ``p(w | x)``, shape ``(batch,)``.

        The latitude ``w`` has density proportional to ``exp(-E(w))`` on ``[-1, 1]``;
        modes are the segments between its local minima (the pre-registered check that
        no ``x`` has a second mode with more than 5% of the mass).
        """
        grid = torch.linspace(-1.0, 1.0, num_nodes, dtype=x.dtype)
        energy = ((x[:, None, :] - self.features(grid)[None]) ** 2).sum(-1)
        density = torch.exp(
            -(energy - energy.min(-1, keepdim=True).values) / 2 / self.sigma**2
        )
        interior = (density[:, 1:-1] < density[:, :-2]) & (
            density[:, 1:-1] <= density[:, 2:]
        )
        is_min = torch.zeros_like(density, dtype=torch.long)
        is_min[:, 1:-1] = interior.long()
        segment = torch.cumsum(is_min, dim=-1)
        masses = torch.zeros(x.shape[0], num_nodes + 1, dtype=x.dtype)
        masses.scatter_add_(1, segment, density)
        ordered = masses.sort(dim=-1, descending=True).values
        return ordered[:, 1] / masses.sum(-1)

    def sample_posterior(
        self,
        x: Tensor,
        cls: Tensor,
        num: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draws ``z ~ p(z | x, c)`` per row: inverse-CDF latitude, uniform azimuth.

        Args:
            x: Observations, shape ``(batch, 3)``.
            cls: Class indices, shape ``(batch,)``.
            num: Draws per row.
            generator: Optional RNG.

        Returns:
            Tensor of shape ``(num, batch, 3)``.
        """
        x = x.double()
        theta, density = self.latitude_density(x)
        delta = math.pi / self.grid_size
        cdf = torch.cumsum(density * delta, dim=-1)
        cdf = cdf / cdf[:, -1:]
        u = torch.rand(x.shape[0], num, generator=generator, dtype=x.dtype)
        index = torch.searchsorted(cdf, u).clamp(max=self.grid_size - 1)
        low = torch.where(index > 0, cdf.gather(1, (index - 1).clamp_min(0)), 0.0)
        high = cdf.gather(1, index)
        fraction = ((u - low) / (high - low).clamp_min(1e-300)).clamp(0, 1)
        angle = (index.to(x.dtype) + fraction) * delta  # (batch, num)
        phi = (
            torch.rand(x.shape[0], num, generator=generator, dtype=x.dtype)
            * 2
            * math.pi
        )
        axis = self.axis_of(cls, x.dtype)
        e1, e2 = _orthonormal_frame(axis)
        sin_t = torch.sin(angle)
        z = (
            (sin_t * torch.cos(phi))[..., None] * e1[:, None]
            + (sin_t * torch.sin(phi))[..., None] * e2[:, None]
            + torch.cos(angle)[..., None] * axis[:, None]
        )
        return z.transpose(0, 1)

    def _log_normaliser(self, x: Tensor) -> Tensor:
        """``log Z'(x)`` for the surface-measure density, shape ``(batch,)``."""
        theta, delta = latitude_grid(self.grid_size, x.dtype)
        log_weight = (DIM - 2) * torch.log(torch.sin(theta)) + math.log(delta)
        log_integral = torch.logsumexp(-self.energy_grid(x, theta) + log_weight, -1)
        return sphere_log_area(DIM - 1) + log_integral


def ring_class_task(design: int, grid_size: int = _DEFAULT_GRID) -> RingClassTask:
    """Builds the task for ``design`` (1 or 2) with sigma 0.2."""
    return RingClassTask(design, SIGMA, grid_size)


@torch.no_grad()
def exact_class_kl(
    posterior: Distribution,
    task: RingClassTask,
    x: Tensor,
    cls: Tensor,
    num_samples: int = 1000,
    max_draws_per_pass: int = 2000,
) -> tuple[Tensor, Tensor]:
    """Monte Carlo ``KL(q || p_true(. | x, c))`` per observation.

    Uses ``q.rsample`` and the exact ``log q`` (:func:`sample_with_log_prob`);
    ``p_true`` is exact.

    Args:
        posterior: A batch ``q(. | x_i)`` over 3-vectors, batch shape ``(batch,)``.
        task: The task defining the exact posterior.
        x: Observations, shape ``(batch, 3)``.
        cls: Class indices, shape ``(batch,)``.
        num_samples: ``z ~ q`` draws per observation.
        max_draws_per_pass: Draws per observation made in one pass (bounds memory).

    Returns:
        ``(kl, standard_error)``, each of shape ``(batch,)``.
    """
    batch = x.shape[0]
    pass_size = max(1, min(max_draws_per_pass, 4_000_000 // max(1, batch * DIM)))
    total = torch.zeros(batch, dtype=x.dtype)
    total_sq = torch.zeros_like(total)
    remaining = num_samples
    while remaining > 0:
        count = min(pass_size, remaining)
        z, log_q = sample_with_log_prob(posterior, torch.Size([count]))
        assert log_q is not None
        integrand = log_q - task.log_posterior(x, z, cls)
        total += integrand.sum(0)
        total_sq += integrand.pow(2).sum(0)
        remaining -= count
    mean = total / num_samples
    variance = (total_sq / num_samples - mean.pow(2)).clamp_min(0)
    return mean, (variance / num_samples).sqrt()


@dataclass(frozen=True)
class OffsetRingClassTask:
    """Design 2': design 2 plus class offsets, ``x = h(a_c . z) + o_c + sigma * eps``.

    The offsets ``o_c = scale * 1.5 * [e_1, e_2, e_3, -(1, 1, 1)]`` make the class
    identifiable from ``x``. ``z ~ Uniform(S^2)`` is independent of the class ``c ~
    Uniform{0..3}``, so ``p(x | c)`` is the design-2 evidence of ``x - o_c`` (a 1-D
    quadrature, ``>= 4001`` nodes), ``p(x) = (1/4) sum_c p(x | c)``, and the class
    posterior is the softmax of ``log p(x | c)``. Design 1 and the original design 2
    (:class:`RingClassTask`) are untouched.

    Attributes:
        offset_scale: Multiplier on the nominal offsets (``1`` is ``1.5 * [...]``).
        sigma: Observation noise standard deviation.
        grid_size: Number of midpoint-rule nodes over the polar angle.
    """

    offset_scale: float = 1.0
    sigma: float = SIGMA
    grid_size: int = _DEFAULT_GRID

    @property
    def base(self) -> RingClassTask:
        """The offset-free design-2 task this one shifts."""
        return RingClassTask(2, self.sigma, self.grid_size)

    @property
    def input_dim(self) -> int:
        """Class-given encoder input size: ``x`` plus the one-hot class."""
        return NUM_FEATURES + NUM_CLASSES

    def offsets(self, dtype: torch.dtype = torch.float64) -> Tensor:
        """The class offsets ``o_c``, shape ``(4, 3)``."""
        directions = torch.tensor(OFFSET_DIRECTIONS, dtype=dtype)
        return BASE_OFFSET * self.offset_scale * directions

    def decode(self, z: Tensor, cls: Tensor) -> Tensor:
        """The true decoder mean ``h(a_c . z) + o_c``; ``z`` is ``(..., batch, 3)``."""
        return self.base.decode(z, cls) + self.offsets(z.dtype).to(z.device)[cls]

    def sample(
        self,
        num: int,
        generator: torch.Generator | None = None,
        balanced: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Draws ``(x, z, cls, w)`` like :meth:`RingClassTask.sample`, plus offsets."""
        x, z, cls, w = self.base.sample(num, generator, balanced)
        return x + self.offsets(x.dtype)[cls], z, cls, w

    def log_evidence_given_class(self, x: Tensor, cls: Tensor) -> Tensor:
        """Exact ``log p(x | c)`` for each row's class, shape ``(batch,)``."""
        return self.log_evidence_by_class(x).gather(1, cls[:, None])[:, 0]

    def log_evidence_by_class(self, x: Tensor) -> Tensor:
        """Exact ``log p(x | c)`` for every class, shape ``(batch, 4)``."""
        offsets = self.offsets(x.dtype)
        base = self.base
        return torch.stack(
            [base.log_evidence(x - offsets[c]) for c in range(NUM_CLASSES)], dim=-1
        )

    def log_evidence(self, x: Tensor) -> Tensor:
        """Exact mixture ``log p(x) = log((1/4) sum_c p(x | c))``, ``(batch,)``."""
        by_class = self.log_evidence_by_class(x)
        return torch.logsumexp(by_class, dim=-1) - math.log(NUM_CLASSES)

    def class_posterior(self, x: Tensor) -> Tensor:
        """The exact class posterior ``p(c | x)``, shape ``(batch, 4)``."""
        return torch.softmax(self.log_evidence_by_class(x), dim=-1)


def offset_ring_class_task(
    offset_scale: float = 1.0, grid_size: int = _DEFAULT_GRID
) -> OffsetRingClassTask:
    """Builds design 2' with sigma 0.2 and the offsets scaled by ``offset_scale``."""
    return OffsetRingClassTask(offset_scale, SIGMA, grid_size)


@torch.no_grad()
def class_identifiability(
    task: OffsetRingClassTask,
    num: int = 20_000,
    seed: int = 7,
    threshold: float = 0.99,
    chunk: int = 1000,
) -> tuple[float, float]:
    """The pre-registered check that ``p(c | x) > threshold`` for most test ``x``.

    Args:
        task: Design 2'.
        num: Number of test ``x`` (a fresh generator, class random).
        seed: Generator seed of the check's draw.
        threshold: Class-posterior level.
        chunk: Observations per quadrature pass.

    Returns:
        ``(fraction, mean_max_posterior)``: the fraction of ``x`` whose largest class
        posterior exceeds ``threshold``, and the mean largest class posterior.
    """
    generator = torch.Generator().manual_seed(seed)
    x, _, _, _ = task.sample(num, generator)
    x = x.double()
    top = torch.cat(
        [
            task.class_posterior(x[i : i + chunk]).max(-1).values
            for i in range(0, num, chunk)
        ]
    )
    return float((top > threshold).double().mean()), float(top.mean())


def _orthonormal_frame(axis: Tensor) -> tuple[Tensor, Tensor]:
    """Two unit vectors orthogonal to each row of ``axis`` and to each other."""
    pole = torch.zeros_like(axis)
    alt = torch.zeros_like(axis)
    pole[..., 0] = 1.0
    alt[..., 1] = 1.0
    helper = torch.where((axis[..., :1].abs() < 0.9), pole, alt)
    e1 = torch.linalg.cross(axis, helper)
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    return e1, torch.linalg.cross(axis, e1)
