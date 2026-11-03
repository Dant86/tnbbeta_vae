"""Antipodal spiked-matrix task whose exact posterior is a Bingham distribution.

A latent ``z ~ Uniform(S^(d-1))`` is observed through the rank-one matrix ``z z^T`` plus
iid Gaussian noise::

    X = z z^T + sigma E,    E_ij ~ N(0, 1) iid,    X in R^(d x d)  (not symmetrised).

Because ``||z z^T||_F^2 = 1`` for every unit ``z``,

    log p(X | z) = z^T X z / sigma^2 + c(X) = z^T X_sym z / sigma^2 + c(X),
    c(X) = -(||X||_F^2 + 1) / (2 sigma^2) - (d^2 / 2) log(2 pi sigma^2),

and only the symmetric part ``X_sym = (X + X^T) / 2`` enters. With the uniform prior the
exact posterior is the **Bingham** density ``p(z | X) = exp(z^T A z) / Z(A)`` with ``A =
X_sym / sigma^2``, exactly antipodal: ``p(z | X) = p(-z | X)``. ``Z`` is the expectation
of ``exp(z^T A z)`` over the uniform sphere (the normalised Bingham constant), which for
``d = 2, 3, 4`` reduces, in the eigenbasis of ``A`` (``lam_1 >= ... >= lam_d``), to a
one-dimensional integral with modified Bessel functions (below), evaluated by
Gauss-Legendre quadrature, so the exact log evidence ``log p(X)``, the exact posterior
and the exact KL of any ``q`` to it are available.

Every density here is with respect to the surface measure of ``S^(d-1)`` (the same
convention as ``TNBBetaSpherical.log_prob``).
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import math

import numpy as np
from scipy import special
import torch
from torch import Tensor

__all__ = [
    "AxialBinghamTask",
    "bingham_log_normaliser",
    "num_sym",
    "sphere_quadrature",
    "vec_sym",
]

_GL_NODES = 2000


@dataclass(frozen=True)
class AxialBinghamTask:
    """The task: ambient dimension and noise scale.

    The sphere is ``S^(dim-1)``.

    Attributes:
        dim: Ambient dimension ``d`` (2, 3 or 4 for the exact quantities).
        sigma: Observation noise standard deviation.
    """

    dim: int
    sigma: float

    @property
    def num_features(self) -> int:
        """Length of a flattened observation, ``d * d``."""
        return self.dim * self.dim

    def sample(
        self, num: int, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Draws ``num`` observations.

        Args:
            num: Number of draws.
            generator: Optional RNG.

        Returns:
            ``(x, z)``: ``x`` flattened to ``(num, d * d)`` (row-major ``d x d``), and
            the true latents ``z`` of shape ``(num, d)``.
        """
        z = torch.randn(num, self.dim, generator=generator)
        z = z / z.norm(dim=-1, keepdim=True)
        noise = torch.randn(num, self.dim, self.dim, generator=generator)
        x = z[:, :, None] * z[:, None, :] + self.sigma * noise
        return x.reshape(num, -1), z

    def true_mean(self, z: Tensor) -> Tensor:
        """The true decoder mean ``z z^T`` flattened, shape ``(..., d * d)``."""
        return (z[..., :, None] * z[..., None, :]).reshape(*z.shape[:-1], -1)

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """``log N(x; z z^T, sigma^2 I)`` over the ``d * d`` entries.

        Args:
            x: Observations ``(batch, d * d)``.
            z: Latents ``(..., batch, d)``.

        Returns:
            Tensor of shape ``z.shape[:-1]``.
        """
        residual = (x - self.true_mean(z)) / self.sigma
        constant = -self.num_features * (
            math.log(self.sigma) + 0.5 * math.log(2 * math.pi)
        )
        return constant - 0.5 * residual.pow(2).sum(-1)

    def symmetric_part(self, x: Tensor) -> Tensor:
        """``X_sym`` as ``(batch, d, d)`` from flattened observations."""
        matrix = x.reshape(-1, self.dim, self.dim)
        return 0.5 * (matrix + matrix.transpose(-1, -2))

    def bingham_matrix(self, x: Tensor) -> Tensor:
        """The exact posterior's matrix ``A = X_sym / sigma^2``, ``(batch, d, d)``."""
        return self.symmetric_part(x) / self.sigma**2

    def log_posterior(self, x: Tensor, z: Tensor) -> Tensor:
        """Exact ``log p(z | X)`` (surface-measure density).

        Args:
            x: Observations ``(batch, d * d)``.
            z: Points ``(..., batch, d)``.

        Returns:
            Tensor of shape ``z.shape[:-1]``.
        """
        matrix = self.bingham_matrix(x)
        quad = torch.einsum("...bi,bij,...bj->...b", z, matrix, z)
        log_area = _log_sphere_area(self.dim)
        return quad - bingham_log_normaliser(torch.linalg.eigvalsh(matrix)) - log_area

    def log_evidence(self, x: Tensor) -> Tensor:
        """Exact ``log p(X) = log E_z N(X; z z^T, sigma^2 I)``, shape ``(batch,)``."""
        matrix = self.bingham_matrix(x)
        eigenvalues = torch.linalg.eigvalsh(matrix)
        return self._log_constant(x) + bingham_log_normaliser(eigenvalues)

    def top_axis(self, x: Tensor) -> Tensor:
        """Unit eigenvector of ``X_sym`` for its largest eigenvalue, ``(batch, d)``."""
        _, vectors = torch.linalg.eigh(self.symmetric_part(x))
        return vectors[..., -1]

    def _log_constant(self, x: Tensor) -> Tensor:
        """``c(X)`` of the module docstring."""
        sq = x.pow(2).sum(-1)
        return -(sq + 1) / (2 * self.sigma**2) - (self.num_features / 2) * math.log(
            2 * math.pi * self.sigma**2
        )


def num_sym(dim: int) -> int:
    """Number of unique entries of a symmetric ``dim x dim`` matrix."""
    return dim * (dim + 1) // 2


def vec_sym(z: Tensor) -> Tensor:
    """The unique entries of ``z z^T``, off-diagonals times ``sqrt(2)``.

    Upper triangle, row-major.

    The scaling makes the vector's Euclidean norm the Frobenius norm of ``z z^T`` (1).

    Args:
        z: Points, shape ``(..., d)``.

    Returns:
        Tensor of shape ``(..., d (d + 1) / 2)``.
    """
    dim = z.shape[-1]
    rows, cols = torch.triu_indices(dim, dim).unbind(0)
    scale = torch.where(rows == cols, 1.0, math.sqrt(2.0)).to(z)
    return z[..., rows] * z[..., cols] * scale


def bingham_log_normaliser(eigenvalues: Tensor, num_nodes: int = _GL_NODES) -> Tensor:
    """``log E_{z ~ Uniform(S^(d-1))} exp(sum_i lam_i z_i^2)`` for ``d = 2, 3, 4``.

    In the eigenbasis (``lam`` sorted ascending as ``eigvalsh`` returns it, internally
    reordered descending) the expectation is, with ``I_0`` the modified Bessel function:

    * ``d = 2``: ``e^((l1 + l2) / 2) I_0((l1 - l2) / 2)`` (closed form);
    * ``d = 3``: ``int_0^1 du exp(l1 u^2 + (l2 + l3)(1 - u^2) / 2) I_0((l2 - l3)(1 -
      u^2) / 2)`` (``u`` the coordinate along the top axis, uniform on ``[-1, 1]`` on
      ``S^2``);
    * ``d = 4``: ``int_0^1 ds exp(s (l1 + l2) / 2 + (1 - s)(l3 + l4) / 2)
      I_0(s (l1 - l2) / 2) I_0((1 - s)(l3 - l4) / 2)`` (``s = cos^2 eta`` in Hopf-type
      coordinates is uniform on ``[0, 1]`` on ``S^3``).

    The integrals use ``num_nodes`` Gauss-Legendre nodes on ``[0, 1]``.

    Args:
        eigenvalues: Eigenvalues, shape ``(batch, d)`` (float64 recommended).
        num_nodes: Gauss-Legendre nodes for ``d = 3, 4``.

    Returns:
        Tensor of shape ``(batch,)``.

    Raises:
        ValueError: If ``d`` is not 2, 3 or 4.
    """
    dim = eigenvalues.shape[-1]
    lam = torch.flip(eigenvalues, dims=(-1,)).detach().cpu().double().numpy()
    if dim == 2:
        out = lam[:, 0] + np.log(special.ive(0, (lam[:, 0] - lam[:, 1]) / 2))
    elif dim in (3, 4):
        nodes, weights = _gauss_legendre(num_nodes)
        nodes, weights = (nodes + 1) / 2, weights / 2
        log_w = np.log(weights)
        if dim == 3:
            u2 = nodes**2
            r = (lam[:, 1:2] - lam[:, 2:3]) * (1 - u2) / 2
            expo = lam[:, 0:1] * u2 + (lam[:, 1:2] + lam[:, 2:3]) * (1 - u2) / 2
            log_f = expo + r + np.log(special.ive(0, r))
        else:
            r1 = nodes * (lam[:, 0:1] - lam[:, 1:2]) / 2
            r2 = (1 - nodes) * (lam[:, 2:3] - lam[:, 3:4]) / 2
            expo = (
                nodes * (lam[:, 0:1] + lam[:, 1:2]) / 2
                + (1 - nodes) * (lam[:, 2:3] + lam[:, 3:4]) / 2
            )
            log_f = (
                expo + r1 + r2 + np.log(special.ive(0, r1)) + np.log(special.ive(0, r2))
            )
        out = special.logsumexp(log_f + log_w, axis=-1)
    else:
        raise ValueError(f"exact Bingham normaliser needs d in (2, 3, 4); got {dim}.")
    return torch.tensor(out, dtype=eigenvalues.dtype)


def sphere_quadrature(dim: int, size: int = 1) -> tuple[Tensor, Tensor]:
    """Quadrature nodes on ``S^(dim-1)`` for the uniform probability measure.

    Supports ``d = 2, 3``.

    ``d = 2``: ``720 * size`` equispaced angles (spectrally accurate for periodic
    integrands). ``d = 3``: Gauss-Legendre in ``u = z_1`` (``200 * size`` nodes,
    clustered at the poles) times ``400 * size`` equispaced azimuths.

    Args:
        dim: 2 or 3.
        size: Refinement multiplier.

    Returns:
        ``(z, log_w)``: nodes ``(K, dim)`` (float64) and log weights ``(K,)`` summing
        (in ``exp``) to 1.

    Raises:
        ValueError: If ``dim`` is not 2 or 3.
    """
    if dim == 2:
        count = 720 * size
        angle = 2 * math.pi * (torch.arange(count, dtype=torch.float64) + 0.5) / count
        z = torch.stack([angle.cos(), angle.sin()], dim=-1)
        return z, torch.full((count,), -math.log(count), dtype=torch.float64)
    if dim == 3:
        n_u, n_phi = 200 * size, 400 * size
        nodes, weights = _gauss_legendre(n_u)
        u = torch.tensor(nodes, dtype=torch.float64)
        w_u = torch.tensor(weights / 2, dtype=torch.float64)
        phi = 2 * math.pi * (torch.arange(n_phi, dtype=torch.float64) + 0.5) / n_phi
        radius = (1 - u**2).sqrt()
        z = torch.stack(
            [
                u[:, None].expand(n_u, n_phi),
                radius[:, None] * phi.cos()[None],
                radius[:, None] * phi.sin()[None],
            ],
            dim=-1,
        ).reshape(-1, 3)
        log_w = (w_u.log()[:, None] - math.log(n_phi)).expand(n_u, n_phi).reshape(-1)
        return z, log_w
    raise ValueError(f"sphere_quadrature supports d in (2, 3); got {dim}.")


@functools.cache
def _gauss_legendre(num: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    nodes, weights = special.roots_legendre(num)
    return np.asarray(nodes), np.asarray(weights)


def _log_sphere_area(dim: int) -> float:
    """Log surface area of ``S^(dim-1)``."""
    return math.log(2) + (dim / 2) * math.log(math.pi) - math.lgamma(dim / 2)
