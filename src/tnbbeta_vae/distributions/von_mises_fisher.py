"""Von Mises-Fisher distribution, as in the hyperspherical VAE (S-VAE).

Ported from ``hyperspherical_vae/distributions/von_mises_fisher.py`` in
nicola-decao/s-vae-pytorch (Davidson et al., 2018, "Hyperspherical
Variational Auto-Encoders"; MIT License, Copyright (c) 2018 Nicola De Cao).
The sampling (Wood's rejection sampler, Householder rotation) and the
analytic KL to the uniform distribution (via Bessel functions) follow the
original; only style/typing changes were made.

As in the original, ``rsample`` draws the accepted Beta variate without
gradient, so the pathwise gradient w.r.t. the concentration flows only
through the proposal parameters ``b``, ``a``, ``d`` (the paper's additional
correction term is not implemented in the reference code).

Shapes follow the original: ``loc`` is ``(..., m)`` and ``scale`` (the
concentration ``kappa``) is ``(..., 1)``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
from torch.distributions.kl import register_kl

from tnbbeta_vae.distributions._ive import ive
from tnbbeta_vae.distributions.hyperspherical_uniform import HypersphericalUniform

__all__ = ["VonMisesFisher"]


class VonMisesFisher(Distribution):
    """Von Mises-Fisher distribution on S^(m-1), as in the original S-VAE.

    Attributes:
        loc: Unit-norm mean direction, shape ``(..., m)``.
        scale: Concentration ``kappa > 0``, shape ``(..., 1)``.
    """

    arg_constraints = {  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    has_rsample = True
    _mean_carrier_measure = 0

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        validate_args: bool | None = None,
        k: int = 1,
    ) -> None:
        """Initializes the distribution.

        Args:
            loc: Unit-norm mean direction, shape ``(..., m)``.
            scale: Concentration ``kappa``, shape ``(..., 1)``.
            validate_args: Whether to validate arguments.
            k: Number of candidate proposals drawn per rejection-sampling
                round.
        """
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self._m = loc.shape[-1]
        self._e1 = torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1)).to(self.device)
        self.k = k

        super().__init__(self.loc.size(), validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """Mean of the distribution."""
        return self.loc * (
            ive(self._m / 2, self.scale) / ive(self._m / 2 - 1, self.scale)
        )

    @property
    def stddev(self) -> Tensor:
        """Concentration (as in the original)."""
        return self.scale

    def sample(self, sample_shape: torch.Size | int = torch.Size()) -> Tensor:  # noqa: B008  # pyright: ignore[reportIncompatibleMethodOverride]
        """Draws samples without tracking gradients.

        Args:
            sample_shape: Shape of the sample batch.

        Returns:
            Unit-norm samples.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape: torch.Size | int = torch.Size()) -> Tensor:  # noqa: B008  # pyright: ignore[reportIncompatibleMethodOverride]
        """Draws samples: rejection-sample ``w``, then Householder-rotate.

        Args:
            sample_shape: Shape of the sample batch.

        Returns:
            Unit-norm samples, shape ``sample_shape + loc.shape``.
        """
        shape = (
            sample_shape
            if isinstance(sample_shape, torch.Size)
            else torch.Size([sample_shape])
        )

        w = (
            self._sample_w3(shape=shape)
            if self._m == 3
            else self._sample_w_rej(shape=shape)
        )

        v = (
            torch.distributions.Normal(0, 1)
            .sample(shape + torch.Size(self.loc.shape))
            .to(self.device)
            .transpose(0, -1)[1:]
        ).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1 - (w**2), 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self._householder_rotation(x)

        return z.type(self.dtype)

    def entropy(self) -> Tensor:
        """Analytic entropy, via Bessel functions."""
        output = (
            -self.scale
            * ive(self._m / 2, self.scale)
            / ive((self._m / 2) - 1, self.scale)
        )
        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def log_prob(self, value: Tensor) -> Tensor:
        """Computes the log-density.

        Args:
            value: Points on the sphere, shape ``(..., m)``.

        Returns:
            Log-density, shape ``value.shape[:-1]``.
        """
        return self._log_unnormalized_prob(value) - self._log_normalization()

    def _sample_w3(self, shape: torch.Size) -> Tensor:
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        return (
            1
            + torch.stack(
                [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0
            ).logsumexp(0)
            / self.scale
        )

    def _sample_w_rej(self, shape: torch.Size) -> Tensor:
        c = torch.sqrt((4 * (self.scale**2)) + (self._m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self._m - 1)

        # Taylor approximation with a smooth switch from 10 < scale < 11 to
        # avoid numerical errors for large scale.
        b_app = (self._m - 1) / (4 * self.scale)
        s = torch.min(
            torch.max(
                torch.tensor([0.0], dtype=self.dtype, device=self.device),
                self.scale - 10,
            ),
            torch.tensor([1.0], dtype=self.dtype, device=self.device),
        )
        b = b_app * s + b_true * (1 - s)

        a = (self._m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self._m - 1) * math.log(self._m - 1)

        _, w = self._while_loop(b, a, d, shape, k=self.k)
        return w

    def _while_loop(
        self,
        b: Tensor,
        a: Tensor,
        d: Tensor,
        shape: torch.Size,
        k: int = 20,
        eps: float = 1e-20,
    ) -> tuple[Tensor, Tensor]:
        # Matrix while loop: samples an [A, k] matrix to avoid looping over rows.
        b, a, d = (
            e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
            for e in (b, a, d)
        )
        w, e, bool_mask = (
            torch.zeros_like(b).to(self.device),
            torch.zeros_like(b).to(self.device),
            (torch.ones_like(b) == 1).to(self.device),
        )

        sample_shape = torch.Size([b.shape[0], k])
        shape = shape + torch.Size(self.scale.shape)

        while bool_mask.sum() != 0:
            con1 = torch.tensor((self._m - 1) / 2, dtype=torch.float64)
            con2 = torch.tensor((self._m - 1) / 2, dtype=torch.float64)
            e_ = (
                torch.distributions.Beta(con1, con2)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            u = (
                torch.distributions.Uniform(0 + eps, 1 - eps)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self._m - 1.0) * t.log() - t + d) > torch.log(u)
            accept_idx = self._first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(
                1
            )
            accept_idx_clamped = accept_idx.clamp(0)
            # .clamp(0) avoids -1 index issues; the -1 is still used afterwards.
            w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
            e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))

            reject = accept_idx < 0
            accept = ~reject

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]

        return e.reshape(shape), w.reshape(shape)

    @staticmethod
    def _first_nonzero(x: Tensor, dim: int, invalid_val: int = -1) -> Tensor:
        mask = x > 0
        return torch.where(
            mask.any(dim=dim),
            mask.float().argmax(dim=1).squeeze(),
            torch.tensor(invalid_val, device=x.device),
        )

    def _householder_rotation(self, x: Tensor) -> Tensor:
        u = self._e1 - self.loc
        # Exact normalization keeps the reflection orthonormal, so samples stay on
        # the sphere even when loc is close to e1 (common at latent_dim=2). u = 0
        # (loc == e1) gives the identity, which is the right rotation.
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x - 2 * (x * u).sum(-1, keepdim=True) * u

    def _log_unnormalized_prob(self, x: Tensor) -> Tensor:
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)
        return output.view(*(output.shape[:-1]))

    def _log_normalization(self) -> Tensor:
        output = -(
            (self._m / 2 - 1) * torch.log(self.scale)
            - (self._m / 2) * math.log(2 * math.pi)
            - (self.scale + torch.log(ive(self._m / 2 - 1, self.scale)))
        )
        return output.view(*(output.shape[:-1]))


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf: VonMisesFisher, hyu: HypersphericalUniform) -> Tensor:
    return -vmf.entropy() + hyu.entropy()
