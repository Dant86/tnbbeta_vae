"""Exponentially scaled modified Bessel function of the first kind, with gradient.

Ported from ``hyperspherical_vae/ops/ive.py`` in nicola-decao/s-vae-pytorch
(Davidson et al., 2018; MIT License, Copyright (c) 2018 Nicola De Cao).
"""

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np
import scipy.special
import torch
from torch import Tensor

__all__ = ["ive"]


class IveFunction(torch.autograd.Function):
    """``ive(v, z) = I_v(z) * exp(-|z|)`` for a scalar order ``v``."""

    @staticmethod
    def forward(ctx: Any, v: float, z: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        assert isinstance(v, Number), "v must be a scalar"

        ctx.save_for_backward(z)
        ctx.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)

        return torch.Tensor(output).to(z.device)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[None, Tensor]:  # pyright: ignore[reportIncompatibleMethodOverride]
        (z,) = ctx.saved_tensors
        return (
            None,
            grad_outputs[0] * (ive(ctx.v - 1, z) - ive(ctx.v, z) * (ctx.v + z) / z),
        )


def ive(v: float, z: Tensor) -> Tensor:
    """Computes the exponentially scaled Bessel function ``ive(v, z)``.

    Args:
        v: Scalar order.
        z: Argument tensor.

    Returns:
        ``I_v(z) * exp(-|z|)``, same shape as ``z``, differentiable in ``z``.
    """
    return IveFunction.apply(v, z)
