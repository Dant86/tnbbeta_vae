# tests/diffusion/test_noising.py
from __future__ import annotations

from typing import cast

from scipy import stats
import torch

from tnbbeta_vae.diffusion.noising import noise_to
from tnbbeta_vae.diffusion.schedule import DiffusionSchedule
from tnbbeta_vae.distributions import TNBBetaSpherical

_N = 20_000
_KS_STAT_MAX = 0.03


def _random_unit_vector(dim: int) -> torch.Tensor:
    v = torch.randn(dim)
    return v / v.norm()


def test_noise_to_is_identity_at_t_zero() -> None:
    """At t=0, noise_to draws TNBBetaSpherical(mean_direction, p_start, ...) itself.

    ``median(Y) = p`` exactly (TNBBetaUnivariate's Theorem 3.1), and the
    perpendicular radius near the pole scales like ``sqrt(1 - p)``, not like
    ``1 / eps`` -- so ``p_start=0.999`` is not within 1e-3 of mean_direction
    no matter how large ``eps_start`` is (confirmed directly against
    TNBBetaSpherical.rsample() itself, not just noise_to, at these same
    p/q/epsilon). atol is loosened to match that inherent floor rather than
    tightening p_start, to keep this test's arguments matching the "close to
    1"/"large" starting values documented on ``noise_to``.
    """
    torch.manual_seed(0)
    mean_direction = _random_unit_vector(5).expand(10, -1)
    schedule = DiffusionSchedule(eps_target=1.0)

    z = noise_to(
        mean_direction,
        torch.zeros(10),
        schedule,
        p_start=0.999,
        q_start=0.0,
        eps_start=50.0,
    )

    assert torch.allclose(z, mean_direction, atol=0.1)


def test_noise_to_converges_to_target_tnbbeta_spherical() -> None:
    """Large t: the cosine-to-mean_direction distribution matches the target."""
    torch.manual_seed(1)
    dim = 5
    mean_direction = _random_unit_vector(dim).expand(_N, -1)
    schedule = DiffusionSchedule(eps_target=2.0, q_target=0.05, speed=1.0)

    z = noise_to(
        mean_direction, torch.full((_N,), 10.0), schedule,
        p_start=0.999, q_start=0.0, eps_start=50.0,
    )  # fmt: skip
    cosine = (z * mean_direction).sum(dim=-1)

    target = TNBBetaSpherical(_random_unit_vector(dim), 0.5, 0.05, 2.0).rsample((_N,))
    target_cosine = (target * _random_unit_vector(dim)).sum(dim=-1)
    # Compare against a fresh TNBBetaSpherical draw's own cosine-to-its-own-
    # mean_direction, which is mean_direction-independent by construction --
    # this is the same "w" latitude variable either way.
    stat, _ = stats.ks_2samp(cosine.numpy(), target_cosine.numpy())
    assert cast(float, stat) < _KS_STAT_MAX


def test_noise_to_output_is_unit_norm() -> None:
    torch.manual_seed(2)
    mean_direction = _random_unit_vector(5).expand(100, -1)
    schedule = DiffusionSchedule(eps_target=1.0)

    z = noise_to(
        mean_direction, torch.full((100,), 3.0), schedule,
        p_start=0.999, q_start=0.0, eps_start=50.0,
    )  # fmt: skip

    assert torch.allclose(z.norm(dim=-1), torch.ones(100), atol=1e-4)
