"""Tests for DiffusionSchedule."""

from __future__ import annotations

from typing import cast

import pytest
from scipy import stats
import torch

from tnbbeta_vae.diffusion.schedule import DiffusionSchedule
from tnbbeta_vae.distributions import TNBBetaUnivariate

_N = 20_000
_KS_STAT_MAX = 0.03


def test_fields_default_as_specified() -> None:
    schedule = DiffusionSchedule(eps_target=1.0)

    assert schedule.eps_target == 1.0
    assert schedule.q_target == 0.05
    assert schedule.speed == 1.0


def test_draw_latitude_matches_forward_draw_latitude_directly() -> None:
    torch.manual_seed(0)
    schedule = DiffusionSchedule(eps_target=1.0, q_target=0.05, speed=1.0)

    w = schedule.draw_latitude(0.7, 0.3, 2.0, torch.full((_N,), 8.0))
    y = (w + 1) / 2

    target = TNBBetaUnivariate(
        torch.tensor(0.5), torch.tensor(0.05), torch.tensor(1.0)
    ).rsample((_N,))
    stat, _ = stats.ks_2samp(y.numpy(), target.numpy())
    assert cast(float, stat) < _KS_STAT_MAX


def test_is_immutable() -> None:
    schedule = DiffusionSchedule(eps_target=1.0)

    with pytest.raises(AttributeError):
        schedule.eps_target = 2.0  # type: ignore[misc]
