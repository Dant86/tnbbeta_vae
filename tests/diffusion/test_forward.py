"""Tests for tnbbeta_vae.diffusion.forward.

Statistical checks (KS test against TNBBetaUnivariate's closed-form
density) at N=20,000 -- smaller than the N=200,000 used during the
exploratory notebooks/ scripts this is promoted from, traded for test-suite
speed; the threshold is loosened to match (stat < 0.03 rather than 0.01).
"""

from __future__ import annotations

from scipy import stats
import torch

from tnbbeta_vae.diffusion.forward import draw_latitude, leisen_step, resize_eps
from tnbbeta_vae.distributions import TNBBetaUnivariate

_N = 20_000
_KS_STAT_MAX = 0.03


def _ks_match(x: torch.Tensor, y: torch.Tensor) -> bool:
    stat, _ = stats.ks_2samp(x.numpy(), y.numpy())
    return stat < _KS_STAT_MAX


def _closed_form_y(p: float, q: float, eps: float, n: int) -> torch.Tensor:
    return TNBBetaUnivariate(
        torch.tensor(p), torch.tensor(q), torch.tensor(eps)
    ).rsample((n,))


def test_resize_eps_split_direction_matches_closed_form() -> None:
    """eps_to < eps_from: splitting off a smaller piece keeps the same q."""
    torch.manual_seed(0)
    p, q, eps_from, eps_to = 0.4, 0.6, 5.0, 2.0
    c = torch.distributions.NegativeBinomial(
        torch.full((_N,), eps_from), probs=torch.full((_N,), q)
    ).sample()

    resized = resize_eps(
        c, torch.full((_N,), eps_from), torch.full((_N,), eps_to), torch.full((_N,), q)
    )
    a_t = torch.distributions.NegativeBinomial(eps_to + resized, probs=p).sample()
    b_t = torch.distributions.NegativeBinomial(eps_to + resized, probs=1 - p).sample()
    y = torch.distributions.Beta(
        eps_to + resized + a_t, eps_to + resized + b_t
    ).sample()

    assert _ks_match(y, _closed_form_y(p, q, eps_to, _N))


def test_resize_eps_merge_direction_matches_closed_form() -> None:
    """eps_to > eps_from: merging in a fresh piece keeps the same q."""
    torch.manual_seed(1)
    p, q, eps_from, eps_to = 0.4, 0.6, 1.0, 4.0
    c = torch.distributions.NegativeBinomial(
        torch.full((_N,), eps_from), probs=torch.full((_N,), q)
    ).sample()

    resized = resize_eps(
        c, torch.full((_N,), eps_from), torch.full((_N,), eps_to), torch.full((_N,), q)
    )
    a_t = torch.distributions.NegativeBinomial(eps_to + resized, probs=p).sample()
    b_t = torch.distributions.NegativeBinomial(eps_to + resized, probs=1 - p).sample()
    y = torch.distributions.Beta(
        eps_to + resized + a_t, eps_to + resized + b_t
    ).sample()

    assert _ks_match(y, _closed_form_y(p, q, eps_to, _N))


def test_leisen_step_is_identity_at_t_zero() -> None:
    torch.manual_seed(2)
    c0 = torch.distributions.NegativeBinomial(
        torch.full((_N,), 3.0), probs=torch.full((_N,), 0.5)
    ).sample()

    c_t = leisen_step(
        c0, torch.zeros(_N), torch.full((_N,), 3.0), q_target=0.1, speed=1.0
    )

    assert torch.equal(c_t, c0)


def test_leisen_step_converges_regardless_of_starting_q() -> None:
    """Large t: C_t's law -> NB(eps, q_target), independent of C_0's own q."""
    torch.manual_seed(3)
    eps, q_target = 3.0, 0.05
    c0_a = torch.distributions.NegativeBinomial(
        torch.full((_N,), eps), probs=torch.full((_N,), 0.9)
    ).sample()
    c0_b = torch.distributions.NegativeBinomial(
        torch.full((_N,), eps), probs=torch.full((_N,), 0.1)
    ).sample()

    c_t_a = leisen_step(
        c0_a, torch.full((_N,), 8.0), torch.full((_N,), eps), q_target, 1.0
    )
    c_t_b = leisen_step(
        c0_b, torch.full((_N,), 8.0), torch.full((_N,), eps), q_target, 1.0
    )

    stat, _ = stats.ks_2samp(c_t_a.numpy(), c_t_b.numpy())
    assert stat < _KS_STAT_MAX


def test_draw_latitude_recovers_data_at_t_zero() -> None:
    torch.manual_seed(4)
    p, q, eps = 0.85, 0.6, 2.5

    w = draw_latitude(
        p, q, eps, torch.zeros(_N), eps_target=1.0, q_target=0.05, speed=1.0
    )
    y = (w + 1) / 2

    assert _ks_match(y, _closed_form_y(p, q, eps, _N))


def test_draw_latitude_converges_to_target_regardless_of_data_params() -> None:
    torch.manual_seed(5)
    eps_target, q_target = 1.0, 0.05
    target_y = _closed_form_y(0.5, q_target, eps_target, _N)

    for p_data, q_data, eps_data in [
        (0.85, 0.6, 2.5),
        (0.1, 0.0, 0.5),
        (0.5, 0.9, 5.0),
    ]:
        w = draw_latitude(
            p_data, q_data, eps_data, torch.full((_N,), 10.0),
            eps_target=eps_target, q_target=q_target, speed=1.0,
        )  # fmt: skip
        y = (w + 1) / 2
        assert _ks_match(y, target_y), (p_data, q_data, eps_data)


def test_draw_latitude_is_finite_near_t_zero() -> None:
    """Review Focus: near-point-mass params (eps=50, p=0.999) must not overflow."""
    torch.manual_seed(6)

    w = draw_latitude(
        0.999,
        0.0,
        50.0,
        torch.full((1000,), 0.01),
        eps_target=1.0,
        q_target=0.05,
        speed=1.0,
    )

    assert torch.isfinite(w).all()
    assert (w > -1).all() and (w < 1).all()
