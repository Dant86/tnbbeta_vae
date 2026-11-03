"""Tests for the sign-symmetrised wrapper."""

from __future__ import annotations

import torch

from tnbbeta_vae.distributions import TNBBetaSpherical, VonMisesFisher
from tnbbeta_vae.distributions.sign_symmetrised import SignSymmetrised


def _bases() -> list[torch.distributions.Distribution]:
    mu = torch.nn.functional.normalize(torch.randn(3, 3, dtype=torch.float64), dim=-1)
    vmf = VonMisesFisher(mu, torch.full((3, 1), 8.0, dtype=torch.float64))
    tnb = TNBBetaSpherical(
        mu,
        torch.full((3,), 0.7, dtype=torch.float64),
        torch.full((3,), 0.3, dtype=torch.float64),
        torch.full((3,), 0.8, dtype=torch.float64),
    )
    return [vmf, tnb]


def test_log_prob_is_antipodal_and_matches_exact_log_q() -> None:
    torch.manual_seed(0)
    for base in _bases():
        dist = SignSymmetrised(base)
        z, log_q = dist.rsample_with_log_prob(torch.Size([50]))
        assert z.shape == (50, 3, 3)
        assert torch.allclose(z.norm(dim=-1), torch.ones(50, 3, dtype=torch.float64))
        assert torch.allclose(log_q, dist.log_prob(z), atol=1e-6)
        assert torch.allclose(dist.log_prob(z), dist.log_prob(-z), atol=1e-9)


def test_both_hemispheres_are_visited() -> None:
    torch.manual_seed(1)
    for base in _bases():
        z = SignSymmetrised(base).rsample(torch.Size([400]))
        mean = getattr(base, "mean_direction", None)
        if mean is None:
            mean = base.loc  # pyright: ignore[reportAttributeAccessIssue]
        share = float(((z * mean).sum(-1) > 0).double().mean())
        assert 0.35 < share < 0.65


def test_density_integrates_to_one_on_circle() -> None:
    mu = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    base = VonMisesFisher(mu, torch.tensor([[5.0]], dtype=torch.float64))
    angle = torch.linspace(0, 2 * torch.pi, 20001, dtype=torch.float64)[:-1]
    points = torch.stack([angle.cos(), angle.sin()], dim=-1)
    density = SignSymmetrised(base).log_prob(points).exp()
    assert abs(float(density.mean() * 2 * torch.pi) - 1) < 1e-6
