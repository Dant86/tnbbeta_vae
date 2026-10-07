from __future__ import annotations

import torch

from tnbbeta_vae.models.architectures.denoiser_mlp import SphereDenoiserMLP


def test_output_is_unit_norm_and_correct_shape() -> None:
    torch.manual_seed(0)
    model = SphereDenoiserMLP(latent_dim=8, hidden_dim=32, depth=2)
    z = torch.randn(6, 8)
    z = z / z.norm(dim=-1, keepdim=True)
    t = torch.rand(6) * 10

    out = model(z, t)

    assert out.shape == (6, 8)
    assert torch.allclose(out.norm(dim=-1), torch.ones(6), atol=1e-5)


def test_different_t_gives_different_output() -> None:
    """Sanity check that t actually conditions the network, not just z."""
    torch.manual_seed(1)
    model = SphereDenoiserMLP(latent_dim=4, hidden_dim=16, depth=2)
    z = torch.randn(3, 4)
    z = z / z.norm(dim=-1, keepdim=True)

    out_t0 = model(z, torch.zeros(3))
    out_t1 = model(z, torch.full((3,), 5.0))

    assert not torch.allclose(out_t0, out_t1)


def test_gradients_flow_to_every_parameter() -> None:
    torch.manual_seed(2)
    model = SphereDenoiserMLP(latent_dim=4, hidden_dim=16, depth=2)
    z = torch.randn(5, 4)
    z = z / z.norm(dim=-1, keepdim=True)
    t = torch.rand(5) * 10

    out = model(z, t)
    out.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
