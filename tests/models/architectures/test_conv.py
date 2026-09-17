"""Tests for tnbbeta_vae.models.architectures.conv."""

from __future__ import annotations

import pytest
import torch

from tnbbeta_vae.models.architectures.conv import ConvDecoder, ConvEncoder


def test_encoder_output_shape() -> None:
    encoder = ConvEncoder(image_channels=3, image_size=32, hidden_channels=8)
    x = torch.randn(5, 3, 32, 32)

    features = encoder(x)

    assert features.shape == (5, encoder.out_features)


def test_decoder_output_shape_and_range() -> None:
    decoder = ConvDecoder(
        latent_dim=6, image_channels=3, image_size=32, hidden_channels=8
    )
    z = torch.randn(5, 6)

    images = decoder(z)

    assert images.shape == (5, 3, 32, 32)
    assert torch.all(images >= 0.0)
    assert torch.all(images <= 1.0)


def test_decoder_preserves_arbitrary_leading_batch_dims() -> None:
    """Decoder is used with both (batch, dim) and (batch, dim) posterior samples."""
    decoder = ConvDecoder(
        latent_dim=6, image_channels=3, image_size=32, hidden_channels=8
    )
    z = torch.randn(2, 3, 6)

    images = decoder(z)

    assert images.shape == (2, 3, 3, 32, 32)


def test_encoder_rejects_image_size_not_divisible_by_eight() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        ConvEncoder(image_channels=3, image_size=30, hidden_channels=8)


def test_decoder_rejects_image_size_not_divisible_by_eight() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        ConvDecoder(latent_dim=6, image_channels=3, image_size=30, hidden_channels=8)


def test_encoder_decoder_round_trip_shapes_match_input() -> None:
    """Sanity check that encoder -> linear -> decoder recovers the input shape."""
    image_size = 32
    encoder = ConvEncoder(image_channels=3, image_size=image_size, hidden_channels=8)
    decoder = ConvDecoder(
        latent_dim=6, image_channels=3, image_size=image_size, hidden_channels=8
    )
    to_latent = torch.nn.Linear(encoder.out_features, 6)
    x = torch.randn(4, 3, image_size, image_size)

    z = to_latent(encoder(x))
    reconstruction = decoder(z)

    assert reconstruction.shape == x.shape
