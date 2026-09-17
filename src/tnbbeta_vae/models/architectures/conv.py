"""A simple convolutional encoder/decoder pair for image VAEs.

The encoder maps images to a flat feature vector; squashing that into
valid TNBBetaSpherical parameters (a unit ``mean_direction``, ``p``/``q``
in (0, 1), ``epsilon`` > 0) is the caller's job (see
:class:`tnbbeta_vae.models.conv_vae.ConvTNBBetaSphericalVAE`), so this
module stays free of any distribution-specific logic and could equally
well feed a different latent family.
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["ConvDecoder", "ConvEncoder"]

_NUM_DOWNSAMPLES = 3  # three stride-2 convs: image_size -> image_size / 8


class ConvEncoder(nn.Module):
    """Downsamples an image to a flat feature vector, three stride-2 convs deep."""

    def __init__(
        self, image_channels: int, image_size: int, hidden_channels: int
    ) -> None:
        """Initializes the encoder.

        Args:
            image_channels: Number of input image channels (e.g. 3 for RGB).
            image_size: Height/width of the (square) input image. Must be
                divisible by 8.
            hidden_channels: Base channel width; doubled at each
                downsampling stage.

        Raises:
            ValueError: If ``image_size`` isn't divisible by 8.
        """
        super().__init__()
        if image_size % (2**_NUM_DOWNSAMPLES) != 0:
            raise ValueError(
                f"image_size must be divisible by {2**_NUM_DOWNSAMPLES} "
                f"(three stride-2 convs); got {image_size}."
            )
        self.feature_size = image_size // (2**_NUM_DOWNSAMPLES)
        self.out_channels = hidden_channels * 4
        self.out_features = self.out_channels * self.feature_size**2

        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Encodes a batch of images to flat features.

        Args:
            images: Batch of images, shape ``(batch, image_channels,
                image_size, image_size)``.

        Returns:
            Flat features, shape ``(batch, out_features)``.
        """
        return self.conv(images).flatten(1)


class ConvDecoder(nn.Module):
    """Upsamples a latent vector back to an image, mirroring :class:`ConvEncoder`."""

    def __init__(
        self,
        latent_dim: int,
        image_channels: int,
        image_size: int,
        hidden_channels: int,
    ) -> None:
        """Initializes the decoder.

        Args:
            latent_dim: Dimensionality of the input latent vector.
            image_channels: Number of output image channels.
            image_size: Height/width of the (square) output image. Must be
                divisible by 8.
            hidden_channels: Base channel width, matching the encoder's.

        Raises:
            ValueError: If ``image_size`` isn't divisible by 8.
        """
        super().__init__()
        if image_size % (2**_NUM_DOWNSAMPLES) != 0:
            raise ValueError(
                f"image_size must be divisible by {2**_NUM_DOWNSAMPLES} "
                f"(three stride-2 transposed convs); got {image_size}."
            )
        self.feature_size = image_size // (2**_NUM_DOWNSAMPLES)
        self.in_channels = hidden_channels * 4

        self.project = nn.Linear(latent_dim, self.in_channels * self.feature_size**2)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels * 2, hidden_channels, 4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, image_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decodes a batch of latent vectors to reconstructed images.

        Args:
            z: Latent vectors, shape ``(..., latent_dim)``.

        Returns:
            Reconstructed image means in [0, 1], shape ``(..., image_channels,
            image_size, image_size)``.
        """
        batch_shape = z.shape[:-1]
        features = self.project(z)
        features = features.reshape(
            *batch_shape, self.in_channels, self.feature_size, self.feature_size
        )
        flat_features = features.reshape(
            -1, self.in_channels, self.feature_size, self.feature_size
        )
        reconstruction = self.deconv(flat_features)
        return reconstruction.reshape(*batch_shape, *reconstruction.shape[1:])
