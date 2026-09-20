"""A simple convolutional encoder/decoder pair for image VAEs.

The encoder maps images to a flat feature vector; squashing that into
valid TNBBetaSpherical parameters (a unit ``mean_direction``, ``p``/``q``
in (0, 1), ``epsilon`` > 0) is the caller's job (see
:class:`tnbbeta_vae.models.conv_vae.ConvTNBBetaSphericalVAE`), so this
module stays free of any distribution-specific logic and could equally
well feed a different latent family.

Every conv/deconv/projection layer (other than the final output) is
followed by GroupNorm. Without it, an early sweep along a single geodesic
direction in latent space showed the
decoder swinging output color wildly for *any* direction, not some
dedicated "color" subspace -- a hair-trigger, poorly-conditioned mapping
that left no safe direction for the encoder to route weakly-rewarded
information (e.g. color) through without also disturbing whatever
strongly-rewarded information (e.g. position) was already encoded there.
GroupNorm (not BatchNorm) is used so behavior doesn't depend on batch
size or differ between train/eval mode.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["ConvDecoder", "ConvEncoder"]

_NUM_DOWNSAMPLES = 3  # three stride-2 convs: image_size -> image_size / 8
_NUM_GROUPS = 8


def _padding(image_size: int) -> tuple[int, int]:
    """Returns the (before, after) zero padding that makes a size divisible by 8.

    Sizes already divisible by 8 (e.g. CIFAR-10's 32) get none; 28 (MNIST) gets 2
    on each side, i.e. it is padded to 32.
    """
    total = -image_size % 2**_NUM_DOWNSAMPLES
    return total // 2, total - total // 2


class ConvEncoder(nn.Module):
    """Downsamples an image to a flat feature vector, three stride-2 convs deep."""

    def __init__(
        self, image_channels: int, image_size: int, hidden_channels: int
    ) -> None:
        """Initializes the encoder.

        Args:
            image_channels: Number of input image channels (e.g. 3 for RGB).
            image_size: Height/width of the (square) input image. If it is not
                divisible by 8 it is zero-padded up to the next multiple of 8
                (28 becomes 32).
            hidden_channels: Base channel width; doubled at each
                downsampling stage.

        Raises:
            ValueError: If ``hidden_channels`` isn't divisible by 8 (needed for
                GroupNorm).
        """
        super().__init__()
        before, after = _padding(image_size)
        self._pad = (before, after, before, after)
        if hidden_channels % _NUM_GROUPS != 0:
            raise ValueError(
                f"hidden_channels must be divisible by {_NUM_GROUPS} "
                f"(GroupNorm); got {hidden_channels}."
            )
        self.feature_size = (image_size + before + after) // (2**_NUM_DOWNSAMPLES)
        self.out_channels = hidden_channels * 4
        self.out_features = self.out_channels * self.feature_size**2

        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, hidden_channels, 4, stride=2, padding=1),
            nn.GroupNorm(_NUM_GROUPS, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(_NUM_GROUPS, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(_NUM_GROUPS, hidden_channels * 4),
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
        return self.conv(nn.functional.pad(images, self._pad)).flatten(1)


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
            image_size: Height/width of the (square) output image. If it is not
                divisible by 8 the deconv stack runs at the next multiple of 8
                and the output is cropped back (32 -> 28).
            hidden_channels: Base channel width, matching the encoder's.

        Raises:
            ValueError: If ``hidden_channels`` isn't divisible by 8 (needed for
                GroupNorm).
        """
        super().__init__()
        before, after = _padding(image_size)
        self._crop = slice(before, before + image_size)
        if hidden_channels % _NUM_GROUPS != 0:
            raise ValueError(
                f"hidden_channels must be divisible by {_NUM_GROUPS} "
                f"(GroupNorm); got {hidden_channels}."
            )
        self.feature_size = (image_size + before + after) // (2**_NUM_DOWNSAMPLES)
        self.in_channels = hidden_channels * 4

        self.project = nn.Linear(latent_dim, self.in_channels * self.feature_size**2)
        # Normalizes the projection's output before it fans out into the
        # deconv stack -- this is the single dense layer through which
        # every latent dimension mixes into every initial spatial/channel
        # position, so it's the most direct point to condition.
        self.project_norm = nn.GroupNorm(_NUM_GROUPS, self.in_channels)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1
            ),
            nn.GroupNorm(_NUM_GROUPS, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels * 2, hidden_channels, 4, stride=2, padding=1
            ),
            nn.GroupNorm(_NUM_GROUPS, hidden_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels, image_channels, 4, stride=2, padding=1),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decodes a batch of latent vectors to reconstructed images.

        Args:
            z: Latent vectors, shape ``(..., latent_dim)``.

        Returns:
            Reconstructed image means in [0, 1], shape ``(..., image_channels,
            image_size, image_size)``.
        """
        return torch.sigmoid(self.logits(z))

    def logits(self, z: Tensor) -> Tensor:
        """Decodes latent vectors to pre-sigmoid pixel logits.

        Args:
            z: Latent vectors, shape ``(..., latent_dim)``.

        Returns:
            Logits with the same shape as :meth:`forward`'s output. A Bernoulli
            likelihood uses these directly, which is more stable than taking the
            log of a saturated sigmoid.
        """
        batch_shape = z.shape[:-1]
        features = self.project(z)
        flat_features = features.reshape(
            -1, self.in_channels, self.feature_size, self.feature_size
        )
        flat_features = torch.relu(self.project_norm(flat_features))
        logits = self.deconv(flat_features)[..., self._crop, self._crop]
        return logits.reshape(*batch_shape, *logits.shape[1:])
