"""Writes a side-by-side PNG comparing a VAE's fixed-prior samples against
a diffusion prior's samples, for the qualitative "sharper than blurry VAE
samples" check FID alone doesn't show.

Usage:
    uv run python -m apps.eval.sample_grid --vae-run NAME --diffusion-run NAME \
        [--num-samples 8] [--checkpoint final] [--device cpu] [--output grid.png]
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, cast

import torch
import torchvision.utils as vutils

from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import load_model_checkpoint

if TYPE_CHECKING:
    from torch import Tensor


def main(argv: list[str] | None = None) -> None:
    """Parses CLI args and writes the comparison grid.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-run", required=True)
    parser.add_argument("--diffusion-run", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="sample_grid.png")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    vae_model, _ = load_model_checkpoint(
        checkpoint_dir() / args.vae_run / f"{args.checkpoint}.pt", device
    )
    diffusion_model, _ = load_model_checkpoint(
        checkpoint_dir() / args.diffusion_run / f"{args.checkpoint}.pt", device
    )

    with torch.no_grad():
        vae_images = cast("Tensor", cast("Any", vae_model).generate(args.num_samples))
        diffusion_images = cast(
            "Tensor", cast("Any", diffusion_model).generate(args.num_samples)
        )

    _save_comparison_grid(vae_images, diffusion_images, args.output)
    print(f"Wrote {args.output}")


def _save_comparison_grid(
    top_row: Tensor, bottom_row: Tensor, output_path: str
) -> None:
    """Saves a two-row PNG: top=top_row's images, bottom=bottom_row's images."""
    combined = torch.cat([top_row.cpu(), bottom_row.cpu()], dim=0)
    grid = vutils.make_grid(combined, nrow=top_row.shape[0])
    vutils.save_image(grid, output_path)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
