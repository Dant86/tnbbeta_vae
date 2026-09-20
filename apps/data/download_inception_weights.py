"""Downloads the pytorch-fid Inception weights used by ``apps.eval.fid``.

Run this once before computing FID (it needs network access, so use the login
node if compute nodes have none); ``apps.eval.fid`` reads the weights from disk.
The file (about 91 MB) is cached under ``$TNBBETA_DATA_DIR/torch_hub``.

Usage:
    uv run python -m apps.data.download_inception_weights
"""

from __future__ import annotations

from pathlib import Path
import sys

from pytorch_fid.inception import FID_WEIGHTS_URL
import torch

from tnbbeta_vae.paths import torch_hub_dir

_MIN_BYTES = 50_000_000


def main(argv: list[str] | None = None) -> None:
    """Downloads and verifies the Inception weights.

    Args:
        argv: Unused; accepted for symmetry with the other CLI scripts.

    Raises:
        RuntimeError: If the cached file is missing or implausibly small.
    """
    del argv
    torch.hub.set_dir(str(torch_hub_dir()))
    torch.hub.load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(FID_WEIGHTS_URL).name
    if not path.exists() or path.stat().st_size < _MIN_BYTES:
        raise RuntimeError(f"Inception weights were not cached at {path}.")
    print(f"Inception weights OK at {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1:])
