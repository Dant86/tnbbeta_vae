"""Side-by-side Hammer projection of vMF and TNBBeta S^2 latents, MNIST test split.

Usage:
    uv run python -m apps.eval.hammer_projection --vmf-run NAME --tnb-run NAME \\
        [--checkpoint final] [--output PATH]

Reproduces the style of Figure 2 in Davidson et al. (2018) -- a Hammer (equal-area)
projection of the S^2 latent space, one point per test image, colored by class -- for
one vMF ("S-VAE") run and one TNBBeta run side by side in a single PNG, sized for
``\\includegraphics`` in a paper. Both runs must have ``latent_dim == 3`` (S^2 in R^3,
the paper-convention ambient dimension for its d=2), since the Hammer projection is a
map from the 2-sphere; there is no plotting for any other model or dimension.

Each posterior's centre direction is plotted (not a sample): ``VonMisesFisher.loc`` for
vMF, and TNBBeta's `mode_direction` -- the mean direction, negated when ``p < 0.5`` to
undo the ``(mu, p) ~ (-mu, 1 - p)`` alias, so a single class doesn't get arbitrarily
split across antipodal poles by that gauge freedom (see :func:`posterior_centre`).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tnbbeta_vae.data.mnist import load_mnist  # noqa: E402
from tnbbeta_vae.models.heads import LatentFamily, posterior_centre  # noqa: E402
from tnbbeta_vae.paths import checkpoint_dir, data_dir  # noqa: E402
from tnbbeta_vae.training import load_model_checkpoint  # noqa: E402

_FAMILY_BY_MODEL: dict[str, LatentFamily] = {
    "conv_vmf_vae": "vmf",
    "conv_tnbbeta_spherical_vae": "tnbbeta",
}
_TITLE_BY_MODEL = {
    "conv_vmf_vae": "S-VAE (vMF)",
    "conv_tnbbeta_spherical_vae": "TNBBeta",
}
_SPHERE_DIM = 3  # S^2 in R^3.


def main(argv: list[str] | None = None) -> None:
    """Plots the two-panel Hammer projection figure and writes it as a PNG.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmf-run", required=True)
    parser.add_argument("--tnb-run", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    test_set = load_mnist(data_dir(), split="test")
    labels = test_set.labels().numpy()

    panels = [
        _panel(run_name, args, device, test_set)
        for run_name in (args.vmf_run, args.tnb_run)
    ]

    output = args.output or str(
        checkpoint_dir() / f"hammer_{args.vmf_run}_vs_{args.tnb_run}.png"
    )
    _plot(panels, labels, output, args.point_size)
    print(f"Wrote {output}")


def _panel(
    run_name: str, args: argparse.Namespace, device: torch.device, test_set: Any
) -> tuple[str, np.ndarray]:
    """Returns ``(panel title, (lon, lat) array)`` for one run's test-set centres."""
    run_dir = checkpoint_dir() / run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)
    kind = checkpoint["model_name"]
    if kind not in _FAMILY_BY_MODEL:
        raise ValueError(
            f"{run_name}: model is {kind}, which has no spherical latent to project."
        )
    latent_dim = checkpoint["config"]["latent_dim"]
    if latent_dim != _SPHERE_DIM:
        raise ValueError(
            f"{run_name}: latent_dim is {latent_dim}, not {_SPHERE_DIM} (S^2); "
            "the Hammer projection only applies to a run trained on S^2."
        )
    directions = _centres(model, _FAMILY_BY_MODEL[kind], test_set, args, device)
    return _TITLE_BY_MODEL[kind], _to_lon_lat(directions)


@torch.no_grad()
def _centres(
    model: Any,
    family: LatentFamily,
    dataset: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    """Returns every test image's posterior centre direction, shape ``(n, 3)``."""
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    parts = []
    for batch in loader:
        posterior, _ = model.posterior_and_prior(batch.to(device))
        parts.append(posterior_centre(family, posterior).cpu())
    return torch.cat(parts).numpy()


def _to_lon_lat(directions: np.ndarray) -> np.ndarray:
    """Maps unit vectors in R^3 to (longitude, latitude) in radians for a Hammer axes.

    ``matplotlib``'s ``"hammer"`` projection expects longitude in ``[-pi, pi]`` and
    latitude in ``[-pi/2, pi/2]``; the third coordinate is treated as the polar axis.
    """
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    longitude = np.arctan2(y, x)
    latitude = np.arcsin(np.clip(z, -1.0, 1.0))
    return np.stack([longitude, latitude], axis=1)


def _plot(
    panels: list[tuple[str, np.ndarray]],
    labels: np.ndarray,
    output: str,
    point_size: float,
) -> None:
    """Draws both panels' Hammer projections, one shared class legend, and saves."""
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6 * len(panels), 3.6),
        subplot_kw={"projection": "hammer"},
    )
    colormap = plt.get_cmap("tab10")
    for axis, (title, lon_lat) in zip(np.atleast_1d(axes), panels, strict=True):
        for class_index in range(10):
            selected = labels == class_index
            axis.scatter(
                lon_lat[selected, 0],
                lon_lat[selected, 1],
                s=point_size,
                color=colormap(class_index),
                label=str(class_index),
                alpha=0.6,
                linewidths=0,
            )
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.set_xticklabels([])
        axis.set_yticklabels([])
    axes_list = np.atleast_1d(axes)
    axes_list[-1].legend(
        title="class",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        markerscale=3,
        fontsize="small",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main(sys.argv[1:])
