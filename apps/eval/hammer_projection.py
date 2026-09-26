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

Uses plotly's ``Scattergeo`` with a ``"hammer"`` geo projection (this project's other
plots -- ``apps.eval.svae_latitude``, ``notebooks/plot_sphere_3d.py`` -- are plotly
too), rendered to a static PNG via ``kaleido``, styled with the shared
``tnbbeta_vae.plotting`` theme matching the paper's LaTeX template.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader

from tnbbeta_vae.data.mnist import load_mnist
from tnbbeta_vae.models.heads import LatentFamily, posterior_centre
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.plotting import TEMPLATE_NAME
from tnbbeta_vae.training import load_model_checkpoint

_FAMILY_BY_MODEL: dict[str, LatentFamily] = {
    "conv_vmf_vae": "vmf",
    "conv_tnbbeta_spherical_vae": "tnbbeta",
}
_TITLE_BY_MODEL = {
    "conv_vmf_vae": "S-VAE (vMF)",
    "conv_tnbbeta_spherical_vae": "TNBBeta",
}
_SPHERE_DIM = 3  # S^2 in R^3.
CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]  # fmt: skip
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]  # fmt: skip


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
    figure = _figure(panels, labels, args.point_size)
    figure.write_image(output, width=1500, height=490, scale=2)
    print(f"Wrote {output}")


def _panel(
    run_name: str, args: argparse.Namespace, device: torch.device, test_set: Any
) -> tuple[str, np.ndarray]:
    """Returns ``(panel title, (lon, lat) array in degrees)`` for one run's centres."""
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
    """Maps unit vectors in R^3 to (longitude, latitude) in degrees, for Scattergeo.

    The third coordinate is treated as the polar axis.
    """
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    longitude = np.degrees(np.arctan2(y, x))
    latitude = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return np.stack([longitude, latitude], axis=1)


def _figure(
    panels: list[tuple[str, np.ndarray]], labels: np.ndarray, point_size: float
) -> go.Figure:
    """Builds the two-panel Hammer-projection figure, one shared class legend."""
    figure = make_subplots(
        rows=1,
        cols=len(panels),
        specs=[[{"type": "scattergeo"}] * len(panels)],
        subplot_titles=[title for title, _ in panels],
        horizontal_spacing=0.02,
    )
    for column, (_, lon_lat) in enumerate(panels, start=1):
        for class_index, name in enumerate(CLASSES):
            selected = labels == class_index
            figure.add_trace(
                go.Scattergeo(
                    lon=lon_lat[selected, 0],
                    lat=lon_lat[selected, 1],
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    marker={
                        "size": point_size,
                        "color": COLORS[class_index],
                        "opacity": 0.6,
                    },
                ),
                row=1,
                col=column,
            )
    # A dummy trace per class purely for the legend: a thick line swatch reads far
    # better as a small color key than the tiny marker dots the real traces use, so
    # the legend is drawn from these (invisible, no real lon/lat) instead.
    for class_index, name in enumerate(CLASSES):
        figure.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode="lines",
                name=name,
                legendgroup=name,
                showlegend=True,
                line={"width": 14, "color": COLORS[class_index]},
            ),
            row=1,
            col=1,
        )
    figure.update_geos(
        projection_type="hammer",
        showland=False,
        showocean=False,
        showcountries=False,
        showcoastlines=False,
        showlakes=False,
        showrivers=False,
        showframe=True,
        # Fixed full-sphere range: otherwise plotly fits the viewport to the data's
        # bounding box, so a tightly-clustered panel gets zoomed in and its oval
        # Hammer frame is cropped away -- both panels need the same fixed framing to
        # be visually comparable regardless of how spread out each one's data is.
        lonaxis={"showgrid": True, "gridcolor": "lightgray", "range": [-180, 180]},
        lataxis={"showgrid": True, "gridcolor": "lightgray", "range": [-90, 90]},
        bgcolor="rgba(0,0,0,0)",
    )
    figure.update_layout(
        template=TEMPLATE_NAME,
        paper_bgcolor="white",
        margin={"l": 10, "r": 10, "t": 35, "b": 45},
        legend={
            "title_text": "class",
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.02,
            "yanchor": "top",
            "itemwidth": 40,
        },
    )
    return figure


if __name__ == "__main__":
    main(sys.argv[1:])
