"""Recover an axis from noisy R^100 embeddings with a built-in sign ambiguity.

Usage:
    uv run python -m apps.synthetic.axial_recovery [--out-dir DIR] \
        [--epochs 200] [--seed 0] \
        [--families gaussian vmf power_spherical tnbbeta]

A sibling of ``apps.synthetic.circle_recovery``: trains an MLP VAE per latent
family (``latent_dim=2``) on ``tnbbeta_vae.data.axial_mixture``'s axial von
Mises mixture, where the observation cannot distinguish an angle ``phi``
from ``phi + pi`` by construction (see that module's docstring and
``docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md``).
Reports, for each family:

* ``angle_error``/``angle_error_sample``: as in ``circle_recovery``, but
  wrapped mod ``pi`` (not mod ``2*pi``) -- the recoverable signal here is
  the axis, not a signed angle. Small means the axis was recovered;
  ``~pi/4`` means it was not.
* ``reconstruction_angle_error``: as in ``circle_recovery``, mod ``pi``.
* ``prior_manifold_ratio``, ``test_ll``, ``test_kl``: unchanged from
  ``circle_recovery``.
* TNBBeta only: ``p_mean``, ``p_std``, ``centre_axis_resultant`` (as in
  ``circle_recovery``) plus ``m_mean = epsilon_mean - (latent_dim - 1) / 2``
  -- negative is the exact condition (proved in
  ``writeup/weekly_markdown_summaries/week_2/
  tnbbeta_vs_power_spherical_expressivity.md``) for a bimodal posterior at
  ``{mean_direction, -mean_direction}``, which is what this task's
  Bayes-optimal posterior actually looks like.

Writes ``axial_recovery.json`` and ``axial_recovery.html`` (latent scatter
plots coloured by the true axial component) into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from tnbbeta_vae.data.axial_mixture import axial_mixture_data
from tnbbeta_vae.models import MlpVAE, MlpVAEConfig
from tnbbeta_vae.models.losses import importance_weighted_metrics
from tnbbeta_vae.training import Trainer

_TITLES = {
    "gaussian": "N-VAE",
    "vmf": "S-VAE (vMF)",
    "power_spherical": "Power Spherical",
    "tnbbeta": "TNBBeta",
}


def main(argv: list[str] | None = None) -> None:
    """Runs the experiment and writes the JSON and HTML outputs.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("axial_recovery"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-train", type=int, default=10_000)
    parser.add_argument("--num-test", type=int, default=2_000)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["gaussian", "vmf", "power_spherical", "tnbbeta"],
        choices=list(_TITLES),
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = axial_mixture_data(seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_x, _, _ = data.sample(args.num_train, generator)
    test_x, test_angle, test_component = data.sample(args.num_test, generator)
    manifold_angles = torch.linspace(-math.pi, math.pi, 4000)
    manifold = data.embed(manifold_angles)
    batches = list(train_x.split(args.batch_size))

    results: dict[str, Any] = {}
    latents: dict[str, dict[str, torch.Tensor]] = {}
    for family in args.families:
        torch.manual_seed(args.seed)
        model = MlpVAE(MlpVAEConfig(family=cast("Any", family), latent_dim=2))
        trainer = Trainer(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            "mlp_vae",
            model.config,
            runs_dir=args.out_dir / "runs",
            run_id=family,
        )
        trainer.fit(batches, args.epochs, kl_warmup_epochs=args.kl_warmup_epochs)
        model.eval()
        centre = _centre(model, test_x, family)
        sample = _sample(model, test_x)
        latents[family] = {"centre": centre, "sample": sample}
        metrics = importance_weighted_metrics(model, test_x, num_samples=500)
        results[family] = {
            "angle_error": axial_angle_error(centre, test_angle),
            "angle_error_sample": axial_angle_error(sample, test_angle),
            "reconstruction_angle_error": _reconstruction_angle_error(
                model, sample, manifold, manifold_angles, test_angle
            ),
            "prior_manifold_ratio": _prior_manifold_ratio(model, manifold, test_x),
            "test_ll": metrics["ll"].mean().item(),
            "test_kl": metrics["kl"].mean().item(),
            "learned_sigma": model.learned_scale().item(),
        }
        if family == "tnbbeta":
            results[family].update(_tnbbeta_diagnostics(model, test_x, centre))
        print(family, json.dumps(results[family]))

    (args.out_dir / "axial_recovery.json").write_text(json.dumps(results, indent=2))
    _write_figure(latents, test_angle, test_component, args.out_dir)
    print(f"Wrote {args.out_dir / 'axial_recovery.json'}")


def axial_angle_error(latent: torch.Tensor, true_angle: torch.Tensor) -> float:
    """Mean absolute angular error mod pi, after the best rotation and reflection.

    Like ``circle_recovery.angle_error``, but wraps to ``(-pi/2, pi/2]``
    instead of ``(-pi, pi]`` -- the recoverable signal here is the axis
    (angle mod pi), not the signed angle, since ``phi`` and ``phi + pi`` are
    equally valid. The best rotation offset is found via the standard
    axial-mean trick (double the residual, take its circular mean, halve),
    the same doubling idea the dataset itself uses to erase the sign.

    Args:
        latent: Points in the plane, shape ``(n, 2)``; only their angle is
            used.
        true_angle: The true angles, shape ``(n,)``.

    Returns:
        The smallest mean absolute circular difference mod pi (radians)
        over a reflection and the best rotation.
    """
    estimate = torch.atan2(latent[:, 1], latent[:, 0])
    best = math.inf
    for sign in (1.0, -1.0):
        residual = true_angle - sign * estimate
        doubled = 2 * residual
        offset = torch.atan2(doubled.sin().mean(), doubled.cos().mean()) / 2
        wrapped = (residual - offset + math.pi / 2) % math.pi - math.pi / 2
        best = min(best, wrapped.abs().mean().item())
    return best


@torch.no_grad()
def _centre(model: MlpVAE, x: torch.Tensor, family: str) -> torch.Tensor:
    posterior, _ = model.posterior_and_prior(x)
    if family == "gaussian":
        return posterior.base_dist.loc  # pyright: ignore[reportAttributeAccessIssue]
    if family == "vmf":
        return posterior.loc  # pyright: ignore[reportAttributeAccessIssue]
    if family == "power_spherical":
        return posterior.mean_direction  # pyright: ignore[reportAttributeAccessIssue]
    direction = posterior.mean_direction  # pyright: ignore[reportAttributeAccessIssue]
    return torch.where((posterior.p > 0.5)[:, None], direction, -direction)  # pyright: ignore[reportAttributeAccessIssue]


@torch.no_grad()
def _tnbbeta_diagnostics(
    model: MlpVAE, x: torch.Tensor, centre: torch.Tensor
) -> dict[str, float]:
    """How TNBBeta encodes the axis: p/epsilon and the bimodality indicator.

    ``m_mean`` is ``epsilon_mean - (latent_dim - 1) / 2``; negative is the
    exact bimodality condition (expressivity write-up, Theorem
    5.1/Corollary 5.3).
    """
    posterior, _ = model.posterior_and_prior(x)
    p = posterior.p  # pyright: ignore[reportAttributeAccessIssue]
    epsilon = posterior.epsilon  # pyright: ignore[reportAttributeAccessIssue]
    doubled = 2 * torch.atan2(centre[:, 1], centre[:, 0])
    latent_dim = model.config.latent_dim
    return {
        "p_mean": p.mean().item(),
        "p_std": p.std().item(),
        "centre_axis_resultant": torch.hypot(
            doubled.cos().mean(), doubled.sin().mean()
        ).item(),
        "m_mean": epsilon.mean().item() - (latent_dim - 1) / 2,
    }


@torch.no_grad()
def _reconstruction_angle_error(
    model: MlpVAE,
    sample: torch.Tensor,
    manifold: torch.Tensor,
    manifold_angles: torch.Tensor,
    true_angle: torch.Tensor,
) -> float:
    nearest = torch.cdist(model.decoder(sample), manifold).argmin(dim=1)
    difference = manifold_angles[nearest] - true_angle
    return ((difference + math.pi / 2) % math.pi - math.pi / 2).abs().mean().item()


@torch.no_grad()
def _sample(model: MlpVAE, x: torch.Tensor) -> torch.Tensor:
    posterior, _ = model.posterior_and_prior(x)
    return posterior.sample()


@torch.no_grad()
def _prior_manifold_ratio(
    model: MlpVAE, manifold: torch.Tensor, real: torch.Tensor, num: int = 2000
) -> float:
    generated = model.generate(num)
    generated_distance = torch.cdist(generated, manifold).min(dim=1).values.mean()
    real_distance = torch.cdist(real, manifold).min(dim=1).values.mean()
    return (generated_distance / real_distance).item()


def _write_figure(
    latents: dict[str, dict[str, torch.Tensor]],
    angle: torch.Tensor,
    component: torch.Tensor,
    out_dir: Path,
) -> None:
    """Scatter plots of the posterior centres (top) and samples (bottom)."""
    figure = make_subplots(
        rows=2, cols=len(latents), subplot_titles=[_TITLES[f] for f in latents]
    )
    marker = {
        "size": 4,
        "color": np.asarray(component),
        "colorscale": "Turbo",
        "opacity": 0.7,
    }
    for column, points in enumerate(latents.values(), start=1):
        for row, kind in enumerate(("centre", "sample"), start=1):
            array = points[kind].numpy()
            figure.add_trace(
                go.Scatter(
                    x=array[:, 0],
                    y=array[:, 1],
                    mode="markers",
                    marker=marker,
                    text=[f"angle {a:.2f}" for a in angle.numpy()],
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
            index = (row - 1) * len(latents) + column
            axis = "y" if index == 1 else f"y{index}"
            figure.update_xaxes(scaleanchor=axis, row=row, col=column)
    figure.update_layout(
        title="Latent space by true axial component (top: centres, bottom: samples)"
    )
    figure.write_html(out_dir / "axial_recovery.html", include_plotlyjs=True)


if __name__ == "__main__":
    main(sys.argv[1:])
