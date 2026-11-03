"""Plots exact vs fitted posteriors on S^2 for the d = 3 ring task (condition A).

Loads the stage-3 checkpoints written by ``apps.synthetic.ring_recovery stage3`` and
draws, for a few held-out test observations, the exact posterior density next to the
fitted vMF / PowerSpherical / free-TNBBeta posteriors in a Hammer projection whose
north pole is the task axis ``a`` (so the true posterior is a horizontal band).

Example:
    uv run --with matplotlib python -m apps.synthetic.ring_sphere_plot --seed 0 9

matplotlib is not a project dependency, hence ``--with``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib  # pyright: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  # pyright: ignore[reportMissingImports]
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from tnbbeta_vae.data.ring_posterior import RingPosteriorTask, ring_task  # noqa: E402
from tnbbeta_vae.models.mlp_vae import MlpVAE, MlpVAEConfig  # noqa: E402

FAMILIES = (
    ("vmf", "vMF"),
    ("power_spherical", "PowerSpherical"),
    ("tnbbeta", "TNBBeta (free)"),
)
SIGMA = 0.2
DIM = 3
TEST_SEED = 424242
NUM_TEST = 2000


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir", type=Path, default=Path("ring_recovery/l23_d3/stage3")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("writeup/results/figures"))
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--latitudes", type=float, nargs="+", default=[30.0, 60.0, 90.0]
    )
    args = parser.parse_args(argv)

    task = ring_task(DIM, SIGMA)
    generator = torch.Generator().manual_seed(TEST_SEED)
    test_x, _ = task.sample(NUM_TEST, generator)
    indices = _pick_indices(task, test_x, args.latitudes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seed:
        path = _plot_seed(args.ckpt_dir, args.out_dir, task, test_x, indices, seed)
        print(path)


def _pick_indices(
    task: RingPosteriorTask, x: Tensor, latitudes: list[float]
) -> list[int]:
    """Test x with exact posterior mean latitude closest to each target (degrees)."""
    theta = torch.linspace(0, math.pi, 2001, dtype=torch.float64)
    density = torch.exp(
        task.log_posterior_at_latitude(
            x.double()[:, None, :], torch.cos(theta)[None, :]
        )
    )
    density = density * torch.sin(theta)
    density = density / density.sum(-1, keepdim=True)
    mean = (density * theta).sum(-1) * 180 / math.pi
    return [int((mean - t).abs().argmin()) for t in latitudes]


def _load(ckpt_dir: Path, family: str, seed: int) -> MlpVAE:
    model = MlpVAE(
        MlpVAEConfig(
            family=family,  # pyright: ignore[reportArgumentType]
            input_dim=3,
            hidden_dims=[128, 128],
            latent_dim=DIM,
            likelihood_scale=SIGMA,
            num_elbo_samples=4,
        )
    )
    state = torch.load(
        ckpt_dir / "ckpt" / f"d3_true_{family}_s{seed}.pt", map_location="cpu"
    )
    state = {k: v for k, v in state.items() if not k.startswith("decoder.")}
    model.load_state_dict(state, strict=False)
    return model.eval()


def _frame(axis: Tensor) -> Tensor:
    """Orthonormal basis rows (e1, e2, a)."""
    helper = (
        torch.tensor([1.0, 0.0, 0.0])
        if abs(float(axis[0])) < 0.9
        else torch.tensor([0.0, 1.0, 0.0])
    )
    e1 = helper - (helper @ axis) * axis
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(axis, e1)
    return torch.stack([e1, e2, axis])


def _plot_seed(
    ckpt_dir: Path,
    out_dir: Path,
    task: RingPosteriorTask,
    test_x: Tensor,
    indices: list[int],
    seed: int,
) -> Path:
    frame = _frame(task.axis)
    lon = torch.linspace(-math.pi, math.pi, 361)
    lat = torch.linspace(-math.pi / 2, math.pi / 2, 181)
    lon_c, lat_c = (lon[:-1] + lon[1:]) / 2, (lat[:-1] + lat[1:]) / 2
    grid_lon, grid_lat = torch.meshgrid(lon_c, lat_c, indexing="xy")
    local = torch.stack(
        [
            torch.cos(grid_lat) * torch.cos(grid_lon),
            torch.cos(grid_lat) * torch.sin(grid_lon),
            torch.sin(grid_lat),
        ],
        dim=-1,
    )
    points = (local @ frame).reshape(-1, 1, DIM)  # (N, 1, 3) in ambient coordinates
    models = {fam: _load(ckpt_dir, fam, seed) for fam, _ in FAMILIES}
    summaries = {
        fam: json.loads((ckpt_dir / f"d3_true_{fam}_s{seed}.json").read_text())
        for fam, _ in FAMILIES
    }
    fig = plt.figure(figsize=(15, 3.6 * len(indices)))
    for row, index in enumerate(indices):
        x = test_x[index : index + 1]
        with torch.no_grad():
            exact = (
                task.log_posterior(x, points.squeeze(1).unsqueeze(1))
                .exp()
                .reshape(grid_lon.shape)
            )
        theta_mean = _mean_latitude(task, x)
        panels: list[tuple[str, Any, Any]] = [
            (f"Exact posterior\nmean lat {theta_mean:.0f} deg", exact, None)
        ]
        for fam, label in FAMILIES:
            with torch.no_grad():
                posterior, _ = models[fam].posterior_and_prior(x)
                density = posterior.log_prob(points).exp().reshape(grid_lon.shape)
                mu = getattr(posterior, "mean_direction", None)
                if mu is None:
                    mu = posterior.loc  # pyright: ignore[reportAttributeAccessIssue]
            kl = summaries[fam]["kl_per_x"][index]
            panels.append((f"{label}\nexact KL {kl:.2f} nats", density, mu.reshape(-1)))
        for col, (title, density, mu) in enumerate(panels):
            ax = fig.add_subplot(
                len(indices), 4, row * 4 + col + 1, projection="hammer"
            )
            values = density.numpy()
            ax.pcolormesh(
                grid_lon.numpy(),
                grid_lat.numpy(),
                values / values.max(),
                cmap="Blues",
                vmin=0,
                vmax=1,
                shading="auto",
            )
            if mu is not None:
                local_mu = frame @ mu
                ax.plot(
                    [math.atan2(float(local_mu[1]), float(local_mu[0]))],
                    [math.asin(max(-1.0, min(1.0, float(local_mu[2]))))],
                    marker="x",
                    color="#c0392b",
                    markersize=9,
                    mew=2,
                    label="mu",
                )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid(True, color="0.8", lw=0.4)
            ax.set_title(title, fontsize=10)
        fig.text(
            0.005,
            1 - (row + 0.5) / len(indices),
            f"test x #{index}",
            rotation=90,
            va="center",
            fontsize=10,
        )
    fig.suptitle(
        f"d = 3 ring task, true decoder, seed {seed}: posterior density on S^2 "
        "(Hammer, north pole = axis a; "
        "each panel scaled to its own max; red x = fitted mean direction)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    path = out_dir / f"ring_d3_sphere_seed{seed}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _mean_latitude(task: RingPosteriorTask, x: Tensor) -> float:
    theta = torch.linspace(0, math.pi, 2001, dtype=torch.float64)
    density = torch.exp(
        task.log_posterior_at_latitude(x.double(), torch.cos(theta)[:, None])[:, 0]
    )
    density = density * torch.sin(theta)
    density = density / density.sum()
    return float((density * theta).sum()) * 180 / math.pi


if __name__ == "__main__":
    main()
