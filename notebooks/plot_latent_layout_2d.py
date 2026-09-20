"""Renders class layouts from a 2-D latent export (apps/eval/export_latents.py).

Usage:
    uv run python notebooks/plot_latent_layout_2d.py --npz latents_final_test.npz \
        --out-dir plots_out

Pure NumPy (no plotting dependency). Class colors, in CIFAR-10 label order:
0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse,
8 ship, 9 truck. The legend image shows the color of each class as a row of
swatches in that order.

For latent_dim=3 use notebooks/plot_sphere_3d.py (interactive).

Gaussian (plane): scatter of the posterior means and of posterior samples.
TNBBeta (circle): points on the unit circle at the angle of the mean
direction and of a posterior sample (with a random radial jitter so points
do not overprint), class-by-angle heatmaps, and the histogram of the angle
between a sample and its mean direction (a two-point "ring" in 2-D).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import zlib

import numpy as np

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
COLORS = np.array(
    [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
        (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    ],
    dtype=np.float64,
)  # fmt: skip
SIZE = 720


def write_png(path: Path, image: np.ndarray) -> None:
    """Writes an (H, W, 3) uint8 array as a PNG."""
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    )  # fmt: skip


def scatter(xy: np.ndarray, labels: np.ndarray, extent: float, radius: int = 2) -> np.ndarray:
    """Alpha-blended scatter of points in [-extent, extent]^2, colored by class."""
    canvas = np.full((SIZE, SIZE, 3), 255.0)
    scale = (SIZE - 1) / (2 * extent)
    cols = np.clip(((xy[:, 0] + extent) * scale).astype(int), 0, SIZE - 1)
    rows = np.clip(((extent - xy[:, 1]) * scale).astype(int), 0, SIZE - 1)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r = np.clip(rows + dr, 0, SIZE - 1)
            c = np.clip(cols + dc, 0, SIZE - 1)
            canvas[r, c] = 0.45 * canvas[r, c] + 0.55 * COLORS[labels]
    return canvas.astype(np.uint8)


def circle_points(angle: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Points at the given angles on a ring of radius ~1 with random radial jitter."""
    radius = rng.uniform(0.8, 1.2, size=angle.shape)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


def heatmap(angle: np.ndarray, labels: np.ndarray, bins: int = 72) -> np.ndarray:
    """Class-by-angle density (rows: classes, normalized per row), upscaled."""
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    grid = np.stack([np.histogram(angle[labels == k], edges)[0] for k in range(10)]).astype(float)
    grid /= grid.max(axis=1, keepdims=True).clip(1)
    ramp = np.array([[0.98, 0.98, 0.98], [0.99, 0.87, 0.3], [0.85, 0.2, 0.15], [0.25, 0.0, 0.1]])
    positions = np.linspace(0, 1, len(ramp))
    rgb = np.stack([np.interp(grid, positions, ramp[:, ch]) for ch in range(3)], axis=-1)
    image = np.repeat(np.repeat(rgb, 36, axis=0), 10, axis=1)
    for k in range(10):  # class color strip on the left edge
        image[k * 36 : (k + 1) * 36, :14] = COLORS[k] / 255.0
    return (image * 255).astype(np.uint8)


def legend() -> np.ndarray:
    """A row of 10 class-color swatches in label order."""
    return np.repeat(COLORS[None, :, :].astype(np.uint8), 60, axis=1).repeat(40, axis=0)


def wrap(angle: np.ndarray) -> np.ndarray:
    """Wraps angles to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def resultant_length(angle: np.ndarray) -> float:
    """Mean resultant length of a set of angles (1 = all identical, 0 = uniform)."""
    return float(np.hypot(np.cos(angle).mean(), np.sin(angle).mean()))


def main() -> None:
    """Loads an export and writes the layout images and a text summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    data = np.load(args.npz)
    kind = str(data["model_name"])
    labels = data["labels"]
    tag = args.tag or args.npz.parent.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    dim = data["direction"].shape[1]
    if dim != 2:
        raise SystemExit("This script is for latent_dim 2; see plot_sphere_3d.py for 3.")

    write_png(args.out_dir / "legend.png", legend())
    print("class order:", ", ".join(f"{k}={name}" for k, name in enumerate(CLASSES)))
    print(f"{tag}: kNN class accuracy on direction/mean = "
          f"see latent_probe_*.json; KL mean = {data['kl'].mean():.2f} nats")  # fmt: skip

    if kind == "conv_gaussian_vae":
        mu, z = data["direction"], data["z"]
        extent = float(np.percentile(np.abs(np.concatenate([mu, z])), 99)) * 1.05
        write_png(args.out_dir / f"{tag}_mu.png", scatter(mu, labels, extent))
        write_png(args.out_dir / f"{tag}_z.png", scatter(z, labels, extent))
        print("per-class mean of mu (x, y) and mean posterior std:")
        for k in range(10):
            m = mu[labels == k]
            print(f"  {k} {CLASSES[k]:10s} centroid=({m[:, 0].mean():6.2f},{m[:, 1].mean():6.2f}) "
                  f"spread={m.std(axis=0).mean():.2f} sigma={data['concentration'][labels == k].mean():.3f}")  # fmt: skip
        return

    phi_mu = np.arctan2(data["direction"][:, 1], data["direction"][:, 0])
    phi_z = np.arctan2(data["z"][:, 1], data["z"][:, 0])
    delta = wrap(phi_z - phi_mu)
    write_png(args.out_dir / f"{tag}_mu_circle.png", scatter(circle_points(phi_mu, rng), labels, 1.4))
    write_png(args.out_dir / f"{tag}_z_circle.png", scatter(circle_points(phi_z, rng), labels, 1.4))
    write_png(args.out_dir / f"{tag}_mu_heatmap.png", heatmap(phi_mu, labels))
    write_png(args.out_dir / f"{tag}_z_heatmap.png", heatmap(phi_z, labels))
    hist = np.histogram(np.abs(delta), bins=12, range=(0, np.pi))[0]
    print("|angle between z and its mean direction| histogram (12 bins over 0..pi):", hist.tolist())
    print("class  name        mu-angle R   mean p   mean q  mean eps  mean KL")
    for k in range(10):
        sel = labels == k
        print(f"  {k}    {CLASSES[k]:10s} {resultant_length(phi_mu[sel]):9.3f}   {data['p'][sel].mean():6.3f}  "
              f"{data['q'][sel].mean():6.3f}  {data['epsilon'][sel].mean():8.2f}  {data['kl'][sel].mean():7.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
