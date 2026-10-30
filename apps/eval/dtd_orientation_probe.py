"""DTD oriented textures: Table-1 metrics plus (TNBBeta only) a bimodality probe.

Usage:
    uv run python -m apps.eval.dtd_orientation_probe --run-name NAME \
        [--checkpoint final] [--split test]

DTD has no labeled orientation, so ground truth is derived, for evaluation only
(never for training): each test image's dominant line orientation and a
coherence score (how strongly oriented it actually is) come from the structure
tensor (``tnbbeta_vae.data.orientation``), not a label.

Writes ``dtd_orientation_probe_<checkpoint>.json`` (Table-1-style
``test_ll``/``test_elbo``/``test_kl``, via the shared
``importance_weighted_metrics``, plus TNBBeta-only p/q/m summary statistics)
and ``dtd_orientation_probe_<checkpoint>.html`` next to the checkpoint. For a
``conv_tnbbeta_spherical_vae`` run, the HTML also scatters ``p``/``q`` and
``m = epsilon - (latent_dim - 1) / 2`` against each image's coherence -- the
design's prediction is that posteriors go bimodal (``m < 0``, ``p`` near
``0.5``) specifically on strongly-oriented images. Non-TNBBeta runs still get
the Table-1 comparison and a plain coherence histogram, just without the p/q/m
part (no such parameters).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader

from tnbbeta_vae.data.dtd import load_dtd
from tnbbeta_vae.data.orientation import structure_tensor_orientation
from tnbbeta_vae.models.losses import importance_weighted_metrics
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.plotting import TEMPLATE_NAME
from tnbbeta_vae.training import load_model_checkpoint

_TNBBETA_MODEL = "conv_tnbbeta_spherical_vae"


def main(argv: list[str] | None = None) -> None:
    """Computes Table-1 metrics and the coherence-vs-posterior probe for one run.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Importance samples per image for the Table-1 metrics.",
    )
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_dir = checkpoint_dir() / args.run_name
    model, checkpoint = load_model_checkpoint(run_dir / f"{args.checkpoint}.pt", device)
    kind = checkpoint["model_name"]
    is_tnbbeta = kind == _TNBBETA_MODEL

    test_set = load_dtd(
        data_dir(), split=args.split, image_size=checkpoint["config"]["image_size"]
    )
    ll, elbo, kl, coherence, latitude = _evaluate(
        model, test_set, args, device, is_tnbbeta, checkpoint["config"]["latent_dim"]
    )

    results: dict[str, Any] = {
        "run_name": args.run_name,
        "model_name": kind,
        "split": args.split,
        "num_images": len(ll),
        "test_ll": float(ll.mean()),
        "test_elbo": float(elbo.mean()),
        "test_kl": float(kl.mean()),
        "mean_coherence": float(coherence.mean()),
    }
    if latitude is not None:
        p, q, m = (part.numpy() for part in latitude)
        coherence_np = coherence.numpy()
        results["tnbbeta"] = {
            "p_mean": float(p.mean()),
            "p_std": float(p.std()),
            "q_mean": float(q.mean()),
            "q_std": float(q.std()),
            "m_mean": float(m.mean()),
            "m_std": float(m.std()),
            "coherence_m_correlation": _safe_corr(coherence_np, m),
            "coherence_p_distance_correlation": _safe_corr(
                coherence_np, np.abs(p - 0.5)
            ),
        }

    output = run_dir / f"dtd_orientation_probe_{args.checkpoint}.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")

    figure = _figure(coherence, latitude, args.run_name, args.max_points, args.seed)
    html = run_dir / f"dtd_orientation_probe_{args.checkpoint}.html"
    figure.write_html(html, include_plotlyjs=True)
    print(f"Wrote {html}")


@torch.no_grad()
def _evaluate(
    model: Any,
    dataset: Any,
    args: argparse.Namespace,
    device: torch.device,
    is_tnbbeta: bool,
    latent_dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
]:
    """Runs Table-1 metrics, orientation coherence, and (TNBBeta) p/q/m over a split."""
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    ll_parts, elbo_parts, kl_parts, coherence_parts = [], [], [], []
    p_parts, q_parts, m_parts = [], [], []
    for batch in loader:
        x = batch.to(device)
        metrics = importance_weighted_metrics(model, x, num_samples=args.num_samples)
        ll_parts.append(metrics["ll"].cpu())
        elbo_parts.append(metrics["elbo"].cpu())
        kl_parts.append(metrics["kl"].cpu())
        _, batch_coherence = structure_tensor_orientation(batch)
        coherence_parts.append(batch_coherence)
        if is_tnbbeta:
            posterior, _ = model.posterior_and_prior(x)
            p_parts.append(posterior.p.cpu())
            q_parts.append(posterior.q.cpu())
            m_parts.append(posterior.epsilon.cpu() - (latent_dim - 1) / 2)

    ll = torch.cat(ll_parts)
    elbo = torch.cat(elbo_parts)
    kl = torch.cat(kl_parts)
    coherence = torch.cat(coherence_parts)
    if not is_tnbbeta:
        return ll, elbo, kl, coherence, None
    latitude = (torch.cat(p_parts), torch.cat(q_parts), torch.cat(m_parts))
    return ll, elbo, kl, coherence, latitude


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, or ``nan`` if either input is degenerate (zero variance)."""
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _figure(
    coherence: torch.Tensor,
    latitude: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    title: str,
    max_points: int,
    seed: int,
) -> go.Figure:
    """A p-vs-q and m-vs-coherence scatter (TNBBeta), or a coherence histogram."""
    keep = np.arange(len(coherence))
    if len(keep) > max_points:
        keep = np.sort(
            np.random.default_rng(seed).choice(keep, max_points, replace=False)
        )
    coherence_np = coherence.numpy()[keep]

    if latitude is None:
        figure = go.Figure(go.Histogram(x=coherence_np))
        figure.update_layout(
            template=TEMPLATE_NAME,
            title=f"{title}: structure-tensor coherence (no TNBBeta p/q/m)",
            xaxis={"title": "coherence"},
        )
        return figure

    p, q, m = (part.numpy()[keep] for part in latitude)
    marker = {
        "size": 4,
        "color": coherence_np,
        "colorscale": "Viridis",
        "opacity": 0.6,
    }
    figure = make_subplots(rows=1, cols=2, subplot_titles=["p vs q", "m vs coherence"])
    figure.add_trace(
        go.Scatter(
            x=p,
            y=q,
            mode="markers",
            marker={**marker, "colorbar": {"title": "coherence"}},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=coherence_np, y=m, mode="markers", marker=marker, showlegend=False
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="p", range=[0, 1], row=1, col=1)
    figure.update_yaxes(title_text="q", range=[0, 1], row=1, col=1)
    figure.update_xaxes(title_text="coherence", row=1, col=2)
    figure.update_yaxes(title_text="m = epsilon - (latent_dim - 1) / 2", row=1, col=2)
    figure.update_layout(
        template=TEMPLATE_NAME, title=f"{title}: p/q/m vs structure-tensor coherence"
    )
    return figure


if __name__ == "__main__":
    main(sys.argv[1:])
