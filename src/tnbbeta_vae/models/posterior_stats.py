"""Dataset-agnostic posterior-shape statistics for any sphere-latent GraphVAE run.

Originally written inside ``apps/eval/dblp_bridge_diagnostic.py`` (where it still
takes a bridge/non-bridge mask), this is the whole-graph-agnostic core: entropy and
per-node concentration (``r_bar``) for any family, plus TNBBeta's own ``p``/``q``/
``epsilon``/``m`` marginals and the fraction of nodes past the proven bimodal
threshold (``tnbbeta_vs_power_spherical_expressivity.md``, Theorem 5.1/Corollary
5.3). Factored out so ``apps/eval/graph_posterior_shape.py`` -- which has no
community structure to split on, since it also runs on plain Planetoid graphs -- can
reuse it on the whole node set instead of duplicating it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.distributions import Distribution

__all__ = ["posterior_stats"]


def posterior_stats(
    posterior: Distribution, mask: torch.Tensor, config: dict[str, Any]
) -> dict[str, float]:
    """Computes posterior statistics for a subset of nodes.

    Args:
        posterior: Posterior distribution over all nodes.
        mask: Boolean tensor of shape (num_nodes,) selecting the subset.
        config: Model config dict (for family and latent_dim).

    Returns:
        Dict with entropy, concentration, and (TNBBeta only) parameter marginals and
        the fraction of nodes in the proven bimodal regime.
    """
    # Entropy (per-node). posterior already has batch_shape == (num_nodes,) with the
    # whole latent vector as a single event (not num_nodes independent per-dimension
    # events), so wrapping in Independent(posterior, 1) -- as a previous version of
    # this function did -- reinterprets that batch dimension into the event instead,
    # collapsing entropy() to a 0-d scalar and crashing on entropy[mask] ("too many
    # indices for tensor of dimension 0"). TNBBetaSpherical has no closed-form entropy
    # (NotImplementedError, caught below) and masked this for TNBBeta; vMF and Power
    # Spherical both implement entropy() and hit it directly.
    try:
        entropy = posterior.entropy()  # type: ignore[union-attr]
        entropy = entropy[mask].mean().item()
    except (NotImplementedError, AttributeError):
        entropy = float("nan")

    # For sphere models: concentration via r-bar (resultant length), computed PER NODE
    # -- each node's own sampled mean direction is normed first, and those norms are
    # then averaged across the subset. A previous version averaged sample means across
    # nodes before taking one norm, which measures how aligned the subset's posterior
    # mean directions are WITH EACH OTHER (a cross-node alignment statistic), not each
    # node's own posterior concentration -- the two coincide only if every node in the
    # subset already shares close to the same mean direction, which a bridge/non-bridge
    # split of a real graph has no reason to.
    r_bar = float("nan")
    try:
        with torch.no_grad():
            # torch.Size, not a plain tuple: VonMisesFisher.rsample (a stable,
            # restored baseline -- see CLAUDE.md -- not touched here) only accepts
            # torch.Size or a bare int, and silently does the wrong thing with any
            # other iterable (wraps the whole tuple as one non-int "size", raising
            # TypeError at torch.Size construction). TNBBetaSpherical/PowerSpherical
            # accept a plain tuple too, so torch.Size works uniformly across all
            # three families.
            samples = posterior.rsample(torch.Size([100]))  # type: ignore
            if len(samples.shape) > 2:
                # samples shape: (n_samples, batch, latent_dim)
                subset_samples = samples[:, mask, :]  # (n_samples, num_selected, dim)
                per_node_mean = subset_samples.mean(dim=0)  # (num_selected, dim)
                per_node_r_bar = torch.linalg.norm(per_node_mean, dim=-1)
                r_bar = float(per_node_r_bar.mean().item())
    except (AttributeError, RuntimeError):
        pass

    stats: dict[str, float] = {
        "entropy_mean": entropy,
        "r_bar": r_bar,
        "num_nodes": int(mask.sum().item()),
    }

    # TNBBeta-specific: p, q, epsilon marginals, and the fraction of nodes whose
    # epsilon has crossed into the proven bimodal regime
    # (tnbbeta_vs_power_spherical_expressivity.md's Theorem 5.1/Corollary 5.3:
    # m = epsilon - (latent_dim - 1) / 2 < 0 is necessary and sufficient for a bimodal
    # posterior) -- the actual mechanism TNBBeta has that vMF/Power Spherical are
    # structurally incapable of, and the direct test of whether a subset of nodes is
    # actually using it.
    if config.get("family") == "tnbbeta" and hasattr(posterior, "epsilon"):
        p = posterior.p[mask]  # type: ignore[attr-defined]
        q = posterior.q[mask]  # type: ignore[attr-defined]
        epsilon = posterior.epsilon[mask]  # type: ignore[attr-defined]
        latent_dim = config["latent_dim"]
        m = epsilon - (latent_dim - 1) / 2
        stats.update(
            {
                "p_mean": float(p.mean().item()),
                "p_std": float(p.std().item()),
                "q_mean": float(q.mean().item()),
                "q_std": float(q.std().item()),
                "epsilon_mean": float(epsilon.mean().item()),
                "epsilon_std": float(epsilon.std().item()),
                "m_mean": float(m.mean().item()),
                "frac_bimodal": float((m < 0).float().mean().item()),
            }
        )

    return stats
