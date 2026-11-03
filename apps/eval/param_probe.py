"""Probes what each posterior family's per-node parameters communicate, on GraphVAEs.

Usage:
    uv run python -m apps.eval.param_probe --dataset cora [--seeds 0 1 2 3 4]

For each checkpoint of ``fixed_m_sweep_graphs.sbatch`` (TNBBeta, free epsilon and
``m = 0.5 * threshold``) and ``fixed_m_baselines_graphs.sbatch`` (vMF, power-spherical,
Gaussian) -- all featureless, at TNBBeta's dimension -- this extracts every node's
posterior parameters from a forward pass over the *training* adjacency and reports:

1. Parameter-to-covariate map. Spearman correlation of each parameter with log training
   degree, local clustering coefficient and (``mag_*`` only) the number of distinct
   neighbor communities and neighbor-community entropy; the same correlations
   *partialling out* log degree; the cross-validated ``R^2`` of predicting each
   covariate from the parameters, and how much that ``R^2`` improves on degree alone;
   and the parameters' effective rank (participation ratio of their covariance
   spectrum), i.e. how many independent scalars they amount to.
2. Decoder-visible noise. Test AUC scoring each held-out edge from the posterior
   *centres* (what ``run_once`` reports) versus from posterior *samples*: the mean over
   ``S`` draws of ``sigmoid(link_logit(z_i, z_j))``, for ``S = 1`` and ``S = 16``.
   A family whose samples carry little node information has a sampled AUC far below
   its centre AUC. ``r_bar`` (norm of the mean of 32 samples) is reported for the
   sphere families; exactly uniform samples give about ``1 / sqrt(32) = 0.177``.

Reads ``$TNBBETA_CHECKPOINT_DIR/<run>_seed<seed>/final.pt`` and writes
``param_probe.json`` beside each.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata, spearmanr
import torch

from apps.eval.bucketed_link_prediction import _load
from apps.eval.graph_posterior_shape import graph_to_batch
from apps.link_prediction.main import score_edges
from tnbbeta_vae.data.neighbor_heterogeneity import (
    neighbor_community_count,
    neighbor_heterogeneity,
)
from tnbbeta_vae.data.planetoid import normalized_adjacency, split_edges
from tnbbeta_vae.models.losses.ranking import roc_auc
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import load_model_checkpoint, select_device

_R_BAR_SAMPLES = 32
_MC_SAMPLES = (1, 16)
_CV_FOLDS = 5


def main(argv: list[str] | None = None) -> None:
    """Runs the probes for every condition x seed of one dataset.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    device = select_device(args.device)
    graph, communities = _load(args.dataset, True)
    for seed in args.seeds:
        split = split_edges(graph.adjacency, seed=seed)
        covariates = _covariates(split.train_adjacency, communities)
        batch = graph_to_batch(graph, device)
        batch = dataclasses.replace(
            batch,
            norm_adjacency=normalized_adjacency(split.train_adjacency).to(device),
        )
        for run in _runs(args.dataset):
            run_dir = checkpoint_dir() / f"{run}_seed{seed}"
            if not (run_dir / "final.pt").exists():
                print(f"skipping {run_dir}: no checkpoint")
                continue
            torch.manual_seed(seed)
            model, checkpoint = load_model_checkpoint(run_dir / "final.pt", device)
            model.eval()
            with torch.no_grad():
                posterior, _ = model.posterior_and_prior(batch)  # type: ignore[attr-defined]
                centres = model.embeddings(batch).cpu()  # type: ignore[attr-defined]
                parameters, r_bar = _node_parameters(
                    posterior, checkpoint["config"]["family"]
                )
                result = {
                    "run": run,
                    "seed": seed,
                    "family": checkpoint["config"]["family"],
                    "r_bar": r_bar,
                    "parameter_map": parameter_map(parameters, covariates),
                    "decoding": _decoding(
                        model, posterior, centres, split, device, batch.num_nodes
                    ),
                }
            (run_dir / "param_probe.json").write_text(json.dumps(result, indent=2))
            print(f"Wrote {run_dir / 'param_probe.json'}")


def _runs(dataset: str) -> list[str]:
    """Run names (minus ``_seed<seed>``) of one dataset's five compared conditions."""
    return [
        f"fixedmg_{dataset}_free",
        f"fixedmg_{dataset}_0.5",
        f"fixedmgbase_{dataset}_vmf",
        f"fixedmgbase_{dataset}_power_spherical",
        f"fixedmgbase_{dataset}_gaussian",
    ]


def parameter_map(
    parameters: dict[str, npt.NDArray[np.float64]],
    covariates: dict[str, npt.NDArray[np.float64]],
) -> dict[str, Any]:
    """Relates per-node parameters to per-node covariates.

    Args:
        parameters: Parameter name -> value per node. Constant parameters (a fixed
            epsilon) are dropped.
        covariates: Covariate name -> value per node; must contain ``"log_degree"``.

    Returns:
        ``spearman`` and ``partial_spearman`` (given log degree) per covariate and
        parameter, ``r2`` (cross-validated, from all parameters) and ``r2_over_degree``
        (the gain over degree alone) per covariate, and ``effective_rank``.
    """
    parameters = {k: v for k, v in parameters.items() if v.std() > 1e-12}
    names = list(parameters)
    result: dict[str, Any] = {
        "parameters": names,
        "means": {name: float(parameters[name].mean()) for name in names},
        "spearman": {},
        "r2": {},
    }
    result["partial_spearman"] = {}
    result["r2_over_degree"] = {}
    if not names:
        return result
    matrix = np.column_stack([parameters[k] for k in names])
    degree = covariates["log_degree"]
    for covariate, values in covariates.items():
        result["spearman"][covariate] = {
            name: _spearman(parameters[name], values) for name in names
        }
        result["r2"][covariate] = cv_r2(matrix, values)
        if covariate != "log_degree":
            result["partial_spearman"][covariate] = {
                name: partial_spearman(parameters[name], values, degree)
                for name in names
            }
            base = cv_r2(degree[:, None], values)
            full = cv_r2(np.column_stack([degree, matrix]), values)
            result["r2_over_degree"][covariate] = {"degree_only": base, "plus": full}
    if len(names) > 1:
        result["effective_rank"] = effective_rank(matrix)
    return result


def partial_spearman(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    control: npt.NDArray[np.float64],
) -> float:
    """Spearman correlation of ``a`` and ``b`` after regressing out ``control``'s rank.

    Args:
        a: First variable.
        b: Second variable.
        control: Variable to partial out.

    Returns:
        The correlation of the rank residuals (``nan`` if either is constant).
    """
    ranks = [rankdata(v) for v in (a, b, control)]
    design = np.column_stack([ranks[2], np.ones(len(a))])
    residuals = [
        r - design @ np.linalg.lstsq(design, r, rcond=None)[0] for r in ranks[:2]
    ]
    if residuals[0].std() < 1e-12 or residuals[1].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(residuals[0], residuals[1])[0, 1])


def cv_r2(features: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> float:
    """Cross-validated ``R^2`` of ridge regression on rank-transformed variables.

    Args:
        features: Shape ``(n, k)``.
        target: Shape ``(n,)``.

    Returns:
        ``1 - SSE / SST`` pooled over ``_CV_FOLDS`` folds (can be negative).
    """
    n = len(target)
    x = np.column_stack([rankdata(c) / n for c in features.T])
    y = rankdata(target) / n
    order = np.random.default_rng(0).permutation(n)
    sse = sst = 0.0
    for fold in np.array_split(order, _CV_FOLDS):
        train = np.setdiff1d(order, fold)
        a = np.c_[x[train], np.ones(len(train))]
        weights = np.linalg.solve(a.T @ a + np.eye(a.shape[1]), a.T @ y[train])
        prediction = np.c_[x[fold], np.ones(len(fold))] @ weights
        sse += float(((y[fold] - prediction) ** 2).sum())
        sst += float(((y[fold] - y[train].mean()) ** 2).sum())
    return 1.0 - sse / sst


def effective_rank(features: npt.NDArray[np.float64]) -> float:
    """Participation ratio of the standardized covariance spectrum.

    That is ``(sum l)^2 / sum l^2`` over the eigenvalues ``l``.

    1.0 means the columns are one scalar in disguise; ``k`` means ``k`` independent
    ones.
    """
    standardized = (features - features.mean(0)) / (features.std(0) + 1e-12)
    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(standardized.T)), 0, None)
    return float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum())


def clustering_coefficient(adjacency: Any) -> npt.NDArray[np.float64]:
    """Each node's local clustering coefficient (0 for degree < 2).

    Args:
        adjacency: Symmetric binary adjacency, any scipy.sparse format.

    Returns:
        Triangles through each node over the possible ``deg * (deg - 1) / 2``.
    """
    binary = (adjacency.tocsr() > 0).astype(np.float64)
    triangles = np.asarray(binary.multiply(binary @ binary).sum(axis=1)).ravel() / 2
    degree = np.asarray(binary.sum(axis=1)).ravel()
    possible = degree * (degree - 1) / 2
    return np.divide(
        triangles, possible, out=np.zeros_like(possible), where=possible > 0
    )


def _spearman(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    return float(spearmanr(a, b).statistic)  # type: ignore[attr-defined]


def _covariates(
    adjacency: Any, communities: dict[int, set[int]] | None
) -> dict[str, npt.NDArray[np.float64]]:
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    covariates = {
        "log_degree": np.log1p(degree),
        "clustering": clustering_coefficient(adjacency),
    }
    if communities is not None:
        covariates["neighbor_communities"] = neighbor_community_count(
            adjacency, communities
        ).astype(np.float64)
        covariates["neighbor_entropy"] = np.nan_to_num(
            neighbor_heterogeneity(adjacency, communities)
        )
    return covariates


def _node_parameters(
    posterior: Any, family: str
) -> tuple[dict[str, npt.NDArray[np.float64]], float | None]:
    """Each family's per-node confidence parameters (float64), and its mean ``r_bar``.

    ``r_bar`` is computed from samples and is a function of the parameters, so it is
    returned separately and kept out of the regressions and the effective rank.
    """
    if family == "tnbbeta":
        values = {
            "abs_p_minus_half": (posterior.p - 0.5).abs(),
            "q": posterior.q,
            "epsilon": posterior.epsilon,
        }
    elif family == "vmf":
        values = {"kappa": posterior.scale.reshape(-1)}
    elif family == "power_spherical":
        values = {"kappa": posterior.kappa}
    else:
        values = {"sigma_mean": posterior.base_dist.scale.mean(-1)}
    r_bar = None
    if family != "gaussian":
        samples = posterior.rsample(torch.Size([_R_BAR_SAMPLES]))
        r_bar = float(samples.mean(0).norm(dim=-1).mean())
    arrays = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in values.items()}
    return arrays, r_bar


def _decoding(
    model: Any,
    posterior: Any,
    centres: torch.Tensor,
    split: Any,
    device: torch.device,
    num_nodes: int,
) -> dict[str, float]:
    """Test AUC from centres vs. from posterior samples (see the module docstring)."""
    centre_auc, _ = score_edges(centres, split.test_positive, split.test_negative)
    positive = torch.as_tensor(split.test_positive, dtype=torch.long, device=device)
    negative = torch.as_tensor(split.test_negative, dtype=torch.long, device=device)
    results: dict[str, float] = {"centre_auc": float(centre_auc)}
    for num_samples in _MC_SAMPLES:
        positive_scores = torch.zeros(positive.shape[1], device=device)
        negative_scores = torch.zeros(negative.shape[1], device=device)
        for _ in range(num_samples):
            z = posterior.rsample()
            positive_scores += model.link_logits(z, positive).sigmoid()
            negative_scores += model.link_logits(z, negative).sigmoid()
        results[f"sampled_auc_{num_samples}"] = roc_auc(
            (positive_scores / num_samples).cpu().numpy(),
            (negative_scores / num_samples).cpu().numpy(),
        )
    results["num_nodes"] = float(num_nodes)
    return results


if __name__ == "__main__":
    main()
