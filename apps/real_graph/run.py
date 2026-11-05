"""Training and scoring runs of the real-graph revision (pre-registered).

Usage (``writeup/results/real_graph_revision_preregistration_2026-11-03.md``):
    uv run python -m apps.real_graph.run manifest --stage tune
    uv run python -m apps.real_graph.run run --stage tune --shard 0 --num-shards 96
    uv run python -m apps.real_graph.run select   # tuned.json from the tune runs
    uv run python -m apps.real_graph.run manifest --stage main  # or thin/community/pin
    uv run python -m apps.real_graph.run run --stage main --shard 0 --num-shards 96

Everything lands under ``$TNBBETA_CHECKPOINT_DIR/real_graph/<stage>/``:
``manifest.jsonl``, ``runs/<id>.json`` (metrics and shape summaries), and for the
``main`` and ``pin`` stages ``pairs/<id>.npz`` (test-pair scores and labels) and
``nodes/<id>.npz`` (training degree and per-node posterior statistics). Finished runs
are skipped, so resubmitting is safe.

One run trains a ``GraphVAE`` of one family on one (possibly thinned or community)
graph and one seed, selects the epoch by validation sampled log loss, and scores the
validation and test pairs with 32 posterior samples. The edge split, the thinning and
the model initialisation of a run depend only on its seed, so all families of a seed
are paired.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import random
import time
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
import torch

from tnbbeta_vae.data.dense_graphs import DENSE_GRAPH_NAMES, load_dense_graph
from tnbbeta_vae.data.mag_coauthor import load_mag_coauthor
from tnbbeta_vae.data.planetoid import (
    Graph,
    LinkSplit,
    load_planetoid,
    normalized_adjacency,
    split_edges,
)
from tnbbeta_vae.data.snap_community import identity_features
from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.heads import epsilon_from_m
from tnbbeta_vae.models.posterior_stats import posterior_stats
from tnbbeta_vae.models.predictive import bernoulli_log_loss
from tnbbeta_vae.models.real_graph_eval import (
    degree_buckets,
    graph_statistics,
    node_shape_stats,
    pair_metrics,
    pair_probabilities,
    thin_edges,
    vmf_sampler_name,
)
from tnbbeta_vae.paths import checkpoint_dir, data_dir
from tnbbeta_vae.training import select_device

DATASETS: tuple[str, ...] = ("cora", "citeseer", "pubmed", "mag_cs")
FEATURES = ("on", "off")
FAMILIES = ("tnbbeta", "vmf", "power_spherical", "gaussian")
EPOCHS: dict[str, int] = {"cora": 200, "citeseer": 200, "pubmed": 400, "mag_cs": 200}
TUNE_SEEDS = (100, 101, 102)
FINAL_SEEDS = tuple(range(10))
COMMUNITY_SEEDS = tuple(range(5))
TUNE_GRID: dict[str, tuple[Any, ...]] = {
    "lr": (0.003, 0.01, 0.03),
    "dropout": (0.0, 0.4),
    "dim": (16, 32),
}
# ``REAL_GRAPH_STUDY=dense`` switches to the dense real-graph study
# (writeup/results/dense_real_graph_preregistration_2026-11-04.md): three dense graphs,
# a wider grid, its own output directory. The default reproduces the revision exactly.
STUDY = os.environ.get("REAL_GRAPH_STUDY", "revision")
if STUDY == "dense":
    DATASETS = tuple(DENSE_GRAPH_NAMES)
    EPOCHS = dict.fromkeys(DATASETS, 200)
    TUNE_GRID = {
        "lr": (0.003, 0.01, 0.03),
        "dropout": (0.0, 0.4, 0.6),
        "dim": (16, 32, 64),
    }
KEEP_LEVELS = (0.7, 0.5, 0.3)
PIN_FRACTIONS = (-0.6, -0.2, 0.0, 0.5, 1.0)
SELECT_SAMPLES = 16
FINAL_SAMPLES = 32
SHAPE_SAMPLES = 64
NUM_COMMUNITIES = 12
COMMUNITY_SIZE = (300, 1200)
COMMUNITY_DATASETS = ("mag_cs", "pubmed")
STAGES = ("tune", "main", "thin", "community", "pin")
_SAVE_PAIRS = ("main",)
_SAVE_NODES = ("main", "pin")

_graph_cache: dict[str, Graph] = {}
_tensor_cache: dict[tuple[Any, ...], torch.Tensor] = {}


def main(argv: list[str] | None = None) -> None:
    """Dispatches the ``manifest``, ``select`` and ``run`` subcommands.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="List the runs of a stage.")
    manifest.add_argument("--stage", choices=STAGES, required=True)
    sub.add_parser("select", help="Freeze the validation-selected configurations.")
    run = sub.add_parser("run", help="Run one shard of a stage's manifest.")
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--num-shards", type=int, required=True)
    run.add_argument("--device", default=None)
    run.add_argument(
        "--cell",
        default=None,
        help="Only the jobs of one cell, 'dataset|features'.",
    )
    args = parser.parse_args(argv)
    if args.command == "manifest":
        write_manifest(args.stage)
    elif args.command == "select":
        select_configurations()
    else:
        run_shard(args.stage, args.shard, args.num_shards, args.device, args.cell)


def root_dir() -> Path:
    """Returns (without creating) the study's output directory."""
    return checkpoint_dir() / ("real_graph_dense" if STUDY == "dense" else "real_graph")


def stage_dir(stage: str) -> Path:
    """Returns (without creating) the output directory of ``stage``."""
    return root_dir() / stage


def latent_dim_for(family: str, dim: int) -> int:
    """Ambient latent dimension: sphere families live on ``S^dim`` (``dim + 1``)."""
    return dim if family == "gaussian" else dim + 1


def model_config_for(
    family: str,
    in_features: int,
    dim: int,
    dropout: float,
    pin_fraction: float | None = None,
) -> GraphVAEConfig:
    """Builds the ``GraphVAE`` config of one condition.

    Args:
        family: One of :data:`FAMILIES`.
        in_features: Number of node features.
        dim: Manifold dimension ``d``.
        dropout: Dropout of both GCN layers' inputs.
        pin_fraction: TNBBeta only: pin ``m = pin_fraction * (latent_dim - 1) / 2``
            (``None`` leaves epsilon free).

    Returns:
        The config: decoder bias on for every family, vMF and power-spherical kappa
        initialised at the ambient dimension and trained.
    """
    latent_dim = latent_dim_for(family, dim)
    extra: dict[str, Any] = {}
    if family in ("vmf", "power_spherical"):
        extra["initial_kappa"] = float(latent_dim)
    if pin_fraction is not None:
        if family != "tnbbeta":
            raise ValueError("pin_fraction is TNBBeta only.")
        extra["fixed_epsilon"] = epsilon_from_m(
            pin_fraction * (latent_dim - 1) / 2, latent_dim
        )
    return GraphVAEConfig(
        family=cast("Any", family),
        in_features=in_features,
        latent_dim=latent_dim,
        dropout=dropout,
        decoder_bias=True,
        **extra,
    )


def run_id(job: dict[str, Any]) -> str:
    """Returns the stable file stem of a manifest entry."""
    parts = [job["stage"], job["dataset"], job["features"], job["family"]]
    if job.get("keep") is not None:
        parts.append(f"keep{job['keep']:g}")
    if job.get("community") is not None:
        parts.append(f"comm{job['community']}")
    if job.get("pin") is not None:
        parts.append(f"f{job['pin']:g}")
    if job["stage"] == "tune":
        parts += [f"lr{job['lr']:g}", f"do{job['dropout']:g}", f"d{job['dim']}"]
    parts.append(f"s{job['seed']}")
    return "_".join(parts)


def write_manifest(stage: str) -> Path:
    """Writes the manifest of ``stage`` (and, for ``community``, its graph list).

    Args:
        stage: One of :data:`STAGES`.

    Returns:
        The manifest path.
    """
    directory = stage_dir(stage)
    for sub in ("runs", "pairs", "nodes"):
        (directory / sub).mkdir(parents=True, exist_ok=True)
    if stage == "tune":
        jobs = _tune_jobs()
    else:
        tuned = json.loads((root_dir() / "tuned.json").read_text())
        if stage == "community":
            _write_communities()
        jobs = _final_jobs(stage, tuned)
    random.Random(20261103).shuffle(jobs)
    for job in jobs:
        job["run_id"] = run_id(job)
    path = directory / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(job) for job in jobs) + "\n")
    print(f"Wrote {len(jobs)} runs to {path}")
    return path


def select_configurations() -> Path:
    """Freezes, per (dataset, features, family), the best-validation configuration.

    The criterion is the mean validation sampled log loss over the tuning seeds
    (diverged runs count as infinite loss). Writes ``tuned.json``.

    Returns:
        The path of ``tuned.json``.
    """
    runs = [
        json.loads(path.read_text())
        for path in (stage_dir("tune") / "runs").glob("*.json")
    ]
    grouped: dict[tuple[str, str, str, float, float, int], list[float]] = {}
    for run in runs:
        job = run["job"]
        loss = run["val_log_loss"] if run["diverged"] == 0 else float("inf")
        key = (
            job["dataset"],
            job["features"],
            job["family"],
            job["lr"],
            job["dropout"],
            job["dim"],
        )
        grouped.setdefault(key, []).append(loss)
    tuned: dict[str, Any] = {}
    table: dict[str, list[dict[str, Any]]] = {}
    for key, losses in sorted(grouped.items()):
        cell = "|".join(key[:3])
        entry = {
            "lr": key[3],
            "dropout": key[4],
            "dim": key[5],
            "val_log_loss": float(np.mean(losses)),
            "num_seeds": len(losses),
        }
        table.setdefault(cell, []).append(entry)
    expected = len(TUNE_SEEDS)
    grid_size = len(list(itertools.product(*TUNE_GRID.values())))
    for cell, entries in table.items():
        complete = [entry for entry in entries if entry["num_seeds"] == expected]
        if len(complete) != grid_size:
            raise SystemExit(
                f"{cell}: {len(complete)} of {grid_size} configurations are complete."
            )
        best = min(complete, key=lambda entry: entry["val_log_loss"])
        tuned[cell] = {**best, "grid": entries}
    path = root_dir() / "tuned.json"
    path.write_text(json.dumps(tuned, indent=2))
    for cell, chosen in sorted(tuned.items()):
        print(
            cell,
            {k: chosen[k] for k in ("lr", "dropout", "dim", "val_log_loss")},
            f"({len(chosen['grid'])} configs)",
        )
    print(f"Wrote {path}")
    return path


def run_shard(
    stage: str,
    shard: int,
    num_shards: int,
    device_name: str | None,
    cell: str | None = None,
) -> None:
    """Runs this shard's share of the manifest, skipping finished runs.

    Args:
        stage: Stage name.
        shard: Index of this shard.
        num_shards: Number of shards.
        device_name: Device name, or ``None`` for ``select_device``'s choice.
        cell: If given (``"dataset|features"``), only that cell's jobs are sharded.
    """
    directory = stage_dir(stage)
    lines = (directory / "manifest.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in lines if line]
    if cell is not None:
        entries = [e for e in entries if f"{e['dataset']}|{e['features']}" == cell]
    jobs = entries[shard::num_shards]
    device = select_device(device_name)
    failures = 0
    for job in jobs:
        output = directory / "runs" / f"{job['run_id']}.json"
        if output.exists():
            continue
        try:
            result = run_job(job, device)
        except Exception as error:  # keep the shard going; report at the end
            failures += 1
            print(f"FAILED {job['run_id']}: {error!r}", flush=True)
            continue
        arrays = result.pop("_arrays", {})
        for kind, content in arrays.items():
            np.savez_compressed(directory / kind / f"{job['run_id']}.npz", **content)
        output.write_text(json.dumps(result))
        print(f"done {job['run_id']} in {result['seconds']:.0f}s", flush=True)
    if failures:
        raise SystemExit(f"{failures} run(s) failed")


def run_job(job: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Trains one condition on one graph and one seed, and scores it.

    Args:
        job: A manifest entry.
        device: Device to train on.

    Returns:
        The job, validation and test metrics, selection and shape summaries and the
        wall-clock time; under ``"_arrays"`` the per-pair and per-node arrays to save.
    """
    start = time.time()
    seed = int(job["seed"])
    graph, statistics = build_graph(job)
    split = split_edges(graph.adjacency, seed=seed)
    batch = _make_batch(job, graph, split, device)
    config = model_config_for(
        job["family"],
        batch.features.shape[1],
        job["dim"],
        job["dropout"],
        job.get("pin"),
    )
    epochs = EPOCHS[job["dataset"]]
    trained = train(config, batch, split, lr=job["lr"], epochs=epochs, seed=seed)
    result: dict[str, Any] = {
        "job": job,
        "graph_stats": statistics,
        "best_epoch": trained["best_epoch"],
        "diverged": trained["diverged"],
        "val_log_loss": trained["val_log_loss_select"],
    }
    if trained["state"] is None:
        result["seconds"] = time.time() - start
        return result
    model = GraphVAE(config).to(device)
    model.load_state_dict(trained["state"])
    result.update(score(model, batch, split, seed=seed, keep_arrays=job["stage"]))
    result["seconds"] = time.time() - start
    return result


def build_graph(job: dict[str, Any]) -> tuple[Graph, dict[str, float]]:
    """Builds the graph of a job: parent dataset, optional community, optional thinning.

    Args:
        job: A manifest entry.

    Returns:
        The graph (features on or off) and the statistics of its adjacency (after
        thinning, before the edge split).
    """
    parent = _load_graph(job["dataset"])
    adjacency = parent.adjacency.tocsr()
    features: Any = parent.features
    if job.get("community") is not None:
        selected = json.loads(
            (stage_dir("community") / f"communities_{job['dataset']}.json").read_text()
        )
        members = np.array(
            next(item for item in selected if item["community"] == job["community"])[
                "nodes"
            ]
        )
        adjacency = adjacency[members][:, members].tocsr()
        features = features.tocsr()[members]
    if job.get("keep") is not None:
        adjacency = thin_edges(adjacency, float(job["keep"]), int(job["seed"]))
    if job["features"] == "off":
        features = identity_features(adjacency.shape[0])
    return Graph(adjacency, features), graph_statistics(adjacency)


def train(
    config: GraphVAEConfig,
    batch: GraphBatch,
    split: LinkSplit,
    *,
    lr: float,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    """Trains a model, keeping the epoch with the best validation sampled log loss.

    Args:
        config: Model config.
        batch: The training graph.
        split: Edge split (only its validation pairs are read here).
        lr: Adam learning rate.
        epochs: Full-graph training steps.
        seed: Seed of initialisation, negative sampling and dropout.

    Returns:
        ``state`` (the best epoch's ``state_dict``, ``None`` if no finite epoch),
        ``best_epoch``, ``val_log_loss_select`` (inf when none) and ``diverged``.
    """
    device = batch.features.device
    torch.manual_seed(seed)
    model = GraphVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pairs = np.concatenate([split.val_positive, split.val_negative], axis=1)
    labels = np.concatenate(
        [np.ones(split.val_positive.shape[1]), np.zeros(split.val_negative.shape[1])]
    )
    targets = torch.as_tensor(labels, dtype=torch.float64)
    best = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    diverged = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.training_step(batch)["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
            diverged = 1.0
            break
        optimizer.step()
        model.eval()
        with torch.no_grad():
            posterior, _ = model.posterior_and_prior(batch)
            if (
                config.family == "vmf"
                and vmf_sampler_name(posterior) != "exact_envelope"
            ):
                raise RuntimeError("vMF run is not using the exact envelope sampler.")
            probability = pair_probabilities(
                model,
                batch,
                pairs,
                num_samples=SELECT_SAMPLES,
                seed=seed + 1,
                posterior=posterior,
            )["sampled"]
        value = float(bernoulli_log_loss(torch.as_tensor(probability), targets))
        if np.isfinite(value) and value < best:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {
        "state": best_state,
        "best_epoch": best_epoch,
        "val_log_loss_select": best,
        "diverged": diverged,
    }


@torch.no_grad()
def score(
    model: GraphVAE,
    batch: GraphBatch,
    split: LinkSplit,
    *,
    seed: int,
    keep_arrays: str,
) -> dict[str, Any]:
    """Scores a trained model on validation and test pairs and summarises its posterior.

    Args:
        model: The trained (best-validation) model.
        batch: The training graph.
        split: Edge split.
        seed: Run seed (the sampling seeds derive from it).
        keep_arrays: The stage name; decides which per-pair and per-node arrays are
            returned for saving.

    Returns:
        ``val`` and ``test`` metric dicts, posterior summaries, the decoder scalars,
        the vMF sampler name and (under ``_arrays``) the arrays to save.
    """
    model.eval()
    posterior, _ = model.posterior_and_prior(batch)
    sampler = vmf_sampler_name(posterior)
    if model.config.family == "vmf" and sampler != "exact_envelope":
        raise RuntimeError("vMF run is not using the exact envelope sampler.")
    val_pairs = np.concatenate([split.val_positive, split.val_negative], axis=1)
    test_pairs = np.concatenate([split.test_positive, split.test_negative], axis=1)
    val_labels = np.concatenate(
        [np.ones(split.val_positive.shape[1]), np.zeros(split.val_negative.shape[1])]
    )
    test_labels = np.concatenate(
        [np.ones(split.test_positive.shape[1]), np.zeros(split.test_negative.shape[1])]
    )
    everything = np.concatenate([val_pairs, test_pairs], axis=1)
    probabilities = pair_probabilities(
        model,
        batch,
        everything,
        num_samples=FINAL_SAMPLES,
        seed=seed + 17,
        posterior=posterior,
    )
    split_at = val_pairs.shape[1]
    val = pair_metrics(
        probabilities["sampled"][:split_at],
        probabilities["centre"][:split_at],
        val_labels,
    )
    test_sampled = probabilities["sampled"][split_at:]
    test_centre = probabilities["centre"][split_at:]
    test = pair_metrics(test_sampled, test_centre, test_labels)
    degree = np.asarray(split.train_adjacency.sum(axis=1)).ravel().astype(np.float64)
    nodes = node_shape_stats(
        model, posterior, num_samples=SHAPE_SAMPLES, seed=seed + 29
    )
    everyone = torch.ones(
        degree.shape[0], dtype=torch.bool, device=batch.features.device
    )
    stats = posterior_stats(posterior, everyone, model.config.model_dump())
    shape = _shape_summary(nodes, degree, model.config.latent_dim)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    if keep_arrays in _SAVE_PAIRS:
        arrays["pairs"] = {
            "pairs": test_pairs,
            "labels": test_labels,
            "sampled": test_sampled.astype(np.float32),
            "centre": test_centre.astype(np.float32),
            "spread": probabilities["spread"][split_at:].astype(np.float32),
            "bucket": degree_buckets(degree, test_pairs),
        }
    if keep_arrays in _SAVE_NODES:
        arrays["nodes"] = {"degree": degree, **nodes}
    return {
        "val": val,
        "test": test,
        "shape": shape,
        "post_r_bar": stats["r_bar"],
        "temperature": 1.0
        if model.config.family == "gaussian"
        else float(model.temperature()),
        "logit_bias": float(model.logit_bias()),
        "vmf_sampler": sampler,
        "num_test_pairs": int(test_pairs.shape[1]),
        "_arrays": arrays,
    }


def _base_job(stage: str, dataset: str, features: str, family: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "dataset": dataset,
        "features": features,
        "family": family,
        "keep": None,
        "community": None,
        "pin": None,
    }


def _tune_jobs() -> list[dict[str, Any]]:
    jobs = []
    for dataset, features, family in itertools.product(DATASETS, FEATURES, FAMILIES):
        for lr, dropout, dim in itertools.product(
            TUNE_GRID["lr"], TUNE_GRID["dropout"], TUNE_GRID["dim"]
        ):
            for seed in TUNE_SEEDS:
                job = _base_job("tune", dataset, features, family)
                jobs.append(
                    {**job, "lr": lr, "dropout": dropout, "dim": dim, "seed": seed}
                )
    return jobs


def _final_jobs(stage: str, tuned: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    def add(
        dataset: str,
        features: str,
        family: str,
        seed: int,
        **extra: Any,
    ) -> None:
        chosen = tuned[f"{dataset}|{features}|{family}"]
        job = _base_job(stage, dataset, features, family)
        job.update(
            lr=chosen["lr"], dropout=chosen["dropout"], dim=chosen["dim"], seed=seed
        )
        job.update(extra)
        jobs.append(job)

    for dataset, features in itertools.product(DATASETS, FEATURES):
        if stage == "main":
            for family, seed in itertools.product(FAMILIES, FINAL_SEEDS):
                add(dataset, features, family, seed)
        elif stage == "thin":
            for family, seed, keep in itertools.product(
                FAMILIES, FINAL_SEEDS, KEEP_LEVELS
            ):
                add(dataset, features, family, seed, keep=keep)
        elif stage == "pin":
            for seed, fraction in itertools.product(FINAL_SEEDS, PIN_FRACTIONS):
                add(dataset, features, "tnbbeta", seed, pin=fraction)
        elif stage == "community" and dataset in COMMUNITY_DATASETS:
            selected = json.loads(
                (stage_dir("community") / f"communities_{dataset}.json").read_text()
            )
            for item, family, seed in itertools.product(
                selected, FAMILIES, COMMUNITY_SEEDS
            ):
                add(dataset, features, family, seed, community=item["community"])
    return jobs


def _write_communities() -> None:
    """Louvain communities of the community-stage datasets, selected by density only."""
    import networkx  # noqa: PLC0415

    for dataset in COMMUNITY_DATASETS:
        graph = _load_graph(dataset)
        adjacency = graph.adjacency.tocsr()
        network = networkx.from_scipy_sparse_array(adjacency)
        parts = networkx.community.louvain_communities(network, resolution=1.0, seed=0)
        low, high = COMMUNITY_SIZE
        eligible = []
        for index, nodes in enumerate(parts):
            if low <= len(nodes) <= high:
                members = np.array(sorted(nodes))
                statistics = graph_statistics(adjacency[members][:, members])
                eligible.append(
                    {"community": index, "nodes": members.tolist(), **statistics}
                )
        eligible.sort(key=lambda item: (item["mean_degree"], item["community"]))
        count = min(NUM_COMMUNITIES, len(eligible))
        ranks = np.unique(
            np.round(np.linspace(0, len(eligible) - 1, count)).astype(int)
        )
        selected = [eligible[int(rank)] for rank in ranks]
        out = stage_dir("community") / f"communities_{dataset}.json"
        out.write_text(json.dumps(selected))
        print(
            f"{dataset}: {len(parts)} Louvain communities, {len(eligible)} eligible, "
            f"{len(selected)} selected (mean degrees "
            f"{[round(item['mean_degree'], 2) for item in selected]})"
        )


def _load_graph(dataset: str) -> Graph:
    if dataset not in _graph_cache:
        if dataset in DENSE_GRAPH_NAMES:
            loaded: Any = load_dense_graph(data_dir() / "dense_graphs", dataset)
        elif dataset == "mag_cs":
            loaded = load_mag_coauthor(data_dir() / "mag_coauthor", "cs")
        else:
            loaded = load_planetoid(data_dir() / "planetoid", dataset)
        _graph_cache[dataset] = Graph(loaded.adjacency, loaded.features)
    return _graph_cache[dataset]


def _make_batch(
    job: dict[str, Any], graph: Graph, split: LinkSplit, device: torch.device
) -> GraphBatch:
    key = (job["dataset"], job["features"], job.get("community"))
    if key not in _tensor_cache:
        if hasattr(graph.features, "tocoo"):
            tensor = torch.as_tensor(graph.features.toarray(), dtype=torch.float32)
        else:
            tensor = cast("torch.Tensor", graph.features).to(dtype=torch.float32)
        _tensor_cache[key] = tensor.to(device)
    upper = _upper_edges(split.train_adjacency)
    return GraphBatch(
        features=_tensor_cache[key],
        norm_adjacency=normalized_adjacency(split.train_adjacency),
        positive_edges=torch.as_tensor(upper, dtype=torch.long),
    ).to(device)


def _upper_edges(adjacency: Any) -> np.ndarray:
    upper: Any = sp.triu(adjacency, k=1).tocoo()
    return np.stack([upper.row, upper.col])


def _shape_summary(
    nodes: dict[str, np.ndarray], degree: np.ndarray, latent_dim: int
) -> dict[str, float]:
    """Run-level summaries of per-node shape statistics and degree correlations."""
    out: dict[str, float] = {
        "second_mode_mass": float(np.mean(nodes["second_mode_mass"])),
        "frac_nodes_second_mode": float(np.mean(nodes["second_mode_mass"] > 0.1)),
    }

    def correlation(values: np.ndarray) -> float:
        if np.std(values) == 0 or np.std(degree) == 0:
            return float("nan")
        return float(cast("Any", spearmanr(values, degree))[0])

    if "m" in nodes:
        out.update(
            m_mean=float(nodes["m"].mean()),
            frac_m_negative=float((nodes["m"] < 0).mean()),
            m_degree_spearman=correlation(nodes["m"]),
            p_mean=float(nodes["p"].mean()),
            q_mean=float(nodes["q"].mean()),
            epsilon_mean=float(nodes["epsilon"].mean()),
            second_mode_degree_spearman=correlation(nodes["second_mode_mass"]),
        )
    if "kappa" in nodes:
        out.update(
            kappa_mean=float(nodes["kappa"].mean()),
            kappa_median=float(np.median(nodes["kappa"])),
        )
    return out


if __name__ == "__main__":
    main()
