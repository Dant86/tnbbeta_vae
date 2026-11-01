"""A wide search over synthetic graph types for where TNBBeta posteriors might help.

Usage:
    uv run python -m apps.synthetic.graph_search configs --name discovery \
        --per-family 12
    uv run python -m apps.synthetic.graph_search manifest --stage discovery --draws 16
    uv run python -m apps.synthetic.graph_search run --stage discovery --shard 0 \
        --num-shards 576

Three stages share one code path: ``pilot`` tunes learning rates and measures variance,
``discovery`` is the wide test, and ``confirmation`` repeats rejected configurations on
fresh graphs (see ``writeup/results/graph_search_preregistration_*.md``). A *config* is
a calibrated point of ``tnbbeta_vae.data.graph_space``, a *draw* an independent graph
from it, and a *run* one trained model. All conditions of one (config, draw) see the
same graph, edge split and seed, so differences between them are paired.

Everything lands under ``$TNBBETA_CHECKPOINT_DIR/graph_search/<stage>/``:
``configs.json``, ``manifest.jsonl`` and ``runs/<run id>.json`` (finished runs are
skipped on resubmission).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from scipy.stats import qmc, spearmanr
import torch

from apps.link_prediction.main import run_once
from apps.synthetic.graph_families import second_mode_mass
from tnbbeta_vae.data.graph_space import (
    FAMILIES,
    GraphSpec,
    SyntheticGraph,
    calibrate_contrast,
    sample_graph,
)
from tnbbeta_vae.data.planetoid import (
    Graph,
    LinkSplit,
    normalized_adjacency,
    split_edges,
)
from tnbbeta_vae.models import GraphBatch, GraphVAE, GraphVAEConfig
from tnbbeta_vae.models.losses.ranking import roc_auc
from tnbbeta_vae.models.posterior_stats import posterior_stats
from tnbbeta_vae.models.predictive import (
    balanced_truth,
    bernoulli_kl,
    bernoulli_log_loss,
    calibration_slope,
    dense_edge_probabilities,
    uncertainty_error_auroc,
)
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import select_device

DIM = 16
EPOCHS = 500
SAMPLES = 32
LEARNING_RATES = (0.003, 0.01, 0.03)
SEED_BASE = {
    "pilot": 9_000_000,
    "discovery": 10_000_000,
    "confirmation": 20_000_000,
    "kappa_grid": 30_000_000,
}
# Fixed-kappa grid of the ``kappa_grid`` stage (diffuse to tight; ``DIM`` is the pin).
KAPPA_GRID = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
# Conditions and the model each trains. ``_init`` starts kappa at ``DIM`` and trains it
# (the primary comparator), ``_pin`` fixes it there, ``_default`` is the repo's default.
CONDITIONS = {
    "tnbbeta": ("tnbbeta", {}),
    "vmf_init": ("vmf", {"initial_kappa": float(DIM)}),
    "power_spherical_init": ("power_spherical", {"initial_kappa": float(DIM)}),
    "vmf_pin": ("vmf", {"fixed_kappa": float(DIM)}),
    "power_spherical_pin": ("power_spherical", {"fixed_kappa": float(DIM)}),
    "vmf_default": ("vmf", {}),
    "power_spherical_default": ("power_spherical", {}),
    "gaussian": ("gaussian", {}),
}
for _kappa in KAPPA_GRID:
    CONDITIONS[f"vmf_k{_kappa:g}"] = ("vmf", {"fixed_kappa": _kappa})
    CONDITIONS[f"power_spherical_k{_kappa:g}"] = (
        "power_spherical",
        {"fixed_kappa": _kappa},
    )
_TUNED = ("tnbbeta", "vmf_init", "power_spherical_init", "gaussian")
_SECONDARY_LR_FROM = {
    "vmf_pin": "vmf_init",
    "vmf_default": "vmf_init",
    "power_spherical_pin": "power_spherical_init",
    "power_spherical_default": "power_spherical_init",
}
_AUC_TOLERANCE = 0.02
for _name, (_family, _extra) in list(CONDITIONS.items()):
    if "fixed_kappa" in _extra and _name not in _SECONDARY_LR_FROM:
        _SECONDARY_LR_FROM[_name] = f"{_family}_init"


def main(argv: list[str] | None = None) -> None:
    """Dispatches the ``configs``, ``manifest`` and ``run`` subcommands.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    configs = sub.add_parser("configs", help="Sample and calibrate graph configs.")
    configs.add_argument("--name", required=True)
    configs.add_argument("--per-family", type=int, default=12)
    configs.add_argument("--seed", type=int, default=0)
    manifest = sub.add_parser("manifest", help="List the runs of a stage.")
    manifest.add_argument("--stage", choices=sorted(SEED_BASE), required=True)
    manifest.add_argument(
        "--configs", default=None, help="Config set (default: stage)."
    )
    manifest.add_argument("--config-ids", nargs="*", default=None)
    manifest.add_argument("--draws", type=int, required=True)
    manifest.add_argument("--first-draw", type=int, default=0)
    manifest.add_argument("--conditions", nargs="+", default=None)
    run = sub.add_parser("run", help="Run one shard of a stage's manifest.")
    run.add_argument("--stage", choices=sorted(SEED_BASE), required=True)
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--num-shards", type=int, required=True)
    run.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.command == "configs":
        write_configs(args.name, args.per_family, args.seed)
    elif args.command == "manifest":
        write_manifest(
            args.stage,
            args.configs or args.stage,
            draws=args.draws,
            first_draw=args.first_draw,
            conditions=args.conditions,
            config_ids=args.config_ids,
        )
    else:
        run_shard(args.stage, args.shard, args.num_shards, args.device)


def stage_dir(stage: str) -> Path:
    """Returns (without creating) the output directory of ``stage``."""
    return checkpoint_dir() / "graph_search" / stage


def sample_configs(name: str, per_family: int, seed: int = 0) -> list[dict[str, Any]]:
    """Samples reachable, calibrated configs per family from a scrambled Sobol sequence.

    Axes: size (log-uniform 300-1200), mean degree (log-uniform 4-40), groups or
    intrinsic dimension (log-uniform 2-16), heterogeneity, features and a target oracle
    AUC (uniform 0.65-0.95). Targets a family cannot reach within ``_AUC_TOLERANCE``
    (few blocks cap the AUC) are rejected and the next Sobol point is used, so a
    family's targets are confined to what it can express.

    Args:
        name: Config-set name (seeds the sequence together with ``seed``).
        per_family: Configs to keep per family.
        seed: Extra seed offset.

    Returns:
        Dicts with ``config_id`` and the calibrated ``GraphSpec`` fields.
    """
    configs = []
    for family_index, family in enumerate(FAMILIES):
        sobol = qmc.Sobol(
            d=6, scramble=True, rng=_stable_seed(name, family_index, seed)
        )
        kept = 0
        for u in sobol.random_base2(m=7):
            if kept == per_family:
                break
            spec = GraphSpec(
                family=family,
                num_nodes=int(50 * round(_log_uniform(u[0], 300, 1200) / 50)),
                mean_degree=round(_log_uniform(u[1], 4, 40), 1),
                num_groups=int(round(_log_uniform(u[2], 2, 16))),
                heterogeneity=("none", "mild", "heavy")[min(int(u[3] * 3), 2)],
                features="none" if u[4] < 0.5 else "informative",
            )
            if family in ("scale_free", "degree_corrected"):
                spec = GraphSpec(**{**asdict(spec), "heterogeneity": "none"})
            calibrated = calibrate_contrast(spec, 0.65 + 0.30 * float(u[5]))
            if abs(calibrated.achieved_auc - calibrated.target_auc) > _AUC_TOLERANCE:
                continue
            configs.append({"config_id": f"{family}-{kept:02d}", **asdict(calibrated)})
            kept += 1
    return configs


def write_configs(name: str, per_family: int, seed: int) -> Path:
    """Writes ``graph_search/configs_<name>.json`` and returns its path."""
    configs = sample_configs(name, per_family, seed)
    path = checkpoint_dir() / "graph_search" / f"configs_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configs, indent=2))
    print(f"Wrote {len(configs)} configs to {path}")
    return path


def build_manifest(
    stage: str,
    configs: list[dict[str, Any]],
    *,
    draws: int,
    first_draw: int = 0,
    conditions: list[str] | None = None,
    tuned_lr: dict[str, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    """Lists every run of a stage, largest graphs first so shards balance.

    Args:
        stage: ``pilot`` (every condition at every learning rate), or ``discovery`` /
            ``confirmation`` (each condition at its tuned rate).
        configs: Calibrated configs (see :func:`sample_configs`).
        draws: Independent graph draws per config.
        first_draw: Index of the first draw (to extend a stage with fresh draws).
        conditions: Conditions to run (default: the tuned four for the pilot, all else).
        tuned_lr: ``{family: {condition: lr}}`` from the pilot (required off the pilot).

    Returns:
        Run descriptions with a unique ``run_id``.
    """
    default = [c for c in CONDITIONS if "_k" not in c or stage == "kappa_grid"]
    names = conditions or (list(_TUNED) if stage == "pilot" else default)
    items = []
    for index, config in enumerate(configs):
        for draw in range(first_draw, first_draw + draws):
            for condition in names:
                if stage == "pilot":
                    rates: list[float] = list(LEARNING_RATES)
                else:
                    assert tuned_lr is not None, "tuned learning rates are required"
                    source = _SECONDARY_LR_FROM.get(condition, condition)
                    rates = [tuned_lr[config["family"]][source]]
                for lr in rates:
                    run_id = f"{config['config_id']}_d{draw:03d}_{condition}_lr{lr}"
                    items.append(
                        {
                            "run_id": run_id,
                            "config_id": config["config_id"],
                            "draw": draw,
                            "seed": SEED_BASE[stage] + 1000 * index + draw,
                            "condition": condition,
                            "lr": lr,
                            "num_nodes": config["num_nodes"],
                        }
                    )
    items.sort(key=lambda item: (-item["num_nodes"], item["run_id"]))
    return items


def write_manifest(
    stage: str,
    config_set: str,
    *,
    draws: int,
    first_draw: int,
    conditions: list[str] | None,
    config_ids: list[str] | None,
) -> Path:
    """Writes ``<stage dir>/manifest.jsonl`` and copies the configs next to it."""
    configs = json.loads(
        (checkpoint_dir() / "graph_search" / f"configs_{config_set}.json").read_text()
    )
    if config_ids:
        configs = [c for c in configs if c["config_id"] in set(config_ids)]
    tuned = None
    if stage != "pilot":
        tuned = json.loads((stage_dir("pilot") / "tuned_lr.json").read_text())
    items = build_manifest(
        stage,
        configs,
        draws=draws,
        first_draw=first_draw,
        conditions=conditions,
        tuned_lr=tuned,
    )
    directory = stage_dir(stage)
    (directory / "runs").mkdir(parents=True, exist_ok=True)
    (directory / "configs.json").write_text(json.dumps(configs, indent=2))
    path = directory / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in items) + "\n")
    print(f"Wrote {len(items)} runs ({len(configs)} configs) to {path}")
    return path


def run_shard(stage: str, shard: int, num_shards: int, device_name: str | None) -> None:
    """Runs this shard's share of the manifest, skipping finished runs."""
    directory = stage_dir(stage)
    configs = {
        c["config_id"]: c for c in json.loads((directory / "configs.json").read_text())
    }
    lines = (directory / "manifest.jsonl").read_text().splitlines()
    items = [json.loads(line) for line in lines if line][shard::num_shards]
    device = select_device(device_name)
    failures = 0
    for item in items:
        output = directory / "runs" / f"{item['run_id']}.json"
        if output.exists():
            continue
        try:
            result = run_item(item, configs[item["config_id"]], device)
        except Exception as error:  # keep the shard going; report at the end
            failures += 1
            print(f"FAILED {item['run_id']}: {error!r}", flush=True)
            continue
        output.write_text(json.dumps(result))
        print(f"done {item['run_id']} in {result['seconds']:.0f}s", flush=True)
    if failures:
        raise SystemExit(f"{failures} run(s) failed")


def run_item(
    item: dict[str, Any], config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Trains one condition on one graph draw and scores it.

    Args:
        item: A manifest entry.
        config: The matching calibrated config.
        device: Device to train on.

    Returns:
        The manifest entry, the config's key fields, metrics (see :func:`evaluate`),
        and wall-clock ``seconds``.
    """
    start = time.time()
    spec = GraphSpec(**{k: v for k, v in config.items() if k != "config_id"})
    sample = sample_graph(spec, item["seed"])
    split = split_edges(sample.adjacency, seed=item["seed"])
    model_config = model_config_for(
        item["condition"], int(sample.features.get_shape()[1])
    )
    state: dict[str, torch.Tensor] = {}
    trained = run_once(
        Graph(sample.adjacency, sample.features),
        split,
        model_config,
        lr=item["lr"],
        epochs=EPOCHS,
        seed=item["seed"],
        device=device,
        best_state=state,
    )
    model = GraphVAE(model_config).to(device)
    model.load_state_dict(state)
    metrics = evaluate(model, sample, split, seed=item["seed"])
    return {
        **item,
        "family": spec.family,
        "metrics": metrics,
        "val_auc": trained["val_auc"],
        "best_epoch": trained["best_epoch"],
        "diverged": trained.get("diverged", 0.0),
        "seconds": time.time() - start,
    }


def model_config_for(condition: str, in_features: int) -> GraphVAEConfig:
    """Builds the ``GraphVAE`` config of ``condition`` (decoder bias on for all)."""
    family, extra = CONDITIONS[condition]
    return GraphVAEConfig(
        family=cast("Any", family),
        in_features=in_features,
        latent_dim=DIM - 1 if family == "gaussian" else DIM,
        decoder_bias=True,
        **extra,
    )


@torch.no_grad()
def evaluate(
    model: GraphVAE, sample: SyntheticGraph, split: LinkSplit, *, seed: int
) -> dict[str, float]:
    """Scores a trained model on held-out pairs and against the true probabilities.

    Returns:
        ``centre_auc``, ``sampled_auc``, ``sampled_log_loss``, ``cal_slope``
        (calibration slope of the sampled prediction) and ``cal_slope_dev`` (its
        distance from 1),
        ``uncertainty_auroc`` (does the predictive spread flag wrong edges),
        ``truth_kl`` (sampled prediction against the balanced true probability over
        non-training pairs), ``rank_corr`` (Spearman of centre logits against log true
        probability over a pair sample), and posterior-shape diagnostics.
    """
    device = next(model.parameters()).device
    model.eval()
    features = torch.as_tensor(sample.features.toarray(), dtype=torch.float32)
    batch = GraphBatch(
        features=features,
        norm_adjacency=normalized_adjacency(split.train_adjacency),
        positive_edges=torch.as_tensor(
            np.stack(sp.triu(split.train_adjacency, k=1).nonzero()), dtype=torch.long
        ),
    ).to(device)
    posterior, _ = model.posterior_and_prior(batch)
    centres = model.embeddings(batch)
    torch.manual_seed(seed)
    samples = posterior.rsample(torch.Size([SAMPLES]))

    positive, negative = split.test_positive, split.test_negative
    pairs = torch.as_tensor(np.concatenate([positive, negative], axis=1), device=device)
    targets = torch.cat([torch.ones(positive.shape[1]), torch.zeros(negative.shape[1])])
    centre_probability = torch.sigmoid(model.link_logits(centres, pairs)).cpu()
    draws = torch.stack(
        [torch.sigmoid(model.link_logits(z, pairs)) for z in samples]
    ).cpu()
    mean, spread = draws.mean(0), draws.std(0)
    num_positive = positive.shape[1]
    slope, intercept = calibration_slope(mean, targets)
    recalibrate = _validation_recalibration(model, samples, split, device)

    truth = balanced_truth(torch.as_tensor(sample.probability, dtype=torch.float32))
    train_dense = split.train_adjacency.toarray() == 0
    held = np.triu(train_dense, k=1)
    dense = dense_edge_probabilities(model, batch, num_samples=SAMPLES)
    held_mask = torch.as_tensor(held)
    rank_pairs = _sample_pairs(held, seed)
    centre_logits = model.link_logits(
        centres, torch.as_tensor(rank_pairs, device=device)
    ).cpu()
    log_truth = np.log(
        np.clip(sample.probability[rank_pairs[0], rank_pairs[1]], 1e-12, None)
    )
    statistic = cast("Any", spearmanr(centre_logits.numpy(), log_truth))[0]
    stats = posterior_stats(
        posterior,
        torch.ones(centres.shape[0], dtype=torch.bool, device=device),
        model.config.model_dump(),
    )
    return {
        "centre_auc": roc_auc(
            centre_probability[:num_positive].numpy(),
            centre_probability[num_positive:].numpy(),
        ),
        "sampled_auc": roc_auc(
            mean[:num_positive].numpy(), mean[num_positive:].numpy()
        ),
        "sampled_log_loss": float(bernoulli_log_loss(mean, targets)),
        "cal_slope": slope,
        "cal_intercept": intercept,
        "cal_slope_dev": abs(slope - 1.0),
        "uncertainty_auroc": uncertainty_error_auroc(mean, spread, targets),
        "truth_kl": float(bernoulli_kl(truth[held_mask], dense[held_mask])),
        "recal_log_loss": float(bernoulli_log_loss(recalibrate(mean), targets)),
        "recal_truth_kl": float(
            bernoulli_kl(truth[held_mask], recalibrate(dense)[held_mask])
        ),
        "rank_corr": float(statistic),
        **{f"post_{k}": float(v) for k, v in stats.items() if k != "num_nodes"},
        **second_mode_mass(posterior, SAMPLES),
    }


def _validation_recalibration(
    model: GraphVAE, samples: torch.Tensor, split: LinkSplit, device: torch.device
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a one-parameter-pair Platt map fitted on the validation pairs.

    The map is ``p -> sigmoid(intercept + slope * logit(p))`` with both fitted on the
    mean sampled prediction at the validation edges and non-edges, then applied to
    test-time predictions, so it separates a model's ranking and shape from its
    overall temperature.
    """
    pairs = torch.as_tensor(
        np.concatenate([split.val_positive, split.val_negative], axis=1), device=device
    )
    targets = torch.cat(
        [
            torch.ones(split.val_positive.shape[1]),
            torch.zeros(split.val_negative.shape[1]),
        ]
    )
    mean = (
        torch.stack([torch.sigmoid(model.link_logits(z, pairs)) for z in samples])
        .mean(0)
        .cpu()
    )
    slope, intercept = calibration_slope(mean, targets)

    def recalibrate(probability: torch.Tensor) -> torch.Tensor:
        clipped = probability.clamp(1e-6, 1 - 1e-6)
        return torch.sigmoid(intercept + slope * torch.logit(clipped))

    return recalibrate


def _sample_pairs(mask: np.ndarray, seed: int, count: int = 200_000) -> np.ndarray:
    """Returns up to ``count`` random ``(row, col)`` pairs where ``mask`` is true."""
    rows, cols = mask.nonzero()
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows.shape[0], size=min(count, rows.shape[0]), replace=False)
    return np.stack([rows[pick], cols[pick]])


def _log_uniform(u: float, low: float, high: float) -> float:
    """Maps ``u`` in [0, 1) log-uniformly onto ``[low, high]``."""
    return math.exp(math.log(low) + float(u) * (math.log(high) - math.log(low)))


def _stable_seed(name: str, family_index: int, seed: int) -> int:
    """A reproducible integer seed from a name (``hash`` is salted per process)."""
    return (
        sum(ord(c) * 31**i for i, c in enumerate(name)) + 1009 * family_index + seed
    ) % (2**31)


if __name__ == "__main__":
    main()
