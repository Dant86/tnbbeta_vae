"""ITD-only HRTF, arm V follow-up: collapse-controlled bound comparison.

Pre-registration:
``writeup/results/hrtf_itd_armv_followup_preregistration_2026-11-01.md`` (frozen;
deviations are appended there). The registered study it follows is
:mod:`apps.synthetic.hrtf_itd`, which is imported from and left untouched. Changes to
arm V: the primary metric is the held-out importance-weighted bound (``K = 64``), the
KL warm-up lasts 100 epochs, the learned sigma starts at 0.1, and a second decoder
fixes the axis to the interaural axis ``u = +y``.

Usage (one run per call; the shell loops over the grid)::

    uv run python -m apps.synthetic.hrtf_itd_armv_followup train \
        --decoder fixed_axis --family tnbbeta --seed 0 --out-dir hrtf_runs/armv_followup
    uv run python -m apps.synthetic.hrtf_itd_armv_followup summarize \
        --out-dir hrtf_runs/armv_followup

Each run writes ``<out-dir>/<decoder>/<family>/s<seed>.{json,npz,pt}``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy import stats
import torch
from torch import Tensor
from torch.distributions import Distribution

from apps.synthetic.hrtf_itd import (
    EVAL_SEED,
    FAMILIES,
    GRID_NODES,
    KL_DRAWS,
    LAT_DRAWS,
    HrtfModel,
    evaluate,
    family_spec,
    load_split,
    summarize_run,
)
from tnbbeta_vae.data.hrtf import AreaWeightedSampler, HrtfSplit
from tnbbeta_vae.models.heads import P_RAW_INIT, LatentFamily, standard_prior
from tnbbeta_vae.models.losses.elbo import pixel_log_likelihood
from tnbbeta_vae.models.losses.importance_weighted import importance_weighted_metrics
from tnbbeta_vae.training import select_device

DECODERS = ("learned_axis", "fixed_axis")
COMPARATORS = ("vmf", "power_spherical")
INITIAL_SIGMA = 0.1
WARMUP_EPOCHS = 100
IWAE_SAMPLES = 64
COLLAPSE_SIGMA = 0.5
MARGIN_NATS = 0.02
HOLM_ALPHA = 0.05
NUM_SPLITS = 10
_Y = (0.0, 1.0, 0.0)

Cell = dict[tuple[str, str], dict[int, dict[str, Any]]]


class FollowupModel(HrtfModel):
    """Arm V of the registered study with an optional exactly fixed decoder axis.

    ``learned_axis`` is the registered model. ``fixed_axis`` replaces the learned
    ``raw_u`` parameter by a buffer equal to ``+y`` (so ``u = +y`` exactly and only
    ``g`` and sigma are learned). The registered model's random draws are consumed
    either way, so the encoder, head and ``g`` start identically in both decoders.

    Attributes:
        decoder: ``"learned_axis"`` or ``"fixed_axis"``.
    """

    def __init__(
        self,
        decoder: str,
        family: LatentFamily,
        p_init_offset: float = P_RAW_INIT,
        initial_sigma: float = INITIAL_SIGMA,
    ) -> None:
        """Builds the model.

        Args:
            decoder: ``"learned_axis"`` or ``"fixed_axis"``.
            family: ``"vmf"``, ``"power_spherical"`` or ``"tnbbeta"``.
            p_init_offset: TNBBeta only: initial raw-p bias of the head.
            initial_sigma: Starting value of the learned likelihood sigma.
        """
        if decoder not in DECODERS:
            raise ValueError(f"unknown decoder {decoder!r}.")
        super().__init__("V", family, p_init_offset, initial_sigma)
        self.decoder = decoder
        if decoder == "fixed_axis":
            del self.raw_u
            self.register_buffer("raw_u", torch.tensor(_Y))


def run_path(out_dir: Path, decoder: str, family: str, seed: int) -> Path:
    """The stem (no suffix) of one run's output files."""
    return out_dir / decoder / family / f"s{seed}"


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``train`` or ``summarize``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--decoder", choices=list(DECODERS), required=True)
    train.add_argument("--family", choices=list(FAMILIES), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--num-elbo-samples", type=int, default=4)
    train.add_argument("--iwae-samples", type=int, default=IWAE_SAMPLES)
    train.add_argument("--kl-draws", type=int, default=KL_DRAWS)
    train.add_argument("--lat-draws", type=int, default=LAT_DRAWS)
    train.add_argument("--grid-nodes", type=int, default=GRID_NODES)
    train.add_argument("--data-path", type=Path, default=None)
    train.add_argument(
        "--subjects",
        type=int,
        default=None,
        help="Smoke tests only: keep the first N subject ids.",
    )
    train.add_argument("--device", default=None, help="Default: select_device().")
    train.add_argument("--out-dir", type=Path, required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    if args.command == "train":
        run_train(args)
    else:
        summarize(args.out_dir)


def run_train(args: argparse.Namespace) -> None:
    """Trains one (decoder, family, seed) on the GPU and evaluates it on the CPU."""
    device = select_device(args.device)
    started = time.time()
    split = load_split("xcorr", args.seed, args.data_path, args.subjects)
    spec = family_spec(args.family)
    torch.manual_seed(args.seed)
    model = FollowupModel(args.decoder, **spec)
    model.to(device)
    losses = train_model(
        model,
        split,
        args.epochs,
        args.warmup_epochs,
        args.batch_size,
        args.num_elbo_samples,
        device,
    )
    model.cpu().double().eval()
    train_seconds = time.time() - started
    diverged = not (
        all(math.isfinite(v) for v in losses)
        and all(bool(torch.isfinite(t).all()) for t in model.state_dict().values())
    )
    sigma = float(model.learned_scale().detach())
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    if not diverged:
        torch.manual_seed(EVAL_SEED)
        arrays, extra = evaluate(
            model, split, args.kl_draws, args.lat_draws, args.grid_nodes
        )
        summary = summarize_run(arrays, extra, split)
        arrays.update(iwae_arrays(model, split, args.iwae_samples))
        summary.update(bound_summary(arrays, args.iwae_samples))
        diverged = not math.isfinite(summary["iwae"])
    summary.update(
        {
            "decoder": args.decoder,
            "family": args.family,
            "seed": args.seed,
            "estimator": "xcorr",
            "sigma": sigma,
            "collapsed": is_collapsed(sigma),
            "diverged": diverged,
            "p_init_offset": spec["p_init_offset"],
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "initial_sigma": INITIAL_SIGMA,
            "n_train": len(split.train_weight),
            "n_test": len(split.test_weight),
            "train_subjects": split.train_subjects.tolist(),
            "test_subjects": split.test_subjects.tolist(),
            "itd_mean_us": split.mean,
            "itd_sd_us": split.sd,
            "num_elbo_samples": args.num_elbo_samples,
            "iwae_samples": args.iwae_samples,
            "kl_draws": args.kl_draws,
            "lat_draws": args.lat_draws,
            "grid_nodes": args.grid_nodes,
            "final_train_loss": losses[-1],
            "loss_curve": losses[:: max(1, args.epochs // 30)],
            "train_seconds": train_seconds,
            "total_seconds": time.time() - started,
        }
    )
    stem = run_path(args.out_dir, args.decoder, args.family, args.seed)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(summary))
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # pyright: ignore[reportArgumentType]
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    print(
        f"{args.decoder} {args.family} s{args.seed}: "
        f"IWAE {summary.get('iwae', float('nan')):.4f} sigma {sigma:.3f} "
        f"collapsed {summary['collapsed']} diverged {diverged} "
        f"({summary['total_seconds']:.0f}s)",
        flush=True,
    )


def train_model(
    model: FollowupModel,
    split: HrtfSplit,
    epochs: int,
    warmup_epochs: int,
    batch_size: int,
    num_samples: int,
    device: torch.device,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Trains ``model`` with Adam, a linear KL warm-up and the area-weighted sampler.

    Args:
        model: The model (already on ``device``).
        split: The split (its training rows are used).
        epochs: Number of epochs.
        warmup_epochs: The KL weight rises linearly to 1 over this many epochs.
        batch_size: Batch size.
        num_samples: ELBO samples per example.
        device: Training device.
        learning_rate: Adam learning rate.

    Returns:
        The mean training loss of each epoch.
    """
    x_all = split.train_x.to(device)
    z_all = split.train_z.to(device)
    sampler = AreaWeightedSampler(split.train_weight, batch_size, 200 + split.seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    warmup = max(1, warmup_epochs)
    losses = []
    for epoch in range(epochs):
        kl_weight = min(1.0, (epoch + 1) / warmup)
        running = torch.zeros((), device=device)
        batches = sampler.epoch()
        for index in batches:
            index = index.to(device)
            optimiser.zero_grad()
            loss = model.loss(x_all[index], z_all[index], kl_weight, num_samples)
            loss.backward()
            optimiser.step()
            running += loss.detach()
        losses.append(float(running) / len(batches))
    return losses


def is_collapsed(sigma: float) -> bool:
    """A run is collapsed if its learned sigma exceeds 0.5 at the end of training."""
    return bool(sigma > COLLAPSE_SIGMA)


@torch.no_grad()
def iwae_arrays(
    model: FollowupModel,
    x: Tensor | HrtfSplit,
    num_samples: int = IWAE_SAMPLES,
    chunk: int = 500,
) -> dict[str, np.ndarray]:
    """Per-point IWAE bound, ELBO, reconstruction term and KL to the uniform prior.

    The proposal is ``q(z | x)`` with its exact ``log q`` (:func:`sample_with_log_prob`
    through :func:`importance_weighted_metrics`).

    Args:
        model: The float64 CPU model.
        x: Standardised ITDs ``(n, 1)`` or a split (its test rows).
        num_samples: Importance samples ``K``.
        chunk: Points per pass.

    Returns:
        Arrays of shape ``(n,)``: ``ll`` (the bound), ``elbo``, ``re``, ``kl_prior``.
    """
    points = x.test_x if isinstance(x, HrtfSplit) else x
    adapter = _BoundAdapter(model)
    pieces: dict[str, list[Tensor]] = {"ll": [], "elbo": [], "re": [], "kl": []}
    for start in range(0, points.shape[0], chunk):
        out = importance_weighted_metrics(
            adapter, points[start : start + chunk], num_samples=num_samples
        )
        for key in pieces:
            pieces[key].append(out[key])
    result = {key: torch.cat(value).numpy() for key, value in pieces.items()}
    result["kl_prior"] = result.pop("kl")
    return result


def bound_summary(arrays: dict[str, np.ndarray], num_samples: int) -> dict[str, Any]:
    """Area-weighted scalar summaries of the per-point bound arrays."""
    weight = arrays["weight"]
    return {
        "iwae": float((arrays["ll"] * weight).sum()),
        "iwae_unweighted": float(arrays["ll"].mean()),
        "iwae_k": num_samples,
        "neg_elbo": float(-(arrays["elbo"] * weight).sum()),
        "recon": float((arrays["re"] * weight).sum()),
        "kl_prior": float((arrays["kl_prior"] * weight).sum()),
        "kl_p_model": float((arrays["kl"] * weight).sum()),
    }


def oriented_latitude_check(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """POST-HOC latitude check with a label-free orientation of q's latitude.

    The registered-style check fixes the axis sign from the training labels, but the
    latent's own orientation is set by the sign of the learned ``g`` (a decreasing ``g``
    mirrors the latent). Here q's latitude is oriented against the *observation*: if the
    saved posterior-median latitude is negatively correlated with the ITD, the true
    ``w`` is compared with q after flipping its sign. The held-out true directions
    enter only the coverage itself, not the choice of orientation.

    Args:
        arrays: A run's per-point arrays (``lat_q*``, ``w_true``, ``itd_us``,
            ``weight``).

    Returns:
        ``flipped`` (1.0 if the latent was mirrored), area-weighted ``cov50``,
        ``cov95`` and ``median_abs_error``.
    """
    flipped = bool(np.corrcoef(arrays["lat_q50"], arrays["itd_us"])[0, 1] < 0)
    w = -arrays["w_true"] if flipped else arrays["w_true"]
    weight = arrays["weight"]
    in50 = (w >= arrays["lat_q25"]) & (w <= arrays["lat_q75"])
    in95 = (w >= arrays["lat_q025"]) & (w <= arrays["lat_q975"])
    return {
        "flipped": float(flipped),
        "cov50": float((in50 * weight).sum()),
        "cov95": float((in95 * weight).sum()),
        "median_abs_error": float((np.abs(arrays["lat_q50"] - w) * weight).sum()),
    }


# ----------------------------------------------------------------------------- analysis


def bound_of(record: dict[str, Any]) -> float:
    """The primary metric of a run; a diverged run counts as ``-inf`` (worst)."""
    if record.get("diverged") or not math.isfinite(record["iwae"]):
        return float("-inf")
    return float(record["iwae"])


def wilcoxon_greater(diff: np.ndarray) -> float:
    """One-sided (greater) Wilcoxon signed-rank p-value (scipy) of paired diffs."""
    if len(diff) == 0:
        return float("nan")
    if not np.any(diff != 0):
        return 1.0
    return float(stats.wilcoxon(diff, alternative="greater").pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def wilcoxon_greater_enumerated(diff: np.ndarray) -> float:
    """The same exact p-value by enumerating all ``2^n`` sign patterns."""
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 1.0
    ranks = stats.rankdata(np.abs(diff))
    observed = ranks[diff > 0].sum()
    count = 0
    total = 0
    for signs in itertools.product((0, 1), repeat=len(diff)):
        total += 1
        if (ranks * np.array(signs)).sum() >= observed - 1e-9:
            count += 1
    return count / total


def restricted_seeds(
    tnb: dict[int, dict[str, Any]], other: dict[int, dict[str, Any]]
) -> list[int]:
    """Splits where both runs of a pair exist and are uncollapsed (``sigma <= 0.5``)."""
    both = sorted(set(tnb) & set(other))
    return [
        s
        for s in both
        if not is_collapsed(tnb[s]["sigma"]) and not is_collapsed(other[s]["sigma"])
    ]


def paired_tests(runs: Cell, restricted: bool = False) -> list[dict[str, Any]]:
    """The four ``tnbbeta - comparator`` bound tests (decoders x comparators).

    Args:
        runs: Run records keyed ``(decoder, family)`` then split seed.
        restricted: Keep only splits where both runs are uncollapsed.

    Returns:
        One dict per test with the paired differences and both p-values. Pairs with
        fewer than one split are omitted.
    """
    tests = []
    for decoder in DECODERS:
        tnb = runs.get((decoder, "tnbbeta"), {})
        for other in COMPARATORS:
            cap = runs.get((decoder, other), {})
            seeds = (
                restricted_seeds(tnb, cap)
                if restricted
                else sorted(set(tnb) & set(cap))
            )
            if not seeds:
                continue
            diff = np.array([bound_of(tnb[s]) - bound_of(cap[s]) for s in seeds])
            tests.append(
                {
                    "decoder": decoder,
                    "other": other,
                    "n": len(seeds),
                    "seeds": seeds,
                    "diff": diff,
                    "p": wilcoxon_greater(diff),
                    "p_enum": wilcoxon_greater_enumerated(diff),
                }
            )
    return tests


def apply_holm(tests: list[dict[str, Any]], alpha: float = HOLM_ALPHA) -> None:
    """Adds Holm thresholds/decisions and the 0.02-nat margin flag, in place."""
    order = np.argsort([t["p"] for t in tests], kind="stable")
    still = True
    for rank, index in enumerate(order):
        test = tests[index]
        test["threshold"] = alpha / (len(tests) - rank)
        still = still and test["p"] <= test["threshold"]
        test["holm_reject"] = bool(still)
    for test in tests:
        test["margin"] = bool(test["diff"].mean() >= MARGIN_NATS)


def verdict(tests: list[dict[str, Any]]) -> dict[str, bool]:
    """The registered decision: all four tests present, survive Holm, margin holds."""
    complete = len(tests) == len(DECODERS) * len(COMPARATORS)
    holm = complete and all(t["holm_reject"] for t in tests)
    margin = complete and all(t["margin"] for t in tests)
    return {
        "complete": complete,
        "holm_all": holm,
        "margin_all": margin,
        "rule_met": holm and margin,
    }


def summarize(out_dir: Path) -> None:
    """Prints the pre-registered analysis as markdown from the saved run summaries."""
    runs = _load_runs(out_dir)
    if not runs:
        print(f"no runs found under {out_dir}")
        return
    print(
        "Note: the 10 split replicates overlap in subjects (each of 18 subjects is "
        "held out in about 1/6 of the splits and trains in the rest), so the Wilcoxon "
        "p-values are optimistic; the range over splits accompanies every effect.\n"
    )
    _print_main_table(runs)
    _print_per_split(runs)
    print(
        "\n## Confirmatory tests (one-sided paired Wilcoxon on the IWAE bound, Holm)\n"
    )
    tests = paired_tests(runs)
    apply_holm(tests)
    _print_tests(tests, holm=True)
    decision = verdict(tests)
    print(
        "\nDecision rule 'TNBBeta improves a collapse-controlled ITD VAE' (all four "
        f"tests survive Holm and the {MARGIN_NATS}-nat margin holds): Holm all = "
        f"{decision['holm_all']}, margin all = {decision['margin_all']}, rule met = "
        f"{decision['rule_met']}"
        + ("" if decision["complete"] else " (fewer than four tests available)")
        + "\n"
    )
    _print_collapse(runs)
    print(
        "\n## Restricted comparison (both runs uncollapsed; outside the Holm family)\n"
    )
    _print_tests(paired_tests(runs, restricted=True), holm=False)
    print("\n## Descriptive comparisons (bound, no p-values)\n")
    _print_descriptive(runs)
    _print_diagnostics(runs)
    _print_posthoc_latitude(out_dir, runs)


# ----------------------------------------------------------------------------- helpers


class _BoundAdapter:
    """The interface :func:`importance_weighted_metrics` expects, for one model."""

    def __init__(self, model: FollowupModel) -> None:
        self.model = model
        self.prior = standard_prior(model.family, 3, torch.device("cpu"))

    def posterior_and_prior(self, x: Tensor) -> tuple[Distribution, Distribution]:
        return self.model.posterior(x), self.prior

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        return pixel_log_likelihood(
            x, self.model.decode(z), "gaussian", self.model.learned_scale()
        )


def _load_runs(out_dir: Path) -> Cell:
    runs: Cell = {}
    paths = [p for d in DECODERS for p in sorted(out_dir.glob(f"{d}/*/s*.json"))]
    for path in paths:
        record = json.loads(path.read_text())
        key = (record["decoder"], record["family"])
        runs.setdefault(key, {})[record["seed"]] = record
    return runs


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _print_main_table(runs: Cell) -> None:
    for decoder in DECODERS:
        if not any(k[0] == decoder for k in runs):
            continue
        print(
            f"\n## Decoder {decoder}: test IWAE bound (K = 64, nats; higher better)\n"
        )
        print(
            "| family | splits | collapsed (sigma > 0.5) | bound mean +- sd | median | "
            "range | negative ELBO mean | KL(q || p_model) mean | "
            "sigma median (range) |"
        )
        print("|" + "---|" * 9)
        for family in FAMILIES:
            seeds = runs.get((decoder, family), {})
            if not seeds:
                continue
            b = np.array([bound_of(s) for s in seeds.values()])
            e = np.array([s.get("neg_elbo", float("nan")) for s in seeds.values()])
            k = np.array([s.get("kl_p_model", float("nan")) for s in seeds.values()])
            sig = np.array([s["sigma"] for s in seeds.values()])
            collapsed = sum(s["collapsed"] for s in seeds.values())
            print(
                f"| {family} | {len(seeds)} | {collapsed}/{len(seeds)} | "
                f"{b.mean():.4f} +- {_sd(b):.4f} | {np.median(b):.4f} | "
                f"{b.min():.4f} to {b.max():.4f} | {np.nanmean(e):.4f} | "
                f"{np.nanmean(k):.4f} | {np.median(sig):.3f} "
                f"({sig.min():.3f}-{sig.max():.3f}) |"
            )


def _print_per_split(runs: Cell) -> None:
    for decoder in DECODERS:
        present = [f for f in FAMILIES if runs.get((decoder, f))]
        if not present:
            continue
        print(
            f"\nPer split, decoder {decoder}: IWAE bound / negative ELBO / "
            "KL(q || p_model) / sigma (c = collapsed)\n"
        )
        print("| split | test subjects | " + " | ".join(present) + " |")
        print("|" + "---|" * (2 + len(present)))
        seeds = sorted({s for f in present for s in runs[(decoder, f)]})
        for seed in seeds:
            cells, subjects = [], ""
            for family in present:
                r = runs[(decoder, family)].get(seed)
                if r is None:
                    cells.append("-")
                    continue
                subjects = ",".join(f"H{s}" for s in r["test_subjects"])
                flag = " c" if r["collapsed"] else ""
                flag += " DIVERGED" if r["diverged"] else ""
                cells.append(
                    f"{r.get('iwae', float('nan')):.4f} / "
                    f"{r.get('neg_elbo', float('nan')):.4f} / "
                    f"{r.get('kl_p_model', float('nan')):.4f} / {r['sigma']:.3f}{flag}"
                )
            print(f"| {seed} | {subjects} | " + " | ".join(cells) + " |")


def _print_tests(tests: list[dict[str, Any]], holm: bool) -> None:
    head = (
        "| decoder | comparator | n | mean paired diff (tnbbeta - comparator) | "
        "range over splits | splits tnbbeta higher | p (scipy) | p (enumerated) |"
    )
    if holm:
        head += f" Holm threshold | Holm reject | margin >= {MARGIN_NATS} |"
    else:
        head += " splits used |"
    print(head)
    print("|" + "---|" * (11 if holm else 9))
    if not tests:
        print("| (no pairs) |")
    for test in sorted(tests, key=lambda t: (t["decoder"], t["other"])):
        diff = test["diff"]
        wins = int((diff > 0).sum())
        row = (
            f"| {test['decoder']} | {test['other']} | {test['n']} | "
            f"{diff.mean():.4f} | "
            f"{diff.min():.4f} to {diff.max():.4f} | {wins}/{test['n']} | "
            f"{test['p']:.5f} | {test['p_enum']:.5f} |"
        )
        if holm:
            row += (
                f" {test['threshold']:.5f} | {test['holm_reject']} | {test['margin']} |"
            )
        else:
            row += " " + ",".join(map(str, test["seeds"])) + " |"
        print(row)
        if abs(test["p"] - test["p_enum"]) > 1e-9:
            print(f"WARNING: scipy and enumerated p differ for {test['decoder']}")


def _print_collapse(runs: Cell) -> None:
    print("\n## Collapse accounting (sigma > 0.5 at the end of training)\n")
    print("| decoder | family | collapsed / runs | collapsed splits | diverged runs |")
    print("|---|---|---|---|---|")
    for decoder in DECODERS:
        for family in FAMILIES:
            seeds = runs.get((decoder, family), {})
            if not seeds:
                continue
            which = [s for s, r in sorted(seeds.items()) if r["collapsed"]]
            diverged = sum(r["diverged"] for r in seeds.values())
            print(
                f"| {decoder} | {family} | {len(which)}/{len(seeds)} | "
                f"{','.join(map(str, which)) or '-'} | {diverged} |"
            )


def _print_descriptive(runs: Cell) -> None:
    print(
        "| comparison | group | n | mean bound (A) | mean bound (B) | mean paired diff "
        "(A - B) | range of diff | splits A higher |"
    )
    print("|" + "---|" * 8)
    rows = [
        ("tnbbeta vs tnbbeta_init0", d, (d, "tnbbeta"), (d, "tnbbeta_init0"))
        for d in DECODERS
    ] + [
        (
            "learned_axis vs fixed_axis",
            f,
            ("learned_axis", f),
            ("fixed_axis", f),
        )
        for f in FAMILIES
    ]
    for name, group, a, b in rows:
        ra, rb = runs.get(a, {}), runs.get(b, {})
        seeds = sorted(set(ra) & set(rb))
        if not seeds:
            continue
        ka = np.array([bound_of(ra[s]) for s in seeds])
        kb = np.array([bound_of(rb[s]) for s in seeds])
        d = ka - kb
        print(
            f"| {name} | {group} | {len(seeds)} | {ka.mean():.4f} | {kb.mean():.4f} | "
            f"{d.mean():.4f} | {d.min():.4f} to {d.max():.4f} | "
            f"{int((ka > kb).sum())}/{len(seeds)} |"
        )


def _print_posthoc_latitude(out_dir: Path, runs: Cell) -> None:
    print(
        "\n## POST-HOC (not registered): latitude check with q oriented by the ITD "
        "(label-free; see the Deviations of the pre-registration)\n"
    )
    print(
        "| decoder | family | latent mirrored (splits) | central 50% coverage "
        "(mean, range) | central 95% coverage (mean, range) | mean abs error of "
        "the median |"
    )
    print("|" + "---|" * 6)
    for decoder in DECODERS:
        for family in FAMILIES:
            checks = []
            for seed in sorted(runs.get((decoder, family), {})):
                stem = run_path(out_dir, decoder, family, seed).with_suffix(".npz")
                with np.load(stem) as data:
                    if "lat_q50" in data:
                        checks.append(oriented_latitude_check(dict(data)))
            if not checks:
                continue
            c50 = np.array([c["cov50"] for c in checks])
            c95 = np.array([c["cov95"] for c in checks])
            err = np.mean([c["median_abs_error"] for c in checks])
            flips = int(sum(c["flipped"] for c in checks))
            print(
                f"| {decoder} | {family} | {flips}/{len(checks)} | {c50.mean():.3f} "
                f"({c50.min():.3f}-{c50.max():.3f}) | {c95.mean():.3f} "
                f"({c95.min():.3f}-{c95.max():.3f}) | {err:.4f} |"
            )


def _print_diagnostics(runs: Cell) -> None:
    present = [f for f in FAMILIES if runs.get(("fixed_axis", f))]
    if present:
        print(
            "\n## Label-based latitude check, decoder fixed_axis (q's latitude "
            "marginal about the sign-oriented +y against the true y . z; mean over "
            "splits; "
            "nominal coverage 0.50 / 0.95)\n"
        )
        print(
            "| family | central 50% coverage (range) | central 95% coverage (range) | "
            "median abs error of the marginal's median (area-weighted mean) | "
            "median |cos(mu, y)| over splits (range) |"
        )
        print("|" + "---|" * 5)
        for family in present:
            seeds = runs[("fixed_axis", family)]
            c50 = np.array([r["cov50"] for r in seeds.values() if "cov50" in r])
            c95 = np.array([r["cov95"] for r in seeds.values() if "cov95" in r])
            err = np.array(
                [r["median_abs_error"] for r in seeds.values() if "cov50" in r]
            )
            cos = np.array(
                [r["median_abs_cos"] for r in seeds.values() if "cov50" in r]
            )
            if len(c50) == 0:
                continue
            print(
                f"| {family} | {c50.mean():.3f} ({c50.min():.3f}-{c50.max():.3f}) | "
                f"{c95.mean():.3f} ({c95.min():.3f}-{c95.max():.3f}) | "
                f"{err.mean():.4f} | {np.median(cos):.3f} "
                f"({cos.min():.3f}-{cos.max():.3f}) |"
            )
        print(
            "\nPer split, fixed_axis: coverage 50% / 95% / median abs error / "
            "|cos(mu, y)|\n"
        )
        print("| split | " + " | ".join(present) + " |")
        print("|" + "---|" * (1 + len(present)))
        seeds_all = sorted({s for f in present for s in runs[("fixed_axis", f)]})
        for seed in seeds_all:
            cells = []
            for family in present:
                r = runs[("fixed_axis", family)].get(seed)
                cells.append(
                    "-"
                    if r is None or "cov50" not in r
                    else (
                        f"{r['cov50']:.2f} / {r['cov95']:.2f} / "
                        f"{r['median_abs_error']:.3f} / {r['median_abs_cos']:.3f}"
                    )
                )
            print(f"| {seed} | " + " | ".join(cells) + " |")
    present = [f for f in FAMILIES if runs.get(("learned_axis", f))]
    if present:
        print(
            "\n## Decoder learned_axis: |cos(u, +/- y)| and |cos(mu, u)|, per split\n"
        )
        print("| split | " + " | ".join(present) + " |")
        print("|" + "---|" * (1 + len(present)))
        seeds_all = sorted({s for f in present for s in runs[("learned_axis", f)]})
        for seed in seeds_all:
            cells = []
            for family in present:
                r = runs[("learned_axis", family)].get(seed)
                cells.append(
                    "-"
                    if r is None or "u_abs_cos_y" not in r
                    else f"{r['u_abs_cos_y']:.3f} / {r['median_abs_cos']:.3f}"
                )
            print(f"| {seed} | " + " | ".join(cells) + " |")
        print(
            "\nLearned-axis latitude check (about the signed u; mean over splits): "
            "family: coverage 50% / 95%\n"
        )
        for family in present:
            seeds = runs[("learned_axis", family)]
            rows = [r for r in seeds.values() if "cov50" in r]
            if rows:
                print(
                    f"- {family}: {np.mean([r['cov50'] for r in rows]):.3f} / "
                    f"{np.mean([r['cov95'] for r in rows]):.3f}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
