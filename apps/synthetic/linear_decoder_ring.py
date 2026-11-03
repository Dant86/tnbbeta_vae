"""Sphere probabilistic PCA with a LINEAR learned decoder: does TNBBeta find the ring?

Pre-registration: ``writeup/results/linear_decoder_ring_preregistration_2026-11-02.md``
(grid, decoders, arms, metrics, tests and decision rules are fixed there; deviations are
appended at its end). Task: :mod:`tnbbeta_vae.data.linear_ring`.

Usage (one run per call; the sbatch script loops over the grid)::

    uv run python -m apps.synthetic.linear_decoder_ring train --decoder L --dim 5 \
        --sigma 0.2 --arm tnbbeta --seed 0 --out-dir ring_recovery/linear_decoder
    uv run python -m apps.synthetic.linear_decoder_ring summarize \
        --out-dir ring_recovery/linear_decoder

Each run writes ``<out-dir>/<decoder>/d<d>_s<sigma>/<arm>/s<seed>.{json,npz,pt}``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy import stats
import torch
from torch import Tensor, nn
from torch.distributions import Distribution, kl_divergence

from tnbbeta_vae.data.linear_ring import (
    information_nats,
    kl_uniform_to_posterior,
    latitude_std,
    log_evidence,
    sample_observations,
)
from tnbbeta_vae.data.ring_posterior import sphere_log_area
from tnbbeta_vae.distributions import HypersphericalUniform, VonMisesFisher
from tnbbeta_vae.models.architectures.mlp import mlp_stack
from tnbbeta_vae.models.heads import (
    P_RAW_INIT,
    LatentFamily,
    head_size,
    init_tnbbeta_p_bias,
    posterior_centre,
    posterior_from_raw,
    power_spherical_kappa_inverse,
    standard_prior,
    tnbbeta_posterior,
    vmf_kappa_inverse,
)
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo, pixel_log_likelihood
from tnbbeta_vae.models.losses.importance_weighted import importance_weighted_metrics
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.models.losses.sampling import sample_with_log_prob
from tnbbeta_vae.training import select_device

DECODERS = ("L", "Lb", "M", "F")
ARMS = ("vmf", "power_spherical", "tnbbeta", "tnbbeta_init0")
PRIMARY_COMPARATORS = ("vmf", "power_spherical")
DIMS = (3, 5, 9, 17)
SIGMAS = (0.1, 0.2, 0.4)
CONTRAST_SIGMA = 0.2
SIGMA_START = 0.2
NUM_SEEDS = 10
MARGIN_NATS = 0.05
HOLM_ALPHA = 0.05
ALIGNED_COS = 0.99
TEST_SEED = 424242
ELBO_DRAWS = 256
IWAE_K = 64
JACOBIAN_POINTS = 20_000
_HIDDEN = [128, 128]
_LINK_HIDDEN = 32
_EPSILON_FLOOR = 1e-4
_EVAL_CHUNK = 250

Runs = dict[tuple[str, int, float, str], dict[int, dict[str, Any]]]


class ExactVonMisesFisher(VonMisesFisher):
    """vMF whose rejection sampler always uses Wood's exact envelope parameter ``b``.

    The library sampler switches ``b`` to a Taylor value for ``kappa > 11`` while ``a``
    and ``d`` keep the exact-``b`` formulas, which biases draws for kappa of about 10-40
    at ambient dimension 9 and above. ``m = 3`` uses the library's closed-form sampler.
    """

    def _sample_w_rej(self, shape: torch.Size) -> Tensor:
        c = torch.sqrt(4 * self.scale**2 + (self._m - 1) ** 2)
        b = (self._m - 1) / (c + 2 * self.scale)
        a = (self._m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self._m - 1) * math.log(self._m - 1)
        _, w = self._while_loop(b, a, d, shape, k=self.k)
        return w


class LinearRingVAE(nn.Module):
    """A study-local VAE: MLP encoder + posterior head and a scalar-output decoder.

    Attributes:
        decoder_kind: ``"L"`` (``u . z``), ``"Lb"`` (``u . z + b``), ``"M"``
            (``g(u . z)``) or ``"F"`` (free MLP of ``z``).
        dim: Ambient dimension ``d``.
        family: Latent family of the posterior head.
    """

    def __init__(self, decoder: str, dim: int, arm: str) -> None:
        """Builds encoder, decoder, learned sigma and head (in that RNG order).

        The head is built last so the encoder and decoder initial weights of one seed
        are identical across arms (only the head differs in size).

        Args:
            decoder: One of :data:`DECODERS`.
            dim: Ambient dimension.
            arm: One of :data:`ARMS`.
        """
        super().__init__()
        spec = arm_spec(arm, dim)
        self.decoder_kind = decoder
        self.dim = dim
        self.family: LatentFamily = spec["family"]
        self.encoder = mlp_stack([1, *_HIDDEN])
        if decoder == "F":
            self.decoder_net = nn.Sequential(
                mlp_stack([dim, *_HIDDEN]), nn.Linear(_HIDDEN[-1], 1)
            )
        else:
            self.axis_raw = nn.Parameter(torch.randn(dim))
        if decoder == "Lb":
            self.bias = nn.Parameter(torch.zeros(()))
        if decoder == "M":
            self.link = nn.Sequential(
                nn.Linear(1, _LINK_HIDDEN),
                nn.Tanh(),
                nn.Linear(_LINK_HIDDEN, _LINK_HIDDEN),
                nn.Tanh(),
                nn.Linear(_LINK_HIDDEN, 1),
            )
        self.learned_scale = LearnedLikelihoodScale(SIGMA_START)
        self.posterior_head = nn.Linear(_HIDDEN[-1], head_size(self.family, dim))
        self._init_head(spec)

    def posterior(self, y: Tensor) -> Distribution:
        """``q(z | y)`` for scalar observations ``y`` of shape ``(batch,)``."""
        raw = self.posterior_head(self.encoder(y[:, None]))
        if self.family == "tnbbeta":
            return tnbbeta_posterior(raw, self.dim, epsilon_floor=_EPSILON_FLOOR)
        posterior = posterior_from_raw(self.family, raw, self.dim)
        if self.family == "vmf":
            return ExactVonMisesFisher(posterior.loc, posterior.scale)  # pyright: ignore[reportAttributeAccessIssue]
        return posterior

    def decode(self, z: Tensor) -> Tensor:
        """The decoder mean for ``z`` of shape ``(..., d)``; shape ``(..., 1)``."""
        if self.decoder_kind == "F":
            return self.decoder_net(z)
        index = (z * self.axis()).sum(-1, keepdim=True)
        if self.decoder_kind == "M":
            return self.link(index)
        if self.decoder_kind == "Lb":
            return index + self.bias
        return index

    def axis(self) -> Tensor:
        """The learned unit axis ``u`` (not defined for the free decoder ``F``)."""
        return nn.functional.normalize(self.axis_raw, dim=-1)

    def latitude_mean(self, w: Tensor) -> Tensor:
        """The decoder mean as a function of the latitude ``w = u . z``."""
        if self.decoder_kind == "M":
            return self.link(w[:, None]).squeeze(-1)
        if self.decoder_kind == "Lb":
            return w + self.bias
        return w

    def log_likelihood(self, y: Tensor, z: Tensor) -> Tensor:
        """``log p(y | z)`` with the learned sigma, shape ``z.shape[:-1]``."""
        return pixel_log_likelihood(
            y[:, None], self.decode(z), "gaussian", self.learned_scale()
        )

    def negative_elbo(self, y: Tensor, kl_weight: float, num_samples: int) -> Tensor:
        """The training loss: mean over the batch of ``-(log p(y|z) - w * KL)``."""
        posterior = self.posterior(y)
        prior = standard_prior(self.family, self.dim, y.device)
        terms = monte_carlo_elbo(
            y[:, None],
            posterior,
            prior,
            self.decode,
            self.learned_scale(),
            num_samples,
            analytic_kl=self.family != "tnbbeta",
        )
        return -(terms["log_likelihood"] - kl_weight * terms["kl"]).mean()

    def _init_head(self, spec: dict[str, Any]) -> None:
        """Applies the arm's head initialisation (kappa / p biases)."""
        head = self.posterior_head
        with torch.no_grad():
            if self.family == "vmf" and spec["kappa_init"] is not None:
                head.bias[-1] = vmf_kappa_inverse(spec["kappa_init"])
            if self.family == "power_spherical" and spec["kappa_init"] is not None:
                head.bias[-1] = power_spherical_kappa_inverse(spec["kappa_init"])
        if self.family == "tnbbeta":
            init_tnbbeta_p_bias(head, self.dim, offset=spec["p_init_offset"])


def arm_spec(arm: str, dim: int) -> dict[str, Any]:
    """Model settings of an arm: family, initial kappa (vMF / PS: ``d``), p offset."""
    if arm == "vmf":
        return {"family": "vmf", "kappa_init": float(dim), "p_init_offset": 0.0}
    if arm == "power_spherical":
        return {
            "family": "power_spherical",
            "kappa_init": float(dim),
            "p_init_offset": 0.0,
        }
    if arm == "tnbbeta":
        return {"family": "tnbbeta", "kappa_init": None, "p_init_offset": P_RAW_INIT}
    if arm == "tnbbeta_init0":
        return {"family": "tnbbeta", "kappa_init": None, "p_init_offset": 0.0}
    raise ValueError(f"unknown arm {arm!r}.")


def grid_cells() -> list[tuple[str, int, float, str, int]]:
    """Every ``(decoder, dim, sigma, arm, seed)`` of the pre-registered grid.

    Ordered by priority (L, F, M, Lb), then seed, so a shortfall of compute truncates
    the descriptive contrasts first.
    """
    cells = []
    for decoder in ("L", "F", "M", "Lb"):
        sigmas = SIGMAS if decoder == "L" else (CONTRAST_SIGMA,)
        for seed, dim, sigma, arm in itertools.product(
            range(NUM_SEEDS), DIMS, sigmas, ARMS
        ):
            cells.append((decoder, dim, sigma, arm, seed))
    return cells


def sigma_label(sigma: float) -> str:
    """The directory label of a noise scale."""
    return f"{sigma:g}"


def run_path(
    out_dir: Path, decoder: str, dim: int, sigma: float, arm: str, seed: int
) -> Path:
    """The stem (no suffix) of one run's output files."""
    return out_dir / decoder / f"d{dim}_s{sigma_label(sigma)}" / arm / f"s{seed}"


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``train``, ``summarize`` or ``cells``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--decoder", choices=list(DECODERS), required=True)
    train.add_argument("--dim", type=int, required=True)
    train.add_argument("--sigma", type=float, required=True)
    train.add_argument("--arm", choices=list(ARMS), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--num-train", type=int, default=20_000)
    train.add_argument("--num-test", type=int, default=2_000)
    train.add_argument("--num-elbo-samples", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--device", default=None, help="Default: select_device().")
    train.add_argument("--out-dir", type=Path, required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--out-dir", type=Path, required=True)
    sub.add_parser("cells", help="Print the grid, one cell per line.")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "train":
        run_train(args)
    elif args.command == "cells":
        for cell in grid_cells():
            print(*cell)
    else:
        summarize(args.out_dir)


def run_train(args: argparse.Namespace) -> None:
    """Trains one cell on the GPU, evaluates it on the CPU in float64."""
    device = select_device(args.device)
    started = time.time()
    train_gen = torch.Generator().manual_seed(100 + args.seed)
    train_y = sample_observations(args.dim, args.sigma, args.num_train, train_gen)
    torch.manual_seed(args.seed)
    model = LinearRingVAE(args.decoder, args.dim, args.arm)
    model.to(device)
    y_dev = train_y.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    shuffle = torch.Generator().manual_seed(200 + args.seed)
    warmup = max(1, args.epochs // 10)
    losses: list[float] = []
    diverged = False
    for epoch in range(args.epochs):
        kl_weight = min(1.0, (epoch + 1) / warmup)
        order = torch.randperm(args.num_train, generator=shuffle).to(device)
        running = torch.zeros((), device=device)
        for index in order.split(args.batch_size):
            optimiser.zero_grad()
            loss = model.negative_elbo(y_dev[index], kl_weight, args.num_elbo_samples)
            loss.backward()
            optimiser.step()
            running += loss.detach() * len(index)
        losses.append(float(running) / args.num_train)
        if not math.isfinite(losses[-1]):
            diverged = True
            break
    model.cpu().double().eval()
    train_seconds = time.time() - started

    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    if not diverged:
        test_gen = torch.Generator().manual_seed(TEST_SEED)
        test_y = sample_observations(args.dim, args.sigma, args.num_test, test_gen)
        torch.manual_seed(31337)
        arrays, summary = evaluate(model, args.sigma, test_y.double())
        diverged = not all(
            np.isfinite(arrays[k]).all() for k in ("elbo", "iwae", "log_p_true")
        )
    summary.update(
        {
            "decoder": args.decoder,
            "dim": args.dim,
            "sigma_true": args.sigma,
            "arm": args.arm,
            "seed": args.seed,
            "family": model.family,
            "epochs": args.epochs,
            "num_train": args.num_train,
            "num_test": args.num_test,
            "num_elbo_samples": args.num_elbo_samples,
            "elbo_draws": ELBO_DRAWS,
            "iwae_k": IWAE_K,
            "diverged": bool(diverged),
            "epochs_run": len(losses),
            "final_train_loss": losses[-1] if losses else float("nan"),
            "loss_curve": losses[:: max(1, args.epochs // 30)],
            "train_seconds": train_seconds,
            "total_seconds": time.time() - started,
        }
    )
    stem = run_path(
        args.out_dir, args.decoder, args.dim, args.sigma, args.arm, args.seed
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(summary))
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # pyright: ignore[reportArgumentType]
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    gap = summary.get("gap_elbo", float("nan"))
    print(
        f"{args.decoder} d={args.dim} sigma={args.sigma} {args.arm} s{args.seed}: gap "
        f"{gap:.4f} sigma_hat {summary.get('sigma_hat', float('nan')):.4f} |cos| "
        f"{summary.get('abs_cos_axis', float('nan')):.3f} diverged {diverged} "
        f"({summary['total_seconds']:.0f}s)",
        flush=True,
    )


@torch.no_grad()
def evaluate(
    model: LinearRingVAE, sigma: float, y: Tensor
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per-``y`` ELBO / IWAE gaps, decomposition, ring statistic and axis alignment.

    Args:
        model: The (float64, CPU) model.
        sigma: The true noise standard deviation.
        y: Test observations ``(n,)``, float64.

    Returns:
        ``(arrays, summary)``: per-``y`` numpy arrays and a JSON-ready scalar summary.
    """
    dim = model.dim
    log_p_true = log_evidence(y, dim, sigma)
    axis, axis_fraction = learned_axis(model)
    sigma_hat = float(model.learned_scale())
    has_model_posterior = model.decoder_kind != "F"
    prior = _UniformLogProb(dim)
    columns: dict[str, list[Tensor]] = {}
    for start in range(0, y.shape[0], _EVAL_CHUNK):
        ys = y[start : start + _EVAL_CHUNK]
        posterior = model.posterior(ys)
        z, log_q = sample_with_log_prob(
            posterior, torch.Size([ELBO_DRAWS]), reparameterize=False
        )
        recon = model.log_likelihood(ys, z).mean(0)
        if model.family == "tnbbeta":
            assert log_q is not None
            kl = (log_q + sphere_log_area(dim)).mean(0)
        else:
            kl = kl_divergence(posterior, HypersphericalUniform(dim - 1))
        weighted = importance_weighted_metrics(
            _Bound(model, posterior, prior),
            ys,
            num_samples=IWAE_K,
            chunk_size=IWAE_K,
        )
        latitude = (z * axis).sum(-1)
        perp = z - latitude[..., None] * axis
        unit = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        nan = torch.full_like(ys, float("nan"))
        centre = posterior_centre(model.family, posterior)
        pieces = {
            "abs_cos_centre": (centre * axis).sum(-1).abs(),
            "elbo": recon - kl,
            "recon": recon,
            "kl": kl,
            "iwae": weighted["ll"],
            "ring_r": unit.mean(0).norm(dim=-1),
            "w_std_q": latitude.std(0),
            "log_p_model": nan,
            "w_std_model": nan,
            **_fitted_parameters(posterior, ys.shape[0]),
        }
        if has_model_posterior:
            pieces["log_p_model"] = log_evidence(
                ys, dim, sigma_hat, model.latitude_mean
            )
            pieces["w_std_model"] = latitude_std(
                ys, dim, sigma_hat, model.latitude_mean
            )
        for key, value in pieces.items():
            columns.setdefault(key, []).append(value)
    arrays = {key: torch.cat(value).numpy() for key, value in columns.items()}
    arrays["y"] = y.numpy()
    arrays["log_p_true"] = log_p_true.numpy()
    summary = summarize_run(arrays, model, sigma, axis, axis_fraction)
    summary["info_nats"] = information_nats(y, dim, sigma)
    summary["collapse_gap"] = float(kl_uniform_to_posterior(y, dim, sigma).mean())
    return arrays, summary


def summarize_run(
    arrays: dict[str, np.ndarray],
    model: LinearRingVAE,
    sigma: float,
    axis: Tensor,
    axis_fraction: float,
) -> dict[str, Any]:
    """Scalar summaries of one run's per-``y`` arrays."""
    gap = arrays["log_p_true"] - arrays["elbo"]
    iwae_gap = arrays["log_p_true"] - arrays["iwae"]
    cosine = abs(float(axis[0]))
    summary: dict[str, Any] = {
        "gap_elbo": float(gap.mean()),
        "gap_elbo_median": float(np.median(gap)),
        "gap_elbo_se": float(gap.std(ddof=1) / math.sqrt(len(gap))),
        "gap_iwae": float(iwae_gap.mean()),
        "elbo_mean": float(arrays["elbo"].mean()),
        "log_p_true_mean": float(arrays["log_p_true"].mean()),
        "recon_mean": float(arrays["recon"].mean()),
        "kl_mean": float(arrays["kl"].mean()),
        "sigma_hat": float(model.learned_scale()),
        "sigma_ratio": float(model.learned_scale()) / sigma,
        "axis": [float(v) for v in axis],
        "abs_cos_axis": cosine,
        "aligned": bool(cosine >= ALIGNED_COS),
        "axis_fraction": axis_fraction,
        "ring_r_median": float(np.median(arrays["ring_r"])),
        "abs_cos_centre_mean": float(arrays["abs_cos_centre"].mean()),
        "w_std_q_mean": float(arrays["w_std_q"].mean()),
    }
    if model.decoder_kind != "F":
        misfit = arrays["log_p_true"] - arrays["log_p_model"]
        summary["model_misfit"] = float(misfit.mean())
        summary["kl_to_model_posterior"] = float(
            (arrays["log_p_model"] - arrays["elbo"]).mean()
        )
        summary["w_std_model_mean"] = float(arrays["w_std_model"].mean())
    if model.decoder_kind == "Lb":
        summary["bias"] = float(model.bias)
    for key in ("p", "q", "epsilon", "kappa"):
        values = arrays[key]
        if np.isfinite(values).any():
            summary[f"{key}_median"] = float(np.nanmedian(values))
    if np.isfinite(arrays["epsilon"]).any():
        summary["m_median"] = (
            float(np.nanmedian(arrays["epsilon"])) - (model.dim - 1) / 2
        )
    return summary


def learned_axis(model: LinearRingVAE) -> tuple[Tensor, float]:
    """The decoder's axis and how single-index it is.

    ``L`` / ``Lb`` / ``M``: the learned unit vector (fraction 1). ``F``: the leading
    eigenvector of ``E_z[grad f grad f^T]`` over uniform ``z`` and ``lambda_1 / trace``
    (sign fixed so the largest component is positive).

    Args:
        model: The (float64, CPU) model.

    Returns:
        ``(axis, fraction)`` with ``axis`` a unit vector of shape ``(d,)``.
    """
    if model.decoder_kind != "F":
        return model.axis().detach().double(), 1.0
    generator = torch.Generator().manual_seed(5)
    z = torch.randn(
        JACOBIAN_POINTS, model.dim, generator=generator, dtype=torch.float64
    )
    z = (z / z.norm(dim=-1, keepdim=True)).requires_grad_(True)
    with torch.enable_grad():
        gradient = torch.autograd.grad(model.decode(z).sum(), z)[0]
    gram = gradient.T @ gradient / z.shape[0]
    values, vectors = torch.linalg.eigh(gram)
    axis = vectors[:, -1]
    if axis[torch.argmax(axis.abs())] < 0:
        axis = -axis
    return axis, float(values[-1] / values.sum().clamp_min(1e-300))


# ---------------------------------------------------------------------------- summarize


def load_runs(out_dir: Path) -> Runs:
    """All finished run summaries under ``out_dir``, keyed by cell."""
    runs: Runs = {}
    for path in sorted(out_dir.glob("*/d*_s*/*/s*.json")):
        record = json.loads(path.read_text())
        key = (record["decoder"], record["dim"], record["sigma_true"], record["arm"])
        runs.setdefault(key, {})[record["seed"]] = record
    return runs


def valid(runs: dict[int, dict[str, Any]], key: str = "gap_elbo") -> dict[int, float]:
    """Seed -> finite value of ``key`` for non-diverged runs."""
    return {
        s: float(r[key])
        for s, r in runs.items()
        if not r["diverged"] and math.isfinite(r.get(key, float("nan")))
    }


def wilcoxon_greater(diff: np.ndarray) -> float:
    """One-sided (greater) Wilcoxon signed-rank p-value (scipy) of paired diffs."""
    return float(stats.wilcoxon(diff, alternative="greater").pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def wilcoxon_greater_enumerated(diff: np.ndarray) -> float:
    """The same exact p-value by enumerating all ``2^n`` sign patterns."""
    diff = diff[diff != 0]
    ranks = stats.rankdata(np.abs(diff))
    observed = ranks[diff > 0].sum()
    count = 0
    total = 0
    for signs in itertools.product((0, 1), repeat=len(diff)):
        total += 1
        if (ranks * np.array(signs)).sum() >= observed - 1e-9:
            count += 1
    return count / total


def confirmatory_tests(runs: Runs) -> list[dict[str, Any]]:
    """The 24 tests of decoder ``L`` with Holm step-down and the margin rule."""
    tests: list[dict[str, Any]] = []
    for dim, sigma in itertools.product(DIMS, SIGMAS):
        tnb = valid(runs.get(("L", dim, sigma, "tnbbeta"), {}))
        for other in PRIMARY_COMPARATORS:
            cap = valid(runs.get(("L", dim, sigma, other), {}))
            seeds = sorted(set(tnb) & set(cap))
            if len(seeds) < 5:
                continue
            diff = np.array([cap[s] - tnb[s] for s in seeds])
            tests.append(
                {
                    "dim": dim,
                    "sigma": sigma,
                    "other": other,
                    "n": len(seeds),
                    "diff": diff,
                    "p": wilcoxon_greater(diff),
                    "p_enum": wilcoxon_greater_enumerated(diff),
                }
            )
    family = 2 * len(DIMS) * len(SIGMAS)
    order = np.argsort([t["p"] for t in tests], kind="stable")
    still = True
    for rank, index in enumerate(order):
        test = tests[index]
        test["threshold"] = HOLM_ALPHA / (family - rank)
        still = still and test["p"] <= test["threshold"]
        test["holm_reject"] = bool(still)
        test["margin"] = bool(test["diff"].mean() >= MARGIN_NATS)
        test["confirmed"] = bool(test["holm_reject"] and test["margin"])
    return tests


def summarize(out_dir: Path) -> None:
    """Prints the pre-registered analysis as markdown from the saved run summaries."""
    runs = load_runs(out_dir)
    if not runs:
        print(f"no runs found under {out_dir}")
        return
    print("\n## Per-cell test ELBO gap (nats; lower is better)\n")
    _print_arm_table(runs)
    print(
        "\n## Confirmatory tests, decoder L "
        "(one-sided paired Wilcoxon, Holm 0.05, 24)\n"
    )
    tests = confirmatory_tests(runs)
    _print_confirmatory(tests)
    print(
        "\n## Descriptive trend in d (mean paired gap difference, "
        "comparator - tnbbeta)\n"
    )
    _print_trends(runs)
    print("\n## Per-seed ELBO gap\n")
    _print_per_seed(runs, "gap_elbo")
    print(
        "\n## Per-seed |cos(u, a)| (symmetry check: not identified, see Deviations)\n"
    )
    _print_per_seed(runs, "abs_cos_axis")
    print("\n## Per-seed encoder-decoder consistency: mean |centre . u|\n")
    _print_per_seed(runs, "abs_cos_centre_mean")
    print("\n## Per-seed IWAE (K=64) gap\n")
    _print_per_seed(runs, "gap_iwae")
    print("\n## Exact decomposition: gap = model misfit + KL(q || model posterior)\n")
    _print_decomposition(runs)
    print("\n## Diagnostics: sigma_hat, alignment, ring statistic, ring thickness\n")
    _print_diagnostics(runs)
    print("\n## Fitted posterior parameters\n")
    _print_parameters(runs)
    print("\n## Descriptive contrasts (sigma = 0.2; paired by seed)\n")
    _print_contrasts(runs)


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _cells(runs: Runs) -> list[tuple[str, int, float, str]]:
    return sorted(
        runs,
        key=lambda k: (DECODERS.index(k[0]), k[2], k[1], ARMS.index(k[3])),
    )


def _print_arm_table(runs: Runs) -> None:
    print(
        "| decoder | d | sigma | arm | seeds (valid) | gap mean +- sd | median | "
        "min / max | IWAE gap mean | collapse gap | I(z;y) |"
    )
    print("|" + "---|" * 11)
    for decoder, dim, sigma, arm in _cells(runs):
        group = runs[(decoder, dim, sigma, arm)]
        gap = np.array(list(valid(group).values()))
        iw = np.array(list(valid(group, "gap_iwae").values()))
        info = np.mean([r["info_nats"] for r in group.values() if "info_nats" in r])
        collapse = np.mean(
            [r["collapse_gap"] for r in group.values() if "collapse_gap" in r]
        )
        if len(gap) == 0:
            print(
                f"| {decoder} | {dim} | {sigma} | {arm} | 0/{len(group)} | "
                "- | - | - | - | - | - |"
            )
            continue
        print(
            f"| {decoder} | {dim} | {sigma} | {arm} | {len(gap)}/{len(group)} | "
            f"{gap.mean():.4f} +- {_sd(gap):.4f} | {np.median(gap):.4f} | "
            f"{gap.min():.4f} / {gap.max():.4f} | {iw.mean():.4f} | "
            f"{collapse:.4f} | {info:.3f} |"
        )


def _print_per_seed(runs: Runs, key: str) -> None:
    groups = sorted(
        {(k[0], k[1], k[2]) for k in runs},
        key=lambda g: (DECODERS.index(g[0]), g[2], g[1]),
    )
    for decoder, dim, sigma in groups:
        arms = [a for a in ARMS if (decoder, dim, sigma, a) in runs]
        print(f"\n{decoder}, d = {dim}, sigma = {sigma}\n")
        print("| seed | " + " | ".join(arms) + " |")
        print("|" + "---|" * (1 + len(arms)))
        seeds = sorted({s for a in arms for s in runs[(decoder, dim, sigma, a)]})
        for seed in seeds:
            cells = []
            for arm in arms:
                r = runs[(decoder, dim, sigma, arm)].get(seed)
                if r is None:
                    cells.append("-")
                elif r["diverged"]:
                    cells.append("DIVERGED")
                else:
                    cells.append(f"{r[key]:.4f}")
            print(f"| {seed} | " + " | ".join(cells) + " |")


def _print_confirmatory(tests: list[dict[str, Any]]) -> None:
    print(
        "| d | sigma | comparator | n | mean paired diff (comparator - tnbbeta) | "
        "seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | "
        "Holm reject | margin >= 0.05 | confirmed |"
    )
    print("|" + "---|" * 12)
    for test in sorted(tests, key=lambda t: (t["sigma"], t["dim"], t["other"])):
        wins = int((test["diff"] > 0).sum())
        print(
            f"| {test['dim']} | {test['sigma']} | {test['other']} | {test['n']} | "
            f"{test['diff'].mean():.4f} | {wins}/{test['n']} | {test['p']:.5f} | "
            f"{test['p_enum']:.5f} | {test['threshold']:.5f} | "
            f"{test['holm_reject']} | {test['margin']} | {test['confirmed']} |"
        )
        if abs(test["p"] - test["p_enum"]) > 1e-9:
            print(f"WARNING: scipy and enumerated p differ for d={test['dim']}")
    print(
        f"\nConfirmed (Holm and margin): "
        f"{sum(t['confirmed'] for t in tests)} of {len(tests)}."
    )
    print("\n| d | sigma | confirmed against both comparators |")
    print("|---|---|---|")
    for dim, sigma in itertools.product(DIMS, SIGMAS):
        mine = [t for t in tests if t["dim"] == dim and t["sigma"] == sigma]
        if len(mine) == 2:
            print(f"| {dim} | {sigma} | {all(t['confirmed'] for t in mine)} |")


def _print_trends(runs: Runs) -> None:
    print(
        "| decoder | sigma | comparator | "
        + " | ".join(f"d={d}" for d in DIMS)
        + " | slope vs log2(d-1) |"
    )
    print("|" + "---|" * (4 + len(DIMS)))
    for decoder in DECODERS:
        sigmas = SIGMAS if decoder == "L" else (CONTRAST_SIGMA,)
        for sigma, other in itertools.product(sigmas, PRIMARY_COMPARATORS):
            means = []
            for dim in DIMS:
                tnb = valid(runs.get((decoder, dim, sigma, "tnbbeta"), {}))
                cap = valid(runs.get((decoder, dim, sigma, other), {}))
                seeds = sorted(set(tnb) & set(cap))
                means.append(
                    np.mean([cap[s] - tnb[s] for s in seeds]) if seeds else np.nan
                )
            ok = np.isfinite(means)
            slope = (
                float(
                    np.polyfit(np.log2(np.array(DIMS)[ok] - 1), np.array(means)[ok], 1)[
                        0
                    ]
                )
                if ok.sum() >= 2
                else float("nan")
            )
            cells = " | ".join("-" if np.isnan(m) else f"{m:.4f}" for m in means)
            print(f"| {decoder} | {sigma} | {other} | {cells} | {slope:.4f} |")


def _print_decomposition(runs: Runs) -> None:
    print(
        "| decoder | d | sigma | arm | gap | model misfit (log p_true - log p_model) | "
        "KL(q || p_model(.|y)) | I(z;y) |"
    )
    print("|" + "---|" * 8)
    for decoder, dim, sigma, arm in _cells(runs):
        group = [
            r for r in runs[(decoder, dim, sigma, arm)].values() if not r["diverged"]
        ]
        if not group or "model_misfit" not in group[0]:
            continue
        gap, mis, kl, info = (
            np.mean([r[k] for r in group])
            for k in ("gap_elbo", "model_misfit", "kl_to_model_posterior", "info_nats")
        )
        print(
            f"| {decoder} | {dim} | {sigma} | {arm} | {gap:.4f} | {mis:.4f} | "
            f"{kl:.4f} | {info:.3f} |"
        )


def _print_diagnostics(runs: Runs) -> None:
    print(
        "| decoder | d | sigma | arm | diverged | sigma_hat / sigma (mean +- sd) | "
        "|cos(u, a)| median [min-max] (symmetry check) | "
        "mean |centre . u| (seed mean) | axis eigen-fraction (F) | "
        "ring R (seed median [min-max]) | std(u.z) q / exact model (seed mean) |"
    )
    print("|" + "---|" * 10)
    for decoder, dim, sigma, arm in _cells(runs):
        group = runs[(decoder, dim, sigma, arm)]
        ok = [r for r in group.values() if not r["diverged"]]
        if not ok:
            continue
        ratio = np.array([r["sigma_ratio"] for r in ok])
        cos = np.array([r["abs_cos_axis"] for r in ok])
        ring = np.array([r["ring_r_median"] for r in ok])
        frac = np.mean([r["axis_fraction"] for r in ok])
        centre_cos = np.mean([r["abs_cos_centre_mean"] for r in ok])
        q_std = np.mean([r["w_std_q_mean"] for r in ok])
        m_std = (
            np.mean([r["w_std_model_mean"] for r in ok])
            if "w_std_model_mean" in ok[0]
            else float("nan")
        )
        print(
            f"| {decoder} | {dim} | {sigma} | {arm} | "
            f"{len(group) - len(ok)}/{len(group)} | "
            f"{ratio.mean():.3f} +- {_sd(ratio):.3f} | "
            f"{np.median(cos):.3f} [{cos.min():.3f}-{cos.max():.3f}] | "
            f"{centre_cos:.3f} | {frac:.3f} | "
            f"{np.median(ring):.3f} [{ring.min():.3f}-{ring.max():.3f}] | "
            f"{q_std:.4f} / {m_std:.4f} |"
        )


def _print_parameters(runs: Runs) -> None:
    print("| decoder | d | sigma | arm | p | q | epsilon | m = eps - (d-1)/2 | kappa |")
    print("|" + "---|" * 9)
    for decoder, dim, sigma, arm in _cells(runs):
        ok = [r for r in runs[(decoder, dim, sigma, arm)].values() if not r["diverged"]]

        def med(key: str, ok: list[dict[str, Any]] = ok) -> str:
            values = [r[key] for r in ok if key in r]
            return f"{np.median(values):.3g}" if values else "-"

        print(
            f"| {decoder} | {dim} | {sigma} | {arm} | {med('p_median')} | "
            f"{med('q_median')} | "
            f"{med('epsilon_median')} | {med('m_median')} | {med('kappa_median')} |"
        )


def _paired(
    runs: Runs, first: tuple[str, int, float, str], second: tuple[str, int, float, str]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    ga = valid(runs.get(first, {}))
    gb = valid(runs.get(second, {}))
    seeds = sorted(set(ga) & set(gb))
    return seeds, np.array([ga[s] for s in seeds]), np.array([gb[s] for s in seeds])


def _print_contrasts(runs: Runs) -> None:
    print(
        "| comparison | d | n | mean gap (A) | mean gap (B) | "
        "mean paired diff (A - B) | seeds A better |"
    )
    print("|" + "---|" * 7)
    pairs = [
        ("tnbbeta", "vmf"),
        ("tnbbeta", "power_spherical"),
        ("tnbbeta", "tnbbeta_init0"),
        ("vmf", "power_spherical"),
    ]
    for decoder in DECODERS:
        for a, b in pairs:
            for dim in DIMS:
                seeds, ka, kb = _paired(
                    runs,
                    (decoder, dim, CONTRAST_SIGMA, a),
                    (decoder, dim, CONTRAST_SIGMA, b),
                )
                if seeds:
                    print(
                        f"| {decoder}: {a} vs {b} | {dim} | {len(seeds)} | "
                        f"{ka.mean():.4f} | "
                        f"{kb.mean():.4f} | {(ka - kb).mean():.4f} | "
                        f"{int((ka < kb).sum())}/{len(seeds)} |"
                    )
    for other in ("Lb", "M", "F"):
        for arm in ARMS:
            for dim in DIMS:
                seeds, ka, kb = _paired(
                    runs,
                    (other, dim, CONTRAST_SIGMA, arm),
                    ("L", dim, CONTRAST_SIGMA, arm),
                )
                if seeds:
                    print(
                        f"| {arm}: {other} vs L | {dim} | {len(seeds)} | "
                        f"{ka.mean():.4f} | {kb.mean():.4f} | "
                        f"{(ka - kb).mean():.4f} | "
                        f"{int((ka < kb).sum())}/{len(seeds)} |"
                    )


# ----------------------------------------------------------------------------- helpers


class _UniformLogProb:
    """The uniform-sphere prior's ``log_prob`` (a constant) for the IWAE machinery."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def log_prob(self, value: Tensor) -> Tensor:
        return torch.full(
            value.shape[:-1], -sphere_log_area(self.dim), dtype=value.dtype
        )


class _Bound:
    """Adapts a model to the ``importance_weighted_metrics`` API."""

    def __init__(
        self, model: LinearRingVAE, posterior: Distribution, prior: _UniformLogProb
    ) -> None:
        self.model, self.posterior, self.prior = model, posterior, prior

    def posterior_and_prior(self, y: Tensor) -> tuple[Distribution, _UniformLogProb]:
        del y
        return self.posterior, self.prior

    def log_likelihood(self, y: Tensor, z: Tensor) -> Tensor:
        return self.model.log_likelihood(y, z)


def _fitted_parameters(posterior: Distribution, num: int) -> dict[str, Tensor]:
    """Per-``y`` fitted parameters (NaN where the family has none)."""
    nan = torch.full((num,), float("nan"), dtype=torch.float64)
    out = {"p": nan, "q": nan, "epsilon": nan, "kappa": nan}
    if hasattr(posterior, "epsilon"):
        out["p"] = posterior.p  # pyright: ignore[reportAttributeAccessIssue]
        out["q"] = posterior.q  # pyright: ignore[reportAttributeAccessIssue]
        out["epsilon"] = posterior.epsilon  # pyright: ignore[reportAttributeAccessIssue]
    elif hasattr(posterior, "kappa"):
        out["kappa"] = posterior.kappa.reshape(num)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        out["kappa"] = posterior.scale.reshape(num)  # pyright: ignore[reportAttributeAccessIssue]
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
