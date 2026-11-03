"""Does TNBBeta's ring-posterior advantage scale with the ambient dimension?

Pre-registration:
``writeup/results/ring_dimension_scaling_preregistration_2026-11-02.md``
(task, conditions, arms, training, metrics, tests, margin and risks are fixed there;
deviations are appended at its end). Task: :mod:`tnbbeta_vae.data.ring_dimension`.

Usage (one run per call; the shell loops over the grid)::

    uv run python -m apps.synthetic.ring_dimension_scaling train --condition B \
        --dim 9 --arm tnbbeta --seed 0 --out-dir ring_recovery/dimension_scaling
    uv run python -m apps.synthetic.ring_dimension_scaling info \
        --out-dir ring_recovery/dimension_scaling
    uv run python -m apps.synthetic.ring_dimension_scaling summarize \
        --out-dir ring_recovery/dimension_scaling

Each run writes ``<out-dir>/<condition>/d<dim>/<arm>/s<seed>.{json,npz,pt}``.
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

from tnbbeta_vae.data.ring_dimension import (
    DIMENSIONS,
    SIGMA,
    DimensionRingTask,
    dimension_ring_task,
    information_summary,
)
from tnbbeta_vae.data.ring_posterior import exact_kl, sphere_log_area
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

CONDITIONS = ("A", "B", "C")
ARMS = (
    "vmf",
    "power_spherical",
    "tnbbeta",
    "tnbbeta_init0",
    "tnbbeta_epsinit",
    "vmf_untuned",
    "power_spherical_untuned",
)
C_ARMS = ("vmf", "power_spherical", "tnbbeta", "tnbbeta_init0")
COMPARATORS = ("vmf", "power_spherical")
NUM_SEEDS = 10
MARGIN_NATS = 0.05
HOLM_ALPHA = 0.05
HOLM_FAMILY_SIZE = 8
TEST_SEED = 424242
ELBO_DRAWS = 256
IWAE_K = 64
KL_DRAWS = 2000
JACOBIAN_POINTS = 2000
BOOTSTRAPS = 2000
_HIDDEN = [128, 128]
_LINK_HIDDEN = 32
_EPSILON_FLOOR = 1e-4
_EVAL_CHUNK = 250
_NUM_FEATURES = 3


class DimensionRingVAE(nn.Module):
    """A study-local VAE: MLP encoder + posterior head and a condition-specific decoder.

    Attributes:
        condition: ``"A"`` (true frozen decoder, sigma fixed), ``"B"`` (free MLP
            decoder ``[d, 128, 128, 3]``, learned sigma) or ``"C"`` (learned
            single-index decoder).
        dim: Ambient dimension ``d`` of the sphere ``S^(d-1)``.
        family: Latent family of the posterior head.
    """

    def __init__(
        self, condition: str, dim: int, arm: str, task: DimensionRingTask
    ) -> None:
        """Builds encoder, decoder, learned sigma and head (in that RNG order).

        The head is built last so the encoder and decoder initial weights of one seed
        are identical across arms (only the head differs in size).

        Args:
            condition: ``"A"``, ``"B"`` or ``"C"``.
            dim: Ambient dimension.
            arm: One of :data:`ARMS`.
            task: The task (its ``h`` is the frozen decoder in condition A).
        """
        super().__init__()
        spec = arm_spec(arm, dim)
        self.condition = condition
        self.dim = dim
        self.family: LatentFamily = spec["family"]
        self.task = task
        self.encoder = mlp_stack([_NUM_FEATURES, *_HIDDEN])
        if condition == "B":
            self.decoder_net = nn.Sequential(
                mlp_stack([dim, *_HIDDEN]), nn.Linear(_HIDDEN[-1], _NUM_FEATURES)
            )
        elif condition == "C":
            self.axis_raw = nn.Parameter(torch.randn(dim))
            self.link = nn.Sequential(
                nn.Linear(1, _LINK_HIDDEN),
                nn.Tanh(),
                nn.Linear(_LINK_HIDDEN, _LINK_HIDDEN),
                nn.Tanh(),
                nn.Linear(_LINK_HIDDEN, _NUM_FEATURES),
            )
        if condition == "A":
            self.register_buffer("fixed_scale", torch.tensor(SIGMA))
        else:
            self.learned_scale = LearnedLikelihoodScale(SIGMA)
        self.posterior_head = nn.Linear(_HIDDEN[-1], head_size(self.family, dim))
        self._init_head(spec)

    def sigma(self) -> Tensor:
        """The likelihood standard deviation (fixed 0.2 in A, learned otherwise)."""
        if self.condition == "A":
            return self.fixed_scale  # pyright: ignore[reportReturnType]
        return self.learned_scale()

    def posterior(self, x: Tensor) -> Distribution:
        """``q(z | x)`` for a batch."""
        raw = self.posterior_head(self.encoder(x))
        if self.family == "tnbbeta":
            return tnbbeta_posterior(raw, self.dim, epsilon_floor=_EPSILON_FLOOR)
        posterior = posterior_from_raw(self.family, raw, self.dim)
        if self.family == "vmf":
            return ExactVonMisesFisher(posterior.loc, posterior.scale)  # pyright: ignore[reportAttributeAccessIssue]
        return posterior

    def decode(self, z: Tensor) -> Tensor:
        """The decoder mean for ``z`` of shape ``(..., d)``."""
        if self.condition == "A":
            return self.task.features(z[..., 0])
        if self.condition == "B":
            return self.decoder_net(z)
        return self.link((z * self.axis()).sum(-1, keepdim=True))

    def axis(self) -> Tensor:
        """The learned unit axis of a ``C`` decoder, ``(d,)``."""
        return nn.functional.normalize(self.axis_raw, dim=-1)

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """``log p(x | z)``, shape ``z.shape[:-1]``."""
        return pixel_log_likelihood(x, self.decode(z), "gaussian", self.sigma())

    def negative_elbo(self, x: Tensor, kl_weight: float, num_samples: int) -> Tensor:
        """The training loss: mean over the batch of ``-(log p(x|z) - w * KL)``."""
        posterior = self.posterior(x)
        prior = standard_prior(self.family, self.dim, x.device)
        terms = monte_carlo_elbo(
            x,
            posterior,
            prior,
            self.decode,
            self.sigma(),
            num_samples,
            analytic_kl=self.family != "tnbbeta",
        )
        return -(terms["log_likelihood"] - kl_weight * terms["kl"]).mean()

    def _init_head(self, spec: dict[str, Any]) -> None:
        """Applies the arm's head initialisation (kappa / p / epsilon biases)."""
        head, dim = self.posterior_head, self.dim
        with torch.no_grad():
            if self.family == "vmf" and spec["kappa_init"] is not None:
                head.bias[-1] = vmf_kappa_inverse(spec["kappa_init"])
            if self.family == "power_spherical" and spec["kappa_init"] is not None:
                head.bias[-1] = power_spherical_kappa_inverse(spec["kappa_init"])
        if self.family == "tnbbeta":
            init_tnbbeta_p_bias(head, dim, offset=spec["p_init_offset"])
            if spec["epsilon_init"] is not None:
                target = spec["epsilon_init"] - _EPSILON_FLOOR
                with torch.no_grad():
                    head.bias[dim + 2] = math.log(math.expm1(target))


class ExactVonMisesFisher(VonMisesFisher):
    """vMF whose rejection sampler always uses Wood's exact envelope parameter ``b``.

    The library sampler (a port of the S-VAE reference code) switches ``b`` to the
    Taylor value ``(m - 1) / (4 kappa)`` for ``kappa > 11`` while ``a`` and ``d`` keep
    the exact-``b`` formulas; the mismatch biases the draws for ``kappa`` of about
    10-40 at ``m >= 9`` (e.g. ``E[cos]`` 0.608 instead of 0.627 at ``m = 17,
    kappa = 17``). Here ``b = (m - 1) / (c + 2 kappa)`` (the exact value, without
    the cancellation of ``(c - 2 kappa) / (m - 1)``) is used for every kappa. ``m = 3``
    uses the library's closed-form sampler, which is exact.
    """

    def _sample_w_rej(self, shape: torch.Size) -> Tensor:
        c = torch.sqrt(4 * self.scale**2 + (self._m - 1) ** 2)
        b = (self._m - 1) / (c + 2 * self.scale)
        a = (self._m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self._m - 1) * math.log(self._m - 1)
        _, w = self._while_loop(b, a, d, shape, k=self.k)
        return w


def arm_spec(arm: str, dim: int) -> dict[str, Any]:
    """Model settings of an arm: family and initial kappa / p offset / epsilon."""
    base: dict[str, Any] = {
        "kappa_init": None,
        "p_init_offset": P_RAW_INIT,
        "epsilon_init": None,
    }
    if arm == "vmf":
        return {**base, "family": "vmf", "kappa_init": float(dim)}
    if arm == "power_spherical":
        return {**base, "family": "power_spherical", "kappa_init": float(dim)}
    if arm == "vmf_untuned":
        return {**base, "family": "vmf"}
    if arm == "power_spherical_untuned":
        return {**base, "family": "power_spherical"}
    if arm == "tnbbeta":
        return {**base, "family": "tnbbeta"}
    if arm == "tnbbeta_init0":
        return {**base, "family": "tnbbeta", "p_init_offset": 0.0}
    if arm == "tnbbeta_epsinit":
        return {**base, "family": "tnbbeta", "epsilon_init": (dim - 1) / 2}
    raise ValueError(f"unknown arm {arm!r}.")


def run_path(out_dir: Path, condition: str, dim: int, arm: str, seed: int) -> Path:
    """The stem (no suffix) of one run's output files."""
    return out_dir / condition / f"d{dim}" / arm / f"s{seed}"


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``train``, ``info`` or ``summarize``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--condition", choices=list(CONDITIONS), required=True)
    train.add_argument("--dim", type=int, required=True)
    train.add_argument("--arm", choices=list(ARMS), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--num-train", type=int, default=20_000)
    train.add_argument("--num-test", type=int, default=2_000)
    train.add_argument("--num-elbo-samples", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--device", default=None, help="Default: select_device().")
    train.add_argument("--out-dir", type=Path, required=True)
    info = sub.add_parser("info")
    info.add_argument("--out-dir", type=Path, required=True)
    info.add_argument("--num", type=int, default=20_000)
    summ = sub.add_parser("summarize")
    summ.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "train":
        run_train(args)
    elif args.command == "info":
        run_info(args.out_dir, args.num)
    else:
        summarize(args.out_dir)


def run_info(out_dir: Path, num: int) -> None:
    """Writes ``information.json``: the per-d information in ``x`` about ``z``."""
    result = {
        str(d): information_summary(dimension_ring_task(d), num) for d in DIMENSIONS
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "information.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def run_train(args: argparse.Namespace) -> None:
    """Trains one (condition, d, arm, seed) on the GPU, evaluates on the CPU."""
    device = select_device(args.device)
    started = time.time()
    task = dimension_ring_task(args.dim)
    train_gen = torch.Generator().manual_seed(100 + args.seed)
    train_x, _ = task.sample(args.num_train, train_gen)
    torch.manual_seed(args.seed)
    model = DimensionRingVAE(args.condition, args.dim, args.arm, task)
    model.to(device)
    x_dev = train_x.to(device)
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
            loss = model.negative_elbo(x_dev[index], kl_weight, args.num_elbo_samples)
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
        test_x, _ = task.sample(args.num_test, test_gen)
        torch.manual_seed(31337)
        arrays, summary = evaluate(model, task, test_x.double())
        diverged = not all(np.isfinite(arrays[k]).all() for k in ("elbo", "iwae"))
    spec = arm_spec(args.arm, args.dim)
    summary.update(
        {
            "condition": args.condition,
            "dim": args.dim,
            "arm": args.arm,
            "seed": args.seed,
            "family": spec["family"],
            "kappa_init": spec["kappa_init"],
            "p_init_offset": spec["p_init_offset"],
            "epsilon_init": spec["epsilon_init"],
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
    stem = run_path(args.out_dir, args.condition, args.dim, args.arm, args.seed)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(summary))
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # pyright: ignore[reportArgumentType]
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    gap = summary.get("gap_elbo", float("nan"))
    print(
        f"{args.condition} d{args.dim} {args.arm} s{args.seed}: gap {gap:.4f} "
        f"sigma {summary.get('sigma', float('nan')):.4f} diverged {diverged} "
        f"({summary['total_seconds']:.0f}s)",
        flush=True,
    )


@torch.no_grad()
def evaluate(
    model: DimensionRingVAE, task: DimensionRingTask, x: Tensor
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per-x ELBO / IWAE gaps, exact KL (A), ring statistic and fitted parameters.

    Args:
        model: The (float64, CPU) model.
        task: The task (exact evidence and posterior).
        x: Test observations ``(n, 3)``, float64.

    Returns:
        ``(arrays, summary)``: per-x numpy arrays and a JSON-ready scalar summary.
    """
    dim = model.dim
    log_ref = task.log_evidence(x)
    axis, single_index = decoder_axis(model)
    prior = _UniformLogProb(dim)
    columns: dict[str, list[Tensor]] = {}
    for start in range(0, x.shape[0], _EVAL_CHUNK):
        xs = x[start : start + _EVAL_CHUNK]
        posterior = model.posterior(xs)
        z, log_q = sample_with_log_prob(
            posterior, torch.Size([ELBO_DRAWS]), reparameterize=False
        )
        recon = model.log_likelihood(xs, z).mean(0)
        if model.family == "tnbbeta":
            assert log_q is not None
            kl = (log_q + sphere_log_area(dim)).mean(0)
        else:
            kl = kl_divergence(posterior, HypersphericalUniform(dim - 1))
        weighted = importance_weighted_metrics(
            _Bound(model, posterior, prior), xs, num_samples=IWAE_K, chunk_size=IWAE_K
        )
        perp = z - (z * axis).sum(-1, keepdim=True) * axis
        unit = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        centre = posterior_centre(model.family, posterior)
        pieces = {
            "elbo": recon - kl,
            "recon": recon,
            "kl": kl,
            "iwae": weighted["ll"],
            "ring_r": unit.mean(0).norm(dim=-1),
            "abs_cos_u": (centre * axis).sum(-1).abs(),
            **_fitted_parameters(posterior, xs.shape[0]),
        }
        if model.condition == "A":
            torch.manual_seed(777 + start)
            kl_exact, kl_se = exact_kl(posterior, task, xs, num_samples=KL_DRAWS)
            pieces["kl_exact"] = kl_exact
            pieces["kl_exact_se"] = kl_se
        for key, value in pieces.items():
            columns.setdefault(key, []).append(value)
    arrays = {key: torch.cat(value).numpy() for key, value in columns.items()}
    arrays["log_ref"] = log_ref.numpy()
    arrays["axis"] = axis.numpy()
    summary = summarize_run(arrays, model, single_index)
    return arrays, summary


def summarize_run(
    arrays: dict[str, np.ndarray], model: DimensionRingVAE, single_index: float
) -> dict[str, Any]:
    """Scalar summaries of one run's per-x arrays."""
    gap = arrays["log_ref"] - arrays["elbo"]
    iwae_gap = arrays["log_ref"] - arrays["iwae"]
    summary: dict[str, Any] = {
        "gap_elbo": float(gap.mean()),
        "gap_elbo_median": float(np.median(gap)),
        "gap_elbo_se": float(gap.std(ddof=1) / math.sqrt(len(gap))),
        "gap_iwae": float(iwae_gap.mean()),
        "elbo_mean": float(arrays["elbo"].mean()),
        "log_ref_mean": float(arrays["log_ref"].mean()),
        "recon_mean": float(arrays["recon"].mean()),
        "kl_mean": float(arrays["kl"].mean()),
        "sigma": float(model.sigma()),
        "single_index_fraction": single_index,
        "axis_cos_true": float(abs(arrays["axis"][0])),
        "ring_r_median": float(np.median(arrays["ring_r"])),
        "ring_r_mean": float(arrays["ring_r"].mean()),
        "abs_cos_u_mean": float(arrays["abs_cos_u"].mean()),
    }
    if "kl_exact" in arrays:
        summary["kl_exact"] = float(arrays["kl_exact"].mean())
        summary["kl_exact_se"] = float(
            np.sqrt((arrays["kl_exact_se"] ** 2).sum()) / len(gap)
        )
    for key in ("p", "q", "epsilon", "kappa"):
        values = arrays[key]
        if np.isfinite(values).any():
            summary[f"{key}_median"] = float(np.nanmedian(values))
    if np.isfinite(arrays["epsilon"]).any():
        summary["m_median"] = (
            float(np.nanmedian(arrays["epsilon"])) - (model.dim - 1) / 2
        )
    return summary


@torch.no_grad()
def decoder_axis(model: DimensionRingVAE) -> tuple[Tensor, float]:
    """The decoder's dominant axis and how single-index it is.

    Condition A: ``e_1`` (fraction 1). Condition C: the learned axis (fraction 1).
    Condition B: the leading eigenvector of ``E_z[J^T J]`` (``J`` the decoder
    Jacobian at uniform ``z``) and the fraction ``lambda_1 / trace`` of it.

    Args:
        model: The (float64, CPU) model.

    Returns:
        ``(axis, fraction)`` with ``axis`` a unit vector ``(d,)``.
    """
    if model.condition == "A":
        axis = torch.zeros(model.dim, dtype=torch.float64)
        axis[0] = 1.0
        return axis, 1.0
    if model.condition == "C":
        return model.axis().double(), 1.0
    generator = torch.Generator().manual_seed(5)
    z = torch.randn(
        JACOBIAN_POINTS, model.dim, generator=generator, dtype=torch.float64
    )
    z = z / z.norm(dim=-1, keepdim=True)
    with torch.enable_grad():
        z = z.requires_grad_(True)
        output = model.decode(z)
        rows = [
            torch.autograd.grad(output[:, k].sum(), z, retain_graph=True)[0]
            for k in range(_NUM_FEATURES)
        ]
    jacobian = torch.stack(rows, dim=1)  # (n, 3, d)
    gram = torch.einsum("nkd,nke->de", jacobian, jacobian) / z.shape[0]
    values, vectors = torch.linalg.eigh(gram)
    axis = vectors[:, -1]
    if axis[torch.argmax(axis.abs())] < 0:
        axis = -axis
    return axis, float(values[-1] / values.sum())


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
        self, model: DimensionRingVAE, posterior: Distribution, prior: _UniformLogProb
    ) -> None:
        self.model, self.posterior, self.prior = model, posterior, prior

    def posterior_and_prior(self, x: Tensor) -> tuple[Distribution, _UniformLogProb]:
        del x
        return self.posterior, self.prior

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        return self.model.log_likelihood(x, z)


def _fitted_parameters(posterior: Distribution, num: int) -> dict[str, Tensor]:
    """Per-x fitted parameters (NaN where the family has none)."""
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


# ---------------------------------------------------------------------------- summarize

Runs = dict[tuple[str, int, str], dict[int, dict[str, Any]]]


def load_runs(out_dir: Path) -> Runs:
    """Loads every run summary: ``(condition, d, arm) -> seed -> record``."""
    runs: Runs = {}
    for path in sorted(out_dir.glob("*/d*/*/s*.json")):
        record = json.loads(path.read_text())
        key = (record["condition"], record["dim"], record["arm"])
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
    """One-sided (greater) exact Wilcoxon signed-rank p-value (scipy)."""
    return float(stats.wilcoxon(diff, alternative="greater").pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def wilcoxon_greater_enumerated(diff: np.ndarray) -> float:
    """The same exact p-value by enumerating all ``2^n`` sign patterns."""
    diff = diff[diff != 0]
    ranks = stats.rankdata(np.abs(diff))
    observed = ranks[diff > 0].sum()
    count = total = 0
    for signs in itertools.product((0, 1), repeat=len(diff)):
        total += 1
        if (ranks * np.array(signs)).sum() >= observed - 1e-9:
            count += 1
    return count / total


def confirmatory_tests(runs: Runs, condition: str) -> list[dict[str, Any]]:
    """The 8 pre-registered tests of one condition, Holm-adjusted."""
    tests: list[dict[str, Any]] = []
    for dim in DIMENSIONS:
        tnb = valid(runs.get((condition, dim, "tnbbeta"), {}))
        for other in COMPARATORS:
            cap = valid(runs.get((condition, dim, other), {}))
            seeds = sorted(set(tnb) & set(cap))
            if len(seeds) < 5:
                continue
            diff = np.array([cap[s] - tnb[s] for s in seeds])
            tests.append(
                {
                    "dim": dim,
                    "other": other,
                    "n": len(seeds),
                    "diff": diff,
                    "p": wilcoxon_greater(diff),
                    "p_enum": wilcoxon_greater_enumerated(diff),
                }
            )
    order = np.argsort([t["p"] for t in tests], kind="stable")
    still = True
    for rank, index in enumerate(order):
        test = tests[index]
        test["threshold"] = HOLM_ALPHA / (HOLM_FAMILY_SIZE - rank)
        still = still and test["p"] <= test["threshold"]
        test["holm_reject"] = bool(still)
        test["margin"] = bool(test["diff"].mean() >= MARGIN_NATS)
        test["confirmed"] = bool(test["holm_reject"] and test["margin"])
    return tests


def trend(
    runs: Runs, condition: str, other: str, key: str = "gap_elbo"
) -> dict[str, Any]:
    """Descriptive trend of ``gap(other) - gap(tnbbeta)`` with ``d`` (A or B)."""
    per_dim: dict[int, dict[int, float]] = {}
    for dim in DIMENSIONS:
        tnb = valid(runs.get((condition, dim, "tnbbeta"), {}), key)
        cap = valid(runs.get((condition, dim, other), {}), key)
        per_dim[dim] = {s: cap[s] - tnb[s] for s in set(tnb) & set(cap)}
    seeds = sorted(set.intersection(*(set(v) for v in per_dim.values())))
    out: dict[str, Any] = {"per_dim": per_dim, "seeds": seeds}
    if len(seeds) < 3:
        return out
    dims = np.array(DIMENSIONS, dtype=float)
    matrix = np.array([[per_dim[d][s] for d in DIMENSIONS] for s in seeds])
    slopes = np.array([np.polyfit(dims, row, 1)[0] for row in matrix])
    rng = np.random.default_rng(0)
    boots = np.array(
        [
            slopes[rng.integers(0, len(slopes), len(slopes))].mean()
            for _ in range(BOOTSTRAPS)
        ]
    )
    pooled_d = np.repeat(dims[None], len(seeds), 0).ravel()
    rho = stats.spearmanr(pooled_d, matrix.ravel())
    out.update(
        {
            "slopes": slopes,
            "slope_mean": float(slopes.mean()),
            "slope_ci": (
                float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5)),
            ),
            "spearman": float(rho.statistic),  # pyright: ignore[reportAttributeAccessIssue]
            "spearman_p": float(rho.pvalue),  # pyright: ignore[reportAttributeAccessIssue]
        }
    )
    return out


def summarize(out_dir: Path) -> None:
    """Prints the pre-registered analysis as markdown from the saved run summaries."""
    runs = load_runs(out_dir)
    if not runs:
        print(f"no runs found under {out_dir}")
        return
    information = out_dir / "information.json"
    if information.exists():
        print("\n## Information in x about z (descriptive, per d)\n")
        _print_information(json.loads(information.read_text()))
    print("\n## Per-arm test ELBO gap (nats; lower is better)\n")
    _print_arm_table(runs)
    for condition in CONDITIONS:
        print(f"\n## Per-seed ELBO gap, condition {condition}\n")
        _print_per_seed(runs, condition, "gap_elbo")
    for condition in ("A", "B"):
        print(f"\n## Confirmatory tests, condition {condition} (Holm 0.05, 8 tests)\n")
        _print_confirmatory(confirmatory_tests(runs, condition))
    print("\n## Descriptive trend of the paired difference with d\n")
    _print_trends(runs)
    for condition in CONDITIONS:
        print(f"\n## Per-seed IWAE (K=64) gap, condition {condition}\n")
        _print_per_seed(runs, condition, "gap_iwae")
    print("\n## Descriptive arm comparisons (mean paired diff = A - B)\n")
    _print_descriptive(runs)
    print("\n## Ring statistic, decoder axis and fitted parameters\n")
    _print_diagnostics(runs)
    print("\n## Condition A: exact KL(q || p) cross-check\n")
    _print_exact_kl(runs)


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _cells(runs: Runs) -> list[tuple[str, int, str]]:
    return [
        (c, d, a)
        for c in CONDITIONS
        for d in DIMENSIONS
        for a in ARMS
        if (c, d, a) in runs
    ]


def _print_information(info: dict[str, dict[str, float]]) -> None:
    print(
        "| d | I(z; x) (nats) | E KL(Uniform to exact posterior) | "
        "posterior std of w | prior std of w |"
    )
    print("|" + "---|" * 5)
    for dim, row in info.items():
        print(
            f"| {dim} | {row['mutual_information']:.3f} | "
            f"{row['kl_uniform_to_posterior']:.3f} | {row['posterior_w_std']:.3f} | "
            f"{row['prior_w_std']:.3f} |"
        )


def _print_arm_table(runs: Runs) -> None:
    print(
        "| cond | d | arm | seeds (valid) | gap mean +- sd | median | min / max | "
        "IWAE gap mean +- sd |"
    )
    print("|" + "---|" * 8)
    for condition, dim, arm in _cells(runs):
        group = runs[(condition, dim, arm)]
        gap = np.array(list(valid(group).values()))
        iw = np.array(list(valid(group, "gap_iwae").values()))
        if len(gap) == 0:
            print(f"| {condition} | {dim} | {arm} | 0/{len(group)} | - | - | - | - |")
            continue
        print(
            f"| {condition} | {dim} | {arm} | {len(gap)}/{len(group)} | "
            f"{gap.mean():.4f} +- {_sd(gap):.4f} | {np.median(gap):.4f} | "
            f"{gap.min():.4f} / {gap.max():.4f} | {iw.mean():.4f} +- {_sd(iw):.4f} |"
        )


def _print_per_seed(runs: Runs, condition: str, key: str) -> None:
    for dim in DIMENSIONS:
        arms = [a for a in ARMS if (condition, dim, a) in runs]
        if not arms:
            continue
        print(f"\n{condition} / d={dim}\n")
        print("| seed | " + " | ".join(arms) + " |")
        print("|" + "---|" * (1 + len(arms)))
        seeds = sorted({s for a in arms for s in runs[(condition, dim, a)]})
        for seed in seeds:
            cells = []
            for arm in arms:
                r = runs[(condition, dim, arm)].get(seed)
                if r is None:
                    cells.append("-")
                elif r["diverged"]:
                    cells.append("DIVERGED")
                else:
                    cells.append(f"{r[key]:.4f}")
            print(f"| {seed} | " + " | ".join(cells) + " |")


def _print_confirmatory(tests: list[dict[str, Any]]) -> None:
    print(
        "| d | comparator | n | mean paired diff (comparator - tnbbeta) | "
        "seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | "
        "Holm reject | "
        "margin >= 0.05 | confirmed |"
    )
    print("|" + "---|" * 11)
    for test in sorted(tests, key=lambda t: (t["dim"], t["other"])):
        wins = int((test["diff"] > 0).sum())
        print(
            f"| {test['dim']} | {test['other']} | {test['n']} | "
            f"{test['diff'].mean():.4f} | {wins}/{test['n']} | {test['p']:.5f} | "
            f"{test['p_enum']:.5f} | {test['threshold']:.5f} | {test['holm_reject']} | "
            f"{test['margin']} | {test['confirmed']} |"
        )
        if abs(test["p"] - test["p_enum"]) > 1e-9:
            print(f"WARNING: scipy and enumerated p differ at d={test['dim']}")
    print(
        f"\nConfirmed advantages: {sum(t['confirmed'] for t in tests)} of {len(tests)}."
    )


def _print_trends(runs: Runs) -> None:
    print(
        "| cond | comparator | metric | mean diff by d (3, 5, 9, 17) | "
        "per-seed OLS slope (per unit d): mean [95% seed-bootstrap] | "
        "Spearman(d, diff), pooled | "
        "descriptive |"
    )
    print("|" + "---|" * 7)
    for condition in CONDITIONS:
        for other in COMPARATORS:
            for key in ("gap_elbo", "gap_iwae"):
                result = trend(runs, condition, other, key)
                if "slope_mean" not in result:
                    continue
                means = ", ".join(
                    f"{np.mean(list(result['per_dim'][d].values())):.3f}"
                    for d in DIMENSIONS
                )
                low, high = result["slope_ci"]
                print(
                    f"| {condition} | {other} | {key} | {means} | "
                    f"{result['slope_mean']:.4f} [{low:.4f}, {high:.4f}] | "
                    f"{result['spearman']:.3f} (p {result['spearman_p']:.3g}) | yes |"
                )


def _print_descriptive(runs: Runs) -> None:
    print(
        "| cond | d | comparison | n | mean gap (A) | mean gap (B) | mean paired diff "
        "(A - B) | seeds A better | IWAE paired diff (A - B) |"
    )
    print("|" + "---|" * 9)
    pairs = [
        ("tnbbeta_init0", "tnbbeta"),
        ("tnbbeta_epsinit", "tnbbeta"),
        ("tnbbeta_epsinit", "vmf"),
        ("tnbbeta_init0", "vmf"),
        ("vmf_untuned", "vmf"),
        ("power_spherical_untuned", "power_spherical"),
        ("vmf", "power_spherical"),
    ]
    for condition in CONDITIONS:
        for dim in DIMENSIONS:
            for first, second in pairs:
                ga = valid(runs.get((condition, dim, first), {}))
                gb = valid(runs.get((condition, dim, second), {}))
                seeds = sorted(set(ga) & set(gb))
                if not seeds:
                    continue
                ka = np.array([ga[s] for s in seeds])
                kb = np.array([gb[s] for s in seeds])
                ia = valid(runs[(condition, dim, first)], "gap_iwae")
                ib = valid(runs[(condition, dim, second)], "gap_iwae")
                iw = [ia[s] - ib[s] for s in seeds if s in ia and s in ib]
                iw_text = f"{np.mean(iw):.4f}" if iw else "-"
                print(
                    f"| {condition} | {dim} | {first} vs {second} | {len(seeds)} | "
                    f"{ka.mean():.4f} | {kb.mean():.4f} | {(ka - kb).mean():.4f} | "
                    f"{int((ka < kb).sum())}/{len(seeds)} | {iw_text} |"
                )


def _print_diagnostics(runs: Runs) -> None:
    print(
        "| cond | d | arm | diverged | sigma (mean +- sd; true 0.2) | "
        "ring R (median over x; seed median [min-max]) | single-index frac | "
        "abs cos(axis, e_1) | "
        "p | q | epsilon | m = eps - (d-1)/2 | kappa |"
    )
    print("|" + "---|" * 13)
    for condition, dim, arm in _cells(runs):
        group = runs[(condition, dim, arm)]
        ok = [r for r in group.values() if not r["diverged"]]
        if not ok:
            print(
                f"| {condition} | {dim} | {arm} | {len(group)}/{len(group)} |"
                + " - |" * 10
            )
            continue
        sigma = np.array([r["sigma"] for r in ok])
        ring = np.array([r["ring_r_median"] for r in ok])
        si = np.array([r["single_index_fraction"] for r in ok])
        cos = np.array([r["axis_cos_true"] for r in ok])

        def med(key: str, ok: list[dict[str, Any]] = ok) -> str:
            values = [r[key] for r in ok if key in r]
            return f"{np.median(values):.3g}" if values else "-"

        print(
            f"| {condition} | {dim} | {arm} | {len(group) - len(ok)}/{len(group)} | "
            f"{sigma.mean():.4f} +- {_sd(sigma):.4f} | {np.median(ring):.3f} "
            f"[{ring.min():.3f}-{ring.max():.3f}] | {np.median(si):.3f} | "
            f"{np.median(cos):.3f} | {med('p_median')} | {med('q_median')} | "
            f"{med('epsilon_median')} | {med('m_median')} | {med('kappa_median')} |"
        )


def _print_exact_kl(runs: Runs) -> None:
    print(
        "| d | arm | ELBO gap mean (seeds) | exact KL(q||p) mean (seeds) | "
        "max abs diff over seeds |"
    )
    print("|" + "---|" * 5)
    for dim in DIMENSIONS:
        for arm in ARMS:
            group = runs.get(("A", dim, arm), {})
            ok = [r for r in group.values() if not r["diverged"] and "kl_exact" in r]
            if not ok:
                continue
            gap = np.array([r["gap_elbo"] for r in ok])
            kl = np.array([r["kl_exact"] for r in ok])
            print(
                f"| {dim} | {arm} | {gap.mean():.4f} | {kl.mean():.4f} | "
                f"{np.abs(gap - kl).max():.4f} |"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
