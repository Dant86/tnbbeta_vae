"""Four-class ring designs with a LEARNED decoder: does TNBBeta give a better bound?

Pre-registration: ``writeup/results/ring_class_designs_learned_decoder_preregistration_
2026-11-01.md`` (settings, decoders, families, training, metrics, tests and decision
rules are fixed there; deviations are appended at its end). Tasks:
:mod:`tnbbeta_vae.data.ring_classes` (design 1 and design 2').

Usage (one run per call; the shell loops over the grid)::

    uv run python -m apps.synthetic.ring_classes_learned train --setting D2l \
        --decoder mlp --arm tnbbeta --seed 0 --out-dir ring_recovery/classes_learned
    uv run python -m apps.synthetic.ring_classes_learned summarize \
        --out-dir ring_recovery/classes_learned

Each run writes ``<out-dir>/<setting>/<decoder>/<arm>/s<seed>.{json,npz,pt}``: the
summary, the per-x arrays and the checkpoint.
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
from scipy import optimize, stats
import torch
from torch import Tensor, nn
from torch.distributions import Distribution, kl_divergence

from tnbbeta_vae.data.ring_classes import (
    NUM_CLASSES,
    OffsetRingClassTask,
    RingClassTask,
    offset_ring_class_task,
    ring_class_task,
)
from tnbbeta_vae.data.ring_posterior import sphere_log_area
from tnbbeta_vae.distributions import HypersphericalUniform
from tnbbeta_vae.models.architectures.mlp import mlp_stack
from tnbbeta_vae.models.heads import (
    P_RAW_INIT,
    LatentFamily,
    head_size,
    init_tnbbeta_p_bias,
    posterior_centre,
    posterior_from_raw,
    standard_prior,
    tnbbeta_posterior,
)
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo, pixel_log_likelihood
from tnbbeta_vae.models.losses.importance_weighted import importance_weighted_metrics
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.models.losses.sampling import sample_with_log_prob
from tnbbeta_vae.training import select_device

SETTINGS = ("D1", "D2g", "D2l")
DECODERS = ("mlp", "single_index")
ARMS = ("vmf", "power_spherical", "tnbbeta", "tnbbeta_init0")
PRIMARY_COMPARATORS = ("vmf", "power_spherical")
SIGMA_TRUE = 0.2
NUM_SEEDS = 10
MARGIN_NATS = 0.05
HOLM_ALPHA = 0.05
HOLM_FAMILY_SIZE = 6
TEST_SEED = 424242
ELBO_DRAWS = 256
IWAE_K = 64
KNN_K = 5
ETA_STARTS = 20
ETA_POINTS = 20_000
ETA_BINS = 40
_HIDDEN = [128, 128]
_LINK_HIDDEN = 32
_EPSILON_FLOOR = 1e-4
_EVAL_CHUNK = 250
_CLASS_SETTINGS = {"D1": "band", "D2g": "class", "D2l": "class"}


class LearnedRingVAE(nn.Module):
    """A study-local VAE: MLP encoder + posterior head and a learned decoder.

    Attributes:
        setting: ``"D1"``, ``"D2g"`` (class is an input to encoder and decoder) or
            ``"D2l"`` (latent class: neither sees it).
        decoder_kind: ``"mlp"`` or ``"single_index"``.
        family: Latent family of the posterior head.
        class_input: Whether the class is an input (``D2g``).
    """

    def __init__(
        self,
        setting: str,
        decoder: str,
        family: LatentFamily,
        p_init_offset: float = P_RAW_INIT,
        likelihood_scale: float = SIGMA_TRUE,
    ) -> None:
        """Builds encoder, decoder, learned sigma and head (in that RNG order).

        The head is built last so the encoder and decoder initial weights of one seed
        are identical across families (only the head differs in size).

        Args:
            setting: ``"D1"``, ``"D2g"`` or ``"D2l"``.
            decoder: ``"mlp"`` or ``"single_index"``.
            family: ``"vmf"``, ``"power_spherical"`` or ``"tnbbeta"``.
            p_init_offset: TNBBeta only: initial raw-p bias of the head.
            likelihood_scale: Starting value of the learned sigma.
        """
        super().__init__()
        self.setting = setting
        self.decoder_kind = decoder
        self.family: LatentFamily = family
        self.class_input = setting == "D2g"
        extra = NUM_CLASSES if self.class_input else 0
        self.encoder = mlp_stack([3 + extra, *_HIDDEN])
        if decoder == "mlp":
            self.decoder_net = nn.Sequential(
                mlp_stack([3 + extra, *_HIDDEN]), nn.Linear(_HIDDEN[-1], 3)
            )
        else:
            self.axis_raw = nn.Parameter(torch.randn(NUM_CLASSES if extra else 1, 3))
            self.link = nn.Sequential(
                nn.Linear(1 + extra, _LINK_HIDDEN),
                nn.Tanh(),
                nn.Linear(_LINK_HIDDEN, _LINK_HIDDEN),
                nn.Tanh(),
                nn.Linear(_LINK_HIDDEN, 3),
            )
        self.learned_scale = LearnedLikelihoodScale(likelihood_scale)
        self.posterior_head = nn.Linear(_HIDDEN[-1], head_size(family, 3))
        if family == "tnbbeta":
            init_tnbbeta_p_bias(self.posterior_head, 3, offset=p_init_offset)

    def posterior(self, x: Tensor, cls: Tensor) -> Distribution:
        """``q(z | x)`` (``q(z | x, c)`` in D2g) for a batch."""
        raw = self.posterior_head(self.encoder(self._with_class(x, cls)))
        if self.family == "tnbbeta":
            return tnbbeta_posterior(raw, 3, epsilon_floor=_EPSILON_FLOOR)
        return posterior_from_raw(self.family, raw, 3)

    def decode(self, z: Tensor, cls: Tensor) -> Tensor:
        """The decoder mean; ``z`` is ``(..., batch, 3)``, ``cls`` is ``(batch,)``."""
        if self.decoder_kind == "mlp":
            if self.class_input:
                z = torch.cat([z, self._onehot(cls, z)], dim=-1)
            return self.decoder_net(z)
        return self.link(self._index_input(z, cls))

    def axes(self) -> Tensor:
        """The learned unit axes of a ``single_index`` decoder, ``(n_axes, 3)``."""
        return nn.functional.normalize(self.axis_raw, dim=-1)

    def log_likelihood(self, x: Tensor, z: Tensor, cls: Tensor) -> Tensor:
        """``log p(x | z)`` with the learned sigma, shape ``z.shape[:-1]``."""
        return pixel_log_likelihood(
            x, self.decode(z, cls), "gaussian", self.learned_scale()
        )

    def negative_elbo(
        self, x: Tensor, cls: Tensor, kl_weight: float, num_samples: int
    ) -> Tensor:
        """The training loss: mean over the batch of ``-(log p(x|z) - w * KL)``.

        The KL to the uniform sphere is analytic for vMF / Power Spherical and Monte
        Carlo (exact ``log q``) for TNBBeta.
        """
        posterior = self.posterior(x, cls)
        prior = standard_prior(self.family, 3, x.device)
        terms = monte_carlo_elbo(
            x,
            posterior,
            prior,
            lambda z: self.decode(z, cls),
            self.learned_scale(),
            num_samples,
            analytic_kl=self.family != "tnbbeta",
        )
        return -(terms["log_likelihood"] - kl_weight * terms["kl"]).mean()

    def _with_class(self, x: Tensor, cls: Tensor) -> Tensor:
        if self.class_input:
            return torch.cat([x, self._onehot(cls, x)], dim=-1)
        return x

    def _index_input(self, z: Tensor, cls: Tensor) -> Tensor:
        axes = self.axes()
        if self.class_input:
            index = (z * axes[cls]).sum(-1, keepdim=True)
            return torch.cat([index, self._onehot(cls, index)], -1)
        return (z * axes[0]).sum(-1, keepdim=True)

    @staticmethod
    def _onehot(labels: Tensor, like: Tensor) -> Tensor:
        """One-hot classes broadcast to ``like``'s leading dims, shape ``(..., 4)``."""
        hot = nn.functional.one_hot(labels, NUM_CLASSES).to(like.dtype)
        return hot.expand(*like.shape[:-1], NUM_CLASSES)


def arm_spec(arm: str) -> dict[str, Any]:
    """Model settings of an arm: family and p-init offset."""
    if arm in ("vmf", "power_spherical"):
        return {"family": arm, "p_init_offset": P_RAW_INIT}
    if arm == "tnbbeta":
        return {"family": "tnbbeta", "p_init_offset": P_RAW_INIT}
    if arm == "tnbbeta_init0":
        return {"family": "tnbbeta", "p_init_offset": 0.0}
    raise ValueError(f"unknown arm {arm!r}.")


def make_task(setting: str) -> RingClassTask | OffsetRingClassTask:
    """Design 1 (``D1``) or design 2' with the final offset scale (``D2g``, ``D2l``)."""
    return ring_class_task(1) if setting == "D1" else offset_ring_class_task()


def sample_data(
    setting: str,
    task: RingClassTask | OffsetRingClassTask,
    num: int,
    generator: torch.Generator,
    balanced: bool = False,
) -> tuple[Tensor, Tensor]:
    """Draws ``(x, cls)``: D1's class is the latitude band, D2's is the drawn class."""
    x, _, cls, _ = task.sample(num, generator, balanced=balanced and setting != "D1")
    return x, cls


def log_reference(
    setting: str, task: RingClassTask | OffsetRingClassTask, x: Tensor, cls: Tensor
) -> Tensor:
    """The true ``log p(x)`` (D1, D2l) or ``log p(x | c)`` (D2g), in float64."""
    x = x.double()
    if isinstance(task, OffsetRingClassTask):
        if setting == "D2g":
            return task.log_evidence_given_class(x, cls)
        return task.log_evidence(x)
    return task.log_evidence(x)


def run_path(out_dir: Path, setting: str, decoder: str, arm: str, seed: int) -> Path:
    """The stem (no suffix) of one run's output files."""
    return out_dir / setting / decoder / arm / f"s{seed}"


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``train`` or ``summarize``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--setting", choices=list(SETTINGS), required=True)
    train.add_argument("--decoder", choices=list(DECODERS), required=True)
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
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "train":
        run_train(args)
    else:
        summarize(args.out_dir)


def run_train(args: argparse.Namespace) -> None:
    """Trains one (setting, decoder, arm, seed) on the GPU, evaluates on the CPU."""
    device = select_device(args.device)
    started = time.time()
    task = make_task(args.setting)
    train_gen = torch.Generator().manual_seed(100 + args.seed)
    train_x, train_cls = sample_data(args.setting, task, args.num_train, train_gen)
    spec = arm_spec(args.arm)
    torch.manual_seed(args.seed)
    model = LearnedRingVAE(args.setting, args.decoder, **spec)
    model.to(device)
    x_dev, cls_dev = train_x.to(device), train_cls.to(device)
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
            loss = model.negative_elbo(
                x_dev[index], cls_dev[index], kl_weight, args.num_elbo_samples
            )
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
        test_x, test_cls = sample_data(
            args.setting, task, args.num_test, test_gen, balanced=True
        )
        torch.manual_seed(31337)
        arrays, summary = evaluate(
            model, args.setting, task, test_x.double(), test_cls, train_x, train_cls
        )
        diverged = not all(
            np.isfinite(arrays[k]).all() for k in ("elbo", "iwae", "log_ref")
        )
    summary.update(
        {
            "setting": args.setting,
            "decoder": args.decoder,
            "arm": args.arm,
            "seed": args.seed,
            "family": spec["family"],
            "p_init_offset": spec["p_init_offset"],
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
    stem = run_path(args.out_dir, args.setting, args.decoder, args.arm, args.seed)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(summary))
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # pyright: ignore[reportArgumentType]
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    gap = summary.get("gap_elbo", float("nan"))
    print(
        f"{args.setting} {args.decoder} {args.arm} s{args.seed}: gap {gap:.4f} "
        f"sigma {summary.get('sigma', float('nan')):.4f} diverged {diverged} "
        f"({summary['total_seconds']:.0f}s)",
        flush=True,
    )


@torch.no_grad()
def evaluate(
    model: LearnedRingVAE,
    setting: str,
    task: RingClassTask | OffsetRingClassTask,
    x: Tensor,
    cls: Tensor,
    train_x: Tensor,
    train_cls: Tensor,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per-x ELBO / IWAE gaps, decoder single-indexness, ring statistic, class probe.

    Args:
        model: The (float64, CPU) model.
        setting: ``"D1"``, ``"D2g"`` or ``"D2l"``.
        task: The task.
        x: Test observations ``(n, 3)``, float64.
        cls: Test classes ``(n,)``.
        train_x: The run's training observations (for the k-NN class probe).
        train_cls: Their classes.

    Returns:
        ``(arrays, summary)``: per-x numpy arrays and a JSON-ready scalar summary.
    """
    log_ref = log_reference(setting, task, x, cls)
    u_star, eta2 = decoder_single_index(model, per_class=model.class_input)
    u_x = torch.tensor(u_star)[cls if model.class_input else torch.zeros_like(cls)]
    prior = _UniformLogProb()
    columns: dict[str, list[Tensor]] = {}
    for start in range(0, x.shape[0], _EVAL_CHUNK):
        sl = slice(start, start + _EVAL_CHUNK)
        xs, cs, us = x[sl], cls[sl], u_x[sl]
        posterior = model.posterior(xs, cs)
        z, log_q = sample_with_log_prob(
            posterior, torch.Size([ELBO_DRAWS]), reparameterize=False
        )
        recon = model.log_likelihood(xs, z, cs).mean(0)
        if model.family == "tnbbeta":
            assert log_q is not None
            kl = (log_q + sphere_log_area(3)).mean(0)
        else:
            kl = _analytic_kl(posterior)
        bound = _Bound(model, posterior, prior, cs)
        weighted = importance_weighted_metrics(
            bound, xs, num_samples=IWAE_K, chunk_size=IWAE_K
        )
        perp = z - (z * us).sum(-1, keepdim=True) * us
        unit = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        centre = posterior_centre(model.family, posterior)
        pieces = {
            "elbo": recon - kl,
            "recon": recon,
            "kl": kl,
            "iwae": weighted["ll"],
            "ring_r": unit.mean(0).norm(dim=-1),
            "centre": centre,
            "abs_cos_u": (centre * us).sum(-1).abs(),
            **_fitted_parameters(posterior, xs.shape[0]),
        }
        for key, value in pieces.items():
            columns.setdefault(key, []).append(value)
    arrays = {key: torch.cat(value).numpy() for key, value in columns.items()}
    arrays["x"] = x.numpy()
    arrays["cls"] = cls.numpy()
    arrays["log_ref"] = log_ref.numpy()
    arrays["u_star"] = u_star
    summary = summarize_run(arrays, model, eta2)
    if setting != "D2g":
        summary["knn_accuracy"] = _knn_accuracy(
            model, arrays["centre"], cls.numpy(), train_x, train_cls
        )
    return arrays, summary


def summarize_run(
    arrays: dict[str, np.ndarray], model: LearnedRingVAE, eta2: np.ndarray
) -> dict[str, Any]:
    """Scalar summaries of one run's per-x arrays (overall and by class)."""
    cls = arrays["cls"]
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
        "gap_by_class": [float(gap[cls == c].mean()) for c in range(NUM_CLASSES)],
        "sigma": float(model.learned_scale()),
        "eta2_by_axis": [float(v) for v in eta2],
        "eta2_mean": float(eta2.mean()),
        "u_star": arrays["u_star"].tolist(),
        "ring_r_median": float(np.median(arrays["ring_r"])),
        "abs_cos_u_mean": float(arrays["abs_cos_u"].mean()),
    }
    if model.decoder_kind == "single_index":
        learned = model.axes().detach().double().numpy()
        cosines = np.abs((learned * arrays["u_star"]).sum(-1))
        summary["learned_axis_cos_ustar"] = [float(v) for v in cosines]
    for key in ("p", "q", "epsilon", "kappa"):
        values = arrays[key]
        if np.isfinite(values).any():
            summary[f"{key}_median"] = float(np.nanmedian(values))
    if np.isfinite(arrays["epsilon"]).any():
        summary["m_median"] = float(np.nanmedian(arrays["epsilon"])) - 1.0
    return summary


@torch.no_grad()
def decoder_single_index(
    model: LearnedRingVAE, per_class: bool, seed: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Best single-index axis of the decoder and the variance fraction it explains.

    ``eta^2(u)`` is the fraction of the decoder output's total variance (over uniform
    ``z`` on ``S^2``) explained by the mean in each of ``ETA_BINS`` equal-mass bins of
    ``u . z`` (a binned function of ``u . z``); it is maximised over ``u`` by
    Nelder-Mead from ``ETA_STARTS`` random starts. With a class input (``per_class``)
    the decoder is evaluated at each class in turn, giving four axes.

    Args:
        model: The (float64, CPU) model.
        per_class: One axis per class (D2g) or a single axis.
        seed: RNG seed of the points and the random starts.

    Returns:
        ``(u_star, eta2)``: unit axes ``(n_axes, 3)`` (sign fixed so the largest
        component is positive) and their ``eta^2`` ``(n_axes,)``.
    """
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(ETA_POINTS, 3, generator=generator, dtype=torch.float64)
    z = z / z.norm(dim=-1, keepdim=True)
    rng = np.random.default_rng(seed)
    classes = range(NUM_CLASSES) if per_class else range(1)
    axes, values = [], []
    for c in classes:
        cls = torch.full((ETA_POINTS,), c, dtype=torch.long)
        output = model.decode(z, cls).numpy()
        axis, eta2 = _best_axis(z.numpy(), output, rng)
        axes.append(axis)
        values.append(eta2)
    return np.stack(axes), np.array(values)


# ----------------------------------------------------------------------------- helpers


class _UniformLogProb:
    """The uniform-sphere prior's ``log_prob`` (a constant) for the IWAE machinery."""

    def log_prob(self, value: Tensor) -> Tensor:
        return torch.full(value.shape[:-1], -sphere_log_area(3), dtype=value.dtype)


class _Bound:
    """Adapts a model and a class batch to the ``importance_weighted_metrics`` API."""

    def __init__(
        self,
        model: LearnedRingVAE,
        posterior: Distribution,
        prior: _UniformLogProb,
        cls: Tensor,
    ) -> None:
        self.model, self.posterior, self.prior, self.cls = model, posterior, prior, cls

    def posterior_and_prior(self, x: Tensor) -> tuple[Distribution, _UniformLogProb]:
        del x
        return self.posterior, self.prior

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        return self.model.log_likelihood(x, z, self.cls)


def _analytic_kl(posterior: Distribution) -> Tensor:
    """Closed-form ``KL(q || Uniform(S^2))`` for vMF / Power Spherical."""
    return kl_divergence(posterior, HypersphericalUniform(2))


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


def _eta_squared(t: np.ndarray, output: np.ndarray, total: float) -> float:
    """Between-bin fraction of output variance, equal-width bins of ``t``."""
    bins = np.clip(((t + 1.0) / 2.0 * ETA_BINS).astype(np.int64), 0, ETA_BINS - 1)
    counts = np.bincount(bins, minlength=ETA_BINS).astype(np.float64)
    centred = output - output.mean(0)
    explained = 0.0
    for k in range(output.shape[1]):
        sums = np.bincount(bins, weights=centred[:, k], minlength=ETA_BINS)
        explained += float((sums[counts > 0] ** 2 / counts[counts > 0]).sum())
    return explained / total


def _best_axis(
    z: np.ndarray, output: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    total = float(((output - output.mean(0)) ** 2).sum())
    if not np.isfinite(total) or total <= 0:
        return np.array([1.0, 0.0, 0.0]), float("nan")

    def direction(angles: np.ndarray) -> np.ndarray:
        theta, phi = angles
        return np.array(
            [
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ]
        )

    def objective(angles: np.ndarray) -> float:
        return -_eta_squared(z @ direction(angles), output, total)

    best_value, best_axis = 0.0, np.array([1.0, 0.0, 0.0])
    for _ in range(ETA_STARTS):
        start = np.array([math.acos(rng.uniform(-1, 1)), rng.uniform(0, 2 * math.pi)])
        result = optimize.minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 400},
        )
        if result.fun < best_value:
            best_value, best_axis = float(result.fun), direction(result.x)
    if best_axis[np.argmax(np.abs(best_axis))] < 0:
        best_axis = -best_axis
    return best_axis, -best_value


def _knn_accuracy(
    model: LearnedRingVAE,
    test_centre: np.ndarray,
    test_cls: np.ndarray,
    train_x: Tensor,
    train_cls: Tensor,
) -> float:
    """5-NN class accuracy from posterior centres (fit on the training set)."""
    centres = []
    for start in range(0, train_x.shape[0], 2000):
        xs = train_x[start : start + 2000].double()
        cs = train_cls[start : start + 2000]
        centres.append(posterior_centre(model.family, model.posterior(xs, cs)))
    train_centre = torch.cat(centres)
    distance = torch.cdist(torch.tensor(test_centre), train_centre)
    nearest = distance.topk(KNN_K, largest=False).indices
    votes = nn.functional.one_hot(train_cls[nearest], NUM_CLASSES).sum(1)
    predicted = votes.argmax(-1).numpy()
    return float((predicted == test_cls).mean())


# ---------------------------------------------------------------------------- summarize


def summarize(out_dir: Path) -> None:
    """Prints the pre-registered analysis as markdown from the saved run summaries."""
    runs = _load_runs(out_dir)
    if not runs:
        print(f"no runs found under {out_dir}")
        return
    print("\n## Per-arm test ELBO gap (nats; lower is better)\n")
    _print_arm_table(runs)
    print("\n## Per-seed ELBO gap\n")
    _print_per_seed(runs, "gap_elbo")
    print("\n## Confirmatory tests (one-sided paired Wilcoxon, Holm at 0.05)\n")
    tests = _confirmatory_tests(runs)
    _print_confirmatory(tests, runs)
    print("\n## Per-seed importance-weighted (K=64) gap\n")
    _print_per_seed(runs, "gap_iwae")
    print("\n## Descriptive comparisons (no p-values)\n")
    _print_descriptive(runs)
    print("\n## Decoder, ring and class-structure diagnostics\n")
    _print_diagnostics(runs)
    print("\n## Learned sigma, divergences and fitted parameters\n")
    _print_sigma_params(runs)


def _load_runs(out_dir: Path) -> dict[tuple[str, str, str], dict[int, dict[str, Any]]]:
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    paths = [p for setting in SETTINGS for p in (out_dir / setting).glob("*/*/s*.json")]
    for path in sorted(paths):
        record = json.loads(path.read_text())
        key = (record["setting"], record["decoder"], record["arm"])
        runs.setdefault(key, {})[record["seed"]] = record
    return runs


def _valid(runs: dict[int, dict[str, Any]], key: str = "gap_elbo") -> dict[int, float]:
    """Seed -> finite value of ``key`` for non-diverged runs."""
    return {
        s: float(r[key])
        for s, r in runs.items()
        if not r["diverged"] and math.isfinite(r.get(key, float("nan")))
    }


def _sd(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _cells(
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
) -> list[tuple[str, str, str]]:
    return [
        (s, d, a) for s in SETTINGS for d in DECODERS for a in ARMS if (s, d, a) in runs
    ]


def _print_arm_table(
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
) -> None:
    print(
        "| setting | decoder | arm | seeds (valid) | gap mean +- sd | median seed | "
        "min / max | IWAE gap mean +- sd |"
    )
    print("|" + "---|" * 8)
    for setting, decoder, arm in _cells(runs):
        group = runs[(setting, decoder, arm)]
        gap = np.array(list(_valid(group).values()))
        iw = np.array(list(_valid(group, "gap_iwae").values()))
        if len(gap) == 0:
            print(f"| {setting} | {decoder} | {arm} | 0/{len(group)} | - | - | - | - |")
            continue
        print(
            f"| {setting} | {decoder} | {arm} | {len(gap)}/{len(group)} | "
            f"{gap.mean():.4f} +- {_sd(gap):.4f} | {np.median(gap):.4f} | "
            f"{gap.min():.4f} / {gap.max():.4f} | "
            f"{iw.mean():.4f} +- {_sd(iw):.4f} |"
        )


def _print_per_seed(
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]], key: str
) -> None:
    for setting in SETTINGS:
        for decoder in DECODERS:
            arms = [a for a in ARMS if (setting, decoder, a) in runs]
            if not arms:
                continue
            print(f"\n{setting} / {decoder}\n")
            print("| seed | " + " | ".join(arms) + " |")
            print("|" + "---|" * (1 + len(arms)))
            seeds = sorted({s for a in arms for s in runs[(setting, decoder, a)]})
            for seed in seeds:
                cells = []
                for arm in arms:
                    r = runs[(setting, decoder, arm)].get(seed)
                    if r is None:
                        cells.append("-")
                    elif r["diverged"]:
                        cells.append("DIVERGED")
                    else:
                        cells.append(f"{r[key]:.4f}")
                print(f"| {seed} | " + " | ".join(cells) + " |")


def _wilcoxon_greater(diff: np.ndarray) -> float:
    """One-sided (greater) Wilcoxon signed-rank p-value (scipy) of paired diffs."""
    return float(stats.wilcoxon(diff, alternative="greater").pvalue)  # pyright: ignore[reportAttributeAccessIssue]


def _wilcoxon_greater_enumerated(diff: np.ndarray) -> float:
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


def _confirmatory_tests(
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for setting in SETTINGS:
        tnb = _valid(runs.get((setting, "mlp", "tnbbeta"), {}))
        for other in PRIMARY_COMPARATORS:
            cap = _valid(runs.get((setting, "mlp", other), {}))
            seeds = sorted(set(tnb) & set(cap))
            if len(seeds) < 5:
                continue
            diff = np.array([cap[s] - tnb[s] for s in seeds])
            tests.append(
                {
                    "setting": setting,
                    "other": other,
                    "n": len(seeds),
                    "diff": diff,
                    "p": _wilcoxon_greater(diff),
                    "p_enum": _wilcoxon_greater_enumerated(diff),
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
    return tests


def _print_confirmatory(
    tests: list[dict[str, Any]],
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
) -> None:
    del runs
    print(
        "| setting | comparator | n | mean paired diff (comparator - tnbbeta) | "
        "seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | "
        "Holm reject | margin >= 0.05 |"
    )
    print("|" + "---|" * 10)
    for test in sorted(tests, key=lambda t: (t["setting"], t["other"])):
        wins = int((test["diff"] > 0).sum())
        print(
            f"| {test['setting']} | {test['other']} | {test['n']} | "
            f"{test['diff'].mean():.4f} | {wins}/{test['n']} | {test['p']:.5f} | "
            f"{test['p_enum']:.5f} | {test['threshold']:.5f} | "
            f"{test['holm_reject']} | {test['margin']} |"
        )
        if abs(test["p"] - test["p_enum"]) > 1e-9:
            print(f"WARNING: scipy and enumerated p differ for {test['setting']}")
    print(
        "\nVerdict 'TNBBeta helps a learned-decoder VAE' (both Holm tests survive and "
    )
    print("both mean paired differences are at least 0.05 nats):\n")
    print("| setting | Holm (both) | margin (both) | verdict |")
    print("|" + "---|" * 4)
    for setting in SETTINGS:
        mine = [t for t in tests if t["setting"] == setting]
        if len(mine) < 2:
            continue
        holm = all(t["holm_reject"] for t in mine)
        margin = all(t["margin"] for t in mine)
        print(f"| {setting} | {holm} | {margin} | {holm and margin} |")


def _print_descriptive(
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
) -> None:
    print(
        "| setting | decoder | comparison | n | mean gap (A) | mean gap (B) | mean "
        "paired diff (A - B) | seeds A better | IWAE paired diff (A - B) |"
    )
    print("|" + "---|" * 9)
    pairs = [
        ("tnbbeta", "tnbbeta_init0"),
        ("tnbbeta", "vmf"),
        ("tnbbeta", "power_spherical"),
        ("tnbbeta_init0", "vmf"),
        ("tnbbeta_init0", "power_spherical"),
        ("vmf", "power_spherical"),
    ]
    for setting in SETTINGS:
        for decoder in DECODERS:
            for a, b in pairs:
                _descriptive_row(runs, (setting, decoder, a), (setting, decoder, b))
    for setting in SETTINGS:
        for arm in ARMS:
            _descriptive_row(
                runs, (setting, "single_index", arm), (setting, "mlp", arm), "SI vs MLP"
            )


def _descriptive_row(
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
    first: tuple[str, str, str],
    second: tuple[str, str, str],
    label: str | None = None,
) -> None:
    ga = _valid(runs.get(first, {}))
    gb = _valid(runs.get(second, {}))
    seeds = sorted(set(ga) & set(gb))
    if not seeds:
        return
    ka = np.array([ga[s] for s in seeds])
    kb = np.array([gb[s] for s in seeds])
    ia = _valid(runs.get(first, {}), "gap_iwae")
    ib = _valid(runs.get(second, {}), "gap_iwae")
    iw = [ia[s] - ib[s] for s in seeds if s in ia and s in ib]
    iw_text = f"{np.mean(iw):.4f}" if iw else "-"
    name = f"{first[2]} ({label})" if label else f"{first[2]} vs {second[2]}"
    print(
        f"| {first[0]} | {first[1]}{'/' + second[1] if label else ''} | {name} | "
        f"{len(seeds)} | {ka.mean():.4f} | {kb.mean():.4f} | {(ka - kb).mean():.4f} | "
        f"{int((ka < kb).sum())}/{len(seeds)} | {iw_text} |"
    )


def _print_diagnostics(
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
) -> None:
    print(
        "| setting | decoder | arm | eta2 (mean over axes; seed median [min-max]) | "
        "ring R (median over x; seed median [min-max]) | 5-NN acc (seed mean +- sd) | "
        "mean abs(u* . centre) (seed mean) |"
    )
    print("|" + "---|" * 7)
    for setting, decoder, arm in _cells(runs):
        group = [r for r in runs[(setting, decoder, arm)].values() if not r["diverged"]]
        if not group:
            continue
        eta = np.array([r["eta2_mean"] for r in group])
        ring = np.array([r["ring_r_median"] for r in group])
        cos = np.array([r["abs_cos_u_mean"] for r in group])
        knn = np.array([r["knn_accuracy"] for r in group if "knn_accuracy" in r])
        knn_text = f"{knn.mean():.3f} +- {_sd(knn):.3f}" if len(knn) else "-"
        print(
            f"| {setting} | {decoder} | {arm} | {np.median(eta):.3f} "
            f"[{eta.min():.3f}-{eta.max():.3f}] | {np.median(ring):.3f} "
            f"[{ring.min():.3f}-{ring.max():.3f}] | {knn_text} | {cos.mean():.3f} |"
        )


def _print_sigma_params(
    runs: dict[tuple[str, str, str], dict[int, dict[str, Any]]],
) -> None:
    print(
        "| setting | decoder | arm | diverged | learned sigma (mean +- sd; true 0.2) | "
        "p | q | epsilon | m = eps - 1 | kappa |"
    )
    print("|" + "---|" * 10)
    for setting, decoder, arm in _cells(runs):
        group = runs[(setting, decoder, arm)]
        ok = [r for r in group.values() if not r["diverged"]]
        diverged = len(group) - len(ok)
        sigma = np.array([r["sigma"] for r in ok]) if ok else np.array([np.nan])

        def med(key: str, ok: list[dict[str, Any]] = ok) -> str:
            values = [r[key] for r in ok if key in r]
            return f"{np.median(values):.3g}" if values else "-"

        print(
            f"| {setting} | {decoder} | {arm} | {diverged}/{len(group)} | "
            f"{sigma.mean():.4f} +- {_sd(sigma):.4f} | {med('p_median')} | "
            f"{med('q_median')} | {med('epsilon_median')} | {med('m_median')} | "
            f"{med('kappa_median')} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
