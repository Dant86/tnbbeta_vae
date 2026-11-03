"""Axial Bingham study: does TNBBeta's antipodal capacity help when the posterior IS
Bingham?

Pre-registration: ``writeup/results/axial_bingham_preregistration_2026-11-02.md``
(design, decoders, arms, metrics, tests and decision rules are fixed there; deviations
are appended at its end). Task: :mod:`tnbbeta_vae.data.axial_bingham` (``X = z z^T +
sigma E``, exact posterior Bingham, exactly antipodal).

Usage (one run per call; the shell loops over the grid)::

    uv run python -m apps.synthetic.axial_bingham train --dim 3 --sigma 0.25 \\
        --decoder linear --arm tnbbeta --seed 0 --out-dir ring_recovery/axial_bingham
    uv run python -m apps.synthetic.axial_bingham summarize \\
        --out-dir ring_recovery/axial_bingham

Each run writes ``<out-dir>/d<dim>/s<sigma>/<decoder>/<arm>/s<seed>.{json,npz,pt}``.
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

from tnbbeta_vae.data.axial_bingham import (
    AxialBinghamTask,
    num_sym,
    sphere_quadrature,
    vec_sym,
)
from tnbbeta_vae.data.ring_posterior import sphere_log_area
from tnbbeta_vae.distributions import HypersphericalUniform
from tnbbeta_vae.distributions.sign_symmetrised import SignSymmetrised
from tnbbeta_vae.models.architectures.mlp import mlp_stack
from tnbbeta_vae.models.heads import (
    P_RAW_INIT,
    LatentFamily,
    head_size,
    init_tnbbeta_p_bias,
    posterior_from_raw,
    power_spherical_kappa_inverse,
    tnbbeta_posterior,
    vmf_kappa_inverse,
)
from tnbbeta_vae.models.losses.elbo import monte_carlo_elbo, pixel_log_likelihood
from tnbbeta_vae.models.losses.importance_weighted import importance_weighted_metrics
from tnbbeta_vae.models.losses.likelihood import LearnedLikelihoodScale
from tnbbeta_vae.models.losses.sampling import sample_with_log_prob
from tnbbeta_vae.training import select_device

DIMS = (2, 3, 4)
EXACT_DIMS = (2, 3)
SIGMAS = (0.1, 0.25, 0.5, 1.0)
DECODERS = ("fixed", "linear", "mlp")
ARMS = ("vmf", "power_spherical", "tnbbeta", "tnbbeta_init0", "vmf_sym", "ps_sym")
PRIMARY_COMPARATORS = ("vmf", "power_spherical")
SECONDARY_COMPARATORS = ("vmf_sym", "ps_sym")
NUM_SEEDS = 10
MARGIN_NATS = 0.05
HOLM_ALPHA = 0.05
TEST_SEED = 424242
ELBO_DRAWS = 256
IWAE_K = 64
INITIAL_LEARNED_SIGMA = 1.0
_HIDDEN = [128, 128]
_EPSILON_FLOOR = 1e-4
_EVAL_CHUNK = 250
_QUAD_CHUNK = 100
_QUAD_REFINE_X = 100


class AxialVAE(nn.Module):
    """MLP encoder + spherical posterior head and a fixed or learned sign-invariant
    decoder.

    Attributes:
        dim: Ambient latent dimension ``d``.
        decoder_kind: ``"fixed"`` (the true ``z z^T`` with the true sigma), ``"linear"``
            (``W vec_sym(z z^T) + b`` with a learned sigma) or ``"mlp"`` (an MLP of
            ``vec_sym(z z^T)`` with a learned sigma).
        arm: The posterior arm (see :data:`ARMS`).
    """

    def __init__(
        self,
        dim: int,
        sigma_true: float,
        decoder: str,
        arm: str,
        p_init_offset: float = P_RAW_INIT,
    ) -> None:
        """Builds encoder, decoder, sigma and head (in that RNG order).

        The head is built last, so one seed's encoder and decoder initial weights are
        the same across arms.

        Args:
            dim: Ambient latent dimension.
            sigma_true: The data's true noise scale (used only by the fixed decoder).
            decoder: ``"fixed"``, ``"linear"`` or ``"mlp"``.
            arm: One of :data:`ARMS`.
            p_init_offset: TNBBeta only: initial raw-p bias of the head.
        """
        super().__init__()
        self.dim, self.decoder_kind, self.arm = dim, decoder, arm
        self.task = AxialBinghamTask(dim, sigma_true)
        self.encoder = mlp_stack([dim * dim, *_HIDDEN])
        if decoder == "linear":
            self.decoder_net: nn.Module = nn.Linear(num_sym(dim), dim * dim)
        elif decoder == "mlp":
            self.decoder_net = nn.Sequential(
                mlp_stack([num_sym(dim), *_HIDDEN]), nn.Linear(_HIDDEN[-1], dim * dim)
            )
        else:
            self.decoder_net = nn.Identity()
        self.learned_scale = LearnedLikelihoodScale(INITIAL_LEARNED_SIGMA)
        self.base_family: LatentFamily = arm_base_family(arm)
        self.symmetrised = arm in ("vmf_sym", "ps_sym")
        self.posterior_head = nn.Linear(_HIDDEN[-1], head_size(self.base_family, dim))
        if self.base_family == "tnbbeta":
            init_tnbbeta_p_bias(self.posterior_head, dim, offset=p_init_offset)
        else:
            self._init_kappa_bias()

    def posterior(self, x: Tensor) -> Distribution:
        """``q(z | X)`` for a batch (sign-symmetrised for the ``*_sym`` arms)."""
        raw = self.posterior_head(self.encoder(x))
        if self.base_family == "tnbbeta":
            base: Distribution = tnbbeta_posterior(
                raw, self.dim, epsilon_floor=_EPSILON_FLOOR
            )
        else:
            base = posterior_from_raw(self.base_family, raw, self.dim)
        return SignSymmetrised(base) if self.symmetrised else base

    def sigma(self) -> Tensor:
        """The likelihood scale: the true one for the fixed decoder, else learned."""
        if self.decoder_kind == "fixed":
            return torch.tensor(
                self.task.sigma, dtype=self.learned_scale.log_variance.dtype
            ).to(self.learned_scale.log_variance.device)
        return self.learned_scale()

    def decode(self, z: Tensor) -> Tensor:
        """The decoder mean for ``z`` of shape ``(..., d)``, shape ``(..., d * d)``."""
        if self.decoder_kind == "fixed":
            return self.task.true_mean(z)
        return self.decoder_net(vec_sym(z))

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """``log p(x | z)``, shape ``z.shape[:-1]``."""
        return pixel_log_likelihood(x, self.decode(z), "gaussian", self.sigma())

    def posterior_and_prior(self, x: Tensor) -> tuple[Distribution, Distribution]:
        """``(q(. | x), Uniform(S^(d-1)))`` for the importance-weighted metrics."""
        return self.posterior(x), HypersphericalUniform(self.dim - 1, device=x.device)

    def negative_elbo(self, x: Tensor, kl_weight: float, num_samples: int) -> Tensor:
        """The training loss: mean over the batch of ``-(log p(x|z) - w * KL)``.

        The KL to the uniform sphere is analytic for plain vMF / Power Spherical and
        Monte Carlo (exact ``log q``) for TNBBeta and the sign-symmetrised arms.
        """
        posterior, prior = self.posterior_and_prior(x)
        terms = monte_carlo_elbo(
            x,
            posterior,
            prior,
            self.decode,
            self.sigma(),
            num_samples,
            analytic_kl=self.base_family != "tnbbeta" and not self.symmetrised,
        )
        return -(terms["log_likelihood"] - kl_weight * terms["kl"]).mean()

    def _init_kappa_bias(self) -> None:
        """Starts vMF / PS at ``kappa = d`` (dimension-aware), as CLAUDE.md
        prescribes."""
        kappa = float(self.dim)
        raw = (
            vmf_kappa_inverse(kappa + 1.0)
            if self.base_family == "vmf"
            else power_spherical_kappa_inverse(kappa)
        )
        with torch.no_grad():
            assert self.posterior_head.bias is not None
            self.posterior_head.bias[-1] = raw


def arm_base_family(arm: str) -> LatentFamily:
    """The latent family of an arm's base posterior."""
    if arm in ("vmf", "vmf_sym"):
        return "vmf"
    if arm in ("power_spherical", "ps_sym"):
        return "power_spherical"
    if arm in ("tnbbeta", "tnbbeta_init0"):
        return "tnbbeta"
    raise ValueError(f"unknown arm {arm!r}.")


def arm_p_init(arm: str) -> float:
    """The TNBBeta raw-p bias of an arm (library default except ``tnbbeta_init0``)."""
    return 0.0 if arm == "tnbbeta_init0" else P_RAW_INIT


def run_path(
    out_dir: Path, dim: int, sigma: float, decoder: str, arm: str, seed: int
) -> Path:
    """The stem (no suffix) of one run's output files."""
    return out_dir / f"d{dim}" / f"s{sigma:g}" / decoder / arm / f"s{seed}"


def main(argv: list[str] | None = None) -> None:
    """Dispatches to ``train`` or ``summarize``.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--dim", type=int, choices=list(DIMS), required=True)
    train.add_argument("--sigma", type=float, required=True)
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
    """Trains one (dim, sigma, decoder, arm, seed), evaluates it on the CPU in
    float64."""
    device = select_device(args.device)
    started = time.time()
    task = AxialBinghamTask(args.dim, args.sigma)
    train_gen = torch.Generator().manual_seed(100 + args.seed)
    train_x, _ = task.sample(args.num_train, train_gen)
    torch.manual_seed(args.seed)
    model = AxialVAE(
        args.dim, args.sigma, args.decoder, args.arm, p_init_offset=arm_p_init(args.arm)
    )
    model.to(device)
    x_dev = train_x.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    shuffle = torch.Generator().manual_seed(200 + args.seed)
    warmup = max(1, args.epochs // 10)
    losses: list[float] = []
    diverged = False
    skipped_steps = 0
    total_steps = 0
    for epoch in range(args.epochs):
        kl_weight = min(1.0, (epoch + 1) / warmup)
        order = torch.randperm(args.num_train, generator=shuffle).to(device)
        running = torch.zeros((), device=device)
        for index in order.split(args.batch_size):
            total_steps += 1
            optimiser.zero_grad()
            try:
                loss = model.negative_elbo(
                    x_dev[index], kl_weight, args.num_elbo_samples
                )
                loss.backward()
            except ValueError:  # a non-finite sample reached a validated distribution
                skipped_steps += 1
                continue
            gradient = nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            if not (torch.isfinite(loss) and torch.isfinite(gradient)):
                skipped_steps += 1
                continue
            optimiser.step()
            running += loss.detach() * len(index)
        losses.append(float(running) / args.num_train)
        if not math.isfinite(losses[-1]):
            diverged = True
            break
    # A run that had to skip more than 1% of its steps is not a clean run.
    diverged = diverged or skipped_steps > 0.01 * total_steps
    model.cpu().double().eval()
    train_seconds = time.time() - started

    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    if not diverged:
        test_gen = torch.Generator().manual_seed(TEST_SEED)
        test_x, _ = task.sample(args.num_test, test_gen)
        torch.manual_seed(31337)
        arrays, summary = evaluate(model, task, test_x.double())
        diverged = not all(np.isfinite(arrays[k]).all() for k in ("elbo", "log_ref"))
    summary.update(
        {
            "dim": args.dim,
            "sigma_true": args.sigma,
            "decoder": args.decoder,
            "arm": args.arm,
            "seed": args.seed,
            "family": model.base_family,
            "symmetrised": model.symmetrised,
            "p_init_offset": arm_p_init(args.arm),
            "epochs": args.epochs,
            "num_train": args.num_train,
            "num_test": args.num_test,
            "num_elbo_samples": args.num_elbo_samples,
            "elbo_draws": ELBO_DRAWS,
            "iwae_k": IWAE_K,
            "diverged": bool(diverged),
            "skipped_steps": skipped_steps,
            "total_steps": total_steps,
            "epochs_run": len(losses),
            "final_train_loss": losses[-1] if losses else float("nan"),
            "loss_curve": losses[:: max(1, args.epochs // 30)],
            "device": str(device),
            "train_seconds": train_seconds,
            "total_seconds": time.time() - started,
        }
    )
    stem = run_path(
        args.out_dir, args.dim, args.sigma, args.decoder, args.arm, args.seed
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(summary))
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # pyright: ignore[reportArgumentType]
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    gap = summary.get("gap_elbo", float("nan"))
    print(
        f"d{args.dim} s{args.sigma:g} {args.decoder} {args.arm} s{args.seed}: gap "
        f"{gap:.4f} sigma {summary.get('sigma', float('nan')):.4f} "
        f"diverged {diverged} ({summary['total_seconds']:.0f}s)",
        flush=True,
    )


@torch.no_grad()
def evaluate(
    model: AxialVAE, task: AxialBinghamTask, x: Tensor
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per-x ELBO / IWAE gaps against the true evidence, exact-model quantities, shape
    stats.

    Args:
        model: The (float64, CPU) model.
        task: The task.
        x: Test observations ``(n, d * d)``, float64.

    Returns:
        ``(arrays, summary)``: per-x numpy arrays and a JSON-ready scalar summary.
    """
    log_ref = task.log_evidence(x)
    top_axis = task.top_axis(x)
    columns: dict[str, list[Tensor]] = {}
    nonfinite = 0
    for start in range(0, x.shape[0], _EVAL_CHUNK):
        sl = slice(start, start + _EVAL_CHUNK)
        xs = x[sl]
        posterior, prior = model.posterior_and_prior(xs)
        z, log_q = sample_with_log_prob(
            posterior, torch.Size([ELBO_DRAWS]), reparameterize=False
        )
        assert log_q is not None
        valid = torch.isfinite(z).all(-1) & torch.isfinite(log_q)
        nonfinite += int((~valid).sum())
        safe = torch.where(valid[..., None], z, torch.eye(model.dim)[0].to(z))
        recon_all = model.log_likelihood(xs, safe)
        count = valid.sum(0).clamp_min(1)
        recon = torch.where(valid, recon_all, 0.0).sum(0) / count
        if model.base_family != "tnbbeta" and not model.symmetrised:
            kl = kl_divergence(posterior, prior)
        else:
            kl = (torch.where(valid, log_q, 0.0).sum(0) / count) + sphere_log_area(
                model.dim
            )
        try:
            weighted = importance_weighted_metrics(
                model, xs, num_samples=IWAE_K, chunk_size=IWAE_K
            )["ll"]
        except ValueError:
            weighted = torch.full_like(recon, float("nan"))
        pieces = {
            "elbo": recon - kl,
            "recon": recon,
            "kl": kl,
            "iwae": weighted,
            **_mass_split(z, top_axis[sl]),
            **_fitted_parameters(posterior, xs.shape[0]),
        }
        for key, value in pieces.items():
            columns.setdefault(key, []).append(value)
    arrays = {key: torch.cat(value).numpy() for key, value in columns.items()}
    arrays["x"] = x.numpy()
    arrays["log_ref"] = log_ref.numpy()
    summary: dict[str, Any] = {
        "sigma": float(model.sigma()),
        "eval_nonfinite_draws": nonfinite,
        "eval_nonfinite_iwae_x": int((~np.isfinite(arrays["iwae"])).sum()),
    }
    if model.dim in EXACT_DIMS:
        log_model, check = model_log_evidence(model, task, x)
        arrays["log_model"] = log_model.numpy()
        summary.update(check)
    summarize_run(arrays, summary, model)
    return arrays, summary


@torch.no_grad()
def model_log_evidence(
    model: AxialVAE, task: AxialBinghamTask, x: Tensor, size: int = 1
) -> tuple[Tensor, dict[str, float]]:
    """The model's own ``log p_theta(x) = log E_z p_theta(x | z)`` by sphere quadrature.

    Also checks the quadrature itself: the TRUE model's evidence on the same grid
    against the closed-form Bingham value, and the learned model's evidence on a grid
    refined 2x.

    Args:
        model: The (float64, CPU) model.
        task: The task.
        x: Observations ``(n, d * d)``.
        size: Grid refinement multiplier of the main evaluation.

    Returns:
        ``(log_model, checks)``.
    """
    z, log_w = sphere_quadrature(model.dim, size)
    mean = model.decode(z)
    log_model = _grid_evidence(x, mean, log_w, float(model.sigma()))
    true_grid = _grid_evidence(x, task.true_mean(z), log_w, task.sigma)
    z2, log_w2 = sphere_quadrature(model.dim, 2 * size)
    refined = _grid_evidence(
        x[:_QUAD_REFINE_X], model.decode(z2), log_w2, float(model.sigma())
    )
    return log_model, {
        "quad_true_model_maxerr": float((true_grid - task.log_evidence(x)).abs().max()),
        "quad_refine_maxdiff": float(
            (refined - log_model[:_QUAD_REFINE_X]).abs().max()
        ),
    }


def _grid_evidence(x: Tensor, mean: Tensor, log_w: Tensor, sigma: float) -> Tensor:
    """``log sum_k w_k N(x; mean_k, sigma^2 I)`` for each row of ``x``."""
    features = x.shape[-1]
    constant = -features * (math.log(sigma) + 0.5 * math.log(2 * math.pi))
    mean_sq = mean.pow(2).sum(-1)
    out = []
    for start in range(0, x.shape[0], _QUAD_CHUNK):
        xs = x[start : start + _QUAD_CHUNK]
        dist_sq = xs.pow(2).sum(-1, keepdim=True) - 2 * xs @ mean.T + mean_sq
        out.append(torch.logsumexp(-dist_sq / (2 * sigma**2) + log_w, dim=-1))
    return constant + torch.cat(out)


def summarize_run(
    arrays: dict[str, np.ndarray], summary: dict[str, Any], model: AxialVAE
) -> None:
    """Fills ``summary`` with scalar summaries of one run's per-x arrays (in place)."""
    gap = arrays["log_ref"] - arrays["elbo"]
    iwae_gap = arrays["log_ref"] - arrays["iwae"]
    summary.update(
        {
            "gap_elbo": float(gap.mean()),
            "gap_elbo_median": float(np.median(gap)),
            "gap_elbo_se": float(gap.std(ddof=1) / math.sqrt(len(gap))),
            "gap_elbo_min": float(gap.min()),
            "gap_iwae": float(np.nanmean(iwae_gap)),
            "elbo_mean": float(arrays["elbo"].mean()),
            "log_ref_mean": float(arrays["log_ref"].mean()),
            "recon_mean": float(arrays["recon"].mean()),
            "kl_mean": float(arrays["kl"].mean()),
            "mass_min_q_axis_median": float(np.median(arrays["mass_min_q_axis"])),
            "mass_min_true_axis_median": float(np.median(arrays["mass_min_true_axis"])),
            "axis_align_median": float(np.median(arrays["axis_align"])),
        }
    )
    if "log_model" in arrays:
        kl_model = arrays["log_model"] - arrays["elbo"]
        summary.update(
            {
                "kl_to_model_mean": float(kl_model.mean()),
                "kl_to_model_min": float(kl_model.min()),
                "model_deficit_mean": float(
                    (arrays["log_ref"] - arrays["log_model"]).mean()
                ),
            }
        )
    for key in ("p", "q", "epsilon", "kappa"):
        values = arrays[key]
        if np.isfinite(values).any():
            summary[f"{key}_median"] = float(np.nanmedian(values))
    if np.isfinite(arrays["epsilon"]).any():
        threshold = (model.dim - 1) / 2
        m = arrays["epsilon"] - threshold
        summary["m_median"] = float(np.nanmedian(m))
        summary["frac_m_negative"] = float(np.nanmean(m < 0))


# ----------------------------------------------------------------------------- helpers


def _mass_split(z: Tensor, true_axis: Tensor) -> dict[str, Tensor]:
    """Antipodal mass balance of the posterior draws, per ``x``.

    ``mass_min_q_axis`` is the smaller hemisphere share about the posterior's OWN axis
    (the top eigenvector of its second-moment matrix; frame-free, so valid for a learned
    decoder whose latent frame is arbitrary): 0.5 is exactly balanced, 0 one-sided.
    ``mass_min_true_axis`` is the same about the top eigenvector of the data's ``X_sym``
    (meaningful only where the latent frame is the data's, i.e. the fixed decoder), and
    ``axis_align`` is ``|v_q . v_true|``.
    """
    second = torch.einsum("sbi,sbj->bij", z, z) / z.shape[0]
    _, vectors = torch.linalg.eigh(second)
    own = vectors[..., -1]
    share_own = ((z * own).sum(-1) > 0).to(z.dtype).mean(0)
    share_true = ((z * true_axis).sum(-1) > 0).to(z.dtype).mean(0)
    return {
        "mass_min_q_axis": torch.minimum(share_own, 1 - share_own),
        "mass_min_true_axis": torch.minimum(share_true, 1 - share_true),
        "axis_align": (own * true_axis).sum(-1).abs(),
    }


def _fitted_parameters(posterior: Distribution, num: int) -> dict[str, Tensor]:
    """Per-x fitted parameters (NaN where the family has none)."""
    base = posterior.base if isinstance(posterior, SignSymmetrised) else posterior
    nan = torch.full((num,), float("nan"), dtype=torch.float64)
    out = {"p": nan, "q": nan, "epsilon": nan, "kappa": nan}
    if hasattr(base, "epsilon"):
        out["p"] = base.p  # pyright: ignore[reportAttributeAccessIssue]
        out["q"] = base.q  # pyright: ignore[reportAttributeAccessIssue]
        out["epsilon"] = base.epsilon  # pyright: ignore[reportAttributeAccessIssue]
    elif hasattr(base, "kappa"):
        out["kappa"] = base.kappa.reshape(num)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        out["kappa"] = base.scale.reshape(num)  # pyright: ignore[reportAttributeAccessIssue]
    return out


# ---------------------------------------------------------------------------- summarize

Runs = dict[tuple[int, float, str, str], dict[int, dict[str, Any]]]


def summarize(out_dir: Path) -> None:
    """Prints the pre-registered analysis as markdown from the saved run summaries."""
    runs = _load_runs(out_dir)
    if not runs:
        print(f"no runs found under {out_dir}")
        return
    print(
        "\n## Per-arm test ELBO gap against the true log p(X) (nats; lower is better)\n"
    )
    _print_arm_table(runs)
    print("\n## Per-seed ELBO gap\n")
    _print_per_seed(runs, "gap_elbo")
    print("\n## Confirmatory tests (one-sided paired Wilcoxon, Holm at 0.05)\n")
    tests = _paired_tests(runs, PRIMARY_COMPARATORS)
    _holm(tests)
    _print_tests(tests)
    print("\n## Secondary tests vs the sign-symmetrised baselines (own Holm family)\n")
    secondary = _paired_tests(runs, SECONDARY_COMPARATORS)
    _holm(secondary)
    _print_tests(secondary)
    print("\n## Exact KL(q || p_model) where the model evidence is exact (d = 2, 3)\n")
    _print_value_table(
        runs,
        "kl_to_model_mean",
        "mean KL to the model's own posterior",
        exact_only=True,
    )
    print("\n## Model deficit log p_true(X) - log p_model(X) (d = 2, 3, learned)\n")
    _print_value_table(
        runs, "model_deficit_mean", "mean deficit", exact_only=True, skip_fixed=True
    )
    print("\n## Importance-weighted (K = 64) gap\n")
    _print_value_table(runs, "gap_iwae", "mean IWAE gap")
    print("\n## Descriptive paired differences and the ln 2 prediction\n")
    _print_descriptive(runs)
    print("\n## Antipodal mass balance (median over x, seed median; 0.5 = balanced)\n")
    _print_mass(runs)
    print("\n## Learned sigma, divergences and fitted parameters\n")
    _print_sigma_params(runs)
    print("\n## Quadrature checks (worst case over runs)\n")
    _print_quadrature(runs)


def _load_runs(out_dir: Path) -> Runs:
    runs: Runs = {}
    for path in sorted(out_dir.glob("d*/s*/*/*/s*.json")):
        record = json.loads(path.read_text())
        key = (record["dim"], record["sigma_true"], record["decoder"], record["arm"])
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


def _cells(runs: Runs) -> list[tuple[int, float, str, str]]:
    return sorted(
        runs,
        key=lambda k: (k[0], k[1], DECODERS.index(k[2]), ARMS.index(k[3])),
    )


def _print_arm_table(runs: Runs) -> None:
    print(
        "| d | sigma | decoder | arm | seeds (valid) | gap mean +- sd | median seed | "
        "min / max | IWAE gap mean |"
    )
    print("|" + "---|" * 9)
    for dim, sigma, decoder, arm in _cells(runs):
        group = runs[(dim, sigma, decoder, arm)]
        gap = np.array(list(_valid(group).values()))
        iw = np.array(list(_valid(group, "gap_iwae").values()))
        head = f"| {dim} | {sigma:g} | {decoder} | {arm} |"
        if len(gap) == 0:
            print(f"{head} 0/{len(group)} | - | - | - | - |")
            continue
        print(
            f"{head} {len(gap)}/{len(group)} | {gap.mean():.4f} +- {_sd(gap):.4f} | "
            f"{np.median(gap):.4f} | {gap.min():.4f} / {gap.max():.4f} | "
            f"{iw.mean():.4f} |"
        )


def _print_per_seed(runs: Runs, key: str) -> None:
    groups = sorted({(d, s, dec) for d, s, dec, _ in runs})
    for dim, sigma, decoder in sorted(
        groups, key=lambda g: (g[0], g[1], DECODERS.index(g[2]))
    ):
        arms = [a for a in ARMS if (dim, sigma, decoder, a) in runs]
        print(f"\nd={dim}, sigma={sigma:g}, {decoder}\n")
        print("| seed | " + " | ".join(arms) + " |")
        print("|" + "---|" * (1 + len(arms)))
        seeds = sorted({s for a in arms for s in runs[(dim, sigma, decoder, a)]})
        for seed in seeds:
            cells = []
            for arm in arms:
                r = runs[(dim, sigma, decoder, arm)].get(seed)
                if r is None:
                    cells.append("-")
                elif r["diverged"]:
                    cells.append("DIVERGED")
                else:
                    cells.append(f"{r[key]:.4f}")
            print(f"| {seed} | " + " | ".join(cells) + " |")


def _print_value_table(
    runs: Runs,
    key: str,
    label: str,
    exact_only: bool = False,
    skip_fixed: bool = False,
) -> None:
    print(f"| d | sigma | decoder | arm | {label} (seed mean +- sd) | seeds |")
    print("|" + "---|" * 6)
    for dim, sigma, decoder, arm in _cells(runs):
        if (exact_only and dim not in EXACT_DIMS) or (
            skip_fixed and decoder == "fixed"
        ):
            continue
        values = np.array(list(_valid(runs[(dim, sigma, decoder, arm)], key).values()))
        if len(values):
            print(
                f"| {dim} | {sigma:g} | {decoder} | {arm} | "
                f"{values.mean():.4f} +- {_sd(values):.4f} | {len(values)} |"
            )


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


def _paired_tests(runs: Runs, comparators: tuple[str, ...]) -> list[dict[str, Any]]:
    """One test per (d in {2, 3}, sigma, decoder, comparator): TNBBeta smaller gap."""
    tests: list[dict[str, Any]] = []
    for dim in EXACT_DIMS:
        for sigma in SIGMAS:
            for decoder in DECODERS:
                tnb = _valid(runs.get((dim, sigma, decoder, "tnbbeta"), {}))
                for other in comparators:
                    cap = _valid(runs.get((dim, sigma, decoder, other), {}))
                    seeds = sorted(set(tnb) & set(cap))
                    if len(seeds) < 5:
                        continue
                    diff = np.array([cap[s] - tnb[s] for s in seeds])
                    tests.append(
                        {
                            "dim": dim,
                            "sigma": sigma,
                            "decoder": decoder,
                            "other": other,
                            "n": len(seeds),
                            "diff": diff,
                            "p": _wilcoxon_greater(diff),
                            "p_enum": _wilcoxon_greater_enumerated(diff),
                        }
                    )
    return tests


def _holm(tests: list[dict[str, Any]]) -> None:
    """Holm step-down at 0.05 over a family of ``len(tests)`` tests (in place)."""
    order = np.argsort([t["p"] for t in tests], kind="stable")
    still = True
    for rank, index in enumerate(order):
        test = tests[index]
        test["threshold"] = HOLM_ALPHA / (len(tests) - rank)
        still = still and test["p"] <= test["threshold"]
        test["holm_reject"] = bool(still)
        test["margin"] = bool(test["diff"].mean() >= MARGIN_NATS)
        test["confirmed"] = test["holm_reject"] and test["margin"]


def _print_tests(tests: list[dict[str, Any]]) -> None:
    print(
        "| d | sigma | decoder | comparator | n | "
        "mean paired diff (comparator - tnbbeta) "
        "| seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | "
        "Holm reject | margin >= 0.05 | confirmed |"
    )
    print("|" + "---|" * 13)
    key = lambda t: (t["dim"], t["sigma"], DECODERS.index(t["decoder"]), t["other"])  # noqa: E731
    for test in sorted(tests, key=key):
        wins = int((test["diff"] > 0).sum())
        print(
            f"| {test['dim']} | {test['sigma']:g} | {test['decoder']} | "
            f"{test['other']} | "
            f"{test['n']} | {test['diff'].mean():.4f} | {wins}/{test['n']} | "
            f"{test['p']:.5f} | {test['p_enum']:.5f} | {test['threshold']:.5f} | "
            f"{test['holm_reject']} | {test['margin']} | {test['confirmed']} |"
        )
        if abs(test["p"] - test["p_enum"]) > 1e-9:
            print(f"WARNING: scipy and enumerated p differ for {test}")
    confirmed = sum(t["confirmed"] for t in tests)
    reverse = sum(t["diff"].mean() <= -MARGIN_NATS for t in tests)
    print(
        f"\nConfirmed (TNBBeta better, Holm and margin): {confirmed} of {len(tests)}; "
        f"comparator better by at least the margin (descriptive): {reverse}."
    )


def _print_descriptive(runs: Runs) -> None:
    print(
        "| d | sigma | decoder | comparison | n | mean gap (A) | mean gap (B) | "
        "mean paired diff (A - B) | seeds A better | IWAE paired diff (A - B) |"
    )
    print("|" + "---|" * 10)
    pairs = [
        ("tnbbeta", "tnbbeta_init0"),
        ("tnbbeta", "vmf"),
        ("tnbbeta", "power_spherical"),
        ("tnbbeta", "vmf_sym"),
        ("tnbbeta", "ps_sym"),
        ("tnbbeta_init0", "vmf_sym"),
        ("vmf", "vmf_sym"),
        ("power_spherical", "ps_sym"),
        ("vmf", "power_spherical"),
    ]
    for dim, sigma, decoder in sorted(
        {(d, s, dec) for d, s, dec, _ in runs},
        key=lambda g: (g[0], g[1], DECODERS.index(g[2])),
    ):
        for a, b in pairs:
            first = _valid(runs.get((dim, sigma, decoder, a), {}))
            second = _valid(runs.get((dim, sigma, decoder, b), {}))
            seeds = sorted(set(first) & set(second))
            if not seeds:
                continue
            ka = np.array([first[s] for s in seeds])
            kb = np.array([second[s] for s in seeds])
            ia = _valid(runs.get((dim, sigma, decoder, a), {}), "gap_iwae")
            ib = _valid(runs.get((dim, sigma, decoder, b), {}), "gap_iwae")
            iw = [ia[s] - ib[s] for s in seeds if s in ia and s in ib]
            iw_text = f"{np.mean(iw):.4f}" if iw else "-"
            print(
                f"| {dim} | {sigma:g} | {decoder} | {a} vs {b} | {len(seeds)} | "
                f"{ka.mean():.4f} | {kb.mean():.4f} | {(ka - kb).mean():.4f} | "
                f"{int((ka < kb).sum())}/{len(seeds)} | {iw_text} |"
            )
    print(f"\nln 2 = {math.log(2):.4f}.")


def _print_mass(runs: Runs) -> None:
    print(
        "| d | sigma | decoder | arm | min hemisphere share about q's own axis | "
        "about the true X_sym axis (fixed decoder) | "
        "median abs cos(q axis, true axis) |"
    )
    print("|" + "---|" * 7)
    for dim, sigma, decoder, arm in _cells(runs):
        ok = [r for r in runs[(dim, sigma, decoder, arm)].values() if not r["diverged"]]
        if not ok:
            continue
        own = np.median([r["mass_min_q_axis_median"] for r in ok])
        true = np.median([r["mass_min_true_axis_median"] for r in ok])
        align = np.median([r["axis_align_median"] for r in ok])
        true_text = f"{true:.3f}" if decoder == "fixed" else "-"
        align_text = f"{align:.3f}" if decoder == "fixed" else "-"
        print(
            f"| {dim} | {sigma:g} | {decoder} | {arm} | {own:.3f} | {true_text} | "
            f"{align_text} |"
        )


def _print_sigma_params(runs: Runs) -> None:
    print(
        "| d | sigma | decoder | arm | diverged | learned sigma (mean +- sd) | p | q | "
        "epsilon | m = eps - (d-1)/2 | frac m<0 | kappa |"
    )
    print("|" + "---|" * 12)
    for dim, sigma, decoder, arm in _cells(runs):
        group = runs[(dim, sigma, decoder, arm)]
        ok = [r for r in group.values() if not r["diverged"]]
        sig = np.array([r["sigma"] for r in ok]) if ok else np.array([np.nan])

        def med(key: str, ok: list[dict[str, Any]] = ok) -> str:
            values = [r[key] for r in ok if key in r]
            return f"{np.median(values):.3g}" if values else "-"

        sigma_text = (
            f"{sig.mean():.4f} +- {_sd(sig):.4f}"
            if decoder != "fixed"
            else "(true, fixed)"
        )
        print(
            f"| {dim} | {sigma:g} | {decoder} | {arm} | {len(group) - len(ok)}/"
            f"{len(group)} | {sigma_text} | {med('p_median')} | {med('q_median')} | "
            f"{med('epsilon_median')} | {med('m_median')} | {med('frac_m_negative')} | "
            f"{med('kappa_median')} |"
        )


def _print_quadrature(runs: Runs) -> None:
    worst: dict[tuple[int, float], tuple[float, float]] = {}
    for (dim, sigma, _, _), group in runs.items():
        for r in group.values():
            if "quad_true_model_maxerr" in r:
                a, b = worst.get((dim, sigma), (0.0, 0.0))
                worst[(dim, sigma)] = (
                    max(a, r["quad_true_model_maxerr"]),
                    max(b, r["quad_refine_maxdiff"]),
                )
    print(
        "| d | sigma | max abs err of true-model quadrature vs Bingham | "
        "max abs refine diff |"
    )
    print("|---|---|---|---|")
    for (dim, sigma), (a, b) in sorted(worst.items()):
        print(f"| {dim} | {sigma:g} | {a:.2e} | {b:.2e} |")


if __name__ == "__main__":
    main(sys.argv[1:])
