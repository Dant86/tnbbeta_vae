# Axial Bimodality Datasets (Part 1: `axial_mixture`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the synthetic `axial_mixture` proof-of-concept from the
spec: a provably axial (sign-ambiguous) task on `S^1` where only
`TNBBetaSpherical` can represent the Bayes-optimal posterior, train the
existing `MlpVAE` across all four latent families on it, and record the
result in `writeup/results/` per project convention.

**Architecture:** A new data module (`axial_mixture.py`) composes the
existing `CircleMixtureData`'s embedding network under a doubled angle,
which is what makes `phi` and `phi + pi` exactly indistinguishable to the
observation without writing any new embedding code. A new app
(`axial_recovery.py`), a close sibling of `apps/synthetic/circle_recovery.py`,
trains the existing `MlpVAE` (`gaussian`/`vmf`/`power_spherical`/`tnbbeta`)
on it and reports both comparative (`test_ll`/`test_kl`) and mechanistic
(`m = epsilon - (latent_dim - 1) / 2`, `p`) diagnostics. No model or
architecture code changes anywhere in this plan.

**Tech Stack:** PyTorch (`torch.distributions`), Pydantic, Plotly, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md`
(Part 1 only — Part 2, the real DTD dataset, is explicitly deferred there
until Part 1's result is in, and is out of scope for this plan).

## Global Constraints

- Line length 88 chars, `ruff format`/`ruff check` clean, Google-style
  docstrings on all public module/class/function definitions outside
  `apps/`/`tests/` (which are docstring-exempt, but this plan documents them
  anyway to match `circle_recovery.py`'s existing style), `pyright` clean,
  `pytest` green — run all four before any task is considered done.
- Private (underscore-prefixed) helpers go after the public API in their
  module, not before (`docs/STYLE_GUIDE.md`).
- The exact bimodality condition (proved, not a heuristic):
  `m = epsilon - (latent_dim - 1) / 2 < 0` is necessary and sufficient for a
  `TNBBetaSpherical` posterior to be bimodal at
  `{mean_direction, -mean_direction}`
  (`writeup/weekly_markdown_summaries/week_2/tnbbeta_vs_power_spherical_expressivity.md`,
  Theorem 5.1/Corollary 5.3). For `latent_dim = 2`, that threshold is
  `epsilon < 0.5`. Every diagnostic in this plan uses `m`, never raw
  `epsilon`, as the thing to check.
- Reuse `CircleMixtureData.embed` by composition; do not duplicate the
  embedding network.
- No new model, architecture, or shared-diagnostics (`models/diagnostics.py`)
  code — everything here is a new dataset module and a new eval/experiment
  script built entirely on existing primitives (`MlpVAE`,
  `importance_weighted_metrics`, `Trainer`).
- `writeup/results/` entries are committed, not scratch: every result there
  records its date, the exact command(s) used, and the commit it was run
  against (`CLAUDE.md`).

## Review Focus

- **`embed`'s invariance must hold for arbitrary angles, not just the
  `-pi`/`pi` wrap boundary.** `circle_mixture.py`'s own test only checks
  that one boundary case; the whole task depends on invariance holding
  everywhere. Covered in Task 1.
- **The new `power_spherical` family (never exercised by
  `circle_recovery.py`) must not crash `_centre` or leak a `p_mean` key it
  has no business having** — it has a `mean_direction` attribute but no
  `p`/`q`/`epsilon`. Covered in Task 2.
- **`axial_angle_error` must treat `phi` and `phi + pi` as the *same*
  answer, not an error of ~pi.** A naive mod-`2*pi` reuse of
  `circle_recovery.angle_error` would score a perfectly-recovered axis
  pointing the "other way" as maximally wrong. Covered in Task 2.
- **`m_mean`'s sign convention must use `(latent_dim - 1) / 2`, not `1` or
  `latent_dim / 2`.** Getting this constant wrong silently inverts the
  mechanistic success criterion (reporting "bimodal" when the posterior is
  actually a unimodal cap, or vice versa). Pinned directly (at two
  different `latent_dim`s, independent of training) in Task 2, and
  cross-checked by hand in Task 3 against the raw `epsilon` value.
- **The real experiment run (Task 3) must not silently paste non-finite or
  diverged numbers into the write-up.** The spec's own "open risks" section
  flags that `m < 0` hands the Monte Carlo ELBO unbounded log-density
  terms; if that destabilizes training, the write-up must say so
  explicitly rather than report whatever number came out. Covered in
  Task 3.

---

### Task 1: `axial_mixture` data module

**Files:**
- Create: `src/tnbbeta_vae/data/axial_mixture.py`
- Test: `tests/data/test_axial_mixture.py`

**Interfaces:**
- Consumes: `tnbbeta_vae.data.circle_mixture.CircleMixtureData` (existing;
  constructor `CircleMixtureData(ambient_dim: int, noise_std: float,
  seed: int)`, method `embed(angle: Tensor) -> Tensor`).
- Produces: `AxialMixtureData` (attributes `ambient_dim: int`,
  `noise_std: float`; methods `embed(angle: Tensor) -> Tensor` and
  `sample(num: int, generator: Generator | None) -> tuple[Tensor, Tensor,
  Tensor]` returning `(x, angle, component)`) and
  `axial_mixture_data(ambient_dim: int = 100, noise_std: float = 0.05,
  seed: int = 0) -> AxialMixtureData`, both used by Task 2.

- [ ] **Step 1: Write the failing test file**

Create `tests/data/test_axial_mixture.py`:

```python
"""Tests for tnbbeta_vae.data.axial_mixture."""

from __future__ import annotations

import math

import torch

from tnbbeta_vae.data.axial_mixture import axial_mixture_data
from tnbbeta_vae.data.circle_mixture import circle_mixture_data


def test_embed_is_invariant_under_a_shift_of_pi() -> None:
    data = axial_mixture_data(ambient_dim=20, seed=1)
    angle = torch.linspace(-3.0, 3.0, 13)

    assert torch.allclose(data.embed(angle), data.embed(angle + math.pi), atol=1e-5)


def test_embed_reuses_circle_mixture_on_the_doubled_angle() -> None:
    angle = torch.linspace(-3.0, 3.0, 13)

    axial = axial_mixture_data(ambient_dim=20, seed=1)
    circle = circle_mixture_data(ambient_dim=20, seed=1)

    assert torch.equal(axial.embed(angle), circle.embed(2 * angle))


def test_samples_are_reproducible_and_shaped() -> None:
    data = axial_mixture_data(ambient_dim=20, seed=1)

    first = data.sample(50, torch.Generator().manual_seed(3))
    second = data.sample(50, torch.Generator().manual_seed(3))

    x, angle, component = first
    assert x.shape == (50, 20)
    assert angle.shape == component.shape == (50,)
    assert set(component.tolist()) <= {0, 1}
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_components_are_roughly_balanced() -> None:
    _, _, component = axial_mixture_data(seed=0).sample(
        2000, torch.Generator().manual_seed(0)
    )

    fraction_ones = component.float().mean().item()
    assert 0.4 < fraction_ones < 0.6


def test_component_is_not_recoverable_from_the_observation_alone() -> None:
    """The whole point: a component-0 and a component-1 angle at the same
    recovered axis give (noise-free) identical observations."""
    data = axial_mixture_data(ambient_dim=20, seed=1)
    angle = torch.tensor([0.4])

    component_0 = data.embed(angle)
    component_1 = data.embed(angle + math.pi)

    assert torch.allclose(component_0, component_1, atol=1e-5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/data/test_axial_mixture.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named
'tnbbeta_vae.data.axial_mixture'`.

- [ ] **Step 3: Write the implementation**

Create `src/tnbbeta_vae/data/axial_mixture.py`:

```python
"""Noisy embeddings of an axial (sign-ambiguous) von Mises mixture on the circle.

A sibling of :mod:`tnbbeta_vae.data.circle_mixture`'s S-VAE-paper replication,
built to need the opposite of what that task needs: instead of a *directed*
angle, the true latent here is an *axis* -- ``phi`` and ``phi + pi`` are
equally likely a priori and genuinely indistinguishable from the
observation, by construction. The angle is drawn from a 50/50 mixture of
two von Mises components at ``0`` and ``pi`` (same concentration), then
embedded through :meth:`CircleMixtureData.embed` on the *doubled* angle:
``embed(2*phi) == embed(2*(phi+pi))`` exactly, since ``2*phi`` and
``2*phi + 2*pi`` are the same angle. That reuses the existing fixed random
embedding network unchanged; only the generative distribution over the
angle and the doubling are new.

The Bayes-optimal posterior over ``phi`` given an observation is therefore
exactly bimodal with equal mass at the two components -- the shape a single
``TNBBetaSpherical`` component can represent (via ``epsilon`` below the
uniform-prior threshold) and no other latent family in this project can.
See
``docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md``.
"""

from __future__ import annotations

import math

import torch
from torch.distributions import VonMises

from tnbbeta_vae.data.circle_mixture import CircleMixtureData

__all__ = ["AxialMixtureData", "axial_mixture_data"]

_CONCENTRATION = 20.0


class AxialMixtureData:
    """A fixed axial embedding of the circle into R^``ambient_dim`` plus a sampler.

    Attributes:
        ambient_dim: Dimension of the observed vectors.
        noise_std: Standard deviation of the Gaussian observation noise.
    """

    def __init__(
        self, ambient_dim: int = 100, noise_std: float = 0.05, seed: int = 0
    ) -> None:
        """Draws the fixed embedding weights.

        Args:
            ambient_dim: Dimension of the observed vectors.
            noise_std: Standard deviation of the observation noise.
            seed: Seed for the embedding weights (not for :meth:`sample`),
                forwarded to
                :class:`~tnbbeta_vae.data.circle_mixture.CircleMixtureData`.
        """
        self.ambient_dim = ambient_dim
        self.noise_std = noise_std
        self._circle = CircleMixtureData(ambient_dim, noise_std, seed)

    def embed(self, angle: torch.Tensor) -> torch.Tensor:
        """Maps angles to noise-free points, invariant under ``angle + pi``.

        Args:
            angle: Angles in radians, shape ``(n,)``.

        Returns:
            Tensor of shape ``(n, ambient_dim)``. Equal for ``angle`` and
            ``angle + pi`` (reuses
            :meth:`~tnbbeta_vae.data.circle_mixture.CircleMixtureData.embed`
            on the doubled angle, which is exactly periodic in ``pi``).
        """
        return self._circle.embed(2 * angle)

    def sample(
        self, num: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draws noisy observations.

        Args:
            num: Number of points.
            generator: Optional RNG for the component, angle and noise.

        Returns:
            ``(x, angle, component)``: observations ``(num, ambient_dim)``,
            the true angles in ``(-pi, pi]`` and which of the two axial
            components (0 or 1, for the von Mises centred at 0 or at pi)
            produced it. ``component`` is never recoverable from ``x``
            alone by construction -- that is the point of the task.
        """
        component = torch.randint(2, (num,), generator=generator)
        centres = component.float() * math.pi
        # VonMises has no generator argument, so seed the global RNG from
        # ours (same workaround as CircleMixtureData.sample).
        if generator is not None:
            torch.manual_seed(int(torch.randint(2**31, (1,), generator=generator)))
        angle = VonMises(centres, torch.tensor(_CONCENTRATION)).sample()
        noise = self.noise_std * torch.randn(num, self.ambient_dim, generator=generator)
        return self.embed(angle) + noise, angle, component


def axial_mixture_data(
    ambient_dim: int = 100, noise_std: float = 0.05, seed: int = 0
) -> AxialMixtureData:
    """Builds the dataset object (see :class:`AxialMixtureData`)."""
    return AxialMixtureData(ambient_dim, noise_std, seed)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/data/test_axial_mixture.py -v`
Expected: 5 passed.

- [ ] **Step 5: Format, lint, type-check**

Run: `uv run ruff format src/tnbbeta_vae/data/axial_mixture.py tests/data/test_axial_mixture.py && uv run ruff check src/tnbbeta_vae/data/axial_mixture.py tests/data/test_axial_mixture.py && uv run pyright src/tnbbeta_vae/data/axial_mixture.py`
Expected: no diffs from format, no lint errors, no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/tnbbeta_vae/data/axial_mixture.py tests/data/test_axial_mixture.py
git commit -m "Add axial_mixture: an axial (sign-ambiguous) sibling of circle_mixture

Composes CircleMixtureData.embed on a doubled angle, which exactly erases
phi vs phi+pi -- the Bayes-optimal posterior is then provably bimodal.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `axial_recovery` experiment script

**Files:**
- Create: `apps/synthetic/axial_recovery.py`
- Test: `tests/apps/test_axial_recovery.py`

**Interfaces:**
- Consumes: `axial_mixture_data` and `AxialMixtureData` from Task 1;
  `tnbbeta_vae.models.MlpVAE`/`MlpVAEConfig` (existing, `family:
  Literal["gaussian","vmf","tnbbeta","power_spherical"]`, `latent_dim: int`);
  `tnbbeta_vae.models.losses.importance_weighted_metrics(model, x, *,
  num_samples, chunk_size) -> dict[str, Tensor]` (keys `"ll"`, `"elbo"`,
  `"re"`, `"kl"`); `tnbbeta_vae.training.Trainer` (existing).
- Produces: `main(argv: list[str] | None) -> None` (CLI entry point, writes
  `<out-dir>/axial_recovery.json` and `<out-dir>/axial_recovery.html`) and
  `axial_angle_error(latent: Tensor, true_angle: Tensor) -> float`, both
  used by Task 3 (via the CLI) and by this task's own tests.

- [ ] **Step 1: Write the failing test file**

Create `tests/apps/test_axial_recovery.py`:

```python
"""Tests for the axial-mixture data and the S^1 axial-recovery experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from apps.synthetic import axial_recovery
from tnbbeta_vae.models import MlpVAE, MlpVAEConfig


def test_tnbbeta_diagnostics_m_mean_uses_latent_dim_minus_one_over_two() -> None:
    """Pins the bimodality-indicator formula directly, independent of training.

    ``m_mean`` must be ``epsilon_mean - (latent_dim - 1) / 2``, not off by a
    constant -- getting this wrong would silently invert the mechanistic
    success criterion (see the plan's Review Focus).
    """
    torch.manual_seed(0)
    x = torch.randn(16, 100)
    for latent_dim in (2, 4):
        model = MlpVAE(
            MlpVAEConfig(family=cast("Any", "tnbbeta"), latent_dim=latent_dim)
        )
        posterior, _ = model.posterior_and_prior(x)
        centre = axial_recovery._centre(model, x, "tnbbeta")

        diagnostics = axial_recovery._tnbbeta_diagnostics(model, x, centre)

        expected = posterior.epsilon.mean().item() - (latent_dim - 1) / 2
        assert diagnostics["m_mean"] == pytest.approx(expected)


def test_axial_angle_error_treats_phi_and_phi_plus_pi_as_equal() -> None:
    generator = torch.Generator().manual_seed(0)
    true = torch.rand(500, generator=generator) * 2 * math.pi - math.pi
    rotated = true + 1.3
    reflected = -true + 0.4
    flipped = true + math.pi
    good = [
        torch.stack([a.cos(), a.sin()], dim=-1)
        for a in (true, rotated, reflected, flipped)
    ]
    random = torch.randn(500, 2, generator=generator)

    for latent in good:
        assert axial_recovery.axial_angle_error(latent, true) < 1e-3
    assert axial_recovery.axial_angle_error(random, true) > 0.5


def test_experiment_writes_metrics_and_figure(tmp_path: Path) -> None:
    axial_recovery.main(
        [
            "--out-dir",
            str(tmp_path),
            "--epochs",
            "2",
            "--num-train",
            "256",
            "--num-test",
            "64",
            "--batch-size",
            "64",
            "--kl-warmup-epochs",
            "1",
        ]  # fmt: skip
    )

    results = json.loads((tmp_path / "axial_recovery.json").read_text())
    assert set(results) == {"gaussian", "vmf", "power_spherical", "tnbbeta"}
    for metrics in results.values():
        for key in (
            "angle_error",
            "angle_error_sample",
            "reconstruction_angle_error",
            "prior_manifold_ratio",
            "test_ll",
            "test_kl",
        ):
            assert math.isfinite(metrics[key])
    assert {"p_mean", "p_std", "centre_axis_resultant", "m_mean"} <= set(
        results["tnbbeta"]
    )
    assert "p_mean" not in results["vmf"]
    assert "p_mean" not in results["power_spherical"]
    html = (tmp_path / "axial_recovery.html").read_text()
    assert "N-VAE" in html and "TNBBeta" in html and "Power Spherical" in html
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/apps/test_axial_recovery.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named
'apps.synthetic.axial_recovery'`.

- [ ] **Step 3: Write the implementation**

Create `apps/synthetic/axial_recovery.py`:

```python
"""Recover an axis from noisy R^100 embeddings with a built-in sign ambiguity.

Usage:
    uv run python -m apps.synthetic.axial_recovery [--out-dir DIR] \
        [--epochs 200] [--seed 0] \
        [--families gaussian vmf power_spherical tnbbeta]

A sibling of ``apps.synthetic.circle_recovery``: trains an MLP VAE per latent
family (``latent_dim=2``) on ``tnbbeta_vae.data.axial_mixture``'s axial von
Mises mixture, where the observation cannot distinguish an angle ``phi``
from ``phi + pi`` by construction (see that module's docstring and
``docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md``).
Reports, for each family:

* ``angle_error``/``angle_error_sample``: as in ``circle_recovery``, but
  wrapped mod ``pi`` (not mod ``2*pi``) -- the recoverable signal here is
  the axis, not a signed angle. Small means the axis was recovered;
  ``~pi/4`` means it was not.
* ``reconstruction_angle_error``: as in ``circle_recovery``, mod ``pi``.
* ``prior_manifold_ratio``, ``test_ll``, ``test_kl``: unchanged from
  ``circle_recovery``.
* TNBBeta only: ``p_mean``, ``p_std``, ``centre_axis_resultant`` (as in
  ``circle_recovery``) plus ``m_mean = epsilon_mean - (latent_dim - 1) / 2``
  -- negative is the exact condition (proved in
  ``writeup/weekly_markdown_summaries/week_2/
  tnbbeta_vs_power_spherical_expressivity.md``) for a bimodal posterior at
  ``{mean_direction, -mean_direction}``, which is what this task's
  Bayes-optimal posterior actually looks like.

Writes ``axial_recovery.json`` and ``axial_recovery.html`` (latent scatter
plots coloured by the true axial component) into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from tnbbeta_vae.data.axial_mixture import axial_mixture_data
from tnbbeta_vae.models import MlpVAE, MlpVAEConfig
from tnbbeta_vae.models.losses import importance_weighted_metrics
from tnbbeta_vae.training import Trainer

_TITLES = {
    "gaussian": "N-VAE",
    "vmf": "S-VAE (vMF)",
    "power_spherical": "Power Spherical",
    "tnbbeta": "TNBBeta",
}


def main(argv: list[str] | None = None) -> None:
    """Runs the experiment and writes the JSON and HTML outputs.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("axial_recovery"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-train", type=int, default=10_000)
    parser.add_argument("--num-test", type=int, default=2_000)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["gaussian", "vmf", "power_spherical", "tnbbeta"],
        choices=list(_TITLES),
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = axial_mixture_data(seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_x, _, _ = data.sample(args.num_train, generator)
    test_x, test_angle, test_component = data.sample(args.num_test, generator)
    manifold_angles = torch.linspace(-math.pi, math.pi, 4000)
    manifold = data.embed(manifold_angles)
    batches = list(train_x.split(args.batch_size))

    results: dict[str, Any] = {}
    latents: dict[str, dict[str, torch.Tensor]] = {}
    for family in args.families:
        torch.manual_seed(args.seed)
        model = MlpVAE(MlpVAEConfig(family=cast("Any", family), latent_dim=2))
        trainer = Trainer(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            "mlp_vae",
            model.config,
            runs_dir=args.out_dir / "runs",
            run_id=family,
        )
        trainer.fit(batches, args.epochs, kl_warmup_epochs=args.kl_warmup_epochs)
        model.eval()
        centre = _centre(model, test_x, family)
        sample = _sample(model, test_x)
        latents[family] = {"centre": centre, "sample": sample}
        metrics = importance_weighted_metrics(model, test_x, num_samples=500)
        results[family] = {
            "angle_error": axial_angle_error(centre, test_angle),
            "angle_error_sample": axial_angle_error(sample, test_angle),
            "reconstruction_angle_error": _reconstruction_angle_error(
                model, sample, manifold, manifold_angles, test_angle
            ),
            "prior_manifold_ratio": _prior_manifold_ratio(model, manifold, test_x),
            "test_ll": metrics["ll"].mean().item(),
            "test_kl": metrics["kl"].mean().item(),
            "learned_sigma": model.learned_scale().item(),
        }
        if family == "tnbbeta":
            results[family].update(_tnbbeta_diagnostics(model, test_x, centre))
        print(family, json.dumps(results[family]))

    (args.out_dir / "axial_recovery.json").write_text(json.dumps(results, indent=2))
    _write_figure(latents, test_angle, test_component, args.out_dir)
    print(f"Wrote {args.out_dir / 'axial_recovery.json'}")


def axial_angle_error(latent: torch.Tensor, true_angle: torch.Tensor) -> float:
    """Mean absolute angular error mod pi, after the best rotation and reflection.

    Like ``circle_recovery.angle_error``, but wraps to ``(-pi/2, pi/2]``
    instead of ``(-pi, pi]`` -- the recoverable signal here is the axis
    (angle mod pi), not the signed angle, since ``phi`` and ``phi + pi`` are
    equally valid. The best rotation offset is found via the standard
    axial-mean trick (double the residual, take its circular mean, halve),
    the same doubling idea the dataset itself uses to erase the sign.

    Args:
        latent: Points in the plane, shape ``(n, 2)``; only their angle is
            used.
        true_angle: The true angles, shape ``(n,)``.

    Returns:
        The smallest mean absolute circular difference mod pi (radians)
        over a reflection and the best rotation.
    """
    estimate = torch.atan2(latent[:, 1], latent[:, 0])
    best = math.inf
    for sign in (1.0, -1.0):
        residual = true_angle - sign * estimate
        doubled = 2 * residual
        offset = torch.atan2(doubled.sin().mean(), doubled.cos().mean()) / 2
        wrapped = (residual - offset + math.pi / 2) % math.pi - math.pi / 2
        best = min(best, wrapped.abs().mean().item())
    return best


@torch.no_grad()
def _centre(model: MlpVAE, x: torch.Tensor, family: str) -> torch.Tensor:
    posterior, _ = model.posterior_and_prior(x)
    if family == "gaussian":
        return posterior.base_dist.loc  # pyright: ignore[reportAttributeAccessIssue]
    if family == "vmf":
        return posterior.loc  # pyright: ignore[reportAttributeAccessIssue]
    if family == "power_spherical":
        return posterior.mean_direction  # pyright: ignore[reportAttributeAccessIssue]
    direction = posterior.mean_direction  # pyright: ignore[reportAttributeAccessIssue]
    return torch.where((posterior.p > 0.5)[:, None], direction, -direction)  # pyright: ignore[reportAttributeAccessIssue]


@torch.no_grad()
def _tnbbeta_diagnostics(
    model: MlpVAE, x: torch.Tensor, centre: torch.Tensor
) -> dict[str, float]:
    """How TNBBeta encodes the axis: p/epsilon and the bimodality indicator.

    ``m_mean`` is ``epsilon_mean - (latent_dim - 1) / 2``; negative is the
    exact bimodality condition (expressivity write-up, Theorem
    5.1/Corollary 5.3).
    """
    posterior, _ = model.posterior_and_prior(x)
    p = posterior.p  # pyright: ignore[reportAttributeAccessIssue]
    epsilon = posterior.epsilon  # pyright: ignore[reportAttributeAccessIssue]
    doubled = 2 * torch.atan2(centre[:, 1], centre[:, 0])
    latent_dim = model.config.latent_dim
    return {
        "p_mean": p.mean().item(),
        "p_std": p.std().item(),
        "centre_axis_resultant": torch.hypot(
            doubled.cos().mean(), doubled.sin().mean()
        ).item(),
        "m_mean": epsilon.mean().item() - (latent_dim - 1) / 2,
    }


@torch.no_grad()
def _reconstruction_angle_error(
    model: MlpVAE,
    sample: torch.Tensor,
    manifold: torch.Tensor,
    manifold_angles: torch.Tensor,
    true_angle: torch.Tensor,
) -> float:
    nearest = torch.cdist(model.decoder(sample), manifold).argmin(dim=1)
    difference = manifold_angles[nearest] - true_angle
    return ((difference + math.pi / 2) % math.pi - math.pi / 2).abs().mean().item()


@torch.no_grad()
def _sample(model: MlpVAE, x: torch.Tensor) -> torch.Tensor:
    posterior, _ = model.posterior_and_prior(x)
    return posterior.sample()


@torch.no_grad()
def _prior_manifold_ratio(
    model: MlpVAE, manifold: torch.Tensor, real: torch.Tensor, num: int = 2000
) -> float:
    generated = model.generate(num)
    generated_distance = torch.cdist(generated, manifold).min(dim=1).values.mean()
    real_distance = torch.cdist(real, manifold).min(dim=1).values.mean()
    return (generated_distance / real_distance).item()


def _write_figure(
    latents: dict[str, dict[str, torch.Tensor]],
    angle: torch.Tensor,
    component: torch.Tensor,
    out_dir: Path,
) -> None:
    """Scatter plots of the posterior centres (top) and samples (bottom)."""
    figure = make_subplots(
        rows=2, cols=len(latents), subplot_titles=[_TITLES[f] for f in latents]
    )
    marker = {
        "size": 4,
        "color": np.asarray(component),
        "colorscale": "Turbo",
        "opacity": 0.7,
    }
    for column, points in enumerate(latents.values(), start=1):
        for row, kind in enumerate(("centre", "sample"), start=1):
            array = points[kind].numpy()
            figure.add_trace(
                go.Scatter(
                    x=array[:, 0],
                    y=array[:, 1],
                    mode="markers",
                    marker=marker,
                    text=[f"angle {a:.2f}" for a in angle.numpy()],
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
            index = (row - 1) * len(latents) + column
            axis = "y" if index == 1 else f"y{index}"
            figure.update_xaxes(scaleanchor=axis, row=row, col=column)
    figure.update_layout(
        title="Latent space by true axial component (top: centres, bottom: samples)"
    )
    figure.write_html(out_dir / "axial_recovery.html", include_plotlyjs=True)


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/apps/test_axial_recovery.py -v`
Expected: 3 passed. (The end-to-end test actually trains 4 tiny models for
2 epochs each; it should finish in well under a minute on CPU.)

- [ ] **Step 5: Format, lint, type-check**

Run: `uv run ruff format apps/synthetic/axial_recovery.py tests/apps/test_axial_recovery.py && uv run ruff check apps/synthetic/axial_recovery.py tests/apps/test_axial_recovery.py && uv run pyright apps/synthetic/axial_recovery.py`
Expected: no diffs from format, no lint errors, no type errors.

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest`
Expected: all tests pass, including the two new files and everything
already in the repo (confirms nothing in Task 1/2 broke an existing test).

- [ ] **Step 7: Commit**

```bash
git add apps/synthetic/axial_recovery.py tests/apps/test_axial_recovery.py
git commit -m "Add axial_recovery: train gaussian/vmf/power_spherical/tnbbeta on axial_mixture

Sibling of apps.synthetic.circle_recovery; adds the m = epsilon -
(latent_dim-1)/2 bimodality diagnostic and mod-pi angle error metrics.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: Run the experiment and record the result

**Files:**
- Create: `writeup/results/axial_mixture_recovery_<today's date, YYYY-MM-DD>.md`

**Interfaces:**
- Consumes: `apps.synthetic.axial_recovery`'s CLI (Task 2) via `python -m`.
- Produces: a committed, dated result write-up — the final deliverable of
  this plan, and the thing that determines whether Part 2 (DTD, out of
  scope here) is worth starting.

- [ ] **Step 1: Get the commit hash to cite**

Run: `git log -1 --format='%h %s'`
Record the hash (and branch name, via `git branch --show-current`) for the
write-up's header — this is a feature-branch commit, not yet on `master`,
so CLAUDE.md's convention is to cite it as such.

- [ ] **Step 2: Run the real experiment**

Run (defaults match `circle_recovery.py`'s own defaults — 200 epochs,
10k/2k train/test, which trains in well under a minute per family on CPU
given the tiny MLP and 2D latent):

```bash
uv run python -m apps.synthetic.axial_recovery --out-dir runs/axial_recovery
```

`runs/` is already gitignored (`.gitignore`'s `runs/` entry), so this
output is local scratch, not something to commit directly — only the
transcribed results below get committed.

- [ ] **Step 3: Inspect the raw output and check for non-finite results**

Run: `cat runs/axial_recovery/axial_recovery.json`

Before writing anything up, check by hand that every `test_ll`, `test_kl`,
`angle_error`, and (for `tnbbeta`) `m_mean`/`p_mean` value is finite and
not wildly implausible (e.g. `test_kl` in the tens of thousands, or
`angle_error` exactly `0` or exactly `pi/4`, would indicate a bug or a
training collapse, not a result). If anything looks broken, treat it as a
bug to fix in Task 1 or 2 (re-open those, don't patch around it here)
before proceeding — per this plan's Review Focus, a non-finite or diverged
number must be reported as such, never silently pasted into the write-up
as if it were a clean result.

- [ ] **Step 4: Write the dated results doc**

Create `writeup/results/axial_mixture_recovery_<YYYY-MM-DD>.md` (use
today's actual date; this plan was written assuming same-day execution on
2026-10-07, but use whatever date Task 3 actually runs on). Use this
structure, filling the `<...>` placeholders from Steps 1-3's actual
output — every number in the table must come from the real run, not be
invented:

```markdown
# Axial mixture recovery: does TNBBeta's bimodality get used? <YYYY-MM-DD>

- **Date:** <YYYY-MM-DD>
- **Branch commit:** `<hash>` ("<commit subject>", branch `<branch name>`;
  not yet on `master`)
- **Spec:** `docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md`
  (Part 1)
- **Command:**
  ```
  uv run python -m apps.synthetic.axial_recovery --out-dir runs/axial_recovery
  ```

`axial_mixture` is a synthetic task where the true angle `phi` is drawn
from a 50/50 mixture of von Mises components at `0` and `pi`, then observed
through an embedding that is exactly invariant under `phi -> phi + pi` (see
the spec). The Bayes-optimal posterior over `phi` given an observation is
therefore exactly bimodal at the two components -- a shape only
`TNBBetaSpherical` can represent (`PowerSpherical` is *proved* strictly
unimodal for every kappa; `vmf` and `gaussian` are unimodal for the same
structural reason).

## Results (single seed)

| family | test_ll | test_kl | angle_error | angle_error_sample | reconstruction_angle_error | prior_manifold_ratio |
|---|---|---|---|---|---|---|
| gaussian | <val> | <val> | <val> | <val> | <val> | <val> |
| vmf | <val> | <val> | <val> | <val> | <val> | <val> |
| power_spherical | <val> | <val> | <val> | <val> | <val> | <val> |
| tnbbeta | <val> | <val> | <val> | <val> | <val> | <val> |

TNBBeta-only mechanistic diagnostic: `p_mean = <val>`, `p_std = <val>`,
`m_mean = <val>` (bimodal iff negative; the uniform-prior boundary is
`m = 0`, i.e. `epsilon = 0.5` at this `latent_dim = 2`).

## Reading it

<Fill in after seeing the real numbers -- do not pre-write a conclusion.
Specifically address, honestly, whichever of these actually happened:>

- Did `tnbbeta`'s `m_mean` come out negative (criterion 1 from the spec:
  the model actually finds and uses the bimodal regime), or did it stay at
  or above the `m = 0` uniform-prior boundary the way MNIST training does?
- Did `tnbbeta` beat `vmf`/`power_spherical`/`gaussian` on `test_ll`/
  `test_kl` (criterion 2)? By how much, and is it the direction the spec
  predicted?
- If `tnbbeta` did *not* show `m_mean < 0` but still matched or beat the
  other families on `test_ll`/`test_kl`: that is the spec's flagged
  caveat (a unimodal family's decoder learning approximate sign-invariance
  to work around the ambiguity) potentially showing up here, or could mean
  the training signal/warm-up schedule just never pushes epsilon down far
  enough to reach the bimodal regime (as MNIST training does not) -- say
  which, with evidence (e.g. check `power_spherical` and `vmf`'s own
  `angle_error` vs `reconstruction_angle_error`: if `reconstruction_angle_error`
  stays small while `angle_error` on the *centre* is large/unstable across
  reruns, that's the "decoder learned sign-invariance, encoder picks
  arbitrarily" pattern).
- One-paragraph verdict: is this a "satisfying result" by the spec's own
  bar (criteria 1-2 both hold clearly), justifying moving to Part 2
  (DTD), or not yet -- and if not, what's the next concrete thing to try
  (e.g. an epsilon-initialization or KL-warmup change, analogous to the
  `fixed_epsilon`/`fixed_mean_direction` ablations already in
  `ConvTNBBetaSphericalVAEConfig`)?
```

- [ ] **Step 5: Commit**

```bash
git add writeup/results/axial_mixture_recovery_*.md
git commit -m "Record axial_mixture recovery result (Part 1 of the bimodality-datasets spec)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
