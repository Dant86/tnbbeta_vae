# TNBBeta-Spherical Diffusion Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a learned, score-matching-flavored prior for
`ConvTNBBetaSphericalVAE`'s latent space, trained via a closed-form
forward-noising process native to the `TNBBetaSpherical` family, so
`generate()`-quality CIFAR-10 samples can be tested against the existing
fixed-`Uniform(sphere)`-prior baseline.

**Architecture:** A new `src/tnbbeta_vae/diffusion/` package implements the
verified forward-noising math (epsilon merge/split, the Leisen et al. (2019)
closed-form `q`-kernel, the exact `p`-logit-decoupling identity) as batched
torch functions. A new registered model, `tnbbeta_spherical_diffusion_prior`,
wraps a frozen, pretrained `ConvTNBBetaSphericalVAE` checkpoint and a small
MLP denoiser trained by direct `z_0`-regression; `generate()` runs a
multi-step predict-then-renoise reverse sampler starting from the VAE's own
`Uniform(sphere)` prior, then decodes through the frozen decoder.

**Tech Stack:** PyTorch (`torch.distributions.NegativeBinomial`, `Binomial`,
`Beta`), the existing `tnbbeta_vae` registry/`Trainer`/checkpoint
infrastructure, `scipy.stats` for statistical test assertions (already a
project dependency).

**Spec:** `docs/superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md`

## Global Constraints

- Run all four before considering any task done: `uv run ruff format .`,
  `uv run ruff check .`, `uv run pyright`, `uv run pytest` (CLAUDE.md).
- `master` is protected -- this work stays on `research/sphere-diffusion-prior`.
- `TNBBetaSpherical`/`TNBBetaUnivariate` are finished, validated math
  (CLAUDE.md) -- Task 3's one change to `tnbbeta_spherical.py` is a pure
  visibility rename (no logic change), and is reviewed in isolation for
  exactly that reason.
- Library code lives under `src/tnbbeta_vae/`; CLI/experiment scripts live
  under `apps/` (CLAUDE.md).
- New models register via `@register_model(name, config_cls=...)` and are
  imported from `src/tnbbeta_vae/models/__init__.py` so the registry sees
  them (CLAUDE.md) -- required for `apps/train/main.py --model ...` to find
  the new model by name.
- Docstrings: Google style. Private (`_`-prefixed) functions/methods go
  after the public API in their module (`docs/STYLE_GUIDE.md`).
- Tests mirror the `src/tnbbeta_vae/` package layout under `tests/`.

## Review Focus

- **Logit overflow near the point-mass start.** `eps_start=50`, `p_start=
  0.999` are deliberately tight; `logit(u)` for `u` within float32 epsilon
  of 0 or 1 can produce `inf`/`nan`. A reasonable person expects a forward
  pass near `t=0` to return finite numbers, not silently propagate `nan`
  into training. Covered in Task 1 (`test_draw_latitude_is_finite_near_t_zero`).
- **`t` sampled at or near exactly 0 during training trivializes the
  denoising task** (predicting `z_0` from `z_t` ~= `z_0` is not a useful
  training signal) -- RSGM's own paper samples `t ~ U([epsilon, T])`, not
  `U([0, T])`, for the same reason. Covered in Task 5
  (`test_training_step_never_samples_t_below_t_min`).
- **Pointing `vae_run_name` at a non-`ConvTNBBetaSphericalVAE` checkpoint**
  (e.g. a `conv_gaussian_vae` run) should fail loudly at construction time,
  not produce a model that silently calls methods the loaded object doesn't
  have. Covered in Task 5 (`test_rejects_a_non_tnbbeta_spherical_vae_checkpoint`).
- **The frozen VAE actually receives zero gradient**, not just
  `requires_grad=False` set once -- a reasonable person reading "frozen"
  expects `.grad` to stay `None` after a real `backward()` call, not just
  that the flag was set. Covered in Task 5
  (`test_vae_parameters_receive_no_gradient`).
- **`generate()` at the reverse loop's boundary conditions** (`num_reverse_
  steps=1`, and the final decoded output) should produce finite, valid
  `[0, 1]`-ranged images, not `nan` from an off-by-one in the step
  sequence. Covered in Task 5 (`test_generate_handles_single_reverse_step`
  and `test_generate_returns_valid_images`).

---

## Task 1: Forward-noising primitives (`diffusion/forward.py`)

**Files:**
- Create: `src/tnbbeta_vae/diffusion/__init__.py`
- Create: `src/tnbbeta_vae/diffusion/forward.py`
- Test: `tests/diffusion/__init__.py`
- Test: `tests/diffusion/test_forward.py`

**Interfaces:**
- Consumes: `tnbbeta_vae.distributions.TNBBetaUnivariate` (test oracle only).
- Produces:
  - `resize_eps(c: Tensor, eps_from: Tensor, eps_to: Tensor, q: Tensor) -> Tensor`
  - `leisen_step(c0: Tensor, t: Tensor, eps: Tensor, q_target: float, speed: float) -> Tensor`
  - `draw_latitude(p_data: Tensor | float, q_data: Tensor | float, eps_data: Tensor | float, t: Tensor, eps_target: float, q_target: float, speed: float) -> Tensor` -- returns `w_t`, the latitude in `(-1, 1)`, same shape as `t`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/diffusion/test_forward.py
"""Tests for tnbbeta_vae.diffusion.forward.

Statistical checks (KS test against TNBBetaUnivariate's closed-form
density) at N=20,000 -- smaller than the N=200,000 used during the
exploratory notebooks/ scripts this is promoted from, traded for test-suite
speed; the threshold is loosened to match (stat < 0.03 rather than 0.01).
"""

from __future__ import annotations

import torch
from scipy import stats

from tnbbeta_vae.diffusion.forward import draw_latitude, leisen_step, resize_eps
from tnbbeta_vae.distributions import TNBBetaUnivariate

_N = 20_000
_KS_STAT_MAX = 0.03


def _ks_match(x: torch.Tensor, y: torch.Tensor) -> bool:
    stat, _ = stats.ks_2samp(x.numpy(), y.numpy())
    return stat < _KS_STAT_MAX


def _closed_form_y(p: float, q: float, eps: float, n: int) -> torch.Tensor:
    return TNBBetaUnivariate(
        torch.tensor(p), torch.tensor(q), torch.tensor(eps)
    ).rsample((n,))


def test_resize_eps_split_direction_matches_closed_form() -> None:
    """eps_to < eps_from: splitting off a smaller piece keeps the same q."""
    torch.manual_seed(0)
    p, q, eps_from, eps_to = 0.4, 0.6, 5.0, 2.0
    c = torch.distributions.NegativeBinomial(
        torch.full((_N,), eps_from), probs=torch.full((_N,), 1 - q)
    ).sample()

    resized = resize_eps(
        c, torch.full((_N,), eps_from), torch.full((_N,), eps_to), torch.full((_N,), q)
    )
    a_t = torch.distributions.NegativeBinomial(eps_to + resized, probs=1 - p).sample()
    b_t = torch.distributions.NegativeBinomial(eps_to + resized, probs=p).sample()
    y = torch.distributions.Beta(eps_to + resized + a_t, eps_to + resized + b_t).sample()

    assert _ks_match(y, _closed_form_y(p, q, eps_to, _N))


def test_resize_eps_merge_direction_matches_closed_form() -> None:
    """eps_to > eps_from: merging in a fresh piece keeps the same q."""
    torch.manual_seed(1)
    p, q, eps_from, eps_to = 0.4, 0.6, 1.0, 4.0
    c = torch.distributions.NegativeBinomial(
        torch.full((_N,), eps_from), probs=torch.full((_N,), 1 - q)
    ).sample()

    resized = resize_eps(
        c, torch.full((_N,), eps_from), torch.full((_N,), eps_to), torch.full((_N,), q)
    )
    a_t = torch.distributions.NegativeBinomial(eps_to + resized, probs=1 - p).sample()
    b_t = torch.distributions.NegativeBinomial(eps_to + resized, probs=p).sample()
    y = torch.distributions.Beta(eps_to + resized + a_t, eps_to + resized + b_t).sample()

    assert _ks_match(y, _closed_form_y(p, q, eps_to, _N))


def test_leisen_step_is_identity_at_t_zero() -> None:
    torch.manual_seed(2)
    c0 = torch.distributions.NegativeBinomial(
        torch.full((_N,), 3.0), probs=torch.full((_N,), 0.5)
    ).sample()

    c_t = leisen_step(c0, torch.zeros(_N), torch.full((_N,), 3.0), q_target=0.1, speed=1.0)

    assert torch.equal(c_t, c0)


def test_leisen_step_converges_regardless_of_starting_q() -> None:
    """Large t: C_t's law -> NB(eps, q_target), independent of C_0's own q."""
    torch.manual_seed(3)
    eps, q_target = 3.0, 0.05
    c0_a = torch.distributions.NegativeBinomial(
        torch.full((_N,), eps), probs=torch.full((_N,), 0.9)
    ).sample()
    c0_b = torch.distributions.NegativeBinomial(
        torch.full((_N,), eps), probs=torch.full((_N,), 0.1)
    ).sample()

    c_t_a = leisen_step(c0_a, torch.full((_N,), 8.0), torch.full((_N,), eps), q_target, 1.0)
    c_t_b = leisen_step(c0_b, torch.full((_N,), 8.0), torch.full((_N,), eps), q_target, 1.0)

    stat, _ = stats.ks_2samp(c_t_a.numpy(), c_t_b.numpy())
    assert stat < _KS_STAT_MAX


def test_draw_latitude_recovers_data_at_t_zero() -> None:
    torch.manual_seed(4)
    p, q, eps = 0.85, 0.6, 2.5

    w = draw_latitude(
        p, q, eps, torch.zeros(_N), eps_target=1.0, q_target=0.05, speed=1.0
    )
    y = (w + 1) / 2

    assert _ks_match(y, _closed_form_y(p, q, eps, _N))


def test_draw_latitude_converges_to_target_regardless_of_data_params() -> None:
    torch.manual_seed(5)
    eps_target, q_target = 1.0, 0.05
    target_y = _closed_form_y(0.5, q_target, eps_target, _N)

    for p_data, q_data, eps_data in [(0.85, 0.6, 2.5), (0.1, 0.0, 0.5), (0.5, 0.9, 5.0)]:
        w = draw_latitude(
            p_data, q_data, eps_data, torch.full((_N,), 10.0),
            eps_target=eps_target, q_target=q_target, speed=1.0,
        )  # fmt: skip
        y = (w + 1) / 2
        assert _ks_match(y, target_y), (p_data, q_data, eps_data)


def test_draw_latitude_is_finite_near_t_zero() -> None:
    """Review Focus: near-point-mass params (eps=50, p=0.999) must not overflow."""
    torch.manual_seed(6)

    w = draw_latitude(
        0.999, 0.0, 50.0, torch.full((1000,), 0.01), eps_target=1.0, q_target=0.05, speed=1.0
    )

    assert torch.isfinite(w).all()
    assert (w > -1).all() and (w < 1).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/diffusion/test_forward.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tnbbeta_vae.diffusion'`

- [ ] **Step 3: Write the implementation**

```python
# src/tnbbeta_vae/diffusion/__init__.py
"""Forward-noising process for TNBBetaSpherical, used by the diffusion prior."""
```

```python
# src/tnbbeta_vae/diffusion/forward.py
"""Closed-form forward-noising primitives for the TNBbeta latitude.

Three pieces, each verified against TNBBetaUnivariate's closed-form density
in notebooks/tnbbeta_*.py before being promoted here (see
docs/superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md for the
derivation):

1. ``resize_eps``: TNBbeta's Theorem-4.1 auxiliary count C is exactly
   mergeable/splittable in its ``eps`` (size) parameter, holding ``q``
   fixed -- a Beta-Binomial thinning (eps decreasing) or an independent
   negative-binomial merge (eps increasing).
2. ``leisen_step``: Leisen, Mena, Palma Mancilla & Rossini (2019,
   arXiv:1812.07271)'s closed-form reversible NB(r,q) Markov chain, used as
   a one-shot (no path simulation) forward-noising kernel for C that
   converges to NB(eps, q_target) as t -> infinity, regardless of the
   starting q.
3. ``draw_latitude``: combines both with the exact p-decoupling identity
   (p enters the latitude purely as an additive logit-space shift) into the
   full forward-noising recipe for the latitude ``w = 2y - 1``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["draw_latitude", "leisen_step", "resize_eps"]

_EPS = 1e-6
_LOGIT_CLAMP = 1e-6


def resize_eps(c: Tensor, eps_from: Tensor, eps_to: Tensor, q: Tensor) -> Tensor:
    """One-shot resize of an NB(eps_from, 1-q) count to NB(eps_to, 1-q), same q.

    ``eps_to < eps_from`` (per element): Beta-Binomial splits ``c`` and
    keeps the ``eps_to``-sized piece. ``eps_to > eps_from``: merges in an
    independent fresh ``NB(eps_to - eps_from, 1-q)`` piece. Mixed batches
    (some elements splitting, others merging) are supported.

    Args:
        c: Counts to resize, any shape.
        eps_from: Current size parameter, broadcastable to ``c``.
        eps_to: Target size parameter, broadcastable to ``c``.
        q: TNBbeta concentration parameter (shared, unaffected by the
            resize), broadcastable to ``c``.

    Returns:
        Resized counts, same shape as the broadcast of the inputs.
    """
    c, eps_from, eps_to, q = torch.broadcast_tensors(c, eps_from, eps_to, q)
    delta = eps_to - eps_from
    is_split = delta <= 0

    # Both branches are computed for every element (torch.where evaluates
    # eagerly), so each branch's own inputs are clamped positive even where
    # that branch's result will be discarded -- otherwise an invalid shape
    # parameter (e.g. a negative Beta shape) could raise or produce nan
    # before the where() ever gets to pick the valid branch.
    split_pi = torch.distributions.Beta(
        eps_to.clamp_min(_EPS), (-delta).clamp_min(_EPS)
    ).sample()
    split_result = torch.distributions.Binomial(total_count=c, probs=split_pi).sample()

    merge_piece = torch.distributions.NegativeBinomial(
        delta.clamp_min(_EPS), probs=1 - q
    ).sample()
    merge_result = c + merge_piece

    return torch.where(is_split, split_result, merge_result)


def leisen_step(c0: Tensor, t: Tensor, eps: Tensor, q_target: float, speed: float) -> Tensor:
    """Closed-form, one-shot draw of C_t | C_0 (no intermediate-step simulation).

    Identity at ``t=0``; converges to ``NB(eps, q_target)`` as ``t ->
    infinity``, regardless of ``C_0``'s own distribution.

    Args:
        c0: Starting counts, any shape.
        t: Diffusion time, broadcastable to ``c0``, >= 0.
        eps: Size parameter (held fixed across this call), broadcastable to ``c0``.
        q_target: Target concentration in ``[0, 1)`` -- NB(r, q) requires q < 1,
            so this can be arbitrarily close to but never exactly 0.
        speed: Leisen et al.'s rate constant ``c`` (name avoided here to not
            collide with the count tensor ``c0``).

    Returns:
        C_t, same shape as the broadcast of the inputs.
    """
    c0, t, eps = torch.broadcast_tensors(c0, t, eps)
    theta_t = (1 - q_target) / (torch.exp(speed * t) - q_target)
    y_bin = torch.distributions.Binomial(total_count=c0, probs=theta_t).sample()
    z = torch.distributions.NegativeBinomial(
        eps + y_bin, probs=1 - q_target * (1 - theta_t)
    ).sample()
    return y_bin + z


def draw_latitude(
    p_data: Tensor | float,
    q_data: Tensor | float,
    eps_data: Tensor | float,
    t: Tensor,
    eps_target: float,
    q_target: float,
    speed: float,
) -> Tensor:
    """Draws the forward-noised latitude w_t = 2*Y_t - 1, in (-1, 1).

    At ``t=0`` this recovers ``TNBbeta(p_data, q_data, eps_data)`` (as a
    latitude) exactly; as ``t -> infinity`` it converges to
    ``TNBbeta(0.5, q_target, eps_target)``, regardless of
    ``(p_data, q_data, eps_data)``.

    Args:
        p_data, q_data, eps_data: The starting TNBbeta parameters, scalars
            or tensors broadcastable to ``t``.
        t: Diffusion time, >= 0, any shape.
        eps_target: The schedule's asymptotic size parameter (for the
            spherical model this is ``(latent_dim - 1) / 2``, not an
            arbitrary constant -- see
            ``tnbbeta_vae.models.priors.tnbbeta_spherical.uniform_prior_params``).
        q_target: The schedule's asymptotic concentration, in ``[0, 1)``.
        speed: Leisen et al.'s rate constant.

    Returns:
        w_t, same shape as ``t``.
    """
    p_data, q_data, eps_data, t = torch.broadcast_tensors(
        torch.as_tensor(p_data, dtype=t.dtype),
        torch.as_tensor(q_data, dtype=t.dtype),
        torch.as_tensor(eps_data, dtype=t.dtype),
        t,
    )
    eps_t = eps_target + (eps_data - eps_target) * torch.exp(-t)
    c_data = torch.distributions.NegativeBinomial(eps_data, probs=1 - q_data).sample()
    c0_prime = resize_eps(c_data, eps_data, eps_t, q_data)
    c_t = leisen_step(c0_prime, t, eps_t, q_target, speed)

    a_t = torch.distributions.NegativeBinomial(eps_t + c_t, probs=0.5).sample()
    b_t = torch.distributions.NegativeBinomial(eps_t + c_t, probs=0.5).sample()
    u_t = torch.distributions.Beta(eps_t + c_t + a_t, eps_t + c_t + b_t).rsample()

    psi_t = _logit(p_data) * torch.exp(-t)
    y_t = torch.sigmoid(psi_t + _logit(u_t))
    return 2 * y_t - 1


def _logit(x: Tensor) -> Tensor:
    """log(x / (1-x)), with x clamped away from the boundary for finiteness."""
    x = x.clamp(_LOGIT_CLAMP, 1 - _LOGIT_CLAMP)
    return torch.log(x) - torch.log1p(-x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/diffusion/test_forward.py -v`
Expected: PASS (all 7 tests). If a statistical test is flaky at the chosen
seed, re-run once -- these are randomized at fixed seeds, not adaptive, so a
genuine failure should reproduce; do not loosen `_KS_STAT_MAX` to make a
real mismatch pass.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run pyright`
Expected: no errors in the new files.

- [ ] **Step 6: Commit**

```bash
git add src/tnbbeta_vae/diffusion/__init__.py src/tnbbeta_vae/diffusion/forward.py tests/diffusion/__init__.py tests/diffusion/test_forward.py
git commit -m "Add closed-form forward-noising primitives for TNBbeta's latitude

Promotes the merge/split epsilon semigroup, the Leisen et al. (2019)
closed-form q-kernel, and the p-logit-decoupling identity from
notebooks/tnbbeta_*.py into tested library code."
```

---

## Task 2: Diffusion schedule (`diffusion/schedule.py`)

**Files:**
- Create: `src/tnbbeta_vae/diffusion/schedule.py`
- Test: `tests/diffusion/test_schedule.py`

**Interfaces:**
- Consumes: `tnbbeta_vae.diffusion.forward.draw_latitude` (Task 1).
- Produces: `DiffusionSchedule` (frozen dataclass): fields `eps_target:
  float`, `q_target: float = 0.05`, `speed: float = 1.0`; method
  `draw_latitude(self, p_data, q_data, eps_data, t) -> Tensor`.

Note: the spec describes separate `eps_t(t)`/`psi_t(t)` methods; this task
keeps those as private details inside `forward.draw_latitude` (Task 1)
rather than exposing them a second time here, since nothing else needs
them independently -- `DiffusionSchedule` exists to carry the three tunable
constants as one object and give `draw_latitude` a home that doesn't
require repeating them at every call site.

- [ ] **Step 1: Write the failing tests**

```python
# tests/diffusion/test_schedule.py
from __future__ import annotations

import pytest
import torch
from scipy import stats

from tnbbeta_vae.diffusion.schedule import DiffusionSchedule
from tnbbeta_vae.distributions import TNBBetaUnivariate

_N = 20_000
_KS_STAT_MAX = 0.03


def test_fields_default_as_specified() -> None:
    schedule = DiffusionSchedule(eps_target=1.0)

    assert schedule.eps_target == 1.0
    assert schedule.q_target == 0.05
    assert schedule.speed == 1.0


def test_draw_latitude_matches_forward_draw_latitude_directly() -> None:
    torch.manual_seed(0)
    schedule = DiffusionSchedule(eps_target=1.0, q_target=0.05, speed=1.0)

    w = schedule.draw_latitude(0.7, 0.3, 2.0, torch.full((_N,), 8.0))
    y = (w + 1) / 2

    target = TNBBetaUnivariate(
        torch.tensor(0.5), torch.tensor(0.05), torch.tensor(1.0)
    ).rsample((_N,))
    stat, _ = stats.ks_2samp(y.numpy(), target.numpy())
    assert stat < _KS_STAT_MAX


def test_is_immutable() -> None:
    schedule = DiffusionSchedule(eps_target=1.0)

    with pytest.raises(AttributeError):
        schedule.eps_target = 2.0  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/diffusion/test_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tnbbeta_vae.diffusion.schedule'`

- [ ] **Step 3: Write the implementation**

```python
# src/tnbbeta_vae/diffusion/schedule.py
"""A tunable forward-noising schedule for the TNBbeta latitude."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from tnbbeta_vae.diffusion.forward import draw_latitude

__all__ = ["DiffusionSchedule"]


@dataclass(frozen=True)
class DiffusionSchedule:
    """The three constants that shape the forward-noising process.

    Attributes:
        eps_target: The schedule's asymptotic size parameter. For the
            spherical model this must be ``(latent_dim - 1) / 2`` (see
            ``tnbbeta_vae.models.priors.tnbbeta_spherical.uniform_prior_params``),
            not an arbitrary constant -- it is derived from ``latent_dim``
            by the caller, not defaulted here.
        q_target: The schedule's asymptotic concentration, in ``[0, 1)``.
        speed: Leisen et al.'s rate constant; larger values converge to the
            target faster as a function of ``t``.
    """

    eps_target: float
    q_target: float = 0.05
    speed: float = 1.0

    def draw_latitude(
        self,
        p_data: Tensor | float,
        q_data: Tensor | float,
        eps_data: Tensor | float,
        t: Tensor,
    ) -> Tensor:
        """Draws the forward-noised latitude at time ``t``; see ``forward.draw_latitude``."""
        return draw_latitude(
            p_data, q_data, eps_data, t, self.eps_target, self.q_target, self.speed
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/diffusion/test_schedule.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run pyright`

- [ ] **Step 6: Commit**

```bash
git add src/tnbbeta_vae/diffusion/schedule.py tests/diffusion/test_schedule.py
git commit -m "Add DiffusionSchedule wrapping the forward-noising constants"
```

---

## Task 3: Expose Householder reflection + `noise_to` (`diffusion/noising.py`)

**Files:**
- Modify: `src/tnbbeta_vae/distributions/tnbbeta_spherical.py` (rename
  `_householder_reflect` -> `householder_reflect`, add to `__all__`, update
  its one call site in `rsample`)
- Create: `src/tnbbeta_vae/diffusion/noising.py`
- Test: `tests/diffusion/test_noising.py`

**Interfaces:**
- Consumes: `tnbbeta_vae.distributions.tnbbeta_spherical.householder_reflect`
  (this task), `tnbbeta_vae.diffusion.schedule.DiffusionSchedule` (Task 2).
- Produces: `noise_to(mean_direction: Tensor, t: Tensor, schedule:
  DiffusionSchedule, p_start: float, q_start: float, eps_start: float) ->
  Tensor` -- returns a unit-norm point on `S^(dim-1)`, same leading shape
  as `t`/`mean_direction`'s batch dimensions.

- [ ] **Step 1: Make the pure-rename change to `tnbbeta_spherical.py`**

In `src/tnbbeta_vae/distributions/tnbbeta_spherical.py`:
- Line 46: add `"householder_reflect"` to `__all__` (keep alphabetical: `__all__ = ["TNBBetaSpherical", "householder_reflect"]` -- but Google/isort style here lists symbols as defined; match the existing single-entry list's style by appending).
- Line 197: change `return _householder_reflect(z, mean_direction)` to `return householder_reflect(z, mean_direction)`.
- Line 209: change `def _householder_reflect(z: Tensor, mean_direction: Tensor) -> Tensor:` to `def householder_reflect(z: Tensor, mean_direction: Tensor) -> Tensor:`.
- In that function's docstring, add one sentence: `"Also used directly by tnbbeta_vae.diffusion.noising to noise/reflect points during the diffusion prior's forward process."`

This is a pure visibility rename -- no change to any logic, tensor
operation, or numerical behavior. The existing test suite for
`TNBBetaSpherical` (`tests/distributions/test_tnbbeta_spherical.py`) must
still pass unmodified after this change, since it exercises `rsample()`
end-to-end and does not reference the private name directly.

- [ ] **Step 2: Confirm the existing TNBBetaSpherical tests still pass**

Run: `uv run pytest tests/distributions/test_tnbbeta_spherical.py -v`
Expected: PASS, unchanged (confirms the rename broke nothing).

- [ ] **Step 3: Write the failing tests for `noise_to`**

```python
# tests/diffusion/test_noising.py
from __future__ import annotations

import torch
from scipy import stats

from tnbbeta_vae.diffusion.noising import noise_to
from tnbbeta_vae.diffusion.schedule import DiffusionSchedule
from tnbbeta_vae.distributions import TNBBetaSpherical

_N = 20_000
_KS_STAT_MAX = 0.03


def _random_unit_vector(dim: int) -> torch.Tensor:
    v = torch.randn(dim)
    return v / v.norm()


def test_noise_to_is_identity_at_t_zero() -> None:
    torch.manual_seed(0)
    mean_direction = _random_unit_vector(5).expand(10, -1)
    schedule = DiffusionSchedule(eps_target=1.0)

    z = noise_to(
        mean_direction, torch.zeros(10), schedule, p_start=0.999, q_start=0.0, eps_start=50.0
    )

    assert torch.allclose(z, mean_direction, atol=1e-3)


def test_noise_to_converges_to_target_tnbbeta_spherical() -> None:
    """Large t: the cosine-to-mean_direction distribution matches the target."""
    torch.manual_seed(1)
    dim = 5
    mean_direction = _random_unit_vector(dim).expand(_N, -1)
    schedule = DiffusionSchedule(eps_target=2.0, q_target=0.05, speed=1.0)

    z = noise_to(
        mean_direction, torch.full((_N,), 10.0), schedule,
        p_start=0.999, q_start=0.0, eps_start=50.0,
    )  # fmt: skip
    cosine = (z * mean_direction).sum(dim=-1)

    target = TNBBetaSpherical(
        _random_unit_vector(dim), 0.5, 0.05, 2.0
    ).rsample((_N,))
    target_cosine = (target * _random_unit_vector(dim)).sum(dim=-1)
    # Compare against a fresh TNBBetaSpherical draw's own cosine-to-its-own-
    # mean_direction, which is mean_direction-independent by construction --
    # this is the same "w" latitude variable either way.
    stat, _ = stats.ks_2samp(cosine.numpy(), target_cosine.numpy())
    assert stat < _KS_STAT_MAX


def test_noise_to_output_is_unit_norm() -> None:
    torch.manual_seed(2)
    mean_direction = _random_unit_vector(5).expand(100, -1)
    schedule = DiffusionSchedule(eps_target=1.0)

    z = noise_to(
        mean_direction, torch.full((100,), 3.0), schedule,
        p_start=0.999, q_start=0.0, eps_start=50.0,
    )  # fmt: skip

    assert torch.allclose(z.norm(dim=-1), torch.ones(100), atol=1e-4)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/diffusion/test_noising.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tnbbeta_vae.diffusion.noising'`

- [ ] **Step 5: Write the implementation**

```python
# src/tnbbeta_vae/diffusion/noising.py
"""Forward-noising a point on the sphere toward TNBBetaSpherical's reference.

The latitude (``diffusion.forward``/``diffusion.schedule``) is the whole
problem: the non-latitude direction is always drawn fresh and uniform,
independent of the latitude's own parameters, and the mean direction never
needs to move -- a uniform latitude plus an independent uniform azimuthal
direction, reflected through *any* fixed pole, is exactly rotationally
symmetric (see TNBBetaSpherical's own module docstring, and
docs/superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md point
5). ``noise_to`` is therefore the same two-step construction
``TNBBetaSpherical.rsample()`` already uses, with the forward-noised
latitude from ``diffusion.schedule`` in place of a fresh TNBbeta draw.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tnbbeta_vae.diffusion.schedule import DiffusionSchedule
from tnbbeta_vae.distributions.tnbbeta_spherical import householder_reflect

__all__ = ["noise_to"]


def noise_to(
    mean_direction: Tensor,
    t: Tensor,
    schedule: DiffusionSchedule,
    p_start: float,
    q_start: float,
    eps_start: float,
) -> Tensor:
    """Forward-noises ``mean_direction`` to diffusion time ``t``.

    Args:
        mean_direction: Unit vectors to noise from, shape ``(..., dim)``.
        t: Diffusion time, >= 0, shape ``(...)`` matching ``mean_direction``'s
            batch dimensions.
        schedule: The forward process's target constants.
        p_start: Near-point-mass starting ``p`` (close to 1).
        q_start: Starting ``q`` (0 by design -- no ring structure at the
            point-mass start).
        eps_start: Near-point-mass starting ``eps`` (large).

    Returns:
        Unit-norm points on the sphere, same shape as ``mean_direction``.
    """
    w_t = schedule.draw_latitude(p_start, q_start, eps_start, t)
    dim = mean_direction.shape[-1]
    g = torch.randn(
        (*w_t.shape, dim - 1), dtype=mean_direction.dtype, device=mean_direction.device
    )
    v = g / g.norm(dim=-1, keepdim=True)
    radius = torch.sqrt((1 - w_t**2).clamp_min(0)).unsqueeze(-1)
    pole_frame = torch.cat([w_t.unsqueeze(-1), radius * v], dim=-1)
    return householder_reflect(pole_frame, mean_direction)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/diffusion/test_noising.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run pyright`

- [ ] **Step 8: Commit**

```bash
git add src/tnbbeta_vae/distributions/tnbbeta_spherical.py src/tnbbeta_vae/diffusion/noising.py tests/diffusion/test_noising.py
git commit -m "Expose householder_reflect; add noise_to for the spherical forward process

Pure-rename change to tnbbeta_spherical.py (no logic change, confirmed by
the unmodified existing TNBBetaSpherical test suite still passing)."
```

---

## Task 4: Sphere denoiser MLP (`models/architectures/denoiser_mlp.py`)

**Files:**
- Create: `src/tnbbeta_vae/models/architectures/denoiser_mlp.py`
- Test: `tests/models/architectures/test_denoiser_mlp.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (independent).
- Produces: `SphereDenoiserMLP(nn.Module)`: `__init__(self, latent_dim:
  int, hidden_dim: int = 256, depth: int = 4, time_embed_dim: int = 64) ->
  None`; `forward(self, z: Tensor, t: Tensor) -> Tensor` -- `z` shape
  `(..., latent_dim)`, `t` shape `(...)` matching `z`'s batch dims, returns
  a unit-norm tensor of shape `(..., latent_dim)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/architectures/test_denoiser_mlp.py
from __future__ import annotations

import torch

from tnbbeta_vae.models.architectures.denoiser_mlp import SphereDenoiserMLP


def test_output_is_unit_norm_and_correct_shape() -> None:
    torch.manual_seed(0)
    model = SphereDenoiserMLP(latent_dim=8, hidden_dim=32, depth=2)
    z = torch.randn(6, 8)
    z = z / z.norm(dim=-1, keepdim=True)
    t = torch.rand(6) * 10

    out = model(z, t)

    assert out.shape == (6, 8)
    assert torch.allclose(out.norm(dim=-1), torch.ones(6), atol=1e-5)


def test_different_t_gives_different_output() -> None:
    """Sanity check that t actually conditions the network, not just z."""
    torch.manual_seed(1)
    model = SphereDenoiserMLP(latent_dim=4, hidden_dim=16, depth=2)
    z = torch.randn(3, 4)
    z = z / z.norm(dim=-1, keepdim=True)

    out_t0 = model(z, torch.zeros(3))
    out_t1 = model(z, torch.full((3,), 5.0))

    assert not torch.allclose(out_t0, out_t1)


def test_gradients_flow_to_every_parameter() -> None:
    torch.manual_seed(2)
    model = SphereDenoiserMLP(latent_dim=4, hidden_dim=16, depth=2)
    z = torch.randn(5, 4)
    z = z / z.norm(dim=-1, keepdim=True)
    t = torch.rand(5) * 10

    out = model(z, t)
    out.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/architectures/test_denoiser_mlp.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/tnbbeta_vae/models/architectures/denoiser_mlp.py
"""A small MLP that predicts a clean sphere point from a noised one.

latent_dim is a small vector (project default 8), not spatial, so this is
a plain MLP conditioned on a sinusoidal time embedding -- no convolutions,
unlike models/architectures/conv.py's image encoder/decoder.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["SphereDenoiserMLP"]


class SphereDenoiserMLP(nn.Module):
    """Predicts a unit-norm z_0 estimate from a noised z_t and timestep t."""

    def __init__(
        self, latent_dim: int, hidden_dim: int = 256, depth: int = 4, time_embed_dim: int = 64
    ) -> None:
        """Initializes the network.

        Args:
            latent_dim: Ambient dimension of the sphere S^(latent_dim - 1).
            hidden_dim: Width of each hidden layer.
            depth: Number of hidden layers.
            time_embed_dim: Dimension of the sinusoidal time embedding fed
                in alongside z.
        """
        super().__init__()
        self.time_embed_dim = time_embed_dim
        layers: list[nn.Module] = [nn.Linear(latent_dim + time_embed_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        """Predicts a unit-norm z_0 estimate.

        Args:
            z: Noised points, shape ``(..., latent_dim)``.
            t: Diffusion time, shape ``(...)`` matching ``z``'s batch dims.

        Returns:
            Unit-norm predictions, same shape as ``z``.
        """
        embed = _sinusoidal_embedding(t, self.time_embed_dim)
        prediction = self.net(torch.cat([z, embed], dim=-1))
        return prediction / prediction.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _sinusoidal_embedding(t: Tensor, dim: int) -> Tensor:
    """Standard transformer-style sinusoidal embedding of a scalar time.

    Args:
        t: Shape ``(...)``.
        dim: Output embedding dimension; must be even.

    Returns:
        Shape ``(..., dim)``.
    """
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=t.dtype, device=t.device) / half
    )
    angles = t.unsqueeze(-1) * frequencies
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/architectures/test_denoiser_mlp.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run pyright`

- [ ] **Step 6: Commit**

```bash
git add src/tnbbeta_vae/models/architectures/denoiser_mlp.py tests/models/architectures/test_denoiser_mlp.py
git commit -m "Add SphereDenoiserMLP: sinusoidal-time-conditioned z_0 predictor"
```

---

## Task 5: `tnbbeta_spherical_diffusion_prior` model

**Files:**
- Create: `src/tnbbeta_vae/models/sphere_diffusion.py`
- Modify: `src/tnbbeta_vae/models/__init__.py` (import + `__all__`)
- Test: `tests/models/test_sphere_diffusion.py`

**Interfaces:**
- Consumes: `DiffusionSchedule`, `noise_to` (Tasks 2-3), `SphereDenoiserMLP`
  (Task 4), `tnbbeta_vae.training.load_model_checkpoint`,
  `tnbbeta_vae.paths.checkpoint_dir`, `tnbbeta_vae.models.conv_vae.
  ConvTNBBetaSphericalVAE`, `tnbbeta_vae.models.priors.tnbbeta_spherical.
  uniform_prior_params`, `tnbbeta_vae.registry.register_model`.
- Produces: `SphereDiffusionPriorConfig(BaseModel)`, `SphereDiffusionPrior
  (nn.Module)` registered as `"tnbbeta_spherical_diffusion_prior"`, with
  `training_step(self, images: Tensor) -> dict[str, Tensor]` and
  `generate(self, num_samples: int) -> Tensor` (images, matching
  `ConvTNBBetaSphericalVAE.generate`'s contract so `apps/eval/fid.py` works
  unchanged).

- [ ] **Step 1: Write the failing tests**

```python
# tests/models/test_sphere_diffusion.py
"""Tests for tnbbeta_vae.models.sphere_diffusion."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tnbbeta_vae.models import (
    ConvGaussianVAE,
    ConvGaussianVAEConfig,
    ConvTNBBetaSphericalVAE,
    ConvTNBBetaSphericalVAEConfig,
)
from tnbbeta_vae.models.sphere_diffusion import (
    SphereDiffusionPrior,
    SphereDiffusionPriorConfig,
)
from tnbbeta_vae.registry import build_model, list_registered_models
from tnbbeta_vae.training import Trainer


def _train_tiny_vae(tmp_path: Path, run_name: str = "vae_run") -> str:
    """Writes a real (tiny, 1-epoch) VAE checkpoint under tmp_path; returns its run name."""
    model = ConvTNBBetaSphericalVAE(ConvTNBBetaSphericalVAEConfig(latent_dim=4, hidden_channels=8))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model, optimizer, "conv_tnbbeta_spherical_vae", model.config,
        runs_dir=tmp_path / "runs",
    )  # fmt: skip
    trainer.fit(
        [torch.rand(4, 3, 32, 32) for _ in range(2)],
        num_epochs=1,
        checkpoint_dir=tmp_path / "checkpoints" / run_name,
    )
    return run_name


def _train_tiny_gaussian_vae(tmp_path: Path, run_name: str = "gauss_run") -> str:
    model = ConvGaussianVAE(ConvGaussianVAEConfig(latent_dim=4, hidden_channels=8))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model, optimizer, "conv_gaussian_vae", model.config, runs_dir=tmp_path / "runs"
    )
    trainer.fit(
        [torch.rand(4, 3, 32, 32) for _ in range(2)],
        num_epochs=1,
        checkpoint_dir=tmp_path / "checkpoints" / run_name,
    )
    return run_name


@pytest.fixture(autouse=True)
def _isolated_checkpoint_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    return tmp_path


def _small_diffusion_model(tmp_path: Path, **overrides: object) -> SphereDiffusionPrior:
    run_name = _train_tiny_vae(tmp_path)
    config = SphereDiffusionPriorConfig(
        vae_run_name=run_name,
        denoiser_hidden_dim=16,
        denoiser_depth=2,
        num_reverse_steps=3,
        **overrides,  # pyright: ignore[reportArgumentType]
    )
    return SphereDiffusionPrior(config)


def test_registered_under_expected_name() -> None:
    assert "tnbbeta_spherical_diffusion_prior" in list_registered_models()


def test_build_via_registry_applies_overrides(tmp_path: Path) -> None:
    run_name = _train_tiny_vae(tmp_path)

    model = build_model(
        "tnbbeta_spherical_diffusion_prior", vae_run_name=run_name, denoiser_hidden_dim=16
    )

    assert isinstance(model, SphereDiffusionPrior)
    assert model.config.denoiser_hidden_dim == 16


def test_rejects_a_non_tnbbeta_spherical_vae_checkpoint(tmp_path: Path) -> None:
    """Review Focus: pointing at the wrong model kind must fail clearly."""
    run_name = _train_tiny_gaussian_vae(tmp_path)

    with pytest.raises(TypeError, match="ConvTNBBetaSphericalVAE"):
        SphereDiffusionPrior(SphereDiffusionPriorConfig(vae_run_name=run_name))


def test_training_step_returns_finite_loss(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    images = torch.rand(5, 3, 32, 32)

    outputs = model.training_step(images)

    assert "loss" in outputs
    assert outputs["loss"].dim() == 0
    assert torch.isfinite(outputs["loss"])


def test_training_step_never_samples_t_below_t_min(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review Focus: t must stay bounded away from 0, not Uniform(0, t_max)."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path, t_min=0.5, t_max=1.0)
    seen_t: list[torch.Tensor] = []
    original = torch.distributions.Uniform

    class _RecordingUniform(original):  # type: ignore[misc]
        def sample(self, *args: object, **kwargs: object) -> torch.Tensor:
            value = super().sample(*args, **kwargs)
            seen_t.append(value)
            return value

    monkeypatch.setattr(torch.distributions, "Uniform", _RecordingUniform)
    model.training_step(torch.rand(20, 3, 32, 32))

    assert seen_t, "training_step never drew a t -- test fixture is out of date"
    assert all((t >= 0.5).all() for t in seen_t)


def test_training_step_gradients_flow_only_to_the_denoiser(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    images = torch.rand(5, 3, 32, 32)

    outputs = model.training_step(images)
    outputs["loss"].backward()

    for name, param in model.denoiser.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_vae_parameters_receive_no_gradient(tmp_path: Path) -> None:
    """Review Focus: 'frozen' means .grad stays None after a real backward(), not just
    requires_grad=False."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    images = torch.rand(5, 3, 32, 32)

    outputs = model.training_step(images)
    outputs["loss"].backward()

    for param in model.vae.parameters():
        assert not param.requires_grad
        assert param.grad is None


def test_train_mode_does_not_unfreeze_the_vae(tmp_path: Path) -> None:
    model = _small_diffusion_model(tmp_path)

    model.train()

    assert model.denoiser.training
    assert not model.vae.training


def test_generate_returns_valid_images(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)

    images = model.generate(4)

    assert images.shape == (4, 3, 32, 32)
    assert torch.isfinite(images).all()
    assert images.min() >= 0 and images.max() <= 1


def test_generate_handles_single_reverse_step(tmp_path: Path) -> None:
    """Review Focus: the reverse loop's boundary (K=1) must not produce nan."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path, num_reverse_steps=1)

    images = model.generate(3)

    assert torch.isfinite(images).all()


def test_trainer_runs_end_to_end(tmp_path: Path) -> None:
    """Smoke test: the real model, the real Trainer, unstructured random batches."""
    torch.manual_seed(0)
    model = _small_diffusion_model(tmp_path)
    optimizer = torch.optim.Adam(model.denoiser.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        model_name="tnbbeta_spherical_diffusion_prior",
        config=model.config,
        runs_dir=tmp_path / "runs2",
    )
    dataloader = [torch.rand(4, 3, 32, 32) for _ in range(3)]

    trainer.fit(dataloader, num_epochs=2)

    metrics_path = trainer.run_logger.run_dir / "metrics.jsonl"
    assert metrics_path.exists()
    assert len(metrics_path.read_text().splitlines()) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_sphere_diffusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tnbbeta_vae.models.sphere_diffusion'`

- [ ] **Step 3: Write the implementation**

```python
# src/tnbbeta_vae/models/sphere_diffusion.py
"""A learned diffusion prior for ConvTNBBetaSphericalVAE's latent space.

Wraps a frozen, pretrained ConvTNBBetaSphericalVAE checkpoint and a small
MLP denoiser trained by direct z_0-regression against the closed-form
forward-noising process in tnbbeta_vae.diffusion. Kept frozen rather than
trained jointly -- see
docs/superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md point 8
for why (citing Rombach et al. 2022's explicit finding against joint
training, and this project's own need for an attributable comparison
against the VAE's existing fixed-prior baseline).
"""

from __future__ import annotations

import torch
from pydantic import BaseModel
from torch import Tensor, nn

from tnbbeta_vae.diffusion.noising import noise_to
from tnbbeta_vae.diffusion.schedule import DiffusionSchedule
from tnbbeta_vae.models.architectures.denoiser_mlp import SphereDenoiserMLP
from tnbbeta_vae.models.conv_vae import ConvTNBBetaSphericalVAE
from tnbbeta_vae.models.priors.tnbbeta_spherical import uniform_prior_params
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.registry import register_model
from tnbbeta_vae.training.checkpoint import load_model_checkpoint

__all__ = ["SphereDiffusionPrior", "SphereDiffusionPriorConfig"]


class SphereDiffusionPriorConfig(BaseModel):
    """Hyperparameters for :class:`SphereDiffusionPrior`.

    Attributes:
        vae_run_name: Checkpoint directory name (under
            ``$TNBBETA_CHECKPOINT_DIR``) of the frozen, pretrained
            ConvTNBBetaSphericalVAE to build this prior on top of.
        q_target: Forward process's target concentration; can approach but
            never reach 0 (NB(r,q) requires q < 1).
        p_start: Near-point-mass starting median, close to 1.
        eps_start: Near-point-mass starting size parameter, deliberately
            large -- a fixed diffusion hyperparameter, not read from the
            VAE's own per-example posterior uncertainty.
        t_min: Smallest diffusion time training samples t from; kept above
            0 so the denoising task near t=0 isn't trivial (predicting z_0
            from z_t ~= z_0 carries no training signal).
        t_max: Largest diffusion time training samples t from, and the
            reverse sampler's starting time.
        speed: Leisen et al.'s rate constant.
        num_reverse_steps: Number of predict-then-renoise steps in generate().
        denoiser_hidden_dim: Width of the denoiser MLP's hidden layers.
        denoiser_depth: Number of the denoiser MLP's hidden layers.
    """

    vae_run_name: str
    q_target: float = 0.05
    p_start: float = 0.999
    eps_start: float = 50.0
    t_min: float = 0.01
    t_max: float = 10.0
    speed: float = 1.0
    num_reverse_steps: int = 50
    denoiser_hidden_dim: int = 256
    denoiser_depth: int = 4


@register_model("tnbbeta_spherical_diffusion_prior", config_cls=SphereDiffusionPriorConfig)
class SphereDiffusionPrior(nn.Module):
    """A learned prior over ConvTNBBetaSphericalVAE's latent sphere."""

    def __init__(self, config: SphereDiffusionPriorConfig) -> None:
        """Initializes the model, loading and freezing the pretrained VAE.

        Args:
            config: Hyperparameters; see :class:`SphereDiffusionPriorConfig`.

        Raises:
            TypeError: If the checkpoint at ``config.vae_run_name`` is not a
                ``ConvTNBBetaSphericalVAE``.
        """
        super().__init__()
        self.config = config

        vae_path = checkpoint_dir() / config.vae_run_name / "final.pt"
        vae, _ = load_model_checkpoint(vae_path)
        if not isinstance(vae, ConvTNBBetaSphericalVAE):
            raise TypeError(
                "SphereDiffusionPrior requires a ConvTNBBetaSphericalVAE checkpoint "
                f"at vae_run_name={config.vae_run_name!r}, got {type(vae).__name__}."
            )
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad_(False)

        self.latent_dim = vae.config.latent_dim
        eps_target = uniform_prior_params(self.latent_dim)[2]
        self.schedule = DiffusionSchedule(
            eps_target=eps_target, q_target=config.q_target, speed=config.speed
        )
        self.denoiser = SphereDenoiserMLP(
            self.latent_dim, config.denoiser_hidden_dim, config.denoiser_depth
        )

    def train(self, mode: bool = True) -> SphereDiffusionPrior:
        """Puts the denoiser in train/eval mode; the frozen VAE always stays in eval mode."""
        self.training = mode
        self.denoiser.train(mode)
        self.vae.eval()
        return self

    def training_step(self, images: Tensor) -> dict[str, Tensor]:
        """Computes the z_0-regression loss for one batch of images.

        Args:
            images: Input images, shape ``(batch, 3, image_size, image_size)``.

        Returns:
            A dict with ``"loss"`` (mean ``1 - cosine_similarity``).
        """
        with torch.no_grad():
            posterior = self.vae.posterior_and_prior(images)[0]
            z_0 = posterior.rsample()
        z_0 = z_0.detach()

        batch_size = z_0.shape[0]
        t = torch.distributions.Uniform(self.config.t_min, self.config.t_max).sample(
            (batch_size,)
        )
        z_t = noise_to(
            z_0, t, self.schedule, self.config.p_start, 0.0, self.config.eps_start
        )
        z_0_hat = self.denoiser(z_t, t)
        loss = 1 - torch.nn.functional.cosine_similarity(z_0_hat, z_0, dim=-1)
        return {"loss": loss.mean()}

    @torch.no_grad()
    def generate(self, num_samples: int) -> Tensor:
        """Generates images via reverse (predict-then-renoise) diffusion.

        Args:
            num_samples: Number of images to generate.

        Returns:
            Images of shape ``(num_samples, 3, image_size, image_size)``.
        """
        z = self.vae.prior().sample(torch.Size([num_samples]))
        steps = torch.linspace(self.config.t_max, 0.0, self.config.num_reverse_steps + 1)
        for i in range(self.config.num_reverse_steps):
            t_current, t_next = steps[i], steps[i + 1]
            z_0_hat = self.denoiser(z, t_current.expand(num_samples))
            if t_next <= 0.0:
                z = z_0_hat
            else:
                z = noise_to(
                    z_0_hat, t_next.expand(num_samples), self.schedule,
                    self.config.p_start, 0.0, self.config.eps_start,
                )  # fmt: skip
        return self.vae.decoder(z)
```

- [ ] **Step 4: Update `models/__init__.py`**

```python
# src/tnbbeta_vae/models/__init__.py
```
Add, alphabetically among the existing imports:
```python
from tnbbeta_vae.models.sphere_diffusion import (
    SphereDiffusionPrior,
    SphereDiffusionPriorConfig,
)
```
and add `"SphereDiffusionPrior"`, `"SphereDiffusionPriorConfig"` to `__all__` (alphabetically).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_sphere_diffusion.py -v`
Expected: PASS (all tests). If `test_training_step_never_samples_t_below_
t_min` fails because `torch.distributions.Uniform.sample` isn't the method
actually invoked (e.g. implementation calls `.rsample()` instead), adjust
the monkeypatch target to match -- the two should behave identically for a
non-reparameterized use, but confirm the implementation picks one and the
test patches that one.

- [ ] **Step 6: Run the full existing test suite to check for regressions**

Run: `uv run pytest -v`
Expected: PASS, including every test from before this plan.

- [ ] **Step 7: Lint and type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run pyright`

- [ ] **Step 8: Commit**

```bash
git add src/tnbbeta_vae/models/sphere_diffusion.py src/tnbbeta_vae/models/__init__.py tests/models/test_sphere_diffusion.py
git commit -m "Add tnbbeta_spherical_diffusion_prior: frozen-VAE diffusion prior model

training_step regresses the denoiser against z_0 using the closed-form
forward-noising process; generate() runs multi-step predict-then-renoise
reverse sampling from the VAE's own Uniform(sphere) prior, then decodes
through the frozen decoder. Trainable via the existing apps/train/main.py
CLI and evaluable via the existing apps/eval/fid.py, both unchanged."
```

---

## Task 6: Qualitative sample grid (`apps/eval/sample_grid.py`)

**Files:**
- Create: `apps/eval/sample_grid.py`
- Test: `tests/apps/test_sample_grid.py`

**Interfaces:**
- Consumes: `tnbbeta_vae.training.load_model_checkpoint`,
  `tnbbeta_vae.paths.checkpoint_dir`, any registered model's `.generate(n)`.
- Produces: a CLI writing a single PNG grid image; no importable functions
  needed by later tasks.

- [ ] **Step 1: Write the failing test**

```python
# tests/apps/test_sample_grid.py
"""Smoke test for apps.eval.sample_grid (tiny fake models, CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from apps.eval import sample_grid
from apps.train import main as train_main
from tnbbeta_vae.data.cifar10 import Cifar10Images


class _FakeCifar10Base:
    def __init__(self, n: int = 16) -> None:
        self._images = torch.rand(n, 3, 32, 32)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._images[index], 0


def _fake_load_cifar10(*_args: object, **_kwargs: object) -> Cifar10Images:
    return Cifar10Images(_FakeCifar10Base())


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TNBBETA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TNBBETA_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TNBBETA_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(train_main, "load_cifar10", _fake_load_cifar10)


def _train(model: str, run_name: str, extra_sets: list[str] | None = None) -> None:
    args = [
        "--model", model, "--dataset", "cifar10", "--run-name", run_name,
        "--set", "latent_dim=4", "--set", "hidden_channels=8",
    ]  # fmt: skip
    for setting in extra_sets or []:
        args += ["--set", setting]
    args += [
        "--epochs", "1", "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
    ]  # fmt: skip
    train_main.main(args)


def test_writes_a_png_comparing_both_priors(tmp_path: Path) -> None:
    _train("conv_tnbbeta_spherical_vae", "vae_run")
    _train(
        "tnbbeta_spherical_diffusion_prior", "diffusion_run",
        extra_sets=["vae_run_name=vae_run", "num_reverse_steps=2"],
    )  # fmt: skip

    output = tmp_path / "grid.png"
    sample_grid.main(
        [
            "--vae-run", "vae_run", "--diffusion-run", "diffusion_run",
            "--num-samples", "4", "--device", "cpu", "--output", str(output),
        ]  # fmt: skip
    )

    assert output.exists() and output.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/apps/test_sample_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apps.eval.sample_grid'`

- [ ] **Step 3: Write the implementation**

```python
# apps/eval/sample_grid.py
"""Writes a side-by-side PNG comparing a VAE's fixed-prior samples against
a diffusion prior's samples, for the qualitative "sharper than blurry VAE
samples" check FID alone doesn't show.

Usage:
    uv run python -m apps.eval.sample_grid --vae-run NAME --diffusion-run NAME \
        [--num-samples 8] [--checkpoint final] [--device cpu] [--output grid.png]
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, cast

import torch
import torchvision.utils as vutils

from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.training import load_model_checkpoint

if TYPE_CHECKING:
    from torch import Tensor, nn


def main(argv: list[str] | None = None) -> None:
    """Parses CLI args and writes the comparison grid.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vae-run", required=True)
    parser.add_argument("--diffusion-run", required=True)
    parser.add_argument("--checkpoint", default="final", choices=["final", "latest"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="sample_grid.png")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    vae_model, _ = load_model_checkpoint(
        checkpoint_dir() / args.vae_run / f"{args.checkpoint}.pt", device
    )
    diffusion_model, _ = load_model_checkpoint(
        checkpoint_dir() / args.diffusion_run / f"{args.checkpoint}.pt", device
    )

    with torch.no_grad():
        vae_images = cast("nn.Module", vae_model).generate(args.num_samples)
        diffusion_images = cast("nn.Module", diffusion_model).generate(args.num_samples)

    _save_comparison_grid(vae_images, diffusion_images, args.output)
    print(f"Wrote {args.output}")


def _save_comparison_grid(top_row: Tensor, bottom_row: Tensor, output_path: str) -> None:
    """Saves a two-row PNG: top=top_row's images, bottom=bottom_row's images."""
    combined = torch.cat([top_row.cpu(), bottom_row.cpu()], dim=0)
    grid = vutils.make_grid(combined, nrow=top_row.shape[0])
    vutils.save_image(grid, output_path)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/apps/test_sample_grid.py -v`
Expected: PASS. If `torchvision.utils` isn't already an importable module
in this environment (check: `uv run python -c "import torchvision.utils"`),
add it: `uv add torchvision` (it is a near-certain existing dependency given
`torchvision.datasets`/`torchvision.transforms` are already used in
`tnbbeta_vae/data/cifar10.py`; confirm rather than assume).

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff format . && uv run ruff check . && uv run pyright`

- [ ] **Step 6: Commit**

```bash
git add apps/eval/sample_grid.py tests/apps/test_sample_grid.py
git commit -m "Add apps/eval/sample_grid.py: qualitative VAE-vs-diffusion-prior comparison"
```

---

## Task 7: Full-suite verification and docs

**Files:**
- Modify: `CLAUDE.md` (one paragraph, package structure notes)

**Interfaces:** none (wrap-up task).

- [ ] **Step 1: Run the complete verification suite**

Run: `uv run ruff format . && uv run ruff check . && uv run pyright && uv run pytest`
Expected: all four pass with zero errors/failures across the whole branch,
not just the files touched by this plan.

- [ ] **Step 2: Add a CLAUDE.md package-structure note**

In the `## Package structure notes` section of `CLAUDE.md`, after the
existing `conv_vmf_vae.py` bullet, add:

```markdown
- `diffusion/`: closed-form forward-noising process for `TNBBetaSpherical`'s
  latitude (epsilon merge/split, the Leisen et al. (2019, arXiv:1812.07271)
  closed-form `q`-kernel, the exact `p`-logit-decoupling identity -- see
  `docs/superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md`).
  `models/sphere_diffusion.py`'s `tnbbeta_spherical_diffusion_prior` wraps a
  frozen, pretrained `conv_tnbbeta_spherical_vae` checkpoint with a small
  `SphereDenoiserMLP` trained by direct `z_0`-regression (not literal
  denoising score matching -- the forward process's closed-form density was
  never derived, only its sampling correctness verified), to test whether a
  learned prior produces sharper CIFAR-10 samples than the fixed
  `Uniform(sphere)` baseline. Train it via the same
  `apps/train/main.py --model tnbbeta_spherical_diffusion_prior --set
  vae_run_name=<run>` CLI every other model uses; compare against the
  baseline VAE's own samples via `apps/eval/fid.py` (unchanged, works on
  either checkpoint) and `apps/eval/sample_grid.py` (new, qualitative
  side-by-side).
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Document the diffusion prior in CLAUDE.md's package structure notes"
```

- [ ] **Step 4: Report the follow-up (not part of this plan's scope)**

This plan delivers working, tested code. It does **not** run the actual
CIFAR-10 experiment -- that's real GPU wall-clock time, out of scope for an
implementation plan. Once this branch is reviewed, the follow-up commands
are:

```bash
# Stage 1: train the VAE (skip if a suitable checkpoint already exists)
uv run python -m apps.train.main --model conv_tnbbeta_spherical_vae \
    --dataset cifar10 --run-name cifar_vae --epochs 50

# Stage 2: train the diffusion prior on top of it
uv run python -m apps.train.main --model tnbbeta_spherical_diffusion_prior \
    --dataset cifar10 --run-name cifar_diffusion \
    --set vae_run_name=cifar_vae --epochs 50

# Evaluate
uv run python -m apps.eval.fid --run-name cifar_vae
uv run python -m apps.eval.fid --run-name cifar_diffusion
uv run python -m apps.eval.sample_grid --vae-run cifar_vae --diffusion-run cifar_diffusion
```

Per CLAUDE.md's `writeup/results/` convention, that run's results belong in
a dated `writeup/results/<topic>_<date>.md` entry recording the exact
commands and the commit this plan was merged at -- a separate step after
the run completes, not part of this plan.

---
