# Axial mixture recovery: does TNBBeta's bimodality get used? 2026-10-07

- **Date:** 2026-10-07
- **Branch commit:** `0c25964` ("Fix VonMisesFisher.rsample NaN at m=2 when
  the transverse draw is exactly 0", branch `worktree-axial-bimodality-datasets`,
  based on `research/sphere-diffusion-prior`; not yet on `master`)
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

**Note on a bug hit and fixed en route:** the first attempt at this run
crashed deterministically (seed 0, `vmf` family, epoch 75) with a
`ValueError` from `torch.distributions.Normal` rejecting a NaN
reconstruction. Root cause, confirmed via instrumentation: at
`latent_dim = 2` (`S^1`), `VonMisesFisher.rsample`'s transverse direction is
a single standard-normal scalar normalized by its own absolute value
(`v / v.norm()`); over a few thousand training steps that draw lands on
exactly `0.0` in float32 often enough to be a real `0/0` division, not a
theoretical risk. This is a pre-existing bug in
`src/tnbbeta_vae/distributions/von_mises_fisher.py` (unrelated to this
plan's own new code), fixed with `clamp_min(1e-12)` on the norm -- the same
guard `_householder_rotation` already uses for the identical near-zero-norm
situation in the same file -- plus a regression test
(`tests/distributions/test_von_mises_fisher.py::test_rsample_on_the_circle_is_finite_even_if_the_transverse_draw_is_zero`).
The results below are from the run *after* that fix (commit `0c25964`); no
non-finite or implausible value appears in the raw output reproduced below.

## Results (single seed)

| family | test_ll | test_kl | angle_error | angle_error_sample | reconstruction_angle_error | prior_manifold_ratio |
|---|---|---|---|---|---|---|
| gaussian | 148.901 | 4.1067 | 0.1996 | 0.7229 | 0.00649 | 0.1392 |
| vmf | 151.706 | 4.3965 | 0.7168 | 0.7173 | 0.00504 | 0.5512 |
| power_spherical | 151.710 | 4.3643 | 0.7167 | 0.7171 | 0.00526 | 0.3757 |
| tnbbeta | 151.244 | 5.4028 | 0.6945 | 0.6951 | 0.00567 | 0.7851 |

TNBBeta-only mechanistic diagnostic: `p_mean = 0.99995`, `p_std = 1.67e-5`,
`m_mean = 26.497` (bimodal iff negative; the uniform-prior boundary is
`m = 0`, i.e. `epsilon = 0.5` at this `latent_dim = 2`; the fitted
`epsilon_mean` here is `m_mean + 0.5 = 26.997`).

## Reading it

**Criterion 1 (mechanistic) fails clearly.** `m_mean = 26.497` is deeply
*positive*, not negative -- roughly 53x the uniform-prior boundary
(`epsilon = 0.5`), not even close to it. `p_mean` sits at `0.99995`, pinned
to the saturation floor, not near the `0.5` the spec's symmetric-bimodal
criterion calls for. This is exactly the MNIST pattern CLAUDE.md already
documents (`p -> 1`, `epsilon` well above threshold): even on a task
*designed* so that only the bimodal shape is Bayes-optimal, this training
run never found or used that capability. `TNBBetaSpherical` collapsed into
behaving like an ordinary unimodal cap, just like the other three families.

**Criterion 2 (comparative) also fails, by this single seed's numbers.**
`tnbbeta`'s `test_ll` (151.244) is *worse* than both `vmf` (151.706) and
`power_spherical` (151.710) -- not by much (~0.46-0.47 nats), but in the
wrong direction -- and only beats `gaussian` (148.901). `tnbbeta`'s
`test_kl` (5.403) is the *largest* of the four, not the smallest. Neither
half of the spec's comparative prediction holds here.

**The caveat the spec flagged is what's actually showing up, and the
evidence supports it.** `vmf` and `power_spherical` -- both *provably*
unimodal, with no way to represent the bimodal posterior at all -- have
`angle_error` of `0.7168`/`0.7167`, essentially at the mod-`pi` chance
ceiling (`E|U|` for `U` uniform on `(-pi/2, pi/2]` is `pi/4 ~= 0.785`), while
their `reconstruction_angle_error` is tiny (`0.00504`/`0.00526`). That gap
is exactly the pattern the plan's write-up template asks to check for:
reconstruction is good (the decoder correctly places the point on the data
manifold near the true angle mod `pi`) while the posterior *centre*'s own
angle carries almost no information about which axis it is -- the
"decoder learned to work around the ambiguity, encoder picks an arbitrary
direction" failure mode, not "the family found a genuinely different
useful representation." `tnbbeta` lands in the same place (`angle_error
0.6945`, also near the chance ceiling, `reconstruction_angle_error
0.00567`), consistent with it having collapsed to the *same* unimodal
behavior as the other two rather than doing anything structurally
different. (`gaussian`'s lower `angle_error`, `0.1996`, is an outlier worth
a note but not a contradiction: its *centre* happens to align well with the
true angle more often than chance while its *sample* error, `0.7229`, is
back near the ceiling -- i.e. its posterior variance is wide enough to wash
out that alignment once actually sampled, which is what the decoder
actually sees during training and generation.)

**One-paragraph verdict:** by the spec's own bar (criteria 1-2 both
holding clearly), this is **not yet a satisfying result** -- `tnbbeta`
behaved exactly like the known MNIST collapse rather than demonstrating its
unique capability, even on a task built so that capability should be
necessary. This argues against moving to Part 2 (DTD) yet: spending real
data-collection and pipeline effort on a real dataset is premature while
the synthetic task that was *designed* to force the bimodal regime still
can't reach it. The next concrete thing to try is on the training-dynamics
side the spec's own "Open risks" section anticipated: an epsilon
initialization change (start the raw-epsilon network output below the
`(latent_dim - 1) / 2` threshold instead of at its default, so training
starts inside the bimodal region instead of approaching it from above) or
a `fixed_epsilon` ablation (analogous to the one already in
`ConvTNBBetaSphericalVAEConfig`, forcing `m < 0` and checking whether the
Monte Carlo ELBO's unbounded log-density terms there destabilize training,
per that same risk note) -- either would help isolate whether epsilon
*can't* profitably go below threshold here or simply never gets pushed
there by the current warm-up schedule.

## Follow-up: isolating "not found" from "doesn't help" at latent_dim=5, 2026-10-08

- **Date:** 2026-10-08
- **Commits:** `80688fb` ("Add fixed_epsilon to MlpVAEConfig for the tnbbeta
  family") and `d9d2957` ("Add --latent-dim/--fixed-epsilon to
  axial_recovery, project to best-fit plane"), branch
  `worktree-axial-bimodality-datasets`.
- **Commands:**
  ```
  uv run python -m apps.synthetic.axial_recovery --out-dir runs/axial_recovery_d5_free --latent-dim 5
  uv run python -m apps.synthetic.axial_recovery --out-dir runs/axial_recovery_d5_fixed --latent-dim 5 --fixed-epsilon 1.5
  ```

The 2026-10-07 run above couldn't tell "TNBBeta's bimodality never gets
*found*" apart from "bimodality wouldn't *help* even if found," because at
`latent_dim = 2` the bimodal threshold (`epsilon < 0.5`) and the univariate
TNBBeta's own `epsilon < 1` boundary-divergence instability (Proposition
3.1) overlap completely -- pushing epsilon low enough to test bimodality
also means hitting a regime already known to be numerically fragile for an
unrelated reason. At `latent_dim = 5` the bimodal threshold is
`(5-1)/2 = 2.0`, so `epsilon = 1.5` gives `m = -0.5` (bimodal) while
staying `> 1` (clear of that other instability) -- this isolates the
question. Two runs: free epsilon at `latent_dim = 5` (does the *same*
collapse recur away from the degenerate `latent_dim = 2` case), and
`fixed_epsilon = 1.5` at `latent_dim = 5` (forces the bimodal shape
directly, bypassing whatever training dynamics would otherwise avoid it).

### Results (single seed, latent_dim = 5)

| family | test_ll | test_kl | angle_error | angle_error_sample | reconstruction_angle_error | prior_manifold_ratio |
|---|---|---|---|---|---|---|
| gaussian | 151.967 | 4.7428 | 0.5220 | 0.7609 | 0.00515 | 0.1779 |
| vmf | 149.527 | 7.4804 | 0.7375 | 0.7363 | 0.00680 | 1.2816 |
| power_spherical | 149.455 | 7.7873 | 0.7362 | 0.7374 | 0.00689 | 1.2643 |
| tnbbeta (free epsilon) | 149.435 | 7.0764 | 0.6907 | 0.6988 | 0.00662 | 1.0127 |
| tnbbeta (fixed_epsilon=1.5) | 149.801 | 7.7644 | 0.6323 | 0.6387 | 0.00760 | 0.9635 |

TNBBeta mechanistic diagnostic, both `latent_dim = 5` variants (bimodal
threshold `m = epsilon - 2.0 < 0`):

| variant | p_mean | p_std | centre_axis_resultant | m_mean | epsilon_mean |
|---|---|---|---|---|---|
| free epsilon | 0.011178 | 0.006641 | 0.1046 | 6.195 | 8.195 |
| fixed_epsilon=1.5 | 0.990019 | 0.007627 | 0.1219 | **-0.5** (exact) | 1.5 (fixed) |

Both JSON outputs were inspected by hand before writing this: every
`test_ll`, `test_kl`, `m_mean` and `p_mean` above is finite and within the
range already seen in the 2026-10-07 run; no NaNs, no crash, in either run.

### Reading it

**(a) Does free training at `latent_dim = 5` also collapse to `m_mean >=
0`, the same pattern as `latent_dim = 2`? Yes, clearly.** `m_mean =
6.195` (fitted `epsilon_mean = 8.195`) is deeply positive, the same
qualitative pattern as the `latent_dim = 2` run's `m_mean = 26.497`
(`epsilon_mean = 26.997`) -- well above the (now higher, `2.0`) bimodal
threshold. `p_mean = 0.0112` sits at the opposite saturation floor from
the `latent_dim = 2` run's `p_mean = 0.99995`, but that's the
`(mu, p) ~ (-mu, 1-p)` alias this family has everywhere, not a different
regime: both runs pin `p` at an extreme, just an arbitrary one, of the
parameterization's redundant sign. Free training at a higher `latent_dim`
does not change the outcome: the same collapse recurs away from the
degenerate `latent_dim = 2` overlap.

**(b) Does fixing `epsilon = 1.5` train without the numerical instability
previously associated with `epsilon < 1`? Yes -- this is the thing the
whole design was built to isolate, and it holds.** `m_mean = -0.5` exactly
(`1.5 - 2.0`), confirming the posterior genuinely sits in the bimodal
regime by construction. `test_ll` (149.801) and `test_kl` (7.764) are both
finite and in the same range as every other cell in the table; the run
completed without a NaN or a crash. The univariate boundary-divergence
instability (`epsilon < 1`) and the bimodality condition (`m < 0`) are
successfully decoupled at `latent_dim = 5`: a genuinely bimodal TNBBeta
posterior trains fine here.

**(c) Does forcing `m < 0` via `fixed_epsilon = 1.5` actually improve
`test_ll`/`test_kl` relative to the free-epsilon `latent_dim = 5` run? Not
cleanly -- it's a wash, slightly in the wrong direction on net.**
`test_ll` goes up slightly (149.435 -> 149.801, +0.366 nats) but `test_kl`
goes up more (7.076 -> 7.764, +0.688 nats); net `ll - kl` is 142.359 (free)
vs. 142.037 (fixed) -- the forced-bimodal variant's actual ELBO is
*slightly worse*, not better. The one metric that does move in the
"bimodality helping" direction is `angle_error`: 0.6907 (free) -> 0.6323
(fixed), and `angle_error_sample` 0.6988 -> 0.6387 -- a real but modest
improvement, still far from the mod-`pi` chance ceiling's opposite end
(`0`) and still close to the `~0.785` ceiling itself. Compared to the
original `latent_dim = 2` baseline (`test_ll = 151.244`, `test_kl =
5.4028`, `angle_error = 0.6945`): both `latent_dim = 5` variants have
lower `test_ll` and higher `test_kl` than the `latent_dim = 2` run (expected
-- a higher-dimensional uniform-sphere prior costs more KL regardless of
family, visible in every family's `latent_dim = 5` row above, not just
TNBBeta's), and `fixed_epsilon = 1.5`'s `angle_error` (0.6323) is the best
of all three TNBBeta variants, but only by a modest margin.

**(d) Overall verdict: this does cleanly separate "not found" from
"doesn't help," and the numbers say it's the latter, not the former.**
The free-epsilon `latent_dim = 5` run (a) answers only "training doesn't
find it," same as before. The `fixed_epsilon = 1.5` run answers the
sharper question directly: *handed* a genuinely bimodal posterior shape,
with no gradient signal needed to discover it, training still drives `p`
to an extreme (`0.990`, not the symmetric `0.5` a mass-balanced bimodal
posterior would need) and does not produce a better net ELBO than the
unimodal-collapsed free run. That is a materially stronger result than
"optimizer never gets there": even when the shape is free and the boundary
instability is avoided, nothing in the loss rewards using it symmetrically.
The likely mechanism is structural, not a training-dynamics accident:
`axial_mixture`'s decoder is built to be *exactly* invariant under
`phi -> phi + pi` (`embed(2*phi) == embed(2*(phi+pi))`), so a posterior
that puts all its mass on one of the two axially-equivalent points
reconstructs exactly as well as one that splits it 50/50 across both --
there is no reconstruction reward for symmetric bimodality on this task,
only an extra KL cost from spreading mass across two separated modes
instead of one. Once `mean_direction` is available as a free per-example
escape valve (exactly the "direction carries the signal, p doesn't"
pattern CLAUDE.md already documents for MNIST), gradient descent has every
reason to let `p` collapse to an extreme and none to keep it near `0.5`,
*regardless* of whether `m` is positive or negative. This argues against
spending further effort chasing bimodality through epsilon initialization
or scheduling changes on this particular task family: the mechanism
suppressing it is the task's own reconstruction symmetry, not an
optimizer-reachability problem that a different schedule would fix.
