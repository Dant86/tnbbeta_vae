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
