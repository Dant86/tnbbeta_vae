# TNBBeta raw-p init offset (ln 3) applied as an initialisation convention, and the ring check (2026-11-01)

Date: 2026-11-01. Base commit `ec2677e` (branch `synthetic-graph-families`, uncommitted working tree).
Follows the recommendation of `tnb_init_symmetry_and_head_offset_2026-11-01.md`. The ring runs are
EXPLORATORY and outside the pre-registered Holm family of `ring_posterior_design_2026-11-01.md`
(Deviations item 6); only descriptive p-values are given.

## What changed in code

* `src/tnbbeta_vae/models/heads.py`: `P_RAW_INIT = math.log(3)` and
  `init_tnbbeta_p_bias(head, latent_dim, fixed_mean_direction=False, offset=P_RAW_INIT)`, which sets
  `head.bias[latent_dim]` (index 0 if `fixed_mean_direction`) to `offset`; `offset == 0.0` returns
  without touching anything. `tnbbeta_posterior` and `src/tnbbeta_vae/distributions/` are untouched.
* New config field `p_init_offset: float = P_RAW_INIT` on `MlpVAEConfig`, `PairwiseMlpVAEConfig`,
  `ConvTNBBetaSphericalVAEConfig` and `M1M2Config` (applied to every head whose family is
  `tnbbeta`, including M1M2's `decoder2`). Parameter names and shapes are unchanged; non-TNBBeta
  families are not touched. **Old init back: `p_init_offset=0.0`** (bit-identical state, tested).
* **`GraphVAE` was NOT changed.** Its head is `nn.Linear(..., bias=False)`, so there is no bias to
  initialise, and adding one (or a buffer applied in `_posterior`) would change the state_dict or
  the posterior map, both ruled out. The init study's graph result used a constant added inside
  `_posterior`, which is exactly that map change. If the graph models should get it, that needs
  an explicit decision (e.g. an opt-in non-persistent offset defaulting to 0.0).
* Old checkpoints: only the construction-time init changed; a loaded `state_dict` overwrites the bias,
  and the posterior map is the same, so loaded checkpoints behave exactly as before. (Saved configs
  lack `p_init_offset`; rebuilding one picks up the new default, which is harmless once weights load.)
  The default for NEW runs of those four models now differs from before; pass 0.0 to reproduce earlier
  fresh-init experiments.
* `apps/synthetic/ring_recovery.py`: new stage-3 variant `tnbbeta_p_init` (offset `P_RAW_INIT`); all
  other variants, including `tnbbeta`, pass `p_init_offset=0.0` explicitly, so the finished L3 results
  stay reproducible. (Only `STAGE3_VARIANTS`, the model construction and one import were edited.)
* `scripts/slurm/ring_recovery_p_init.sbatch` (sibling of `ring_recovery.sbatch`, 20 cells).
* Tests: `tests/models/test_p_init_offset.py` (bias value and p = 0.75; helper changes only the p
  entry; fixed_mean_direction / fixed_epsilon layouts; `0.0` reproduces the old init exactly under
  a fixed seed (every tensor except the one bias entry is identical at the default); state_dict
  keys and shapes unchanged; gaussian/vMF/PS untouched; pairwise, conv and M1M2 variants). No
  existing test hard-coded p = 0.5 at raw 0 on these models, so none needed changing, and the
  `GraphVAE` state_dict-key regression tests pass unmodified.

## Checks

* `uv run ruff format .` / `ruff check src tests`: clean. `uv run ruff check .` still reports
  pre-existing issues only in other untracked exploratory files (`apps/synthetic/axial_init_shift.py`,
  `apps/synthetic/init_symmetry_sweep.py`, `scratch_degree_correlation.py`); not touched here.
* `uv run pyright`: my files clean; 3 errors remain in the untracked `apps/synthetic/ring_recovery.py`
  (`_TrueDecoder` assignment at ~763, `.pvalue` at ~989 and ~1035), pre-existing and in code
  another agent is editing.
* `uv run pytest` (SLURM CPU job 2288912, 6 min): 773 passed, 1 skipped.

## Ring run (d = 3, 2 conditions x 10 seeds)

```
DIM=3 PROCS=1 sbatch --array=0-19 scripts/slurm/ring_recovery_p_init.sbatch   # job 2288911, all 20 ran concurrently
```
Outputs `ring_recovery/l23_d3/stage3/d3_{true,learned}_tnbbeta_p_init_s{0..9}.json`.
Condition A (true decoder): exact test KL, lower is better. Condition B (learned decoder): ELBO gap
(`elbo_gap_model`), lower is better. mean +- sd (ddof = 1) over 10 seeds:

| variant | A: KL | B: ELBO gap |
|---|---|---|
| vMF | 2.7015 +- 0.0265 | 0.4316 +- 0.0085 |
| Power Spherical | 2.7797 +- 0.0495 | 0.4297 +- 0.0089 |
| TNBBeta, original init (`tnbbeta`) | 1.0779 +- 0.8840 | -0.2509 +- 2.2684 |
| TNBBeta, `tnbbeta_p_init` | 0.7715 +- 0.5173 | 0.5373 +- 0.0091 |

Paired one-sided Wilcoxon, descriptive (H1: `tnbbeta_p_init` has the lower value), exploratory, not
Holm-adjusted, not a significance claim:

| comparator | A: p (wins/10) | B: p (wins/10) |
|---|---|---|
| vMF | 0.00098 (10/10) | 1.0 (0/10) |
| Power Spherical | 0.00098 (10/10) | 1.0 (0/10) |
| original TNBBeta | 0.28 (5/10) | 0.93 (3/10) |

Reading: in A, `tnbbeta_p_init` has a lower KL than vMF/PS on every seed and a somewhat lower mean and
sd than the original TNBBeta, with no seed-level evidence of a difference from it (5/10). In B it is
worse than vMF/PS on every seed (0.537 vs 0.43 gap) and, unlike the original TNBBeta, shows no
negative-gap outlier seeds (sd 0.009 against 2.27); the original's mean gap is dominated by a few
seeds, so its mean is not comparable to a typical seed. Mean over seeds of `frac_band_ge_0.8`: A 0.969 vs
0.937 (original); B 0.029 vs 0.012. Median m is positive for both variants in both conditions, i.e. not the
bimodal regime.

## Caveats

* The earlier init study was 3 graph draws, assortative-type graphs and d = 16 only; it ran the
  offset as a map constant, not as a trainable bias. Here the offset is a trainable bias init in
  MLP models, a different mechanism from that study.
* Its eps-only and q-only controls also escaped the plateau, so axis symmetry at p = 0.5 is not the only
  cause; the offset is a convention that helps, not a mechanism claim.
* Cost to bimodality (SBM, axial tasks) is untested; p = 0.75 starts away from the p = 0.5 axial
  symmetry.
* The ring variant was added after seeing the L3 default-init results, so the A improvement is a
  post hoc exploration. 10 seeds, one task, d = 3.
* `GraphVAE` (the model the init study actually targeted) does not get the init; see above.
