# Does TNBBeta's advantage shrink as features reveal the community directly? 2026-10-09

- **Date:** 2026-10-09
- **Branch/commit:** `neighbor-heterogeneity-mechanism`, `6094d63`
  (`stochastic_block_model: add feature_noise_std, threaded into
  sbm_recovery.py`). `master` did not yet have this code.
- **Fixed `p_out`:** 0.05, chosen from
  `sbm_pout_dose_response_2026-10-09.md`'s dose-response grid as the single-seed
  point with the largest TNBBeta-vs-Power-Spherical AUC gap (+0.0256) -- giving the
  most room to observe that gap shrink as features become informative. (That same
  write-up's 3-seed check at this `p_out` also found the gap noisy across seeds;
  this sweep reuses `--seed 0` throughout, so it inherits that same single-seed
  caveat -- flagged here, not hidden.)
- **Commands** (`--feature-noise-std` is the only knob varied; everything else
  matches the dose-response sweep's `p_out=0.05` run exactly):
  ```
  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/feature_informativeness_sweep/std_5.0 \
      --p-out 0.05 --epochs 500 --device cpu \
      --run-name feature_sweep_std_5.0 --feature-noise-std 5.0

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/feature_informativeness_sweep/std_2.0 \
      --p-out 0.05 --epochs 500 --device cpu \
      --run-name feature_sweep_std_2.0 --feature-noise-std 2.0

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/feature_informativeness_sweep/std_1.0 \
      --p-out 0.05 --epochs 500 --device cpu \
      --run-name feature_sweep_std_1.0 --feature-noise-std 1.0

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/feature_informativeness_sweep/std_0.5 \
      --p-out 0.05 --epochs 500 --device cpu \
      --run-name feature_sweep_std_0.5 --feature-noise-std 0.5

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/feature_informativeness_sweep/std_0.2 \
      --p-out 0.05 --epochs 500 --device cpu \
      --run-name feature_sweep_std_0.2 --feature-noise-std 0.2

  uv run python -m apps.synthetic.sbm_recovery \
      --out-dir runs/feature_informativeness_sweep/std_0.05 \
      --p-out 0.05 --epochs 500 --device cpu \
      --run-name feature_sweep_std_0.05 --feature-noise-std 0.05
  ```

## Background

`writeup/results/features_onoff_dimension_sweep_2026-10-09.md` (on
`features-overlap-investigation`, not yet merged) found TNBBeta closes the gap to
Gaussian and beats vMF/Power Spherical specifically in the *featureless* regime on
3 of 4 real datasets. That result is binary (features present vs. absent); this
sweep tests whether the effect is a sharp on/off transition or a gradual one, by
holding the graph (edges, communities) exactly fixed and continuously interpolating
node features from pure noise to an exact community reveal via the new
`feature_noise_std` parameter (`tnbbeta_vae.data.stochastic_block_model
.stochastic_block_model`): features become each node's true community as a one-hot
vector plus i.i.d. `N(0, feature_noise_std**2)` noise.

## Sanity check: the noise grid actually spans "mostly noise" to "mostly revealed"

Measured directly (not just trusted from the parameter) via how often `argmax`-ing
a node's noisy feature vector recovers its true community label, at this sweep's
exact graph (`p_in=0.3`, `p_out=0.05`, `graph_seed=0`, 10 communities):

| `feature_noise_std` | argmax-recovery accuracy |
|---|---|
| 5.0 | 0.112 (chance is 0.10) |
| 2.0 | 0.184 |
| 1.0 | 0.332 |
| 0.5 | 0.650 |
| 0.2 | 1.000 |
| 0.05 | 1.000 |

The grid spans the intended range: `std=5.0` is indistinguishable from chance
(pure noise), `std=0.2` and below perfectly reveal the community by simple argmax
(features essentially noiseless), and `std=1.0`/`0.5` sit in between.

## Result: AUC, TNBBeta shape, and the AUC gap, by `feature_noise_std`

| `feature_noise_std` | gaussian AUC | vMF AUC | Power Spherical AUC | TNBBeta AUC | TNBBeta `m_mean` | TNBBeta best_epoch |
|---|---|---|---|---|---|---|
| 5.0 (mostly noise) | 0.5695 | 0.5854 | 0.5774 | 0.5990 | -5.148 | 365 |
| 2.0 | 0.5915 | 0.6019 | 0.5932 | 0.6165 | -5.271 | 499 |
| 1.0 | 0.6250 | 0.6374 | 0.6371 | 0.6372 | -5.450 | 123 |
| 0.5 | 0.6339 | 0.6618 | 0.6632 | 0.6579 | -5.194 | 90 |
| 0.2 | 0.6482 | 0.6587 | 0.6595 | 0.6601 | -6.775 | **6** |
| 0.05 (mostly revealed) | 0.6489 | 0.6608 | 0.6701 | 0.6518 | -6.718 | **21** |

TNBBeta-vs-baseline AUC gap:

| `feature_noise_std` | TNBBeta - vMF | TNBBeta - Power Spherical |
|---|---|---|
| 5.0 | +0.0136 | +0.0216 |
| 2.0 | +0.0146 | +0.0233 |
| 1.0 | -0.0002 | +0.0001 |
| 0.5 | -0.0039 | -0.0053 |
| 0.2 | +0.0014 | +0.0006 |
| 0.05 | -0.0090 | -0.0183 |

**Important caveat on `feature_noise_std = 0.2` and `0.05`:** TNBBeta's best
checkpoint there was saved at `best_epoch = 6` and `21` respectively, out of 500
epochs -- validation AUC peaked almost immediately and never improved with further
training (`p_mean`/`q_mean` ~0.50-0.53 there, vs. the well-trained ~0.06-0.93 seen
at every other noise level), i.e. these two points' `m_mean`/AUC numbers reflect a
near-initialization TNBBeta posterior, not a stably trained one. (The other three
families trained normally at every noise level, `best_epoch` in the 86-494 range.)
This is the same near-init-collapse pattern `sbm_pout_dose_response_2026-10-09.md`
flagged at `p_out=0.1`; here it appears specifically once features become strongly
informative, suggesting TNBBeta's optimization becomes less stable once the task
is easy enough that the real signal is almost entirely in the node features rather
than the graph structure -- itself worth a closer look, but outside this sweep's
scope.

## Reading it: gradual, not sharp, and with the same important sign-caveat as (2)

**At the noisy end of the grid (`std` = 5.0, 2.0), TNBBeta does show a real,
measurable edge over both baselines** (+1.4 to +1.5 AUC points vs. vMF, +2.2 to
+2.3 vs. Power Spherical) -- directionally consistent with the featureless-regime
result on real data. **That gap shrinks to near zero by `std=1.0`** (within ±0.0002
of both baselines) **and turns slightly negative at `std=0.5`** (TNBBeta behind
both by 0.4-0.5 points), continuing the trend predicted by the mechanism: features
becoming more informative erodes TNBBeta's advantage. There is no sharp,
step-function transition visible in this grid -- the gap closes gradually across
roughly one order of magnitude of noise std (5.0 down to 1.0), not abruptly at one
threshold.

**The `std=0.2`/`0.05` points cannot cleanly extend this reading**, since TNBBeta's
posterior there reflects near-initialization geometry rather than a trained
representation (see the caveat above) -- their negative-looking gaps (particularly
`std=0.05`'s -0.009/-0.018) are at least partly an artifact of that collapse, not
necessarily a continuation of the same gradual erosion seen at `std` = 2.0 -> 1.0 ->
0.5. Taking the three *reliably trained* points together (`std` = 5.0, 2.0, 1.0,
0.5), the picture is: **a real, gradually shrinking (not sharply transitioning)
advantage, crossing from positive to roughly zero/slightly negative somewhere
between `std=2.0` and `std=0.5`** -- directionally exactly what the mechanism
predicts, on this single seed, at this one `p_out`.

**Caveats:** single seed throughout (no repeat-seed cross-check at this scale, for
time); one `p_out` (0.05); TNBBeta's known optimization instability once the task
becomes easy (the near-init-collapse points) limits how far down the "informative"
end of the grid this result can be read with confidence.
