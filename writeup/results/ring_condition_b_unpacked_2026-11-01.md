# Ring task, learned-decoder condition B: why TNBBeta does not beat the caps there (2026-11-01)

Date: 2026-11-01. Base commit `ec2677e` (branch `synthetic-graph-families`) plus the UNCOMMITTED working tree. ALL of this is
EXPLORATORY: nothing here is in the pre-registered Holm family of `ring_posterior_design_2026-11-01.md`, no multiplicity correction
is applied, and every statement below is descriptive over the listed seeds (d = 3, sigma = 0.2, 20000 train x, 100 epochs unless
stated; test set seed 424242, 2000 x). Taken as given from `tnb_negative_kl_clamp_and_beta_underflow_2026-11-01.md`: the
`tnbbeta` (original init) learned seed 4 run is invalid (eps at the 1e-4 floor, estimator exploit, true gap ~1e4), so it is
EXCLUDED from every `tnbbeta` mean below ("9 valid seeds"), and any run with `epsilon` median < 0.3 would be excluded (none other).

## Code and commands

Additive edits to `apps/synthetic/ring_recovery.py` (defaults reproduce the finished runs): `stage3 --num-elbo-samples N`
(default 4), `--mc-kl` (vMF/PS trained with a Monte Carlo KL to the uniform prior, like TNBBeta; default off),
`--warmup-epochs` (default `epochs // 10`), and four extra JSON keys (`num_elbo_samples`, `mc_kl`, `epochs`, `loss_curve_full`,
`recon_model`, `kl_prior_model_mc`). The three pyright errors (`_TrueDecoder` assignment, two `.pvalue`) are fixed without changing
behaviour (a `cast` and a small `_wilcoxon_greater` helper). New: `apps/synthetic/ring_condition_b_diagnostics.py` (checkpoint
decomposition), `apps/synthetic/ring_b_perdatum.py` (per-datum posterior optimisation against a learned decoder),
`apps/synthetic/ring_condition_b_summary.py` (tables), `scripts/slurm/ring_b_controls.sbatch`, `ring_b_diagnostics.sbatch`,
`ring_b_perdatum.sbatch`, one new test in `tests/apps/test_ring_recovery.py`.

    # fresh controls (stage 3, d = 3), one cell per worker, 160 cells: job 2289038 (+ no-GPU retries 2289059_4, 2289081_8, 2289082_15)
    PROCS=8 sbatch --array=0-19 scripts/slurm/ring_b_controls.sbatch
    #   -> ring_recovery/b_controls/{n32,n128,mckl_n4,mckl_n32,a_n32,ep300}/stage3/
    # checkpoint decomposition of the finished L3 runs (all 120 learned/true checkpoints, 2000 eval draws, 4e5 prior points)
    STAGE3=ring_recovery/l23_d3/stage3 OUT=ring_recovery/b_diag/l23 sbatch --array=0-4 scripts/slurm/ring_b_diagnostics.sbatch   # job 2289146
    # per-datum optimisation against the learned decoders (300 test x, 4 restarts, 1500 steps, lr 0.05) and refinement from the encoder
    sbatch --array=0-14 scripts/slurm/ring_b_perdatum.sbatch                                                                         # job 2289152
    TAG=refine EXTRA="--refine-encoder --lr 0.01 --steps 1000" sbatch --array=0-14 scripts/slurm/ring_b_perdatum.sbatch              # job 2289188
    uv run python -m apps.synthetic.ring_condition_b_summary controls --dirs ring_recovery/l23_d3 ring_recovery/b_controls/n32 ...
    uv run python -m apps.synthetic.ring_condition_b_summary diag --json ring_recovery/b_diag/l23_all.json

Not finished: the `n128` cells for `tnbbeta` / `tnbbeta_p_init` (20 cells; the 32-sample TNBBeta cells alone took ~1h50 each, so 128
samples would exceed the 6 h task limit), `n32` seed 0 for both TNBBeta variants, and `n128` vMF s0 / PS s0, s9. Everything reported
below used only finished cells; n is given per row.

## What B measures (the first thing that differs from A)

With a learned decoder the model ELBO gap splits exactly as
`log p(x) - ELBO = [log p(x) - log p_model(x)] + [log p_model(x) - ELBO]`, a *decoder misfit* plus an *inference gap*
`= KL(q || p_model(z|x))`, where `p_model(x)` is the learned model's own marginal likelihood (uniform-prior Monte Carlo, 4e5 points,
no use of q; `ring_condition_b_diagnostics.py`). In A the decoder is the truth, so misfit = 0 and the gap is `KL(q || p_true)`.
Means over 10 seeds (TNBBeta original: 9 valid seeds), nats:

| cond | variant | recon `E_q log p(x|z)` | `KL(q||U)` | gap | decoder misfit | inference gap | learned sigma |
|---|---|---|---|---|---|---|---|
| A | vMF | -0.277 | 4.096 | 2.700 | 0 | 2.700 | 0.2 (fixed) |
| A | PS | -0.237 | 4.216 | 2.780 | 0 | 2.780 | 0.2 |
| A | TNBBeta p_init | 0.231 | 2.676 | 0.772 | 0 | 0.771 | 0.2 |
| B | vMF | 0.627 | 2.733 | 0.433 | 0.170 | 0.263 | 0.193 |
| B | PS | 0.564 | 2.665 | 0.429 | 0.165 | 0.264 | 0.197 |
| B | TNBBeta p_init | 0.558 | 2.768 | 0.537 | 0.167 | 0.370 | 0.198 |
| B | TNBBeta orig (9 valid) | 0.217 | 2.355 | 0.466 | 0.163 | 0.302 | 0.223 |

(Evaluation noise is not an issue: the recorded 200-draw ELBO gap and the 2000-draw one differ by -0.0003 on average, max 0.0074
over all 110 valid learned runs; for vMF/PS the MC and analytic KL differ by 0.0002.)

Reading: the A advantage (vMF 2.70 vs TNBBeta 0.77: 1.9 nats paired, 10/10 seeds) is entirely an inference-gap advantage. In B
the decoder reshapes itself around the posterior family and the caps' inference gap collapses from 2.70 to 0.26; the decoder
misfit (0.165-0.170) is the same for vMF, PS and TNBBeta p_init (paired p_init - vMF = -0.003 +- 0.005), so it does not explain the
difference. The whole 0.104 nats paired gap p_init - vMF (0.093-0.119 on every one of 10 seeds, 0/10 wins) sits in the inference gap
(+0.107 +- 0.009), split as reconstruction -0.070 +- 0.049 (TNBBeta lower, 9/10 seeds) and `KL(q||U)` +0.034 +- 0.047.

## (1) Estimator asymmetry: not supported

* Eval noise: negligible (above).
* Training with more ELBO samples, same seeds (paired, 100 epochs, gap mean +- sd over seeds): vMF 0.432 -> n32 0.428 (paired
  -0.004 +- 0.011, n = 10), PS 0.430 -> 0.424 (-0.005 +- 0.007, n = 10), TNBBeta p_init 0.537 -> 0.530 (-0.009 +- 0.007, n = 9; seed 0
  not finished). The p_init - vMF gap at 32 samples is 0.100 (9/9 seeds worse), i.e. unchanged from 0.104. 128 samples: vMF 0.423
  (n = 9), PS 0.422 (n = 8); TNBBeta n128 did not finish.
* Giving the caps TNBBeta's estimator (Monte Carlo KL, `--mc-kl`): vMF 0.432 (analytic, n4) -> 0.432 (MC KL, n4) -> 0.428 (MC KL,
  n32); PS 0.430 -> 0.428 -> 0.423; no effect.
* So the 4-sample MC KL does not explain the 0.1 nat deficit of `tnbbeta_p_init`. It does matter in one place: the invalid
  original-init seed 4 (already diagnosed elsewhere) is not invalid with 32 samples (the 32-sample run of seed 4 is a ring, gap 0.259).
  In A, 32 samples did NOT help `tnbbeta_p_init` (KL 1.10 vs 0.77 at 4 samples, 10 seeds; per seed 1.13 1.18 1.25 1.24 1.18 1.23 0.21 1.22
  1.14 1.27 vs 1.14 0.25 0.35 1.35 0.29 1.29 0.26 1.28 1.24 0.26: 1 of 10 seeds reaches the ring instead of 5 of 10), which I cannot
  explain and report as is (single configuration, n = 10, exploratory).

## (2) What the learned-decoder gap measures: supported, this is most of the answer

Decoder structure (`dec_*`, 40000 uniform z): the learned decoders sit about one noise sigma off the true curve
(rms distance to `g([-1,1])` 0.26 for vMF/PS/p_init, vs sigma = 0.2) and are NOT clean functions of one latitude coordinate for the cap
families: the best single linear axis explains eta^2 = 0.73 of the variance of the nearest-curve parameter for vMF, PS and p_init
(`dec_wstar_eta2_on_best_axis`), and the best axis has |cos| with the true axis a of only 0.45 / 0.42 / 0.74 (sd 0.3: the decoder
rotates freely). Learned sigma: vMF 0.193, PS 0.197, p_init 0.198 (true 0.2). So the decoder is a 2-D-dependent map whose
posterior given x is a blob that a cap represents well (inference gap 0.26), which is why caps lose ~2.4 nats of A-style penalty
and TNBBeta's ring has far less to gain.

Decomposition of the learned TNBBeta (original init, 100 epochs, d = 3) per seed; two regimes, not a continuum
(`ring` = q median > 0.2 and eps median >= 8):

| seed | gap | regime | eps | q | sigma | recon | KL(q||U) | misfit | inference gap | decoder eta^2 on best axis |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.26 | ring | 11.4 | 0.59 | 0.241 | -0.02 | 1.92 | 0.14 | 0.12 | 0.98 |
| 3 | 0.27 | ring | 12.5 | 0.48 | 0.241 | -0.05 | 1.90 | 0.13 | 0.14 | 0.99 |
| 5 | 0.29 | ring | 13.3 | 0.45 | 0.242 | -0.06 | 1.90 | 0.15 | 0.14 | 0.99 |
| 0, 2 | 0.66, 0.58 | hybrid | 4.9, 4.6 | 0.00 | 0.256, 0.235 | -0.20, -0.01 | 2.13, 2.24 | 0.21, 0.17 | 0.45, 0.40 | 0.74, 0.73 |
| 6, 7, 8, 9 | 0.52-0.55 | cap (q = 0) | 3.0-3.1 | 0.00 | 0.195-0.199 | 0.54-0.59 | 2.74-2.80 | 0.16-0.19 | 0.36-0.37 | 0.71-0.73 |

Seed 4 (invalid) omitted. With 32 samples the same picture: ring seeds 1, 3, 4, 5 (gap 0.249, 0.238, 0.259, 0.236), hybrid seed 2 (0.624),
cap seeds 6-9 (0.525-0.531), seed 0 not finished; at 300 epochs ring seeds 1, 3, 4 (0.227, 0.214, 0.232), hybrid seeds 0, 2 (0.500, 0.502).
Reading:

* TNBBeta CAN beat the caps in B: the ring regime gives gap 0.21-0.29 vs vMF/PS 0.41-0.43, by 0.15-0.2 nats, with a decoder that is
  a clean single-coordinate function of the latent (eta^2 0.98-0.99), larger sigma (0.22-0.24) and much lower KL(q||U) (1.9 vs 2.7).
  The seed determines the basin (the same seeds 1, 3, 5 are rings at 4 and 32 samples; seed 4 joins at 32 and 300 epochs).
* But the ring's advantage in B is ~0.15-0.2 nats against ~2.4 nats in A, because the cap families adapt the decoder to make caps
  nearly sufficient; and the cap/hybrid basin is reached in 5-6 of 9 original-init seeds, with an in-basin gap (0.52-0.55) 0.1
  worse than vMF/PS. Averaged over seeds this is what the table in the L3 note shows (original-init mean 0.466, sd 0.15, vs 0.43).
* `tnbbeta_p_init` never reached the ring in B (0 of 24 runs at 4, 32 samples and 300 epochs combined; all p = 0.98, q = 0, eps 3.0), unlike A
  (5/10 at 4 samples). The ln 3 offset therefore turns a ~0.4 chance of the ring (3-4/9 seeds) into 0 in B; I did not test why. This is
  the reason p_init has a uniformly worse gap than the caps, not an estimator or capacity effect: its cap is a `q = 0` TNBBeta cap, and
  at that solution the paired gap to vMF is +0.104 +- 0.007.

## (3) Optimisation / underfitting: not supported as the explanation

* Training loss in the last 20% of epochs changes by 0.001-0.009 in B for every family (table: `loss drop` column of the summary),
  i.e. B is converged at 100 epochs; in A it is still moving (0.04 vMF, 0.07 p_init, 0.16 TNBBeta original).
* 300 epochs, seeds 0-4 (paired with the 100-epoch run of the same seed): B vMF gap 0.431 -> 0.413, PS 0.429 -> 0.410, p_init
  0.535 -> 0.523: all improve by 0.012-0.019 and the p_init - vMF difference stays 0.110. A (true decoder) improves much more:
  vMF KL 2.703 -> 2.580, p_init 0.678 -> 0.473 (n = 5): A is under-trained at 100 epochs; B is not. Original-init B at 300 epochs:
  ring seeds 1, 3, 4, hybrid 0, 2.
* So B is not unfavourable to TNBBeta because of too few epochs; if anything A at 100 epochs underestimates TNBBeta's A advantage.

## (4) Can a ring posterior improve on what the decoder lets caps reach: partly supported

* Fixed-eps_ring and m = -0.5 variants in B (L3 note): gap 1.216 +- 0.230 and 1.096 +- 0.008, with learned sigma 0.316 / 0.345 and
  decoder misfit 0.399 / 0.509 (vs 0.17), i.e. forcing the posterior shape makes the decoder misfit worse, not better; their `dec_*`
  eta^2 on the best axis is 0.29 / 0.71. These shapes do not help in B.
* Per-datum optimisation against the learned decoders (ring_b_perdatum.py, 300 test x, seeds 0-4, family-by-family from random starts;
  mean over seeds of `log p_model(x) - ELBO`, TNBBeta runs with eps < 0.3 excluded, none occurred): on the vMF decoders: vMF 0.361,
  PS 0.750, TNBBeta 0.377 (amortised vMF 0.253); on the PS decoders: vMF 0.351, PS 0.707, TNBBeta 0.385 (amortised 0.258); on the p_init
  decoders: vMF 0.341, PS 0.688, TNBBeta 0.372 (amortised TNBBeta p_init 0.365). So against the decoder's own posterior, TNBBeta is no
  better than vMF per datum (0.01-0.04 worse), consistent with that posterior being cap-like, although this per-datum optimiser is not a
  bound (it is worse than the amortised encoder for the caps, 0.36 vs 0.25: Adam from random starts, 1500 steps, lr 0.05). Refining the
  amortised posterior per datum (restart 0 = encoder output, lr 0.01, 1000 steps) gave vMF 0.270 (amortised 0.256), PS 0.242 (0.258),
  p_init 0.402 (0.361): no improvement, so the p_init deficit is not closed by 1000 extra per-datum steps (inconclusive about capacity
  vs optimisation since the refinement is itself noisy at 16 draws for TNBBeta).

## Which explanations the evidence supports (all exploratory; seeds 0-9 unless stated)

1. Estimator asymmetry (4-sample MC KL): NOT supported. 8x more samples change the gaps by <= 0.01 for every family, giving the
   caps the MC KL changes nothing (0.432 vs 0.432), eval noise <= 0.007. (Only the invalid seed-4 collapse is an estimator effect.)
2. What the gap measures: SUPPORTED, the main explanation. B's gap is decoder misfit (0.165-0.170, identical across families) plus an
   inference gap, and the learned decoder collapses the caps' inference gap from 2.70 (A) to 0.26; TNBBeta's ring regime, when
   found, still wins by 0.15-0.2 nats (gap 0.21-0.29 vs 0.41-0.43; seeds 1, 3, 5 at n4, plus 4 at n32 and 300 epochs), so the true
   B-regime gain is an order of magnitude smaller than A's (~2.4 nats).
3. Optimisation: NOT supported as a cause of the B deficit (B is converged at 100 epochs; 300 epochs keep the 0.11 difference);
   but it is supported as a statement about A (A is still improving at 100 epochs).
4. Learned decoders let caps get most of the way: SUPPORTED (inference gap 0.26 for caps; per-datum TNBBeta not better than vMF
   against the learned decoders; fixed-shape TNBBeta variants make B worse). Plus a basin effect specific to B: ~0.4 of
   original-init seeds find the ring (gap 0.26), the rest sit in a q = 0 cap that is 0.1 worse than vMF/PS, and `tnbbeta_p_init`
   found the ring in 0 of 24 B runs while it found it in 5 of 10 A runs.

Caveats: single task, d = 3, sigma = 0.2; 10 seeds; seeds and sampling streams are shared across variants (paired); the `n128` TNBBeta
cells, `n32` seed 0 for TNBBeta and a few `n128` vMF/PS cells are missing; the per-datum optimiser is not a capacity bound; the ring-vs-cap
split is a descriptive classification (q median > 0.2 and eps median >= 8), not a registered test; no p-values are claimed; the
original-init TNBBeta mean (0.466) mixes two regimes and one run (seed 4) was excluded for the estimator defect.
