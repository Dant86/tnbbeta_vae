# Exact TNBBeta log q in every Monte Carlo ELBO / KL / importance weight, and an optional epsilon floor

Date: 2026-11-01. Code: base commit `ec2677e` (branch `synthetic-graph-families`) plus the UNCOMMITTED working-tree state of
the main tree at the time (the L2/L3 ring code, `p_init_offset` / `head_bias` changes) plus the distributions fix of
`tnb_small_epsilon_fix_2026-11-01.md` (`rsample_with_log_prob`), plus this change (UNCOMMITTED; patch
`tnb_exact_log_q.patch`, written to apply on top of that state). The follow-up was approved by the user.

## Commands

    uv run ruff format . && uv run ruff check . && uv run pyright        # clean
    # full `uv run pytest` as a SLURM CPU job (4 CPUs, 16 GB): 935 passed, 1 skipped, 99% coverage (job 2289343; see Tests)
    sbatch --array=0-2 scripts/slurm/tnb_exact_log_q.sbatch                # job 2289286, the Task C numbers below
      # task 0: python -m apps.synthetic.tnb_exact_log_q_check stage2 --out ring_recovery/exact_log_q/stage2_d3_steps2000.json
      # task 1: python -m apps.synthetic.ring_recovery stage3 --dim 3 --condition learned --family tnbbeta --seed 4 \
      #             --epsilon-floor 1e-4 --out-dir ring_recovery/exact_log_q/stage3_floor1e-4   (then ...tnb_exact_log_q_check stage3 --ckpt ...)
      # task 2: the same with --epsilon-floor 0.3 (--out-dir ring_recovery/exact_log_q/stage3_floor0.3)

## What changed

Task A: exact `log q`.

* New `models/losses/sampling.py`: `sample_with_log_prob(dist, sample_shape=None, *, reparameterize=True, need_log_prob=True)`
  returns `dist.rsample_with_log_prob(shape)` when the distribution has that method (TNBBetaSpherical), otherwise exactly
  `rsample()`/`sample()` followed by `log_prob(z)` (same call, same RNG consumption). `sample_and_kl(posterior, prior)` is
  `kl_divergence` when PyTorch has the pair, else the one-draw `log q - log p` with the exact `log q`. With
  `reparameterize=False` (evaluation only) a TNBBeta draw runs under `no_grad` and its `log q` is detached.
* Switched to the helpers: `models/losses/elbo.py::monte_carlo_elbo` (with `analytic_kl=True` the log-density is not even
  evaluated: that path is unchanged), `models/losses/importance_weighted.py` (the Table 1 weights, `kl` fallback and `elbo`),
  `models/graph_vae.py` and `models/pairwise_mlp_vae.py` (`sample_and_kl`; the previous try/except `kl_divergence` is inside it),
  `models/semi_supervised.py` (`z1` and `z2` terms), `data/ring_posterior.py::exact_kl`, `apps/synthetic/ring_recovery.py`
  (`_evaluate_q`, `_fit_tnbbeta_free`, `_elbo_estimate` -- training ELBO AND the 64-sample restart-selection ELBO --,
  `_evaluate_posteriors`' ELBO, `_model_elbo`), `apps/synthetic/ring_b_perdatum.py`, `apps/synthetic/ring_condition_b_diagnostics.py`,
  `apps/eval/export_latents.py`. `mlp_vae.py` and `conv_vae.py` go through `monte_carlo_elbo` and need no edit.
* Checked and not needing a change (no `log q` at samples): `models/diagnostics.py`, `models/predictive.py`,
  `models/posterior_stats.py`, `apps/eval/param_probe.py`, `apps/synthetic/graph_families.py`, `graph_search.py`. The
  `ring_recovery._evaluate_q_single`, `ring_sphere_plot.py` and `tnb_low_epsilon_check.py` density evaluations are at fixed
  points or use their own exact sampler, so they were left alone.

Task B: epsilon floor.

* `heads.tnbbeta_posterior(..., epsilon_floor=_PARAM_EPS)` and `heads.posterior_from_raw(..., epsilon_floor=_PARAM_EPS)`;
  `epsilon = softplus(raw) + epsilon_floor`. `epsilon_floor: float = 1e-4` added as the last field of `MlpVAEConfig`,
  `PairwiseMlpVAEConfig`, `ConvTNBBetaSphericalVAEConfig`, `GraphVAEConfig`, `M1M2Config`. The default is the old constant, so
  behaviour is bit-identical (test below), and checkpoints load unchanged (no parameter or shape change). **Do not change the
  default**: it silently shifts the epsilon of every existing checkpoint. **0.3 is the suggested value for NEW runs** on tasks
  that can reach small epsilon (it is irrelevant where epsilon stays large). `ring_recovery stage3 --epsilon-floor` exposes it.
* The edits to `heads.py`, `graph_vae.py` and the config classes are small additive hunks (new kwarg, new last field); they
  were made against a copy of the main tree's current (uncommitted) versions of those files.

New verification driver: `apps/synthetic/tnb_exact_log_q_check.py` (stage 2 re-run and stage 3 checkpoint scoring with the
honest evaluator `tnb_low_epsilon_check.honest_kl`, copied in from the main tree) and `scripts/slurm/tnb_exact_log_q.sbatch`.

## Tests

`tests/models/losses/test_exact_log_q.py`, `tests/models/test_epsilon_floor.py`.

* (i) At p = 0.015, q = 3e-3, eps = 1e-4, d = 3 the MC KL-to-uniform from `monte_carlo_elbo` equals the quadrature value
  (`tnb_low_epsilon_check.exact_elq + log 4 pi` = 9989) within 6 SE + 1%; the pre-change estimator (`rsample` then `log_prob`,
  with the already-patched density) gives 675 here, under 20% of it. Also across eps in {1e-4 ... 10} and three (p, q) pairs:
  quadrature agreement and KL >= 0 within MC error; importance-weighted-metrics KL >= 0 for d in {3, 8} and matching
  quadrature at d = 3; at eps = 3 the new and old estimators agree to MC error.
* (ii) vMF, PowerSpherical and Gaussian: `monte_carlo_elbo` (with and without `analytic_kl`), `sample_with_log_prob` (both
  `reparameterize`) and `sample_and_kl` are `torch.equal` to the pre-change code given a seed. `analytic_kl=True` returns exactly
  `kl_divergence` and still raises `NotImplementedError` for TNBBeta.
* (iii) Gradients through the exact ELBO are finite and non-zero (including wrt the epsilon column) for raw epsilon in
  {-12 ... 2}; an end-to-end `MlpVAE.training_step` with epsilon at the floor reports KL > 100 with finite head gradients.
* Floor: default config value is 1e-4 for all five configs; `tnbbeta_posterior` / `posterior_from_raw` at the default are
  `torch.equal` to `softplus(raw) + 1e-4`; floor 0.3 is exactly additive and leaves p, q, direction alone; ignored with
  `fixed_epsilon`; a model with `epsilon_floor=0.3` shifts epsilon by 0.3 - 1e-4 on the same weights.
* Full suite: see Commands (the first run had one failing assertion in my new test -- the old-path value was 7% of exact rather
  than the guessed <5% -- loosened to 20%, nothing else failed).

## Task C results (d = 3, ring task, sigma 0.2, float32; honest = exact E_q[log q] by quadrature minus exact-sampler cross term)

Stage 2, TNBBeta only, 2000 steps, 300 x (data seed 777, `torch.manual_seed(0)`), the training and restart-selection ELBO both
use the exact estimator. Old numbers are from `tnb_negative_kl_clamp_and_beta_underflow_2026-11-01.md` (same seeds, old code).

| init | eps min / median | frac eps at floor (< 2e-4) | frac eps < 0.3 | honest KL mean / median / frac < 1 | library `exact_kl` (exact log q) mean / median |
|---|---|---|---|---|---|
| amortised-like, new | 0.074 / 3.39 | 0 | 0.007 | 1.98 / 0.213 / 0.693 | 1.99 / 0.207 |
| amortised-like, old | -- | 30/300 below 0.3 | 0.10 | 957 / 0.201 / 0.700 | 3.16 / 0.187 (library, wrong) |
| random restarts, new | 1.07 / 3.44 | 0 | 0 | 0.81 / 0.152 / 0.797 | 0.81 / 0.150 |
| random restarts, old | -- | 102/300 below 0.3 | 0.34 | 3263 / 0.356 / 0.550 | -0.74 / 0.090 (library, wrong) |

**Epsilon no longer collapses to the floor in stage 2.** The 10% (amortised-like) and 34% (restarts) of x that sat at the floor
with honest KL about 9.5e3 are gone, and the honest KL now equals the library KL (the exact estimator agrees with the independent
evaluator to about 1%). One row in each init has an honest KL slightly below 0 (count 1/300 each); I did not inspect them (MC
noise of the 2000-draw cross term on rows with KL near 0 is the expected cause, not verified).

Stage 3, condition B (`learned` decoder), TNBBeta, seed 4 (`d3_learned_tnbbeta_s4`; same seeds, RNG consumption differs so not
bit-for-bit), 100 epochs, test set seed 424242, 2000 x:

| run | eps min / median | frac at floor | model ELBO gap to log p(x) | honest KL mean / median / frac < 1 | final train loss |
|---|---|---|---|---|---|
| original (old code) | at 1e-4 on all x | 1.0 | -6.69 (impossible: a negative gap) | 31.25 / 29.64 (library, invalid) | -7.20 (negative loss) |
| exact log q, floor 1e-4 | 3.23 / 12.65 | 0 | +0.268 | 29.35 / 29.04 / 0 | 1.934 |
| exact log q, floor 0.3 | 2.91 / 11.94 | 0 | +0.270 | 18.26 / 14.15 / 0 | 1.930 |

**Without the floor, epsilon no longer collapses either**: the run that had epsilon at the floor on every test x now trains to
epsilon of about 12 (minimum 3), and the model's ELBO gap is a valid positive 0.27 nats. The 0.3 floor is therefore not needed
here (it is never active: minimum epsilon is 2.9); the two runs differ only by training noise/RNG, which also moves the honest
posterior KL between 29 and 18 (single seed each, so I do not claim the floor helps or hurts). Both posteriors are still far
from the true ring posterior (honest KL 18-29 nats, never below 1): the exact log q removes the exploit, it does not teach the
amortised TNBBeta the ring shape (a separate issue, same as before).

## What is NOT fixed / caveats

* Single seed and a single d = 3 configuration for Task C; the other 99 stage-3 cells, the stage-2 10000-step run and the
  graph / MNIST / semi-supervised experiments were not re-run. Any earlier result whose logged epsilon was below about 0.3
  (graph_search, graph_families, `d3_learned_tnbbeta_s*` other than s4) was trained with the biased estimator and still needs a
  re-run. For epsilon >= 1 the new and old estimators agree to MC error, so results at larger epsilon stand.
* The floor is available but not enabled anywhere by default, and no existing script sets it.
* `sample_with_log_prob(reparameterize=False)` for TNBBeta returns a detached `log q` (evaluation use: importance-weighted
  metrics and `export_latents`); do not use it in a training loss.
* The public `TNBBetaSpherical.log_prob(rsample())` is unchanged and still under-states `E_q[log q]` below epsilon about 0.03;
  external code that calls it directly (not through these helpers) keeps that bias.
* RNG consumption changed for TNBBeta (new sampler), so old seeds do not reproduce bit-for-bit; other families are unchanged.
