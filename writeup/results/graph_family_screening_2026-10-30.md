# Which graph structures make latent-family expressivity matter? A screening on synthetic block graphs, 2026-10-30

- **Date:** 2026-10-30
- **Code:** branch `synthetic-graph-families` (stacked on `fixed-m-sweep`, PR #68) at the commit that
  adds this note; the run commands used the working tree at `4c7f334` plus the
  core-periphery replacement of the hub-and-spoke preset (see "Incident"). Follow-up to
  `vmf_ps_fixed_kappa_and_training_dynamics_2026-10-30.md`.
- **Commands** (5 seeds per cell; each seed re-draws the graph, the edge split and the model):

  ```bash
  sbatch --array=0-6 scripts/slurm/graph_families_screen.sbatch   # canary: task 0 (job 2279891), then 2279910 (tasks 1-6)
  sbatch --array=5-6 scripts/slurm/graph_families_screen.sbatch   # job 2279922: core_periphery cells, after the fix
  uv run python -m apps.eval.graph_families_table                 # the tables below
  ```

## Question

Is there a graph structure on which TNBBeta's extra expressivity (the `p`, `q`, `epsilon`
shape family, including antipodal bimodality) is actually needed, i.e. where vMF,
power-spherical and Gaussian posteriors notably fail? The two earlier notes found that free
kappa is the main reason vMF/PS lag on real graphs, and that centre AUC cannot see posterior
shape at all, so this screening adds posterior-predictive metrics and ground truth.

## Design

- **Graphs** (`tnbbeta_vae.data.block_graph`, 500 nodes, featureless identity features):
  `assortative` (5 blocks of 100, `p_in=0.15`, `p_out=0.01`); `assortative_dc` (the same
  with Pareto degree weights, tail index 2.5, `p_in=0.1`); `bipartite` (two sides of 250,
  `p_cross=0.08`, `p_within=0.005`; the extreme disassortative case); `core_periphery` (5
  groups, each a core of 20 and a periphery of 80; cores link densely within a group
  (0.5) and to their own periphery (0.2); everything else 0.004).
- **Conditions** (all ambient dimension 16, Gaussian 15, `lr=0.01`, no dropout, 500 epochs,
  best-validation-AUC epoch): TNBBeta free; TNBBeta with `epsilon` fixed at `m = 0.5 (d-1)/2`
  (`f0.5`, unimodal); vMF and power-spherical free and with kappa pinned at `d`; Gaussian.
  The kappa = `d` and 500-epoch controls come from the previous note.
- **Metrics** (`tnbbeta_vae.models.predictive`, `apps/synthetic/graph_families.py`): centre
  AUC (the standard metric); held-out log loss and calibration (ECE) of the
  *posterior-predictive* edge probability (mean of `sigmoid(logit)` over 32 posterior
  samples), also for the plug-in centre probability; and KL from the model's pair
  probabilities to the *true* ones over every non-training pair, overall and per pair kind.
  The truth is mapped through `P / (P + rho)` (`rho` the mean pair probability), because
  1:1 negative sampling trains a model to output that, not `P`. "Oracle" AUC ranks the test
  pairs by the true probability.

## Summary

1. **No graph here shows a structure where TNBBeta is needed.** Centre AUC is within noise
   across families on both assortative graphs (0.80-0.81 on the plain one, at its oracle
   ceiling of 0.802; 0.76-0.77 on the degree-corrected one, oracle 0.833),
   and the Gaussian is clearly best on core-periphery (0.872 vs. 0.836-0.847 for every sphere
   family, 5/5 seeds, oracle 0.884).
2. **Predictive metrics separate the families modestly on assortative graphs.** On the plain
   assortative graph, free TNBBeta and the Gaussian have the lowest sampled KL to the truth
   (0.099, 0.098) and the best sampled log loss (0.561, 0.560); vMF and power-spherical are
   worse in 4-5 of 5 paired seeds by 0.014-0.066 KL, with or without pinned kappa. On the
   degree-corrected graph the gaps shrink to 0.003-0.012 KL.
3. **Where TNBBeta ties the Gaussian it does so without any tuning,** and the sphere
   families with a fixed kappa are *over-confident at the centre* (centre log loss 0.69 for
   vMF `k1` vs. 0.568 when sampled: the posterior noise is doing the calibrating), while
   TNBBeta's centre and sampled numbers agree (0.557 / 0.561).
4. **Free kappa collapse reappears** on these graphs: free vMF / power-spherical have a
   sampled AUC far below the centre AUC in some seeds (assortative: vMF 0.698 +- 0.134,
   power-spherical 0.614 +- 0.140 vs. centre 0.80), as in the real-graph probes.
5. **Bipartite is a decoder-limit control, and every family fails** (centre AUC 0.38-0.50,
   oracle 0.72). The dot-product decoder cannot give many same-side nodes a negative inner
   product, so no posterior shape can fix it. This was predicted before running.
6. **On core-periphery, TNBBeta `f0.5` is the best sphere family** (sampled AUC 0.866, log
   loss 0.521, best in 5/5 paired seeds vs. free TNBBeta's 0.848 / 0.541), but still below
   the Gaussian (0.871 / 0.514). Free TNBBeta is *not* better than `f0.5` here, so the
   bimodal-capable regime is not the source of any advantage.


## Incident: the first hub-and-spoke preset was ill-posed

The first screening included a `hub_spoke` graph (leaves attached to one hub each, leaves
barely linking to each other). Every model scored about 0.5 and even the oracle only 0.61,
because each leaf's single hub edge is the held-out target: hub membership is not
recoverable from the remaining structure. It was replaced by `core_periphery` (several core
neighbours per periphery node) and its results discarded; the old `gf_hub_spoke_*.json`
files are still in `$TNBBETA_CHECKPOINT_DIR/synthetic_graphs/` and are not used. The table
code also initially matched `gf_assortative_*` against `gf_assortative_dc_*`; it now matches
exact condition names.

## What this supports and what it does not

- Supported: centre AUC alone cannot rank the families on these graphs; predictive
  calibration against ground truth does, and there TNBBeta (free, and `f0.5` on
  core-periphery) is at or near the best sphere family while vMF/PS are worse, partly
  through free-kappa collapse and partly (assortative, kappa pinned) through
  centre over-confidence.
- Not supported: that bimodality or any unique TNBBeta capability is used. `tnbbeta_f0.5`
  matches or beats free TNBBeta on core-periphery, and I did not log `m` / `frac_bimodal`
  in these runs, so I cannot say whether free TNBBeta even entered `m < 0` here. That
  should be logged before any claim about the regime.
- The Gaussian wins where degree structure matters (core-periphery); its unbounded norm is
  plausibly the lever, but I did not test that (e.g. by comparing with `fixed_temperature`
  or a norm-normalized Gaussian).

## Caveats

- 5 seeds, one graph size (500 nodes), one dimension (16), one learning rate and no
  per-condition tuning; the vMF/PS `k1` kappa is `d`, not optimized (the previous note
  shows `0.5 d`-`2 d` all work on real graphs).
- Model selection used validation AUC from centres, not a predictive metric; a family whose
  best-AUC epoch is poorly calibrated is penalized in log loss / KL.
- Oracle AUC is low on `bipartite` and `assortative` because the generating probabilities
  are modest; the test negatives are sampled uniformly from non-edges as in the standard
  split.
- The KL to the balanced truth assumes the 1:1 negative sampling target `P / (P + rho)`
  with `rho` the full-graph mean; the training graph has 85% of the edges, so the target
  is only approximately that.
- `periphery-periphery-same` and `within` (bipartite) KL stay large for every family: that
  is the decoder limit again, not a family difference.

## Next

1. Log `p`, `q`, `epsilon`, `m`, `frac_bimodal` and `r_bar` in these runs to see whether free
   TNBBeta uses `m < 0` and what the trajectory looks like.
2. The sign-invariant (squared inner product) decoder on an axial graph: the one setting where
   bimodality has a clean, falsifiable payoff (at most `ln 2` nats of KL per node). Needs a
   decoder option in `GraphVAE`.
3. Tune per condition (lr, kappa) before concluding anything about the 0.01-0.02 gaps.

## Results (seed means; AUC also shows the std over seeds)

`centre` scores posterior centres, `sampled` averages the edge probability over 32 posterior
samples. Oracle AUC is the AUC of ranking the test pairs by the true probability.


#### assortative

Oracle (true-probability) test AUC: 0.802

| condition | seeds | centre AUC | sampled AUC | centre logloss | sampled logloss | sampled ECE | centre KL | sampled KL |
|---|---|---|---|---|---|---|---|---|
| tnbbeta_free | 5 | 0.804 ± 0.005 | 0.805 ± 0.004 | 0.557 | 0.561 | 0.125 | 0.094 | 0.099 |
| tnbbeta_f0.5 | 5 | 0.803 ± 0.004 | 0.753 ± 0.100 | 0.579 | 0.581 | 0.094 | 0.122 | 0.113 |
| vmf_free | 5 | 0.807 ± 0.009 | 0.698 ± 0.134 | 0.598 | 0.617 | 0.096 | 0.153 | 0.133 |
| vmf_k1 | 5 | 0.806 ± 0.007 | 0.805 ± 0.003 | 0.690 | 0.568 | 0.143 | 0.176 | 0.112 |
| power_spherical_free | 5 | 0.800 ± 0.007 | 0.614 ± 0.140 | 0.649 | 0.665 | 0.077 | 0.235 | 0.165 |
| power_spherical_k1 | 5 | 0.802 ± 0.013 | 0.795 ± 0.013 | 0.678 | 0.602 | 0.178 | 0.219 | 0.140 |
| gaussian | 5 | 0.805 ± 0.005 | 0.806 ± 0.005 | 0.560 | 0.560 | 0.123 | 0.097 | 0.098 |

Sampled KL to the true pair probabilities, by pair kind

| condition | between | within |
|---|---|---|
| tnbbeta_free | 0.117 | 0.013 |
| tnbbeta_f0.5 | 0.128 | 0.041 |
| vmf_free | 0.145 | 0.082 |
| vmf_k1 | 0.135 | 0.007 |
| power_spherical_free | 0.169 | 0.146 |
| power_spherical_k1 | 0.161 | 0.042 |
| gaussian | 0.116 | 0.016 |

#### assortative_dc

Oracle (true-probability) test AUC: 0.833

| condition | seeds | centre AUC | sampled AUC | centre logloss | sampled logloss | sampled ECE | centre KL | sampled KL |
|---|---|---|---|---|---|---|---|---|
| tnbbeta_free | 5 | 0.769 ± 0.013 | 0.771 ± 0.013 | 0.593 | 0.595 | 0.120 | 0.120 | 0.120 |
| tnbbeta_f0.5 | 5 | 0.759 ± 0.028 | 0.761 ± 0.028 | 0.609 | 0.597 | 0.101 | 0.144 | 0.130 |
| vmf_free | 5 | 0.772 ± 0.013 | 0.773 ± 0.015 | 0.600 | 0.607 | 0.139 | 0.138 | 0.129 |
| vmf_k1 | 5 | 0.767 ± 0.014 | 0.766 ± 0.015 | 0.628 | 0.607 | 0.135 | 0.158 | 0.123 |
| power_spherical_free | 5 | 0.771 ± 0.012 | 0.774 ± 0.012 | 0.599 | 0.606 | 0.135 | 0.138 | 0.128 |
| power_spherical_k1 | 5 | 0.774 ± 0.011 | 0.769 ± 0.010 | 0.597 | 0.626 | 0.163 | 0.129 | 0.131 |
| gaussian | 5 | 0.764 ± 0.011 | 0.764 ± 0.011 | 0.600 | 0.600 | 0.112 | 0.132 | 0.132 |

Sampled KL to the true pair probabilities, by pair kind

| condition | between | within |
|---|---|---|
| tnbbeta_free | 0.143 | 0.022 |
| tnbbeta_f0.5 | 0.150 | 0.040 |
| vmf_free | 0.152 | 0.031 |
| vmf_k1 | 0.144 | 0.031 |
| power_spherical_free | 0.150 | 0.033 |
| power_spherical_k1 | 0.150 | 0.047 |
| gaussian | 0.155 | 0.031 |

#### bipartite

Oracle (true-probability) test AUC: 0.723

| condition | seeds | centre AUC | sampled AUC | centre logloss | sampled logloss | sampled ECE | centre KL | sampled KL |
|---|---|---|---|---|---|---|---|---|
| tnbbeta_free | 5 | 0.498 ± 0.012 | 0.485 ± 0.016 | 0.774 | 0.694 | 0.018 | 0.339 | 0.207 |
| tnbbeta_f0.5 | 5 | 0.484 ± 0.018 | 0.488 ± 0.010 | 0.861 | 0.695 | 0.019 | 0.477 | 0.208 |
| vmf_free | 5 | 0.502 ± 0.026 | 0.514 ± 0.011 | 0.730 | 0.695 | 0.014 | 0.281 | 0.212 |
| vmf_k1 | 5 | 0.489 ± 0.021 | 0.488 ± 0.020 | 0.932 | 0.728 | 0.106 | 0.466 | 0.247 |
| power_spherical_free | 5 | 0.379 ± 0.079 | 0.491 ± 0.011 | 0.982 | 0.695 | 0.017 | 0.715 | 0.208 |
| power_spherical_k1 | 5 | 0.478 ± 0.014 | 0.481 ± 0.009 | 1.038 | 0.720 | 0.089 | 0.682 | 0.260 |
| gaussian | 5 | 0.500 ± 0.014 | 0.495 ± 0.021 | 0.693 | 0.697 | 0.027 | 0.210 | 0.212 |

Sampled KL to the true pair probabilities, by pair kind

| condition | cross | within |
|---|---|---|
| tnbbeta_free | 0.048 | 0.357 |
| tnbbeta_f0.5 | 0.048 | 0.358 |
| vmf_free | 0.049 | 0.365 |
| vmf_k1 | 0.071 | 0.412 |
| power_spherical_free | 0.048 | 0.359 |
| power_spherical_k1 | 0.043 | 0.464 |
| gaussian | 0.046 | 0.369 |

#### core_periphery

Oracle (true-probability) test AUC: 0.884

| condition | seeds | centre AUC | sampled AUC | centre logloss | sampled logloss | sampled ECE | centre KL | sampled KL |
|---|---|---|---|---|---|---|---|---|
| tnbbeta_free | 5 | 0.836 ± 0.015 | 0.848 ± 0.011 | 0.549 | 0.541 | 0.156 | 0.274 | 0.245 |
| tnbbeta_f0.5 | 5 | 0.847 ± 0.017 | 0.866 ± 0.016 | 0.553 | 0.521 | 0.152 | 0.328 | 0.256 |
| vmf_free | 5 | 0.839 ± 0.022 | 0.790 ± 0.129 | 0.573 | 0.583 | 0.129 | 0.336 | 0.255 |
| vmf_k1 | 5 | 0.845 ± 0.025 | 0.839 ± 0.027 | 0.564 | 0.566 | 0.169 | 0.334 | 0.241 |
| power_spherical_free | 5 | 0.838 ± 0.021 | 0.799 ± 0.126 | 0.561 | 0.603 | 0.176 | 0.312 | 0.246 |
| power_spherical_k1 | 5 | 0.841 ± 0.021 | 0.828 ± 0.022 | 0.620 | 0.592 | 0.195 | 0.408 | 0.247 |
| gaussian | 5 | 0.872 ± 0.022 | 0.871 ± 0.022 | 0.512 | 0.514 | 0.160 | 0.239 | 0.239 |

Sampled KL to the true pair probabilities, by pair kind

| condition | core-core-diff | core-core-same | core-other-periphery | core-own-periphery | periphery-periphery-diff | periphery-periphery-same |
|---|---|---|---|---|---|---|
| tnbbeta_free | 0.129 | 0.052 | 0.174 | 0.081 | 0.203 | 0.667 |
| tnbbeta_f0.5 | 0.110 | 0.014 | 0.161 | 0.038 | 0.190 | 0.852 |
| vmf_free | 0.165 | 0.180 | 0.196 | 0.151 | 0.215 | 0.608 |
| vmf_k1 | 0.172 | 0.178 | 0.188 | 0.129 | 0.200 | 0.581 |
| power_spherical_free | 0.204 | 0.259 | 0.217 | 0.210 | 0.226 | 0.414 |
| power_spherical_k1 | 0.197 | 0.244 | 0.207 | 0.175 | 0.216 | 0.494 |
| gaussian | 0.157 | 0.014 | 0.183 | 0.051 | 0.208 | 0.581 |


## Update (2026-10-31): posterior shape logged, and four graphs that are not block models

- **Code:** `a66f095` on `synthetic-graph-families` (PR #69). Logging adds `posterior` and
  `posterior_by_block` (`posterior_stats`: `r_bar`, and for TNBBeta `p`/`q`/`epsilon`/`m`/
  `frac_bimodal`) to every cell. New generators are in `tnbbeta_vae/data/structured_graph.py`.
- **Commands** (`GF_PREFIX` keeps these separate from the `gf_*` files above):

  ```bash
  GF_PREFIX=gfp sbatch scripts/slurm/graph_families_screen.sbatch                # job 2282624: the original 4 presets, rerun with logging
  GF_PREFIX=gfp sbatch --array=7-13 scripts/slurm/graph_families_screen.sbatch   # job 2282637: latent_sphere, mixed_membership, scale_free, ring
  uv run python -m apps.eval.graph_families_table --prefix gfp --graph latent_sphere mixed_membership scale_free ring
  ```

### Free TNBBeta is bimodal everywhere (`m < 0`, `frac_bimodal` = 1.0)

| graph | `m` | `r_bar` |
|---|---|---|
| assortative | -4.41 | 0.90 |
| assortative_dc | -4.36 | 0.89 |
| bipartite | -5.74 | 0.10 |
| core_periphery | -4.45 | 0.86 |
| latent_sphere | -4.32 | 0.93 |
| mixed_membership | -5.90 | 0.43 |
| scale_free | -5.88 | 0.33 |
| ring | -4.45 | 0.94 |

- `m < 0` holds on every graph, including ones the model cannot learn (bipartite, and
  mixed_membership at near-chance AUC), so it is the default landing point of free training
  here, not a response to structure. `m < 0` says the *density* has an antipodal mode; it does
  not say the samples split across the sphere (`r_bar` is about 0.9 on four graphs). The mass in
  the second mode was not measured.
- Within a graph `m` tracks degree: core_periphery cores have `m` about -2.6, peripheries about
  -4.9; scale_free `m` is -6.1 for the lowest weight quartile and -5.5 for the highest.
- Reproduction: the `gfp` rerun of the original presets differs from `gf` by at most 0.0083 in
  centre AUC or sampled KL (GPU nondeterminism).

#### latent_sphere

Oracle (true-probability) test AUC: 0.962

| condition | seeds | centre AUC | sampled AUC | centre logloss | sampled logloss | sampled ECE | centre KL | sampled KL |
|---|---|---|---|---|---|---|---|---|
| tnbbeta_free | 5 | 0.947 ± 0.010 | 0.946 ± 0.010 | 0.448 | 0.457 | 0.185 | 0.400 | 0.409 |
| tnbbeta_f0.5 | 5 | 0.945 ± 0.011 | 0.943 ± 0.010 | 0.472 | 0.481 | 0.201 | 0.436 | 0.434 |
| vmf_free | 5 | 0.948 ± 0.009 | 0.948 ± 0.009 | 0.480 | 0.483 | 0.213 | 0.449 | 0.439 |
| vmf_k1 | 5 | 0.946 ± 0.011 | 0.935 ± 0.012 | 0.486 | 0.534 | 0.233 | 0.458 | 0.457 |
| power_spherical_free | 5 | 0.949 ± 0.010 | 0.946 ± 0.009 | 0.482 | 0.487 | 0.210 | 0.453 | 0.440 |
| power_spherical_k1 | 5 | 0.945 ± 0.012 | 0.924 ± 0.015 | 0.484 | 0.579 | 0.272 | 0.456 | 0.476 |
| gaussian | 5 | 0.940 ± 0.010 | 0.939 ± 0.010 | 0.481 | 0.482 | 0.193 | 0.437 | 0.438 |

Sampled KL to the true pair probabilities, by pair kind

| condition | far | mid | near |
|---|---|---|---|
| tnbbeta_free | 0.460 | 0.510 | 0.070 |
| tnbbeta_f0.5 | 0.420 | 0.651 | 0.080 |
| vmf_free | 0.411 | 0.677 | 0.082 |
| vmf_k1 | 0.511 | 0.577 | 0.069 |
| power_spherical_free | 0.418 | 0.672 | 0.080 |
| power_spherical_k1 | 0.565 | 0.542 | 0.084 |
| gaussian | 0.398 | 0.691 | 0.092 |

#### mixed_membership

Oracle (true-probability) test AUC: 0.655

| condition | seeds | centre AUC | sampled AUC | centre logloss | sampled logloss | sampled ECE | centre KL | sampled KL |
|---|---|---|---|---|---|---|---|---|
| tnbbeta_free | 5 | 0.514 ± 0.015 | 0.512 ± 0.014 | 0.812 | 0.700 | 0.052 | 0.162 | 0.048 |
| tnbbeta_f0.5 | 5 | 0.512 ± 0.014 | 0.517 ± 0.013 | 0.937 | 0.732 | 0.092 | 0.294 | 0.077 |
| vmf_free | 5 | 0.518 ± 0.009 | 0.515 ± 0.012 | 0.986 | 0.701 | 0.046 | 0.367 | 0.049 |
| vmf_k1 | 5 | 0.533 ± 0.017 | 0.529 ± 0.015 | 0.797 | 0.702 | 0.045 | 0.149 | 0.052 |
| power_spherical_free | 5 | 0.509 ± 0.005 | 0.508 ± 0.013 | 0.754 | 0.694 | 0.021 | 0.111 | 0.042 |
| power_spherical_k1 | 5 | 0.532 ± 0.022 | 0.536 ± 0.020 | 0.810 | 0.696 | 0.033 | 0.160 | 0.047 |
| gaussian | 5 | 0.513 ± 0.013 | 0.513 ± 0.025 | 0.693 | 0.693 | 0.017 | 0.043 | 0.043 |

Sampled KL to the true pair probabilities, by pair kind

| condition | mixed | pure-diff | pure-same |
|---|---|---|---|
| tnbbeta_free | 0.039 | 0.068 | 0.087 |
| tnbbeta_f0.5 | 0.070 | 0.093 | 0.100 |
| vmf_free | 0.040 | 0.069 | 0.081 |
| vmf_k1 | 0.043 | 0.074 | 0.073 |
| power_spherical_free | 0.032 | 0.063 | 0.084 |
| power_spherical_k1 | 0.038 | 0.069 | 0.073 |
| gaussian | 0.033 | 0.066 | 0.086 |

#### scale_free

Oracle (true-probability) test AUC: 0.707

| condition | seeds | centre AUC | sampled AUC | centre logloss | sampled logloss | sampled ECE | centre KL | sampled KL |
|---|---|---|---|---|---|---|---|---|
| tnbbeta_free | 5 | 0.611 ± 0.024 | 0.637 ± 0.032 | 0.753 | 0.683 | 0.035 | 0.169 | 0.054 |
| tnbbeta_f0.5 | 5 | 0.590 ± 0.029 | 0.539 ± 0.037 | 1.405 | 0.695 | 0.043 | 0.900 | 0.061 |
| vmf_free | 5 | 0.598 ± 0.033 | 0.532 ± 0.034 | 1.300 | 0.692 | 0.034 | 0.809 | 0.057 |
| vmf_k1 | 5 | 0.591 ± 0.035 | 0.558 ± 0.021 | 1.730 | 0.830 | 0.224 | 1.304 | 0.261 |
| power_spherical_free | 5 | 0.599 ± 0.006 | 0.575 ± 0.056 | 1.276 | 0.689 | 0.027 | 0.776 | 0.055 |
| power_spherical_k1 | 5 | 0.590 ± 0.012 | 0.559 ± 0.026 | 1.508 | 0.743 | 0.146 | 1.034 | 0.143 |
| gaussian | 5 | 0.696 ± 0.013 | 0.647 ± 0.062 | 0.679 | 0.681 | 0.097 | 0.064 | 0.064 |

Sampled KL to the true pair probabilities, by pair kind

| condition | hub-hub | hub-leaf | leaf-leaf |
|---|---|---|---|
| tnbbeta_free | 0.174 | 0.042 | 0.055 |
| tnbbeta_f0.5 | 0.207 | 0.056 | 0.060 |
| vmf_free | 0.212 | 0.054 | 0.056 |
| vmf_k1 | 0.039 | 0.071 | 0.303 |
| power_spherical_free | 0.210 | 0.051 | 0.055 |
| power_spherical_k1 | 0.075 | 0.037 | 0.166 |
| gaussian | 0.124 | 0.028 | 0.071 |

#### ring

Oracle (true-probability) test AUC: 0.850

| condition | seeds | centre AUC | sampled AUC | centre logloss | sampled logloss | sampled ECE | centre KL | sampled KL |
|---|---|---|---|---|---|---|---|---|
| tnbbeta_free | 5 | 0.840 ± 0.006 | 0.840 ± 0.005 | 0.525 | 0.525 | 0.162 | 0.186 | 0.175 |
| tnbbeta_f0.5 | 5 | 0.836 ± 0.004 | 0.835 ± 0.004 | 0.538 | 0.535 | 0.160 | 0.200 | 0.184 |
| vmf_free | 5 | 0.839 ± 0.005 | 0.838 ± 0.005 | 0.544 | 0.544 | 0.182 | 0.212 | 0.180 |
| vmf_k1 | 5 | 0.835 ± 0.009 | 0.824 ± 0.010 | 0.650 | 0.593 | 0.205 | 0.390 | 0.195 |
| power_spherical_free | 5 | 0.836 ± 0.004 | 0.772 ± 0.124 | 0.625 | 0.591 | 0.157 | 0.333 | 0.174 |
| power_spherical_k1 | 5 | 0.833 ± 0.009 | 0.824 ± 0.012 | 0.598 | 0.607 | 0.229 | 0.284 | 0.162 |
| gaussian | 5 | 0.823 ± 0.006 | 0.823 ± 0.006 | 0.562 | 0.562 | 0.143 | 0.212 | 0.209 |

Sampled KL to the true pair probabilities, by pair kind

| condition | lattice | shortcut |
|---|---|---|
| tnbbeta_free | 0.033 | 0.178 |
| tnbbeta_f0.5 | 0.033 | 0.187 |
| vmf_free | 0.066 | 0.182 |
| vmf_k1 | 0.160 | 0.196 |
| power_spherical_free | 0.197 | 0.174 |
| power_spherical_k1 | 0.249 | 0.161 |
| gaussian | 0.065 | 0.212 |

### Reading

- **latent_sphere** (oracle AUC 0.962): centre AUC is 0.940-0.949 for every family. On sampled
  KL, free TNBBeta is lowest (0.409 against 0.434-0.476) and better than every other condition
  in **5/5 paired seeds**, by 0.025-0.031, including TNBBeta with `m` pinned unimodal
  (`f0.5`). This is the first graph where the free (bimodal) TNBBeta beats the unimodal one in
  every seed. It is a 7% KL gap, `f0.5` is a single `m` value, and free versus fixed also
  differs in `epsilon` per node, so it does not isolate bimodality. **Caveat on the design:**
  the decoder has no bias term, so the generator's `sigmoid(10 cos - 9)` is not exactly
  representable (every family's KL is about 0.4 and ECE about 0.19). The preset is not "the
  decoder's own generative model" as its docstring implies; an offset-free generator would be
  needed for that claim.
- **scale_free** (oracle 0.707): the Gaussian is clearly best on centre AUC (0.696 against
  0.590-0.611 for every sphere family, 0/5 seeds for TNBBeta), as expected, since a bounded
  norm with one shared temperature cannot express node-specific degree. Free TNBBeta is the best
  sphere family (centre 0.611, sampled 0.637), ahead of `f0.5` and free vMF/PS in 4/5 seeds.
- **ring** (oracle 0.850): sampled KL is 0.174-0.209 for all but the Gaussian (worst, 0.209;
  TNBBeta better in 5/5 seeds). Free TNBBeta's lattice-pair KL is 0.033 against 0.065-0.249,
  but `f0.5` gets the same 0.033, so it is not bimodality. Free PS sampled AUC collapses in
  some seeds (0.772 +- 0.124).
- **mixed_membership** (oracle 0.655): no family learns it (centre AUC 0.51-0.53), and the
  Gaussian's KL (0.043) is the lowest only because it predicts the base rate. This graph is
  uninformative at this size and density; it does not show a difference between families.

**Standing conclusion.** Across eight graph families there is still no graph on which TNBBeta
is required: the sphere families tie on centre AUC except where an unbounded norm matters
(core_periphery, scale_free, both won by the Gaussian). The one signal that favours free
(bimodal) over unimodal TNBBeta is latent_sphere sampled KL, and it needs the confounds above
removed before it can be attributed to bimodality.

### Suggested next steps

1. Re-specify latent_sphere without an unrepresentable offset, and compare free TNBBeta against
   `m` pinned at several values (negative and positive), not just `f0.5`.
2. Measure the second mode's mass for free TNBBeta (`r_bar` cancels for antipodal mixtures).
3. A larger, denser mixed_membership graph so the model can learn it at all.
