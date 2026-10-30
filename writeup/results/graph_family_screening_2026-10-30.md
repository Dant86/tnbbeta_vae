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
