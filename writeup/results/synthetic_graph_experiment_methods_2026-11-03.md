# Synthetic graph experiment: how the graphs, models and metrics are defined

Date: 2026-11-03. Base commit: `4cab2d6` (`master`). A methods note, no new results and no
commands of its own; every statement below was read from the code at that commit
(`src/tnbbeta_vae/data/graph_space.py`, `apps/synthetic/graph_search.py`,
`src/tnbbeta_vae/models/predictive.py`, `src/tnbbeta_vae/models/graph_vae.py`,
`src/tnbbeta_vae/data/planetoid.py`) and the pre-registrations
(`graph_search_preregistration_2026-10-31.md`, `graph_search_metric_first_preregistration_2026-11-02.md`,
`scale_free_density_figure_preregistration_2026-11-03.md`). Where something comes from a
pre-registration and not from the code, the text says so.

Two different things are called "draws" in the study. To keep them apart this note says
**graph draw** for a replicate (a new graph, edge split, initialisation and training run,
paired across conditions) and **posterior sample** for one of the 32 samples of the node
latents used at evaluation.

## 1. Generative model of the graphs

Every graph is undirected, unweighted and has no self-loops. Each pair `i < j` has an
independent edge with probability

    P_ij = clip(c * W_ij * w_i * w_j, 0, 1),     P_ii = 0.

* `W` is a family-specific **shape matrix** (below), possibly random per draw.
* `w` are **degree weights**: mean-one Pareto weights `w = (Pareto(a) + 1) / mean`, with tail
  index `a = 6` (`mild`) or `2.5` (`heavy`), or all ones (`none`). `scale_free` and
  `degree_corrected` set their own tail (below).
* `c` is a scalar chosen so that the mean degree equals the target. Probabilities are clipped
  at 1, so `c` is refit (up to 6 passes, tolerance 0.1%) after the clip. For very high targets
  the realised mean degree can fall slightly short (the density figure's nominal 35 realised 34.9).
* The true `P` of every pair is kept with the sampled graph. This is what makes truth KL possible.
* **Contrast.** Each family has one monotone structure-strength parameter `t`. It is bisected
  (14 steps, log scale, on one fixed calibration draw with seed 0) so that the **expected oracle
  AUC** hits a target. The oracle AUC is that of ranking all pairs `i < j` by their true `P`,
  positives weighted by `P` and negatives by `1 - P`, ties counting half. Targets a family cannot
  reach (within 0.02) are rejected in the config sampler.

### Families (`W`, with `n` nodes and `k` = `num_groups`)

| family | `W_ij` | notes |
|---|---|---|
| `assortative` | `1 + t` if same block, else 1 | `k` equal blocks (SBM) |
| `degree_corrected` | as `assortative` | weights fixed at Pareto tail 2.5 |
| `bipartite` | `1 + t` if on opposite sides, else 1 | two equal halves; a dot-product decoder cannot represent it (negative control) |
| `core_periphery` | `1 + t` core-core, `1 + (sqrt(1+t) - 1)` core-periphery, 1 otherwise | `k` groups; the first 20% of each group is core; the interactions are within a group only |
| `hierarchical` | `1 + t` same block, `1 + sqrt(t)` same super-block, else 1 | `max(4, k + k mod 2)` blocks paired into super-blocks |
| `mixed_membership` | `m_i' (1 + t I) m_j` | memberships `m_i ~ Dirichlet(0.3)` over `k` groups, new per draw |
| `latent_kernel` | `exp(t (z_i . z_j - 1))` | `z_i` uniform on `S^(k-1)`, new per draw |
| `geometric` | `exp(-t d_ij^2)` | points uniform on the 2-torus (3-torus if `k > 10`), toroidal distance, new per draw |
| `ring` | `1 + t` if within `max(2, round(0.02 n))` positions on the cycle | deterministic layout |
| `scale_free` | 1 (all pairs) | Chung-Lu: all structure comes from the weights, Pareto tail `a = 1.4 + 8 / (1 + t)` |

### Node features

* `none`: identity (one-hot node ids). The GCN's first layer is then a learned per-node
  embedding, smoothed by neighbour aggregation.
* `informative`: a random 32-dimensional projection of each node's true probability row plus
  Gaussian noise with the same variance as the signal (signal-to-noise 1). Given the truth they
  are independent of the sampled edges, so no held-out edge is encoded in them.

### Config spaces

* **Discovery** (pre-registration): 12 configs per family (120) from a scrambled Sobol sequence over
  size (log-uniform 300-1200 nodes), mean degree (log-uniform 4-40), blocks or intrinsic dimension
  (log-uniform 2-16), heterogeneity (none / mild / heavy), features (identity / informative) and a
  target oracle AUC (uniform 0.65-0.95). 62 of the 120 use informative features.
* **Fresh** (metric-first study): 20 configs per family (200), 16 graph draws each, for the
  confirmatory test. The `gaussian` reference was dropped from this stage. I did not re-read
  the fresh-config generator for this note; I assume it samples the same space as discovery.
* **Density figure**: three `scale_free` configs, `n = 600`, contrast 13 (tail index about 1.97),
  identity features, mean degree 6 / 15 / 35 (realised 5.99 / 15.00 / 34.92), 32 graph draws each.
  `num_groups` and `heterogeneity` are ignored by this family. The drawn illustrations use the same
  generator with 36 nodes and mean degrees 2 / 8 / 20 (a drawing seed of 7).

## 2. Model and training

* **Encoder.** A 2-layer GCN on the symmetrically normalised *training* adjacency: hidden width
  32, ReLU, no dropout, and a posterior head from `models/heads.py` per node. Node posteriors
  are independent given the graph.
* **Latent space and prior.** Ambient dimension 16 (15 for the Gaussian reference); the prior is
  uniform on the sphere.
* **Decoder.** `sigmoid(temperature * z_i . z_j + bias)` with one learned `temperature = exp(log_temperature)`
  and one learned scalar `bias` (`decoder_bias=True` in every condition); for the Gaussian,
  `z_i . z_j` without the temperature.
* **Objective.** A Monte Carlo ELBO. Optimiser: Adam. Reconstruction is binary cross-entropy on all training edges
  (positives) plus one uniformly random node pair per training edge as a negative, re-drawn each
  step (these negatives are not screened against the training graph, so a few are true edges).
  The KL term is analytic for vMF and PowerSpherical and uses the exact `log q` for TNBBeta.
* **Training.** 500 epochs, Adam, the learning rate tuned per family and condition on pilot draws
  and then frozen (grid 0.003 / 0.01 / 0.03), keeping the state with the best validation AUC.
* **Edge split.** 5% of the edges are held out for validation and 10% for test. For each held-out
  set an equal number of node pairs that are not edges of the *full* graph are drawn uniformly
  without duplicates, disjoint between validation and test. The remaining 85% of the edges form the
  training graph that the encoder sees. The split is seeded by the run's seed, so it is shared
  by all conditions of one graph draw.
* **Conditions.** `tnbbeta` (free); `vmf_init` / `power_spherical_init` (kappa starts at 16 and
  is trained; the **primary comparators**); `vmf_pin` / `power_spherical_pin` (kappa fixed at 16);
  `vmf_default` / `power_spherical_default` (the repository's default initial kappa); `gaussian`
  (reference). All conditions of one graph draw share the graph, split and seed (paired).
* **vMF sampler.** The fresh and discovery stages used the library `VonMisesFisher` sampler
  before its default was changed to Wood's exact envelope (PR #70), which is biased for kappa above
  10 at ambient dimension 5 and higher. The density figure's bars used the exact sampler. The
  audit's rerun of 40 configs found no change in the graph conclusions
  (`vmf_sampler_bias_audit_2026-11-03.md`).

## 3. How the posterior samples are drawn at evaluation

`evaluate` (`apps/synthetic/graph_search.py`) runs the encoder once over the training
adjacency to get the per-node posteriors, then:

1. `torch.manual_seed(seed)` and **one joint draw** `samples = posterior.rsample(32)`, shape
   `(32, n, 16)`: 32 posterior samples, each an embedding of **all** nodes. Within a sample the nodes are
   independent draws from their own posteriors; the 32 samples are independent of each other.
2. For one posterior sample `s`, the logit of any pair `(i, j)` is `temperature * z_i^s . z_j^s + bias`,
   so a node's sample `z_i^s` is shared by every pair that contains it in that sample.
3. The **posterior-predictive probability** of a pair is the average of the sigmoid over the 32
   posterior samples, `p_hat_ij = (1/32) * sum_s sigmoid(temperature * z_i^s . z_j^s + bias)`.
   Averaging happens on the probability scale, not on the logit.
4. The **centre** score is `sigmoid(temperature * c_i . c_j + bias)` with one point per node,
   `c_i = posterior_centre` (the mean direction; negated for TNBBeta when `p < 0.5`). No sampling
   is involved.

There are two separate sets of 32 posterior samples per run. The test-pair metrics use the
joint draw in step 1. The truth-KL matrix `p_hat` over all pairs comes from
`dense_edge_probabilities`, which calls `posterior.rsample(32)` again (the RNG state continues from
the seeded state of step 1, so it is deterministic but not the same 32 samples).

## 4. The two metrics

### Sampled AUROC

* **Pairs scored:** the test positives (the 10% held-out edges) and the same number of test negatives
  (random non-edges of the full graph, drawn at split time). No other pairs.
* **Score:** `p_hat_ij` from the 32-sample joint draw in step 1 of section 3.
* **Statistic:** the AUROC of positives against negatives from the rank sum of all scores, with ties
  counting one half.
* **Centre AUROC** is the same computation with the centre score instead of `p_hat`. TNBBeta's
  non-cap posteriors make the centre a poor summary, which is why sampled AUROC is shown as well.

### Truth KL

* **Pairs scored:** every pair `i < j` that is **not a training edge** (`held` mask =
  `train_adjacency == 0`, upper triangle). That is the validation and test positives plus all
  non-edges, `n(n-1)/2` minus the training edges. It is not restricted to the test pairs.
* **Truth:** the balanced true probability `q_ij = P_ij / (P_ij + rho)`, where `rho` is the mean of
  `P` over all off-diagonal pairs. Training pairs positives (drawn in proportion to `P`) with the same
  number of uniform random pairs, so the Bayes-optimal output is `P / (P + rho)`, not `P`.
* **Prediction:** `p_hat_ij` from the second 32-sample set above, averaged over the posterior
  samples before it enters the KL.
* **Statistic:** the mean over those pairs of `KL(Bernoulli(q_ij) || Bernoulli(p_hat_ij))`, with both
  arguments clamped to `[1e-6, 1 - 1e-6]`. All pairs have equal weight; lower is better.
* **Advantage `E_c`** (the confirmatory outcome, per config): the mean over its 16 graph draws of the
  paired difference in truth KL between the less favourable of `vmf_init` and
  `power_spherical_init` and `tnbbeta` (positive = TNBBeta better). The registered H1 uses
  `S_c = E_c / KL0`, where `KL0` is the truth KL of the better comparator on the config.

### Other probability metrics (for reference)

* **Sampled log loss:** `bernoulli_log_loss(p_hat, target)` on the test pairs, with clamping at 1e-6.
* **ECE:** equal-width 10-bin expected calibration error on the same test pairs, using `p_hat` from the
  dense 32-sample matrix (the second sample set; extended stages only: `fresh` and `density`).

## 5. Limits to keep in mind when reading the results

* Truth KL is computed on graphs with known `P`, so it cannot be computed on real data.
* The metrics use only 32 posterior samples; the sampling noise in `p_hat` is not separated from
  the training noise (each graph draw is one trained model).
* Identity features make the model a free per-node embedding. The informative-feature graphs are
  a different regime, and the density figure used only identity features.
* The bipartite family is a deliberate negative control for the dot-product decoder.
