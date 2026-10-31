# Pre-registration: a wide search over synthetic graph types for where TNBBeta helps, 2026-10-31

- **Date:** 2026-10-31 (written and committed before any discovery run started).
- **Code:** branch `synthetic-graph-families` (PR #69) at `d92b275`; this file is committed after it.
  Analysis code is `apps/eval/graph_search_stats.py` and must not change between now and the
  confirmation stage; any change is logged under "Deviations".
- **Pilot:** job 2283700 (960 runs: 20 configs x 4 draws x {tnbbeta, vmf_init, power_spherical_init, gaussian} x 3 learning rates; seeds from 9,000,000).
  Graphs in the pilot are disjoint from discovery and confirmation graphs.
- **Commands:**

  ```bash
  uv run python -m apps.synthetic.graph_search configs --name pilot --per-family 2 --seed 7
  uv run python -m apps.synthetic.graph_search manifest --stage pilot --draws 4
  STAGE=pilot sbatch --array=0-23 scripts/slurm/graph_search.sbatch
  uv run python -m apps.eval.graph_search_stats tune
  uv run python -m apps.eval.graph_search_stats variance
  uv run python -m apps.synthetic.graph_search configs --name discovery --per-family 12 --seed 11
  uv run python -m apps.synthetic.graph_search manifest --stage discovery --draws 32
  STAGE=discovery sbatch --array=0-95 scripts/slurm/graph_search.sbatch
  uv run python -m apps.eval.graph_search_stats test --stage discovery
  ```

## Question and null hypotheses

Where, in a wide space of synthetic graphs, does a TNBBeta posterior (free `p`, `q`, `epsilon`)
give a better graph VAE than a power-spherical (PS) or a von Mises-Fisher (vMF) posterior? We are
not committed to any mechanism (the modality `m` is one candidate explanation among several).

For a graph configuration `c` and a metric `m` (oriented so larger is better) the null is:
**TNBBeta is no better than PS by more than the margin, or TNBBeta is no better than vMF by more
than the margin.** This is an intersection-union test; its p-value is the larger of the two
one-sided p-values, so no further correction is needed within a pair. The Gaussian posterior is
reported as a descriptive reference only and is not part of any null.

## Graph space

Ten families (`tnbbeta_vae.data.graph_space.FAMILIES`): assortative SBM, bipartite (a negative
control: a dot-product decoder cannot represent it), core-periphery, hierarchical SBM,
mixed-membership, latent-sphere kernel, random geometric, small-world ring, scale-free
(Chung-Lu), degree-corrected SBM. Every graph is **undirected and unweighted**. 12 configs per
family (120 in total) from a scrambled Sobol sequence over: size (log-uniform 300-1200 nodes),
mean degree (log-uniform 4-40), blocks or intrinsic dimension (log-uniform 2-16), degree
heterogeneity (none / mild / heavy), node features (identity or informative) and a target oracle
AUC (uniform 0.65-0.95). The structure strength is bisected so the expected oracle AUC hits the
target; a target a family cannot reach (within 0.02) is rejected and the next Sobol point used,
so each family's targets are confined to what it can express (e.g. mixed-membership 0.66-0.75).
The realized discovery configs span 300-1200 nodes, mean degree 4-38 and oracle AUC 0.65-0.94;
62 of 120 use informative features. Configs are in `graph_search/configs_discovery.json`.

## Models (all conditions)

GraphVAE, 500 epochs, best-validation-AUC state, ambient dimension 16 (Gaussian 15), no dropout,
learned decoder bias, 32 posterior samples at evaluation. Conditions: `tnbbeta` (free);
`vmf_init` / `power_spherical_init` (**primary comparators**: kappa starts at `d` = 16 and is
trained); `vmf_pin` / `power_spherical_pin` (kappa fixed at 16) and `vmf_default` /
`power_spherical_default` (the repo's default initialisation) as secondary comparator sets;
`gaussian` (reference). The learning rate is tuned per graph family and condition on pilot draws
only (lowest mean sampled log loss) and then frozen; the secondary comparators inherit their
`_init` condition's rate. Tuned rates:

| graph family | tnbbeta | vmf_init | power_spherical_init | gaussian |
|---|---|---|---|---|
| assortative | 0.01 | 0.003 | 0.003 | 0.03 |
| bipartite | 0.003 | 0.003 | 0.003 | 0.01 |
| core_periphery | 0.01 | 0.003 | 0.01 | 0.03 |
| hierarchical | 0.03 | 0.01 | 0.03 | 0.01 |
| mixed_membership | 0.01 | 0.003 | 0.003 | 0.01 |
| latent_kernel | 0.01 | 0.003 | 0.03 | 0.003 |
| geometric | 0.01 | 0.003 | 0.003 | 0.003 |
| ring | 0.003 | 0.003 | 0.003 | 0.003 |
| scale_free | 0.01 | 0.03 | 0.003 | 0.003 |
| degree_corrected | 0.003 | 0.003 | 0.003 | 0.003 |

## Metrics, orientation and practical margins

Each metric is computed on the held-out test pairs (balanced positives and random negatives) or
against the known truth. A difference must exceed the margin, not merely zero.

| metric | better | margin | what it checks |
|---|---|---|---|
| `sampled_log_loss` (**primary**) | lower | 0.005 | posterior-predictive proper score |
| `centre_auc` | higher | 0.005 | standard link prediction; blind to posterior shape |
| `cal_slope_dev` = abs(calibration slope - 1) | lower | 0.05 | calibration, no binning |
| `uncertainty_auroc` | higher | 0.02 | does the predictive spread flag wrong edges |
| `truth_kl` | lower | 0.005 | sampled prediction vs the balanced true probability |
| `rank_corr` | higher | 0.01 | Spearman of centre logits vs log true probability |

Margins were fixed from interpretability before looking at any discovery data.

## Design and tests

- **Unit of replication:** the graph draw. Per config, **K = 32 independent draws** (seeds from
  10,000,000), each with its own edge split; all conditions of a draw share the graph, split and
  seed (paired). 120 configs x 32 draws x 8 conditions = 30,720 runs.
- **Test:** one-sided paired Wilcoxon signed-rank test of (oriented difference - margin) > 0,
  separately against each comparator; draws with a non-finite metric on either side are dropped;
  p = the maximum over the two comparators.
- **Discovery family:** every (config, metric) test on the primary comparator pair
  (120 x 6 = 720 tests), corrected with **Benjamini-Hochberg at q = 0.05**; Holm and the
  Bonferroni threshold (0.05 / 720 = 6.9e-5) are reported alongside. BH assumes positive
  dependence between tests, which is plausible for metrics on the same graph; configs are
  independent.
- **Pooled family tests:** per (graph family, metric), config as replicate (its mean difference
  over draws), same test, Holm over these tests. These answer "does TNBBeta help on this family".
- **Secondary comparator sets** (pinned, default) are analysed identically as separate,
  descriptive families. **Reverse direction** (TNBBeta significantly worse than either
  comparator, mirrored test) is reported as its own descriptive family.
- **Confirmation:** every (config, metric) rejected at BH q <= 0.05 on the primary set is rerun
  on 32 fresh draws (seeds from 20,000,000), tested on exactly the rejected pairs, with Holm at
  0.05. Only confirmed pairs are called findings; the explanation phase starts from those.
- **Data handling:** diverged runs (non-finite loss or gradient; the model then keeps its best
  earlier state or chance level) are kept at their metrics, since failing to train is a real
  property of the model; no run is excluded for any other reason.

## Power, from the pilot (K = 4 draws per config, so these SDs are themselves rough)

| metric | margin | median SD of paired diff | draws for 80% power at alpha = 0.001, true effect = 2 x margin |
|---|---|---|---|
| centre_auc | 0.005 | 0.0147 | 134 |
| sampled_log_loss | 0.005 | 0.0216 | 290 |
| cal_slope_dev | 0.05 | 0.2795 | 483 |
| uncertainty_auroc | 0.02 | 0.0330 | 42 |
| truth_kl | 0.005 | 0.0342 | 722 |
| rank_corr | 0.01 | 0.0357 | 197 |

K = 32 is far below these counts, so per-config tests will only detect large effects
(roughly 0.015 nats of log loss beyond the margin at alpha = 0.001, about 0.011 at 0.01). A small
true advantage will go undetected; the pooled family tests (12 configs per family) have more
power for consistent effects. We chose K = 32 for cost (about 31k runs) and say so. A non-rejection
is **not** evidence that TNBBeta does not help.

## Known limits

All graphs are undirected and unweighted and the decoder is a symmetric dot product, so asymmetric
or weighted structure is out of scope here. The dot-product decoder cannot represent bipartite
structure, which is therefore a control. Pilot observations (K = 4, exploratory, not used to select
anything but learning rates): 6 of 960 runs diverged, all TNBBeta (scale-free and one
core-periphery draw), and the Gaussian had the lowest mean log loss in most families.

## Deviations

- **2026-11-01, after discovery:** `graph_search_stats.py` gained `confirmation_report`
  (`test --stage confirmation`), which restricts the confirmation tests to the discovery-rejected
  (config, metric) pairs and applies Holm over just those. This implements the confirmation rule
  stated above, which the analysis code had not yet encoded; the discovery analysis (`render_report`,
  `run_config_tests`, margins, BH, Holm) is unchanged. The code change is in the commit that adds
  this entry.
- **2026-11-01, not pre-registered (exploratory):** a sensitivity analysis that removes every draw
  on which TNBBeta diverged (146 TNBBeta runs diverged in discovery; no vMF, PS or Gaussian run did),
  reported separately from the pre-registered results.
