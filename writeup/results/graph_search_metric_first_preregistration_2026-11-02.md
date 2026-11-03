# Pre-registration: a metric-first re-analysis of the wide graph-type search, 2026-11-02

**Status: FROZEN** (2026-11-02), before any fresh-stage training and before any analysis that
relates an outcome to a covariate. After this point the file is edited only by appending to the
"Deviations" section at the end. Base commit: `6b1880f` on branch `synthetic-graph-families`
(the new code of this study is uncommitted at freeze time; its files are listed under "Code").
Items that were marked **[CHOICE]** in the draft were confirmed by the user, who approved the
draft's choices 1-9 (outcome `S_c`, leave-family-out primary CV, block-permutation null, `truth_kl_w`,
HeaRT-style negatives, covariate sets G and M, fresh-stage design, decision rule, no pooling of
confirmation draws).

## Amendments to the approved draft (all before freezing)

1. **Metric dimension is a first-class objective (user instruction).** The end goal is to identify
   graph types **and** metrics on which TNBBeta succeeds, so besides the per-config `truth_kl`
   effect the study maps a config x metric effect matrix, with pre-stated orientation,
   standardisation, margins, multiplicity control and a graph family x metric summary table
   (section "The metric dimension"). `truth_kl` stays primary; `sampled_log_loss` is its twin; the
   rest are descriptive but reported with their intervals.
2. **Pre-freeze power simulation done** (section "Fresh-stage size, from the power simulation"). It
   recommends a larger fresh stage than the draft's 12 configs per family: **20 configs per family
   (200 configs), 16 draws each**, and drops `gaussian` from the fresh stage (reference only).
3. **Implementation details fixed that the draft left open:** the gradient-boosted secondary model
   and its partial dependence are dropped (no scikit-learn in the environment, and the environment
   is not changed); the elastic net is implemented in numpy; the family-block permutation test uses
   2,000 permutations (not 5,000); posterior-predictive graph statistics use 30 sampled graphs per
   matrix and are computed on **all** pairs from the implied true-scale probability
   `P_hat = rho * p / (1 - p)` (the model is trained on balanced targets `t = P / (P + rho)`, so
   its raw `p` is not an edge probability; `rho` is the true mean pair probability, the same `rho`
   `truth_kl` uses); latent recovery uses the Gram-Spearman score for all three continuous latents and
   Procrustes for `latent_kernel` only (the draft's wording on the geometric latent is replaced by this).
4. **The `second_mode_mass` finding is carried forward.** In the discovery runs TNBBeta's
   `second_mode_mass` is not 0 (config-level median 0.07, 90th percentile 0.43, above 0.01 in
   94 of 120 configs), so it enters Set M as an explanatory model-side covariate, never as a
   graph property.

## Design (the approved draft, with the amendments applied)

- **Status line of the draft:** nothing in this file had been run when it was drafted.
- **Code:** branch `synthetic-graph-families`, HEAD `6b1880f`. Existing search code
  (`apps/synthetic/graph_search.py`, `apps/eval/graph_search_stats.py`, `apps/eval/graph_search_explore.py`,
  `src/tnbbeta_vae/data/graph_space.py`, `src/tnbbeta_vae/models/predictive.py`) is the baseline; new code
  (covariate computation, `evaluate` extension, the analysis script) will be committed before freezing and its
  commit hash recorded here.
- **Commands:** to be listed at freezing (stages: `covariates`, `fresh` configs/manifest/run, `analyse`).
- **Relation to earlier work:** `graph_search_preregistration_2026-10-31.md` (design, margins, tuned
  learning rates, all inherited unchanged) and `graph_search_results_2026-10-31.md`. That search asked
  "for which (config, metric) pairs is TNBBeta significantly better", with six metrics and a win count. This
  one fixes one proper metric first and asks what the graphs on which TNBBeta does well have in common.

## Question

With a posterior-aware metric fixed in advance, on which synthetic graphs does TNBBeta beat tuned vMF
and power-spherical (PS) posteriors, and what do those graphs have in common? The deliverable is an
effect size per graph configuration and a regularised account of how it depends on graph properties,
not a count of significant configs.

**Honesty about what is and is not blind.** The 120 discovery configs and 32 draws each were analysed
before (the 9-better / 28-worse truth-KL counts and the exploratory attribute tables are in the earlier
write-up). The choice of `truth_kl` as primary is made on principle (below), but the user has seen those
counts, so the discovery-stage analysis here is **exploratory** by construction. Only the fresh-config
stage is confirmatory.

## Metrics

### Primary: `truth_kl` (unchanged implementation)

Mean over node pairs `i < j` that are not *training* edges (held-out true edges, validation edges and all
non-edges) of `KL(Bernoulli(t_ij) || Bernoulli(p_ij))`, with `t = P / (P + rho)` (`balanced_truth`, `rho` the
mean pair probability: what 1:1 negative sampling teaches a Bayes-optimal classifier) and `p_ij` the
posterior-predictive edge probability `mean_s sigmoid(temperature * z_i^s . z_j^s + bias)` over **32**
posterior draws (`SAMPLES = 32` in `graph_search.py`; the brief's "64" is not what the code does).
`eps = 1e-6` clamp on both arguments. Lower is better.

Why this is the primary: it scores the whole posterior-predictive distribution (not the centre), it is a
proper divergence from the generator's own probabilities, and it uses all ~n^2/2 pairs, so it is far
less noisy than a score on the ~10% held-out edges plus equally many sampled negatives (60 test edges at
n = 300, mean degree 4).

### Proper-score twin: `sampled_log_loss`

Held-out log loss on the test positives and equally many sampled negatives, on the mean sampled
prediction. It is the metric the comparators' and TNBBeta's learning rates were tuned on (pilot), which
is a conflict of interest for the model-selection step, not for the analysis; see Risks. It is also the
only one of the two available on real graphs.

**Bridge to real graphs [CHOICE, new descriptive metric `truth_kl_w`].** With test positives drawn
proportionally to `P` and test negatives uniformly from non-edges, expected held-out log loss equals a
graph-specific irreducible term plus a pair-weighted truth KL with weights
`w_ij = 1/2 * (P_ij / sum P + (1 - P_ij) / sum (1 - P))` and targets
`t^w_ij = (P_ij / sum P) / (P_ij / sum P + (1 - P_ij) / sum (1 - P))`. `truth_kl_w` computes exactly that
from the dense predictive matrix. It differs from `truth_kl` only in the pair weighting (up-weighting
likely pairs, instead of the uniform weight that lets the many unlikely pairs dominate) and the target's
`(1 - P)` factor, so reporting both shows whether a conclusion survives the change of weighting that
a real-graph held-out log loss would impose. It is the formal link between "what we can compute here"
and "what we could compute on a real graph".

### Descriptive (reported per config, never entering the primary outcome, covariates or decision rule)

| metric | definition | why |
|---|---|---|
| `recal_truth_kl` | `truth_kl` after a Platt map fitted on validation pairs (already in `evaluate`; not stored in discovery runs) | separates posterior shape from overall temperature |
| calibration slope, `cal_slope_dev`, ECE | slope as implemented; ECE = `expected_calibration_error`, 10 equal-width bins, on test pairs (Guo et al. 2017; GNN calibration, Wang et al. NeurIPS 2021) | calibration; ECE is in `predictive.py` but not called by `evaluate` |
| Hits@K, MRR | HeaRT-style hard negatives (Li et al. NeurIPS 2023): per test positive `(u, v)`, 100 negatives `(u, k)` from non-edges of the full graph chosen by training-graph common-neighbour count and personalised PageRank (50 each, highest-ranked first; random if fewer candidates), scored by centre logit and by the mean sampled prediction; K in {10, 50}; ties get their average rank | replaces AUC; **[CHOICE]** the exact negative recipe is mine, not a reproduction of HeaRT's code |
| posterior-predictive graph statistics (`gs_degree_w1`, `gs_clustering_err`, `gs_triangle_err`) | 30 graphs are drawn (independent Bernoulli per pair, all pairs) from the implied probability matrix `P_hat = min(1, rho p / (1 - p))` of the mean sampled prediction `p`, and 30 from the true `P` of the same draw. `gs_degree_w1` = 1-Wasserstein distance between the pooled degree samples divided by the true mean degree; `gs_clustering_err` and `gs_triangle_err` = `abs(a - b) / (a + b)` of the mean global clustering and the mean triangles per node. Hunter, Goodreau & Handcock (2008) framing; MMD only as an aside, not computed | does the model reproduce graph-level structure |
| latent recovery (`latent_gram_rho`, `latent_procrustes`, `latent_nmi`) | `latent_gram_rho` (latent_kernel, geometric, ring): Spearman of the centres' cosine similarity with the true latent kernel (inner product, negative squared toroidal distance, negative ring gap) over 200,000 random pairs; `latent_procrustes` (latent_kernel only): `||C R - Z||_F^2 / n` after orthogonal alignment of unit centres to the zero-padded true unit latent, lower is better; `latent_nmi` (block families, mixed_membership): overlapping NMI, LFK-normalised (McDaid et al. 2011), between k-means clusters of the unit centres (k = number of true blocks, 5 restarts) and the true cover (block labels; for mixed_membership every block with at least 0.25 of the node's membership, always including the largest); undefined (nan) for scale_free | recovers the latent, not just the edges |
| `centre_auc`, `sampled_auc`, `rank_corr`, `uncertainty_auroc` | as before | descriptive only; centre-based metrics cannot see posterior shape |
| `truth_kl` by true-probability decile | KL split by deciles of `t` | where in probability space the difference comes from |
| `truth_kl` at `eps = 1e-4` | same, different clamp | sensitivity |

## Outcome (not a win count)

For each config `c`, comparator `a` in the primary set `{vmf_init, power_spherical_init}`, draw `k`
(K draws, all conditions share graph, split and seed):

`d_{c,a,k} = truth_kl(a) - truth_kl(tnbbeta)`   (positive = TNBBeta better, nats per pair)

- `effect_{c,a} = mean_k d_{c,a,k}`; the config's **headline effect** is `E_c = min_a effect_{c,a}`
  (intersection-union logic: TNBBeta is only as good as it is against the better comparator). Both
  comparator-specific values are also reported.
- **Interval:** paired percentile bootstrap over draws (10,000 resamples, draws resampled jointly across
  comparators, `E_c` recomputed each time, so the selection of the less favourable comparator is inside
  the interval). Reported as 95% interval. No p-value is attached to any config; configs are described
  by `E_c` and its interval.
- **Scale-free companion [CHOICE]:** `S_c = E_c / KL0_c`, where `KL0_c` is the truth KL of the best
  *constant* predictor, `KL(t || mean(t))` over the same pair set, computed from the true `P` alone
  (headroom). Absolute nats scale with the graph's own sparsity and size and are dominated by the
  few high-effect configs (geometric configs reached +0.2 nats earlier); `S_c` is the fraction of
  available headroom TNBBeta gains over the better comparator. **Primary regression outcome is `S_c`;**
  `E_c` in nats is the secondary outcome with the identical analysis.
- **Reliability ceiling:** split-half correlation of `E_c` across configs, between draws 0-15 and 16-31
  (Spearman-Brown corrected), reported before any covariate analysis. It bounds every R^2 below.
- **Other comparator sets, descriptive:** the same `E_c` against the pinned-kappa set
  (`vmf_pin`, `power_spherical_pin`) and the default-initialised set, to show how much of the effect is
  kappa optimisation (the earlier write-up found 45 of 46 confirmed primary wins also better than pinned).
- **Descriptive tier labels:** for the sentence "the graphs where TNBBeta wins", a config is labelled
  `win` if the lower 95% bound of `E_c` exceeds 0.005 nats (the earlier truth-KL margin), `loss` if the
  upper bound is below -0.005, else `inconclusive`. Labels are used only for tables and plots, never as a
  regression target or inside a decision rule, and counts of them carry no inferential weight.
- Diverged TNBBeta runs (146 of 3,840 in discovery) are kept at their metrics, as pre-registered before;
  a sensitivity analysis drops draws with a TNBBeta divergence.

## The metric dimension (user-requested amendment): where does each metric favour TNBBeta?

The second objective is the **config x metric effect matrix** and its graph family x metric summary.

**Metric set and orientation.** Every metric is oriented so that positive = TNBBeta better. For
metric `m`, comparator `a`, draw `k`: `d = m(a) - m(tnbbeta)` for lower-is-better metrics and
`d = m(tnbbeta) - m(a)` for higher-is-better ones. Draws with a non-finite value on either side are
dropped. Margins are in native units, fixed now, before any data exist for the new metrics:

| metric | better | margin | tier |
|---|---|---|---|
| `truth_kl` | lower | 0.005 | **confirmatory (primary)** |
| `sampled_log_loss` | lower | 0.005 | **confirmatory (proper-score twin)** |
| `truth_kl_w`, `recal_truth_kl`, `truth_kl_eps1e4` | lower | 0.005 | descriptive |
| `cal_slope_dev` | lower | 0.05 | descriptive |
| `ece` | lower | 0.01 | descriptive |
| `hard_hits10`, `hard_hits50` (mean sampled prediction) and `hardc_hits10`, `hardc_hits50` (centre) | higher | 0.02 | descriptive |
| `hard_mrr`, `hardc_mrr` | higher | 0.01 | descriptive |
| `gs_degree_w1`, `gs_clustering_err`, `gs_triangle_err` | lower | 0.05 | descriptive |
| `latent_gram_rho`, `latent_nmi` | higher | 0.02 | descriptive (where defined) |
| `latent_procrustes` | lower | 0.02 | descriptive (latent_kernel) |
| `centre_auc`, `sampled_auc` | higher | 0.005 | descriptive contrast |
| `rank_corr` | higher | 0.01 | descriptive contrast |
| `uncertainty_auroc` | higher | 0.02 | descriptive contrast |

**Effect records.** For each (config, metric) and comparator in the primary set, report the mean `d`
in native units and the standardised effect `dz = mean(d) / sd(d)` over draws (each metric divided
by its own between-draw SD); the (config, metric) record is the **less favourable comparator's**
values (intersection-union), with its paired-bootstrap 95% interval (10,000 resamples of draws,
recomputing the min over comparators).

**Test and classification.** One-sided paired Wilcoxon signed-rank test of `d > margin`, separately for
each comparator, `p = max` over the two comparators (the intersection-union p-value, as in the
2026-10-31 pre-registration, using `graph_search_stats.one_sided_p`); the reverse test of
`d < -margin` likewise gives `p_reverse`. A (config, metric) is **better** if the multiplicity-adjusted
`p <= 0.05`, **worse** if the adjusted `p_reverse <= 0.05`, otherwise **neither**.
Forward and reverse p-values are each adjusted within their own family.

**Multiplicity control.**

| stage | family | correction |
|---|---|---|
| fresh, confirmatory | (200 configs) x {`truth_kl`, `sampled_log_loss`} = 400 tests | Holm, alpha 0.05 |
| fresh, descriptive | every (config, other metric) pair with data | Benjamini-Hochberg, q 0.05 |
| discovery, exploratory | every (config, stored metric) pair; stored metrics are `truth_kl`, `sampled_log_loss`, `cal_slope_dev`, `centre_auc`, `sampled_auc`, `rank_corr`, `uncertainty_auroc` | Benjamini-Hochberg, q 0.05 |

**Pre-stated summary tables.** (1) `graph family x metric`, each cell the counts of configs
`better / worse / neither` under the rule above, one table for discovery (exploratory, stored
metrics) and one for fresh (confirmatory metrics under Holm, others under BH, marked); (2) the same
table with the median standardised effect `dz` of the less favourable comparator; (3) the
config x metric matrix of `dz` in full; (4) the pinned-kappa comparator set repeated as
descriptive. Counts carry no inferential weight beyond the tests that produced them; a
confirmatory claim "TNBBeta is better on metric `m` in family `f`" requires the family-level
statement: the median `E_c` of the family's fresh configs against the less favourable comparator has a
95% bootstrap interval (over configs, 10,000 resamples) above 0, only evaluated for `truth_kl` and
`sampled_log_loss`.

**What counts as the metric answer.** Metrics on which TNBBeta "succeeds" are those whose fresh
`better` count exceeds the `worse` count in at least one family (descriptive), and, confirmatory, the
two metrics above under the family-level statement. The write-up must report the metrics on which it
fails with the same prominence as the ones on which it succeeds.

## Covariates (fixed before any outcome-covariate relationship is examined)

Each covariate is computed **per draw** and averaged over a config's draws (K = 32 discovery; all draws
used for that config), from the regenerated graph (`sample_graph(spec, seed)` is deterministic from the
seed and spec; no model needed). `A` = full sampled adjacency, `P` = the draw's true probability matrix,
`n` nodes, `U` = upper-triangle pairs, `rho = sum_U P / |U|`, `t = P / (P + rho)`.

### Set G (graph-side, available before any model is trained; the only set the prediction rule may use)

| id | covariate | exact definition |
|---|---|---|
| g1 | `log_n` | log `num_nodes` |
| g2 | `log_nominal_degree` | log of the config's `mean_degree` |
| g3 | `log_groups` | log `max(2, num_groups)` for families that use it (all but `bipartite`, `scale_free`, `ring`); 0 otherwise; with indicator g3b `uses_groups` |
| g4 | `hetero_level` | 0/1/2 for none/mild/heavy as in `GraphSpec`; `scale_free` and `degree_corrected` carry their own weights, see g8 |
| g5 | `informative_features` | 1 if features informative, else 0 |
| g6 | `oracle_auc` | `achieved_auc` of the config |
| g7 | `realised_degree` | mean degree of `A` |
| g8 | `expected_degree_cv` | `sd / mean` of `d_i = sum_j P_ij` (degree dispersion without sampling noise) |
| g9 | `clustering` | global clustering of `A`: 3 x triangles / connected triples |
| g10 | `assortativity` | Newman degree assortativity of `A` |
| g11 | `spectral_gap` | `1 - lambda_2` of `D^{-1/2} P D^{-1/2}`, `D = diag(P 1)` |
| g12 | `affinity_erank` | effective rank of `P` (zero diagonal): `exp(-sum s_i log s_i)`, `s_i = sigma_i / sum sigma` |
| g13 | `logit_rank16` | `1 - ||L - L_16||_F^2 / ||L - mean(L)||_F^2` over off-diagonal entries, `L = logit(t)` clamped at 1e-6, diagonal filled with the off-diagonal mean, `L_16` the best rank-16 symmetric approximation (16 = `DIM`): how much of the target logit a 16-dimensional dot-product decoder can represent |
| g14 | `high_prob_fraction` | `|{(i,j) in U: t_ij > 0.5}| / |U|` (equivalently `P > rho`) |
| g15 | `mass_top1pct` | share of `sum_U P` carried by the top 1% of pairs |
| g16 | `clip_fraction` | share of `U` with `P = 1` |
| g17 | `log_kl0` | log of `KL0` (headroom of the constant predictor, defined above) |
| g18 | `leaf_fraction` | share of nodes with training degree <= 1 (the encoder sees them only through features); training graph from `split_edges(A, seed)` |
| g19 | `giant_fraction` | share of nodes in the largest connected component of the training graph |

Excluded with reasons: the structure `contrast` (family-specific meaning, not comparable across
families); `family` identity (not usable under leave-family-out CV; used only in the secondary analysis
below); `density` (collinear with g1, g2: `d / (n - 1)`).

Collinearity among G is expected (g2/g7/g15, g11/g12). The elastic net handles it for prediction;
for *interpretation* the report gives the correlation matrix of G and groups covariates with
|Spearman| > 0.8 into clusters, reporting a cluster as the unit of any claim.

### Set M (model-side, per run, explanatory only; post-treatment, so never in the prediction rule)

Computed from the stored run JSONs (config mean over draws): TNBBeta `post_p_mean`, `post_q_mean`,
`post_epsilon_mean`, `post_m_mean` (`m = epsilon - (latent_dim - 1)/2`), `post_p_std`,
`post_epsilon_std`, `post_frac_bimodal`, `post_r_bar`, `second_mode_mass`, `frac_nodes_second_mode`,
`best_epoch`, divergence indicator; the vMF and PS `post_r_bar` (kappa proxy; kappa itself is **not**
stored in the discovery runs, and the fresh stage will log mean log kappa and the learned temperature and
bias). These variables tell us what TNBBeta did on a graph, not what the graph is, and they are
mediators or outcomes of training. They are used only to *describe* winning configs (Analysis step 5),
never to claim a graph property causes a win.

### Pre-declared hypothesis (one, directional)

H1: within family, higher mean degree goes with a larger `S_c` (earlier lead: confirmed log-loss wins had
mean degree 20.5 against 15.2 over all 120 configs; pooled Spearman +0.25 to +0.28, within-family
+0.15 to +0.20, not significant). Tested on the fresh configs only, as a within-family permutation test
of the Spearman correlation between `log_nominal_degree` and `S_c`, one-sided, alpha = 0.05, a single test
so no correction. All other covariate findings are exploratory.

## Analysis plan

### Step 0 (before any covariate is related to any outcome)

Compute and archive: `E_c`, `S_c`, intervals, reliability ceiling, covariate table G and M with their
marginal distributions and correlation matrix, per-family spread of `E_c` (marginals only). Freeze the
covariate table's hash in this file.

### Step 1 (exploratory, discovery configs, outcome `S_c`): predictability

- **Model class:** elastic net (`l1_ratio = 0.5`), covariates standardised (log-transformed ones as
  listed; g8, g11, g12, g17 are right-skewed and enter after log), outcome `S_c` unweighted
  (weighted-by-inverse-variance as sensitivity).
- **Penalty `lambda`:** chosen by an inner leave-one-family-out CV over the training families, minimising
  MSE with the one-standard-error rule.
- **Primary summary:** outer **leave-one-family-out** cross-validated `R^2_LOFO = 1 - SSE_model / SSE_base`,
  where the baseline predicts each held-out config by the **mean of the training families' configs**
  (a family-mean baseline is unavailable for an unseen family by construction). This answers "what do
  graphs on which TNBBeta wins have in common *across graph types*".
- **Null:** family-block permutation. Every discovery family has 12 configs, so reassign each family's 12
  outcomes (random order within the block) to another family's 12 covariate rows, 2,000 draws of a
  random permutation of the 10 blocks, recompute `R^2_LOFO`. This keeps both the family clustering of
  covariates and of outcomes, so it is the honest null for a design whose between-family effective sample
  is 10. A plain permutation of configs is also reported, labelled optimistic.
  `R^2_LOFO` is declared nonzero when its block-permutation p <= 0.05.
- **Secondary summaries:** (a) within-family increment: repeated (20x) 4-fold CV over configs stratified
  by family, covariates plus family dummies against family dummies only; (b) (dropped: the
  gradient-boosted secondary model of the draft, see Amendments); (c) `E_c` in nats as the outcome; (d) excluding `bipartite` (the control
  family whose structure a dot-product decoder cannot represent) and excluding divergence-affected
  configs.
- Report `R^2_LOFO` next to the reliability ceiling, so a low value is read against how much is knowable.

### Step 2: selection and stability

Fit the elastic net on all 120 configs (`lambda` from CV as above). Report nonzero coefficients; a
covariate cluster is **selected** only if it is nonzero in at least 80% of the 10 leave-one-family-out
fits with a constant sign.

### Step 3: partial dependence

For the (up to) three selected clusters with largest standardised coefficient, the elastic net's partial
dependence (linear: the slope per standard deviation with a bootstrap-over-configs interval) and a
nonparametric check, the mean of the partial residual (outcome minus the other selected covariates'
linear contribution) within covariate tertiles; report whether the tertile pattern agrees in sign.

### Step 4: descriptive characterisation of the wins

Tabulate G medians and Set M medians by the `win` / `inconclusive` / `loss` tiers and by family, plus
the descriptive metrics above for each tier. This produces hypotheses; it is not tested.

### Step 5: model-side description

Relate `S_c` to Set M within the tiers, and report how M varies with G. Mediation language
("because TNBBeta's posterior is diffuse") is forbidden in the write-up; only "co-occurs with".

### Exploratory vs confirmatory

| item | status |
|---|---|
| Step 0, Steps 1-5 on the 120 discovery configs | exploratory |
| Any covariate selected, any partial-dependence shape | exploratory |
| H1 (mean degree, within family) on fresh configs | confirmatory |
| Fresh-config out-of-sample performance of the frozen prediction rule | confirmatory |
| All descriptive metrics (Hits@K, MRR, graph statistics, latent recovery, ECE) | descriptive on every stage |

## Fresh-config confirmation

### Prediction rule (frozen before the fresh stage runs)

The rule is the Step 2 elastic net on **Set G only**, refit on all 120 discovery configs, with the
`S_c` outcome, the standardisation constants, `lambda` and coefficients written to
`graph_search/metric_first/prediction_rule.json`. Because the rule is a function of discovery outcomes it
cannot exist at freeze time: its sha256 is appended to the Deviations section ("Records") as soon as
Step 2 produces it and **before any fresh-stage outcome is read**; fresh-stage training and the fresh
covariate computation (neither reads any outcome-covariate relation) may run before or after. Its prediction for a fresh
config is a function of the fresh config's G covariates (computed from regenerated graphs before any
training). The 34 confirmation-stage configs are not fresh: they are the same configs as discovery with
new draws; their new draws may be pooled into `E_c` for the discovery fit (more draws, same configs),
which this design does not do (confirmed choice, simpler and keeps the discovery data identical to what was
analysed before).

### Fresh stage

- **Configs:** newly sampled, `uv run python -m apps.synthetic.graph_search configs --name fresh
  --per-family N_c --seed 13` (name and seed not used before; pilot used `pilot`/7, discovery
  `discovery`/11). Verified before running that no fresh spec equals a discovery or pilot spec.
- **Seeds:** a new stage `fresh` with `SEED_BASE = 50_000_000` (existing: pilot 9M, discovery 10M,
  confirmation 20M, kappa_grid 30M, axis_diagnostic 40M; seeds are `base + 1000 * config index + draw`
  and the config index stays under 1,000, so the ranges are disjoint).
- **Learning rates:** the frozen pilot-tuned rates, unchanged (pilot set was disjoint).
- **Conditions:** `tnbbeta`, `vmf_init`, `power_spherical_init` (primary), plus the pinned pair
  (descriptive); `gaussian` is dropped (a reference only). Every run also computes the extended
  metrics (ECE, hard-negative Hits@K/MRR, graph statistics, latent recovery, `truth_kl_w`, the clamp and
  decile variants, `recal_*`, kappa, temperature, bias) and saves the best-validation state to disk (new
  behaviour, see audit) so that later metrics need no retraining.
- **Size (set by the power simulation below): 20 configs per family (200 configs), K = 16 draws
  per config, 5 conditions: 16,000 runs.**
  - `configs --name fresh --per-family 20 --seed 13`; manifest `--stage fresh --draws 16 --conditions
    tnbbeta vmf_init power_spherical_init vmf_pin power_spherical_pin`.

### Fresh-stage size, from the power simulation

Run on 2026-11-02 before freezing, from marginals of the stored discovery runs only
(`discovery/config_aggregates.json`: per config and comparator, the mean, per-draw SD and n of the paired
`truth_kl` difference; no covariate, no outcome-covariate relation). `E_c` in nats is used (`S_c` needs
the covariate job). Findings: the reliability of the config effect `E_c` (1 - mean SE^2 / var(E_c)) is
0.92 over all configs and 0.92 within family (variance between family means is only 11% of the total), the
median SE of `E_c` at K = 32 is 0.0115 nats against a between-config SD of 0.10, so **draws per config
matter little beyond K = 16; the number of configs is what limits power**. Simulation (300
replications; true effects drawn from the empirically deflated distribution of `E_c`, noise SE scaled
by `sqrt(32 / K)`; the rule's prediction correlated with the true effect at `rho_true`; fresh
interval by a stratified bootstrap in the simulation that resamples whole families, which is more
conservative than the pre-registered within-family resampling):

| K | configs per family | rho_true | P(rho_fresh >= 0.30) | P(lower 95% bound > 0) | P(R2_fresh > 0) | P(all three) |
|---|---|---|---|---|---|---|
| 16 | 12 | 0.3 / 0.5 / 0.7 | 0.21 / 0.80 / 1.00 | 0.77 / 1.00 / 1.00 | 0.92 / 1.00 / 1.00 | 0.21 / 0.80 / 1.00 |
| 16 | 20 | 0.3 / 0.5 / 0.7 | 0.14 / 0.85 / 1.00 | 0.97 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.14 / 0.85 / 1.00 |
| 16 | 30 | 0.3 / 0.5 / 0.7 | 0.07 / 0.87 / 1.00 | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.07 / 0.87 / 1.00 |
| 32 | 12 | 0.3 / 0.5 / 0.7 | 0.20 / 0.82 / 1.00 | 0.80 / 1.00 / 1.00 | 0.95 / 1.00 / 1.00 | 0.20 / 0.82 / 1.00 |
| 32 | 20 | 0.3 / 0.5 / 0.7 | 0.12 / 0.88 / 1.00 | 0.93 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 0.12 / 0.88 / 1.00 |

H1 (within-family Spearman of mean degree with the effect, one-sided 0.05, analytic Fisher-z with the
within-family reliability, 0.84 at K = 16): power at true within-family rho = 0.15 / 0.20 / 0.30 is
0.42 / 0.61 / 0.90 with 12 configs per family, **0.60 / 0.82 / 0.99 with 20**, 0.76 / 0.93 / 1.00 with 30.

Reading: with 12 configs per family the lower-bound criterion has only 77-80% power at `rho_true = 0.3`
and H1 only 61% at the earlier lead's strength (0.2), so 12 is underpowered for the decision rule.
**20 per family at K = 16 reaches 97% for the lower bound at 0.3 and 82% for H1 at 0.2.** The point
estimate criterion (`rho_fresh >= 0.30`) is a deliberate effect-size floor, not a significance test: it
passes with 85% probability when the true correlation is 0.5 and is expected to fail when it is 0.3, and
no sample size changes that. 30 per family would add little (0.93 for H1 at 0.2) for 50% more compute,
so 20 is chosen. Cost: 16,000 runs at about 17 s (legacy) plus the extended evaluation, about
25-30 GPU-task-hours at 6 processes per GPU; it queues behind the dimension-scaling array in the shared
24-running quota.

### Pre-specified evaluation on the fresh stage

1. `R^2_fresh = 1 - SSE_rule / SSE_base`, baseline = mean of the 120 discovery `S_c`; plus Spearman
   `rho_fresh` between predicted and observed `S_c`. 95% interval by cluster bootstrap over fresh
   configs (families as strata, 10,000 resamples).
2. Within-fresh-family correlation: Spearman between predicted and observed `S_c` after removing
   fresh-family means from both.
3. H1 as above.
4. Descriptive: all metrics and the tier tables, on the fresh data, with the same definitions.

### Decision rule for "a replicable pattern" (all must hold)

(1) Discovery `R^2_LOFO > 0` with block-permutation p <= 0.05 and exceeding 25% of the reliability
ceiling; (2) at least one covariate cluster selected in at least 80% of leave-family-out fits with a
constant sign; (3) on the fresh stage, `rho_fresh` has a lower 95% bound above 0 **and** the point
estimate is at least 0.30, **and** `R^2_fresh > 0`; (4) the sign of the Spearman correlation, on the fresh configs, between the top selected cluster's
representative covariate and `S_c` matches the sign of its elastic-net coefficient. If (1) or (2) fail, the fresh stage is reported as a descriptive
replication of the effect distribution and no pattern is claimed. If (1)-(2) hold and (3) fails, the
claim is "predictable in discovery, not replicated". We commit to publishing the outcome whichever way it
falls.

### Order of operations

1. Freeze this file. 2. Implement and test (uncommitted; code listed in the results note).
3. Fresh configs, manifest, fresh training, and covariates (discovery and fresh) on SLURM.
4. Per-draw run cache for the discovery stage (SLURM, thread pool). 5. Step 0 and the exploratory
analysis (Steps 1-5) on discovery; writes `prediction_rule.json`; its hash is appended below.
6. Only then the fresh-stage analysis. Any departure from this is a Deviation.

### Compute (SLURM)

`scripts/slurm/graph_search.sbatch` as is, `STAGE=fresh sbatch --array=0-23` (6 processes per GPU, the
24-running / 200-submitted `general` QOS limit; one array job, no per-cell tasks). Covariate computation
(eigendecompositions of ~1,200 x 1,200 matrices, regenerated graphs, 120 + 120 configs x 32 / 16 draws)
is CPU only and goes through SLURM as a single job (thread pool), not the login node.

## Data availability audit

Facts established by reading `graph_search.py` and the run JSONs (discovery: 30,720 runs; confirmation:
3,264; stored in `$TNBBETA_CHECKPOINT_DIR/graph_search/<stage>/runs/<run id>.json`). Each JSON holds
`metrics`, `val_auc`, `best_epoch`, `diverged`, `seconds`, `seed`, `lr`, `condition`, `config_id`, `draw`,
`family`, `num_nodes`.

| item | in stored JSONs? | to obtain |
|---|---|---|
| `truth_kl`, `sampled_log_loss`, `centre_auc`, `sampled_auc`, `cal_slope`, `cal_slope_dev`, `cal_intercept`, `uncertainty_auroc`, `rank_corr` | yes, all 8 conditions, 32 draws (discovery) | stored |
| TNBBeta `post_p/q/epsilon/m` mean and std, `post_frac_bimodal`, `post_r_bar`, `post_entropy_mean`, `second_mode_mass`, `frac_nodes_second_mode` | yes (non-TNBBeta: `r_bar`, entropy only) | stored |
| `diverged`, `best_epoch`, `val_auc`, `seconds` | yes | stored |
| vMF / PS kappa, learned temperature and bias | **no** | retrain |
| `recal_log_loss`, `recal_truth_kl` | **no** (the code computes them now; discovery runs pre-date that) | retrain |
| `axis_abs_cos*` | no | retrain |
| ECE, Hits@K, MRR, truth_kl by decile, `truth_kl_w`, eps = 1e-4 variant | no (ECE exists in `predictive.py` but is not called by `evaluate`) | retrain |
| posterior-predictive graph statistics, latent recovery | no | retrain |
| graph draws, edge splits, true `P`, features | not stored, but **reproducible**: `sample_graph(spec, seed)` and `split_edges(A, seed)` are deterministic from the manifest's `seed` and the config (`configs.json` stored) | regenerate (CPU) |
| model weights | **no**: `run_item` passes no `run_name`, so `run_once` does not write a checkpoint (`best_state` is only held in memory for `evaluate`) | retrain |
| per-pair predictions | no | retrain |

Consequences.

1. **Without retraining**: `truth_kl`, `sampled_log_loss`, calibration, the TNBBeta parameter set M, all
   of Set G (graphs regenerate from seeds), the reliability ceiling, `E_c`, `S_c`, and Steps 0-5 on
   the 120 discovery configs are all computable from what is on disk. That is enough for the whole
   exploratory stage.
2. **Retraining is required for every new descriptive metric** (Hits@K, MRR, ECE, graph statistics,
   latent recovery, `truth_kl_w`, `recal_*`, kappa). Training is seeded but run on GPU, so a rerun will
   not reproduce the stored weights bit for bit: new-metric values are not paired with stored ones.
   Rerun cost (from sampled run times of 9-53 s per run with 6 processes per GPU, median about 15 s,
   mean about 17 s; 24 tasks): the primary 3 conditions on discovery = 120 x 32 x 3 = 11,520 runs
   ~ 55 process-hours ~ 9-10 GPU-task-hours, ~25 min of wall clock on 24 tasks plus queue; adding the
   pinned pair 7,680 runs ~ +6 GPU-task-hours. The full original 8-condition discovery was 30,720 runs
   (~25 GPU-task-hours). The fresh stage (16,000 runs, see the power simulation) ~ 25-30 GPU-task-hours. **[CHOICE]** Recommendation: do
   *not* rerun discovery. Compute the new descriptive metrics on the fresh stage only (they come out of
   its first run, since evaluation is online), and save fresh checkpoints so anything further needs
   no retraining. If the discovery descriptive tables are wanted too, add the 11,520-run rerun (with a
   new `evaluate` and saved checkpoints); the stored `truth_kl` stays the discovery analysis value and the
   rerun's `truth_kl` is reported only as a reproducibility check.
3. Implementation notes (code not yet written): a new evaluation function in a new module (the legacy
   `evaluate` and stored schema stay untouched, per the "reproduce prior behaviour exactly" convention),
   which reuses `dense_edge_probabilities`, `balanced_truth` and `bernoulli_kl` so `truth_kl` is
   byte-identical, and a `run_name`-style checkpoint save of `best_state`.

## Risks and known weaknesses

- **Forking paths in the covariate search.** 19 graph covariates, 120 configs in 10 families
  (effective sample for between-family structure is 10): any selection is fragile. Mitigations are
  family-grouped CV, a block-permutation null, an 80% stability criterion, selected *clusters* instead of
  single variables, and a fresh confirmation; they reduce but cannot remove it. Many G covariates
  (`assortativity`, `clustering`, `erank`) are essentially family identifiers, so "commonality" across
  families may in effect be a re-description of "which family".
- **Replicates are graph draws, not configs.** The per-config interval is over 32 draws of one config;
  conclusions about graph types rest on the 120 configs, nested in 10 families. The bootstrap and
  permutation schemes above resample at the config / family level where it matters. The Wilcoxon tests
  of the earlier pre-registration are not used here.
- **`truth_kl` clamping.** `eps = 1e-6` on both arguments caps a pair's penalty near 13.8 nats for a
  saturated confident error and caps tiny truths at 1e-6 (relevant where `P` underflows, e.g.
  geometric with strong contrast). The `eps = 1e-4` variant and the decile decomposition are reported for
  robustness (fresh stage).
- **`balanced_truth` assumes 1:1 negative sampling teaches `P / (P + rho)`.** True only at the model's
  optimum with negatives uniform over all pairs, and the decoder bias is learned, so a systematic
  bias-vs-slope difference between families shifts `truth_kl` without being a "shape" effect. The
  `recal_truth_kl` and calibration-slope descriptives address temperature but not a changed target.
- **No real-graph analogue of `truth_kl`.** On real graphs only held-out log loss exists. `truth_kl_w`
  makes the link explicit (expected log loss = irreducible term + pair-weighted truth KL up to the
  `(1 - P)` factor and the training-negative mismatch), but real-graph generators are unknown, so a
  covariate-effect map found here does not transfer without checking.
- **Comparators' learning rates were tuned on `sampled_log_loss` on 20 pilot configs x 4 draws**, a
  3-point grid per family, shared by every config of a family. A comparator that is under-tuned for a
  given config (not for the family average) inflates TNBBeta's effect on that config, and
  `truth_kl` and log loss are strongly rank-correlated across configs (rho 0.91 earlier), so the twin
  adds little independent evidence about tuning. The pinned set (descriptive) is the control for
  kappa optimisation, not for learning-rate mis-tuning.
- **`E_c` takes the minimum over comparators,** a downward-biased estimator of the true minimum effect
  (selection among noisy means): this is conservative for a win and anti-conservative for describing a
  "loss"; the bootstrap includes the selection but is not exact.
- **Metric chosen with partial knowledge.** The user had seen the truth-KL and sampled-AUC counts before
  specifying the design; the primary is justified on principle (all-pairs, proper divergence, posterior-
  aware) and the metric-first order is respected for the outcome relationships, which have not been
  looked at, but the discovery stage cannot be called blind.
- **`second_mode_mass` is not zero in the discovery runs.** The background for this task says it is 0 in
  these graph runs. In the stored discovery JSONs the config-level TNBBeta `second_mode_mass` has median
  0.07 and 90th percentile 0.43, and exceeds 0.01 in 94 of 120 configs (the 2026-10-31 results table
  reports 0.31 on bipartite, 0.23 on mixed-membership, 0.20 degree-corrected). The "single cap" statement
  (CLAUDE.md, graph-families screening) appears to hold for geometric / core-periphery / hierarchical
  configs and for the earlier screening presets, not for every discovery config. This matters because
  the claim "TNBBeta's posterior is a cap, so the metric need not be posterior-aware" is not
  uniformly true; it is a reason the posterior-aware metric is the right primary, and Set M's
  `second_mode_mass` is carried forward.
- **Truth KL has a large per-draw SD in discovery** (median 0.073, 90th percentile 0.27, against the
  pilot's 0.034), mostly from poorly learned or diverging configs; low-reliability configs add noise to
  the regression outcome. The reliability ceiling quantifies this; configs with a TNBBeta divergence
  rate above 10% are reported separately (sensitivity (d)).
- **Descriptive ranking metrics on small graphs.** Hits@K and MRR with 100 negatives per positive are
  noisy at the smallest graphs (n = 300, mean degree 4, about 60 test positives); they are descriptive.
- **Run-time drift.** The fresh stage uses the same code path and tuned rates as discovery; if the base
  code (`run_once`, `evaluate`) changes between now and then, a bridging check (rerun a handful of
  discovery configs, compare `truth_kl`) is part of freezing.

## Deviations

(None yet. Appended below, if any: the sha256 of `prediction_rule.json` once produced, and any departure from the text above with its reason.)

### Records (appended)

- 2026-11-02, freeze: the text above this line has sha256 `71facbf012f049c5020bcbe6a09e300912e20a3e49fc398f9323590dfd8a0875`; base commit `6b1880f`; frozen before any fresh-stage training and before any outcome-covariate analysis.
- 2026-11-02, after the exploratory discovery analysis (SLURM job 2297912; no fresh-stage outcome read yet): `prediction_rule.json` has sha256 `cba8430faa2e5e344aa6a28772dc34cc896b7f3a4c2dcd16084c1992e17cd6b5`. **All its coefficients are zero** (the one-standard-error rule chose penalty 2.02, the leave-one-family-out R^2 is exactly 0.000 with block-permutation p = 0.998), so the rule predicts one constant for every fresh config. Consequences, as pre-registered: decision-rule criterion (1) fails, so the fresh stage is reported as a descriptive replication of the effect distribution and no covariate pattern is claimed; `rho_fresh` is undefined for a constant prediction and criterion (3) is not met.

### Deviations (appended)

1. **Implementation constants not fixed at freeze:** the elastic-net penalty grid has 12 log-spaced values from the smallest all-zero penalty down by a factor 10^2.5, coordinate descent uses the Gram-matrix update with tolerance 1e-6 and at most 200 sweeps per penalty (chosen to make 3,000 leave-family-out refits affordable; an earlier 15-point grid to 1e-3 with 500 sweeps took 35 s per refit).
2. **Fresh training array:** tasks 2-23 of the first array (job 2297873) failed within seconds on a node problem (`configs.json` not readable); they were resubmitted as job 2298193 (`--array=2-23`, `TOTAL_TASKS=24` so the shard mapping is unchanged, `--exclude=k003`). Finished runs are skipped, so nothing is duplicated.
3. **Post hoc supplement (not pre-registered), added after seeing the discovery result and before any fresh outcome was read:** because the pre-registered rule is degenerate, the fresh report also scores "discovery family means plus a within-family elastic net on family-centred covariates" on the fresh configs (they share the ten families). It is labelled post hoc in the results and carries no confirmatory weight.
4. **Degenerate deciles:** `truth_kl_dec*` split the KL by deciles of the balanced truth; for block-structured families the truth takes only a few distinct values, so most deciles are empty (nan). The columns are kept and reported as such; they are descriptive.
5. **CPU training for most fresh runs.** The GPU array stayed pending behind the shared quota for hours (estimated start the next morning). The first 1,344 fresh runs (the largest graphs: all runs of tasks 0 and 1, job 2297873, GPU) were trained on GPU; the remaining 14,656 on CPU with `scripts/slurm/graph_search_cpu.sbatch` (job 2298489, `--device cpu`, 4 threads per process). A fresh (config, draw) therefore sits almost always wholly on one device, but device numerics differ from the GPU-trained discovery stage; a device effect on paired differences is not tested. The CPU job ran about 30 s per large run, as fast as the GPU.
6. **Fresh-stage analysis run on the login node.** Its SLURM job (2298870) stayed pending on `QOSMaxJobsPerUserLimit` (the 24-running quota was held by two other studies' arrays); it was cancelled and the single-core cache build and analysis (about 15 minutes, niced, `OMP_NUM_THREADS=1`) ran on the login node. Results are identical to what the SLURM job would produce (same code, fixed seeds).
7. **Caveat recorded (risks).** The library `VonMisesFisher` rejection sampler was reported biased for kappa about 10-40 at manifold dimension m >= 9 (dimension-scaling study, its Deviation D2). The graph runs here use ambient dimension 16 with vMF kappa starting at and pinned near 16, i.e. inside that regime, so the vMF comparators (`vmf_init`, `vmf_pin`) may have biased posterior samples (in training, evaluation and `truth_kl`). The power-spherical comparators are not affected. Not edited in the library; conclusions against vMF should be read with this caveat, and the intersection-union effect (the less favourable of the two comparators) is not protected by it when vMF is the less favourable one.
