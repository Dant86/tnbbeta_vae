# Real-graph revision: results (Cora, Citeseer, Pubmed, MAG-CS), 2026-11-03

- **Date:** 2026-11-03. Pre-registration (frozen before any run): `real_graph_revision_preregistration_2026-11-03.md` (deviations appended there). Full generated tables: `real_graph_revision_tables_2026-11-03.md`.
- **Code version:** base `master` commit `d5bb583` (PR #72), branch `real-graph-revision`, **uncommitted working tree** (new files only: `apps/real_graph/{run,stats,analyze,figures}.py`, `src/tnbbeta_vae/models/real_graph_eval.py`, `scripts/slurm/real_graph{,_aux}.sbatch`, `tests/apps/test_real_graph{,_analysis}.py`; no shared file was edited).
- **Commands:**
  ```
  uv run python -m apps.real_graph.run manifest --stage tune
  STAGE=tune TOTAL_TASKS=6 sbatch --exclude=k003 --array=0-5 scripts/slurm/real_graph.sbatch   # job 2305622 (tasks 0-2), tasks 3-5 resubmitted as 2306197
  uv run python -m apps.real_graph.run select                                                  # freezes tuned.json
  for s in main thin pin community; do uv run python -m apps.real_graph.run manifest --stage $s; done
  # sbatch jobs 2306502 (main), 2306503 (thin), 2306504 (pin), 2306505 (community), --exclude=k003
  AUX=analyze sbatch scripts/slurm/real_graph_aux.sbatch                                       # job 2307498 (final analysis)
  uv run python -m apps.real_graph.figures --fig-dir writeup/results/figures                   # login node
  AUX=full sbatch scripts/slurm/real_graph_aux.sbatch                                          # job 2306510: 1191 passed, 1 skipped
  ```
  Run records: `$TNBBETA_CHECKPOINT_DIR/real_graph/{tune,main,thin,pin,community}/`, analysis in `.../real_graph/analysis.json`. Node k003 has broken CUDA and was excluded. Run counts: tune 1152, main 320, thin 960, pin 400, community 800; no run failed or diverged-without-state.
- **Figures** (`writeup/results/figures/`): `real_graph_revision_main_bars.png`, `_degree_buckets.png`, `_thinning.png`, `_m_pinning.png`, `_community.png`.

## What was done (short; the design is the pre-registration)

Families: TNBBeta (free), vMF and PowerSpherical with kappa initialised at the ambient dimension and trained, Gaussian reference; all with `decoder_bias=True`, the exact vMF sampler (each vMF run asserts and records `vmf_sampler = exact_envelope`; all 80 main vMF runs do). Per (dataset, features, family) the configuration (lr in {0.003, 0.01, 0.03} x dropout {0, 0.4} x d in {16, 32}) was chosen by mean validation sampled log loss on 3 separate tuning seeds, then 10 final seeds (0-9) paired by seed. Primary metric: sampled test log loss (32 posterior samples) on the balanced test pairs. Truth KL does not exist on real graphs; the paired log-loss difference is its analogue (E[log loss] = entropy + KL to the best predictor for those pairs; the entropy cancels in a paired difference), with the caveats in the pre-registration. Advantage is positive when TNBBeta is better; the primary comparator is the better of vMF/PowerSpherical by cell mean (never looking at TNBBeta).

Selected configurations sit at the edge of the grid often (dropout 0.4 in 27 of 32 cells, lr 0.03 in 17, d = 32 in 25; the smallest lr, 0.003, never): the grids were not wide enough to bound the optimum, for any family. The full selection table is in `tuned.json`.

## 1. Main bars (P1 and P2): TNBBeta does not beat tuned vMF/PowerSpherical on log loss in 7 of 8 cells

Sampled test log loss, mean over 10 seeds (95% bootstrap intervals are in the tables file):

| cell | TNBBeta | vMF | PowerSpherical | Gaussian |
|---|---|---|---|---|
| Cora, features on | 0.327 | 0.316 | 0.314 | 0.371 |
| Cora, off | 0.486 | 0.486 | 0.472 | 0.473 |
| Citeseer, on | 0.339 | 0.319 | 0.312 | 0.398 |
| Citeseer, off | 0.549 | 0.534 | 0.553 | 0.535 |
| Pubmed, on | 0.249 | 0.255 | 0.261 | **0.210** |
| Pubmed, off | 0.496 | 0.482 | 0.482 | **0.448** |
| MAG-CS, on | 0.263 | 0.237 | 0.230 | 0.276 |
| MAG-CS, off | 0.293 | 0.278 | 0.290 | 0.283 |

P1 (TNBBeta minus primary comparator, exact two-sided sign-flip, Holm over 8): advantage (positive = TNBBeta better) Cora on -0.013 (Holm 0.066, not significant), Cora off -0.013 (Holm 0.029, significant loss, below the 0.05-nat margin), Citeseer on -0.027 (0.020), Citeseer off -0.015 (0.016), Pubmed on **+0.006** [0.001, 0.011] (p = 0.051, Holm 0.066, not significant), Pubmed off -0.014 (0.016), MAG-CS on -0.033 (0.020), MAG-CS off -0.016 (0.016). **Confirmed wins: 0. Confirmed losses (beyond the 0.05-nat margin): 0. Significant losses below margin: 6. Not significant: 2 (Cora on, Pubmed on).** The smallest attainable two-sided p with 10 seeds is 2/1024 = 0.002 (checked against an independent enumeration in the tests).

Rankings (P2, BH over 32 tests, TNBBeta minus primary comparator): sampled AUROC is lower for TNBBeta in 7 of 8 cells (Cora on -0.0068, Cora off -0.0086, Citeseer on -0.0134, Citeseer off -0.0089, MAG-CS on -0.0053, MAG-CS off -0.0043, Pubmed off -0.0013; BH-significant: Cora on, Citeseer on, MAG-CS off) and higher in Pubmed on (+0.0025, BH 0.0125, significant). Centre AUROC is lower in 7 of 8 cells (BH-significant in five: Cora off -0.019, Citeseer on and off, Pubmed off -0.018, MAG-CS off; Cora on at BH = 0.050) and higher in Pubmed on (+0.0058, significant). ECE: TNBBeta is better calibrated by this binned measure than the comparator nowhere significantly; it is significantly worse in Citeseer off (-0.038), Pubmed off (-0.016) and MAG-CS on (-0.035). Against the **Gaussian**, TNBBeta has the lower log loss in Cora on (+0.043), Citeseer on (+0.059) and MAG-CS on (+0.012, p = 0.08) and the higher in Pubmed on and off, Cora off, Citeseer off, MAG-CS off (descriptive; Gaussian is not a primary comparator).

Sensitivity to the comparator (descriptive): against vMF alone TNBBeta is tied in Cora off (+0.0002) and slightly ahead in Pubmed on (+0.006, p = 0.051); against PowerSpherical alone it is ahead in Pubmed on (+0.012, p = 0.002) and tied in Citeseer off (+0.004, p = 0.40).

## 2. Degree buckets: no uniform shift, and the pattern differs by graph

Buckets by the lower endpoint's training degree (0-1, 2-3, 4-7, 8-15, 16+); label-balanced per-bucket log-loss advantage with seed-level intervals (the edge-level intervals in the tables file are narrower and anti-conservative; edges share endpoints). Buckets above degree 7 have fewer than 20 negatives in Cora and Citeseer and are not reported there (degree-gradient contrast G is undefined for Citeseer; BH over the 6 defined cells for each outcome).

* Pubmed on: advantage grows with degree, from -0.024 [-0.031, -0.017] (0-1) to +0.020 (2-3), +0.026 (4-7), +0.033 [0.017, 0.048] (8-15); G (min degree >= 4 minus <= 3) = +0.038 (BH 0.004). This is the one cell where the data look like "TNBBeta helps more where the graph is denser around the node", and also the one cell where TNBBeta ties or edges the comparators overall. TNBBeta's AUROC advantage is +0.006 to +0.008 in buckets >= 2-3 (0.000 in 0-1).
* MAG-CS on: the opposite: +0.073 [0.056, 0.090] in the lowest-degree bucket, then -0.012, -0.036, -0.046, -0.045 (G = -0.061, BH 0.004).
* MAG-CS off: -0.266 [-0.303, -0.229] in 0-1 and -0.052 in 2-3, then +0.007 (4-7), +0.016 (8-15), -0.004 (16+); G = +0.135 (BH 0.004). Pubmed off: negative in all buckets (-0.037 to -0.011), G = +0.011 (BH 0.19). Cora off G = -0.034 (BH 0.004); Cora on G = +0.011 (BH 0.53).
* **The Oct-30 note's roughly uniform AUC shift across buckets does not hold here**: the AUROC gradient G is BH-significant in 5 of 6 defined cells with varying sign (Cora on +0.0095, Pubmed on +0.0047, Pubmed off -0.022, MAG-CS on -0.005, MAG-CS off +0.0096).
* Log loss and AUROC buckets can disagree (Pubmed off 0-1: log loss -0.037, AUROC +0.022), and the lowest-degree bucket is where the largest swings, positive and negative, occur. Per-bucket log-loss values are label-balanced advantages, so they are not the pooled log loss.

## 3. Density by edge thinning: the pre-registered monotone trend is not there for log loss

Edges retained 0.3 / 0.5 / 0.7 / 1.0 (realised mean degrees in the tables file; e.g. MAG-CS 2.6 / 4.4 / 6.2 / 8.8, Pubmed 1.35 / 2.25 / 3.15 / 4.50; thinned graphs also have more isolated nodes, up to 46-49% at 0.3 in Citeseer and Pubmed, lower transitivity and, for the sparser cases, a higher degree CV). All thinned runs reuse the full-density tuned configuration (a limitation, below).

* Log-loss advantage versus density, combined within-dataset Spearman (4 blocks, permutation within seed rows, 100,000 draws): features on **-0.017** (one-sided p = 0.59), features off **-0.080** (p = 0.83). Per dataset (on): Cora -0.17, Citeseer -0.25, Pubmed **+0.72**, MAG-CS -0.37; (off): Cora -0.38, Citeseer -0.34, Pubmed -0.23, MAG-CS +0.63. Only Pubmed on (+0.72; OLS slope +0.020 nats per ln mean degree [0.013, 0.028]) and MAG-CS off follow the synthetic direction; Cora, Citeseer, MAG-CS on and Cora, Citeseer, Pubmed off have the advantage getting less negative as the graph gets sparser or flat. With four blocks the test is weak by construction; the point estimates, not the p-values, are the finding: **no consistent advantage-with-density effect on real graphs under thinning**.
* AUROC advantage versus density: combined Spearman **+0.247** (on, p = 0.0005, Holm 0.0009) and **+0.235** (off, p = 0.0026): TNBBeta's AUROC deficit shrinks as graphs get denser, mostly driven by MAG-CS off (+0.969; the advantage goes from -0.125 at 30% of edges to -0.004 at 100%) and Pubmed on. This is the opposite of the synthetic finding (no advantage on ranking metrics regardless of density).
* A caution on the thinned MAG-CS off and Pubmed off runs: TNBBeta's deficit there is large (log-loss advantage down to -0.109, AUROC down to -0.125 at 50% and 30% of edges) with 10 tight seeds; with the learning rate and dropout frozen at full density, these are more likely optimisation mismatches than a statement about density, and they affect every family of the same thinned graph equally only if the optima shift equally, which was not checked.
* Louvain community subgraphs (descriptive; MAG-CS 8 eligible communities, Pubmed 12; 300-1200 nodes; induced mean degree only 2.5-7.3; 5 seeds): Spearman of the log-loss advantage with mean degree is -0.07 (MAG-CS on), +0.76 (MAG-CS off), +0.20 (Pubmed on), +0.76 (Pubmed off), with 8-12 points per panel and the transitivity and degree-CV confounds in the same table (e.g. MAG-CS off: CV +0.86, size +0.69). Advantage values are mostly within +-0.05 and of both signs. No claim.

## 4. Shape and mechanism (descriptive only; no causal claim about m)

Free TNBBeta (10 seeds per cell): the fraction of nodes at `m < 0` is 0.978-1.000 in all 8 cells; the mean second-mode mass is 0.0000-0.0005 in the features-on cells and 0.0005-0.020 (Citeseer off 0.0200, Pubmed off 0.0127, Cora off 0.0062) in the features-off cells, i.e. one cap per node (fraction of nodes with more than 10% second-mode mass: up to 0.08 in Citeseer off and 0.05 in Pubmed off). The per-run Spearman of `m` with training degree is positive in all 8 cells (0.49-0.92, higher with features off), and the second-mode mass is negatively associated with degree where it is nonzero (-0.52 Citeseer off, -0.47 Pubmed off): the nodes carrying the second mode are the low-degree ones (the 0-1 bucket in the features-off cells has second-mode mass 0.007-0.069 (Cora 0.039, Citeseer 0.069, Pubmed 0.056, MAG-CS 0.007), versus about 0.000 in the other buckets).

m-pinning dose-response (`f = m / ((d-1)/2)` in {-0.6, -0.2, 0, 0.5, 1}; full tables in the tables file; pinned runs use the free run's frozen hyperparameters, 10 seeds each): within a cell the log loss varies by at most 0.004-0.016 across the five values of f, without a monotone pattern (e.g. Cora on 0.3340 / 0.3335 / 0.3368 / 0.3325 / 0.3341; Citeseer on 0.3451 / 0.3441 / 0.3408 / 0.3418 / 0.3569; Pubmed off 0.4709 / 0.4695 / 0.4686 / 0.4649 / 0.4677), AUROC varies by 0.001-0.01, and the second-mode mass is at most 0.0094 (Pubmed off at f = -0.6) in every pinned cell, including the bimodal-regime pins (`m < 0`). Pinning epsilon also limits concentration, so it is not a pure bimodality switch. Relative to the **free** TNBBeta runs, the pinned runs have a lower log loss in the features-off cells (e.g. MAG-CS off 0.258-0.262 versus free 0.293; Cora off 0.461-0.465 versus 0.486; Citeseer off 0.515-0.520 versus 0.549; Pubmed off 0.465-0.471 versus 0.496) and in Pubmed on and MAG-CS on at some f, and a similar or higher log loss in Cora on and Citeseer on at f = 1; the pinned runs use the free run's tuned configuration so this is not a like-for-like tuned comparison, and no test was run (descriptive). The pinned values at f = 0 or 0.5 are also below the tuned vMF/PowerSpherical log loss in every features-off cell, but this comparison is also not paired-tested and uses TNBBeta runs with a free parameter fewer and untuned hyperparameters; I do not interpret it here.

## Predictions scored (from the pre-registration)

1. **Features on:** *missed.* Hit: no features-on confirmed win or loss; TNBBeta is within 1.5 AUROC points of the best family in all four cells; MAG-CS on TNBBeta is behind on sampled AUROC (-0.0053, not BH-significant). Miss: `|mean a| < 0.02` held in only 2 of 4 features-on cells (Cora -0.013, Pubmed +0.006; Citeseer -0.027 and MAG-CS -0.033 exceed it) and MAG-CS on log loss and ECE advantages are negative (-0.033, -0.035), not >= 0.
2. **Features off:** *mixed.* Hit: vMF/PS AUROC gaps to TNBBeta are under 2 points in all 4 datasets (the Oct-9 5-8 point gaps in the paper's free-kappa bars were largely a kappa-initialisation artifact), and the Gaussian is above TNBBeta on AUROC in 3 of 4 (Cora, Pubmed, MAG-CS). Miss: TNBBeta's log-loss advantage is positive in 0 of 4 features-off cells (1 of 8 overall: Pubmed on), Holm-significant wins 0 of 8.
3. **Density:** *missed.* Hit: no confirmed win on Cora or Citeseer on. Miss: MAG-CS advantage is not larger than Cora's or Citeseer's (it is more negative on features on and off); positive per-dataset Spearman in only 1 of 4 datasets on each features setting; the AUROC Spearman is +0.25 and +0.24, above the 0.1 limit.
4. **Degree buckets:** *missed.* Hit: the low-degree-bucket advantage exceeds the >= 4 advantage in only 2 of 6 defined cells (Cora off, MAG-CS on). Miss: the log-loss advantage is negative in many buckets with min degree >= 4 and the AUROC gradient G is BH-significant in 5 of 6 defined cells.
5. **Shape:** *mostly hit.* More than half of the nodes have `m < 0` in all 8 cells; mean second-mode mass is at most 0.0200 (below the 0.05 limit; equal to the "below 0.02" claim only at its boundary, Citeseer off); `m`-degree Spearman positive in all 8. The weak dose-response expectation (f = 1 within 0.02 nats of free in 6 of 8 cells) is *missed*: it holds in 4 of 8, because the pinned runs are better than free in four features-off cells by 0.023-0.036.

## Where do TNBBeta's wins show up on real graphs, and where not

**Not on the main comparison.** Against vMF and PowerSpherical tuned equally (kappa started at d and trained), TNBBeta has no confirmed win on log loss in any of the 8 (dataset, features) cells, is significantly but modestly worse in 6, and is not significantly different in 2. It is behind on ranking metrics in 7 of 8 cells (AUROC, centre AUROC, AP), as in the synthetic study, and no better calibrated by ECE. The paper's current statement that Gaussian and TNBBeta "dominate without node features" over vMF/PS does not survive: with kappa initialised at d the vMF/PS AUROC is within 1 point of TNBBeta (sometimes above it) in all four featureless graphs, and the Gaussian is best in Pubmed on and off.

**Where it is closest or ahead:** Pubmed with node features (the densest-in-features, lowest-clustering graph here): TNBBeta ties or edges vMF/PS on log loss (+0.006 vs vMF, p = 0.051 not significant; +0.012 vs PowerSpherical, p = 0.002) and has significantly higher sampled AUROC (+0.0025), centre AUROC (+0.006) and AP; the within-graph bucket pattern there is the synthetic pattern (advantage rising from -0.024 at degree 0-1 to +0.03 at degree 8-15). Against the Gaussian reference TNBBeta is ahead on log loss with features on in Cora, Citeseer and (not significantly) MAG-CS. Those are the only favourable spots; the Gaussian is better than all sphere families on Pubmed.

**Where it does not show up:** the density manipulation. Thinning edges does not make TNBBeta's log-loss advantage rise with density (combined Spearman -0.02 and -0.08), the one-bucket-by-degree and natural-subgraph looks are heterogeneous in sign, and the synthetic Spearman +0.46 does not transfer (it would have to be read with 4 dataset blocks and with the limitation that thinned graphs reuse full-density hyperparameters). The AUROC deficit does shrink with density, which the synthetic study did not predict.

**Mechanism (descriptive, no causal claim):** TNBBeta lands at `m < 0` for essentially every node of every real graph and yet is a single cap (second-mode mass <= 0.02 on average; only low-degree nodes in featureless graphs carry some), the same as on the synthetic families, and pinning m to any value in the tested range changes log loss by under 0.016 within a cell and leaves the second-mode mass tiny.

## Limitations and caveats

* Hyperparameter grids are small and hit their edges (dropout 0.4, lr 0.03, d = 32 are frequent winners) for all families alike; a wider grid could move any comparison, and the margin by which vMF/PS beat TNBBeta (0.01-0.03 nats) is within what tuning could change. Selection was by validation log loss for every family (the paper used validation AUC); a sensitivity analysis using validation AUC was not run.
* Thinning, pinning and community runs reuse hyperparameters tuned at full density (documented in the pre-registration); the larger thinned-graph deficits in MAG-CS and Pubmed with features off probably partly reflect that.
* One graph and one dataset split family per dataset: the 10 seeds vary the split, initialisation and thinning, not the underlying graph; "significance" refers to that variation only. Four datasets means the density analysis has four blocks and no generalisation across graphs is licensed.
* The balanced test pairs use random non-edges, so log loss and AUROC refer to the easy "random negative" task (the HeaRT-style hard-negative metrics of the synthetic study were not computed). Real graphs have no true edge-probability matrix, so the log-loss difference is excess loss on these pairs, not a distance to a known truth.
* Posterior samples are 32 per run and the per-seed sampling noise is not separated from training noise.
* The pre-registered tests that returned non-significant or null results are reported above; no analysis was changed after seeing results except the figure legend position and the deviations listed in the pre-registration.
