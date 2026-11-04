# Pre-registration: dense real graphs (Amazon Computers, Amazon Photo, Squirrel), 2026-11-04

**Status: FROZEN** (2026-11-04), written before any model of this study was trained for a result. After this point the file is edited only by appending to "Deviations" at the end.
Base commit: `ed47071` (branch `real-graph-revision`, PR #74, on `master` `d5bb583`); working branch `dense-real-graphs`. The code of this study is additive to PR #74: `src/tnbbeta_vae/data/dense_graphs.py` (loaders), a `REAL_GRAPH_STUDY=dense` switch in `apps/real_graph/run.py` (default reproduces the revision exactly), a margin/BH switch in `apps/real_graph/analyze.py`, labels in `apps/real_graph/figures.py`, `scripts/slurm/dense_graph*.sbatch`.
What had been seen before freezing: the dataset statistics below; the synthetic study; the previous real-graph results; and a smoke test (job 2313075: one run per family on Amazon Computers features-on and Squirrel features-off at one grid point, tuning seed 100, used only to time the runs and to size the SLURM arrays; no result of it was used for any choice below).

## Why this study exists

The synthetic graph study (ten graph families, 200 fresh configurations, TNBBeta against trained-kappa vMF and PowerSpherical) found that TNBBeta's advantage depends on mean degree. Median paired advantage over the better of vMF/PS (positive = TNBBeta better), by nominal mean degree. This table is the only source of the predictions below.

| mean degree | truth KL | sampled log loss (share of configs won) | sampled AUROC |
|---|---|---|---|
| < 6 | -0.040 | -0.043 (12%) | -0.022 |
| 6-12 | -0.018 | -0.019 (23%) | -0.010 |
| 12-24 | -0.003 | -0.005 (44%) | -0.005 |
| >= 24 | +0.015 | +0.013 (73%) | -0.001 (50%) |

At degree >= 24 the log-loss effect is +0.0137 with identity features and +0.0024 (win share 63%) with informative features. The earlier real-graph revision (Cora 3.9, Citeseer 2.7, Pubmed 4.5, MAG-CS 8.8 mean degree) found log-loss losses of -0.013 to -0.033 nats (only Pubmed with features about +0.006), consistent with the synthetic low-degree rows. The sharpest remaining test is a dense real graph. A null or negative result is a legitimate outcome; nothing here is tuned toward a win.

## Data

All three are undirected, symmetrised, binarised, without self-loops, deduplicated, **all nodes kept** (no largest-component restriction; isolated nodes stay), identical for the three. Features on: the stored binary features (bag of words or page-feature indicators). Features off: identity features (as in the revision). Verified statistics after this preprocessing (`src/tnbbeta_vae/data/dense_graphs.py`; the adjacency is symmetric, binary, zero diagonal):

| graph | nodes | undirected edges | mean degree | degree CV | transitivity | isolated nodes | components (largest) | max degree | features |
|---|---|---|---|---|---|---|---|---|---|
| Amazon Computers | 13,752 | 245,861 | 35.76 | 1.97 | 0.108 | 281 (2.0%) | 314 (13,381) | 2,992 | 767 |
| Amazon Photo | 7,650 | 119,081 | 31.13 | 1.52 | 0.177 | 115 (1.5%) | 136 (7,487) | 1,434 | 745 |
| Squirrel | 5,201 | 198,353 | 76.27 | 2.12 | 0.348 | 0 | 1 (5,201) | 1,903 | 3,148 |

**Third graph and why.** BlogCatalog and Flickr (the requested options) were not obtainable cleanly: the mirror `EdisonLeeeee/GraphData` ships them as `.npz` files whose adjacency and attributes are pickled object arrays (`numpy.load(..., allow_pickle=False)` refuses them, and unpickling downloaded content is excluded by the handling rules); the Renchi Yang dataset page links Google Drive folders and the `.mat` files of Tang and Liu (BlogCatalog3, 10,312 nodes, no node attributes), and the one Drive file id I tried (taken from memory of the PyG loader, not verified) returned an error page. Facebook Page-Page (the stated fallback) has mean degree 15.2 and is excluded by the rule that every graph must have verified mean degree at least 24. I therefore used the **Wikipedia Squirrel** page-page graph of MUSAE (Rozemberczki, Allen and Sarkar, 2019): plain `squirrel_edges.csv` and `squirrel.json` from `raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input/`, parsed with `csv`/`json` only. The file has 217,073 rows, of which 140 are self-loops and the rest list some pairs in both orders; the deduplicated undirected graph has 198,353 edges, mean degree 76.3 (the "217,073 edges" of the literature counts rows). It is attributed (3,148 binary features), connected, small (5,201 nodes, like BlogCatalog's 5,196) and the densest of the three, so the study spans mean degree 31-76. Crocodile (11,631 nodes, mean degree about 29) was the alternative; I chose Squirrel for the wider density range. Squirrel is known to be heterophilous; this is a caveat on the third graph.
Amazon Computers and Photo: Shchur et al., "Pitfalls of Graph Neural Network Evaluation" (2018), `amazon_electronics_{computers,photo}.npz` from `github.com/shchur/gnn-benchmark` (flat plain arrays; the stored adjacency is not symmetric and is symmetrised). **Licences as far as read:** the `gnn-benchmark` repository is MIT; the MUSAE repository is GPL-3.0 (for the code), the data is also distributed through SNAP, whose Wikipedia-networks page I read for terms and found none stated; I make no claim about a data licence beyond that. Nothing of the raw data is committed to git. Downloads were put in a new directory under `slurm_logs/` and read by an isolated interpreter (`uv run python -I`); they now live in `$TNBBETA_DATA_DIR/dense_graphs/<graph>/`.

## Protocol (as the revision, `real_graph_revision_preregistration_2026-11-03.md`, unless stated)

* **Cells:** 3 graphs x features on/off = 6 cells. **Families:** `tnbbeta` (free epsilon), `vmf` and `power_spherical` (`initial_kappa` = ambient dimension, kappa trained; the primary comparators), `gaussian` (reference, never a primary comparator). `GraphVAEConfig(decoder_bias=True)`, hidden width 32, learned temperature, uniform-sphere prior, ambient `latent_dim = d + 1` for sphere families and `d` for Gaussian. **Exact vMF sampler**: every vMF run records `vmf_sampler` and aborts unless it is `exact_envelope` (checked every epoch and at scoring).
* **Edge split:** `split_edges` (5% validation, 10% test, an equal number of random non-edges each); seed `s` fixes split and initialisation for all families (paired). **Seeds:** final runs 0-9; tuning runs 100-102 (disjoint).
* **Training:** Adam, full batch, 200 epochs for all three graphs (the revision's default for the non-Pubmed graphs), one random negative per training edge per step; the epoch of each run is selected by validation sampled log loss (16 posterior samples), the test metrics are then computed once with 32 samples.
* **Tuning (changed from the revision):** per (graph, features, family), learning rate in {0.003, 0.01, 0.03} x dropout in {0, 0.4, 0.6} x d in {16, 32, 64} = 27 configurations, each on 3 tuning seeds; the configuration with the lowest mean validation sampled log loss (diverged runs count as infinite) is frozen into `tuned.json` and used by every main run of that cell. Every family gets the identical grid and criterion. The previous grid hit its edges for all families (dropout 0.4 won 27 of 32 cells, d = 32 25 of 32, lr 0.03 17 of 32); the results note states, per hyperparameter, how many selections still sit on an edge of this grid and does not tune further. The full grid is run (1,944 tuning runs); the smoke test showed it fits the budget, so nothing is trimmed.
* **Metrics:** on the test pairs (held-out positives plus an equal number of random non-edges): **primary sampled log loss** (mean over 32 joint posterior samples of the predicted probability, clamped to [1e-6, 1-1e-6]); secondary sampled AUROC, centre AUROC, average precision, 10-bin ECE. Truth KL does not exist on real graphs; the paired log-loss difference is the real-data analogue (the entropy term cancels in expectation; the labels are one realisation, label noise about 0.003-0.005 nats per seed).

## Statistics (pre-specified)

Advantage `a = comparator - TNBBeta` for lower-is-better metrics (log loss, ECE), `TNBBeta - comparator` for higher-is-better; **`a > 0` means TNBBeta is better**. The primary comparator of a (cell, metric) is the better, by the cell's 10-seed mean of that metric, of `vmf` and `power_spherical` (chosen without looking at TNBBeta). Each comparator and the Gaussian are also reported descriptively.

* Paired p-values: exact two-sided sign-flip test over the 10 per-seed differences (all 1,024 sign patterns; checked against an independent enumeration in the tests); effect sizes with a 95% percentile bootstrap over seeds (10,000 resamples, generator seed 20261104).
* **P1 (primary, confirmatory): sampled test log loss, 6 cells**, TNBBeta against the primary comparator. Holm over the 6 two-sided tests at 0.05 defines "significant". **Confirmed win / loss at the primary margin 0.01 nats** (set before any run: the previous 0.05-nat margin is larger than the synthetic effect, about 0.013 nats at degree >= 24, so it could not be met even if the synthetic prediction held; 0.01 is just below that effect): exact one-sided sign-flip tests of `mean(a - 0.01) > 0` (win) and `mean(-a - 0.01) > 0` (loss), Holm over these 12 one-sided tests. The **0.05-nat margin is reported as a secondary label** (its own Holm over 12). A cell significant without the margin is "significant, below margin". The margin only enters labels; it changes no test.
* **P2 (secondary): sampled AUROC, centre AUROC, AP, ECE, 6 cells each**, same test and comparator rule, Benjamini-Hochberg at q = 0.05 **within each metric** (6 tests per family). AUROC losses are shown whatever their size.
* **Degree buckets (descriptive):** each test pair goes to the bucket of the lower training degree of its endpoints, bins {0-1, 2-3, 4-7, 8-15, 16+}; label-balanced per-bucket log-loss and AUROC advantage with seed-level bootstrap intervals (and the anti-conservative edge-level interval for comparison); at least 20 pairs of each label per bucket. The degree-gradient contrast `G` (advantage at min degree >= 4 minus advantage at <= 3) is computed where defined and sign-flip tested with BH over the defined cells; with dense graphs the lowest buckets are small and `G` is undefined in some cells (reported as such).
* **TNBBeta shape (descriptive only):** fraction of nodes with `m = epsilon - (latent_dim - 1)/2 < 0`, second-mode mass, Spearman of `m` with training degree, mean `p`, `q`, `epsilon`. No m-pinning, no community stage, **no edge thinning** in this task. No causal claim about `m`.

## PREDICTIONS (written before any run; sole source: the synthetic table above)

1. **Log loss:** TNBBeta's mean advantage over the primary comparator is positive in at least 4 of the 6 cells. *Miss:* fewer than 4 positive cells. *The synthetic degree effect does not transfer to dense real graphs* if TNBBeta has a negative mean advantage in 4 or more of 6 cells.
2. **Features:** the advantage with features on is smaller than with features off for at least 2 of the 3 graphs (synthetic: +0.0024 informative versus +0.0137 identity; the effect with informative features is small and may be indistinguishable from 0). *Miss:* features-on advantage at least as large as features-off for 2 or 3 graphs.
3. **Ranking metrics:** roughly tied rather than won: sampled AUROC advantage |a| < 0.01 in at least 4 of 6 cells (synthetic median -0.001). *Miss:* |a| >= 0.01 in 3 or more cells (a clear win or loss on ranking).
4. **ECE:** advantage positive in at least 4 of 6 cells. *Miss:* fewer than 4.
5. **Confirmation:** no cell is a confirmed win at the 0.05 margin (the synthetic median effect is 0.013). *Miss:* any cell confirmed at 0.05. No prediction is made on how many cells are confirmed at 0.01 (the test has limited power at 10 seeds; the honest expectation is zero to two features-off cells).

No prediction is made about shape statistics or degree buckets; they are descriptive.

## Caveats carried

Three graphs; the 10 seeds vary splits and initialisation but not the graph, so the seed-level inference is about training and split variability only, not about graphs in general; negatives are uniformly random pairs, not hard negatives; the Amazon graphs are hub-heavy (max degree 1,434 and 2,992) and Squirrel is dense, small and heterophilous; all three graphs are far larger than the synthetic ones (13.8k, 7.7k and 5.2k nodes against about 1k); mean degree is confounded with graph type, size and feature kind here, so even a result in line with the synthetic table is not evidence that degree is the cause.

## Compute and tests

All training and scoring on SLURM (GPU array tasks with several processes per GPU, `OMP_NUM_THREADS=1`, no `--device cuda`, exit 75 resubmits through `scripts/slurm/no_gpu_retry.sh`, node k003 excluded). Tune stage first (manifest of 1,944 runs), then `tuned.json` frozen, then the main stage as one array of 6 tasks (one per cell, each running its 4 families x 10 seeds). Analysis on SLURM; figures on the login node.

## Deviations

(none yet)

### Addendum 1 (2026-11-04): Stage 2, edge thinning, registered before any thinning run

Written after the main-stage results were seen and BEFORE any thinning run. The main-stage results do not enter the thinning predictions below except through the frozen `tuned.json`.

* **Design.** As the revision's Stage 3: before the train/val/test split, delete a uniformly random fraction of the edges so that **0.7, 0.5, 0.3** of them are retained (1.0 is the main stage); nodes, features and graph size fixed; thinning seeded by the run seed. Same 3 graphs x features on/off x 4 families x 10 seeds (720 runs). **Hyperparameters are the frozen full-density `tuned.json`** (not re-tuned per density): the large TNBBeta deficits seen at low density in the revision may be partly that. Expected realised mean degrees: Computers about 25 / 18 / 11, Photo 22 / 16 / 9, Squirrel 53 / 38 / 23.
* **Statistic (as `apps/real_graph/analyze.py`, `analyze_thinning`).** For each (graph, features) cell, the Spearman correlation of the seed-level log-loss advantage with the realised mean degree over the four density levels (10 seeds x 4 levels); the combined statistic is the mean over cells (separately for features on and off); one-sided permutation test (100,000 draws, outcomes permuted across levels within each seed row), H1: the advantage rises with density. Per-cell correlations are descriptive; BH over the cells for any per-cell p-values.
* **Predictions (from the synthetic degree table: the advantage falls as degree falls through 24 to 12).** (T1) the combined Spearman is positive for features off and for features on. (T2) the advantage at 0.3 retention is lower than at 1.0 in at least 2 of 3 graphs for each features setting. (T3) at 0.3 retention (degrees about 9-23) the features-off log-loss advantage is smaller than at full density by at least 0.01 nats in at least 2 of 3 graphs. **Misses:** T1 if either combined Spearman is <= 0 or its one-sided p > 0.05; T2 or T3 if the stated count is not reached. The revision's own thinning (Cora, Citeseer, Pubmed, MAG-CS, mean degree 1.4-8.8) was flat (combined Spearman -0.02 and -0.08), so a flat result here is a live possibility and would count against the transfer of the synthetic density effect.
* **Caveats.** Hyperparameters frozen at full density; thinning changes the degree distribution differently from the synthetic generator; 3 graphs; seeds vary splits and thinning, not the graph.
