# Dense real graphs: results (main stage), 2026-11-04

Date: 2026-11-04. Base commit: `ed47071` (branch `real-graph-revision`, PR #74, on `master` `d5bb583`), working branch `dense-real-graphs`, uncommitted. Pre-registration (frozen before any run): `dense_real_graph_preregistration_2026-11-04.md`. All tables: `dense_real_graph_tables_2026-11-04.md`. Figures: `figures/dense_real_graph_{main_bars,advantage,degree_buckets}.png`. Edge thinning was **not** run (by instruction).

## Commands

```bash
export REAL_GRAPH_STUDY=dense
uv run python -m apps.real_graph.run manifest --stage tune                       # 1,944 runs
STAGE=tune TOTAL_TASKS=12 sbatch --array=0-11 scripts/slurm/dense_graph.sbatch   # job 2313099 (tasks 4,5,7-10 resubmitted: 2313152)
uv run python -m apps.real_graph.run select                                      # tuned.json
uv run python -m apps.real_graph.run manifest --stage main                       # 240 runs
STAGE=main TOTAL_TASKS=6 PROCS=8 sbatch --array=0-5 scripts/slurm/dense_graph.sbatch   # job 2313299
ANALYZE_ARGS="--stages main --workers 16" AUX=analyze sbatch scripts/slurm/dense_graph_aux.sbatch   # job 2313352
uv run python -c "...figures.render_all(analysis, Path('writeup/results/figures'), prefix='dense_real_graph')"   # login node
```

Data: Amazon Computers/Photo, `.npz` from `github.com/shchur/gnn-benchmark` (MIT); Squirrel, `squirrel_edges.csv` + `squirrel.json` from `github.com/benedekrozemberczki/MUSAE` (code GPL-3.0; no data terms found on the SNAP page). Checkpoints/records: `$TNBBETA_CHECKPOINT_DIR/real_graph_dense/`. Verified graph statistics are in the pre-registration (Computers 13,752 nodes / 245,861 edges / mean degree 35.8; Photo 7,650 / 119,081 / 31.1; Squirrel 5,201 / 198,353 / 76.3). BlogCatalog and Flickr could not be obtained without unpickling and Facebook Page-Page has mean degree 15.2, so Squirrel is the third graph (see the pre-registration).

All 240 main runs finished; 0 diverged; all 60 vMF runs used `exact_envelope`. Tests: full pytest, ruff format/check and pyright run on SLURM (see "Checks").

## Main result: log-loss advantage of TNBBeta over the primary comparator (a > 0: TNBBeta better), 10 seeds

| cell | comparator | advantage [95% CI] | exact p | Holm (6) | label at 0.01 | label at 0.05 |
|---|---|---|---|---|---|---|
| Computers, features on | power_spherical | +0.0072 [-0.0177, 0.0257] | 0.613 | 1.00 | not significant | not confirmed |
| Computers, features off | vmf | +0.0647 [0.0631, 0.0664] | 0.002 | 0.012 | confirmed win | confirmed win |
| Photo, features on | power_spherical | +0.0050 [-0.0237, 0.0300] | 0.738 | 1.00 | not significant | not confirmed |
| Photo, features off | vmf | +0.0589 [0.0580, 0.0596] | 0.002 | 0.012 | confirmed win | confirmed win |
| Squirrel, features on | power_spherical | +0.0016 [-0.0162, 0.0165] | 0.871 | 1.00 | not significant | not confirmed |
| Squirrel, features off | vmf | +0.0680 [0.0654, 0.0708] | 0.002 | 0.012 | confirmed win | confirmed win |

0.002 is the smallest attainable exact p (2/1024). With features **on** TNBBeta ties the better of vMF/PS on log loss in all three graphs (point estimates +0.002 to +0.007, intervals spanning about +-0.02); with features **off** it beats it by about 0.06 nats in all three, the three cells passing Holm and both margins.

## The other metrics (advantage over the primary comparator; BH within each metric over 6 cells)

| cell | sampled AUROC | centre AUROC | AP | ECE |
|---|---|---|---|---|
| Computers on | -0.0058 (ns) | -0.0068 (ns) | -0.0052 (ns) | +0.0163 (q 0.002) |
| Computers off | +0.0060 (q 0.004) | +0.0046 (q 0.003) | +0.0066 (q 0.004) | +0.0587 (q 0.002) |
| Photo on | -0.0047 (ns) | -0.0026 (ns) | -0.0027 (ns) | +0.0201 (q 0.006) |
| Photo off | +0.0044 (q 0.004) | +0.0023 (q 0.003) | +0.0051 (q 0.004) | +0.0545 (q 0.002) |
| Squirrel on | -0.0035 (ns) | -0.0240 (q 0.003) | +0.0006 (ns) | +0.0151 (q 0.002) |
| Squirrel off | +0.0092 (q 0.004) | +0.0036 (q 0.003) | +0.0095 (q 0.004) | +0.0397 (q 0.002) |

Ranking metrics are small (|a| < 0.01 everywhere): features-on slightly negative and not significant, features-off slightly positive and significant. Squirrel features-on centre AUROC is the one clear ranking loss (-0.024). ECE is better for TNBBeta in all six cells (significantly).

## Gaussian reference (descriptive, not a comparator)

Log-loss advantage of TNBBeta over the Gaussian: Computers off -0.0015 (ns), Photo off +0.0043 (small win), Squirrel off -0.0109 (loss); features on: +0.0743 and +0.0572 (wins, Computers and Photo) but -0.0524 on Squirrel (a clear loss). The Gaussian also has the best AUROC on Squirrel (0.981 on, 0.987 off). So the large features-off margin is a margin over vMF/PS, not over the Gaussian, which is on par with TNBBeta in the three features-off cells.

## Predictions: hit or miss (criteria in the pre-registration)

| # | prediction | outcome | verdict |
|---|---|---|---|
| 1 | advantage > 0 in at least 4 of 6 cells | positive in 6 of 6 (3 significant, 3 with intervals including 0) | hit (the "does not transfer" criterion, negative in 4 or more cells, is not met) |
| 2 | features-on advantage smaller than features-off in at least 2 of 3 graphs | smaller in 3 of 3 (about 0.005 versus 0.06) | hit |
| 3 | sampled AUROC \|a\| < 0.01 in at least 4 of 6 cells | \|a\| < 0.01 in 6 of 6 (but the 3 features-off values are BH-significant) | hit on the magnitude criterion |
| 4 | ECE advantage positive in at least 4 of 6 | positive in 6 of 6 | hit |
| 5 | no confirmed win at the 0.05 margin | 3 cells confirmed at 0.05 (features off) | **miss** |

Honest reading: the direction and the features-on/off ordering match the synthetic table, but the features-off effect (about 0.06 nats) is four to five times the synthetic median at degree >= 24 (+0.0137), and the features-on effect (+0.002 to +0.007) is compatible with both zero and the synthetic +0.0024. Prediction 5 missed because the size is far larger than predicted, which is a reason for suspicion, not celebration.

## Where TNBBeta wins and loses

* **Wins:** features-off log loss and ECE over vMF/PS in every graph (about 0.06 nats; ECE 0.015-0.03 against 0.07-0.09); features-off ranking by 0.004-0.009 AUROC; ECE with features on (+0.015 to +0.020).
* **Ties:** features-on log loss and AUROC against power_spherical (and Photo/Squirrel against vMF, p 0.08-0.13, descriptive).
* **Losses:** Squirrel features-on centre AUROC (-0.024); against the Gaussian reference on Squirrel (both feature settings) and, with identity features, no gain on Computers; low-degree pairs, below.
* **Degree buckets (descriptive, lower endpoint's training degree):** with features off the pooled win is carried by pairs whose lower-degree endpoint has training degree >= 4 (advantage +0.04 to +0.08); pairs with degree 0-1 are heavily negative (-0.22 to -0.41 nats), 2-3 about zero to -0.12. With features on, the advantage over power_spherical rises from -0.10 (Computers), -0.06 (Photo) and -0.36 (Squirrel) at degree 0-1 to +0.003, +0.015 and +0.017 at degree 16+. The degree-gradient contrast G on log loss is positive and BH-significant in all six cells; on AUROC it is positive with features on (Computers, Squirrel) and negative with features off. Seed-level intervals are the honest ones; the edge-level ones are anti-conservative.

## TNBBeta shape (descriptive; no causal reading)

Free TNBBeta has `m < 0` for 93-100% of nodes in five cells (identity-feature cells 99.98-100%); Photo features-on is the exception (14% of nodes, mean `m` +25, `q` mean 0.22, a different regime). The second-mode mass is near 0 (0.0005-0.0033), i.e. a single cap, as in every earlier study. The Spearman of `m` with training degree is positive (0.27-0.81). These are statistics of fitted posteriors, not an explanation of the log-loss differences.

## Tuning and edge hits

27 configurations x 3 seeds (100-102) per (graph, features, family), 1,944 runs, nothing trimmed (the smoke test showed vMF at 10-56 s per run on these graphs). 42 tuning runs diverged (counted as infinite loss). **Grid-edge selections, 24 cells (graph x features x family): learning rate 19 (the maximum 0.03 in 17 of them, the minimum 0.003 in 2), dropout 16, d 15.** Dropout 0.6 was selected only 3 times (TNBBeta Computers-on, vMF Computers-on and Photo-on), dropout 0 in 13: the widened grid moved the dropout optimum back to 0 for identity features. The tuning still ends on its edge for the learning rate in most cells, especially with identity features, where the 200-epoch budget with learning rate 0.03 is the best of the three and the optimum may lie beyond it. Full table in `dense_real_graph_tables_2026-11-04.md`.

## Caveats that matter for reading the features-off result

1. **vMF and PS are far behind the Gaussian with identity features** (log loss 0.19-0.26 against 0.14-0.16; AUROC 0.97-0.98 against 0.985-0.988), and their selected learning rate sits at the grid edge. As in `vmf_ps_fixed_kappa_and_training_dynamics_2026-10-30.md`, trained kappa moves little in 200 epochs; this study has no pinned-kappa arm, so part of the "TNBBeta wins" with identity features may measure comparator optimisation (kappa and learning rate), not family shape. The Gaussian reference (tied with TNBBeta) is the better yardstick for that cell type.
2. **Seed variance with features on:** TNBBeta's log loss varies a lot across seeds (for example Computers 0.295-0.405, Photo 0.264-0.402), whereas power_spherical's interval is a few thousandths wide, which is why features-on intervals are about +-0.02 and the cells are not significant. Dropout 0.6 was selected for TNBBeta Computers-on.
3. Three graphs; seeds vary splits and initialisation, not the graph; random negatives; hub-heavy Amazon graphs (max degree 1,434 and 2,992) and a small, dense, heterophilous Squirrel; mean degree is confounded with graph type, size and feature kind, so nothing here isolates degree as a cause. The third graph is Squirrel because BlogCatalog/Flickr could not be loaded without unpickling.
4. Truth KL is unavailable; the paired log-loss difference includes label noise of about 0.003-0.005 nats per seed.

## Deviations

None in the pre-registered analysis. Implementation notes: (a) tuning array tasks 4, 5, 7-10 failed in seconds on node `m001` (shared storage not readable there; the workers exited with status 1 rather than 75, so the no-GPU resubmit did not trigger) and were resubmitted excluding `k003,m001` (job 2313152; finished runs are skipped, the manifest is unchanged); `scripts/slurm/dense_graph.sbatch` now excludes both. (b) The main-stage array maps one task to one cell through a new `--cell` filter of `apps.real_graph.run run`. (c) Hyperparameters of the four-dataset revision study are untouched: all new behaviour is behind `REAL_GRAPH_STUDY=dense`.

## Checks

Full pytest, `ruff format --check`, `ruff check .` and full `pyright` on SLURM: format/lint/pyright jobs 2313353 passed (0 errors); full pytest (job 2313354) 1,194 passed, 1 failed (the figure-count assertion, caused by the new advantage figure); after updating that assertion the revision and dense test files pass (22 passed, job 2313419) and ruff/pyright pass again (job 2313420). The full suite was not rerun after the one-line fix. Tests added: `tests/data/test_dense_graphs.py` (3) and a secondary-margin test in `tests/apps/test_real_graph_analysis.py`; the sign-flip enumeration check is the revision's existing test.

## Edge thinning (stage 3, run after the main-stage results were reviewed)

Date: 2026-11-04. Code: branch `dense-real-graphs` (this PR), commit of the thinning figure and this section on top of `5433ea0`. Comparators are tuned once at full density (frozen `tuned.json`); each graph keeps 0.3, 0.5 and 0.7 of its edges plus the full-density main-stage runs, 10 paired seeds, 720 thinning runs (jobs 2313433, analysis 2313434; all 720 run files present, all tasks exit 0).

```bash
export REAL_GRAPH_STUDY=dense
uv run python -m apps.real_graph.run manifest --stage thin     # 720 runs, uses the frozen tuned.json
STAGE=thin TOTAL_TASKS=12 sbatch --array=0-11 scripts/slurm/dense_graph.sbatch
ANALYZE_ARGS="--stages main thin --workers 16" AUX=analyze sbatch scripts/slurm/dense_graph_aux.sbatch
uv run python -m apps.real_graph.thinning_lines    # login node; figure below
```

Per-graph figure (columns are graphs, rows features on and off; no pooling): `figures/dense_thinning_lines_per_graph_2026-11-05.png`, same script, same 10-seed bootstrap intervals.

Figure: `figures/dense_thinning_lines_2026-11-04.png` (test log loss per family against fraction of edges kept, averaged over the three graphs within each seed, 95% seed-bootstrap intervals). Full tables: `$TNBBETA_CHECKPOINT_DIR/real_graph_dense/analysis.md`.

### Predictions T1-T3 (Addendum 1 of the pre-registration): all met

| Test | Outcome |
|---|---|
| T1 combined Spearman > 0, one-sided p <= 0.05 | Features off: 0.962 (log loss), 0.780 (AUROC), p < 0.001. Features on: 0.167, p = 0.016 (AUROC 0.172, p = 0.015). Met. |
| T2 advantage at 0.3 below full density in >= 2 of 3 graphs | Features off 3 of 3; features on 2 of 3 (Squirrel is the exception: +0.0086 at 0.3, +0.0016 at 1). Met. |
| T3 features-off log-loss advantage smaller at 0.3 by >= 0.01 nats in >= 2 of 3 graphs | 3 of 3 (drops 0.057 Computers, 0.043 Photo, 0.055 Squirrel). Met. |

### Log-loss advantage over the primary comparator (a > 0: TNBBeta better)

| Graph | Kept | Mean degree | Features on | Features off |
|---|---|---|---|---|
| Computers | 0.3 | 10.7 | -0.0006 [-0.0095, 0.0085] | +0.0078 [0.0058, 0.0099] |
| Computers | 0.5 | 17.9 | +0.0040 [-0.0043, 0.0119] | +0.0416 [0.0394, 0.0439] |
| Computers | 0.7 | 25.0 | +0.0082 [0.0001, 0.0150] | +0.0547 [0.0535, 0.0559] |
| Computers | 1 | 35.8 | +0.0072 [-0.0177, 0.0257] | +0.0647 [0.0631, 0.0664] |
| Photo | 0.3 | 9.3 | -0.0317 [-0.0993, 0.0266] | +0.0158 [0.0124, 0.0193] |
| Photo | 0.5 | 15.6 | -0.0139 [-0.0499, 0.0197] | +0.0389 [0.0369, 0.0408] |
| Photo | 0.7 | 21.8 | +0.0006 [-0.0235, 0.0271] | +0.0503 [0.0489, 0.0518] |
| Photo | 1 | 31.1 | +0.0050 [-0.0237, 0.0300] | +0.0589 [0.0580, 0.0596] |
| Squirrel | 0.3 | 22.9 | +0.0086 [0.0050, 0.0124] | +0.0133 [0.0102, 0.0162] |
| Squirrel | 0.5 | 38.1 | +0.0063 [-0.0027, 0.0134] | +0.0448 [0.0413, 0.0481] |
| Squirrel | 0.7 | 53.4 | +0.0098 [0.0043, 0.0149] | +0.0586 [0.0556, 0.0617] |
| Squirrel | 1 | 76.3 | +0.0016 [-0.0162, 0.0165] | +0.0680 [0.0654, 0.0708] |

### Reading

* **Features off:** the advantage rises almost monotonically with retained density in every graph, with tight intervals. This is the density trend the synthetic degree table predicts, unlike the flat thinning on the four sparse graphs.
* **Features on:** the trend is weak (combined Spearman 0.17); most per-graph intervals include 0, advantages stay under about 0.01 nats (the pre-registered margin), and Photo is a point-estimate loss at 0.3 and 0.5. Sampled AUROC is slightly worse for TNBBeta in most cells.
* **Comparator caveat (unchanged and now sharper):** in the line figure, features off, TNBBeta and the Gaussian are nearly identical at every retention level and both sit far below vMF and PS, whose learning rate was at the grid edge. The gradient is the gap between TNBBeta/Gaussian and under-tuned vMF/PS growing with density; it does not show a TNBBeta-specific density mechanism. Tuning was done at full density only, so vMF/PS may fit sparser training sets worse still.
* Thinning changes transitivity, isolated fraction and degree spread together with mean degree (see the analysis tables), so degree is not isolated as a cause.
* Not run: a pinned-kappa and extended-learning-rate control for vMF/PS on the features-off cells; it is the check that separates the two readings.
