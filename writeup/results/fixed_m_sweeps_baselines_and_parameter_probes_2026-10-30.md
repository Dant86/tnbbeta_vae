# Fixed-m sweeps, matched-dimension baselines, and what the posterior parameters communicate, 2026-10-30

- **Date:** 2026-10-30
- **Base:** `master` at `9eb8071` (merge of PR #67). The code is on the feature branch
  `fixed-m-sweep`; the last code commit these results use is `5b7f60f`
  (`Add param_probe: per-node posterior parameter and decoding probes`). The jobs ran
  from that branch's working tree *before* it was committed, so one difference matters:
  `--results-name` (commit `63d2b45`) was added **after** the real-graph sweep ran (see
  "Incident: overwritten result files"). Nothing else in the scripts changed after
  submission.
- **Scope:** TNBBeta's modality offset `m = epsilon - (latent_dim - 1) / 2` held fixed
  (SBM, real graphs), matched-dimension vMF / power-spherical / Gaussian baselines on the
  same splits, a per-bucket decomposition (degree, neighbor communities), and probes of
  the per-node posterior parameters. **MNIST is not covered:** its sweep is incomplete
  (see "Open items") and gets its own note.

## Summary

1. **`m < 0` is not needed for link prediction.** Free-epsilon TNBBeta lands deep in the
   bimodal regime on every graph (`m` fraction of the threshold `f = m / ((d-1)/2)` from
   -0.58 to -0.90, `frac_bimodal` about 1). Forcing it unimodal (`f >= 0`) gives the same
   AUC, or slightly better, overall and in every degree bucket, on four real graphs and
   the SBM.
2. **The TNBBeta advantage over vMF / power-spherical is not a dimension effect and not
   a node-type effect.** At TNBBeta's own dimension it is +0.02 to +0.10 AUC, and it is a
   roughly uniform shift across degree and neighbor-community buckets.
3. **vMF and power-spherical posteriors are at or near the prior on Cora and MAG-CS.**
   Decoding from posterior *samples* instead of centres gives chance AUC (0.49-0.53) for
   them, versus no loss for TNBBeta and the Gaussian. Where they collapse less (Pubmed,
   Citeseer) the gap to TNBBeta is smaller.
4. **TNBBeta's parameters communicate about one scalar, a degree-tracking confidence.**
   Their effective rank is 1.2-1.5 of 3 (1.1-1.5 of 2 with epsilon fixed), and they
   correlate at Spearman 0.85-0.98 with log degree. The Gaussian's sigma does the same. No
   family carries substantial independent information beyond degree.

The mechanism behind (3) is **not established**; see "Interpretation".

## Commands

Training (all on the `general` partition; `k003` excluded, see below):

```
sbatch scripts/slurm/fixed_m_sweep_sbm.sbatch                      # job 2278360, 55 cells
sbatch scripts/slurm/fixed_m_sweep_mnist.sbatch                    # job 2278361, 21 cells (not analyzed here)
sbatch --exclude=k003 scripts/slurm/fixed_m_sweep_graphs.sbatch    # job 2278729, 200 cells (25 tasks x 8)
sbatch --exclude=k003 scripts/slurm/fixed_m_baselines_graphs.sbatch # job 2278789, 60 cells (8 tasks)
```

Job `2278702` was the first submission of the real-graph sweep; it was cancelled after
one node (`k003`) failed most tasks (below), and `2278729` re-ran everything that had not
finished (52 cells had).

Analysis (CPU, against the checkpoints those jobs left in `$TNBBETA_CHECKPOINT_DIR`):

```
uv run python -m apps.eval.fixed_m_sweep_table --root runs/fixed_m_sweep_sbm
uv run python -m apps.eval.fixed_m_graphs_table --breakdown degree        # and --breakdown communities
for ds in cora citeseer pubmed mag_cs; do
    uv run python -m apps.eval.param_probe --dataset $ds --device cpu
done
uv run python -m apps.eval.param_probe_table [--partial]
```

Configuration: SBM at `latent_dim=16`, `p_in=0.3`, `p_out=0.05`, 10 communities x 50 nodes,
`--graph-seed 0`, 500 epochs, 5 seeds. Real graphs featureless (`--ignore-features`), 5
seeds, each dataset at TNBBeta's own selected `(lr, dropout, latent_dim, epochs)` from the
Figure 5 grid search: Cora (0.01, 0.4, 65, 200), Citeseer (0.01, 0.0, 17, 200), Pubmed
(0.01, 0.4, 17, 400), MAG-CS (0.01, 0.2, 33, 200). The real-graph `m` grid is in fractions
`f` of the threshold: `m = f * (latent_dim - 1) / 2` for
`f in {-0.9, -0.75, -0.6, -0.4, -0.2, 0, 0.2, 0.5, 1}` plus a free-epsilon reference.

## Results

### SBM, fixed m (test AUC, 5 seeds; TNBBeta only)

| m | AUC | q | temperature |
|---|---|---|---|
| free (learned m = -4.8) | 0.656 ± 0.009 | 0.91 | 1.6 |
| -7 | 0.614 ± 0.022 | 0.60 | 3.8 |
| -6 | 0.656 ± 0.007 | 0.96 | 1.5 |
| -4.5 | 0.638 ± 0.023 | 0.75 | 2.2 |
| -3 | 0.620 ± 0.026 | 0.59 | 2.7 |
| -1.5 | 0.653 ± 0.004 | 0.65 | 1.9 |
| -0.5 | 0.654 ± 0.007 | 0.58 | 2.0 |
| 0 | 0.653 ± 0.002 | 0.53 | 2.0 |
| 0.5 | 0.655 ± 0.004 | 0.53 | 2.0 |
| 2 | 0.657 ± 0.004 | 0.44 | 2.1 |
| 5 | 0.648 ± 0.004 | 0.24 | 2.2 |

Flat for `m` in [-1.5, 2], equal to the free run, which learned `m = -4.8`; worse with
much larger seed variance at `m = -7, -4.5, -3`. The free solution matches the `m = -6`
runs in AUC, `q` and temperature. `q` falls monotonically as `m` rises, so fixing epsilon
moves work into `q`. The SBM does **not** show TNBBeta beating vMF robustly (the
dose-response note's 3-seed check had it 0.018 behind vMF, 0.028 ahead of
power-spherical), which is why the real graphs matter.

### Real graphs, fixed m (featureless, test AUC, 5 seeds)

| dataset | threshold | free-epsilon learned `m` (`f`) | free AUC | fixed-`f` AUC range, `f` in [-0.9, 1] |
|---|---|---|---|---|
| Cora | 32 | -28.9 (-0.90) | 0.838 ± 0.014 | 0.841-0.850 |
| Citeseer | 8 | -5.3 (-0.67) | 0.770 ± 0.018 | 0.762-0.772 |
| Pubmed | 8 | -4.7 (-0.58) | 0.830 ± 0.003 | 0.828-0.832 |
| MAG-CS | 16 | -12.7 (-0.79) | 0.943 ± 0.002 | 0.948-0.950 |

Free training has `frac_bimodal` 0.99-1.00 and `q` 0.91-0.96 everywhere. Per-condition
means and standard deviations are in `apps.eval.fixed_m_graphs_table`'s output; the
differences among fixed conditions are mostly within one standard deviation. On MAG-CS
every fixed condition beats free epsilon by about 0.005-0.007.

### Per-bucket AUC does not depend on m

By degree (training-degree quintiles; b0 is the lowest-degree fifth), TNBBeta at `f = 0.5`
differs from free epsilon by at most about 0.02 in any bucket of the Cora and MAG-CS
tables I tabulated (the bucket-level seed spread was not computed, and Citeseer / Pubmed
were not tabulated by bucket); e.g. MAG-CS free `0.822 0.921 0.945 0.954 0.957` vs. fixed `f in
[-0.9, 1]` `0.829-0.835 0.928-0.933 0.950-0.954 0.959-0.961 0.963-0.964`. AUC rises with
degree identically for every `m` (Cora 0.70 -> 0.88; MAG-CS 0.83 -> 0.96). By number of
distinct neighbor communities (MAG-CS only) AUC also rises with community count
(0.92 -> 0.95) and is likewise independent of `m`, but community count is confounded with
degree (mean degree 2.9 -> 9.7 across the four buckets).

### Matched-dimension baselines (overall test AUC, 5 seeds)

Every family at TNBBeta's own `(lr, dropout, epochs)` and dimension (sphere families ambient
`latent_dim`; the Gaussian `latent_dim - 1`).

| dataset | d | TNBBeta `f=0.5` | TNBBeta free | vMF | power-spherical | Gaussian |
|---|---|---|---|---|---|---|
| Cora | 65 | 0.850 ± 0.007 | 0.838 ± 0.014 | 0.756 ± 0.013 | 0.751 ± 0.013 | 0.851 ± 0.013 |
| Citeseer | 17 | 0.772 ± 0.007 | 0.770 ± 0.018 | 0.732 ± 0.008 | 0.705 ± 0.020 | 0.784 ± 0.011 |
| Pubmed | 17 | 0.832 ± 0.005 | 0.830 ± 0.003 | 0.814 ± 0.007 | 0.790 ± 0.006 | 0.843 ± 0.003 |
| MAG-CS | 33 | 0.949 ± 0.002 | 0.943 ± 0.002 | 0.853 ± 0.004 | 0.836 ± 0.002 | 0.939 ± 0.003 |

Each family at its **own** selected dimension (the Figure 5 grid search): vMF selects
`latent_dim = 9` (d = 8, the floor of the grid) on all four datasets, power-spherical 65, 65,
9, 65, the Gaussian 32-64, TNBBeta 17-65. Test AUC there: Cora vMF 0.781 / PS 0.747 /
Gaussian 0.848 / TNBBeta 0.840; Citeseer 0.748 / 0.700 / 0.786 / 0.770; Pubmed 0.823 / 0.813 /
0.853 / 0.827; MAG-CS 0.894 / 0.861 / 0.943 / 0.943. So giving vMF its preferred low
dimension narrows the gap (Cora +0.06, MAG-CS +0.05, Citeseer +0.02, Pubmed +0.004) but
does not close it on Cora or MAG-CS.

The matched-dimension gap TNBBeta (`f=0.5`) - vMF per degree bucket is +0.10 in every
bucket but the lowest on Cora (+0.046) and +0.10-0.11 in all five on MAG-CS; versus the
Gaussian it is within about 0.02 everywhere except the lowest-degree Citeseer bucket
(-0.048).

### What the parameters communicate (`apps.eval.param_probe`, all five seeds)

Decoding: test AUC from posterior centres (what `run_once` reports) vs. the mean over `S`
posterior samples of `sigmoid(link_logit)`.

| dataset | condition | centre AUC | sampled S=1 | sampled S=16 | `r_bar` (32 draws) |
|---|---|---|---|---|---|
| Cora (d=65) | TNBBeta free | 0.838 | 0.836 | 0.844 | 0.91 |
| | TNBBeta `f=0.5` | 0.850 | 0.847 | 0.854 | 0.82 |
| | vMF | 0.756 | **0.508** | **0.493** | 0.178 |
| | power-spherical | 0.751 | **0.508** | **0.507** | 0.177 |
| | Gaussian | 0.851 | 0.833 | 0.852 | n/a |
| MAG-CS (d=33) | TNBBeta free | 0.942 | 0.941 | 0.944 | 0.97 |
| | vMF | 0.853 | 0.509 | 0.531 | 0.194 |
| | power-spherical | 0.836 | 0.500 | 0.504 | 0.177 |
| Citeseer (d=17) | vMF | 0.732 | 0.541 | 0.607 | 0.253 |
| | power-spherical | 0.705 | 0.507 | 0.522 | 0.193 |
| Pubmed (d=17) | vMF | 0.814 | 0.798 | 0.829 | 0.697 |
| | power-spherical | 0.790 | 0.777 | 0.816 | 0.604 |

`r_bar` is the norm of the mean of 32 samples; exactly uniform samples in high dimension
give about `1/sqrt(32) = 0.177`. Cora's and MAG-CS's vMF / power-spherical posteriors sit
at that floor, i.e. indistinguishable from the prior, while their centres still score
0.75-0.85 (plausibly from the GCN's structure alone, which the sampled path does not
see; untested). The
size of the TNBBeta-vs-vMF gap orders with how collapsed vMF is: Cora and MAG-CS (full
collapse, gap +0.09 / +0.10), Citeseer (partial, +0.04), Pubmed (mild, +0.02). That is four
datasets, correlational.

Parameters vs. covariates (seed means; the three TNBBeta parameters share one `R^2` row
because it is from regressing on all of them jointly):

- **Degree.** Spearman with log training degree is 0.85-0.89 for `|p - 0.5|`, `q` and
  `epsilon` (free) across the four datasets; at `f=0.5` it is 0.90-0.98 for `|p - 0.5|` and
  0.61-0.97 for `q` (Cora's 0.61 is the low end). The Gaussian's mean sigma is -0.88 to
  -0.94. Held-out ridge `R^2` for degree: TNBBeta 0.78-0.97, Gaussian 0.78-0.89.
- **Effective rank** of TNBBeta's parameters (`|p - 0.5|`, `q`, `epsilon`): 1.2-1.5 of 3
  free, 1.1-1.5 of 2 with epsilon fixed. About one scalar. (The Gaussian is probed through
  its mean sigma alone, so it has no effective rank here.)
- **vMF / power-spherical kappa** barely leaves its initial value (vMF `softplus(0) + 1 =
  1.69`, power-spherical `softplus(0) = 0.69`) on Cora and MAG-CS: seed means 1.69-1.99 /
  0.69-1.07 on Cora and 2.72-2.75 / 0.69-0.74 on MAG-CS. On Citeseer they reach 3.1-3.5 /
  0.7-2.0, and only on Pubmed (d=17) do they become concentrated, about 25-31 for both. Where
  kappa moves it tracks degree: Spearman 0.72 (MAG-CS, Citeseer), 0.88 (Pubmed), 0.24 (Cora)
  for vMF; 0.21 / 0.53 / 0.82 / 0.28 for power-spherical on MAG-CS / Citeseer / Pubmed /
  Cora.
- **Beyond degree** (gain in cross-validated `R^2` of degree + parameters over degree
  alone), MAG-CS neighbor communities / neighbor entropy: TNBBeta free +0.07 / +0.10,
  `f=0.5` +0.05 / +0.07, vMF +0.04 / +0.06, power-spherical +0.02 / +0.04, Gaussian +0.01 /
  +0.02. Clustering coefficient: at most +0.02 for the sphere families, +0.07 to +0.11 for the
  Gaussian on Cora and MAG-CS (+0.005 on Pubmed). Small everywhere, and not exclusive to TNBBeta.

## Interpretation

Supported by these data:

- Bimodality is a byproduct of training, not what makes TNBBeta succeed on link prediction.
  This is consistent with how the evaluation works: edges are scored from posterior centres,
  so the posterior's shape reaches the result only through training noise and the KL term.
- TNBBeta and the Gaussian learn a node-specific, degree-tracking confidence and use it
  (sampled AUC about equals centre AUC). vMF and power-spherical, at this dimension and
  budget, mostly do not.

**Not established:** *why* vMF / power-spherical stay at the prior. One hypothesis I raised
in discussion was that kappa is `softplus(raw)` (+1 for vMF), linear in `raw`, so reaching
`kappa ~ d` takes a proportionally large `raw`, whereas `p = sigmoid(raw)` saturates at a
logit of about 3. That is unverified, and the vMF dimension-aware-kappa note in `CLAUDE.md`
is attributed to numeric / training instability, not to this, so it should not be read as
supporting evidence. Distinguishing weak gradient, KL pull-back and parameterization reach
needs per-epoch logging of `raw`, `r_bar`, KL and reconstruction, and a vMF / power-spherical
run with an initialization or parameterization that lets kappa reach the needed range.

## Caveats

- Every fixed-m cell reuses the free-epsilon run's `(lr, dropout, latent_dim)`; none was
  re-tuned per `m`. Baselines likewise reuse TNBBeta's `lr` and `dropout`, so the matched-
  dimension lead is an upper bound relative to each baseline's own tuned result (the "own
  dimension" numbers above include its tuning).
- Five seeds, means and standard deviations only; no significance tests. Real graphs
  featureless only. Community bucketing is MAG-CS only (Planetoid class labels were never
  downloaded).
- `r_bar` at d=17 is not near the uniform floor even for a fully uniform posterior (the
  floor argument is a high-dimensional one); the Citeseer and Pubmed rows are less clear-cut
  than Cora and MAG-CS.
- The ridge `R^2` is on rank-transformed variables and is a simple, linear probe; a
  nonlinear one could recover more.
- The earlier whole-graph `m` and degree-vs-`m` numbers (`apps.eval.graph_posterior_shape`,
  and the SBM figure in the paper draft) normalize the **full** graph's adjacency, but the
  models were trained on the 85% train-edge graph. Everything in this note uses the train
  adjacency, matching `run_once`; those earlier figures should be re-derived the same way.

## Incidents

**Node `k003`.** Most tasks of the first real-graph submission (`2278702`) that landed on
`k003` died within seconds with `No CUDA GPUs are available`. They passed `--device cuda`
explicitly, which bypasses the repo's no-GPU handling (exit 75 and resubmit excluding the
node), so the failures exited 1 and the scheduler kept feeding the array to the bad node.
Fixed by dropping the explicit flag; resubmitted with `--exclude=k003`. `k003` looks
unhealthy and is worth reporting.

**Incident: overwritten result files.** The free-epsilon cells of the real-graph sweep
wrote their summary to `link_prediction/<dataset>_nofeat_tnbbeta.json`, the same path that
holds the full TNBBeta featureless grid search behind Figure 5, replacing the four files
(Cora, Citeseer, Pubmed, MAG-CS) with single-seed, single-configuration summaries between
13:52 and 13:57 on 2026-10-30. They were rebuilt from the original jobs' logs
(`slurm_logs/link-prediction-run_2120348/2120352/2120356/2120360.out`, which hold all 36
per-configuration summaries and the `Selected` line each), with `seeds` set to the script's
default `[0, 1, 2, 3, 4]`. Verified three ways: for each dataset the best-validation
configuration equals the logged `Selected` line, the selected `(lr, dropout, latent_dim)`
equal what was read from the files before the overwrite, and the selected test AUCs agree to
four decimals (Cora 0.8402, Citeseer 0.7699, Pubmed 0.8265, MAG-CS 0.9428). The overwritten
versions were kept in the session scratchpad. `apps.link_prediction.main --results-name`
(with a regression test) now keeps any single-run or per-seed invocation off those paths;
the sweep and baseline scripts pass it. The sweep as run did not have it.

## Open items

- **MNIST fixed-m (job `2278361`):** at the time of writing 12 of 21 tasks completed, 7
  failed with `FloatingPointError: Non-finite loss or gradient` (epochs 1-10), 2 still
  running; not analyzed here. All three `m = -2` (`epsilon = 0.5`) cells failed, plausibly
  expected; seed 2 failed at `m = 1, 3, 8, 20` too, which is unexplained (no shared node; a
  CPU rerun of `m = 20`, seed 2 completed two epochs without a non-finite value, pointing
  at GPU numerics). A diagnostic job (`2278689`, anomaly detection on a GPU) is still queued
  on the single-node `dev` partition.
- **Experiments not run:** per-epoch trajectory logging for vMF / power-spherical / TNBBeta
  (`raw`, `r_bar`, KL, reconstruction); a vMF / power-spherical run with a parameterization
  or initialization that lets kappa reach the required range; a KL-weight sweep across
  families; a degrees-of-freedom ladder for TNBBeta (`q` fixed); Planetoid label download
  for community bucketing on Cora / Citeseer / Pubmed.
