# Bipartite slope check: TNBBeta advantage vs mean degree, degree CV, clustering (2026-11-03)

**DESCRIPTIVE ONLY.** 20 fresh bipartite configs, one family, 16 draws each, no
confirmatory claim, no multiplicity correction. Permutation p-values are uncorrected
two-sided and only say how unusual the rank correlation is among 20 points.

- Base commit: `dd8ede0` (branch `synthetic-graph-families`; working tree had other uncommitted files).
- Command (about 20 s CPU, login node; no training, reads cached `runs_fresh.json` / `covariates_fresh.json`):
  `uv run python -m apps.eval.bipartite_slope_check` (script `apps/eval/bipartite_slope_check.py`, uncommitted).
- Advantage per config and metric: per draw `comparator - tnbbeta` (lower-better metrics) or
  `tnbbeta - comparator` (AUC), for the comparators `vmf_init` and `power_spherical_init`; the
  per-config **median** over the 16 draws for each comparator, then the minimum over the two
  (the less favourable comparator, as in the study). Positive = TNBBeta better. Note the study's `E_c`
  uses the mean over draws; this check uses the median as requested, so values differ slightly.
- Covariates are the study's (mean over draws): `realised_degree` (mean degree), `expected_degree_cv`
  (CV of expected degree), `clustering`. Permutations: 2,000 for Spearman, 2,000 for partials; seed 20261103.

## Result

The slope is **flat** for truth KL: Spearman vs mean degree +0.057 (perm p 0.82), vs degree CV +0.089
(p 0.72), vs clustering +0.314 (p 0.19). Nothing within bipartite reaches even uncorrected
significance on any of the five metrics. Signs for the probability metrics (truth KL, ECE, log loss)
are slightly positive in degree, but all |rho| <= 0.16 for degree. The only largest-magnitude
associations are sampled AUC (degree +0.38, CV -0.38, p about 0.10 each) and clustering vs truth KL
(+0.31), none distinguishable from noise at n = 20. So the H1 within-family degree slope
(+0.46 over all 200 fresh configs) is **not visible within bipartite alone**; this does not show it is absent
(n = 20 gives Spearman SE about 0.23).

| metric (advantage) | median | configs > 0 | Spearman vs degree (p) | vs CV (p) | vs clustering (p) | Kendall deg / CV / clust | partial deg given CV (p) | partial CV given deg (p) |
|---|---|---|---|---|---|---|---|---|
| truth_kl | +0.0057 | 13/20 | +0.057 (0.82) | +0.089 (0.72) | +0.314 (0.19) | +0.042 / +0.063 / +0.232 | +0.080 (0.74) | +0.105 (0.67) |
| ece | +0.0442 | 18/20 | +0.161 (0.50) | -0.023 (0.92) | +0.206 (0.38) | +0.095 / -0.011 / +0.095 | +0.160 (0.51) | +0.016 (0.95) |
| sampled_log_loss | +0.0138 | 17/20 | +0.137 (0.58) | -0.092 (0.70) | +0.087 (0.72) | +0.116 / -0.074 / +0.074 | +0.119 (0.61) | -0.062 (0.80) |
| sampled_auc | -0.0048 | 8/20 | +0.382 (0.11) | -0.380 (0.09) | -0.039 (0.89) | +0.274 / -0.253 / -0.105 | +0.326 (0.17) | -0.324 (0.18) |
| centre_auc | -0.0118 | 6/20 | -0.041 (0.86) | +0.248 (0.28) | +0.311 (0.17) | -0.011 / +0.200 / +0.179 | +0.018 (0.94) | +0.246 (0.33) |

Kendall p-values (truth KL): 0.82 / 0.73 / 0.17. Partial = rank-based (Spearman on residuals of ranks on the ranks of the other covariate).

Within bipartite, mean degree and degree CV are only weakly (negatively) related: Spearman -0.233
(p 0.32), Kendall -0.179, so they are not collinear here and the partials are about the marginals.
Clustering correlates with CV (+0.47) and weakly with degree (+0.14).

## Reachability of the low-density / high-heterogeneity cell

Generator (`src/tnbbeta_vae/data/graph_space.py`, `apps/synthetic/graph_search.py::sample_configs`):
bipartite is a two-side block model (`1 + t * [side_i != side_j]`), mean degree log-uniform 4-40, and
`heterogeneity in {none, mild, heavy}` supplies mean-one Pareto degree weights with tail index
{none, 6, 2.5} (`block_graph(degree_exponent=...)`); only `scale_free` / `degree_corrected` override it.
So degree-correction is available to bipartite and the cell is reachable by design. Realised in the 20
fresh configs: mean degree 4.95-38.9, degree CV 0.00-0.78 (heterogeneity none 5, mild 6, heavy 9;
CV about 0 / 0.20 / 0.63-0.78), clustering 0.000-0.136. Covariate levels are essentially three CV
bands, not a continuum. Low density (< 11) x high CV (> 0.6): bipartite-09, -18, -02, -13 (4 configs;
truth-KL advantage +0.034, +0.030, -0.013, +0.141). High density (> 20) x high CV: bipartite-15, -04, -01
(-0.023, -0.003, +0.091). Low density x CV 0: -16, -05 (+0.000, +0.006). High density x CV 0: -08, -07 (+0.011, +0.003).
Only 2-4 configs per corner, so the corners cannot be compared with any power.

## Scatter table (sorted by degree; het = heterogeneity level 0/1/2; advantages = median, less favourable comparator)

| config | het | degree | CV | clustering | truth_kl | ece | sampled_log_loss | sampled_auc | centre_auc |
|---|---|---|---|---|---|---|---|---|---|
| bipartite-09 | 2 | 5.0 | 0.68 | 0.007 | +0.0340 | +0.0505 | +0.0175 | -0.0263 | -0.0322 |
| bipartite-18 | 2 | 6.4 | 0.71 | 0.011 | +0.0302 | +0.0441 | +0.0149 | -0.0326 | -0.0282 |
| bipartite-00 | 1 | 6.5 | 0.20 | 0.006 | -0.0033 | +0.0196 | +0.0036 | -0.0013 | -0.0237 |
| bipartite-16 | 0 | 6.7 | 0.00 | 0.004 | +0.0003 | +0.0080 | +0.0082 | +0.0050 | +0.0001 |
| bipartite-05 | 0 | 7.2 | 0.00 | 0.004 | +0.0055 | +0.0547 | +0.0172 | -0.0091 | -0.0109 |
| bipartite-11 | 1 | 8.1 | 0.20 | 0.004 | -0.0019 | +0.0523 | +0.0173 | +0.0004 | -0.0068 |
| bipartite-02 | 2 | 9.8 | 0.72 | 0.054 | -0.0130 | -0.0257 | -0.0016 | +0.0358 | +0.0015 |
| bipartite-13 | 2 | 10.7 | 0.77 | 0.036 | +0.1406 | +0.1633 | +0.0943 | -0.0239 | +0.0043 |
| bipartite-03 | 0 | 13.6 | 0.00 | 0.015 | +0.0059 | +0.0442 | +0.0103 | -0.0197 | -0.0127 |
| bipartite-19 | 2 | 14.4 | 0.76 | 0.051 | +0.0520 | +0.0847 | +0.0177 | -0.0652 | -0.0208 |
| bipartite-10 | 2 | 17.8 | 0.78 | 0.054 | -0.0077 | +0.0259 | -0.0026 | -0.0577 | -0.0068 |
| bipartite-06 | 1 | 18.6 | 0.20 | 0.031 | -0.0038 | +0.0343 | +0.0102 | -0.0227 | -0.0351 |
| bipartite-15 | 2 | 21.0 | 0.63 | 0.000 | -0.0233 | +0.0099 | -0.0006 | -0.0398 | -0.0867 |
| bipartite-04 | 2 | 23.1 | 0.70 | 0.001 | -0.0029 | -0.0009 | +0.0002 | +0.0091 | +0.2639 |
| bipartite-14 | 1 | 26.3 | 0.20 | 0.017 | +0.0076 | +0.0168 | +0.0038 | +0.0179 | +0.1784 |
| bipartite-17 | 1 | 28.0 | 0.20 | 0.028 | +0.0125 | +0.0384 | +0.0133 | -0.0076 | -0.0201 |
| bipartite-01 | 2 | 29.7 | 0.66 | 0.136 | +0.0905 | +0.0753 | +0.0795 | +0.0517 | +0.0008 |
| bipartite-12 | 1 | 34.3 | 0.19 | 0.013 | +0.0201 | +0.0634 | +0.0235 | +0.0212 | -0.0746 |
| bipartite-08 | 0 | 35.8 | 0.00 | 0.028 | +0.0106 | +0.0498 | +0.0143 | -0.0019 | -0.0068 |
| bipartite-07 | 0 | 38.9 | 0.00 | 0.000 | +0.0026 | +0.0718 | +0.0236 | +0.0194 | -0.0921 |

## Caveats

- Descriptive: 20 configs, one family, median instead of the study's mean, uncorrected p-values.
- Two bipartite configs (-13: +0.141, -01: +0.091) dominate truth-KL advantage; rank statistics reduce but do not remove their leverage.
- Covariates are realised/measured, not the sampled design axes; bipartite CV is effectively 3-level.
- Within the 20 configs TNBBeta's truth-KL median advantage is positive (+0.0057) but its AUC advantage is negative, consistent with the family table in `graph_search_metric_first_results_2026-11-02.md`.
