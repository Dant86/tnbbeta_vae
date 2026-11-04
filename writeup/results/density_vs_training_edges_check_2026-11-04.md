# Does the truth-KL advantage track mean degree, node count or edge count? (exploratory)

Date: 2026-11-04. Code base: `master` commit `d5bb583` (PR #72) plus the new script
`apps/eval/edges_vs_degree_check.py` (this PR). Exploratory, not pre-registered, no new training:
it re-analyses the 200 fresh-stage configs of the metric-first graph search
(`graph_search_metric_first_results_2026-11-02.md`).

Command (login node, light; needs the fresh-stage runs under `$TNBBETA_CHECKPOINT_DIR/graph_search`):

    uv run python -m apps.eval.edges_vs_degree_check

## Why

In the density figure (`scale_free_density_figure_results_2026-11-03.md`) the node count is fixed
(n = 600), so a higher mean degree also means more edges to train on (about 1,530 / 3,825 / 8,900
training edges, 85% of the realised edges) and more held-out test pairs (about 180 / 450 / 1,050 test
positives, each with an equal number of negatives). The registered H1 (within-family Spearman +0.462
of `S_c = E_c / KL0` against log nominal degree) has the same exposure to the extent that degree and
edge count co-vary across configs. Truth KL itself is scored on every non-training pair (about 171k to
178k pairs at n = 600), so test-set size cannot bias it; the open question is training-set size.

## Method

For the 200 fresh configs (10 families, 20 each): within-family rank correlations (family means of the
ranks removed, as in the registered H1 statistic) of `S_c` with log nominal degree, log n and log
realised edges `n * realised_degree / 2`; partial rank correlations of `S_c` with each of log degree
and log n given the other; ranks of `S_c` regressed on the standardised ranks of log degree and log n.
95% intervals: 4,000 bootstrap resamples of configs within family. One permutation p-value (3,000
within-family shuffles of `S_c`) for the partial correlation of degree given n; it is an unadjusted
permutation, not Freedman-Lane, so the bootstrap intervals are the primary evidence.

## Results

Within a family the three predictors are not collinear in the way the density figure's design is:
the configs sampled n and degree independently (Sobol grid), so corr(log degree, log n) = -0.02;
corr(log degree, log edges) = +0.85 and corr(log n, log edges) = +0.48.

| quantity | estimate | 95% bootstrap interval |
|---|---|---|
| rho(`S_c`, log degree) (reproduces the registered +0.462) | +0.462 | [+0.351, +0.562] |
| rho(`S_c`, log n) | -0.125 | [-0.263, +0.016] |
| rho(`S_c`, log edges) | +0.328 | [+0.195, +0.446] |
| partial rho(`S_c`, log degree given log n) | +0.463 | [+0.351, +0.566] |
| partial rho(`S_c`, log n given log degree) | -0.129 | [-0.262, +0.006] |
| rank regression beta, log degree | +0.460 | [+0.347, +0.563] |
| rank regression beta, log n | -0.114 | [-0.235, +0.005] |
| rho(`E_c` in nats, log degree) | +0.479 | [+0.368, +0.576] |
| rho(`E_c` in nats, log edges) | +0.357 | [+0.228, +0.470] |

One-sided permutation p for partial rho(`S_c`, log degree given log n): 0.0003 (unadjusted).

## Reading

* Controlling for n leaves the degree association unchanged (+0.463 against +0.462). n itself has
  no positive association; its estimate is slightly negative with an interval that includes 0. If
  the advantage came from the amount of training data, n would be expected to help about as much
  as degree does.
* Log edges (+0.33) is a weaker predictor than log degree (+0.46), consistent with degree being the
  driver and edge count riding along with it.
* Across the 200 configs, therefore, the advantage follows mean degree and not network size or
  training-edge count.

## Limits

* n spans only about a 4x range (300 to 1,200) against a 10x range of degree (4 to 40); the interval
  for log n includes small positive values, so low power on n.
* Across-config variation with other factors varying too (blocks, heterogeneity, target AUC); this
  separates degree from n and edge count, but it is not an intervention.
* It says nothing about the three-config scale-free figure at fixed n = 600, where density and
  training edges move together by construction, nor about the test-set-size dependence of sampled
  AUROC and sampled log loss (test pairs grow with edges).
* A controlled n x degree design (for example n in {600, 3,000}, degree in {6, 30}, which has two cells
  with about 9,000 edges at different densities) would confirm it with an intervention; not run.
* Real graphs: the real-graph revision (`real_graph_revision_results_2026-11-03.md`) found no such
  density effect under edge thinning, so this result concerns the synthetic generator only.
