# Real-checkpoint neighbor-heterogeneity correlation: infrastructure only, 2026-10-09

- **Date:** 2026-10-09
- **Branch/commit:** `neighbor-heterogeneity-mechanism`, `e153cf5`
  (`Add apps/eval/neighbor_heterogeneity_correlation.py`). `master` did not yet
  have this code.

## What this is

`apps/eval/neighbor_heterogeneity_correlation.py` is the real-checkpoint analogue
of `sbm_pout_dose_response_2026-10-09.md`'s per-node heterogeneity-vs-`m`
correlation, built to run against an actual community-labeled real-world graph
(com-DBLP or MAG CS) rather than the synthetic SBM. **No com-DBLP or MAG-CS
checkpoint exists locally** -- these require the UChicago DSI cluster (com-DBLP:
317,080 nodes; MAG CS: 21,957 nodes, per `src/tnbbeta_vae/data/mag_coauthor.py`'s
docstring) -- so this script was built and tested entirely against mocked tiny
graphs/checkpoints (`tests/apps/test_neighbor_heterogeneity_correlation.py`),
exactly like this project's other `apps/eval` scripts (e.g.
`graph_posterior_shape.py`), and has not been run for real.

## What it computes, once real checkpoints exist

For a trained `GraphVAE` checkpoint and the matching real graph: per-node `m =
epsilon - (latent_dim - 1) / 2` (TNBBeta only; prints a message and skips for any
other family) and per-node neighbor-heterogeneity
(`tnbbeta_vae.data.neighbor_heterogeneity.neighbor_heterogeneity`, fed the graph's
real overlapping community labels via `load_snap_communities`/`remap_communities`
for `dblp`/`amazon`, or `load_mag_communities` directly for `mag_cs`/`mag_eng`/
`mag_chem`/`mag_med`), then the Pearson and Spearman correlation between them.
Writes `neighbor_heterogeneity_<checkpoint>.json` and, for TNBBeta, a plotly
heterogeneity-vs-`m` scatter (`neighbor_heterogeneity_<checkpoint>.html`) next to
the checkpoint.

## The exact command to run this for real (once a cluster checkpoint exists)

Against an existing com-DBLP TNBBeta checkpoint (e.g. the one
`writeup/results/dblp_bridge_diagnostic_2026-10-01.md` already trained):

```
uv run python -m apps.eval.neighbor_heterogeneity_correlation \
    --run-name <dblp-tnbbeta-run-name> --dataset dblp --device cpu
```

Or against a MAG CS TNBBeta checkpoint (e.g. from the
`features_onoff_dimension_sweep_2026-10-09.md` sweep, on
`features-overlap-investigation`):

```
uv run python -m apps.eval.neighbor_heterogeneity_correlation \
    --run-name <mag_cs-tnbbeta-run-name> --dataset mag_cs --device cpu
```

Both need `apps/data/download_snap_community.py`/`apps/data/download_mag_coauthor.py`
to have already populated `$TNBBETA_DATA_DIR/snap_community`/`mag_coauthor` (same
prerequisite `graph_posterior_shape.py` and `dblp_bridge_diagnostic.py` already
have for these datasets), and the relevant checkpoint directory under
`$TNBBETA_CHECKPOINT_DIR/<run-name>_seed<N>/` (or whatever naming the specific
training run used) to already exist.

## Why this one is infra-only

Per the task's own framing, this script "needs REAL cluster checkpoints (com-DBLP,
MAG CS) that don't exist locally"; it was explicitly scoped to be "built and
tested with mocked tiny graphs/checkpoints only" and not run for real in this
round. `sbm_pout_dose_response_2026-10-09.md` covers the equivalent correlation on
fully-controlled synthetic data instead, where every ground-truth quantity is
known exactly -- see that write-up for the actual (mixed/honest) result on the
sign of this correlation.
