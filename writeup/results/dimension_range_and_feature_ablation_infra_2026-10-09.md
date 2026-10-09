# Dimension-range fix and feature-ablation infrastructure, 2026-10-09

**This is an infrastructure log, not a results write-up.** No real training ran as
part of this entry -- the two methodology fixes below (and the consolidated sweep
command) are deferred to the user's cluster. Read this as "what got built and why,"
not "what it found."

- **Date:** 2026-10-09
- **Branch:** `dtd-oriented-textures`, built on top of this branch's own
  `dimension_sweep_and_mag_coauthor_infra_2026-10-09.md` entry (commit range
  `924c3d5`..`4cc47e9`). This entry's own commits are listed at the end.

## Why

`dimension_sweep_and_mag_coauthor_infra_2026-10-09.md`'s sweep and MAG co-authorship
infrastructure produced real results that are genuinely inconsistent across
datasets -- TNBBeta ties on Cora/Citeseer, wins clearly on com-DBLP, loses clearly on
MAG CS, and Pubmed is currently uninterpretable (stale data, exploding variance). The
user does not want a premature tidy narrative forced onto that inconsistency -- the
inconsistency itself needs explaining. Two concrete fixes push that investigation
further without yet resolving it:

1. **The dimension range was wrong relative to the paper.** Checked directly against
   the S-VAE paper (Davidson et al. 2018, Section F.3): the real protocol is `dz in
   {8, 16, 32, 64}` for the Gaussian family, with the paper itself dropping `dz=64`
   for the sphere family only, due to a 2018-era memory constraint. This project's
   `link_prediction.sbatch` used `(16 32 64)` for every family, missing `d=8`
   entirely -- inherited unexamined from this project's own earlier default, never
   actually checked against the paper until now.
2. **"Real features vs. none" was confounded with "which dataset."** Whether TNBBeta
   wins or loses differs across datasets that differ in many other ways (size, degree
   distribution, whether features exist at all). Toggling a single dataset's real
   features on/off, with its graph held exactly fixed, isolates that one variable.

## What was built

### Fix 1: the dimension range (`scripts/slurm/link_prediction.sbatch`)

`MANIFOLD_DIMS=(16 32 64)` -> `MANIFOLD_DIMS=(8 16 32 64)`, applied uniformly to
every family (the user explicitly chose NOT to reproduce the paper's `dz=64`
sphere-memory omission -- every family keeps `64` here). The array also grew a
fourth dataset, `mag_cs` (this branch's own prior commit added the loader and wired
it into `apps.link_prediction.main`/`apps.eval.graph_posterior_shape`, but hadn't yet
added it to this sbatch script's array): 4 datasets x 4 families = 16 tasks (`0-15`,
previously `0-11`). Tested the same way as before (see
`writeup/results/dimension_sweep_and_mag_coauthor_infra_2026-10-09.md`'s Part 1): a
stubbed `uv` records the exact command line built for each `SLURM_ARRAY_TASK_ID`,
now covering all four datasets and the new dimension list
(`tests/scripts/test_link_prediction_sbatch.py`).

### Fix 2: an `--ignore-features` toggle

- **`tnbbeta_vae.data.snap_community.identity_features(num_nodes) ->
  torch.Tensor`**: factored out of `load_snap_community`'s inline sparse-identity
  construction (a pure refactor -- `load_snap_community`'s own tests pass
  unchanged) so it's reusable wherever a featureless stand-in is needed.
- **`apps/link_prediction/main.py --ignore-features`**: after loading the graph
  (any `--dataset`), replaces `graph.features` with
  `identity_features(graph.features.shape[0])` before the grid search loop reads
  `graph.features.shape[1]` for `GraphVAEConfig.in_features` -- confirmed directly
  (not assumed) that this makes `in_features` track `num_nodes` automatically, with
  no special-casing needed anywhere else in the grid search or `run_once`. Every
  output path and checkpoint name derived from the dataset uses
  `label = f"{args.dataset}_nofeat" if args.ignore_features else args.dataset`
  instead of `args.dataset` directly, so a features-off run's
  `<label>_<family>.json` (and any `--run-name`-based checkpoint the user names
  following the same convention) never collides with its features-on counterpart.
  Confirmed `apps/link_prediction/table.py` needs no changes: it already takes an
  arbitrary `--datasets` list of plain strings and looks up `<string>_<family>.json`
  directly, so `--datasets cora cora_nofeat` works once a features-off run has
  written `cora_nofeat_<family>.json` -- verified with an actual test, not just
  trusted.
- **`apps/eval/graph_posterior_shape.py --ignore-features`**: same override,
  applied in `_load_batch` before `graph_to_batch` builds the `GraphBatch`, so the
  diagnostic can rebuild the identical kind of batch (identity features) a
  features-off checkpoint was actually trained with. `--dataset` still names the
  real graph topology to load; `--run-name` is unaffected (whatever the user named
  the training run, e.g. `cora_nofeat_tnbbeta_seed0`).

Only Cora, Citeseer, Pubmed and MAG CS can take `--ignore-features` meaningfully
(they have real features to toggle off); com-DBLP and Amazon already have no
features (SNAP's own identity-features convention, now literally the same
`identity_features` call) and are out of scope for this change, not something to
add real features to.

Tests mock a tiny fake graph exactly as this project's other
`apps/link_prediction`/`apps/eval` tests already do -- asserting directly on: default
behavior (`--ignore-features` omitted) is unchanged; with the flag, the batch
actually built has identity features of the right shape; output file naming uses the
`_nofeat` suffix; `in_features` in the resulting config is `num_nodes`, not the
original feature count.

## Verification

`uv run ruff format .`, `uv run ruff check .`, `uv run pyright`, `uv run pytest`: all
clean, full suite (581 passed).

## The factorial sweep (the actual next step, not run here)

4 datasets (Cora, Citeseer, Pubmed, MAG CS) x 2 feature settings (real / identity,
MAG CS and Planetoid only -- com-DBLP has no features to toggle) x 4 dimensions
(8, 16, 32, 64 manifold, i.e. `d`/`d+1` per family) x 4 families (Gaussian, vMF,
TNBBeta, Power Spherical).

### Prerequisite downloads

```
uv run python -m apps.data.download_planetoid
uv run python -m apps.data.download_mag_coauthor --datasets cs
```

(com-DBLP is already covered by the prior round's `apps.data.download_snap_community`
if a com-DBLP features-on/off comparison is wanted alongside this sweep -- com-DBLP
itself only ever runs in its one (featureless) form.)

### Kicking off the sweep

`scripts/slurm/link_prediction.sbatch`'s array already covers
(dataset x family) x 4 dimensions in one `sbatch` submission; it does not yet have a
features-on/off axis built in (its array would double to 32 tasks for an
8-dimension-list-doubled job, which is a reasonable size but changes what every
existing array index means and a running job referencing a fixed array size --
more disruptive than it's worth for what is, after all, a single boolean flag on
top of an already-correct per-dataset command). The next-cleanest alternative: run
the existing array job for features-on (unchanged), then one more pass over just the
four feature-toggleable datasets with `--ignore-features` via
`link_prediction_run.sbatch` (which already forwards arbitrary extra args,
`--ignore-features` included, with no script changes needed):

```
# Features-on: all 4 datasets x 4 families x the 4-dimension sweep, one sbatch call.
sbatch scripts/slurm/link_prediction.sbatch

# Features-off: the 4 feature-toggleable datasets x 4 families, same dimension
# convention, one call per (dataset, family) cell (ambient dims = manifold dims
# directly for gaussian, +1 for vmf/tnbbeta/power_spherical, per the same d/d+1
# convention link_prediction.sbatch uses).
for DATASET in cora citeseer pubmed mag_cs; do
    sbatch scripts/slurm/link_prediction_run.sbatch "${DATASET}" gaussian \
        --latent-dims 8 16 32 64 --ignore-features
    sbatch scripts/slurm/link_prediction_run.sbatch "${DATASET}" vmf \
        --latent-dims 9 17 33 65 --ignore-features
    sbatch scripts/slurm/link_prediction_run.sbatch "${DATASET}" tnbbeta \
        --latent-dims 9 17 33 65 --ignore-features
    sbatch scripts/slurm/link_prediction_run.sbatch "${DATASET}" power_spherical \
        --latent-dims 9 17 33 65 --ignore-features
done
```

(16 tasks via the one `sbatch` call above, plus 16 more via the loop's 16
`link_prediction_run.sbatch` submissions -- 32 total cells, matching the factorial
design; each writes its own `$TNBBETA_CHECKPOINT_DIR/link_prediction/
<label>_<family>.json`, `<label>` being `<dataset>` or `<dataset>_nofeat`.)

### `graph_posterior_shape` diagnostic, with matching `--ignore-features`

Needs a `--run-name` checkpoint, so run `apps.link_prediction.main` directly (not
through the table-only commands above) for whichever cells need a saved checkpoint,
e.g.:

```
uv run python -m apps.link_prediction.main --dataset cora --family tnbbeta \
    --ignore-features --lrs 0.01 --dropouts 0 --latent-dims 17 --epochs 200 \
    --seeds 0 --run-name cora_nofeat_tnbbeta

uv run python -m apps.eval.graph_posterior_shape \
    --run-name cora_nofeat_tnbbeta_seed0 --dataset cora --ignore-features \
    --device cpu
```

(Omit `--ignore-features` from both commands for the matching features-on
checkpoint/diagnostic pair; repeat per dataset/family/seed as needed.)

### Viewing the results

```
uv run python -m apps.link_prediction.table \
    --datasets cora cora_nofeat citeseer citeseer_nofeat pubmed pubmed_nofeat \
    mag_cs mag_cs_nofeat \
    --by-dimension
```

(`table.py` needed no code changes for this -- it already resolves each dataset
string to its own `<string>_<family>.json` independently, so the features-on and
features-off runs for the same underlying dataset show up as separate adjacent rows,
not merged or compared automatically.)

## Deviations from the task description

- `link_prediction.sbatch`'s array was not extended to a features-on/off axis
  in-place (which would have required doubling its array size and reinterpreting
  every existing index). The sweep command set above runs features-on through the
  existing array job unchanged and features-off through a small loop of
  `link_prediction_run.sbatch` calls (which already forwards `--ignore-features`
  with no script changes) -- fewer total commands than enumerating every cell by
  hand, without changing what the existing array job's indices mean.
