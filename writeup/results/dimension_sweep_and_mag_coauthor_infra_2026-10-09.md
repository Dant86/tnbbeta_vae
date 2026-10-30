# Dimension-sweep and MAG co-authorship infrastructure, 2026-10-09

**This is an infrastructure log, not a results write-up.** No training ran on a
real Cora/Citeseer/Pubmed dimension sweep and no real training ran on the MAG
co-authorship data as part of this entry -- both are explicitly deferred to the
user's cluster. There is no results table below because there are no real results
yet. Read this as "what got built and why," not "what it found."

- **Date:** 2026-10-09
- **Branch/commit range:** `dtd-oriented-textures`, `924c3d5`..`7c438cf` (four
  commits: the `mag_coauthor` loader, its download script, wiring the dataset into
  `apps/link_prediction/main.py` and `apps/eval/graph_posterior_shape.py`, and the
  `link_prediction.sbatch` dimension-sweep fix).

## Why

`graph_bimodality_investigation_summary_2026-10-08.md` closed with a triangulated
mechanistic conclusion (a GCN encoder aggregating over community-correlated graph
structure is sufficient for `TNBBetaSpherical`'s bimodality on `GraphVAE`
link-prediction runs) and an open, increasingly narrow next ablation. Continuing
to isolate the mechanism with more synthetic ablations was getting harder to do
cleanly, so the user chose to pivot: build a broader, systematic empirical picture
(a proper 4-family dimension sweep on the existing Planetoid datasets, plus a new
dataset that is less synthetic/confound-prone than the ablations so far) instead of
another single narrow mechanistic test. Both pieces below are infrastructure only
-- the sweeps and real training run on the user's cluster, not here.

## Part 1: Planetoid dimension-sweep infrastructure

`scripts/slurm/link_prediction.sbatch` already ran one `apps.link_prediction.main`
grid search per (dataset, family) task, and `apps.link_prediction.main` already
supports `--latent-dims` as a list and runs its own internal
(lr, dropout, latent_dim) grid search, keeping every cell's result (not just the
validation-selected one) in its output JSON's `"configurations"` list -- i.e. the
per-dimension breakdown a dimension sweep needs was already there once
`--latent-dims` is passed correctly. So the actual gap, once read carefully, was
exactly what the task predicted it might be: the sbatch script wasn't passing a
family-aware dimension list at all (every family defaulted to
`apps.link_prediction.main`'s own default `--latent-dims 16 32 64`, identical
across families), and `power_spherical` wasn't in the family array.

Fixed both, mirroring `scripts/slurm/mnist_sweep.sbatch`'s own d/d+1 convention
(CLAUDE.md: "Sphere models' `latent_dim` is the AMBIENT dimension... To compare
with the paper at the same d as the Gaussian, give the sphere models
`latent_dim = d + 1`"): `MANIFOLD_DIMS=(16 32 64)` are the Gaussian's own "d"; the
Gaussian family's `--latent-dims` are those dims directly, and every sphere family
(`vmf`, `tnbbeta`, `power_spherical`) gets each dim + 1. The job array grew from 9
tasks (3 datasets x 3 families) to 12 (3 datasets x 4 families).

No changes were needed to `apps/link_prediction/main.py`'s grid search itself, or
to `run_once` -- confirming the task's own suspicion that duplicating that logic
wasn't necessary.

Tested the same way `tests/scripts/test_mnist_sweep_sbatch.py` tests its sbatch
script: a stubbed `uv` records the exact command line built for each
`SLURM_ARRAY_TASK_ID`, asserting the right dataset/family/dimension-list
combination at several indices spanning all three datasets and all four families
(see `tests/scripts/test_link_prediction_sbatch.py`).

## Part 2: MAG co-authorship dataset (`src/tnbbeta_vae/data/mag_coauthor.py`)

### Why this dataset

The investigation's synthetic ablations (Gaussian-blob clusters, a synthetic SBM
graph) were good for mechanistic isolation but are, definitionally, confounds for
the eventual paper's broader story: a reviewer can always ask whether the
synthetic structure happens to favor TNBBeta's bimodality by construction. com-DBLP
has real overlapping communities but no real features (SNAP ships it with sparse
identity "features" as a featureless stand-in); Planetoid has real features but no
overlapping community structure. The NOCD paper's (Shchur & Guennemann,
"Overlapping Community Detection with Graph Neural Networks") four Microsoft
Academic Graph co-authorship networks have both at once: real bag-of-keyword
features AND genuine overlapping ground-truth community memberships, on a single
large real graph. That is the missing combination.

### The real `.npz` format (verified directly, not assumed)

A single file was downloaded once
(`https://raw.githubusercontent.com/shchur/overlapping-community-detection/master/data/mag_cs.npz`,
confirmed live via `curl -sIL`, HTTP 200, ~15.2MB) specifically to inspect its
structure with `numpy.load(path, allow_pickle=True).files` and each array's
shape/dtype, then deleted immediately after recording what follows (no multi-MB
data file is committed or left in this worktree). The guessed key structure in the
task description (the "Pitfalls of Graph Neural Network Evaluation" SparseGraph
lineage) was correct in substance but not in exact key spelling -- the real keys
use a `<matrix_name>.<field>` naming, not `adj_data`/`adj_indices` etc.:

| Key | dtype | shape (on `mag_cs.npz`) | Meaning |
|---|---|---|---|
| `adj_matrix.data` | float32 | (193500,) | scipy CSR adjacency, binary `{0,1}` data |
| `adj_matrix.indices` | int32 | (193500,) | CSR column indices |
| `adj_matrix.indptr` | int32 | (21958,) | CSR row pointers (`n + 1` = 21,958 rows) |
| `adj_matrix.shape` | int64 | (2,) = `[21957, 21957]` | 21,957 nodes |
| `attr_matrix.data` | float32 | (1267484,) | scipy CSR features, bag-of-keyword **counts** (values 1-10+, not binary) |
| `attr_matrix.indices` | int32 | (1267484,) | CSR column indices |
| `attr_matrix.indptr` | int32 | (21958,) | CSR row pointers |
| `attr_matrix.shape` | int64 | (2,) = `[21957, 7793]` | 7,793 keyword columns |
| `labels.data` | float32 | (41243,) | scipy CSR community membership, binary `{0,1}` |
| `labels.indices` | int32 | (41243,) | CSR column indices |
| `labels.indptr` | int32 | (21958,) | CSR row pointers |
| `labels.shape` | int64 | (2,) = `[21957, 18]` | 18 overlapping communities |
| `node_names` | `<U8` | (21957,) | each row's original (opaque) MAG author ID; unused |
| `attr_names` | `<U70` | (7793,) | keyword vocabulary; unused |
| `class_names` | object (`str`) | (18,) | community names (`bioinformatics`, `machine_learning`, ... ); unused |
| `edge_attr_matrix`, `edge_attr_names`, `metadata` | object | scalar | always `None` |
| `type` | `<U11` | scalar | always the literal string `"SparseGraph"` |

Confirmed directly against the file (not assumed): `adj_matrix` is symmetric
(`(adj != adj.T).nnz == 0`) with zero diagonal (no self-loops); 96,750 edges
(`193500 / 2`, matching the task's "~97K edges"); every node belongs to at least
one community (no node with zero rows set in `labels`), up to 13 communities for
one node -- genuinely overlapping, not single-label. Row/column order in all three
matrices is already the graph's dense `0..n-1` internal node space (`node_names[i]`
is just node `i`'s original author ID, never needed elsewhere) -- unlike SNAP's
com-DBLP/com-amazon, no ID remapping step is needed.

### What was built

- `src/tnbbeta_vae/data/mag_coauthor.py`: `load_mag_coauthor(root, name)` for
  `name in {"cs", "eng", "chem", "med"}`, returning
  `tnbbeta_vae.data.snap_community.Graph` unmodified (adjacency and features both
  fit it as-is) with an identity `node_id_map` (no remapping needed, per above);
  `load_mag_communities(root, name)` returning the same
  `node_id -> set[community_index]` shape `load_snap_communities`/
  `remap_communities` already use, built directly from the `labels` CSR matrix's
  rows rather than reimplementing community-set logic.
- `apps/data/download_mag_coauthor.py`: mirrors `download_planetoid.py`'s/
  `download_snap_community.py`'s pattern -- downloads the four real `.npz` files
  and verifies by loading. Run for real against `mag_cs` as part of this task
  (not just the one-time format-inspection step above) to confirm the download
  path and parser work end to end: `cs: 21957 nodes, 96750 edges, 7793 features,
  18 communities OK` -- then the downloaded file was deleted again. (This
  sandbox's Python needed `SSL_CERT_FILE` pointed at `certifi`'s bundle for
  `urlretrieve` to pass TLS verification -- a local cert-store quirk, not
  something the script itself needs to handle; `curl` worked without it.)
- Wired into `apps/link_prediction/main.py` (`--dataset mag_cs`/`mag_eng`/
  `mag_chem`/`mag_med`, same single-flag convention as `cora`/`dblp`) and
  `apps/eval/graph_posterior_shape.py` (same dataset dispatch, so
  `posterior_stats` runs on MAG checkpoints unchanged).
- `apps/link_prediction/table.py` additionally gained `power_spherical`'s display
  label, since the dimension sweep now runs it as a fourth family.

All new code is tested against tiny synthetic `.npz` fixtures built with the exact
real key structure above (and tiny synthetic graphs for the dataset-dispatch
wiring) -- no test touches the network or the real multi-MB file, following this
project's existing `download_cifar10.py`/`download_dtd.py` testing discipline.

## Cluster commands (the actual next step, not run here)

### Part 1: Cora/Citeseer/Pubmed dimension sweep, all 4 families

```
sbatch scripts/slurm/link_prediction.sbatch
```

(Requires `apps.data.download_planetoid` already run. 12 tasks: 3 datasets x 4
families, each family already carrying its own `--latent-dims` per the d/d+1
convention above. Then `uv run python -m apps.link_prediction.table` prints the
validation-selected cell per family; the full per-dimension breakdown is in each
`$TNBBETA_CHECKPOINT_DIR/link_prediction/<dataset>_<family>.json`'s
`"configurations"` list.)

### Part 2: MAG Computer-Science co-authorship training, all 4 families

```
uv run python -m apps.data.download_mag_coauthor --datasets cs

sbatch scripts/slurm/link_prediction_run.sbatch mag_cs gaussian \
    --lrs 0.01 --dropouts 0 0.2 --latent-dims 16 32 64 --epochs 200 \
    --seeds 0 1 2 3 4 --run-name mag_cs_gaussian

sbatch scripts/slurm/link_prediction_run.sbatch mag_cs vmf \
    --lrs 0.01 --dropouts 0 0.2 --latent-dims 17 33 65 --epochs 200 \
    --seeds 0 1 2 3 4 --run-name mag_cs_vmf

sbatch scripts/slurm/link_prediction_run.sbatch mag_cs tnbbeta \
    --lrs 0.01 --dropouts 0 0.2 --latent-dims 17 33 65 --epochs 200 \
    --seeds 0 1 2 3 4 --run-name mag_cs_tnbbeta

sbatch scripts/slurm/link_prediction_run.sbatch mag_cs power_spherical \
    --lrs 0.01 --dropouts 0 0.2 --latent-dims 17 33 65 --epochs 200 \
    --seeds 0 1 2 3 4 --run-name mag_cs_power_spherical
```

(`--run-name` with more than one (lr, dropout, latent_dim) configuration
overwrites each seed's checkpoint with the last configuration processed -- per
`apps/link_prediction/main.py`'s own warning, run one configuration per call if
every cell's checkpoint is needed for the diagnostic below, e.g. add
`--latent-dims 64`/`--latent-dims 65` alone per call instead of the full list.
`com-DBLP`'s 317,080-node run needed `--mem=64G` on `link_prediction_run.sbatch`
for sequential multi-configuration calls; MAG CS at 21,957 nodes is roughly 14x
smaller, so the script's current `--mem=64G` default has ample headroom.)

### `graph_posterior_shape` diagnostic, against the resulting checkpoints

```
uv run python -m apps.eval.graph_posterior_shape \
    --run-name mag_cs_tnbbeta_seed0 --dataset mag_cs --device cpu

uv run python -m apps.eval.graph_posterior_shape \
    --run-name mag_cs_vmf_seed0 --dataset mag_cs --device cpu

uv run python -m apps.eval.graph_posterior_shape \
    --run-name mag_cs_power_spherical_seed0 --dataset mag_cs --device cpu

uv run python -m apps.eval.graph_posterior_shape \
    --run-name mag_cs_gaussian_seed0 --dataset mag_cs --device cpu
```

(Repeat per seed/configuration as needed; writes
`posterior_shape_final.json` next to each checkpoint, the same `m_mean`/
`frac_bimodal` summary already reported for Cora/Citeseer/Pubmed/com-DBLP in
`graph_bimodality_investigation_summary_2026-10-08.md`'s table.)

## Deviations from the task description

- The task offered a choice between `--dataset mag_cs`/`mag_eng`/`mag_chem`/
  `mag_med` and a single `mag_coauthor` dataset with a `--mag-name` flag. Chose
  the former: `apps/link_prediction/main.py` and `apps/eval/graph_posterior_shape.py`
  already use a single `--dataset` string for every other dataset (`cora`, `dblp`,
  ...), so a second flag would be inconsistent with the existing convention rather
  than matching it.
- The real `.npz` keys (`adj_matrix.data`/`.indices`/`.indptr`/`.shape`, etc.) use
  a `<matrix_name>.<field>` naming rather than the task description's guessed
  `adj_data`/`adj_indices`/`adj_indptr`/`adj_shape` flat naming -- documented above
  and in `mag_coauthor.py`'s module docstring; the underlying SparseGraph lineage
  guess was otherwise correct.
- Added `power_spherical`'s display label to `apps/link_prediction/table.py`
  (a one-line addition) so the dimension sweep's fourth family shows up in the
  printed table; not explicitly requested, but the table would otherwise silently
  drop a quarter of the sweep's results.
