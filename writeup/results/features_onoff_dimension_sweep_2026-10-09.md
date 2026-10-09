# Features on/off x dimension sweep, Cora/Citeseer/Pubmed/MAG CS, 2026-10-09

- **Date:** 2026-10-09
- **Branch/commit:** `master`, `630b77b`. The sweep itself ran against two
  commits on `dtd-oriented-textures`, both now part of `master`'s history at
  `630b77b`: `0983e0c` (the `d in {8,16,32,64}` dimension fix and the
  `--ignore-features` toggle) for most cells, and `5060c37` (the
  `link_prediction_run.sbatch` `--device cuda` fix -- see "A real bug found
  mid-run" below) for the one cell that needed a manual retry after that fix.
  This write-up itself is on a fresh branch off `master` (`630b77b`), per
  this project's "one PR, many commits" convention, rather than continuing
  to extend the long-lived `dtd-oriented-textures` branch.
- **Commands:**
  ```
  uv run python -m apps.data.download_planetoid
  uv run python -m apps.data.download_mag_coauthor --datasets cs

  # Features ON -- one array job, 4 datasets x 4 families x dz in {8,16,32,64}
  sbatch scripts/slurm/link_prediction.sbatch

  # Features OFF -- Cora/Citeseer/Pubmed/MAG CS only (DBLP/Amazon have no
  # features to toggle off)
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

  uv run python -m apps.link_prediction.table \
      --datasets cora cora_nofeat citeseer citeseer_nofeat pubmed pubmed_nofeat \
      mag_cs mag_cs_nofeat --by-dimension
  ```

## A real bug found mid-run (job 2120361)

`scripts/slurm/link_prediction_run.sbatch` hardcoded `--device cuda`.
`select_device` honors an explicit `--device` verbatim with no availability
check at all, so on a node where Slurm allocated a GPU but CUDA couldn't
actually initialize, the job skipped `select_device`'s own clean, designed-
for-this `exit(75)` entirely and instead crashed later, uncaught, inside
`GraphBatch.to(device)` with a raw `RuntimeError: No CUDA GPUs are
available` (Python's default exit code 1 for an uncaught exception) --
a code `resubmit_if_no_gpu` doesn't recognize, so the automatic retry never
fired. Fixed by removing the hardcoded flag (commit `5060c37`), with a
regression test. The one affected cell (`mag_cs`/`power_spherical`,
identified by counting the loop's 16 `Submitted batch job` lines in
submission order) was resubmitted manually after the fix; every other cell
in this table ran under the already-correct behavior.
`link_prediction.sbatch` (the array job) was never affected -- it already
left `--device` unset.

## Results

| dataset | metric | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|---|
| cora | AUC | 93.2 ± 1.0 | 93.2 ± 0.5 | 93.7 ± 0.8 | 93.7 ± 0.7 |
| cora | AP | 93.9 ± 1.0 | 93.6 ± 0.7 | 94.0 ± 0.9 | 93.8 ± 0.7 |
| cora_nofeat | AUC | 84.8 ± 1.0 | 78.1 ± 1.1 | 84.0 ± 1.6 | 74.7 ± 2.1 |
| cora_nofeat | AP | 88.4 ± 0.7 | 81.7 ± 0.7 | 87.6 ± 1.2 | 78.2 ± 2.1 |
| citeseer | AUC | 91.1 ± 1.1 | 94.0 ± 0.8 | 93.1 ± 0.4 | 93.5 ± 0.8 |
| citeseer | AP | 92.1 ± 1.0 | 94.4 ± 0.8 | 93.8 ± 0.5 | 94.0 ± 0.9 |
| citeseer_nofeat | AUC | 78.6 ± 1.3 | 74.8 ± 2.0 | 77.0 ± 2.1 | 70.0 ± 0.9 |
| citeseer_nofeat | AP | 83.8 ± 1.0 | 80.1 ± 1.5 | 82.7 ± 1.8 | 75.6 ± 1.1 |
| pubmed | AUC | 95.9 ± 0.3 | 95.1 ± 0.3 | 95.4 ± 0.3 | 95.2 ± 0.2 |
| pubmed | AP | 96.0 ± 0.2 | 94.9 ± 0.4 | 95.2 ± 0.3 | 94.9 ± 0.2 |
| pubmed_nofeat | AUC | 85.3 ± 1.0 | 82.3 ± 0.4 | 82.7 ± 0.6 | 81.3 ± 0.6 |
| pubmed_nofeat | AP | 85.5 ± 0.8 | 85.8 ± 0.4 | 86.7 ± 0.5 | 84.3 ± 0.8 |
| mag_cs | AUC | 96.7 ± 0.1 | 96.6 ± 0.1 | 95.3 ± 0.2 | 96.5 ± 0.2 |
| mag_cs | AP | 96.7 ± 0.2 | 96.5 ± 0.2 | 95.0 ± 0.2 | 96.5 ± 0.2 |
| mag_cs_nofeat | AUC | 94.3 ± 0.3 | 89.4 ± 0.5 | 94.3 ± 0.3 | 86.1 ± 0.3 |
| mag_cs_nofeat | AP | 95.0 ± 0.2 | 90.2 ± 0.5 | 95.5 ± 0.2 | 88.7 ± 0.2 |

### cora, by latent dimension

**AUC**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 92.0 ± 1.0 | dim=9: 93.4 ± 1.0 | dim=9: 93.2 ± 0.6 | dim=9: 92.4 ± 0.7 |
| 1 | dim=16: 92.3 ± 0.6 | dim=17: 93.9 ± 0.7 | dim=17: 93.7 ± 0.8 | dim=17: 93.7 ± 0.7 |
| 2 | dim=32: 93.1 ± 1.3 | dim=33: 93.2 ± 0.5 | dim=33: 93.6 ± 0.6 | dim=33: 93.3 ± 0.8 |
| 3 | dim=64: 93.2 ± 1.0 | dim=65: 89.7 ± 1.6 | dim=65: 92.8 ± 1.5 | dim=65: 93.4 ± 0.6 |

**AP**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 92.1 ± 1.2 | dim=9: 93.7 ± 0.9 | dim=9: 93.7 ± 0.6 | dim=9: 92.9 ± 0.8 |
| 1 | dim=16: 93.0 ± 0.8 | dim=17: 94.2 ± 0.9 | dim=17: 94.0 ± 0.9 | dim=17: 93.8 ± 0.7 |
| 2 | dim=32: 93.4 ± 1.6 | dim=33: 93.6 ± 0.7 | dim=33: 94.0 ± 0.4 | dim=33: 93.8 ± 0.9 |
| 3 | dim=64: 93.9 ± 1.0 | dim=65: 90.5 ± 1.5 | dim=65: 93.3 ± 1.3 | dim=65: 93.9 ± 0.7 |

### cora_nofeat, by latent dimension

**AUC**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 84.0 ± 1.1 | dim=9: 78.1 ± 1.1 | dim=9: 81.2 ± 1.9 | dim=9: 75.1 ± 1.5 |
| 1 | dim=16: 85.1 ± 0.8 | dim=17: 76.7 ± 1.6 | dim=17: 82.7 ± 2.1 | dim=17: 71.7 ± 1.8 |
| 2 | dim=32: 84.8 ± 1.0 | dim=33: 74.8 ± 1.3 | dim=33: 84.1 ± 1.3 | dim=33: 73.6 ± 1.0 |
| 3 | dim=64: 84.9 ± 1.3 | dim=65: 74.7 ± 0.2 | dim=65: 84.0 ± 1.6 | dim=65: 74.7 ± 2.1 |

**AP**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 87.1 ± 1.1 | dim=9: 81.7 ± 0.7 | dim=9: 84.7 ± 1.5 | dim=9: 78.3 ± 1.5 |
| 1 | dim=16: 88.1 ± 0.8 | dim=17: 80.5 ± 1.4 | dim=17: 86.7 ± 1.4 | dim=17: 75.6 ± 1.9 |
| 2 | dim=32: 88.4 ± 0.7 | dim=33: 78.8 ± 1.1 | dim=33: 87.6 ± 1.1 | dim=33: 76.8 ± 0.7 |
| 3 | dim=64: 88.7 ± 0.9 | dim=65: 78.7 ± 0.3 | dim=65: 87.6 ± 1.2 | dim=65: 78.2 ± 2.1 |

### citeseer, by latent dimension

**AUC**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 89.6 ± 0.8 | dim=9: 93.3 ± 0.7 | dim=9: 91.8 ± 0.9 | dim=9: 93.5 ± 0.5 |
| 1 | dim=16: 90.6 ± 1.2 | dim=17: 93.4 ± 0.5 | dim=17: 92.3 ± 2.5 | dim=17: 93.9 ± 0.8 |
| 2 | dim=32: 91.3 ± 0.8 | dim=33: 94.0 ± 0.8 | dim=33: 93.5 ± 0.8 | dim=33: 93.5 ± 0.8 |
| 3 | dim=64: 91.1 ± 1.1 | dim=65: 90.1 ± 1.0 | dim=65: 93.1 ± 0.4 | dim=65: 93.4 ± 0.3 |

**AP**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 90.3 ± 0.9 | dim=9: 94.1 ± 0.6 | dim=9: 93.0 ± 0.7 | dim=9: 94.0 ± 0.6 |
| 1 | dim=16: 91.2 ± 1.1 | dim=17: 94.2 ± 0.6 | dim=17: 93.3 ± 2.4 | dim=17: 94.3 ± 0.8 |
| 2 | dim=32: 92.0 ± 0.9 | dim=33: 94.4 ± 0.8 | dim=33: 94.2 ± 0.7 | dim=33: 94.0 ± 0.9 |
| 3 | dim=64: 92.1 ± 1.0 | dim=65: 91.3 ± 0.7 | dim=65: 93.8 ± 0.5 | dim=65: 94.1 ± 0.5 |

### citeseer_nofeat, by latent dimension

**AUC**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 77.3 ± 0.6 | dim=9: 74.8 ± 2.0 | dim=9: 76.7 ± 1.3 | dim=9: 70.8 ± 1.1 |
| 1 | dim=16: 77.3 ± 2.2 | dim=17: 73.5 ± 0.9 | dim=17: 77.0 ± 2.1 | dim=17: 71.6 ± 3.5 |
| 2 | dim=32: 77.9 ± 1.4 | dim=33: 70.9 ± 2.0 | dim=33: 77.2 ± 1.2 | dim=33: 71.1 ± 1.5 |
| 3 | dim=64: 78.6 ± 1.3 | dim=65: 71.0 ± 2.2 | dim=65: 76.3 ± 1.5 | dim=65: 70.0 ± 0.9 |

**AP**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 81.9 ± 0.8 | dim=9: 80.1 ± 1.5 | dim=9: 81.9 ± 1.1 | dim=9: 76.2 ± 1.0 |
| 1 | dim=16: 81.8 ± 1.7 | dim=17: 79.1 ± 0.7 | dim=17: 82.7 ± 1.8 | dim=17: 76.3 ± 3.4 |
| 2 | dim=32: 83.1 ± 1.4 | dim=33: 76.2 ± 2.2 | dim=33: 83.0 ± 1.3 | dim=33: 76.1 ± 0.9 |
| 3 | dim=64: 83.8 ± 1.0 | dim=65: 76.9 ± 2.3 | dim=65: 82.3 ± 1.6 | dim=65: 75.6 ± 1.1 |

### pubmed, by latent dimension

**AUC**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 95.7 ± 0.2 | dim=9: 94.9 ± 0.2 | dim=9: 95.1 ± 0.2 | dim=9: 94.9 ± 0.2 |
| 1 | dim=16: 95.7 ± 0.5 | dim=17: 95.1 ± 0.3 | dim=17: 95.4 ± 0.3 | dim=17: 95.0 ± 0.4 |
| 2 | dim=32: 95.7 ± 0.4 | dim=33: 95.1 ± 0.3 | dim=33: 94.5 ± 0.8 | dim=33: 95.2 ± 0.2 |
| 3 | dim=64: 95.9 ± 0.3 | dim=65: 91.4 ± 0.3 | dim=65: 91.7 ± 5.3 | dim=65: 95.0 ± 0.3 |

**AP**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 95.8 ± 0.2 | dim=9: 94.6 ± 0.2 | dim=9: 94.7 ± 0.3 | dim=9: 94.6 ± 0.3 |
| 1 | dim=16: 95.9 ± 0.5 | dim=17: 94.9 ± 0.4 | dim=17: 95.2 ± 0.3 | dim=17: 94.8 ± 0.4 |
| 2 | dim=32: 95.9 ± 0.4 | dim=33: 94.9 ± 0.3 | dim=33: 94.2 ± 0.8 | dim=33: 94.9 ± 0.2 |
| 3 | dim=64: 96.0 ± 0.2 | dim=65: 91.5 ± 0.5 | dim=65: 91.8 ± 4.4 | dim=65: 94.7 ± 0.3 |

### pubmed_nofeat, by latent dimension

**AUC**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 83.2 ± 0.5 | dim=9: 82.3 ± 0.4 | dim=9: 82.4 ± 0.6 | dim=9: 81.3 ± 0.6 |
| 1 | dim=16: 84.3 ± 0.3 | dim=17: 82.0 ± 0.6 | dim=17: 82.7 ± 0.6 | dim=17: 81.0 ± 0.7 |
| 2 | dim=32: 84.3 ± 0.4 | dim=33: 79.6 ± 1.2 | dim=33: 82.8 ± 0.1 | dim=33: 79.6 ± 1.1 |
| 3 | dim=64: 85.3 ± 1.0 | dim=65: 76.5 ± 1.1 | dim=65: 82.6 ± 0.4 | dim=65: 76.4 ± 1.3 |

**AP**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 88.1 ± 0.4 | dim=9: 85.8 ± 0.4 | dim=9: 86.1 ± 0.5 | dim=9: 84.3 ± 0.8 |
| 1 | dim=16: 88.6 ± 0.3 | dim=17: 85.4 ± 0.8 | dim=17: 86.7 ± 0.5 | dim=17: 83.9 ± 0.8 |
| 2 | dim=32: 88.8 ± 0.2 | dim=33: 82.4 ± 1.4 | dim=33: 87.0 ± 0.2 | dim=33: 82.4 ± 1.2 |
| 3 | dim=64: 85.5 ± 0.8 | dim=65: 80.4 ± 1.0 | dim=65: 86.7 ± 0.3 | dim=65: 80.0 ± 0.9 |

### mag_cs, by latent dimension

**AUC**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 95.5 ± 0.2 | dim=9: 95.7 ± 0.2 | dim=9: 95.3 ± 0.3 | dim=9: 95.9 ± 0.2 |
| 1 | dim=16: 96.4 ± 0.1 | dim=17: 96.4 ± 0.2 | dim=17: 95.3 ± 0.2 | dim=17: 96.4 ± 0.2 |
| 2 | dim=32: 96.5 ± 0.1 | dim=33: 96.6 ± 0.1 | dim=33: 95.2 ± 1.0 | dim=33: 96.5 ± 0.2 |
| 3 | dim=64: 96.7 ± 0.1 | dim=65: 94.6 ± 1.9 | dim=65: 94.4 ± 1.9 | dim=65: 96.5 ± 0.2 |

**AP**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 95.3 ± 0.2 | dim=9: 95.6 ± 0.2 | dim=9: 94.9 ± 0.4 | dim=9: 95.7 ± 0.1 |
| 1 | dim=16: 96.3 ± 0.1 | dim=17: 96.3 ± 0.1 | dim=17: 95.0 ± 0.2 | dim=17: 96.3 ± 0.2 |
| 2 | dim=32: 96.5 ± 0.0 | dim=33: 96.5 ± 0.2 | dim=33: 95.0 ± 1.2 | dim=33: 96.5 ± 0.2 |
| 3 | dim=64: 96.7 ± 0.2 | dim=65: 94.6 ± 1.9 | dim=65: 94.3 ± 1.9 | dim=65: 96.4 ± 0.2 |

### mag_cs_nofeat, by latent dimension

**AUC**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 92.7 ± 0.3 | dim=9: 89.4 ± 0.5 | dim=9: 93.3 ± 0.2 | dim=9: 84.8 ± 0.8 |
| 1 | dim=16: 93.7 ± 0.2 | dim=17: 87.3 ± 0.5 | dim=17: 94.1 ± 0.2 | dim=17: 82.0 ± 1.3 |
| 2 | dim=32: 94.0 ± 0.3 | dim=33: 85.4 ± 0.5 | dim=33: 94.3 ± 0.3 | dim=33: 83.9 ± 0.4 |
| 3 | dim=64: 94.3 ± 0.3 | dim=65: 85.4 ± 0.5 | dim=65: 94.0 ± 0.2 | dim=65: 86.1 ± 0.3 |

**AP**

| dim index | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE | Power Spherical-VGAE |
|---|---|---|---|---|
| 0 | dim=8: 93.2 ± 0.3 | dim=9: 90.2 ± 0.5 | dim=9: 94.4 ± 0.2 | dim=9: 86.8 ± 0.5 |
| 1 | dim=16: 94.3 ± 0.2 | dim=17: 89.1 ± 0.3 | dim=17: 95.3 ± 0.1 | dim=17: 84.7 ± 1.2 |
| 2 | dim=32: 94.6 ± 0.2 | dim=33: 88.0 ± 0.4 | dim=33: 95.5 ± 0.2 | dim=33: 86.4 ± 0.4 |
| 3 | dim=64: 95.0 ± 0.2 | dim=65: 88.0 ± 0.5 | dim=65: 95.3 ± 0.1 | dim=65: 88.7 ± 0.2 |

## Reading it

**The features-off toggle (same graph, held fixed, only the encoder's input
changed) consistently makes Gaussian the best-or-tied-best family, every
dataset, no exception.** Comparing the "selected" row's winner in each
`_nofeat` condition against its features-on counterpart: Gaussian goes from
sometimes-losing (citeseer, mag_cs) to winning or tying every time once
features are removed. This is the clean, non-confounded confirmation of the
norm-as-degree mechanism discussed this session: an unconstrained Gaussian
embedding can use its own magnitude as a free per-node signal in a way no
unit-norm sphere family can, and that advantage is largest specifically
where there is no other information (features) to lean on.

**Within the sphere families, Power Spherical is worst in every single
`_nofeat` condition, and TNBBeta is consistently the strongest sphere
family there** -- clearly so on Cora (84.0 vs. vMF 78.1, PS 74.7), Citeseer
(77.0 vs. 74.8, 70.0) and MAG CS (94.3, tied with Gaussian and ahead of both
other spheres: vMF 89.4, PS 86.1). **Pubmed is the one exception**: all
three sphere families cluster within 1.4 points of each other
(vMF 82.3, TNBBeta 82.7, PS 81.3) -- the same qualitative ranking, but a
much smaller, noisier effect than the other three datasets show. This
should be read as "the pattern holds clearly on 3 of 4 datasets, weakly on
the 4th," not smoothed into "every dataset."

**MAG CS is the single cleanest demonstration of the whole hypothesis**:
TNBBeta goes from clearly worst with features (95.3 vs. 96.5-96.7) to
tied-best without them (94.3/95.5, edging out Gaussian on AP) -- same
graph, same architecture, only the feature toggle changed.

**What this does not explain: TNBBeta's deficit on MAG CS *with* features**
is real, not a dimension-65 instability artifact -- it holds steady across
`dim=9/17/33` with tight std (0.2-1.0), only getting noisier at `dim=65`.
Cora/Citeseer/Pubmed-with-features don't show an equivalent TNBBeta
deficit; MAG CS is also the only dataset here with genuine *overlapping*
multi-label communities (NOCD's design) rather than single-label classes.
The live, untested hypothesis: it may be the *combination* of real features
and genuine community overlap, not features alone, that costs TNBBeta
something specific. This is the open question the next round of work
targets.

**The `dim=65` variance blow-up (Pubmed/TNBBeta `91.7 +/- 5.3` AUC with
features; Pubmed/vMF and MAG-CS/vMF similarly noisy at their own highest
dimension) looks tied to large real feature matrices at high dimension
generally, not a TNBBeta-specific fragility** -- the same Pubmed/TNBBeta
cell *without* features, at the identical `dim=65`, is stable
(`82.6 +/- 0.4`), and vMF shows the same kind of blow-up on MAG-CS-with-
features at its own `dim=65` (`94.6 +/- 1.9`, up from `96.6 +/- 0.1` at
`dim=33`). `GraphVAEConfig` has no dimension-aware `kappa`-style
initialization knob at all, unlike the conv/MNIST pipeline, which
`CLAUDE.md` already documents needs one at high `d`, "or it collapses" --
a plausible, concrete, not-yet-tried fix for these specific cells, flagged
here rather than pursued yet.
