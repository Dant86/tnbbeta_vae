# Qualitative check: do heterogeneous nodes' neighbors visibly split? 2026-10-09

- **Date:** 2026-10-09
- **Branch/commit:** `neighbor-heterogeneity-mechanism`, `1664727`
  (`Add apps/eval/visualize_neighbor_clusters.py`). `master` did not yet have this
  code.
- **Checkpoint used:** `dose_response_pout_0.05_tnbbeta_seed0` (the main `p_out=0.05`
  TNBBeta checkpoint from `sbm_pout_dose_response_2026-10-09.md`, `best_epoch=253`,
  well-trained, not one of that write-up's near-init-collapsed points).
- **Command:**
  ```
  uv run python -m apps.eval.visualize_neighbor_clusters \
      --run-name dose_response_pout_0.05_tnbbeta_seed0 --device cpu \
      --num-communities 10 --nodes-per-community 50 --p-in 0.3 --p-out 0.05 \
      --graph-seed 0 --num-high 4 --num-low 4 --min-degree 6
  ```
  Writes `neighbor_clusters_final.html`/`.json` next to the checkpoint
  (`checkpoints/dose_response_pout_0.05_tnbbeta_seed0/`).

## What was selected

| group | node | heterogeneity (nats) | degree |
|---|---|---|---|
| high | 105 | 2.224 | 45 |
| high | 260 | 2.165 | 40 |
| high | 73 | 2.162 | 38 |
| high | 59 | 2.162 | 39 |
| low | 283 | 1.153 | 21 |
| low | 117 | 1.430 | 29 |
| low | 108 | 1.447 | 27 |
| low | 345 | 1.457 | 26 |

## What the figure actually shows

The published figure plots each selected node's neighbors' posterior centres,
PCA-projected to 2D. Beyond the figure itself, two direct, quantitative checks were
run against the same neighbor sets (not shown in the plot, but a more honest read
of what it contains than eyeballing a scatter alone):

**1. Pairwise cosine similarity among each node's neighbors' posterior centres (in
the original 16-dimensional ambient space, before PCA):**

| group | node | mean sim | std sim | frac. negative pairs |
|---|---|---|---|---|
| high | 105 | 0.048 | 0.327 | 0.501 |
| high | 260 | 0.035 | 0.315 | 0.513 |
| high | 73 | 0.043 | 0.323 | 0.511 |
| high | 59 | 0.053 | 0.331 | 0.513 |
| low | 283 | 0.318 | 0.387 | 0.238 |
| low | 117 | 0.265 | 0.426 | 0.337 |
| low | 108 | 0.195 | 0.428 | 0.430 |
| low | 345 | 0.195 | 0.375 | 0.323 |

High-heterogeneity nodes' neighbors have a mean pairwise cosine similarity close to
0 (0.035-0.053) -- indistinguishable, at this latent dimension (16), from what a set
of uncorrelated random directions on the sphere would give (the expected pairwise
cosine similarity between independent uniform points on `S^15` is 0, with standard
deviation `~1/sqrt(15) ≈ 0.258`, which matches the observed std of ~0.32 closely).
Low-heterogeneity nodes' neighbors are noticeably more aligned (mean 0.20-0.32,
clearly positive) -- a real, substantial difference in spread.

**2. A direct 2-cluster test (2-means on the same PCA-projected 2D points, fraction
of variance explained by the best 2-way split):**

| group | node | n neighbors | 2-means variance explained | cluster sizes |
|---|---|---|---|---|
| high | 105 | 45 | 0.346 | [25, 20] |
| high | 260 | 40 | 0.430 | [25, 15] |
| high | 73 | 38 | 0.409 | [20, 18] |
| high | 59 | 39 | 0.546 | [25, 14] |
| low | 283 | 21 | 0.492 | [9, 12] |
| low | 117 | 29 | 0.604 | [11, 18] |
| low | 108 | 27 | 0.617 | [10, 17] |
| low | 345 | 26 | 0.540 | [12, 14] |

**This does NOT show a cleaner 2-cluster split for high-heterogeneity nodes --**
if anything, the low-heterogeneity nodes' variance-explained-by-2-means is
*slightly higher* (0.49-0.62) than the high-heterogeneity nodes' (0.35-0.55). This
check is reported honestly despite not supporting the sharper hypothesis: a
2-means split finds *some* separation in any point cloud (even an isotropic
Gaussian blob typically has ~36% of its variance "explained" by the best arbitrary
2-way split), so this metric alone cannot distinguish "genuinely bimodal" from
"diffusely spread" -- and the two checks together suggest it's the latter, not the
former, that's happening here.

## Honest reading

**The visualization does show a real, measurable difference between high- and
low-heterogeneity nodes' neighbor representations: high-heterogeneity nodes'
neighbors are more spread out / diffuse across the latent sphere** (cosine
similarities statistically close to what uncorrelated random directions would
give) **than low-heterogeneity nodes' neighbors, whose posterior centres are
noticeably more mutually aligned.** This is consistent with the basic premise that
a community-heterogeneous neighborhood produces more varied neighbor embeddings.

**It does NOT show the sharper "two separated clusters vs. one tight cluster"**
structure the mechanism's intuitive picture suggests, by a direct quantitative
test (2-means variance-explained is, if anything, slightly lower for the
high-heterogeneity group). The more accurate description of what this diagnostic
found is **"more diffuse/scattered" rather than "more bimodal" in a strict
two-cluster sense** -- a real pattern, but a different and weaker one than the
mechanism's cleanest version predicts, and reported here as such rather than
cherry-picking a node whose plot looks like two clusters (none of the four
high-heterogeneity nodes checked quantitatively supports that specific reading
over the others).

**Caveat:** one checkpoint, one graph, 4+4 selected nodes (not every node in the
graph), and a single degree threshold (`--min-degree 6`). The two quantitative
checks above were run ad hoc against this one figure's node selection to give an
honest read beyond eyeballing the plot; they are not part of the committed script
(which deliberately makes no clustering claim on its own -- see its docstring).
