# Link prediction on citation graphs (S-VAE paper Table 4), partial snapshot, 2026-09-21

Test AUC and AP (%), mean ± std over 5 seeds (fresh 85/5/10 edge split and initialization per seed), at
each family's hyperparameter configuration chosen by mean validation AUC over the grid (learning rate
{0.01, 0.005, 0.001} x dropout {0, 0.2, 0.4} x latent dim {16, 32, 64}); within a run the epoch with the
best validation AUC is used (200 epochs for Cora and Citeseer, 400 for Pubmed). GCN encoder with one hidden
layer of 32 units, inner-product decoder, one random negative per training edge, KL divided by the number
of nodes (Kipf and Welling's VGAE normalization). The sphere models multiply the inner product of unit
vectors by a learned temperature (initially 5), which the paper does not describe. Produced by:

```
uv run --no-sync python -m apps.link_prediction.table
```

The Pubmed vMF run had not finished when this was copied.

| dataset | metric | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE |
|---|---|---|---|---|
| cora | AUC | 92.8 ± 0.6 | 93.7 ± 0.8 | 93.5 ± 0.4 |
| cora | AP | 93.3 ± 0.8 | 93.9 ± 1.0 | 93.9 ± 0.4 |
| citeseer | AUC | 92.0 ± 0.5 | 94.1 ± 0.7 | 93.4 ± 0.9 |
| citeseer | AP | 92.7 ± 0.5 | 94.5 ± 0.8 | 94.3 ± 0.8 |
| pubmed | AUC | 96.0 ± 0.4 | - | 95.4 ± 0.3 |
| pubmed | AP | 96.1 ± 0.4 | - | 95.2 ± 0.3 |

## Reference: the paper's Table 4 (5 repeats)

| dataset | metric | N-VGAE | S-VGAE |
|---|---|---|---|
| Cora | AUC | 92.7 ± 0.2 | 94.1 ± 0.1 |
| Cora | AP | 93.2 ± 0.4 | 94.1 ± 0.3 |
| Citeseer | AUC | 90.3 ± 0.5 | 94.7 ± 0.2 |
| Citeseer | AP | 91.5 ± 0.5 | 95.2 ± 0.2 |
| Pubmed | AUC | 97.1 ± 0.0 | 96.0 ± 0.1 |
| Pubmed | AP | 97.1 ± 0.0 | 96.0 ± 0.1 |

## Observations to check before writing them up

- Our N-VGAE reproduces the paper's on Cora (92.8 vs 92.7); on Citeseer ours is higher (92.0 vs 90.3) and on
  Pubmed lower (96.0 vs 97.1).
- The sphere models beat the Gaussian on Cora (+0.7 to +0.9 AUC) and Citeseer (+1.4 to +2.1 AUC), as in the
  paper. TNBBeta is level with vMF on Cora and slightly behind on Citeseer AUC (93.4 vs 94.1, within about one
  standard deviation); the AP gap on Citeseer is 0.2.
- On Pubmed TNBBeta is below the Gaussian (95.4 vs 96.0), the direction the paper found for vMF.
- Runs that hit exactly-zero node outputs (featureless nodes) use a fixed pole direction after PR #21; the
  original jobs crashed on them. Divergence counts are in each JSON's `num_diverged`.
