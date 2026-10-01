# Link prediction on citation graphs, complete results, 2026-09-23

Same protocol as the earlier partial snapshot (`link_prediction_table4_partial_2026-09-21.md`);
Pubmed vMF now uses the paper's own selected configuration (lr=0.01, dropout=0, latent dim 32,
5 seeds) since the full 27-configuration grid search kept timing out / crashing on Pubmed's size.
Produced by:

```
uv run --no-sync python -m apps.link_prediction.table
```

| dataset | metric | N-VGAE | S-VGAE (vMF) | TNBBeta-VGAE |
|---|---|---|---|---|
| cora | AUC | 92.8 ± 0.6 | 93.7 ± 0.8 | 93.5 ± 0.4 |
| cora | AP | 93.3 ± 0.8 | 93.9 ± 1.0 | 93.9 ± 0.4 |
| citeseer | AUC | 92.0 ± 0.5 | 94.1 ± 0.7 | 93.4 ± 0.9 |
| citeseer | AP | 92.7 ± 0.5 | 94.5 ± 0.8 | 94.3 ± 0.8 |
| pubmed | AUC | 96.0 ± 0.4 | 95.2 ± 0.2 | 95.4 ± 0.3 |
| pubmed | AP | 96.1 ± 0.4 | 95.1 ± 0.3 | 95.2 ± 0.3 |

## Reference: the paper's Table 4

| dataset | metric | N-VGAE | S-VGAE |
|---|---|---|---|
| Cora | AUC | 92.7 ± 0.2 | 94.1 ± 0.1 |
| Cora | AP | 93.2 ± 0.4 | 94.1 ± 0.3 |
| Citeseer | AUC | 90.3 ± 0.5 | 94.7 ± 0.2 |
| Citeseer | AP | 91.5 ± 0.5 | 95.2 ± 0.2 |
| Pubmed | AUC | 97.1 ± 0.0 | 96.0 ± 0.1 |
| Pubmed | AP | 97.1 ± 0.0 | 96.0 ± 0.1 |

## Reading it

- The qualitative pattern matches the paper on all three datasets: sphere latents beat the
  Gaussian on Cora and Citeseer, and lose to it on Pubmed (the paper's own explanation: Pubmed is
  larger/more complex and benefits from the Gaussian's extra free parameters).
- **TNBBeta and vMF are statistically indistinguishable on every dataset and metric** -- nominal
  gaps of 0.2 (Cora AUC), 0.7 (Citeseer AUC), 0.2 (Pubmed AUC, TNBBeta nominally ahead here) are
  all well within their combined standard deviations.
- Note the Pubmed vMF cell used only the paper's single selected configuration, not our own
  27-point grid search like the other two cells and like TNBBeta's Pubmed result -- a mildly
  unfair comparison in vMF's favor (its number benefits from the paper's own tuning), which makes
  the tie with TNBBeta there slightly more notable, not less.
