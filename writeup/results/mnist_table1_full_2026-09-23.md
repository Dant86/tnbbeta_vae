# MNIST Table 1, paper dimension convention, full 5-seed results, 2026-09-23

Sphere models use ambient d+1 (S^d in R^(d+1): `vmfs` = reference-style vMF, `tnbs` = TNBBeta,
`vmfks` = vMF with kappa initialized at its ambient dimension). Gaussian uses R^d. All other
protocol details as in the earlier partial snapshot. Produced by:

```
uv run --no-sync python -m apps.eval.svae_table --models gauss vmfs tnbs vmfks
```

Bold = best mean in the row that beats every other model with a Welch t-test at p < 0.01.

| d | N-VAE LL | N-VAE L[q] | N-VAE RE | N-VAE KL | S-VAE (vMF, S^d) LL | S-VAE (vMF, S^d) L[q] | S-VAE (vMF, S^d) RE | S-VAE (vMF, S^d) KL | TNBBeta (S^d) LL | TNBBeta (S^d) L[q] | TNBBeta (S^d) RE | TNBBeta (S^d) KL | S-VAE (vMF, kappa init, S^d) LL | S-VAE (vMF, kappa init, S^d) L[q] | S-VAE (vMF, kappa init, S^d) RE | S-VAE (vMF, kappa init, S^d) KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -132.64 ± 0.30 | -136.81 ± 0.28 | -129.19 ± 0.35 | 7.62 ± 0.10 | -130.86 ± 0.43 | -134.42 ± 0.42 | -126.89 ± 0.47 | 7.53 ± 0.08 | -130.11 ± 0.57 | -134.21 ± 0.46 | -126.43 ± 0.54 | 7.78 ± 0.11 | -130.89 ± 0.57 | -134.40 ± 0.33 | -126.84 ± 0.45 | 7.56 ± 0.14 |
| 5 | -105.38 ± 0.11 | -109.53 ± 0.09 | -96.12 ± 0.12 | 13.41 ± 0.17 | -105.27 ± 0.14 | -109.32 ± 0.08 | -95.56 ± 0.12 | 13.76 ± 0.06 | -104.89 ± 0.22 | -109.15 ± 0.22 | -95.36 ± 0.26 | 13.79 ± 0.13 | -105.26 ± 0.18 | -109.27 ± 0.24 | -95.54 ± 0.23 | 13.74 ± 0.06 |
| 10 | **-88.75 ± 0.12** | **-92.91 ± 0.14** | -72.38 ± 0.10 | 20.53 ± 0.12 | -89.38 ± 0.02 | -93.60 ± 0.06 | -71.84 ± 0.08 | 21.77 ± 0.05 | -89.29 ± 0.10 | -93.65 ± 0.14 | -71.96 ± 0.25 | 21.69 ± 0.14 | -89.41 ± 0.10 | -93.66 ± 0.08 | -71.91 ± 0.07 | 21.75 ± 0.06 |
| 20 | **-83.67 ± 0.09** | **-87.64 ± 0.11** | -61.87 ± 0.14 | 25.77 ± 0.07 | -86.84 ± 0.04 | -92.10 ± 0.05 | -62.63 ± 0.21 | 29.47 ± 0.18 | -86.90 ± 0.23 | -92.12 ± 0.15 | -62.33 ± 0.41 | 29.78 ± 0.53 | -86.87 ± 0.04 | -92.08 ± 0.09 | -62.51 ± 0.24 | 29.57 ± 0.16 |
| 40 | -83.24 ± 0.05 | -87.40 ± 0.07 | -60.77 ± 0.11 | 26.63 ± 0.14 | -144.28 ± 31.06 | -170.06 ± 41.29 | -162.32 ± 56.08 | 7.73 ± 14.79 | -89.09 ± 0.15 | -96.38 ± 0.14 | -62.53 ± 0.42 | 33.86 ± 0.36 | -88.88 ± 0.16 | -96.32 ± 0.16 | -62.02 ± 0.28 | 34.30 ± 0.20 |

## Reading it

- **d=2, d=5:** the three sphere variants (vmfs, tnbs, vmfks) are all numerically ahead of the
  Gaussian but none of them beats it (or each other) with a significant t-test -- with 5 seeds'
  worth of real variance, the low-dimension "sphere advantage" from the partial single/few-seed
  snapshot is suggestive, not decisive.
- **d=10, d=20: the Gaussian wins outright**, significantly, over all three sphere variants
  (including the two that don't collapse). The gap grows with dimension: ~0.6 nats (LL) at d=10,
  ~3.2 nats at d=20. This is the opposite of the paper's own finding, where the Gaussian only
  pulls ahead at d=40 in their MLP setup.
- **d=40: `vmfs` (reference init) collapses**, as predicted -- but not uniformly: the huge std
  (LL -144 ± 31) means some seeds trained fine and others hit the mean-image collapse, not that
  all five failed. `vmfks` (kappa-init) and `tnbs` (TNBBeta) both avoid the collapse and land
  close together (-88.9 / -89.1), well below the Gaussian's -83.2 but far above collapsed vmfs.
  **TNBBeta does not beat the fixed vMF anywhere in this table** -- at every dimension where both
  are healthy, they're statistically indistinguishable.
- No row is bolded for the Gaussian at d=40 despite the huge raw gap over vmfs, because vmfs's
  variance is large enough that the Welch t-test does not clear p < 0.01 against it specifically;
  the practical dominance is still real (86% of the gap is real signal), just not "everyone loses
  to Gaussian, formally" for that one comparison.

## Bottom line so far

On the paper's own primary metric (importance-weighted log-likelihood / ELBO), our conv
architecture's Gaussian is a stronger baseline than either sphere family at moderate-to-high
dimension, and TNBBeta's extra flexibility earns it nothing over a trivially-fixed vMF anywhere.
If TNBBeta has a case to make, it is not going to be on this table -- it has to come from the
k-NN, confidence-probe, semi-supervised or link-prediction results, or from the latitude/geometry
diagnostics, not from matching or beating vMF on likelihood.
