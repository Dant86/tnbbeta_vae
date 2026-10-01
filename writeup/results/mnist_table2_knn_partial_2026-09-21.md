# MNIST latent k-NN (S-VAE paper Table 2), partial snapshot, 2026-09-21

Test accuracy (fraction) of a 5-NN vote in latent space with N randomly labelled training images
(class-balanced draws are not used here; 20 random subsets per model, then mean over seeds),
mean ± std over seeds. Euclidean distance on the Gaussian mean; geodesic distance on the centre of
the sphere models' posterior (vMF mean direction; TNBBeta mode direction). Same runs and protocol as
the Table 1 snapshot (`mnist_table1_partial_2026-09-21.md`); the sweep was still running ("7 run(s)
had no metrics file and were skipped"). Produced by:

```
uv run --no-sync python -m apps.eval.svae_table --kind knn
```

Bold = best mean in the row that beats every other model with a Welch t-test at p < 0.01 (rows
without all three models get no bold).

| d | N-VAE N=100 | N-VAE N=600 | N-VAE N=1000 | S-VAE (vMF) N=100 | S-VAE (vMF) N=600 | S-VAE (vMF) N=1000 | TNBBeta N=100 | TNBBeta N=600 | TNBBeta N=1000 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.73 ± 0.01 | 0.83 ± 0.01 | 0.84 ± 0.01 | 0.69 ± 0.02 | 0.73 ± 0.03 | 0.73 ± 0.03 | - | - | - |
| 5 | 0.81 ± 0.00 | 0.92 ± 0.00 | 0.94 ± 0.00 | 0.86 ± 0.01 | 0.94 ± 0.00 | 0.94 ± 0.00 | 0.85 ± 0.02 | 0.93 ± 0.01 | 0.94 ± 0.01 |
| 10 | 0.74 ± 0.01 | 0.90 ± 0.00 | 0.92 ± 0.00 | 0.77 ± 0.01 | 0.92 ± 0.00 | 0.93 ± 0.00 | 0.77 ± 0.00 | 0.92 ± 0.00 | 0.93 ± 0.00 |
| 20 | 0.58 ± 0.01 | 0.82 ± 0.01 | 0.86 ± 0.01 | 0.66 ± 0.00 | 0.87 ± 0.00 | 0.90 ± 0.00 | 0.65 ± 0.00 | 0.87 ± 0.00 | 0.90 ± 0.00 |
| 40 | 0.52 ± 0.01 | 0.78 ± 0.01 | 0.82 ± 0.01 | **0.62 ± 0.00** | 0.74 ± 0.00 | 0.76 ± 0.00 | 0.57 ± 0.01 | **0.83 ± 0.00** | **0.87 ± 0.00** |

## Reference: the paper's Table 2 (accuracy %, MLP models, 20 runs)

| d | N-VAE N=100 | S-VAE N=100 | N-VAE N=600 | S-VAE N=600 | N-VAE N=1000 | S-VAE N=1000 |
|---|---|---|---|---|---|---|
| 2 | 72.6 | 77.9 | 80.8 | 84.9 | 81.7 | 85.6 |
| 5 | 81.8 | 87.5 | 90.9 | 92.8 | 92.0 | 93.4 |
| 10 | 75.7 | 80.6 | 88.4 | 91.2 | 90.2 | 92.8 |
| 20 | 71.3 | 72.8 | 88.3 | 89.1 | 90.1 | 91.1 |
| 40 | 72.3 | 67.7 | 88.0 | 87.4 | 90.3 | 90.4 |

## Observations to check before writing them up

- The Gaussian at d=5 matches the paper's (0.81 / 0.92 / 0.94 vs 81.8 / 90.9 / 92.0). The sphere models beat
  the Gaussian at d=5-20 (N=100: 0.86 and 0.85 vs 0.81 at d=5; 0.66 and 0.65 vs 0.58 at d=20), in line with
  the paper's low-dimension advantage.
- vMF and TNBBeta are indistinguishable at d=5-20. At d=40 TNBBeta is best for N=600 and N=1000 (0.83, 0.87)
  while vMF is best only at N=100 (0.62); note the d=40 vMF runs are the collapsed ones (Table 1: KL 1.15).
- At d=2 vMF is worse than the Gaussian (0.69 vs 0.73 at N=100), unlike the paper; this is the same d=2 vMF
  cell that looks under-trained in Table 1. Our high-dimension Gaussian is worse than the paper's (d=20,
  N=100: 0.58 vs 71.3%; d=40: 0.52 vs 72.3%), probably an architecture or centre-of-posterior effect.

## Correction

Same as the Table 1 snapshot: the sphere rows use ambient dimension d (S^(d-1)); the paper's d is the
manifold dimension (ambient d + 1). Re-run with the `vmfs`/`tnbs`/`vmfks` models for a like-for-like
comparison.
