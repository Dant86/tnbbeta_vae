# MNIST Table 1 in the paper's dimension convention, partial snapshot, 2026-09-21

Same protocol and metrics as `mnist_table1_partial_2026-09-21.md`, but the sphere models use the paper's
convention: the row's d is the manifold dimension, so the sphere is S^d in R^(d+1) (`vmfs`, `tnbs`,
`vmfks` in `scripts/slurm/mnist_sweep.sbatch`; `latent_dim = d + 1`). The Gaussian columns are the same runs
as before (its d was already the manifold dimension). `vmfs` = reference-style vMF (kappa starts near 1.7),
`vmfks` = vMF with kappa starting at its ambient dimension. Produced by:

```
uv run --no-sync python -m apps.eval.svae_table --models gauss vmfs tnbs vmfks --dims 2 5
```

The sweep was still running: only d = 2 and 5 have finished runs, and TNBBeta d=2 has a single seed
(`± 0.00` means one value); `vmfks` had not started. Mean ± std over seeds, nats per image, test split.

| d | N-VAE LL | N-VAE L[q] | N-VAE RE | N-VAE KL | vMF (S^d) LL | vMF (S^d) L[q] | vMF (S^d) RE | vMF (S^d) KL | TNBBeta (S^d) LL | TNBBeta (S^d) L[q] | TNBBeta (S^d) RE | TNBBeta (S^d) KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -132.64 ± 0.30 | -136.81 ± 0.28 | -129.19 ± 0.35 | 7.62 ± 0.10 | -130.86 ± 0.43 | -134.42 ± 0.42 | -126.89 ± 0.47 | 7.53 ± 0.08 | -130.61 ± 0.00 | -134.62 ± 0.00 | -127.03 ± 0.00 | 7.59 ± 0.00 |
| 5 | -105.38 ± 0.11 | -109.53 ± 0.09 | -96.12 ± 0.12 | 13.41 ± 0.17 | -105.27 ± 0.14 | -109.32 ± 0.08 | -95.56 ± 0.12 | 13.76 ± 0.06 | - | - | - | - |

## Reference: the paper's Table 1 (MLP encoder/decoder, 10 runs)

| d | N-VAE LL | N-VAE L[q] | N-VAE RE | N-VAE KL | S-VAE LL | S-VAE L[q] | S-VAE RE | S-VAE KL |
|---|---|---|---|---|---|---|---|---|
| 2 | -135.73 | -137.08 | -129.84 | 7.24 | -132.50 | -133.72 | -126.43 | 7.28 |
| 5 | -110.21 | -112.98 | -100.16 | 12.82 | -108.43 | -111.19 | -97.84 | 13.35 |

## Observations

- With the paper's convention the d=2 vMF is at the paper's level (RE -126.9 vs -126.4, KL 7.53 vs 7.28) and
  beats our Gaussian by 2.4 nats on the ELBO (paper: 3.4). The earlier ambient-d version (a circle) scored
  LL -150.0.
- At d=5 the vMF is level with our Gaussian (LL -105.27 ± 0.14 vs -105.38 ± 0.11), whereas the paper's
  S-VAE was 1.8 nats ahead of its Gaussian. Our conv Gaussian is a stronger baseline than the paper's MLP one.
- TNBBeta at d=2 (one seed) is level with vMF (LL -130.61 vs -130.86 ± 0.43). Not yet enough seeds to say more.
