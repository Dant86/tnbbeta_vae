# MNIST Table 1 (S-VAE paper protocol), partial snapshot, 2026-09-21

Unsupervised binarized MNIST, conv encoder/decoder (`hidden_channels=32`), test split, nats per
image, mean ± std over seeds (up to 5 per cell; the sweep was still running when this was
copied, so the number of finished seeds per cell was not recorded). Produced by:

```
uv run --no-sync python -m apps.eval.svae_table
```

from `svae_metrics_final_test.json` in `mnist_<model>_d<dim>_seed<seed>/` under
`/net/spaces/scratch/vpathak/tnbbeta_checkpoints/`, trained with `scripts/slurm/mnist_sweep.sbatch`
(batch 64, Adam 1e-3, 100-epoch linear KL warm-up, early stopping with patience 50 on the 10k
validation images, best-validation epoch evaluated). LL is the 500-sample importance-weighted
log-likelihood, L[q] the ELBO, RE the reconstruction term, KL the exact KL (Gaussian, vMF) or
its Monte Carlo estimate (TNBBeta). Bold = best mean in the row that beats every other model with
a Welch t-test at p < 0.01 (KL is never bolded); a row with a missing model has no bold for it.

| d | N-VAE LL | N-VAE L[q] | N-VAE RE | N-VAE KL | S-VAE (vMF) LL | S-VAE (vMF) L[q] | S-VAE (vMF) RE | S-VAE (vMF) KL | TNBBeta LL | TNBBeta L[q] | TNBBeta RE | TNBBeta KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -132.64 ± 0.30 | -136.81 ± 0.28 | -129.19 ± 0.35 | 7.62 ± 0.10 | -150.01 ± 0.91 | -152.92 ± 1.37 | -148.34 ± 1.40 | 4.58 ± 0.06 | - | - | - | - |
| 5 | **-105.38 ± 0.11** | **-109.53 ± 0.09** | **-96.12 ± 0.12** | 13.41 ± 0.17 | -111.64 ± 0.06 | -115.59 ± 0.13 | -103.78 ± 0.13 | 11.80 ± 0.05 | -111.75 ± 0.53 | -115.99 ± 0.53 | -104.12 ± 0.66 | 11.87 ± 0.27 |
| 10 | **-88.75 ± 0.12** | **-92.91 ± 0.14** | **-72.38 ± 0.10** | 20.53 ± 0.12 | -90.93 ± 0.14 | -95.05 ± 0.17 | -74.66 ± 0.24 | 20.39 ± 0.08 | -90.73 ± 0.12 | -95.12 ± 0.11 | -74.88 ± 0.19 | 20.23 ± 0.11 |
| 20 | **-83.67 ± 0.09** | **-87.64 ± 0.11** | **-61.87 ± 0.14** | 25.77 ± 0.07 | -86.83 ± 0.08 | -91.96 ± 0.08 | -62.96 ± 0.13 | 29.00 ± 0.06 | -86.80 ± 0.08 | -91.92 ± 0.06 | -62.67 ± 0.32 | 29.25 ± 0.31 |
| 40 | **-83.24 ± 0.05** | **-87.40 ± 0.07** | **-60.77 ± 0.11** | 26.63 ± 0.14 | -157.63 ± 0.39 | -188.14 ± 0.03 | -187.00 ± 0.03 | 1.15 ± 0.00 | -89.00 ± 0.11 | -96.09 ± 0.10 | -62.02 ± 0.18 | 34.08 ± 0.27 |

## Reference: the paper's Table 1 (MLP encoder/decoder, 10 runs)

| d | N-VAE LL | N-VAE RE | N-VAE KL | S-VAE LL | S-VAE RE | S-VAE KL |
|---|---|---|---|---|---|---|
| 2 | -135.73 | -129.84 | 7.24 | -132.50 | -126.43 | 7.28 |
| 5 | -110.21 | -100.16 | 12.82 | -108.43 | -97.84 | 13.35 |
| 10 | -93.84 | -78.93 | 19.44 | -93.16 | -77.03 | 20.67 |
| 20 | -88.90 | -71.29 | 23.50 | -89.02 | -67.65 | 28.50 |
| 40 | -88.93 | -71.14 | 23.77 | -90.87 | -67.75 | 33.50 |

## Observations to check before writing them up

- The Gaussian reproduces the paper's Gaussian closely (KL within ~1 nat, RE within ~1-4 nats), so the
  pipeline looks sound; our conv model is a few nats better on LL at d=5-40.
- The vMF baseline does **not** reproduce the paper's S-VAE: at d=2 LL -150.0 (paper -132.5) with KL 4.58 (paper
  7.28), and at d=40 it is fully collapsed (KL 1.15, RE -187, i.e. the mean image). The d=2 and d=40 vMF cells
  should be treated as a suspect baseline until investigated (posterior collapse in the restored vMF code
  or its interaction with the warm-up / early-stopping protocol), not as evidence against spheres.
- TNBBeta matches vMF at d=5-20 (within noise) and, unlike vMF, does not collapse at d=40 (LL -89.0);
  the Gaussian is best in every row that has all three models. TNBBeta at d=2 was not yet finished.

## Correction (found by reading the authors' code, s-vae-pytorch)

The vMF and TNBBeta rows above give the sphere **ambient dimension d** (S^(d-1)), i.e. one degree of
freedom fewer than the Gaussian's R^d. The paper's d is the **manifold dimension**: its d=2 S-VAE is S^2
in R^3 (Figure 2: "Hammer projection of S^2 latent space of the S-VAE"), and the reference example trains
the vMF model with `z_dim + 1`. So our d=2 vMF was a circle, which explains its poor result (LL -150 vs the
Gaussian's -132.6), and none of the sphere rows are like-for-like with the Gaussian or the paper. The
comparable runs use ambient d + 1 (`vmfs`, `tnbs`, `vmfks` in `scripts/slurm/mnist_sweep.sbatch`).
The separate d=40 collapse is a real initialization problem (see the vMF kappa notes) and is fixed by
`initial_kappa`.
