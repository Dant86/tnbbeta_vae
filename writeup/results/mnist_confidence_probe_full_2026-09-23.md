# MNIST confidence-only k-NN probe, full 5-seed results, 2026-09-23

k-NN class probe using only the posterior's confidence scalar (Gaussian mean std, vMF/TNBBeta
kappa/p), no direction. Paper-convention sphere models. Produced by:

```
uv run --no-sync python -m apps.eval.svae_table --kind confidence --models gauss vmfs tnbs vmfks
```

| d | N-VAE N=100 | N-VAE N=600 | N-VAE N=1000 | S-VAE (vMF, S^d) N=100 | S-VAE (vMF, S^d) N=600 | S-VAE (vMF, S^d) N=1000 | TNBBeta (S^d) N=100 | TNBBeta (S^d) N=600 | TNBBeta (S^d) N=1000 | S-VAE (vMF, kappa init, S^d) N=100 | S-VAE (vMF, kappa init, S^d) N=600 | S-VAE (vMF, kappa init, S^d) N=1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.25 ± 0.03 | 0.25 ± 0.03 | 0.25 ± 0.03 | 0.21 ± 0.01 | 0.21 ± 0.00 | 0.21 ± 0.00 | 0.15 ± 0.03 | 0.15 ± 0.03 | 0.15 ± 0.04 | 0.21 ± 0.01 | 0.21 ± 0.01 | 0.21 ± 0.01 |
| 5 | 0.19 ± 0.01 | 0.19 ± 0.01 | 0.19 ± 0.01 | 0.22 ± 0.00 | 0.22 ± 0.01 | 0.22 ± 0.01 | 0.20 ± 0.01 | 0.20 ± 0.01 | 0.20 ± 0.01 | 0.22 ± 0.00 | 0.22 ± 0.00 | 0.22 ± 0.00 |
| 10 | 0.24 ± 0.00 | 0.24 ± 0.00 | 0.24 ± 0.00 | 0.24 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.23 ± 0.01 | 0.24 ± 0.01 | 0.24 ± 0.01 | 0.24 ± 0.00 | 0.25 ± 0.00 | 0.24 ± 0.00 |
| 20 | **0.29 ± 0.01** | **0.29 ± 0.01** | **0.29 ± 0.01** | 0.26 ± 0.00 | 0.26 ± 0.00 | 0.26 ± 0.00 | 0.26 ± 0.00 | 0.26 ± 0.00 | 0.26 ± 0.00 | 0.26 ± 0.00 | 0.26 ± 0.00 | 0.26 ± 0.00 |
| 40 | **0.30 ± 0.00** | **0.30 ± 0.00** | **0.30 ± 0.00** | 0.16 ± 0.06 | 0.16 ± 0.06 | 0.16 ± 0.06 | 0.25 ± 0.00 | 0.26 ± 0.00 | 0.26 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 | 0.25 ± 0.00 |

## Reading it

- The hypothesis motivating this probe -- that TNBBeta's p routes class information through a
  channel vMF's kappa structurally lacks -- **is not supported**. At d=2, TNBBeta's p carries
  *less* class signal than every other family's confidence scalar (0.15 vs 0.21-0.25). At every
  other dimension TNBBeta tracks vMF/vmfk almost exactly.
- **The Gaussian's mean posterior std is the most class-informative confidence scalar at d=20 and
  d=40**, significantly beating every sphere variant. This is a genuine, unanticipated finding
  (not something the paper measures) -- worth a sentence in the write-up, low priority to chase
  further.
- At d=40, `vmfs`'s collapse shows up here too (0.16 ± 0.06, both a lower mean and a much larger
  std than `tnbs`/`vmfks`), consistent with the same partially-collapsed seeds seen in Tables 1-2.
- All chance-level accuracy is 0.10 (10 classes); every value here is only modestly above chance,
  confirming a single confidence scalar was never going to carry much class information for any
  family -- the interesting result is the *relative* comparison, not the absolute numbers.
