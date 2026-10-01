# Semi-supervised MNIST (100 labels), first-look results, 2026-09-23

3 seeds, z1/z2 dim in {10, 50} (paper's manifold dimension; vMF's ambient is d+1), the paper's
three variants only (N+N, S+S, S+N; TNBBeta's T+T/T+N not yet run). Produced by:

```
uv run --no-sync python -m apps.semi_supervised.table --variants nn ss sn --dims 10 50 --seeds 0 1 2
```

| z1 dim | z2 dim | N+N | S+S | S+N |
|---|---|---|---|---|
| 10 | 10 | 85.24 ± 0.64 | 91.11 ± 2.80 | 94.14 ± 0.47 |
| 10 | 50 | 82.38 ± 3.75 | 84.72 ± 2.90 | 93.95 ± 0.25 |
| 50 | 10 | 82.24 ± 2.38 | - | 96.62 ± 0.65 |
| 50 | 50 | 82.60 ± 0.93 | 84.24 ± 0.30 | **96.81 ± 0.18** |

S+S at (z1=50, z2=10) is missing (some of its 3 tasks never produced `semi_supervised_final.json`,
likely lost in the earlier m001/timeout/corrupted-checkpoint chaos on that sweep) -- worth
backfilling if we want that cell, not urgent.

## Reading it

- **S+N (vMF z1 + Gaussian z2) is the clear winner everywhere it was run**, and the only variant
  that improves with dimension: 94.1% at (10,10) up to 96.8% at (50,50), a ~14 point margin over
  pure Gaussian (N+N, flat at 82-85% regardless of dimension). This reproduces the paper's
  headline M1+M2 claim well -- mixing a circular z1 (style/identity) with a Euclidean z2 gives the
  best semi-supervised classifier.
- Pure S+S is inconsistent (91.1 at (10,10) but only 84.2-84.7 elsewhere) and does not reliably
  beat N+N, unlike S+N.
- This is the first unambiguous "sphere latent helps" result that isn't a wash with the Gaussian --
  much stronger than Table 2's k-NN margins. It says nothing about TNBBeta yet: T+T and T+N were
  not part of this first-look sweep.
