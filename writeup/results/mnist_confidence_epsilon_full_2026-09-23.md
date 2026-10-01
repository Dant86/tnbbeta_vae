# MNIST confidence-only k-NN probe, TNBBeta epsilon variant, 2026-09-23

Same protocol as the p/kappa confidence probe, but using TNBBeta's epsilon (the cap's thickness
in the collapsed q=0 special case -- see `mnist_confidence_probe_full_2026-09-23.md`) instead of
p, since p saturates near 1 with almost no spread and cannot carry class information by
construction. Produced by:

```
uv run --no-sync python -m apps.eval.svae_table --kind confidence_epsilon --models tnbs
```

| d | TNBBeta (S^d) N=100 | TNBBeta (S^d) N=600 | TNBBeta (S^d) N=1000 |
|---|---|---|---|
| 2 | 0.13 ± 0.01 | 0.13 ± 0.01 | 0.13 ± 0.01 |
| 5 | 0.14 ± 0.01 | 0.14 ± 0.01 | 0.14 ± 0.01 |
| 10 | 0.16 ± 0.03 | 0.16 ± 0.03 | 0.16 ± 0.03 |
| 20 | 0.26 ± 0.02 | 0.27 ± 0.02 | 0.27 ± 0.02 |
| 40 | 0.30 ± 0.00 | 0.30 ± 0.00 | 0.30 ± 0.00 |

## Compared against the p/kappa table (mnist_confidence_probe_full_2026-09-23.md)

| d | TNBBeta p | TNBBeta epsilon | vMF kappa (S^d) | vMF kappa (kappa-init, S^d) | N-VAE mean std |
|---|---|---|---|---|---|
| 2 | 0.15 | 0.13 | 0.21 | 0.21 | 0.25 |
| 5 | 0.20 | 0.14 | 0.22 | 0.22 | 0.19 |
| 10 | 0.23 | 0.16 | 0.24 | 0.24 | 0.24 |
| 20 | 0.26 | 0.26-0.27 | 0.26 | 0.26 | **0.29** |
| 40 | 0.25-0.26 | **0.30** | 0.16 (collapsed) | 0.25 | **0.30** |

(N=1000 column shown; the pattern is the same at N=100/600.)

## Reading it

- **At d=2, 5, 10: epsilon carries *less* class signal than p, not more.** Both are weak (well
  below vMF's kappa and the Gaussian's mean std), meaning at low/mid dimension neither of
  TNBBeta's scalar parameters is doing meaningful per-example class-discriminative work -- the
  direction alone is carrying essentially all of it (consistent with Table 2's high direction-
  based accuracy at these dimensions).
- **At d=40: epsilon (0.30) matches the Gaussian's best confidence result and clearly beats
  vMF's own (properly-initialized, non-collapsed) kappa (0.25) by 5 points with ~0 std on both
  sides** -- this is a real, reproducible gap, not noise. This is the first genuinely
  TNBBeta-specific advantage found anywhere in the reproduction: not "ties vMF," but "beats it,"
  and only once the correct parameter is probed.
- Epsilon "wakes up" sharply between d=10 and d=20 (0.16 -> 0.26), right where TNBBeta's other
  results (Tables 1-2) also start separating from low-dimension behavior. Whatever mechanism
  makes epsilon class-informative appears to be a higher-dimension phenomenon specifically.

## Implication for a fixed-epsilon ablation

Since epsilon carries close to no information at low dimension (0.13-0.16, barely above the 0.10
chance level) but real information at d=40 (0.30), fixing epsilon to a constant is a much lower-
stakes intervention at low dimension (there is little class-relevant signal to lose there, so any
resulting change in p/q behavior would be a clean reallocation to observe) than at d=40 (where
epsilon now appears to be doing real, valuable work, and removing its freedom risks destroying a
capability rather than revealing a hidden one). Suggests starting the {0.5, 1, 1.5} fixed-epsilon
sweep at a low dimension (e.g. d=5) before risking it at d=40.
