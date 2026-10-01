# MNIST Table 1, with Power Spherical as a fourth model, 2026-10-01

- **Date:** 2026-10-01
- **Master commit:** `ff2a12b` (merge of PR #39/#40, "Add Power Spherical as a
  fourth latent baseline", 2026-10-01)
- **Training command:** `sbatch --array=210-234 scripts/slurm/mnist_sweep.sbatch`
  (the `pss` row: `conv_power_spherical_vae`, ambient $d+1$, 5 dims $\times$ 5
  seeds)
- **Table command:**
  ```
  uv run python -m apps.eval.svae_table --models gauss vmfs tnbs pss
  ```

Reproduces the existing `mnist_table1_full_2026-09-23.md` three-model table
(N-VAE / S-VAE (vMF) / TNBBeta, paper-convention ambient $d+1$ sphere models),
with Power Spherical added as a fourth column after the `pss` sweep above.

| d | N-VAE LL | N-VAE L[q] | N-VAE RE | N-VAE KL | S-VAE (vMF, S^d) LL | S-VAE (vMF, S^d) L[q] | S-VAE (vMF, S^d) RE | S-VAE (vMF, S^d) KL | TNBBeta (S^d) LL | TNBBeta (S^d) L[q] | TNBBeta (S^d) RE | TNBBeta (S^d) KL | Power Spherical (S^d) LL | Power Spherical (S^d) L[q] | Power Spherical (S^d) RE | Power Spherical (S^d) KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | -132.64 ± 0.30 | -136.81 ± 0.28 | -129.19 ± 0.35 | 7.62 ± 0.10 | -130.86 ± 0.43 | -134.42 ± 0.42 | -126.89 ± 0.47 | 7.53 ± 0.08 | -130.11 ± 0.57 | -134.21 ± 0.46 | -126.43 ± 0.54 | 7.78 ± 0.11 | -131.02 ± 0.39 | -134.56 ± 0.28 | -127.03 ± 0.40 | 7.53 ± 0.13 |
| 5 | -105.38 ± 0.11 | -109.53 ± 0.09 | -96.12 ± 0.12 | 13.41 ± 0.17 | -105.27 ± 0.14 | -109.32 ± 0.08 | -95.56 ± 0.12 | 13.76 ± 0.06 | -104.89 ± 0.22 | -109.15 ± 0.22 | -95.36 ± 0.26 | 13.79 ± 0.13 | -105.37 ± 0.17 | -109.38 ± 0.22 | -95.67 ± 0.22 | 13.71 ± 0.08 |
| 10 | **-88.75 ± 0.12** | **-92.91 ± 0.14** | -72.38 ± 0.10 | 20.53 ± 0.12 | -89.38 ± 0.02 | -93.60 ± 0.06 | -71.84 ± 0.08 | 21.77 ± 0.05 | -89.29 ± 0.10 | -93.65 ± 0.14 | -71.96 ± 0.25 | 21.69 ± 0.14 | -89.42 ± 0.10 | -93.69 ± 0.09 | -71.97 ± 0.10 | 21.71 ± 0.03 |
| 20 | **-83.67 ± 0.09** | **-87.64 ± 0.11** | -61.87 ± 0.14 | 25.77 ± 0.07 | -86.84 ± 0.04 | -92.10 ± 0.05 | -62.63 ± 0.21 | 29.47 ± 0.18 | -86.90 ± 0.23 | -92.12 ± 0.15 | -62.33 ± 0.41 | 29.78 ± 0.53 | -86.95 ± 0.08 | -92.17 ± 0.07 | -62.72 ± 0.18 | 29.45 ± 0.15 |
| 40 | -83.24 ± 0.05 | -87.40 ± 0.07 | -60.77 ± 0.11 | 26.63 ± 0.14 | -144.28 ± 31.06 | -170.06 ± 41.29 | -162.32 ± 56.08 | 7.73 ± 14.79 | -89.09 ± 0.15 | -96.38 ± 0.14 | -62.53 ± 0.42 | 33.86 ± 0.36 | -89.04 ± 0.11 | -96.37 ± 0.13 | -62.17 ± 0.23 | 34.20 ± 0.23 |

(Bold = best mean in the row, significant at $p<0.01$, Welch's $t$-test, among
the boldable metrics LL/L[q]/RE. Same convention as the three-model table.)

## Reading it

**d=2, 5, 10, 20: all three spherical models are statistically
indistinguishable from each other on every metric.** No cell among vMF,
TNBBeta, or Power Spherical is ever bolded relative to the others at these
dimensions -- this was already true for vMF vs. TNBBeta in the three-model
table, and Power Spherical slots into the same tie rather than breaking it.
N-VAE remains the only significant winner, at d=10 and d=20, exactly as
before.

**d=40 is the headline result: Power Spherical also avoids vMF's collapse,
and matches TNBBeta almost exactly.** Reference vMF collapses catastrophically
here (LL $-144.28\pm31.06$, a known failure mode without kappa-init -- see
`tnbbeta_vmf_geometric_convergence_2026-09-23.md`). Both TNBBeta
($-89.09\pm0.15$) and Power Spherical ($-89.04\pm0.11$) stay stable, with
means within noise of each other and neither bolded (consistent with them
being statistically tied, not with either being compared against vMF's wide,
unstable distribution). If anything, Power Spherical's standard deviation is
slightly *tighter* than TNBBeta's here (0.11 vs. 0.15 on LL), though this
difference is itself well within what 5 seeds can resolve.

**This directly confirms, rather than just anticipates, something flagged as
a real risk before Power Spherical was implemented:** "I would not assume
TNBBeta wins on high-dimension stability without running it -- Power
Spherical was explicitly designed to fix this exact vMF weakness." The data
now shows this plainly: the "TNBBeta is more stable than vMF at d=40" result
is a (TNBBeta, Power Spherical) vs. (uninitialized vMF) story, not a TNBBeta
vs. Power Spherical one.

**Minor secondary pattern, not significant anywhere:** Power Spherical's KL
tracks vMF's closely at d=2, 5, 20 (even very slightly below it at d=2 and
d=5), while TNBBeta tends to pay a touch more KL in exchange for a slightly
better RE at d=5 and d=10. Consistent with Power Spherical's construction
being "closer" to vMF's own shape (an asymmetric-Beta tilt vs. vMF's
exponential tilt) than TNBBeta's p/q-based route is, but the effect is small
and never survives the significance test.

## Implication for the week's plan

Table 1 now gives **zero points of aggregate differentiation between TNBBeta
and Power Spherical, anywhere in the sweep, including the one dimension
(d=40) where TNBBeta looked most impressive against plain vMF.** This makes
the provable expressivity-gap argument (non-containment + the
unimodal-vs-bimodal result, in progress) and a task that can actually exploit
that gap (the com-DBLP bridge-node experiment, in progress) the entire
remaining case for TNBBeta over Power Spherical -- not a nice-to-have on top
of a Table-1 win, the whole thing.
