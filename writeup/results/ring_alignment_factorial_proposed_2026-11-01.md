# Proposed, not run: why did the single-axis ring study fail to align on half the seeds?

Date: 2026-11-01. Base commit `667cfe1`. This is a note, not a result: no run was made.

## The open question

Single-axis levels 2-3 (`ring_posterior_l23_d3_2026-11-01.md`): free TNBBeta recovered single-ring
posteriors on 5 of 10 seeds (mean exact KL 0.12 on those x) and stayed near cap level on the other
5 (mean direction not aligned with the axis, |cos| 0.63-0.72). The four-class study
(`ring_class_designs_results_2026-11-01.md`) aligned on 10 of 10 seeds in both designs
(mean KL 0.065 and 0.127). The cause of the difference is not identified.

## What differed between the two studies (confounded)

| | single-axis study | four-class study |
|---|---|---|
| observation map `h` | `[w, sin 3w, cos 5w]` (steep: posterior sd in `w` about 0.035-0.12) | `[w, sin 1.5w, cos 2w]` (gentler, thicker rings) |
| second latitude mode | 18.9% of test x | 0.000 |
| epochs | 100 | 300 |
| TNBBeta head init | raw 0 (`p = q = 0.5`), later also ln 3 on raw p | library default ln 3 on raw p |
| log q in the ELBO | clamped, biased at eps < 0.3 | exact (patched sampler) |
| latent class / axes | one axis | one axis (design 1) or four (design 2) |

## Proposed factorial (not scheduled)

Single axis, `d = 3`, true frozen decoder, TNBBeta only, 10 seeds, exact KL on 2000 test x:
`h` steep versus gentle x epochs 100 versus 300 x p init offset 0 versus ln 3 (8 cells x 10 seeds
= 80 runs, about 20 minutes on SLURM with fan-out), reporting the fraction of seeds aligned
(median |cos(mu, a)| >= 0.99) and the mean KL. Separately, with the steep `h`, the oracle mean
direction arm (`fixed_mean_direction`) to see whether alignment is the whole gap.

## Why it matters for the HRTF study

The ITD-only HRTF task has thin rings (sd of `w` about 0.085, about 5 degrees) set by the data, not
chosen; if thin rings plus short training cause misalignment, the HRTF run may reproduce the
single-axis failure rather than the four-class success. The HRTF study therefore reports alignment
per split and fixes the same 300 epochs as the four-class study.
