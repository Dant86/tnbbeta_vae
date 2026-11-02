# Four-class ring designs with a learned decoder: results (2026-11-01)

Date: 2026-11-01. Base commit `945f133` (branch `synthetic-graph-families`) plus the UNCOMMITTED working tree (new:
`apps/synthetic/ring_classes_learned.py`, `ring_classes_learned_plot.py`, `scripts/slurm/ring_classes_learned.sbatch`,
`tests/apps/test_ring_classes_learned.py`, `tests/data/test_ring_classes_offsets.py`; additive edit to
`src/tnbbeta_vae/data/ring_classes.py`: `OffsetRingClassTask`, `class_identifiability`; the ITD/HRTF files in the tree are unrelated).
Pre-registration (frozen; Deviations appended before any run): `ring_class_designs_learned_decoder_preregistration_2026-11-01.md`.

## Commands

    # identifiability check (login node): offsets scale 1.0 passed, 99.425% of 20,000 x have p(c|x) > 0.99
    PYTHONPATH=src python -c "from tnbbeta_vae.data.ring_classes import *; print(class_identifiability(offset_ring_class_task(1.0)))"
    # 240 runs, 14 array tasks x 6 workers sharing a GPU
    PROCS=6 sbatch --array=0-13 scripts/slurm/ring_classes_learned.sbatch      # job 2290776, then 2291018 (see Deviations)
    uv run python -m apps.synthetic.ring_classes_learned summarize --out-dir ring_recovery/classes_learned
    uv run python -m apps.synthetic.ring_classes_learned_plot --out-dir ring_recovery/classes_learned --fig-dir writeup/results/figures
    # quality: ruff format/check, pyright clean on everything added; full pytest as SLURM CPU job 2291536: 990 passed, 1 skipped

All 240 runs finished, 0 divergences. Raw outputs: `ring_recovery/classes_learned/<setting>/<decoder>/<arm>/s<seed>.{json,npz,pt}` (gitignored).
The tables in "Full tables" below are the verbatim output of `summarize`.

## Headline against the pre-registered rules

**TNBBeta does NOT help a learned-decoder VAE (`mlp` decoder) in any setting.** All six one-sided paired Wilcoxon tests (H1: `tnbbeta`
has the smaller ELBO gap) give p = 1.0000 (scipy and sign-pattern enumeration agree), TNBBeta better on 0/10 seeds in every comparison;
nothing survives Holm and the margin fails. The sign is the OPPOSITE of the hypothesis (mean paired differences comparator - tnbbeta,
nats; negative = TNBBeta worse):

| setting | vs vMF | vs PowerSpherical | verdict |
|---|---|---|---|
| D1 | -0.113 (p 1.0, 0/10) | -0.119 (p 1.0, 0/10) | not helped |
| D2g | -0.096 (p 1.0, 0/10) | -0.105 (p 1.0, 0/10) | not helped |
| D2l | -0.101 (p 1.0, 0/10) | -0.087 (p 1.0, 0/10) | not helped |

Mean test ELBO gaps (nats, mlp decoder): D1 vMF 0.414, PS 0.408, TNBBeta 0.527; D2g 0.455 / 0.446 / 0.551; D2l 0.579 / 0.592 / 0.680.
The importance-weighted (K=64) gaps shrink to 0.13-0.26 and the family differences largely vanish there (D2g TNBBeta 0.158 vs vMF 0.172;
D2l 0.264 vs 0.255 vs 0.260), i.e. much of the ELBO deficit is looseness of the TNBBeta posterior bound, not a worse model (descriptive).

## Descriptive findings (no p-values; all exploratory except the six tests above)

1. **`tnbbeta` (library p init) always converges to a cap**: median q = 1e-6 (floor), p ~ 0.95-0.99, `mlp` decoder, ring statistic
   0.88-0.98 (cap off the decoder axis); learned decoders with 128-128 hidden units absorb whatever posterior shape the family gives.
2. **`tnbbeta_init0` (p init 0) sometimes finds a ring, and then wins.** mlp decoder, seeds with ring R ~ 0.05 and eta^2 ~ 1 (decoder
   became single-index): D1 7/10 seeds (gap 0.038-0.045 vs caps 0.41), D2g 4/10 (gap ~0.19 vs caps 0.45), D2l 4/10 (gaps 0.664, 0.637,
   0.701, 0.423, still not better than caps 0.58 on most). The other seeds are caps with gaps ~0.5-0.57. Mean paired gap
   `tnbbeta_init0` - vMF: D1 -0.229 (7/10 seeds better), D2g -0.048 (4/10), D2l +0.054 (2/10). This arm is descriptive (not in the
   confirmatory family) and the ring/cap split is bimodal over seeds, so means hide it. Where the ring forms the decoder is
   single-index (eta^2 0.88-1.0) versus 0.5-0.6 for cap runs: the data are single-index-representable, and what is learned is a joint
   (decoder, posterior) solution.
3. **Forced single-index decoder (`single_index`)**: caps lose by ~1 nat (D1 vMF 1.104, PS 0.988; D2g 1.115, 1.003; D2l 2.050, 1.947) and
   learn sigma 0.35-0.94 (true 0.2); TNBBeta reaches 0.08 / 0.11 / 0.46 (init0: 0.047 / 0.086 / 0.452), sigma 0.20-0.22, ring R 0.05, 10/10
   seeds better than both caps. That is the ring-forcing result of the frozen-decoder study and says the families differ in representing
   the ring, not that the data require one. Note the learned `mlp` decoder reaches a cap-based gap of 0.41 on D1, far better than
   the cap-based `single_index` (1.10).
4. **Class structure.** D2l `mlp`: 5-NN accuracy from posterior centres 1.000 for vMF, PS and TNBBeta (0.971 for init0; the lower-accuracy
   init0 seeds are the collapsed/ring ones, fig. D2l mlp s8 shows init0 latent collapsing to two points). D1 (band recovery, chance 0.25): 0.87-0.88
   caps and TNBBeta, 0.84 init0. D1 mean |u*.centre|: 0.57 caps/TNBBeta (centres not on the axis), 0.87 init0 (mixture of ring seeds 1.0 and cap seeds 0.57).
5. Learned sigma (mlp) is 0.17-0.22 in all arms (true 0.2); TNBBeta (p init ln 3) D1 0.173 +- 0.027.
6. Decoder single-indexness: eta^2 of `mlp` decoders is 0.50-0.60 for cap runs and ~1 for ring runs; in D2g the per-class axes `u*_c` are
   used (mean over classes).

## What this does and does not show

* Shows, for this task, optimizer, 300 epochs and 10 seeds: with a free 128-128 decoder, TNBBeta with its default initialisation gives a
  WORSE ELBO than vMF and PS in every setting, on every seed, and the IWAE bound narrows this to near zero in two of three settings.
  A ring-shaped posterior is not what the free-decoder optimum uses by default.
* Shows that a ring-capable posterior can be found (init0) and then gives a markedly better bound on D1 (and partly D2g), but only on
  some seeds and with a decoder that happened to become single-index; it is exploratory, selected by a descriptive arm, not confirmed.
* Does not show that the data require TNBBeta (the cap-based mlp gap is small: ~0.4 nats of a log-likelihood of order 2 and is
  largely inference looseness: IWAE gap 0.13-0.26), nor that TNBBeta is intrinsically worse (the loss is an ELBO/optimisation effect;
  the exact-log-q estimator used for TNBBeta vs the analytic KL for caps is not equalised here, cf. `ring_condition_b_unpacked_2026-11-01.md`
  where that was ruled out for the single-axis task).
* Latents are identified only up to rotations/reparametrisations; the figures use the decoder's estimated axis `u*` per run and family,
  no metric compares to the true `z`. Posterior means understate rings (compare the draws figure, D1 mlp).
* Gap for TNBBeta vs caps compares different estimators (MC log q vs analytic KL); evaluation uses 256 draws so noise is small
  (per-run SE of the per-x gap is in the JSON, `gap_elbo_se`) relative to the 0.1-nat differences.

## Deviations (see also the Deviations section of the pre-registration)

* Offset scale 1.0 (identifiability check passed, 99.425%); no rescaling.
* Job 2290776 (the 240-run array) was cancelled by another agent about 38 minutes in, by mistake; the same command was resubmitted as job
  2291018 with identical code and seeds. Finished cells were kept (skipped), cells in progress restarted from scratch; results are
  otherwise unaffected up to GPU nondeterminism.
* Operationalisations (posterior mean = `posterior_centre`, eta^2 and ring statistic definitions, IWAE stub prior, fixed Holm family
  of six, D2g per-class axes) are in the pre-registration's Deviations section. No run diverged; none was excluded.
* Figures: one figure per setting x decoder (median-gap, lower median, `tnbbeta` seed), columns vMF, PowerSpherical, TNBBeta and
  TNBBeta (p init 0) (the fourth is an addition); D2g draws four arrows `u*_c`; the draws-check figure is D1 / mlp.

## Figures (`writeup/results/figures/`)

* `figures/ring_classes_learned_D1_mlp_s1_2026-11-01.png`
* `figures/ring_classes_learned_D1_single_index_s4_2026-11-01.png`
* `figures/ring_classes_learned_D2g_mlp_s2_2026-11-01.png`
* `figures/ring_classes_learned_D2g_single_index_s2_2026-11-01.png`
* `figures/ring_classes_learned_D2l_mlp_s8_2026-11-01.png`
* `figures/ring_classes_learned_D2l_single_index_s1_2026-11-01.png`
* `figures/ring_classes_learned_draws_D1_mlp_s1_2026-11-01.png` (posterior draws instead of means)

## Full tables (verbatim `summarize` output)

## Per-arm test ELBO gap (nats; lower is better)

| setting | decoder | arm | seeds (valid) | gap mean +- sd | median seed | min / max | IWAE gap mean +- sd |
|---|---|---|---|---|---|---|---|
| D1 | mlp | vmf | 10/10 | 0.4138 +- 0.0062 | 0.4141 | 0.4035 / 0.4253 | 0.1505 +- 0.0135 |
| D1 | mlp | power_spherical | 10/10 | 0.4079 +- 0.0104 | 0.4079 | 0.3894 / 0.4225 | 0.1269 +- 0.0202 |
| D1 | mlp | tnbbeta | 10/10 | 0.5268 +- 0.0181 | 0.5252 | 0.5030 / 0.5681 | 0.1867 +- 0.0482 |
| D1 | mlp | tnbbeta_init0 | 10/10 | 0.1846 +- 0.2305 | 0.0440 | 0.0382 / 0.5355 | 0.0574 +- 0.0709 |
| D1 | single_index | vmf | 10/10 | 1.1040 +- 0.0023 | 1.1038 | 1.1012 / 1.1081 | 0.7187 +- 0.0202 |
| D1 | single_index | power_spherical | 10/10 | 0.9884 +- 0.0062 | 0.9861 | 0.9821 / 1.0025 | 0.6671 +- 0.0187 |
| D1 | single_index | tnbbeta | 10/10 | 0.0817 +- 0.0579 | 0.0531 | 0.0376 / 0.2041 | 0.0263 +- 0.0206 |
| D1 | single_index | tnbbeta_init0 | 10/10 | 0.0471 +- 0.0161 | 0.0416 | 0.0351 / 0.0886 | 0.0162 +- 0.0075 |
| D2g | mlp | vmf | 10/10 | 0.4550 +- 0.0099 | 0.4545 | 0.4386 / 0.4727 | 0.1716 +- 0.0116 |
| D2g | mlp | power_spherical | 10/10 | 0.4460 +- 0.0077 | 0.4469 | 0.4349 / 0.4633 | 0.1300 +- 0.0164 |
| D2g | mlp | tnbbeta | 10/10 | 0.5511 +- 0.0082 | 0.5525 | 0.5346 / 0.5610 | 0.1583 +- 0.0096 |
| D2g | mlp | tnbbeta_init0 | 10/10 | 0.4071 +- 0.1903 | 0.5439 | 0.1826 / 0.5702 | 0.1216 +- 0.0536 |
| D2g | single_index | vmf | 10/10 | 1.1154 +- 0.0018 | 1.1154 | 1.1124 / 1.1182 | 0.7704 +- 0.0189 |
| D2g | single_index | power_spherical | 10/10 | 1.0027 +- 0.0080 | 1.0037 | 0.9894 / 1.0179 | 0.7026 +- 0.0147 |
| D2g | single_index | tnbbeta | 10/10 | 0.1142 +- 0.0366 | 0.1087 | 0.0548 / 0.1774 | 0.0367 +- 0.0192 |
| D2g | single_index | tnbbeta_init0 | 10/10 | 0.0862 +- 0.0346 | 0.0835 | 0.0470 / 0.1511 | 0.0304 +- 0.0147 |
| D2l | mlp | vmf | 10/10 | 0.5786 +- 0.0347 | 0.5686 | 0.5503 / 0.6656 | 0.2547 +- 0.0104 |
| D2l | mlp | power_spherical | 10/10 | 0.5922 +- 0.0212 | 0.5933 | 0.5693 / 0.6355 | 0.2595 +- 0.0119 |
| D2l | mlp | tnbbeta | 10/10 | 0.6795 +- 0.0158 | 0.6794 | 0.6531 / 0.7126 | 0.2644 +- 0.0110 |
| D2l | mlp | tnbbeta_init0 | 10/10 | 0.6324 +- 0.0814 | 0.6555 | 0.4230 / 0.7007 | 0.2653 +- 0.0416 |
| D2l | single_index | vmf | 10/10 | 2.0502 +- 0.0026 | 2.0493 | 2.0478 / 2.0556 | 1.8929 +- 0.0079 |
| D2l | single_index | power_spherical | 10/10 | 1.9471 +- 0.0034 | 1.9458 | 1.9439 / 1.9544 | 1.8509 +- 0.0062 |
| D2l | single_index | tnbbeta | 10/10 | 0.4580 +- 0.1762 | 0.4551 | 0.2175 / 0.7347 | 0.2353 +- 0.1334 |
| D2l | single_index | tnbbeta_init0 | 10/10 | 0.4517 +- 0.1873 | 0.3856 | 0.2876 / 0.9027 | 0.2431 +- 0.1278 |

## Per-seed ELBO gap


D1 / mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.4186 | 0.4071 | 0.5379 | 0.0452 |
| 1 | 0.4035 | 0.3954 | 0.5248 | 0.5064 |
| 2 | 0.4143 | 0.4060 | 0.5255 | 0.0451 |
| 3 | 0.4253 | 0.4157 | 0.5681 | 0.0382 |
| 4 | 0.4139 | 0.4225 | 0.5030 | 0.5135 |
| 5 | 0.4069 | 0.4211 | 0.5315 | 0.0403 |
| 6 | 0.4171 | 0.4087 | 0.5141 | 0.0428 |
| 7 | 0.4164 | 0.4101 | 0.5343 | 0.0382 |
| 8 | 0.4127 | 0.3894 | 0.5148 | 0.5355 |
| 9 | 0.4095 | 0.4033 | 0.5137 | 0.0409 |

D1 / single_index

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 1.1022 | 0.9877 | 0.0540 | 0.0543 |
| 1 | 1.1046 | 0.9852 | 0.2041 | 0.0371 |
| 2 | 1.1057 | 0.9860 | 0.0459 | 0.0449 |
| 3 | 1.1012 | 0.9932 | 0.0415 | 0.0499 |
| 4 | 1.1022 | 0.9837 | 0.0522 | 0.0351 |
| 5 | 1.1056 | 0.9821 | 0.0447 | 0.0491 |
| 6 | 1.1056 | 1.0025 | 0.1453 | 0.0383 |
| 7 | 1.1029 | 0.9928 | 0.1339 | 0.0886 |
| 8 | 1.1081 | 0.9861 | 0.0576 | 0.0375 |
| 9 | 1.1014 | 0.9845 | 0.0376 | 0.0360 |

D2g / mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.4678 | 0.4446 | 0.5427 | 0.5574 |
| 1 | 0.4727 | 0.4491 | 0.5576 | 0.5702 |
| 2 | 0.4534 | 0.4474 | 0.5519 | 0.1908 |
| 3 | 0.4483 | 0.4464 | 0.5346 | 0.5470 |
| 4 | 0.4464 | 0.4410 | 0.5532 | 0.1826 |
| 5 | 0.4386 | 0.4474 | 0.5549 | 0.1865 |
| 6 | 0.4560 | 0.4379 | 0.5466 | 0.1848 |
| 7 | 0.4530 | 0.4477 | 0.5610 | 0.5530 |
| 8 | 0.4578 | 0.4633 | 0.5486 | 0.5574 |
| 9 | 0.4555 | 0.4349 | 0.5603 | 0.5408 |

D2g / single_index

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 1.1137 | 0.9940 | 0.1269 | 0.1151 |
| 1 | 1.1141 | 1.0056 | 0.0548 | 0.0858 |
| 2 | 1.1161 | 1.0056 | 0.0995 | 0.1511 |
| 3 | 1.1180 | 0.9966 | 0.1179 | 0.0812 |
| 4 | 1.1182 | 1.0046 | 0.1774 | 0.1233 |
| 5 | 1.1144 | 1.0016 | 0.0840 | 0.0632 |
| 6 | 1.1162 | 1.0179 | 0.0942 | 0.0550 |
| 7 | 1.1124 | 1.0028 | 0.1475 | 0.0901 |
| 8 | 1.1160 | 0.9894 | 0.0910 | 0.0506 |
| 9 | 1.1148 | 1.0087 | 0.1486 | 0.0470 |

D2l / mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.5555 | 0.5940 | 0.6751 | 0.6735 |
| 1 | 0.5718 | 0.6355 | 0.6796 | 0.6747 |
| 2 | 0.5747 | 0.6086 | 0.6908 | 0.6140 |
| 3 | 0.5504 | 0.5695 | 0.6531 | 0.6643 |
| 4 | 0.6656 | 0.6018 | 0.6803 | 0.5916 |
| 5 | 0.5503 | 0.5752 | 0.6638 | 0.6367 |
| 6 | 0.5845 | 0.5693 | 0.6860 | 0.6988 |
| 7 | 0.6036 | 0.5925 | 0.7126 | 0.7007 |
| 8 | 0.5654 | 0.5725 | 0.6792 | 0.4230 |
| 9 | 0.5639 | 0.6029 | 0.6747 | 0.6467 |

D2l / single_index

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 2.0510 | 1.9511 | 0.2560 | 0.9027 |
| 1 | 2.0488 | 1.9446 | 0.4073 | 0.4060 |
| 2 | 2.0484 | 1.9439 | 0.2175 | 0.3443 |
| 3 | 2.0478 | 1.9440 | 0.5628 | 0.3043 |
| 4 | 2.0494 | 1.9452 | 0.7347 | 0.5226 |
| 5 | 2.0492 | 1.9468 | 0.5029 | 0.2876 |
| 6 | 2.0556 | 1.9544 | 0.6539 | 0.3651 |
| 7 | 2.0505 | 1.9458 | 0.3372 | 0.4777 |
| 8 | 2.0478 | 1.9457 | 0.5872 | 0.3175 |
| 9 | 2.0538 | 1.9490 | 0.3207 | 0.5890 |

## Confirmatory tests (one-sided paired Wilcoxon, Holm at 0.05)

| setting | comparator | n | mean paired diff (comparator - tnbbeta) | seeds tnbbeta better | p (scipy) | p (enumerated) | Holm threshold | Holm reject | margin >= 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| D1 | power_spherical | 10 | -0.1188 | 0/10 | 1.00000 | 1.00000 | 0.01000 | False | False |
| D1 | vmf | 10 | -0.1130 | 0/10 | 1.00000 | 1.00000 | 0.00833 | False | False |
| D2g | power_spherical | 10 | -0.1052 | 0/10 | 1.00000 | 1.00000 | 0.01667 | False | False |
| D2g | vmf | 10 | -0.0962 | 0/10 | 1.00000 | 1.00000 | 0.01250 | False | False |
| D2l | power_spherical | 10 | -0.0874 | 0/10 | 1.00000 | 1.00000 | 0.05000 | False | False |
| D2l | vmf | 10 | -0.1010 | 0/10 | 1.00000 | 1.00000 | 0.02500 | False | False |

Verdict 'TNBBeta helps a learned-decoder VAE' (both Holm tests survive and 
both mean paired differences are at least 0.05 nats):

| setting | Holm (both) | margin (both) | verdict |
|---|---|---|---|
| D1 | False | False | False |
| D2g | False | False | False |
| D2l | False | False | False |

## Per-seed importance-weighted (K=64) gap


D1 / mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1702 | 0.1468 | 0.2131 | 0.0166 |
| 1 | 0.1386 | 0.1226 | 0.2110 | 0.1290 |
| 2 | 0.1432 | 0.1341 | 0.1601 | 0.0167 |
| 3 | 0.1768 | 0.1501 | 0.2425 | 0.0153 |
| 4 | 0.1429 | 0.1101 | 0.1022 | 0.1951 |
| 5 | 0.1447 | 0.0913 | 0.2209 | 0.0131 |
| 6 | 0.1510 | 0.1389 | 0.1517 | 0.0162 |
| 7 | 0.1555 | 0.1002 | 0.2181 | 0.0108 |
| 8 | 0.1469 | 0.1431 | 0.2236 | 0.1483 |
| 9 | 0.1351 | 0.1316 | 0.1240 | 0.0124 |

D1 / single_index

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.7415 | 0.6882 | 0.0177 | 0.0221 |
| 1 | 0.7151 | 0.6724 | 0.0674 | 0.0103 |
| 2 | 0.7130 | 0.6497 | 0.0193 | 0.0171 |
| 3 | 0.7033 | 0.6576 | 0.0176 | 0.0149 |
| 4 | 0.7134 | 0.6534 | 0.0164 | 0.0102 |
| 5 | 0.7087 | 0.6620 | 0.0116 | 0.0134 |
| 6 | 0.6966 | 0.6917 | 0.0617 | 0.0128 |
| 7 | 0.7641 | 0.6748 | 0.0235 | 0.0348 |
| 8 | 0.7061 | 0.6351 | 0.0184 | 0.0162 |
| 9 | 0.7253 | 0.6867 | 0.0095 | 0.0107 |

D2g / mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.1748 | 0.1226 | 0.1555 | 0.1580 |
| 1 | 0.1943 | 0.1663 | 0.1686 | 0.1814 |
| 2 | 0.1766 | 0.1182 | 0.1507 | 0.0628 |
| 3 | 0.1558 | 0.1398 | 0.1507 | 0.1595 |
| 4 | 0.1551 | 0.1319 | 0.1696 | 0.0549 |
| 5 | 0.1727 | 0.1342 | 0.1482 | 0.0554 |
| 6 | 0.1638 | 0.1107 | 0.1658 | 0.0666 |
| 7 | 0.1752 | 0.1400 | 0.1565 | 0.1619 |
| 8 | 0.1691 | 0.1215 | 0.1463 | 0.1587 |
| 9 | 0.1788 | 0.1145 | 0.1710 | 0.1568 |

D2g / single_index

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.7536 | 0.6812 | 0.0348 | 0.0335 |
| 1 | 0.7526 | 0.7102 | 0.0204 | 0.0347 |
| 2 | 0.7900 | 0.7110 | 0.0344 | 0.0471 |
| 3 | 0.7663 | 0.6978 | 0.0321 | 0.0295 |
| 4 | 0.7719 | 0.6919 | 0.0873 | 0.0570 |
| 5 | 0.7465 | 0.6905 | 0.0209 | 0.0188 |
| 6 | 0.7545 | 0.7092 | 0.0302 | 0.0186 |
| 7 | 0.7768 | 0.6924 | 0.0451 | 0.0383 |
| 8 | 0.7898 | 0.7089 | 0.0341 | 0.0154 |
| 9 | 0.8018 | 0.7325 | 0.0274 | 0.0111 |

D2l / mlp

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 0.2600 | 0.2638 | 0.2622 | 0.2795 |
| 1 | 0.2648 | 0.2743 | 0.2753 | 0.2626 |
| 2 | 0.2541 | 0.2638 | 0.2558 | 0.2218 |
| 3 | 0.2474 | 0.2564 | 0.2549 | 0.2044 |
| 4 | 0.2613 | 0.2507 | 0.2690 | 0.2114 |
| 5 | 0.2339 | 0.2481 | 0.2671 | 0.3087 |
| 6 | 0.2447 | 0.2352 | 0.2620 | 0.2662 |
| 7 | 0.2656 | 0.2712 | 0.2877 | 0.2846 |
| 8 | 0.2644 | 0.2679 | 0.2602 | 0.2830 |
| 9 | 0.2512 | 0.2631 | 0.2499 | 0.3309 |

D2l / single_index

| seed | vmf | power_spherical | tnbbeta | tnbbeta_init0 |
|---|---|---|---|---|
| 0 | 1.9001 | 1.8546 | 0.1037 | 0.5635 |
| 1 | 1.8851 | 1.8482 | 0.1675 | 0.2335 |
| 2 | 1.8968 | 1.8511 | 0.1164 | 0.1928 |
| 3 | 1.8901 | 1.8456 | 0.2850 | 0.1468 |
| 4 | 1.8935 | 1.8498 | 0.4168 | 0.2816 |
| 5 | 1.8932 | 1.8503 | 0.2772 | 0.1222 |
| 6 | 1.8949 | 1.8592 | 0.4219 | 0.2250 |
| 7 | 1.8983 | 1.8529 | 0.1218 | 0.2224 |
| 8 | 1.8754 | 1.8384 | 0.3603 | 0.1360 |
| 9 | 1.9022 | 1.8593 | 0.0820 | 0.3068 |

## Descriptive comparisons (no p-values)

| setting | decoder | comparison | n | mean gap (A) | mean gap (B) | mean paired diff (A - B) | seeds A better | IWAE paired diff (A - B) |
|---|---|---|---|---|---|---|---|---|
| D1 | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.5268 | 0.1846 | 0.3422 | 2/10 | 0.1294 |
| D1 | mlp | tnbbeta vs vmf | 10 | 0.5268 | 0.4138 | 0.1130 | 0/10 | 0.0362 |
| D1 | mlp | tnbbeta vs power_spherical | 10 | 0.5268 | 0.4079 | 0.1188 | 0/10 | 0.0599 |
| D1 | mlp | tnbbeta_init0 vs vmf | 10 | 0.1846 | 0.4138 | -0.2292 | 7/10 | -0.0932 |
| D1 | mlp | tnbbeta_init0 vs power_spherical | 10 | 0.1846 | 0.4079 | -0.2233 | 7/10 | -0.0695 |
| D1 | mlp | vmf vs power_spherical | 10 | 0.4138 | 0.4079 | 0.0059 | 2/10 | 0.0236 |
| D1 | single_index | tnbbeta vs tnbbeta_init0 | 10 | 0.0817 | 0.0471 | 0.0346 | 3/10 | 0.0101 |
| D1 | single_index | tnbbeta vs vmf | 10 | 0.0817 | 1.1040 | -1.0223 | 10/10 | -0.6924 |
| D1 | single_index | tnbbeta vs power_spherical | 10 | 0.0817 | 0.9884 | -0.9067 | 10/10 | -0.6408 |
| D1 | single_index | tnbbeta_init0 vs vmf | 10 | 0.0471 | 1.1040 | -1.0569 | 10/10 | -0.7025 |
| D1 | single_index | tnbbeta_init0 vs power_spherical | 10 | 0.0471 | 0.9884 | -0.9413 | 10/10 | -0.6509 |
| D1 | single_index | vmf vs power_spherical | 10 | 1.1040 | 0.9884 | 0.1156 | 0/10 | 0.0516 |
| D2g | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.5511 | 0.4071 | 0.1441 | 4/10 | 0.0367 |
| D2g | mlp | tnbbeta vs vmf | 10 | 0.5511 | 0.4550 | 0.0962 | 0/10 | -0.0133 |
| D2g | mlp | tnbbeta vs power_spherical | 10 | 0.5511 | 0.4460 | 0.1052 | 0/10 | 0.0283 |
| D2g | mlp | tnbbeta_init0 vs vmf | 10 | 0.4071 | 0.4550 | -0.0479 | 4/10 | -0.0500 |
| D2g | mlp | tnbbeta_init0 vs power_spherical | 10 | 0.4071 | 0.4460 | -0.0389 | 4/10 | -0.0084 |
| D2g | mlp | vmf vs power_spherical | 10 | 0.4550 | 0.4460 | 0.0090 | 2/10 | 0.0417 |
| D2g | single_index | tnbbeta vs tnbbeta_init0 | 10 | 0.1142 | 0.0862 | 0.0279 | 2/10 | 0.0063 |
| D2g | single_index | tnbbeta vs vmf | 10 | 0.1142 | 1.1154 | -1.0012 | 10/10 | -0.7337 |
| D2g | single_index | tnbbeta vs power_spherical | 10 | 0.1142 | 1.0027 | -0.8885 | 10/10 | -0.6659 |
| D2g | single_index | tnbbeta_init0 vs vmf | 10 | 0.0862 | 1.1154 | -1.0292 | 10/10 | -0.7400 |
| D2g | single_index | tnbbeta_init0 vs power_spherical | 10 | 0.0862 | 1.0027 | -0.9164 | 10/10 | -0.6722 |
| D2g | single_index | vmf vs power_spherical | 10 | 1.1154 | 1.0027 | 0.1127 | 0/10 | 0.0678 |
| D2l | mlp | tnbbeta vs tnbbeta_init0 | 10 | 0.6795 | 0.6324 | 0.0471 | 2/10 | -0.0009 |
| D2l | mlp | tnbbeta vs vmf | 10 | 0.6795 | 0.5786 | 0.1010 | 0/10 | 0.0097 |
| D2l | mlp | tnbbeta vs power_spherical | 10 | 0.6795 | 0.5922 | 0.0874 | 0/10 | 0.0050 |
| D2l | mlp | tnbbeta_init0 vs vmf | 10 | 0.6324 | 0.5786 | 0.0538 | 2/10 | 0.0105 |
| D2l | mlp | tnbbeta_init0 vs power_spherical | 10 | 0.6324 | 0.5922 | 0.0402 | 2/10 | 0.0058 |
| D2l | mlp | vmf vs power_spherical | 10 | 0.5786 | 0.5922 | -0.0136 | 7/10 | -0.0047 |
| D2l | single_index | tnbbeta vs tnbbeta_init0 | 10 | 0.4580 | 0.4517 | 0.0063 | 4/10 | -0.0078 |
| D2l | single_index | tnbbeta vs vmf | 10 | 0.4580 | 2.0502 | -1.5922 | 10/10 | -1.6577 |
| D2l | single_index | tnbbeta vs power_spherical | 10 | 0.4580 | 1.9471 | -1.4890 | 10/10 | -1.6157 |
| D2l | single_index | tnbbeta_init0 vs vmf | 10 | 0.4517 | 2.0502 | -1.5985 | 10/10 | -1.6499 |
| D2l | single_index | tnbbeta_init0 vs power_spherical | 10 | 0.4517 | 1.9471 | -1.4954 | 10/10 | -1.6079 |
| D2l | single_index | vmf vs power_spherical | 10 | 2.0502 | 1.9471 | 0.1032 | 0/10 | 0.0420 |
| D1 | single_index/mlp | vmf (SI vs MLP) | 10 | 1.1040 | 0.4138 | 0.6901 | 0/10 | 0.5682 |
| D1 | single_index/mlp | power_spherical (SI vs MLP) | 10 | 0.9884 | 0.4079 | 0.5804 | 0/10 | 0.5403 |
| D1 | single_index/mlp | tnbbeta (SI vs MLP) | 10 | 0.0817 | 0.5268 | -0.4451 | 10/10 | -0.1604 |
| D1 | single_index/mlp | tnbbeta_init0 (SI vs MLP) | 10 | 0.0471 | 0.1846 | -0.1376 | 6/10 | -0.0411 |
| D2g | single_index/mlp | vmf (SI vs MLP) | 10 | 1.1154 | 0.4550 | 0.6604 | 0/10 | 0.5988 |
| D2g | single_index/mlp | power_spherical (SI vs MLP) | 10 | 1.0027 | 0.4460 | 0.5567 | 0/10 | 0.5726 |
| D2g | single_index/mlp | tnbbeta (SI vs MLP) | 10 | 0.1142 | 0.5511 | -0.4370 | 10/10 | -0.1216 |
| D2g | single_index/mlp | tnbbeta_init0 (SI vs MLP) | 10 | 0.0862 | 0.4071 | -0.3208 | 10/10 | -0.0912 |
| D2l | single_index/mlp | vmf (SI vs MLP) | 10 | 2.0502 | 0.5786 | 1.4717 | 0/10 | 1.6382 |
| D2l | single_index/mlp | power_spherical (SI vs MLP) | 10 | 1.9471 | 0.5922 | 1.3549 | 0/10 | 1.5915 |
| D2l | single_index/mlp | tnbbeta (SI vs MLP) | 10 | 0.4580 | 0.6795 | -0.2215 | 9/10 | -0.0291 |
| D2l | single_index/mlp | tnbbeta_init0 (SI vs MLP) | 10 | 0.4517 | 0.6324 | -0.1807 | 9/10 | -0.0222 |

## Decoder, ring and class-structure diagnostics

| setting | decoder | arm | eta2 (mean over axes; seed median [min-max]) | ring R (median over x; seed median [min-max]) | 5-NN acc (seed mean +- sd) | mean abs(u* . centre) (seed mean) |
|---|---|---|---|---|---|---|
| D1 | mlp | vmf | 0.521 [0.517-0.525] | 0.935 [0.932-0.937] | 0.870 +- 0.004 | 0.570 |
| D1 | mlp | power_spherical | 0.524 [0.509-0.529] | 0.923 [0.897-0.929] | 0.875 +- 0.006 | 0.571 |
| D1 | mlp | tnbbeta | 0.567 [0.503-0.605] | 0.953 [0.888-0.967] | 0.874 +- 0.003 | 0.571 |
| D1 | mlp | tnbbeta_init0 | 0.998 [0.513-0.999] | 0.053 [0.052-0.955] | 0.838 +- 0.048 | 0.871 |
| D1 | single_index | vmf | 0.999 [0.998-0.999] | 0.066 [0.063-0.074] | 0.860 +- 0.011 | 0.880 |
| D1 | single_index | power_spherical | 0.999 [0.998-0.999] | 0.054 [0.052-0.058] | 0.855 +- 0.026 | 0.999 |
| D1 | single_index | tnbbeta | 0.999 [0.999-0.999] | 0.052 [0.050-0.054] | 0.820 +- 0.050 | 1.000 |
| D1 | single_index | tnbbeta_init0 | 0.999 [0.999-0.999] | 0.052 [0.051-0.054] | 0.825 +- 0.044 | 1.000 |
| D2g | mlp | vmf | 0.538 [0.527-0.544] | 0.925 [0.912-0.929] | - | 0.591 |
| D2g | mlp | power_spherical | 0.523 [0.511-0.547] | 0.894 [0.882-0.912] | - | 0.580 |
| D2g | mlp | tnbbeta | 0.525 [0.498-0.550] | 0.884 [0.877-0.893] | - | 0.575 |
| D2g | mlp | tnbbeta_init0 | 0.533 [0.512-0.893] | 0.872 [0.064-0.885] | - | 0.706 |
| D2g | single_index | vmf | 0.999 [0.999-0.999] | 0.077 [0.071-0.084] | - | 0.898 |
| D2g | single_index | power_spherical | 0.999 [0.999-0.999] | 0.056 [0.054-0.061] | - | 0.997 |
| D2g | single_index | tnbbeta | 0.999 [0.999-0.999] | 0.052 [0.051-0.054] | - | 1.000 |
| D2g | single_index | tnbbeta_init0 | 0.999 [0.999-0.999] | 0.053 [0.052-0.054] | - | 1.000 |
| D2l | mlp | vmf | 0.554 [0.523-0.569] | 0.984 [0.984-0.986] | 1.000 +- 0.000 | 0.523 |
| D2l | mlp | power_spherical | 0.549 [0.536-0.565] | 0.984 [0.983-0.985] | 1.000 +- 0.000 | 0.527 |
| D2l | mlp | tnbbeta | 0.554 [0.549-0.563] | 0.982 [0.982-0.984] | 1.000 +- 0.000 | 0.522 |
| D2l | mlp | tnbbeta_init0 | 0.730 [0.543-0.991] | 0.544 [0.052-0.984] | 0.971 +- 0.031 | 0.730 |
| D2l | single_index | vmf | 0.999 [0.999-0.999] | 0.063 [0.060-0.064] | 0.871 +- 0.044 | 0.930 |
| D2l | single_index | power_spherical | 0.999 [0.999-0.999] | 0.053 [0.052-0.055] | 0.888 +- 0.038 | 0.996 |
| D2l | single_index | tnbbeta | 0.989 [0.986-0.991] | 0.053 [0.051-0.057] | 0.935 +- 0.049 | 0.987 |
| D2l | single_index | tnbbeta_init0 | 0.988 [0.983-0.991] | 0.053 [0.051-0.058] | 0.871 +- 0.057 | 0.992 |

## Learned sigma, divergences and fitted parameters

| setting | decoder | arm | diverged | learned sigma (mean +- sd; true 0.2) | p | q | epsilon | m = eps - 1 | kappa |
|---|---|---|---|---|---|---|---|---|---|
| D1 | mlp | vmf | 0/10 | 0.1823 +- 0.0008 | - | - | - | - | 13.8 |
| D1 | mlp | power_spherical | 0/10 | 0.1925 +- 0.0083 | - | - | - | - | 22.7 |
| D1 | mlp | tnbbeta | 0/10 | 0.1727 +- 0.0267 | 0.982 | 1e-06 | 2.99 | 1.99 | - |
| D1 | mlp | tnbbeta_init0 | 0/10 | 0.1958 +- 0.0143 | 0.498 | 0.522 | 13.1 | 12.1 | - |
| D1 | single_index | vmf | 0/10 | 0.3618 +- 0.0042 | - | - | - | - | 3.08 |
| D1 | single_index | power_spherical | 0/10 | 0.3547 +- 0.0013 | - | - | - | - | 3.52 |
| D1 | single_index | tnbbeta | 0/10 | 0.2018 +- 0.0026 | 0.747 | 0.574 | 10.6 | 9.62 | - |
| D1 | single_index | tnbbeta_init0 | 0/10 | 0.2001 +- 0.0015 | 0.495 | 0.546 | 13 | 12 | - |
| D2g | mlp | vmf | 0/10 | 0.1902 +- 0.0034 | - | - | - | - | 12.5 |
| D2g | mlp | power_spherical | 0/10 | 0.2106 +- 0.0059 | - | - | - | - | 16.2 |
| D2g | mlp | tnbbeta | 0/10 | 0.2160 +- 0.0040 | 0.953 | 1e-06 | 3.67 | 2.67 | - |
| D2g | mlp | tnbbeta_init0 | 0/10 | 0.2112 +- 0.0084 | 0.326 | 1e-06 | 3.79 | 2.79 | - |
| D2g | single_index | vmf | 0/10 | 0.3737 +- 0.0018 | - | - | - | - | 2.95 |
| D2g | single_index | power_spherical | 0/10 | 0.3606 +- 0.0028 | - | - | - | - | 3.81 |
| D2g | single_index | tnbbeta | 0/10 | 0.2055 +- 0.0036 | 0.706 | 0.664 | 8.38 | 7.38 | - |
| D2g | single_index | tnbbeta_init0 | 0/10 | 0.2031 +- 0.0030 | 0.503 | 0.615 | 9.93 | 8.93 | - |
| D2l | mlp | vmf | 0/10 | 0.1996 +- 0.0021 | - | - | - | - | 50.3 |
| D2l | mlp | power_spherical | 0/10 | 0.2009 +- 0.0013 | - | - | - | - | 99.7 |
| D2l | mlp | tnbbeta | 0/10 | 0.2023 +- 0.0017 | 0.991 | 1e-06 | 4.19 | 3.19 | - |
| D2l | mlp | tnbbeta_init0 | 0/10 | 0.2168 +- 0.0141 | 0.498 | 0.0818 | 5.9 | 4.9 | - |
| D2l | single_index | vmf | 0/10 | 0.9374 +- 0.0029 | - | - | - | - | 1.8 |
| D2l | single_index | power_spherical | 0/10 | 0.9150 +- 0.0031 | - | - | - | - | 1.77 |
| D2l | single_index | tnbbeta | 0/10 | 0.2208 +- 0.0095 | 0.767 | 0.941 | 25.3 | 24.3 | - |
| D2l | single_index | tnbbeta_init0 | 0/10 | 0.2244 +- 0.0077 | 0.503 | 0.966 | 18.1 | 17.1 | - |
