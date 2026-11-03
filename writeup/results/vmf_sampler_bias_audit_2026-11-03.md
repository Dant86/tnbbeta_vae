# vMF sampler bias audit, 2026-11-03

- **Date:** 2026-11-03. **Pre-registration:** `vmf_sampler_bias_audit_preregistration_2026-11-03.md` (written after the grid jobs were submitted but before any grid, audit or rerun result was read; no deviations).
- **Base commit:** `dd8ede0` (branch `synthetic-graph-families`) plus the UNCOMMITTED working tree: modified `src/tnbbeta_vae/distributions/von_mises_fisher.py` (additive `exact_envelope` flag) and `tests/distributions/test_von_mises_fisher.py`; new `apps/synthetic/vmf_sampler_bias.py`, `apps/synthetic/vmf_exact_rerun.py`, `scripts/slurm/vmf_sampler_bias.sbatch`, `scripts/slurm/vmf_exact_rerun.sbatch`, `scripts/slurm/vmf_bias_aux.sbatch`.
- **Commands:**

  ```bash
  sbatch --array=0-3 scripts/slurm/vmf_sampler_bias.sbatch                                   # job 2304330 (grid: float32/float64 x library/exact)
  uv run python -m apps.synthetic.vmf_sampler_bias summarize                                   # tables below (login node, reads 4 JSONs)
  uv run python -m apps.synthetic.vmf_exact_rerun manifest                                     # 2560 runs
  TOTAL_TASKS=12 sbatch --array=0-11 scripts/slurm/vmf_exact_rerun.sbatch                     # job 2304340 (task 3 hit a node, m001, that could not see the scratch dir: rerun as 2304362 with --exclude=m001)
  AUX=analyze sbatch scripts/slurm/vmf_bias_aux.sbatch                                         # jobs 2304405 (superseded: mislabelled one column), 2304421 (tables below)
  AUX=pytest sbatch scripts/slurm/vmf_bias_aux.sbatch                                          # 2304353 (found 5 test bugs of mine, fixed), 2304433 (final)
  ```

  Raw outputs: `$TNBBETA_CHECKPOINT_DIR/vmf_sampler_bias/*.json`, `$TNBBETA_CHECKPOINT_DIR/graph_search/vmf_exact/runs/*.json`.

## Headline

1. **The library `VonMisesFisher` rejection sampler is biased, and the fix is exact.** Over 15 ambient dimensions x 21 concentrations (315 cells, N = 400,000 draws each, float32 and float64) the default sampler is flagged in 123 (float64) / 125 (float32) cells; with `exact_envelope=True` **0 of 315** in both dtypes (max |mean bias| 0.0009, max |second-moment bias| 0.0007, max KS 0.0030 vs a 0.0031 critical value).
2. **Biased region:** `kappa > 10` (the Taylor switch starts at 10, is complete at 11; every cell with `kappa <= 10` is clean), `m >= 5` (m = 3 uses a closed form and is exact, m = 4 clean; m = 5, 6 flagged only at the 4e-4 level), and `kappa` below roughly `2 m`. The draws are always **too diffuse** (mean cosine too small). Mean-cosine bias, float64: m = 9: -0.005 at kappa 11; m = 17: -0.053 at 11, -0.023 at 16, -0.011 at 20, -0.0018 at 30; m = 33: -0.151 at 11, -0.083 at 16, -0.013 at 40, -0.0007 at 80; m = 65: -0.347 at 11, -0.233 at 16, -0.055 at 40, -0.0077 at 100, -0.0019 at 150. At kappa >= 300 it is below 0.0001 at every m. At m = 65, kappa = 11 the true mean cosine is +0.165 but the sampler's is about -0.18 (the draws point slightly away from `loc`).
3. **Cause (verified numerically):** for `kappa > 10` the sampler replaces Wood's envelope parameter `b` by the Taylor value `(m-1)/(4 kappa)` while the acceptance test's constants are those of the exact-`b` envelope. With the exact `b` the maximum of the acceptance ratio `exp((m-1) log t - t + d)` over the proposal is 1.0000 (a valid envelope). With the Taylor `b` it is 1.98 at (m, kappa) = (17, 17), 4.00 at (17, 11), 3.97 at (65, 100), 1.19 at (65, 300): the "probability" of acceptance exceeds 1 over part of the proposal, so Wood's test no longer corrects the proposal to the vMF law. The Taylor approximation is only good for `kappa >> m`, but the switch is at a fixed `kappa = 10-11` regardless of `m`; this is why the bias grows with `m` and is absent at small `m`. It also shows the comment in the code ("to avoid numerical errors for large scale") is about the cancellation in `(c - 2 kappa)/(m-1)`, which `b = (m-1)/(c + 2 kappa)` avoids exactly.
4. **Effect on the graph metric-first comparison: none detectable.** On 40 configs x 16 draws, TNBBeta's confirmed advantage over vMF is the same with the library and the exact sampler: ECE only (4 of 4 comparator x sampler rows Holm-confirmed, dz 0.46-0.57); no other of the 32 tests passes Holm; the truth-KL / log-loss advantages stay null (dz 0.19-0.26 truth KL, -0.03-0.08 log loss) with both samplers. The sampler fix on vMF alone moves metrics by -0.005..+0.003 (probabilities) with only one Holm-significant change (`cal_slope_dev`, pinned vMF, exact better on 28 of 40 configs, dz +0.60).
5. **Recommendation:** add the flag (done, default off) and **make `exact_envelope=True` the default only after the owner decides**; I did not change the default so every old checkpoint/result reproduces. If flipped, results with vMF at `m >= 8, kappa > 10` (audit table) should be treated as superseded, or rerun.

## (1) Sampler grid

Design as pre-registered. Flag rule: |mean - exact| > 4 SE, or |second moment - exact| > 4 SE, or KS sqrt(N) > 1.95. Exact moments `E[w] = I_(m/2)(k)/I_(m/2-1)(k)`, `E[w^2] = 1 - (m-1)E[w]/k`; exact marginal CDF by quadrature. `norm_err` (unit norm of the samples) <= 6.3e-7 in all cells: the bias is in the distribution of `w = <z, loc>`, not in the rotation.

Library default, float64 (float32 is almost identical: 125 flagged cells, max |mean bias| 0.3473, max KS 0.8462):

Flagged cells: 123 of 315. Max |mean bias| 0.3472, max |second-moment bias| 0.0662, max KS 0.8464 (1e-3 critical 0.0031), max |norm - 1| 2.0e-07.

Mean-cosine bias x 1000 (sample - exact); `*` = flagged (rows m, columns kappa):

| m \ kappa | 1 | 2 | 4 | 8 | 10 | 11 | 12 | 14 | 16 | 20 | 25 | 30 | 40 | 50 | 60 | 80 | 100 | 150 | 200 | 300 | 500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | +0.8 | -0.1 | +0.2 | +0.2 | -0.1 | +0.0 | +0.1 | +0.1 | -0.0 | +0.1 | +0.1 | +0.0 | -0.0 | -0.0 | +0.0 | +0.0 | -0.0 | -0.0 | -0.0 | -0.0 | +0.0 |
| 4 | +0.7 | -0.4 | -0.1 | -0.1 | +0.0 | -0.2 | -0.1 | -0.2 | +0.0 | +0.0 | -0.0 | -0.0 | -0.0 | +0.1 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.0 |
| 5 | +0.1 | -0.0 | -0.1 | +0.1 | +0.1 | -0.4* | -0.2 | +0.0 | -0.2 | +0.0 | -0.1 | -0.1 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | -0.0 | +0.0 | -0.0 | +0.0 |
| 6 | -0.4 | +0.7 | +0.6 | +0.2 | -0.0 | -0.4* | -0.0* | -0.3* | +0.1* | -0.0 | -0.0 | +0.1 | -0.1 | +0.1 | -0.0 | +0.0 | -0.1 | +0.0 | -0.0 | +0.0 | -0.0 |
| 8 | +0.1 | -0.0 | -0.1 | -0.1 | +0.2 | -2.1* | -2.1* | -0.9* | -0.6* | -0.3* | -0.2* | +0.0 | -0.0 | +0.0 | +0.0 | +0.1 | +0.0 | +0.0 | -0.0 | -0.0 | +0.0 |
| 9 | -0.4 | +0.5 | +0.1 | -0.4 | -0.4 | -5.3* | -3.4* | -1.7* | -0.9* | -0.3* | -0.3* | +0.1 | -0.1 | +0.0 | -0.0 | +0.0 | +0.1 | -0.0 | +0.0 | +0.0 | -0.0 |
| 10 | +0.4 | -1.3 | +0.1 | -0.3 | +0.4 | -9.1* | -6.1* | -3.7* | -1.7* | -0.5* | -0.3* | -0.2* | -0.0 | +0.1 | -0.1 | -0.1 | -0.0 | +0.0 | -0.0 | +0.0 | -0.0 |
| 12 | +0.1 | -0.1 | +0.5 | -0.4 | -0.3 | -21.3* | -15.0* | -8.2* | -5.0* | -1.4* | -0.7* | -0.5* | -0.1* | -0.1 | -0.0 | +0.0 | -0.1 | -0.0 | -0.0 | +0.0 | -0.0 |
| 16 | -0.1 | +0.1 | +0.8 | +0.5 | +0.5 | -46.4* | -39.5* | -27.6* | -18.5* | -8.0* | -2.9* | -1.4* | -0.5* | -0.2* | +0.1* | -0.0 | +0.0 | +0.0 | -0.0 | -0.0 | -0.0 |
| 17 | -0.1 | +0.5 | -0.2 | +0.1 | -0.0 | -51.6* | -44.7* | -33.0* | -22.8* | -10.7* | -4.4* | -1.8* | -0.5* | -0.3* | -0.1* | +0.0* | -0.1* | +0.0 | -0.0 | +0.0 | +0.0 |
| 20 | +0.4 | -0.1 | -0.1 | +0.1 | +0.1 | -67.6* | -58.5* | -44.9* | -34.9* | -20.0* | -9.5* | -4.4* | -1.2* | -0.4* | -0.3* | -0.1* | -0.1* | -0.0 | +0.0 | +0.0 | -0.0 |
| 25 | -0.0 | -0.5 | +0.2 | -0.2 | -0.3 | -97.4* | -84.1* | -65.1* | -52.9* | -35.5* | -21.5* | -12.3* | -4.1* | -1.4* | -0.6* | -0.2* | -0.1* | -0.0 | +0.0 | +0.0 | +0.0 |
| 33 | +0.1 | +0.2 | +0.4 | -0.0 | -0.4 | -150.9* | -132.8* | -104.3* | -83.3* | -57.0* | -38.4* | -26.8* | -13.0* | -5.9* | -3.0* | -0.7* | -0.2* | +0.0* | +0.0* | -0.0 | +0.0 |
| 50 | -0.0 | -0.0 | +0.0 | +0.2 | +0.1 | -263.1* | -238.2* | -196.4* | -163.6* | -116.1* | -79.6* | -57.2* | -33.6* | -21.8* | -14.6* | -6.2* | -2.6* | -0.4* | -0.1* | +0.0* | -0.0 |
| 65 | +0.1 | -0.0 | +0.3 | -0.2 | -0.1 | -347.2* | -319.6* | -271.8* | -233.0* | -174.2* | -124.5* | -92.2* | -54.6* | -35.3* | -24.6* | -13.9* | -7.7* | -1.9* | -0.3* | -0.0* | -0.0* |


Library default, float32: Flagged cells: 125 of 315. Max |mean bias| 0.3473, max |second-moment bias| 0.0662, max KS 0.8462 (1e-3 critical 0.0031), max |norm - 1| 6.3e-07.

`exact_envelope=True`, float64: Flagged cells: 0 of 315. Max |mean bias| 0.0009, max |second-moment bias| 0.0006, max KS 0.0028 (1e-3 critical 0.0031), max |norm - 1| 2.2e-07.

`exact_envelope=True`, float32: Flagged cells: 0 of 315. Max |mean bias| 0.0009, max |second-moment bias| 0.0007, max KS 0.0030 (1e-3 critical 0.0031), max |norm - 1| 6.1e-07.

The second moment shows the same pattern (max |bias| 0.066 library, 0.0007 fixed); KS distance of the library sampler reaches 0.85 at (m, kappa) = (65, 11) (the empirical law is nowhere near the vMF law).

## (2) Audit: which earlier results used the library vMF sampler in the biased region

Method: grep of `writeup/results`, `apps/`, `scripts/slurm/` for vMF use, then read each study's ambient dimension `m` and concentration. **Bias** is the mean-cosine bias of the sampler (float64 grid above; interpolated between grid points, approximate). "Region" = flagged by the grid rule. Concentrations that a write-up does not report are marked unknown rather than guessed; this table was compiled by reading write-ups and sbatch scripts, not by re-extracting checkpoints, so the "learned kappa" rows are the weakest.

| Study (write-up / script) | m (ambient) | kappa | In biased region? | Approx. bias (mean cos) | Comment |
|---|---|---|---|---|---|
| `graph_search_*` discovery, confirmation, pilot, fresh (`vmf_init`, `vmf_pin`; apps `graph_search.py`) | 16 | init / pin 16; learned values ~16-17 (this audit's rerun: median exp(mean log kappa) 16.7) | **Yes** | -0.019 | Rerun here (section 4): conclusion unchanged. The `vmf_pin` and `vmf_init` arms at all four stages are affected. |
| `graph_search_kappa_grid_*` (`vmf_k4 ... vmf_k128`) | 16 | 4, 8, 16, 32, 64, 128 | kappa = 16 only (-0.019); 32: -0.0015 (negligible); 4, 8: no; 64, 128: no | <= 0.02 | One of six grid points affected, and it is the pinned point `DIM`. |
| `graph_search_*` `vmf_default` (kappa ~ 1.7 start) | 16 | starts 1.69, learns slowly | Only if kappa escapes above 10 | unknown | Per `graph_head_bias_sweep`, free vMF reaches kappa ~ 2-91 depending on graph in 495 epochs; escape cases sit at kappa far above `2m`, where bias is < 0.002. |
| `graph_family_screening_2026-10-30`, `latent_kernel_m_sweep`, `graph_head_bias_sweep_2026-11-01`, `tnb_init_symmetry_and_head_offset_2026-11-01` (vMF `kappa = d` arms) | 16 (init-symmetry also 3, 8, 32) | pin / start at `d` | Yes for `d = 16` (-0.019) and `d = 32` (~-0.022); `d = 3` exact; `d = 8` (kappa 8 <= 10) clean | up to -0.02 | Free-kappa `vmf_def` arms start at 1.69: outside the region unless they learn kappa in 11-30 at m = 16. The init-symmetry table reports E[w] by `d` at init (computed analytically from Bessel ratios, unaffected). |
| `fixed_m_sweeps_baselines_and_parameter_probes_2026-10-30`, `vmf_ps_fixed_kappa_and_training_dynamics_2026-10-30` fixed-kappa sweep (kappa = 0.5, 1, 2, 5 x d) | Cora 65, Citeseer 17, Pubmed 17, MAG-CS 33 | 0.5d, d, 2d, 5d | **Yes at kappa = d for m = 17, 33, 65 and 0.5d for m = 65** | m = 65: 32.5 -> ~-0.085, 65 -> ~-0.022, 130 -> ~-0.0035; m = 33: 33 -> ~-0.028, 16.5 -> ~-0.08; m = 17: 17 -> -0.023, 8.5 clean | The reported "pinned `kappa = d` matches TNBBeta" finding used the biased sampler (too diffuse at `kappa = d`). The matched-dimension baselines (`fixed_m_baselines_graphs`) use the same dimensions. |
| Link-prediction free-kappa vMF (Figure 5 grid, `link_prediction.sbatch`, manifold d in 8, 16, 32, 64 so m = 9, 17, 33, 65; `kappa_dynamics`) | 9-65 | learned from 1.69: ~2.5 on Cora/MAG-CS after 200 epochs, escapes on Pubmed (400 epochs; value not extracted) | Mostly **no** (kappa < 10); possible only for runs/epochs where kappa passes 10 | unknown | Selected Figure-5 vMF configs mostly sit at m = 9 (`own_vmf`), where nothing is biased even at kappa > 10 (m = 9: -0.005 at 11). |
| MNIST Table 1 / Table 2 k-NN / confidence probe / concentration check (`vmfs`, `vmfks`; 2026-09-21 .. 09-26) | 3, 6, 11, 21, 41 (d = 2, 5, 10, 20, 40) | learned; reported through `r_bar` as kappa_eq: 9.2, 9.6, 10.4, 11.7, 62 (healthy seeds ~74; the collapsed d=40 seed 15.8). `vmfks` starts at kappa = m | d = 2, 5: **no** (m = 3 exact; m = 6 bias <= 0.0004). d = 10: borderline. d = 20: **yes**. d = 40: healthy seeds small bias, the collapsed seed **yes** | m = 11, kappa ~ 11: -0.009; m = 21, kappa 12-20: about -0.07..-0.02; m = 41: kappa 74 about -0.003, kappa 16: about -0.14 | `kappa_eq` is computed from `r_bar` of the (biased) samples, so the nominal kappa is larger than `kappa_eq`, which moves the real operating point out of the worst corner but is not quantified. The analytic KL is exact and the IWAE log-weights use the nominal `log q`, so for d >= 10 the vMF IWAE LL is not an unbiased-bound estimator (proposal != nominal q). The d=40 partial-collapse seed is in the worst corner; whether the sampler bias contributes to that "known `vmfs` instability" is untested (it would be a rerun with `exact_envelope=True`). The S-VAE reference code has the same sampler, so this is also a faithful reproduction of the reference behaviour. |
| Semi-supervised M1+M2 sweep (`semi_supervised_sweep.sbatch`, vMF `kappa_init = ambient`) | z dims 5, 10, 50 manifold -> m = 6, 11, 51 | starts at m; learned values not extracted | m = 6: no; m = 11 (kappa 11): borderline (-0.009); m = 51 (kappa 51): **yes** | m = 51, kappa 51: ~-0.022 | Whether kappa moves off its start was not extracted. |
| `ring_recovery` / `ring_posterior_*` (Level 1-3, d = 3) | 3 | cap fits | **No** (closed form) | 0 | `_fit_cap` for d = 8, 16 would use the library sampler for its Monte Carlo KL; those follow-ups are listed as not run. |
| `ring_class_designs_*`, `hrtf_itd_*`, `axial_bingham_*`, `axial_mixture_recovery`, `circle_recovery` | 3 (ring classes d=3 default), 2-5 | HRTF kappa ~ 1 | **No** | <= 0.0004 | All m <= 5 or kappa <= 10. Ring-class designs list d in {3, 8, 16}; the d = 8, 16 arms were not run in the write-ups I read. |
| `ring_dimension_scaling_*`, `linear_decoder_ring_*` | 3, 5, 9, 17 | all | **No** | 0 | Already use a local exact-`b` subclass (these studies found the bias). |
| `tnbbeta_vmf_convergence_check`, `tnbbeta_vmf_geometric_convergence` | S^2 | up to 1000 | **No** (m = 3, and r_bar from Bessel ratios) | 0 | Analytic / closed-form. |
| `dblp_bridge_diagnostic`, `sbm_*`, `features_onoff_dimension_sweep`, `dimension_range_*` (vMF arms of link prediction) | 9-65 (d + 1) | learned (or pinned at d in the later sweeps) | as for link prediction | unknown | The 2026-10-09 write-ups predate the fixed-kappa work; vMF there is free-kappa (mostly kappa < 10). |

Reading of the table: the results most exposed are (a) every graph-search stage's `vmf_init` / `vmf_pin` arms (m = 16, kappa = 16) and the graph-family / latent-kernel screens, (b) the pinned-`kappa = d` fixed-kappa link-prediction sweep at m >= 17, (c) MNIST d = 20 (and the d = 40 collapsed seed) and the M1+M2 m = 51 runs. Results with free kappa that stays below 10, all m <= 5, and all analytic uses are not affected. Direction of the bias: draws too diffuse, so vMF as sampled is *less* concentrated than the nominal law; against TNBBeta this would make vMF look worse than it is, which is the direction of the "TNBBeta beats vMF" claims, but section 4 shows no measurable effect in the one setting I could rerun.

## (3) Library change

`VonMisesFisher(loc, scale, k=1, exact_envelope=False)`: with `exact_envelope=True`, `_sample_w_rej` uses `b = (m-1)/(c + 2 kappa)` for every `kappa` (algebraically `(c - 2 kappa)/(m - 1)`, without its cancellation), with `a`, `d` as before. Default `False` is byte-for-byte the previous code path (the Taylor blend moved into `_blended_envelope_b`). Verified: grid above (0/315 flagged, both dtypes); tests `test_exact_envelope_matches_bessel_mean` (m, kappa) in {(5,30), (17,17), (33,17), (65,40)} within 0.006 (5 SE at N = 200,000), `test_default_sampler_keeps_taylor_switch_bias_documented` (pins the default's known bias at (17, 17) so a silent default change would fail), `test_exact_envelope_equals_default_below_the_taylor_switch` (same seed, kappa = 6: identical draws to 1e-4, unit norm).  Sampling speed with the flag was not benchmarked. The reparameterisation gradient path is unchanged (the accepted `w` is still drawn without gradient; gradient flows through `b, a, d`, which now carry the exact formula, a change only when the flag is on).

Decision for the owner: (i) default `False` retained, so nothing old changes; (ii) flipping the default to `True` is a one-line change but silently changes every vMF training run at `m >= 5, kappa > 10`; I recommend flipping it with a changelog note and keeping `False` reachable for reproducing archived results, but this is a decision about validated math and was not made here; (iii) `models/heads.py` does not expose the flag yet (it constructs `VonMisesFisher(...)` directly); the rerun here patches `heads.VonMisesFisher` in-process. A `GraphVAEConfig`/head option would be the next step if the flag becomes the default for new work.


## (4) Graph metric-first rerun

Subset (as pre-registered): configs `-00, -05, -10, -15` of each of the 10 families of the `fresh` stage (40 configs), all 16 draws, `vmf_init` and `vmf_pin` (m = 16, kappa 16), each run twice from the same manifest entry (same seed, learning rate, graph and edge split): `lib` (library sampler) and `exact` (`exact_envelope=True`). TNBBeta and PowerSpherical are the original `fresh` runs; 2560 reruns finished, 0 failures, CPU. Unit = config (mean over 16 draws).

Reproducibility control: the `lib` rerun matches the original fresh run exactly (truth KL) in 48% of runs and to a median 4e-9, but a few differ by up to 0.46 in truth KL (some originals ran on GPU; training is chaotic once any floating-point difference appears). So paired differences between arms carry rerun noise of this kind, which is why a `lib` control is in the design. Because TNBBeta runs were not redone, comparisons across `lib` and `exact` are not subject to TNBBeta noise.

Power note: 40 configs give 80% power at alpha = 0.05 (one-sided, unadjusted) for dz about 0.45; after Holm over 32 tests the detectable dz is about 0.6, so effects smaller than that are not excluded.

**Answer: TNBBeta's wins over vMF do not change.** Holm-confirmed TNBBeta-better rows are identical in set with either sampler (ECE for both comparators, 4 of 4); the two metrics where the earlier searches saw TNBBeta ahead and behind (truth KL, log loss) stay null with both; centre AUC, sampled AUC and rank correlation stay in vMF's favour (dz -0.18 to -0.46, p ~ 1 one-sided, with both samplers). With the exact sampler TNBBeta's median truth-KL advantage is slightly larger (vmf_init +0.0025 -> +0.0072; vmf_pin +0.0044 -> +0.0058; one-sided p 0.16 -> 0.11 and 0.12 -> 0.079), but none of this is significant; the exact sampler gives vMF slightly *worse* truth KL (median -0.003, not significant) and a better calibration slope (pinned arm, Holm p 0.016). Per family the truth-KL pattern is the same under both samplers (scale-free, assortative, degree-corrected TNBBeta ahead; latent-kernel, ring, hierarchical behind).

Tables (verbatim output of `apps.synthetic.vmf_exact_rerun analyze`, job 2304421; in the second table "better/worse" counts configs where the exact sampler improved / worsened vMF, "change" is positive when exact is better):


Configs: 40; runs finished: 2560.

Library-sampler rerun vs the original fresh run (truth_kl, same run): median |diff| 3.73e-09, max 4.57e-01, exactly equal in 48% of 1280 runs.

### TNBBeta advantage over vMF (positive = TNBBeta better), config-level means over 16 draws

One-sided Wilcoxon signed-rank over configs (H1: TNBBeta better), Holm over all rows. dz = mean/sd of the 40 per-config advantages; better/worse = configs with advantage > / < 0.

| comparator | sampler | metric | median adv | mean adv | dz | better/worse | p | Holm p | verdict |
|---|---|---|---|---|---|---|---|---|---|
| vmf_init | lib | truth_kl | +0.0025 | +0.0152 | +0.19 | 21/19 | 0.1602 | 1.0000 | null |
| vmf_init | lib | sampled_log_loss | +0.0011 | -0.0031 | -0.03 | 22/18 | 0.3425 | 1.0000 | null |
| vmf_init | lib | truth_kl_w | -0.0018 | -0.0061 | -0.07 | 19/21 | 0.5764 | 1.0000 | null |
| vmf_init | lib | ece | +0.0219 | +0.0272 | +0.46 | 28/12 | 0.0012 | 0.0347 | TNBBeta better |
| vmf_init | lib | cal_slope_dev | -0.0629 | -0.1234 | -0.21 | 15/25 | 0.9128 | 1.0000 | null |
| vmf_init | lib | centre_auc | -0.0065 | -0.0177 | -0.45 | 7/33 | 1.0000 | 1.0000 | null |
| vmf_init | lib | sampled_auc | -0.0058 | -0.0111 | -0.18 | 13/27 | 0.9751 | 1.0000 | null |
| vmf_init | lib | rank_corr | -0.0121 | -0.0273 | -0.46 | 10/30 | 0.9996 | 1.0000 | null |
| vmf_init | exact | truth_kl | +0.0072 | +0.0203 | +0.23 | 22/18 | 0.1107 | 1.0000 | null |
| vmf_init | exact | sampled_log_loss | +0.0049 | +0.0014 | +0.02 | 21/19 | 0.2593 | 1.0000 | null |
| vmf_init | exact | truth_kl_w | -0.0041 | -0.0014 | -0.01 | 19/21 | 0.4393 | 1.0000 | null |
| vmf_init | exact | ece | +0.0251 | +0.0306 | +0.49 | 29/11 | 0.0007 | 0.0205 | TNBBeta better |
| vmf_init | exact | cal_slope_dev | -0.0853 | -0.1543 | -0.27 | 15/25 | 0.9486 | 1.0000 | null |
| vmf_init | exact | centre_auc | -0.0062 | -0.0178 | -0.43 | 9/31 | 1.0000 | 1.0000 | null |
| vmf_init | exact | sampled_auc | -0.0065 | -0.0135 | -0.23 | 11/29 | 0.9861 | 1.0000 | null |
| vmf_init | exact | rank_corr | -0.0098 | -0.0269 | -0.46 | 10/30 | 0.9992 | 1.0000 | null |
| vmf_pin | lib | truth_kl | +0.0044 | +0.0210 | +0.23 | 21/19 | 0.1212 | 1.0000 | null |
| vmf_pin | lib | sampled_log_loss | +0.0102 | +0.0048 | +0.05 | 22/18 | 0.1383 | 1.0000 | null |
| vmf_pin | lib | truth_kl_w | +0.0019 | -0.0014 | -0.01 | 20/20 | 0.2680 | 1.0000 | null |
| vmf_pin | lib | ece | +0.0309 | +0.0333 | +0.52 | 30/10 | 0.0004 | 0.0111 | TNBBeta better |
| vmf_pin | lib | cal_slope_dev | -0.1207 | +0.0216 | +0.03 | 18/22 | 0.6575 | 1.0000 | null |
| vmf_pin | lib | centre_auc | -0.0062 | -0.0178 | -0.43 | 13/27 | 1.0000 | 1.0000 | null |
| vmf_pin | lib | sampled_auc | -0.0054 | -0.0129 | -0.23 | 12/28 | 0.9796 | 1.0000 | null |
| vmf_pin | lib | rank_corr | -0.0172 | -0.0274 | -0.46 | 13/27 | 0.9992 | 1.0000 | null |
| vmf_pin | exact | truth_kl | +0.0058 | +0.0240 | +0.26 | 23/17 | 0.0789 | 1.0000 | null |
| vmf_pin | exact | sampled_log_loss | +0.0088 | +0.0080 | +0.08 | 23/17 | 0.0961 | 1.0000 | null |
| vmf_pin | exact | truth_kl_w | +0.0033 | +0.0020 | +0.02 | 22/18 | 0.2063 | 1.0000 | null |
| vmf_pin | exact | ece | +0.0339 | +0.0365 | +0.57 | 28/12 | 0.0003 | 0.0091 | TNBBeta better |
| vmf_pin | exact | cal_slope_dev | -0.1057 | -0.0272 | -0.03 | 16/24 | 0.7821 | 1.0000 | null |
| vmf_pin | exact | centre_auc | -0.0040 | -0.0163 | -0.39 | 15/25 | 0.9993 | 1.0000 | null |
| vmf_pin | exact | sampled_auc | -0.0048 | -0.0128 | -0.23 | 15/25 | 0.9250 | 1.0000 | null |
| vmf_pin | exact | rank_corr | -0.0094 | -0.0269 | -0.44 | 14/26 | 0.9979 | 1.0000 | null |

### Effect of the sampler fix on vMF alone (positive = vMF better with the exact sampler)

Two-sided Wilcoxon over configs, Holm over these rows. Per-config means over 16 draws, vMF(lib) vs vMF(exact), same seeds.

| arm | metric | median change | mean change | dz | exact better/worse | p | Holm p |
|---|---|---|---|---|---|---|---|
| vmf_init | truth_kl | -0.0029 | -0.0050 | -0.27 | 15/25 | 0.0867 | 0.8665 |
| vmf_init | sampled_log_loss | -0.0017 | -0.0045 | -0.34 | 14/26 | 0.0451 | 0.5412 |
| vmf_init | truth_kl_w | -0.0016 | -0.0047 | -0.41 | 14/26 | 0.0098 | 0.1407 |
| vmf_init | ece | -0.0020 | -0.0034 | -0.32 | 16/24 | 0.0770 | 0.8470 |
| vmf_init | cal_slope_dev | +0.0179 | +0.0308 | +0.51 | 24/16 | 0.0094 | 0.1407 |
| vmf_init | centre_auc | -0.0002 | +0.0000 | +0.00 | 19/21 | 0.3541 | 1.0000 |
| vmf_init | sampled_auc | +0.0006 | +0.0024 | +0.32 | 24/16 | 0.0892 | 0.8665 |
| vmf_init | rank_corr | -0.0003 | -0.0004 | -0.05 | 19/21 | 0.7047 | 1.0000 |
| vmf_pin | truth_kl | -0.0005 | -0.0030 | -0.11 | 19/21 | 0.7852 | 1.0000 |
| vmf_pin | sampled_log_loss | -0.0020 | -0.0032 | -0.16 | 16/24 | 0.3900 | 1.0000 |
| vmf_pin | truth_kl_w | -0.0006 | -0.0034 | -0.17 | 17/23 | 0.3203 | 1.0000 |
| vmf_pin | ece | -0.0012 | -0.0032 | -0.16 | 19/21 | 0.3403 | 1.0000 |
| vmf_pin | cal_slope_dev | +0.0476 | +0.0488 | +0.60 | 28/12 | 0.0010 | 0.0159 |
| vmf_pin | centre_auc | -0.0010 | -0.0015 | -0.30 | 15/25 | 0.0269 | 0.3491 |
| vmf_pin | sampled_auc | +0.0002 | -0.0001 | -0.02 | 21/19 | 0.7446 | 1.0000 |
| vmf_pin | rank_corr | -0.0008 | -0.0005 | -0.05 | 17/23 | 0.2163 | 1.0000 |

### Per family, truth_kl advantage of TNBBeta over vMF (median over the family's 4 configs)

| family | vmf_init lib | vmf_init exact | vmf_pin lib | vmf_pin exact |
|---|---|---|---|---|
| assortative | +0.0205 | +0.0235 | +0.0299 | +0.0342 |
| bipartite | +0.0097 | +0.0156 | +0.0012 | +0.0050 |
| core_periphery | -0.0133 | -0.0047 | -0.0002 | -0.0056 |
| degree_corrected | +0.0341 | +0.0348 | +0.0295 | +0.0398 |
| geometric | -0.0009 | -0.0023 | +0.0107 | +0.0147 |
| hierarchical | -0.0066 | -0.0052 | -0.0062 | -0.0009 |
| latent_kernel | -0.0688 | -0.0736 | -0.0721 | -0.0723 |
| mixed_membership | +0.0008 | +0.0022 | -0.0016 | +0.0038 |
| ring | -0.0127 | -0.0146 | -0.0139 | -0.0051 |
| scale_free | +0.1869 | +0.1874 | +0.1710 | +0.2021 |

### Learned posterior concentration (mean log kappa, median over configs)

- vmf_init lib: median exp(mean log kappa) = 16.71
- vmf_init exact: median exp(mean log kappa) = 16.75
- vmf_pin lib: median exp(mean log kappa) = 16.00
- vmf_pin exact: median exp(mean log kappa) = 16.00


## Deviations, caveats

* No deviation from the pre-registration. Job 2304405's analysis printed the "exact better/worse" counts swapped (a labelling bug in my script, fixed; the tables here come from job 2304421).
* The rerun covers one setting (graph search, m = 16, kappa 16) and 40 of 200 configs; the audit table's other rows were not rerun and their bias sizes are from the sampler grid, not from retraining. In particular nothing here shows how much MNIST d = 20 / d = 40 or the fixed-kappa Cora (m = 65) results move.
* Audit "kappa" entries for learned-kappa studies come from write-up text, not checkpoints; where unknown I say so.
* Full pytest as a CPU job (2304353, before the test fix): 5 of 6 failures were my test bugs (fixed, file re-run: 24 passed); the sixth, `tests/apps/test_linear_decoder_kappa_control.py::test_initial_kappa_is_applied_and_trainable` (kappa 25.16 vs 25), belongs to another agent's in-progress, uncommitted study and is not touched. Final run 2304433 (after my test fix): 1168 passed, 1 skipped, 0 failed (the other agent's test passed by then).
* The `exact_envelope` flag does not change `rsample`'s gradient structure; no gradient-correctness test was added.
