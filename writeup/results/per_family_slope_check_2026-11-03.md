# Per-family slope check: TNBBeta advantage vs degree, CV, clustering, all 10 families (2026-11-03)

**EXPLORATORY AND DESCRIPTIVE ONLY.** Pre-registered H1 is the *pooled* within-family Spearman (+0.462, 200 fresh configs); nothing here
replaces or re-tests it. Each family has 20 fresh configs x 16 draws (Spearman SE about 0.23 at n = 20). Permutation p-values are
two-sided, 10,000 permutations, **uncorrected** unless labelled; Holm/BH (section B) cover only the 10 truth-KL-vs-degree tests and are a
descriptive guide, not confirmatory.

**Selection effect.** Picking the family for a figure *because* its slope looks good is a selection on the outcome. Any slope quoted for the
chosen family (e.g. scale_free +0.47, hierarchical +0.63) is a **biased-upward estimate**; with 10 families and SE about 0.23, the best of 10 null
slopes would typically sit near +0.3 to +0.4. Only the pooled +0.462 is an unbiased confirmatory number, and the figure caption should say the
family was chosen for visibility.

- Base commit: `dd8ede0` on `synthetic-graph-families` (working tree had other uncommitted files; the new files are uncommitted).
- Commands (no training; reads cached `runs_fresh.json` / `covariates_fresh.json`; about 80 s CPU so run via sbatch, job 2304466, log `slurm_logs/pf-slope_2304466.out`):
  `sbatch scripts/slurm/per_family_slope_check.sbatch` which runs `uv run --no-sync python -m apps.eval.per_family_slope_check`
  (script `apps/eval/per_family_slope_check.py`, reuses `bipartite_slope_check._partial` and the metric-first loaders).
- Definitions as in `bipartite_slope_check_2026-11-03.md`: advantage per config = median over draws of `comparator - tnbbeta` (lower-better) or `tnbbeta - comparator`
  (AUC), min over `vmf_init`/`power_spherical_init`; positive = TNBBeta better. Covariates are the study's realised mean degree, expected-degree CV, clustering.
  Bootstrap CI: 2,000 resamples of configs (percentile). Partials: rank-based, permutation p from 2,000 permutations. "Drop top-2": greedily remove the two configs whose
  removal moves the degree Spearman most, recompute. CV bands = groups of sorted CV split at gaps > 0.1. Corner thresholds: low degree < 11, high degree > 20, high CV > 0.5.

## Findings

- **Slopes of truth-KL advantage vs mean degree (Spearman):** hierarchical +0.63 (p 0.004; Holm 0.039, BH 0.039), scale_free +0.47 (0.036; BH 0.16), mixed_membership +0.44 (0.058),
  core_periphery +0.42 (0.064), assortative +0.31, ring +0.23, degree_corrected +0.22, geometric +0.16, **bipartite +0.06 (flat)**, **latent_kernel -0.08 (flat/slightly negative)**.
  No family is significantly negative; 8 of 10 are positive, consistent with the pooled +0.46 but only hierarchical survives a Holm guide.
- **Positive-advantage region.** Median truth-KL advantage is positive only in scale_free (+0.026, 16/20 configs), bipartite (+0.006, 13/20), assortative (+0.004, 11/20) and geometric (+0.003, 13/20);
  negative in the other six (hierarchical, core_periphery, latent_kernel, mixed_membership, ring, degree_corrected). So in hierarchical, mixed_membership and core_periphery the slope is real-looking but TNBBeta is mostly worse and only reaches parity or a win at the densest end.
- **AUROC loss.** Centre-AUC advantage is negative in the median in all 10 families (mixed_membership about 0, 10/20 positive) and in 19/20 configs for scale_free, ring, assortative and degree_corrected; sampled AUC is negative in the median in 8/10 (positive in scale_free +0.012, 11/20, and mixed_membership +0.002). AUC advantage tends to
  *rise* with degree too (geometric +0.74, hierarchical +0.47, latent_kernel +0.53 for sampled AUC), so AUROC loss is largest at low density.
- **Leverage.** Dropping the top-2 influential configs leaves the degree slope positive in 8/10 families and larger in several (scale_free +0.76, core_periphery +0.73, mixed_membership +0.68, hierarchical +0.63),
  i.e. the slopes are not driven by two points; the top points tend to be outliers working *against* the trend (e.g. hierarchical-02, -0.134 at degree 12). Bipartite stays weak-to-moderate (+0.33) but with CI [-0.37, +0.51].
- **Partials.** Degree given CV mirrors the marginal in every family (CV and degree are weakly related: |corr| <= 0.28 except geometric -0.45); CV given degree is weakly positive everywhere except scale_free (-0.09) and
  nowhere significant for truth KL (smallest p 0.10, core_periphery). The heterogeneity effect on truth KL is not detected in any family; the slope story is about density.
- **Design reach.** scale_free has the most continuous heterogeneity (8 CV bands, CV 0.52-2.22) but **no low-CV configs at all** (degree_corrected likewise: single level, CV 0.51-0.77); a low-vs-high heterogeneity contrast is impossible within those two.
  All other families have 3-4 CV bands and CV from about 0 to 0.8 (core_periphery 0.40-1.42). Both (low degree, high CV) and (high degree, high CV) corners exist (n >= 3 each) in assortative (3/4), bipartite (4/3), core_periphery (8/5), mixed_membership (3/4), latent_kernel (3/3), degree_corrected (9/5), scale_free (10/5);
  hierarchical has 4/1, ring 2/1, geometric 3/0. Mean advantage in those corners: assortative +0.031 / +0.043, bipartite +0.048 / +0.021, mixed_membership +0.002 / +0.053, scale_free +0.009 / +0.034.

## Suitability ranking for the 2x2 figure (judgement, not a test)

Criteria: clear positive truth-KL slope vs degree; visible positive-advantage region; AUROC loss visible; continuous-enough heterogeneity knob; small drawable graphs easy.

| rank | family | slope (truth KL vs degree) | positive region | AUROC loss | heterogeneity | drawing | verdict |
|---|---|---|---|---|---|---|---|
| 1 | scale_free | +0.47 (biased up; BH 0.16; drop-2 +0.76) | yes: 16/20 > 0, median +0.026 | centre AUC 19/20 negative; sampled AUC mixed | continuous (CV 0.5-2.2) but **no low-CV cell** | trivial (preferential attachment, hubs draw well) | best for the density axis and a visible win; heterogeneity contrast must be moderate (CV < 1) vs heavy (CV > 1.2), not none vs heavy |
| 2 | assortative | +0.31 (ns) | yes, 11/20, corners all populated (3/4/6/3) | centre AUC 19/20 negative | 3 levels, CV 0-0.82 | easy (SBM) | best if a full 2x2 incl. a homogeneous row is required, but slope is weak and unconfirmed in-family; the largest two wins (+0.14, +0.11) carry the corner means |
| 3 | hierarchical | +0.63 (biased up; Holm 0.039) | weak: median -0.005, 7/20; wins only at degree > 20, one high-CV corner config | centre AUC 16/20 negative | 3 levels, CV 0-0.8 | easy (nested blocks) | strongest, cleanest slope (loss at low density to about 0 or win at high), but mostly shows "TNBBeta loses" with one big win (hierarchical-09, +0.12) |
| 4 | bipartite | +0.06 (flat) | yes, 13/20, but flat in degree | mixed | 3 levels | easy | rejected for the density axis |
| 5 | mixed_membership / core_periphery | +0.44 / +0.42 (ns) | no, median negative | present | 3 bands | moderate | not recommended |
| 6 | remaining (geometric, ring, degree_corrected, latent_kernel) | +0.16, +0.23, +0.22, -0.08 | no | present | geometric/ring 3 levels | geometric/ring easy | not recommended; latent_kernel flat/negative |

**Recommendation.** Use **scale_free** for the "denser graphs => TNBBeta better" panel (positive slope, clear win region, centre-AUROC loss visible), and define the heterogeneity factor *within* its realised range
(moderate CV 0.5-0.9 vs heavy CV > 1.2; this is a tail-exponent contrast, not "none vs heavy"), or use **assortative** if the 2x2 must include a truly homogeneous row, accepting its weaker slope. Do not use bipartite
(flat) or latent_kernel (flat/negative). In captions: state the families were compared and this one chosen for visibility; the in-family slope is a biased-upward illustration; the unbiased evidence is pooled H1 (+0.462)
and, for scale_free, the study's own family E_c +0.029 with a 95% interval that spans 0.

## Caveats

- n = 20 per family, 10 families, a selection of the best look; uncorrected p-values apart from section B. Corner counts are 0-10 configs, mean advantages in corners are descriptive only.
- Advantage here uses the per-config median (the study uses the mean), so values differ slightly from the study's `E_c`.
- Covariates are realised, not design axes; CV in scale_free is realised heterogeneity driven by the tail exponent and so is not an independent knob from graph shape.
- In scale_free the CV-conditional statement is conditional on CV >= 0.5; there is no data for homogeneous scale-free graphs.

# Full output (slurm_logs/pf-slope_2304466.out)

## A. truth-KL advantage vs covariates (Spearman / Kendall; perm p uncorrected)

| family | median adv | n>0 | rho deg (p) [boot 95%] | tau deg | rho CV (p) | tau CV | rho clust (p) | tau clust | partial deg\|CV (p) | partial CV\|deg (p) | rho deg drop top-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| assortative | +0.0040 | 11/20 | +0.305 (0.191) [-0.20,+0.69] | +0.221 | +0.104 (0.663) | +0.053 | +0.504 (0.025) | +0.326 | +0.302 (0.218) | +0.093 (0.685) | +0.346 |
| bipartite | +0.0057 | 13/20 | +0.057 (0.813) [-0.37,+0.51] | +0.042 | +0.089 (0.709) | +0.063 | +0.314 (0.181) | +0.232 | +0.080 (0.749) | +0.105 (0.663) | +0.325 |
| core_periphery | -0.0057 | 7/20 | +0.423 (0.064) [-0.07,+0.80] | +0.326 | +0.257 (0.267) | +0.147 | +0.081 (0.737) | +0.053 | +0.496 (0.035) | +0.377 (0.102) | +0.734 |
| degree_corrected | -0.0181 | 8/20 | +0.223 (0.340) [-0.29,+0.64] | +0.147 | +0.140 (0.562) | +0.063 | +0.045 (0.849) | +0.053 | +0.266 (0.277) | +0.204 (0.401) | +0.238 |
| geometric | +0.0027 | 13/20 | +0.155 (0.505) [-0.38,+0.64] | +0.126 | +0.110 (0.639) | +0.116 | +0.507 (0.021) | +0.368 | +0.231 (0.365) | +0.204 (0.395) | +0.232 |
| hierarchical | -0.0050 | 7/20 | +0.629 (0.004) [+0.29,+0.82] | +0.432 | +0.068 (0.771) | +0.053 | +0.642 (0.003) | +0.463 | +0.646 (0.006) | +0.203 (0.391) | +0.628 |
| latent_kernel | -0.0032 | 8/20 | -0.075 (0.748) [-0.62,+0.40] | -0.137 | +0.293 (0.207) | +0.221 | -0.054 (0.825) | -0.074 | -0.079 (0.738) | +0.294 (0.218) | -0.090 |
| mixed_membership | -0.0132 | 7/20 | +0.436 (0.058) [-0.02,+0.78] | +0.305 | +0.053 (0.828) | +0.053 | +0.362 (0.114) | +0.232 | +0.442 (0.059) | +0.093 (0.684) | +0.680 |
| ring | -0.0197 | 5/20 | +0.230 (0.321) [-0.24,+0.65] | +0.158 | +0.296 (0.208) | +0.200 | +0.143 (0.538) | +0.137 | +0.184 (0.442) | +0.264 (0.266) | -0.034 |
| scale_free | +0.0255 | 16/20 | +0.469 (0.036) [+0.02,+0.79] | +0.337 | -0.212 (0.370) | -0.158 | +0.286 (0.222) | +0.200 | +0.436 (0.066) | -0.093 (0.695) | +0.761 |

## B. exploratory multiplicity guide over the 10 truth-KL-vs-degree tests (two-sided perm p)

| family | rho | p | Holm adj | BH adj |
|---|---|---|---|---|
| assortative | +0.305 | 0.1911 | 1.0000 | 0.3822 |
| bipartite | +0.057 | 0.8129 | 1.0000 | 0.8129 |
| core_periphery | +0.423 | 0.0640 | 0.4680 | 0.1600 |
| degree_corrected | +0.223 | 0.3400 | 1.0000 | 0.4857 |
| geometric | +0.155 | 0.5045 | 1.0000 | 0.6307 |
| hierarchical | +0.629 | 0.0039 | 0.0390 | 0.0390 |
| latent_kernel | -0.075 | 0.7478 | 1.0000 | 0.8129 |
| mixed_membership | +0.436 | 0.0585 | 0.4680 | 0.1600 |
| ring | +0.230 | 0.3215 | 1.0000 | 0.4857 |
| scale_free | +0.469 | 0.0364 | 0.3276 | 0.1600 |

## C. ece: advantage vs covariates (higher = TNBBeta better)

| family | median adv | n>0 | rho deg (p) [boot 95%] | rho CV (p) | rho clust (p) | partial deg\|CV (p) | partial CV\|deg (p) | rho deg drop top-2 |
|---|---|---|---|---|---|---|---|---|
| assortative | +0.0215 | 14/20 | +0.386 (0.096) [-0.11,+0.76] | +0.102 (0.663) | +0.588 (0.008) | +0.384 (0.092) | +0.090 (0.711) | +0.426 |
| bipartite | +0.0442 | 18/20 | +0.161 (0.498) [-0.32,+0.54] | -0.023 (0.926) | +0.206 (0.379) | +0.160 (0.505) | +0.016 (0.946) | -0.046 |
| core_periphery | -0.0144 | 6/20 | +0.304 (0.189) [-0.14,+0.63] | +0.397 (0.087) | +0.564 (0.011) | +0.419 (0.072) | +0.485 (0.035) | +0.519 |
| degree_corrected | +0.0329 | 14/20 | +0.295 (0.202) [-0.18,+0.67] | -0.502 (0.027) | +0.441 (0.051) | +0.209 (0.390) | -0.466 (0.059) | +0.082 |
| geometric | -0.0029 | 9/20 | +0.311 (0.175) [-0.19,+0.69] | +0.009 (0.971) | +0.287 (0.222) | +0.354 (0.147) | +0.177 (0.467) | +0.055 |
| hierarchical | -0.0149 | 6/20 | +0.704 (0.001) [+0.38,+0.85] | -0.072 (0.760) | +0.538 (0.017) | +0.702 (0.001) | +0.039 (0.879) | +0.721 |
| latent_kernel | -0.0224 | 3/20 | -0.105 (0.650) [-0.55,+0.43] | +0.108 (0.647) | -0.122 (0.605) | -0.106 (0.679) | +0.109 (0.631) | +0.174 |
| mixed_membership | -0.0292 | 5/20 | +0.350 (0.130) [-0.17,+0.75] | +0.129 (0.580) | +0.304 (0.192) | +0.363 (0.131) | +0.165 (0.511) | +0.635 |
| ring | +0.0218 | 14/20 | -0.350 (0.130) [-0.75,+0.14] | +0.162 (0.487) | -0.131 (0.583) | -0.395 (0.089) | +0.251 (0.288) | -0.641 |
| scale_free | +0.0250 | 17/20 | +0.329 (0.155) [-0.21,+0.82] | -0.483 (0.035) | +0.102 (0.657) | +0.229 (0.348) | -0.430 (0.066) | +0.721 |

## C. sampled_log_loss: advantage vs covariates (higher = TNBBeta better)

| family | median adv | n>0 | rho deg (p) [boot 95%] | rho CV (p) | rho clust (p) | partial deg\|CV (p) | partial CV\|deg (p) | rho deg drop top-2 |
|---|---|---|---|---|---|---|---|---|
| assortative | -0.0057 | 9/20 | +0.439 (0.056) [-0.02,+0.79] | +0.012 (0.963) | +0.568 (0.011) | +0.439 (0.060) | -0.012 (0.963) | +0.692 |
| bipartite | +0.0138 | 17/20 | +0.137 (0.560) [-0.32,+0.55] | -0.092 (0.705) | +0.087 (0.712) | +0.119 (0.606) | -0.062 (0.804) | -0.115 |
| core_periphery | -0.0075 | 8/20 | +0.454 (0.048) [-0.05,+0.81] | +0.391 (0.090) | +0.298 (0.207) | +0.583 (0.008) | +0.543 (0.017) | +0.723 |
| degree_corrected | -0.0075 | 8/20 | -0.032 (0.892) [-0.52,+0.44] | +0.182 (0.444) | -0.086 (0.708) | +0.012 (0.955) | +0.180 (0.435) | -0.094 |
| geometric | -0.0011 | 10/20 | +0.323 (0.162) [-0.24,+0.74] | +0.116 (0.628) | +0.292 (0.205) | +0.424 (0.073) | +0.311 (0.190) | +0.071 |
| hierarchical | -0.0141 | 9/20 | +0.666 (0.002) [+0.34,+0.83] | -0.006 (0.983) | +0.526 (0.018) | +0.672 (0.004) | +0.119 (0.630) | +0.773 |
| latent_kernel | -0.0102 | 3/20 | -0.015 (0.953) [-0.44,+0.43] | +0.180 (0.451) | -0.233 (0.317) | -0.016 (0.950) | +0.180 (0.447) | +0.265 |
| mixed_membership | -0.0102 | 7/20 | +0.495 (0.029) [+0.01,+0.82] | +0.179 (0.453) | +0.444 (0.051) | +0.517 (0.022) | +0.247 (0.303) | +0.748 |
| ring | -0.0059 | 6/20 | +0.080 (0.737) [-0.43,+0.57] | +0.000 (1.000) | -0.003 (0.994) | +0.081 (0.740) | -0.016 (0.947) | -0.240 |
| scale_free | +0.0192 | 18/20 | +0.397 (0.087) [-0.15,+0.81] | -0.039 (0.868) | +0.313 (0.172) | +0.403 (0.091) | +0.084 (0.736) | +0.717 |

## C. sampled_auc: advantage vs covariates (higher = TNBBeta better)

| family | median adv | n>0 | rho deg (p) [boot 95%] | rho CV (p) | rho clust (p) | partial deg\|CV (p) | partial CV\|deg (p) | rho deg drop top-2 |
|---|---|---|---|---|---|---|---|---|
| assortative | -0.0026 | 7/20 | +0.156 (0.511) [-0.34,+0.59] | +0.132 (0.571) | +0.244 (0.308) | +0.151 (0.539) | +0.126 (0.583) | +0.432 |
| bipartite | -0.0048 | 8/20 | +0.382 (0.098) [-0.08,+0.72] | -0.380 (0.099) | -0.039 (0.871) | +0.326 (0.178) | -0.324 (0.165) | +0.404 |
| core_periphery | -0.0011 | 8/20 | +0.495 (0.028) [+0.05,+0.80] | +0.448 (0.050) | +0.203 (0.392) | +0.658 (0.003) | +0.633 (0.006) | +0.659 |
| degree_corrected | -0.0318 | 4/20 | -0.023 (0.925) [-0.54,+0.54] | +0.149 (0.537) | -0.056 (0.813) | +0.013 (0.953) | +0.148 (0.557) | -0.340 |
| geometric | -0.0012 | 8/20 | +0.743 (0.000) [+0.39,+0.92] | +0.000 (1.000) | +0.439 (0.052) | +0.833 (0.000) | +0.563 (0.014) | +0.864 |
| hierarchical | -0.0023 | 9/20 | +0.469 (0.039) [+0.05,+0.74] | +0.241 (0.306) | +0.250 (0.288) | +0.524 (0.027) | +0.351 (0.141) | +0.313 |
| latent_kernel | -0.0054 | 3/20 | +0.528 (0.020) [+0.10,+0.78] | +0.074 (0.758) | +0.553 (0.013) | +0.529 (0.027) | +0.086 (0.712) | +0.692 |
| mixed_membership | +0.0023 | 13/20 | +0.394 (0.084) [-0.05,+0.71] | +0.580 (0.009) | +0.429 (0.063) | +0.536 (0.020) | +0.663 (0.002) | +0.432 |
| ring | -0.0106 | 1/20 | +0.326 (0.152) [-0.15,+0.72] | -0.371 (0.109) | +0.322 (0.168) | +0.437 (0.051) | -0.469 (0.043) | +0.567 |
| scale_free | +0.0122 | 11/20 | +0.123 (0.606) [-0.41,+0.67] | +0.335 (0.143) | +0.374 (0.109) | +0.242 (0.315) | +0.389 (0.103) | +0.476 |

## C. centre_auc: advantage vs covariates (higher = TNBBeta better)

| family | median adv | n>0 | rho deg (p) [boot 95%] | rho CV (p) | rho clust (p) | partial deg\|CV (p) | partial CV\|deg (p) | rho deg drop top-2 |
|---|---|---|---|---|---|---|---|---|
| assortative | -0.0035 | 1/20 | +0.424 (0.065) [-0.04,+0.85] | -0.147 (0.533) | +0.546 (0.016) | +0.437 (0.055) | -0.187 (0.441) | +0.794 |
| bipartite | -0.0118 | 6/20 | -0.041 (0.869) [-0.49,+0.50] | +0.248 (0.285) | +0.311 (0.181) | +0.018 (0.940) | +0.246 (0.315) | +0.284 |
| core_periphery | -0.0054 | 4/20 | +0.096 (0.684) [-0.44,+0.60] | +0.433 (0.057) | +0.063 (0.795) | +0.200 (0.411) | +0.461 (0.052) | +0.441 |
| degree_corrected | -0.0200 | 1/20 | +0.144 (0.539) [-0.40,+0.66] | +0.069 (0.776) | +0.144 (0.542) | +0.166 (0.500) | +0.108 (0.655) | +0.220 |
| geometric | -0.0031 | 3/20 | +0.702 (0.001) [+0.39,+0.90] | -0.462 (0.039) | +0.220 (0.348) | +0.624 (0.004) | -0.227 (0.353) | +0.862 |
| hierarchical | -0.0025 | 4/20 | +0.517 (0.021) [+0.13,+0.76] | +0.110 (0.637) | +0.299 (0.207) | +0.541 (0.022) | +0.216 (0.379) | +0.678 |
| latent_kernel | -0.0068 | 1/20 | +0.326 (0.166) [-0.14,+0.67] | +0.108 (0.653) | +0.418 (0.066) | +0.328 (0.175) | +0.114 (0.640) | +0.567 |
| mixed_membership | -0.0000 | 10/20 | -0.089 (0.703) [-0.55,+0.43] | +0.068 (0.776) | -0.060 (0.796) | -0.084 (0.729) | +0.062 (0.806) | +0.236 |
| ring | -0.0080 | 1/20 | +0.281 (0.230) [-0.23,+0.72] | -0.370 (0.111) | +0.311 (0.182) | +0.387 (0.104) | -0.451 (0.053) | +0.329 |
| scale_free | -0.0113 | 1/20 | +0.197 (0.403) [-0.29,+0.59] | +0.078 (0.742) | +0.385 (0.098) | +0.229 (0.344) | +0.143 (0.565) | +0.441 |

## D. design ranges and corners

low degree < 11, high degree > 20, high CV > 0.5; bands = groups of sorted CV split at gaps > 0.1

| family | degree range | CV range | CV bands | het levels | corr(deg,CV) | low-deg&highCV n (mean adv) | high-deg&highCV n (mean adv) | low-deg&lowCV n | high-deg&lowCV n | frac TNB>0 low-half / high-half deg | frac AUC adv<0 (sampled/centre) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| assortative | 4.5-39.1 | 0.00-0.82 | 3 | 3 | +0.05 | 3 (+0.031) | 4 (+0.043) | 6 | 3 | 0.40 / 0.70 | 0.65 / 0.95 |
| bipartite | 5.0-38.9 | 0.00-0.78 | 3 | 3 | -0.23 | 4 (+0.048) | 3 (+0.021) | 4 | 5 | 0.70 / 0.60 | 0.60 / 0.70 |
| core_periphery | 4.1-36.5 | 0.40-1.42 | 3 | 3 | -0.19 | 8 (-0.007) | 5 (+0.016) | 2 | 1 | 0.20 / 0.50 | 0.60 / 0.80 |
| degree_corrected | 4.1-39.6 | 0.51-0.77 | 1 | 1 | -0.24 | 9 (-0.023) | 5 (-0.013) | 0 | 0 | 0.30 / 0.50 | 0.80 / 0.95 |
| geometric | 4.0-36.3 | 0.02-0.77 | 4 | 3 | -0.45 | 3 (-0.003) | 0 (n/a) | 5 | 6 | 0.50 / 0.80 | 0.60 / 0.85 |
| hierarchical | 4.0-36.9 | 0.00-0.80 | 4 | 3 | -0.14 | 4 (-0.021) | 1 (+0.120) | 6 | 4 | 0.10 / 0.60 | 0.55 / 0.80 |
| latent_kernel | 4.0-35.2 | 0.03-0.79 | 3 | 3 | +0.00 | 3 (+0.014) | 3 (-0.002) | 6 | 3 | 0.40 / 0.40 | 0.85 / 0.95 |
| mixed_membership | 4.1-34.2 | 0.02-0.86 | 3 | 3 | -0.07 | 3 (+0.002) | 4 (+0.053) | 3 | 3 | 0.20 / 0.50 | 0.35 / 0.50 |
| ring | 4.3-37.6 | 0.00-0.73 | 3 | 3 | +0.19 | 2 (-0.021) | 1 (-0.012) | 7 | 5 | 0.30 / 0.20 | 0.95 / 0.95 |
| scale_free | 4.4-33.4 | 0.52-2.22 | 8 | 1 | -0.28 | 10 (+0.009) | 5 (+0.034) | 0 | 0 | 0.60 / 1.00 | 0.45 / 0.95 |

## E. per-config scatter (family, config, het, degree, CV, clustering, truth_kl, ece, sampled_auc)

```

assortative-02,0,4.5,0.01,0.049,-0.0253,+0.0215,-0.0245  
assortative-08,0,4.9,0.00,0.007,-0.0437,-0.0482,+0.0037  
assortative-04,2,6.1,0.77,0.019,-0.0195,-0.0309,-0.0047  
assortative-13,2,6.6,0.64,0.173,+0.1359,+0.1189,+0.0043  
assortative-15,1,7.0,0.21,0.027,+0.0156,+0.0186,-0.0215  
assortative-18,0,8.0,0.00,0.031,+0.0347,+0.0296,-0.0017  
assortative-00,1,9.1,0.21,0.042,-0.0081,-0.0355,-0.0074  
assortative-11,1,9.5,0.19,0.019,-0.0463,-0.0614,-0.0029  
assortative-17,2,10.3,0.82,0.060,-0.0224,-0.0231,+0.0077  
assortative-06,2,11.8,0.69,0.108,+0.0119,+0.0302,-0.0076  
assortative-07,1,13.7,0.21,0.015,+0.0014,+0.0357,-0.0053  
assortative-12,1,14.8,0.20,0.118,+0.0588,+0.0215,-0.0023  
assortative-01,1,18.5,0.20,0.110,-0.1214,+0.0818,-0.1260  
assortative-10,2,20.2,0.72,0.071,-0.0268,-0.0180,+0.0085  
assortative-19,2,21.3,0.68,0.128,+0.0828,+0.0968,+0.0743  
assortative-14,0,23.9,0.00,0.038,+0.0164,+0.0256,+0.0035  
assortative-16,0,25.8,0.00,0.056,-0.0028,+0.0042,-0.0038  
assortative-05,0,29.9,0.00,0.608,+0.0277,+0.0552,-0.0014  
assortative-09,2,31.3,0.61,0.160,+0.0067,+0.0126,+0.0067  
assortative-03,2,39.1,0.70,0.113,+0.1110,+0.1312,-0.0572  
bipartite-09,2,5.0,0.68,0.007,+0.0340,+0.0505,-0.0263  
bipartite-18,2,6.4,0.71,0.011,+0.0302,+0.0441,-0.0326  
bipartite-00,1,6.5,0.20,0.006,-0.0033,+0.0196,-0.0013  
bipartite-16,0,6.7,0.00,0.004,+0.0003,+0.0080,+0.0050  
bipartite-05,0,7.2,0.00,0.004,+0.0055,+0.0547,-0.0091  
bipartite-11,1,8.1,0.20,0.004,-0.0019,+0.0523,+0.0004  
bipartite-02,2,9.8,0.72,0.054,-0.0130,-0.0257,+0.0358  
bipartite-13,2,10.7,0.77,0.036,+0.1406,+0.1633,-0.0239  
bipartite-03,0,13.6,0.00,0.015,+0.0059,+0.0442,-0.0197  
bipartite-19,2,14.4,0.76,0.051,+0.0520,+0.0847,-0.0652  
bipartite-10,2,17.8,0.78,0.054,-0.0077,+0.0259,-0.0577  
bipartite-06,1,18.6,0.20,0.031,-0.0038,+0.0343,-0.0227  
bipartite-15,2,21.0,0.63,0.000,-0.0233,+0.0099,-0.0398  
bipartite-04,2,23.1,0.70,0.001,-0.0029,-0.0009,+0.0091  
bipartite-14,1,26.3,0.20,0.017,+0.0076,+0.0168,+0.0179  
bipartite-17,1,28.0,0.20,0.028,+0.0125,+0.0384,-0.0076  
bipartite-01,2,29.7,0.66,0.136,+0.0905,+0.0753,+0.0517  
bipartite-12,1,34.3,0.19,0.013,+0.0201,+0.0634,+0.0212  
bipartite-08,0,35.8,0.00,0.028,+0.0106,+0.0498,-0.0019  
bipartite-07,0,38.9,0.00,0.000,+0.0026,+0.0718,+0.0194  
core_periphery-05,0,4.1,0.50,0.109,-0.0332,-0.0270,-0.0126  
core_periphery-16,0,4.8,0.92,0.083,+0.0338,+0.0348,-0.0003  
core_periphery-14,0,5.1,1.42,0.305,-0.0302,-0.0149,-0.0034  
core_periphery-09,1,5.6,0.58,0.134,-0.0062,-0.0219,-0.0124  
core_periphery-03,1,7.0,1.14,0.154,+0.0215,-0.0019,+0.0099  
core_periphery-07,0,7.2,1.05,0.424,-0.0242,-0.0181,-0.0000  
core_periphery-17,0,8.7,0.42,0.031,-0.0152,-0.0479,-0.0111  
core_periphery-13,1,8.8,0.73,0.107,-0.0224,-0.0539,+0.0020  
core_periphery-11,2,10.0,0.87,0.246,-0.0036,+0.0209,-0.0058  
core_periphery-19,2,10.6,0.90,0.259,-0.0239,-0.0234,-0.0100  
core_periphery-01,2,12.4,0.73,0.065,-0.0305,-0.0383,-0.0055  
core_periphery-02,0,13.5,0.68,0.264,-0.0170,-0.0050,-0.0137  
core_periphery-08,0,14.7,1.09,0.152,+0.0545,-0.0125,+0.0342  
core_periphery-15,2,17.4,0.88,0.080,-0.0070,-0.0241,-0.0020  
core_periphery-04,1,20.4,1.05,0.373,+0.0333,+0.0465,+0.0028  
core_periphery-00,1,24.9,0.80,0.296,+0.0018,+0.0900,+0.0055  
core_periphery-10,1,26.8,0.40,0.110,-0.0052,-0.0189,-0.0093  
core_periphery-18,0,29.2,0.62,0.198,-0.0008,+0.0015,+0.0078  
core_periphery-12,2,34.2,0.74,0.185,+0.0145,+0.0102,+0.0124  
core_periphery-06,2,36.5,0.69,0.245,+0.0294,-0.0139,+0.0667  
degree_corrected-11,0,4.1,0.61,0.125,+0.0003,+0.0327,+0.0017  
degree_corrected-16,0,4.6,0.75,0.029,-0.0541,-0.0229,-0.0070  
degree_corrected-07,0,5.1,0.75,0.016,-0.0115,-0.0025,-0.0566  
degree_corrected-01,0,5.6,0.68,0.035,-0.0353,+0.0346,-0.0857  
degree_corrected-18,0,6.3,0.68,0.044,-0.0178,+0.0332,-0.0754  
degree_corrected-05,0,7.2,0.63,0.067,-0.0777,+0.0487,-0.2224  
degree_corrected-14,0,8.6,0.71,0.035,+0.0161,+0.0555,-0.0224  
degree_corrected-13,0,8.8,0.72,0.029,+0.0029,+0.0461,-0.0411  
degree_corrected-19,0,10.3,0.71,0.048,-0.0327,-0.0254,-0.0095  
degree_corrected-03,0,12.3,0.72,0.033,-0.0457,-0.0355,-0.0057  
degree_corrected-09,0,12.7,0.62,0.105,-0.0184,-0.0030,-0.0035  
degree_corrected-17,0,14.3,0.56,0.227,+0.0176,+0.0470,+0.0005  
degree_corrected-00,0,15.9,0.75,0.048,+0.0053,+0.0032,+0.0104  
degree_corrected-06,0,17.6,0.53,0.204,-0.1079,+0.1370,-0.3326  
degree_corrected-15,0,19.3,0.71,0.120,+0.0362,+0.0524,-0.0621  
degree_corrected-10,0,22.0,0.77,0.052,-0.0198,-0.0016,-0.0956  
degree_corrected-02,0,23.0,0.51,0.285,-0.2256,+0.1008,-0.3480  
degree_corrected-08,0,28.7,0.64,0.093,-0.0279,+0.0058,-0.1573  
degree_corrected-12,0,31.9,0.58,0.201,+0.0161,+0.0302,-0.0011  
degree_corrected-04,0,39.6,0.69,0.081,+0.1907,+0.1635,+0.0776  
geometric-02,1,4.0,0.24,0.046,+0.0534,-0.0742,-0.0369  
geometric-08,1,4.9,0.20,0.007,-0.0403,-0.0608,-0.0350  
geometric-04,2,5.8,0.77,0.020,-0.0019,+0.0039,-0.0197  
geometric-16,1,6.6,0.21,0.023,+0.0038,+0.0227,-0.0019  
geometric-14,2,7.0,0.75,0.068,-0.0183,-0.0341,-0.0104  
geometric-06,1,8.0,0.21,0.055,+0.0368,-0.0089,-0.0042  
geometric-12,2,8.9,0.77,0.026,+0.0110,+0.0364,+0.0021  
geometric-00,0,10.2,0.02,0.011,-0.0331,-0.0608,-0.0051  
geometric-10,0,11.3,0.07,0.052,-0.0009,-0.0009,-0.0005  
geometric-19,0,12.1,0.09,0.091,+0.0246,+0.0174,+0.0000  
geometric-09,2,13.6,0.54,0.200,+0.0274,+0.0144,-0.0019  
geometric-03,2,15.1,0.72,0.049,-0.0172,-0.0436,+0.0054  
geometric-17,0,17.7,0.05,0.073,+0.0009,-0.0117,-0.0111  
geometric-15,0,19.0,0.02,0.018,+0.0055,+0.0260,-0.0025  
geometric-05,1,21.0,0.21,0.065,+0.0238,+0.0158,+0.0080  
geometric-18,1,23.1,0.21,0.055,+0.0015,-0.0121,+0.0001  
geometric-13,0,24.6,0.08,0.143,+0.0160,+0.0051,+0.0003  
geometric-07,0,29.5,0.03,0.043,-0.0029,-0.0186,-0.0002  
geometric-11,1,30.4,0.21,0.061,+0.0013,-0.0049,+0.0023  
geometric-01,1,36.3,0.20,0.177,+0.0455,+0.0527,+0.0112  
hierarchical-19,0,4.0,0.00,0.014,-0.0328,-0.0305,-0.0128  
hierarchical-10,1,4.6,0.20,0.013,-0.0255,-0.0302,-0.0028  
hierarchical-00,0,4.7,0.00,0.017,-0.0007,-0.0246,-0.0128  
hierarchical-17,2,5.6,0.73,0.050,-0.0187,-0.0603,-0.0375  
hierarchical-12,2,5.8,0.56,0.165,-0.0058,-0.0199,-0.0157  
hierarchical-06,1,6.5,0.20,0.009,-0.0739,-0.0893,+0.0003  
hierarchical-14,2,7.2,0.80,0.018,-0.0606,-0.0671,+0.0041  
hierarchical-16,1,8.0,0.20,0.018,-0.0228,-0.0303,-0.0162  
hierarchical-04,2,9.3,0.72,0.095,+0.0011,-0.0058,+0.0070  
hierarchical-08,0,10.0,0.00,0.078,-0.0332,-0.0352,-0.0018  
hierarchical-02,0,12.0,0.00,0.035,-0.1341,-0.1267,-0.0406  
hierarchical-11,1,12.8,0.19,0.088,+0.0109,+0.0204,+0.0067  
hierarchical-01,1,16.7,0.20,0.029,+0.0016,+0.0077,-0.0046  
hierarchical-13,0,17.9,0.00,0.030,-0.0078,-0.0045,+0.0003  
hierarchical-18,0,18.6,0.00,0.097,-0.0005,-0.0100,-0.0050  
hierarchical-07,1,21.4,0.19,0.360,+0.0076,+0.0043,+0.0007  
hierarchical-15,0,25.7,0.00,0.155,+0.0051,+0.0216,-0.0030  
hierarchical-05,0,26.5,0.00,0.056,-0.0042,+0.0109,+0.0000  
hierarchical-09,2,32.8,0.72,0.078,+0.1196,+0.0807,+0.1320  
hierarchical-03,2,36.9,0.47,0.329,+0.0072,-0.0013,+0.0065  
latent_kernel-13,0,4.0,0.04,0.009,-0.0232,-0.0267,-0.0127  
latent_kernel-19,1,4.4,0.20,0.014,+0.0296,+0.0780,-0.0025  
latent_kernel-07,0,4.8,0.07,0.020,-0.0141,-0.0083,-0.0123  
latent_kernel-01,2,6.0,0.79,0.013,+0.0497,+0.0794,-0.0177  
latent_kernel-11,2,6.5,0.66,0.126,+0.0255,-0.0124,-0.0051  
latent_kernel-03,1,7.7,0.21,0.038,-0.0347,-0.0656,-0.0057  
latent_kernel-09,2,9.2,0.74,0.027,-0.0331,-0.0399,-0.0212  
latent_kernel-15,1,9.9,0.23,0.049,-0.0734,-0.1744,-0.0031  
latent_kernel-17,0,10.8,0.06,0.024,-0.0005,-0.0219,-0.0150  
latent_kernel-05,1,11.0,0.20,0.017,+0.0248,+0.0579,-0.0104  
latent_kernel-08,1,12.9,0.21,0.041,+0.0044,-0.0338,-0.0143  
latent_kernel-02,0,15.3,0.03,0.029,-0.0094,-0.0524,-0.0111  
latent_kernel-04,1,18.9,0.22,0.153,+0.0016,-0.0229,-0.0039  
latent_kernel-16,2,19.7,0.72,0.093,+0.0026,-0.0288,+0.0065  
latent_kernel-14,1,21.2,0.20,0.023,-0.0105,-0.0295,+0.0019  
latent_kernel-06,2,24.4,0.71,0.060,+0.0008,-0.0165,-0.0068  
latent_kernel-18,2,27.1,0.64,0.140,-0.0023,-0.0111,+0.0032  
latent_kernel-12,2,29.2,0.60,0.176,-0.0041,-0.0175,-0.0029  
latent_kernel-10,0,31.5,0.03,0.041,-0.0153,-0.0384,-0.0049  
latent_kernel-00,0,35.2,0.09,0.220,-0.0133,-0.0065,-0.0009  
mixed_membership-18,1,4.1,0.20,0.006,-0.0400,-0.0748,+0.0008  
mixed_membership-07,1,4.2,0.19,0.015,-0.2089,-0.1442,+0.0036  
mixed_membership-10,2,4.3,0.65,0.020,+0.0024,+0.0583,-0.0317  
mixed_membership-12,1,5.4,0.20,0.026,-0.0146,-0.0185,+0.0008  
mixed_membership-03,2,5.6,0.75,0.014,+0.0176,+0.0325,-0.0168  
mixed_membership-01,2,8.4,0.77,0.022,-0.0129,-0.0351,+0.0145  
mixed_membership-16,2,11.4,0.73,0.024,-0.0705,-0.0856,+0.0097  
mixed_membership-04,1,12.2,0.19,0.028,-0.0167,-0.0752,-0.0218  
mixed_membership-14,1,12.5,0.21,0.030,-0.0392,-0.0617,-0.0075  
mixed_membership-17,2,15.0,0.86,0.037,-0.0506,-0.0698,+0.0078  
mixed_membership-06,0,15.5,0.03,0.060,-0.0134,-0.0377,-0.0277  
mixed_membership-13,2,15.7,0.76,0.062,-0.0162,-0.0397,+0.0183  
mixed_membership-11,0,19.7,0.03,0.061,+0.0003,+0.0043,-0.0072  
mixed_membership-02,2,20.3,0.81,0.050,+0.1816,+0.1670,+0.0714  
mixed_membership-08,1,22.6,0.20,0.057,+0.0012,-0.0052,+0.0034  
mixed_membership-05,2,24.5,0.63,0.091,+0.0460,+0.0331,+0.0225  
mixed_membership-15,0,25.3,0.02,0.049,-0.0017,-0.0052,+0.0012  
mixed_membership-19,2,31.4,0.76,0.071,-0.0165,-0.0233,+0.0145  
mixed_membership-00,0,32.9,0.02,0.051,-0.0113,-0.0355,-0.0089  
mixed_membership-09,2,34.2,0.68,0.140,+0.0028,-0.0052,+0.0239  
ring-05,1,4.3,0.21,0.046,-0.0945,+0.0484,-0.2001  
ring-15,0,4.9,0.00,0.018,-0.0025,+0.0452,-0.0104  
ring-17,0,5.1,0.00,0.013,-0.0208,+0.0413,-0.0343  
ring-03,2,6.0,0.62,0.045,+0.0045,+0.0280,-0.0230  
ring-09,1,6.3,0.20,0.062,-0.0248,+0.0044,-0.0107  
ring-07,0,7.4,0.00,0.039,-0.0468,-0.0074,-0.0055  
ring-19,1,8.7,0.20,0.122,-0.0214,+0.0055,-0.0062  
ring-13,0,9.0,0.00,0.172,+0.0018,+0.0255,-0.0021  
ring-01,2,9.5,0.68,0.047,-0.0475,+0.0332,-0.1665  
ring-11,2,12.4,0.73,0.073,+0.0377,+0.0629,-0.0560  
ring-08,1,14.5,0.20,0.083,-0.0294,+0.0238,-0.1927  
ring-02,1,14.7,0.19,0.223,-0.1182,+0.1625,-0.3799  
ring-16,2,18.1,0.31,0.292,-0.0093,-0.0011,-0.0037  
ring-14,1,18.8,0.20,0.111,-0.0172,-0.0033,-0.0003  
ring-04,1,20.4,0.19,0.086,-0.0478,-0.0534,+0.0053  
ring-10,0,23.2,0.00,0.142,-0.0186,+0.0024,-0.0036  
ring-00,0,29.5,0.00,0.136,-0.0244,-0.0021,-0.0012  
ring-12,2,31.9,0.64,0.142,-0.0119,-0.0087,-0.0761  
ring-18,1,33.2,0.21,0.059,+0.0026,+0.0198,-0.0427  
ring-06,2,37.6,0.26,0.370,+0.0498,+0.0607,-0.0017  
scale_free-01,0,4.4,0.73,0.027,+0.0184,+0.0692,-0.0593  
scale_free-06,0,4.9,1.60,0.035,-0.0003,-0.0491,-0.0241  
scale_free-18,0,5.7,1.57,0.069,-0.0242,-0.0505,+0.0440  
scale_free-11,0,6.2,0.87,0.033,+0.0025,+0.0196,-0.0257  
scale_free-12,0,6.7,1.25,0.031,+0.0611,+0.0800,-0.0671  
scale_free-09,0,7.6,2.22,0.066,-0.0125,-0.0440,+0.0679  
scale_free-17,0,8.3,0.65,0.020,+0.0157,+0.0394,-0.0260  
scale_free-04,0,8.9,1.39,0.062,+0.0262,+0.0189,+0.0244  
scale_free-08,0,9.6,0.54,0.038,+0.0109,+0.0125,+0.0266  
scale_free-14,0,10.2,1.06,0.057,-0.0123,+0.0139,+0.0029  
scale_free-13,0,13.1,0.93,0.040,+0.0249,+0.0202,-0.0007  
scale_free-10,0,13.9,0.95,0.114,+0.0938,+0.0574,+0.0464  
scale_free-05,0,15.2,1.99,0.084,+0.0414,+0.0179,+0.0251  
scale_free-15,0,17.6,1.25,0.133,+0.0439,+0.0277,+0.0397  
scale_free-00,0,20.0,0.52,0.071,+0.0315,+0.0511,-0.0219  
scale_free-02,0,23.4,1.44,0.106,+0.0424,+0.0412,+0.0461  
scale_free-16,0,26.7,0.53,0.045,+0.0432,+0.0371,+0.0215  
scale_free-07,0,28.6,0.90,0.133,+0.0494,+0.0459,+0.0660  
scale_free-03,0,30.5,0.63,0.151,+0.0307,+0.0591,-0.0308  
scale_free-19,0,33.4,1.17,0.143,+0.0023,+0.0224,-0.0751  
```
