# ITD-only HRTF, arm V follow-up (collapse-controlled): pre-registration

Date: 2026-11-01. Base commit `10d30b9`. STATUS: FROZEN (approved by the user) before any run of this
follow-up; edited afterwards only by appending a "Deviations" section. Results go in
`hrtf_itd_armv_followup_results_<date>.md`.

**Provenance.** This follow-up was designed AFTER seeing the arm V result of
`hrtf_itd_ring_results_2026-11-01.md` (the registered rule was not met; vMF and PowerSpherical
collapsed). It is a second look at the same data, motivated by that result, and must be read as
such: its p-values are pre-registered for this follow-up but are not independent of the first
analysis.

## What went wrong in arm V, and what this follow-up changes

In the registered arm V, vMF and PowerSpherical collapsed to a nearly uniform `q` (kappa about 1,
learned sigma 0.76-0.83, negative ELBO about 1.42-1.46, the value of a decoder that ignores
`z`); their `p_model` was then also nearly uniform, so the registered primary metric
(`KL(q || p_model)`) was trivially small for them. TNBBeta used the latent in most splits (sigma
0.10-0.27, negative ELBO 1.37-1.40). The decoder axis `u` was also not tied to the interaural
axis, so the label-based latitude check was meaningful only in a few splits. Changes, applied
identically to every family:
1. **Primary metric: the held-out importance-weighted bound on `log p(x)`** (the VAE's own
   objective), not the KL to the model's own posterior.
2. **Collapse controls:** KL warm-up over the first 100 epochs (a third of training) instead of
   10%, and the initial learned sigma 0.1 (sigma is still learned; this is only its starting
   value, as in the project's convention).
3. **A second decoder, `fixed_axis`:** the single-index decoder with the axis fixed to the
   interaural axis `u = +y` (only `g` and sigma are learned), so the latent is tied to the
   physical frame and the label-based latitude check is meaningful in every split. This uses
   knowledge of the axis (not of any direction label).

## Data, splits, estimator (unchanged from `hrtf_itd_ring_preregistration_2026-11-01.md`)

SADIE II human subjects, 96 kHz HRIRs, cross-correlation ITD standardised by the training
subjects' area-weighted mean/sd, area-weighted sampling and evaluation, 10 random subject splits
(seeds 0-9, 3 test subjects each), no early stopping, no tuning. Only the xcorr estimator
(no onset sensitivity run).

## Models and arms

Encoder MLP `[1, 128, 128]` + posterior head, Uniform(S^2) prior, Gaussian likelihood with one
learned sigma. Families: `vmf`, `power_spherical`, `tnbbeta` (library default), `tnbbeta_init0`
(descriptive). Decoders: `learned_axis` (the registered single-index decoder `g(u . z)`, `u`
learned) and `fixed_axis` (`u = +y`). Training: Adam lr 1e-3, batch 256, 300 epochs,
4 ELBO samples, linear KL warm-up over 100 epochs, initial sigma 0.1, paired by split seed.
2 decoders x 4 families x 10 splits = 80 runs.

## Metrics, tests, decision rules

* **Primary: test IWAE bound with K = 64** on `log p(x)` (nats per test point, area-weighted mean
  over the test subjects' directions; exact log q for TNBBeta; float64 on CPU). Higher is better;
  the comparisons below use `tnbbeta` minus the comparator.
* **Confirmatory family (4 tests):** one-sided paired Wilcoxon over the 10 splits, H1: `tnbbeta`
  has a higher bound than `vmf`, and than `power_spherical`, with each decoder; Holm at 0.05 over
  the four (smallest attainable p is 2^-10 = 0.00098, below 0.05/4); margin **0.02 nats** on the
  mean paired difference as well (the registered arm V's descriptive negative-ELBO difference
  between uncollapsed TNBBeta and the collapsed caps was about 0.05; 0.02 is set below it so the
  rule is not trivially met by collapse alone). The 10 splits share subjects, so the p-values are
  optimistic and effect sizes with ranges over splits are reported.
* **"TNBBeta improves a collapse-controlled ITD VAE"** iff all four tests survive Holm and the
  margin holds. All runs are analysed (none excluded).
* **Collapse accounting (descriptive, headline for interpretation):** a run is collapsed if its
  learned sigma exceeds 0.5 at the end of training; the number of collapsed runs per family and
  decoder. **Trainability versus capacity:** the same four comparisons restricted to splits where
  both runs of a pair are uncollapsed, reported with their `n` and p-values but NOT part of the
  Holm family (a descriptive check of whether a difference survives without collapse).
* **Descriptive, per arm and split:** the exact `KL(q || p_model)` (continuity with the first
  analysis); the negative ELBO; learned sigma; for `fixed_axis`, the label-based latitude check
  against the true `y . z` (central 50% and 95% coverage of the true latitude under `q`'s
  latitude marginal with 4000 draws per test point, median absolute error), and the median
  `|cos(mu, y)|`; for `learned_axis`, `|cos(u, +/- y)|` and `|cos(mu, u)|`;
  `tnbbeta` against `tnbbeta_init0`; `learned_axis` against `fixed_axis`.

## Known risks recorded in advance

* The caps may still collapse despite the warm-up and the small initial sigma; that is then a
  trainability result for vMF/PS, reported as such, and the bound comparison (all runs) would be
  decided by it. The restricted comparison is the guard against reading that as capacity.
* `fixed_axis` presumes the interaural axis; it is informative about posterior shape in the
  physical frame but is not an unsupervised discovery of the axis.
* A higher bound is not a better ring: a TNBBeta posterior can improve the bound for reasons other
  than the ring; the latitude check and the restricted comparison are descriptive guards.
* The decisions in this follow-up (the warm-up length, the initial sigma, the margin) were chosen
  after the first analysis and are not tuned; no value other than the stated ones is run.

## Deviations

Appended 2026-11-01 before any full run of this follow-up (only a 3-epoch smoke test on the login
node had been executed); none of the registered quantities above is changed. These are choices the
note left open, made minimally:

1. **Signed axis in the `fixed_axis` latitude check.** `u = +y` is exact in the decoder. The
   label-based latitude check (like the registered arm V's) is taken about the *signed* axis
   `s * y`, with `s = +/-1` chosen so that `s * y . z` correlates positively with the ITD on the
   TRAINING subjects (labels never enter training; `apps.synthetic.hrtf_itd._axis_sign`,
   reused unchanged). Without it, a decoder `g` that learns a decreasing function of `y . z`
   would be scored against a flipped axis. `|cos(mu, y)|` is sign-free. The sign is stored in
   each run's JSON (`axis_sign`).
2. **Diverged runs.** A run with non-finite losses, weights or bound is flagged `diverged` and
   counts as `-inf` on the bound (worst) in all tests (none are excluded, per the rule "all runs
   are analysed"); a diverged run is skipped by the evaluation (no other metrics). None is
   expected.
3. **Prior and proposal of the bound.** The bound uses the head's own `q(z|x)` as proposal with
   its exact `log q` (`sample_with_log_prob`, non-reparameterised draws, through the library's
   `importance_weighted_metrics`) and `standard_prior(family, 3)` (the Uniform(S^2) prior;
   TNBBeta's own uniform-prior object for TNBBeta), K = 64, one pass per test point. The
   per-split value is the area-weighted mean over the test points. The "negative ELBO" reported
   is `-(E_q log p(x|z) - KL(q || prior))` from the same call (analytic KL for vMF/PS,
   64-sample mean of `log q - log p` for TNBBeta) and is a test-set quantity, unlike the
   training loss stored in the JSON.
4. **Wilcoxon details.** `scipy.stats.wilcoxon(diff, alternative="greater")` (default zero
   handling) is the reported p; the sign-pattern enumeration (midranks, zero differences dropped)
   is the independent check, as in the registered analysis. If every difference is zero both give
   p = 1; the restricted comparison is skipped for a pair with no uncollapsed pair of runs.
5. **Seeds and pairing.** Per split `s`: `torch.manual_seed(s)` before building the model, the
   area-weighted sampler seed `200 + s`, evaluation seed 31337, exactly as in the registered
   study, so a split's four families and two decoders share the batch stream. The `fixed_axis`
   model draws the same initial random numbers as `learned_axis` (the learned `raw_u` is created,
   then replaced by the fixed `+y` buffer), so encoder, head and `g` start identically in the two
   decoders.
6. **Descriptive diagnostics kept from the registered study:** exact `KL(q || p_model)` with 1000
   draws and 4001 quadrature nodes, label-based latitude check with 4000 draws per test point.
7. **Compute layout.** 80 cells in a 10-task SLURM array with 8 worker processes sharing a GPU per
   task (`scripts/slurm/hrtf_itd_armv_followup.sbatch`); no effect on any result.
8. **Post-hoc addition after the results were seen (not registered).** In the `fixed_axis` runs the
   registered-style latitude check scored badly in many splits for TNBBeta (and vMF/PS) even though
   `corr(median latitude of q, true y . z)` is about +/-0.96: the latent was mirrored, because the
   sign of `g` (and hence of the latent latitude) is arbitrary, and the sign rule of item 1 orients
   the *label axis*, not the latent. A label-free fix was added to `summarize` as a clearly separate
   "POST-HOC" table (`oriented_latitude_check`): q's saved latitude quantiles are oriented against
   the observation (flip if the posterior-median latitude correlates negatively with the ITD) and
   the true `w` is flipped accordingly. It does not enter any registered test, the verdict, or the
   collapse accounting; the registered-style table is reported unchanged next to it.
