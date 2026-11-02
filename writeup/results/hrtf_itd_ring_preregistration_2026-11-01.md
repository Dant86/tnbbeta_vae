# ITD-only HRTF task (design 1 on real data): pre-registration

Date: 2026-11-01. Base commit: `667cfe1`. FROZEN before any model is trained on these data;
edited afterwards only by appending a "Deviations" section. Supersedes
`hrtf_itd_ring_preregistration_DRAFT_2026-11-01.md` (changes listed at the end). Results go in
`hrtf_itd_ring_results_<date>.md`. Feasibility gate (already computed, passed):
`hrtf_itd_gate_2026-11-01.md`.

## Question

Given only the scalar ITD of a measured head-related impulse response, the source direction lies
(approximately) on a latitude ring about the interaural axis. On real measurements (subject
variation, head asymmetry), (1) does TNBBeta beat vMF and PowerSpherical, and (2) does a VAE
trained without direction labels get there?

## Data

SADIE II (Zenodo record 10886409, Apache-2.0; cite DOI 10.3390/app8112029): the 18 human subjects
H3-H20, 96 kHz SimpleFreeFieldHRIR files (2818 directions each, 1.2 m), downloaded by
`apps.data.download_sadie2`. Dummy heads excluded.
* **Direction `z`**: head-frame unit vector (x front, y left, z up) from azimuth/elevation.
  **Ring axis `a = +y`.**
* **Observation `x`**: scalar ITD in microseconds (right ear minus left ear, positive = left),
  from HRIRs low-passed at 3 kHz (4th-order Butterworth, zero phase), by **cross-correlation lag
  with parabolic interpolation** (as in `apps/eval/hrtf_itd_gate.py`). Standardised by the
  training subjects' mean and sd. No per-subject normalisation. Sensitivity (descriptive): the
  -30 dB onset estimator, which nearly fails the gate (ring better than cap by 0.242 nats; see
  the gate note).
* **Area weights**: spherical-Voronoi cell areas (duplicate pole positions share their cell).
  Training batches are drawn with probability proportional to the weight; every evaluation mean
  below is area-weighted.
* **Preprocessing**: a script (`apps/data/prepare_sadie2_itd.py`, needs `h5py`, run once with
  `uv run --with h5py`) writes the (subject, direction, ITD, weight) table to one `.npz` in the data
  directory; training and evaluation read only that file.
* **Splits**: 10 random subject splits (seeds 0-9): 3 held-out test subjects per split, the other
  15 subjects train. No validation set and no early stopping: the hyperparameters below are fixed
  in advance.

## Models and arms (families identical in both arms)

Families: `vmf`, `power_spherical`, `tnbbeta` (library default: `p_init_offset = ln 3`,
`epsilon_floor = 1e-4`, exact log q), and, descriptive only, `tnbbeta_init0` (`p_init_offset = 0`).
Encoder: MLP `[1, 128, 128]` + the posterior head from `models/heads.py`. Prior: Uniform(S^2).
Training (both arms): Adam lr 1e-3, batch 256, 300 epochs (an epoch is
`n_train / 256` area-weighted batches), 10% linear KL warm-up (arm V), 4 ELBO samples (arm V),
paired by seed (same splits, same data stream, same initial seed per family).

* **Arm V (primary, a VAE).** Unsupervised: direction labels are never used in training.
  Decoder: a single-index decoder `x_hat = g(u . z)` with a learned unit vector `u` (a raw
  3-vector, normalised) and a small 1-D MLP `g` (`[1, 32, 32, 1]`, tanh), Gaussian likelihood with
  learned sigma (`LearnedLikelihoodScale`, the project's convention). Because `x` is scalar and
  `g` depends on `z` only through `u . z`, the model's own exact posterior
  `p_model(z | x) proportional to N(x; g(u . z), sigma^2)` times the uniform prior is a latitude
  ring about `u`, computable by 1-D quadrature in `w' = u . z` (4001 nodes).
* **Arm L (secondary, not a VAE).** The same encoder and heads, no decoder, no prior and no KL:
  trained by minimising `-log q(z_true | x)` in the head frame (area-weighted). Tests posterior
  family capacity; its scale is not comparable with arm V.

## Metrics, tests, decision rules

* **Arm V primary: exact `KL(q || p_model(. | x))`** (nats), mean over the test subjects'
  directions, area-weighted, with 1000 `q` draws and exact log q (the same `exact_kl` machinery as
  the synthetic studies, with the model's own learned `g`, `u`, sigma). This is the inference gap
  of the model the VAE fitted and needs no alignment between the latent and the head frame
  (rotating the latent leaves the VAE unchanged).
* **Arm L primary: held-out mean `-log q(z_true | x)`** (nats, surface measure, area-weighted).
* **Confirmatory family (4 tests):** one-sided paired Wilcoxon signed-rank over the 10 splits, H1:
  `tnbbeta` has a lower primary metric than `vmf`, and than `power_spherical`, in each of arm V and
  arm L; Holm at 0.05 over the four (the smallest attainable p with 10 splits is
  2^-10 = 0.00098, below 0.05/4); margin **0.25 nats** on the mean paired difference as well.
  Because the 18 subjects overlap across splits, the 10 replicates are not independent: the
  p-values are optimistic and effect sizes with the range over splits are reported alongside.
* **"TNBBeta uses its ring capacity" on this task** iff all four tests survive Holm and the margin
  holds. (The earlier total-variation rule is dropped: caps also pass it.)
* **Descriptive, no p-values:**
  * arm V alignment: per split, the median over test x of `|cos(mu(x), u)|` (aligned if >= 0.99) and
    the fraction of aligned splits per family; and whether `u` aligns with `+/- y`
    (`|cos|`, per split);
  * arm V latitude check against the labels: sign of `u` chosen on the training subjects by the sign
    of the correlation of `u . z_true` with `x`, then the central 50% and 95% coverage of the true
    `w' = u . z_true` under `q`'s latitude marginal (4000 draws per test x) and the absolute error
    of the marginal's median;
  * the held-out NLL of the true direction under the oracle ring, oracle cap and uniform baselines
    of the gate note, computed on the same splits, for scale;
  * `tnbbeta` versus `tnbbeta_init0`; the onset-estimator sensitivity run of the whole analysis.

## Known risks recorded in advance

* ITD is not exactly a function of `a . z` (head asymmetry, elevation dependence, subject head
  size): the gate showed a free-form per-bin estimator 0.15 nats better than the oracle ring, so
  arm L's NLL for every family includes structure a ring about `y` cannot represent. The paired
  differences between families are what the tests use.
* The single-index decoder forces rings in arm V; arm V therefore tests whether the family can
  represent and find the ring the decoder implies, not whether the data prefer a ring.
* Mean-direction alignment (arm V: between `mu` and `u`; arm L: `mu` and `+/- y`) failed on half the
  seeds in the single-axis synthetic study and not in the four-class study; the cause of the
  difference is unresolved (`ring_alignment_factorial_proposed_2026-11-01.md`).
* About 0.5% of cross-correlation ITDs exceed 750 us (outliers, not removed).
* Real ITDs depart from the Gaussian-noise model of the decoder; sigma is learned and absorbs
  that, so arm V's KL to `p_model` is not an estimate of the error against the data.

## Changes from the draft

Primary ITD estimator: cross-correlation (draft: onset; changed after the gate, see the gate note);
margin and gate threshold both 0.25 nats; no validation split, fixed 300 epochs and lr 1e-3;
arm V's primary metric is the exact KL to the model's own posterior (the draft's label-based NLL is
not defined for a latent whose azimuth about `u` is arbitrary), with the label-based latitude
coverage kept as a descriptive check; the total-variation ring rule removed; 10 splits with 3 test
subjects each.

## Deviations

Appended on 2026-11-01 while implementing, before any model was trained on these data. Nothing
above is changed; each item is a choice the text left open or a fact found in the data.

1. **Directions per subject.** The gate note and this note say 2818 directions per subject. The
   prepared table has 2818 for H3-H9 and 2114 for H10-H20 (7 x 2818 + 11 x 2114 = 42 980 rows).
   Subjects therefore differ in direction count; an epoch is `n_train / 256` batches with `n_train`
   the split's own number of training rows, so epoch length (and steps) varies by split. Area
   weights sum to 1 within each subject, so every subject has the same total sampling and test weight
   regardless of its grid; the number of rows in the table is not what the weights follow.
2. **Standardisation** uses the *area-weighted* mean and sd of the training subjects (the text says
   only "training mean and sd"); the weighted version matches the training distribution defined by the
   sampler.
3. **Arm V initial values** not given in the text: `LearnedLikelihoodScale` starts at sigma 0.5 (data sd
   is 1 after standardisation); the raw decoder vector `u` starts as a standard-normal draw from the
   run's seed. The decoder's `g` and the encoder use PyTorch's default initialisation.
4. **Exact-KL grid**: exactly 4001 midpoint nodes in `w'` (the minimum the text allows).
5. **Data stream pairing**: the sampler is seeded `200 + split seed`, the model initialiser `split seed`
   and the evaluation draws with fixed seeds, identically for all families and both estimators.
6. **Arm L has no decoder**, so its alignment is `|cos(mu, +/- y)|` only; the label-based latitude
   check (descriptive) is run for arm V as specified, and for arm L it is reported in the head frame
   with `a = +y` (no sign choice needed).
