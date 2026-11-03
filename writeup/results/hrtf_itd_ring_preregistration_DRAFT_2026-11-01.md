# ITD-only HRTF task (design 1 on real data): pre-registration DRAFT

Date: 2026-11-01. STATUS: SUPERSEDED by `hrtf_itd_ring_preregistration_2026-11-01.md` (frozen). Kept for the record.
Original status line: DRAFT, not yet frozen. Nothing had been downloaded or run. Items
marked [OPEN] are decided after the four-class study
(`ring_class_designs_preregistration_2026-11-01.md`) reports, or after the data are in hand and
the feasibility gate below has been computed. Once frozen, this note changes only by an appended
"Deviations" section. Results go in `hrtf_itd_ring_results_<date>.md`.

## Question

An interaural time difference (ITD) depends on a source direction mainly through its angle to
the interaural axis (the "cone of confusion"), so given ITD alone the direction lies on a
latitude ring about that axis. On real measurements, with subject variation and head asymmetry,
(1) is the posterior over direction ring-shaped enough for TNBBeta to beat vMF/PS, and
(2) does a VAE trained without direction labels find it?

## Data (access checked by web search only; nothing downloaded)

* **SADIE II** (University of York; Zenodo record 10886409): HRTFs/BRIRs as SOFA and WAV, 18
  human subjects (H3-H20) plus 2 dummy heads per the SOFA-conventions file list (verify the
  count against the download index), measured on a regular latitude-longitude grid (15 degree
  elevation steps, variable azimuth resolution) plus Ambisonic layouts. Terms: cite the original
  dataset (DOI 10.3390/app8112029).
* **CIPIC** (45 subjects, 25 azimuths x 50 elevations = 1250 directions per ear in release 1.0):
  more subjects, but the original site is down, a GitHub mirror
  (`amini-allight/cipic-hrtf-database`) exists, and the licence is unclear (a 2001 announcement
  says public domain, the mirror lists a UC Regents copyright). [OPEN] use as a replication only
  after the copy's readme licence is read.
* Primary: SADIE II, human subjects only (dummy heads excluded). Other SOFA collections
  (sofacoustics.org/data/database) are possible replications.

## Variables

* **Direction `z`**: unit vector in the head frame (SOFA Cartesian: x front, y left, z up) from
  the measured azimuth/elevation, `z in S^2`. **Ring axis `a = +y` (interaural axis).**
* **Observation `x`**: scalar ITD of the pair (subject, direction), from the 96 kHz HRIRs
  low-passed at 3 kHz. **Primary estimator: the lag maximising the interaural cross-correlation**
  (changed from the first draft's -30 dB onset threshold after the feasibility gate,
  `hrtf_itd_gate_2026-11-01.md`: with this dataset's linear-phase, windowed HRIRs the onset ITD
  is ~500 us at the interaural poles and explains 69% of its variance by lateral angle, against
  ~700 us and 97% for cross-correlation). Sensitivity: the -30 dB onset estimator, which nearly
  fails the gate (ring better than cap by 0.242 nats). `x` is standardised by the training
  mean/std. No per-subject normalisation (that would use other directions of the test subject).
* **Measurement grids are not uniform on S^2.** Training and evaluation use area weights,
  proportional to the spherical-Voronoi cell area of each measured direction, so that the data
  distribution approximates the model's uniform prior.
* **Splits:** by subject (held-out subjects never seen in training), with train / validation
  / test subject sets and 10 random subject splits (seeds 0-9). [OPEN] the exact counts once the
  subject number is confirmed; at least 3 test subjects per split.

## Feasibility gate (computed before any model is trained; RESULT RECORDED, see below)

Non-parametric oracles, using ITD quantile bins (50): (a) an oracle ring: latitude (`a . z`)
density estimated per bin, uniform azimuth about `a`; (b) an oracle cap: per bin, the
maximum-likelihood vMF (mean direction and kappa). Report the cross-fitted held-out NLL of the true
direction for both, and for the uniform sphere (`log 4 pi = 2.531` nats). **Stop** if the oracle
ring is not better than the oracle cap by at least [OPEN: 0.25] nats: the ring is then too thick
for any family difference to be visible. This is the analogue of the Level-1 stop rule.

**Result (`hrtf_itd_gate_2026-11-01.md`, SADIE II, 18 subjects, 6 folds of 3 held-out
subjects):** cross-correlation ITD: ring 0.520 vs cap 1.860 nats (ring better by 1.34, gate
passed), median within-bin sd of `w` 0.085 (a thin ring); onset ITD: ring better by 0.242
(just below 0.25). The free-form per-bin KDE (kappa 100) reaches 0.373, 0.15 nats better than the
oracle ring, so the real conditional is not exactly axisymmetric.

## Arms (identical families in both)

Families: `vmf`, `power_spherical`, `tnbbeta` (library default `p_init_offset = ln 3`),
`tnbbeta_init0` (descriptive).
* **Arm V (primary): unsupervised VAE.** Encoder MLP `[1, 128, 128]` + posterior head from
  `models/heads.py`; uniform-sphere prior; a SINGLE-INDEX decoder `g(u . z)` with a learned unit
  vector `u` (raw 3-vector, normalised) and a small 1-D MLP `g` (`[1, 32, 32, 1]`); Gaussian
  likelihood with learned sigma (`LearnedLikelihoodScale`); 4 ELBO samples with exact log q for
  TNBBeta; direction labels are NOT used in training. Because `x` is scalar, the decoder cannot
  make a cap posterior adequate: the model's posterior lies on level circles of `u . z`.
  Evaluation aligns the latent to the head frame by the decoder axis only: `u` is mapped to
  `+/- y` (sign chosen on the training subjects by the sign of the correlation of `u . z_true`
  with `x`); the azimuth about `y` is unobservable from `x`, so no further rotation is fitted.
* **Arm L (secondary, not a VAE): labelled-direction NLL.** Same encoder and heads, no decoder,
  trained by `-log q(z_true | x)` in the head frame (no prior, no KL). Tests posterior-family
  capacity only; its NLL scale is not comparable to arm V's ELBO.
* [OPEN] epochs, learning rate and early stopping chosen on the validation subjects with the
  same grid for all families, set after the four-class study (its 300-epoch choice is a
  starting point).

## Metrics, tests, decision rules

Primary metric: held-out NLL (nats, w.r.t. the sphere's surface measure, area-weighted) of the
true direction under the fitted posterior, mean over the test subjects of a split.
* Replicates are the 10 subject splits (paired across families). **Confirmatory family (4 tests):**
  one-sided paired Wilcoxon, H1: `tnbbeta` has lower NLL than `vmf`, and than `power_spherical`,
  in each of arm V and arm L; Holm at 0.05 over the four; margin [OPEN: 0.10] nats on the mean
  paired difference.
* Reported alongside (descriptive): the NLL gap of each family to the oracle ring and the oracle
  cap of the gate; calibration of the latitude marginal (coverage of the central 50% and 95%
  intervals of `a . z`); TV between each family's latitude marginal and the empirical conditional
  on ITD bins; for arm V, alignment between the decoder axis `u` and `+/- y`
  (|cos|, per split) and whether TNBBeta's mean direction is aligned with `u` (the failure mode of
  the single-axis synthetic study).
* ITD estimator sensitivity: repeat the primary analysis with the cross-correlation ITD
  (descriptive).

## Known risks recorded in advance

* ITD is not exactly a function of `a . z`: head asymmetry, near-field effects and subject head
  size add noise, thickening the ring (this is what the gate measures).
* Elevation/front-back structure inside the cone is real signal that ITD does not carry; the
  azimuth-about-axis uncertainty is therefore large, and `q`'s NLL includes `log(2 pi)`-scale
  azimuth entropy that is identical across families and cancels in paired differences.
* Few subjects (18 humans in SADIE II): subject splits overlap across the 10 replicates, so the
  replicates are not independent; treat p-values as optimistic and report effect sizes with
  intervals over splits.
* Mean-direction alignment may fail on some seeds, as in the synthetic study.
* The `tnbbeta` default init (ln 3 on raw p) was adopted from the graph study and did not
  reliably help in the synthetic ring task.
