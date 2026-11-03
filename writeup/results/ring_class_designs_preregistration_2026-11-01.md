# Four-class ring designs on S^2: pre-registration

Date: 2026-11-01. Base commit: `6e2e00d` (HEAD) plus the uncommitted working tree, which
includes the small-epsilon distribution fix, the exact-log-q ELBO paths, the `p_init_offset`
head init and `epsilon_floor`. Written BEFORE any run of this study; edited afterwards only by
appending a "Deviations" section. Results go in `ring_class_designs_results_2026-11-01.md`.
Illustration of the two designs: `ring_class_designs_illustration_2026-11-01.md`.

## Question

On the earlier single-axis ring task (`ring_posterior_l23_d3_2026-11-01.md`,
`ring_posterior_strata_2026-11-01.md`) TNBBeta recovered single-ring posteriors almost exactly
(KL 0.12 against 2.85 for vMF) on the 5 of 10 seeds whose mean direction aligned with the axis,
and stayed near cap level on the other 5; about 19% of test x had a second latitude mode that
no single-ring family can represent. Do four-class versions, with a single-mode posterior,
(1) show TNBBeta beating vMF and PS, and (2) show it reliably, across seeds and classes?

## Generative models (d = 3, ambient R^3, sigma = 0.2 throughout)

Common: `h(w) = [w, sin(1.5 w), cos(2 w)]` in R^3, `x = h(a . z) + N(0, sigma^2 I_3)`, prior
`z ~ Uniform(S^2)` (so the latitude `w = a . z` is uniform on [-1, 1]). The exact posterior
given `x` and the axis is a latitude ring about `a`, with density in `w` proportional to
`exp(-||x - h(w)||^2 / (2 sigma^2))` and a uniform azimuth about `a`; `log p(x)` and the exact
KL come from a 1-D quadrature in `w` (>= 4001 nodes). With this `h` no test x has a second
latitude mode with more than 5% mass (0.000 of 2000, checked by the illustration script).

* **Design 1 (shared axis).** One axis `a = e_1` (the pole `models/heads.py::_fixed_pole` uses, so the oracle-mean arm below sees the same data stream). `z ~ Uniform(S^2)`; the class is the latitude
  band of `w` (edges -1, -0.5, 0, 0.5, 1; four equal-mass classes). The class is a function of
  `z`, NOT an input to the model. The exact posterior given `x` is the ring above.
* **Design 2 (one axis per class).** Axes `a_c` = the tetrahedron vertices
  `(1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)` / sqrt(3). `c ~ Uniform{1..4}`, then
  `z ~ Uniform(S^2)` independent of `c` (so `p(z | c)` is uniform and the aggregate prior is
  exactly uniform), `x = h(a_c . z) + noise`. The class IS an input: the one-hot `c` is
  concatenated to `x` at the encoder input and selects the axis in the (frozen, true) decoder,
  so the exact posterior `p(z | x, c)` is a ring about `a_c`. (The latent-class alternative needs
  a mixture likelihood and is left to a later study.)

## Models and arms

Encoder: MLP `[in, 128, 128]` + linear posterior head from `models/heads.py` (`in` = 3 for design
1, 7 for design 2), TNBBeta heads built with the library defaults. Decoder: frozen true
decoder, sigma frozen at 0.2 (no free decoder parameters), the project's prior (Uniform sphere).
Condition B (learned decoder) is NOT run: with a learned decoder the model can choose a
decoder under which cap posteriors suffice (`ring_condition_b_unpacked_2026-11-01.md`), so it is
not a ring-recovery test.

Arms (10 seeds, 0-9, paired by seed and data stream):
* Primary: `vmf`, `power_spherical`, `tnbbeta` (library default: `p_init_offset = ln 3`,
  `epsilon_floor = 1e-4`).
* Descriptive (no p-values): `tnbbeta_init0` (`p_init_offset = 0`); design 1 only,
  `tnbbeta_fixed_mu` (mean direction fixed to the pole `e_1`, which is the design-1 axis,
  via `fixed_mean_direction`; p/q/eps free), which isolates whether mean-direction alignment is the
  failure mode.

Training: Adam lr 1e-3, batch 256, 300 epochs (the 100-epoch runs of the earlier study were
still improving), 10% linear KL warm-up, 4 ELBO samples (exact log q for TNBBeta via the
patched sampler), 20,000 fresh training x per seed (generator seed `100 + seed`; design 2
also draws the class), test set 2000 x (seed 424242; design 2 balanced 500 per class).
Evaluation: exact `KL(q || p_true)` per test x with 1000 q draws (float64 posterior,
`exact_kl`), per-x results saved (x, class, true latitude, KL, fitted parameters, mu angle to
the class axis) so every stratified table can be recomputed.

## Metrics, tests, decision rules

Primary metric: mean exact test KL over the 2000 test x, per (design, family, seed).
* **Confirmatory family (4 tests):** one-sided paired Wilcoxon signed-rank over the 10 seeds,
  H1: `tnbbeta` has lower KL than `vmf`, and than `power_spherical`, in each design; Holm at
  0.05 over the four (smallest attainable p with 10 seeds is 2^-10 = 0.00098, below 0.05/4).
  Margin X3 = 0.25 nats on the mean paired difference as well.
* **"TNBBeta uses its ring capacity" in a design** iff both of its Holm tests survive, the
  margin holds, AND for at least 8 of the 10 seeds at least half of the test x have a fitted
  latitude marginal within total variation 0.25 of the exact one (a pre-chosen, deliberately
  strict threshold; the earlier study could not assess it because TV was not stored).
* **Alignment (descriptive, reported per seed and per class):** |cos| between the fitted mean
  direction and the class axis, median over test x; a seed counts as aligned if the median is
  >= 0.99. The fraction of aligned seeds per design and arm is the headline diagnostic for
  the five-of-ten pattern.
* Stratified tables: mean KL by class (design 1: by band; design 2: by class), by arm.
* Exploratory (no p-values): `tnbbeta` vs `tnbbeta_init0`; `tnbbeta_fixed_mu` vs `tnbbeta` in
  design 1; if `tnbbeta_fixed_mu` is far better than `tnbbeta`, that is evidence alignment, not
  representation, limits the free model.

Not tested here: Level 1 (representation) and Level 2 (per-datum optimisation); d > 3; a latent
class; a learned decoder; a class-conditional prior.

## Known risks recorded in advance

* The mean direction must reach `+/- a` (design 1) or `+/- a_c` per class (design 2); the
  single-axis study showed this fails on half the seeds. Design 2 needs four alignments from
  one network and is expected to be harder.
* With sigma = 0.2 and this `h` the rings are bands of about 10 degrees, so neighbouring
  design-1 classes overlap in latitude; the class label is a function of `z`, not of `x`.
* Both `vmf` and `power_spherical` caps already do well near the poles; only the thickness of
  the ring distinguishes the families elsewhere.
* The palette used in the figures is unvalidated (no `node` for the validator).

## Deviations

Appended 2026-11-01 BEFORE any full-length run of the study had finished (only 3-10 epoch
smoke runs of the code, whose results were discarded, had been run). These are choices
for points the text above leaves unspecified; none changes a generative model, arm, test or
decision rule.

1. **Design 2 has no class offset.** The illustration note describes design 2 with a class
   offset `o_c`; the specification above (class one-hot as an encoder input, decoder
   `h(a_c . z)`) has none, and none is used.
2. **Evaluation precision and sample sets.** Models are trained in float32 on the GPU and
   evaluated on the CPU in float64 (the "float64 posterior" above). The design-1 test set is the
   2000 x drawn with generator seed 424242 as they fall (not class-balanced; band counts are
   recorded); the design-2 test set is class `i mod 4` for `i = 0..1999` (500 per class) with `z`
   and noise from the same seed.
3. **Latitude-marginal TV.** TV is between the exact latitude (polar-angle) marginal and a
   90-bin histogram of 8000 draws from the fitted `q`, separate from the 1000 draws used for
   the KL (histogram noise on an exactly correct `q` is about 0.05 in TV, well under 0.25).
   The "alignment" statistic uses the fitted mean direction itself (for TNBBeta this is
   `mean_direction`, with no p < 0.5 flip), `|cos|` against the class axis.
4. **Arm details the text leaves open.** `tnbbeta_fixed_mu` keeps the library default
   `p_init_offset = ln 3` (its head has no direction outputs; p/q/epsilon free);
   `tnbbeta_init0` is `p_init_offset = 0.0`, which leaves the plain random `nn.Linear` bias
   in place. `epsilon_floor = 1e-4` for every TNBBeta arm. `MlpVAE` has no
   `fixed_mean_direction`, so all arms use a study-local wrapper
   (`apps/synthetic/ring_classes.py::RingClassVAE`) built from `models/heads.py`; the
   loss is `monte_carlo_elbo` with 4 samples and the same KL choices as `MlpVAE` (analytic for
   vMF / Power Spherical, Monte Carlo with exact `log q` for TNBBeta). Model initialisation uses
   `torch.manual_seed(seed)`, the shuffle uses seed `200 + seed`; the training data stream
   (seed `100 + seed`) is shared by every arm of a seed.
5. **Test implementation.** The one-sided Wilcoxon is `scipy.stats.wilcoxon(diff,
   alternative="greater")` on `KL(comparator) - KL(tnbbeta)` per seed, cross-checked against a
   direct enumeration of the `2^n` sign patterns (the two must agree). Holm is applied to the
   sorted p-values with thresholds `0.05 / (4 - rank)` and stops at the first failure.
6. **Second-mode check** is run on 20,000 draws of x per design in
   `tests/data/test_ring_classes.py` (not only 2000), with the illustration script's definition
   (segments between local minima of the density in `w`, second-largest segment mass).
7. **Run layout.** 90 runs, one array of 23 tasks with 4 worker processes per task sharing a GPU
   (`scripts/slurm/ring_classes.sbatch`), outputs under `ring_recovery/classes/`.
