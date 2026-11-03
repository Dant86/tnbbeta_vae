# Four-class ring designs with a learned decoder: pre-registration

Date: 2026-11-01. Base commit: `945f133` plus the uncommitted tree (the ITD-study agent's files are
unrelated). FROZEN before any run of this study; edited afterwards only by appending a
"Deviations" section. Results go in `ring_class_designs_learned_decoder_results_2026-11-01.md`.
Builds on `ring_class_designs_preregistration_2026-11-01.md` and its results note (frozen true
decoder, where TNBBeta recovered the rings) and on `ring_condition_b_unpacked_2026-11-01.md`
(single-axis learned decoder, where caps closed most of the gap).

## Question

With a LEARNED decoder, the posterior's shape is whatever the learned decoder implies, so a ring
is not forced. (1) On the four-class tasks, does a VAE with a TNBBeta posterior reach a better
bound on the data log-likelihood than vMF and PowerSpherical? (2) What latent structure does each
family learn per class (rings, bands or blobs)? A null answer to (1) is a legitimate result.

## Identifiability (why the design is what it is)

With a learned decoder only the law of `x` is identified, not `z`. (i) In design 2 as first
registered, `x = h(a_c . z) + noise` with `z` uniform gives the same law of `x` for every class
(`w = a_c . z` is uniform), so the four axes are unlearnable and the class is invisible. (ii) With
class-specific observation maps the data fix only the mixture over classes; a single latitude
coordinate split into four equal-mass bands reproduces it exactly, so a learned model may use
bands about ONE axis rather than four axes. Latent rotations and such reparametrisations leave the
ELBO unchanged, so per-class latent figures are drawn in a frame aligned to the learned decoder's
estimated axis, and no metric compares the learned latent with the true `z`.

## Generative models (d = 3, sigma = 0.2, `h(w) = [w, sin 1.5w, cos 2w]`)

* **Setting D1 (design 1):** one axis `a = e_1`, `z` uniform, class = latitude band of `w` (edges
  -1, -0.5, 0, 0.5, 1); the class is not an input anywhere.
* **Setting D2g (design 2', class given):** tetrahedron axes `a_c`; `c` uniform; `z` uniform
  independent of `c`; `x = h(a_c . z) + o_c + noise` with class offsets
  `o_c = 1.5 * [e_1, e_2, e_3, -(1, 1, 1)]` so that the class is identifiable from `x` (this
  changes design 2 of the earlier study, which had no offsets). Encoder input `[x, onehot(c)]`,
  decoder input `[z, onehot(c)]`. Reference: `log p(x | c)` by 1-D quadrature.
* **Setting D2l (design 2', latent class):** the same data; encoder input `x` only, decoder input
  `z` only. Reference: `log p(x) = log( (1/4) sum_c p(x | c) )` by quadrature per class.
* **Class-identifiability check (before any training):** under the true model, the class posterior
  `p(c | x)` must exceed 0.99 for at least 99% of 20,000 test `x`. If not, the offsets are
  multiplied by 1.5 repeatedly until it does (recorded under Deviations with the final scale).

## Models and arms

Encoder: MLP `[in, 128, 128]` + the posterior head of `models/heads.py` (`in` = 3, or 7 with the
one-hot class). Prior: Uniform(S^2). Gaussian likelihood on `R^3` with ONE learned scalar sigma
(`LearnedLikelihoodScale`). Decoders:
* **`mlp`** (the real VAE): MLP `[in_dec, 128, 128, 3]`, ReLU, `in_dec` = 3 (D1, D2l) or 7 (D2g).
* **`single_index`** (learned but ring-forcing): `x_hat = g(u . z, onehot(c))` with a learned unit
  vector `u` (one per class in D2g; one in D1 and D2l) and a small MLP `g` (`[1 (+4), 32, 32, 3]`,
  tanh); descriptive only.
Families: `vmf`, `power_spherical`, `tnbbeta` (library default), `tnbbeta_init0`
(`p_init_offset = 0`, descriptive).
Training: Adam lr 1e-3, batch 256, 300 epochs, 10% linear KL warm-up, 4 ELBO samples (exact log q
for TNBBeta; analytic KL for vMF/PS), 20,000 fresh training `x` per seed (generator seed
`100 + seed`), 10 seeds (0-9), paired by seed (same data stream and decoder/encoder initial seed
across families). Test set: 2000 fixed `x` (seed 424242; design 2' balanced 500 per class).
3 settings x 2 decoders x 4 families x 10 seeds = 240 runs.

## Metrics, tests, decision rules

* **Primary: test ELBO gap** `= mean_x [ log p_ref(x) - ELBO(x) ]` (nats; `p_ref` as above; ELBO from
  256 posterior draws per test `x`, exact log q; float64 on CPU). Lower is better.
* **Check: importance-weighted bound gap** with `K = 64` (the repository's
  `models/losses/importance_weighted.py` machinery), descriptive.
* **Confirmatory family (6 tests):** for the `mlp` decoder, one-sided paired Wilcoxon signed-rank
  over the 10 seeds, H1: `tnbbeta` has a smaller ELBO gap than `vmf`, and than `power_spherical`,
  in each of D1, D2g, D2l; Holm at 0.05 over the six (smallest attainable p with 10 seeds is
  2^-10 = 0.00098, below 0.05/6 = 0.0083); margin **0.05 nats** on the mean paired difference as
  well. **"TNBBeta helps a learned-decoder VAE" in a setting** iff both of its tests survive Holm
  and the margin holds. All other decoder, family and variant comparisons are descriptive.
* **Descriptive diagnostics, per arm and seed:**
  * learned sigma against the true 0.2; training divergences (count of NaN/inf runs);
  * decoder single-indexness (`mlp`): `eta^2_max`, the best fraction of decoder-output variance
    explained by a binned function of `u . z` over axes `u` (maximised from 20 random starts), and
    that best axis `u*`;
  * ring versus cap in the learned posterior: the mean resultant length of `q`'s draws projected
    onto the plane orthogonal to `u*` (about 0 for a ring about `u*`, large for a cap off the
    axis), median over test `x`;
  * latent class structure (D1, D2l): accuracy of a 5-nearest-neighbour classifier of the class
    from the posterior mean direction (train on the training-set posterior means, test on the test
    set), and for D1 the mean `|u* . mu|`;
  * `tnbbeta` against `tnbbeta_init0`; `single_index` against `mlp`.
* **Figures** (plotly, the project's `alptex` template, the style of
  `apps/synthetic/ring_classes_plot.py`): for each setting and decoder, one seed (the median-gap
  `tnbbeta` seed), the posterior means of 1500 test `x` coloured by class on orthographic views of
  `S^2`, rotated so the estimated axis `u*` is the arrow from the origin, one column per family;
  no text under the spheres, legend at the bottom with "Class k", title
  "Synthetic Experiment: Learned-Decoder Latents by Class", 3x resolution.

## Known risks recorded in advance

* The earlier single-axis condition B showed caps within about 0.1 nats of TNBBeta and TNBBeta
  losing on some seeds; a null or negative result here is plausible, not a failure of the setup.
* The reference `p_ref` is the true generative model; a learned model cannot beat it, so gaps are
  non-negative up to estimator noise; negative per-run gaps are investigated, not discarded.
* The single-index decoder forces rings by construction; its results say how well the families
  represent and find that ring, not whether the data need one.
* Latent-class D2l needs the encoder to infer the class from `x` and the decoder to encode it in
  `z`; a failure to separate classes is reported, not repaired.
* Posterior means of ring-shaped posteriors lie near the axis, so a figure of means understates the
  rings; the figure uses posterior means because they are what a downstream use sees. (A figure
  of posterior draws is added for one setting as a check.)

## Deviations

Appended 2026-11-01 BEFORE any run of the study (the identifiability check below is a
pre-registered pre-training check, not a result). Nothing above is changed.

1. **Class-identifiability check passed at the nominal offsets; final offset scale = 1.0.**
   `class_identifiability(offset_ring_class_task(1.0))` on 20,000 fresh `x` (generator seed 7, class
   random, 4001-node quadrature): `p(c | x) > 0.99` for **99.425%** of `x` (>= 99% required), mean
   largest class posterior 0.99936. Offsets stay `o_c = 1.5 * [e_1, e_2, e_3, -(1,1,1)]`, no 1.5x
   rescaling. (For information: scale 1.5 gives 100.000%, 2.25 gives 100.000%; not used.)
2. **Operationalisations of under-specified quantities** (minimal choices):
   * "Posterior mean (direction)" for the figures, the 5-NN probe and `|u* . mu|` is
     `heads.posterior_centre` of each posterior (the mean direction, negated for TNBBeta when
     `p < 0.5`), a unit vector; the 5-NN classifier is fit on the centres of the run's own 20,000
     training `x` and scored on the 2000 test `x` (Euclidean distance, majority vote, ties to the
     lowest class index).
   * `eta^2(u)`: 20,000 uniform `z` on `S^2`, 40 equal-width (hence equal-mass) bins of `u . z`,
     between-bin sum of squares over the total sum of squares of the decoder output (3 outputs);
     maximised over `u` by Nelder-Mead on the two sphere angles from 20 random starts; the sign of
     `u*` is fixed so its largest component is positive. In D2g the decoder is evaluated at each
     class in turn, giving four axes `u*_c` and four `eta^2`; the summary uses their mean, and the
     D2g figures draw all four arrows. The same procedure is applied to the `single_index`
     decoders (descriptive; the cosine between the learned `u` and `u*` is also recorded).
   * Ring statistic: the 256 evaluation draws of `q` per test `x` are projected orthogonal to the
     `u*` of that `x` (class-specific in D2g), renormalised to unit azimuth vectors, and the mean
     resultant length of those vectors is taken; median over test `x`.
   * Evaluation ELBO: reconstruction averaged over 256 draws; KL analytic for vMF/PS, Monte
     Carlo with exact `log q` over the same 256 draws for TNBBeta (as in training). IWAE: the
     repository's `importance_weighted_metrics` with `K = 64` draws, a constant-density stub for the
     uniform-sphere prior's `log_prob`.
   * The learned sigma starts at 0.2; the TNBBeta head's epsilon floor is the library default
     `1e-4`. Encoder and decoder initial weights are identical across families for one seed (the
     posterior head, whose size depends on the family, is built last); the head differs necessarily.
   * Holm uses a fixed family size of six even if a test is dropped for lack of valid seeds
     (a dropped test counts as not rejected).
   * A run whose epoch loss becomes non-finite is stopped, flagged `diverged`, and excluded from
     paired tests (the exclusion and its seeds are reported); runs that finish are never excluded
     (including negative or large gaps, which are investigated and reported).
   * D1 test `x` are drawn at random (band classes are equal-mass); design 2' test `x` are balanced.
   * The extra posterior-draw figure (one setting, as a check) is D1 with the `mlp` decoder.
