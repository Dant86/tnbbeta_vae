# Demonstrating TNBBetaSpherical's Bimodality: Two Datasets

- **Date:** 2026-10-07
- **Branch:** `research/sphere-diffusion-prior`
- **Status:** design, not yet implemented

## Motivation

`TNBBetaSpherical` can represent a shape no other latent family in this
project can: simultaneous concentration at a mean direction **and** its
antipode (and, for suitable parameters, a third "ring" mode between them).
Nothing in the project currently exercises this. MNIST training so far pins
`epsilon` at or above the value that makes multimodality impossible
(`writeup/weekly_markdown_summaries/week_2/tnbbeta_vs_power_spherical_expressivity.md`
§7(i)), so the capability has only ever been demonstrated on static
parameter grids, never learned by an encoder from data that actually needs
it.

This spec designs two datasets to close that gap, in order:

1. **`axial_mixture`** (synthetic): a minimal, provably-correct
   proof-of-concept where the Bayes-optimal posterior is *exactly* the
   bimodal shape `TNBBetaSpherical` can represent and no other family here
   can. Its job is to answer one question cheaply: can training actually
   find and use that capability, or does it stay theoretical?
2. **DTD oriented textures** (real): only attempted once (1) gives a
   satisfying result. A real-world analogue -- texture images whose
   dominant line orientation is axial (defined mod `pi`, no head/tail) --
   run through the project's existing conv-VAE pipeline unchanged.

Both reuse existing infrastructure as much as possible; neither needs new
model or architecture code.

## Background: the exact bimodality condition

From the expressivity write-up (proved, not numerical): writing
`m = epsilon - (latent_dim - 1) / 2`,

- `m > 0`: `TNBBetaSpherical` is a unimodal cap at `mean_direction` (density
  vanishes at both poles).
- `m = 0`: still unimodal; this is exactly where `uniform_prior_params`
  (the prior used everywhere in this project) sits, and where the posterior
  head has been empirically found to saturate on MNIST.
- `m < 0`: bimodal at `{mean_direction, -mean_direction}` for every `p`,
  `q`; for suitable `(p, q)` a third interior "ring" mode appears too.
  `p` controls the *balance* between the two antipodal lobes (`p = 0.5` is
  the symmetric case, equal mass at both poles); `p != 0.5` gives a large
  cap at one pole and a small spike at the other.

`PowerSpherical` is proved strictly unimodal on the sphere for *every*
`kappa` and `d >= 2` (Theorem 6.1) -- a provable, not just empirical,
baseline failure mode. `vMF` and a Gaussian-VAE posterior are unimodal for
the same structural reason (both are monotone functions of a single
direction's cosine similarity).

**Implication for both datasets below:** the posterior head's `epsilon`
must actually be free to go below `(latent_dim - 1) / 2`, and `m` (not raw
`epsilon`) is the quantity that should be logged and checked. Neither
`models/diagnostics.py`'s `tnbbeta_spherical_posterior_diagnostics` nor
`apps/eval/svae_latitude.py` currently compute `m`; both datasets' eval
scripts add it rather than touching the shared diagnostics module (keeping
the change scoped to the new scripts, not the training-time logging path
every other model also uses).

## Part 1: `axial_mixture` (synthetic)

### Generative model

A direct sibling of `src/tnbbeta_vae/data/circle_mixture.py`, reusing its
embedding network rather than duplicating it.

1. Draw a component `c ~ Bernoulli(0.5)` and angle
   `phi ~ VonMises(c * pi, kappa)` (`kappa = 20`, matching
   `circle_mixture.py`'s concentration) -- i.e. a 50/50 mixture of a von
   Mises at `0` and one at `pi`. By construction this *generative* process
   is itself invariant under `phi -> phi + pi`: both the component at `0`
   and the one at `pi` are equally likely a priori, with identical shape.
   This is what makes the Bayes-optimal posterior over `phi` given an
   observation *exactly* symmetric-bimodal, not merely "hard to see" -- it
   is not a continuity/branch-cut argument about one ambiguous seam, it
   holds at every sample.
2. Embed via `CircleMixtureData.embed(2 * phi mod 2*pi)` -- the **existing**
   fixed random two-layer tanh network from `circle_mixture.py`, called on
   the doubled angle. Doubling erases the sign because
   `2*phi == 2*(phi + pi) mod 2*pi`, so the embedding (plus the same
   Gaussian observation noise already in `circle_mixture.py`,
   `noise_std = 0.05`) cannot distinguish `phi` from `phi + pi`. This reuses
   `CircleMixtureData.embed` directly; no new embedding code.
3. Observation `x = embed(2*phi) + noise in R^ambient_dim` (default
   `ambient_dim = 100`, matching `circle_mixture.py`'s default).

The dataset module returns `(x, phi, component)` just as
`CircleMixtureData.sample` does, for evaluation use only -- `phi` is never
given to the model.

### Training

`models/mlp_vae.py`, `latent_dim = 2`, across
`family in {gaussian, vmf, power_spherical, tnbbeta}`. Zero new model code.

### Evaluation and success criteria

New app `apps/synthetic/axial_recovery.py`, mirroring
`apps/synthetic/circle_recovery.py`'s structure (same CLI shape, same
JSON + HTML output convention, same `importance_weighted_metrics` usage).
Differences from `circle_recovery.py`:

- Ground truth is `phi` from the antipodal mixture above, not the
  three-component directional mixture.
- `_tnbbeta_diagnostics` adds `m_mean = epsilon_mean - 0.5` (since
  `latent_dim = 2`) alongside the existing `p_mean`/`p_std`/
  `centre_axis_resultant`.
- `angle_error`/`reconstruction_angle_error` are computed mod `pi` (not mod
  `2*pi`), since the task's recoverable signal is the axis, not the sign.

Falsifiable success criteria, in order of importance:

1. **Mechanistic:** `tnbbeta`'s fitted posterior shows `m_mean < 0` and
   `p_mean` near `0.5` on held-out data -- the model actually finds and uses
   the bimodal regime the task calls for, not just a cap.
2. **Comparative:** `tnbbeta`'s `test_ll`/`test_elbo`/`test_kl` (from
   `importance_weighted_metrics`, already used by `circle_recovery.py`) are
   better than `vmf`, `power_spherical`, and `gaussian`'s.
3. **Caveat to report either way, not assume:** it is possible for a
   unimodal family's *decoder* to learn approximate sign-invariance
   (`decoder(z) ~= decoder(-z)`) and partially compensate -- this is a
   legitimate empirical outcome to report, not a failure of the experiment.
   If it happens, criterion 1 (the mechanistic check) is what still
   distinguishes "`tnbbeta` represents the ambiguity directly" from "the
   unimodal families worked around it architecturally."

If criteria 1-2 hold clearly, that is the "satisfying result" that
justifies moving to the real dataset.

## Part 2: DTD oriented textures (real)

Only built once Part 1's result is satisfying.

### Data

[DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) (Describable Textures
Dataset): 5,640 real photographs, 47 categories, 120 images/category, sizes
300x300 to 640x640. Candidate oriented-texture categories (a dominant line
with no head/tail, i.e. genuinely axial): `banded`, `braided`, `cracked`,
`fibrous`, `grooved`, `lined`, `striped`, `veined`, `wrinkled`, `zigzagged`.
The final category subset is decided during implementation by checking
which candidates actually produce a strong structure-tensor coherence
signal (below) -- not fixed in advance by inspection.

### Pipeline

Following the existing `apps/data/download_*.py` + `src/tnbbeta_vae/data/*.py`
convention (`mnist.py`, `cifar10.py`):

- `apps/data/download_dtd.py`: downloads and extracts the official tarball
  once (never at training time, matching every other `download_*.py`).
- `src/tnbbeta_vae/data/dtd.py`: loads images, resizes to a conv-friendly
  square (64x64, divisible by 8 per `ConvTNBBetaSphericalVAEConfig`'s
  requirement), filters to the oriented-category subset, same
  `Dataset` + `labels()` shape as `Cifar10Images` (labels are the texture
  category, for inspection only, not used in training or in the orientation
  evaluation below).
- Training: `apps/train/main.py --model conv_tnbbeta_spherical_vae` /
  `conv_gaussian_vae` / `conv_vmf_vae`, pointed at the new dataset, RGB,
  `likelihood="gaussian"`. No new model or architecture code.

### Evaluation

DTD has no labeled orientation, so ground truth is derived, for evaluation
only, never for training -- the same role `gaussian_blob_batch`'s known
factors play for the synthetic collapse diagnostic, but measured rather
than generated. New app `apps/eval/dtd_orientation_probe.py`, mirroring
`apps/eval/svae_latitude.py`'s scatter-plot structure and
`apps/eval/confidence_probe.py`'s evaluation-script shape:

1. For each test image, compute the dominant line orientation mod `pi` via
   a structure tensor (gradient second-moment matrix; standard technique
   for local orientation fields in image processing) -- this also yields a
   coherence score (how strongly oriented the image actually is; near 0 for
   images with no dominant direction).
2. Table-1-style `test_ll`/`test_elbo`/`test_kl` comparison across
   `{gaussian, vmf, tnbbeta}` via the existing, unchanged
   `importance_weighted_metrics`.
3. A `p` vs. `q` vs. `m` scatter (extending `svae_latitude.py`'s pattern),
   colored by structure-tensor coherence instead of class label. The
   testable prediction, carried over from Part 1: `tnbbeta`'s posteriors
   should go bimodal (`m < 0`, `p` near `0.5`) specifically on the
   strongly-oriented images, and stay capped (`m >= 0`, vMF-like) on weakly
   oriented ones -- the real-data analogue of Part 1's clean synthetic
   result.

### New code

`apps/data/download_dtd.py`, `src/tnbbeta_vae/data/dtd.py`,
`apps/eval/dtd_orientation_probe.py`. No new model/architecture code --
this is new dataset + new diagnostic only, reusing every existing
VAE/training/eval primitive.

## Testing

- `axial_mixture`: there is no existing `test_circle_mixture.py` to mirror
  (only the app, `circle_recovery.py`, is tested) -- a new
  `tests/data/test_axial_mixture.py` checking: the embedding is exactly
  invariant under `phi -> phi + pi` (bitwise via the doubled-angle
  construction, not approximately); the component split is ~50/50 over a
  large sample; `sample()`'s shapes match `circle_mixture_data`'s.
- `axial_recovery.py`: a fast-smoke test (few epochs, tiny `num_train`)
  mirroring the existing `tests/apps/test_circle_recovery.py`, checking the
  script runs end-to-end and writes its JSON/HTML outputs with the
  expected keys (including the new `m_mean`).
- `dtd.py`: a unit test against a tiny fixture (a handful of fake images on
  disk, not the real download) checking resizing, channel count, and
  category filtering, following the existing `tests/data/test_cifar10.py`'s
  pattern.
- `dtd_orientation_probe.py`: a unit test of the structure-tensor
  orientation function alone (e.g. a synthetic striped image at a known
  angle recovers that angle mod `pi` to within a tolerance), independent of
  any trained checkpoint.

## Open risks / explicitly out of scope

- **Trainability, not just expressivity.** The expressivity write-up's
  caveat (iii) applies here too: `m < 0` makes the density unbounded at two
  points, which is fine measure-theoretically but hands the Monte Carlo
  ELBO unbounded log-density terms. If `axial_mixture` training also never
  approaches `m < 0` (as MNIST training has not), that is itself a result
  worth recording, not a bug to silently work around -- see whether the
  `fixed_epsilon` / `fixed_mean_direction` ablations already in
  `ConvTNBBetaSphericalVAEConfig` (and an equivalent, if needed, added to
  `MlpVAE`) help isolate why.
- **DTD category curation is deliberately deferred to implementation,**
  not pre-decided here -- the structure-tensor coherence check is the
  actual filter, and categories may need dropping or adding after seeing
  real coherence scores.
- **No new model/architecture code in either part.** If Part 1's result is
  unsatisfying, the fix is more likely in training dynamics (warm-up
  schedule, epsilon parameterization/initialization) than in a new
  architecture; that investigation is out of scope for this spec and would
  get its own.
- **Writeup:** per `writeup/results/` convention, both parts' results get
  a dated write-up there (command(s) used, the `master`/branch commit),
  not just this spec or chat.
