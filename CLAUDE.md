# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repo.

## What this project is

A research package extending the TNBbeta distribution (Lederman & Schein,
2026, arXiv:2606.11624) -- currently defined only on the unit interval --
to the unit hypersphere, and building VAEs whose latent prior/posterior
uses that extension. See [README.md](README.md) for the project layout.

## Conventions

All code style (formatting, docstrings, imports, line length, type
checking) is defined in [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) and
enforced by tooling, not by convention alone:

```bash
uv run ruff format .   # format
uv run ruff check .    # lint (includes import sorting, Google docstrings)
uv run pyright          # type check
uv run pytest           # tests + coverage
```

Run all four before considering a change done. Pre-commit hooks
(`uv run pre-commit install`) run `ruff format` + re-stage + `ruff check`
automatically on commit, excluding `notebooks/`.

## Package structure notes

- `src/tnbbeta_vae/distributions/`: probability distributions.
  `TNBBetaUnivariate` is a finished, paper-sourced implementation
  (closed-form `log_prob`, reparameterized `rsample`). `TNBBetaSpherical`
  lifts it to the hypersphere via a Householder reflection -- original
  research for this project, not in the source paper. Both are finished,
  tested implementations; treat changes to either as touching validated
  math, not a stub to fill in.
- `src/tnbbeta_vae/registry.py`: models register via
  `@register_model(name, config_cls=SomePydanticConfig)`. New models
  should follow this pattern rather than being wired up ad hoc. Importing
  `tnbbeta_vae.models` runs every model module's decorator (see that
  package's `__init__.py`), so anything that needs the registry populated
  (e.g. `apps/train/main.py`) must import `tnbbeta_vae.models`, not just
  `tnbbeta_vae.registry`.
- `src/tnbbeta_vae/models/`: `conv_vae.py`'s `ConvTNBBetaSphericalVAE` is
  the first concrete model -- a simple conv encoder/decoder with a
  `TNBBetaSpherical` posterior/prior, trained via
  `models/losses/elbo.py`'s `monte_carlo_elbo` (a generic Monte Carlo
  ELBO averaged over `num_samples` draws, not a closed-form KL --
  TNBBetaSpherical has none, for the same reason von Mises-Fisher's KL
  between differing mean directions doesn't reduce to one). `p -> 0, q ->
  1` is a known posterior-collapse failure mode in this parameterization
  (the point mass lands wherever the prior already is, independent of
  `x`) -- `models/diagnostics.py`'s `tnbbeta_spherical_posterior_diagnostics`
  logs the statistics to watch for it, and `tnbbeta_vae.data.gaussian_blob_batch`
  is a synthetic dataset (known hue/position factors) for testing against
  it locally before touching real data. `conv_gaussian_vae.py`'s
  `ConvGaussianVAE` is a Gaussian baseline on the same encoder/decoder
  (closed-form KL via `monte_carlo_elbo(..., analytic_kl=True)`) for
  separating latent-family effects from architecture/data effects.
  The TNBBeta prior is always Uniform(sphere), derived from `latent_dim`.
  `conv_vmf_vae.py`'s `ConvVonMisesFisherVAE` (the S-VAE baseline, analytic KL to the
  uniform sphere) was restored from the git tag `vmf-baseline-archive` for the MNIST
  reproduction of the S-VAE paper (Davidson et al. 2018); it is otherwise a baseline,
  not a place for new knobs. All three conv VAEs share `posterior_and_prior` and
  `log_likelihood`, which `models/losses/importance_weighted.py` uses to compute the
  paper's Table 1 metrics. `apps/eval/confidence_probe.py` (k-NN from the posterior's
  confidence scalar alone -- p, kappa or mean std -- no direction) and
  `apps/eval/svae_latitude.py` (TNBBeta-only p-vs-q "cap vs ring" diagnostic; a no-op for
  other models) probe how each family's extra parameters differ geometrically, once the
  aggregate metrics alone stop distinguishing them. On MNIST, TNBBeta's `p` saturates near
  1 with `q` at its floor (the `q=0` special case: a cap at the mean direction, same shape
  vMF always has), so `p` carries almost no class signal by construction; `epsilon` (the
  cap's thickness there) is the parameter actually analogous to vMF's kappa, and
  `confidence_probe.py --param epsilon` probes that instead.
- Sphere models' `latent_dim` is the AMBIENT dimension (S^(latent_dim - 1) in R^latent_dim).
  The S-VAE paper's "d" is the manifold dimension: its d=2 S-VAE is S^2 in R^3 (Figure 2 shows
  a Hammer projection of S^2, and the reference code trains the vMF model with `z_dim + 1`).
  To compare with the paper at the same d as the Gaussian, give the sphere models
  `latent_dim = d + 1` (the sweep's `vmfs`/`tnbs`/`vmfks` models); at equal `latent_dim` the
  sphere has one degree of freedom fewer than the Gaussian. The vMF head also needs a
  dimension-aware `initial_kappa` (about the ambient dimension) at high d, or it collapses.
- MNIST (`--dataset mnist`) uses a Bernoulli likelihood on dynamically binarized images
  (`likelihood="bernoulli"`, set automatically), 28x28 padded to 32 inside the conv
  encoder/decoder, a 50k/10k train/val split, per-epoch validation, KL warm-up and early
  stopping (`Trainer`); `final.pt` is then the best-validation epoch, not the last.
- `models/mlp_vae.py` (vectors), `models/graph_vae.py` (link prediction) and
  `models/semi_supervised.py` (M1+M2) take a latent `family` of `gaussian`, `vmf`,
  `power_spherical` or `tnbbeta`; the posterior heads, priors and centres live in
  `models/heads.py` (`posterior_from_raw`, `standard_prior`, `posterior_centre`), so a
  new family or model should reuse them instead of copying. Entry points pick their
  device with `tnbbeta_vae.training.select_device`, which exits with code 75 on a
  GPU-less node so the sbatch scripts can resubmit (`scripts/slurm/no_gpu_retry.sh`).
- `models/graph_vae.py`'s `GraphVAE` decoder is *just* the dot product (`link_logits`,
  factored into the stateless `models/pairwise.py::pairwise_logits` so other models can
  reuse the same scoring formula): `sigmoid(temperature * (z_i . z_j))` for the sphere
  families (`temperature` a single learned scalar shared across every node -- the only
  per-node substitute for the free per-node norm an unconstrained Gaussian embedding
  gets from an unbounded dot product; see the bimodality investigation below), plain
  `z_i . z_j` for Gaussian. `GraphVAEConfig.feature_reconstruction_weight` (default
  `0.0`, meaning no decoder exists at all) is an ablation knob, not a feature to build
  on casually: it was added specifically to test whether adding a reconstruction term
  changes TNBBeta's posterior shape, and its decoder submodule is constructed only when
  the weight is `> 0`, so default-config checkpoints keep their exact pre-existing
  `state_dict` shape -- do not change that invariant without re-checking it against
  real cluster checkpoints (`tests/models/test_graph_vae.py`'s state_dict-key
  regression test exists exactly for this). `apps/link_prediction/main.py`'s
  `run_once` similarly has an additive-only `encoder_adjacency` override (lets the
  GCN's aggregation input differ from the real training graph, for isolating
  aggregation itself) -- same "default must reproduce prior behavior exactly" rule.
- `models/posterior_stats.py::posterior_stats` (entropy/concentration for any family;
  TNBBeta's own `p`/`q`/`epsilon`/`m = epsilon - (latent_dim-1)/2`/`frac_bimodal`,
  where `m < 0` is the proven-necessary-and-sufficient bimodality condition from
  `tnbbeta_vs_power_spherical_expressivity.md`) is shared by
  `apps/eval/dblp_bridge_diagnostic.py` (bridge-node-specific) and
  `apps/eval/graph_posterior_shape.py` (whole-graph, dataset-agnostic -- works on any
  `GraphVAE` checkpoint, Planetoid/SNAP community/MAG co-authorship alike). Reuse this
  rather than recomputing p/q/epsilon/m a third time.
- The Gaussian likelihood scale sigma is always learned (`LearnedLikelihoodScale`, log
  sigma^2); `likelihood_scale` in each model config is only its starting value. Do not
  add a fixed-sigma option back: it made results depend on a hand-picked number.
- `tnbbeta_vae.data.axial_mixture` (and `apps/synthetic/axial_recovery.py`) is a
  sibling of `circle_mixture.py`/`circle_recovery.py`: a synthetic task where the true
  angle is drawn from a 50/50 *antipodal* von Mises mixture, observed through an
  embedding exactly invariant under `phi -> phi + pi`, so the Bayes-optimal posterior
  is provably bimodal -- a minimal proof-of-concept for `TNBBetaSpherical`'s
  antipodal-bimodality capability
  (`docs/superpowers/specs/2026-10-07-axial-bimodality-datasets-design.md`). Free
  training does not find the bimodal regime on it (same collapse as MNIST); forcing it
  via `MlpVAEConfig.fixed_epsilon` (mirrors `GraphVAEConfig`'s `fixed_temperature`
  pattern) doesn't help either, since the task's decoder is exactly sign-invariant and
  rewards no reconstruction benefit for splitting mass across both modes --
  see `writeup/results/axial_mixture_recovery_2026-10-07.md` and its `latent_dim=5`
  follow-up.
- `tnbbeta_vae.data.dtd` (DTD oriented textures -- `apps/data/download_dtd.py`,
  `apps/data/curate_dtd_categories.py`, `apps/eval/dtd_orientation_probe.py`) is the
  real-data counterpart to `axial_mixture`: texture categories with a genuine,
  no-head/tail dominant line orientation (defined mod `pi`). Orientation/coherence
  ground truth for evaluation only (never training) comes from
  `tnbbeta_vae.data.orientation.structure_tensor_orientation`, a general structure-
  tensor function, not DTD-specific. `ORIENTED_CATEGORIES` is deliberately the full,
  *uncurated* candidate list -- `curate_dtd_categories.py` is a tool for whoever runs
  it (on real downloaded data, e.g. on the cluster) to pick the final subset from real
  measured coherence, not something decided in source ahead of time.
- **The TNBBeta-bimodality investigation**
  (`writeup/results/graph_bimodality_investigation_summary_2026-10-08.md` is the
  narrative entry point; read it before touching any of the modules below).
  `TNBBetaSpherical`'s bimodality (`m < 0`) shows up reliably on real `GraphVAE`
  link-prediction runs (Cora/Citeseer/Pubmed/com-DBLP) and not on any reconstruction
  task tried. Three synthetic ablations (each its own dated `writeup/results/` entry)
  isolate the mechanism: `models/pairwise_mlp_vae.py` + `data/cluster_mixture.py`
  (the exact `GraphVAE` loss, no graph, plain MLP -- stays unimodal);
  `GraphVAEConfig.feature_reconstruction_weight` on real Cora (adding reconstruction
  back -- barely moves it); `data/stochastic_block_model.py` +
  `apps/synthetic/sbm_recovery.py` (a synthetic graph through `GraphVAE`'s real,
  unmodified GCN -- reproduces bimodality, *more* so than real Cora, at 1/634th
  com-DBLP's scale). Triangulated conclusion: GCN aggregation over community-
  correlated structure is sufficient, independent of scale/features/decoder presence;
  the exact mechanistic "why" remains open. `tnbbeta_vae.data.mag_coauthor` (the NOCD
  paper's Microsoft Academic Graph co-authorship networks, `mag_cs`/`mag_eng`/
  `mag_chem`/`mag_med` -- genuine overlapping research-area communities *with* real
  keyword features, unlike com-DBLP) and `scripts/slurm/link_prediction.sbatch`'s
  per-family `d`/`d+1` dimension sweep are the next, broader empirical step, not yet
  run (`writeup/results/dimension_sweep_and_mag_coauthor_infra_2026-10-09.md` is
  infrastructure-only, no results yet).
- `src/tnbbeta_vae/training/`: `Trainer` is a minimal, model-agnostic
  epoch loop -- model-specific logic belongs in the model's
  `training_step`, not in `Trainer`. It also writes/reads checkpoints
  (`latest.pt`, `final.pt`) so preempted cluster jobs can resume.
- `src/tnbbeta_vae/paths.py`: data/checkpoint/runs directories come from
  `.env` (see `.env.sample`); never hard-code paths in scripts.
- `scripts/slurm/`: `sbatch` scripts for the UChicago DSI cluster (see the
  README); they call `apps/train`, `apps/eval` and `apps/data`.
- `apps/` holds CLI scripts (not part of the installed package); library
  code belongs in `src/tnbbeta_vae/`.
- `writeup/results/`: dated per-experiment write-ups,
  `<descriptive_topic>_<YYYY-MM-DD>.md`. Unlike the other `writeup/`
  subdirectories, these ARE committed to git, not gitignored and not local
  scratch -- the project wants a transparent, permanent chain of every
  experiment's results, not just the ones that made it into a weekly
  summary. Every result that gets generated belongs here (not left in a
  stray location, not only pasted into chat), and every one must record,
  near the top: the date, the exact command(s) used to produce the result
  (training and/or table/eval command), and the `master` commit the code was
  run against (or the relevant feature-branch commit/PR if `master` didn't
  yet have the code) -- a result without its code version attached can't be
  reproduced or correctly attributed later.
- `notebooks/` is excluded from ruff/pyright/pre-commit -- don't hold it
  to the same style standard as `src/`.
- Private (underscore-prefixed) helper functions/methods go after the
  public API in their module/class, not before -- see
  [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md#code-organization-private-helpers-go-at-the-bottom)
  for the one exception (definitions referenced at class-definition time).

## Branch/PR policy

`master` is protected: no direct pushes, and PRs require the `test`,
`ruff`, and `pyright` CI checks (`.github/workflows/ci.yml`) to pass
before merging.
