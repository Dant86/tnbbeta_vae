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
  `conv_vmf_vae.py`'s `ConvVonMisesFisherVAE` is a hyperspherical
  baseline using a port of the original S-VAE vMF (Wood rejection
  sampler, Bessel-function KL via `scipy`); keep it faithful to that
  reference rather than "improving" it, since it is the comparison point.
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
