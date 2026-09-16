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
  `TNBBetaUnivariate` is a finished, paper-sourced implementation.
  `TNBBetaSpherical` is an intentional design stub (raises
  `NotImplementedError`) -- the hyperspherical extension is original
  research for this project, not in the source paper. Don't "complete" it
  without discussing the mathematical construction first.
- `src/tnbbeta_vae/registry.py`: models register via
  `@register_model(name, config_cls=SomePydanticConfig)`. New models
  should follow this pattern rather than being wired up ad hoc.
- `src/tnbbeta_vae/training/`: `Trainer` is a minimal, model-agnostic
  epoch loop -- model-specific logic belongs in the model's
  `training_step`, not in `Trainer`.
- `apps/` holds CLI scripts (not part of the installed package); library
  code belongs in `src/tnbbeta_vae/`.
- `notebooks/` is excluded from ruff/pyright/pre-commit -- don't hold it
  to the same style standard as `src/`.

## Branch/PR policy

`master` is protected: no direct pushes, and PRs require the `test`,
`ruff`, and `pyright` CI checks (`.github/workflows/ci.yml`) to pass
before merging.
