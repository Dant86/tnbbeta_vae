# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

### Added

- Project scaffolding: `uv`-managed package, ruff + pyright + pre-commit,
  GitHub Actions CI, branch protection.
- `tnbbeta_vae.distributions.TNBBetaUnivariate`: the univariate TNBbeta
  distribution (Lederman & Schein, 2026, arXiv:2606.11624), with
  closed-form `log_prob` and exact sampling via its auxiliary
  negative-binomial construction.
- `tnbbeta_vae.distributions.TNBBetaSpherical`: design stub for the
  hyperspherical extension (not yet implemented).
- `tnbbeta_vae.registry`: Pydantic-config-based model registry
  (`register_model`, `build_model`, `list_registered_models`).
- `tnbbeta_vae.training`: generic `Trainer` loop and local JSON/YAML
  `RunLogger` for train-run metadata.
