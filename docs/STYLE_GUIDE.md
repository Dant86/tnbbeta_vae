# Style Guide

This project enforces its conventions with tooling rather than review
discipline alone. Everything below is checked by `ruff`, `pyright`, or the
pre-commit hooks -- if your code passes those, it conforms.

## Formatting

- **Line length**: 88 characters, enforced by `ruff format` and `ruff
  check` (`E501`).
- **Formatter**: `ruff format` (a Black-compatible formatter). Double
  quotes, space indentation.
- Run it yourself with:

  ```bash
  uv run ruff format .
  ```

## Linting

- **Linter**: `ruff check`, configured in `pyproject.toml` under
  `[tool.ruff.lint]`. Enabled rule sets: `E`/`W` (pycodestyle), `F`
  (pyflakes), `I` (isort), `D` (pydocstyle), `UP` (pyupgrade), `B`
  (bugbear), `SIM` (simplify), `N` (pep8-naming).

  ```bash
  uv run ruff check .
  ```

## Docstrings: Google style

All public modules, classes, and functions get a docstring in **Google
style**, enforced by `ruff`'s `pydocstyle` rules (`convention = "google"`).

```python
def build_model(name: str, **config_overrides: object) -> object:
    """Builds a registered model by name from keyword config overrides.

    Args:
        name: Registry key of the model to build.
        **config_overrides: Fields to pass to the model's config class.

    Returns:
        An instance of the registered model class.

    Raises:
        KeyError: If no model is registered under ``name``.
    """
```

Tests and `apps/` scripts are exempt from the docstring requirement
(`D` rules are ignored there via `per-file-ignores`) -- their names should
be descriptive enough on their own.

## Imports: Google style, isort-sorted

Imports are grouped and sorted automatically by `ruff`'s `isort`
implementation (`I` rules), following the standard Google/PEP 8 grouping:

1. `__future__` imports
2. Standard library
3. Third-party packages
4. First-party (`tnbbeta_vae`) imports

Each group is alphabetized, with `from __future__ import annotations` first
when needed for deferred type-annotation evaluation. Don't use wildcard
imports (`from x import *`) outside of `__init__.py` re-exports, and
prefer importing modules/names explicitly over deep dotted access.

```python
from __future__ import annotations

import json
from pathlib import Path

import torch
from pydantic import BaseModel

from tnbbeta_vae.registry import build_model
```

Run `uv run ruff check --fix .` to auto-sort imports.

## Type checking

- **Type checker**: `pyright`, in `standard` mode, configured under
  `[tool.pyright]` in `pyproject.toml`. Covers `src/`, `apps/`, and
  `tests/`.
- All new public functions and methods should have full type annotations
  on parameters and return values.

  ```bash
  uv run pyright
  ```

## Tests and coverage

- Tests live under `tests/`, mirroring the `src/tnbbeta_vae/` package
  layout, and run with `pytest` (`uv run pytest`).
- `pytest-cov` is configured (see `[tool.pytest.ini_options]`) to report
  coverage against `src/tnbbeta_vae` on every run, and CI reports it in the
  README (see [Reporting coverage](#reporting-coverage)).

## Pre-commit hooks

Install the hooks once per clone:

```bash
uv run pre-commit install
```

On each commit, the hooks:

1. Run `ruff format` over staged files and re-stage the result.
2. Run `ruff check` against the (now-formatted) staged files.

Notebooks (`notebooks/`) are excluded from both hooks -- they're not
expected to conform to this style guide.

## Reporting coverage

CI computes coverage on every push to `main`/`master` and updates the
badge/table in [README.md](../README.md#test-coverage) via
`.github/workflows/ci.yml`. Coverage is not a merge gate by itself, but the
`test` job (which must pass) fails on test failures or import errors, so a
broken build can't merge.
