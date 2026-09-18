# tnbbeta_vae

![CI](https://github.com/Dant86/tnbbeta_vae/actions/workflows/ci.yml/badge.svg?branch=master)
![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)

Extending the **triply-randomized negative binomial beta (TNBbeta)**
distribution -- introduced in Lederman & Schein (2026),
["The Triply-Randomized Negative Binomial Beta for Robust Regression and
Conjugate Models of Bounded Support Data"](https://arxiv.org/pdf/2606.11624)
-- to the unit hypersphere, and building VAEs around the result.

## Project layout

```
apps/train/       # CLI entrypoint for training runs
scripts/          # Shell scripts (e.g. scripts/train.sh)
src/tnbbeta_vae/
  distributions/  # TNBBetaUnivariate, TNBBetaSpherical
  models/         # architectures/, priors/, losses/
  registry.py     # model registry (Pydantic configs -> model classes)
  training/       # Trainer loop, RunLogger
  data/           # dataset loaders
tests/            # mirrors src/tnbbeta_vae/
notebooks/        # exploratory notebooks (excluded from lint/format)
docs/STYLE_GUIDE.md
writeup/          # LaTeX writeup
```

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync --all-groups
uv run pre-commit install
```

## Development

```bash
uv run pytest              # tests + coverage
uv run ruff format .       # format
uv run ruff check .        # lint
uv run pyright              # type check
```

See [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md) for the full set of coding
conventions (formatting, docstrings, imports, type checking) enforced by
these tools and by the pre-commit hooks.

## Training

```bash
./scripts/train.sh --list          # list registered models
./scripts/train.sh --model conv_tnbbeta_spherical_vae --set latent_dim=8
```

`conv_tnbbeta_spherical_vae` is the first registered model: a simple
conv encoder/decoder VAE with a `TNBBetaSpherical` latent posterior/prior
(see `tnbbeta_vae.models.conv_vae`), trained by maximizing a generic
Monte Carlo ELBO (`tnbbeta_vae.models.losses.monte_carlo_elbo`,
averaged over `num_elbo_samples` draws).
CIFAR-10 loading isn't wired up yet (`tnbbeta_vae.data.cifar10` is a
stub), so `apps/train/main.py` currently builds the model and stops --
driving `Trainer.fit()` end-to-end needs a real dataloader first.

Each run's config and metrics are logged locally under `runs/<run_id>/`
(see `tnbbeta_vae.training.RunLogger`) for later inspection.

### Posterior collapse diagnostics

`p -> 0, q -> 1` (the posterior collapsing to a point mass sitting
wherever the prior already is, independent of `x`) is a known failure
mode worth watching for. Every `training_step` call logs
`posterior_p_{mean,min,max}`, `posterior_q_{mean,min,max}`, and
`posterior_direction_pairwise_cosine_mean` (see
`tnbbeta_vae.models.diagnostics`) alongside `loss`/`kl` -- a collapse
shows up as `q` trending toward 1 and/or the pairwise cosine trending
toward 1 (the encoder converging to ~the same direction for every input)
in `runs/<run_id>/metrics.jsonl`.

For fast local iteration before touching real data,
`tnbbeta_vae.data.gaussian_blob_batch` generates small synthetic images
(a colored Gaussian blob with known hue/position) with ground-truth
generative factors, so reconstructing well genuinely requires the latent
code to carry information about `x`:

```python
import torch
from tnbbeta_vae.data import gaussian_blob_batch
from tnbbeta_vae.models import ConvTNBBetaSphericalVAE, ConvTNBBetaSphericalVAEConfig
from tnbbeta_vae.training import Trainer

model = ConvTNBBetaSphericalVAE(ConvTNBBetaSphericalVAEConfig(latent_dim=8))
trainer = Trainer(
    model,
    torch.optim.Adam(model.parameters(), lr=1e-3),
    "conv_tnbbeta_spherical_vae",
    model.config,
)
dataloader = [gaussian_blob_batch(batch_size=32)[0] for _ in range(200)]
trainer.fit(dataloader, num_epochs=10)
```

## Test coverage

Current coverage: **99%** (`src/tnbbeta_vae`). Generated locally via:

```bash
uv run pytest
```

CI computes this on every push to `master` and posts a full per-file
breakdown to the workflow run's summary (Actions tab); update the badge
above when coverage changes materially.

## Contributing

- All changes land via pull request; direct pushes to `master` are
  blocked (see branch protection).
- A PR can only be merged once the `test`, `ruff`, and `pyright` CI checks
  are passing.
