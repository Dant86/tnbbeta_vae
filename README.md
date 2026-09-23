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
apps/eval/        # CLI entrypoint for evaluating a checkpoint
apps/data/        # CLI entrypoint for downloading CIFAR-10
scripts/          # Shell scripts (scripts/train.sh)
scripts/slurm/    # sbatch scripts for the UChicago DSI cluster
src/tnbbeta_vae/
  distributions/  # TNBBetaUnivariate, TNBBetaSpherical
  paths.py        # data/checkpoint/runs directories (from .env)
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

### CIFAR-10 workflow (local or cluster)

Directories are configured in `.env` (copy `.env.sample`): 
`TNBBETA_DATA_DIR`, `TNBBETA_CHECKPOINT_DIR`, `TNBBETA_RUNS_DIR`. Real
environment variables override `.env`.

```bash
uv run python -m apps.data.download_cifar10          # once; needs network
uv run python -m apps.train.main --model conv_tnbbeta_spherical_vae --run-name tnb_test \
    --set latent_dim=32 --set likelihood_scale=0.1 --epochs 20 --resume
uv run python -m apps.eval.main --run-name tnb_test  # writes eval_final_test.json + PNGs
```

The Gaussian likelihood's scale sigma is always learned (one scalar shared by
all pixels); `--set likelihood_scale=...` only sets its starting value.

Training writes `latest.pt` every epoch and `final.pt` at the end under
`$TNBBETA_CHECKPOINT_DIR/<run-name>/`; `--resume` continues from
`latest.pt` (and does nothing if `final.pt` exists), so re-running after a
preemption is safe. TNBBeta's prior is always Uniform(sphere)
(p=0.5, q=0, epsilon=(d-1)/2).

### MNIST and the S-VAE reproduction

Reproduces the unsupervised MNIST experiment of the S-VAE paper (Davidson et al. 2018)
with three models on the same conv architecture: Gaussian, vMF and TNBBeta.

```bash
uv run python -m apps.data.download_mnist            # once; needs network
uv run python -m apps.train.main --model conv_tnbbeta_spherical_vae --dataset mnist \
    --set latent_dim=10 --batch-size 64 --epochs 1000 --patience 50 \
    --kl-warmup-epochs 100 --run-name mnist_tnb_d10_seed0 --resume
uv run python -m apps.eval.svae_metrics --run-name mnist_tnb_d10_seed0
sbatch scripts/slurm/mnist_sweep.sbatch              # 90-task array; edit the grid inside
uv run python -m apps.eval.svae_table                # Table 1: mean +- std over seeds
uv run python -m apps.eval.svae_knn --run-name mnist_tnb_d10_seed0   # Table 2: latent k-NN
uv run python -m apps.eval.svae_table --kind knn     # Table 2 aggregated over seeds
uv run python -m apps.eval.confidence_probe --run-name mnist_tnb_d10_seed0  # k-NN from p/kappa alone
uv run python -m apps.eval.confidence_probe --run-name mnist_tnb_d10_seed0 --param epsilon  # TNBBeta only
uv run python -m apps.eval.svae_table --kind confidence          # aggregated over seeds
uv run python -m apps.eval.svae_table --kind confidence_epsilon  # aggregated, TNBBeta epsilon vs vMF kappa
uv run python -m apps.eval.svae_latitude --run-name mnist_tnb_d10_seed0    # TNBBeta only: p vs q plot
```

Ablation: does fixing TNBBeta's epsilon force p and/or q to pick up its role (see above)?

```bash
sbatch scripts/slurm/fixed_epsilon_sweep.sbatch      # 9 tasks: epsilon in {0.5, 1, 1.5} x 3 seeds, d=5
uv run python -m apps.eval.svae_table --prefix mnistfix --models eps0p5 eps1p0 eps1p5 --dims 5 --seeds 0 1 2
```

Two more experiments from the paper:

```bash
uv run python -m apps.synthetic.circle_recovery      # sec. 5.1: recover a circle from R^100 (CPU, ~1 min)
uv run python -m apps.data.download_planetoid        # once; Cora, Citeseer, Pubmed raw files
sbatch scripts/slurm/link_prediction.sbatch          # Table 4: 9 tasks (dataset x latent family)
uv run python -m apps.link_prediction.table          # test AUC / AP, mean +- std over seeds
```

Semi-supervised classification with 100 labels (Table 3) uses the stacked M1+M2 model,
with any latent family for `z1` and `z2` (nn = Gaussian+Gaussian, ss = vMF+vMF, sn =
vMF+Gaussian, tt/tn = TNBBeta):

```bash
uv run python -m apps.semi_supervised.main --run-name semi_ss_z10_10_seed0 \
    --z1-family vmf --z2-family vmf --z1-dim 10 --z2-dim 10
sbatch scripts/slurm/semi_supervised_sweep.sbatch    # 225 tasks (5 variants x 3x3 dims x 5 seeds)
uv run python -m apps.semi_supervised.table          # test accuracy (%), mean +- std over seeds
```

MNIST is binarized dynamically for training and once (fixed seed) for validation and test.
`apps.eval.svae_metrics` reports the importance-weighted log-likelihood (500 samples), the
ELBO, the reconstruction term and the KL, in nats per image.

`apps.eval.confidence_probe` runs the same k-NN class probe as `svae_knn`, but using only
the posterior's confidence scalar (the Gaussian's mean std, vMF's kappa, or TNBBeta's p) --
no direction -- to test whether a family routes class information through it separately
from the mean direction. On MNIST, TNBBeta's training saturates p near 1 with q near its
floor (the q=0, p->1 special case: a cap at the mean direction, the same shape vMF always
has), leaving p with almost no spread to carry class information; `--param epsilon` probes
TNBBeta's cap-thickness parameter instead, the fairer comparison to vMF's kappa in that
regime. `apps.eval.svae_latitude` (TNBBeta only; a no-op for other models) writes a
per-class p-vs-q scatter and summary stats, the "cap vs ring" diagnostic: p near 0 or 1 is
a tight cap at the mean direction like a vMF posterior, p away from the poles is a ring at
some angular distance from it, a shape vMF cannot represent.

### Running on the UChicago DSI cluster

One-time setup on the login node (see
[the cluster docs](https://cluster-policy.ds.uchicago.edu/)): clone the
repo, install `uv`, run `uv sync`, copy `.env.sample` to `.env` and point
the three paths at storage you own, then `mkdir -p slurm_logs`.

```bash
sbatch scripts/slurm/download_cifar10.sbatch        # or run the download on the login node
sbatch scripts/slurm/train.sbatch conv_tnbbeta_spherical_vae tnb_d32_seed0 \
    --set latent_dim=32 --set likelihood_scale=0.1 --epochs 50
sbatch scripts/slurm/sweep.sbatch                   # 12-task array; edit the grid inside
sbatch scripts/slurm/eval.sbatch tnb_d32_seed0
```

For FID (`apps.eval.fid`), download the Inception weights once first, on the
login node if compute nodes have no internet access:

```bash
uv run python -m apps.data.download_inception_weights
uv run --no-sync python -m apps.eval.fid --run-name tnb_d32_seed0
```

The default QoS is preemptable, so the scripts use `--requeue` together
with `--resume`. The resource requests in the scripts (1 GPU, 4 CPUs,
16 GB, partition `general`) are starting points to adjust.

Each run's config and metrics are logged locally under `runs/<run_id>/`
(see `tnbbeta_vae.training.RunLogger`) for later inspection.

### Logged posterior statistics

Every `training_step` logs `posterior_p_{mean,min,max}`,
`posterior_q_{mean,min,max}`, `posterior_epsilon_mean` (see
`tnbbeta_vae.models.diagnostics`) and the current `likelihood_scale`
alongside `loss`/`kl` in `runs/<run_id>/metrics.jsonl`. Things to watch
for: p or q pinned at its clamp (1e-6), or q drifting toward 1 (a thin
ring around the mean direction).

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
