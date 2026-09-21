"""Prints the semi-supervised MNIST results table (S-VAE paper, Table 3).

Usage:
    uv run python -m apps.semi_supervised.table [--prefix semi] \
        [--variants nn ss sn tt tn] [--dims 5 10 50] [--seeds 0 1 2 3 4]

Reads ``$TNBBETA_CHECKPOINT_DIR/<prefix>_<variant>_z<z1>_<z2>_seed<seed>/
semi_supervised_final.json`` (the names used by
``scripts/slurm/semi_supervised_sweep.sbatch``) and prints the test accuracy (%), mean
+- standard deviation over seeds, for every (z1 dim, z2 dim) pair. Variants name the
(z1, z2) latent families: nn = Gaussian+Gaussian, ss = vMF+vMF, sn = vMF+Gaussian,
tt = TNBBeta+TNBBeta, tn = TNBBeta+Gaussian. A mean is bold if it is the best in its row
and beats every other variant with a Welch t-test at p < ``--alpha``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

from apps.eval.svae_table import format_mean_std, significant_winner
from tnbbeta_vae.paths import checkpoint_dir

VARIANTS = {
    "nn": ("gaussian", "gaussian"),
    "ss": ("vmf", "vmf"),
    "sn": ("vmf", "gaussian"),
    "tt": ("tnbbeta", "tnbbeta"),
    "tn": ("tnbbeta", "gaussian"),
}
_TITLES = {"nn": "N+N", "ss": "S+S", "sn": "S+N", "tt": "T+T", "tn": "T+N"}


def main(argv: list[str] | None = None) -> None:
    """Prints the markdown table for the runs found in the checkpoint directory.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="semi")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--dims", nargs="+", type=int, default=[5, 10, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--alpha", type=float, default=0.01)
    args = parser.parse_args(argv)

    header = ["z1 dim", "z2 dim", *(_TITLES[v] for v in args.variants)]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    missing = 0
    for z1, z2 in itertools.product(args.dims, args.dims):
        series = {}
        for variant in args.variants:
            values = []
            for seed in args.seeds:
                run = f"{args.prefix}_{variant}_z{z1}_{z2}_seed{seed}"
                path = checkpoint_dir() / run / "semi_supervised_final.json"
                if path.exists():
                    values.append(100 * json.loads(path.read_text())["test_accuracy"])
                else:
                    missing += 1
            series[variant] = values
        winner = significant_winner(series, args.alpha)
        cells = [str(z1), str(z2)]
        cells += [format_mean_std(series[v], winner == v) for v in args.variants]
        print("| " + " | ".join(cells) + " |")
    if missing:
        print(
            f"\n{missing} run(s) had no results file and were skipped.", file=sys.stderr
        )


if __name__ == "__main__":
    main(sys.argv[1:])
