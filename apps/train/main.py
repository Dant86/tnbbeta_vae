"""CLI entrypoint for training a registered model.

Usage:
    uv run python -m apps.train.main --list
    uv run python -m apps.train.main --model <name> [--set key=value ...]
"""

from __future__ import annotations

import argparse
import sys

from tnbbeta_vae.registry import build_model, list_registered_models


def main(argv: list[str] | None = None) -> None:
    """Parses CLI args and either lists or builds/trains a registered model.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List registered model names and exit."
    )
    parser.add_argument("--model", type=str, help="Registered model name to train.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Config override, may be repeated.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name in list_registered_models():
            print(name)
        return

    if not args.model:
        parser.error("--model is required unless --list is passed.")

    overrides = _parse_overrides(args.set)
    model = build_model(args.model, **overrides)
    print(f"Built model {args.model!r}: {model}")

    raise NotImplementedError(
        "Training is not yet wired up: dataset loading (see "
        "tnbbeta_vae.data.cifar10) and a concrete Trainer.fit() call are "
        "still needed."
    )


def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    """Parses ``key=value`` strings into a dict.

    Args:
        pairs: Strings of the form ``"key=value"``.

    Returns:
        Mapping from key to value.
    """
    overrides: dict[str, str] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        overrides[key] = value
    return overrides


if __name__ == "__main__":
    main(sys.argv[1:])
