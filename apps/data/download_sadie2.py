"""Downloads the SADIE II human-subject HRIR SOFA files into the data directory.

Fetches ``H<k>_HRIR_SOFA.zip`` for the human subjects H3-H20 (the two dummy heads D1
and D2 are not used) from Zenodo record 10886409 (SADIE II database, University of York;
Apache-2.0 on Zenodo; cite the original paper, DOI 10.3390/app8112029).
Each archive is checked against the md5 Zenodo lists and extracted into its own
directory under ``<data dir>/hrtf/sadie2/``; archive members are never executed and
path traversal in a member name is refused. Run once; analysis reads from disk and never
downloads.

Usage:
    uv run python -m apps.data.download_sadie2 [--data-dir PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

from tnbbeta_vae.paths import data_dir

_RECORD_URL = "https://zenodo.org/api/records/10886409"
_SUBJECTS = tuple(f"H{k}" for k in range(3, 21))


def main(argv: list[str] | None = None) -> None:
    """Downloads, verifies and extracts the 18 human-subject HRIR SOFA archives.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Raises:
        RuntimeError: If a wanted archive is missing from the record or fails its md5.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.data_dir or data_dir()) / "hrtf" / "sadie2"
    root.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(_RECORD_URL, timeout=60) as response:  # noqa: S310
        record = json.load(response)
    files = {entry["key"]: entry for entry in record["files"]}
    for subject in _SUBJECTS:
        key = f"{subject}_HRIR_SOFA.zip"
        if key not in files:
            raise RuntimeError(f"{key} is not in Zenodo record 10886409.")
        target_dir = root / subject
        if any(target_dir.glob("*.sofa")):
            print(f"{subject}: already extracted")
            continue
        archive = root / key
        if not archive.exists():
            _download(files[key]["links"]["self"], archive)
        expected = files[key]["checksum"].split(":", 1)[1]
        if _md5(archive) != expected:
            raise RuntimeError(f"{key}: md5 mismatch; delete {archive} and retry.")
        _extract(archive, target_dir)
        print(f"{subject}: {len(list(target_dir.rglob('*.sofa')))} sofa file(s)")


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(".part")
    with (
        urllib.request.urlopen(url, timeout=300) as response,
        partial.open("wb") as out,
    ):  # noqa: S310
        while chunk := response.read(1 << 20):
            out.write(chunk)
    partial.rename(destination)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    base = target_dir.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not (base / member).resolve().is_relative_to(base):
                raise RuntimeError(f"{archive.name}: unsafe member path {member!r}.")
        bundle.extractall(target_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
