"""Writes the (subject, direction, ITD, area weight) table of the SADIE II ITD study.

Reads the 18 human subjects' 96 kHz SimpleFreeFieldHRIR SOFA files that
``apps.data.download_sadie2`` extracted, computes the interaural time difference
with both estimators of the pre-registration (cross-correlation lag: primary;
-30 dB onset: sensitivity) by reusing :mod:`apps.eval.hrtf_itd_gate`'s functions, and
writes one ``.npz`` that every downstream step reads (no ``h5py`` there).

The SOFA files are downloaded data and are only parsed here (HDF5 reads); nothing in
the data directory is executed. ``h5py`` is not a project dependency::

    uv run --with h5py python -m apps.data.prepare_sadie2_itd

Arrays in the file (``N = 18 * 2818`` rows): ``subject`` (int, 3..20 for H3..H20),
``direction`` ``(N, 3)`` head-frame unit vectors (x front, y left, z up),
``itd_xcorr_us`` and ``itd_onset_us`` (microseconds, positive = source on the left),
``weight`` (spherical-Voronoi cell areas, summing to 1 within each subject).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from apps.eval.hrtf_itd_gate import SUBJECTS, _load_subject
from tnbbeta_vae.paths import data_dir

NPZ_NAME = "sadie2_itd.npz"


def main(argv: list[str] | None = None) -> None:
    """Computes both ITD estimators for every subject and writes the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    base = args.data_dir or data_dir()
    out = args.out or base / "hrtf" / NPZ_NAME
    root = base / "hrtf" / "sadie2"

    columns: dict[str, list[np.ndarray]] = {
        k: []
        for k in ("subject", "direction", "itd_xcorr_us", "itd_onset_us", "weight")
    }
    for name in SUBJECTS:
        xcorr = _load_subject(root, name, "xcorr")
        onset = _load_subject(root, name, "onset")
        if not np.allclose(xcorr["direction"], onset["direction"]):
            raise RuntimeError(f"{name}: estimator passes disagree on directions.")
        num = len(xcorr["itd"])
        columns["subject"].append(np.full(num, int(name[1:])))
        columns["direction"].append(xcorr["direction"])
        columns["itd_xcorr_us"].append(xcorr["itd"])
        columns["itd_onset_us"].append(onset["itd"])
        columns["weight"].append(xcorr["weight"])
        print(f"{name}: {num} directions, left receiver first: {xcorr['left_first']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.concatenate(v) for k, v in columns.items()})  # pyright: ignore[reportArgumentType]
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
