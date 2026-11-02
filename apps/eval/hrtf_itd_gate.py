"""Feasibility gate for the ITD-only HRTF task (see the draft pre-registration).

Computed on the SADIE II human subjects before any model is trained: how ring-shaped
is the posterior over source direction given only the scalar ITD? Reads the 96 kHz
SimpleFreeFieldHRIR SOFA files written by ``apps.data.download_sadie2`` and cross-fits
by held-out subject groups.

For each held-out group, fitted on the other subjects' (ITD, direction) pairs in 50
ITD quantile bins, the area-weighted mean held-out negative log-likelihood (nats, with
respect to the sphere's surface measure) of the true direction under:

* **uniform:** ``log 4 pi``;
* **oracle ring:** the latitude ``w = a . z`` (``a`` = the interaural axis ``+y``) has
  a per-bin weighted-histogram density and the azimuth about ``a`` is uniform, so the
  density is ``f_W(w) / (2 pi)``;
* **oracle cap:** a per-bin maximum-likelihood von Mises-Fisher on ``S^2``;
* **free-form (context only):** a per-bin von Mises-Fisher kernel density estimate.

Also the ITD summary, the share of ITD variance explained by ``w``, and the ring
thickness. Needs ``h5py`` (not a project dependency): run with ``uv run --with h5py``.

Usage:
    uv run --with h5py python -m apps.eval.hrtf_itd_gate --estimator onset|xcorr
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import optimize, signal
from scipy.spatial import SphericalVoronoi

from tnbbeta_vae.paths import data_dir

NUM_BINS = 50
W_BINS = 40
LOWPASS_HZ = 3000.0
ONSET_DB = -30.0
SOFA_NAME = "{s}_96K_24bit_512tap_FIR_SOFA.sofa"
SUBJECTS = tuple(f"H{k}" for k in range(3, 21))
KDE_KAPPAS = (30.0, 100.0)
NUM_FOLDS = 6
SPLIT_SEED = 0


def main(argv: list[str] | None = None) -> None:
    """Loads the subjects, extracts ITDs and prints the gate."""
    parser = argparse.ArgumentParser(description="ITD-only HRTF feasibility gate.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--estimator", choices=["onset", "xcorr"], default="onset")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output.")
    args = parser.parse_args(argv)
    root = (args.data_dir or data_dir()) / "hrtf" / "sadie2"

    subjects = [_load_subject(root, s, args.estimator) for s in SUBJECTS]
    report = _sanity(subjects)
    report.update(_gate(subjects))
    for key, value in report.items():
        print(f"{key}: {value}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))


def _load_subject(root: Path, name: str, estimator: str) -> dict[str, Any]:
    import h5py  # pyright: ignore[reportMissingImports]

    path = root / name / f"{name}_HRIR_SOFA" / SOFA_NAME.format(s=name)
    with h5py.File(path, "r") as handle:
        irs = np.asarray(handle["Data.IR"])  # (M, 2, N)
        rate = float(np.asarray(handle["Data.SamplingRate"]).ravel()[0])
        position = np.asarray(handle["SourcePosition"])  # azimuth, elevation, radius
        receivers = np.asarray(handle["ReceiverPosition"])  # (2, 3, 1)
    azimuth = np.radians(position[:, 0])
    elevation = np.radians(position[:, 1])
    direction = np.stack(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        axis=-1,
    )
    # SOFA receiver 0 is the left ear (+y); flip so ITD > 0 means a source on the left.
    left_first = bool(receivers[0, 1, 0] > receivers[1, 1, 0])
    left, right = (0, 1) if left_first else (1, 0)
    filtered = _lowpass(irs, rate)
    if estimator == "onset":
        itd = _onset(filtered[:, right], rate) - _onset(filtered[:, left], rate)
    else:
        itd = _xcorr_lag(filtered[:, left], filtered[:, right], rate)
    return {
        "name": name,
        "direction": direction,
        "itd": itd * 1e6,  # microseconds
        "weight": _area_weights(direction),
        "left_first": left_first,
    }


def _lowpass(irs: np.ndarray, rate: float) -> np.ndarray:
    sos = signal.butter(4, LOWPASS_HZ, btype="low", fs=rate, output="sos")
    return signal.sosfiltfilt(sos, irs, axis=-1)


def _onset(irs: np.ndarray, rate: float) -> np.ndarray:
    """First crossing of ``ONSET_DB`` below the per-IR peak (s), interpolated."""
    magnitude = np.abs(irs)
    threshold = magnitude.max(-1, keepdims=True) * 10 ** (ONSET_DB / 20)
    above = magnitude >= threshold
    first = above.argmax(-1)
    previous = np.clip(first - 1, 0, None)
    rows = np.arange(len(first))
    low = magnitude[rows, previous]
    high = magnitude[rows, first]
    gap = np.where(
        high > low, (threshold[:, 0] - low) / np.maximum(high - low, 1e-30), 0
    )
    return (previous + np.clip(gap, 0, 1) * (first > 0) + (first == 0) * 0.0) / rate


def _xcorr_lag(left: np.ndarray, right: np.ndarray, rate: float) -> np.ndarray:
    """Lag (s) of the interaural cross-correlation maximum (right vs left ear)."""
    num = left.shape[-1]
    spectrum = np.fft.rfft(left, 2 * num) * np.conj(np.fft.rfft(right, 2 * num))
    corr = np.fft.fftshift(np.fft.irfft(spectrum, 2 * num), axes=-1)
    peak = corr.argmax(-1)
    rows = np.arange(len(peak))
    before = corr[rows, np.clip(peak - 1, 0, 2 * num - 1)]
    after = corr[rows, np.clip(peak + 1, 0, 2 * num - 1)]
    centre = corr[rows, peak]
    denominator = before - 2 * centre + after
    shift = np.where(
        np.abs(denominator) > 1e-30, 0.5 * (before - after) / denominator, 0
    )
    return -((peak - num) + shift) / rate


def _area_weights(direction: np.ndarray) -> np.ndarray:
    """Spherical-Voronoi cell areas (sum 1); duplicate poles share one cell."""
    rounded = np.round(direction, 6)
    unique, inverse, counts = np.unique(
        rounded, axis=0, return_inverse=True, return_counts=True
    )
    areas = SphericalVoronoi(unique, radius=1).calculate_areas()
    weight = areas[inverse.ravel()] / counts[inverse.ravel()]
    return weight / weight.sum()


def _sanity(subjects: list[dict[str, Any]]) -> dict[str, Any]:
    itd = np.concatenate([s["itd"] for s in subjects])
    weight = np.concatenate([s["weight"] for s in subjects])
    lateral = np.concatenate([s["direction"][:, 1] for s in subjects])  # w = a . z
    mean_w = float(np.average(lateral, weights=weight))
    mean_itd = float(np.average(itd, weights=weight))
    cov = float(np.average((lateral - mean_w) * (itd - mean_itd), weights=weight))
    var_w = float(np.average((lateral - mean_w) ** 2, weights=weight))
    var_itd = float(np.average((itd - mean_itd) ** 2, weights=weight))
    # Variance of ITD explained by a monotone (binned) function of w.
    edges = np.linspace(-1, 1, 41)
    index = np.clip(np.digitize(lateral, edges) - 1, 0, 39)
    fitted = np.zeros_like(itd)
    for b in range(40):
        mask = index == b
        if mask.any():
            fitted[mask] = np.average(itd[mask], weights=weight[mask])
    resid = float(np.average((itd - fitted) ** 2, weights=weight))
    left = np.concatenate([s["direction"][:, 1] > 0.95 for s in subjects])
    right = np.concatenate([s["direction"][:, 1] < -0.95 for s in subjects])
    return {
        "subjects": len(subjects),
        "directions_per_subject_min_max": [
            min(len(s["itd"]) for s in subjects),
            max(len(s["itd"]) for s in subjects),
        ],
        "left_receiver_first": [s["left_first"] for s in subjects].count(True),
        "itd_us_min_median_max": [
            round(float(v), 1) for v in (itd.min(), np.median(itd), itd.max())
        ],
        "itd_us_mean_where_w>0.95": round(float(itd[left].mean()), 1),
        "itd_us_mean_where_w<-0.95": round(float(itd[right].mean()), 1),
        "frac |ITD| > 750 us": round(float((np.abs(itd) > 750).mean()), 4),
        "corr(ITD, w)": round(cov / math.sqrt(var_w * var_itd), 4),
        "ITD variance explained by binned w": round(1 - resid / var_itd, 4),
    }


def _fit_bins(train_itd: np.ndarray) -> np.ndarray:
    return np.quantile(train_itd, np.linspace(0, 1, NUM_BINS + 1))


def _assign(itd: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(itd, edges[1:-1]), 0, NUM_BINS - 1)


def _gate(subjects: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(subjects))
    folds = np.array_split(order, NUM_FOLDS)
    totals = {"uniform": 0.0, "ring": 0.0, "cap": 0.0}
    for kappa in KDE_KAPPAS:
        totals[f"kde{kappa:g}"] = 0.0
    thickness = []
    weight_sum = 0.0
    for fold in folds:
        test = [subjects[i] for i in fold]
        train = [s for i, s in enumerate(subjects) if i not in set(fold.tolist())]
        z_tr = np.concatenate([s["direction"] for s in train])
        x_tr = np.concatenate([s["itd"] for s in train])
        w_tr = np.concatenate([s["weight"] for s in train])
        z_te = np.concatenate([s["direction"] for s in test])
        x_te = np.concatenate([s["itd"] for s in test])
        w_te = np.concatenate([s["weight"] for s in test])
        edges = _fit_bins(x_tr)
        bin_tr, bin_te = _assign(x_tr, edges), _assign(x_te, edges)
        nll = {k: np.zeros(len(x_te)) for k in totals}
        nll["uniform"][:] = math.log(4 * math.pi)
        for b in range(NUM_BINS):
            tr, te = bin_tr == b, bin_te == b
            if not te.any():
                continue
            nll["ring"][te] = _ring_nll(z_tr[tr], w_tr[tr], z_te[te])
            nll["cap"][te] = _cap_nll(z_tr[tr], w_tr[tr], z_te[te])
            for kappa in KDE_KAPPAS:
                nll[f"kde{kappa:g}"][te] = _kde_nll(z_tr[tr], w_tr[tr], z_te[te], kappa)
            lat = z_tr[tr, 1]
            mean = np.average(lat, weights=w_tr[tr])
            thickness.append(math.sqrt(np.average((lat - mean) ** 2, weights=w_tr[tr])))
        for key in totals:
            totals[key] += float((nll[key] * w_te).sum())
        weight_sum += float(w_te.sum())
    result = {f"NLL {k}": round(v / weight_sum, 4) for k, v in totals.items()}
    result["ring minus cap (negative = ring better)"] = round(
        (totals["ring"] - totals["cap"]) / weight_sum, 4
    )
    result["median within-bin sd of w (ring thickness)"] = round(
        float(np.median(thickness)), 4
    )
    result["folds"] = NUM_FOLDS
    return result


def _ring_nll(
    z_train: np.ndarray, w_train: np.ndarray, z_test: np.ndarray
) -> np.ndarray:
    """``log 2 pi - log f_W(w)`` with a smoothed weighted histogram of ``w = z_y``."""
    edges = np.linspace(-1, 1, W_BINS + 1)
    mass, _ = np.histogram(z_train[:, 1], bins=edges, weights=w_train)
    mass = mass / mass.sum()
    mass = (mass + 1e-3 / W_BINS) / (1 + 1e-3)
    density = mass / (2.0 / W_BINS)
    index = np.clip(np.digitize(z_test[:, 1], edges) - 1, 0, W_BINS - 1)
    return math.log(2 * math.pi) - np.log(density[index])


def _cap_nll(
    z_train: np.ndarray, w_train: np.ndarray, z_test: np.ndarray
) -> np.ndarray:
    resultant = (z_train * w_train[:, None]).sum(0) / w_train.sum()
    rbar = float(np.linalg.norm(resultant))
    if rbar < 1e-6:
        return np.full(len(z_test), math.log(4 * math.pi))
    mu = resultant / rbar

    def mean_resultant(kappa: float) -> float:
        return 1.0 / math.tanh(kappa) - 1.0 / kappa

    kappa = float(
        cast("float", optimize.brentq(lambda k: mean_resultant(k) - rbar, 1e-6, 1e5))
    )
    return -(kappa * (z_test @ mu - 1.0) + _vmf_log_norm(kappa))


def _vmf_log_norm(kappa: float) -> float:
    """``log(kappa / (4 pi (1 - exp(-2 kappa)) / 2))``: the S^2 vMF density is
    ``exp(kappa (mu . z - 1) + this)``, stable for large kappa."""
    return (
        math.log(kappa)
        - math.log(4 * math.pi)
        - math.log((1 - math.exp(-2 * kappa)) / 2)
    )


def _kde_nll(
    z_train: np.ndarray, w_train: np.ndarray, z_test: np.ndarray, kappa: float
) -> np.ndarray:
    logits = kappa * (z_test @ z_train.T - 1.0) + _vmf_log_norm(kappa)
    shift = logits.max(1, keepdims=True)
    log_density = shift[:, 0] + np.log(
        (np.exp(logits - shift) * (w_train / w_train.sum())).sum(1) + 1e-300
    )
    return -log_density


if __name__ == "__main__":
    main()
