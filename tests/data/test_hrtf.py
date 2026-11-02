"""Tests for the SADIE II ITD table, splits, standardisation and sampler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tnbbeta_vae.data.hrtf import AreaWeightedSampler, load_hrtf, make_split

SUBJECT_IDS = (3, 4, 5, 6, 7, 8, 9, 10)
ROWS = (30, 24)


@pytest.fixture
def npz_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    subject, direction, xcorr, onset, weight = [], [], [], [], []
    for i, sid in enumerate(SUBJECT_IDS):
        n = ROWS[i % 2]
        z = rng.normal(size=(n, 3))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        w = rng.uniform(0.5, 2.0, size=n)
        subject.append(np.full(n, sid))
        direction.append(z)
        xcorr.append(600 * z[:, 1] + 10 * sid)  # subject-dependent offset
        onset.append(400 * z[:, 1])
        weight.append(w / w.sum())
    path = tmp_path / "table.npz"
    np.savez(
        path,
        subject=np.concatenate(subject),
        direction=np.concatenate(direction),
        itd_xcorr_us=np.concatenate(xcorr),
        itd_onset_us=np.concatenate(onset),
        weight=np.concatenate(weight),
    )
    return path


def test_load_selects_estimator(npz_path: Path) -> None:
    xcorr = load_hrtf(npz_path, "xcorr")
    onset = load_hrtf(npz_path, "onset")
    assert xcorr.direction.shape == (sum(ROWS[i % 2] for i in range(8)), 3)
    assert np.allclose(onset.itd, 400 * onset.direction[:, 1])
    assert not np.allclose(xcorr.itd, onset.itd)
    # Weights sum to 1 within each subject.
    for sid in SUBJECT_IDS:
        assert xcorr.weight[xcorr.subject == sid].sum() == pytest.approx(1.0)


def test_split_is_disjoint_by_subject_and_reproducible(npz_path: Path) -> None:
    table = load_hrtf(npz_path)
    for seed in range(10):
        split = make_split(table, seed, num_test=3)
        assert len(split.test_subjects) == 3
        assert len(split.train_subjects) == 5
        assert not set(split.test_subjects) & set(split.train_subjects)
        assert set(split.test_subjects) | set(split.train_subjects) == set(SUBJECT_IDS)
        assert set(split.test_subject.tolist()) == set(split.test_subjects.tolist())
        again = make_split(table, seed, num_test=3)
        assert np.array_equal(split.test_subjects, again.test_subjects)
    assert (
        len({tuple(make_split(table, s, 3).test_subjects.tolist()) for s in range(10)})
        > 1
    )


def test_test_weights_sum_to_one_with_equal_subject_share(npz_path: Path) -> None:
    split = make_split(load_hrtf(npz_path), 0, num_test=3)
    assert split.test_weight.sum() == pytest.approx(1.0)
    for sid in split.test_subjects:
        assert split.test_weight[split.test_subject == sid].sum() == pytest.approx(
            1 / 3
        )
    # Training weights: 1 per subject, so 5 in total.
    assert split.train_weight.sum() == pytest.approx(5.0)


def test_standardisation_uses_only_training_subjects(npz_path: Path) -> None:
    table = load_hrtf(npz_path)
    split = make_split(table, 1, num_test=3)
    train = np.isin(table.subject, split.train_subjects)
    mean = np.average(table.itd[train], weights=table.weight[train])
    sd = np.sqrt(
        np.average((table.itd[train] - mean) ** 2, weights=table.weight[train])
    )
    assert split.mean == pytest.approx(mean)
    assert split.sd == pytest.approx(sd)
    standardised = split.train_x[:, 0].double().numpy()
    assert np.average(standardised, weights=split.train_weight) == pytest.approx(
        0.0, abs=1e-5
    )
    assert np.average(standardised**2, weights=split.train_weight) == pytest.approx(
        1.0, abs=1e-4
    )
    # Changing a held-out subject's ITDs leaves the standardisation untouched.
    altered = table.itd.copy()
    altered[~train] += 1e4
    shifted = type(table)(table.subject, table.direction, altered, table.weight)
    other = make_split(shifted, 1, num_test=3)
    assert other.mean == pytest.approx(split.mean)
    assert other.sd == pytest.approx(split.sd)
    assert not np.allclose(other.test_x.numpy(), split.test_x.numpy())


def test_sampler_follows_weights_and_is_reproducible() -> None:
    weight = np.array([0.0, 1.0, 3.0, 6.0])
    sampler = AreaWeightedSampler(weight, batch_size=100, seed=3)
    assert sampler.batches_per_epoch == 1
    draws = torch.cat([b for _ in range(200) for b in sampler.epoch()])
    freq = torch.bincount(draws, minlength=4).double() / len(draws)
    assert freq[0] == 0
    assert np.allclose(freq.numpy(), weight / weight.sum(), atol=0.01)
    first = AreaWeightedSampler(weight, 50, 7).epoch()
    second = AreaWeightedSampler(weight, 50, 7).epoch()
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert AreaWeightedSampler(np.ones(1000), 256, 0).batches_per_epoch == 4
