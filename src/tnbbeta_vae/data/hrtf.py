"""SADIE II ITD table: loading, subject splits, standardisation and the sampler.

The table is written once by ``apps/data/prepare_sadie2_itd.py`` (the only step that
needs ``h5py``); everything here reads that ``.npz`` with numpy. Rows are (subject,
head-frame direction, ITD in microseconds, area weight), with weights summing to 1
within each subject. Pre-registration: ``writeup/results/hrtf_itd_ring_preregistration_
2026-11-01.md``.

A split (seed ``s``) draws ``NUM_TEST`` random test subjects and trains on the rest.
ITDs are standardised by the *training* subjects' area-weighted mean and sd. Training
batches are sampled with probability proportional to the area weight and every
evaluation mean is area-weighted (each held-out subject counts equally).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from tnbbeta_vae.paths import data_dir

__all__ = [
    "AXIS",
    "NPZ_NAME",
    "NUM_TEST",
    "AreaWeightedSampler",
    "HrtfSplit",
    "HrtfTable",
    "default_path",
    "load_hrtf",
    "make_split",
]

NPZ_NAME = "sadie2_itd.npz"
NUM_TEST = 3
# The interaural axis in the head frame (x front, y left, z up): the ring axis.
AXIS = (0.0, 1.0, 0.0)
Estimator = Literal["xcorr", "onset"]


@dataclass(frozen=True)
class HrtfTable:
    """The whole table for one ITD estimator.

    Attributes:
        subject: Subject ids, shape ``(n,)``.
        direction: Head-frame unit vectors, shape ``(n, 3)``.
        itd: ITD in microseconds, shape ``(n,)``.
        weight: Area weights, summing to 1 within each subject, shape ``(n,)``.
    """

    subject: np.ndarray
    direction: np.ndarray
    itd: np.ndarray
    weight: np.ndarray

    @property
    def subjects(self) -> np.ndarray:
        """The sorted distinct subject ids."""
        return np.unique(self.subject)


@dataclass(frozen=True)
class HrtfSplit:
    """One subject split with standardised ITDs.

    Attributes:
        seed: The split seed.
        train_subjects: Training subject ids (sorted).
        test_subjects: Held-out subject ids (sorted).
        mean: Training area-weighted ITD mean (microseconds).
        sd: Training area-weighted ITD sd (microseconds).
        train_x: Standardised training ITD, shape ``(n_train, 1)``, float32.
        train_z: Training directions, shape ``(n_train, 3)``, float32.
        train_weight: Training area weights, shape ``(n_train,)``, float64.
        train_itd: Raw training ITD (microseconds), shape ``(n_train,)``.
        test_x: Standardised test ITD, shape ``(n_test, 1)``, float64.
        test_z: Test directions, shape ``(n_test, 3)``, float64.
        test_weight: Test weights normalised to sum 1 (each subject equal), float64.
        test_itd: Raw test ITD (microseconds), shape ``(n_test,)``.
        test_subject: Subject id of each test row.
    """

    seed: int
    train_subjects: np.ndarray
    test_subjects: np.ndarray
    mean: float
    sd: float
    train_x: Tensor
    train_z: Tensor
    train_weight: np.ndarray
    train_itd: np.ndarray
    test_x: Tensor
    test_z: Tensor
    test_weight: np.ndarray
    test_itd: np.ndarray
    test_subject: np.ndarray


def default_path() -> Path:
    """Where ``prepare_sadie2_itd`` writes the table."""
    return data_dir() / "hrtf" / NPZ_NAME


def load_hrtf(path: Path | None = None, estimator: Estimator = "xcorr") -> HrtfTable:
    """Loads the table, selecting one ITD estimator.

    Args:
        path: The ``.npz`` (default: :func:`default_path`).
        estimator: ``"xcorr"`` (cross-correlation lag, primary) or ``"onset"``.

    Returns:
        The table.
    """
    key = {"xcorr": "itd_xcorr_us", "onset": "itd_onset_us"}[estimator]
    with np.load(path or default_path()) as data:
        return HrtfTable(
            subject=np.asarray(data["subject"]),
            direction=np.asarray(data["direction"], dtype=np.float64),
            itd=np.asarray(data[key], dtype=np.float64),
            weight=np.asarray(data["weight"], dtype=np.float64),
        )


def make_split(table: HrtfTable, seed: int, num_test: int = NUM_TEST) -> HrtfSplit:
    """The seed's split: ``num_test`` random test subjects, the rest train.

    Args:
        table: The table.
        seed: Split seed (the study uses 0-9).
        num_test: Number of held-out subjects.

    Returns:
        The split with standardised ITDs.
    """
    subjects = table.subjects
    order = np.random.default_rng(seed).permutation(len(subjects))
    test_subjects = np.sort(subjects[order[:num_test]])
    train_subjects = np.sort(subjects[order[num_test:]])
    is_test = np.isin(table.subject, test_subjects)
    train = ~is_test
    weight = table.weight[train]
    mean = float(np.average(table.itd[train], weights=weight))
    sd = float(np.sqrt(np.average((table.itd[train] - mean) ** 2, weights=weight)))
    test_weight = table.weight[is_test]
    test_weight = test_weight / test_weight.sum()

    def standardised(mask: np.ndarray, dtype: torch.dtype) -> Tensor:
        return torch.tensor(((table.itd[mask] - mean) / sd)[:, None], dtype=dtype)

    return HrtfSplit(
        seed=seed,
        train_subjects=train_subjects,
        test_subjects=test_subjects,
        mean=mean,
        sd=sd,
        train_x=standardised(train, torch.float32),
        train_z=torch.tensor(table.direction[train], dtype=torch.float32),
        train_weight=weight,
        train_itd=table.itd[train],
        test_x=standardised(is_test, torch.float64),
        test_z=torch.tensor(table.direction[is_test], dtype=torch.float64),
        test_weight=test_weight,
        test_itd=table.itd[is_test],
        test_subject=table.subject[is_test],
    )


class AreaWeightedSampler:
    """Batches of row indices drawn with probability proportional to the area weight.

    One epoch is ``ceil(n / batch_size)`` batches of ``batch_size`` draws with
    replacement; the stream depends only on ``seed`` and the weights, so it is the same
    for every model family trained on a split.
    """

    def __init__(self, weight: np.ndarray, batch_size: int, seed: int) -> None:
        """Stores the weights and seeds the sampler.

        Args:
            weight: Non-negative sampling weights, shape ``(n,)``.
            batch_size: Draws per batch.
            seed: Seed of the sampler's own generator.
        """
        self.weight = torch.tensor(weight / weight.sum(), dtype=torch.float64)
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    @property
    def batches_per_epoch(self) -> int:
        """``ceil(n / batch_size)``: an epoch's worth of batches."""
        return -(-len(self.weight) // self.batch_size)

    def epoch(self) -> list[Tensor]:
        """The next epoch: ``batches_per_epoch`` index tensors of ``batch_size``."""
        draws = torch.multinomial(
            self.weight,
            self.batches_per_epoch * self.batch_size,
            replacement=True,
            generator=self.generator,
        )
        return list(draws.split(self.batch_size))
