"""Tests for design 2' (class offsets) of the four-class ring task."""

from __future__ import annotations

import math

import pytest
import torch

from tnbbeta_vae.data.ring_classes import (
    NUM_CLASSES,
    RingClassTask,
    class_identifiability,
    offset_ring_class_task,
    ring_class_task,
)


def test_offsets_are_the_registered_ones() -> None:
    offsets = offset_ring_class_task().offsets()
    expected = 1.5 * torch.tensor(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, -1.0, -1.0]],
        dtype=torch.float64,
    )
    assert torch.equal(offsets, expected)
    scaled = offset_ring_class_task(1.5).offsets()
    assert torch.allclose(scaled, 1.5 * expected)


def test_old_designs_are_untouched() -> None:
    for design in (1, 2):
        task = ring_class_task(design)
        generator = torch.Generator().manual_seed(5)
        old, *_ = task.sample(64, generator)
        again, *_ = ring_class_task(design).sample(64, torch.Generator().manual_seed(5))
        assert torch.equal(old, again)
    with pytest.raises(ValueError):
        RingClassTask(3)
    task = offset_ring_class_task()
    x, _, cls, _ = task.sample(64, torch.Generator().manual_seed(5))
    old, _, old_cls, _ = ring_class_task(2).sample(64, torch.Generator().manual_seed(5))
    assert torch.equal(cls, old_cls)
    assert torch.allclose(x - task.offsets(x.dtype)[cls], old)
    assert ring_class_task(2).input_dim == 7
    assert ring_class_task(1).input_dim == 3


def test_balanced_sampling_has_equal_class_counts() -> None:
    _, _, cls, _ = offset_ring_class_task().sample(
        2000, torch.Generator().manual_seed(1), balanced=True
    )
    assert [int((cls == c).sum()) for c in range(NUM_CLASSES)] == [500] * 4


def test_class_posterior_normalises_and_is_mostly_identifiable() -> None:
    task = offset_ring_class_task()
    x, _, cls, _ = task.sample(300, torch.Generator().manual_seed(2))
    posterior = task.class_posterior(x.double())
    assert torch.allclose(posterior.sum(-1), torch.ones(300, dtype=torch.float64))
    assert float((posterior.argmax(-1) == cls).double().mean()) > 0.98
    fraction, mean_top = class_identifiability(task, num=1000, chunk=500)
    assert fraction > 0.98
    assert mean_top > 0.99


def test_mixture_evidence_matches_monte_carlo() -> None:
    task = offset_ring_class_task()
    x, _, cls, _ = task.sample(4, torch.Generator().manual_seed(3))
    x = x.double()
    generator = torch.Generator().manual_seed(11)
    num = 400_000
    z = torch.randn(num, 3, generator=generator, dtype=torch.float64)
    z = z / z.norm(dim=-1, keepdim=True)
    log_const = -1.5 * math.log(2 * math.pi * task.sigma**2)
    by_class = []
    for c in range(NUM_CLASSES):
        mean = task.decode(z, torch.full((num,), c))
        energy = ((x[:, None, :] - mean[None]) ** 2).sum(-1) / (2 * task.sigma**2)
        by_class.append(log_const + torch.logsumexp(-energy, dim=1) - math.log(num))
    mc = torch.stack(by_class, dim=-1)
    assert torch.allclose(task.log_evidence_by_class(x), mc, atol=0.05)
    mixture = torch.logsumexp(mc, dim=-1) - math.log(NUM_CLASSES)
    assert torch.allclose(task.log_evidence(x), mixture, atol=0.05)
    given = task.log_evidence_given_class(x, cls)
    assert torch.allclose(given, mc.gather(1, cls[:, None])[:, 0], atol=0.05)
    assert bool((task.log_evidence(x) >= given - math.log(NUM_CLASSES) - 1e-9).all())


def test_grid_has_at_least_4001_nodes() -> None:
    assert offset_ring_class_task().grid_size >= 4001
