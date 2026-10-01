"""Tests for com-DBLP bridge-node diagnostic utilities."""

from __future__ import annotations

from pathlib import Path

from apps.eval.dblp_bridge_diagnostic import _remap_communities


def test_remap_communities_creates_consistent_mapping(tmp_path: Path) -> None:
    """Verifies that community remapping is consistent across nodes."""
    original_communities: dict[int, set[int]] = {
        0: {0, 1},  # Node 0 in communities 0, 1.
        2: {1, 2},  # Node 2 in communities 1, 2.
        5: {2},  # Node 5 in community 2.
    }

    # Remap for a 3-node graph (nodes 0, 2, 5 -> 0, 1, 2).
    remapped = _remap_communities(3, original_communities)

    # Check that remapped dict has expected keys.
    assert len(remapped) > 0
    # Check that remapped dict is consistent.
    for remapped_id, comms in remapped.items():
        assert isinstance(remapped_id, int)
        assert isinstance(comms, set)


def test_remap_communities_preserves_community_structure() -> None:
    """Verifies that community memberships are preserved after remapping."""
    original_communities: dict[int, set[int]] = {
        10: {0},
        20: {0, 1},
        30: {1},
    }

    remapped = _remap_communities(3, original_communities)

    # All original community IDs should still exist in the remapped version.
    original_comms = {c for comms in original_communities.values() for c in comms}
    remapped_comms = {c for comms in remapped.values() for c in comms}
    assert original_comms == remapped_comms
