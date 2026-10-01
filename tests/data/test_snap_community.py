"""Tests for SNAP community graph loading and community membership parsing."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from tnbbeta_vae.data.snap_community import (
    bridge_nodes,
    load_snap_communities,
    load_snap_community,
    primary_secondary_communities,
    remap_communities,
)


def _write_edge_list(path: Path, edges: list[tuple[int, int]]) -> None:
    """Writes a synthetic edge list in SNAP ungraph format."""
    with gzip.open(path, "wt") as f:
        f.write("# Test edge list\n")
        for a, b in edges:
            f.write(f"{a}\t{b}\n")


def _write_community_file(path: Path, communities: list[list[int]]) -> None:
    """Writes a synthetic community membership file in SNAP format."""
    with gzip.open(path, "wt") as f:
        f.write("# Test communities\n")
        for community in communities:
            f.write("\t".join(str(x) for x in community) + "\n")


def test_load_snap_community_remaps_node_ids_and_builds_undirected_adjacency(
    tmp_path: Path,
) -> None:
    """Verifies node ID remapping and adjacency construction from edge list."""
    # Edge list with non-contiguous IDs: {0, 2, 5, 7} -> remapped to {0, 1, 2, 3}.
    edges = [(0, 2), (2, 5), (5, 7)]
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", edges)
    # Write a dummy community file (not used in this test).
    _write_community_file(tmp_path / "com-test.all.cmty.txt.gz", [[0, 1, 2, 3]])

    graph = load_snap_community(tmp_path, "test")

    assert graph.adjacency.shape == (4, 4)
    assert graph.features.shape == (4, 4)
    # Raw IDs {0, 2, 5, 7} remap to 0..3 in the order they are first seen in the edge
    # list, which happens to coincide with sorted order here.
    assert graph.node_id_map == {0: 0, 2: 1, 5: 2, 7: 3}
    # Check undirected (symmetric): edges appear twice.
    assert graph.adjacency.sum() == 6  # 3 edges * 2 directions
    # Sparse identity features (torch.sparse for snap_community).
    # Verify it's sparse identity by converting to dense.
    if hasattr(graph.features, "to_dense"):
        feat_dense = graph.features.to_dense().numpy()
    elif hasattr(graph.features, "toarray"):
        feat_dense = graph.features.toarray()  # type: ignore
    else:
        feat_dense = np.asarray(graph.features)
    assert np.allclose(feat_dense, np.eye(4))


def test_load_snap_community_handles_comments_and_blank_lines(
    tmp_path: Path,
) -> None:
    """Verifies that comments and blank lines are skipped."""
    edges_text = "# Comment line\n0\t1\n\n# Another comment\n1\t2\n"
    with gzip.open(tmp_path / "com-test.ungraph.txt.gz", "wt") as f:
        f.write(edges_text)
    _write_community_file(tmp_path / "com-test.all.cmty.txt.gz", [[0, 1, 2]])

    graph = load_snap_community(tmp_path, "test")

    assert graph.adjacency.shape == (3, 3)
    assert graph.adjacency.sum() == 4  # 2 edges * 2 directions


def test_load_snap_communities_parses_membership_and_returns_dict(
    tmp_path: Path,
) -> None:
    """Verifies community membership parsing returns correct dict."""
    # Edges are not used for community loading, but file must exist.
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", [(0, 1)])
    # Three communities: line 0 has nodes 0, 1; line 1 has 1, 2; line 2 has 2, 3.
    # Communities are indexed by line number.
    with gzip.open(tmp_path / "com-test.all.cmty.txt.gz", "wt") as f:
        f.write("0 1\n")  # Line 0
        f.write("1 2\n")  # Line 1
        f.write("2 3\n")  # Line 2

    memberships = load_snap_communities(tmp_path, "test")

    # Community indices correspond to line numbers
    assert memberships[0] == {0}  # Node 0 in community 0.
    assert memberships[1] == {0, 1}  # Node 1 in communities 0 and 1.
    assert memberships[2] == {1, 2}  # Node 2 in communities 1 and 2.
    assert memberships[3] == {2}  # Node 3 in community 2.


def test_bridge_nodes_identifies_nodes_in_multiple_communities(
    tmp_path: Path,
) -> None:
    """Verifies bridge-node identification by multiplicity."""
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", [(0, 1)])
    communities = [[0], [1], [1, 2], [2]]
    _write_community_file(tmp_path / "com-test.all.cmty.txt.gz", communities)

    memberships = load_snap_communities(tmp_path, "test")
    bridges = bridge_nodes(memberships, min_multiplicity=2)

    # Node 1 belongs to communities 1 and 2 (multiplicity 2).
    # Node 2 belongs to communities 2 and 3 (multiplicity 2).
    assert bridges == {1, 2}


def test_bridge_nodes_with_different_thresholds(tmp_path: Path) -> None:
    """Verifies that bridge threshold is correctly applied."""
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", [(0, 1)])
    communities = [[0, 1, 2], [1, 2], [2]]
    _write_community_file(tmp_path / "com-test.all.cmty.txt.gz", communities)

    memberships = load_snap_communities(tmp_path, "test")

    # min_multiplicity=2: nodes 1, 2 (in 2+ communities).
    assert bridge_nodes(memberships, min_multiplicity=2) == {1, 2}
    # min_multiplicity=3: only node 2 (in 3 communities).
    assert bridge_nodes(memberships, min_multiplicity=3) == {2}
    # min_multiplicity=4: no nodes.
    assert bridge_nodes(memberships, min_multiplicity=4) == set()


def test_sparse_identity_features_are_orthonormal(tmp_path: Path) -> None:
    """Verifies that sparse identity features form an orthonormal basis."""
    edges = [(0, 1), (1, 2), (2, 0)]
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", edges)
    _write_community_file(tmp_path / "com-test.all.cmty.txt.gz", [[0, 1, 2]])

    graph = load_snap_community(tmp_path, "test")

    # Features should be identity: features @ features^T = I.
    # Note: graph.features might be torch.sparse, need to handle both types.
    if hasattr(graph.features, "toarray"):
        feat_dense = graph.features.toarray()  # type: ignore
    else:
        feat_dense = graph.features.to_dense().numpy()  # type: ignore
    product = feat_dense @ feat_dense.T  # type: ignore
    assert np.allclose(product, np.eye(3))


def test_empty_communities_are_handled(tmp_path: Path) -> None:
    """Verifies that malformed community lines (empty, comment-only) are skipped."""
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", [(0, 1)])
    # Write community file with blank and comment lines.
    # Line numbers: 0=comment, 1=blank, 2="0 1", 3=comment, 4=blank, 5="1 2"
    # So communities are indexed 2 and 5.
    with gzip.open(tmp_path / "com-test.all.cmty.txt.gz", "wt") as f:
        f.write("# Comment\n")
        f.write("\n")
        f.write("0 1\n")
        f.write("# Another comment\n")
        f.write("\n")
        f.write("1 2\n")

    memberships = load_snap_communities(tmp_path, "test")

    # Only two communities should be present (communities 2 and 5).
    assert len(set(c for communities in memberships.values() for c in communities)) == 2
    assert memberships[0] == {2}  # Node 0 in community 2 (line 2).
    assert memberships[1] == {2, 5}  # Node 1 in communities 2 and 5 (lines 2 and 5).
    assert memberships[2] == {5}  # Node 2 in community 5 (line 5).


def test_load_snap_community_node_id_map_is_not_sorted_order(tmp_path: Path) -> None:
    """Verifies the remap follows first-seen order, not sorted raw-ID order.

    A naive "sorted node IDs map contiguously" assumption (reconstructing the mapping
    from ``sorted(raw_ids)`` elsewhere, instead of using the loader's real mapping)
    would give a different -- wrong -- mapping here, since the raw IDs are not
    encountered in sorted order while reading the edge list.
    """
    # Raw IDs 5, 2, 9, 1 are first seen in that order, not sorted order (1, 2, 5, 9).
    edges = [(5, 2), (2, 9), (9, 1)]
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", edges)
    _write_community_file(tmp_path / "com-test.all.cmty.txt.gz", [[5], [9, 1]])

    graph = load_snap_community(tmp_path, "test")

    assert graph.node_id_map == {5: 0, 2: 1, 9: 2, 1: 3}
    # A sorted-order assumption would instead give {1: 0, 2: 1, 5: 2, 9: 3}.
    assert graph.node_id_map != {1: 0, 2: 1, 5: 2, 9: 3}


def test_remap_communities_uses_the_real_mapping_not_sorted_order(
    tmp_path: Path,
) -> None:
    """Verifies community membership lands on the loader's real internal node IDs."""
    edges = [(5, 2), (2, 9), (9, 1)]
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", edges)
    _write_community_file(tmp_path / "com-test.all.cmty.txt.gz", [[5], [9, 1]])

    graph = load_snap_community(tmp_path, "test")
    memberships = load_snap_communities(tmp_path, "test")
    community_of_5 = next(iter(memberships[5]))
    community_of_9_and_1 = next(iter(memberships[9]))

    remapped = remap_communities(graph.node_id_map, memberships)

    # Raw node 5 is internal node 0, not internal node 2 as a naive sorted-order
    # reconstruction of the mapping would place it.
    assert remapped[0] == {community_of_5}
    assert remapped[2] == {community_of_9_and_1}  # raw node 9 -> internal id 2.
    assert remapped[3] == {community_of_9_and_1}  # raw node 1 -> internal id 3.
    assert 1 not in remapped  # raw node 2 (internal id 1) is in no community.


def test_remap_communities_drops_ids_absent_from_the_node_id_map() -> None:
    """Verifies raw IDs with no graph edges (hence no internal ID) are dropped."""
    memberships = {10: {0}, 999: {0}}

    remapped = remap_communities({10: 0}, memberships)

    assert remapped == {0: {0}}


def test_primary_secondary_communities_counts_real_shared_edges(
    tmp_path: Path,
) -> None:
    """Verifies the primary community is picked by actual neighbor overlap, not luck."""
    # Node 0 is a bridge node: a member of community 0 ({0, 1, 2, 3}) and community 1
    # ({0, 4}). Of node 0's five neighbors, three (1, 2, 3) are members of community 0
    # and only one (4) is a member of community 1, so community 0 must win.
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", edges)
    _write_community_file(tmp_path / "com-test.all.cmty.txt.gz", [[0, 1, 2, 3], [0, 4]])

    graph = load_snap_community(tmp_path, "test")
    memberships = load_snap_communities(tmp_path, "test")
    community_a = next(iter(memberships[1]))  # {0, 1, 2, 3}'s community index.
    community_b = next(iter(memberships[4]))  # {0, 4}'s community index.
    remapped = remap_communities(graph.node_id_map, memberships)
    bridges = bridge_nodes(remapped, min_multiplicity=2)
    bridge_node = graph.node_id_map[0]

    assert bridges == {bridge_node}

    result = primary_secondary_communities(graph.adjacency, remapped, bridges)

    primary, secondary = result[bridge_node]
    assert primary == community_a
    assert secondary == {community_b}


def test_primary_secondary_communities_breaks_ties_by_smaller_index(
    tmp_path: Path,
) -> None:
    """Verifies a tied edge count is broken toward the smaller community index."""
    # Node 0 belongs to communities 0 and 1, each with exactly one qualifying
    # neighbor (1 and 2, respectively) -- an exact tie.
    edges = [(0, 1), (0, 2)]
    _write_edge_list(tmp_path / "com-test.ungraph.txt.gz", edges)
    _write_community_file(tmp_path / "com-test.all.cmty.txt.gz", [[0, 1], [0, 2]])

    graph = load_snap_community(tmp_path, "test")
    memberships = load_snap_communities(tmp_path, "test")
    community_a = next(iter(memberships[1]))  # {0, 1}'s community index (listed first).
    community_b = next(
        iter(memberships[2])
    )  # {0, 2}'s community index (listed second).
    remapped = remap_communities(graph.node_id_map, memberships)
    bridges = bridge_nodes(remapped, min_multiplicity=2)
    bridge_node = graph.node_id_map[0]

    result = primary_secondary_communities(graph.adjacency, remapped, bridges)

    # Both communities have exactly one qualifying neighbor; the tie is broken toward
    # the smaller community index, which is community_a's (it was listed first).
    assert result[bridge_node] == (community_a, {community_b})


def test_primary_secondary_communities_skips_bridges_absent_from_memberships() -> None:
    """A node ID with no recorded community membership is silently skipped."""
    adjacency = sp.csr_matrix(([1.0], ([0], [1])), shape=(2, 2))

    result = primary_secondary_communities(adjacency, {}, {0, 1})

    assert result == {}
