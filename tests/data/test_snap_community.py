"""Tests for SNAP community graph loading and community membership parsing."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from tnbbeta_vae.data.snap_community import (
    bridge_nodes,
    load_snap_communities,
    load_snap_community,
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
