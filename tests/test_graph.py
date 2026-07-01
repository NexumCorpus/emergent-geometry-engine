"""Graph construction: kNN build + shortcut-edge pruning (Constitution sec 2.1,
robustness fixes #2 and #3).

Pruning rule (ORC-MANL style): an edge with extreme-negative Ollivier curvature is a
shortcut candidate; remove it ONLY if its endpoints remain within a hop threshold in
the pruned graph (i.e. a redundant parallel path exists). A true bridge — whose
removal disconnects or pushes the endpoints far — is kept.
"""
import numpy as np
import networkx as nx
import pytest

from geometry.graph import (
    zero_diagonal,
    similarity_to_distance,
    build_knn_graph,
    prune_shortcut_edges,
)


def _circle_distance_matrix(n: int) -> np.ndarray:
    ang = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(ang), np.sin(ang)])
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return D


def test_zero_diagonal():
    M = np.array([[5.0, 1.0], [1.0, 5.0]])
    Z = zero_diagonal(M)
    assert Z[0, 0] == 0.0 and Z[1, 1] == 0.0
    assert Z[0, 1] == 1.0           # off-diagonal untouched
    assert M[0, 0] == 5.0           # original not mutated


def test_similarity_to_distance_monotone():
    S = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    D = similarity_to_distance(S)
    assert np.allclose(np.diag(D), 0.0)
    # higher similarity -> smaller distance
    assert D[0, 1] < D[0, 2]


def test_build_knn_ring():
    D = _circle_distance_matrix(8)
    G = build_knn_graph(D, k=2)
    assert nx.is_connected(G)
    assert G.number_of_nodes() == 8
    # 8 points on a circle, k=2 nearest -> the ring C_8 (every node degree 2).
    assert all(deg == 2 for _, deg in G.degree())
    assert G.number_of_edges() == 8


def test_knn_no_self_loops_and_symmetric():
    D = _circle_distance_matrix(10)
    G = build_knn_graph(D, k=3)
    assert not any(u == v for u, v in G.edges())
    assert isinstance(G, nx.Graph)  # undirected => symmetric


def _two_cliques(bridges):
    G = nx.Graph()
    G.add_edges_from((i, j) for i in range(4) for j in range(i + 1, 4))
    G.add_edges_from((i, j) for i in range(4, 8) for j in range(i + 1, 8))
    G.add_edges_from(bridges)
    return G


def test_prune_keeps_single_bridge():
    # One inter-clique edge: removing it disconnects -> must be KEPT.
    G = _two_cliques([(3, 4)])
    pruned, removed = prune_shortcut_edges(G)
    assert nx.is_connected(pruned)
    assert (3, 4) in pruned.edges() or (4, 3) in pruned.edges()
    assert (3, 4) not in removed and (4, 3) not in removed


def test_prune_removes_one_redundant_parallel_bridge():
    # Two inter-clique edges: one is redundant (parallel path exists) -> removed;
    # the graph must stay connected and keep exactly one of the two bridges.
    G = _two_cliques([(3, 4), (2, 5)])
    pruned, removed = prune_shortcut_edges(G)
    assert nx.is_connected(pruned)
    bridges_left = [e for e in [(3, 4), (2, 5)] if pruned.has_edge(*e)]
    assert len(bridges_left) == 1
    assert len(removed) == 1
