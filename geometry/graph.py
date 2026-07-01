"""Graph construction from a similarity/distance matrix.

Robustness fix #2: zero the dissimilarity diagonal (a nonzero diagonal fakes
structure). Robustness fix #3: build a kNN graph and prune extreme-negative-curvature
shortcut edges BEFORE computing global curvature / delta (shortcut edges fake the
geometry).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from geometry.config import KNN_K, SHORTCUT_CURVATURE, SHORTCUT_DISTANCE_THRESHOLD
from geometry.curvature import ollivier_ricci


def zero_diagonal(M: np.ndarray) -> np.ndarray:
    M = np.array(M, dtype=float, copy=True)
    np.fill_diagonal(M, 0.0)
    return M


def similarity_to_distance(S: np.ndarray) -> np.ndarray:
    """Monotone-decreasing map from a similarity matrix to a distance matrix,
    normalized so the largest off-diagonal similarity maps to distance 0 and the
    diagonal is exactly 0."""
    S = np.asarray(S, dtype=float)
    off = S - np.diag(np.diag(S))
    smax = off.max()
    if smax <= 0:
        raise ValueError("similarity matrix has no positive off-diagonal entries")
    D = 1.0 - off / smax
    np.fill_diagonal(D, 0.0)
    return D


def build_knn_graph(D: np.ndarray, k: int = KNN_K) -> nx.Graph:
    """Symmetric (union) k-nearest-neighbour graph from a distance matrix D.

    Each node connects to its k nearest others; an edge is kept if either endpoint
    has the other in its kNN set. Edge weight is the distance. Diagonal is ignored.
    """
    D = zero_diagonal(D)
    n = D.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be < n={n}")
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        order = np.argsort(D[i])
        nbrs = [j for j in order if j != i][:k]
        for j in nbrs:
            G.add_edge(i, int(j), weight=float(D[i, j]))
    return G


def prune_shortcut_edges(
    G: nx.Graph,
    curv_threshold: float = SHORTCUT_CURVATURE,
    dist_threshold: int = SHORTCUT_DISTANCE_THRESHOLD,
    curvature_fn=ollivier_ricci,
):
    """Remove redundant negative-curvature shortcut edges (ORC-MANL style).

    Candidates are edges with Ollivier curvature < curv_threshold, processed
    most-negative first. A candidate is removed only if, after its removal, its
    endpoints remain connected within dist_threshold hops (a parallel path exists).
    Bridges — whose removal disconnects the endpoints — are kept.

    Returns (pruned_graph, removed_edges).
    """
    H = G.copy()
    curv = curvature_fn(H)
    candidates = sorted(
        (e for e, c in curv.items() if c < curv_threshold),
        key=lambda e: curv[e],
    )
    removed = []
    for u, v in candidates:
        if not H.has_edge(u, v):
            continue
        H.remove_edge(u, v)
        try:
            d = nx.shortest_path_length(H, u, v)
            redundant = d <= dist_threshold
        except nx.NetworkXNoPath:
            redundant = False
        if redundant:
            removed.append((u, v))
        else:
            H.add_edge(u, v, **G.get_edge_data(u, v))  # restore the bridge
    return H, removed
