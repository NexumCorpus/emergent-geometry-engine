"""Adapter Q: mutual-information matrix -> per-trajectory graph geometry.

Constitution sec 4.2 crux: the MIPT lives in the trajectory ensemble, so every
observable here is computed on ONE trajectory's MI matrix; averaging happens
downstream in the sweep driver. Annealed averaging of MI matrices is suspect
and is deliberately not offered.

Pre-registered choices (robustness fix #1 — frozen BEFORE any sweep result):

- MI_FLOOR = 1e-9 bits. Pairs with MI <= MI_FLOOR are NON-edges. The generic
  kNN builder would connect zero-MI pairs through argsort ties at maximal
  distance, faking connectivity exactly when measurement fragments the state;
  fragmentation is signal, so it must survive into the graph.
- kNN degree cap KNN_K (frozen kernel recipe) applied among genuine
  (MI > floor) candidates only; union-symmetric, edge weight = distance.
- Shortcut pruning: the kernel's frozen ORC-MANL recipe, unchanged.
- Global observables on the LARGEST CONNECTED COMPONENT, with lcc_frac
  recorded so no fragmentation is hidden. Forman (augmented) is the
  per-trajectory workhorse (fix #4); ORC runs only inside the pruning recipe.
- Global metric = HOP metric (unweighted shortest path), matching the kernel's
  delta convention and unweighted Forman. MI-distances rank kNN candidates but
  are NOT a metric: maximal-similarity pairs sit at distance 0 (Bell pair,
  GHZ), which would collapse any weighted Laplacian / path length to zero.
- Degenerate trajectories (no edges / LCC < 2) return NaN observables with
  lcc_frac = 1/n — an honest record, never an exception.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from geometry.config import KNN_K
from geometry.curvature import forman_ricci
from geometry.delta import gromov_delta_from_distances
from geometry.embedding import (avg_distortion, euclidean_mds,
                                hmds, hyperboloid_distances)
from geometry.graph import prune_shortcut_edges, similarity_to_distance
from geometry.spectral import graph_spectral_dimension

MI_FLOOR = 1e-9          # bits; at or below this a pair is a NON-edge


def mi_graph(M: np.ndarray) -> nx.Graph:
    """kNN graph over an MI matrix, connecting only genuinely-correlated pairs."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if not (M > MI_FLOOR).any():
        return G                                  # no structure at all
    D = similarity_to_distance(M)
    for i in range(n):
        cand = [j for j in np.argsort(D[i]) if j != i and M[i, j] > MI_FLOOR]
        for j in cand[:KNN_K]:
            G.add_edge(i, int(j), weight=float(D[i, j]))
    return G


_NAN_KEYS = ("forman_mean", "forman_min", "delta", "delta_norm",
             "spectral_dim", "hmds_distortion", "mds_stress")


def trajectory_observables(M: np.ndarray) -> dict:
    """All Phase 3 geometric observables for ONE trajectory's MI matrix."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    obs = {"n": n, "mean_mi": float(M[iu].mean())}

    G = mi_graph(M)
    if G.number_of_edges() == 0:
        obs.update({k: float("nan") for k in _NAN_KEYS})
        obs["lcc_frac"] = 1.0 / n
        return obs

    G, _removed = prune_shortcut_edges(G)         # frozen kernel recipe
    comps = list(nx.connected_components(G))
    lcc = max(comps, key=len)
    obs["lcc_frac"] = len(lcc) / n
    if len(lcc) < 2:
        obs.update({k: float("nan") for k in _NAN_KEYS})
        return obs
    # Unweighted view of the LCC: hop metric everywhere downstream.
    H = nx.Graph()
    H.add_nodes_from(lcc)
    H.add_edges_from(G.subgraph(lcc).edges())

    # Forman (augmented) — the cheap per-trajectory curvature (fix #4).
    f = forman_ricci(H, method="augmented")
    vals = np.array(list(f.values()), dtype=float)
    obs["forman_mean"] = float(vals.mean())
    obs["forman_min"] = float(vals.min())

    # Hop-metric global geometry on the LCC.
    Dh = np.asarray(nx.floyd_warshall_numpy(H), dtype=float)
    diam = float(Dh.max())
    obs["delta"] = gromov_delta_from_distances(Dh)
    obs["delta_norm"] = obs["delta"] / diam if diam > 0 else 0.0
    obs["spectral_dim"] = graph_spectral_dimension(H)

    # Q2: hyperbolic vs Euclidean fit quality of the emergent metric.
    X = hmds(Dh, rank=2)
    obs["hmds_distortion"] = avg_distortion(Dh, hyperboloid_distances(X)) \
        if diam > 0 else float("nan")
    _, stress = euclidean_mds(Dh, dim=2)
    obs["mds_stress"] = float(stress)
    return obs
