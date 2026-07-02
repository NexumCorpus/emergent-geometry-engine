"""Q2 like-for-like embedding comparison (POST-HOC EXPLORATORY — 2026-07-02).

The Phase 3 report could not decide Q2's hyperbolic-vs-flat reading because
h-MDS distortion and Euclidean stress are different functionals. Here both
embeddings are scored with the SAME functional (avg_distortion, mean relative
error over pairs) on the same per-trajectory LCC hop metric:

  - Euclidean: classical MDS, dim 2. avg_distortion is scale-invariant here
    (scaling D scales the embedding linearly), so no scale search is needed.
  - Hyperbolic: h-MDS, rank 2, with a per-trajectory scale search
    tau in {0.25, 0.5, 1, 2, 4} (h-MDS is not scale-invariant; Sala et al.
    sec 4 — the search gives hyperbolic its natural scale, best-of reported).

Fresh trajectories (base seed 777 — NOT the sweep's), so this is an
independent look, but the analysis is post-hoc: it informs Q2 language in the
report and is labeled exploratory there; it feeds no certified claim.

Usage: python -m experiments.mipt.q2_embedding
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from adapters.quantum import mi_graph
from experiments.mipt.circuits import run_trajectory
from experiments.mipt.stabilizer import block_mi_matrix
from geometry.embedding import (avg_distortion, euclidean_mds,
                                hmds, hyperboloid_distances)
from geometry.graph import prune_shortcut_edges

BASE_SEED = 777
NTRAJ = 100
TAUS = (0.25, 0.5, 1.0, 2.0, 4.0)


def lcc_hop_matrix(sim):
    G = mi_graph(block_mi_matrix(sim, 4))
    if G.number_of_edges() == 0:
        return None
    G, _ = prune_shortcut_edges(G)
    lcc = max(nx.connected_components(G), key=len)
    if len(lcc) < 4:
        return None
    H = nx.Graph()
    H.add_nodes_from(lcc)
    H.add_edges_from(G.subgraph(lcc).edges())
    return np.asarray(nx.floyd_warshall_numpy(H), dtype=float)


def both_distortions(Dh):
    coords, _ = euclidean_mds(Dh, dim=2)
    Demb = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    e = avg_distortion(Dh, Demb)
    h = min(avg_distortion(tau * Dh,
                           hyperboloid_distances(hmds(tau * Dh, rank=2))) / 1.0
            for tau in TAUS)
    return e, h


def main():
    print(f"# Q2 like-for-like: avg_distortion, Euclid dim2 vs h-MDS rank2 "
          f"(tau best-of {TAUS}), ntraj={NTRAJ}, base_seed={BASE_SEED}")
    for L in (32, 48, 64):
        for ip, p in enumerate((0.08, 0.12, 0.16)):
            es, hs = [], []
            for k in range(NTRAJ):
                seed = int(np.random.SeedSequence(
                    [BASE_SEED, L, ip, k]).generate_state(1)[0])
                Dh = lcc_hop_matrix(run_trajectory(L=L, p=p, T=2 * L, seed=seed))
                if Dh is None:
                    continue
                e, h = both_distortions(Dh)
                es.append(e)
                hs.append(h)
            es, hs = np.array(es), np.array(hs)
            d = es - hs                     # >0 means hyperbolic fits better
            sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"L={L} p={p:.2f} n={len(es):3d} | "
                  f"euclid={es.mean():.4f} hyper={hs.mean():.4f} | "
                  f"euclid-hyper={d.mean():+.4f} +/- {sem:.4f}")


if __name__ == "__main__":
    main()
