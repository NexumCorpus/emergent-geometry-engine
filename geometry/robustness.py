"""Graph-construction robustness harness (Constitution sec 2.2, 3).

Graph curvature/delta are notoriously sensitive to how the graph is built. This is
the referee's main attack, so the harness re-runs the CHEAP invariants (Forman,
Gromov delta, spectral dimension) across a grid of (k, alpha) construction choices
and reports how stable each invariant is (coefficient of variation across configs).

This is a kernel self-diagnostic — it imports the kernel deliberately. It is NOT the
physics grader (that lives in verify/ and never touches the kernel).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import networkx as nx

from geometry.graph import build_knn_graph, prune_shortcut_edges
from geometry.curvature import forman_ricci, ollivier_ricci
from geometry.delta import gromov_delta
from geometry.spectral import graph_spectral_dimension


@dataclass
class SweepResult:
    configs: list
    stability: dict


def _largest_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or nx.is_connected(G):
        return G
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes).copy()


def _invariants(G: nx.Graph) -> dict:
    forman = forman_ricci(G, method="augmented")
    mean_forman = float(np.mean(list(forman.values()))) if forman else 0.0
    core = _largest_component(G)
    delta = gromov_delta(core) if core.number_of_nodes() >= 4 else 0.0
    sdim = graph_spectral_dimension(core) if core.number_of_nodes() >= 3 else 0.0
    return {
        "mean_forman": mean_forman,
        "delta": float(delta),
        "spectral_dim": float(sdim),
        "n_edges": G.number_of_edges(),
    }


def robustness_sweep(
    D: np.ndarray,
    k_values=(5, 8, 10),
    alpha_values=(0.0, 0.5),
    prune: bool = True,
) -> SweepResult:
    """Sweep (k, alpha) graph-build choices and report invariant stability.

    For each config: build the kNN graph at k, optionally prune shortcut edges using
    Ollivier curvature at the given idleness alpha, then compute the cheap invariants.
    Stability is reported per invariant as mean / std / coefficient-of-variation
    across all configs.
    """
    configs = []
    for k in k_values:
        for alpha in alpha_values:
            G = build_knn_graph(D, k=k)
            if prune:
                G, _ = prune_shortcut_edges(
                    G, curvature_fn=lambda g, a=alpha: ollivier_ricci(g, alpha=a)
                )
            inv = _invariants(G)
            inv.update({"k": int(k), "alpha": float(alpha)})
            configs.append(inv)

    stability = {}
    for key in ("mean_forman", "delta", "spectral_dim"):
        vals = np.array([c[key] for c in configs], dtype=float)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cov = float(std / abs(mean)) if abs(mean) > 1e-12 else float("inf") if std > 1e-12 else 0.0
        stability[key] = {"mean": mean, "std": std, "cov": cov}

    return SweepResult(configs=configs, stability=stability)
