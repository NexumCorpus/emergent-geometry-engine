"""Discrete Ricci curvature: Forman (cheap, per-step workhorse) and Ollivier/
Lin-Lu-Yau (tied to the spectral-gap bound, used at anchor checkpoints).

Ollivier is implemented from scratch via optimal transport (POT), deliberately NOT
through GraphRicciCurvature.OllivierRicci — that path hard-codes a 'fork'
multiprocessing context which raises on Windows. Rolling our own also keeps the
library free to act as an independent grader for the Forman path.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import ot

from geometry.config import OLLIVIER_ALPHA, LLY_ALPHA_NEAR_ONE


def forman_ricci(G: nx.Graph, method: str = "augmented") -> dict:
    """Per-edge Forman-Ricci curvature (unweighted), matching GraphRicciCurvature.

    1d:        F(e) = 4 - deg(u) - deg(v)
    augmented: F(e) = 3*T + 4 - deg(u) - deg(v)   with T = #common neighbours of u,v
    """
    if method not in ("1d", "augmented"):
        raise ValueError("method must be '1d' or 'augmented'")
    deg = dict(G.degree())
    out = {}
    for u, v in G.edges():
        base = 4 - deg[u] - deg[v]
        if method == "augmented":
            t = len(list(nx.common_neighbors(G, u, v)))
            out[(u, v)] = float(3 * t + base)
        else:
            out[(u, v)] = float(base)
    return out


def _all_pairs_hops(G: nx.Graph) -> dict:
    return dict(nx.all_pairs_shortest_path_length(G))


def _lazy_measure(G: nx.Graph, x, alpha: float):
    """alpha-lazy random-walk measure m_x^alpha: mass alpha at x, (1-alpha)/deg
    spread uniformly over neighbours."""
    nbrs = list(G.neighbors(x))
    deg = len(nbrs)
    support = [x] + nbrs
    mass = np.array([alpha] + [(1.0 - alpha) / deg] * deg, dtype=float)
    return support, mass


def ollivier_ricci_edge(G: nx.Graph, x, y, alpha: float, dist: dict) -> float:
    """kappa_alpha(x,y) = 1 - W1(m_x^alpha, m_y^alpha)/d(x,y), W1 the L1 Wasserstein
    distance computed by exact optimal transport."""
    sx, mx = _lazy_measure(G, x, alpha)
    sy, my = _lazy_measure(G, y, alpha)
    cost = np.array([[dist[a][b] for b in sy] for a in sx], dtype=float)
    w1 = ot.emd2(mx, my, cost)
    return 1.0 - w1 / dist[x][y]


def ollivier_ricci(G: nx.Graph, alpha: float = OLLIVIER_ALPHA) -> dict:
    """Per-edge fixed-idleness Ollivier-Ricci curvature."""
    dist = _all_pairs_hops(G)
    return {(u, v): ollivier_ricci_edge(G, u, v, alpha, dist) for u, v in G.edges()}


def lly_curvature(G: nx.Graph, alpha_near_one: float = LLY_ALPHA_NEAR_ONE) -> dict:
    """Per-edge Lin-Lu-Yau curvature, the alpha->1 derivative limit
    kappa_LLY = lim_{alpha->1} kappa_alpha / (1 - alpha), estimated at alpha near 1.
    This is the curvature tied to the spectral-gap (Lichnerowicz) bound."""
    dist = _all_pairs_hops(G)
    scale = 1.0 - alpha_near_one
    return {
        (u, v): ollivier_ricci_edge(G, u, v, alpha_near_one, dist) / scale
        for u, v in G.edges()
    }
