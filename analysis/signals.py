"""Early-warning / order-parameter signals for the grokking head-to-head.

Two families:
  * 2RDM spectral heat capacity (Hennick-Corlouer arXiv:2603.29805) — the competitor.
    C = n x n covariance of per-sample LOSSES across a checkpoint window; SHC = Var of
    its raw eigenvalues (NOT a thermodynamic quantity); PR = (tr C)^2 / tr(C^2).
  * The geometric order parameter — mean Forman curvature + spectral dimension of the
    cosine-similarity graph of a representation (reuses the certified kernel).
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from adapters.neural import cosine_similarity_matrix
from geometry.graph import build_knn_graph, similarity_to_distance
from geometry.curvature import forman_ricci
from geometry.delta import gromov_delta
from geometry.spectral import graph_spectral_dimension


# --- 2RDM spectral heat capacity (competitor signal) ---

def _loss_covariance(loss_window: np.ndarray) -> np.ndarray:
    """n x n sample covariance over the W checkpoint snapshots in the window.

    loss_window: (W, n) array of per-sample losses (W checkpoints, n probe samples).
    """
    L = np.asarray(loss_window, dtype=float)
    return np.atleast_2d(np.cov(L, rowvar=False))


def spectral_heat_capacity(loss_window: np.ndarray) -> float:
    """SHC = Var({lambda_k}) of the eigenvalues of the loss covariance C.

    A scalar that spikes when loss fluctuations concentrate into a single dominant
    mode (a forming circuit). Variance is shift-invariant, so no regularization is
    needed for the value itself.
    """
    lam = np.linalg.eigvalsh(_loss_covariance(loss_window))
    return float(np.var(lam))


def participation_ratio(loss_window: np.ndarray) -> float:
    """PR = (sum lambda)^2 / sum(lambda^2), in [1, n]; how many modes participate."""
    lam = np.linalg.eigvalsh(_loss_covariance(loss_window))
    s1 = float(lam.sum())
    s2 = float((lam ** 2).sum())
    return s1 * s1 / s2 if s2 > 0 else 0.0


# --- geometric order parameter (our signal) ---

def _largest_component(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0 or nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def embedding_graph_signal(features: np.ndarray, k: int = 8) -> dict:
    """Geometric invariants of the cosine-similarity graph of a representation.

    features: (n_nodes, d) — e.g. the p token-embedding rows W_E at one checkpoint.
    Uses the cheap per-checkpoint invariants (Forman + Gromov delta + spectral dim);
    exact Ollivier/LLY is reserved for anchor checkpoints. Frozen recipe across calls.
    """
    S = cosine_similarity_matrix(features)
    D = similarity_to_distance(S)
    G = build_knn_graph(D, k=k)

    forman = forman_ricci(G, method="augmented")
    fv = np.array(list(forman.values())) if forman else np.array([0.0])
    core = _largest_component(G)
    delta = gromov_delta(core) if core.number_of_nodes() >= 4 else 0.0
    sdim = graph_spectral_dimension(core) if core.number_of_nodes() >= 3 else 0.0
    return {
        "mean_forman": float(np.mean(fv)),
        "var_forman": float(np.var(fv)),
        "gromov_delta": float(delta),
        "spectral_dim": float(sdim),
        "n_edges": G.number_of_edges(),
    }
