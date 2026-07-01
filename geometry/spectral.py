"""Spectral dimension from the graph-Laplacian heat trace (Constitution sec 2.1).

The spectral dimension is read off the heat-kernel return probability

    P(t) = (1/N) Tr exp(-t L) = (1/N) sum_i exp(-t lambda_i),

via the running logarithmic slope

    d_s(t) = -2 d ln P(t) / d ln t,

evaluated on the intermediate-t plateau (the median of the central portion of the
running estimate). Anchored numerically: cycle C_n -> d_s ~ 1, 2D periodic torus
-> d_s ~ 2. The plateau-median estimator (rather than a single global linear fit)
is what keeps the 2D case from overshooting.
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def spectral_dimension(L) -> float:
    """Spectral dimension of a graph-Laplacian matrix L via the heat-trace slope.

    Parameters
    ----------
    L : array-like, shape (N, N)
        A symmetric graph Laplacian (combinatorial D - A or a normalized variant).

    Returns
    -------
    float
        The plateau-median estimate of the spectral dimension.
    """
    L = np.asarray(L, dtype=float)
    w = np.linalg.eigvalsh(L)
    N = len(w)

    # Spectral gap (smallest nonzero eigenvalue) and largest eigenvalue set the
    # diffusion-time window: short times resolve the spectral top, long times the
    # gap. Below/above this window the slope is contaminated by lattice/finite-size
    # effects, so we sample only the intermediate regime.
    gap = np.sort(w[w > 1e-9])[0]
    lam_max = w.max()
    t_lo, t_hi = 5.0 / lam_max, 0.1 / gap
    if t_hi <= 3 * t_lo:  # tiny-graph guard: ensure a usable window
        t_hi = 10 * t_lo

    ts = np.logspace(np.log10(t_lo), np.log10(t_hi), 60)
    lnP = np.log(np.array([np.mean(np.exp(-t * w)) for t in ts]))
    lnt = np.log(ts)

    ds_point = -2 * np.gradient(lnP, lnt)
    plateau = ds_point[len(ds_point) // 4 : 3 * len(ds_point) // 4]
    return float(np.median(plateau))


def graph_spectral_dimension(G: nx.Graph, laplacian: str = "combinatorial") -> float:
    """Spectral dimension of a networkx graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph. Node labels may be arbitrary (e.g. tuples from grid_2d_graph);
        nx.laplacian_matrix handles arbitrary labels.
    laplacian : {"combinatorial", "normalized"}
        "combinatorial" -> L = D - A. "normalized" -> the symmetric normalized
        Laplacian I - D^{-1/2} A D^{-1/2}.

    Returns
    -------
    float
        The spectral dimension (see ``spectral_dimension``).
    """
    if laplacian == "combinatorial":
        L = nx.laplacian_matrix(G).toarray().astype(float)
    elif laplacian == "normalized":
        L = nx.normalized_laplacian_matrix(G).toarray().astype(float)
    else:
        raise ValueError(
            f"unknown laplacian={laplacian!r}; expected 'combinatorial' or 'normalized'"
        )
    return spectral_dimension(L)
