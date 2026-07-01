"""Gromov delta-hyperbolicity via the 4-point condition (Constitution sec 2.1).

This module computes the (Gromov) delta-hyperbolicity of a graph using the
standard SageMath / research-verified 4-point condition.

Convention -- the ``(M1 - M2) / 2`` rule
----------------------------------------
For every unordered 4-tuple of vertices ``{a, b, c, d}`` we form the three
pairwise-distance sums of the perfect matchings::

    S1 = D[a, b] + D[c, d]
    S2 = D[a, c] + D[b, d]
    S3 = D[a, d] + D[b, c]

We sort these so that ``M1 >= M2 >= M3``. The hyperbolicity contributed by the
tuple is the half-difference of the two largest sums::

    delta(a, b, c, d) = (M1 - M2) / 2

The graph's delta-hyperbolicity is the maximum of this quantity over all
4-tuples::

    delta(G) = max_{a,b,c,d} (M1 - M2) / 2

This is always ``>= 0``. Trees (and paths) are 0-hyperbolic, complete graphs
are 0-hyperbolic (all distances equal so all three sums coincide), and the
4-cycle ``C_4`` has delta ``1``.

The all-pairs shortest-path step is ``O(n^3)`` (Floyd-Warshall) and the 4-tuple
enumeration is ``O(n^4)``; both are fine for the small graphs used here.
"""

from itertools import combinations

import numpy as np
import networkx as nx

__all__ = ["gromov_delta", "gromov_delta_from_distances"]


def gromov_delta_from_distances(D):
    """Gromov delta-hyperbolicity from an ``N x N`` distance matrix.

    Parameters
    ----------
    D : array_like, shape (n, n)
        A symmetric matrix of pairwise (shortest-path) distances.

    Returns
    -------
    float
        ``delta(G) = max over all 4-tuples of (M1 - M2) / 2``, where ``M1 >= M2``
        are the two largest of the three matching-sum distances for the tuple.
        Always ``>= 0``.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]

    if n < 4:
        # Fewer than 4 vertices: no 4-tuple exists, so delta is 0.
        return 0.0

    best = 0.0
    for a, b, c, d in combinations(range(n), 4):
        s1 = D[a, b] + D[c, d]
        s2 = D[a, c] + D[b, d]
        s3 = D[a, d] + D[b, c]

        # Two largest of the three sums (M1 >= M2 >= M3).
        m1, m2, m3 = sorted((s1, s2, s3), reverse=True)
        delta = (m1 - m2) / 2.0
        if delta > best:
            best = delta

    return float(best)


def gromov_delta(G):
    """Gromov delta-hyperbolicity of a networkx graph.

    Computes all-pairs shortest-path distances via ``nx.floyd_warshall_numpy``
    and applies the ``(M1 - M2) / 2`` 4-point condition (see module docstring).

    Parameters
    ----------
    G : networkx.Graph
        The graph. Assumed connected for a finite delta; isolated components
        would yield infinite distances.

    Returns
    -------
    float
        The graph's delta-hyperbolicity, ``>= 0``.
    """
    D = nx.floyd_warshall_numpy(G)
    return gromov_delta_from_distances(D)
