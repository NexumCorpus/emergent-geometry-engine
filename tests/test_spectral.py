"""Spectral dimension via the graph-Laplacian heat trace (Constitution sec 2.1).

d_s = -2 d ln P(t) / d ln t, with P(t) = (1/N) Tr exp(-tL) = (1/N) sum_i exp(-t lambda_i),
read off the intermediate-t plateau. Combinatorial Laplacian for regular graphs.
Sanity anchors (research-verified numerically): cycle C_n -> d_s ~ 1; 2D periodic
torus -> d_s ~ 2.
"""
import numpy as np
import networkx as nx
import pytest

from geometry.spectral import spectral_dimension, graph_spectral_dimension


def test_cycle_spectral_dimension_is_one():
    G = nx.cycle_graph(64)
    ds = graph_spectral_dimension(G, laplacian="combinatorial")
    assert abs(ds - 1.0) < 0.15


def test_torus_spectral_dimension_is_two():
    G = nx.grid_2d_graph(20, 20, periodic=True)
    ds = graph_spectral_dimension(G, laplacian="combinatorial")
    assert abs(ds - 2.0) < 0.3


def test_spectral_dimension_accepts_laplacian_matrix():
    # Direct Laplacian entry point: build C_64's combinatorial Laplacian by hand.
    n = 64
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[i, (i - 1) % n] = 1.0
    L = np.diag(A.sum(1)) - A
    ds = spectral_dimension(L)
    assert abs(ds - 1.0) < 0.15


def test_cycle_laplacian_eigenvalues_are_analytic():
    # Independent correctness check on the Laplacian itself: C_n eigenvalues are
    # 2 - 2 cos(2 pi k / n). (This pins that the Laplacian is built correctly.)
    n = 32
    G = nx.cycle_graph(n)
    L = nx.laplacian_matrix(G).toarray().astype(float)
    w = np.sort(np.linalg.eigvalsh(L))
    analytic = np.sort([2 - 2 * np.cos(2 * np.pi * k / n) for k in range(n)])
    assert np.allclose(w, analytic, atol=1e-9)
