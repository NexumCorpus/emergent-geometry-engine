"""Discrete Ricci curvature kernel (Constitution sec 2.1).

Graders for this module are INDEPENDENT of our implementation:
  - Forman: cross-checked against the GraphRicciCurvature library (different codebase).
  - Ollivier/LLY: cross-checked against closed-form analytic values (K_n, Q_n) AND
    against the verifier's spectral gap (the curvature<->spectral-gap tie, sec 5.2).
"""
import numpy as np
import networkx as nx
import pytest

from geometry.curvature import forman_ricci, ollivier_ricci, lly_curvature
from verify.spectral_gap import spectral_gap


def _triangle_with_tail():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3)])
    return G


def test_forman_1d_closed_form():
    G = _triangle_with_tail()
    f = forman_ricci(G, method="1d")
    assert abs(f[(0, 1)] - 0.0) < 1e-12   # 4 - deg(0) - deg(1) = 4-2-2
    assert abs(f[(2, 3)] - 0.0) < 1e-12   # 4 - 3 - 1


def test_forman_augmented_closed_form():
    G = _triangle_with_tail()
    f = forman_ricci(G, method="augmented")
    assert abs(f[(0, 1)] - 3.0) < 1e-12   # 3*T + 4 - deg(0) - deg(1), T=1
    assert abs(f[(2, 3)] - 0.0) < 1e-12   # T=0


def test_forman_augmented_matches_library():
    # Independent grader: the GraphRicciCurvature library (separate implementation).
    # Installed 0.4.5 computes the AUGMENTED Forman variant by default (verified: its
    # output matches our 3T+4-deg-deg formula on every karate-club edge).
    from GraphRicciCurvature.FormanRicci import FormanRicci

    G = nx.karate_club_graph()
    mine = forman_ricci(G, method="augmented")
    frc = FormanRicci(G.copy(), verbose="ERROR")
    frc.compute_ricci_curvature()
    for u, v in G.edges():
        key = (u, v) if (u, v) in mine else (v, u)
        assert abs(mine[key] - frc.G[u][v]["formanCurvature"]) < 1e-9


def test_ollivier_K5_alpha_half():
    # Analytic anchor: fixed-alpha (idleness 0.5) Ollivier curvature of K_5 is 0.625.
    G = nx.complete_graph(5)
    orc = ollivier_ricci(G, alpha=0.5)
    for e in G.edges():
        assert abs(orc[e] - 0.625) < 1e-6


def test_lly_K5_is_tight_curvature():
    # LLY curvature of K_n is n/(n-1); for K_5 = 1.25.
    G = nx.complete_graph(5)
    lly = lly_curvature(G)
    for e in G.edges():
        assert abs(lly[e] - 1.25) < 1e-4


def test_lly_hypercube_Q3():
    # LLY curvature of Q_n is 2/n; for Q_3 = 2/3.
    G = nx.hypercube_graph(3)
    lly = lly_curvature(G)
    for e in G.edges():
        assert abs(lly[e] - 2.0 / 3.0) < 1e-4


def test_lly_min_equals_spectral_gap_on_K5():
    # The Track-B claim made concrete: min LLY curvature == lambda_1 of the normalized
    # Laplacian (curvature is the upstream object). Computed by two separate code paths.
    G = nx.complete_graph(5)
    kappa_min = min(lly_curvature(G).values())
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    lam1 = spectral_gap(A)
    assert abs(kappa_min - lam1) < 1e-3
    assert lam1 >= kappa_min - 1e-9   # the Lichnerowicz bound holds on real data
