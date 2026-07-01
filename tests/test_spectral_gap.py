"""Spectral-gap >= curvature check: the discrete Lichnerowicz bound (Constitution
sec 2.2, 5.1). Lin-Lu-Yau (Tohoku 2011, Thm 4.2): if the LLY Ricci curvature
kappa(x,y) >= k > 0 on every edge, then the first nonzero eigenvalue lambda_1 of the
NORMALIZED Laplacian L = I - D^{-1/2} A D^{-1/2} satisfies lambda_1 >= k.

Closed-form LLY curvatures (research-confirmed): K_n -> n/(n-1), Q_n -> 2/n (both
TIGHT), C_5 -> 1/2 (strict), Petersen -> 0 (loose). The tightness tests on K_n/Q_n
are what actually pin down that the NORMALIZED Laplacian is used: the combinatorial
Laplacian gives lambda_1 = 5 for K_5 (!= 1.25), which the negative-control test
exploits.
"""
import numpy as np
import networkx as nx
import pytest

from verify.spectral_gap import spectral_gap, check_lichnerowicz


def _adj(G) -> np.ndarray:
    return nx.to_numpy_array(G, nodelist=sorted(G.nodes()))


def test_normalized_gap_complete_graph_is_tight():
    # K_5: lambda_1(normalized) = n/(n-1) = 1.25 exactly.
    A = _adj(nx.complete_graph(5))
    assert abs(spectral_gap(A) - 1.25) < 1e-9


def test_normalized_gap_hypercube_is_tight():
    # Q_3: lambda_1(normalized) = 2/n = 2/3.
    A = _adj(nx.hypercube_graph(3))
    assert abs(spectral_gap(A) - 2.0 / 3.0) < 1e-9


def test_lichnerowicz_tight_on_K5():
    A = _adj(nx.complete_graph(5))
    res = check_lichnerowicz(A, kappa_min=5.0 / 4.0, expect_tight=True)
    assert res.ok
    assert abs(res.slack) < 1e-9  # equality


def test_lichnerowicz_tight_on_Q3():
    A = _adj(nx.hypercube_graph(3))
    res = check_lichnerowicz(A, kappa_min=2.0 / 3.0, expect_tight=True)
    assert res.ok
    assert abs(res.slack) < 1e-9


def test_lichnerowicz_strict_on_C5():
    # C_5: kappa = 1/2, lambda_1 = 1 - cos(72deg) ~ 0.691 -> strict (positive slack).
    A = _adj(nx.cycle_graph(5))
    res = check_lichnerowicz(A, kappa_min=0.5, expect_tight=False)
    assert res.ok
    assert res.slack > 0.1


def test_lichnerowicz_loose_on_petersen():
    # Petersen: kappa = 0 everywhere; bound 0 <= 2/3 holds but is loose.
    A = _adj(nx.petersen_graph())
    res = check_lichnerowicz(A, kappa_min=0.0, expect_tight=False)
    assert res.ok


def test_combinatorial_laplacian_breaks_tightness_negative_control():
    # The research-flagged bug: using the COMBINATORIAL Laplacian gives lambda_1 = 5
    # for K_5, which is NOT the curvature 1.25. The normalized gap is the right one;
    # this confirms our tightness test would catch a wrong-Laplacian implementation.
    A = _adj(nx.complete_graph(5))
    lam_comb = spectral_gap(A, normalized=False)
    assert abs(lam_comb - 5.0) < 1e-9
    assert abs(lam_comb - 1.25) > 1.0  # would fail an equality-to-curvature test


def test_spectral_gap_rejects_disconnected_graph():
    A = _adj(nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3)))
    with pytest.raises(ValueError):
        spectral_gap(A)
