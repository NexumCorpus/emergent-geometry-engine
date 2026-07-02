"""Adapter Q: per-trajectory MI-graph geometry (Phase 3, Constitution sec 4.2).

Anchors are exactly-known stabilizer states whose MI structure is derived by
hand beside each assert. The degenerate cases (product state, fragmented
Bell-pair chain) are first-class: measurement-induced fragmentation is signal,
and the adapter must report it honestly rather than crash or fake connectivity.
"""
import numpy as np
import pytest

from adapters.quantum import MI_FLOOR, mi_graph, trajectory_observables
from experiments.mipt.stabilizer import StabilizerSim, mi_matrix, tripartite_mi
from experiments.mipt.circuits import run_trajectory


def _bell_chain(L):
    """Disjoint Bell pairs (0,1), (2,3), ...: MI = 2 bits inside a pair, 0 across."""
    sim = StabilizerSim(L, seed=0)
    for a in range(0, L - 1, 2):
        sim.h(a)
        sim.cnot(a, a + 1)
    return sim


def _ghz(L):
    sim = StabilizerSim(L, seed=0)
    sim.h(0)
    for a in range(L - 1):
        sim.cnot(a, a + 1)
    return sim


def test_bell_chain_fragments_honestly():
    # 4 disjoint Bell pairs on L=8: the MI graph must be exactly 4 disconnected
    # dimers — zero-MI pairs are NON-edges (pre-registered MI_FLOOR rule), so
    # kNN ties at maximal distance must not fake connectivity.
    sim = _bell_chain(8)
    M = mi_matrix(sim)
    G = mi_graph(M)
    assert G.number_of_edges() == 4
    assert sorted(tuple(sorted(e)) for e in G.edges()) == [
        (0, 1), (2, 3), (4, 5), (6, 7)]
    obs = trajectory_observables(M)
    assert obs["lcc_frac"] == pytest.approx(2 / 8)
    # mean MI over the 28 unordered pairs: 4 pairs at 2 bits -> 8/28.
    assert obs["mean_mi"] == pytest.approx(8 / 28)
    # connectivity diagnostics: 4 dimer edges, all adjacent-index -> 0 long-range
    assert obs["n_edges"] == 4
    assert obs["lr_edges"] == 0


def test_ghz_is_one_component_zero_delta():
    # GHZ_8: every pair has MI exactly 1 bit -> complete MI graph at uniform
    # distance -> single component, hop-metric delta = 0 (all 4-point sums tie).
    sim = _ghz(8)
    obs = trajectory_observables(mi_matrix(sim))
    assert obs["lcc_frac"] == pytest.approx(1.0)
    assert obs["delta"] == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(obs["forman_mean"])


def test_product_state_degenerate_record():
    # p=1 limit: all MI zero -> no graph. The adapter must return a degenerate
    # record (NaNs + lcc_frac = 1/L), never raise — fragmentation IS data.
    sim = StabilizerSim(8, seed=0)
    obs = trajectory_observables(mi_matrix(sim))
    assert obs["lcc_frac"] == pytest.approx(1 / 8)
    assert np.isnan(obs["delta"])
    assert np.isnan(obs["forman_mean"])
    assert obs["mean_mi"] == 0.0


def test_random_trajectory_observables_finite():
    sim = run_trajectory(L=12, p=0.15, T=24, seed=9)
    M = mi_matrix(sim)
    obs = trajectory_observables(M)
    for key in ("forman_mean", "forman_min", "delta", "delta_norm",
                "spectral_dim", "hmds_distortion", "mds_stress"):
        assert np.isfinite(obs[key]), key
    assert obs["delta"] >= 0.0
    assert 0.0 < obs["lcc_frac"] <= 1.0
    assert np.isfinite(tripartite_mi(sim))


def test_mi_floor_is_strict():
    # An MI value at exactly the floor is a non-edge; just above it is an edge.
    M = np.zeros((5, 5))
    M[0, 1] = M[1, 0] = MI_FLOOR
    M[2, 3] = M[3, 2] = 10 * MI_FLOOR
    G = mi_graph(M)
    assert not G.has_edge(0, 1)
    assert G.has_edge(2, 3)
