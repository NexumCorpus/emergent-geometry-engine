"""Phase 3 slice 1: stabilizer (Clifford) platform for the MIPT sweep.

Graders are INDEPENDENT of the implementation (Constitution: generator != grader):
  - analytic anchors (product / Bell / GHZ / measurement determinism), each value
    derived by hand in the comment beside its assert;
  - a dense statevector executor built HERE on verify/ primitives only (it shares
    no code with experiments/mipt/stabilizer.py) replays the identical op list,
    with the tableau's measurement outcomes FORCED by projectors, and every
    region entropy must agree to 1e-9.
"""
import numpy as np
import pytest

from experiments.mipt.stabilizer import StabilizerSim, gf2_rank, mi_matrix, tripartite_mi
from experiments.mipt.circuits import C1_WORDS, build_brickwork_ops, run_trajectory
from verify.entropy import von_neumann_entropy, partial_trace


# --------------------------------------------------------------------------- #
# Dense statevector executor (the independent grader). numpy only.
# --------------------------------------------------------------------------- #
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)


def _apply_1q(psi, U, q, n):
    t = psi.reshape([2] * n)
    t = np.tensordot(U, t, axes=([1], [q]))       # gate index becomes axis 0
    t = np.moveaxis(t, 0, q)
    return t.reshape(-1)


def _apply_cnot(psi, a, b, n):
    t = psi.reshape([2] * n).copy()
    # control a = 1 slice: flip target axis b
    idx = [slice(None)] * n
    idx[a] = 1
    sub = t[tuple(idx)]
    t[tuple(idx)] = np.flip(sub, axis=b if b < a else b - 1)
    return t.reshape(-1)


def _project(psi, q, outcome, n):
    """Force measurement outcome; return (renormalized psi, outcome probability)."""
    t = psi.reshape([2] * n).copy()
    idx = [slice(None)] * n
    idx[q] = 1 - outcome
    t[tuple(idx)] = 0.0
    v = t.reshape(-1)
    prob = float(np.vdot(v, v).real)
    if prob > 1e-12:
        v = v / np.sqrt(prob)
    return v, prob


def dense_replay(n, ops, outcomes):
    """Replay an op list on a dense statevector, forcing the given outcomes.
    Returns (psi, min_forced_probability)."""
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    outcomes = list(outcomes)
    min_prob = 1.0
    for op in ops:
        kind = op[0]
        if kind == "h":
            psi = _apply_1q(psi, _H, op[1], n)
        elif kind == "s":
            psi = _apply_1q(psi, _S, op[1], n)
        elif kind == "cnot":
            psi = _apply_cnot(psi, op[1], op[2], n)
        elif kind == "m":
            psi, prob = _project(psi, op[1], outcomes.pop(0), n)
            min_prob = min(min_prob, prob)
        else:
            raise ValueError(f"unknown op {op!r}")
    assert not outcomes, "outcome list longer than measurement count"
    return psi, min_prob


def dense_entropy_bits(psi, n, region):
    rho = np.outer(psi, psi.conj())
    rho_A = partial_trace(rho, dims=[2] * n, keep=sorted(region))
    return von_neumann_entropy(rho_A, base=2)


# --------------------------------------------------------------------------- #
# Analytic anchors
# --------------------------------------------------------------------------- #
def test_gf2_rank_anchors():
    assert gf2_rank(np.zeros((3, 4), dtype=np.uint8)) == 0
    assert gf2_rank(np.eye(3, dtype=np.uint8)) == 3
    # row3 = row1 XOR row2 -> rank 2
    M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    assert gf2_rank(M) == 2


def test_product_state_no_structure():
    sim = StabilizerSim(4, seed=0)
    for r in ([0], [1], [0, 1], [0, 2], [0, 1, 2]):
        assert sim.entropy(r) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(mi_matrix(sim), 0.0)


def test_bell_pair_anchors():
    # H(0), CNOT(0,1) -> (|00>+|11>)/sqrt2. S({0}) = 1 bit;
    # MI(0:1) = S0 + S1 - S01 = 1 + 1 - 0 = 2 bits.
    sim = StabilizerSim(2, seed=0)
    sim.h(0)
    sim.cnot(0, 1)
    assert sim.entropy([0]) == pytest.approx(1.0, abs=1e-12)
    assert sim.entropy([1]) == pytest.approx(1.0, abs=1e-12)
    assert sim.entropy([0, 1]) == pytest.approx(0.0, abs=1e-12)
    M = mi_matrix(sim)
    assert M[0, 1] == pytest.approx(2.0, abs=1e-12)
    assert M[0, 0] == 0.0


def test_ghz4_entropies_and_i3():
    # GHZ_4: any proper nonempty region has S = 1 bit. Quarters of n=4 are single
    # sites A={0},B={1},C={2}: I3 = S_A+S_B+S_C - S_AB-S_AC-S_BC + S_ABC
    #                             = 1+1+1 - 1-1-1 + 1 = +1 bit (GHZ anchor).
    sim = StabilizerSim(4, seed=0)
    sim.h(0)
    sim.cnot(0, 1)
    sim.cnot(1, 2)
    sim.cnot(2, 3)
    for r in ([0], [3], [0, 1], [1, 3], [0, 1, 2]):
        assert sim.entropy(r) == pytest.approx(1.0, abs=1e-12)
    assert tripartite_mi(sim) == pytest.approx(1.0, abs=1e-12)


def test_measurement_determinism_and_collapse():
    # |0> measures 0 deterministically, repeatably.
    sim = StabilizerSim(1, seed=1)
    assert sim.measure(0) == 0
    assert sim.measure(0) == 0
    # H|0> measures randomly, then repeats deterministically (collapse).
    sim = StabilizerSim(1, seed=1)
    sim.h(0)
    o = sim.measure(0)
    assert o in (0, 1)
    assert sim.measure(0) == o
    # Bell pair: measuring site 0 collapses site 1 to the SAME outcome
    # ((|00>+|11>)/sqrt2 has perfectly correlated Z outcomes) and kills MI.
    sim = StabilizerSim(2, seed=3)
    sim.h(0)
    sim.cnot(0, 1)
    o0 = sim.measure(0)
    assert sim.entropy([0]) == pytest.approx(0.0, abs=1e-12)
    assert sim.entropy([1]) == pytest.approx(0.0, abs=1e-12)
    assert sim.measure(1) == o0


def test_mi_matrix_properties():
    rng = np.random.default_rng(7)
    sim = StabilizerSim(6, seed=11)
    ops = build_brickwork_ops(6, p=0.2, T=8, rng=rng)
    sim.apply(ops)
    M = mi_matrix(sim)
    assert M.shape == (6, 6)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)
    assert (M > -1e-12).all()                    # MI is nonnegative


# --------------------------------------------------------------------------- #
# The independent grader: dense replay of identical circuits, forced outcomes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_circuit_matches_dense_statevector(seed):
    n = 5
    rng = np.random.default_rng(seed)
    ops = build_brickwork_ops(n, p=0.25, T=6, rng=rng)
    sim = StabilizerSim(n, seed=seed + 100)
    outcomes = sim.apply(ops)

    psi, min_prob = dense_replay(n, ops, outcomes)
    # Every forced outcome must have been genuinely possible (prob 0.5 or 1.0
    # for stabilizer states; anything near 0 means the tableau lied).
    assert min_prob > 0.25

    regions = [[q] for q in range(n)]
    regions += [list(range(k)) for k in range(1, n)]
    regions += [[0, 2], [1, 4], [0, 3, 4]]
    for r in regions:
        assert sim.entropy(r) == pytest.approx(
            dense_entropy_bits(psi, n, r), abs=1e-9), f"region {r}"


def test_c1_words_cover_the_single_qubit_clifford_group():
    # 24 distinct single-qubit Cliffords mod global phase, each as an H/S word.
    assert len(C1_WORDS) == 24
    seen = set()
    for word in C1_WORDS:
        U = np.eye(2, dtype=complex)
        for g in word:
            U = (_H if g == "h" else _S) @ U
        # canonicalize mod global phase: divide by phase of first nonzero entry
        flat = U.reshape(-1)
        piv = flat[np.argmax(np.abs(flat) > 1e-9)]
        key = tuple(np.round(flat / (piv / abs(piv)), 9))
        seen.add(key)
    assert len(seen) == 24


# --------------------------------------------------------------------------- #
# Trajectory driver: determinism + physical limits
# --------------------------------------------------------------------------- #
def test_trajectory_is_deterministic_given_seed():
    a = run_trajectory(L=8, p=0.15, T=16, seed=42)
    b = run_trajectory(L=8, p=0.15, T=16, seed=42)
    assert np.allclose(mi_matrix(a), mi_matrix(b))
    assert tripartite_mi(a) == pytest.approx(tripartite_mi(b), abs=1e-12)


def test_trajectory_limits():
    # p=1: every site measured every layer -> product state -> zero MI.
    sim = run_trajectory(L=8, p=1.0, T=16, seed=5)
    assert np.allclose(mi_matrix(sim), 0.0)
    # p=0: pure unitary brickwork scrambles -> half-chain entropy >= 1 bit.
    sim = run_trajectory(L=8, p=0.0, T=16, seed=5)
    assert sim.entropy(list(range(4))) >= 1.0
