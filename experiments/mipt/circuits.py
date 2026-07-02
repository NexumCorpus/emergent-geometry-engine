"""Hybrid brickwork circuits for the MIPT sweep (Phase 3, Track A).

Circuit ensemble: alternating-offset brickwork of two-qubit blocks, each block
= independent uniform random single-qubit Cliffords on both qubits followed by
CNOT, then per-site Z measurement with probability p after each layer.

Honest ensemble note: gates are NOT uniform over the full two-qubit Clifford
group (they are C1xC1 * CNOT). The circuit is still scrambling Clifford
dynamics with a measurement-induced transition; the critical rate p_c is
ensemble-specific and is LOCATED BY the I3 control on identical runs — no
literature p_c value is assumed anywhere in Phase 3.

Single-qubit Cliffords are enumerated exactly (24 elements mod global phase)
as words over {H, S} by breadth-first closure at import time.
"""
from __future__ import annotations

import numpy as np

from experiments.mipt.stabilizer import StabilizerSim

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)


def _canonical(U: np.ndarray):
    """Canonical key of a 2x2 unitary mod global phase."""
    flat = U.reshape(-1)
    piv = flat[np.argmax(np.abs(flat) > 1e-9)]
    return tuple(np.round(flat / (piv / abs(piv)), 9))


def _enumerate_c1() -> list[tuple[str, ...]]:
    gates = {"h": _H, "s": _S}
    words = {_canonical(np.eye(2, dtype=complex)): ()}
    frontier = [((), np.eye(2, dtype=complex))]
    while frontier:
        nxt = []
        for word, U in frontier:
            for name, g in gates.items():
                U2 = g @ U                      # word applied left-to-right
                k = _canonical(U2)
                if k not in words:
                    w2 = word + (name,)
                    words[k] = w2
                    nxt.append((w2, U2))
        frontier = nxt
    assert len(words) == 24, f"C1 closure found {len(words)} != 24"
    return list(words.values())


C1_WORDS: list[tuple[str, ...]] = _enumerate_c1()


def build_brickwork_ops(L: int, p: float, T: int, rng) -> list:
    """Op list for T layers of brickwork + rate-p measurements on L qubits.
    Deterministic given the rng state."""
    ops = []
    for t in range(T):
        for a in range(t % 2, L - 1, 2):
            b = a + 1
            for q in (a, b):
                for g in C1_WORDS[int(rng.integers(len(C1_WORDS)))]:
                    ops.append((g, q))
            ops.append(("cnot", a, b))
        for q in range(L):
            if rng.random() < p:
                ops.append(("m", q))
    return ops


def run_trajectory(L: int, p: float, T: int, seed: int) -> StabilizerSim:
    """One monitored-circuit trajectory, fully reproducible from `seed`
    (circuit structure, gate choices, measurement locations AND outcomes)."""
    rng = np.random.default_rng(seed)
    ops = build_brickwork_ops(L, p, T, rng)
    sim = StabilizerSim(L, seed=int(rng.integers(2 ** 63)))
    sim.apply(ops)
    return sim
