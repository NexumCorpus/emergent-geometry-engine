"""Independent exact-diagonalization cross-check for the XX-chain entropy.

This computes the SAME entanglement entropy as verify/peschel.py by a completely
different route: build the many-body spin Hamiltonian via Jordan-Wigner, diagonalize
the full 2^N matrix, take the ground state, partial-trace, and apply the standard
von Neumann entropy. It shares no derivation with the correlation-matrix method, so
agreement to <1e-9 certifies both.

Jordan-Wigner (open chain): H_fermion = -sum_i (c_i^dag c_{i+1} + h.c.) maps to
    H_spin = -sum_i (1/2)(X_i X_{i+1} + Y_i Y_{i+1}).
Pure numpy (no QuTiP). Limited to small N (2^N dense); N<=12 is comfortable.
"""
from __future__ import annotations

import numpy as np

from verify.entropy import von_neumann_entropy, partial_trace

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)


def _op_at(op: np.ndarray, site: int, N: int) -> np.ndarray:
    """Embed a single-site operator at `site` into the N-qubit Hilbert space."""
    out = op if site == 0 else _I
    for i in range(1, N):
        out = np.kron(out, op if i == site else _I)
    return out


def xx_hamiltonian(N: int, bc: str = "open") -> np.ndarray:
    """H_spin = -sum (1/2)(X_i X_{i+1} + Y_i Y_{i+1}) over the chain bonds."""
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)
    bonds = [(i, i + 1) for i in range(N - 1)]
    if bc == "periodic":
        bonds.append((N - 1, 0))
    elif bc != "open":
        raise ValueError(f"bc must be 'open' or 'periodic', got {bc!r}")
    for i, j in bonds:
        Xi, Xj = _op_at(_X, i, N), _op_at(_X, j, N)
        Yi, Yj = _op_at(_Y, i, N), _op_at(_Y, j, N)
        H += -0.5 * (Xi @ Xj + Yi @ Yj)
    return H


def xx_ground_state(N: int, bc: str = "open") -> np.ndarray:
    """Global ground-state vector of the XX chain (half filling for even N)."""
    H = xx_hamiltonian(N, bc)
    _, vecs = np.linalg.eigh(H)
    return vecs[:, 0]


def xx_reduced_entropy_ed(N: int, L: int, bc: str = "open") -> float:
    """Entanglement entropy (nats) of the first L sites, via exact diagonalization."""
    psi = xx_ground_state(N, bc)
    rho = np.outer(psi, psi.conj())
    rho_A = partial_trace(rho, dims=[2] * N, keep=list(range(L)))
    return von_neumann_entropy(rho_A)
