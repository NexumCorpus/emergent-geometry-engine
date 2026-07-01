"""Free-fermion entanglement entropy via the Peschel correlation-matrix method.

Peschel, J. Phys. A 36 (2003) L205. For a free-fermion ground state, the reduced
density matrix of a block A is a thermal state of a quadratic entanglement
Hamiltonian whose entire spectrum is fixed by the restricted correlation matrix
C_ij = <c_i^dag c_j>, i,j in A. With nu_k = eig(C_A):

    S = - sum_k [ nu_k ln nu_k + (1 - nu_k) ln(1 - nu_k) ]   (nats)

This module is pure numpy and shares no code with the exact-diagonalization
cross-check in verify/ed.py — that independence is the whole point.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import eigh

_FLOOR = 1e-15  # nu_k at 0 or 1 contribute exactly 0 (x ln x -> 0)


def xx_correlation_matrix(N: int, n_fill: int | None = None, bc: str = "open") -> np.ndarray:
    """Equal-time correlation matrix C_ij = <c_i^dag c_j> for the half-filled XX /
    nearest-neighbour hopping chain H = -sum_i (c_i^dag c_{i+1} + h.c.).

    Open boundaries are the default and the only setting the project anchors trust
    (periodic is degenerate at the Fermi level for N divisible by 4).
    """
    if n_fill is None:
        n_fill = N // 2
    h = np.zeros((N, N))
    for i in range(N - 1):
        h[i, i + 1] = -1.0
        h[i + 1, i] = -1.0
    if bc == "periodic":
        h[0, N - 1] = -1.0
        h[N - 1, 0] = -1.0
    elif bc != "open":
        raise ValueError(f"bc must be 'open' or 'periodic', got {bc!r}")

    eps, U = eigh(h)                       # ascending single-particle energies
    occ = np.argsort(eps)[:n_fill]         # fill the lowest (most negative) modes
    U_occ = U[:, occ]
    return U_occ @ U_occ.conj().T          # C_ij = sum_{k in occ} U_ik conj(U_jk)


def free_fermion_entropy(C: np.ndarray, L: int) -> float:
    """Entanglement entropy (nats) of the first L sites from the correlation matrix."""
    nu = np.clip(np.real(eigh(C[:L, :L])[0]), 0.0, 1.0)
    s = 0.0
    for x in nu:
        if _FLOOR < x < 1.0 - _FLOOR:
            s -= x * np.log(x) + (1.0 - x) * np.log(1.0 - x)
    return float(s)


def xx_entanglement_entropy(N: int, L: int, bc: str = "open",
                            n_fill: int | None = None) -> float:
    """Convenience: entanglement entropy of the first L sites of the XX chain."""
    C = xx_correlation_matrix(N, n_fill=n_fill, bc=bc)
    return free_fermion_entropy(C, L)
