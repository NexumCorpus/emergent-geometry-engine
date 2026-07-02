"""Aaronson-Gottesman stabilizer simulator + region entropies (Phase 3, Track A).

Exact simulation of Clifford circuits with projective Z measurements, following
Aaronson & Gottesman, PRA 70 052328 (2004): a (2n+1) x (2n+1) binary tableau of
destabilizer rows (0..n-1), stabilizer rows (n..2n-1) and a scratch row, with
phase bits. Region entanglement entropy uses the stabilizer-rank formula
(Fattal et al., quant-ph/0406168):

    S_A [bits] = rank_GF2( stabilizer rows restricted to A's (x|z) columns ) - |A|

Derivation check (Bell pair, stabilizers XX and ZZ, A = site 0): restricted
rows are (x=1|z=0) and (x=0|z=1) -> rank 2 -> S = 2 - 1 = 1 bit. Correct.

Graded independently by tests/test_stabilizer.py against a dense statevector
executor built on verify/ primitives (shares no code with this module).
"""
from __future__ import annotations

import numpy as np


def gf2_rank(M: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2), by Gaussian elimination."""
    M = (np.array(M, dtype=np.uint8) & 1).copy()
    rows, cols = M.shape
    rank = 0
    for c in range(cols):
        piv = None
        for rr in range(rank, rows):
            if M[rr, c]:
                piv = rr
                break
        if piv is None:
            continue
        M[[rank, piv]] = M[[piv, rank]]
        mask = M[:, c].astype(bool)
        mask[rank] = False
        M[mask] ^= M[rank]
        rank += 1
        if rank == rows:
            break
    return rank


class StabilizerSim:
    """n-qubit stabilizer state, initialized to |0...0>.

    `seed` drives ONLY the intrinsic randomness of measurements on
    non-eigenstate qubits, so a trajectory is fully reproducible.
    """

    def __init__(self, n: int, seed=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.z = np.zeros((2 * n + 1, n), dtype=np.uint8)
        self.r = np.zeros(2 * n + 1, dtype=np.uint8)
        for i in range(n):
            self.x[i, i] = 1          # destabilizer i = X_i
            self.z[n + i, i] = 1      # stabilizer  i = Z_i

    # --- Clifford gates (AG04 update rules) --------------------------------
    def h(self, q: int):
        self.r ^= self.x[:, q] & self.z[:, q]
        xq = self.x[:, q].copy()
        self.x[:, q] = self.z[:, q]
        self.z[:, q] = xq

    def s(self, q: int):
        self.r ^= self.x[:, q] & self.z[:, q]
        self.z[:, q] ^= self.x[:, q]

    def cnot(self, a: int, b: int):
        self.r ^= self.x[:, a] & self.z[:, b] & (self.x[:, b] ^ self.z[:, a] ^ 1)
        self.x[:, b] ^= self.x[:, a]
        self.z[:, a] ^= self.z[:, b]

    # --- rowsum: row h <- row i * row h, with exact phase tracking ----------
    def _rowsum(self, h: int, i: int):
        x1 = self.x[i].astype(np.int64)
        z1 = self.z[i].astype(np.int64)
        x2 = self.x[h].astype(np.int64)
        z2 = self.z[h].astype(np.int64)
        # AG04 g-function, per column: exponent of i contributed by multiplying
        # the column's Paulis (0 if row i has I there; Y: z2-x2; X: z2(2x2-1);
        # Z: x2(1-2z2)).
        g = (x1 * z1 * (z2 - x2)
             + x1 * (1 - z1) * (z2 * (2 * x2 - 1))
             + (1 - x1) * z1 * (x2 * (1 - 2 * z2)))
        ph = (2 * int(self.r[h]) + 2 * int(self.r[i]) + int(g.sum())) % 4
        self.r[h] = 1 if ph == 2 else 0     # ph is provably 0 or 2
        self.x[h] ^= self.x[i]
        self.z[h] ^= self.z[i]

    # --- projective Z measurement -------------------------------------------
    def measure(self, q: int) -> int:
        n = self.n
        anticommuting = np.nonzero(self.x[n:2 * n, q])[0]
        if anticommuting.size:
            # Random outcome: some stabilizer anticommutes with Z_q.
            p = int(anticommuting[0]) + n
            for i in range(2 * n):
                if i != p and self.x[i, q]:
                    self._rowsum(i, p)
            self.x[p - n] = self.x[p].copy()
            self.z[p - n] = self.z[p].copy()
            self.r[p - n] = self.r[p]
            self.x[p] = 0
            self.z[p] = 0
            self.z[p, q] = 1
            outcome = int(self.rng.integers(2))
            self.r[p] = outcome
            return outcome
        # Deterministic outcome: reconstruct the sign via the scratch row.
        s = 2 * n
        self.x[s] = 0
        self.z[s] = 0
        self.r[s] = 0
        for i in range(n):
            if self.x[i, q]:
                self._rowsum(s, i + n)
        return int(self.r[s])

    def apply(self, ops) -> list[int]:
        """Run an op list [('h',q)|('s',q)|('cnot',a,b)|('m',q), ...];
        returns measurement outcomes in order."""
        outcomes = []
        for op in ops:
            kind = op[0]
            if kind == "h":
                self.h(op[1])
            elif kind == "s":
                self.s(op[1])
            elif kind == "cnot":
                self.cnot(op[1], op[2])
            elif kind == "m":
                outcomes.append(self.measure(op[1]))
            else:
                raise ValueError(f"unknown op {op!r}")
        return outcomes

    # --- entanglement -------------------------------------------------------
    def entropy(self, region) -> float:
        """Entanglement entropy of `region` in BITS (exact integer for
        stabilizer states)."""
        cols = sorted(set(region))
        stab = slice(self.n, 2 * self.n)
        M = np.concatenate(
            [self.x[stab][:, cols], self.z[stab][:, cols]], axis=1)
        return float(gf2_rank(M) - len(cols))


def mi_matrix(sim: StabilizerSim) -> np.ndarray:
    """Pairwise mutual information I(i:j) = S_i + S_j - S_ij, in bits.
    Symmetric, zero diagonal, nonnegative."""
    n = sim.n
    S1 = [sim.entropy([i]) for i in range(n)]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = S1[i] + S1[j] - sim.entropy([i, j])
    return M


def tripartite_mi(sim: StabilizerSim) -> float:
    """I3(A:B:C) over the first three contiguous quarters of the chain — the
    Constitution's mandated MIPT diagnostic (NOT raw entropy scaling)."""
    n = sim.n
    q = n // 4
    A = list(range(0, q))
    B = list(range(q, 2 * q))
    C = list(range(2 * q, 3 * q))
    S = sim.entropy
    return (S(A) + S(B) + S(C)
            - S(A + B) - S(A + C) - S(B + C)
            + S(A + B + C))
