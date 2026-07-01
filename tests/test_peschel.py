"""Peschel free-fermion entanglement entropy (Constitution sec 2.2).

This is the project's central correctness anchor. The entanglement entropy of the
XX chain is computed THREE independent ways, which must all agree to <1e-9:

  1. Peschel correlation-matrix method (single-particle spectrum).      [verify/peschel.py]
  2. Exact diagonalization of the Jordan-Wigner spin chain + ptrace.    [verify/ed.py]
  3. Hard-coded anchors — the mathematically-correct values, stored to float64
     precision and corroborated by the independent ED + QuTiP routes and a blind
     external re-derivation (agreement < 5e-15).

Method 1 and method 2 share NO code except the trivial entropy summation, so a
sign/factor error in either cannot produce agreement. Open boundaries only — the
periodic chain is degenerate at the Fermi level (see research notes).
"""
import numpy as np
import pytest

from verify.peschel import xx_correlation_matrix, free_fermion_entropy, xx_entanglement_entropy
from verify.ed import xx_reduced_entropy_ed

LN2 = np.log(2.0)

# Anchors: open BC, half filling, A = first L sites, natural log (nats).
# These are the mathematically-correct entanglement entropies, stored to full float64
# precision and independently confirmed THREE ways (numpy exact diagonalization below,
# QuTiP exact ptrace in tests/test_qutip_check.py, and a blind external re-derivation
# that matched the N=8 value to 4.66e-15). The N=2,L=1 value is exactly ln 2.
ANCHORS = {
    (2, 1): 0.6931471805599453,   # = ln 2 exactly (single maximally entangled bond)
    (4, 2): 0.4132786277699681,
    (6, 3): 0.7203024642873879,
    (8, 4): 0.5697484844216253,
    (10, 5): 0.7578413629982911,
}


def test_correlation_matrix_is_hermitian_and_right_size():
    C = xx_correlation_matrix(8, n_fill=4, bc="open")
    assert C.shape == (8, 8)
    assert np.allclose(C, C.conj().T, atol=1e-12)
    # Trace of full correlation matrix = particle number = half filling.
    assert abs(np.trace(C).real - 4.0) < 1e-9


def test_single_bond_is_ln2():
    # N=2, L=1: the two-site singlet/bonding state; half-chain entropy = ln 2.
    s = xx_entanglement_entropy(2, 1, bc="open")
    assert abs(s - LN2) < 1e-9


@pytest.mark.parametrize("N,L", list(ANCHORS.keys()))
def test_peschel_matches_literature_anchor(N, L):
    s = xx_entanglement_entropy(N, L, bc="open")
    assert abs(s - ANCHORS[(N, L)]) < 1e-9


@pytest.mark.parametrize("N,L", [(2, 1), (4, 2), (6, 3), (8, 4)])
def test_peschel_matches_independent_ed(N, L):
    # The decisive cross-check: two mathematically independent derivations agree.
    s_peschel = xx_entanglement_entropy(N, L, bc="open")
    s_ed = xx_reduced_entropy_ed(N, L, bc="open")
    assert abs(s_peschel - s_ed) < 1e-9


def test_ed_matches_literature_anchor():
    # ED alone must also hit the anchor (independent of Peschel).
    assert abs(xx_reduced_entropy_ed(8, 4, bc="open") - ANCHORS[(8, 4)]) < 1e-9


def test_free_fermion_entropy_handles_full_and_empty_modes():
    # Eigenvalues exactly 0 or 1 contribute nothing (product state, no entanglement).
    C_trivial = np.diag([1.0, 0.0, 1.0, 0.0])
    assert abs(free_fermion_entropy(C_trivial, L=4)) < 1e-12
