"""Tests for the numpy-only quantum-information primitives in verify/entropy.py.

These are the foundation every other entropy check builds on, so they are pinned
to exactly-known analytic values (ln 2, 0) rather than to each other.
"""
import numpy as np
import pytest

from verify.entropy import von_neumann_entropy, partial_trace


LN2 = np.log(2.0)


def _bell_state_density() -> np.ndarray:
    """|Phi+> = (|00> + |11>)/sqrt(2) as a 4x4 density matrix."""
    psi = np.zeros(4, dtype=complex)
    psi[0b00] = 1.0
    psi[0b11] = 1.0
    psi /= np.sqrt(2.0)
    return np.outer(psi, psi.conj())


def test_entropy_of_pure_state_is_zero():
    psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert abs(von_neumann_entropy(rho)) < 1e-12


def test_entropy_of_maximally_mixed_qubit_is_ln2():
    rho = np.eye(2, dtype=complex) / 2.0
    assert abs(von_neumann_entropy(rho) - LN2) < 1e-12


def test_partial_trace_of_bell_state_is_maximally_mixed():
    rho = _bell_state_density()
    rho_A = partial_trace(rho, dims=[2, 2], keep=[0])
    assert np.allclose(rho_A, np.eye(2) / 2.0, atol=1e-12)
    assert abs(von_neumann_entropy(rho_A) - LN2) < 1e-12


def test_partial_trace_keep_second_subsystem():
    rho = _bell_state_density()
    rho_B = partial_trace(rho, dims=[2, 2], keep=[1])
    assert np.allclose(rho_B, np.eye(2) / 2.0, atol=1e-12)


def test_entropy_handles_tiny_negative_eigenvalues():
    # Numerically, a valid rho can have eigenvalues like -1e-16; entropy must stay
    # finite (clip, do not blow up on log of a negative).
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # pure, eigvals {1, 0}
    s = von_neumann_entropy(rho)
    assert np.isfinite(s)
    assert abs(s) < 1e-12


def test_entropy_base2_option():
    rho = np.eye(2, dtype=complex) / 2.0
    assert abs(von_neumann_entropy(rho, base=2) - 1.0) < 1e-12
