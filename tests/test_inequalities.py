"""The entropy-inequality suite (Constitution sec 2.2).

Each inequality is checked on explicitly constructed states with a small numerical
slack tolerance. We use a fixed RNG so failures are reproducible.
"""
import numpy as np
import pytest

from verify.inequalities import (
    check_trace_one,
    check_positive_semidefinite,
    check_entropy_nonnegative,
    check_subadditivity,
    check_strong_subadditivity,
    check_araki_lieb,
    check_pure_state_symmetry,
)


def _random_density(dim: int, rank: int, rng) -> np.ndarray:
    """Random mixed density matrix of given rank: M M^dagger / tr, M is dim x rank."""
    M = rng.standard_normal((dim, rank)) + 1j * rng.standard_normal((dim, rank))
    rho = M @ M.conj().T
    return rho / np.trace(rho).real


def _random_pure(dim: int, rng) -> np.ndarray:
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return psi / np.linalg.norm(psi)


def test_trace_one_holds_for_density():
    rng = np.random.default_rng(0)
    rho = _random_density(8, 8, rng)
    assert check_trace_one(rho).ok


def test_positive_semidefinite_holds_for_density():
    rng = np.random.default_rng(1)
    rho = _random_density(8, 5, rng)
    assert check_positive_semidefinite(rho).ok


def test_positive_semidefinite_fails_for_non_psd():
    bad = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=complex)
    bad[0, 0] = -0.1  # deliberately not PSD
    bad[1, 1] = 1.1
    assert not check_positive_semidefinite(bad).ok


def test_entropy_nonnegative():
    rng = np.random.default_rng(2)
    rho = _random_density(8, 4, rng)
    assert check_entropy_nonnegative(rho).ok


def test_subadditivity_two_qubits():
    rng = np.random.default_rng(3)
    rho = _random_density(4, 4, rng)  # 2 qubits, mixed
    res = check_subadditivity(rho, dims=[2, 2])
    assert res.ok
    assert res.slack >= -1e-12


def test_strong_subadditivity_three_qubits():
    rng = np.random.default_rng(4)
    rho = _random_density(8, 8, rng)  # 3 qubits, mixed
    res = check_strong_subadditivity(rho, dims=[2, 2, 2])
    assert res.ok
    assert res.slack >= -1e-12


def test_araki_lieb_two_qubits():
    rng = np.random.default_rng(5)
    rho = _random_density(4, 3, rng)
    res = check_araki_lieb(rho, dims=[2, 2])
    assert res.ok


def test_pure_state_symmetry():
    rng = np.random.default_rng(6)
    psi = _random_pure(8, rng)  # 3 qubits pure; partition A={0}, complement={1,2}
    res = check_pure_state_symmetry(psi, dims=[2, 2, 2], partition=[0])
    assert res.ok
    assert abs(res.slack) < 1e-9  # S(A) == S(Abar) to machine precision


def test_pure_state_symmetry_unbalanced_partition():
    rng = np.random.default_rng(7)
    psi = _random_pure(16, rng)  # 4 qubits; A={0,1}
    res = check_pure_state_symmetry(psi, dims=[2, 2, 2, 2], partition=[0, 1])
    assert res.ok
