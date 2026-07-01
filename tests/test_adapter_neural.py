"""Adapter N: transformer activations -> representation-similarity matrix
(Constitution sec 2.3). Robustness fix #2: use cosine / CKA, NOT estimated mutual
information (MI estimation on neural activations is biased).
"""
import numpy as np
import pytest

from adapters.neural import linear_cka, cosine_similarity_matrix


def test_cka_self_is_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8))
    assert abs(linear_cka(X, X) - 1.0) < 1e-9


def test_cka_invariant_to_orthogonal_transform():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 8))
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))  # random orthogonal
    assert abs(linear_cka(X, X @ Q) - 1.0) < 1e-9


def test_cka_invariant_to_isotropic_scaling():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 8))
    assert abs(linear_cka(X, 7.5 * X) - 1.0) < 1e-9


def test_cka_in_unit_interval_and_low_for_unrelated():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((80, 6))
    Y = rng.standard_normal((80, 6))  # independent
    val = linear_cka(X, Y)
    assert 0.0 - 1e-9 <= val <= 1.0 + 1e-9
    assert val < 0.5  # unrelated representations -> low CKA


def test_cosine_similarity_matrix_properties():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((10, 5))   # 10 nodes, 5 features
    S = cosine_similarity_matrix(A)
    assert S.shape == (10, 10)
    assert np.allclose(np.diag(S), 1.0, atol=1e-9)   # unit self-similarity
    assert np.allclose(S, S.T, atol=1e-12)           # symmetric
    assert S.min() >= -1.0 - 1e-9 and S.max() <= 1.0 + 1e-9


def test_cosine_matrix_identical_rows_all_ones():
    A = np.tile(np.array([[1.0, 2.0, 3.0]]), (4, 1))
    S = cosine_similarity_matrix(A)
    assert np.allclose(S, 1.0, atol=1e-9)


def test_cosine_matrix_orthogonal_rows():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    S = cosine_similarity_matrix(A)
    assert abs(S[0, 1]) < 1e-12
