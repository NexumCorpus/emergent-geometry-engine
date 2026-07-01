"""Embeddings: classical Euclidean MDS and hyperbolic h-MDS (Constitution sec 2.1).

Euclidean MDS recovers points that live in R^dim with ~zero stress. h-MDS (Sala et
al., ICML 2018) exactly recovers genuinely-hyperbolic distance matrices: map random
Poincare-disk points to hyperboloid distances, then h-MDS reconstructs them to
machine precision. Both report a stress / distortion comparable across embeddings.
"""
import numpy as np
import pytest

from geometry.embedding import (
    euclidean_mds,
    mds_stress,
    hmds,
    hyperboloid_distances,
    avg_distortion,
)


def _euclidean_distance_matrix(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)


def test_euclidean_mds_recovers_planar_points():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 2))
    D = _euclidean_distance_matrix(X)
    coords, stress = euclidean_mds(D, dim=2)
    # Distances are recoverable in 2D, so stress ~ 0.
    assert stress < 1e-9
    D_rec = _euclidean_distance_matrix(coords)
    assert np.allclose(D_rec, D, atol=1e-6)


def test_euclidean_mds_stress_positive_when_dim_too_low():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((15, 3))   # genuinely 3D
    D = _euclidean_distance_matrix(X)
    _, stress = euclidean_mds(D, dim=1)  # forced into 1D -> must distort
    assert stress > 1e-3


def _random_poincare_points(n, dim, rng):
    p = rng.standard_normal((n, dim))
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    scale = rng.uniform(0.1, 0.8, size=(n, 1))
    return p / norms * scale  # inside the unit ball


def _poincare_to_hyperboloid_xvec(p):
    nsq = (p ** 2).sum(1, keepdims=True)
    return 2 * p / (1 - nsq)  # spatial (Gans) coords; x0 = (1+nsq)/(1-nsq)


def test_hmds_exact_recovery_of_hyperbolic_points():
    rng = np.random.default_rng(2)
    p = _random_poincare_points(12, 2, rng)
    Xvec = _poincare_to_hyperboloid_xvec(p)
    D = hyperboloid_distances(Xvec)
    Xrec = hmds(D, rank=2)
    D_rec = hyperboloid_distances(Xrec)
    assert avg_distortion(D, D_rec) < 1e-9
    assert np.max(np.abs(D - D_rec)) < 1e-7


def test_hyperboloid_distances_zero_diagonal_and_symmetric():
    rng = np.random.default_rng(3)
    Xvec = _poincare_to_hyperboloid_xvec(_random_poincare_points(8, 2, rng))
    D = hyperboloid_distances(Xvec)
    assert np.allclose(np.diag(D), 0.0, atol=1e-9)
    assert np.allclose(D, D.T, atol=1e-9)


def test_mds_stress_is_zero_for_exact_embedding():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((10, 2))
    D = _euclidean_distance_matrix(X)
    assert mds_stress(D, X) < 1e-9
