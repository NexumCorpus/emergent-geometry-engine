"""Early-warning signals for the grokking head-to-head (Constitution sec 5.2).

Two signals are tested here:
  - 2RDM spectral heat capacity (Hennick-Corlouer arXiv:2603.29805): SHC = Var(eigvals)
    of the n x n covariance of per-sample LOSSES across a checkpoint window. NOT a
    thermodynamic quantity — literally the variance of the raw eigenvalues.
  - The geometric order parameter: mean Forman curvature / spectral dimension of the
    cosine-similarity graph of a representation, which should read ~1D for a Fourier
    ring and higher-D for unstructured features.
"""
import numpy as np
import pytest

from analysis.signals import (
    spectral_heat_capacity,
    participation_ratio,
    embedding_graph_signal,
)


def test_shc_rank_one_window_matches_analytic():
    # Loss window L[t,i] = f(t) * w[i]: every probe co-fluctuates with one temporal
    # mode -> covariance is rank-1 -> one nonzero eigenvalue lambda1, SHC = Var([l1,0..]).
    rng = np.random.default_rng(0)
    W, n = 20, 5
    f = rng.standard_normal(W)
    w = rng.standard_normal(n)
    L = np.outer(f, w)
    lam1 = np.var(f, ddof=1) * np.sum(w ** 2)        # the single nonzero eigenvalue
    expected = np.var(np.array([lam1] + [0.0] * (n - 1)))  # population var of spectrum
    assert abs(spectral_heat_capacity(L) - expected) < 1e-9


def test_shc_low_for_uniform_high_for_concentrated():
    rng = np.random.default_rng(1)
    W, n = 40, 8
    independent = rng.standard_normal((W, n))                 # ~isotropic spectrum
    f = rng.standard_normal(W)
    concentrated = np.outer(f, rng.standard_normal(n)) + 1e-3 * rng.standard_normal((W, n))
    # A single dominant mode gives a much larger eigenvalue-variance than an isotropic
    # spectrum (empirically ~8x here; assert a clear, conservative separation).
    assert spectral_heat_capacity(concentrated) > 5 * spectral_heat_capacity(independent)


def test_participation_ratio_rank_one_is_about_one():
    rng = np.random.default_rng(2)
    L = np.outer(rng.standard_normal(20), rng.standard_normal(6))
    assert abs(participation_ratio(L) - 1.0) < 1e-6


def test_participation_ratio_isotropic_is_about_n():
    # Independent isotropic columns -> covariance ~ identity -> PR ~ n.
    rng = np.random.default_rng(3)
    n = 6
    L = rng.standard_normal((4000, n))
    assert participation_ratio(L) > 0.7 * n


def _fourier_ring(p, freqs=(1,)):
    n = np.arange(p)
    cols = []
    for k in freqs:
        cols.append(np.cos(2 * np.pi * k * n / p))
        cols.append(np.sin(2 * np.pi * k * n / p))
    return np.column_stack(cols)


def test_geometric_signal_ring_is_low_dimensional():
    # A single-frequency Fourier ring should read ~1D (the grokked asymptote).
    feats = _fourier_ring(48, freqs=(1,))
    sig = embedding_graph_signal(feats, k=4)
    assert set(["mean_forman", "var_forman", "spectral_dim", "gromov_delta", "n_edges"]) <= set(sig)
    assert 0.5 <= sig["spectral_dim"] <= 1.8


def test_geometric_signal_separates_ring_from_random():
    # The order parameter must distinguish the grokked ring from pre-grok noise. The
    # ROBUST discriminator at this scale is mean Forman curvature (ring ~ +1, uniform
    # positive cycle; noise ~ -7, negatively-curved/bottlenecked) and Gromov delta
    # (ring ~ 0, noise > 0). Spectral dimension is too noisy at p~48 to discriminate.
    rng = np.random.default_rng(5)
    ring = _fourier_ring(48, freqs=(1,))
    noise = rng.standard_normal((48, 32))
    s_ring = embedding_graph_signal(ring, k=6)
    s_noise = embedding_graph_signal(noise, k=6)
    assert s_ring["mean_forman"] > s_noise["mean_forman"] + 2.0   # large clean gap
    assert s_ring["gromov_delta"] <= s_noise["gromov_delta"]
