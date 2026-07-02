"""Finite-size-scaling machinery (Phase 3 stretch): crossing finder + collapse.

Graders are synthetic datasets with KNOWN critical parameters: the tools must
recover p_c and nu from data generated with them, and must NOT find a good
collapse in unscalable (structureless) data. No sweep data enters these tests.
"""
import numpy as np
import pytest

from experiments.mipt.fss import crossing_points, collapse_quality, fit_collapse


def _synthetic(p_c=0.12, nu=1.3, sizes=(32, 48, 64, 96), noise=0.0, seed=0):
    """O(p, L) = f((p - p_c) * L^(1/nu)) with f a smooth sigmoid-like curve.
    Curves for all L cross exactly at p_c (f(0) shared)."""
    rng = np.random.default_rng(seed)
    ps = np.linspace(0.05, 0.20, 31)
    data = {}
    for L in sizes:
        x = (ps - p_c) * L ** (1.0 / nu)
        y = np.tanh(x)                       # f(0) = 0 for every L -> crossing
        y = y + noise * rng.standard_normal(len(ps))
        data[L] = (ps, y)
    return data


def test_crossings_locate_pc_exactly_on_clean_data():
    data = _synthetic(p_c=0.12, nu=1.3)
    cs = crossing_points(data)
    assert len(cs) >= 3                       # one per size pair (at least)
    assert np.allclose(cs, 0.12, atol=1e-3)


def test_crossings_tolerate_noise():
    data = _synthetic(p_c=0.12, nu=1.3, noise=0.01, seed=1)
    cs = crossing_points(data)
    assert abs(np.median(cs) - 0.12) < 0.01


def test_collapse_quality_prefers_true_parameters():
    data = _synthetic(p_c=0.12, nu=1.3)
    good = collapse_quality(data, p_c=0.12, nu=1.3)
    bad_pc = collapse_quality(data, p_c=0.16, nu=1.3)
    bad_nu = collapse_quality(data, p_c=0.12, nu=3.0)
    assert good < bad_pc
    assert good < bad_nu


def test_fit_recovers_known_exponents():
    data = _synthetic(p_c=0.125, nu=1.4, noise=0.005, seed=2)
    p_c, nu, q = fit_collapse(data)
    assert p_c == pytest.approx(0.125, abs=0.01)
    assert nu == pytest.approx(1.4, abs=0.35)   # nu is the softer direction
    assert np.isfinite(q)


def test_no_false_collapse_on_structureless_data():
    # Independent noise per size: no scaling form exists. The fitted collapse
    # quality must be clearly worse than for genuinely-scaling data.
    rng = np.random.default_rng(3)
    ps = np.linspace(0.05, 0.20, 31)
    data = {L: (ps, rng.standard_normal(len(ps))) for L in (32, 48, 64, 96)}
    _, _, q_noise = fit_collapse(data)
    _, _, q_real = fit_collapse(_synthetic(noise=0.005, seed=4))
    assert q_real < 0.1 * q_noise
