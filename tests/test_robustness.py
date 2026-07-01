"""Graph-construction robustness harness (Constitution sec 2.2, 3 — the main referee
attack surface). Re-run the cheap invariants across a grid of (k, alpha) graph-build
choices and report signal stability. Uses Forman + Gromov delta + spectral dimension
(the cheap, per-trajectory invariants); exact LLY is reserved for anchor checkpoints.
"""
import numpy as np
import pytest

from geometry.robustness import robustness_sweep


def _ring_distance_matrix(n: int) -> np.ndarray:
    ang = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(ang), np.sin(ang)])
    return np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)


def test_sweep_covers_all_configs():
    D = _ring_distance_matrix(16)
    res = robustness_sweep(D, k_values=(2, 3, 4), alpha_values=(0.0, 0.5))
    assert len(res.configs) == 6
    for cfg in res.configs:
        assert set(["k", "alpha", "mean_forman", "delta", "spectral_dim", "n_edges"]) <= set(cfg)
        assert np.isfinite(cfg["mean_forman"])
        assert np.isfinite(cfg["delta"]) and cfg["delta"] >= -1e-12
        assert np.isfinite(cfg["spectral_dim"])
        assert cfg["n_edges"] > 0


def test_sweep_reports_stability_summary():
    D = _ring_distance_matrix(16)
    res = robustness_sweep(D, k_values=(2, 3, 4), alpha_values=(0.0, 0.5))
    for key in ("mean_forman", "delta", "spectral_dim"):
        assert key in res.stability
        s = res.stability[key]
        assert np.isfinite(s["mean"]) and np.isfinite(s["std"]) and s["std"] >= 0.0
        assert np.isfinite(s["cov"])


def test_ring_dimension_is_robustly_about_one():
    # A ring is intrinsically 1D; the recovered spectral dimension should stay near 1
    # across graph-construction choices — a genuine (not contrived) stability claim.
    D = _ring_distance_matrix(24)
    res = robustness_sweep(D, k_values=(2, 3), alpha_values=(0.0, 0.5))
    dims = [cfg["spectral_dim"] for cfg in res.configs]
    assert all(0.5 <= d <= 2.0 for d in dims)
