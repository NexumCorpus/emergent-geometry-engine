"""Trusted-oracle check for the Phase 3 MIPT-geometry claim (grader contract).

Usage: python check.py <seed>

Exits 0 iff, for a FRESH ensemble of 50 monitored-circuit trajectories drawn
from base seed <seed> (L=32, T=64, block=4, the ensemble of PHASE3_REPORT.md),
the declared signal structure reproduces:

  1. I3 control locates the phases:    mean I3(p=0.02) < -2.0 bits
                                        |mean I3(p=0.30)| <  0.3 bits
  2. spectral dimension peaks at the transition region:
        d_s(0.12) > d_s(0.02) + 0.1   and   d_s(0.12) > d_s(0.30) + 0.1
  3. mean Forman curvature dips at the transition region:
        F(0.12) < F(0.02) - 0.2       and   F(0.12) < F(0.30) - 0.2

Margins are ~4x smaller than the gaps measured in the claimed sweep (e.g. the
d_s gap at L=32 was ~0.4), so a real effect passes easily and a null fails.
All simulation/geometry logic is imported from the EGE repo (commit recorded
in claim.json) — nothing is reimplemented here. Ordering assertions only; no
exact values, so the check cannot be satisfied by memorized numbers.
"""
import sys
from pathlib import Path

EGE = r"E:\emergent-geometry-engine"

L, T, BLOCK, NTRAJ = 32, 64, 4, 50
P_LOW, P_CRIT, P_HIGH = 0.02, 0.12, 0.30


def point(base_seed, p, ip):
    import numpy as np
    from adapters.quantum import trajectory_observables
    from experiments.mipt.circuits import run_trajectory
    from experiments.mipt.stabilizer import block_mi_matrix, tripartite_mi

    i3s, dss, fms = [], [], []
    for k in range(NTRAJ):
        seed = int(np.random.SeedSequence([base_seed, L, ip, k]).generate_state(1)[0])
        sim = run_trajectory(L=L, p=p, T=T, seed=seed)
        obs = trajectory_observables(block_mi_matrix(sim, BLOCK))
        i3s.append(tripartite_mi(sim))
        dss.append(obs["spectral_dim"])
        fms.append(obs["forman_mean"])
    import numpy as np
    mean = lambda a: float(np.nanmean(np.array(a, dtype=float)))
    return mean(i3s), mean(dss), mean(fms)


def main():
    base_seed = int(sys.argv[1])
    sys.path.insert(0, EGE)

    i3_lo, ds_lo, fm_lo = point(base_seed, P_LOW, 0)
    i3_cr, ds_cr, fm_cr = point(base_seed, P_CRIT, 1)
    i3_hi, ds_hi, fm_hi = point(base_seed, P_HIGH, 2)

    checks = [
        ("i3 volume-law", i3_lo < -2.0, f"I3({P_LOW})={i3_lo:+.2f} < -2.0"),
        ("i3 area-law", abs(i3_hi) < 0.3, f"|I3({P_HIGH})|={abs(i3_hi):.2f} < 0.3"),
        ("ds peak vs low", ds_cr > ds_lo + 0.1,
         f"ds({P_CRIT})={ds_cr:.3f} > ds({P_LOW})={ds_lo:.3f} + 0.1"),
        ("ds peak vs high", ds_cr > ds_hi + 0.1,
         f"ds({P_CRIT})={ds_cr:.3f} > ds({P_HIGH})={ds_hi:.3f} + 0.1"),
        ("forman dip vs low", fm_cr < fm_lo - 0.2,
         f"F({P_CRIT})={fm_cr:.3f} < F({P_LOW})={fm_lo:.3f} - 0.2"),
        ("forman dip vs high", fm_cr < fm_hi - 0.2,
         f"F({P_CRIT})={fm_cr:.3f} < F({P_HIGH})={fm_hi:.3f} - 0.2"),
    ]
    ok = True
    for name, passed, detail in checks:
        print(("ok  " if passed else "FAIL") + f" {name}: {detail}")
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
