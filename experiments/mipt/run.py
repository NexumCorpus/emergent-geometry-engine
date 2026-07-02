"""Phase 3 MIPT-geometry sweep driver (Constitution sec 6, Phase 3 gate).

Per (L, p): run `ntraj` independent monitored-circuit trajectories, compute the
geometric observables PER TRAJECTORY (Adapter Q) plus the I3 control on the
IDENTICAL trajectory, then aggregate mean/sem. Every trajectory is
reproducible: seed = SeedSequence([base_seed, L, p_index, traj_index]).

Usage:
  python -m experiments.mipt.run --sizes 12,16,20 --ntraj 100 \
      --ps 0.02,0.05,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.24,0.30 \
      --tmult 2 --base-seed 20260702 --out artifacts/mipt_sweep

Writes one JSON per (L, p) with per-trajectory arrays (nothing hidden by the
aggregation) and a sweep-level summary.json including the frozen recipe.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from adapters.quantum import trajectory_observables
from experiments.mipt.circuits import run_trajectory
from experiments.mipt.stabilizer import block_mi_matrix, tripartite_mi
from geometry.config import FROZEN_RECIPE

# Geometry reads the BLOCK MI graph (see block_mi_matrix: site-level pairwise
# MI is degenerate in scrambled stabilizer states); the I3 control reads site
# quarters on the identical trajectory.
BLOCK_SITES = 4


def _traj_seed(base: int, L: int, ip: int, k: int) -> int:
    return int(np.random.SeedSequence([base, L, ip, k]).generate_state(1)[0])


def sweep_point(L: int, p: float, ip: int, ntraj: int, tmult: int, base_seed: int):
    keys = None
    rows = []
    for k in range(ntraj):
        sim = run_trajectory(L=L, p=p, T=tmult * L,
                             seed=_traj_seed(base_seed, L, ip, k))
        obs = trajectory_observables(block_mi_matrix(sim, BLOCK_SITES))
        obs["i3"] = tripartite_mi(sim)
        if keys is None:
            keys = [x for x in obs if x != "n"]
        rows.append(obs)
    out = {"L": L, "p": p, "ntraj": ntraj, "T": tmult * L,
           "per_trajectory": {key: [r[key] for r in rows] for key in keys},
           "mean": {}, "sem": {}, "n_valid": {}}
    for key in keys:
        a = np.array([r[key] for r in rows], dtype=float)
        v = a[np.isfinite(a)]
        out["n_valid"][key] = int(v.size)
        out["mean"][key] = float(v.mean()) if v.size else None
        out["sem"][key] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="32,48,64")
    ap.add_argument("--ps", default="0.02,0.05,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.24,0.30")
    ap.add_argument("--ntraj", type=int, default=100)
    ap.add_argument("--tmult", type=int, default=2)
    ap.add_argument("--base-seed", type=int, default=20260702)
    ap.add_argument("--out", default="artifacts/mipt_sweep")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    ps = [float(s) for s in args.ps.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"sizes": sizes, "ps": ps, "ntraj": args.ntraj,
                "tmult": args.tmult, "base_seed": args.base_seed,
                "block_sites": BLOCK_SITES,
                "frozen_recipe": FROZEN_RECIPE, "points": []}
    t0 = time.time()
    for L in sizes:
        for ip, p in enumerate(ps):
            t1 = time.time()
            point = sweep_point(L, p, ip, args.ntraj, args.tmult, args.base_seed)
            name = f"L{L}_p{p:.3f}.json"
            (out_dir / name).write_text(json.dumps(point), encoding="utf-8")
            manifest["points"].append(name)
            print(f"[sweep] L={L} p={p:.3f} done in {time.time()-t1:.1f}s | "
                  f"i3={point['mean']['i3']:+.3f} "
                  f"delta_norm={point['mean']['delta_norm']} "
                  f"lcc={point['mean']['lcc_frac']:.2f}", flush=True)
    manifest["wall_s"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=1),
                                          encoding="utf-8")
    print(f"[sweep] complete in {manifest['wall_s']}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
