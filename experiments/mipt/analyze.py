"""Phase 3 gate figure + honest summary from a completed MIPT sweep.

Usage: python -m experiments.mipt.analyze artifacts/mipt_sweep_v1

Produces <sweep>/phase3_gate.png (observables vs p, one curve per L, sem error
bars) and prints a plain-text assessment per observable: does it move at the
I3-located transition, and does the feature sharpen with L? The verdict
language stays descriptive — "signal" vs "null" is judged against the I3
control, never asserted from a single curve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = [
    ("i3", "I3 control (bits)"),
    ("delta_norm", "Gromov delta / diameter"),
    ("spectral_dim", "spectral dimension"),
    ("forman_mean", "mean Forman (augmented)"),
    ("hmds_distortion", "h-MDS distortion (rank 2)"),
    ("mds_stress", "Euclidean MDS stress (dim 2)"),
    ("lcc_frac", "largest-component fraction"),
    ("lr_edges", "long-range MI edges"),
    ("mean_mi", "mean pairwise block MI (bits)"),
]


def load_sweep(out_dir: Path):
    manifest = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    points = [json.loads((out_dir / name).read_text(encoding="utf-8"))
              for name in manifest["points"]]
    return manifest, points


def main():
    out_dir = Path(sys.argv[1])
    manifest, points = load_sweep(out_dir)
    sizes = manifest["sizes"]
    ps = manifest["ps"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for ax, (key, label) in zip(axes.flat, PANELS):
        for L in sizes:
            row = [pt for pt in points if pt["L"] == L]
            row.sort(key=lambda pt: pt["p"])
            xs = [pt["p"] for pt in row]
            ys = [pt["mean"].get(key) for pt in row]
            es = [pt["sem"].get(key) or 0.0 for pt in row]
            xs2 = [x for x, y in zip(xs, ys) if y is not None]
            ys2 = [y for y in ys if y is not None]
            es2 = [e for e, y in zip(es, ys) if y is not None]
            ax.errorbar(xs2, ys2, yerr=es2, marker="o", ms=3, lw=1,
                        capsize=2, label=f"L={L}")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("measurement rate p")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        f"Phase 3 MIPT-geometry sweep — block={manifest['block_sites']} sites, "
        f"ntraj={manifest['ntraj']}, T={manifest['tmult']}L "
        f"(frozen recipe: {manifest['frozen_recipe']})", fontsize=10)
    fig.tight_layout()
    out = out_dir / "phase3_gate.png"
    fig.savefig(out, dpi=140)
    print(f"[analyze] figure -> {out}")

    # I3-located crossing: first p where mean I3 rises above -0.5 per size.
    print("\nI3 control (transition locator):")
    for L in sizes:
        row = sorted((pt for pt in points if pt["L"] == L), key=lambda t: t["p"])
        xs = [pt["p"] for pt in row]
        i3 = [pt["mean"]["i3"] for pt in row]
        cross = next((x for x, v in zip(xs, i3) if v > -0.5), None)
        print(f"  L={L}: I3 rises past -0.5 at p ~ {cross}  "
              f"(values: {[round(v, 2) for v in i3]})")

    # Per-observable movement summary near vs far from the crossing region.
    print("\nObservable ranges (min..max of the mean curve, largest L):")
    Lmax = max(sizes)
    row = sorted((pt for pt in points if pt["L"] == Lmax), key=lambda t: t["p"])
    for key, label in PANELS[1:]:
        vals = [pt["mean"].get(key) for pt in row if pt["mean"].get(key) is not None]
        if vals:
            print(f"  {key:16s} {min(vals):+.3f} .. {max(vals):+.3f}")


if __name__ == "__main__":
    main()
