"""Quick honest inspection of a grokking run's signal trajectories."""
import os
import sys

import numpy as np

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
tag = sys.argv[1] if len(sys.argv) > 1 else "frac030"
d = np.load(os.path.join(ART, f"grokking_results_{tag}.npz"))
s = d["step"]


def row(i):
    return (f"{int(s[i]):6d}  te={d['test_acc'][i]:5.2f}  delta={d['gromov_delta'][i]:5.2f}  "
            f"varF={d['var_forman'][i]:6.2f}  mForm={d['mean_forman'][i]:6.2f}  "
            f"sdim={d['spectral_dim'][i]:4.2f}  shc={d['shc_test'][i]:9.2e}")


print(f"=== {tag} ===")
print("  step  test  delta  varForman  meanForman  sdim    SHC")
for st in (0, 500, 1000, 2000, 3000, 5000, 8000, 9000, 11000, 13000, 13800, 16000, 20000, 25000):
    i = int(np.argmin(np.abs(s - st)))
    print(row(i))

de = d["gromov_delta"]
vf = d["var_forman"]
print(f"\ndelta: start~{np.median(de[:5]):.3f}  end~{np.median(de[-5:]):.3f}  "
      f"min={de.min():.3f}  max={de.max():.3f}")
print(f"varForman: start~{np.median(vf[:5]):.2f}  end~{np.median(vf[-5:]):.2f}")
# fraction of total delta fall achieved by the long plateau midpoint vs at grokking
top = np.median(de[:5]); bot = np.median(de[-5:])
half = top - 0.5 * (top - bot)
first_half = next((int(s[i]) for i in range(len(s)) if de[i] <= half), None)
print(f"delta half-fall step={first_half}  (test acc there = "
      f"{d['test_acc'][int(np.argmin(np.abs(s-(first_half or 0))))]:.3f})")
