import sys, os, numpy as np
sys.path.insert(0, ".")
from num21_two_class import run_ep, D2R, NMET, NTGT, NINP
VM, VT, DT, TMAX = 250.0, 240.0, 0.02, 40.0
L = float(os.environ["L_OVR"])
PLANTS = [(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0), (1, 1, 0.12)]   # P0 P1 P2 P3
out = np.zeros((4, NTGT, NINP, NMET))
for pi, (pl, rg, cs) in enumerate(PLANTS):
    for ti in range(NTGT):
        for ii in range(NINP):
            out[pi, ti, ii] = run_ep(3, pl, ti, ii, 1200.0, 34.4*D2R, 0.0, VM, VT,
                                     DT, TMAX, 1000+7*ti+13*ii, rg, 1, 0.667, cs)
np.save(f"_ls4_{L:.3f}.npy", out)
