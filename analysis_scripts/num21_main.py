"""num21_main.py —— num21 的驱动：先冒烟，再正式跑全场景对照。"""
import sys, time, math
import numpy as np

sys.path.insert(0, ".")
from num21_two_class import (run_ep, run_ep_dbg, CTRL_NAMES, TGT_NAMES, INP_NAMES,
                             NMET, NCTRL, NTGT, NINP, D2R, R2D, L_CAL)

VM, VT = 250.0, 240.0
DT = 0.02
TMAX = 40.0


def hdr():
    print("控制器                             " +
          "  ".join(f"{n:>8}" for n in
                    ["t_las", "t_lead", "follow均", "e_t均", "jerk", "tv", "flip",
                     "超调", "t_set", "|b|max", "e_t_p95"]))


def row(ci, o):
    print(f"{CTRL_NAMES[ci]:<32}" +
          "  ".join(f"{v:8.2f}" for v in
                    [o[0], o[1], o[13], o[10], o[2], o[3] / 100.0, o[4], o[6], o[7],
                     o[12], o[11]]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # 8 控制器 x 8 目标机动 x 4 鼠标档 x 4 被控对象档 = 1024 episode
        PLANTS = [(0, 0, 1.0, "P0 降阶 rate-command（无耦合）"),
                  (0, 1, 1.0, "P1 +RotGravR 重力投影耦合"),
                  (1, 1, 1.00, "P2 完整刚体·裸机（耦合上界）"),
                  (1, 1, 0.12, "P3 完整刚体·SAS 残余 12%")]
        res = np.zeros((len(PLANTS), NCTRL, NTGT, NINP, NMET))
        t0 = time.time()
        for pi, (pl, rg, cs, tag) in enumerate(PLANTS):
            for ti in range(NTGT):
                for ii in range(NINP):
                    for ci in range(NCTRL):
                        seed = 1000 + 7 * ti + 13 * ii
                        res[pi, ci, ti, ii] = run_ep(
                            ci, pl, ti, ii, 1200.0, 34.4 * D2R, 0.0, VM, VT, DT,
                            TMAX, seed, rg, 3 if ci >= 5 else 1, L_CAL, cs)
            print(f"  [{tag}] done  {time.time()-t0:.0f}s", flush=True)
        np.savez("num21_full.npz", res=res,
                 ctrl=np.array(CTRL_NAMES), tgt=np.array(TGT_NAMES),
                 inp=np.array(INP_NAMES),
                 plant=np.array([p[3] for p in PLANTS]))
        print("saved num21_full.npz", res.shape, f"{time.time()-t0:.0f}s")
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        for (plant, rg, cs, tag) in [(0, 0, 1.0, "P0 降阶 rate-command（无耦合）"),
                                     (0, 1, 1.0, "P1 +RotGravR 重力投影耦合"),
                                     (1, 1, 1.00, "P2 完整刚体·裸机（耦合上界）"),
                                     (1, 1, 0.12, "P3 完整刚体·SAS 残余 12%")]:
            print(f"=== 冒烟 R0=1200 aspect0=34.4deg th0=0 ninp=0 tgt=0 | {tag} ===")
            hdr()
            for ci in range(NCTRL):
                o = run_ep(ci, plant, 0, 0, 1200.0, 34.4 * D2R, 0.0, VM, VT, DT,
                           TMAX, 12345, rg, 3 if ci >= 5 else 1, L_CAL, cs)
                row(ci, o)
            print()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "dbg":
        ci = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        cs = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12
        tr = run_ep_dbg(ci, 1, 0, 0, 1200.0, 34.4 * D2R, 0.0, VM, VT, DT, 14.0,
                        12345, 1, 3 if ci >= 5 else 1, L_CAL, cs)
        print("k     t     e_t     e_f    om_r   om_p   om_y    ur     up     uy"
              "     R     lamdot   gam    th")
        for k in range(0, len(tr), 25):
            t = k * DT
            print(f"{k:4d} {t:5.2f} " + " ".join(f"{v:6.2f}" for v in tr[k]))
        sys.exit(0)
