# -*- coding: utf-8 -*-
"""汇总 L 灵敏度扫描 (_ls4_*.npy) -> 表格 + 比值表。
成功率口径 = (t_lead > 0) 的比例 (与 num21_report.py 一致);
t_lead / t_settle 的 -1 与 t_las 的哨兵值在统计时剔除。"""
import numpy as np, os

M = dict(t_las=0, t_lead=1, jerk=2, tv=3, flips=4, tv_u=5, over=6, tset=7,
         smooth=8, t=9, et=10, etp=11, beta=12, fol=13, folp=14, nk=15)

Ls = [0.150, 0.250, 0.350, 0.450, 0.550, 0.667, 0.800]
PL = ["P0 降阶 rate-command（无耦合）", "P1 +RotGravR 重力投影", "P2 完整刚体·裸机（上界）", "P3 完整刚体·SAS12%"]
D = {}
for L in Ls:
    f = f"_ls4_{L:.3f}.npy"
    if not os.path.exists(f):
        print(f"!! missing {f}"); continue
    D[L] = np.load(f)          # (3, 8, 4, 16)

OK_FREE = ["fol", "folp", "jerk", "tv", "flips", "over", "tv_u", "t_las"]
OK_ONLY = ["tset", "t_lead"]

def row_of(a):
    r = {}
    for k in OK_FREE:
        v = a[..., M[k]].ravel()
        v = v[np.isfinite(v)]
        r[k] = np.median(v) if v.size else np.nan
    for k in OK_ONLY:
        v = a[..., M[k]].ravel()
        v = v[(v > 0) & np.isfinite(v)]
        r[k] = np.median(v) if v.size else np.nan
    r["succ"] = (a[..., M["t_lead"]] > 0).mean() * 100.0
    return r

hdr = [("fol", "follow"), ("folp", "fol_p95"), ("jerk", "jerk"), ("tv", "tv"),
       ("flips", "flips"), ("over", "over"), ("tset", "t_settle"), ("t_lead", "t_lead"),
       ("t_las", "t_las"), ("succ", "succ%")]
ks = [k for k, _ in hdr]

for pi, pname in enumerate(PL[:3]):
    print("=" * 112)
    print(f"被控对象 {pname}   (控制器 = C3 +标定 L, 只扫控制器侧 L 估计; 被控对象恒用 L_plant=0.667)")
    print("=" * 112)
    rows = {}
    print(f"{'L':>6} " + " ".join(f"{h:>10}" for _, h in hdr))
    for L in Ls:
        if L not in D: continue
        rows[L] = row_of(D[L][pi])
        print(f"{L:>6.3f} " + " ".join(f"{rows[L][k]:>10.2f}" for k in ks))
    b = rows.get(0.150)
    print(f"\n  --- 相对 L=0.15 的比值 (越小越好, succ% 列为绝对百分点) ---")
    print(f"{'L':>6} " + " ".join(f"{h:>10}" for _, h in hdr))
    for L in Ls:
        if L not in D: continue
        r = rows[L]
        cells = []
        for k in ks:
            if k == "succ":
                cells.append(f"{r[k]:>10.1f}")
            elif abs(b[k]) > 1e-9 and np.isfinite(b[k]):
                cells.append(f"{r[k]/b[k]:>10.2f}")
            else:
                cells.append(f"{'—':>10}")
        print(f"{L:>6.3f} " + " ".join(cells))
    print()

print("=" * 112)
print("综合归一化代价 (三档被控对象平均; 指标 follow/fol_p95/jerk/tv/flips/over/t_lead/t_las 相对 L=0.15)")
print("=" * 112)
base = {pi: row_of(D[0.150][pi]) for pi in range(4)}
for L in Ls:
    if L not in D: continue
    acc = []
    for pi in range(4):
        r = row_of(D[L][pi])
        for k in ["fol", "folp", "jerk", "tv", "flips", "over", "t_lead", "t_las"]:
            bb = base[pi][k]
            if np.isfinite(bb) and abs(bb) > 1e-9 and np.isfinite(r[k]):
                acc.append(r[k] / bb)
    print(f"L={L:.3f}  归一化代价 = {np.mean(acc):.3f}   中位 = {np.median(acc):.3f}")
