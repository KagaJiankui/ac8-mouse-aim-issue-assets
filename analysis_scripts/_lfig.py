# -*- coding: utf-8 -*-
"""图 F: L 灵敏度 —— 平滑 vs 跟手的显式权衡曲线。"""
import numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Heiti SC",
                                   "LXGW WenKai GB", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

M = dict(t_las=0, t_lead=1, jerk=2, tv=3, flips=4, tv_u=5, over=6, tset=7,
         smooth=8, t=9, et=10, etp=11, beta=12, fol=13, folp=14, nk=15)
Ls = [0.150, 0.250, 0.350, 0.450, 0.550, 0.667, 0.800]
PL = ["P0 降阶 rate-command（无耦合）", "P1 +RotGravR 重力投影", "P2 完整刚体·裸机（上界）", "P3 完整刚体·SAS12%（决策档）"]
COL = ["#2E7D32", "#1565C0", "#EF6C00", "#C62828"]
MK = ["o", "s", "^", "D"]

D = {L: np.load(f"_ls4_{L:.3f}.npy") for L in Ls if os.path.exists(f"_ls4_{L:.3f}.npy")}
Ls = [L for L in Ls if L in D]

def med(a, k):
    v = a[..., M[k]].ravel()
    return np.nanmedian(v)

fig, axs = plt.subplots(1, 3, figsize=(17.4, 5.4))

# --- (a) 平滑性: jerk 归一化 ---
ax = axs[0]
for pi, (nm, c, m) in enumerate(zip(PL, COL, MK)):
    y = [med(D[L][pi], "jerk") for L in Ls]
    y = np.array(y) / y[0]
    ax.plot(Ls, y, marker=m, color=c, lw=2.2, ms=7, label=nm)
ax.set_xlabel("控制器侧延迟估计 $L_{est}$ [s]"); ax.set_ylabel("jerk 归一化（相对 L=0.15）")
ax.set_title("(a) 平滑性：角加速度 RMS 随 L 单调下降", fontsize=11, fontweight="bold")
ax.grid(alpha=.3); ax.legend(fontsize=9, framealpha=.9)
ax.axvspan(0.45, 0.70, color="#FFC107", alpha=.16, zorder=0)
ax.text(0.575, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.04,
        "拐点区\n0.45–0.70", ha="center", fontsize=8.5, color="#8D6E00")

# --- (b) 跟手性: follow 归一化 ---
ax = axs[1]
for pi, (nm, c, m) in enumerate(zip(PL, COL, MK)):
    y = [med(D[L][pi], "fol") for L in Ls]
    y = np.array(y) / y[0]
    ax.plot(Ls, y, marker=m, color=c, lw=2.2, ms=7, label=nm)
ax.axhline(1.0, color="#888", ls="--", lw=1, zorder=0)
ax.set_xlabel("控制器侧延迟估计 $L_{est}$ [s]"); ax.set_ylabel("跟随误差 归一化（相对 L=0.15）")
ax.set_title("(b) 跟手性：L 越大越滞后（P2/P3 代价最大）", fontsize=11, fontweight="bold")
ax.grid(alpha=.3); ax.legend(fontsize=9, framealpha=.9)
ax.axvspan(0.45, 0.70, color="#FFC107", alpha=.16, zorder=0)

# --- (c) Pareto: follow vs jerk (绝对量) ---
ax = axs[2]
for pi, (nm, c, m) in enumerate(zip(PL, COL, MK)):
    x = [med(D[L][pi], "jerk") for L in Ls]
    y = [med(D[L][pi], "fol") for L in Ls]
    ax.plot(x, y, marker=m, color=c, lw=2.0, ms=8, label=nm)
    for xi, yi, L in zip(x, y, Ls):
        ax.annotate(f"{L:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(5, 5), fontsize=7.4, color=c)
    # 标出 0.15 起点
    ax.plot(x[0], y[0], "o", ms=13, mfc="none", mec=c, mew=2.2)
ax.set_xlabel("jerk（角加速度 RMS）↓ 越平滑"); ax.set_ylabel("follow [°] ↓ 越跟手")
ax.set_title("(c) 权衡前沿：点内数字为 L；空心圈 = 当前 mod 值 0.15", fontsize=11, fontweight="bold")
ax.grid(alpha=.3); ax.legend(fontsize=9, framealpha=.9)
ax.invert_xaxis()

fig.suptitle("图 F  L 灵敏度扫描（控制器 = C3 标定档；被控对象恒用真实 $L_{plant}$=0.667 s；128 episode/点）",
             fontsize=12.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig("figs/fig21F_Lsweep.png", dpi=150, bbox_inches="tight")
print("saved figs/fig21F_Lsweep.png")

# --- 控制台再打一份关键数字 ---
print(f"\n{'L':>6} | {'P0 jerk':>8} {'P0 fol':>7} | {'P1 jerk':>8} {'P1 fol':>7} | {'P3 jerk':>8} {'P3 fol':>7}")
for L in Ls:
    print(f"{L:>6.3f} | " + " | ".join(
        f"{med(D[L][pi],'jerk'):>8.1f} {med(D[L][pi],'fol'):>7.2f}" for pi in range(4)))
