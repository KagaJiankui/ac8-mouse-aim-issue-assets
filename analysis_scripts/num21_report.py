"""num21_report.py —— 读 num21_full.npz + fr_calib.npz，出表 + 出图。"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Heiti SC",
                                   "LXGW WenKai GB", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

D = np.load("num21_full.npz", allow_pickle=True)
res = D["res"]                       # (3, 8, 8, 4, 16)
CTRL = [str(c) for c in D["ctrl"]]
TGT = [str(t) for t in D["tgt"]]
INP = [str(i) for i in D["inp"]]
PLANT = [str(p) for p in D["plant"]]
NP_, NC, NT, NI, NM = res.shape

M = dict(t_las=0, t_lead=1, jerk=2, tv=3, flips=4, tv_u=5, over=6, tset=7,
         smooth=8, t=9, et=10, etp=95 if False else 11, beta=12, fol=13, folp=14, nk=15)


def agg(a, k, pct=50, only_ok=False):
    """对 (NC,NT,NI) 的某个指标做统计；only_ok 只统计 t_lead>0 的样本。"""
    v = a[:, :, :, k]
    if only_ok:
        ok = a[:, :, :, M["t_lead"]] > 0
        v = np.where(ok, v, np.nan)
    f = np.nanmedian if pct == 50 else (lambda x, axis: np.nanpercentile(x, pct, axis=axis))
    return f(v, axis=(1, 2))


# ============================================================ 表 1: FR 标定
def table1():
    C = np.load("fr_calib.npz", allow_pickle=True)
    print("\n" + "=" * 96)
    print("表 1  AC7 PPDT 速率标定（n=28 架，GameFAQs 实测表）  vs  mod flight_math.h 的硬编码")
    print("=" * 96)
    rows = [("等效速率斜坡 L [s]", C["L"], 0.15, "Δ"),
            ("俯仰 FR_p [°/s]", C["FRp"], 55.0, "Δ"),
            ("滚转 FR_r [°/s]", C["FRr"], 170.0, "Δ"),
            ("偏航 FR_y [°/s]", C["FRy"], 10.0, "Δ")]
    print(f"{'量':<22}{'min':>9}{'p25':>9}{'中位':>9}{'p75':>9}{'max':>9}"
          f"{'mod':>10}{'中位/mod':>11}")
    for nm, v, md, _ in rows:
        q = np.percentile(v, [0, 25, 50, 75, 100])
        print(f"{nm:<22}" + "".join(f"{x:9.3f}" for x in q) +
              f"{md:10.3f}{md/max(q[2],1e-9):11.2f}")
    print(f"  三轴比例（以 pitch 归一）  AC7 中位: roll {np.median(C['FRr'])/np.median(C['FRp']):.2f}"
          f" / pitch 1.00 / yaw {np.median(C['FRy'])/np.median(C['FRp']):.2f}"
          f"   |  mod: roll {170/55:.2f} / pitch 1.00 / yaw {10/55:.2f}")


# ============================================================ 表 2: 消融阶梯
def table2():
    print("\n" + "=" * 96)
    print("表 2  消融阶梯：8 控制器 x 8 目标机动 x 4 鼠标档（中位数），按被控对象分块")
    print("=" * 96)
    keys = [("t_las", "t_las"), ("t_lead", "t_lead"), ("fol", "跟随[°]"),
            ("over", "超调[°]"), ("jerk", "jerk"), ("tv", "tv"), ("flips", "翻转"),
            ("tset", "t_settle")]
    for pi in range(NP_):
        print(f"\n--- {PLANT[pi]} ---")
        print(f"{'控制器':<24}" + "".join(f"{lbl:>10}" for _, lbl in keys) + f"{'成功率':>9}")
        for ci in range(NC):
            a = res[pi, ci]
            vals = []
            for k, _ in keys:
                v = a[:, :, M[k]]
                m = np.median(v, axis=(0, 1))
                vals.append(m)
            ok = (a[:, :, M["t_lead"]] > 0).mean() * 100
            print(f"{CTRL[ci]:<24}" + "".join(f"{x:10.2f}" for x in vals) + f"{ok:8.0f}%")


# ============================================================ 表 3: 每目标
def table3():
    print("\n" + "=" * 96)
    print("表 3  每目标机动：follow[°] / t_lead[s]（P3 完整刚体·SAS残余12%，4 档鼠标取中位数）")
    print("=" * 96)
    a2 = res[3]
    print(f"{'控制器':<24}" + "".join(f"{t[:7]:>14}" for t in TGT))
    for ci in range(NC):
        a = a2[ci]
        s = ""
        for ti in range(NT):
            fo = np.median(a[ti, :, M["fol"]])
            tl = np.median(a[ti, :, M["t_lead"]])
            s += f"{fo:7.1f}/{tl:6.1f}"
        print(f"{CTRL[ci]:<24}{s}")


# ============================================================ 表 4: 鼠标档
def table4():
    print("\n" + "=" * 96)
    print("表 4  鼠标输入档位敏感性（P3；8 目标中位）")
    print("=" * 96)
    a2 = res[3]
    print(f"{'控制器':<24}" + "".join(f"{i[:10]:>12}" for i in INP) + f"{'抖/理想':>10}")
    for ci in range(NC):
        a = a2[ci]
        fo = [np.median(a[:, ii, M["fol"]]) for ii in range(NI)]
        jk = [np.median(a[:, ii, M["jerk"]]) for ii in range(NI)]
        print(f"{CTRL[ci]:<24}" + "".join(f"{x:7.1f}/{y:4.0f}" for x, y in zip(fo, jk)) +
              f"{fo[2]/max(fo[0],1e-6):10.2f}")


# ============================================================ 图
def figs():
    COL = ["#B0B7C3", "#5B8FF9", "#9270CA", "#F6BD16", "#5AD8A6",
           "#E8684A", "#6DC8EC", "#FF9D4D"]
    # --- fig A: FR 标定 ---
    C = np.load("fr_calib.npz", allow_pickle=True)
    f, axs = plt.subplots(2, 2, figsize=(11, 7))
    spec = [("L", "等效速率斜坡 L [s]", 0.15, C["L"]),
            ("FRp", "俯仰速率 FR_p [°/s]", 55.0, C["FRp"]),
            ("FRr", "滚转速率 FR_r [°/s]", 170.0, C["FRr"]),
            ("FRy", "偏航速率 FR_y [°/s]", 10.0, C["FRy"])]
    for ax, (_, lb, md, v) in zip(axs.ravel(), spec):
        ax.hist(v, bins=12, color="#5B8FF9", edgecolor="white")
        ax.axvline(md, color="#E8684A", lw=2)
        ax.axvline(np.median(v), color="#F6BD16", lw=2, ls="--")
        ax.text(md, ax.get_ylim()[1] * 0.92, f"  mod={md}", color="#E8684A", fontsize=9)
        ax.text(np.median(v), ax.get_ylim()[1] * 0.72, f"  AC7中位={np.median(v):.2f}",
                color="#B08900", fontsize=9)
        ax.set_title(lb, fontsize=11)
        ax.set_ylabel("架数" if ax is axs[0, 0] else "")
    f.suptitle("图 A  AC7 实测标定 vs mod 硬编码（n=28）", fontsize=13)
    f.tight_layout()
    f.savefig("figs/fig21A_fr_calib.png")
    plt.close(f)

    # --- fig B: 跟踪 vs 顺滑 帕累托 ---
    f, axs = plt.subplots(1, 4, figsize=(16.5, 4.4))
    for pi in range(NP_):
        ax = axs[pi]
        for ci in range(NC):
            fo = np.median(res[pi, ci, :, :, M["fol"]])
            tv = np.median(res[pi, ci, :, :, M["tv"]])
            ax.scatter(fo, tv, s=90, color=COL[ci], zorder=3,
                       marker="o" if ci < 5 else ("s" if ci == 5 else "^"))
            ax.annotate(f"C{ci}", (fo, tv), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)
        ax.set_title(PLANT[pi], fontsize=11)
        ax.set_xlabel("跟随误差 follow 中位 [°]  (越小越跟手)")
        ax.set_ylabel("角速率总变差 tv  (越小越顺)" if pi == 0 else "")
        ax.grid(alpha=.3)
    f.suptitle("图 B  手感帕累托：跟随精度 vs 平滑度（8 目标 x 4 鼠标档，中位）", fontsize=13)
    f.tight_layout()
    f.savefig("figs/fig21B_pareto.png")
    plt.close(f)

    # --- fig C: 热图 follow (P2) ---
    f, axs = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for k, (ax, met, lb, cm) in enumerate([
            (axs[0], "fol", "跟随误差 follow [°] (越小越好)", "RdYlGn_r"),
            (axs[1], "jerk", "角加速度 RMS jerk [°/s²] (越小越好)", "RdYlGn_r")]):
        Z = np.zeros((NC, NT))
        for ci in range(NC):
            for ti in range(NT):
                Z[ci, ti] = np.median(res[3, ci, ti, :, M[met]])
        im = ax.imshow(Z, aspect="auto", cmap=cm)
        ax.set_xticks(range(NT), TGT, rotation=38, ha="right", fontsize=8)
        ax.set_yticks(range(NC), [c.split(" ")[0] for c in CTRL], fontsize=9)
        ax.set_title(lb, fontsize=11)
        plt.colorbar(im, ax=ax)
    f.suptitle("图 C  完整刚体·SAS残余12%（P3） 控制器 x 目标机动 指标热图", fontsize=13)
    f.tight_layout()
    f.savefig("figs/fig21C_heat.png")
    plt.close(f)

    # --- fig D: 鼠标档位鲁棒性 ---
    f, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(NI)
    for ci in range(NC):
        y = [np.median(res[2, ci, :, ii, M["fol"]]) for ii in range(NI)]
        ax.plot(x, y, "-o", color=COL[ci], label=f"C{ci} {CTRL[ci].split(' ')[1]}", lw=1.8)
    ax.set_xticks(x, INP, fontsize=9)
    ax.set_ylabel("跟随误差 follow 中位 [°]")
    ax.set_title("图 D  鼠标输入档位鲁棒性（P3，8 目标中位）", fontsize=12)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=2)
    f.tight_layout()
    f.savefig("figs/fig21D_mouse.png")
    plt.close(f)

    # --- fig E: 两类目标 击杀 vs 手感 ---
    f, axs = plt.subplots(1, 3, figsize=(13.5, 4.3))
    ax = axs[0]
    w = 0.38
    idx = [2, 3, 4, 5, 6, 7]        # C2 之后：标定 L / 前馈 / A类 / B类
    tl = [np.median(res[3, c, :, :, M["t_lead"]][res[3, c, :, :, M["t_lead"]] > 0])
          if (res[3, c, :, :, M["t_lead"]] > 0).any() else np.nan for c in idx]
    fo = [np.median(res[3, c, :, :, M["fol"]]) for c in idx]
    x = np.arange(len(idx))
    ax.bar(x - w / 2, tl, w, color="#E8684A", label="建立前置 t_lead [s]")
    ax.bar(x + w / 2, fo, w, color="#5B8FF9", label="跟随误差 follow [°]")
    ax.set_xticks(x, [f"C{c}" for c in idx])
    ax.set_title("击杀侧 vs 手感侧（P3）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axs[1]
    w = 0.2
    for pi in range(NP_):
        ok = [(res[pi, ci, :, :, M["t_lead"]] > 0).mean() * 100 for ci in range(NC)]
        ax.bar(np.arange(NC) + (pi - 1.5) * w, ok, w, color=COL[pi], label=PLANT[pi][:2])
    ax.set_xticks(range(NC), [f"C{i}" for i in range(NC)])
    ax.set_ylabel("建立前置成功率 [%]")
    ax.set_title("成功率：四档被控对象", fontsize=11)
    ax.set_ylim(0, 92)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3, axis="y")

    ax = axs[2]
    for pi, mk in zip(range(NP_), ["o", "s", "^", "D"]):
        y = [np.median(res[pi, ci, :, :, M["over"]]) for ci in range(NC)]
        ax.plot(range(NC), y, mk + "-", label=PLANT[pi], lw=1.6, ms=5)
    ax.set_xticks(range(NC), [f"C{i}" for i in range(NC)])
    ax.set_ylabel("超调 [°]")
    ax.set_title("四档被控对象下的超调", fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    f.suptitle("图 E  击杀侧 / 手感侧 / 成功率 对照", fontsize=13)
    f.tight_layout()
    f.savefig("figs/fig21E_two_class.png")
    plt.close(f)
    print("\nfigs saved: fig21A_fr_calib / fig21B_pareto / fig21C_heat / fig21D_mouse / fig21E_two_class")


if __name__ == "__main__":
    table1()
    table2()
    table3()
    table4()
    figs()
