"""num20_fr_calib.py —— 用社区逆向/实测数据标定 FR（每轴权限）与 τ（速率斜坡）

未验证假设 #1 的兑现：第一轮报告用了 FR = 170/55/10 °/s、τ = 0.1 s、ULIM = 1.0/0.85/0.7，
但从来没拿真实游戏数据标定过。这里用三份公开材料：

  S1  GameFAQs「Some basic performance stats for all aircraft」(CloverKitten, 2019)
      28 架机型，16,404 ft 平飞起测。字段：Pitch(俯仰 90° 耗时)、Loop(俯仰 360° 耗时)、
      Yaw(偏航 20° 耗时)、Roll(横滚 360° 耗时)。实测精度自述 ±0.1 s。
  S2  r/acecombat 玩家由 G 值反推的转向率（"400 kt → 13G，1000 kt → 33G"）。
  S3  PPDT 字段逆向（Nexus 2539 / 2075 / 1400）：SpeedRot = (pitch, yaw, roll) 三轴角速率能力，
      RotGravR = 有机体坡度时的俯仰/偏航「重力相对」加速度，第 3 分量恒为 0（**没有滚转加速度**）。

标定模型（与游戏 rate-command 结构一致）
----------------------------------------
    满杆阶跃从零速率起：  theta(t) = FR * [ t - L ]        (t > L)
其中 L 是「有效速率斜坡时间」——它把两种可能的实现合并成一个可辨识量：
    a) 一阶滞后  omega_dot = (FR - omega)/tau     -> L = tau * (1 - e^{-t/tau}) -> t 大时 L -> tau
    b) 纯传输延迟 delta                            -> L = delta
两者在「到达固定角度所需时间」上给出相同的面积亏损，用 ±0.1 s 精度的两点测量
（俯仰 90° 与 360°）无法区分。所以下面只报 L，并注明两种解释。

闭式解（本脚本的核心结果）
--------------------------
    俯仰有两点测量 (T90, T360)，两点必须给出同一个 FR：
        90/(T90 - L) = 360/(T360 - L)
    =>  L = (4*T90 - T360) / 3
    =>  FR_p = 270 / (T360 - T90)
横滚/偏航只有一点测量，用同一 L（或 L 的中位）反解：
        FR_r = 360/(T_roll - L)      FR_y = 20/(T_yaw - L)
"""
import numpy as np

# --------------------------------------------------------------------------
# S1: AC7 GameFAQs 实测表（28 架）。列: name, top_mph, short_accel_s, long_accel_s,
#     yaw20_s, pitch90_s, loop360_s, roll360_s
# --------------------------------------------------------------------------
AC7 = [
    ("F-104C",       1410, 11.80, 40.10, 4.10, 3.15, 9.60, 3.15),
    ("A-10C",        1069, 40.20, 53.75, 2.45, 2.70, 8.60, 2.70),
    ("F-2A",         1312, 11.35, 36.45, 3.30, 2.65, 8.40, 2.70),
    ("F-4E",         1410, 11.75, 39.95, 4.10, 3.20, 9.70, 3.30),
    ("F-14D",        1385,  9.40, 33.45, 3.60, 2.95, 9.35, 3.10),
    ("F-15C",        1556,  6.80, 26.85, 3.60, 2.80, 9.10, 3.00),
    ("F-15J",        1556,  6.80, 26.85, 3.30, 2.80, 9.10, 3.00),
    ("F-15E",        1458,  8.40, 31.85, 3.40, 2.35, 7.00, 2.65),
    ("F-16C",        1264, 16.05, 43.20, 4.10, 2.90, 9.65, 3.10),
    ("F/A-18F",      1385,  9.70, 34.40, 3.00, 2.65, 8.40, 2.65),
    ("F-22A",        1507,  7.50, 28.05, 3.10, 2.45, 8.45, 2.70),
    ("F-35C",        1361,  9.55, 31.20, 3.10, 2.55, 8.20, 2.75),
    ("YF-23",        1556,  6.90, 27.15, 3.00, 2.55, 8.65, 2.75),
    ("Mirage 2000-5",1337, 10.55, 35.55, 3.70, 2.50, 8.45, 2.70),
    ("Rafale M",     1410,  9.20, 33.40, 2.95, 2.10, 6.85, 2.60),
    ("Typhoon",      1434,  8.40, 32.10, 3.20, 2.80, 8.85, 2.85),
    ("Gripen E",     1385,  9.20, 31.05, 3.35, 2.60, 8.30, 2.70),
    ("MiG-21bis",    1264, 13.05, 37.90, 3.45, 2.60, 8.10, 2.15),
    ("MiG-29A",      1361, 10.35, 35.30, 3.45, 2.70, 9.20, 2.90),
    ("MiG-31B",      1678,  6.15, 26.55, 2.45, 2.80, 9.25, 3.00),
    ("Su-30M2",      1434,  8.60, 32.45, 3.85, 2.70, 9.05, 2.90),
    ("Su-30SM",      1449,  8.75, 32.85, 3.40, 2.60, 8.70, 2.80),
    ("Su-33",        1410,  9.10, 32.90, 3.75, 2.75, 9.00, 2.80),
    ("Su-34",        1312, 11.15, 36.00, 2.85, 2.60, 8.10, 2.65),
    ("Su-35S",       1458,  8.50, 32.25, 3.05, 2.60, 8.65, 2.65),
    ("Su-37",        1410,  9.25, 33.80, 3.45, 2.75, 8.85, 2.80),
    ("Su-47",        1361,  9.60, 33.60, 2.95, 2.50, 8.40, 2.45),
    ("Su-57",        1483,  7.55, 28.15, 3.10, 2.45, 7.95, 2.75),
    ("X-02S",        1604,  6.70, 28.70, 2.95, 2.35, 7.65, 2.35),
]

# 第一轮报告 / AC8-Mouse-Aim 仿真里用的参数（待标定对象）
MODEL_FR = dict(p=55.0, r=170.0, y=10.0)
MODEL_L = 0.15          # flight_math.h 里 arrival_rate() 硬写的 0.15f
MODEL_TAU = 0.10        # 第一轮建模用的一阶滞后


def calib():
    names = [a[0] for a in AC7]
    yaw = np.array([a[4] for a in AC7])
    p90 = np.array([a[5] for a in AC7])
    loop = np.array([a[6] for a in AC7])
    roll = np.array([a[7] for a in AC7])

    # ---- 俯仰: 两点测量 -> L 与 FR_p 的闭式解 ----
    L = (4.0 * p90 - loop) / 3.0
    FRp = 270.0 / (loop - p90)

    # ---- 横滚/偏航: 单点 + 同 L ----
    FRr = 360.0 / (roll - L)
    FRy = 20.0 / (yaw - L)

    print("=" * 104)
    print("A. 逐机标定（S1: AC7 GameFAQs 实测表, 28 架）")
    print("=" * 104)
    print(f"{'机型':>14s} {'T90':>5s} {'T360':>5s} {'Troll':>6s} {'Tyaw':>5s} | "
          f"{'L[s]':>6s} {'FR_p':>6s} {'FR_r':>6s} {'FR_y':>6s}")
    for i, n in enumerate(names):
        print(f"{n:>14s} {p90[i]:5.2f} {loop[i]:5.2f} {roll[i]:6.2f} {yaw[i]:5.2f} | "
              f"{L[i]:6.3f} {FRp[i]:6.1f} {FRr[i]:6.1f} {FRy[i]:6.2f}")

    print("\n" + "=" * 104)
    print("B. 分布（28 架）")
    print("=" * 104)
    print(f"{'量':>8s} {'min':>8s} {'p25':>8s} {'中位':>8s} {'p75':>8s} {'max':>8s} "
          f"{'均值':>8s} {'std':>7s}")
    for lab, v in (("L [s]", L), ("FR_p [°/s]", FRp),
                   ("FR_r [°/s]", FRr), ("FR_y [°/s]", FRy)):
        q = np.percentile(v, [0, 25, 50, 75, 100])
        print(f"{lab:>8s} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f} {q[3]:8.3f} {q[4]:8.3f} "
              f"{v.mean():8.3f} {v.std(ddof=1):7.3f}")

    print("\n" + "=" * 104)
    print("C. 与待标定模型的对照")
    print("=" * 104)
    med = dict(L=float(np.median(L)), p=float(np.median(FRp)),
               r=float(np.median(FRr)), y=float(np.median(FRy)))
    print(f"{'量':>10s} {'模型值':>10s} {'AC7 中位':>10s} {'AC7 区间':>20s} {'模型/中位':>10s} {'判定':>8s}")
    rows = [
        ("FR_p", MODEL_FR['p'], med['p'], (FRp.min(), FRp.max())),
        ("FR_r", MODEL_FR['r'], med['r'], (FRr.min(), FRr.max())),
        ("FR_y", MODEL_FR['y'], med['y'], (FRy.min(), FRy.max())),
        ("L (τ)", MODEL_L, med['L'], (L.min(), L.max())),
    ]
    for lab, mv, cv, (lo, hi) in rows:
        ratio = mv / cv
        verdict = "OK" if 0.7 <= ratio <= 1.4 else ("偏小" if ratio < 1 else "偏大")
        print(f"{lab:>10s} {mv:10.3f} {cv:10.3f} {f'[{lo:.2f}, {hi:.2f}]':>20s} "
              f"{ratio:10.2f} {verdict:>8s}")

    # ---- 三轴比例（PPDT 里 SpeedRot 存的是相对值，比例比绝对值更重要） ----
    print("\n" + "=" * 104)
    print("D. 三轴比例 FR_r : FR_p : FR_y （PPDT 的 SpeedRot 存相对值）")
    print("=" * 104)
    rel = np.vstack([FRr / FRp, np.ones_like(FRp), FRy / FRp])
    print(f"{'':>22s} {'roll':>8s} {'pitch':>8s} {'yaw':>8s}")
    print(f"{'AC7 中位（以 pitch=1）':>22s} {np.median(rel[0]):8.2f} {1.0:8.2f} "
          f"{np.median(rel[2]):8.2f}")
    print(f"{'AC7 均值':>22s} {rel[0].mean():8.2f} {1.0:8.2f} {rel[2].mean():8.2f}")
    print(f"{'mod 当前 FR=170/55/10':>22s} {MODEL_FR['r']/MODEL_FR['p']:8.2f} {1.0:8.2f} "
          f"{MODEL_FR['y']/MODEL_FR['p']:8.2f}")
    print(f"  -> AC7 区间 roll/pitch = [{rel[0].min():.2f}, {rel[0].max():.2f}]，"
          f"yaw/pitch = [{rel[2].min():.2f}, {rel[2].max():.2f}]")

    # ---- S2: r/acecombat 的 G 值反推 ----
    print("\n" + "=" * 104)
    print("E. 来源交叉核对（三份来源互相矛盾，这是本节的诚实结论）")
    print("=" * 104)
    g = 9.80665
    for kt, G in ((400, 13.0), (1000, 33.0)):
        V = kt * 0.514444
        om = G * g / V
        print(f"  S2: {kt:5d} kt 拉 {G:5.1f}G  -> 稳态角速率 {np.degrees(om):6.2f} °/s "
              f"（推出 ω 与速度无关 => 社区认为转向率不随速度变）")
    print(f"  S1: 俯仰 90° 耗时 2.35~3.20 s -> 平均 {90/p90.mean():.1f} °/s"
          f"（区间 {90/p90.max():.1f}~{90/p90.min():.1f}）")
    print(f"  S1: 筋斗 360° 耗时 6.85~9.70 s -> 平均 {360/loop.mean():.1f} °/s"
          f"（区间 {360/loop.max():.1f}~{360/loop.min():.1f}）")
    print(f"  => S2 与 S1 的 T90 一致（~36 °/s），但与 S1 的 T360（~46 °/s）差 "
          f"{(1 - (90/p90.mean())/(360/loop.mean()))*100:.0f}%。")
    print("     S2 是玩家「dirty math」反推，并非实测；本脚本以 S1 为准，S2 只作为量级核对。")

    # ---- F. 对求解器/控制器的影响 ----
    print("\n" + "=" * 104)
    print("F. 对 mod 控制律的直接影响")
    print("=" * 104)
    print(f"  flight_math.h 的 arrival_rate() 用固定 delay_speed = decel * {MODEL_L}f，")
    print(f"  即假设被控对象是「L={MODEL_L}s 延迟的双积分器」。")
    print(f"  标定结果显示 AC7 的等效斜坡 L = {L.min():.2f} ~ {L.max():.2f} s（中位 {med['L']:.2f}）。")
    print(f"  L 被低估 {med['L']/MODEL_L:.1f}x  ->  制动包络 "
          f"sqrt((La)^2+2ad)-La 严重偏小  ->  典型症状是【到位后停不住/超调振荡】。")
    print(f"  这与用户报的『方向稍微歪一点机体就会一直滚转』在机理上一致。")
    if med['L'] > 0:
        print(f"  建议: L 取 0.6~0.9 s（中位 {med['L']:.2f}），并做成可配置项。")
    np.savez("fr_calib.npz", names=np.array(names), L=L, FRp=FRp, FRr=FRr, FRy=FRy,
             p90=p90, loop=loop, roll=roll, yaw=yaw)
    return dict(L=L, FRp=FRp, FRr=FRr, FRy=FRy, med=med)


if __name__ == "__main__":
    calib()
