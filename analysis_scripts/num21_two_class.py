"""num21_two_class.py —— 两类目标函数 + 双指标（击杀效率 / 操纵手感）+ 真实鼠标输入

兑现「未验证假设」#2 与 #4，并落实用户新增的「分两类讨论」要求。

两类目标
--------
A 类：**鼠标操纵体验**（顺滑感，模仿战争雷霆专利的目标函数，不直接追求击杀效率）
    专利 US8770979B2 / US9858830B2 的内环 = 对候选控制做 N 步前向仿真，
    用「机头方向 · 目标方向」的点积评分挑最优（说明书 FIG.7 的 303C 实现，
    配 ×2/÷2 自适应步长）。这里形式化为
        J_A(u) = sum_k w_k * (1 - f_pred(k) . aim)          <- 专利的点积评分
               + lam_j * |u(k) - u(k-1)|^2                  <- 平滑正则（顺滑感）
               + lam_u * |u(k)|^2                           <- 控制能量
    注意它的 aim 是「玩家看到的指示框」= 目标本身，所以 A 类在几何上等价于纯追踪
    （第二轮报告的 J1）。它**不会**主动去建前置角，因为那会让机头偏离鼠标。

B 类：**击杀航线调整**（适当放弃操纵体验感）
    J6 分层：未进发射包线 -> 最小化进包线时间 T_LAS；已进包线 -> 最小化 ZEM（PPN）。
    代价是机头会主动离开鼠标方向去建前置 -> 「跟随度」指标会变差。

手感指标（本脚本新引入，之前两轮报告完全没有）
    jerk_rms   角加速度 RMS [deg/s^2]        —— 顿挫感
    tv_om      角速率总变差 sum|d om|        —— 抖动/来回修
    n_flip     角速率符号翻转次数             —— 来回摆
    overshoot  进入 +-2deg 后的最大超调 [deg]
    t_settle   稳定到 5deg 内所需时间 [s]
    smooth     1/(1 + tv_om/1000) 归一化顺滑分
    follow_e   机头与玩家鼠标方向(aim)的夹角均值 [deg] —— **直接对应玩家的"跟手"感受**

真实鼠标输入（未验证假设 #4 的兑现）
    三段式：抽拽(flick) + 周期性拉动(pull) + 微调(micro)，幅度服从幂律分布；
    手抖：OU 过程，相关时间由生理震颤 8~12 Hz 决定；
    轨迹形状：两-thirds 幂律（速度 ∝ 曲率^{-1/3}）。

几何：全套用 SO(3)（3x3 机体基），不用降阶 2D 平面角。plant=1 时额外打开
      刚体陀螺耦合 + 侧滑动力学 + 副翼不利偏航 —— 即「非初等滚转-横向耦合」。
"""
import math
import os
import numpy as np
from numba import njit

# 符号由 _symtest.py 的 ±1 对照实验确定: -1 让"坡度 -> 朝目标转弯"
# (C5 A 类 follow 19.45 -> 2.77 deg, C0 1.91 -> 1.61 deg)
PSI_SIGN = float(os.environ.get("PSI_SIGN", "-1"))
# L 灵敏度扫描用: >0 时强制覆盖控制器侧的 L 估计 (被控对象仍用 L_plant)
L_OVR = float(os.environ.get("L_OVR", "0"))

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

# ---------------------------------------------------------------- 标定参数
FR_CAL = dict(r=168.75, p=44.63, y=7.69)     # 中位（num20_fr_calib.py）
L_CAL = 0.667                                # 等效速率斜坡中位
ULIM = dict(r=1.00, p=0.85, y=0.70)
# numba 不能用 dict 全局, 展开成标量常量
FR_R = 168.75
FR_P = 44.63
FR_Y = 7.69
UL_R = 1.00
UL_P = 0.85
UL_Y = 0.70

RMIN, RMAX = 250.0, 1500.0
GAM_REAR, GAM_FRONT, BORESIGHT = 30 * D2R, 150 * D2R, 45 * D2R

# F-16 惯量 [kg m^2]（Stevens & Lewis）
IXX, IYY, IZZ = 12876.0, 75674.0, 85552.0
RHO, S_REF, B_REF = 0.7364, 27.87, 9.96
G0 = 9.80665
C_NB = 0.18          # 风标稳定性 /rad
C_LB = -0.10         # 上反效应（Dutch roll 来源）
C_NDA = 0.0069       # 副翼不利偏航
NMAX = 9.0           # 最大过载（AC7 一般 9~13G）

NMET = 16


# ================================================================ 小工具
@njit(cache=True, inline="always")
def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


@njit(cache=True, inline="always")
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True, inline="always")
def crossn(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


@njit(cache=True, inline="always")
def norm3(a):
    n = math.sqrt(dot(a, a))
    return a / n if n > 1e-12 else a


@njit(cache=True, inline="always")
def rot_by_omega(R, om, dt):
    """机体基 R(世界<-机体系) 按机体角速率 om[rad/s] 推进 dt。

    dR/dt = R [om]_x  =>  R_new = R @ expm([om]_x dt)
    约定: om[0]>0 右翼上抬(绕前轴右手) ; om[1]>0 低头(绕右轴) ; om[2]>0 机头右偏(绕机背轴)
    """
    w = math.sqrt(om[0] * om[0] + om[1] * om[1] + om[2] * om[2])
    if w * dt < 1e-14:
        return R
    th = w * dt
    ax = om / w
    kx, ky, kz = ax[0], ax[1], ax[2]
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    c = math.cos(th)
    s = math.sin(th)
    dR = np.eye(3) + K * s + (K @ K) * (1.0 - c)
    return R @ dR


@njit(cache=True, inline="always")
def col(R, j):
    return np.array([R[0, j], R[1, j], R[2, j]])


@njit(cache=True, inline="always")
def roll_angle(R):
    return math.atan2(R[2, 1], R[2, 2])


@njit(cache=True, inline="always")
def grav_rel_rates(up_, phi, vm):
    """AC7 PPDT `RotGravR` —— 非初等滚转-横向耦合。

    【建模说明 / 假设】`RotGravR` 是「重力相对」加速度：世界重力在世界系是 [0,0,g]，
    投影到有坡度的机体上就得到两个非零分量，除以 V 即为等效姿态角速度率。
        Value  = 机头朝转弯方向  -> 绕世界 z 的转弯率   psi_dot   = (g/V)*sin(phi)
        Value2 = 机头朝地面      -> 不拉杆时的掉头率   theta_dot = -(g/V)*(1-cos(phi))
        Value3 = 0               -> 重力不产生滚转                                  ✓

    量级: |g/V| = 9.80665/250 = 0.0392 rad/s = 2.25 °/s —— 与 mod 原作者早先用的
    3 °/s 同量级, 佐证这个解释。

    **不**乘过载 n：`SpeedRot` 的角速率权限本身已经等价于 20G 级杆量（
    FR_P=44.6°/s @ 250 m/s 对应 n≈21），再乘 n 会严重重复计数。
    """
    psi_dot = PSI_SIGN * (G0 / vm) * math.sin(phi)
    theta_dot = -(G0 / vm) * (1.0 - math.cos(phi))
    return psi_dot, theta_dot


@njit(cache=True, inline="always")
def world_z_rot(R, psi):
    c, s = math.cos(psi), math.sin(psi)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ R


@njit(cache=True, inline="always")
def apply_coupling(Rm, up_, vm, dt):
    """把 RotGravR 的重力相对项施加到姿态上（水平转弯 + 机头俯仰）。"""
    phi = roll_angle(Rm)
    psd, thd = grav_rel_rates(up_, phi, vm)
    Rm = world_z_rot(Rm, psd * dt)
    Rm = rot_by_omega(Rm, np.array([0.0, thd, 0.0]), dt)
    return Rm


@njit(cache=True, inline="always")
def las_ok(R, gam, th):
    if R < RMIN or R > RMAX:
        return False
    ag = abs(gam)
    if not (ag <= GAM_REAR or ag >= GAM_FRONT):
        return False
    return abs(th) <= BORESIGHT


@njit(cache=True, inline="always")
def geom(Rn, vdir, vm, ptgt, vtgt, pm):
    """返回 R, lamdot[rad/s], Vc, gam, th（th 用机头 Rn 算）"""
    rx = ptgt[0] - pm[0]
    ry = ptgt[1] - pm[1]
    rz = ptgt[2] - pm[2]
    R = math.sqrt(rx * rx + ry * ry + rz * rz)
    if R < 1e-6:
        return 1e-6, 0.0, 0.0, 0.0, 0.0
    ux, uy, uz = rx / R, ry / R, rz / R
    vx = vtgt[0] - vdir[0] * vm
    vy = vtgt[1] - vdir[1] * vm
    vz = vtgt[2] - vdir[2] * vm
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    lamdot = math.sqrt(cx * cx + cy * cy + cz * cz) / R
    Vc = -(ux * vx + uy * vy + uz * vz)
    vn = math.sqrt(vtgt[0] ** 2 + vtgt[1] ** 2 + vtgt[2] ** 2)
    gam = math.acos(clamp((ux * vtgt[0] + uy * vtgt[1] + uz * vtgt[2])
                          / max(vn, 1e-9), -1.0, 1.0))
    th = math.acos(clamp(ux * Rn[0] + uy * Rn[1] + uz * Rn[2], -1.0, 1.0))
    return R, lamdot, Vc, gam, th


# ================================================================ 目标机动
TGT_NAMES = ["直线", "匀速转 12°/s", "多轴复合正弦", "螺旋 barrel",
             "OU 有色抖振", "剪刀方波", "大离轴反向", "复合垂直机动"]
NTGT = 8


@njit(cache=True)
def tgt_vel(N, dt, mode, seed, vt, aspect0):
    """生成目标速度序列 (N,3)。目标保持真空速；机动在速度方向上施加角速率。"""
    out = np.zeros((N, 3))
    v = np.empty(3)
    v[0] = vt * math.cos(aspect0)
    v[1] = vt * math.sin(aspect0)
    v[2] = 0.0
    rng_s = np.uint32(seed * 2654435761 + 1)
    t = 0.0
    for k in range(N):
        out[k] = v
        om_y = 0.0
        om_p = 0.0
        if mode == 1:
            om_y = 12.0 * D2R
        elif mode == 2:
            om_y = 10.0 * D2R * math.sin(2 * math.pi * 0.06 * t)
            om_p = 8.0 * D2R * math.sin(2 * math.pi * 0.09 * t + 1.1)
        elif mode == 3:
            om_y = 14.0 * D2R
            om_p = 10.0 * D2R * math.sin(2 * math.pi * 0.11 * t)
        elif mode == 4:
            rng_s = np.uint32(rng_s * 1664525 + 1013904223)
            u1 = (rng_s >> 8) / 16777216.0 * 2.0 - 1.0
            rng_s = np.uint32(rng_s * 1664525 + 1013904223)
            u2 = (rng_s >> 8) / 16777216.0 * 2.0 - 1.0
            om_y = 18.0 * D2R * u1
            om_p = 9.0 * D2R * u2
        elif mode == 5:
            om_y = (20.0 * D2R) * (1.0 if int(t / 2.5) % 2 == 0 else -1.0)
        elif mode == 6:
            om_y = (25.0 * D2R) * (1.0 if int(t / 1.5) % 2 == 0 else -0.6)
            om_p = 6.0 * D2R * math.sin(2 * math.pi * 0.2 * t)
        elif mode == 7:
            om_y = 8.0 * D2R * math.sin(2 * math.pi * 0.05 * t)
            om_p = 22.0 * D2R * math.sin(2 * math.pi * 0.07 * t + 0.7)
        # 先把速度绕世界 z 转 om_y
        c, s = math.cos(om_y * dt), math.sin(om_y * dt)
        vx = v[0] * c - v[1] * s
        vy = v[0] * s + v[1] * c
        vz = v[2]
        if om_p != 0.0:
            hz = math.sqrt(vx * vx + vy * vy)
            if hz > 1e-6:
                ax = np.array([-vy / hz, vx / hz, 0.0])
                ang = om_p * dt
                cc, ss = math.cos(ang), math.sin(ang)
                tmp = np.array([vx, vy, vz])
                v = tmp * cc + crossn(ax, tmp) * ss + ax * (dot(ax, tmp) * (1 - cc))
            else:
                v = np.array([vx, vy, vz])
        else:
            v = np.array([vx, vy, vz])
        v = norm3(v) * vt
        t += dt
    return out


# ================================================================ 真实鼠标输入
INP_NAMES = ["理想（无手抖）", "基准玩家", "新手（高抖）", "职业（低抖）"]
NINP = 4


@njit(cache=True)
def gen_disturbance(N, dt, mode, seed):
    """返回 (dyaw, dpitch)，单位度：玩家指示框相对真实视线的偏移。"""
    dy = np.zeros(N)
    dp = np.zeros(N)
    if mode == 0:
        return dy, dp
    if mode == 1:
        sig, f0 = 0.55, 10.0
    elif mode == 2:
        sig, f0 = 1.40, 8.0
    else:
        sig, f0 = 0.22, 11.5
    rng_s = np.uint32(seed * 40503 + 7)
    tau_c = 1.0 / (2 * math.pi * f0)
    a_ou = dt / (tau_c + dt)
    oy = 0.0
    op = 0.0
    ty = 0.0
    tp = 0.0
    spd_y = 0.0
    spd_p = 0.0
    phase_end = 0.0
    phase = 0
    t = 0.0
    for k in range(N):
        rng_s = np.uint32(rng_s * 1664525 + 1013904223)
        n1 = (rng_s >> 8) / 16777216.0 * 2.0 - 1.0
        rng_s = np.uint32(rng_s * 1664525 + 1013904223)
        n2 = (rng_s >> 8) / 16777216.0 * 2.0 - 1.0
        oy += -a_ou * oy + math.sqrt(2.0 * a_ou) * sig * n1
        op += -a_ou * op + math.sqrt(2.0 * a_ou) * sig * n2
        if t >= phase_end:
            rng_s = np.uint32(rng_s * 1664525 + 1013904223)
            u = (rng_s >> 8) / 16777216.0
            if phase == 0:                      # 抽拽 flick
                A = min(0.6 * math.pow(max(u, 1e-4), -1.0 / 1.6), 40.0)
                dur = 0.10 + 0.045 * math.pow(A, 0.45)
                phase = 1
            elif phase == 1:                    # 周期性拉动 pull
                A = min(0.25 * math.pow(max(u, 1e-4), -1.0 / 2.0), 12.0)
                dur = 0.35 + 0.5 * u
                phase = 2
            else:                               # 微调 micro
                A = min(0.06 * math.pow(max(u, 1e-4), -1.0 / 2.5), 2.5)
                dur = 0.18 + 0.25 * u
                phase = 0
            rng_s = np.uint32(rng_s * 1664525 + 1013904223)
            sgn = 1.0 if (rng_s >> 8) & 1 else -1.0
            rng_s = np.uint32(rng_s * 1664525 + 1013904223)
            sgn2 = 1.0 if (rng_s >> 8) & 1 else -1.0
            amp_y = A * sgn
            amp_p = A * sgn2 * 0.75
            phase_end = t + dur
            kappa = 1.0 / (1.0 + 6.0 * u)       # 两-thirds 幂律: v ~ kappa^(1/3)
            spd_y = amp_y / max(dur, 1e-3) * math.pow(kappa, 1.0 / 3.0)
            spd_p = amp_p / max(dur, 1e-3) * math.pow(kappa, 1.0 / 3.0)
            rng_s = np.uint32(rng_s * 1664525 + 1013904223)
            vf = min(0.5 + 2.2 * math.pow(max((rng_s >> 8) / 16777216.0, 1e-3),
                                          -1.0 / 2.2), 6.0)
            spd_y *= vf
            spd_p *= vf
        spd_y *= 0.945
        spd_p *= 0.945
        ty = (ty + spd_y * dt) * (1.0 - 0.9 * dt)
        tp = (tp + spd_p * dt) * (1.0 - 0.9 * dt)
        dy[k] = oy + ty
        dp[k] = op + tp
        t += dt
    return dy, dp


# ================================================================ 控制律
# 消融阶梯：每个 C_k 只比 C_{k-1} 多一处改动 —— 这样每一处 patch 的收益可单独归因
CTRL_NAMES = [
    "C0 AC8 原律(基线)",
    "C1 +平滑正则(软 den)",
    "C2 +连续权重(去迟滞)",
    "C3 +标定 L=0.667",
    "C4 +L*wdot 逆模型前馈",
    "C5 A类 WT专利内环",
    "C6 B类 J6分层(进包线->ZEM)",
    "C7 B类 纯ZEM/PPN",
]
NCTRL = 8
C0, C1, C2, C3, C4, C5, C6, C7 = 0, 1, 2, 3, 4, 5, 6, 7
L_REF = 0.15        # flight_math.h 里硬编码的旧值


@njit(cache=True, inline="always")
def arrival_rate(error, max_rate, deceleration, gain, zone, delay_L):
    distance = max(0.0, abs(error) - zone)
    ds = deceleration * delay_L
    stoppable = math.sqrt(ds * ds + 2.0 * deceleration * distance) - ds
    w = min(max_rate, gain * distance)
    if stoppable < w:
        w = stoppable
    return math.copysign(w, error)


@njit(cache=True, inline="always")
def arrival_command(error, actual, max_rate, deceleration, gain, zone,
                    full_rate, limit, delay_L):
    wanted = arrival_rate(error, max_rate, deceleration, gain, zone, delay_L)
    command = wanted / full_rate + (wanted - actual) / full_rate
    if abs(error) <= zone or actual * math.copysign(1.0, error) > abs(wanted) + 3.0:
        command = (wanted - actual) / (full_rate * 0.65)
    return clamp(command, -limit, limit)


@njit(cache=True, inline="always")
def tracking_weight(angle):
    t = clamp((angle - 0.75) / 4.25, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True)
def ctrl_out(ci, aim, Rf, Rr, Ru, om, dt, st, L_est, ptgt, vtgt, pm, vm, vdir, cpl):
    f = clamp(dot(aim, Rf), -1.0, 1.0)
    right = dot(aim, Rr)
    up = dot(aim, Ru)
    ang = math.acos(f) * R2D

    # ---- 纯追踪几何解（C2 及 A 类共用，A 类把它当"先验偏置中心"） ----
    pitch_error = math.atan2(up, max(0.02, f)) * R2D
    yaw_error = math.atan2(right, max(0.02, f)) * R2D
    level_error = math.atan2(Rr[2], Ru[2]) * R2D
    near_blend = tracking_weight(ang)
    cc = 0.06 / (1.0 + (ang / 3.0) * (ang / 3.0))
    tbe_geo = clamp(math.atan2(right, up + cc) * R2D, -90.0, 90.0)
    # 【符号】mod 的 LevelBlend: roll_blend=1 = 转弯-坡度跟踪(大角度),
    # roll_blend=0 = 水平回正(小角度, 避开 atan2(right,up) 在 up≈0 的病态区)。
    # 所以 rb = tracking_weight(ang) —— 不是 1 - tracking_weight。
    rb_geo = tracking_weight(ang)
    roll_speed_geo = 70.0 + 70.0 * rb_geo
    roll_err_geo = level_error * (1.0 - rb_geo) + tbe_geo * rb_geo
    # 几何解对应的归一化杆量（先验中心）：只取比例项 + 角速率阻尼
    gr0 = clamp((roll_err_geo - (-om[0] * R2D)) / roll_speed_geo, -1.0, 1.0)
    gp0 = clamp((pitch_error - (-om[1] * R2D)) / 45.0, -1.0, 1.0)

    if ci <= C4:
        # 消融开关
        soft_den = (ci >= C1)      # 平滑正则替换硬钳位 max(0.12, up)
        cont_w = (ci >= C2)        # LevelBlend 迟滞 -> 连续权重
        cal_L = (ci >= C3)         # arrival_rate 的 0.15f -> 标定 L
        lead_ff = (ci >= C4)       # 逆模型前馈 L*wdot
        L_use = L_est if cal_L else L_REF

        if soft_den:
            den = up + cc
        else:
            den = max(0.12, up)
        tbe = clamp(math.atan2(right, den) * R2D, -90.0, 90.0)
        if cont_w:
            rb = rb_geo
        else:
            if ang <= 3.0:
                st[1] = 1.0
            elif ang >= 6.0:
                st[1] = 0.0
            wanted = 0.0 if st[1] > 0.5 else tracking_weight(ang)
            delta = clamp(wanted - st[0], -4.0 * dt, 4.0 * dt)
            st[0] = clamp(st[0] + delta, 0.0, 1.0)
            rb = st[0]
        roll_speed = 70.0 + 70.0 * rb
        roll_error = level_error * (1.0 - rb) + tbe * rb
        fg = 1.0 - near_blend
        # 阻尼项符号: 本模拟 om 为标准机体系角速率 (x 前 y 右 z 下),
        # error 用「右为正 / 上为正」; omc = [-FR_R*ur, -FR_P*up_, FR_Y*uy]
        # 故 actual 必须取负号才能构成负反馈。
        rcmd = arrival_command(roll_error, -om[0] * R2D, roll_speed, 240.0, 3.5,
                               1.0, FR_R, UL_R, L_use)
        pcmd = arrival_command(pitch_error, -om[1] * R2D, 45.0, 90.0,
                               1.8 + 0.8 * fg, 0.2, FR_P, UL_P, L_use)
        if lead_ff:
            # 被控对象 L*wdot + w = FR*u  ->  精确逆: FR*u = w + L*dw/dt
            # 消除「标定 L 变大后前馈不足 -> 到不了期望角速率」这一滞后亏损
            wr = arrival_rate(roll_error, roll_speed, 240.0, 3.5, 1.0, L_use)
            wp = arrival_rate(pitch_error, 45.0, 90.0, 1.8 + 0.8 * fg, 0.2, L_use)
            dwr = clamp((wr - st[2]) / dt, -6000.0, 6000.0)
            dwp = clamp((wp - st[3]) / dt, -6000.0, 6000.0)
            rcmd = clamp(rcmd + L_use * dwr / FR_R, -UL_R, UL_R)
            pcmd = clamp(pcmd + L_use * dwp / FR_P, -UL_P, UL_P)
            st[2] = wr
            st[3] = wp
        yrate = clamp(math.copysign(max(0.0, abs(yaw_error) - 0.2), yaw_error)
                      * (1.2 + 0.6 * fg), -7.0, 7.0)
        damp = 0.6 if ci <= C0 else 1.15
        ycmd = clamp((yrate - om[2] * R2D * damp) / 10.0, -UL_Y, UL_Y)
        return rcmd, pcmd, ycmd

    if ci == C5:
        # ============ A 类: 战争雷霆专利内环 (US8770979B2 FIG.7 / 303C) ============
        # 候选控制做 N 步前向仿真 -> 用「机头方向 · 目标方向」点积评分挑最优,
        # 配 x2/÷2 自适应步长(= 这里的粗搜+细搜), 再加平滑正则与控制能量。
        #
        # 【关键修正】候选网格的中心不能取 0: 在 rate-command 被控对象上,
        # 当候选控制恒定时 roll 不改变机头指向 -> 点积代价对 roll 维度**完全平坦**,
        # 零中心网格会退化到"不动作"(jerk≡0, 完全不跟踪)。
        # 因此把网格中心放在【纯追踪几何解】上, 网格只搜索几何解邻域内的
        # 增益修正量 —— 这既保留专利的数值内环结构, 又让 roll 通道非奇异。
        lam_j = 0.010
        lam_u = 0.002
        NSTEP, HSTEP = 16, 0.12            # 1.92 s 前向视界
        best = 1e18
        bu_r, bu_p = gr0, gp0
        for i in range(5):                  # 粗搜 step=0.4
            ur = clamp(gr0 + (-0.8 + 0.4 * i), -1.0, 1.0)
            for j in range(5):
                uc = clamp(gp0 + (-0.8 + 0.4 * j), -1.0, 1.0)
                Rm = np.empty((3, 3))
                Rm[:, 0] = Rf
                Rm[:, 1] = Rr
                Rm[:, 2] = Ru
                omw = om.copy()
                omc = np.array([-FR_R * ur, -FR_P * uc, 0.0]) * D2R
                cost = 0.0
                for kk in range(NSTEP):
                    omw = omw + (omc - omw) * (HSTEP / L_est)
                    Rm = rot_by_omega(Rm, omw, HSTEP)
                    if cpl == 1:
                        Rm = apply_coupling(Rm, uc, vm, HSTEP)
                    fw = col(Rm, 0)
                    cost += (1.0 + 1.0 * kk / NSTEP) * (1.0 - dot(fw, aim))
                cost += lam_j * 1e2 * ((ur - st[0]) ** 2 + (uc - st[1]) ** 2)
                cost += lam_u * 1e1 * (ur * ur + uc * uc)
                if cost < best:
                    best = cost
                    bu_r, bu_p = ur, uc
        # 细搜 step=0.1 (对应专利的 ÷2 自适应步长)
        c_r, c_p = bu_r, bu_p
        for i in range(3):
            ur = clamp(c_r + (-0.1 + 0.1 * i), -1.0, 1.0)
            for j in range(3):
                uc = clamp(c_p + (-0.1 + 0.1 * j), -1.0, 1.0)
                Rm = np.empty((3, 3))
                Rm[:, 0] = Rf
                Rm[:, 1] = Rr
                Rm[:, 2] = Ru
                omw = om.copy()
                omc = np.array([-FR_R * ur, -FR_P * uc, 0.0]) * D2R
                cost = 0.0
                for kk in range(NSTEP):
                    omw = omw + (omc - omw) * (HSTEP / L_est)
                    Rm = rot_by_omega(Rm, omw, HSTEP)
                    if cpl == 1:
                        Rm = apply_coupling(Rm, uc, vm, HSTEP)
                    fw = col(Rm, 0)
                    cost += (1.0 + 1.0 * kk / NSTEP) * (1.0 - dot(fw, aim))
                cost += lam_j * 1e2 * ((ur - st[0]) ** 2 + (uc - st[1]) ** 2)
                cost += lam_u * 1e1 * (ur * ur + uc * uc)
                if cost < best:
                    best = cost
                    bu_r, bu_p = ur, uc
        st[0] = bu_r
        st[1] = bu_p
        return bu_r * UL_R, bu_p * UL_P, 0.0

    # ---------------- B 类 ----------------
    Rn, lamdot, Vc, gam, th_ = geom(Rf, vdir, vm, ptgt, vtgt, pm)
    in_las = las_ok(Rn, gam, th_) and Vc > 0.0
    if ci == C7 or in_las:
        om_c = 3.0 * lamdot * R2D
        ux = (ptgt[0] - pm[0]) / max(Rn, 1e-9)
        uy = (ptgt[1] - pm[1]) / max(Rn, 1e-9)
        uz = (ptgt[2] - pm[2]) / max(Rn, 1e-9)
        lat = ux * Rr[0] + uy * Rr[1] + uz * Rr[2]
        ver = ux * Ru[0] + uy * Ru[1] + uz * Ru[2]
        pcmd = clamp(om_c * ver / FR_P, -UL_P, UL_P)
        tbe = clamp(math.atan2(lat, max(0.12, ver)) * R2D, -90.0, 90.0)
        rcmd = arrival_command(tbe, -om[0] * R2D, 140.0, 240.0, 3.5, 1.0,
                               FR_R, UL_R, L_est)
        return rcmd, pcmd, 0.0
    # 未进包线 -> 最短进包线时间 T_LAS (roll/pitch 网格 + 带一阶滞后的短视界 rollout)
    NC, H, DH = 5, 50, 0.08
    best = 1e18
    bu_r, bu_p = gr0, gp0
    for i in range(NC):
        ur = clamp(gr0 + (-0.6 + 0.3 * i), -1.0, 1.0)
        for j in range(NC):
            uc = clamp(gp0 + (-0.6 + 0.3 * j), -1.0, 1.0)
            Rm = np.empty((3, 3))
            Rm[:, 0] = Rf
            Rm[:, 1] = Rr
            Rm[:, 2] = Ru
            omw = om.copy()
            omc = np.array([-FR_R * ur, -FR_P * uc, 0.0]) * D2R
            px, py, pz = pm[0], pm[1], pm[2]
            tx, ty, tz = ptgt[0], ptgt[1], ptgt[2]
            RR = Rn
            hit = 1e6
            for kk in range(H):
                omw = omw + (omc - omw) * (DH / L_est)
                Rm = rot_by_omega(Rm, omw, DH)
                if cpl == 1:
                    Rm = apply_coupling(Rm, uc, vm, DH)
                fw = col(Rm, 0)
                px += fw[0] * vm * DH
                py += fw[1] * vm * DH
                pz += fw[2] * vm * DH
                tx += vtgt[0] * DH
                ty += vtgt[1] * DH
                tz += vtgt[2] * DH
                dx, dy_, dz = tx - px, ty - py, tz - pz
                RR = math.sqrt(dx * dx + dy_ * dy_ + dz * dz)
                if RR > 1e-6:
                    uxx, uyy, uzz = dx / RR, dy_ / RR, dz / RR
                    vnn = math.sqrt(vtgt[0] ** 2 + vtgt[1] ** 2 + vtgt[2] ** 2)
                    gm = math.acos(clamp((uxx * vtgt[0] + uyy * vtgt[1] + uzz * vtgt[2])
                                         / max(vnn, 1e-9), -1.0, 1.0))
                    tt = math.acos(clamp(uxx * fw[0] + uyy * fw[1] + uzz * fw[2],
                                         -1.0, 1.0))
                    if las_ok(RR, gm, tt):
                        hit = (kk + 1) * DH
                        break
            cost = hit if hit < 1e5 else (1e5 + 40.0 * RR)
            # 控制变化正则: 否则每 dec 帧重取 argmin 会在网格上跳格 -> 抖振
            cost += 0.6 * ((ur - st[0]) ** 2 + (uc - st[1]) ** 2)
            if cost < best:
                best = cost
                bu_r, bu_p = ur, uc
    st[0] = bu_r
    st[1] = bu_p
    return bu_r * UL_R, bu_p * UL_P, 0.0


# ================================================================ 仿真核心
@njit(cache=True)
def run_ep(ci, plant, ntgt, ninp, R0, aspect0, th0, vm, vt, dt, tmax, seed,
           use_rgrav, dec, L_plant, cpl_s):
    N = int(tmax / dt)
    pm = np.zeros(3)
    ptgt0 = np.array([R0, 0.0, 0.0])
    Rm = np.empty((3, 3))
    Rm[:, 0] = np.array([math.cos(th0), math.sin(th0), 0.0])
    Rm[:, 1] = np.array([-math.sin(th0), math.cos(th0), 0.0])
    Rm[:, 2] = np.array([0.0, 0.0, 1.0])
    om = np.zeros(3)
    vtt = tgt_vel(N, dt, ntgt, seed, vt, aspect0)
    ptgt = np.zeros((N, 3))
    for k in range(1, N):
        ptgt[k] = ptgt[k - 1] + vtt[k - 1] * dt
    ptgt = ptgt + ptgt0
    dy, dp = gen_disturbance(N, dt, ninp, seed)
    if ci <= C1:
        st = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        st = np.zeros(5)
    L_est = L_OVR if L_OVR > 0.0 else (L_REF if ci <= C2 else L_plant)

    rec_e = np.zeros(N)
    rec_f = np.zeros(N)
    rec_om = np.zeros((N, 3))
    rec_u = np.zeros((N, 3))
    rec_beta = np.zeros(N)
    rec_alpha = np.zeros(N)
    t_las = -1.0
    t_lead = -1.0
    t_enter2 = -1.0
    overshoot = 0.0
    t_settle = -1.0
    settle_start = -1.0
    beta = 0.0
    alpha0 = 4.0 * D2R
    ur = 0.0
    up_ = 0.0
    uy = 0.0
    t = 0.0
    k = 0
    for k in range(N):
        Rf = col(Rm, 0)
        Rr = col(Rm, 1)
        Ru = col(Rm, 2)
        # ---- 真实视线 ----
        dx = ptgt[k, 0] - pm[0]
        dyy = ptgt[k, 1] - pm[1]
        dz = ptgt[k, 2] - pm[2]
        Rn = math.sqrt(dx * dx + dyy * dyy + dz * dz)
        los = np.array([dx / Rn, dyy / Rn, dz / Rn])
        # ---- 玩家指示框 = 真实视线 + 手抖/亚运动 ----
        if ninp == 0:
            aim = los
        else:
            ref = np.array([0.0, 0.0, 1.0])
            if abs(los[2]) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            p1 = norm3(crossn(los, ref))
            p2 = crossn(los, p1)
            aim = norm3(los + p1 * (dy[k] * D2R) + p2 * (dp[k] * D2R))
        # ---- 速度方向（plant=1 时含 alpha/beta） ----
        if plant == 1:
            ca, sa = math.cos(alpha0 + 12.0 * D2R * clamp(up_, -1.0, 1.0)), math.sin(
                alpha0 + 12.0 * D2R * clamp(up_, -1.0, 1.0))
            cb, sb = math.cos(beta), math.sin(beta)
            vdir = norm3(Rf * (ca * cb) + Rr * sb + Ru * sa)
        else:
            vdir = Rf
        # ---- 控制（按 dec 抽帧，模拟决策周期） ----
        cpl = 1 if (plant == 1 or use_rgrav == 1) else 0
        if k % dec == 0:
            ur, up_, uy = ctrl_out(ci, aim, Rf, Rr, Ru, om, dt, st, L_est,
                                   ptgt[k], vtt[k], pm, vm, vdir, cpl)
        # ---- 被控对象 ----
        omc = np.array([-FR_R * ur, -FR_P * up_,
                        FR_Y * uy]) * D2R
        if plant == 0:
            omd = (omc - om) / L_plant
            om = om + omd * dt
            Rm = rot_by_omega(Rm, om, dt)
            if use_rgrav:
                Rm = apply_coupling(Rm, clamp(up_, -1.0, 1.0), vm, dt)
            beta = 0.0
            alpha = alpha0
        else:
            omd = (omc - om) / L_plant
            p, q, r = om[0], om[1], om[2]
            omd[0] += (IYY - IZZ) / IXX * q * r
            omd[1] += (IZZ - IXX) / IYY * r * p
            omd[2] += (IXX - IYY) / IZZ * p * q
            qbar = 0.5 * RHO * vm * vm
            # cpl_s = 气动交叉耦合的「残余系数」: 1.0 = 裸机(无 SAS) 上界,
            # 0.12 = 代表 FBW/SAS 已抑制 88% 的残余。陀螺项不乘(运动学量, 速率环消不掉)
            omd[2] += cpl_s * qbar * S_REF * B_REF * (C_NB * beta + C_NDA * ur) / IZZ
            omd[0] += cpl_s * qbar * S_REF * B_REF * (C_LB * beta) / IXX
            om = om + omd * dt
            Rm = rot_by_omega(Rm, om, dt)
            Rm = apply_coupling(Rm, clamp(up_, -1.0, 1.0), vm, dt)
            alpha = alpha0 + 12.0 * D2R * clamp(up_, -1.0, 1.0)
            phi = roll_angle(Rm)
            beta = clamp(beta + ((G0 / vm) * math.sin(phi) - om[2]
                                 + om[0] * alpha) * dt, -25.0 * D2R, 25.0 * D2R)
        Rf = col(Rm, 0)
        Rr = col(Rm, 1)
        Ru = col(Rm, 2)
        if plant == 1:
            ca, sa = math.cos(alpha), math.sin(alpha)
            cb, sb = math.cos(beta), math.sin(beta)
            vdir = norm3(Rf * (ca * cb) + Rr * sb + Ru * sa)
        else:
            vdir = Rf
        pm = pm + vdir * vm * dt
        # ---- 指标 ----
        Rn2, lamdot2, Vc2, gam2, th2 = geom(Rf, vdir, vm, ptgt[k], vtt[k], pm)
        if t_las < 0.0 and las_ok(Rn2, gam2, th2):
            t_las = t
        if t_lead < 0.0 and las_ok(Rn2, gam2, th2) and lamdot2 * R2D <= 1.0 and Vc2 > 0.0:
            t_lead = t
        e_t = math.acos(clamp(dot(los, Rf), -1.0, 1.0)) * R2D
        e_f = math.acos(clamp(dot(aim, Rf), -1.0, 1.0)) * R2D
        rec_e[k] = e_t
        rec_f[k] = e_f
        rec_om[k] = om * R2D
        rec_u[k] = np.array([ur, up_, uy])
        rec_beta[k] = beta * R2D
        rec_alpha[k] = alpha * R2D
        if t_enter2 < 0.0 and e_t <= 2.0:
            t_enter2 = t
            settle_start = t
        if t_enter2 >= 0.0:
            if e_t > overshoot:
                overshoot = e_t
            if e_t > 5.0:
                settle_start = t
            elif t_settle < 0.0 and t - settle_start > 1.0:
                t_settle = t - t_enter2
        t += dt
        if t > 22.0 and t_las > 0.0 and t_lead > 0.0 and e_t < 1.5 and k > 400:
            break
    nk = k + 1
    # ---- 手感聚合 ----
    oj = rec_om[:nk]
    dif = oj[1:] - oj[:-1]
    jerk = math.sqrt(float(np.mean(dif[:, 0] ** 2 + dif[:, 1] ** 2 + dif[:, 2] ** 2))) / dt
    tv = float(np.sum(np.abs(dif)))
    flips = 0
    for i in range(1, nk):
        for ax in range(3):
            if abs(oj[i, ax]) > 1.0 and (oj[i, ax] > 0.0) != (oj[i - 1, ax] > 0.0):
                flips += 1
    uj = rec_u[:nk]
    tv_u = float(np.sum(np.abs(uj[1:] - uj[:-1])))
    smooth = 1.0 / (1.0 + tv / 1000.0)
    es = np.sort(rec_e[:nk])
    fs = np.sort(rec_f[:nk])
    out = np.zeros(NMET)
    out[0] = t_las
    out[1] = t_lead
    out[2] = jerk
    out[3] = tv
    out[4] = flips
    out[5] = tv_u
    out[6] = overshoot
    out[7] = t_settle
    out[8] = smooth
    out[9] = t
    out[10] = float(np.mean(rec_e[:nk]))
    out[11] = float(es[min(nk - 1, int(0.95 * nk))])
    out[12] = float(np.max(np.abs(rec_beta[:nk])))
    out[13] = float(np.mean(rec_f[:nk]))
    out[14] = float(fs[min(nk - 1, int(0.95 * nk))])
    out[15] = float(nk)
    return out


@njit(cache=True)
def run_ep_dbg(ci, plant, ntgt, ninp, R0, aspect0, th0, vm, vt, dt, tmax, seed,
               use_rgrav, dec, L_plant, cpl_s):
    """调试版：返回逐步轨迹 (e_t, e_f, om(3), u(3), R, lamdot, gam, th)"""
    N = int(tmax / dt)
    pm = np.zeros(3)
    ptgt0 = np.array([R0, 0.0, 0.0])
    Rm = np.empty((3, 3))
    Rm[:, 0] = np.array([math.cos(th0), math.sin(th0), 0.0])
    Rm[:, 1] = np.array([-math.sin(th0), math.cos(th0), 0.0])
    Rm[:, 2] = np.array([0.0, 0.0, 1.0])
    om = np.zeros(3)
    vtt = tgt_vel(N, dt, ntgt, seed, vt, aspect0)
    ptgt = np.zeros((N, 3))
    for k in range(1, N):
        ptgt[k] = ptgt[k - 1] + vtt[k - 1] * dt
    ptgt = ptgt + ptgt0
    dy, dp = gen_disturbance(N, dt, ninp, seed)
    if ci <= C1:
        st = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        st = np.zeros(5)
    L_est = L_OVR if L_OVR > 0.0 else (L_REF if ci <= C2 else L_plant)
    tr = np.zeros((N, 12))
    beta = 0.0
    alpha0 = 4.0 * D2R
    ur = 0.0
    up_ = 0.0
    uy = 0.0
    for k in range(N):
        Rf = col(Rm, 0); Rr = col(Rm, 1); Ru = col(Rm, 2)
        dx = ptgt[k, 0] - pm[0]; dyy = ptgt[k, 1] - pm[1]; dz = ptgt[k, 2] - pm[2]
        Rn = math.sqrt(dx * dx + dyy * dyy + dz * dz)
        los = np.array([dx / Rn, dyy / Rn, dz / Rn])
        if ninp == 0:
            aim = los
        else:
            ref = np.array([0.0, 0.0, 1.0])
            if abs(los[2]) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            p1 = norm3(crossn(los, ref)); p2 = crossn(los, p1)
            aim = norm3(los + p1 * (dy[k] * D2R) + p2 * (dp[k] * D2R))
        if plant == 1:
            aa = alpha0 + 12.0 * D2R * clamp(up_, -1.0, 1.0)
            ca, sa = math.cos(aa), math.sin(aa)
            cb, sb = math.cos(beta), math.sin(beta)
            vdir = norm3(Rf * (ca * cb) + Rr * sb + Ru * sa)
        else:
            vdir = Rf
        if k % dec == 0:
            ur, up_, uy = ctrl_out(ci, aim, Rf, Rr, Ru, om, dt, st, L_est,
                                   ptgt[k], vtt[k], pm, vm, vdir,
                                   1 if (plant == 1 or use_rgrav == 1) else 0)
        omc = np.array([-FR_R * ur, -FR_P * up_, FR_Y * uy]) * D2R
        if plant == 0:
            om = om + (omc - om) / L_plant * dt
            Rm = rot_by_omega(Rm, om, dt)
            if use_rgrav:
                Rm = apply_coupling(Rm, clamp(up_, -1.0, 1.0), vm, dt)
        else:
            omd = (omc - om) / L_plant
            p_, q_, r_ = om[0], om[1], om[2]
            omd[0] += (IYY - IZZ) / IXX * q_ * r_
            omd[1] += (IZZ - IXX) / IYY * r_ * p_
            omd[2] += (IXX - IYY) / IZZ * p_ * q_
            qbar = 0.5 * RHO * vm * vm
            omd[2] += cpl_s * qbar * S_REF * B_REF * (C_NB * beta + C_NDA * ur) / IZZ
            omd[0] += cpl_s * qbar * S_REF * B_REF * C_LB * beta / IXX
            om = om + omd * dt
            Rm = rot_by_omega(Rm, om, dt)
            Rm = apply_coupling(Rm, clamp(up_, -1.0, 1.0), vm, dt)
            phi = roll_angle(Rm)
            beta = clamp(beta + ((G0 / vm) * math.sin(phi) - om[2] + om[0] * alpha0) * dt,
                         -25.0 * D2R, 25.0 * D2R)
        Rf = col(Rm, 0); Rr = col(Rm, 1); Ru = col(Rm, 2)
        pm = pm + (Rf if plant == 0 else vdir) * vm * dt
        Rn2, lamdot2, Vc2, gam2, th2 = geom(Rf, vdir, vm, ptgt[k], vtt[k], pm)
        tr[k, 0] = math.acos(clamp(dot(los, Rf), -1.0, 1.0)) * R2D
        tr[k, 1] = math.acos(clamp(dot(aim, Rf), -1.0, 1.0)) * R2D
        tr[k, 2:] = np.array([om[0], om[1], om[2], ur, up_, uy,
                              Rn2, lamdot2 * R2D, gam2 * R2D, th2 * R2D])
    return tr
