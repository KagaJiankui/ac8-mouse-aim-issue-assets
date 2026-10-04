# Patch：简化版战争雷霆飞控（WT 专利内环）

> 目标仓库：`FletcherMiya/AC8-Mouse-Aim`  
> 本文只列代码段，不改动任何源文件。  
> 收益数字来自 `analysis_scripts/num21_two_class.py` 的 1024 场景对照（8 控制器 × 8 目标机动 × 4 鼠标档 × 4 被控对象），指标定义见 `analysis_reports_zem-wtpatent.md`。

---

## 1. 它是什么

本方案把 Gaijin 的战争雷霆鼠标飞控专利内环（US8770979B2 / US9858830B2，说明书 FIG.7 的 303C 实现）简化为一个可直接接入的控制器：

1. 对候选 (roll, pitch) 归一化杆量做 N 步前向仿真。
2. 用「机头方向 · 目标方向」的点积评分挑最优（专利的点积评分）。
3. 粗搜 + 细搜（对应专利的 ×2 / ÷2 自适应步长）。
4. 代价里加平滑正则 `λ_j·|Δu|²` 与控制能量 `λ_u·|u|²`。

它不主动建立前置角。目标就是玩家看到的鼠标指示框，所以几何上等价于纯追踪。代价是机头始终追鼠标，不会主动离开去预判拦截点。

## 2. 为什么值得考虑

在 P3 决策档（完整刚体 + SAS 残余 12%）上，它是唯一在所有指标上都不输基线、且在多数指标上赢的方案：

| 控制器          | 跟随误差       | 建立前置    | jerk      | 成功率     | 进包线时间      |
| ------------ | ---------- | ------- | --------- | ------- | ---------- |
| C0 基线（原律）    | 21.57°     | 2.48 s  | 53.32     | 53%     | 3.62 s     |
| **C5 WT 飞控** | **17.84°** | 10.26 s | **31.96** | **69%** | **2.96 s** |

- 跟随误差 −17%，是全场最好。
- jerk −40%（手感侧改善）。
- 成功率 +16 个百分点，并列最高。
- 代价：建立前置慢到 10.26 s（这是「先跟手、后瞄准」的设计取向）。

它没有手调增益。roll/pitch 杆量由前向仿真 + 点积评分选出。

## 3. 需要的辅助函数

上游 `flight_math.h` 只有欧拉角 `basis()` 与 `rotate()`。本方案要新增三个小函数（全部可放进 `flight` 命名空间）：

```cpp
// 按机体角速率 om [rad/s] 推进基底 dt。Rodrigues 旋转。
inline Basis rot_by_omega(Basis b, V om, float dt) {
    float w = std::sqrt(dot(om, om));
    if (w * dt < 1e-9f) return b;
    V ax = om * (1.0f / w);
    float th = w * dt;
    return {rotate(b.f, ax, th), rotate(b.r, ax, th), rotate(b.u, ax, th)};
}

// 绕世界 Z 轴转 psi。
inline Basis world_z_rot(Basis b, float psi) {
    V z{0, 0, 1};
    return {rotate(b.f, z, psi), rotate(b.r, z, psi), rotate(b.u, z, psi)};
}

// AC7 PPDT 的 RotGravR 重力投影（滚转 -> 转弯 / 掉头的运动学耦合）。
// PSI_SIGN 由 ±1 对照实验定为 -1（坡度产生朝目标的转弯）。
inline void grav_rel(float phi, float V, float& psi_dot, float& theta_dot) {
    const float PSI_SIGN = -1.0f;
    psi_dot   = PSI_SIGN * (9.80665f / V) * std::sin(phi);
    theta_dot = -(9.80665f / V) * (1.0f - std::cos(phi));
}

inline Basis apply_coupling(Basis b, float V, float dt) {
    float phi = std::atan2(b.r.z, b.u.z);   // 滚转角
    float psd, thd;
    grav_rel(phi, V, psd, thd);
    b = world_z_rot(b, psd * dt);
    b = rot_by_omega(b, {0, thd, 0}, dt);
    return b;
}
```

## 4. 内环参考实现

```cpp
// ---- 简化版战争雷霆飞控：WT 专利内环 (US8770979B2 FIG.7 / 303C) ----
struct WtInnerLoop {
    static constexpr int   NSTEP = 16;     // 前向步数，1.92 s @ HSTEP
    static constexpr float HSTEP = 0.12f;  // 前向步长 [s]
    static constexpr int   GRID  = 5;      // 粗搜 5×5，步长 0.4
    static constexpr int   FINE  = 3;      // 细搜 3×3，步长 0.1（专利 ÷2 步长）
    static constexpr float LAM_J = 0.010f; // 平滑正则系数
    static constexpr float LAM_U = 0.002f; // 控制能量系数
    float prev_r = 0.0f, prev_p = 0.0f;

    float cost(const flight::Basis& b, const flight::V& om, float L,
               float ur, float up, const flight::V& aim, float V) const {
        flight::Basis R = b;
        flight::V w = om;
        const float fr_r = 168.75f, fr_p = 44.63f;   // 标定 FR [°/s]，需重标定
        flight::V wc = {-fr_r * ur * flight::rad, -fr_p * up * flight::rad, 0.0f};
        float c = 0.0f;
        for (int k = 0; k < NSTEP; ++k) {
            w = w + (wc - w) * (HSTEP / L);
            R = rot_by_omega(R, w, HSTEP);
            R = apply_coupling(R, V, HSTEP);
            const float fw = flight::dot(R.f, aim);
            c += (1.0f + 1.0f * k / NSTEP) * (1.0f - fw);  // 远期步权重更高
        }
        c += LAM_J * 1e2f * ((ur - prev_r) * (ur - prev_r)
                           + (up - prev_p) * (up - prev_p));
        c += LAM_U * 1e1f * (ur * ur + up * up);
        return c;
    }

    void step(const flight::Basis& b, const flight::V& om, float L,
              const flight::V& aim, float V, float gr0, float gp0,
              float& ur, float& up) {
        float best = 1e18f, br = gr0, bp = gp0;
        for (int i = 0; i < GRID; ++i)
        for (int j = 0; j < GRID; ++j) {
            float a = std::clamp(gr0 + (-0.8f + 0.4f * i), -1.0f, 1.0f);
            float c = std::clamp(gp0 + (-0.8f + 0.4f * j), -1.0f, 1.0f);
            float v = cost(b, om, L, a, c, aim, V);
            if (v < best) { best = v; br = a; bp = c; }
        }
        for (int i = 0; i < FINE; ++i)
        for (int j = 0; j < FINE; ++j) {
            float a = std::clamp(br + (-0.1f + 0.1f * i), -1.0f, 1.0f);
            float c = std::clamp(bp + (-0.1f + 0.1f * j), -1.0f, 1.0f);
            float v = cost(b, om, L, a, c, aim, V);
            if (v < best) { best = v; br = a; bp = c; }
        }
        prev_r = br; prev_p = bp; ur = br; up = bp;
    }
};
```

## 5. 接入点

在 `mouse_aim.cpp` 的 `update_commands()` 尾部替换俯仰/滚转输出：

```diff
-    command_pitch.store(manual?0:pcmd*config.pitch_sign);
-    command_roll.store(manual?0:rcmd*config.roll_sign);
+    if (config.inner_loop == 1) {          // 0 = 现有几何律; 1 = WT 飞控
+        float ur, up;
+        wt.step(b, omega, config.delay_l_roll, aim,
+                /*V=*/config.aircraft_speed, gr0, gp0, ur, up);
+        command_roll.store(manual?0:ur*config.roll_sign);
+        command_pitch.store(manual?0:up*config.pitch_sign);
+    } else {
+        command_pitch.store(manual?0:pcmd*config.pitch_sign);
+        command_roll.store(manual?0:rcmd*config.roll_sign);
+    }
```

其中 `gr0` / `gp0` 是现有 `turn_bank_error` / `pitch_error` 的归一化杆量（纯追踪几何解，作搜索中心）：

```cpp
const float gr0 = std::clamp((roll_error  - filtered_roll_rate) / roll_speed,  -1.0f, 1.0f);
const float gp0 = std::clamp((pitch_error - filtered_pitch_rate) / 45.0f,       -1.0f, 1.0f);
```

## 6. 两个关键点

1. **候选网格中心不应取 0。** 在 rate-command 被控对象上，候选控制恒定时 roll 不改变机头指向（绕前轴旋转不改变前轴方向），点积代价对 roll 维度完全平坦。零中心网格会退化成「不动作」。所以网格中心应放在纯追踪几何解上，网格只搜邻域修正量。
2. **前向仿真应带滚转 -> 转弯耦合**（`apply_coupling`），否则 roll 通道同样退化。

## 7. 性能与优化（访存 + 向量化）

### 7.1 热点在哪

每帧 25 + 9 = 34 次前向仿真 × 16 步。先讲清热点的性质：

- **访存不是瓶颈**。一个候选的前向状态是 1 个 `Basis`（36 B）+ 1 个角速率 `V`（12 B），34 个候选合计约 1.6 KB，全程在 L1。`_mm_prefetch` 在这里没有收益。
- **瓶颈是超越函数**。原版每一步（`rot_by_omega` + `apply_coupling`）累计约 40 个慢指令：`sqrt`/`div` 各 11 次、`sin`/`cos` 各 10 次、`atan2` 1 次，全部走 30–70 周期的硬件/软件实现。
- **16 步递推是严格串行的**（每步 `R`、`w` 依赖上一步），只有候选之间相互独立。

所以优化顺序是：先砍超越函数（标量层，收益最大），再谈向量化（收益小且受兼容约束）。

### 7.2 标量优化：砍掉约 80% 的慢指令

四个改动都是纯代数等价变换，不改变控制律语义：

1. **去掉 `rotate()` 里的 `unit()` 归一化**。`rot_by_omega` 旋转的是正交基底（`f/r/u` 两两正交且单位长），Rodrigues 旋转矩阵正交，结果天然保持单位长度。原版每步 6–9 次 `unit()`（每次 1 个 `sqrt` + 1 个 `div`）是白做的。
2. **`1-cos(θ)` 用半角**。`1 − cos θ = 2·sin²(θ/2)`。前向步长 0.12 s、角速率通常 < 3 rad/s，`θ = |ω|·dt < 0.36 rad` 是小角度，直接算 `1−cos(θ)` 会减法抵消（约 40 ulp 误差，见 Roblox 旋转向量文章的实测）。半角形式既稳又只需一次 `sin`+`cos`。
3. **`atan2` + `sin` + `cos` 三连 → 一次 `sqrt` + 有理化**。`phi = atan2(b.r.z, b.u.z)` 后的 `sin(phi)`/`cos(phi)` 直接用分量比：`sin(phi) = b.r.z/d`，`1−cos(phi) = b.r.z²/(d·(d+b.u.z))`，`d = |(b.r.z, b.u.z)|`。三个超越函数换成一个 `sqrt`。
4. **绕 Z / 绕 Y 的旋转写成 2D 特化**。`world_z_rot` 和 `rot_y` 只动两个分量，一次 `sin`/`cos` + 纯乘加，不走通用 Rodrigues 的 `cross`+`dot`。

```cpp
#include <immintrin.h>   // _mm_rsqrt_ss（SSE2）

// rsqrt：SSE2 硬件近似 + 一次 Newton 迭代（UE 的 InvSqrt 同款）。
// 相对误差 ~1e-6。候选评分是相对比较，这个精度足够。
inline float inv_sqrt(float x) {
    __m128 v = _mm_set_ss(x);
    __m128 r = _mm_rsqrt_ss(v);
    r = _mm_mul_ss(r, _mm_sub_ss(_mm_set_ss(1.5f),
        _mm_mul_ss(_mm_mul_ss(_mm_set_ss(0.5f), v), _mm_mul_ss(r, r))));
    return _mm_cvtss_f32(r);
}

// Rodrigues 一步：v' = v + (ax×v)·s + (ax×(ax×v))·omc
// s = sin(th)，omc = 1 - cos(th) = 2·sin²(th/2)，无减法抵消。
inline flight::V rodrigues(flight::V v, flight::V ax, float s, float omc) {
    flight::V k  = flight::cross(ax, v);
    flight::V k2 = flight::cross(ax, k);
    return { v.x + k.x*s + k2.x*omc, v.y + k.y*s + k2.y*omc, v.z + k.z*s + k2.z*omc };
}

inline flight::Basis rot_by_omega_fast(flight::Basis b, flight::V om, float dt) {
    const float w2 = flight::dot(om, om);
    if (w2 * dt * dt < 1e-18f) return b;   // |om|·dt ≈ 0
    const float rw   = inv_sqrt(w2);       // 1/|om|
    const float th   = w2 * rw * dt;       // |om|·dt（复用 rw，不再单独 sqrt）
    const float half = 0.5f * th;
    const float sh = std::sin(half), ch = std::cos(half);
    const float s   = 2.0f * sh * ch;      // sin(th)
    const float omc = 2.0f * sh * sh;      // 1 - cos(th)
    flight::V ax = { om.x*rw, om.y*rw, om.z*rw };
    return { rodrigues(b.f, ax, s, omc), rodrigues(b.r, ax, s, omc), rodrigues(b.u, ax, s, omc) };
}

inline flight::Basis world_z_rot_fast(flight::Basis b, float psi) {
    const float s = std::sin(psi), c = std::cos(psi);
    return { { c*b.f.x - s*b.f.y, s*b.f.x + c*b.f.y, b.f.z },
             { c*b.r.x - s*b.r.y, s*b.r.x + c*b.r.y, b.r.z },
             { c*b.u.x - s*b.u.y, s*b.u.x + c*b.u.y, b.u.z } };
}

inline flight::Basis rot_y_fast(flight::Basis b, float angle) {
    const float s = std::sin(angle), c = std::cos(angle);
    return { { c*b.f.x + s*b.f.z, b.f.y, c*b.f.z - s*b.f.x },
             { c*b.r.x + s*b.r.z, b.r.y, c*b.r.z - s*b.r.x },
             { c*b.u.x + s*b.u.z, b.u.y, c*b.u.z - s*b.u.x } };
}

inline flight::Basis apply_coupling_fast(flight::Basis b, float V, float dt) {
    const float rz = b.r.z, uz = b.u.z;
    const float d2 = rz*rz + uz*uz;
    if (d2 < 1e-12f) return b;             // 机头竖直，投影退化，跳过耦合
    const float d = std::sqrt(d2);
    const float sin_phi = rz / d;
    const float omc_phi = (rz*rz) / (d * (d + uz));   // 1 - cos(phi)，有理化，无抵消
    const float gV = 9.80665f / V;
    const float PSI_SIGN = -1.0f;
    return rot_y_fast(world_z_rot_fast(b, PSI_SIGN * gV * sin_phi * dt),
                      -gV * omc_phi * dt);
}
```

`cost()` 把循环不变量提到外面：

```cpp
float cost_fast(const flight::Basis& b, const flight::V& om, float L,
                float ur, float up, const flight::V& aim, float V) const {
    flight::Basis R = b;
    flight::V w = om;
    const float fr_r = 168.75f, fr_p = 44.63f;
    flight::V wc = { -fr_r*ur*flight::rad, -fr_p*up*flight::rad, 0.0f };
    const float alpha  = HSTEP / L;        // 一阶滞后系数，提到循环外
    const float one_ma = 1.0f - alpha;
    const float invN   = 1.0f / NSTEP;
    float c = 0.0f;
    for (int k = 0; k < NSTEP; ++k) {
        w = { w.x*one_ma + wc.x*alpha, w.y*one_ma + wc.y*alpha, w.z*one_ma + wc.z*alpha };
        R = rot_by_omega_fast(R, w, HSTEP);
        R = apply_coupling_fast(R, V, HSTEP);
        const float fw = flight::dot(R.f, aim);
        c += (1.0f + (float)k * invN) * (1.0f - fw);
    }
    c += LAM_J * 1e2f * ((ur - prev_r)*(ur - prev_r) + (up - prev_p)*(up - prev_p));
    c += LAM_U * 1e1f * (ur*ur + up*up);
    return c;
}
```

预期收益：慢指令从每步约 40 个（`sqrt`/`div` 各 11、`sin`/`cos` 各 10、`atan2` 1）降到约 8 个（`rsqrt` 1、`sqrt` 1、`sin`/`cos` 各 3）。这是纯标量层约 5× 的慢指令削减，且半角/有理化/rsqrt 的误差都在候选评分的容差内，控制律数值不变。

### 7.3 编译器侧：让 MSVC 榨自动向量化

不改代码结构的前提下，用编译开关 + pragma：

| 手段 | 作用 |
|---|---|
| `/O2`（含 `/Og /Oi /Ot /Oy /Ob2 /GF /Gy`） | 基础优化 + 内联 intrinsic |
| `/arch:AVX2` | 启用 AVX2 + FMA（目标机必支持 SSE4.2） |
| `/fp:fast` | 允许 FMA、重结合、近似超越函数——候选评分安全 |
| `/GL` | 链接期跨 TU 内联 `dot`/`cross`/`rodrigues` |
| `/Qvec-report:2` | 打印向量化诊断，确认哪些循环真向量化了 |
| `__forceinline` | 强制内联 `dot`/`cross`/`rodrigues`（候选循环最内层） |
| `__restrict` | 声明候选数组指针不别名，解除自动向量化的主要障碍 |
| `#pragma loop(ivdep)` | 提示编译器忽略候选循环的向量依赖 |
| `__declspec(align(16))` | 让 `V`/`Basis` 16 字节对齐，load/store 合并 |

候选循环的写法（`step()` 把 25 个粗搜候选放进数组，提示向量化）：

```cpp
__declspec(align(16)) float costs[GRID * GRID];
#pragma loop(ivdep)
for (int i = 0; i < GRID * GRID; ++i) {
    costs[i] = cost_fast(b, om, L, cand[i].ur, cand[i].up, aim, V);
}
````

两点要诚实说明：

1. **16 步前向递推本身向量化不了**（串行依赖），`/Qvec-report:2` 会报告内层 `for (int k...)` 未向量化——这是预期，不是问题。
2. **候选间并行要做成 SoA（结构体数组）+ 显式 SIMD 才有效**，但那需要手写 AVX2 intrinsic 和向量版 `sin/cos`（SVML），超出本 patch 的兼容约束，不展开。真要走到那一步，先用 `/Qvec-report` 确认候选循环已是热点再说。

### 7.4 限频兜底

如果接入后仍想压 CPU，保留 `inner_loop_hz` 限频（例如 60 Hz），中间帧沿用上次的 `(ur, up)`。优化后的每帧成本足够低，多数情况不需要。

## 8. 风险与重标定

| 项          | 说明                                | 缓解                                     |
| ---------- | --------------------------------- | -------------------------------------- |
| 对鼠标抖动敏感    | 实测「新手/理想」跟随比值 1.13，为最差一档          | 对 `aim` 低通后再进 rollout                  |
| 空速未知       | 上游 mod 不读空速，`V` 需给名义值             | 用 `config.aircraft_speed`（标定 AC8 真实空速） |
| 参数用 AC7 标定 | FR=168.75/44.63、L=0.667 来自 AC7 实测 | **接入前应按 AC8 重标定**                      |

符号约定：`omega` 的机体系分量与 `filtered_*_rate` 的符号关系沿用上游 `update_commands()` 现有写法。若实机出现坡度反号，翻转 `grav_rel` 的 `PSI_SIGN`。
