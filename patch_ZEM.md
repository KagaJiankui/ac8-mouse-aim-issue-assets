# Patch：纯 ZEM（比例导引 N=3）

> 目标仓库：`FletcherMiya/AC8-Mouse-Aim`
> 本文只列代码段，不改动任何源文件。
> 收益数字来自 `analysis_scripts/num21_two_class.py` 的 1024 场景对照，指标定义见 `analysis_reports_zem-wtpatent.md`。

---

## 1. 它是什么

ZEM 是零控脱靶量（Zero Effort Miss）。比例导引（Proportional Navigation, PN）是它的变分法最优解：要零脱靶且最小化 `∫a²dt`，横向加速度是

```text
a* = 3 · Vc · λ̇          （N=3；含目标机动时 + 1.5 · a_t）
```

`λ̇` 是视线（Line-of-Sight, LOS）角速度，`Vc` 是接近速度。角速率指令形式就是 `ω_cmd = 3 · λ̇`。本方案把这条指令投影到俯仰（垂直分量）与横向（滚转分量）上。

它和 WT 飞控的差别：它主动建前置角（机头会离开鼠标去预判拦截点），所以击杀更快、更稳，但「跟手」会变差。实测数据见 §6。

## 2. 为什么值得考虑

在 P3 决策档上，它是「手感与击杀」最均衡的解：

| 控制器 | 进包线时间 | 建立前置 | 跟随误差 | 超调 | jerk | 成功率 |
|---|---|---|---|---|---|---|
| C0 基线（原律） | 3.62 s | 2.48 s | 21.57° | 39.20° | 53.32 | 53% |
| **C7 纯 ZEM** | **2.93 s** | 5.52 s | 20.21° | **38.18°** | **23.95** | **62%** |

- jerk 23.95，是全场手感侧最好。
- 超调 38.18°，是全场最低。
- 进包线时间 2.93 s，比基线快 19%。
- 成功率 62%，比基线高 9 个百分点。

## 3. 前提：它需要视线角速度 λ̇

上游 mod 只读姿态、相机与鼠标指向，不读目标位置与速度。而 `λ̇` 需要目标与自身的相对位置、速度。这是本方案与 WT 飞控的关键差别：WT 飞控只用 `aim` 方向，可直接接入；本方案要先解决 `λ̇` 的来源。

有两种补法，二选一：

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 从 aim 历史估 λ̇ | 对 `target_pitch/yaw` 序列差分求视线角速度 | 鼠标输入噪声大，需强低通；仿真用的是干净 LOS，实机收益会打折 |
| B. 读目标状态 | 从游戏内存读目标 3D 位置与速度，再算 λ̇ | 超出当前 mod 的输入/相机定位，改动范围大 |

仿真的 `λ̇` 来自干净几何，所以本文数字是上界。方案 A 的估计误差需要单独研究。

## 4. 参考实现

### 4.1 几何量（供方案 B，对应仿真里的 `geom()`）

```cpp
// 输入：自身位置 pm、速度方向 vdir、速度 vm；目标位置 pt、速度 vt。
// 输出：R, lamdot, Vc, gam（目标航向角）, th（偏离机头角）。
struct Geom {
    float R, lamdot, Vc, gam, th;
};

inline Geom geom(flight::V pm, flight::V vdir, float vm,
                 flight::V pt, flight::V vt, flight::V fwd) {
    flight::V d = pt - pm;
    float R = std::sqrt(flight::dot(d, d));
    if (R < 1e-6f) return {1e-6f, 0, 0, 0, 0};
    flight::V u = d * (1.0f / R);                       // LOS 单位向量
    flight::V vrel = vt - vdir * vm;                    // 相对速度
    flight::V c = flight::cross(u, vrel);               // LOS 旋转
    float lamdot = std::sqrt(flight::dot(c, c)) / R;    // [rad/s]
    float Vc = -flight::dot(u, vrel);                   // 接近速度
    float vn = std::sqrt(flight::dot(vt, vt));
    float gam = std::acos(std::clamp(flight::dot(u, vt) / std::max(vn, 1e-9f),
                                     -1.0f, 1.0f));
    float th  = std::acos(std::clamp(flight::dot(u, fwd), -1.0f, 1.0f));
    return {R, lamdot, Vc, gam, th};
}
```

### 4.2 导引律（C7）

```cpp
// 角速率指令：N=3 的比例导引，投影到俯仰（ver）与横向（lat）。
const float FR_R = 168.75f, FR_P = 44.63f, UL_P = 0.85f, UL_R = 1.0f;  // 重标定
const float om_c = 3.0f * lamdot / flight::rad;         // [°/s]
const float ver = flight::dot(los, b.u);                // 视线上分量
const float lat = flight::dot(los, b.r);                // 视线右分量
const float pcmd = std::clamp(om_c * ver / FR_P, -UL_P, UL_P);
const float tbe  = std::clamp(std::atan2(lat, std::max(0.12f, ver)) / flight::rad,
                              -90.0f, 90.0f);
const float rcmd = flight::arrival_command(tbe, filtered_roll_rate,
                                           140.0f, 240.0f, 3.5f, 1.0f,
                                           FR_R, UL_R);
command_pitch.store(manual ? 0 : pcmd * config.pitch_sign);
command_roll.store(manual ? 0 : rcmd * config.roll_sign);
command_yaw.store(0);
```

`los` 是目标视线单位向量（方案 A 里就是 `aim`，方案 B 里是 `u`）。

## 5. 与 WT 飞控的组合建议

两者可以做成 config 开关，按目标距离切换：

- 远距 / 目标持续机动：用 WT 飞控（跟手，稳定）。
- 进入发射包线后：切到 ZEM（快速建立拦截点）。

这正是 §6 表里 C6（NemoFC）做的事，见 `patch_NemoFC.md`。

## 6. 收益（P3 档，8 目标 × 4 鼠标档中位）

| 指标 | C0 基线 | C7 纯 ZEM | 变化 |
|---|---|---|---|
| 进包线时间 t_las [s] | 3.62 | 2.93 | −19% |
| 建立前置 t_lead [s] | 2.48 | 5.52 | +123% |
| 跟随误差 follow [°] | 21.57 | 20.21 | −6% |
| 超调 overshoot [°] | 39.20 | 38.18 | −3% |
| jerk [°/s²] | 53.32 | 23.95 | −55% |
| tv（角速率总变差） | 1794 | 873 | −51% |
| 成功率 | 53% | 62% | +9 pp |

弱点与 WT 飞控一致：在「匀速转」「螺旋 barrel」这类持续大角速度目标上跟随误差会到 25–31°（表 3），且没有任何控制器能在螺旋 barrel 上建立前置。

## 7. 重标定清单

| 量 | 现值（AC7 标定） | 说明 |
|---|---|---|
| FR_R / FR_P / FR_Y | 168.75 / 44.63 / 7.69 °/s | 应按 AC8 重测 T90/T360/roll/yaw |
| L（等效速率斜坡） | 0.667 s | 进入 `arrival_rate` 制动包络 |
| 空速 V | 250 m/s（名义） | 影响 `geom` 与耦合，应按 AC8 真实空速 |
| N（导引增益） | 3.0 | 变分法最优；`|θ_lead| < 70.53°` 稳定 |

符号约定：`om_c` 的俯仰/滚转方向沿用上游 `filtered_pitch_rate = -dot(omega, b.r)`、`filtered_roll_rate = -dot(omega, b.f)` 的约定。若实机俯仰反号，翻转 `om_c * ver` 的符号。

## 8. 访存与向量化优化

### 8.1 结论先说

ZEM 每帧只跑一次，计算量极小（`geom` 3 个 `sqrt` + 2 个 `acos` + 1 个 `div`，导引律 1 个 `atan2` + 1 个 `sqrt`）。它不是热点，不值得上 SIMD。优化只做两件低成本的事：合并 `sqrt`/`div`；给方案 A 的 `λ̇` 估计用环形缓冲。

### 8.2 `geom` 的 `sqrt`/`div` 合并

`geom` 里出现三次 `sqrt(dot(...))` 和一次 `1/R`，全部合并成 rsqrt：

```cpp
#include <immintrin.h>

inline float inv_sqrt(float x) {
    __m128 v = _mm_set_ss(x);
    __m128 r = _mm_rsqrt_ss(v);
    r = _mm_mul_ss(r, _mm_sub_ss(_mm_set_ss(1.5f),
        _mm_mul_ss(_mm_mul_ss(_mm_set_ss(0.5f), v), _mm_mul_ss(r, r))));
    return _mm_cvtss_f32(r);
}

inline Geom geom_fast(flight::V pm, flight::V vdir, float vm,
                      flight::V pt, flight::V vt, flight::V fwd) {
    flight::V d = pt - pm;
    const float d2 = flight::dot(d, d);
    if (d2 < 1e-12f) return { 1e-6f, 0, 0, 0, 0 };
    const float invR = inv_sqrt(d2);                 // 1/R
    flight::V u = { d.x*invR, d.y*invR, d.z*invR };  // LOS 单位向量
    flight::V vrel = { vt.x - vdir.x*vm, vt.y - vdir.y*vm, vt.z - vdir.z*vm };
    flight::V c = flight::cross(u, vrel);
    const float c2 = flight::dot(c, c);
    const float lamdot = c2 * invR * inv_sqrt(c2);   // |c|/R，复用 rsqrt
    const float Vc = -flight::dot(u, vrel);
    const float vn2 = flight::dot(vt, vt);
    const float vn = vn2 * inv_sqrt(vn2);
    const float gam = std::acos(std::clamp(flight::dot(u, vt) / std::max(vn, 1e-9f), -1.0f, 1.0f));
    const float th  = std::acos(std::clamp(flight::dot(u, fwd), -1.0f, 1.0f));
    return { d2 * invR, lamdot, Vc, gam, th };       // R = d2·invR
}
```

`dot`/`cross` 的 3 元运算交给 MSVC `/arch:AVX2 /fp:fast` 自动向量化/FMA，无需手写 intrinsic。

### 8.3 方案 A 的 `λ̇` 估计：环形缓冲 + 低通

方案 A（从 `aim` 历史估 `λ̇`）的瓶颈不在计算，在鼠标输入噪声。要强低通，且别每帧重新扫历史。用固定大小的环形缓冲存 `target_pitch/yaw`，差分只取最新两帧，低通用一阶 IIR：

```cpp
// 环形缓冲只存最近 N 帧 aim，增量更新，不重扫。
struct AimHistory {
    static constexpr int N = 32;
    float pitch[N]{}, yaw[N]{};
    int head = 0;
    float lam_pitch = 0.0f, lam_yaw = 0.0f;   // 一阶低通后的视线角速度

    void push(float p, float y, float dt, float tau) {
        head = (head + 1) % N;
        pitch[head] = p;
        yaw[head] = y;
        int prev = (head + N - 1) % N;
        float a = 1.0f - std::exp(-dt / tau);          // 低通系数
        lam_pitch += ((p - pitch[prev]) / dt - lam_pitch) * a;
        lam_yaw   += ((y - yaw[prev]) / dt - lam_yaw) * a;
    }
};
```

`tau` 是低通时间常数（例如 0.1 s），按实机鼠标噪声调。环形缓冲 `N=32` 约 256 B，L1 命中，无需 `_mm_prefetch`。

### 8.4 编译开关

与 WT 飞控一致：`/O2 /arch:AVX2 /fp:fast /GL`。`/Qvec-report:2` 可确认 `dot`/`cross` 是否被 FMA 融合。ZEM 没有值得显式 SIMD 的循环，不写 AVX2 intrinsic。
