# Patch：NemoFC（击杀效率飞控，B 类 J6 分层）

> 目标仓库：`FletcherMiya/AC8-Mouse-Aim`
> 本文只列代码段，不改动任何源文件。
> 收益数字来自 `analysis_scripts/num21_two_class.py` 的 1024 场景对照，指标定义见 `analysis_reports_zem-wtpatent.md`。
>
> 注意：本方案是「击杀优先」控制器，不在 `issue_main.md` 正文里。它单独记录于此，供有意追求击杀效率的读者参考。

---

## 1. 它是什么

NemoFC 是 J6 分层目标函数的落地：

1. **未进发射包线**：最小化进包线时间 `T_LAS`。对 (roll, pitch) 杆量做网格搜索 + 短视界前向 rollout，谁先进包线就选谁。
2. **已进发射包线**：切换到 ZEM（比例导引 N=3），最小化脱靶。

它主动放弃操纵体验：机头会离开鼠标去抢最短进包线，所以「跟随」会明显变差。这是「要击杀」的量化形态。

## 2. 收益与代价

P3 决策档（8 目标 × 4 鼠标档中位）：

| 控制器 | 进包线时间 | 建立前置 | 跟随误差 | 超调 | 成功率 |
|---|---|---|---|---|---|
| C0 基线（原律） | 3.62 s | 2.48 s | 21.57° | 39.20° | 53% |
| **C6 NemoFC** | **2.45 s** | 5.11 s | 29.27° | 62.18° | 53% |

- 进包线 2.45 s，四档被控对象上都是全场最快（比基线快 32%）。
- 代价：跟随 29.27°（+36%）、超调 62.18°（+59%）。这正是「适当放弃操纵体验感」的量化结果。
- 成功率 53%，与基线持平——它换的是速度，不是成功率。

## 3. 发射包线（LAS）判定

沿用前几轮定义：距离门 `[250, 1500] m` ∧（|γ| ≤ 30° ∨ |γ| ≥ 150°）∧ |θ| ≤ 45°。γ 是目标航向角，θ 是偏离机头角。

## 4. 参考实现

```cpp
// ---- NemoFC：J6 分层（未进包线 -> 最短进包线；已进包线 -> ZEM）----
struct NemoFC {
    static constexpr int   NC = 5;       // 网格 5×5
    static constexpr int   H  = 50;      // 前向视界步数
    static constexpr float DH = 0.08f;   // 前向步长 [s]（4 s 视界）
    float prev_r = 0.0f, prev_p = 0.0f;

    bool las_ok(float R, float gam, float th) const {
        if (R < 250.0f || R > 1500.0f) return false;
        float ag = std::fabs(gam);
        if (!(ag <= 30.0f * flight::rad || ag >= 150.0f * flight::rad)) return false;
        return std::fabs(th) <= 45.0f * flight::rad;
    }

    // 未进包线：网格搜最短进包线时间
    void step(const flight::Basis& b, const flight::V& om, float L,
              const flight::V& pm, const flight::V& pt, const flight::V& vt,
              float V, float gr0, float gp0, float& ur, float& up) {
        const float FR_R = 168.75f, FR_P = 44.63f;
        float best = 1e18f, br = gr0, bp = gp0;
        for (int i = 0; i < NC; ++i)
        for (int j = 0; j < NC; ++j) {
            float a = std::clamp(gr0 + (-0.6f + 0.3f * i), -1.0f, 1.0f);
            float c = std::clamp(gp0 + (-0.6f + 0.3f * j), -1.0f, 1.0f);
            // 短视界 rollout（带一阶滞后的刚体 + 位置推进）
            flight::Basis R = b;
            flight::V w = om;
            flight::V wc = {-FR_R * a * flight::rad, -FR_P * c * flight::rad, 0.0f};
            flight::V px = pm, tx = pt;
            float hit = 1e6f;
            for (int k = 0; k < H; ++k) {
                w = w + (wc - w) * (DH / L);
                R = rot_by_omega(R, w, DH);
                R = apply_coupling(R, V, DH);
                px = px + R.f * (V * DH);
                tx = tx + vt * DH;
                flight::V d = tx - px;
                float RR = std::sqrt(flight::dot(d, d));
                if (RR > 1e-6f) {
                    flight::V u = d * (1.0f / RR);
                    float vn = std::sqrt(flight::dot(vt, vt));
                    float gm = std::acos(std::clamp(flight::dot(u, vt) / std::max(vn, 1e-9f),
                                                    -1.0f, 1.0f));
                    float tt = std::acos(std::clamp(flight::dot(u, R.f), -1.0f, 1.0f));
                    if (las_ok(RR, gm, tt)) { hit = (k + 1) * DH; break; }
                }
            }
            float cost = hit < 1e5f ? hit : (1e5f + 40.0f * RR);
            cost += 0.6f * ((a - prev_r) * (a - prev_r) + (c - prev_p) * (c - prev_p));
            if (cost < best) { best = cost; br = a; bp = c; }
        }
        prev_r = br; prev_p = bp; ur = br; up = bp;
    }
};
```

已进包线后复用 `patch_ZEM.md` §4.2 的 ZEM 律。

## 5. 接入点

`update_commands()` 里按 `in_las` 分支：

```cpp
if (config.inner_loop == 2) {                       // 2 = NemoFC
    bool in_las = las_ok(R, gam, th) && (Vc > 0.0f);
    if (in_las) {
        // 复用 patch_ZEM.md 的 ZEM 律
    } else {
        nemo.step(b, omega, config.delay_l_roll, pm, pt, vt,
                  config.aircraft_speed, gr0, gp0, ur, up);
        command_roll.store(manual ? 0 : ur * config.roll_sign);
        command_pitch.store(manual ? 0 : up * config.pitch_sign);
    }
}
```

与 ZEM 一样，本方案需要目标位置与速度（见 `patch_ZEM.md` §3）。

## 6. 风险

| 项 | 说明 |
|---|---|
| 跟随差 | 主动放弃角度匹配，跟随 29.27°，比基线 +36% |
| 超调大 | 62.18°，是基线的 1.6 倍 |
| 数据依赖 | 需要目标位置/速度 + 空速，改动范围大 |
| 参数用 AC7 标定 | FR / L / 空速应按 AC8 重标定 |

## 7. 结论

NemoFC 只换速度不换成功率。如果目标是「手感 + 击杀」双优，应看 `patch_wtpatent.md`（WT 飞控）与 `patch_ZEM.md`（纯 ZEM）的组合；NemoFC 适合「只求最快击杀、不计手抖」的场景。
