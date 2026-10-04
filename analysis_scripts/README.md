# AC8 制导律替换 · 分析脚本

本目录是 `issue-wt-zem/` 的复现脚本，对应分析报告
`../analysis_reports_zem-wtpatent.md`。

## 脚本清单

| 文件 | 作用 |
|---|---|
| `num20_fr_calib.py` | 用 AC7 28 架实测表标定 FR 与等效速率斜坡 L |
| `num21_two_class.py` | 被控对象 4 档 + 8 控制器（C0–C7）+ 鼠标/目标模型库 |
| `num21_main.py` | 主实验驱动：8 控制器 × 8 目标 × 4 鼠标 × 4 被控对象 = 1024 episode |
| `num21_report.py` | 出表 1–4 与图 A–E |
| `_lsweep2.py` | L 灵敏度扫描（128 episode/点） |
| `_lsum.py` | 汇总 L 灵敏度表 |
| `_lfig.py` | 出图 F（L 灵敏度曲线） |

## 复现

```bash
uv venv .venv --python 3.12
uv pip install -e .
uv run python num20_fr_calib.py        # -> fr_calib.npz
uv run python num21_main.py full       # ~189 s -> num21_full.npz
uv run python num21_report.py          # 表 1-4 + 图 A-E
for L in 0.15 0.25 0.35 0.45 0.55 0.667 0.80; do
  L_OVR=$L uv run python _lsweep2.py   # -> _ls4_$L.npy
done
uv run python _lsum.py                 # L 灵敏度表
uv run python _lfig.py                 # 图 F
```

主实验是 numba CPU 单线程，约 189 s，不触发 CUDA 路径。

## 重要口径说明

- 控制器 C5 = A 类（简化版战争雷霆飞控）、C7 = 纯 ZEM/PPN、
  C6 = B 类 J6 分层（击杀效率，NemoFC）。
- 被控对象四档 P0–P3，其中 P3 是决策档（完整刚体 + SAS 残余 12%）。
- 全部参数标定于 AC7 数据，AC8 上需要重标定（见分析报告 §局限）。
