"""经验版世界难度标定 —— 实验 N1 的第一步。

`calibrate.py` 只算理论平衡能量 E* = K·light²/(4Cγ)。它告诉你「亮处能不能活」，
但**不告诉你一个随机初始化的个体实际能活多久**。实验 13 造世界 C 时用的就是
「8 个随机初始化、不学习」的实测分布，这里把它做成一个可扫参数的脚本。

为什么要扫：实验 14 发现世界 C 不是「有梯度」，而是**双峰**（~230 死 / ~348 活，
2400 条命只有 3 条活满）。世界 B 修好呼吸后又直接饱和（第 2 代 8/8）。
两端都探到了，中间没有。这个脚本用来找窗口内部那一段。

判据（论文 §3.7 用）：
    - ``full_rate``：活满 350 拍的比例。想要 20–50%。
    - ``bimodal``：低模态（<250）与高模态（≥300）之间有没有中间地带。
      全都挤在一端 = 饱和或全灭；真正可演化的世界应该**连续铺开**。

用法
----
    python -m v11_digital_life.calibrate_world --gains 0.5,0.6,0.7,0.8
    python -m v11_digital_life.calibrate_world --gains 0.65 --individuals 8 --worlds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from v11_digital_life import physics
from v11_digital_life.config import Config, EvolutionConfig
from v11_digital_life.evolution import Population

# 留出世界（和 exp11_evolution 的 held 评估同一组种子，保证可比）
HELDOUT_BASE = 700000


def make_cfg(args) -> Config:
    cfg = Config()
    cfg.device = args.device
    cfg.life.max_steps = args.max_steps
    cfg.life.breath_period = physics.BREATH_PERIOD
    cfg.life.energy_reward_coef = args.energy_reward_coef
    cfg.life.breath_objective = args.breath_objective
    cfg.life.enable_write = False
    cfg.life.write_objective = "none"
    cfg.interface.write_gain_max = 0.0
    cfg.field.breath_mode = True
    cfg.field.breath_k = physics.BREATH_K
    cfg.field.breath_c = physics.BREATH_C
    # 和实验 13/14 一致：修好之后的呼吸行为
    cfg.life.breath_gamma_mode = "step"
    cfg.life.breath_limit_useful = True
    cfg.life.breath_account_move = True
    cfg.interface.arch = "cnn"
    cfg.interface.cnn_channels = (16, 32, 64)
    cfg.interface.cnn_blocks = 1
    cfg.interface.split_pass2 = False
    return cfg


def world_kw(gain: float, sigma: float, amp: float, period: float,
             spawn_min: float, spawn_span: float) -> dict:
    return dict(sigma=sigma, light_amp=amp, light_period=period,
                spawn_min=spawn_min, spawn_span=spawn_span, light_gain=gain)


def bimodality(beats: np.ndarray) -> float:
    """低模态(<250)与高模态(≥300)之间「中间地带」的占比。

    0 = 完全双峰（中间没人）；1 = 均匀铺开。
    世界 C 实测是双峰，这个值会接近 0。
    """
    if len(beats) == 0:
        return float("nan")
    mid = ((beats >= 250) & (beats < 300)).sum()
    return float(mid) / len(beats)


def main() -> int:
    ap = argparse.ArgumentParser(description="世界难度经验标定")
    ap.add_argument("--gains", default="1.0,0.75,0.7,0.65,0.6,0.55,0.5",
                    help="逗号分隔的 light_gain 列表")
    ap.add_argument("--sigma", type=float, default=200.0)
    ap.add_argument("--amp", type=float, default=300.0)
    ap.add_argument("--period", type=float, default=150.0)
    ap.add_argument("--spawn-min", type=float, default=800.0)
    ap.add_argument("--spawn-span", type=float, default=200.0)
    ap.add_argument("--individuals", type=int, default=8,
                    help="每档用几个独立随机初始化的个体")
    ap.add_argument("--worlds", type=int, default=3,
                    help="每个个体在几个留出世界里各活一次")
    ap.add_argument("--seeds", default="0", help="种群随机种子，逗号分隔")
    ap.add_argument("--max-steps", type=int, default=350)
    ap.add_argument("--energy-reward-coef", type=float, default=0.3)
    ap.add_argument("--breath-objective", default="homeostasis")
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    gains = [float(g) for g in args.gains.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = make_cfg(args)
    held = [HELDOUT_BASE + k for k in range(args.worlds)]

    print("=" * 88)
    print("  世界难度经验标定（随机初始化 · 不学习 · 多留出世界）")
    print("=" * 88)
    print(f"  个体/档   : {args.individuals}")
    print(f"  留出世界  : {args.worlds} 个  {held}")
    print(f"  单命上限  : {args.max_steps} 拍")
    print(f"  光斑      : σ={args.sigma}  幅={args.amp}  周期={args.period}")
    print(f"  出生      : [{args.spawn_min}, {args.spawn_min + args.spawn_span}]")
    print()
    print(f"  {'λ':>5} {'n':>4} {'均值':>7} {'标准差':>8} {'活满%':>7} "
          f"{'<250':>6} {'250-300':>8} {'≥300':>6}  {'BIC(2峰)':>9}")
    print("  " + "-" * 84)

    rows = []
    t0 = time.time()
    for gain in gains:
        beats: list[int] = []
        for s in seeds:
            evo = EvolutionConfig(population=args.individuals, generations=1,
                                  selection="none", inheritance="darwinian",
                                  mutation_rate=0.0, elitism=0)
            pop = Population(cfg, evo, seed=s,
                             world_kw=world_kw(gain, args.sigma, args.amp,
                                               args.period, args.spawn_min,
                                               args.spawn_span),
                             device=args.device)
            for gi, gm in enumerate(pop.genomes):
                for wi, w in enumerate(held):
                    r = pop.run_life(gm, int(w), train=False, sample=True,
                                     reset_optimizers=True,
                                     life_seed=s * 999983 + gi * 31 + wi)
                    beats.append(int(r["beats"]))
        b = np.array(beats, float)
        full = float((b >= args.max_steps).mean() * 100)
        low = float((b < 250).mean() * 100)
        mid = float(((b >= 250) & (b < 300)).mean() * 100)
        high = float((b >= 300).mean() * 100)
        rows.append(dict(gain=gain, n=len(b), mean=float(b.mean()),
                         std=float(b.std(ddof=1)) if len(b) > 1 else 0.0,
                         full_rate=full, low=low, mid=mid, high=high,
                         mid_frac=bimodality(b)))
        r = rows[-1]
        print(f"  {gain:>5.2f} {r['n']:>4d} {r['mean']:>7.1f} {r['std']:>8.1f} "
              f"{full:>6.1f}% {low:>5.1f}% {mid:>7.1f}% {high:>5.1f}%  "
              f"{r['mid_frac']:>8.3f}")

    print()
    print(f"  耗时 {time.time() - t0:.1f} 分钟")

    # 推荐档：活满率落在 20–50%、且中间地带比例最高的
    cand = [r for r in rows if 20 <= r["full_rate"] <= 50]
    if cand:
        best = max(cand, key=lambda r: r["mid_frac"])
        print(f"\n  ★ 推荐 λ = {best['gain']:.2f}"
              f"（活满 {best['full_rate']:.0f}%，中间地带 {best['mid_frac']:.3f}）")
    else:
        near = min(rows, key=lambda r: abs(r["full_rate"] - 35))
        print(f"\n  ⚠ 没有落在 20–50% 的档；最接近的是 λ = {near['gain']:.2f}"
              f"（活满 {near['full_rate']:.0f}%）")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"  结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
