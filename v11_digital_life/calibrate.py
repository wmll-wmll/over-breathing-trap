"""实验 8 的世界/物理标定脚本 —— 跑之前先确认「难度」是对的。

要回答三件事：
    1. 出生点用**初始策略**（k≈0.30）能活多久？太短学不动，太长没压力。
    2. 一路上的理论平衡能量 E* = max_k(gain)/γ 是多少？（光强 → 生死）
    3. 光强分布对不对：暗处必死、中间活、太亮撑死？

用法
----
    python -m v11_digital_life.calibrate --sigma 600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from v11_digital_life import physics
from v11_digital_life.ab_test import make_cfg
from v11_digital_life.brain import Brain
from v11_digital_life.sandbox import SandboxWorld


def light_at(x: float, light_x: float, sigma: float) -> float:
    d = (x - light_x) / sigma
    return float(np.exp(-d * d))


def main() -> int:
    ap = argparse.ArgumentParser(description="实验 8 标定")
    ap.add_argument("--sigma", type=float, default=600.0)
    ap.add_argument("--light-x", type=float, default=900.0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    cfg = make_cfg("D", argparse.Namespace(
        device=args.device, max_steps=args.steps, energy_reward_coef=0.0, seed=0))
    cap = physics.energy_capacity(physics.mass_of(cfg.field.grid))
    gamma = cfg.field.dissipation

    print("=" * 76)
    print(f"  标定：光源 x={args.light_x}  σ={args.sigma}  能量容量 E0={cap:.0f}")
    print(f"  呼吸：K={physics.BREATH_K} C={physics.BREATH_C}  γ={gamma}")
    print("=" * 76)
    print()
    print("  「最优呼吸下的平衡能量」E* = max(gain)/γ = K·light²/(4Cγ)")
    print()
    print(f"  {'x':>6s} {'light':>7s} {'k*':>6s} {'max_gain':>9s} {'E*/E0':>7s}  判定")
    print("  " + "-" * 60)
    for x in (0, 200, 300, 400, 500, 540, 570, 600, 700, 900):
        L = light_at(x, args.light_x, args.sigma)
        kstar = min(physics.BREATH_MAX, physics.BREATH_K * L / (2 * physics.BREATH_C))
        gain = physics.breath_gain(L, kstar)
        estar = gain / gamma / cap
        if estar < physics.LOWER:
            verdict = "饿死"
        elif estar > physics.UPPER:
            verdict = "撑死"
        else:
            verdict = f"活 (带内 {(estar - physics.LOWER) / (physics.UPPER - physics.LOWER):.0%})"
        print(f"  {x:>6d} {L:>7.4f} {kstar:>6.3f} {gain:>9.1f} {estar:>7.3f}  {verdict}")
    print()

    # ── 初始策略能活多久 ────────────────────────────────────────────
    print("  ── 初始策略（k≈0.30，不做任何学习）在走廊各点的存活 ──")
    print()
    brain = Brain(cfg, seed=0)
    print(f"  初始呼吸深度 k0 = {brain._k_breath:.3f}")
    print()
    print(f"  {'x0':>6s} {'light0':>8s} {'存活拍':>7s} {'死因':>6s} "
          f"{'末x':>7s} {'末能量':>8s}")
    print("  " + "-" * 56)
    for x0 in (0, 200, 400, 480, 520, 540, 555, 570, 600, 650):
        env = SandboxWorld(cfg, seed=1, sigma=args.sigma,
                           spawn_min=float(x0), spawn_span=1.0)
        obs = env.reset()
        brain.rebirth(keep_brain=True)
        t = 0
        info = None
        while True:
            a, info = brain.heartbeat(obs, sample=False)
            obs, done, _ = env.step(a)
            t += 1
            if (not info["alive"]) or done or t >= args.steps:
                break
            if t % 40 == 0:      # 偶尔清一下图（heartbeat 会建图）
                pass
        cause = info["death_cause"] or "活满"
        print(f"  {x0:>6d} {light_at(x0, args.light_x, args.sigma):>8.4f} "
              f"{t:>7d} {cause:>6s} {env.pos[0]:>7.0f} {info['norm_frac']:>8.3f}")
    print()
    print("  判读：")
    print("    · 出生点存活 150~250 拍 = 难度合适（有学习时间，但确实有压力）")
    print("    · < 80 拍 = 太快死，学不到东西")
    print("    · = 活满上限 = 没压力，呼吸这一维动作没有存在意义")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
