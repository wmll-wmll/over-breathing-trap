"""实验 12 —— 修呼吸目标（γ 用错 + 目标有两个解）。

★ 要验证的两个修复点（实验 11 暴露）
------------------------------------
实验 11 发现：带学习的臂在**移动光源**的世界里完全失控 ——
呼吸深度 k 在 0.1 和 1.9 之间来回跳，LAM 的 120 条命里存活拍数 21~350 乱走。
查下去是两个独立的错误叠在一起：

  错误 1（γ 用错）
      稳态公式写成 E∞ = gain·(1−γ)/γ，γ = dissipation = 0.010。
      但场每拍真实的乘子是 r = exp(−γ·dt) = exp(−0.005) = 0.9950125，
      正确的稳态是 E∞ = gain·r/(1−r) = **199.5·gain**（而不是 99·gain）。
      → 目标增益被高估 **2.01 倍**（161.6 而不是 80.2）。
      → 161.6 对应的真实稳态是 3.22×E₀，**已经越过撑死线 3.00×E₀**。

  错误 2（目标有两个解）
      breath_gain 在 k 上是凹的，所以「某个目标增益」总有两个 k：

          light=0.9:  450·0.9·k − 200k² = 161.6  →  k = 0.55  或  k = 1.48
                                               └ 浅解      └ 深解

      两个解给出**同一个稳态能量**，所以目标函数根本区分不了它们。
      从初值 k=1.50 出发，梯度先把它推到最大增益点 k*=1.125·light ≈ 1.01，
      在那里两个解的梯度**大小相等、方向相反**（对称分岔），
      策略被噪声推向任一侧。静态光下落到浅解看着没事；**光源一动，
      深解（k≈1.8）就在暗处饿死**。

★ 修法
------
  γ：  r = exp(−γ·dt)，E∞ = gain·r/(1−r)，目标增益 g* = 80.2
  动作区间： 把 z 重参数化到 [BREATH_MIN, k*(light)]
             —— k* 是呼吸增益的最大值点，比它更深的呼吸**永远**更差
             （同样甚至更少的增益、更大的代价），
             所以 (k*, BREATH_MAX] 是一段**完全冗余**的动作区间。
             只保留增益单调的那一段，目标就一一可解了。

  ★ 为什么用「重参数化」而不是「夹紧 min(k, k*)」
    夹紧会让钳位区 dk_eff/dz = 0 —— 网络既学不到「该减小 k」，
    也没有任何力量把自己推出来（本实验的第一版就是这么卡死的）。
    重参数化让 dk/dz 处处非零，T15 里逐点验证过（z=±5 仍有 9e-3 ~ 1.3e-2）。

★ 设计：2×2 析因（这样才知道是哪个修复在起作用）
------------------------------------------------
    臂             γ        动作区间
    OLD_FULL      旧(错)    全开 [0.05, 2.0]     ← 实验 8-11 的配置
    OLD_MONO      旧(错)    单调 [0.05, k*]
    NEW_FULL      新(对)    全开 [0.05, 2.0]
    NEW_MONO      新(对)    单调 [0.05, k*]     ← ★ 完整修复

  两个世界都要跑：
    世界 A（固定光源）—— 实验 8-10 的世界，检查「修完还认得原来的活」
    世界 B（移动光源）—— 实验 11 的世界，这里是旧配置翻车的地方

用法
----
    python -m v11_digital_life.exp12_breath_fix --seeds 0,1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from v11_digital_life import physics
from v11_digital_life.brain import Brain
from v11_digital_life.config import Config
from v11_digital_life.sandbox import SandboxWorld


# ══════════════════════════════════════════════════════════════════════
# 世界（与实验 11 完全相同）
# ══════════════════════════════════════════════════════════════════════

WORLDS = {
    "A": dict(sigma=600.0, light_amp=0.0, light_period=0.0,
              spawn_min=505.0, spawn_span=40.0),
    "B": dict(sigma=250.0, light_amp=300.0, light_period=150.0,
              spawn_min=800.0, spawn_span=200.0),
}
WORLD_DESC = {
    "A": "固定光源 σ=600（实验 8-10 的世界）",
    "B": "★ 移动光源 σ=250，±300 单位 / 150 拍（实验 11 的世界）",
}


# ══════════════════════════════════════════════════════════════════════
# 四支臂（2×2 析因）
# ══════════════════════════════════════════════════════════════════════

ARMS: dict[str, dict] = {
    "OLD_FULL": dict(gamma="coef", limit=False, move=False),
    "OLD_MONO": dict(gamma="coef", limit=True, move=False),
    "NEW_FULL": dict(gamma="step", limit=False, move=False),
    "NEW_MONO": dict(gamma="step", limit=True, move=False),
    "FULL": dict(gamma="step", limit=True, move=True),
    # ★ 自查修正之后这个才是「推荐配置」：目标增益用实测的 exp(-2*gamma*dt)
    "FIXED": dict(gamma="step", limit=True, move=True),
    # ══════════════════════════════════════════════════════════════════
    # ★★ 实验 F（第五轮）：补齐 2×2 的另外两格
    #
    #   修复用的映射 k = k_min + (k*(light) − k_min)·u 同时做了两件事：
    #     ① 把可达区间压到增益峰值以下；
    #     ② 让区间**锚定到光强**（上界随光变）。
    #   此前没有任何一臂把这两件事分开，所以「是单调性在起作用」没有证据。
    #
    #   2×2：{上界固定, 上界随光} × {≤k*, >k*}
    #     OLD_FULL      固定（全开）/ >k*      ← 已有
    #     FIXED_BOUND   固定（k*(L_ref)）/ ≤k*  ← 新增
    #     TRACK_EXCESS  随光（2·k*）/ >k*      ← 新增
    #     FIXED         随光（k*）/ ≤k*        ← 已有
    # ══════════════════════════════════════════════════════════════════
    "FIXED_BOUND": dict(gamma="step", limit=True, move=True, mode="fixed"),
    "TRACK_EXCESS": dict(gamma="step", limit=True, move=True, mode="excess"),
    # ★★ 第六轮：审稿人指出 2×2 缺一个**配置匹配**的「固定/越峰」格。
    #   OLD_FULL 虽然也是固定/越峰，但它同时缺 γ 修正与运动耗能，
    #   拿它当基线会把三个变化混在一起。这一臂补上：同样的 step/move，
    #   上界固定为 2·k*(L_ref) = 0.925，与 FIXED_BOUND 的 0.462 正好差一倍，
    #   和追踪行里 2k*(L) 对 k*(L) 的倍率一致。
    "FIXED_EXCESS": dict(gamma="step", limit=True, move=True, mode="fixed_excess"),
}
ARM_DESC = {
    "OLD_FULL": "旧 γ + 全开区间（实验 8-11 的配置）",
    "OLD_MONO": "旧 γ + 单调区间",
    "NEW_FULL": "新 γ + 全开区间",
    "NEW_MONO": "新 γ + 单调区间",
    "FULL": "新 γ + 单调区间 + 运动耗能",
    "FIXED": "★ 推荐配置：实测正确的 γ + 单调区间 + 运动耗能",
    "FIXED_BOUND": "★ 实验 F：上界固定 k*(L_ref)，不随光变（2×2 的「只限制」格）",
    "TRACK_EXCESS": "★ 实验 F：上界 = 2·k*(light)，随光变但允许越过峰值（「只锚定」格）",
    "FIXED_EXCESS": "★ 实验 G：上界固定 2·k*(L_ref)=0.925，配置与其余三格匹配的「固定/越峰」格",
}

GENERATIONS = 12
"""单谱系跑多少代。实验 11 的 LAM 用的是 120 代，但这里只需要看趋势，
12 代就足够看出「稳定 / 震荡」。"""

WORLD_CYCLE = 8
"""世界种子按代轮换（和实验 11 的 Population.WORLD_CYCLE 一致）。"""


# ══════════════════════════════════════════════════════════════════════
# 一条命
# ══════════════════════════════════════════════════════════════════════

def make_cfg(arm: str, args) -> Config:
    spec = ARMS[arm]
    cfg = Config()
    cfg.device = args.device
    cfg.life.max_steps = args.max_steps
    cfg.life.breath_period = physics.BREATH_PERIOD
    cfg.life.energy_reward_coef = args.energy_reward_coef
    cfg.life.breath_objective = "homeostasis"

    cfg.life.enable_write = False
    cfg.life.write_objective = "none"
    cfg.interface.write_gain_max = 0.0

    cfg.field.breath_mode = True
    cfg.field.breath_k = physics.BREATH_K
    cfg.field.breath_c = physics.BREATH_C

    # ★ 本实验的两个自变量
    cfg.life.breath_gamma_mode = spec["gamma"]
    cfg.life.breath_limit_useful = spec["limit"]
    cfg.life.breath_bound_mode = spec.get("mode", "useful")
    cfg.life.breath_light_ref = physics.BREATH_LIGHT_REF
    cfg.life.breath_account_move = spec["move"]

    cfg.interface.arch = "cnn"
    cfg.interface.cnn_channels = (16, 32, 64)
    cfg.interface.cnn_blocks = 1
    cfg.interface.split_pass2 = False
    return cfg


def run_lineage(arm: str, world: str, seed: int, args, out_dir: Path) -> dict:
    cfg = make_cfg(arm, args)
    brain = Brain(cfg, seed=seed)
    wk = WORLDS[world]

    arm_dir = out_dir / f"{world}_{arm}_seed{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    fh = open(arm_dir / "beats.csv", "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    w.writerow(["gen", "step", "light", "k", "k_used", "norm_frac", "energy",
                "meta_gain", "v_norm", "move_cost", "alive", "death_cause"])

    t0 = time.time()
    history: list[dict] = []
    for g in range(1, GENERATIONS + 1):
        wseed = seed * 100000 + (g % WORLD_CYCLE) * 17 + 3
        torch.manual_seed(seed * 1000003 + g * 101)
        env = SandboxWorld(cfg, seed=wseed, **wk)
        obs = env.reset()
        brain.reset(seed=getattr(env, "episode_seed", None))

        steps = 0
        ks: list[float] = []
        lights: list[float] = []
        while True:
            action, info = brain.step(obs, sample=True, train=True)
            obs, env_done, _ = env.step(action)
            steps += 1
            k = float(info["k_breath"])
            light = float(info["light_local"])
            ks.append(k)
            lights.append(light)
            w.writerow([g, steps, f"{light:.5f}", f"{k:.5f}",
                        f"{float(info['k_used']):.5f}",
                        f"{info['norm_frac']:.5f}", f"{info['energy']:.1f}",
                        f"{info['meta_gain']:+.2f}", f"{info['v_norm']:.4f}",
                        f"{info['move_cost']:.2f}",
                        int(bool(info["alive"])), info.get("death_cause", "")])
            if not info["alive"] or env_done or steps >= cfg.life.max_steps:
                break
        cause = info.get("death_cause", "") or "max_steps"
        k_arr = np.asarray(ks, dtype=float)
        # ★ 核心诊断：一条命里 k 的**时间标准差**（慢时钟量的抖动）
        history.append({
            "gen": g,
            "beats": steps,
            "cause": cause,
            "survived": steps >= cfg.life.max_steps,
            "mean_k": float(k_arr.mean()),
            "std_k": float(k_arr.std()),
            "min_k": float(k_arr.min()),
            "max_k": float(k_arr.max()),
            "mean_light": float(np.mean(lights)),
            "final_norm": float(info["norm_frac"]),
        })
    fh.close()

    beats = np.array([h["beats"] for h in history], dtype=float)
    std_k = np.array([h["std_k"] for h in history], dtype=float)
    mean_k = np.array([h["mean_k"] for h in history], dtype=float)
    sec = time.time() - t0
    summary = {
        "arm": arm,
        "desc": ARM_DESC[arm],
        "gamma_mode": ARMS[arm]["gamma"],
        "limit_useful": bool(ARMS[arm]["limit"]),
        "bound_mode": ARMS[arm].get("mode", "full" if not ARMS[arm]["limit"] else "useful"),
        "light_ref": physics.BREATH_LIGHT_REF,
        "account_move": bool(ARMS[arm]["move"]),
        "world": world,
        "world_desc": WORLD_DESC[world],
        "seed": seed,
        "generations": GENERATIONS,
        "max_steps": args.max_steps,
        # ★ 两个主指标
        "mean_beats": float(beats.mean()),
        "std_beats": float(beats.std()),
        "n_survived": int(sum(1 for h in history if h["survived"])),
        "k_temporal_std": float(std_k.mean()),      # 一条命内 k 的抖动
        "k_generation_std": float(mean_k.std()),    # 跨代的 k 抖动
        "k_min": float(min(h["min_k"] for h in history)),
        "k_max": float(max(h["max_k"] for h in history)),
        "k_last3": float(mean_k[-3:].mean()),
        "final_norm_last": float(history[-1]["final_norm"]),
        "wall_seconds": sec,
        "history": history,
        "dir": str(arm_dir),
    }
    print(f"  {world} {arm:<9s} seed{seed}: 存活 {summary['mean_beats']:5.1f} ± "
          f"{summary['std_beats']:5.1f}  活满 {summary['n_survived']:2d}/{GENERATIONS}  "
          f"k 命内σ={summary['k_temporal_std']:.3f}  跨代σ="
          f"{summary['k_generation_std']:.3f}  k末3代={summary['k_last3']:.3f}  "
          f"({sec:.0f}s)", flush=True)
    with open(arm_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="实验 12：修呼吸目标")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--worlds", default="A,B")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--max-steps", type=int, default=350)
    ap.add_argument("--energy-reward-coef", type=float, default=0.3)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=str(_ROOT / "data" / "V11" / "exp12"))
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    worlds = [w.strip().upper() for w in args.worlds.split(",") if w.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for a in arms:
        if a not in ARMS:
            print(f"未知的臂：{a}（可选 {list(ARMS)}）")
            return 2
    for w in worlds:
        if w not in WORLDS:
            print(f"未知的世界：{w}（可选 {list(WORLDS)}）")
            return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    cfg0 = Config()
    g_star = physics.breath_optimal_gain(
        cfg0.field.dissipation, 10000.0, cfg0.field.dt, mode="step")

    print("=" * 92)
    print("  实验 12 —— 修呼吸目标（γ 用错 + 目标有两个解）")
    print("=" * 92)
    print(f"  臂        : {', '.join(arms)}")
    print(f"  世界      : {', '.join(worlds)}")
    print(f"  种子      : {seeds}")
    print(f"  每支臂    : 单谱系 × {GENERATIONS} 代")
    print(f"  主干      : CNN 35k（实验 9 的最优架构）")
    print(f"  呼吸目标  : homeostasis   γ_mode ∈ {{coef(旧), step(新)}}  "
          f"动作区间 ∈ {{全开, 单调}}")
    print(f"  正确目标增益 g* = {g_star:.2f}（旧公式会算成 "
          f"{physics.breath_optimal_gain(cfg0.field.dissipation, 10000.0, cfg0.field.dt, mode='coef'):.2f}）")
    print(f"  输出      : {out}")
    print()

    t0 = time.time()
    results: dict[str, dict] = {}
    for w in worlds:
        print(f"── 世界 {w}：{WORLD_DESC[w]} ──")
        for arm in arms:
            for s in seeds:
                results[f"{w}:{arm}:seed{s}"] = run_lineage(arm, w, s, args, out)
        print()

    # ── 对照表 ──────────────────────────────────────────────────────
    print("=" * 104)
    print("  实验 12 结果对照（单谱系 12 代；k 的两个 σ 越小越稳）")
    print("=" * 104)
    print(f"  {'世界':<3s} {'臂':<9s} {'γ':<5s} {'区间':<5s} {'运动':<3s} "
          f"{'存活':>7s} {'活满':>6s} {'k 命内σ':>8s} {'k 跨代σ':>8s} "
          f"{'k 末3代':>8s} {'k 范围':>13s}")
    print("  " + "-" * 100)
    for w in worlds:
        for arm in arms:
            ks = [k for k in results if k.startswith(f"{w}:{arm}:")]
            if not ks:
                continue
            r = results[ks[0]]

            def m(key):
                return float(np.mean([results[k][key] for k in ks]))

            lo = min(results[k]["k_min"] for k in ks)
            hi = max(results[k]["k_max"] for k in ks)
            print(f"  {w:<3s} {arm:<9s} {r['gamma_mode']:<5s} "
                  f"{'单调' if r['limit_useful'] else '全开':<5s} "
                  f"{'算' if r['account_move'] else '否':<3s} "
                  f"{m('mean_beats'):>7.1f} "
                  f"{sum(results[k]['n_survived'] for k in ks):>3d}/"
                  f"{GENERATIONS * len(ks):<2d} "
                  f"{m('k_temporal_std'):>8.3f} {m('k_generation_std'):>8.3f} "
                  f"{m('k_last3'):>8.3f} [{lo:.2f}, {hi:.2f}]")
        print()

    # ── 预注册判定 ──────────────────────────────────────────────────
    def g(arm, key, w):
        ks = [k for k in results if k.startswith(f"{w}:{arm}:")]
        return float(np.mean([results[k][key] for k in ks])) if ks else float("nan")

    for w in worlds:
        print(f"  ── 世界 {w} 判定 ──────────────────────────────────────")
        a, b = g("OLD_FULL", "k_generation_std", w), g("FULL", "k_generation_std", w)
        print(f"    ① 修完之后呼吸深度的跨代抖动："
              f"σ {a:.3f} → {b:.3f}（{b / a:.3f} 倍）")
        a2, b2 = g("OLD_FULL", "mean_beats", w), g("FULL", "mean_beats", w)
        print(f"    ② 存活拍数：{a2:.1f} → {b2:.1f} 拍  （{b2 - a2:+.1f}）")
        a3 = sum(r["n_survived"] for k2, r in results.items()
                 if k2.startswith(f"{w}:OLD_FULL:")) / (GENERATIONS * len(seeds))
        b3 = sum(r["n_survived"] for k2, r in results.items()
                 if k2.startswith(f"{w}:FULL:")) / (GENERATIONS * len(seeds))
        print(f"    ③ 活满率：{a3:.1%} → {b3:.1%}")
        # 逐项拆解：每一处修复单独加进来值多少
        print(f"    ④ 逐处修复（跨代 σ / 存活 / 活满率）：")
        for arm in ARMS:
            if not any(k2.startswith(f"{w}:{arm}:") for k2 in results):
                continue
            rr = [r for k2, r in results.items() if k2.startswith(f"{w}:{arm}:")]
            surv = sum(r["n_survived"] for r in rr) / (GENERATIONS * len(rr))
            print(f"       {arm:<9s} γ={rr[0]['gamma_mode']:<4s} "
                  f"区间={'单调' if rr[0]['limit_useful'] else '全开'} "
                  f"运动={'算' if rr[0]['account_move'] else '否'}  "
                  f"→ σ={g(arm, 'k_generation_std', w):.3f}  "
                  f"存活={g(arm, 'mean_beats', w):.1f}  活满={surv:.0%}")
        print()

    summary_path = out / f"exp12_{stamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "arms": ARMS, "arm_desc": ARM_DESC,
                   "worlds": WORLDS, "generations": GENERATIONS,
                   "g_star": g_star, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"  总耗时 ：{(time.time() - t0) / 60:.1f} 分钟")
    print(f"  汇总   ：{summary_path}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
