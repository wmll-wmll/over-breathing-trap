"""实验 8 —— 多 Transformer 架构到底适用还是不适用。

★ 要回答的问题
---------------
把「一套 Transformer 同时算两件事」换成「两套 Transformer 各算一件事」，
在**参数量完全相同**的前提下，会不会更好？

★ 为什么这个问题直到现在才问得出来
-----------------------------------
实验 1-7 里，第二趟只有一个输出：力（3 维）。**没有第二个功能可以分。**
「按功能拆模块」在当时根本无从谈起 —— 拆了也是把一个空壳劈成两半。

实验 8 第一次有了第二个功能：**呼吸深度**。

    ┌────────────────────────────────────────────────────────┐
    │  快时钟（1 拍）  力  → 改位置  → 改光强                 │
    │  慢时钟（10 拍） 呼吸深度 k → 改能量吸收                │
    └────────────────────────────────────────────────────────┘

这两个功能在三个维度上不同：
    1. **输出的物理量不同**（力 = 方向矢量，呼吸 = 吸收速率）
    2. **时间尺度不同**（1 拍 vs 10 拍）
    3. **需要的信息粒度不同**（力要空间细节，呼吸只要「这里多亮」一个标量）

第 3 点是关键 —— 它让「拆开之后允许两条干路用不同分辨率」成为可能，
这是单干路**结构上做不到**的事。

★ 呼吸的物理
-------------
    dE_breath = BREATH_K × light × k  −  BREATH_C × k²

    线性项 = 吸得越深拿得越多
    二次项 = 吸得越深效率越低

于是每一点光强都有内部最优深度 k* = 2 × light：
    暗处最优深度小（吸也没用，不如省着点）
    亮处最优深度大
**常数 k 一定有损失** —— 这就是这一维动作存在的意义。

★ 五支臂（★ 唯一变量是架构，其它全部相同）
--------------------------------------------
    臂    力的干路   呼吸干路           总层数  相对算力
    A     2 层       和力共用同一套       2     2.000
    E     2 层       独立，1 层 @1/16     3     2.006   ← ★ 同算力对照
    B     1 层       独立，1 层 @满分辨率  2     1.100
    C     1 层       独立，1 层 @1/16     2     1.006
    D     1 层       和力共用同一套       1     1.000   ← ★ 同算力对照

    ★ 最关键的两组对照（同算力，只差「拆不拆」）
        A vs E ：算力 2.000 vs 2.006 —— 几乎完全一样，只差呼吸有没有独立干路
        D vs C ：算力 1.000 vs 1.006 —— 同上，但在更低的算力档位

    B vs C ：同层数、同参数量，只差呼吸干路的分辨率（1 → 1/16 算力）
    A vs D ：单干路多一层值多少（深度本身的收益）

★ 为什么要加臂 E（第一次试跑的教训）
------------------------------------
    第一版只有 A/B/C/D。试跑结果是 A 全部活满、B/C/D 全部饿死 ——
    看起来「单干路完胜」。但仔细看：A 的**力的干路有 2 层**，B/C/D 只有 1 层。
    所以那个差距完全可能是「力的干路太浅」，而不是「拆开不好」。
    **臂 E 就是来拆这个混淆的**：力给足 2 层，呼吸另开一条几乎免费的干路，
    算力与 A 完全相同。如果 E 也打不过 A，那才说明拆分本身没有价值。

判定
----
    两组同算力对照都 E ≥ A、C ≥ D  →  多 Transformer 架构**适用**
    两组都更差                      →    **不适用**
    一高一低                        →    结论依赖算力档位

用法
----
    python -m v11_digital_life.ab_test --arms A,B,C,D --generations 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from v11_digital_life import physics
from v11_digital_life.brain import Brain
from v11_digital_life.config import Config
from v11_digital_life.sandbox import SandboxWorld


# ══════════════════════════════════════════════════════════════════════
# 四支臂的唯一差异
# ══════════════════════════════════════════════════════════════════════

ARMS: dict[str, dict] = {
    # ── 实验 8：Transformer 家族内部的布局对照 ─────────────────────
    "A": dict(arch="transformer", split_pass2=False,
              n_layers_pass2=2, n_layers_meta=0, meta_pool=1),
    "E": dict(arch="transformer", split_pass2=True,
              n_layers_pass2=2, n_layers_meta=1, meta_pool=4),
    "B": dict(arch="transformer", split_pass2=True,
              n_layers_pass2=1, n_layers_meta=1, meta_pool=1),
    "C": dict(arch="transformer", split_pass2=True,
              n_layers_pass2=1, n_layers_meta=1, meta_pool=4),
    "D": dict(arch="transformer", split_pass2=False,
              n_layers_pass2=1, n_layers_meta=0, meta_pool=1),

    # ── ★ 实验 9：换架构家族（卷积）────────────────────────────────
    "N1": dict(arch="cnn", split_pass2=False,
               cnn_channels=(16, 32, 64), cnn_blocks=1),
    "N4": dict(arch="cnn", split_pass2=False,
               cnn_channels=(32, 64, 128), cnn_blocks=2),
    "N5": dict(arch="cnn", split_pass2=False,
               cnn_channels=(48, 96, 192), cnn_blocks=2),
    "N2": dict(arch="cnn", split_pass2=False,
               cnn_channels=(64, 128, 256), cnn_blocks=2),
    "N6": dict(arch="cnn", split_pass2=False,
               cnn_channels=(96, 192, 384), cnn_blocks=2),
    "N3": dict(arch="cnn", split_pass2=True,
               cnn_channels=(64, 128, 256), cnn_blocks=2,
               cnn_breath_channels=16),

    # ── ★ 实验 9 极端对照：完全不做空间处理 ────────────────────────
    "M1": dict(arch="mlp", split_pass2=False),

    # ── ★ 实验 9 消融：同宽度但更深（3 级 × 2 块）──────────────────
    "N7": dict(arch="cnn", split_pass2=False,
               cnn_channels=(16, 32, 64), cnn_blocks=2),
}

ARM_DESC = {
    "A": "TF 单干路 2 层（共享）",
    "E": "TF 力2层 + 呼吸1层(1/16)",
    "B": "TF 双干路 1+1，同分辨率",
    "C": "TF 双干路 1+1，呼吸 1/16",
    "D": "TF 单干路 1 层（算力对照）",
    "N1": "CNN 35k（16/32/64，1 块）",
    "N4": "CNN 305k（32/64/128，2 块）",
    "N5": "CNN 672k（48/96/192，2 块）",
    "N2": "CNN 1.22M（64/128/256，2 块）",
    "N6": "CNN 2.62M（96/192/384，2 块）",
    "N3": "CNN 1.23M + 独立呼吸分支(8×8)",
    "M1": "全局 MLP（只吃 4 个场统计量）",
    "N7": "CNN 71k（16/32/64，2 块＝更深）",
}

# 相对算力的基准：**一个满分辨率 Transformer 层**（16,385 token）的乘加数。
# 注意力 QK^T + AV 是 2·n²·d，MLP 是 n·d·4d·2。
_TF_GRID = 128
_TF_N = _TF_GRID * _TF_GRID + 1
_TF_D = 64
MACS_TF_LAYER = 2 * _TF_N * _TF_N * _TF_D + _TF_N * _TF_D * 4 * _TF_D * 2


def cnn_macs(channels, blocks, grid: int = _TF_GRID) -> float:
    """卷积分支的乘加数（不含池化和头）。"""
    m, cin, size = 0.0, 2, grid
    for c in channels:
        size //= 2
        m += size * size * c * cin * 9
        cin = c
        for _ in range(max(0, blocks - 1)):
            m += size * size * c * c * 9
    return m


def arm_cost(arm: str) -> float:
    """每拍花多少个「满分辨率 Transformer 层」的算力（解析估计）。"""
    a = ARMS[arm]
    if a.get("arch", "transformer") == "mlp":
        return 1e-5          # 只有几个线性层，相对一层 Transformer 可忽略
    if a.get("arch", "transformer") == "cnn":
        c = cnn_macs(a["cnn_channels"], a["cnn_blocks"]) / MACS_TF_LAYER
        if a.get("split_pass2"):
            # 呼吸分支：两次 stride=4，输出 8×8，只占主干极小一部分
            c += 0.0
        return c
    per = 1.0 / physics.BREATH_PERIOD
    n2 = a["n_layers_pass2"]
    if a["split_pass2"]:
        return n2 + (a["n_layers_meta"] * per) / (a["meta_pool"] ** 2)
    return float(n2)


# ══════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════

def make_cfg(arm: str, args) -> Config:
    cfg = Config()
    cfg.device = args.device
    cfg.life.max_steps = args.max_steps
    cfg.life.breath_period = physics.BREATH_PERIOD
    cfg.life.energy_reward_coef = args.energy_reward_coef
    cfg.life.breath_objective = getattr(args, "breath_objective", "policy")

    # ★ 实验 7 的结论：写场只会制造不稳定。四支臂统一关掉，
    #   并且跳过整个第一趟（省一半算力，不影响架构对照）。
    cfg.life.enable_write = False
    cfg.life.write_objective = "none"
    cfg.interface.write_gain_max = 0.0

    # ★ 呼吸：第二个可学习的动作
    cfg.field.breath_mode = True
    cfg.field.breath_k = physics.BREATH_K
    cfg.field.breath_c = physics.BREATH_C

    # ★★ 实验 12 之后 physics/config 的**默认行为变了**（改成正确的稳态公式
    #    和单调分支的动作区间）。实验 8-11 是在旧行为下跑的，这里显式钉住旧值，
    #    否则同一份代码跑不出报告里的数字。
    cfg.life.breath_gamma_mode = "coef"      # 旧的（错的）γ
    cfg.life.breath_limit_useful = False
    cfg.life.breath_account_move = False     # 旧的（全开）动作区间

    a = ARMS[arm]
    cfg.interface.arch = a.get("arch", "transformer")
    cfg.interface.split_pass2 = a["split_pass2"]
    if cfg.interface.arch == "cnn":
        cfg.interface.cnn_channels = tuple(a["cnn_channels"])
        cfg.interface.cnn_blocks = int(a["cnn_blocks"])
        cfg.interface.cnn_breath_channels = int(a.get("cnn_breath_channels", 16))
    elif cfg.interface.arch == "mlp":
        pass        # 只有全局统计量 + 世界观测，没有层数/分辨率可配
    else:
        cfg.interface.n_layers_pass2 = a["n_layers_pass2"]
        cfg.interface.n_layers_meta = max(1, a["n_layers_meta"])
        cfg.interface.meta_pool = a["meta_pool"]
    return cfg


# ══════════════════════════════════════════════════════════════════════
# 跑一支臂
# ══════════════════════════════════════════════════════════════════════

def run_arm(arm: str, args, out_dir: Path) -> dict:
    cfg = make_cfg(arm, args)
    brain = Brain(cfg, seed=args.seed)

    print()
    print("─" * 78)
    print(f"  臂 {arm}：{ARM_DESC[arm]}")
    print("─" * 78)
    print(f"  参数量        : {brain.n_params():,}")
    if cfg.interface.arch == "cnn":
        print(f"  主干          : CNN 通道 {tuple(cfg.interface.cnn_channels)} × "
              f"{cfg.interface.cnn_blocks} 块  → 池化 {brain.interface.final_size}×"
              f"{brain.interface.final_size}")
        print(f"  呼吸分支      : " + (
            f"独立（通道 {cfg.interface.cnn_breath_channels}，输出 8×8）"
            if cfg.interface.split_pass2 else "与主干共享池化特征"))
    else:
        print(f"  第二趟层数    : {cfg.interface.n_layers_pass2}"
              + (f"（+ 呼吸干路 {cfg.interface.n_layers_meta} 层）"
                 if cfg.interface.split_pass2 else "（共享）"))
        n_tok = (brain.interface.n_cells + 1 if cfg.interface.meta_pool == 1
                 else (cfg.field.grid // cfg.interface.meta_pool) ** 2 + 1)
        print(f"  呼吸干路分辨率: {n_tok:,} token")
    print(f"  相对算力      : {arm_cost(arm):.4f} 个 Transformer 层/拍")
    print(f"  呼吸系数      : K={cfg.field.breath_k:.0f}  C={cfg.field.breath_c:.0f}  "
          f"k ∈ [{physics.BREATH_MIN}, {physics.BREATH_MAX}]")
    print(f"  呼吸时钟      : 每 {cfg.life.breath_period} 拍决策一次")
    print()

    csv_path = out_dir / f"arm{arm}_beats.csv"
    fh = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    w.writerow(["gen", "step", "pos_x", "light", "norm_frac", "energy",
                "meta_gain", "k_breath", "v_norm", "move_cost", "force_norm",
                "std", "loss", "reward", "death_cause"])

    history: list[dict] = []
    t0 = time.time()

    for gen in range(1, args.generations + 1):
        # ★ rebirth 会重置场、呼吸状态、缓冲；keep_brain=True 保留网络和优化器
        brain.rebirth(keep_brain=True)

        env = SandboxWorld(
            cfg,
            seed=(gen if getattr(args, 'world_fixed', False) else args.seed * 100000 + gen),
            sigma=args.sigma,
            spawn_min=args.spawn_min,
            spawn_span=args.spawn_span,
        )
        obs = env.reset()

        steps = 0
        losses: list[float] = []
        norms: list[float] = []
        ks: list[float] = []
        lights: list[float] = []
        xs: list[float] = []
        n_decisions = 0

        while True:
            action, info = brain.step(obs, sample=True, train=True)
            obs, env_done, _einfo = env.step(action)
            steps += 1

            if "loss" in info:
                losses.append(float(info["loss"]))
            norms.append(float(info["norm_frac"]))
            ks.append(float(info["k_breath"]))
            lights.append(float(info["light_local"]))
            xs.append(float(env.pos[0]))
            if info["breath_decision"]:
                n_decisions += 1

            w.writerow([
                gen, steps,
                f"{env.pos[0]:.1f}",
                f"{info['light_local']:.5f}",
                f"{info['norm_frac']:.5f}",
                f"{info['energy']:.1f}",
                f"{info['meta_gain']:+.2f}",
                f"{info['k_breath']:.4f}",
                f"{info['v_norm']:.4f}",
                f"{info['move_cost']:.2f}",
                f"{info['force_norm']:.4f}",
                f"{info['std']:.4f}",
                f"{info.get('loss', float('nan')):.4f}",
                f"{info['reward']:+.4f}",
                info.get("death_cause", ""),
            ])

            if not info["alive"] or env_done or steps >= args.max_steps:
                break

        fh.flush()

        cause = info.get("death_cause", "") or "max_steps"
        # 能量稳定性：最后 50 拍的标准差
        tail = norms[-50:]
        norm_std = float(np.std(tail)) if tail else float("nan")
        # 呼吸策略是否学到了「跟着光走」：k 与 light 的相关系数
        if len(ks) > 2 and np.std(ks) > 1e-9 and np.std(lights) > 1e-9:
            k_light_corr = float(np.corrcoef(ks, lights)[0, 1])
        else:
            k_light_corr = float("nan")

        rec = {
            "gen": gen,
            "beats": steps,
            "cause": cause,
            "survived": steps >= args.max_steps,
            "final_norm": float(info["norm_frac"]),
            "mean_norm": float(np.mean(norms)),
            "norm_std_tail": norm_std,
            "final_x": float(env.pos[0]),
            "mean_x": float(np.mean(xs)),
            "mean_light": float(np.mean(lights)),
            "mean_k": float(np.mean(ks)),
            "std_k": float(np.std(ks)),
            "k_light_corr": k_light_corr,
            "n_breath_decisions": n_decisions,
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "final_k": float(info["k_breath"]),
        }
        history.append(rec)

        label = {"low": "饿死", "high": "撑死", "max_steps": "活满"}.get(cause, cause)
        print(f"  第 {gen:2d} 代：{steps:3d} 拍  {label:<3s}  "
              f"末能量 {rec['final_norm']:.3f}  末位置 x={rec['final_x']:6.1f}  "
              f"k={rec['mean_k']:.2f}  corr(k,光)={rec['k_light_corr']:+.2f}  "
              f"尾段能量σ={norm_std:.3f}")

    fh.close()
    dt = time.time() - t0

    beats = [h["beats"] for h in history]
    n_high = sum(1 for h in history if h["cause"] == "high")
    n_low = sum(1 for h in history if h["cause"] == "low")
    n_max = sum(1 for h in history if h["cause"] == "max_steps")
    first = float(np.mean(beats[: max(1, len(beats) // 2)]))
    last = float(np.mean(beats[len(beats) // 2:]))

    summary = {
        "arm": arm,
        "desc": ARM_DESC[arm],
        "arch": cfg.interface.arch,
        "params": brain.n_params(),
        "rel_cost": arm_cost(arm),
        "layers_pass2": cfg.interface.n_layers_pass2,
        "layers_meta": cfg.interface.n_layers_meta if cfg.interface.split_pass2 else 0,
        "meta_pool": cfg.interface.meta_pool,
        "cnn_channels": list(cfg.interface.cnn_channels),
        "cnn_blocks": cfg.interface.cnn_blocks,
        "meta_tokens": (brain.interface.n_cells + 1
                        if cfg.interface.meta_pool == 1
                        else (cfg.field.grid // cfg.interface.meta_pool) ** 2 + 1),
        "generations": len(history),
        "mean_beats": float(np.mean(beats)),
        "median_beats": float(np.median(beats)),
        "max_beats": int(max(beats)),
        "n_high": n_high, "n_low": n_low, "n_max": n_max,
        "first_half_beats": first,
        "second_half_beats": last,
        "mean_norm_std_tail": float(np.nanmean([h["norm_std_tail"] for h in history])),
        "mean_k_light_corr": float(np.nanmean([h["k_light_corr"] for h in history])),
        "mean_x": float(np.mean([h["mean_x"] for h in history])),
        "mean_light": float(np.mean([h["mean_light"] for h in history])),
        "mean_k": float(np.mean([h["mean_k"] for h in history])),
        "std_k": float(np.mean([h["std_k"] for h in history])),
        "wall_seconds": dt,
        "total_beats": int(sum(beats)),
        "sec_per_beat": float(dt / max(1, sum(beats))),
        "history": history,
        "csv": str(csv_path),
    }

    print()
    print(f"  汇总：平均存活 {summary['mean_beats']:.1f} 拍  "
          f"（前半 {first:.1f} → 后半 {last:.1f}）")
    print(f"        死因：饿死 {n_low} / 撑死 {n_high} / 活满 {n_max}")
    print(f"        平均 corr(k, 光) = {summary['mean_k_light_corr']:+.3f}  "
          f"尾段能量σ = {summary['mean_norm_std_tail']:.4f}")
    print(f"        耗时 {dt / 60:.1f} 分钟")
    return summary


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="实验 8：多 Transformer 架构对照")
    ap.add_argument("--arms", default="A,E,B,C,D", help="跑哪几支臂，逗号分隔")
    ap.add_argument("--generations", type=int, default=8, help="每支臂跑几代")
    ap.add_argument("--max-steps", type=int, default=350, help="单条命最多几拍")
    ap.add_argument("--device", default="", help="空 = 自动")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world-fixed", action="store_true",
                    help="★ 世界种子不随 --seed 变化（每代固定为 gen）。"
                         "这样多个种子之间只差模型初始化，不差世界的出生点，"
                         "跨架构对照才干净。否则出生点在 [505,545] 里随机，"
                         "光强差 0.63~0.70，会让存活拍数出现上百拍的方差，"
                         "把架构效应完全淹没。")
    ap.add_argument("--sigma", type=float, default=600.0,
                    help="光源的高斯宽度（世界单位）")
    ap.add_argument("--spawn-min", type=float, default=505.0)
    ap.add_argument("--spawn-span", type=float, default=40.0)
    ap.add_argument("--energy-reward-coef", type=float, default=0.3,
                    help="离生存带中心的稠密奖励系数（四支臂相同）")
    ap.add_argument("--breath-objective", default="policy",
                    choices=["policy", "homeostasis", "both"],
                    help="★ 实验 10：呼吸头怎么训。"
                         "policy=只走策略梯度（实验 8/9 的基线，会掉进过度呼吸陷阱）；"
                         "homeostasis=加可微目标（稳态能量 → 生存带中心）；"
                         "both=两者都要")
    ap.add_argument("--out", default=str(_ROOT / "data" / "V11" / "exp8"))
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            print(f"未知的臂：{a}（可选 {list(ARMS)}）")
            return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print("  实验 8 —— 多 Transformer 架构到底适用还是不适用")
    print("=" * 78)
    print(f"  臂            : {', '.join(arms)}")
    print(f"  每臂代数      : {args.generations}")
    print(f"  单命上限      : {args.max_steps} 拍")
    print(f"  世界          : 沙盒走廊，光源 x=900，σ={args.sigma}")
    print(f"  出生点        : x ∈ [{args.spawn_min}, "
          f"{args.spawn_min + args.spawn_span}]（暗处，必须走过去）")
    print(f"  呼吸          : K={physics.BREATH_K:.0f} C={physics.BREATH_C:.0f} "
          f"k∈[{physics.BREATH_MIN},{physics.BREATH_MAX}] 周期={physics.BREATH_PERIOD}")
    print(f"  稠密奖励系数  : {args.energy_reward_coef}")
    print(f"  写场          : 关闭（实验 7 结论）")
    print(f"  输出目录      : {out_dir}")

    t0 = time.time()
    results: dict[str, dict] = {}
    for arm in arms:
        results[arm] = run_arm(arm, args, out_dir)

    # ── 对照表 ──────────────────────────────────────────────────────
    print()
    print("=" * 100)
    print("  实验结果对照")
    print("=" * 100)
    hdr = (f"  {'臂':<3s} {'架构':<28s} {'算力':>7s} {'秒/拍':>6s} {'参数':>10s} "
           f"{'平均拍':>7s} {'饿死':>4s} {'活满':>4s} "
           f"{'能量σ':>7s} {'均k':>6s} {'corr':>7s}")
    print(hdr)
    print("  " + "-" * 98)
    for arm in arms:
        r = results[arm]
        print(f"  {arm:<3s} {r['desc']:<28s} {r['rel_cost']:>7.4f} "
              f"{r['sec_per_beat']:>6.3f} {r['params']:>10,d} "
              f"{r['mean_beats']:>7.1f} {r['n_low']:>4d} {r['n_max']:>4d} "
              f"{r['mean_norm_std_tail']:>7.4f} {r['mean_k']:>6.3f} "
              f"{r['mean_k_light_corr']:>+7.2f}")
    print()

    # ── 预注册的判定 ────────────────────────────────────────────────
    if all(a in results for a in "AEBCD"):
        print("  ── 实验 8 判定：多 Transformer（Transformer 家族内部布局对照）──")
        A, E, B, C, D = (results[k]["mean_beats"] for k in "AEBCD")
        eps = 4.0     # 4 拍以内算「没有区别」（实验 4 实测复现误差 ≤2 拍）
        print(f"    ① ★ 同算力（A 2.000× vs E 2.006×）："
              f"{A:.1f} vs {E:.1f} 拍  （{E - A:+.1f}）")
        print(f"    ② ★ 同算力（D 1.000× vs C 1.006×）："
              f"{D:.1f} vs {C:.1f} 拍  （{C - D:+.1f}）")
        print(f"    ③ 拆开后降分辨率（B vs C）：{B:.1f} → {C:.1f}  （{C - B:+.1f}）")
        print(f"    ④ 深度是不是决定性的（A vs D）：{A:.1f} vs {D:.1f}  （{A - D:+.1f}）")
        print()

    if "N1" in results or "N2" in results or "N3" in results:
        print("  ── ★ 实验 9 判定：换架构家族（CNN）─────────────────────────")
        ref = results.get("D") or results.get("A")
        if ref is not None:
            print(f"    Transformer 参照：{ref['arch']} / {ref['mean_beats']:.1f} 拍 / "
                  f"{ref['params']:,} 参数 / {ref['sec_per_beat']:.3f} 秒每拍")
        for n in ("N1", "N2", "N3"):
            if n in results:
                r = results[n]
                print(f"    {n}：{r['desc']:<28s} {r['mean_beats']:>6.1f} 拍 / "
                      f"{r['params']:>9,d} 参数 / {r['sec_per_beat']:.3f} 秒每拍 / "
                      f"能量σ {r['mean_norm_std_tail']:.4f}")
        print()
        if "N1" in results and "D" in results:
            nd = results["N1"]["mean_beats"] - results["D"]["mean_beats"]
            pr = results["N1"]["params"] / results["D"]["params"]
            sp = results["N1"]["sec_per_beat"] / results["D"]["sec_per_beat"]
            print(f"    ★ 小 CNN（N1，参数只有 D 的 {pr:.1%}，"
                  f"每拍耗时 {sp:.2f} 倍）比 1 层 Transformer 多活 {nd:+.1f} 拍")
            print()
        if "N2" in results and "A" in results:
            nd = results["N2"]["mean_beats"] - results["A"]["mean_beats"]
            sp = results["N2"]["sec_per_beat"] / results["A"]["sec_per_beat"]
            print(f"    ★ 参数匹配的大 CNN（N2）vs 2 层 Transformer（A）："
                  f"{nd:+.1f} 拍，耗时 {sp:.2f} 倍")
            print()

    summary_path = out_dir / f"exp8_{stamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "breath_k": physics.BREATH_K,
            "breath_c": physics.BREATH_C,
            "breath_period": physics.BREATH_PERIOD,
            "macs_tf_layer": MACS_TF_LAYER,
            "arms": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"  总耗时 ：{(time.time() - t0) / 60:.1f} 分钟")
    print(f"  汇总   ：{summary_path}")
    print(f"  逐拍   ：{out_dir}/*_beats.csv")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
