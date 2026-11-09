"""实验 11 —— 种群 + 变异 + 选择：v11 第一次真正「进化」。

★ 要回答的问题
--------------
加上种群、变异、选择之后，系统能做到「单个个体跨代学习」做不到的事吗？

★★ 先导结论（决定了本实验的设计）
--------------------------------
实验 10 把「过度呼吸陷阱」修好之后，原来的世界**已经太容易了**：

    不学习、纯随机初始化的 8 个个体里，有 6 个直接活满 350 拍。

选择压力 ≈ 0。种群里没有「更好」和「更差」，进化无从谈起。
**没有压力就没有选择，没有选择就没有进化** —— 这是人口遗传学的第一课，
也是本实验必须先换世界的原因。

★ 世界 B：移动光源（唯一改动，物理法则一个字节没改）
---------------------------------------------------
    世界 A（实验 8-10）固定光源 → 找到亮处站着不动就赢了
    世界 B（本实验）  移动光源 → 站着不动必死，必须持续跟踪

标定（先导实测，先天 = 不学习直接活一条命）：
    乱走的随机个体      活 ~40 拍
    不动的个体          活 ~106 拍（光自己走开了）
    学会跟踪的个体      活满 350 拍

于是适应度有**两个台阶**，而且都是真实的行为差异：
    乱走 → 不动（省下运动耗能）→ 跟踪（这才是唯一能活满的策略）

★ 六支臂（每支臂消耗**同样多**的生命：120 条命）
------------------------------------------------
    臂          种群  选择    遗传        学习  说明
    EVO_NL       8   适应度   达尔文式    关    ★ 纯进化：只有变异+选择
    NONE_NL      8   无       达尔文式    关    ★ 纯进化对照（每个都繁殖）
    EVO          8   适应度   达尔文式    开    进化 + 一生学习（学习不遗传）
    NONE         8   无       达尔文式    开    选择对照
    EVOLAM       8   适应度   拉马克式    开    进化 + 跨代学习
    LAM          1   —        拉马克式    开    ★ 实验 10 的基线（等算力）

    ★ 三组预注册对照
      ① EVO_NL vs NONE_NL ：**选择**到底有没有用（唯一变量 = 谁当父母）
      ② EVO   vs NONE    ：同上，但个体一生也能学习
      ③ EVOLAM vs LAM   ：**种群**在跨代学习之外还加了多少

    ★ 为什么对照是「每个都繁殖一次」而不是「随机抽几个当父母」
      试跑时用的是随机抽取，结果对照也涨（86 → 195 拍）。
      那不是进化，是**遗传漂变**：8 个个体里抽 2 个当父母，几代之后
      全群都成了同一个祖先的后代，于是平均适应度变成「那个祖先好不好」
      的一次抽样。中性对照必须是「繁殖机会均等」。
      详见 evolution.select_parents 的注释。

    ★ 为什么 LAM 是 120 代而不是 15 代
      它只有一个个体，一代只产生一条命。要让它和种群臂消耗同样的生命，
      就必须跑 120 代。横轴一律用「累计生命数」，不用「代数」。

★ 判定
------
    选择和变异有效    ⇔  EVO_NL 显著 > RND_NL（适应度上升，对照不动）
    种群有净收益      ⇔  EVOLAM > LAM（在留出世界上的平均存活拍数）
    先天适应度能进化  ⇔  纯进化臂的末代基因组，在**不学习**的情况下也活得更久

用法
----
    python -m v11_digital_life.exp11_evolution --arms EVO_NL,RND_NL --world B
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

from v11_digital_life import physics
from v11_digital_life.config import Config, EvolutionConfig
from v11_digital_life.evolution import Population, diversity, genome_distance


# ══════════════════════════════════════════════════════════════════════
# 世界
# ══════════════════════════════════════════════════════════════════════

WORLDS = {
    # 实验 8-10 的世界：固定光源，宽光斑，出生在亮处
    "A": dict(sigma=600.0, light_amp=0.0, light_period=0.0,
              spawn_min=505.0, spawn_span=40.0),
    # ★ 实验 11 的世界：移动光源，窄光斑，出生在光斑附近
    "B": dict(sigma=250.0, light_amp=300.0, light_period=150.0,
              spawn_min=800.0, spawn_span=200.0),
    # ★★ 实验 13 的世界：把光强整体调暗一半（light_gain=0.5），光斑再窄一点。
    #    实验 12 修好呼吸之后，世界 B 变成「8 个随机初始化个体里 7 个直接
    #    活满 350 拍」—— 选择压力又消失了。世界 C 是重新标定过的：
    #    先天适应度 219~332（有梯度），而且一生学习几乎没有增益
    #    （219 → 230）—— 正好能把「进化」和「学习」两条通道分开测。
    "C": dict(sigma=200.0, light_amp=300.0, light_period=150.0,
              spawn_min=800.0, spawn_span=200.0, light_gain=0.5),
    # ★★ 实验 N1：**窗口内部**的世界。
    #    用 calibrate_world.py 扫出来的（8 个体 × 3 留出世界，不学习）：
    #      λ=0.500（世界 C） 活满  0.0%
    #      λ=0.525          活满 29.2%
    #      λ=0.550          活满 33.3%   ← 中间地带最大
    #      λ=0.575          活满 50.0%
    #      λ=0.625          活满 50.0% 但 100% 挤在 ≥300（已经开始饱和）
    #    0.500 → 0.525 是一道**悬崖**：活满率从 0 直接跳到 29%。
    #    世界 C 在悬崖下方，世界 D 在悬崖上方一点。
    "D": dict(sigma=200.0, light_amp=300.0, light_period=150.0,
              spawn_min=800.0, spawn_span=200.0, light_gain=0.55),
}
WORLD_DESC = {
    "A": "固定光源 σ=600（实验 8-10 的世界）",
    "B": "移动光源 σ=250，±300 单位 / 150 拍周期（实验 11 的世界）",
    "C": "★★ 移动光源 + 光强减半（σ=200，light_gain=0.5）—— 实验 13 重新标定",
    "D": "★★★ 实验 N1：窗口内部（σ=200，light_gain=0.55）—— 活满率 33%",
}


# ══════════════════════════════════════════════════════════════════════
# 六支臂
# ══════════════════════════════════════════════════════════════════════

ARMS: dict[str, dict] = {
    # ★ 纯进化：一生不学习，适应度 = 天生的。选择是唯一的改进来源。
    "EVO_NL": dict(population=8, selection="fitness", inheritance="darwinian",
                   mutation_rate=0.05, train=False, elitism=1),
    "NONE_NL": dict(population=8, selection="none", inheritance="darwinian",
                    mutation_rate=0.05, train=False, elitism=0),
    # ★ 审稿修正：**elitism 匹配**的无选择对照。
    #   原来的 EVO_NL（elitism=1）vs NONE_NL（elitism=0）把两种机制混在一起了：
    #   对照组比实验组少了「精英原样保留」。这一臂让两边都保留精英，
    #   于是差异只剩「选谁当父母」这一件事。
    #   顺带修掉基因池指标的子样本偏置：两边第一个位置都是精英。
    "NONE_EL": dict(population=8, selection="none", inheritance="darwinian",
                    mutation_rate=0.05, train=False, elitism=1),
    # ★ 进化 + 一生学习（达尔文式：学习成果不遗传）
    "EVO": dict(population=8, selection="fitness", inheritance="darwinian",
                mutation_rate=0.05, train=True, elitism=1),
    "NONE": dict(population=8, selection="none", inheritance="darwinian",
                 mutation_rate=0.05, train=True, elitism=0),
    # ★ 进化 + 跨代学习（拉马克式：学习成果遗传）
    "EVOLAM": dict(population=8, selection="fitness", inheritance="lamarckian",
                   mutation_rate=0.05, train=True, elitism=1),
    # ★ 实验 10 的基线：一个个体、一代一条命、学到的权重重生后保留
    "LAM": dict(population=1, selection="fitness", inheritance="lamarckian",
                mutation_rate=0.0, train=True, elitism=0,
                inherit_optimizer=True),
}

# ── 实验 N2：LAM 退化的 2×2×2 消融 ───────────────────────────────────
# 实验 14 发现 LAM 越学越差（训练 −16.4，「学完就学不动」）。它的配置是
# population=1 + mutation_rate=0 + inherit_optimizer=True 三件事叠在一起。
# 这三个因素各自在做什么？
#
#   population=1  → 没有选择压力，只有一条谱系在漂
#   mutation=0    → 跨代唯一的变量就是「这一生学到的东西」
#   inherit_optimizer=True → Adam 的一阶/二阶动量跨代保留（记忆，不是基因）
#
# 臂名 L<pop><mut><opt>：L101 就是现有的 LAM。
# 全部用 selection=fitness / lamarckian / train=True / elitism=0，
# 只动这三个因素，构成完整的 2×2×2。
_LAM_ABL = {}
for _pop in (1, 8):
    for _mut in (0.0, 0.05):
        for _opt in (False, True):
            _key = f"L{_pop}{0 if _mut == 0.0 else 1}{1 if _opt else 0}"
            _LAM_ABL[_key] = dict(
                population=_pop, selection="fitness",
                inheritance="lamarckian", mutation_rate=_mut,
                train=True, elitism=0, inherit_optimizer=_opt)
ARMS.update(_LAM_ABL)

ARM_DESC = {
    "EVO_NL": "纯进化（选择 + 变异，一生不学习）",
    "NONE_NL": "纯进化 · 无选择对照（每个都繁殖一次，无精英）",
    "NONE_EL": "★ 审稿修正：无选择对照，但保留精英（elitism 匹配）",
    "EVO": "进化 + 一生学习（达尔文式：学习不遗传）",
    "NONE": "进化 + 一生学习 · 无选择对照",
    "EVOLAM": "进化 + 跨代学习（拉马克式：学习遗传）",
    "LAM": "★ 基线：单谱系跨代学习（实验 10，120 条命同一谱系）",
}
ARM_DESC.update({
    "L100": "N2 消融：种群 1 / 无变异 / 不继承动量",
    "L101": "N2 消融：种群 1 / 无变异 / 继承动量（= LAM）",
    "L110": "N2 消融：种群 1 / 有变异 / 不继承动量",
    "L111": "N2 消融：种群 1 / 有变异 / 继承动量",
    "L800": "N2 消融：种群 8 / 无变异 / 不继承动量",
    "L801": "N2 消融：种群 8 / 无变异 / 继承动量",
    "L810": "N2 消融：种群 8 / 有变异 / 不继承动量",
    "L811": "N2 消融：种群 8 / 有变异 / 继承动量",
})

LIVES_PER_ARM = 120
"""每支臂消耗的生命数。种群臂 = 15 代 × 8 个体；LAM = 120 代 × 1。"""


def generations_for(arm: str, lives: int | None = None) -> int:
    return max(1, (lives or LIVES_PER_ARM) // int(ARMS[arm]["population"]))


# ══════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════

def make_cfg(args) -> Config:
    cfg = Config()
    cfg.device = args.device
    cfg.life.max_steps = args.max_steps
    cfg.life.breath_period = physics.BREATH_PERIOD
    cfg.life.energy_reward_coef = args.energy_reward_coef
    cfg.life.breath_objective = args.breath_objective

    # 写场自实验 7 起关闭（结论：只会制造不稳定）
    cfg.life.enable_write = False
    cfg.life.write_objective = "none"
    cfg.interface.write_gain_max = 0.0

    # 呼吸：第二个可学习动作
    cfg.field.breath_mode = True
    cfg.field.breath_k = physics.BREATH_K
    cfg.field.breath_c = physics.BREATH_C

    # ★★ 实验 12 之后 physics/config 的默认行为变了（正确的稳态公式 +
    #    单调分支的动作区间 + 运动耗能）。实验 11 是在旧行为下跑的，
    #    默认钉住旧值，保证那份报告的数字可复现。
    #    ★ 加 --breath-fixed 就切到「修好之后」的行为（实验 13 用）。
    if getattr(args, "breath_fixed", False):
        cfg.life.breath_gamma_mode = "step"
        cfg.life.breath_limit_useful = True
        cfg.life.breath_account_move = True
    else:
        cfg.life.breath_gamma_mode = "coef"
        cfg.life.breath_limit_useful = False
        cfg.life.breath_account_move = False

    # ★ 实验 9 的结论：35k 的 CNN（N1）是这个任务上最好的主干
    #   （303 拍 vs 1.2M Transformer 的 139 拍，快 17 倍）
    #   进化实验要跑上千条命，必须用它。
    cfg.interface.arch = "cnn"
    cfg.interface.cnn_channels = (16, 32, 64)
    cfg.interface.cnn_blocks = 1
    cfg.interface.split_pass2 = False
    return cfg


# ══════════════════════════════════════════════════════════════════════
# 跑一支臂
# ══════════════════════════════════════════════════════════════════════

def run_arm(arm: str, world: str, seed: int, args, out_dir: Path) -> dict:
    spec = ARMS[arm]
    cfg = make_cfg(args)
    gens = generations_for(arm, getattr(args, "lives", None))
    evo = EvolutionConfig(
        population=int(spec["population"]),
        generations=gens,
        selection=spec["selection"],
        inheritance=spec["inheritance"],
        mutation_rate=float(spec["mutation_rate"]),
        elitism=int(spec.get("elitism", 1)),
        inherit_optimizer=bool(spec.get("inherit_optimizer", False)),
    )
    world_kw = WORLDS[world]
    pop = Population(cfg, evo, seed=seed, world_kw=world_kw)

    arm_dir = out_dir / f"{world}_{arm}_seed{seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    gen_path = arm_dir / "generations.csv"
    ind_path = arm_dir / "individuals.csv"

    fg = open(gen_path, "w", newline="", encoding="utf-8")
    wg = csv.writer(fg)
    wg.writerow(["gen", "lives", "best", "mean", "worst", "std", "n_survived",
                 "diversity", "offspring_distance", "mean_k",
                 "mean_track_err", "mean_action", "parent_fitness"])

    fi = open(ind_path, "w", newline="", encoding="utf-8")
    wi = csv.writer(fi)
    wi.writerow(["gen", "lives", "index", "beats", "cause", "survived",
                 "mean_k", "final_k", "mean_light", "mean_track_err",
                 "final_track_err", "final_norm", "mean_energy",
                 "mean_action", "mean_loss"])

    print()
    print("─" * 78)
    print(f"  臂 {arm}：{ARM_DESC[arm]}")
    print("─" * 78)
    print(f"  种群 {evo.population} × {gens} 代 = {evo.population * gens} 条命")
    print(f"  选择 {evo.selection}   遗传 {evo.inheritance}   "
          f"变异率 {evo.mutation_rate}   学习 {'开' if spec['train'] else '关'}")
    print()

    t0 = time.time()
    lives = 0
    for g in range(1, gens + 1):
        rec = pop.run_generation(g, train=bool(spec["train"]))
        lives += int(evo.population)
        wg.writerow([g, lives, f"{rec['best']:.0f}", f"{rec['mean']:.2f}",
                     f"{rec['worst']:.0f}", f"{rec['std']:.2f}",
                     rec["n_survived"], f"{rec['diversity']:.4f}",
                     f"{rec['offspring_distance']:.4f}", f"{rec['mean_k']:.4f}",
                     f"{rec['mean_track_err']:.1f}", f"{rec['mean_action']:.4f}",
                     " ".join(f"{v:.0f}" for v in rec["parent_fitness"])])
        for r in rec["individuals"]:
            wi.writerow([g, lives, r["index"], r["beats"], r["cause"],
                         int(r["survived"]), f"{r['mean_k']:.4f}",
                         f"{r['final_k']:.4f}", f"{r['mean_light']:.5f}",
                         f"{r['mean_track_err']:.1f}", f"{r['final_track_err']:.1f}",
                         f"{r['final_norm']:.4f}", f"{r['mean_energy']:.4f}",
                         f"{r['mean_action']:.4f}",
                         f"{r['mean_loss']:.4f}" if r["mean_loss"] == r["mean_loss"] else ""])
        fg.flush(); fi.flush()

        # 单谱系基线每 10 代打一行，种群臂每代打一行
        verbose = (evo.population == 1 and (g % 10 == 0 or g == gens)) or \
                  (evo.population > 1 and (g % 2 == 0 or g == 1 or g == gens))
        if verbose:
            print(f"  gen {g:3d} ({lives:3d} 命)：best {rec['best']:5.0f}  "
                  f"mean {rec['mean']:6.1f}  worst {rec['worst']:5.0f}  "
                  f"活满 {rec['n_survived']}/{evo.population}  "
                  f"多样性 {rec['diversity']:.3f}  k {rec['mean_k']:.2f}  "
                  f"跟踪误差 {rec['mean_track_err']:5.0f}  "
                  f"({time.time() - t0:.0f}s)")
    fg.close(); fi.close()

    # ── 留出世界评估：末代最佳基因组 vs 末代种群 ──────────────────
    #   ★ 全部用「没见过的世界」评估（不同 seed → 不同出生点 + 不同光相位），
    #     防止进化只是把训练世界的细节背下来。
    #   ★ 同时测两次：
    #       innate  = 出生后**不学习**直接活（考查「先天知识」被进化写成什么样）
    #       learned = 允许活一辈子边活边学（考查这套基因型的真实生存能力）
    held = [700000 + k for k in range(args.heldout)]
    last = pop.history[-1]
    best_i = int(np.argmax(last["fitness"]))
    best_genome = pop.genomes[best_i]
    ev_inn = pop.evaluate(best_genome, held, train=False)
    ev_lea = pop.evaluate(best_genome, held, train=True)

    # 末代基因池的整体水平（取前 4 个基因型，每个考 5 个留出世界）
    sub = pop.genomes[: min(4, len(pop.genomes))]
    sub_worlds = held[:5]
    pool = [pop.evaluate(gm, sub_worlds, train=False)["mean_beats"] for gm in sub]

    sec = time.time() - t0
    first_mean = float(np.mean([h["mean"] for h in pop.history[:3]]))
    last_mean = float(np.mean([h["mean"] for h in pop.history[-3:]]))
    summary = {
        "arm": arm,
        "desc": ARM_DESC[arm],
        "world": world,
        "world_desc": WORLD_DESC[world],
        "seed": seed,
        "population": evo.population,
        "generations": len(pop.history),
        "lives": int(evo.population) * len(pop.history),
        "selection": evo.selection,
        "inheritance": evo.inheritance,
        "mutation_rate": evo.mutation_rate,
        "train": bool(spec["train"]),
        "max_steps": args.max_steps,
        "params": pop.brain.n_params(),
        # 训练集（各自固定世界）上的适应度
        "fitness_first": float(pop.history[0]["mean"]),
        "fitness_last": float(pop.history[-1]["mean"]),
        "fitness_first3": first_mean,
        "fitness_last3": last_mean,
        "beats_best_last": float(max(h["best"] for h in pop.history)),
        "n_survived_last": int(pop.history[-1]["n_survived"]),
        "diversity_first": float(pop.history[0]["diversity"]),
        "diversity_last": float(pop.history[-1]["diversity"]),
        "mean_k_last": float(pop.history[-1]["mean_k"]),
        "track_err_last": float(pop.history[-1]["mean_track_err"]),
        # ★ 留出世界（没见过）：防过拟合 + 跨臂可比的主指标
        "held_innate_mean": ev_inn["mean_beats"],
        "held_innate_worst": ev_inn["worst_beats"],
        "held_innate_beats": ev_inn["beats"],
        "held_learned_mean": ev_lea["mean_beats"],
        "held_learned_worst": ev_lea["worst_beats"],
        "held_learned_beats": ev_lea["beats"],
        "held_pool_mean": float(np.mean(pool)),
        "wall_seconds": sec,
        "sec_per_life": sec / max(1, int(evo.population) * len(pop.history)),
        "history": [
            {k: v for k, v in h.items() if k != "individuals"}
            for h in pop.history
        ],
        "dir": str(arm_dir),
    }

    print()
    print(f"  训练集适应度：首代 {summary['fitness_first']:.1f} → "
          f"末代 {summary['fitness_last']:.1f}  "
          f"（最好 {summary['beats_best_last']:.0f} 拍，"
          f"活满 {summary['n_survived_last']}/{evo.population}）")
    print(f"  多样性      ：{summary['diversity_first']:.3f} → "
          f"{summary['diversity_last']:.3f}")
    print(f"  ★ 留出世界  ：不学习 {summary['held_innate_mean']:.1f} 拍 / "
          f"学习 {summary['held_learned_mean']:.1f} 拍  "
          f"（最优个体最差 {summary['held_learned_worst']:.0f}）")
    print(f"                末代基因池平均 {summary['held_pool_mean']:.1f} 拍")
    print(f"  耗时        ：{sec / 60:.1f} 分钟（{summary['sec_per_life']:.2f} 秒/命）")

    with open(arm_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="实验 11：种群 + 变异 + 选择")
    ap.add_argument("--arms", default="EVO_NL,NONE_NL,EVO,NONE,EVOLAM,LAM")
    ap.add_argument("--worlds", default="B", help="逗号分隔，A=固定光 B=移动光")
    ap.add_argument("--seeds", default="0,1", help="逗号分隔的随机种子")
    ap.add_argument("--lives", type=int, default=LIVES_PER_ARM,
                    help="每支臂消耗多少条命（各臂相同 → 等算力对照）")
    ap.add_argument("--max-steps", type=int, default=350)
    ap.add_argument("--heldout", type=int, default=10,
                    help="留出世界个数（防过拟合，跨臂可比的主指标）")
    ap.add_argument("--energy-reward-coef", type=float, default=0.3)
    ap.add_argument("--breath-objective", default="homeostasis",
                    choices=["policy", "homeostasis", "both"])
    ap.add_argument("--breath-fixed", action="store_true",
                    help="★ 实验 13：用修好之后的呼吸目标（实验 12 的默认配置），"
                         "而不是实验 11 钉住的旧行为")
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default=str(_ROOT / "data" / "V11" / "exp11"))
    ap.add_argument("--resume", action="store_true",
                    help="续跑：已有 summary.json 的单元直接读回，不再重算")
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 78)
    print("  实验 11 —— 种群 + 变异 + 选择（v11 第一次真正进化）")
    print("=" * 78)
    print(f"  臂        : {', '.join(arms)}")
    print(f"  世界      : {', '.join(w + '=' + WORLD_DESC[w] for w in worlds)}")
    print(f"  种子      : {seeds}")
    print(f"  每臂生命  : {LIVES_PER_ARM} 条（各臂相同 → 等算力对照）")
    print(f"  单命上限  : {args.max_steps} 拍")
    print(f"  留出世界  : {args.heldout} 个（末代基因组在没见过的世界里考一遍）")
    print(f"  主干      : CNN 35k（16/32/64，1 块）—— 实验 9 的最优架构")
    print(f"  呼吸      : K={physics.BREATH_K:.0f} C={physics.BREATH_C:.0f} "
          f"周期={physics.BREATH_PERIOD}  目标={args.breath_objective}")
    print(f"  呼吸修复  : {'★ 已启用（实验 12 的默认配置）' if args.breath_fixed else '旧行为（实验 11 钉住）'}")
    print(f"  输出      : {out_dir}")

    t0 = time.time()
    results: dict[str, dict] = {}
    for w in worlds:
        for arm in arms:
            for s in seeds:
                key = f"{w}:{arm}:seed{s}"
                done = out_dir / f"{w}_{arm}_seed{s}" / "summary.json"
                if args.resume and done.exists():
                    print(f"\n[续跑] {key} 已完成，读回 {done.name}")
                    with open(done, encoding="utf-8") as f:
                        results[key] = json.load(f)
                    continue
                results[key] = run_arm(arm, w, s, args, out_dir)

    # ── 对照表 ──────────────────────────────────────────────────────
    print()
    print("=" * 118)
    print("  实验 11 结果对照（主指标 = 留出世界上的存活拍数：不学习 / 学习）")
    print("=" * 118)
    print(f"  {'世界':<2s} {'臂':<8s} {'选择':<8s} {'遗传':<11s} {'学习':<2s} "
          f"{'首代':>6s} {'末代':>6s} {'活满':>5s} {'多样性':>6s} "
          f"{'留出·先天':>9s} {'留出·学习':>9s} {'基因池':>7s}")
    print("  " + "-" * 116)
    for w in worlds:
        for arm in arms:
            keys = [k for k in results if k.startswith(f"{w}:{arm}:")]
            if not keys:
                continue
            r = results[keys[0]]

            def m(key):
                return float(np.mean([results[k][key] for k in keys]))

            print(f"  {w:<2s} {arm:<8s} {r['selection']:<8s} {r['inheritance']:<11s} "
                  f"{'开' if r['train'] else '关':<2s} "
                  f"{r['fitness_first']:>6.1f} {r['fitness_last']:>6.1f} "
                  f"{r['n_survived_last']:>3d}/{r['population']:<1d} "
                  f"{r['diversity_last']:>6.3f} {m('held_innate_mean'):>9.1f} "
                  f"{m('held_learned_mean'):>9.1f} {m('held_pool_mean'):>7.1f}")
        print()

    # ── 预注册判定 ──────────────────────────────────────────────────
    def g(arm, key, w=None):
        w = w or worlds[0]
        vals = [results[k][key] for k in results if k.startswith(f"{w}:{arm}:")]
        return float(np.mean(vals)) if vals else float("nan")

    for w in worlds:
        print(f"  ── 世界 {w} 判定（留出世界 · 不学习 / 学习）───────────────")
        for a, b, label in [
            ("EVO_NL", "NONE_NL", "① 选择有没有用（纯进化，一生不学习）"),
            ("EVO", "NONE", "② 选择有没有用（个体一生也在学习）"),
            ("EVO", "LAM", "③ 达尔文进化 vs 拉马克学习（同样 120 条命）"),
            ("EVOLAM", "LAM", "④ 种群在跨代学习之外加了多少"),
        ]:
            if a in arms and b in arms:
                for key, tag in [("held_innate_mean", "先天"),
                                 ("held_learned_mean", "学习")]:
                    x, y = g(a, key, w), g(b, key, w)
                    print(f"    {label}［{tag}］："
                          f"{a} {x:.1f} vs {b} {y:.1f} 拍  （{x - y:+.1f}）")
        print()

    summary_path = out_dir / f"exp11_{stamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "arms": ARMS,
            "arm_desc": ARM_DESC,
            "worlds": WORLDS,
            "lives_per_arm": LIVES_PER_ARM,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"  总耗时 ：{(time.time() - t0) / 60:.1f} 分钟")
    print(f"  汇总   ：{summary_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
