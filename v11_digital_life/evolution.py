"""V11 进化 —— 种群、变异、选择、遗传。

★ 这个模块补上了 v11 一直缺的那一半
------------------------------------
实验 1-10 的系统有**学习**，但没有**进化**：

    一个个体 → 活一辈子（边活边学）→ 死 → 带着学到的权重重生 → 再活……

这里面缺三样东西，而这三样恰好是「生命」这个词的定义性成分：

    1. 种群（variation）：同时存在很多个**不一样的**个体
    2. 选择（selection）：活得好的留下后代，活得差的断子绝孙
    3. 遗传（heredity）：后代的基因来自父代，所以「好」可以被积累

本模块把它们全部补上，而且保持 v11 的一条铁律：
**适应度只能由「活着」定义** —— 没有手写的奖励函数、没有人工设计的
行为目标。一个个体好不好，唯一的标准就是它在这条走廊里活了多少拍。

★ 基因是什么
-------------
基因 = 网络权重（`interface.state_dict()`）。

这是一个有意的选择：v11 的「身体」是 NLSE 场，「神经系统」是那个网络。
场是**每条命都重新长出来的**（实验 1-5 已经证明它每次从随机初值开始），
世代之间能传下去的只有网络。所以网络权重就是基因组的物理载体。

★ 三种遗传/学习组合（本模块都支持）
----------------------------------
    Lamarckian 拉马克式：后天学到的权重遗传给后代
                          → 对应实验 1-10 的单谱系跨代学习
    Darwinian  达尔文式：只有出生时的基因遗传，学习成果随死亡消失
                          → 对应生物界；被选择的变成「天赋」和「学习速度」
    两者都开：进化和学习同时作用于同一条谱系（数字生命应有的样子）

用法
----
    from v11_digital_life.evolution import Population
    pop = Population(cfg, world_kw=dict(sigma=250.0, light_amp=300.0,
                                        light_period=150.0))
    for gen in range(15):
        rec = pop.run_generation(gen)
        print(rec["best"], rec["mean"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import Config, EvolutionConfig
from .brain import Brain
from .sandbox import SandboxWorld


# ══════════════════════════════════════════════════════════════════════
# 基因组的基本操作（纯函数，便于单测）
# ══════════════════════════════════════════════════════════════════════

def clone_genome(genome: dict) -> dict:
    """深拷贝一份基因组（权重 + 缓冲区）。"""
    return {k: v.detach().clone() for k, v in genome.items()}


def mutate(genome: dict, rate: float, seed: int = 0,
           floor: float = 1e-5) -> dict:
    """★ 变异：每个参数加一点高斯噪声，幅度按**该张量自己的尺度**定标。

        σ_k = max(rate × RMS(θ_k), floor)

    ★ 为什么必须按张量定标（实验 11 的设计决定）
      同一个网络里参数的尺度差两个数量级：
        卷积核   RMS ≈ 0.1    → σ ≈ 0.005
        呼吸头 bias ≈ 0.95    → σ ≈ 0.048
      如果用统一绝对 σ，小尺度张量被炸掉、大尺度张量又等于没动。
      按 RMS 定标 = 所有张量被扰动的**相对幅度**一致。

    ★ 为什么用独立 generator 而不是全局 manual_seed
      全局 RNG 一改，动作采样的随机序列跟着变 —— 那等于「变异」
      顺带改变了每个个体的命运，选择和变异会缠在一起。
      这里用一把独立的、由 seed 决定的 generator，只碰基因组。
    """
    out: dict = {}
    for i, (k, v) in enumerate(genome.items()):
        v = v.detach()
        if not v.is_floating_point():
            out[k] = v.clone()
            continue
        dev = v.device
        gen = torch.Generator(device=dev.type)
        gen.manual_seed(int(seed) * 100003 + i * 7919)
        rms = float(v.pow(2).mean().sqrt().item()) if v.numel() else 0.0
        sigma = max(float(rate) * rms, float(floor))
        noise = torch.randn(v.shape, generator=gen, device=dev, dtype=v.dtype)
        out[k] = v + noise * sigma
    return out


def genome_distance(a: dict, b: dict) -> float:
    """两个基因组的**相对** L2 距离：‖a−b‖ / ‖a‖。

    0 = 完全相同；0.05 = 每代整体偏了 5%（≈ 一次典型变异）。
    """
    num, den = 0.0, 0.0
    for k in a:
        da = a[k].detach().float()
        db = b[k].detach().float()
        num += float((da - db).pow(2).sum())
        den += float(da.pow(2).sum())
    return math.sqrt(num / max(den, 1e-12))


def diversity(genomes: list[dict], maximum: int = 24) -> float:
    """种群的遗传多样性 = 两两相对距离的均值。

    ★ 为什么这是一个必须记录的诊断量
      选择会**消耗**多样性：每代只让前几名繁殖，差异被迅速抹平。
      多样性归零 = 全种群变成同一个克隆 = 进化停止（只剩突变-漂变平衡）。
      所以「适应度上升」和「多样性保持」必须同时看：
        适应度涨、多样性归零 → 只是收敛，不是持续进化。
    """
    if len(genomes) < 2:
        return 0.0
    idx = list(range(len(genomes)))
    if len(idx) > maximum:
        step = len(idx) / maximum
        idx = [int(i * step) for i in range(maximum)]
    ds = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            ds.append(genome_distance(genomes[idx[i]], genomes[idx[j]]))
    return float(np.mean(ds)) if ds else 0.0


def select_parents(fitness: list[float], n_parents: int, rng,
                   mode: str = "fitness") -> list[int]:
    """选出能繁殖的个体编号。

    mode="fitness"：前 n 名（**选择**）
    mode="none"   ：每个个体各繁殖一次（**无选择对照**）

    ★ 为什么对照必须是「每个都繁殖一次」而不是「随机抽 n 个当事人」
      ——这是实验 11 试跑踩到的坑。

      第一版对照用的是 `rng.choice(n, size=2, replace=False)`。结果它
      也跟着涨（8 代里平均从 86 涨到 195 拍），看起来「没有选择也能进化」。

      原因不是选择，是**遗传漂变**：从 8 个个体里随机抽 2 个当父母，
      代代抽下去，种群很快变成一两个祖先的后代（实测 8 代后多样性
      0.68 → 0.085）。于是整群人的适应度被压缩成「那个祖先好不好」的
      **一次抽样** —— 方差 29~350 的一抽，本来就可能抽到 350。

      正确的中性对照是「每个个体各留下一个后代」：
      繁殖机会完全均等 → 基因频率没有系统性方向 → 适应度只随突变扩散，
      不随选择上升。这才是「没有选择」的严格定义。"""
    n = len(fitness)
    n_parents = max(1, min(int(n_parents), n))
    if mode == "none":
        return list(range(n))
    if mode == "random":
        return [int(i) for i in rng.choice(n, size=n_parents, replace=False)]
    if mode != "fitness":
        raise ValueError(f"未知的 selection：{mode!r}（可选 fitness / random / none）")
    order = sorted(range(n), key=lambda i: fitness[i], reverse=True)
    return order[:n_parents]


# ══════════════════════════════════════════════════════════════════════
# 种群
# ══════════════════════════════════════════════════════════════════════

class Population:
    """一个会繁殖、会变异、会被选择的种群。

    生命流程（每一代）
    -----------------
        for 每个个体：
            装上它的基因组 → 在它自己的世界里过完一生
            → 记录适应度（= 活了多少拍）、死因、平均呼吸深度
        选择：按适应度（或随机，对照组）挑出父母
        繁殖：精英原样保留 + 其余 = 父代基因 ∘ 变异
        遗传：拉马克式用「死前学会的」基因组，达尔文式用「出生时」的
    """

    def __init__(self, cfg: Config | None = None,
                 evo: EvolutionConfig | None = None,
                 seed: int = 0,
                 world_kw: dict | None = None,
                 device: str | None = None):
        self.cfg = cfg or Config()
        self.evo = evo or self.cfg.evolution
        self.seed = int(seed)
        self.device = device
        self.world_kw = dict(world_kw or {})
        self.rng = np.random.default_rng(self.seed * 1009 + 7)

        self.brain = Brain(self.cfg, seed=self.seed, device=device)
        self._build_initial_population()
        self.history: list[dict] = []
        self.generation = 0

    # ── 初始化 ──────────────────────────────────────────────────────
    def _build_initial_population(self) -> None:
        """第 0 代：每个个体都是一次**独立的随机初始化**。

        ★ 为什么不是同一个网络复制 8 份
          那样第 0 代全等 → 适应度全等 → 选择完全无信息。
          真实的种群本来就长得不一样，差异是选择的原料。
        """
        self.genomes: list[dict] = [self.brain.get_genome()]
        for i in range(1, int(self.evo.population)):
            b = Brain(self.cfg, seed=self.seed + i * 7919, device=self.device)
            self.genomes.append(b.get_genome())
            del b

    # ── 一条命 ──────────────────────────────────────────────────────
    WORLD_CYCLE = 8
    """训练期轮换多少个世界（按代数）。

    ★ 同一代**所有个体共用同一个世界**，世界按代数在这个集合里轮换。

    ★ 为什么（实验 11 试跑踩的第二个坑）
      第一版是「每个个体绑定一个固定世界」（岛屿模型），看起来很公平。
      但繁殖之后子代是按**位置**重新排的：子代 k 落在第 k 个位置，
      于是被选出来的冠军基因下一代会**换到另一个出生点上重新评估**。
      等于每一代都在给冠军换赛道 —— 适应度信号里混进了「换赛道」的噪声。
      实测：3 个种子里 1 个完全学不动，另外 2 个曲线剧烈震荡。

      改成「同代同世界」之后，一代之内的适应度差异**只来自基因型**，
      选择才干净；而世界按代轮换提供环境多样性，压一压过拟合。
      所有臂用同一套世界序列，所以跨臂对照仍然严格配对。
    """

    def world_seed(self, index: int = 0, gen: int | None = None) -> int:
        """这一代里所有个体共同面对的世界种子。"""
        g = int(self.generation if gen is None else gen)
        k = g % self.WORLD_CYCLE
        return self.seed * 100000 + k * 17 + 3

    def run_life(self, genome: dict, world_seed: int, train: bool = True,
                 sample: bool = True, reset_optimizers: bool = True,
                 life_seed: int | None = None) -> dict:
        """装上基因组，过完一生。返回这一生的记录。"""
        self.brain.set_genome(genome, reset_optimizers=reset_optimizers)
        # ★ 每条命用一个确定的随机种子：同一个基因型 + 同一个世界
        #   → 完全可复现（这是 v11 一直坚持的性质，见实验 4）
        torch.manual_seed(int(world_seed) * 31 + 11 if life_seed is None else int(life_seed))

        env = SandboxWorld(self.cfg, seed=int(world_seed), **self.world_kw)
        obs = env.reset()
        self.brain.reset(seed=getattr(env, "episode_seed", None))

        steps = 0
        ks: list[float] = []
        lights: list[float] = []
        lat: list[float] = []          # |个体位置 − 光源位置|：跟踪误差
        norms: list[float] = []
        losses: list[float] = []
        cause = ""
        anorm = 0.0
        while True:
            action, info = self.brain.step(obs, sample=sample, train=train)
            obs, env_done, einfo = env.step(action)
            steps += 1
            ks.append(float(info["k_breath"]))
            lights.append(float(info["light_local"]))
            norms.append(float(info["norm_frac"]))
            lat.append(abs(float(einfo["x"]) - float(einfo["light_center"])))
            if "loss" in info:
                losses.append(float(info["loss"]))
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            anorm += float(np.linalg.norm(action))
            if not info["alive"] or env_done or steps >= self.cfg.life.max_steps:
                cause = info.get("death_cause", "") or "max_steps"
                break

        return {
            "beats": steps,
            "cause": cause,
            "survived": steps >= self.cfg.life.max_steps,
            "mean_k": float(np.mean(ks)) if ks else float("nan"),
            "final_k": float(ks[-1]) if ks else float("nan"),
            "mean_light": float(np.mean(lights)) if lights else 0.0,
            "mean_track_err": float(np.mean(lat)) if lat else 0.0,
            "final_track_err": float(lat[-1]) if lat else 0.0,
            "final_norm": float(norms[-1]) if norms else 0.0,
            "mean_energy": float(np.mean(norms)) if norms else 0.0,
            "mean_action": anorm / max(1, steps),
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        }

    # ── 一代 ────────────────────────────────────────────────────────
    def run_generation(self, gen: int | None = None, train: bool = True) -> dict:
        """跑一代：所有个体各活一辈子，然后选择 + 繁殖。"""
        self.generation = self.generation + 1 if gen is None else int(gen)
        g = self.generation
        pop = len(self.genomes)

        birth = [clone_genome(gm) for gm in self.genomes]
        records: list[dict] = []
        learned: list[dict] = []

        # ★ 拉马克式单谱系基线：父代死前的权重就是此刻大脑里的权重，
        #   所以「装入基因组」是数值上的空操作，可以连 Adam 动量一起保留
        #   （这正是实验 1-10 rebirth(keep_brain=True) 的行为）。
        keep_opt = bool(self.evo.inherit_optimizer
                        and self.evo.inheritance == "lamarckian"
                        and pop == 1)

        for i in range(pop):
            rec = self.run_life(birth[i], self.world_seed(i, g), train=train,
                                sample=True, reset_optimizers=not keep_opt,
                                life_seed=self.seed * 1000003 + g * 101 + i)
            rec["index"] = i
            rec["gen"] = g
            records.append(rec)
            # 拉马克式遗传要用「死前学会的」基因组
            learned.append(self.brain.get_genome() if train else clone_genome(birth[i]))

        fitness = [float(r["beats"]) for r in records]
        parents = select_parents(fitness, self.evo.n_parents, self.rng,
                                 mode=self.evo.selection)

        # ── 繁殖 ────────────────────────────────────────────────────
        source = learned if self.evo.inheritance == "lamarckian" else birth
        if self.evo.inheritance not in ("lamarckian", "darwinian"):
            raise ValueError(f"未知的 inheritance：{self.evo.inheritance!r}")

        elite_n = max(0, min(int(self.evo.elitism), pop))
        order = sorted(range(pop), key=lambda i: fitness[i], reverse=True)
        next_genomes: list[dict] = [clone_genome(source[i]) for i in order[:elite_n]]

        child_seed = self.seed * 7777771 + g * 977
        while len(next_genomes) < pop:
            k = len(next_genomes) - elite_n
            parent = parents[k % len(parents)]
            child = mutate(source[parent], self.evo.mutation_rate,
                           seed=child_seed + k, floor=self.evo.mutation_floor)
            next_genomes.append(child)

        # ── 记录 ────────────────────────────────────────────────────
        dists = [genome_distance(birth[i], next_genomes[i]) for i in range(pop)]
        rec = {
            "gen": g,
            "selection": self.evo.selection,
            "inheritance": self.evo.inheritance,
            "fitness": fitness,
            "best": float(max(fitness)),
            "mean": float(np.mean(fitness)),
            "worst": float(min(fitness)),
            "std": float(np.std(fitness)),
            "n_survived": int(sum(1 for r in records if r["survived"])),
            "parents": [int(p) for p in parents],
            "parent_fitness": [fitness[p] for p in parents],
            "diversity": diversity(birth),
            "offspring_distance": float(np.mean(dists)) if dists else 0.0,
            "mean_k": float(np.nanmean([r["mean_k"] for r in records])),
            "mean_track_err": float(np.mean([r["mean_track_err"] for r in records])),
            "mean_light": float(np.mean([r["mean_light"] for r in records])),
            "mean_action": float(np.mean([r["mean_action"] for r in records])),
            "individuals": records,
        }
        self.history.append(rec)

        # ── 换代 ────────────────────────────────────────────────────
        self.genomes = next_genomes
        return rec

    # ── 留出世界评估（防过拟合 + 跨臂可比）────────────────────────
    def evaluate(self, genome: dict, worlds: list[int],
                 train: bool = False) -> dict:
        """把基因组放到**没见过的世界**里考一遍。

        ★ 为什么必须做这一步
          每个个体的适应度是在它自己的固定世界里测的 —— 那是**训练集**。
          选择会把基因型往「这一个世界里的最优」推，包括利用它的
          出生位置、光周期相位等等。只看训练集适应度会高估进化能力。
          留出世界（不同 seed = 不同出生点 + 不同光相位）才是泛化能力。
        """
        beats: list[int] = []
        k: list[float] = []
        err: list[float] = []
        for w in worlds:
            r = self.run_life(genome, int(w), train=train, sample=True,
                              reset_optimizers=True,
                              life_seed=self.seed * 999983 + int(w))
            beats.append(r["beats"])
            k.append(r["mean_k"])
            err.append(r["mean_track_err"])
        return {
            "beats": beats,
            "mean_beats": float(np.mean(beats)),
            "worst_beats": float(np.min(beats)),
            "mean_k": float(np.nanmean(k)),
            "mean_track_err": float(np.mean(err)),
        }

    def summary(self) -> dict:
        h = self.history
        return {
            "population": int(self.evo.population),
            "generations": len(h),
            "selection": self.evo.selection,
            "inheritance": self.evo.inheritance,
            "mutation_rate": float(self.evo.mutation_rate),
            "fitness_first": h[0]["mean"] if h else float("nan"),
            "fitness_last": h[-1]["mean"] if h else float("nan"),
            "fitness_best_last": h[-1]["best"] if h else float("nan"),
            "diversity_first": h[0]["diversity"] if h else float("nan"),
            "diversity_last": h[-1]["diversity"] if h else float("nan"),
        }
