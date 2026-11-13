"""Brain —— 一个心跳里把「场」和「两趟 Transformer」串成闭环。

★ 逐格 token 版（2026-10-18）
----------------------------
全场只有一个状态：NLSE 场 ψ。
**不做任何派生计算** —— 128×128 每个格子直接一个 token 进 Transformer。

    ┌──────────────────────────────────────────────────────────┐
    │  ① 场自己演化一步（没有任何外界输入）                      │
    │  ② ψ 逐格铺成 token（16,384 个）+ 世界观测 1 个 token       │
    │  ③ 第一趟 Transformer → Δψ（逐格，128×128）                │
    │  ④ ψ ← ψ + Δψ   ★ 世界影响本体的唯一路径（付能量代价）      │
    │  ⑤ 第二趟 Transformer → 动作 3 维                          │
    │  ⑥ 动作 → 世界                                            │
    │  ⑦ 生死：‖ψ‖² < 阈值？                                     │
    └──────────────────────────────────────────────────────────┘

训练：两条完全独立的梯度
    路径 A（第二趟 → 动作 → 存活）：A2C
    路径 B（第一趟 → Δψ → 能量）  ：可微目标，最大化 ΔE
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .field import NLSEField
from .interface import EntityInterface, gaussian_entropy, gaussian_logprob
from . import physics


class Brain:
    """数字生命体：一个 NLSE 场 + 一个两趟 Transformer 接口。"""

    def __init__(self, cfg: Config | None = None, seed: int = 0, device: str | None = None):
        self.cfg = cfg or Config()
        self.cfg.validate()
        self.seed = seed
        dev = device or self.cfg.device
        if not dev:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── 体重与能量容量（物理法则）────────────────────────────────
        # 体重 = 矩阵的乘积 = grid²；能量容量 ∝ 体重。
        self.mass: float = physics.mass_of(self.cfg.field.grid)
        self.energy_cap: float = physics.energy_capacity(self.mass)
        self.cfg.field.norm_target = self.energy_cap    # 覆盖配置里的固定值

        self.field = NLSEField(self.cfg.field, self.device, seed=seed)
        self.interface = self._build_interface()
        self._build_optimizers()
        self._step = 0
        self._last: dict = {}
        self._buf: list[dict] = []          # 在线学习的分块缓冲
        self._lives = 0                     # 已经活过几条命

        # ★ 实验 8：当前保持住的呼吸深度（慢时钟，每 breath_period 拍才更新）
        self._k_breath: float = self._default_k()
        self._n_breath_decisions = 0

    def _default_k(self) -> float:
        """还没做过任何呼吸决策时的呼吸深度。

        ★ 实验 12：动作范围改成 [BREATH_MIN, k*(light)] 之后，范围的下界
          是「几乎不呼吸」。出生第一拍用下界最安全（拿不到光也亏不了多少），
          随后第 1 拍立刻就会做第一次决策覆盖它。
          旧行为（范围全开）仍然用 BREATH_INIT_Z 对应的 1.50。
        """
        if self.cfg.life.breath_limit_useful:
            return physics.BREATH_MIN
        return physics.breath_map_float(physics.BREATH_INIT_Z)

    # ══════════════════════════════════════════════════════════════════
    # 优化器（★ 实验 10：主干和呼吸头分开）
    # ══════════════════════════════════════════════════════════════════

    def _breath_params(self) -> list:
        return [p for n, p in self.interface.named_parameters()
                if n.startswith("breath_head")]

    def _main_params(self) -> list:
        return [p for n, p in self.interface.named_parameters()
                if not n.startswith("breath_head")]

    def _build_optimizers(self) -> None:
        """主干一个优化器、呼吸头一个优化器。

        ★ 为什么呼吸头要单独一个（实验 10）
          呼吸头的参数只有 d×1 + 1 ≈ 65 个，但它要做的位移很大
          （z 从 +1.07 走到 −0.47），而主干的学习率是按几百万参数定的 3e-4。
          共用优化器 = 呼吸头被主干的尺度绑死，一整轮只挪 0.0009，等于没学。

          分开之后：主干 3e-4（稳定），呼吸头 0.02（快速）。
          梯度也分开裁 —— `clip_grad_norm_` 是按全部参数的总范数缩放的，
          主干的梯度会把呼吸头那一点点梯度一起按比例压没。
        """
        lr_b = float(getattr(self.cfg.life, "lr_breath", 0.02))
        self.optimizer = torch.optim.AdamW(
            self._main_params(), lr=self.cfg.life.lr, weight_decay=1e-5
        )
        self.opt_breath = torch.optim.AdamW(
            self._breath_params(), lr=lr_b, weight_decay=0.0
        )

    # ══════════════════════════════════════════════════════════════════
    # 接口构建（★ 实验 9：Transformer 还是 CNN）
    # ══════════════════════════════════════════════════════════════════

    def _build_interface(self):
        """按 cfg.interface.arch 选主干网络。

        ★ 这是实验 9 的**唯一**变量：两种接口的契约（propose / act_all）
          完全一致，所以上面的心跳、训练、生死判定代码一行都不用改。
          这是「单一变量对照」能成立的前提。
        """
        arch = getattr(self.cfg.interface, "arch", "transformer")
        if arch == "cnn":
            from .interface_cnn import CNNInterface as cls
        elif arch == "mlp":
            from .interface_cnn import MLPInterface as cls
        elif arch == "transformer":
            cls = EntityInterface
        else:
            raise ValueError(f"未知的主干架构：{arch!r}（可选 transformer / cnn / mlp）")
        net = cls(
            self.cfg.interface,
            grid=self.cfg.field.grid,
            init_log_std=self.cfg.life.init_log_std,
            world_obs_dim=self.cfg.interface.obs_dim,
        ).to(self.device)
        # ★ 实验 10：呼吸头的探索噪声单独设小（慢时钟标量，不能乱撞）
        v = float(getattr(self.cfg.life, "init_log_std_breath",
                          self.cfg.life.init_log_std))
        p = getattr(net, "breath_log_std", None)
        if p is not None:
            with torch.no_grad():
                p.fill_(v)
        return net

    # ══════════════════════════════════════════════════════════════════
    # 一个心跳
    # ══════════════════════════════════════════════════════════════════

    def heartbeat(self, world_obs: np.ndarray, sample: bool = True) -> tuple[np.ndarray, dict]:
        """完成一个心跳。返回（动作, 诊断字典）。

        参数
        ----------
        world_obs : (obs_dim,) float
            世界给的观测。**这是世界唯一的入口。**
        sample : bool
            True = 采样（训练）；False = 用均值（评估）。
        """
        self._step += 1

        # ① 场自己演化（NLSE + 能量中性噪声；AGC 默认关闭）
        finfo = self.field.tick(use_agc=self.cfg.field.agc_rate > 0)

        # ② ★ 基础代谢：按本地光强给场补能量（「光提供能量、距离越远越少」）
        #    世界观测第 2 位就是光强（live 和 sandbox 的约定一致）
        light_local = float(world_obs[2]) if len(world_obs) > 2 else 0.0
        # ★ 自查修正（2026-10-24）：这一拍代谢用的是**上一拍决定并保持住的** k。
        #   原来 info["k_breath"] 是在本拍末尾（第 ⑥ 步可能刚更新过）才写入的，
        #   于是决策拍上记录的 k 和 meta_gain 对不上（实测差到 290）。
        #   现在单独记一个 k_used = 本拍代谢真正用的 k。
        k_used = float(self._k_breath)
        if self.cfg.field.breath_mode:
            # ★ 实验 8：吸收量 = breath_k × light × k − breath_c × k²
            #   k 是**已经定下并保持住**的呼吸深度（慢时钟，下面第 ⑤ 步才更新）。
            #   c > 0 时这个增益可以为负 —— 吸得越深，付出的代价越大。
            meta_gain = self.field.metabolize(
                light_local, self.cfg.field.breath_k,
                k=self._k_breath, c=self.cfg.field.breath_c,
            )
        else:
            meta_gain = self.field.metabolize(light_local, self.cfg.field.meta_k)

        # ② 世界观测 → tensor（唯一额外的 token）
        world = torch.as_tensor(world_obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)

        # ③ 第一趟：场逐格 token + 世界 token → Δψ
        #    ★ 实验 8：`enable_write=False` 时整个跳过。
        #      写场增益上限为 0 时它本来就什么也不写，跑它纯属浪费一半算力。
        if self.cfg.life.enable_write:
            psi_c = self.field.obs_tensor()                    # (n, n) complex
            delta_psi = self.interface.propose(psi_c, world)   # (1, n, n)

            # ④ 应用（★ 世界影响本体的唯一路径）
            #    写场要付能量代价 —— 否则「一直写满」就是平凡解
            #    ★ 用可微版本：ΔE 的梯度只穿过 Δψ，所以第一趟能直接学「高效注入」
            psi_new, dE_t = self.field.write_trainable(
                delta_psi[0], cost_coef=self.cfg.field.write_cost
            )
            self.field.psi = psi_new.detach()              # 切断图，避免跨步累积
            delta_e = float(dE_t)
            e_after = psi_new.abs().pow(2).sum()           # ★ 写入后能量（带梯度）
            write_abs = float(delta_psi.detach().abs().sum())
        else:
            delta_psi = None
            dE_t = torch.zeros((), device=self.device)
            delta_e = 0.0
            e_after = self.field.psi.abs().pow(2).sum()
            write_abs = 0.0

        # ⑥ 场变了，重新取张量 → 第二趟 → F（力 / 意愿矢量）+ 呼吸深度
        #    ★ 呼吸是慢时钟：只在决策拍才跑呼吸干路（双干路时能省 90% 的算力）
        period = max(1, int(self.cfg.life.breath_period))
        breath_mode = bool(self.cfg.field.breath_mode)
        decide_breath = breath_mode and ((self._step - 1) % period == 0)
        psi_c2 = self.field.obs_tensor()
        mean, value, std, z_breath, std_breath = self.interface.act_all(
            psi_c2, world, need_breath=decide_breath,
        )

        # ★ 采样新的呼吸深度，并保持到下一次决策
        breath_logp = None
        breath_ent = None
        if decide_breath:
            if sample:
                z_s = z_breath + std_breath * torch.randn_like(z_breath)
            else:
                z_s = z_breath
            # ★ 策略梯度的采样动作必须当常数：z_s = z_breath + std·ε 是
            #   z_breath 的函数，不 detach 时 ∂logp/∂z_breath ≡ 0（重参数化
            #   把 score-function 项抵消掉了），呼吸均值就永远不更新。
            breath_logp = gaussian_logprob(z_breath, std_breath, z_s.detach())
            breath_ent = gaussian_entropy(std_breath)
            # ★ 实验 12：动作映射。
            #   旧行为（breath_limit_useful=False）：z → [0.05, 2.00] 全开。
            #   新行为：z → [0.05, k*(light)]，只保留增益单调的那一段。
            #   (k*, 2.0] 是完全冗余的动作区间（吸得更深、拿得更少），
            #   留着它稳态能量目标就有两个解，训练会在分岔点上乱跳。
            #   详见 physics.breath_map_range / physics.breath_useful_max。
            if self.cfg.life.breath_limit_useful:
                k_s = float(physics.breath_map_range(
                    z_s, light_local,
                    mode=self.cfg.life.breath_bound_mode,
                    light_ref=self.cfg.life.breath_light_ref,
                )[0, 0].detach())
            else:
                k_s = float(physics.breath_map(z_s)[0, 0].detach())
            self._k_breath = k_s
            self._n_breath_decisions += 1

        # ⑦ 采样力；然后套物理法则算世界速度
        if sample:
            eps = torch.randn_like(mean)
            raw = mean + std * eps
        else:
            raw = mean
        # ★ 同理：raw 是 mean + std·ε 的函数，必须 detach 后再算 logp，
        #   否则 ∂logp/∂mean ≡ 0，力头的均值永远不更新。
        logp = gaussian_logprob(mean, std, raw.detach())
        ent = gaussian_entropy(std)
        force = raw.clamp(-1.0, 1.0)

        # ★ 力 → 速度：方向 + 意愿 × 质量系数 × 物理法则系数
        velocity, v_norm = physics.force_to_velocity(
            force[0].detach().cpu().numpy(),
            self.field.energy(), self.mass, self.energy_cap,
        )

        # ★ 运动耗能：½ m v² （带标定系数）
        move_cost = physics.move_energy_cost(v_norm, self.mass)
        if move_cost > 0:
            e_now = max(1e-9, self.field.energy())
            keep = max(0.0, 1.0 - move_cost / e_now)
            self.field.psi = self.field.psi * math.sqrt(keep)
        move_cost_real = min(move_cost, self.field.energy() + move_cost)

        # ⑧ 生死（上下两条边：饿死 / 撑死）
        alive = self.field.is_alive(self.cfg.life.death_norm_frac,
                                    self.cfg.life.max_norm_frac)
        cause = self.field.death_cause(self.cfg.life.death_norm_frac,
                                       self.cfg.life.max_norm_frac)

        action = np.asarray(velocity, dtype=np.float64)     # 世界单位/秒

        # ★ 奖励：唯一的信号就是活着（+1 / −1）。
        #   可选再加一项「离生存带中心有多近」的**稠密**奖励：
        #   呼吸深度对存活的影响要经过「能量 → 几十拍后是否越界」才体现，
        #   纯存活奖励的信用分配太远，呼吸头几乎学不动。
        #   ★ 所有实验臂用同一个奖励函数，所以它不会偏向任何一支。
        reward = 1.0 if alive else -1.0
        _coef = float(getattr(self.cfg.life, "energy_reward_coef", 0.0))
        if _coef > 0.0:
            _dev = self.field.norm_frac() - physics.BAND_CENTER
            reward = reward - _coef * _dev * _dev

        self._last = {
            "step": self._step,
            "energy": self.field.energy(),
            "norm_frac": self.field.norm_frac(),
            "delta_e": delta_e,
            "alive": alive,
            "death_cause": cause,
            "value": float(value[0, 0]),
            "force_norm": float(force[0].norm()),
            "speed_scale": physics.speed_scale(self.field.energy(),
                                               self.mass, self.energy_cap),
            "v_norm": v_norm,
            "move_cost": move_cost_real,
            "std": float(std[0].mean()),
            "write_abs": write_abs,
            "write_gain": float(torch.clamp(self.interface.write_gain, 0,
                                            self.cfg.interface.write_gain_max)),
            "meta_gain": meta_gain,
            "light_local": light_local,
            "action": action.copy(),
            "energy_before_tick": finfo["energy_before"],
            "reward": reward,
            "k_breath": float(self._k_breath),
            "k_used": k_used,          # ★ 本拍代谢真正用的 k（日志一致性）
            "breath_decision": bool(decide_breath),
        }
        self._last["_logp"] = logp
        self._last["_value"] = value
        self._last["_entropy"] = ent
        self._last["_dE"] = dE_t
        self._last["_E_after"] = e_after        # ★ 写场头的训练信号（可微）
        self._last["_breath_logp"] = breath_logp
        self._last["_breath_entropy"] = breath_ent
        # ★ 实验 10：把呼吸头原始输出（**带梯度**）和当时的光强存下来，
        #   供可微的「稳态能量 → 生存带中心」目标使用。
        self._last["_z_breath"] = z_breath if decide_breath else None
        self._last["_breath_light"] = light_local if decide_breath else None
        return action, self._last

    # ══════════════════════════════════════════════════════════════════
    # 在线一步（真实场景用，没有"回合"概念）
    # ══════════════════════════════════════════════════════════════════

    def step(self, world_obs: np.ndarray, sample: bool = True,
             train: bool = True) -> tuple[np.ndarray, dict]:
        """在线一步：心跳 + 分块训练。

        **这就是「活着」。** 没有单独的"训练阶段" ——
        梯度更新发生在活着的过程中，每 `chunk` 拍做一次。
        """
        action, info = self.heartbeat(world_obs, sample=sample)

        if train:
            self._buf.append(
                {
                    "logp": info["_logp"],
                    "value": info["_value"],
                    "entropy": info["_entropy"],
                    "dE": info["_dE"],
                    "E_after": info["_E_after"],
                    "breath_logp": info["_breath_logp"],
                    "breath_entropy": info["_breath_entropy"],
                    "z_breath": info["_z_breath"],
                    "breath_light": info["_breath_light"],
                    "move_cost": info["move_cost"],
                    "reward": info["reward"],
                }
            )
            if len(self._buf) >= self.cfg.life.chunk:
                # 这一块的最后一个状态就是"当前状态"，用它的价值做 bootstrap
                boot = float(info["_value"][0, 0].detach())
                info["loss"] = self._a2c_update(self._buf, boot)
                self._buf = []              # ★ 释放计算图（否则显存爆）
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.empty_cache()
        return action, info

    def rebirth(self, seed: int | None = None, keep_brain: bool = True) -> None:
        """死了，重生。

        参数
        ----------
        keep_brain : bool
            **True（默认）= 跨代累积学习。** 身体（场）重置，网络和优化器保留。
                于是第 2 代带着第 1 代学到的东西出生 —— 学习能累积。
            False = 选项 B，连脑子一起清空。每条命都是全新个体，不累积。

        ★ 为什么默认改成 True：
          「脑子清空」会让第 1000 代和第 1 代完全一样 —— 永远学不会任何东西。
          跨代累积是「学习能变成改进」的必要条件。

        ★ 注意：这不是「进化」。这里只有一个个体、没有变异、没有种群选择。
          它是「同一个灵魂反复投胎」，不是「一个物种世代演化」。
        """
        self._lives += 1
        s = self.seed + self._lives * 7919 if seed is None else int(seed)

        if not keep_brain:
            torch.manual_seed(s)
            self.interface = self._build_interface()
            self._build_optimizers()
        # keep_brain=True 时：网络与优化器（含 AdamW 动量）原样保留

        self.field.reset(s)
        self._step = 0
        self._buf = []
        # ★ 呼吸深度回到初始值（新的一条命，还没有做过任何呼吸决策）
        self._k_breath = self._default_k()
        self._n_breath_decisions = 0
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════════════
    # ★ 实验 11：基因组接口（种群进化用）
    # ══════════════════════════════════════════════════════════════════

    def get_genome(self) -> dict:
        """导出当前基因组 = 网络权重的一份拷贝。

        ★ 为什么基因组是网络权重而不是别的东西
          身体（NLSE 场）在每一条命开始时都会被重置成随机初值 ——
          实验 1-5 已经证明「场每次都重新长出来」。
          所以世代之间唯一能传下去的东西就是这张网络。
          它就是这个数字生命的 DNA。
        """
        return {k: v.detach().clone() for k, v in self.interface.state_dict().items()}

    def set_genome(self, genome: dict, reset_optimizers: bool = True) -> None:
        """装入一个基因组（出生时调用）。

        参数
        ----------
        reset_optimizers : bool
            True = 连 Adam 动量一起清空。

            ★ 达尔文式遗传里这个必须是 True：
              Adam 的一阶/二阶动量是「这一生走过的梯度路径」，是**记忆**，
              不是基因。让它跟着基因组传下去，等于把学习成果偷偷继承了。
              拉马克式（单谱系基线）才保留动量 —— 那正是实验 1-10 的行为。
        """
        missing, unexpected = self.interface.load_state_dict(genome, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"基因组与当前网络结构不匹配：缺 {len(missing)} 项、"
                f"多 {len(unexpected)} 项（是不是换了 arch？）"
            )
        if reset_optimizers:
            self._build_optimizers()

    def reset_optimizers(self) -> None:
        """清空两个优化器的状态（新个体出生时用）。"""
        self._build_optimizers()

    # ══════════════════════════════════════════════════════════════════
    # 跨代模型文件（记忆的落盘形式）
    # ══════════════════════════════════════════════════════════════════

    def save_checkpoint(self, path, extra: dict | None = None) -> None:
        """把「这一代学到的东西」写进模型文件。

        ★ 必须连优化器状态一起存 —— 否则 AdamW 的动量每次重生清零，
          学习会来回震荡，跨代曲线看不出趋势。
        """
        import os
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        state = {
            "interface": self.interface.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "opt_breath": self.opt_breath.state_dict(),
            "generation": int(self._lives),
            "step": int(self._step),
            "field_energy": float(self.field.energy()),
        }
        if extra:
            state.update(extra)
        torch.save(state, str(path))

    def load_checkpoint(self, path) -> bool:
        """从模型文件恢复（用于回滚或继续训练）。"""
        import os
        if not os.path.exists(str(path)):
            return False
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        # strict=False：新增的呼吸头参数在旧模型文件里不存在，别因此崩掉
        missing, unexpected = self.interface.load_state_dict(
            state["interface"], strict=False
        )
        if missing:
            print(f"[load_checkpoint] 忽略缺失的参数 {len(missing)} 项"
                  f"（旧模型文件没有它们）")
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, KeyError):
                pass    # 结构变了就只用权重，别崩
        if "opt_breath" in state:
            try:
                self.opt_breath.load_state_dict(state["opt_breath"])
            except (ValueError, KeyError):
                pass
        self._lives = int(state.get("generation", 0))
        return True

    # ══════════════════════════════════════════════════════════════════
    # 回合与训练（沙盒用）
    # ══════════════════════════════════════════════════════════════════

    def reset(self, seed: int | None = None) -> None:
        """新回合：场重置（但网络权重保留）。

        ★ 实验 11 修正：这里必须**清空在线学习的缓冲**。

          在实验 1-10 里没人发现这个问题，因为换回合走的是 `rebirth()`，
          而 `rebirth()` 自己清了 `_buf`。但种群实验每个个体都走
          `set_genome(...) + reset(...)`，于是上一条命留在 `_buf` 里的
          计算图（引用着**旧版本**的参数）会和这一条命的图混在一起 backward，
          直接报 inplace/modified version 错误。

          清理是安全的：`run_episode()` 用的是局部 `buf`，不碰 `_buf`。
        """
        self.field.reset(seed)
        self._step = 0
        self._buf = []
        # ★ 实验 11：新的一条命，呼吸状态也要回到出生值
        #   （`rebirth()` 一直是这么做的；种群实验直接走 `reset()`，
        #     如果不在这里重置，上一条命的呼吸深度会被下一条命继承 ——
        #     那等于让「呼吸记忆」跨代传递，污染了对照。）
        self._k_breath = self._default_k()
        self._n_breath_decisions = 0

    def run_episode(self, env, sample: bool = True, train: bool = False) -> dict:
        """跑一个完整回合。

        参数
        ----------
        env : 需要 reset() -> obs, step(action) -> (obs, done, info)
        train : 是否更新网络参数

        ★ 分块更新：每 `chunk` 拍做一次 backward 并清空缓冲。
          逐格 token 后每拍的计算图约 276 MB，整回合攒着会爆显存（实测 33 GB）。
        """
        obs = env.reset()
        self.reset(seed=getattr(env, "episode_seed", None))

        buf: list[dict] = []
        losses: list[float] = []
        done = False
        total_steps = 0
        total_reward = 0.0

        while not done and total_steps < self.cfg.life.max_steps:
            action, info = self.heartbeat(obs, sample=sample)
            obs, env_done, env_info = env.step(action)
            total_steps += 1

            alive = info["alive"]
            reward = info["reward"]              # ★ 唯一信号：活着（+ 可选的稠密项）
            total_reward += reward
            done = bool(env_done) or (not alive) or (not env_info.get("alive", True))

            if train:
                buf.append(
                    {
                        "logp": info["_logp"],
                        "value": info["_value"],
                        "entropy": info["_entropy"],
                        "dE": info["_dE"],
                        "E_after": info["_E_after"],
                        "breath_logp": info["_breath_logp"],
                        "breath_entropy": info["_breath_entropy"],
                        "z_breath": info["_z_breath"],
                        "breath_light": info["_breath_light"],
                        "move_cost": info["move_cost"],
                        "reward": reward,
                    }
                )
                if len(buf) >= self.cfg.life.chunk or done:
                    # 非终止时用下一状态的价值做 bootstrap
                    boot = 0.0 if done else float(info["_value"][0, 0].detach())
                    losses.append(self._a2c_update(buf, boot))
                    buf = []                     # ★ 释放计算图
                    if torch.cuda.is_available() and self.device.type == "cuda":
                        torch.cuda.empty_cache()

        result = {
            "steps": total_steps,
            "survived": total_steps >= self.cfg.life.max_steps,
            "final_norm_frac": self.field.norm_frac(),
            "total_reward": total_reward,
        }
        if losses:
            result["loss"] = sum(losses) / len(losses)
        return result

    def _a2c_update(self, buf: list[dict], bootstrap: float = 0.0) -> float:
        """A2C 更新（n 步回报 + bootstrap）+ 写入的可微目标。"""
        gamma = self.cfg.life.gamma
        rewards = [b["reward"] for b in buf]
        values = torch.cat([b["value"] for b in buf]).squeeze(-1)

        returns = []
        running = float(bootstrap)
        for r in reversed(rewards):
            running = r + gamma * running
            returns.append(running)
        returns = torch.as_tensor(list(reversed(returns)), dtype=torch.float32,
                                  device=self.device)

        advantages = returns - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logps = torch.cat([b["logp"] for b in buf])
        ents = torch.cat([b["entropy"] for b in buf])

        policy_loss = -(logps * advantages).mean()
        value_loss = nn.functional.mse_loss(values, returns)

        # ★ 实验 8：呼吸头的策略梯度
        #   呼吸只在决策拍上采样（每 breath_period 拍一次），所以它的对数概率
        #   也只在那些拍上有定义 —— 分开算，其余拍不贡献梯度。
        #   优势函数用决策那一拍的优势（和力用同一套 n 步回报）：
        #   「这一拍的联合动作（力 + 呼吸）好在哪」，这是标准的多动作 A2C。
        #
        # ★ 实验 10 修正：`breath_objective="homeostasis"` 时**关掉这一项**。
        #   第一版没关，结果两个信号在打架 —— 实测 D 的 k 被推到下限 0.05、
        #   N1 的 k 被推到上限 2.0，两边都卡在 sigmoid 饱和区出不来。
        #   策略梯度采样噪声大、优势函数是全局的；可微目标信号干净且确定。
        #   要两个都留就用 "both"，但要清楚它们会互相拉扯。
        breath_obj = getattr(self.cfg.life, "breath_objective", "policy")
        use_policy_breath = breath_obj in ("policy", "both")
        idx = [i for i, b in enumerate(buf) if b.get("breath_logp") is not None]
        if idx and use_policy_breath:
            bl = torch.cat([buf[i]["breath_logp"] for i in idx])
            be = torch.cat([buf[i]["breath_entropy"] for i in idx])
            ba = advantages[torch.as_tensor(idx, dtype=torch.long,
                                            device=self.device)].detach()
            breath_loss = -(bl * ba).mean() - self.cfg.life.entropy_coef * be.mean()
        else:
            breath_loss = torch.zeros((), device=self.device)

        # ★ 实验 10：呼吸头的**可微目标**（修「过度呼吸陷阱」）
        #   和写场头当年遇到的问题是同一个：慢时钟/非策略输出拿不到足够清晰的
        #   信用分配。解法也一样 —— 给它一个物理导出的、可微的目标。
        #
        #   呼吸的长期后果是稳态能量 E∞ = gain·(1−γ)/γ；
        #   目标是让 E∞ 落在**生存带的几何中心**（LOWER=0.2, UPPER=3.0 → 1.6）。
        #   ★ 目标只说"停在哪"，没说 k 该多少 —— k 完全取决于光强，
        #     而光强要网络自己从世界观测里读出来。所以这不是把答案喂进去。
        if breath_obj in ("homeostasis", "both") and idx:
            zi = [i for i in idx if buf[i].get("z_breath") is not None]
            if zi:
                zs = torch.cat([buf[i]["z_breath"] for i in zi])
                ls = torch.as_tensor([buf[i]["breath_light"] for i in zi],
                                     dtype=torch.float32, device=self.device)
                # ★ 实验 12：行为端换了映射，目标端必须用**同一个映射**，
                #   否则网络会被推到一个「行为上根本做不到」的 k 上。
                if self.cfg.life.breath_limit_useful:
                    k_t = physics.breath_map_range(
                        zs, ls.unsqueeze(-1),
                        self.cfg.field.breath_k, self.cfg.field.breath_c,
                        mode=self.cfg.life.breath_bound_mode,
                        light_ref=self.cfg.life.breath_light_ref,
                    ).squeeze(-1)
                else:
                    k_t = physics.breath_map(zs).squeeze(-1)
                # ★ 实验 12 第三处修正：能量预算里要算上**运动耗能**。
                #   一个不停走动的心跳，呼吸必须多提供 ≈move_cost 的净增益，
                #   否则目标会把呼吸定得太浅（实测：世界 A 里 k 落到 0.28、
                #   然后因为走动的开销而饿死）。
                _mc = float(np.mean([b["move_cost"] for b in buf])) \
                    if getattr(self.cfg.life, "breath_account_move", True) else 0.0
                breath_homeo = physics.breath_homeostasis_loss(
                    ls, k_t, self.cfg.field.dissipation, self.energy_cap,
                    self.cfg.field.dt,
                    mode=getattr(self.cfg.life, "breath_gamma_mode", "step"),
                    move_cost=_mc,
                ).mean()
            else:
                breath_homeo = torch.zeros((), device=self.device)
        else:
            breath_homeo = torch.zeros((), device=self.device)

        # ★ 写场头的可微目标（策略梯度不覆盖它，必须单独给信号）
        #   "speed"  = 最大化速度能力  → 目标 E ≈ 17480（1.75×E₀），不会自爆
        #   "energy" = 最大化 ΔE       → 旧目标，每代写得更猛、死得更快
        #   "none"   = 不训练写场
        objective = getattr(self.cfg.life, "write_objective", "speed")
        if objective == "speed":
            e_after = torch.stack([b["E_after"] for b in buf])
            cap = physics.speed_scale_tensor(e_after, self.mass,
                                             self.energy_cap)
            write_loss = -cap.mean()
        elif objective == "energy":
            dE = torch.stack([b["dE"] for b in buf])
            write_loss = -dE.mean() / max(1e-9, self.cfg.field.norm_target)
        else:                                   # "none"
            write_loss = torch.zeros((), device=self.device)

        coef_obj = float(getattr(self.cfg.life, "breath_objective_coef", 1.0))
        loss = (policy_loss
                + self.cfg.life.value_coef * value_loss
                - self.cfg.life.entropy_coef * ents.mean()
                + breath_loss
                + coef_obj * breath_homeo
                + self.cfg.life.write_coef * write_loss)

        # ★ 实验 10：两个优化器分开走，梯度也分开裁
        #   （clip_grad_norm_ 按全部参数的总范数缩放，主干会把呼吸头压没）
        self.optimizer.zero_grad(set_to_none=True)
        self.opt_breath.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self._main_params(), 1.0)
        nn.utils.clip_grad_norm_(
            self._breath_params(),
            float(getattr(self.cfg.life, "clip_breath", 5.0)),
        )
        self.optimizer.step()
        self.opt_breath.step()
        return float(loss.detach())

    # ══════════════════════════════════════════════════════════════════
    # 诊断
    # ══════════════════════════════════════════════════════════════════

    def n_params(self) -> int:
        return self.interface.n_params()

    def summary(self) -> dict:
        return {
            "params": self.n_params(),
            "field_energy": self.field.energy(),
            "norm_frac": self.field.norm_frac(),
            "death_frac": self.cfg.life.death_norm_frac,
            "last": {k: v for k, v in self._last.items() if not k.startswith("_")},
        }
