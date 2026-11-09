"""Transformer —— 世界与本体之间唯一的通道（逐格 token 版）。

    ┌───────────────────────────────────────────────────────────┐
    │                     外界（Godot / 沙盒）                    │
    └───────────────────────────────────────────────────────────┘
                    │                          ▲
        世界观测     │                          │   三维速度
                    ▼                          │
    ┌───────────────────────────────────────────────────────────┐
    │  第一趟：ψ 逐格 token(16,384) + 世界 token → Δψ             │
    │                          ↓                                  │
    │                    ψ ← ψ + Δψ                               │
    │                          ↓                                  │
    │  第二趟：新 ψ 逐格 token + 世界 token → 动作（3 维）         │
    └───────────────────────────────────────────────────────────┘
                    │
                    ▼
              本体（NLSE 场 ψ —— 唯一的状态）

★ 输入方式（2026-10-18 修订）
------------------------------
**不做任何派生计算。128×128 每个格子直接一个 token。**

旧版把场拆成 [Re ψ, Im ψ, |ψ|²] 三个通道再切 patch —— 那是多余的计算。
现在每个格子只取它自己的复数 (Re, Im)，经一个线性嵌入进 Transformer。

唯一额外的一个 token 是「世界观测」：
世界必须有个入口，否则场完全封闭，agent 移动就没有任何因果意义。

为什么必须两趟：
    第一趟是「对内」的（我该如何改变我自己）
    第二趟是「对外」的（我该如何作用于世界）
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InterfaceConfig
from . import physics


# ══════════════════════════════════════════════════════════════════════
# 轻量 Transformer 块（用 SDPA，省显存）
# ══════════════════════════════════════════════════════════════════════

class CellBlock(nn.Module):
    """pre-norm 的 Transformer 块，注意力用 F.scaled_dot_product_attention。

    为什么不用 nn.TransformerEncoder：
        16,384 个 token 时，标准实现会物化 n×n 的注意力矩阵（1 GB+）。
        SDPA 会走 flash / memory-efficient 后端，显存是 O(n) 而不是 O(n²)。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        assert self.dh * n_heads == d_model, "d_model 必须被 n_heads 整除"
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]          # (b,h,n,dh)
        a = F.scaled_dot_product_attention(q, k, v)   # 自动选 flash / efficient
        a = a.transpose(1, 2).reshape(b, n, d)
        x = x + self.drop(self.proj(a))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
# 唯一接口
# ══════════════════════════════════════════════════════════════════════

class EntityInterface(nn.Module):
    """数字生命体与世界的唯一接口（逐格 token）。

    契约
    ----
    1. 世界只能通过 `world_obs` 进入系统，且必须经过本模块的权重；
    2. 本体（ψ）只能通过 `propose()` 被修改，且只能通过 `act()` 影响世界；
    3. 本模块之外，没有任何代码可以读写场。
    """

    def __init__(self, cfg: InterfaceConfig, grid: int,
                 init_log_std: float = -1.0, world_obs_dim: int = 8):
        super().__init__()
        self.cfg = cfg
        self.grid = int(grid)
        self.n_cells = self.grid * self.grid        # 16,384
        d = cfg.d_model

        # ── 逐格嵌入：每个格子只取它自己的 (Re, Im) ──────────────────
        self.cell_embed = nn.Linear(2, d)

        # ── 世界观测：唯一一个额外的 token ────────────────────────────
        self.world_obs_dim = int(world_obs_dim)
        self.world_proj = nn.Linear(self.world_obs_dim, d)

        # ── 位置编码（每个格子一个，可学习）──────────────────────────
        self.pos = nn.Parameter(torch.randn(1, self.n_cells + 1, d) * 0.02)

        # ── 第一趟：外界 → 对内的自我修改 ────────────────────────────
        self.pass1 = nn.ModuleList(
            [CellBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers_pass1)]
        )
        self.write_head = nn.Linear(d, 1)
        self.write_gain = nn.Parameter(torch.tensor(0.06))
        """★ 决定生死的一路：直接叠加到 ψ 的实部。有界（tanh × gain）。"""

        # ── 第二趟：修改后的自己 → 动作 ──────────────────────────────
        self.pass2 = nn.ModuleList(
            [CellBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers_pass2)]
        )
        self.action_head = nn.Linear(d, cfg.action_dim)
        self.value_head = nn.Linear(d, 1)
        self.action_gain = nn.Parameter(torch.tensor(1.0))

        # ★ 实验 10：场池化 + 世界 token 分开，再过一层投影
        #   （旧写法把世界 token 混进 16,385 个格子里一起平均，光强信号被稀释没了）
        if getattr(cfg, "pool_world_separate", True):
            self.pool_proj = nn.Linear(2 * d, d)
        else:
            self.pool_proj = None

        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), float(init_log_std)))

        # ── ★ 实验 8：呼吸干路 + 呼吸头 ──────────────────────────────
        #   split_pass2=False → 呼吸头挂在 pass2 上（共用一套表示）
        #   split_pass2=True  → 呼吸头挂在独立干路 pass2_meta 上
        self.breath_head = nn.Linear(d, cfg.breath_dim)
        """输出 z（实数），再经 sigmoid 映射到呼吸深度 k ∈ [0.05, 2.0]。

        ★ 偏置初始化为 physics.BREATH_INIT_Z（≈ −1.90 → k ≈ 0.30）。
          不用 0：z=0 对应 k≈1.0，而 k=1 在黑暗里每拍亏 BREATH_C=200，
          38 拍就死，第一条命学不到任何东西。"""

        nn.init.constant_(self.breath_head.bias, physics.BREATH_INIT_Z)
        nn.init.normal_(self.breath_head.weight, std=0.02)

        self.breath_log_std = nn.Parameter(
            torch.full((cfg.breath_dim,), float(init_log_std))
        )

        if cfg.split_pass2:
            self.pass2_meta = nn.ModuleList(
                [CellBlock(d, cfg.n_heads, cfg.dropout)
                 for _ in range(cfg.n_layers_meta)]
            )
            """★ 独立的呼吸干路。

            两个头（力 / 呼吸）不再共用第二趟的表示 ——
            移动的梯度不再把呼吸的表示需求写进同一套特征里，反之亦然。
            **参数量与单干路完全相同**：两条干路的层数之和 = 单干路的层数。"""
        else:
            self.pass2_meta = None

        if cfg.meta_pool > 1:
            self.cell_embed_meta = nn.Linear(3, d)
            """下采样干路的 token 嵌入：[均值 Re, 均值 Im, RMS 幅度]。

            ★ 为什么单独一个嵌入：4×4 平均池化会把 |ψ| 的幅度压掉
              （相干的会保留、随机的会抵消），所以额外送一路 RMS 幅度，
              保证「场有多满」这个信息不被池化吃掉。
              只多 3×d = 192 个参数（占总量的 0.015%）。"""
        else:
            self.cell_embed_meta = None

    # ── 内部：把场逐格铺成 token ────────────────────────────────────
    def _tokens(self, psi: torch.Tensor, world_obs: torch.Tensor) -> torch.Tensor:
        """psi (B, n, n) complex 或 (B, 2, n, n) real；world_obs (B, w)。"""
        if psi.dim() == 2:
            psi = psi.unsqueeze(0)
        if world_obs.dim() == 1:
            world_obs = world_obs.unsqueeze(0)
        b, n = psi.shape[0], self.grid

        if torch.is_complex(psi):
            x = torch.stack([psi.real, psi.imag], dim=-1)      # (B, n, n, 2)
        else:
            # 已经是 (B, 2, n, n) 的实部/虚部
            x = psi.permute(0, 2, 3, 1)                        # (B, n, n, 2)
        x = x.reshape(b, self.n_cells, 2)
        cells = self.cell_embed(x) + self.pos[:, 1:, :]         # (B, cells, d)

        w = self.world_proj(world_obs).unsqueeze(1) + self.pos[:, :1, :]
        return torch.cat([w, cells], dim=1)                     # (B, 1+cells, d)

    def _tokens_meta(self, psi: torch.Tensor, world_obs: torch.Tensor) -> torch.Tensor:
        """呼吸干路的 token 序列：**下采样后的场** + 世界 token。

        ★ 为什么可以下采样（meta_pool=4 → 1024 个格子 token，算力降到 1/16）
          呼吸深度 k 只依赖「这里有多亮、我的场有多满」这两个全局量，
          完全不需要 16,384 个格子的细节。用满分辨率去算一个 1 维的数
          是纯浪费 —— **但单干路做不到**，因为力和呼吸必须共用同一份
          token 序列。必须先拆成两条干路，才允许两条干路用不同的分辨率。

          这正是「多 Transformer」在理论上有优势的地方，实验 8 就是去检验它。

        ★ 不做位置编码：池化之后位置信息已经很弱，而且呼吸决策不需要
          「光在哪个方向」，只需要「有多亮」。省掉 pos 也让两条干路的
          参数量严格相等。
        """
        if psi.dim() == 2:
            psi = psi.unsqueeze(0)
        if world_obs.dim() == 1:
            world_obs = world_obs.unsqueeze(0)
        b, n = psi.shape[0], self.grid
        if torch.is_complex(psi):
            re, im = psi.real, psi.imag
        else:
            re, im = psi[:, 0], psi[:, 1]

        p = int(self.cfg.meta_pool)
        nc = n // p
        re_b = re.reshape(b, nc, p, nc, p).mean(dim=(2, 4))
        im_b = im.reshape(b, nc, p, nc, p).mean(dim=(2, 4))
        rms = (re * re + im * im).reshape(b, nc, p, nc, p).mean(dim=(2, 4))
        rms = rms.clamp_min(0.0).sqrt()

        feat = torch.stack([re_b, im_b, rms], dim=-1).reshape(b, nc * nc, 3)
        cells = self.cell_embed_meta(feat)                      # 无位置编码
        w = self.world_proj(world_obs).unsqueeze(1)
        return torch.cat([w, cells], dim=1)                     # (B, 1+nc², d)

    @staticmethod
    def _run(blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for blk in blocks:
            x = blk(x)
        return x

    # ══════════════════════════════════════════════════════════════════
    # 第一趟：世界 → 对本体的修改
    # ══════════════════════════════════════════════════════════════════

    def propose(self, psi: torch.Tensor, world_obs: torch.Tensor) -> torch.Tensor:
        """输出 Δψ 的实部增量，形状 (B, n, n)。

        **这是世界影响本体的唯一方式。**
        每个格子独立输出一个值（逐格 token → 逐格线头），
        经 tanh 压缩并以可学习 gain 缩放，保证幅度有界。
        """
        h = self._run(self.pass1, self._tokens(psi, world_obs))
        cells = h[:, 1:, :]                                     # 丢掉世界 token
        d = self.write_head(cells).squeeze(-1)                  # (B, cells)
        d = d.view(-1, self.grid, self.grid)
        gain = torch.clamp(self.write_gain, 0.0, self.cfg.write_gain_max)
        return torch.tanh(d) * gain

    # ══════════════════════════════════════════════════════════════════
    # 第二趟：修改后的自己 → 对外的动作
    # ══════════════════════════════════════════════════════════════════

    def act(self, psi: torch.Tensor, world_obs: torch.Tensor):
        """输出动作均值、价值、标准差（旧接口，不含呼吸）。"""
        mean, value, std, _, _ = self.act_all(psi, world_obs)
        return mean, value, std

    def act_all(self, psi: torch.Tensor, world_obs: torch.Tensor,
                need_breath: bool = True):
        """输出动作均值、价值、标准差 + 呼吸深度。

        ★ 实验 8 的三种布局都走这一个函数 —— 这是控制变量的关键：
          除了「干路拆不拆」「呼吸干路分辨率多少」，其它一切相同。

          臂 A（单干路 2 层）  : split=False, n_layers_pass2=2
          臂 B（双干路 1+1，同分辨率）: split=True,  n_layers_pass2=1, n_layers_meta=1, meta_pool=1
          臂 C（双干路 1+1，异分辨率）: split=True,  n_layers_pass2=1, n_layers_meta=1, meta_pool=4
          臂 D（单干路 1 层）  : split=False, n_layers_pass2=1

        ★ need_breath=False 时跳过呼吸干路（呼吸是慢时钟，10 拍才决策一次）。
          这不只是省时间 —— **它本身就是拆分带来的算力收益**：
          单干路每拍都必须跑完整干路（力要用），双干路可以让呼吸干路 90%
          的拍数待机。

        返回
        -------
        mean     : (B, action_dim)  ∈ [-1, 1]
        value    : (B, 1)
        std      : (B, action_dim)
        z_breath : (B, breath_dim)  呼吸深度的**采样前均值**（未经 sigmoid）
                   未计算呼吸时为 None
        std_breath : (B, breath_dim) 同上
        """
        tok = self._tokens(psi, world_obs)
        h = self._run(self.pass2, tok)
        pooled = self._pool(h)                                  # 池化（见下）
        gain = torch.clamp(self.action_gain, 0.0, 1.0)
        mean = torch.tanh(self.action_head(pooled)) * gain
        value = self.value_head(pooled)
        std = torch.exp(self.log_std).expand_as(mean)

        if not need_breath:
            return mean, value, std, None, None

        if self.cfg.split_pass2:
            if self.cfg.meta_pool > 1:
                tok_m = self._tokens_meta(psi, world_obs)
            else:
                tok_m = tok
            hm = self._run(self.pass2_meta, tok_m)
        else:
            hm = h                    # ★ 单干路：两个头共用同一套表示

        z_breath = self.breath_head(self._pool(hm))              # (B, 1)
        std_breath = torch.exp(self.breath_log_std).expand_as(z_breath)
        return mean, value, std, z_breath, std_breath

    # ── ★ 池化方式（实验 10 修正）────────────────────────────────────
    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """把 (B, 1+16384, d) 压成 (B, d)。

        ★ 旧写法的问题（实验 10 发现）
        ------------------------------
        原来是 `h.mean(dim=1)` —— **把世界 token 和 16,384 个格子 token 一起平均**。
        世界 token 是唯一携带「这里多亮、我在哪」的那个 token，
        但它只占 1/16,385 的权重 —— **光强信号被稀释成 0.006%**。

        跨架构实测（实验 9）里 Transformer 的动作头学不会「往亮处走」，
        原因就在这里：N1 那个 35k 的 CNN 是
        `cat(pool(场), 世界观测)` —— 世界观测是**显式拼进去**的，一点不稀释。
        所以 CNN 三代入就能学会跟光走，Transformer 学不会。

        ★ 修法：场单独池化，世界 token 单独取出，再拼起来过一层投影。
          多 2×d×d = 8,256 个参数，占 0.66%。
        """
        if getattr(self.cfg, "pool_world_separate", True):
            return self.pool_proj(torch.cat([h[:, 1:, :].mean(dim=1),
                                             h[:, 0, :]], dim=-1))
        return h.mean(dim=1)

    # ══════════════════════════════════════════════════════════════════
    # 工具
    # ══════════════════════════════════════════════════════════════════

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (f"d_model={self.cfg.d_model}, tokens={self.n_cells + 1}, "
                f"params={self.n_params()}")


def gaussian_logprob(mean: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """对角高斯的对数概率，对动作维度求和。"""
    var = std.pow(2).clamp_min(1e-8)
    logp = -0.5 * ((action - mean).pow(2) / var + 2.0 * torch.log(std) + math.log(2 * math.pi))
    return logp.sum(dim=-1)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    """对角高斯的熵（对维度求和）。"""
    return (0.5 * (2.0 * torch.log(std) + math.log(2 * math.pi * math.e))).sum(dim=-1)
