"""CNN 接口 —— 实验 9：把 Transformer 换成卷积网络（架构族对照）。

★ 为什么要做这个对照
--------------------
实验 8 的结论是「多 Transformer 架构不适用」。但那个结论有一个前提没被检验：
**主干网络为什么非得是 Transformer？**

检查参数量就会发现一个刺眼的事实（当前 Transformer 接口）：

    pos（可学习绝对位置编码）  16,385 × 64 = 1,048,640   ← 占 83.9%
    其余全部（4 层 CellBlock + 各头）        =   201,100

**这个「Transformer」里有 84% 的参数是一张绝对位置查表。**

而我们的任务本身是**平移协变**的：
    · NLSE 的动力学不区分「哪个像素」（空间各向同性、无绝对原点）
    · 个体输出的是**全局量**（3 维力、1 个呼吸标量），不是逐格输出
    · 位置信息已经由世界观测 token 单独提供了（obs[0] = 归一化 x）

所以那张 1M 参数的位置表，学的是一份**用不上**的先验。
它还可能有害：位置表让模型可以「记住位置」而不是「学场的动力学」。

CNN 恰好是反过来的：
    · 3×3 卷积的局部感受野 ↔ NLSE 的拉普拉斯邻域耦合（物理上对得上）
    · 平移等变 —— 不需要位置编码
    · O(n) 而不是 O(n²)
    · 下采样天然给出「低分辨率呼吸分支」，几乎免费

★ 本模块提供的接口和 EntityInterface 完全一致
    propose() / act() / act_all() / n_params()
    所以 brain.py 只按 cfg.interface.arch 选一个类，其余代码一行不用改。
    这是实验 9 能作为「单一变量」对照的前提。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InterfaceConfig
from . import physics


# ══════════════════════════════════════════════════════════════════════
# 卷积块
# ══════════════════════════════════════════════════════════════════════

def _groups(c: int, want: int = 8) -> int:
    """挑一个能整除 c 的组数给 GroupNorm（不依赖 batch size，小 batch 也稳）。"""
    for g in range(min(want, c), 0, -1):
        if c % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """3×3 卷积 + GroupNorm + GELU + 残差。

    ★ 只用 3×3：NLSE 里每个格子只和**最近邻**耦合（拉普拉斯项），
      所以 3×3 的感受野在物理上就是对的尺度。
      两层 3×3 叠起来 = 5×5 感受野，每下采样一次视野翻倍。
    """

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(_groups(cout), cout)
        self.act = nn.GELU()
        if stride != 1 or cin != cout:
            self.skip = nn.Conv2d(cin, cout, 1, stride=stride)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x if self.skip is None else self.skip(x)
        return self.act(self.norm(self.conv(x)) + s)


# ══════════════════════════════════════════════════════════════════════
# CNN 接口
# ══════════════════════════════════════════════════════════════════════

class CNNInterface(nn.Module):
    """卷积主干 + 全局池化 + 两个头。契约与 EntityInterface 相同。"""

    def __init__(self, cfg: InterfaceConfig, grid: int,
                 init_log_std: float = -1.0, world_obs_dim: int = 8):
        super().__init__()
        self.cfg = cfg
        self.grid = int(grid)
        self.n_cells = self.grid * self.grid
        d = cfg.d_model
        ch = tuple(int(c) for c in cfg.cnn_channels)
        nb = max(1, int(cfg.cnn_blocks))
        self.world_obs_dim = int(world_obs_dim)

        # ── 主干：每级 stride=2，最后一级再做 nb−1 个同分辨率块 ──────
        layers: list[nn.Module] = []
        cin = 2
        for i, c in enumerate(ch):
            layers.append(ConvBlock(cin, c, stride=2))
            for _ in range(nb - 1):
                layers.append(ConvBlock(c, c))
            cin = c
        self.encoder = nn.Sequential(*layers)
        self.out_ch = cin
        self.downsample = 2 ** len(ch)
        self.final_size = max(1, self.grid // self.downsample)

        # ── 池化特征 + 世界观测 → 公共干路 ──────────────────────────
        feat_dim = 2 * cin + self.world_obs_dim      # 全局均值 + 全局最大 + 世界观测
        self.trunk = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU())

        # ── 第一趟：逐格 Δψ（需要上采样回原分辨率）─────────────────
        self.write_head = nn.Conv2d(cin, 1, 1)
        self.write_gain = nn.Parameter(torch.tensor(0.06))

        # ── 动作 / 价值 / 呼吸 ──────────────────────────────────────
        self.action_head = nn.Linear(d, cfg.action_dim)
        self.value_head = nn.Linear(d, 1)
        self.action_gain = nn.Parameter(torch.tensor(1.0))
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), float(init_log_std)))

        self.breath_head = nn.Linear(d, cfg.breath_dim)
        nn.init.constant_(self.breath_head.bias, physics.BREATH_INIT_Z)
        nn.init.normal_(self.breath_head.weight, std=0.02)
        self.breath_log_std = nn.Parameter(
            torch.full((cfg.breath_dim,), float(init_log_std))
        )

        # ── ★ 独立呼吸分支（split_pass2=True）──────────────────────
        #   呼吸只需要「这里多亮」这一个标量，所以给它一条**极低分辨率**的
        #   独立卷积分支：两次 stride=4 → 8×8。算力只有主干的 ~1/16，
        #   而且因为它本来就是卷积，不存在 Transformer 那种「切断迁移就学不动」
        #   的问题（梯度只穿过几层卷积，路径短得多）。
        if cfg.split_pass2:
            mc = int(cfg.cnn_breath_channels)
            self.breath_encoder = nn.Sequential(
                ConvBlock(2, mc, stride=4),
                ConvBlock(mc, mc, stride=4),
            )
            self.breath_trunk = nn.Sequential(
                nn.Linear(2 * mc + self.world_obs_dim, d), nn.GELU()
            )
            self._breath_downsample = 16
        else:
            self.breath_encoder = None
            self.breath_trunk = None
            self._breath_downsample = 1

    # ── 内部 ────────────────────────────────────────────────────────
    def _field(self, psi: torch.Tensor) -> torch.Tensor:
        """(n,n) complex / (B,n,n) complex / (B,2,n,n) real → (B,2,n,n) real。"""
        if psi.dim() == 2:
            psi = psi.unsqueeze(0)
        if torch.is_complex(psi):
            return torch.stack([psi.real, psi.imag], dim=1)
        if psi.dim() == 3:                       # (B,n,n) 实数
            psi = psi.unsqueeze(1)
        return psi

    def _world(self, world_obs: torch.Tensor, b: int) -> torch.Tensor:
        w = torch.as_tensor(world_obs, dtype=torch.float32, device=self._dev())
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if w.shape[0] == 1 and b > 1:
            w = w.expand(b, -1)
        return w

    def _dev(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _pool(h: torch.Tensor) -> torch.Tensor:
        """全局均值 + 全局最大，拼起来。均值给「平均有多满」，最大给「哪里最强」。"""
        return torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)

    # ── 第一趟：世界 → 对本体的修改 ─────────────────────────────────
    def propose(self, psi: torch.Tensor, world_obs: torch.Tensor) -> torch.Tensor:
        """输出 Δψ 的实部增量，形状 (B, n, n)。与 EntityInterface 的契约一致。"""
        x = self._field(psi)
        h = self.encoder(x)
        d = self.write_head(h)                                  # (B,1,hz,hz)
        if d.shape[-1] != self.grid:
            d = F.interpolate(d, size=(self.grid, self.grid),
                              mode="bilinear", align_corners=False)
        gain = torch.clamp(self.write_gain, 0.0, self.cfg.write_gain_max)
        return torch.tanh(d).squeeze(1) * gain

    # ── 第二趟：修改后的自己 → 动作 + 呼吸 ─────────────────────────
    def act(self, psi: torch.Tensor, world_obs: torch.Tensor):
        mean, value, std, _, _ = self.act_all(psi, world_obs)
        return mean, value, std

    def act_all(self, psi: torch.Tensor, world_obs: torch.Tensor,
                need_breath: bool = True):
        x = self._field(psi)
        b = x.shape[0]
        w = self._world(world_obs, b)

        h = self.encoder(x)
        z = self.trunk(torch.cat([self._pool(h), w], dim=1))

        gain = torch.clamp(self.action_gain, 0.0, 1.0)
        mean = torch.tanh(self.action_head(z)) * gain
        value = self.value_head(z)
        std = torch.exp(self.log_std).expand_as(mean)

        if not need_breath:
            return mean, value, std, None, None

        if self.breath_encoder is not None:
            hb = self.breath_encoder(x)
            zb = self.breath_trunk(torch.cat([self._pool(hb), w], dim=1))
        else:
            zb = z

        z_breath = self.breath_head(zb)
        std_breath = torch.exp(self.breath_log_std).expand_as(z_breath)
        return mean, value, std, z_breath, std_breath

    # ── 工具 ────────────────────────────────────────────────────────
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (f"arch=cnn, channels={tuple(self.cfg.cnn_channels)}, "
                f"blocks={self.cfg.cnn_blocks}, params={self.n_params()}")


# ══════════════════════════════════════════════════════════════════════
# ★ 极端对照：纯全局 MLP（完全不做空间处理）
# ══════════════════════════════════════════════════════════════════════

class MLPInterface(nn.Module):
    """只吃「全局标量」的接口 —— 用来检验「到底需不需要空间结构」。

    ★ 为什么要这一支臂
    ------------------
    这个任务真正需要的可能只有几个**全局量**：
        · 我还有多少能量？   ← 场的平均幅度（mean|ψ|² = E/m）
        · 这里有多亮？       ← 世界观测第 2 位
        · 刚才光在变亮还是变暗？ ← 世界观测第 3 位
        · 上一拍往哪走了？   ← 世界观测第 4~6 位

    如果这么一支 1 万参数的小 MLP 就能活下来，那就说明：
    **16,384 个格子 token 和 1M 参数的位置表，全都是在处理一个
      用不到空间细节的决策。** 那才是「架构选错了」最直接的证据。

    ★ 契约与另外两个接口一致：
        propose() 写场需要逐格输出，MLP 做不到（返回全零）——
        本实验写场本来就关闭，不影响对照。
    """

    def __init__(self, cfg: InterfaceConfig, grid: int,
                 init_log_std: float = -1.0, world_obs_dim: int = 8):
        super().__init__()
        self.cfg = cfg
        self.grid = int(grid)
        self.n_cells = self.grid * self.grid
        d = cfg.d_model
        self.world_obs_dim = int(world_obs_dim)

        # 4 个场统计量 + 世界观测
        feat_dim = 4 + self.world_obs_dim
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(),
        )
        self.write_gain = nn.Parameter(torch.tensor(0.0))
        self.action_head = nn.Linear(d, cfg.action_dim)
        self.value_head = nn.Linear(d, 1)
        self.action_gain = nn.Parameter(torch.tensor(1.0))
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), float(init_log_std)))

        self.breath_head = nn.Linear(d, cfg.breath_dim)
        nn.init.constant_(self.breath_head.bias, physics.BREATH_INIT_Z)
        nn.init.normal_(self.breath_head.weight, std=0.02)
        self.breath_log_std = nn.Parameter(
            torch.full((cfg.breath_dim,), float(init_log_std))
        )

    def _stats(self, psi: torch.Tensor) -> tuple[torch.Tensor, int]:
        if psi.dim() == 2:
            psi = psi.unsqueeze(0)
        if torch.is_complex(psi):
            re, im = psi.real, psi.imag
        else:
            re, im = psi[:, 0], psi[:, 1]
        b = re.shape[0]
        a2 = re * re + im * im
        return torch.stack([
            re.mean(dim=(1, 2)),
            im.mean(dim=(1, 2)),
            a2.mean(dim=(1, 2)),          # = E / m  → 这就是「我还有多少能量」
            a2.mean(dim=(1, 2)).clamp_min(0).sqrt(),
        ], dim=1), b

    def _feat(self, psi, world_obs):
        st, b = self._stats(psi)
        w = torch.as_tensor(world_obs, dtype=torch.float32,
                            device=next(self.parameters()).device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if w.shape[0] == 1 and b > 1:
            w = w.expand(b, -1)
        return self.trunk(torch.cat([st, w], dim=1))

    def propose(self, psi, world_obs):
        """MLP 没有逐格输出能力 —— 返回全零（本实验写场关闭，不影响对照）。"""
        if psi.dim() == 2:
            psi = psi.unsqueeze(0)
        return torch.zeros(psi.shape[0], self.grid, self.grid,
                           device=psi.device, dtype=torch.float32)

    def act(self, psi, world_obs):
        mean, value, std, _, _ = self.act_all(psi, world_obs)
        return mean, value, std

    def act_all(self, psi, world_obs, need_breath: bool = True):
        z = self._feat(psi, world_obs)
        gain = torch.clamp(self.action_gain, 0.0, 1.0)
        mean = torch.tanh(self.action_head(z)) * gain
        value = self.value_head(z)
        std = torch.exp(self.log_std).expand_as(mean)
        if not need_breath:
            return mean, value, std, None, None
        z_breath = self.breath_head(z)
        std_breath = torch.exp(self.breath_log_std).expand_as(z_breath)
        return mean, value, std, z_breath, std_breath

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return f"arch=mlp, params={self.n_params()}"
