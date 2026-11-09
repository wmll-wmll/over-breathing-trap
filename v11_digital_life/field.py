"""NLSE 场 —— 数字生命体的本体。

这是整个系统里唯一"自己算能量"的东西。

物理：
    i ħ ∂ψ/∂t = −(ħ²/2m)∇²ψ + g|ψ|²ψ − i ħ γ ψ

    动能项   → 场的扩散（FFT 求解）
    非线性项 → 自相互作用，能自发形成局域结构
    耗散项   → 每步流失能量（这就是"活着"的代价）

能量：
    E = Σ|ψ|²

    **这个值完全来自场本身。没有任何外部代码可以给它赋值。**
    唯一的改变途径是 `write()`，而调用 `write()` 的只有 Transformer。

求解方法：分步傅里叶法（split-step Fourier）
    1. 势能半步   ψ *= exp(−i·g|ψ|²·dt/2ħ)
    2. 动能一步   ψ = IFFT( FFT(ψ) · exp(−i·ħk²·dt/2m) )
    3. 势能半步   ψ *= exp(−i·g|ψ|²·dt/2ħ)
    4. 耗散       ψ *= exp(−γ·dt)
    5. 持久噪声   ψ += η·(随机复数)，归一化后加
    6. 弱 AGC     把能量缓慢拉回目标（安全网，不是主要机制）
"""

from __future__ import annotations

import math

import torch

from .config import FieldConfig


class NLSEField:
    """一个自治的耗散 NLSE 场。

    契约（不要违反）：
      * `energy()` 的值只由场自己决定；
      * 外部只能通过 `write()` 影响它；
      * `step()` 不接受任何"外部能量"参数。
    """

    def __init__(self, cfg: FieldConfig, device: torch.device, seed: int = 0):
        self.cfg = cfg
        self.device = device
        self.n = int(cfg.grid)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        # 初始场：低幅随机复数，然后归一化到目标能量
        real = torch.randn(self.n, self.n, generator=gen)
        imag = torch.randn(self.n, self.n, generator=gen)
        psi = torch.complex(real, imag).to(torch.complex64).to(device)
        psi = psi / psi.abs().pow(2).sum().sqrt() * math.sqrt(cfg.norm_target)
        self.psi: torch.Tensor = psi

        # ── 动能相位（FFT 频域，只算一次）────────────────────────────
        freq = torch.fft.fftfreq(self.n, d=1.0)          # cycles/sample
        kx = 2.0 * math.pi * freq
        k2 = kx[:, None] ** 2 + kx[None, :] ** 2
        coeff = cfg.hbar / (2.0 * cfg.mass) * cfg.dt
        self._kinetic_phase = torch.exp(-1j * coeff * k2).to(torch.complex64).to(device)

        # ── 耗散系数 ────────────────────────────────────────────────
        # 注意是 exp(-γ·dt)，不是 (1-γ·dt)，保证永远为正
        self._dissipation = float(math.exp(-cfg.dissipation * cfg.dt))

        self._rng = torch.Generator(device="cpu").manual_seed(seed + 12345)
        self._step_count = 0

    # ══════════════════════════════════════════════════════════════════
    # 演化
    # ══════════════════════════════════════════════════════════════════

    def step(self) -> None:
        """让场自己演化一步。**不接受任何外界输入。**"""
        cfg = self.cfg
        psi = self.psi

        # 1+3. 势能半步（前后各一次 = 一次完整的势能演化）
        half = cfg.dt * 0.5 / cfg.hbar
        phase = torch.exp(-1j * (cfg.non_linearity * psi.abs().pow(2)) * half)
        psi = psi * phase.to(torch.complex64)
        psi = psi * phase.to(torch.complex64)   # 用同一相位做前后两步

        # 2. 动能一步（FFT）
        psi = torch.fft.ifft2(torch.fft.fft2(psi) * self._kinetic_phase)

        # 4. 耗散 —— 这一步让"维持能量"成为真实任务
        psi = psi * self._dissipation

        self.psi = psi

    def add_noise(self) -> None:
        """持久噪声：让场永不静止，但**不改变总能量**。

        没有它，场会收敛到一个静态解，世界模型会变得平凡，
        生命体就失去了"动态生存"的意义。

        注意：加完噪声后会把能量归一化回原值。
        否则噪声会持续泵入能量，耗散就失去意义了 —— 那样"求生"就不成立。
        """
        if self.cfg.noise_eta <= 0:
            return
        amp = self.cfg.noise_eta * math.sqrt(self.cfg.norm_target) / self.n
        real = torch.randn(self.n, self.n, generator=self._rng)
        imag = torch.randn(self.n, self.n, generator=self._rng)
        noise = torch.complex(real, imag).to(torch.complex64).to(self.device) * amp

        e_before = self.energy()
        psi = self.psi + noise
        e_after = float(psi.abs().pow(2).sum())
        if e_after > 1e-12:
            psi = psi * math.sqrt(e_before / e_after)
        self.psi = psi

    def metabolize(self, light: float, k_meta: float,
                   k: float = 1.0, c: float = 0.0) -> float:
        """基础代谢：按**本地光强**给场补充能量。

        ★ 这是「光提供能量、距离越远获取越少」的物理实现。

        物理意义
        --------
        场不是被外力"充能"的，而是像光合作用/代谢一样，
        **自己**从所在位置的辐射里吸取能量：

            dE_meta = k_meta × light × k  −  c × k²
                      └─ 线性项：吸得越深拿得越多
                                     └─ 二次项：吸得越深效率越低

        ★ 实验 8 新增的 k 和 c
            旧行为（实验 1-7）：c = 0、k ≡ 1 → dE = k_meta × light
            新行为（实验 8）  ：k 由网络每 breath_period 拍决定，
                                于是每一点光强都有一个内部最优深度
                                k* = k_meta × light / (2c)。

        ★ 注意 c > 0 时增益**可以为负**（呼吸本身要付出代价）——
          即使完全没有光，吸得越深亏得越多。这是有意的：
          「呼吸」不是免费动作，它和「移动」一样是一个要学的决策。

        参数
        ----------
        light : float
            本地光强（Godot / 沙盒给的世界观测第 2 位）。
        k_meta : float
            代谢系数（每单位光强每拍换多少能量）。
        k : float
            呼吸深度（1.0 = 旧行为）。仅 c > 0 时有意义。
        c : float
            二次代价系数。0 = 关闭呼吸深度这一维。

        返回
        -------
        float
            本拍的名义能量变化（可为负）。
        """
        if k_meta <= 0.0:
            return 0.0
        gain = float(k_meta) * float(light) * float(k) - float(c) * float(k) ** 2
        e_now = self.energy()
        if e_now <= 1e-12:
            return 0.0
        e_new = e_now + gain
        if e_new <= 1e-9:
            e_new = 1e-9
        # 按比例缩放，保持场的形状不变（只改幅度）
        self.psi = self.psi * math.sqrt(e_new / e_now)
        return gain

    def auto_gain(self) -> float:
        """弱 AGC：把能量缓慢拉回目标。
        返回本步的增益因子。

        注意：这是**安全网**，不是主要机制。真正的能量维持必须由
        Transformer 的 `write()` 来做，否则这个系统就没有"求生"可言。
        """
        cfg = self.cfg
        norm = self.energy()
        if norm <= 1e-12:
            return 1.0
        target_scale = math.sqrt(cfg.norm_target / norm)
        scale = 1.0 + cfg.agc_rate * (target_scale - 1.0)
        scale = max(0.5, min(2.0, scale))
        self.psi = self.psi * scale
        return scale

    def tick(self, use_agc: bool = True) -> dict:
        """一个完整的心跳：自演化 + 噪声 +（可选）AGC。

        返回本步的物理量，用于日志/调试。
        """
        e_before = self.energy()
        self.step()
        self.add_noise()
        gain = self.auto_gain() if use_agc else 1.0
        e_after = self.energy()
        self._step_count += 1
        return {
            "step": self._step_count,
            "energy_before": e_before,
            "energy_after": e_after,
            "gain": gain,
        }

    # ══════════════════════════════════════════════════════════════════
    # 唯一的写入口
    # ══════════════════════════════════════════════════════════════════

    def write(self, delta: torch.Tensor, cost_coef: float = 0.0) -> float:
        """把外部（= Transformer）算出的修正叠加到场里。

        **这是世界影响本体的唯一通道。** 没有别的路。

        参数
        ----------
        delta : complex64 (n, n) 或 real (n, n)
            对 ψ 的增量。
        cost_coef : float
            每单位 |Δ| 的能量代价。见 `FieldConfig.write_cost` 的说明：
            **没有代价的话，「一直写满」就是平凡最优解，任务无法学习。**

        返回
        -------
        float
            写入前后能量的**净**变化量（已扣掉代价）。
        """
        if delta.shape != self.psi.shape:
            raise ValueError(f"delta shape {tuple(delta.shape)} != psi {tuple(self.psi.shape)}")
        e0 = self.energy()
        d = delta.to(self.device).to(torch.complex64)
        psi = self.psi + d

        if cost_coef > 0:
            cost = float(cost_coef) * float(d.abs().sum())
            e = float(psi.abs().pow(2).sum())
            if e > 1e-9:
                keep = max(0.0, 1.0 - cost / e)
                psi = psi * math.sqrt(keep)

        self.psi = psi
        return self.energy() - e0

    def write_trainable(self, delta: torch.Tensor, cost_coef: float = 0.0):
        """可微版本的 `write()` —— 供训练使用。

        与 `write()` 的区别：
            `write()` 返回 float（用于记录/诊断）；
            本方法返回 (新 ψ, **可微的 ΔE 张量**)。

        ΔE 的梯度只穿过 `delta`（= Transformer 第一趟的输出），
        不穿过场的历史状态。所以：
            * 每一步的图是独立的，不会跨步累积；
            * 可以直接用 `-ΔE` 作为 loss 训练「如何高效地给场注入能量」。

        调用方约定：把返回的新 ψ **detach** 后赋回 `self.psi`，
        否则下一拍的场演化会把这个图继续接长，导致显存爆炸。

        参数
        ----------
        delta : (n, n) 实数——对 ψ 实部的增量
        cost_coef : 每单位 |Δ| 的能量代价

        返回
        -------
        (psi_new : complex64 (n,n), delta_energy : scalar tensor)
        """
        if delta.shape != self.psi.shape:
            raise ValueError(f"delta shape {tuple(delta.shape)} != psi {tuple(self.psi.shape)}")

        base = self.psi.detach()                       # ★ 切断历史
        d = delta.to(self.device).to(torch.complex64)
        psi = base + d

        if cost_coef > 0:
            cost = float(cost_coef) * d.abs().sum()
            e_now = psi.abs().pow(2).sum()
            keep = torch.clamp(1.0 - cost / e_now.clamp(min=1e-9), min=0.0, max=1.0)
            psi = psi * torch.sqrt(keep)

        e0 = base.abs().pow(2).sum()
        e1 = psi.abs().pow(2).sum()
        return psi, (e1 - e0)

    def psi_normalized(self) -> tuple[torch.Tensor, torch.Tensor]:
        """返回归一化的 (Re ψ, Im ψ)，范围约 [-1,1]。

        供矩阵的 psi_real / psi_imag 通道使用 ——
        **没有这两路信息，Transformer 在原理上就不可能写出高效的能量注入。**
        """
        amp = self.psi.abs()
        m = float(amp.max())
        if m < 1e-12:
            z = torch.zeros_like(amp)
            return z, z
        return (self.psi.real / m).detach(), (self.psi.imag / m).detach()

    # ══════════════════════════════════════════════════════════════════
    # 可观测量
    # ══════════════════════════════════════════════════════════════════

    def energy(self) -> float:
        """场能量 E = Σ|ψ|²。**本体唯一的生命量。**"""
        return float(self.psi.abs().pow(2).sum())

    def energy_tensor(self) -> torch.Tensor:
        """可微版本的能量。**测试用：验证梯度能不能从能量传回 Transformer。**"""
        return self.psi.abs().pow(2).sum()

    def obs_tensor(self) -> torch.Tensor:
        """(n, n) complex —— 直接把场本身交给 Transformer。

        ★ 不做任何派生计算：不取 |ψ|²、不做通道分解、不做归一化。
          场的每个格子就是一个 token 的输入值。
        """
        return self.psi

    def density(self) -> torch.Tensor:
        """归一化密度图 (n, n)，范围 [0,1]。仅用于可视化/诊断。"""
        d = self.psi.abs().pow(2)
        m = float(d.max())
        return d / m if m > 1e-12 else d

    def norm_frac(self) -> float:
        """当前能量 / 目标能量。1.0 = 健康，0 = 死。"""
        return self.energy() / max(1e-12, self.cfg.norm_target)

    def is_alive(self, death_frac: float, max_frac: float | None = None) -> bool:
        """生死判定 —— **唯一的终止条件，而且有上下两条边**。

        参数
        ----------
        death_frac : float
            低边。能量比例低于它就饿死。
        max_frac : float | None
            高边。能量比例高于它就撑死（失控）。
            None = 只判低边（旧行为）。

        ★ 为什么高边也要死：
          写入的能量增益 ∝ √E（线性项 2Σ|ψ|Δ），而代价是常数 L1。
          所以「把场撑大」永远是划算的策略 —— 实测场能量指数爆炸
          （20× 和 27× 目标，然后永远不死）。
          加上高边之后，生命才变成一条带子里的稳态控制。
        """
        f = self.norm_frac()
        if f < death_frac:
            return False
        if max_frac is not None and f > max_frac:
            return False
        return True

    def death_cause(self, death_frac: float, max_frac: float | None = None) -> str:
        """返回死因：'low' / 'high' / ''（还活着）。"""
        f = self.norm_frac()
        if f < death_frac:
            return "low"
        if max_frac is not None and f > max_frac:
            return "high"
        return ""

    def reset(self, seed: int | None = None) -> None:
        """重新初始化（新回合）。"""
        cfg = self.cfg
        s = self.cfg.grid if seed is None else seed
        gen = torch.Generator(device="cpu").manual_seed(int(s))
        real = torch.randn(self.n, self.n, generator=gen)
        imag = torch.randn(self.n, self.n, generator=gen)
        psi = torch.complex(real, imag).to(torch.complex64).to(self.device)
        self.psi = psi / psi.abs().pow(2).sum().sqrt() * math.sqrt(cfg.norm_target)
        self._step_count = 0

    def extra_repr(self) -> str:
        return (f"n={self.n}, E={self.energy():.1f}, "
                f"frac={self.norm_frac():.3f}, steps={self._step_count}")
