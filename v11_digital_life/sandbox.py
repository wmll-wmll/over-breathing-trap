"""沙盒世界 —— 一个不需要 Godot 的玩具环境。

用途：验证架构能不能跑通，而不是验证智能。

世界
----
一条 1 维走廊 x ∈ [0, L]（另有一维 z 但不动）。
一个固定光源在 light_x。
光强随距离高斯衰减。

    light(x) = exp( −((x − light_x)/σ)² )

关键：**世界不产生能量，也不杀死 agent。**
    世界只提供观测 —— 特别是「这里的能量在增还是在减」。
    agent 能不能活，只取决于它自己的场。

这严格符合设计原则：世界只能通过 Transformer 影响本体。
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from .config import Config


class SandboxWorld:
    """极简走廊 + 一个光源。"""

    # ★ 实验 11：世界 v2 —— 移动光源
    #   实验 10 修好呼吸陷阱之后，原地不动、呼吸深度停在 1.5 左右的
    #   **纯随机初始化个体** 已经有 6/8 能活满 350 拍。选择压力为零，
    #   种群里没有「更好」和「更差」之分，进化无从谈起。
    #
    #   ★ 人口遗传学的第一课：没有压力就没有选择。
    #     所以世界必须变难 —— 但不是换一套物理法则（物理是 v11 的身份），
    #     而是换一个**资源分布**：光源不再是固定的，它会在走廊里来回移动。
    #
    #   这样就只有一个办法能活：**跟着光走**。
    #     固定光源 → 找到亮处站着不动就赢了（实验 8-10 的世界）
    #     移动光源 → 站着不动必死，必须持续跟踪（本实验的世界）
    #
    #   物理法则（速度曲线 / 运动耗能 / 生死带）**一个字节都没有改**。

    def __init__(self, cfg: Config, length: float = 1200.0,
                 light_x: float = 900.0, sigma: float = 400.0, seed: int = 0,
                 spawn_min: float = 100.0, spawn_span: float = 200.0,
                 light_amp: float = 0.0, light_period: float = 0.0,
                 light_gain: float = 1.0):
        self.cfg = cfg
        self.length = float(length)
        self.light_x = float(light_x)
        self.sigma = float(sigma)
        self.spawn_min = float(spawn_min)
        self.spawn_span = float(spawn_span)
        self.light_amp = float(light_amp)
        """光源来回移动的振幅（世界单位）。0 = 固定光源（实验 8-10 的世界）。"""
        self.light_period = float(light_period)
        """来回一次的周期（**拍**）。0 = 不移动。"""
        self.light_gain = float(light_gain)
        """光强整体缩放。1.0 = 原样。"""
        self.rng = np.random.default_rng(seed)
        self.episode_seed = seed

        self.pos = np.zeros(2, dtype=np.float64)
        self.prev_light = 0.0
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.t = 0

    # ── 世界自身的物理 ──────────────────────────────────────────────
    def light_center(self, t: float | None = None) -> float:
        """光源中心此刻在哪。

        light_amp=0 或 light_period=0 → 恒为 light_x（实验 8-10 的固定光源）。
        否则按正弦来回移动：center(t) = light_x + amp · sin(2π t / period)
        """
        if self.light_amp == 0.0 or self.light_period <= 0.0:
            return self.light_x
        tt = self.t if t is None else float(t)
        return self.light_x + self.light_amp * math.sin(
            2.0 * math.pi * tt / self.light_period)

    def _light_at(self, x: float, t: float | None = None) -> float:
        d = (x - self.light_center(t)) / self.sigma
        return float(self.light_gain * math.exp(-d * d))

    def reset(self) -> np.ndarray:
        # ★ 实验 8：出生在**远离光源的暗处**，必须自己走过去才活得下去。
        #   旧版出生点固定在 [0, 60)，那里基本是纯黑。
        self.pos = np.array([self.spawn_min + self.spawn_span * self.rng.random(), 0.0])
        self.t = 0
        self.prev_light = self._light_at(self.pos[0])
        self.prev_action = np.zeros(2)
        return self._obs()

    def _obs(self) -> np.ndarray:
        """观测矢量 —— 世界给本体的全部信息。

        [0] 位置 x（归一化到 [-1,1]）
        [1] 位置 z（归一化，这里恒 0）
        [2] 当前光强
        [3] 光强的变化率（★「这里能量在增还是减」）
        [4..5] 上一拍的动作
        [7] 时间进度
        """
        light = self._light_at(self.pos[0])
        slope = (light - self.prev_light)
        return np.array(
            [
                (self.pos[0] / self.length) * 2.0 - 1.0,
                (self.pos[1] / 150.0),
                light,
                slope * 10.0,
                self.prev_action[0],
                self.prev_action[1],
                min(1.0, self.t / max(1, self.cfg.life.max_steps)),
            ],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        """动作 → 移动 → 新观测。**注意：这里没有任何能量注入或死亡判定。**

        ★ 单位修正（实验 8）
        -------------------
        `brain.heartbeat()` 返回的 action 已经是**世界单位/秒**
        （物理法则算出来的速度，幅值 ≤ physics.V_WORLD = 150），
        和 live.py 发给 Godot 的量是同一个东西。

        旧版本这里把它当成归一化动作 `[-1, 1]` 又乘了 150 —— **单位是错的**，
        实际速度被放大/压缩了上百倍。因为旧版本只用来做冒烟测试，
        一直没暴露。实验 8 要用沙盒跑真实对照，必须先修正：

            Δ位置 = 速度(世界单位/秒) × dt(秒)

        """
        a = np.asarray(action, dtype=np.float64).reshape(2)
        # 只裁速度上限，不裁成归一化量
        speed_mag = float(np.linalg.norm(a))
        if speed_mag > 150.0:
            a = a * (150.0 / speed_mag)
        dt = self.cfg.field.dt
        self.pos[0] = float(np.clip(self.pos[0] + a[0] * dt, 0.0, self.length))
        self.pos[1] = float(np.clip(self.pos[1] + a[1] * dt, -150.0, 150.0))

        self.t += 1                       # ★ 先走时间：光源也跟着走
        new_light = self._light_at(self.pos[0])
        self.prev_action = a          # 观测要看到刚做的动作
        obs = self._obs()             # 此时 prev_light 仍是**上一拍**的值
        self.prev_light = new_light   # ★ 必须放在 _obs() 之后，否则 slope 恒为 0
        done = self.t >= self.cfg.life.max_steps
        info = {"alive": True, "light": new_light, "x": self.pos[0],
                "light_center": self.light_center()}
        return obs, done, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--device", default="", help="空 = 自动（有 CUDA 用 CUDA）")
    args = ap.parse_args()

    from .brain import Brain

    cfg = Config()
    cfg.device = args.device
    cfg.life.max_steps = args.steps

    brain = Brain(cfg, seed=0)
    print("=" * 64)
    print("  V11 数字生命体 — 沙盒")
    print("=" * 64)
    print(f"  场网格      : {cfg.field.grid}×{cfg.field.grid}")
    print(f"  token 数    : {cfg.field.grid * cfg.field.grid + 1}（逐格 + 1 个世界 token）")
    print(f"  patch       : {cfg.interface.patch}  → "
          f"{(cfg.field.grid // cfg.interface.patch) ** 2} tokens")
    print(f"  d_model     : {cfg.interface.d_model}")
    print(f"  Transformer : 第一趟 {cfg.interface.n_layers_pass1} 层 / "
          f"第二趟 {cfg.interface.n_layers_pass2} 层")
    print(f"  参数量      : {brain.n_params():,}")
    print(f"  死亡阈值    : 场能量 < {cfg.life.death_norm_frac:.0%} 目标")
    print()

    # 空跑：不训练，只看场能不能活着
    print("── 阶段 A：空跑（不训练），看场能不能自己活着 ──")
    env = SandboxWorld(cfg, seed=0)
    r = brain.run_episode(env, sample=False, train=False)
    print(f"  存活步数 : {r['steps']} / {cfg.life.max_steps}")
    print(f"  最终能量 : {brain.field.norm_frac():.3f} × 目标")
    print(f"  是否存活 : {r['survived']}")
    print()

    if not args.train:
        print("（加 --train 进入训练）")
        return 0

    # 训练
    print(f"── 阶段 B：训练 {args.episodes} 回合 ──")
    histories = []
    for ep in range(args.episodes):
        env = SandboxWorld(cfg, seed=ep + 1)
        r = brain.run_episode(env, sample=True, train=True)
        histories.append(r)
        if (ep + 1) % max(1, args.episodes // 10) == 0:
            recent = histories[-max(1, args.episodes // 10):]
            avg_steps = sum(h["steps"] for h in recent) / len(recent)
            print(f"  第 {ep + 1:4d} 回合  "
                  f"平均存活 {avg_steps:6.1f}  "
                  f"损失 {r.get('loss', 0.0):+.3f}  "
                  f"场能量 {brain.field.norm_frac():.3f}")

    print()
    first = sum(h["steps"] for h in histories[:10]) / 10
    last = sum(h["steps"] for h in histories[-10:]) / 10
    print(f"  前 10 回合平均存活 : {first:.1f}")
    print(f"  后 10 回合平均存活 : {last:.1f}")
    print(f"  变化              : {last - first:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
