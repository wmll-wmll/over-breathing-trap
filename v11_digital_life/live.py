"""V11 在真实场景里活着 —— 没有"训练阶段"，就是放进去，看它活不活。

    python -m v11_digital_life.live --port 9002
    python -m v11_digital_life.live --port 9002 --launch       # 自己拉 Godot
    python -m v11_digital_life.live --port 9002 --lives 20     # 活 20 条命就停

一个心跳（0.5 秒）：
    Godot 发观测 → V11 心跳（场演化 + 第一趟写场 + 第二趟出动作 + 梯度更新）
    → 动作发回 Godot → agent 移动 → 下一拍

死了怎么办（选项 B）：
    **连脑子一起清空。** 新随机权重、新场、新优化器。
    每条命都是全新的个体 —— 纯自然选择，这一辈子学到什么就活多久。

日志：data/V11/live_<时间戳>.csv，每拍一行 + 每次死亡的汇总。
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from v11_digital_life.brain import Brain
from v11_digital_life.config import Config
from v11_digital_life.godot_bridge import GodotBridge

GODOT = _ROOT / "Godot_v4.7" / "Godot_v4.7-stable_mono_win64" / "Godot_v4.7-stable_mono_win64_console.exe"
PROJECT = _ROOT / "V10.7" / "godot_project"


# ══════════════════════════════════════════════════════════════════════
# 观测 → 8 维世界矢量
# ══════════════════════════════════════════════════════════════════════

class ObsEncoder:
    """把 Godot 的 obs 字典压成 8 维矢量。

    ★ 这里不做任何"智能"计算 —— 只是坐标归一化和一阶差分。
      真正的感知在 Transformer 里（场是唯一的状态）。
    """

    def __init__(self, arena_x: float = 600.0, arena_z: float = 150.0):
        self.arena_x = arena_x
        self.arena_z = arena_z
        self.prev_light = 0.0
        self.prev_action = np.zeros(3, dtype=np.float32)

    def reset(self) -> None:
        self.prev_light = 0.0
        self.prev_action = np.zeros(3, dtype=np.float32)

    def __call__(self, obs: dict) -> np.ndarray:
        pos = obs.get("pos", [0.0, 0.0, 0.0])
        px = float(pos[0])
        pz = float(pos[2]) if len(pos) > 2 else float(pos[1])
        light = float(obs.get("light", 0.0))
        dl = light - self.prev_light
        self.prev_light = light
        t = float(obs.get("t", 0.0))
        return np.array(
            [
                px / self.arena_x,
                pz / self.arena_z,
                light,
                dl * 10.0,
                self.prev_action[0],
                self.prev_action[1],
                self.prev_action[2],
                min(1.0, t / 300.0),
            ],
            dtype=np.float32,
        )

    def set_action(self, a: np.ndarray) -> None:
        self.prev_action = np.asarray(a, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# 启动 Godot
# ══════════════════════════════════════════════════════════════════════

def launch_godot(port: int) -> subprocess.Popen | None:
    if not GODOT.exists():
        print(f"[Godot] 找不到: {GODOT}")
        return None
    env = dict(os.environ)
    env["V107_TCP_PORT"] = str(port)
    env["V107_GHOST"] = "1"        # ★ 生死交给 V11 的场能量
    cmd = [str(GODOT), "--headless", "--path", str(PROJECT),
           "--rendering-driver", "dummy", "--no-predator"]
    print("[Godot] 启动:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env, cwd=str(_ROOT))


# ══════════════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="V11 在真实场景里活着")
    ap.add_argument("--port", type=int, default=9002)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--launch", action="store_true", help="自己拉一个 Godot")
    ap.add_argument("--lives", type=int, default=0, help="活多少条命后停（0 = 无限）")
    ap.add_argument("--max-steps", type=int, default=600, help="单条命最多多少拍")
    ap.add_argument("--device", default="", help="空 = 自动")
    ap.add_argument("--speed", type=float, default=150.0)
    ap.add_argument("--generations", type=int, default=0,
                    help="跑多少代后停（0 = 用 --lives 的值）")
    ap.add_argument("--keep-brain", dest="keep_brain", action="store_true",
                    default=True, help="重生时保留网络与优化器（默认开）")
    ap.add_argument("--no-keep-brain", dest="keep_brain", action="store_false",
                    help="重生时清空大脑（旧的选项 B）")
    ap.add_argument("--ckpt-dir", default="", help="模型文件目录（默认 data/V11/models）")
    ap.add_argument("--write-objective", default="speed",
                    choices=["speed", "energy", "none"],
                    help="写场头的训练目标：speed=最大化速度能力(默认) / "
                         "energy=最大化ΔE(旧,会自爆) / none=不训练")
    ap.add_argument("--write-gain-max", type=float, default=0.5,
                    help="写场增益上限（默认 0.5；调小可以削掉写场这个强杠杆）")
    args = ap.parse_args()

    # generations 优先于 lives
    n_generations = args.generations or args.lives

    # ── 日志 ────────────────────────────────────────────────────────
    log_dir = _ROOT / "data" / "V11"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (log_dir / "models")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"live_{stamp}.csv"
    fh = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    w.writerow(["life", "step", "pos_x", "pos_z", "light", "danger",
                "norm_frac", "delta_e", "meta_gain", "force_norm", "v_norm",
                "move_cost", "std", "vx", "vy", "vz", "loss", "death_cause"])

    # 每代的存活记录（最后画学习曲线用）
    history: list[dict] = []

    # ── 起 Godot（可选）──────────────────────────────────────────────
    proc = launch_godot(args.port) if args.launch else None
    if proc is not None:
        time.sleep(3.0)

    # ── 连接 ────────────────────────────────────────────────────────
    bridge = GodotBridge(args.host, args.port)
    if not bridge.connect():
        fh.close()
        if proc:
            proc.terminate()
        return 1

    # ── 大脑 ────────────────────────────────────────────────────────
    cfg = Config()
    cfg.device = args.device
    cfg.life.max_steps = args.max_steps
    cfg.life.write_objective = args.write_objective
    cfg.interface.write_gain_max = args.write_gain_max
    brain = Brain(cfg, seed=0)
    print("=" * 70)
    print("  V11 数字生命体 —— 真实场景")
    print("=" * 70)
    print(f"  设备      : {brain.device}")
    print(f"  参数量    : {brain.n_params():,}")
    print(f"  token 数  : {brain.interface.n_cells + 1:,}（逐格 + 世界）")
    print(f"  死亡线    : 场能量 < {cfg.life.death_norm_frac:.0%}，"
          f"> {cfg.life.max_norm_frac:.0%} 也死")
    print(f"  写场目标  : {args.write_objective}（增益上限 {args.write_gain_max}）")
    print(f"  重生方式  : " + ("跨代累积（脑子保留）" if args.keep_brain
                                else "脑子清空（选项 B）"))
    print(f"  代数      : {n_generations if n_generations else '无限'}")
    print(f"  模型目录  : {ckpt_dir}")
    print(f"  日志      : {csv_path}")
    print()

    enc = ObsEncoder()
    lives = 0
    t_start = time.time()
    life_steps = 0
    last_hb = time.time()

    try:
        while True:
            raw = bridge.recv(timeout=15.0)
            if raw is None:
                print("[live] 收不到观测，退出")
                break

            obs_vec = enc(raw)

            # 一个心跳（含在线梯度更新）
            t0 = time.time()
            action, info = brain.step(obs_vec, sample=True, train=True)
            hb_dt = time.time() - t0
            life_steps += 1

            # ★ action 已经是世界单位/秒（brain 内部套过物理法则了）
            vel = action
            enc.set_action(action)
            bridge.send_action(vel, energy_pool=info["energy"] * 10.0,
                               t=float(raw.get("t", 0.0)),
                               light=float(raw.get("light", 0.0)),
                               dE=info["delta_e"])

            pos = raw.get("pos", [0.0, 0.0, 0.0])
            w.writerow([lives + 1, life_steps,
                        f"{float(pos[0]):.1f}",
                        f"{float(pos[2]) if len(pos) > 2 else 0.0:.1f}",
                        f"{float(raw.get('light', 0.0)):.4f}",
                        f"{float(raw.get('danger', 0.0)):.3f}",
                        f"{info['norm_frac']:.4f}",
                        f"{info['delta_e']:+.3f}",
                        f"{info['meta_gain']:.1f}",
                        f"{info['force_norm']:.4f}",
                        f"{info['v_norm']:.4f}",
                        f"{info['move_cost']:.1f}",
                        f"{info['std']:.4f}",
                        f"{vel[0]:.1f}", f"{vel[1]:.1f}", f"{vel[2]:.1f}",
                        f"{info.get('loss', float('nan')):.4f}",
                        info.get("death_cause", "")])
            fh.flush()

            # ── 死亡 ────────────────────────────────────────────────
            dead = (not info["alive"]) or life_steps >= args.max_steps
            if dead:
                lives += 1
                cause = info.get("death_cause", "") or "max_steps"
                label = {"low": "饿死", "high": "撑死"}.get(cause, cause)

                # 这一代的成绩
                rec = {
                    "gen": lives,
                    "beats": life_steps,
                    "cause": cause,
                    "final_norm": float(info["norm_frac"]),
                }
                history.append(rec)

                # ★ 每代存盘 —— 这就是「记忆」的落盘形式
                ckpt = ckpt_dir / f"gen_{lives:03d}.pt"
                try:
                    brain.save_checkpoint(ckpt, extra={
                        "survival_beats": life_steps,
                        "death_cause": cause,
                        "history": history,
                    })
                    saved = f"　→ {ckpt.name}"
                except Exception as e:  # noqa: BLE001
                    saved = f"　（存盘失败 {type(e).__name__}）"

                # 学习曲线：最近 5 代的平均
                recent = [h["beats"] for h in history[-5:]]
                avg5 = sum(recent) / len(recent)
                trend = ""
                if len(history) >= 2:
                    d = history[-1]["beats"] - history[0]["beats"]
                    trend = f"　(第1代 {history[0]['beats']} → 本代 {life_steps}, {d:+d})"

                print(f"  第 {lives:3d} 代：活 {life_steps:3d} 拍 "
                      f"（{life_steps * 0.5:3.0f} 秒）　{label:<4s} "
                      f"末能量 {info['norm_frac']:.3f}　近5代均 {avg5:.0f}{trend}{saved}")

                if n_generations and lives >= n_generations:
                    break

                # 重生
                brain.rebirth(keep_brain=args.keep_brain)
                enc.reset()
                life_steps = 0

            # 别跑飞（Godot 是 2 Hz）
            elapsed = time.time() - t0
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)

    except KeyboardInterrupt:
        print("\n[live] 手动中断")
    finally:
        fh.close()
        bridge.close()
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                pass

    print()
    print("=" * 70)
    print(f"  共 {lives} 代，总时长 {time.time() - t_start:.0f} 秒")
    if history:
        beats = [h["beats"] for h in history]
        first = sum(beats[:5]) / len(beats[:5])
        last = sum(beats[-5:]) / len(beats[-5:])
        print(f"  前 5 代平均存活 : {first:6.1f} 拍")
        print(f"  后 5 代平均存活 : {last:6.1f} 拍")
        print(f"  变化            : {last - first:+6.1f} 拍")
        n_high = sum(1 for h in history if h["cause"] == "high")
        n_low = sum(1 for h in history if h["cause"] == "low")
        n_max = sum(1 for h in history if h["cause"] == "max_steps")
        print(f"  死因分布        : 撑死 {n_high} / 饿死 {n_low} / 活满 {n_max}")
        print()
        print("  存活曲线：")
        bar_max = max(1, max(beats))
        for h in history:
            bar = "█" * int(40 * h["beats"] / bar_max)
            print(f"    gen {h['gen']:3d}  {h['beats']:3d}  {bar}")
    print(f"  日志   ：{csv_path}")
    print(f"  模型   ：{ckpt_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
