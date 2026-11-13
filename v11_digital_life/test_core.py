"""V11 核心自检 —— 不需要 Godot。有 CUDA 会自动用。

跑法：
    python v11_digital_life/test_core.py

对应 README 的验证顺序：

    T0  配置合法
    T1  场「不主动维持就会死」（求生任务成立）
    T2  write() 有真实后果
    T3  ★ 梯度能从场能量传回 Transformer
    T4  死亡判据会触发
    T5  ★ 128×128 逐格 token：格子数、能量可见性
    T6  一个完整心跳的闭环
    T7  沙盒回合能跑完
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from v11_digital_life.brain import Brain
from v11_digital_life.config import Config
from v11_digital_life.field import NLSEField
from v11_digital_life.sandbox import SandboxWorld

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"{PASS if ok else FAIL}  {name}" + (f"   [{detail}]" if detail else ""))
    return ok


def main() -> int:
    t_start = time.time()
    print("=" * 70)
    print("  V11 数字生命体 — 核心自检（逐格 token 版）")
    print("=" * 70)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {dev}" + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    print()

    cfg = Config()
    cfg.life.max_steps = 40          # 自检用小回合，快

    # ══════════════════════════════════════════════════════════════
    print("── T0  配置合法 ──")
    try:
        cfg.validate()
        ok, err = True, ""
    except Exception as e:  # noqa: BLE001
        ok, err = False, str(e)
    check("配置校验通过", ok, err)
    n_tok = cfg.field.grid * cfg.field.grid + 1
    print(f"       grid={cfg.field.grid}  token={n_tok:,}  "
          f"obs_dim={cfg.interface.obs_dim}  death={cfg.life.death_norm_frac:.0%}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T1  场必须「不主动维持就会死」（求生任务成立）──")
    death_frac = cfg.life.death_norm_frac
    field = NLSEField(cfg.field, torch.device("cpu"), seed=0)
    check("初始能量等于目标", abs(field.norm_frac() - 1.0) < 1e-3, f"{field.energy():.1f}")
    for _ in range(200):
        field.step()
        field.add_noise()
    frac200 = field.norm_frac()
    check("★ 无写入时 200 步内跌破死亡线（否则任务平凡）",
          frac200 < death_frac, f"{frac200:.3f} vs 死亡线 {death_frac:.2f}")
    f2 = NLSEField(cfg.field, torch.device("cpu"), seed=0)
    a = f2.energy()
    for _ in range(100):
        f2.step()
        f2.add_noise()
    check("无写入时能量确实下降", f2.energy() < a, f"{f2.energy() / a:.3f}× 初始")
    f2b = NLSEField(cfg.field, torch.device("cpu"), seed=0)
    for _ in range(100):
        f2b.step()
    rel = abs(f2.energy() - f2b.energy()) / max(1e-9, f2b.energy())
    check("噪声是能量中性的（不往场里泵能量）", rel < 0.02,
          f"加噪声 {f2.energy():.0f} vs 不加 {f2b.energy():.0f}，差 {rel:.2%}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T2  write() 有真实后果 ──")
    f3 = NLSEField(cfg.field, torch.device("cpu"), seed=1)
    before = f3.energy()
    delta = torch.full((cfg.field.grid, cfg.field.grid), 0.05, dtype=torch.float32)
    dE = f3.write(delta)
    check("写入后能量改变量与返回值一致", abs((f3.energy() - before) - dE) < 1e-3,
          f"ΔE={dE:+.3f}")
    check("写入确实改变了能量", abs(dE) > 1.0, f"|ΔE|={abs(dE):.3f}")
    check("单次写入幅度合理（< 20% 目标）",
          abs(dE / cfg.field.norm_target) < 0.20, f"{dE / cfg.field.norm_target:+.2%}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T3  ★ 梯度能从场能量传回 Transformer ──")
    brain = Brain(cfg, seed=0)
    obs = np.zeros(cfg.interface.obs_dim, dtype=np.float32)
    brain.interface.zero_grad(set_to_none=True)
    _, info = brain.heartbeat(obs, sample=False)
    dE = info["_dE"]
    check("心跳返回的 ΔE 是可微张量", bool(dE.requires_grad), str(dE.requires_grad))
    loss = -dE / cfg.field.norm_target
    loss.backward()
    grads = [(n, p.grad) for n, p in brain.interface.named_parameters()
             if p.grad is not None]
    nonzero = [(n, float(g.norm())) for n, g in grads if float(g.norm()) > 0]
    total_gnorm = float(torch.sqrt(sum((g.pow(2).sum() for _, g in grads),
                                       torch.zeros((), device="cpu"))))
    check("有参数收到梯度", len(nonzero) > 0, f"{len(nonzero)}/{len(grads)} 个参数")
    check("梯度范数 > 0（能量对网络可微）", total_gnorm > 0.0, f"‖g‖={total_gnorm:.3e}")
    wf = [(n, float(p.grad.norm())) for n, p in brain.interface.named_parameters()
          if "write_head" in n and p.grad is not None]
    check("第一趟的「写场」头有梯度（★ 世界→本体的通道可训练）",
          len(wf) > 0 and max(g for _, g in wf) > 0,
          f"{wf[0][0]} ‖g‖={wf[0][1]:.3e}" if wf else "无梯度")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T4  死亡判据会触发（上下两条边）──")
    f4 = NLSEField(cfg.field, torch.device("cpu"), seed=2)
    check("满能量时判定为活", f4.is_alive(death_frac, cfg.life.max_norm_frac),
          f"{f4.norm_frac():.3f}")
    f4.psi = f4.psi * 0.05
    check("★ 能量过低判定为死（饿死）",
          not f4.is_alive(death_frac, cfg.life.max_norm_frac),
          f"{f4.norm_frac():.3f}")
    f4b = NLSEField(cfg.field, torch.device("cpu"), seed=2)
    f4b.psi = f4b.psi * 5.0
    check("★ 能量过高也判定为死（撑死/失控）",
          not f4b.is_alive(death_frac, cfg.life.max_norm_frac),
          f"{f4b.norm_frac():.3f} vs 上限 {cfg.life.max_norm_frac}")
    check("死因区分正确",
          f4.death_cause(death_frac, cfg.life.max_norm_frac) == "low"
          and f4b.death_cause(death_frac, cfg.life.max_norm_frac) == "high",
          "low/high")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T5  ★ 128×128 逐格 token ──")
    check("token 数 = 128×128 + 1",
          brain.interface.n_cells == cfg.field.grid ** 2,
          f"{brain.interface.n_cells:,} + 1")
    f5 = NLSEField(cfg.field, brain.device, seed=3)   # 必须和网络在同设备
    world = torch.zeros(1, cfg.interface.obs_dim, device=brain.device)
    with torch.no_grad():
        tok100 = brain.interface._tokens(f5.obs_tensor(), world).clone()
        f5.psi = f5.psi * 0.5                      # 能量降到 25%
        tok025 = brain.interface._tokens(f5.obs_tensor(), world).clone()
    # 逐格 token 直接来自场的 (Re, Im)，能量降一半 → 嵌入应明显变小
    d = float((tok100 - tok025).abs().max())
    check("能量变化直接体现在逐格 token 上", d > 0.1, f"最大差 {d:.3f}")
    check("token 数正确（16,384 个格子 + 1 个世界 token）",
          tok100.shape[1] == cfg.field.grid ** 2 + 1, f"{tok100.shape[1]:,}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T6  一个完整心跳的闭环 ──")
    brain2 = Brain(cfg, seed=1)
    obs = np.zeros(cfg.interface.obs_dim, dtype=np.float32)
    t0 = time.time()
    action, info = brain2.heartbeat(obs, sample=True)
    dt = time.time() - t0
    check("动作形状正确", action.shape == (2,), str(action.shape))
    check("动作是速度矢量（世界单位/秒）", bool(np.all(np.abs(action) <= 150.0 + 1e-6)),
          f"max={float(np.abs(action).max()):.1f}")
    check("归一化速度 v_norm ∈ [0,1]", 0.0 <= info["v_norm"] <= 1.0 + 1e-9,
          f"v_norm={info['v_norm']:.4f}")
    check("力被 tanh 压在 [-1,1]", abs(info["force_norm"]) <= 1.7321 + 1e-6,
          f"|F|={info['force_norm']:.3f}")
    check("运动有能量代价", info["move_cost"] >= 0.0,
          f"cost={info['move_cost']:.1f}")
    check("场能量被心跳改变（写入生效）", abs(info["delta_e"]) > 1e-6,
          f"ΔE={info['delta_e']:+.4f}")
    check("动作是有限值", bool(np.all(np.isfinite(action))), "")
    check("std = σ 而非 1.0（v10 的教训）", abs(info["std"] - 0.368) < 0.05,
          f"σ={info['std']:.3f}")
    print(f"       一个心跳耗时 {dt:.2f} 秒　能量 {info['energy_before_tick']:.1f} → "
          f"{info['energy']:.1f}　norm_frac={info['norm_frac']:.4f}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T7  沙盒回合能跑完 ──")
    env = SandboxWorld(cfg, seed=3)
    brain3 = Brain(cfg, seed=2)
    r = brain3.run_episode(env, sample=False, train=False)
    check("回合正常结束", r["steps"] > 0, f"{r['steps']} 步")
    check("能量有限", bool(np.isfinite(r["final_norm_frac"])),
          f"{r['final_norm_frac']:.4f}")
    print(f"       存活 {r['steps']}/{cfg.life.max_steps} 步  "
          f"最终能量 {r['final_norm_frac']:.3f}× 目标  "
          f"存活={r['survived']}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T8  ★ 物理法则：体重、两个系数、速度 ══")
    from v11_digital_life import physics as ph

    m = ph.mass_of(cfg.field.grid)
    e0 = ph.energy_capacity(m)
    check("体重 = 矩阵乘积 (128×128)", abs(m - 16384.0) < 1e-6, f"m={m:,.0f}")
    check("能量容量 ∝ 体重", abs(e0 - 10000.0) < 1e-6, f"E₀={e0:,.0f}")

    # 物理法则系数 f(x) = (1+sin(x²))/(1+x²)
    c_center = ph.phys_coef(ph.BAND_CENTER)
    c_low = ph.phys_coef(ph.LOWER)
    c_high = ph.phys_coef(ph.UPPER)
    check("物理系数在带子正中 = 1.0", abs(c_center - 1.0) < 1e-9, f"{c_center:.6f}")
    check("物理系数在两端 → 0", c_low < 1e-3 and c_high < 1e-3,
          f"下端 {c_low:.6f} / 上端 {c_high:.6f}")
    # 全区间非负且 ≤1
    samples = [ph.phys_coef(ph.LOWER + (ph.UPPER - ph.LOWER) * i / 400)
               for i in range(401)]
    check("物理系数全区间 ∈ [0,1]",
          min(samples) >= 0.0 and max(samples) <= 1.0 + 1e-12,
          f"[{min(samples):.6f}, {max(samples):.6f}]")

    # 速度缩放
    s_center = ph.speed_scale(e0 * ph.BAND_CENTER, m, e0)
    check("带子正中速度缩放接近 1", abs(s_center - 1.0) < 0.02, f"{s_center:.4f}")

    # 力 → 速度
    vel, vn = ph.force_to_velocity([1.0, 0.0], e0 * 1.6, m, e0)
    check("力→速度：方向不变", abs(vel[0] - vn * 150.0) < 1e-6, f"vx={vel[0]:.1f}")
    check("力→速度：v_norm ≤ 1", vn <= 1.0 + 1e-12, f"v_norm={vn:.4f}")
    vel0, vn0 = ph.force_to_velocity([0.0, 0.0], e0 * 1.6, m, e0)
    check("零力 → 零速度", vn0 == 0.0 and float(np.abs(vel0).max()) == 0.0, "0")

    # 运动耗能
    cost = ph.move_energy_cost(1.0, m)
    check("满速运动耗能 ≈ 1.5% × 5000", 0.5e-2 < cost / (e0 * 1.6) < 3e-2,
          f"cost={cost:.0f}（占 16000 的 {100*cost/16000:.2f}%）")

    # 生死边界与物理模块一致
    check("生死边界与物理模块一致",
          abs(cfg.life.death_norm_frac - ph.LOWER) < 1e-12
          and abs(cfg.life.max_norm_frac - ph.UPPER) < 1e-12,
          f"[{ph.LOWER:.2f}, {ph.UPPER:.2f}]")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T9  ★ 基础代谢：光提供能量（距离越远越少）──")

    fcfg = Config().field
    fld = NLSEField(fcfg, "cpu", seed=0)

    gt = ph.META_K * 0.187 / fcfg.dissipation
    check("中心光强(0.187)打平耗散", abs(gt / 10000.0 - 1.0) < 0.01,
          f"E* = {gt:.0f}（{gt / 10000.0:.3f}× 容量）")

    fld_a = NLSEField(fcfg, "cpu", seed=0)
    g_low = fld_a.metabolize(0.1, ph.META_K)
    g_high = fld_a.metabolize(1.0, ph.META_K)
    check("光越强获得越多", g_high > g_low > 0.0,
          f"光0.1 → +{g_low:.0f}/拍，光1.0 → +{g_high:.0f}/拍")
    check("代谢严格正比于光强", abs(g_high / g_low - 10.0) < 1e-6,
          f"比值 {g_high / g_low:.4f}")

    fld_b = NLSEField(fcfg, "cpu", seed=1)
    check("光强为 0 不获得能量", fld_b.metabolize(0.0, ph.META_K) == 0.0, "0")

    fld_c = NLSEField(fcfg, "cpu", seed=2)
    e_before = fld_c.energy()
    gained = fld_c.metabolize(0.32, ph.META_K)
    e_after = fld_c.energy()
    check("场能量确实增加", abs((e_after - e_before) - gained) < 1e-2,
          f"{e_before:.0f} → {e_after:.0f}（+{gained:.0f}）")

    e_star_hi = ph.META_K * 0.70 / fcfg.dissipation
    check("太靠近光源 → 平衡点超出撑死线", e_star_hi > 3.0 * 10000.0,
          f"光0.70 → E* = {e_star_hi:.0f}（{e_star_hi / 10000.0:.2f}×）")

    # brain 里确实调用了代谢
    b9 = Brain(Config(), seed=0)
    obs9 = np.zeros(Config().interface.obs_dim, dtype=np.float32)
    obs9[2] = 0.32
    _, info9 = b9.heartbeat(obs9, sample=False)
    check("心跳里真的调用代谢", abs(info9["meta_gain"] - ph.META_K * 0.32) < 1e-3,
          f"meta_gain = +{info9['meta_gain']:.1f}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T10  ★ 实验 8：呼吸深度（第二个可学习的动作）──")

    from v11_digital_life import physics as ph2

    # 线性项 / 二次项各自正确
    g = ph2.breath_gain(0.5, 1.0)
    check("呼吸增益 = K·light·k − C·k²",
          abs(g - (ph2.BREATH_K * 0.5 - ph2.BREATH_C)) < 1e-9,
          f"light=0.5,k=1 → {g:+.1f}")

    # 最优深度 = K·light/(2C)
    for L in (0.1, 0.25, 0.5):
        kstar = ph2.breath_optimal_k(L)
        check(f"最优呼吸深度 k*(light={L}) = K·light/(2C)",
              abs(kstar - ph2.BREATH_K * L / (2 * ph2.BREATH_C)) < 1e-12,
              f"k*={kstar:.4f}")

    # 数值验证：k* 确实是极大值点（不是随手写的公式）
    ok_max = True
    detail_max = ""
    for L in (0.15, 0.4, 0.7):
        kstar = min(ph2.BREATH_MAX, ph2.breath_optimal_k(L))
        g_star = ph2.breath_gain(L, kstar)
        ks = np.linspace(ph2.BREATH_MIN, ph2.BREATH_MAX, 4001)
        gs = ph2.BREATH_K * L * ks - ph2.BREATH_C * ks ** 2
        if g_star < gs.max() - 1e-6:
            ok_max = False
            detail_max = f"light={L}: 公式 {g_star:.3f} < 扫描 {gs.max():.3f}"
    check("★ 数值扫描确认 k* 就是最大增益点", ok_max, detail_max or "三个光强均成立")

    # 光越强，最优深度越大（单调）
    k_seq = [ph2.breath_optimal_k(L) for L in (0.05, 0.2, 0.4)]
    check("光越强 → 最优呼吸越深", k_seq[0] < k_seq[1] < k_seq[2],
          f"{k_seq[0]:.3f} < {k_seq[1]:.3f} < {k_seq[2]:.3f}")

    # ★ 常数 k 一定有损失 —— 这是这一维动作存在的意义
    loss_const = []
    for L in (0.2, 0.5, 0.8):
        kstar = min(ph2.BREATH_MAX, ph2.breath_optimal_k(L))
        loss_const.append(ph2.breath_gain(L, kstar) - ph2.breath_gain(L, 1.0))
    check("★ 固定 k=1 一定拿不到最优增益", all(x > 0 for x in loss_const),
          "光 0.2/0.5/0.8 的损失 = " + " / ".join(f"{x:.1f}" for x in loss_const))

    # metabolize 支持负增益（呼吸有代价）
    fld_d = NLSEField(Config().field, "cpu", seed=3)
    e0d = fld_d.energy()
    gd = fld_d.metabolize(0.0, ph2.BREATH_K, k=1.5, c=ph2.BREATH_C)
    check("★ 无光时深呼吸是净亏（增益可为负）",
          gd < 0 and fld_d.energy() < e0d,
          f"{e0d:.0f} → {fld_d.energy():.0f}（{gd:+.1f}）")

    # breath_map 有界且单调
    zs = torch.linspace(-6, 6, 25)
    ks = ph2.breath_map(zs)
    check("breath_map 有界 [MIN, MAX] 且单调递增",
          bool((ks >= ph2.BREATH_MIN - 1e-6).all()
               and (ks <= ph2.BREATH_MAX + 1e-6).all()
               and (ks[1:] > ks[:-1]).all()),
          f"z∈[−6,6] → k∈[{float(ks.min()):.3f}, {float(ks.max()):.3f}]")
    check("初始呼吸深度明显偏深（有学习压力）",
          ph2.breath_map_float(ph2.BREATH_INIT_Z) > 1.2,
          f"k0 = {ph2.breath_map_float(ph2.BREATH_INIT_Z):.3f}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T11  ★ 实验 8：三种 Transformer 布局（唯一变量是架构）──")

    from v11_digital_life.interface import EntityInterface

    def build(split, l2, lm, pool):
        c = Config().interface
        c.split_pass2 = split
        c.n_layers_pass2 = l2
        c.n_layers_meta = lm
        c.meta_pool = pool
        return EntityInterface(c, grid=Config().field.grid,
                               init_log_std=-1.0, world_obs_dim=Config().interface.obs_dim)

    iA = build(False, 2, 0, 1)
    iB = build(True, 1, 1, 1)
    iC = build(True, 1, 1, 4)
    iD = build(False, 1, 0, 1)
    pA, pB, pC, pD = (x.n_params() for x in (iA, iB, iC, iD))

    check("★ 臂 A（单干路 2 层）与臂 B（双干路 1+1）参数量严格相等",
          pA == pB, f"{pA:,} == {pB:,}")
    check("臂 C 只多一个下采样嵌入（+3×d+d = 256）",
          pC - pB == 256, f"{pC:,}（比 B 多 {pC - pB}）")
    check("臂 D 比 A 少一整层", pA - pD > 40000,
          f"A {pA:,} / D {pD:,}（差 {pA - pD:,}）")
    check("臂 A/D 没有独立呼吸干路",
          iA.pass2_meta is None and iD.pass2_meta is None, "pass2_meta = None")
    check("臂 B/C 有独立呼吸干路",
          iB.pass2_meta is not None and len(iB.pass2_meta) == 1, "1 层")

    # 前向形状 + 两种分辨率
    psi = torch.randn(Config().field.grid, Config().field.grid,
                      dtype=torch.complex64) * 0.1
    wob = torch.zeros(1, Config().interface.obs_dim)
    for name, iface in (("A", iA), ("B", iB), ("C", iC), ("D", iD)):
        m, v, s, z, sz = iface.act_all(psi, wob)
        ok = (m.shape == (1, 2) and v.shape == (1, 1) and s.shape == (1, 2)
              and z is not None and z.shape == (1, 1) and sz.shape == (1, 1))
        extra = ""
        if name == "C":
            extra = f" 呼吸干路 token={iface.cfg.meta_pool and (Config().field.grid // 4) ** 2 + 1}"
        check(f"臂 {name} 前向输出形状正确", ok, f"mean{tuple(m.shape)} z{tuple(z.shape)}{extra}")

    # need_breath=False 时不跑呼吸干路
    m2, v2, s2, z2, sz2 = iC.act_all(psi, wob, need_breath=False)
    check("need_breath=False 时跳过呼吸干路（慢时钟省算力）",
          z2 is None and sz2 is None, "z=None")

    # ★ 慢时钟：每 breath_period 拍才决策一次
    cb = Config()
    cb.life.max_steps = 25
    cb.life.breath_period = 5
    cb.life.enable_write = False
    cb.field.breath_mode = True
    cb.field.breath_k = ph2.BREATH_K
    cb.field.breath_c = ph2.BREATH_C
    cb.interface.split_pass2 = True
    cb.interface.n_layers_pass2 = 1
    cb.interface.n_layers_meta = 1
    cb.interface.meta_pool = 4
    brain_b = Brain(cb, seed=0)
    obs_b = np.zeros(Config().interface.obs_dim, dtype=np.float32)
    obs_b[2] = 0.4
    decisions = []
    for t in range(20):
        _, ib = brain_b.step(obs_b, sample=True, train=False)
        decisions.append(1 if ib["breath_decision"] else 0)
    check("★ 呼吸是慢时钟（每 5 拍决策一次，20 拍共 4 次）",
          sum(decisions) == 4,
          f"决策拍 = {[i + 1 for i, d in enumerate(decisions) if d]}")

    # ★ 呼吸真的改变能量收支
    cb2 = Config()
    cb2.life.enable_write = False
    cb2.field.breath_mode = True
    cb2.field.breath_k = ph2.BREATH_K
    cb2.field.breath_c = ph2.BREATH_C
    gains = []
    for kk in (0.2, 1.0, 1.8):
        bk = Brain(cb2, seed=0)
        bk._k_breath = float(kk)
        ob = np.zeros(Config().interface.obs_dim, dtype=np.float32)
        ob[2] = 0.45
        _, i2 = bk.heartbeat(ob, sample=False)
        gains.append(i2["meta_gain"])
    check("★ 呼吸深度真的改变能量收支（k=0.2 / 1.0 / 1.8）",
          gains[0] > gains[1] > gains[2],
          " → ".join(f"{x:+.1f}" for x in gains))
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T12  ★ 实验 9：换架构家族（CNN 主干）──")

    from v11_digital_life.interface_cnn import CNNInterface

    def build_cnn(chans, blocks, split=False, breath_ch=16):
        c = Config().interface
        c.arch = "cnn"
        c.cnn_channels = tuple(chans)
        c.cnn_blocks = blocks
        c.cnn_breath_channels = breath_ch
        c.split_pass2 = split
        return CNNInterface(c, grid=Config().field.grid,
                            init_log_std=-1.0, world_obs_dim=Config().interface.obs_dim)

    n1 = build_cnn((16, 32, 64), 1)
    n2 = build_cnn((64, 128, 256), 2)
    n3 = build_cnn((64, 128, 256), 2, split=True)
    p_n1, p_n2, p_n3 = n1.n_params(), n2.n_params(), n3.n_params()

    check("CNN 小臂参数远少于 Transformer 1 层",
          p_n1 < pD / 10, f"N1 {p_n1:,} vs D {pD:,}（{p_n1 / pD:.1%}）")
    check("★ CNN 大臂与 Transformer 2 层参数基本相当（差 <5%）",
          abs(p_n2 - pA) / pA < 0.05,
          f"N2 {p_n2:,} vs A {pA:,}（{p_n2 / pA:.1%}）")
    check("CNN 独立呼吸分支只多很少参数",
          0 < p_n3 - p_n2 < 20000, f"+{p_n3 - p_n2:,}")

    psi12 = torch.randn(Config().field.grid, Config().field.grid,
                        dtype=torch.complex64) * 0.1
    for nm, iface in (("N1", n1), ("N2", n2), ("N3", n3)):
        m, v, s, z, sz = iface.act_all(psi12, wob)
        ok = (m.shape == (1, 2) and v.shape == (1, 1) and s.shape == (1, 2)
              and z is not None and z.shape == (1, 1))
        check(f"CNN {nm} 前向输出形状正确", ok, f"mean{tuple(m.shape)} z{tuple(z.shape)}")

    dpsi = n2.propose(psi12, wob)
    check("CNN propose() 形状与 Transformer 一致",
          dpsi.shape == (1, Config().field.grid, Config().field.grid),
          f"{tuple(dpsi.shape)}")

    m4, v4, s4, z4, _ = n2.act_all(psi12, wob, need_breath=False)
    check("CNN need_breath=False 时跳过呼吸分支", z4 is None, "z=None")

    # ★ 平移近似不变：全局输出的动作不该依赖绝对位置
    #   （Transformer 有 1M 参数的位置表，本质上会依赖绝对位置）
    psi_shift = torch.roll(psi12, shifts=32, dims=1)
    m_a, _, _, _, _ = n2.act_all(psi12, wob)
    m_b, _, _, _, _ = n2.act_all(psi_shift, wob)
    rel = float((m_a - m_b).abs().max() / (m_a.abs().max() + 1e-9))
    check("★ CNN 的动作对平移近似不变（滑动 32 格）", rel < 0.15,
          f"相对变化 {rel:.3f}")

    # 注意：这里**不**去测「Transformer 对平移更敏感」——
    #   实测未训练的 Transformer 对平移也几乎不敏感（相对变化 0.000），
    #   因为均值池化 + 近似线性的初始化本身就抹掉了平移差异。
    #   位置表的影响要在**训练之后**才体现出来，自检里测不到，不要写一条
    #   看起来深刻、其实在初始化态下没有信息的断言。
    #   关于位置表的结论只基于**结构事实**：它占 83.9% 的参数（见下一条）。

    # ★ 参数量里位置表占多少（Transformer）
    pos_params = int(iA.pos.numel())
    check("★ Transformer 的参数量里有 84% 是绝对位置查表",
          0.80 < pos_params / pA < 0.88,
          f"pos {pos_params:,} / 总 {pA:,} = {pos_params / pA:.1%}")

    # ★ 解析算力：CNN 比一层 Transformer 便宜一个数量级以上
    from v11_digital_life.ab_test import MACS_TF_LAYER, cnn_macs
    r2 = cnn_macs((64, 128, 256), 2) / MACS_TF_LAYER
    r1 = cnn_macs((16, 32, 64), 1) / MACS_TF_LAYER
    check("★ CNN 大臂的解析算力 < 1 层 Transformer 的 5%", r2 < 0.05,
          f"{r2:.4f} 层/拍")
    check("CNN 小臂的解析算力 < 1 层 Transformer 的 1%", r1 < 0.01,
          f"{r1:.5f} 层/拍")

    # CNN 也能进完整的心跳 + 训练闭环
    cb3 = Config()
    cb3.life.max_steps = 6
    cb3.life.enable_write = False
    cb3.field.breath_mode = True
    cb3.field.breath_k = ph2.BREATH_K
    cb3.field.breath_c = ph2.BREATH_C
    cb3.interface.arch = "cnn"
    cb3.interface.cnn_channels = (16, 32, 64)
    cb3.interface.cnn_blocks = 1
    brain_c = Brain(cb3, seed=0)
    ob_c = np.zeros(Config().interface.obs_dim, dtype=np.float32)
    ob_c[2] = 0.5
    ok_run = True
    detail_c = ""
    try:
        for _ in range(8):
            _, ic = brain_c.step(ob_c, sample=True, train=True)
    except Exception as e:  # noqa: BLE001
        ok_run = False
        detail_c = f"{type(e).__name__}: {e}"
    check("CNN 能进完整心跳 + A2C 训练闭环", ok_run, detail_c or "8 拍无异常")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T13  ★ 实验 10：修「过度呼吸陷阱」（呼吸头的可微目标）──")

    # ① 闭式：最优增益 → 稳态能量正好是生存带中心
    #   ★ 实验 12 之后这两个函数必须显式给 dt、并选 γ 模式。
    #     这里用 mode="coef"（旧的错误公式）把 T13 保持成实验 10 的回归测试；
    #     正确公式的测试在 T15。
    _c0, _dt0 = Config(), Config().field.dt
    fcfg2 = _c0.field.dissipation
    g_opt = ph2.breath_optimal_gain(fcfg2, 10000.0, _dt0, mode="coef")
    e_inf = ph2.breath_steady_energy(g_opt, fcfg2, _dt0, mode="coef")
    check("★ 最优呼吸增益的稳态能量 = 生存带中心",
          abs(e_inf / 10000.0 - ph2.BAND_CENTER) < 1e-6,
          f"gain={g_opt:.1f} → E∞={e_inf:.1f}（{e_inf / 10000:.3f}×E₀）")

    # ② 数值扫描：稳态能量目标的最优 k 就是 physics 给的 k*(light)
    ok_k = True
    det_k = ""
    for L in (0.4, 0.6, 0.8):
        ks = torch.linspace(ph2.BREATH_MIN, ph2.BREATH_MAX, 4001)
        Ls = torch.full_like(ks, L)
        losses = ph2.breath_homeostasis_loss(Ls, ks, fcfg2, 10000.0, _dt0,
                                             mode="coef")
        k_best = float(ks[int(torch.argmin(losses))])
        # 闭式解：解 K·L·k − C·k² = gain_optimal 的那个较小根。
        # ★ 如果判别式 < 0，说明这个光强下**最大增益都达不到目标**
        #   （比如 light=0.6 时最大增益只有 91，目标需要 161.6）——
        #   此时最优解就是最大增益点 k=y 即 K·L/(2C)，不是无穷大。
        import math as _m
        K, Cc = ph2.BREATH_K, ph2.BREATH_C
        disc = (K * L) ** 2 - 4 * Cc * g_opt
        if disc >= 0:
            k_close = ((K * L) - _m.sqrt(disc)) / (2 * Cc)
        else:
            k_close = min(ph2.BREATH_MAX, max(ph2.BREATH_MIN, K * L / (2 * Cc)))
        if not (abs(k_best - k_close) < 0.02):
            ok_k = False
            det_k = f"light={L}: 扫描 {k_best:.3f} vs 闭式 {k_close:.3f}"
    check("★ 数值扫描的最优 k 与闭式解一致（含「目标不可达 → 取最大增益点」分支）",
          ok_k, det_k or "三个光强均一致")

    # ③ 损失对 k 的梯度**真的存在**（不是常数）
    k_t = torch.tensor([0.8, 1.2], requires_grad=True)
    ls_t = torch.tensor([0.55, 0.55])
    l = ph2.breath_homeostasis_loss(ls_t, k_t, fcfg2, 10000.0, _dt0,
                                    mode="coef").mean()
    l.backward()
    check("★ 呼吸目标对 k 可微（梯度非零）",
          k_t.grad is not None and float(k_t.grad.abs().max()) > 1e-6,
          f"dL/dk = {float(k_t.grad[0]):+.4f} / {float(k_t.grad[1]):+.4f}")

    # ④ 两个优化器：参数不重叠，呼吸头学习率独立且大得多
    cb4 = Config()
    cb4.life.enable_write = False
    cb4.field.breath_mode = True
    cb4.field.breath_k = ph2.BREATH_K
    cb4.field.breath_c = ph2.BREATH_C
    cb4.interface.arch = "cnn"
    cb4.interface.cnn_channels = (16, 32, 64)
    cb4.interface.cnn_blocks = 1
    b4 = Brain(cb4, seed=0)
    main_ids = {id(p) for p in b4._main_params()}
    br_ids = {id(p) for p in b4._breath_params()}
    check("主干与呼吸头的优化器参数不重叠",
          len(main_ids & br_ids) == 0 and len(br_ids) > 0,
          f"主干 {len(main_ids)} 个张量 / 呼吸头 {len(br_ids)} 个")
    lrs = [g["lr"] for g in b4.opt_breath.param_groups]
    check("★ 呼吸头有独立且大得多的学习率",
          abs(lrs[0] - cb4.life.lr_breath) < 1e-12 and lrs[0] > 10 * cb4.life.lr,
          f"呼吸 {lrs[0]:.4f} vs 主干 {cb4.life.lr:.4f}（{lrs[0] / cb4.life.lr:.0f} 倍）")

    # ⑤ 呼吸的探索噪声比主干小
    check("呼吸的探索噪声明显比主干小",
          cb4.life.init_log_std_breath < cb4.life.init_log_std - 0.5,
          f"σ_breath={float(torch.exp(torch.tensor(cb4.life.init_log_std_breath))):.3f} "
          f"vs σ_main={float(torch.exp(torch.tensor(cb4.life.init_log_std))):.3f}")

    # ⑥ ★ 端到端：同一配置下，policy 目标学不动 k，homeostasis 目标学得动
    def run_breath_traj(objective: str, steps: int = 200):
        c = Config()
        c.life.enable_write = False
        c.field.breath_mode = True
        c.field.breath_k = ph2.BREATH_K
        c.field.breath_c = ph2.BREATH_C
        c.life.breath_objective = objective
        # ★ 实验 12 之后默认行为变了；T13 要测的是「目标」本身，
        #   所以这里钉住实验 10 的旧行为（旧 γ + 全开动作区间）。
        c.life.breath_gamma_mode = "coef"
        c.life.breath_limit_useful = False
        c.life.max_steps = steps + 10
        c.interface.arch = "cnn"
        c.interface.cnn_channels = (16, 32, 64)
        c.interface.cnn_blocks = 1
        bb = Brain(c, seed=0)
        ob = np.zeros(Config().interface.obs_dim, dtype=np.float32)
        ob[2] = 0.55
        got = []
        for _ in range(steps):
            _, ii = bb.step(ob, sample=True, train=True)
            got.append(float(ii["k_breath"]))
        return float(np.mean(got[-40:])), float(bb.interface.breath_head.bias)

    k_pol, _ = run_breath_traj("policy")
    k_hom, _ = run_breath_traj("homeostasis")
    check("★ policy 目标：呼吸深度学不动（卡在初值附近）",
          k_pol > 1.3, f"末段均 k = {k_pol:.3f}（初值 1.50）")
    check("★★ homeostasis 目标：呼吸深度显著下降",
          k_hom < 0.95, f"末段均 k = {k_hom:.3f}（目标约 0.62）")
    check("★★ 修复让呼吸深度下降 ≥ 0.45",
          (k_pol - k_hom) > 0.45, f"Δk = {k_pol - k_hom:.3f}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T14  ★ 实验 11：种群 + 变异 + 选择（进化机制）──")
    from v11_digital_life import physics as phy
    from v11_digital_life.config import EvolutionConfig
    from v11_digital_life.evolution import (
        Population, clone_genome, diversity, genome_distance, mutate,
        select_parents)

    # ① ★ 世界 v2：光源会移动（这是整个实验的「压力」来源）
    cw = Config()
    w_fix = SandboxWorld(cw, seed=0, sigma=250.0)
    check("固定光源：中心不随时间变化",
          abs(w_fix.light_center(0.0) - w_fix.light_center(500.0)) < 1e-9)
    w_mov = SandboxWorld(cw, seed=0, sigma=250.0,
                         light_amp=300.0, light_period=150.0)
    c0 = w_mov.light_center(0.0)
    cq = w_mov.light_center(150.0 / 4)
    cfull = w_mov.light_center(150.0)
    check("★ 移动光源：四分之一周期到最远点", abs(cq - (c0 + 300.0)) < 1e-6,
          f"{c0:.1f} → {cq:.1f}")
    check("移动光源：一个周期后回到起点", abs(cfull - c0) < 1e-6, f"{cfull:.1f}")
    # 走一步世界，光心应该真的移动了
    w_mov.reset()
    _x_before = w_mov.light_center()
    w_mov.step(np.zeros(2))
    check("走一拍之后光心位置改变（世界真的有随时间变化的资源）",
          abs(w_mov.light_center() - _x_before) > 1e-6,
          f"{_x_before:.2f} → {w_mov.light_center():.2f}")

    # ② 变异：形状、可复现、按张量自身尺度定标
    g_small = {"w": torch.full((8, 8), 0.10), "b": torch.full((64,), 0.95)}
    g_mut = mutate(g_small, 0.10, seed=3)
    check("变异保持形状与 dtype",
          all(g_mut[k].shape == g_small[k].shape and g_mut[k].dtype == g_small[k].dtype
              for k in g_small))
    check("变异确实改变了数值", genome_distance(g_small, g_mut) > 0.01,
          f"相对距离 {genome_distance(g_small, g_mut):.4f}")
    g_mut2 = mutate(g_small, 0.10, seed=3)
    check("变异可复现（同 seed → 同基因）",
          genome_distance(g_mut, g_mut2) < 1e-12)
    rw = float((g_mut["w"] - g_small["w"]).abs().mean() / 0.10)
    rb = float((g_mut["b"] - g_small["b"]).abs().mean() / 0.95)
    check("★ 变异幅度按张量自身 RMS 定标（0.10 和 0.95 两个尺度同样被扰动 10%）",
          abs(rw - rb) < 0.02, f"小尺度 {rw:.4f} vs 大尺度 {rb:.4f}")
    check("基因组与自身的距离为 0", genome_distance(g_small, g_small) < 1e-12)
    check("相同基因组的种群多样性为 0",
          diversity([clone_genome(g_small) for _ in range(4)]) < 1e-12)

    # ③ 选择
    fit = [10.0, 50.0, 30.0, 90.0, 20.0]
    par = select_parents(fit, 2, np.random.default_rng(0), mode="fitness")
    check("适应度选择 = 前 n 名", set(par) == {1, 3}, f"{par}")
    rng2 = np.random.default_rng(0)
    rnd = [tuple(select_parents(fit, 2, rng2, mode="random")) for _ in range(50)]
    check("随机选择确实随机（对照组不是假的对照）", len(set(rnd)) > 5,
          f"50 次抽到 {len(set(rnd))} 种组合")

    # ④ 基因组接口：往返一致 + 清空 Adam
    cb5 = Config()
    cb5.life.max_steps = 12
    cb5.life.enable_write = False
    cb5.field.breath_mode = True
    cb5.field.breath_k = phy.BREATH_K
    cb5.field.breath_c = phy.BREATH_C
    cb5.interface.arch = "cnn"
    cb5.interface.cnn_channels = (16, 32, 64)
    cb5.interface.cnn_blocks = 1
    b5 = Brain(cb5, seed=0)
    gA = b5.get_genome()
    psi5 = b5.field.obs_tensor()
    world5 = torch.zeros(1, cb5.interface.obs_dim, dtype=torch.float32, device=b5.device)
    with torch.no_grad():
        m1, v1, s1 = b5.interface.act(psi5, world5)
    b5.set_genome(mutate(gA, 0.2, seed=9))      # 换成别的基因
    b5.set_genome(gA)                            # 再换回来
    with torch.no_grad():
        m2, v2, s2 = b5.interface.act(b5.field.obs_tensor(), world5)
    check("★ 基因组往返：同一个基因组给出同一个动作",
          float((m1 - m2).abs().max()) < 1e-6,
          f"最大差 {float((m1 - m2).abs().max()):.2e}")
    check("set_genome 清空 Adam 状态（动量是记忆、不是基因）",
          len(b5.optimizer.state) == 0 and len(b5.opt_breath.state) == 0)

    # ⑤ 种群：初始多样性、繁殖、精英保留
    cp = Config()
    cp.life.max_steps = 20
    cp.life.enable_write = False
    cp.field.breath_mode = True
    cp.field.breath_k = phy.BREATH_K
    cp.field.breath_c = phy.BREATH_C
    cp.interface.arch = "cnn"
    cp.interface.cnn_channels = (8, 16, 32)
    cp.interface.cnn_blocks = 1
    ev = EvolutionConfig(population=4, generations=2, mutation_rate=0.05,
                         selection="fitness", inheritance="darwinian")
    pop = Population(cp, ev, seed=0,
                     world_kw=dict(sigma=250.0, light_amp=300.0, light_period=150.0))
    check("初始种群个体互不相同（选择的原料）", diversity(pop.genomes) > 0.5,
          f"多样性 {diversity(pop.genomes):.3f}")
    birth0 = [clone_genome(g) for g in pop.genomes]
    r1 = pop.run_generation(1, train=False)
    check("一代跑完，记录完整",
          all(k in r1 for k in ("fitness", "best", "mean", "worst", "diversity")))
    check("一代之后种群数量不变", len(pop.genomes) == 4)
    check("适应度有序：best ≥ mean ≥ worst",
          r1["best"] >= r1["mean"] >= r1["worst"],
          f"{r1['best']:.0f} / {r1['mean']:.1f} / {r1['worst']:.0f}")
    best_i = int(np.argmax(r1["fitness"]))
    check("★ 精英个体原样保留（不被变异破坏）",
          genome_distance(pop.genomes[0], birth0[best_i]) < 1e-12)
    check("★ 后代带有变异（不是父代的复印件）",
          genome_distance(pop.genomes[-1], birth0[best_i]) > 1e-4)
    check("★ 同一代所有个体共用同一个世界（精英不会被换到另一条赛道）",
          pop.world_seed(0, 5) == pop.world_seed(3, 5)
          and pop.world_seed(0, 5) != pop.world_seed(0, 6),
          f"gen5 全部 = {pop.world_seed(0, 5)}，gen6 = {pop.world_seed(0, 6)}")
    print()

    # ══════════════════════════════════════════════════════════════
    print("── T15  ★ 实验 12/13：呼吸目标（稳态公式 + 两个解 + 运动耗能）──")
    import math as _math
    math = _math
    c12 = Config()
    gam, dt12 = c12.field.dissipation, c12.field.dt
    e0 = 10000.0

    # ① ★★★ 最重要的一个：**实测**场的能量衰减，去核对公式
    #   field.step() 里 psi *= exp(-γ·dt)，而 E = Σ|ψ|²，
    #   所以能量乘子是振幅乘子的平方 exp(-2γ·dt)。这一步错了会差 2 倍。
    fn = NLSEField(c12.field, torch.device("cpu"), seed=0)
    fn.cfg = c12.field
    _f2 = NLSEField(c12.field, torch.device("cpu"), seed=0)
    _e0, _N = _f2.energy(), 200
    for _ in range(_N):
        _f2.step()                      # 只演化，不加噪声 / 不做代谢
    _ratio = _f2.energy() / _e0
    check("★★ 实测能量衰减 = exp(−2γ·dt)（能量 ∝ 振幅²，不是 exp(−γ·dt)）",
          abs(_ratio - math.exp(-2 * gam * dt12 * _N)) < 2e-3,
          f"实测 {_ratio:.6f} vs exp(−2·γ·dt·N) = "
          f"{math.exp(-2 * gam * dt12 * _N):.6f}")
    check("★ 那个「差 1.5%」的旧公式 (1−γ)/γ 其实够用（不是 2 倍错）",
          abs((1 - gam) / gam / (1 / (1 - math.exp(-2 * gam * dt12))) - 1) < 0.03,
          f"旧 99.0 vs 真 100.5（差 "
          f"{abs(99.0 / (1 / (1 - math.exp(-2 * gam * dt12))) - 1):.2%}）")

    # ② 目标增益：正确 159.2，旧公式 161.6，差 1.5%
    g_step = ph2.breath_optimal_gain(gam, e0, dt12, mode="step")
    g_coef = ph2.breath_optimal_gain(gam, e0, dt12, mode="coef")
    check("★ 正确的目标增益 = 159.2（旧公式 161.6，只差 1.5%）",
          abs(g_step - 159.2) < 0.5 and abs(g_coef / g_step - 1) < 0.03,
          f"step={g_step:.2f}  旧={g_coef:.2f}  比值={g_coef / g_step:.4f}")
    e_step = ph2.breath_steady_energy(g_step, gam, dt12, mode="step")
    check("两种模式下「目标增益 → 稳态 = 生存带中心」都自洽",
          abs(e_step / e0 - ph2.BAND_CENTER) < 1e-6,
          f"step: E∞ = {e_step / e0:.4f}×E₀")

    # ③ 有用上限就是最大增益点；区间内增益单调递增
    L0 = 0.62
    k_useful = ph2.breath_useful_max(L0)
    ks_line = torch.linspace(ph2.BREATH_MIN, k_useful, 500)
    gains = ph2.BREATH_K * L0 * ks_line - ph2.BREATH_C * ks_line * ks_line
    check("★ 有用上限 = 最大增益点 k* = K·L/(2C)",
          abs(k_useful - ph2.breath_optimal_k(L0)) < 1e-9,
          f"k_useful={k_useful:.4f}  k*={ph2.breath_optimal_k(L0):.4f}")
    check("★ 有用区间内呼吸增益单调递增（目标因此一一可解）",
          bool(torch.all(torch.diff(gains) > 0)),
          f"g({ph2.BREATH_MIN:.2f})={float(gains[0]):.1f} → "
          f"g({k_useful:.2f})={float(gains[-1]):.1f}")

    # ④ ★ 旧目标（动作区间全开）在亮光下有两个极小 —— 这是**真正的** bug
    def _local_minima(losses, kk, sep: float = 0.05) -> list:
        v = losses.detach().cpu().numpy()
        out = []
        for i in range(1, len(v) - 1):
            if v[i] < v[i - 1] and v[i] < v[i + 1]:
                k_i = float(kk[i])
                if out and abs(k_i - out[-1]) < sep:
                    continue
                out.append(k_i)
        return out

    Lb = 0.9
    ks_all = torch.linspace(ph2.BREATH_MIN, ph2.BREATH_MAX, 2001)
    old_loss = ph2.breath_homeostasis_loss(torch.full_like(ks_all, Lb), ks_all,
                                           gam, e0, dt12, mode="step")
    mins_old = _local_minima(old_loss, ks_all)
    check("★★ 动作区间全开时，目标在亮光下有两个极小（浅解 + 深解）—— 真 bug",
          len(mins_old) >= 2,
          f"light=0.9 的极小：{[round(m, 3) for m in mins_old]}")

    # ⑤ 限制在单调区间后只剩一个极小，且等于闭式浅解
    ks_use = torch.linspace(ph2.BREATH_MIN, ph2.breath_useful_max(Lb), 2001)
    new_loss = ph2.breath_homeostasis_loss(torch.full_like(ks_use, Lb), ks_use,
                                           gam, e0, dt12, mode="step")
    mins_new = _local_minima(new_loss, ks_use)
    k_th = ph2.breath_shallow_k(Lb, gam, e0, dt12, mode="step")
    check("★★ 限制在单调区间后目标唯一，且等于闭式浅解",
          len(mins_new) == 1 and abs(mins_new[0] - k_th) < 0.03,
          f"数值 {[round(m, 3) for m in mins_new]} vs 闭式 {k_th:.3f}")

    # ⑥ 理论：目标可达时，光越亮呼吸越浅
    k_07 = ph2.breath_shallow_k(0.70, gam, e0, dt12, mode="step")
    k_10 = ph2.breath_shallow_k(1.00, gam, e0, dt12, mode="step")
    check("★ 理论预测：光越亮、呼吸越浅（同一个稳态目标）",
          k_10 < k_07,
          f"light0.70 → k={k_07:.3f}；light1.00 → k={k_10:.3f}")

    # ⑦ 重参数化：值域正确，且 dk/dz 处处非零（不能用夹紧）
    zs_t = torch.tensor([[-5.0], [-0.5], [0.0], [0.95], [5.0]], requires_grad=True)
    for L_ in (0.62, 0.90):
        kk = ph2.breath_map_range(zs_t, L_)
        hi = ph2.breath_useful_max(L_)
        in_range = bool(torch.all(kk >= ph2.BREATH_MIN - 1e-6)) and \
            bool(torch.all(kk <= hi + 1e-6))
        kk.sum().backward(retain_graph=True)
        gz = zs_t.grad.detach().clone()
        zs_t.grad.zero_()
        check(f"★ 重参数化：k 落在 [MIN, k*(light={L_})] 内", in_range,
              f"k ∈ [{float(kk.min()):.3f}, {float(kk.max()):.3f}]，上界 {hi:.3f}")
        check(f"★ 重参数化：dk/dz 处处非零（含 z=±5，light={L_}）",
              bool(torch.all(gz.abs() > 1e-6)),
              f"|dk/dz| 最小 {float(gz.abs().min()):.2e}")

    # ⑧ 端到端：默认配置下 k 收敛到理论浅解
    def run_fixed(light: float, account_move: bool, steps: int = 300):
        c = Config()
        c.life.enable_write = False
        c.field.breath_mode = True
        c.field.breath_k = ph2.BREATH_K
        c.field.breath_c = ph2.BREATH_C
        c.life.breath_objective = "homeostasis"
        c.life.breath_account_move = account_move
        c.life.max_steps = steps + 10
        c.interface.arch = "cnn"
        c.interface.cnn_channels = (16, 32, 64)
        c.interface.cnn_blocks = 1
        bb = Brain(c, seed=0)
        ob = np.zeros(Config().interface.obs_dim, dtype=np.float32)
        ob[2] = light
        got = []
        for _ in range(steps):
            _, ii = bb.step(ob, sample=True, train=True)
            got.append(float(ii["k_breath"]))
        return float(np.mean(got[-40:])), float(bb.field.norm_frac())

    L_e = 0.95                          # 亮光下目标才可达（能验证真正的稳态控制）
    k_fix, norm_fix = run_fixed(L_e, False)
    k_pred = ph2.breath_shallow_k(L_e, gam, e0, dt12, mode="step")
    check("★★ 亮光下 k 收敛到**理论浅解**（±0.10）",
          abs(k_fix - k_pred) < 0.10,
          f"实测 k={k_fix:.3f}，理论 {k_pred:.3f}")
    check("修好之后 k 在有用区间内（不再跑到深解 1.1+）",
          k_fix <= ph2.breath_useful_max(L_e) + 1e-6,
          f"k={k_fix:.3f} ≤ k*={ph2.breath_useful_max(L_e):.3f}")
    check("修好之后能量停在生存带内（没饿死 / 没撑死）",
          ph2.LOWER < norm_fix < ph2.UPPER,
          f"末能量 {norm_fix:.3f}×E₀")

    # ⑨ 运动耗能必须进能量预算
    c_move = 60.0
    k_nomove = ph2.breath_shallow_k(L_e, gam, e0, dt12, mode="step")
    k_move = ph2.breath_shallow_k(L_e, gam, e0, dt12, mode="step",
                                  move_cost=c_move)
    kk_ = torch.linspace(ph2.BREATH_MIN, ph2.breath_useful_max(L_e), 4001)
    loss_m = ph2.breath_homeostasis_loss(
        torch.full_like(kk_, L_e), kk_, gam, e0, dt12, mode="step",
        move_cost=c_move)
    k_argmin = float(kk_[int(torch.argmin(loss_m))])
    check("★ 运动耗能进预算：每拍多烧 60 → 目标 k 从 0.48 加深到 0.85",
          k_move > k_nomove + 0.2 and abs(k_argmin - k_move) < 0.02,
          f"k {k_nomove:.3f} → {k_move:.3f}（数值 argmin {k_argmin:.3f}）")
    print()

    # ══════════════════════════════════════════════════════════════
    print("=" * 70)
    n_ok = sum(1 for _, ok, _ in _results if ok)
    n_all = len(_results)
    print(f"  结果：{n_ok} / {n_all} 通过　　总耗时 {time.time() - t_start:.1f} 秒")
    if n_ok < n_all:
        print("  失败项：")
        for name, ok, detail in _results:
            if not ok:
                print(f"    - {name}  {detail}")
    print("=" * 70)
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
