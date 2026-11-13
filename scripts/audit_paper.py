#!/usr/bin/env python
"""audit_paper.py —— 论文数字的统一重算与审计（V11 部分）。

FCL 部分（0816 的 Table 1-5）由 `scripts/audit_submission_0816.py` 负责，
这里只覆盖 V11 的机制/难度窗口结果。

设计原则
--------
1. **每条主张都必须能指回一个原始文件**。没有原始文件的，标记为 NOT-AUDITED。
2. **配对数据用配对检验**。实验 14 的初稿在这里错过一次（对配对数据用了
   非配对置换检验，把 p=0.125 报成了 0.032），所以这里所有跨臂比较都走
   `paired_t_two_sided` + `exact_sign_flip`。
3. **跨代比较用世界均衡窗口**（gen9-15 vs gen1-7），不用 fitness_first/last
   —— 训练集世界按 `gen % 8` 轮换，首末代不是同一个世界。

用法
----
    python scripts/audit_paper.py
    python scripts/audit_paper.py --out paper_alife/AUDIT.md
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# 交付包里脚本在 code/scripts/，数据在包根 data/；仓库里两者同级。
# 找不到 ROOT/data/V11 时退回上一级，保证两种布局都能跑。
if not (ROOT / "data" / "V11").exists() and (ROOT.parent / "data" / "V11").exists():
    ROOT = ROOT.parent

CHECKS: list[dict] = []
SKIPPED: list[str] = []


# ══════════════════════════════════════════════════════════════════════
# 统计工具（无 scipy，全部自实现并已对数值积分验证）
# ══════════════════════════════════════════════════════════════════════

def _betacf(a: float, b: float, x: float, itmax: int = 300,
            eps: float = 3e-14) -> float:
    """不完全 beta 的连分式（Numerical Recipes 6.4）。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """正则化不完全 beta I_x(a, b)，带对称关系。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def paired_t_two_sided(diff: np.ndarray) -> tuple[float, float]:
    """配对 t 双侧 p（df = n-1）。"""
    n = len(diff)
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        return float("inf"), 0.0
    t = float(diff.mean()) / (sd / math.sqrt(n))
    df = n - 1
    p = betainc_reg(df / 2.0, 0.5, df / (df + t * t))
    return t, p


def exact_sign_flip(diff: np.ndarray) -> float:
    """精确配对符号翻转检验的双侧 p（穷举 2^n 种符号）。"""
    n = len(diff)
    if n > 20:
        raise ValueError("2^n 太大，改用正态近似")
    obs = abs(float(diff.mean()))
    cnt = 0
    for mask in range(1 << n):
        total = 0.0
        for i in range(n):
            total += (-diff[i] if (mask >> i) & 1 else diff[i])
        if abs(total / n) >= obs - 1e-9:
            cnt += 1
    return cnt / float(1 << n)


def check(name: str, claimed, computed, ok: bool, note: str = "") -> None:
    CHECKS.append(dict(name=name, claimed=claimed, computed=computed,
                       ok=bool(ok), note=note))


# ══════════════════════════════════════════════════════════════════════
# 数据装载
# ══════════════════════════════════════════════════════════════════════

def load_results(pattern: str) -> dict:
    """读 exp11_evolution 的聚合 JSON（顶层 'results'）。"""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    merged: dict = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            merged.update(json.load(fh).get("results", {}))
    return merged


def vals(R: dict, world: str, arm: str, key: str) -> np.ndarray:
    ks = sorted(k for k in R if k.startswith(f"{world}:{arm}:"))
    return np.array([R[k][key] for k in ks], float)


def worlds_of(R: dict, arm: str) -> list[str]:
    return sorted({k.split(":")[0] for k in R if f":{arm}:" in k})


# ══════════════════════════════════════════════════════════════════════
# A. 实验 11：世界 B 上的选择效应
# ══════════════════════════════════════════════════════════════════════

def section_a() -> None:
    print("[A] 实验 11：选择效应（世界 B）")
    R = load_results(str(ROOT / "data/V11/exp11/exp11_*.json"))
    x = vals(R, "B", "EVO_NL", "held_pool_mean")
    y = vals(R, "B", "NONE_NL", "held_pool_mean")
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    d = x - y
    t, p = paired_t_two_sided(d)
    ps = exact_sign_flip(d)

    print(f"   EVO_NL {x.mean():.1f} vs NONE_NL {y.mean():.1f}  "
          f"差 {d.mean():+.1f}  n={n}  {int((d > 0).sum())}/{n} 为正")
    print(f"   配对 t={t:.2f} (df={n-1}, p={p:.4f})  符号翻转 p={ps:.4f}")

    check("A1 基因池差", "+27.2", f"{d.mean():+.1f}", abs(d.mean() - 27.2) < 0.5,
          "README/实验11 声称 +17.6")
    check("A2 同向种子数", "7/7", f"{int((d > 0).sum())}/{n}",
          int((d > 0).sum()) == 7 and n == 7)
    check("A3 配对 t", "7.62", f"{t:.2f}", abs(t - 7.62) < 0.3)
    check("A4 符号翻转 p", "0.016", f"{ps:.4f}", abs(ps - 0.0156) < 0.002,
          "7/7 同向时精确双侧 p = 2/128 = 0.0156")

    # 多样性坍缩（最干净的机械信号）
    sel_div = vals(R, "B", "EVO_NL", "diversity_last")
    non_div = vals(R, "B", "NONE_NL", "diversity_last")
    print(f"   末代多样性：有选择 {sel_div.mean():.3f}  无选择 {non_div.mean():.3f}")
    check("A5 多样性坍缩", "0.08 vs 0.98",
          f"{sel_div.mean():.3f} vs {non_div.mean():.3f}",
          sel_div.mean() < 0.2 and non_div.mean() > 0.9)


# ══════════════════════════════════════════════════════════════════════
# B. 实验 13：世界 B 修好呼吸之后饱和
# ══════════════════════════════════════════════════════════════════════

def section_b() -> None:
    print("\n[B] 实验 13：世界 B 的饱和")
    for d in sorted((ROOT / "data/V11/exp13").glob("B_EVO_NL_seed*")):
        p = d / "generations.csv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        g2 = next((r for r in rows if int(r["gen"]) == 2), None)
        if g2 is None:
            continue
        n_surv = int(g2["n_survived"])
        pop = int(round(int(g2["lives"]) / max(1, int(g2["gen"]))))
        std = float(g2["std"])
        print(f"   {d.name}: gen2 活满 {n_surv}/{pop}  mean={float(g2['mean']):.0f}  std={std:.2f}")
        # ★ 修好 sandbox 之后只有 seed0 完全饱和，seed1 留着活的方差。
        #   两个种子分开记，不能再用「全活满 / 方差 0」统一断言。
        _tgt = {"B_EVO_NL_seed0": (8, 0.00), "B_EVO_NL_seed1": (6, 9.11)}
        want = _tgt.get(d.name)
        if want:
            check(f"B {d.name} gen2 活满数", f"{want[0]}/{pop}", f"{n_surv}/{pop}",
                  n_surv == want[0])
            check(f"B {d.name} gen2 方差", f"{want[1]:.2f}", f"{std:.2f}",
                  abs(std - want[1]) < 0.01,
                  "seed0 完全饱和（方差 0，选择没有原料）；seed1 未饱和")


# ══════════════════════════════════════════════════════════════════════
# C. 实验 14：世界 C
# ══════════════════════════════════════════════════════════════════════

def section_c() -> None:
    print("\n[C] 实验 14：世界 C（5 seeds）")
    R = load_results(str(ROOT / "data/V11/exp13_fixed/exp11_*.json"))
    arms = ["EVO_NL", "NONE_NL", "EVO", "LAM"]

    # C1 生存：从 individuals.csv 数活满
    tot = surv = 0
    causes: dict[str, int] = {}
    for d in sorted((ROOT / "data/V11/exp13_fixed").glob("C_*_seed*")):
        p = d / "individuals.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            tot += 1
            surv += int(r["survived"])
            causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    print(f"   生存：{surv}/{tot} 活满  死因={causes}")
    check("C1 世界 C 存活率", "0/2400", f"{surv}/{tot}", tot == 2400 and surv == 0)

    # C2 双峰
    print(f"   {'臂':<9} {'≥340':>6} {'200-250':>8} {'均值':>7}")
    for a in arms:
        bs = np.array([b for k in R if k.startswith(f"C:{a}:")
                       for b in R[k]["held_learned_beats"]], float)
        hi = 100 * (bs >= 340).mean()
        lo = 100 * ((bs >= 200) & (bs < 250)).mean()
        print(f"   {a:<9} {hi:>5.1f}% {lo:>7.1f}% {bs.mean():>7.1f}")

    # C3 配对检验（正确口径）
    print(f"\n   {'对比':<20} {'差':>7} {'配对t':>7} {'t-p':>7} {'符号p':>7}")
    for a, b in [("EVO", "EVO_NL"), ("EVO", "NONE_NL"), ("EVO", "LAM"),
                 ("EVO_NL", "NONE_NL")]:
        x, y = vals(R, "C", a, "held_learned_mean"), vals(R, "C", b, "held_learned_mean")
        d = x - y
        t, p = paired_t_two_sided(d)
        ps = exact_sign_flip(d)
        print(f"   {a+' vs '+b:<20} {d.mean():>+7.1f} {t:>7.2f} {p:>7.3f} {ps:>7.3f}")
        if (a, b) == ("EVO", "LAM"):
            check("C3 EVO vs LAM 配对 t 双侧 p", "0.016", f"{p:.3f}",
                  abs(p - 0.0161) < 0.006,
                  "初稿曾误报 0.032（非配对置换）")
            check("C4 EVO vs LAM 符号翻转 p", "0.062", f"{ps:.3f}",
                  abs(ps - 0.062) < 0.01,
                  "n=5 时最小可能值 0.0625，需 5/5 同向")

    # C5 世界均衡的训练提升（不是首末代）
    # ★ 按臂的代数取窗口：LAM 是 120 代，用 gen9–15 只覆盖前 15 代。
    #   这里曾经把 LAM 的 +9.4 印成 −16.4。
    print()
    print(f"   {'臂':<9} {'代数':>5} {'世界均衡提升':>12}")
    for a in arms:
        g = []
        n_gen = 0
        for k in sorted(kk for kk in R if kk.startswith(f"C:{a}:")):
            H = {h["gen"]: h["mean"] for h in R[k]["history"]}
            gens = sorted(H)
            n_gen = len(gens)
            early = [x for x in gens if 1 <= x <= 7]
            late = [x for x in gens if x % 8 in range(1, 8)][-7:]
            g.append(np.mean([H[x] for x in late]) - np.mean([H[x] for x in early]))
        print(f"   {a:<9} {n_gen:>5} {np.mean(g):>+12.1f}")

# ══════════════════════════════════════════════════════════════════════
# D. 世界轮换口径的修正
# ══════════════════════════════════════════════════════════════════════

def section_d() -> None:
    print("\n[D] 世界轮换统计口径")
    R = load_results(str(ROOT / "data/V11/exp13_fixed/exp11_*.json"))
    worlds = set()
    for k in R:
        for h in R[k]["history"]:
            worlds.add(h["gen"] % 8)
    # 只取 15 代的臂（LAM 是 120 代，混进来会错位）
    gens = sorted({h["gen"] for k in R if len(R[k]["history"]) == 15
                   for h in R[k]["history"]})
    k1, k_last = gens[0] % 8, gens[-1] % 8
    print(f"   gen→世界组 k=g%8，出现过的 k：{sorted(worlds)}")
    print(f"   gen{gens[0]}→k={k1}，gen{gens[-1]}→k={k_last}："
          f"{'不是同一个世界' if k1 != k_last else '是同一个世界'}")
    check("D1 首末代不同世界", "k(gen1) ≠ k(gen15)", f"{k1} vs {k_last}",
          k1 != k_last, "所以 fitness_first/fitness_last 不能直接相减")
    check("D2 世界序列覆盖完整一轮", "0–7 都出现", str(len(worlds)),
          len(worlds) == 8)


def section_e() -> None:
    """实验 12b：呼吸动作的单调重参数化是不是决定性修复。

    ★ 注意数据目录：修复后用 `data/V11/exp12b_fix/`（6 臂 × 世界 B，
      固定策略梯度），旧数据在 `data/V11/exp12b/`。
    """
    print("\n[E] 实验 12b：呼吸修复")
    base = ROOT / "data/V11/exp12b_fix"
    if not base.exists():
        print("   数据目录不存在，跳过")
        check("E0 exp12b_fix 数据存在", "存在", "缺失", False)
        return

    data: dict = {}
    for f in sorted(base.glob("*_seed*/summary.json")):
        d = json.load(open(f, encoding="utf-8"))
        data.setdefault(d["world"], {}).setdefault(d["arm"], {})[d["seed"]] = d

    for w in sorted(data):
        print(f"   世界 {w}:")
        for a in ("OLD_FULL", "OLD_MONO", "FIXED"):
            if a not in data[w]:
                continue
            dd = data[w][a]
            v = np.array([dd[s]["mean_beats"] for s in dd], float)
            sv = sum(dd[s]["n_survived"] for s in dd)
            gen = sum(dd[s]["generations"] for s in dd)
            kg = np.array([dd[s]["k_generation_std"] for s in dd], float)
            print(f"     {a:<9} n={len(v):>2} 存活 {v.mean():>6.1f} "
                  f"活满 {sv}/{gen}={100 * sv / gen:>3.0f}%  k跨代σ {kg.mean():.3f}")
            # 10 个种子（0–9）的论文值。实验 12 报告的 3 种子值在括号里。
            if w == "B" and a == "OLD_FULL":
                check("E1 世界B OLD_FULL 存活", "164.0",
                      f"{v.mean():.1f}", abs(v.mean() - 164.0) < 1.0,
                      f"n={len(v)}")
                check("E2 世界B OLD_FULL 活满率", "9%",
                      f"{100 * sv / gen:.0f}%", abs(100 * sv / gen - 9) <= 1)
            if w == "B" and a == "FIXED":
                check("E3 世界B FIXED 存活", "324.1",
                      f"{v.mean():.1f}", abs(v.mean() - 324.1) < 1.0)
                check("E4 世界B FIXED 活满率", "58%",
                      f"{100 * sv / gen:.0f}%", abs(100 * sv / gen - 58) <= 1)
                check("E4b 世界B FIXED k 跨代 σ", "0.051", f"{kg.mean():.3f}",
                      abs(kg.mean() - 0.051) < 0.005)

    # 配对检验（自动用上全部可用种子）
    if "B" in data and "OLD_FULL" in data["B"] and "FIXED" in data["B"]:
        seeds = sorted(set(data["B"]["OLD_FULL"]) & set(data["B"]["FIXED"]))
        a = np.array([data["B"]["OLD_FULL"][s]["mean_beats"] for s in seeds], float)
        b = np.array([data["B"]["FIXED"][s]["mean_beats"] for s in seeds], float)
        d = b - a
        t, p = paired_t_two_sided(d)
        ps = exact_sign_flip(d)
        print(f"\n   FIXED − OLD_FULL（世界 B，n={len(seeds)}）："
              f"{a.mean():.1f} → {b.mean():.1f}（{100 * d.mean() / a.mean():+.0f}%）")
        print(f"   配对 t={t:.2f}  p={p:.4f}  符号翻转 p={ps:.4f}"
              f"（n={len(seeds)} 时的下限 = {2 / 2 ** len(seeds):.4f}）")
        check("E5 呼吸修复的配对 t", "16.34", f"{t:.2f}", abs(t - 16.34) < 0.5)
        check("E6 呼吸修复的符号翻转 p", "0.00195", f"{ps:.5f}",
              abs(ps - 2 / 2 ** len(seeds)) < 1e-9,
              f"n={len(seeds)} 时为下限；10/10 同向")

        # E7：单调重参数化才是决定性修复 —— OLD_MONO 与 FIXED 应当没有差别
        if "OLD_MONO" in data["B"]:
            ss = sorted(set(data["B"]["OLD_MONO"]) & set(data["B"]["FIXED"]))
            mo = np.array([data["B"]["OLD_MONO"][s]["mean_beats"] for s in ss], float)
            fx = np.array([data["B"]["FIXED"][s]["mean_beats"] for s in ss], float)
            dd = fx - mo
            print(f"   OLD_MONO vs FIXED（n={len(ss)}）：{mo.mean():.1f} vs "
                  f"{fx.mean():.1f}  配对差 {dd.mean():+.1f} ± {dd.std(ddof=1):.1f}")
            check("E7 单调区间≈完整修复", "差值 < 5 拍",
                  f"{abs(dd.mean()):.1f} 拍", abs(dd.mean()) < 5.0,
                  "γ 修正与运动耗能是正确性补丁，无可测增益")

    # E8：§6.1 的「修复前基线」来自 exp11（另一批数据），单独核
    M11: dict = {}
    for f in sorted(glob.glob(str(ROOT / "data/V11/exp11/exp11_*.json"))):
        M11.update(json.load(open(f, encoding="utf-8"))["results"])
    ks11 = sorted(k for k in M11 if k.startswith("B:EVO_NL:"))
    if ks11:
        means = [M11[k]["history"][-1]["mean"] for k in ks11]
        full = sum(M11[k]["history"][-1]["n_survived"] for k in ks11)
        print(f"   §6.1 修复前基线（exp11 世界B EVO_NL，n={len(ks11)}）："
              f"末代均值 {min(means):.0f}–{max(means):.0f}，活满 {full}/{8 * len(ks11)}")
        check("E8 修复前基线范围", "40–105", f"{min(means):.0f}–{max(means):.0f}",
              abs(min(means) - 40.1) < 1 and abs(max(means) - 105.4) < 1)
        check("E9 修复前无个体活满", "0/56", f"{full}/{8 * len(ks11)}", full == 0)

    # ── 2×2：把「压到峰值以下」与「上界随光强」分开 ──────────────────────
    # ★ 第六轮修正：原来拿 OLD_FULL 当基线，但它同时缺 γ 修正与运动耗能，
    #   把三件事混在一起（审稿人指出）。补了配置匹配的 FIXED_EXCESS。
    #   现在四格都是 step γ + 计运动耗能，两行的上界倍率都是 k* vs 2k*。
    b2 = {}
    for arm in ["FIXED_EXCESS", "FIXED_BOUND", "TRACK_EXCESS", "FIXED",
                "OLD_FULL", "OLD_MONO"]:
        vals = []
        for sd in range(10):
            f = ROOT / f"data/V11/exp12b_fix/B_{arm}_seed{sd}/summary.json"
            if f.exists():
                vals.append(float(json.load(open(f, encoding="utf-8"))["mean_beats"]))
        if len(vals) == 10:
            b2[arm] = vals
    if len(b2) >= 6:
        def contrast(a, c):
            diff = [x - y for x, y in zip(b2[a], b2[c])]
            return (float(np.mean(diff)),
                    float(np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))),
                    sum(1 for d in diff if d > 0))
        print()
        print("   2×2（世界 B，各 10 种子，四格配置匹配）：")
        for kk, v in b2.items():
            print(f"     {kk:<13} {np.mean(v):6.1f} ± {np.std(v, ddof=1):4.1f}")
        specs = [("FIXED_BOUND", "FIXED_EXCESS", "封顶@固定", 57.1, 10),
                 ("FIXED", "TRACK_EXCESS", "封顶@随光", 17.3, 9),
                 ("TRACK_EXCESS", "FIXED_EXCESS", "锚定@越峰", 95.0, 10),
                 ("FIXED", "FIXED_BOUND", "锚定@封顶", 55.2, 9),
                 ("FIXED_EXCESS", "OLD_FULL", "匹配格−旧配置", 47.8, 10)]
        for i, (a, c, lab, want, wn) in enumerate(specs, start=10):
            m, t, w = contrast(a, c)
            print(f"     {lab:<14} Δ={m:+6.1f}  t={t:+6.2f}  同向 {w}/10")
            check(f"E{i} {lab}", f"{want:+.1f}", f"{m:+.1f}", abs(m - want) < 0.15)
            check(f"E{i}b {lab} 同向", f"{wn}/10", f"{w}/10", w == wn)
        e = {kk: float(np.mean(v)) for kk, v in b2.items()}
        cap = (e["FIXED_BOUND"] + e["FIXED"]) / 2 - (e["FIXED_EXCESS"] + e["TRACK_EXCESS"]) / 2
        trk = (e["TRACK_EXCESS"] + e["FIXED"]) / 2 - (e["FIXED_EXCESS"] + e["FIXED_BOUND"]) / 2
        inter = (e["FIXED"] - e["FIXED_BOUND"]) - (e["TRACK_EXCESS"] - e["FIXED_EXCESS"])
        print(f"     主效应：封顶 {cap:+.1f}   光强锚定 {trk:+.1f}   交互 {inter:+.1f}")
        check("E18 封顶主效应", "+37.2", f"{cap:+.1f}", abs(cap - 37.2) < 0.2)
        check("E19 锚定主效应", "+75.1", f"{trk:+.1f}", abs(trk - 75.1) < 0.2)
        check("E20 交互项", "-39.8", f"{inter:+.1f}", abs(inter + 39.8) < 0.2)


#: 实验 9 的**纳入清单**。必须写死，否则聚合口径会漂。
#:   排除 `exp9fix_smoke`（冒烟跑）与 `exp9fix_long`（30 代，预算不同）。
#:   剩下的都是「8 代 + 固定世界」的同类运行；mlp30 只进架构图。
EXP9_DIRS = [
    "exp9fix", "exp9fix_seed1", "exp9fix_seed2", "exp9fix_a_seed1",
    "exp9fix_fw_seed0", "exp9fix_fw_seed1", "exp9fix_fw_seed2",
    "exp9fix_n7_seed1", "exp9fix_n7_seed2", "exp9fix_n7_seed3",
    "exp9fix_mlp8", "exp9fix_mlp30",
]


def section_f() -> None:
    """实验 9：架构族对照（参数量 vs 存活，8 代固定预算）。"""
    print("\n[F] 实验 9：架构族")
    per_arm: dict[str, list[float]] = {}
    params: dict[str, int] = {}
    cost: dict[str, float] = {}
    names: dict[str, str] = {}
    for dname in EXP9_DIRS:
        for f in sorted((ROOT / "data/V11" / dname).glob("exp8_*.json")):
            for a, rec in json.load(open(f, encoding="utf-8"))["arms"].items():
                per_arm.setdefault(a, []).append(float(rec["mean_beats"]))
                params[a] = int(rec["params"])
                cost.setdefault(a, float(rec.get("rel_cost", 0.0)))
                names[a] = rec.get("desc", a)

    print(f"   {'臂':<4} {'参数':>10} {'n':>3} {'平均存活':>12}")
    for a in sorted(per_arm, key=lambda x: -params[x]):
        v = np.array(per_arm[a], float)
        print(f"   {a:<4} {params[a]:>10,} {len(v):>3} {v.mean():>7.1f}±{v.std(ddof=1) if len(v) > 1 else 0:>4.1f}")

    if "N1" in per_arm:
        v = np.array(per_arm["N1"], float)
        check("F1 35k CNN 的存活", "132.6", f"{v.mean():.1f}",
              abs(v.mean() - 132.6) < 5 and len(v) == 9,
              "8 代固定世界，纳入清单见 EXP9_DIRS")
        check("F2 35k CNN 参数量", "35,402", f"{params['N1']:,}", params["N1"] == 35402)
    if "A" in per_arm and "D" in per_arm:
        check("F3 Transformer 位置表占比", "84%",
              f"{1048640 / params['A'] * 100:.1f}%",
              abs(1048640 / params["A"] * 100 - 83.4) < 1.0,
              "pos = 16,385 × 64 = 1,048,640")

    # 诚实性：A 只有 2 个种子，必须在论文里标出来
    if "A" in per_arm:
        check("F4 A 臂的样本量", "n=2", f"n={len(per_arm['A'])}",
              len(per_arm["A"]) == 2, "论文里必须标注这个 n")


def section_g() -> None:
    """§5.2 的负结果（场结构条件的反例）：N=29 的 null。

    ★ 口径：只用 **elapsed_s ≈ 3600** 的完整一小时 run（78 条），
      否则会把只有 ~300 s 的截断 run 混进来，把 Δ 从 +1.4% 抬到 +8.2%。
    """
    print("\n[G] N=29 的 null（§5.2）")
    p = ROOT / "data/V10.7/batch/batch_manifest_backup.json"
    if not p.exists():
        p = ROOT / "data/batch_manifest_backup.json"
    if not p.exists():
        print("   manifest 不存在，跳过")
        SKIPPED.append("G（N=29 null 的 manifest 未随包分发）")
        return
    runs = [r for r in json.load(open(p, encoding="utf-8"))["runs"]
            if r["elapsed_s"] > 3500]
    by: dict = {}
    for r in runs:
        by.setdefault(r["seed"], {})[r["profile"]] = r
    seeds = sorted(s for s in by if "warm" in by[s] and "frozen" in by[s])
    w = np.array([by[s]["warm"]["avg_light"] for s in seeds], float)
    f = np.array([by[s]["frozen"]["avg_light"] for s in seeds], float)
    d = w - f
    t, pv = paired_t_two_sided(d)
    print(f"   一小时 run {len(runs)} 条，配对 {len(seeds)} 对")
    print(f"   warm {w.mean():.4f}  frozen {f.mean():.4f}  "
          f"Δ {100 * d.mean() / f.mean():+.1f}%  wins {int((d > 0).sum())}/{len(d)}")
    print(f"   配对 t={t:.3f} (df={len(d) - 1})  p={pv:.4f}")
    check("G1 warm 均值", "0.2976", f"{w.mean():.4f}", abs(w.mean() - 0.2976) < 1e-3)
    check("G2 frozen 均值", "0.2936", f"{f.mean():.4f}", abs(f.mean() - 0.2936) < 1e-3)
    check("G3 胜场", "16/29", f"{int((d > 0).sum())}/{len(d)}", int((d > 0).sum()) == 16)
    check("G4 配对 t", "0.170", f"{t:.3f}", abs(t - 0.170) < 0.01)
    check("G5 null 的 p", "0.866", f"{pv:.3f}", abs(pv - 0.866) < 0.01,
          "原文曾写 0.35（抄错）；按其自身 t=0.223 也应是 0.83")


def section_h() -> None:
    """§6.3 的难度阶梯。

    数据来自 `calibrate_world.py`：8 个随机初始化个体 × 3 个留出世界 = 24 条命，
    **不学习、不选择**，只测「这个 λ 下、出生点随机、什么都不学」的存活分布。
    """
    print("\n[H] 难度阶梯（§6.3）")
    rows: list[dict] = []
    for f in ("data/V11/n1_calibration_low.json",
              "data/V11/n1_calibration.json"):
        p = ROOT / f
        if p.exists():
            rows.extend(json.load(open(p, encoding="utf-8")))
    if not rows:
        print("   标定数据不存在，跳过")
        check("H0 标定数据存在", "存在", "缺失", False)
        return
    # 两份标定文件在 λ=0.60 处有重叠，去重
    dedup: dict[float, dict] = {}
    for r in rows:
        dedup[round(r["gain"], 3)] = r
    rows = [dedup[k] for k in sorted(dedup)]
    by = dedup
    print(f"   {'λ':>6} {'n':>4} {'均值':>7} {'活满%':>7} {'250-300%':>9}")
    for r in rows:
        print(f"   {r['gain']:>6.3f} {r['n']:>4d} {r['mean']:>7.1f} "
              f"{r['full_rate']:>6.1f}% {r['mid']:>8.1f}%")

    if 0.500 in by:
        check("H1 λ=0.5（世界 C）活满率", "0%", f"{by[0.500]['full_rate']:.1f}%",
              by[0.500]["full_rate"] == 0.0)
    if 0.525 in by:
        check("H2 λ=0.525 活满率（悬崖）", "0.0%",
              f"{by[0.525]['full_rate']:.1f}%",
              abs(by[0.525]["full_rate"] - 0.0) < 1.0,
              "0.50 → 0.525 活满率从 0 跳到 29.2%")
    if 0.550 in by:
        check("H3 λ=0.55（世界 D）活满率", "12.5%",
              f"{by[0.550]['full_rate']:.1f}%",
              abs(by[0.550]["full_rate"] - 12.5) < 1.0, "世界 D 选在这里")


def section_i() -> None:
    """§7 的三臂对照，第 1 部分：世界 D。

    与 §6.2 的世界 C 是**同一个统计量**，只有 λ 不同（0.55 vs 0.50），
    所以这两个数可以直接对照。
    """
    print("\n[I] 世界 D 的三臂对照（§7）")
    base = ROOT / "data/V11/n1_window"
    if not base.exists():
        print("   数据不存在，跳过")
        check("I0 世界 D 数据存在", "存在", "缺失", False)
        return
    R: dict = {}
    for f in base.glob("*/summary.json"):
        d = json.load(open(f, encoding="utf-8"))
        R[(d["arm"], d["seed"])] = d
    seeds = sorted({s for (a, s) in R if a == "EVO_NL"}
                   & {s for (a, s) in R if a == "NONE_NL"})
    if not seeds:
        print("   没有配对种子，跳过")
        check("I1 配对种子存在", ">0", "0", False)
        return
    ev = np.array([R[("EVO_NL", s)]["held_pool_mean"] for s in seeds], float)
    no = np.array([R[("NONE_NL", s)]["held_pool_mean"] for s in seeds], float)
    d = ev - no
    t, pv = paired_t_two_sided(d)
    ps = exact_sign_flip(d)
    print(f"   n={len(seeds)}  EVO_NL {ev.mean():.1f}  NONE_NL {no.mean():.1f}  "
          f"Δ {d.mean():+.1f} ± {d.std(ddof=1):.1f}  {int((d > 0).sum())}/{len(d)} 为正")
    print(f"   配对 t={t:.2f}  p={pv:.4f}  符号翻转 p={ps:.4f}（n={len(seeds)} 下限 {2 / 2 ** len(seeds):.4f}）")
    div_ev = np.mean([R[("EVO_NL", s)]["diversity_last"] for s in seeds])
    div_no = np.mean([R[("NONE_NL", s)]["diversity_last"] for s in seeds])
    print(f"   末代多样性：有选择 {div_ev:.3f}  无选择 {div_no:.3f}")

    check("I1 基因池差", "+26.3", f"{d.mean():+.1f}", abs(d.mean() - 26.3) < 0.5)
    check("I2 同向种子数", "5/5", f"{int((d > 0).sum())}/{len(d)}",
          int((d > 0).sum()) == len(d) == 5)
    check("I3 配对 t", "2.56", f"{t:.2f}", abs(t - 2.56) < 0.05)
    check("I4 符号翻转 p", "0.0625", f"{ps:.4f}", abs(ps - 0.0625) < 1e-6,
          "n=5 的下限——符号检验到不了 p<0.05，必须同时报配对 t")
    check("I5 多样性坍缩", "0.083 vs 0.978",
          f"{div_ev:.3f} vs {div_no:.3f}", div_ev < 0.2 and div_no > 0.9)


def section_j() -> None:
    """§7 的 2×2×2 消融（种群 / 变异 / 动量继承）。

    ★ 这里曾经用 `fitness_last - fitness_first`，是错的：训练世界按 `gen % 8`
      轮换，首末代不是同一个世界。而且 pop=1 的臂是 120 代、pop=8 的是 15 代，
      同一个窗口口径对两者都不成立。现在**按臂的代数分别取均衡窗口**。
    """
    print()
    print("[J] 2×2×2 消融（世界均衡窗口，按臂的代数）")
    base = ROOT / "data/V11/n2_lam_ablation"
    if not base.exists():
        print("   数据不存在，跳过")
        check("J0 数据存在", "存在", "缺失", False)
        return
    T: dict[str, dict] = {}
    for f in base.glob("*/summary.json"):
        d = json.load(open(f, encoding="utf-8"))
        T.setdefault(d["arm"], {})[d["seed"]] = d
    arms = ["L100", "L101", "L110", "L111", "L800", "L801", "L810", "L811"]
    if any(a not in T for a in arms):
        print("   缺臂，跳过")
        check("J1 八个格齐全", "8", f"{len(T)}", False)
        return

    def delta(a: str) -> np.ndarray:
        """按该臂的代数取均衡窗口：早段 gen1–7，末段同样是 gen%8∈1..7 的最后七个。"""
        out = []
        for s_ in sorted(T[a]):
            H = {h["gen"]: h["mean"] for h in T[a][s_]["history"]}
            gens = sorted(H)
            early = [g for g in gens if 1 <= g <= 7]
            late = [g for g in gens if g % 8 in range(1, 8)][-7:]
            out.append(float(np.mean([H[g] for g in late])
                             - np.mean([H[g] for g in early])))
        return np.array(out, float)

    print(f"   {'臂':<5} {'pop':>3} {'mut':>5} {'opt':>5} {'均衡Δ':>7}")
    for a in arms:
        print(f"   {a:<5} {a[1]:>3} {'0.05' if a[2] == '1' else '0':>5} "
              f"{'是' if a[3] == '1' else '否':>5} {delta(a).mean():>+7.1f}")

    def main_effect(a_keys, b_keys) -> float:
        x = np.concatenate([delta(a) for a in a_keys])
        y = np.concatenate([delta(a) for a in b_keys])
        return float(x.mean() - y.mean())

    mut = main_effect(["L110", "L111", "L810", "L811"],
                      ["L100", "L101", "L800", "L801"])
    pop = main_effect(["L800", "L801", "L810", "L811"],
                      ["L100", "L101", "L110", "L111"])
    opt = main_effect(["L101", "L111", "L801", "L811"],
                      ["L100", "L110", "L800", "L810"])
    print()
    print(f"主效应：变异 {mut:+.1f}   种群 {pop:+.1f}   动量 {opt:+.1f}")

    free = np.concatenate([delta(a) for a in ["L100", "L101", "L800", "L801"]])
    mu = np.concatenate([delta(a) for a in ["L110", "L111", "L810", "L811"]])
    print(f"   无变异四格 {free.mean():+.1f} ± {free.std(ddof=1):.1f}（n={len(free)}）"
          f"   有变异四格 {mu.mean():+.1f} ± {mu.std(ddof=1):.1f}（n={len(mu)}）")

    d_opt = delta("L101") - delta("L100")
    t_o, p_o = paired_t_two_sided(d_opt)
    d_mut = delta("L810") - delta("L800")
    t_m, p_m = paired_t_two_sided(d_mut)
    ps_m = exact_sign_flip(d_mut)
    print(f"   加动量 L101−L100 : {d_opt.mean():+.1f}  t={t_o:.2f}  p={p_o:.3f}")
    print(f"   加变异 L810−L800 : {d_mut.mean():+.1f}  t={t_m:.2f}  p={p_m:.4f}  "
          f"符号p={ps_m:.4f}  {int((d_mut > 0).sum())}/5")

    check("J2 变异主效应", "+28.2", f"{mut:+.1f}", abs(mut - 28.2) < 0.3)
    check("J3 种群主效应", "-29.6", f"{pop:+.1f}", abs(pop + 29.6) < 0.3,
          "★ 符号为负：只加种群不加变异没有帮助")
    check("J4 动量主效应", "-2.2", f"{opt:+.1f}", abs(opt + 2.2) < 0.3)
    check("J5 加变异 L810−L800", "+30.4", f"{d_mut.mean():+.1f}",
          abs(d_mut.mean() - 30.4) < 0.3, "p=0.076，不显著")
    check("J6 加变异的配对 t", "1.97", f"{t_m:.2f}", abs(t_m - 1.97) < 0.05)
    check("J7 加动量无差别", "-3.4", f"{d_opt.mean():+.1f}", abs(d_opt.mean() - -3.4) < 0.3)
    check("J8 无变异四格均值", "+4.2", f"{free.mean():+.1f}", abs(free.mean() - 4.2) < 0.3)
    check("J9 有变异四格均值", "+32.4", f"{mu.mean():+.1f}", abs(mu.mean() - 32.4) < 0.3)


def section_k() -> None:
    """§7 的三臂对照，第 2 部分：精英 vs 选择。

    ★ 这是抽掉论文初版核心结论的一组数：`EVO_NL − NONE_NL` 看上去是选择效应，
      但那个对照同时改了 elitism。加 `NONE_EL`（无选择但保留精英）之后，
      真正的来源才显现。
    """
    print()
    print("[K] 选择 vs 精英（§7 三臂对照）")
    by: dict = {}
    for f in glob.glob(str(ROOT / "data/V11/review_fix_elitism/*/summary.json")):
        d = json.load(open(f, encoding="utf-8"))
        by.setdefault((d["world"], d["arm"]), {})[d["seed"]] = d["held_pool_mean"]
    for f in glob.glob(str(ROOT / "data/V11/n1_window/*/summary.json")):
        d = json.load(open(f, encoding="utf-8"))
        by.setdefault(("D", d["arm"]), {})[d["seed"]] = d["held_pool_mean"]
    for f in glob.glob(str(ROOT / "data/V11/exp13_fixed/C_*/summary.json")):
        d = json.load(open(f, encoding="utf-8"))
        by.setdefault(("C", d["arm"]), {})[d["seed"]] = d["held_pool_mean"]

    if not by.get(("D", "NONE_EL")):
        print("   NONE_EL 数据缺失，跳过")
        check("K0 NONE_EL 数据存在", "存在", "缺失", False)
        return

    for w in ("D", "C"):
        print(f"   世界 {w}:")
        for a in ("NONE_NL", "NONE_EL", "EVO_NL"):
            v = list(by.get((w, a), {}).values())
            if v:
                print(f"     {a:<8} n={len(v)} 基因池 {np.mean(v):>6.1f}")
        for x, y, lab in (("EVO_NL", "NONE_NL", "选择+精英"),
                          ("EVO_NL", "NONE_EL", "纯选择"),
                          ("NONE_EL", "NONE_NL", "纯精英")):
            if (w, x) in by and (w, y) in by:
                ss = sorted(set(by[(w, x)]) & set(by[(w, y)]))
                d = np.array([by[(w, x)][k] - by[(w, y)][k] for k in ss], float)
                t, pv = paired_t_two_sided(d)
                print(f"       {lab:<8} {x}−{y}: Δ {d.mean():+6.1f}  "
                      f"t={t:5.2f} p={pv:.4f}  {int((d > 0).sum())}/{len(d)}")
                if w == "D" and (x, y) == ("EVO_NL", "NONE_NL"):
                    check("K1 选择+精英 的 Δ", "+26.3", f"{d.mean():+.1f}",
                          abs(d.mean() - 26.3) < 0.5)
                if w == "D" and (x, y) == ("EVO_NL", "NONE_EL"):
                    check("K2 纯选择 的 Δ", "-1.9", f"{d.mean():+.1f}",
                          abs(d.mean() + 1.9) < 0.5,
                          "★ 去掉精英混淆后选择效应消失")
                    check("K3 纯选择 的 p", "0.764", f"{pv:.3f}",
                          abs(pv - 0.764) < 0.01)
                if w == "D" and (x, y) == ("NONE_EL", "NONE_NL"):
                    check("K4 纯精英 的 Δ", "+28.2", f"{d.mean():+.1f}",
                          abs(d.mean() - 28.2) < 0.5,
                          "★ 收益全部来自保留最优个体")


def section_l() -> None:
    """§3.1 陷阱：47 条唯一、同预算、**同一个静态世界**的臂-运行。

    ★ 第七轮修正（策略梯度 detach 之后）
      ① 新数据在 `exp9fix*`；固定学习器下 47 条全部唯一（旧数据里的
         完全重复是坏梯度的产物）。
      ② 主相关改用**末代存活**（图 2a 的口径）；8 代平均作为次要口径
         一起打印，但不作为论文主指标。
      ③ 47 条全部来自世界 A（sigma=600，光源不动），mean_light 是移动
         与存活的结果，不是外生协变量；偏相关无效，已删除。
      ④ 47 条来自 11 个文件 / 4 个种子 / 3 个架构，不独立；用按文件的
         聚类 bootstrap 给区间。
    """
    print()
    print("[L] 过呼吸陷阱")
    dirs = ["exp9fix", "exp9fix_seed1", "exp9fix_seed2", "exp9fix_a_seed1",
            "exp9fix_fw_seed0", "exp9fix_fw_seed1", "exp9fix_fw_seed2",
            "exp9fix_n7_seed1", "exp9fix_n7_seed2", "exp9fix_n7_seed3",
            "exp9fix_mlp8"]
    rows = []
    for d in dirs:
        for f in sorted((ROOT / "data/V11" / d).glob("exp8_*.json")):
            for a, rec in json.load(open(f, encoding="utf-8"))["arms"].items():
                h = rec.get("history") or []
                if len(h) != 8:
                    continue
                k = h[-1].get("mean_k")
                if k is None or k != k:
                    continue
                rows.append(dict(file=d, arm=a, arch=rec.get("arch", "?"),
                                 k=float(k), b=float(rec["mean_beats"]),
                                 bl=float(h[-1]["beats"]),
                                 L=float(h[-1].get("mean_light", 0.0))))

    seen, uniq = set(), []
    for r in rows:
        key = (r["arm"], round(r["k"], 9), round(r["L"], 9),
               round(r["b"], 6), round(r["bl"], 6))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    k = np.array([r["k"] for r in uniq])
    b = np.array([r["b"] for r in uniq])
    bl = np.array([r["bl"] for r in uniq])
    L = np.array([r["L"] for r in uniq])
    files = sorted({r["file"] for r in uniq})

    def corr(x, y):
        mx, my = np.mean(x), np.mean(y)
        return float(np.sum((x - mx) * (y - my))
                     / np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2)))

    r_mean = corr(k, b)
    r_last = corr(k, bl)
    # 按文件的聚类 bootstrap：**有放回**，同一个文件被抽中多次就按多次计入。
    # ★ 初版写的是 `{files[j] for j in pick}`，集合把重复抽中的文件去掉了，
    #   等价于不放回子采样，区间系统性偏窄。
    rng = np.random.default_rng(0)
    row_of_file = [np.array([i for i, row in enumerate(uniq) if row["file"] == f])
                   for f in files]
    boot_last, boot_mean = [], []
    for _ in range(4000):
        pick = rng.integers(0, len(files), size=len(files))
        idx = np.concatenate([row_of_file[j] for j in pick])
        boot_last.append(corr(k[idx], bl[idx]))
        boot_mean.append(corr(k[idx], b[idx]))
    lo, hi = np.percentile(boot_last, [2.5, 97.5])
    lo_m, hi_m = np.percentile(boot_mean, [2.5, 97.5])

    K, C = 450.0, 200.0
    tgt = 1.6 * 10000.0 * (1.0 - np.exp(-0.01))
    gain = K * L * k - C * k * k
    neg = gain < 0
    n_neg = int(neg.sum())
    b_neg = float(bl[neg].mean())
    b_pos = float(bl[~neg].mean())
    b_neg_m = float(b[neg].mean())
    b_pos_m = float(b[~neg].mean())
    n_near = int((np.abs(gain - tgt) < 40).sum())
    unreach = int((L < 0.793).sum())

    print(f"   {len(rows)} 条 → 去重后 {len(uniq)} 条（全部来自世界 A，静态光源）")
    print(f"   末代 r = {r_last:+.4f}   聚类 95% CI [{lo:+.3f}, {hi:+.3f}]"
          f"（{len(files)} 个文件）")
    print(f"   8 代平均 r = {r_mean:+.4f}   CI [{lo_m:+.3f}, {hi_m:+.3f}]")
    print(f"   gain<0：{n_neg}/{len(uniq)}（末代 {b_neg:.0f} 拍 vs {b_pos:.0f} 拍；"
          f"8 代平均 {b_neg_m:.0f} vs {b_pos_m:.0f}）；"
          f"|gain−{tgt:.1f}|<40：{n_near}")
    print(f"   目标不可达（mean_light < 0.793）：{unreach}/{len(uniq)}")

    check("L1 唯一 arm-run 数", "47", str(len(uniq)), len(uniq) == 47)
    check("L2 末代相关 r", "-0.438", f"{r_last:.4f}", abs(r_last + 0.438) < 0.0005)
    check("L3 聚类 bootstrap 下界", "-0.67", f"{lo:+.2f}", abs(lo + 0.67) < 0.03)
    check("L4 聚类 bootstrap 上界", "-0.19", f"{hi:+.2f}", abs(hi + 0.19) < 0.03)
    check("L5 净耗能 run 数", "19", str(n_neg), n_neg == 19)
    check("L6 净耗能组末代存活", "60", f"{b_neg:.0f}", abs(b_neg - 60) < 1)
    check("L7 净收益组末代存活", "282", f"{b_pos:.0f}", abs(b_pos - 282) < 1)
    check("L8 贴近目标增益的 run 数", "2", str(n_near), n_near == 2)
    check("L9 目标不可达的 run 数", "29", str(unreach), unreach == 29)

    print("   架构子组（47 条去重池，末代口径）：")
    for i, (a, lab, wn, wr) in enumerate(
            (("cnn", "CNN", 34, -0.406),
             ("transformer", "Transformer", 9, -0.964),
             ("mlp", "MLP", 4, -0.670)), start=10):
        idx = [j for j, row in enumerate(uniq) if row["arch"] == a]
        if not idx:
            check(f"L{i} {lab} 子组存在", "有", "无", False)
            continue
        ra = corr(k[idx], bl[idx])
        print(f"     {lab:<12} n={len(idx):2d}  r={ra:+.3f}")
        check(f"L{i} {lab} 子组", f"n={wn} r={wr:+.3f}",
              f"n={len(idx)} r={ra:+.3f}",
              len(idx) == wn and abs(ra - wr) < 0.01)


def section_m() -> None:
    """§5 场结构条件的反例：从归档的原始轨迹重算 N=29。"""
    print()
    print("[M] 门 1 的反例：N=29（从归档轨迹重算）")
    man = ROOT / "data/V10.7/batch/batch_manifest_backup.json"
    arch = ROOT / "paper_alife/data/n29_null"
    if not man.exists() or not arch.exists():
        print("   数据缺失，跳过")
        SKIPPED.append("M（N=29 归档轨迹未随包分发）")
        return
    runs = [r for r in json.load(open(man, encoding="utf-8"))["runs"]
            if r["elapsed_s"] > 3500]
    by: dict = {}
    n_read = 0
    for r in runs:
        p = arch / f"{r['id']}_{Path(r['csv']).name}"
        if not p.exists():
            continue
        light = [float(x["light"]) for x in csv.DictReader(open(p, encoding="utf-8"))
                 if x.get("light")]
        by.setdefault(r["seed"], {})[r["profile"]] = float(np.mean(light))
        n_read += 1
    seeds = sorted(s for s in by if "warm" in by[s] and "frozen" in by[s])
    w = np.array([by[s]["warm"] for s in seeds], float)
    f = np.array([by[s]["frozen"] for s in seeds], float)
    d = w - f
    t, pv = paired_t_two_sided(d)
    print(f"   从 CSV 读到 {n_read} 条，配对 {len(seeds)} 对")
    print(f"   warm {w.mean():.4f}  frozen {f.mean():.4f}  Δ {100 * d.mean() / f.mean():+.1f}%  "
          f"{int((d > 0).sum())}/{len(d)}")
    print(f"   配对 t={t:.3f}  p={pv:.4f}")
    check("M1 从原始轨迹读到的 run 数", "78", str(n_read), n_read == 78)
    check("M2 配对种子数", "29", str(len(seeds)), len(seeds) == 29)
    check("M3 warm 均值", "0.2976", f"{w.mean():.4f}", abs(w.mean() - 0.2976) < 1e-3)
    check("M4 frozen 均值", "0.2936", f"{f.mean():.4f}", abs(f.mean() - 0.2936) < 1e-3)
    check("M5 胜场", "16/29", f"{int((d > 0).sum())}/{len(d)}", int((d > 0).sum()) == 16)


def main() -> int:
    ap = argparse.ArgumentParser(description="论文 V11 部分审计")
    ap.add_argument("--out", default="paper_alife/AUDIT_V11.md")
    args = ap.parse_args()

    # Windows 默认控制台是 GBK，遇到 U+2212（减号）等字符会崩。
    # 这里显式切到 UTF-8，让 `python scripts/audit_paper.py` 开箱即用。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    print("=" * 78)
    print("  论文审计（V11 部分）")
    print("=" * 78)

    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    section_g()
    section_h()
    section_i()
    section_j()
    section_k()
    section_l()
    section_m()

    n_pass = sum(1 for c in CHECKS if c["ok"])
    n = len(CHECKS)
    print("\n" + "=" * 78)
    print(f"  结果：{n_pass} / {n} 通过")
    if SKIPPED:
        print("  跳过：" + "；".join(SKIPPED))
    print("=" * 78)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    map_rows = [
        ("A", "历史：world B 修复前的选择效应（**不在当前论文中**）"),
        ("B", "§6.1 世界 B 的饱和"),
        ("C1", "§6.2 世界 C 的 0/2400"),
        ("C3–C4", "历史：EVO vs LAM（**不在当前论文中**）"),
        ("D", "§2 世界轮换口径"),
        ("E", "§4.1–4.2 呼吸修复与 2×2"),
        ("F", "补充表 S1 架构族（**正文只在 §8 讨论里引一句**）"),
        ("G", "历史：N=29 从 manifest 读（由 M 取代）"),
        ("H", "§6.3 难度阶梯（**只查 3/13 个点**）"),
        ("I", "§7 world D 的基因池对比"),
        ("J", "§7 的 2×2×2（世界均衡，按臂代数）"),
        ("K", "§7 world D 的三臂对比"),
        ("L", "§3.1 的陷阱相关（47 arm-runs，末代）"),
        ("M", "§5.2 的 N=29（从归档原始轨迹重算）"),
    ]
    lines = ["# AUDIT — V11 部分", "",
             f"**{n_pass} PASS / {n - n_pass} FAIL**（脚本：`scripts/audit_paper.py`）", ""]
    if SKIPPED:
        lines += ["> 跳过：" + "；".join(SKIPPED), ""]
    lines += ["> FCL 部分见 `scripts/audit_submission_0816.py`。", "",
             "## 各段与论文的对应", "",
             "| 段 | 对应 |", "|---|---|"]
    for k, v in map_rows:
        lines.append(f"| {k} | {v} |")
    lines += ["", "> 标「历史」的段保留下来是为了记录被推翻的结论，**不支撑当前论文的任何主张**。",
              "", "| 检查 | 论文声明 | 重算值 | 结果 | 备注 |", "|---|---|---|---|---|"]
    for c in CHECKS:
        lines.append(f"| {c['name']} | {c['claimed']} | {c['computed']} | "
                     f"{'PASS' if c['ok'] else '**FAIL**'} | {c['note']} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  写入 {out}")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    sys.exit(main())
