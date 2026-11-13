"""V11 的物理法则 —— 全部从这里来，别处不许再写公式。

三条法则
--------
1. **体重**
       m = 矩阵的乘积 = grid²
       128×128 → m = 16,384

2. **质量系数（场强）**
       s = √(E / m)          E = 场的总能量 Σ|ψ|²
                             = 场的**均方根幅度** |ψ|_RMS

3. **物理法则系数（振荡包络）**
       x    = X_MAX · (E/E₀ − 1.6) / 1.4
       f(x) = (1 + sin(x²)) / (1 + x²)

       性质：
         x=0 → 1.000  （生存带正中，满速）
         x→±∞ → 0     （死亡边界，冻结）
         永远 ≥ 0，永远 ≤ 1
         侧瓣清晰可见（5~6 瓣），谷底贴地

速度
----
    T 输出 (Fx, Fy, Fz)  =  力 / 意愿矢量

    方向     = F / |F|
    意愿     = |F|
    v_norm   = |F| · s · f(x)          再裁到 ≤ 1
    v_world  = 方向 · v_norm · V_WORLD

运动耗能
--------
    每拍  E −= K_MOVE · ½ · m · v_norm²

    K_MOVE 标定到「中央满速跑一拍」约烧掉 1.5% 的能量。

生死
----
    E < LOWER · E₀   →  饿死
    E > UPPER · E₀   →  撑死
"""

from __future__ import annotations

import math

# ── 生存带（以 E₀ = norm_target 为单位）──────────────────────────────
BAND_CENTER = 1.6      # 中央峰在 1.6 × E₀（= 16000）
BAND_HALF = 1.4        # 半宽 1.4 × E₀（下 0.2 / 上 3.0）
LOWER = BAND_CENTER - BAND_HALF     # 0.2
UPPER = BAND_CENTER + BAND_HALF     # 3.0

X_MAX = 6.0            # 生存区间映射到 x ∈ [−6, +6]；|x|=6 时 f ≈ 0.0002

K_MOVE = 0.03          # 运动耗能标定（满速一拍 ≈ 1.5% 能量）
V_WORLD = 150.0        # 归一速度 1.0 = 每秒 150 世界单位

# ── 基础代谢（光 → 能量）────────────────────────────────────────────
#   dE_meta = META_K × 本地光强
#   平衡点 E* = META_K × light / γ      （γ = dissipation = 0.01）
#
#   ★ 标定依据：走廊中心的光强实测约 0.187，希望那里刚好打平
#     （E* = 1.0 × E₀ = 10,000）：
#         META_K × 0.187 / 0.01 = 10,000  →  META_K ≈ 535
#
#   于是生存带被光强映射成：
#     位置 0    light 0.187 → E* = 1.00×  （刚打平，一动就亏）
#     位置 300  light 0.320 → E* = 1.71×  （★ 最优）
#     位置 450  light 0.70  → E* = 3.74×  （撑死）
#     位置 500  light 5.58  → E* = 29.8×  （必爆）
META_K = 535.0

ENERGY_PER_MASS = 10000.0 / 16384.0   # E₀ / m（比能量，≈0.6104）


# ── 呼吸深度（实验 8：多 Transformer 架构对照）─────────────────────
#   实验 7 之前，代谢吸收是一个**常数**：dE = META_K × light。
#   也就是说「呼吸」这件事不是个可决策的量 —— 个体只能靠移动去改变光强。
#
#   实验 8 把「呼吸深度」做成第二个可学习的动作：
#
#       dE_breath = BREATH_K × light × k  −  BREATH_C × k²
#                    └── 线性项：吸得越深，拿到的越多
#                                        └── 二次项：吸得越深，提取效率越低
#
#   ★ 为什么必须有这个二次项
#     只有线性项的话「一直吸到底」就是平凡最优解，学不到任何东西。
#     二次项让每一点光强都有一个**内部最优深度**：
#
#           k*(light) = BREATH_K × light / (2 × BREATH_C)
#
#     → 光弱时最优深度小（吸也是白吸，不如省着点）
#     → 光强时最优深度大
#     **常数 k 一定有损失**，这就是这一维动作的存在意义。
#
#   ★ 标定（沙盒走廊，光强 ∈ [0, 1]）
#     BREATH_K=450, BREATH_C=200 → k* = 1.125 × light
#       light = 0.25 → k* = 0.28（再暗就必然饿死）
#       light = 0.75 → k* = 0.84  ← 出生点
#       light = 1.00 → k* = 1.13
#     最优呼吸下的平衡能量 E* = K·light²/(4Cγ) = 25300 × light²：
#       light = 0.25 → 0.16×E₀  饿死
#       light = 0.375 → 0.36×E₀
#       light = 0.75 → 1.42×E₀  ★ 出生点，带宽中间
#       light = 1.00 → 2.53×E₀  ★ 全程都到不了 3.0 —— **撑死被移出环境**
#
#   ★ 为什么要把「撑死」移出环境（实验 8 第一次试跑的教训）
#     原来 K=800 时，最亮处的平衡能量是 8.0×E₀ —— 一旦漂向光源就必爆。
#     试跑里四支臂的失败模式全是「撑死」，而且**和架构毫无关系**：
#     谁的随机漂移恰好往右多一点，谁就先爆。噪声完全盖住了要测的效应。
#     把 K 降到 450 之后，走廊里任何位置都不会撑死 ——
#     失败模式只剩「饿死」（漂到暗处），那是一个**关于位置和呼吸的决策**，
#     这才是实验要测的东西。
BREATH_K = 450.0
BREATH_C = 200.0

BREATH_MIN = 0.05         # 呼吸深度下限（不是 0 —— 0 就是"完全不活"）
BREATH_MAX = 2.0          # 呼吸深度上限
# ★ 实验 F：mode="fixed" 时用于算固定上界的参考光强。
#   取世界 B 逐拍光强的中位数（实测 0.411，见 data/V11/exp12b/B_FIXED_seed*/beats.csv），
#   所以「固定上界」那一格的上界等于策略在典型光照下能拿到的最优深度。
BREATH_LIGHT_REF = 0.411

BREATH_PERIOD = 10        # ★ 慢时钟：每 10 拍才重新决定一次呼吸深度
                          #   移动是 1 拍一次的**快时钟** —— 两个功能不只在
                          #   输出维度上不同，在**时间尺度**上也不同。
                          #   这正是「多 Transformer」在理论上有理由的地方。

BREATH_INIT_Z = 0.95       # 呼吸头输出 z 的初始偏置 → k ≈ 1.50
                          #   （softsign 下 z/(1+z)=0.487 → k = 1.025+0.975×0.487）
                          #   ★ 为什么故意让它**过深**：
                          #     k0=0.30 在暗处恰好接近最优（暗处的最优深度本来就小），
                          #     于是「待在暗处不动」就是一个能活的平凡解 ——
                          #     没有任何压力去学呼吸。实测 k0=0.304 时出生点
                          #     x=200~500 全部活满 400 拍，实验区分不出任何东西。
                          #     k0=1.50 则在中等光强下就是**净亏**的
                          #     （零增益点 light ≈ 0.375），个体必须学会
                          #     把深度调浅（或挪到更亮的地方）才活得下去。
                          #     这就是实验 8 的学习压力来源。


def breath_gain(light: float, k: float,
                k_breath: float = BREATH_K, c: float = BREATH_C) -> float:
    """一次呼吸拿到的净能量（可为负）。

    dE = k_breath × light × k − c × k²
    """
    return float(k_breath) * float(light) * float(k) - float(c) * float(k) * float(k)


def breath_optimal_k(light: float, k_breath: float = BREATH_K,
                     c: float = BREATH_C) -> float:
    """给定光强下的理论最优呼吸深度（未裁剪）。

    k* = k_breath × light / (2c)
    """
    if c <= 0:
        return BREATH_MAX
    return float(k_breath) * float(light) / (2.0 * float(c))


def breath_map(z):
    """网络输出 z（实数）→ 呼吸深度 k ∈ [MIN, MAX]。

    ★ 为什么用 **softsign**（z/(1+|z|)）而不是 sigmoid（实验 10 实测）

      sigmoid 在饱和区的导数是 **指数级**消失的：z=+10 时 σ'(z)=4.5e-5。
      实测代价很直接：第一版用 sigmoid 时，D 的 k 被推到下限 0.05、
      N1 的 k 被推到上限 2.0，然后**两边都卡死出不来** ——
      因为 dk/dz ≈ 0，任何损失梯度都传不过去，连 Adam 都动不了。

      softsign 的导数是 **代数级**衰减：1/(1+|z|)²。
      z=+10 时仍有 0.008 —— 比 sigmoid 大 180 倍，Adam 靠它就能一步步爬回来。

      两者都是有界单调映射，但 softsign 的「饱和」不是死路。
    """
    import torch

    mid = 0.5 * (BREATH_MIN + BREATH_MAX)
    half = 0.5 * (BREATH_MAX - BREATH_MIN)
    return mid + half * z / (1.0 + z.abs())


def breath_map_float(z: float) -> float:
    """breath_map 的标量版（不需要 tensor / 梯度的地方用）。"""
    mid = 0.5 * (BREATH_MIN + BREATH_MAX)
    half = 0.5 * (BREATH_MAX - BREATH_MIN)
    az = abs(float(z))
    return mid + half * float(z) / (1.0 + az)


# ── ★ 实验 10：呼吸的可微目标（修「过度呼吸陷阱」）──────────────────
#
#   问题（实验 9 实测）
#   ------------------
#   57 次运行里 corr(末代呼吸深度 k, 存活拍数) = −0.45：
#   k<1.0 的平均活 300 拍，k>1.6 的只活 118 拍。
#   而且这是**初始化决定**的 —— 同一个 2 层 Transformer，
#   种子 0 → k=0.61、350 拍；种子 1 → k=1.96、155 拍。
#
#   根因：呼吸是**慢时钟**动作（每 10 拍才一次），拿到的只有策略梯度，
#   而它对存活的影响要几十拍后才体现 —— 信用分配太远，梯度基本是噪声。
#   探索噪声一旦把 k 推高，就会掉进「越吸越亏 → 越快越饿」的正反馈里出不来。
#
#   修法（和当年修写场头一模一样的思路）
#   ------------------------------------
#   写场头在实验 1-6 里也拿不到策略梯度，当时的解法是给它一个**可微目标**
#   （最大化 speed_scale，物理法则曲线自己的峰）。呼吸头现在需要同样的待遇。
#
#   呼吸的长期后果是**稳态能量**：
#       E∞ = gain × (1−γ) / γ,      gain = K·light·k − C·k²
#   （因为每拍先加 gain 再乘 (1−γ)，不动点就是上式。）
#
#   而「稳态能量该停在哪」不需要手选 —— **生存带的几何中心**：
#       LOWER = 0.2, UPPER = 3.0  →  BAND_CENTER = 1.6
#   这是死亡规则自己定义的常数，不是我们拍脑袋定的目标。
#
#   于是可微目标就是：
#       L_breath = (E∞/E₀ − BAND_CENTER)²
#
#   ★ 为什么这个目标能自己学会「随光强调深度」
#     目标只说要停在 1.6×E₀，没有说 k 该是多少。
#     k 要多少完全取决于当前光强 —— 光强是网络必须自己看出来的。
#     所以这不是把答案喂进去，而是把**目的**告诉它，路径仍然要学。
#
#   ★★★ 实验 12 的修正（并附 2026-10-24 的自查纠错）
#   ------------------------------------------------
#   原以为有「两处错误」，自查之后发现**只有一处是真的**。
#
#   【真 bug】目标有两个解
#     breath_gain 在 k 上是凹的，所以「某个目标增益」总有两个 k（浅解 / 深解）。
#     从初值 k≈1.5 出发，梯度先被推到最大增益点 k* = 1.125·light，
#     在那里两个解的梯度大小相等、方向相反（对称分岔），策略被噪声推向任一侧。
#     静态光下落到浅解看着没事；**光源一移动，深解（k≈1.8）就在暗处饿死。**
#     实测（区间全开时，light=0.9）：目标函数有两个极小 k = 0.534 与 1.491。
#
#     修法：把动作**重参数化**到增益单调区间 [BREATH_MIN, k*(light)]。
#          超过 k* 的呼吸永远更差（更少的增益、更大的代价），
#          那一段是**完全冗余**的动作空间，去掉之后目标就一一可解了。
#
#   【不是 bug】稳态公式里的 γ
#     曾经把它判成「用错、差 2.03 倍」，**那是错的**。
#     旧写法 E∞ = gain·(1−γ)/γ = 99·gain，而真值是 100.5·gain —— 只差 1.5%。
#     差别小的原因：`field.step()` 衰减的是**振幅**，能量是 |ψ|²，
#     所以真实能量乘子是 exp(−2γ·dt) = 0.99005，不是 exp(−γ·dt) = 0.99501。
#     把后者当成能量乘子才会差 2 倍（200.5 vs 100.5）—— 那是我的错，不是代码的错。
#     实测定标（关噪声与代谢空跑 200 拍）：E(200)/E(0) = 0.135330 = exp(−2) ✓
#
#   【正确性补丁】能量预算里要算上运动耗能
#     E_{t+1} = E_t·r_E + gain_呼吸 − move_cost，所以稳态是
#     E∞ = (gain − move_cost)/(1 − r_E)。
#     不扣 move_cost，目标会把呼吸定得太浅（世界 A 实测 0.40×E₀ 而不是 1.6）。

def decay_multiplier(gamma: float, dt: float, mode: str = "step") -> float:
    """场每拍的**能量**乘子 r_E。

    ★★★ 关键：`field.py` 里耗散是作用在**振幅**上的

        self._dissipation = exp(−γ·dt)
        ψ ← ψ · self._dissipation

    而能量是 E = Σ|ψ|²，所以能量乘子是振幅乘子的**平方**：

        r_E = exp(−2·γ·dt)

    γ=0.010、dt=0.5 时 r_E = exp(−0.01) = 0.9900498。
    **实测验证**（关掉噪声与代谢，空跑 200 拍）：
        E(200)/E₀ = 0.135330  =  exp(−2)  ✓
    `config.py` 里「dissipation=0.0100 → 200 步后 13.5%」这句注释也是这个数。

    mode="step"：r_E = exp(−2γ·dt)   ← **正确**
    mode="coef"：r_E = 1 − γ          ← 实验 8-11 用的近似（差 1.5%，够用）

    ★ 自查记录（2026-10-24）
      这个函数的第一版写成了 exp(−γ·dt) = 0.9950125，把稳态算成 200·gain，
      并把旧公式判成「错 2.03 倍」。**那是错的**：旧公式 99·gain 只差 1.5%，
      真正差 2 倍的是那一版「修正」。错误原因就是漏了「能量 ∝ 振幅²」这一步。
      教训：这类系数必须**实测**（空跑 N 拍看能量衰减），不能靠公式推导。
    """
    if mode == "coef":
        return 1.0 - float(gamma)
    if mode != "step":
        raise ValueError(f"未知的呼吸 γ 模式：{mode!r}（可选 step / coef）")
    amp = math.exp(-float(gamma) * float(dt))
    return amp * amp


def breath_steady_energy(gain, gamma: float, dt: float, mode: str = "step"):
    """给定每拍净增益 gain，长期稳态能量。

    ★ 推导必须按 `Brain.heartbeat()` 里**真实的操作顺序**来：
          ① field.tick()      E ← E·r          （先耗散）
          ② metabolize()      E ← E + gain     （再呼吸补能）
          ③ 运动耗能          E ← E − c        （最后扣掉走动）
      于是 E_{t+1} = E_t·r + gain − c，令 E_{t+1} = E_t = E∞：
          E∞ = (gain − c) / (1 − r)

      调用方把 −c 折进 `gain`（传 gain_eff = 呼吸增益 − 运动耗能）即可。

    ★ 两个都容易写错的地方（实验 12 逐条核对）
      ① dt：旧签名把它隐含成 1.0，等于假设「一个心跳 = 一秒」，实际 dt = 0.5
      ② 顺序：如果假设「先加后乘」会得到 E∞ = gain·r/(1−r)，
         比真实值小 0.5%（199.5 vs 200.5）。量级小，但既然是修错就一起修。

    mode="coef" 是实验 8-11 的旧写法（假设 E_{t+1}=(E_t+gain)(1−γ)），
    只为逐位复现旧结果而保留。
    """
    if mode == "coef":
        return gain * (1.0 - float(gamma)) / float(gamma)
    r = decay_multiplier(gamma, dt, mode)
    return gain / (1.0 - r)


def breath_homeostasis_loss(light, k, gamma: float, energy_cap: float,
                            dt: float, center: float = None,
                            mode: str = "step", move_cost: float = 0.0):
    """★ 可微的呼吸目标：让稳态能量停在生存带中心。

    参数
    ----------
    light : torch.Tensor  (B,)
    k     : torch.Tensor  (B,)   呼吸深度，**带梯度**
    gamma : float       场的耗散率
    energy_cap : float  E₀
    dt    : float       一个心跳的时长（必须给对，见 breath_steady_energy）
    center : float      默认 physics.BAND_CENTER
    mode  : str         "step"（正确）/ "coef"（旧，仅复现实验 8-11）
    move_cost : float
        ★ 实验 12 第三处修正：每拍的运动耗能。
        能量预算里必须包含它 —— 一个不停走动的心跳，呼吸能承受的净增益
        要比「站着不动」高 `move_cost`。不扣这一项，目标就会把呼吸
        定得**太浅**（实测：世界 A 里 k 落到 0.28 然后饿死）。

    返回
    -------
    torch.Tensor  标量损失（越小越好）
    """
    if center is None:
        center = BAND_CENTER
    gain = BREATH_K * light * k - BREATH_C * k * k - float(move_cost)
    e_inf = breath_steady_energy(gain, gamma, dt, mode)
    return ((e_inf / float(energy_cap)) - float(center)) ** 2


def breath_optimal_gain(gamma: float, energy_cap: float, dt: float,
                        center: float = None, mode: str = "step") -> float:
    """稳态能量正好落在生存带中心时，每拍需要的净增益 g*。

        g* = center · E₀ · (1 − r)          （模式 step）
        或者旧写法 center · E₀ · γ/(1−γ)     （模式 coef）

    ★ 如果个体在走动，呼吸要提供的增益还要再加 `move_cost`：
        g_呼吸 = g* + move_cost
    """
    if center is None:
        center = BAND_CENTER
    if mode == "coef":
        return float(center) * float(energy_cap) * float(gamma) / (1.0 - gamma)
    r = decay_multiplier(gamma, dt, mode)
    return float(center) * float(energy_cap) * (1.0 - r)


# ── ★★ 实验 12：呼吸动作的「有用区间」（去掉冗余的深呼吸分支）────────

def breath_useful_max(light: float, k_breath: float = BREATH_K,
                      c: float = BREATH_C, mode: str = "useful",
                      light_ref: float = BREATH_LIGHT_REF) -> float:
    """给定光强下呼吸深度的**有用上限** k*(light) = K·light/(2C)。

    ★ 实验 F（第五轮）：三种上界，用来把「限制到单调支」与「上界随光强变化」
      这两件事分开——修复同时做了两件，此前从未单独隔离。

      mode="useful"（默认，与实验 12 一致）
          上界 = k*(light)。既限制在峰值以下，又让区间跟着光走。
      mode="fixed"
          上界 = k*(light_ref)，light_ref 固定。限制在峰值以下，但区间不随光变。
          这是「只限制、不锚定」那一格。
      mode="excess"
          上界 = min(BREATH_MAX, 2·k*(light))。区间跟着光走，但允许越过峰值。
          这是「只锚定、不限制」那一格。

      2×2 的第四格（固定且越过峰值）就是旧配置 OLD_FULL。

    ★ 为什么这是物理上限，而不是手写的动作规则
      `breath_gain(light, k) = K·light·k − C·k²` 是**凹函数**，
      在 k* = K·light/(2C) 处取最大值，k 再大增益反而下降。

      于是对任意光强，区间 (k*, BREATH_MAX] 里的每一个深度，都存在一个
      **更浅**的深度拿到**同样或更大**的增益。深呼吸分支不提供任何新的
      能力，只提供「用更多代价做同样的事」这一种**冗余表示**。

    ★ 它还有第二个害处（实验 11 实测）
      稳态能量在 k 上是二次的，所以「停在 1.6·E₀」有两个解（浅解 / 深解）。
      旧目标只约束 E∞、不区分两者，策略在对称分岔点上被噪声推向任一侧；
      静态光下落在浅解看着没事，光源一移动就翻到深解（k≈1.8），暗处饿死。
      把动作限制在单调分支上，目标就有唯一解了。
    """
    if c <= 0:
        return BREATH_MAX
    if mode not in ("useful", "fixed", "excess", "fixed_excess"):
        raise ValueError(f"未知的呼吸上界模式：{mode!r}")
    src = float(light_ref) if mode in ("fixed", "fixed_excess") else float(light)
    k = float(k_breath) * src / (2.0 * float(c))
    if mode in ("excess", "fixed_excess"):
        k = 2.0 * k
    return min(BREATH_MAX, max(BREATH_MIN, k))


def breath_useful_max_tensor(light, k_breath: float = BREATH_K,
                             c: float = BREATH_C, mode: str = "useful",
                             light_ref: float = BREATH_LIGHT_REF):
    """`breath_useful_max` 的张量版（可微，供损失函数用）。"""
    import torch

    if c <= 0:
        return torch.full_like(light, BREATH_MAX)
    if mode not in ("useful", "fixed", "excess", "fixed_excess"):
        raise ValueError(f"未知的呼吸上界模式：{mode!r}")
    src = float(light_ref) if mode in ("fixed", "fixed_excess") else light
    k = float(k_breath) * src / (2.0 * float(c))
    if mode in ("excess", "fixed_excess"):
        k = 2.0 * k
    return torch.clamp(k, min=BREATH_MIN, max=BREATH_MAX)


def breath_map_range(z, light, k_breath: float = BREATH_K,
                     c: float = BREATH_C, mode: str = "useful",
                     light_ref: float = BREATH_LIGHT_REF):
    """★ 实验 12：网络输出 z → 呼吸深度，范围 **[BREATH_MIN, k\\*(light)]**。

        u = 0.5 + 0.5·z/(1+|z|)          ← softsign，值域 (0, 1)
        k = BREATH_MIN + (k*(light) − BREATH_MIN)·u

    ★ 为什么上界随光强变
      超过最大增益点 k* 之后「吸得越深拿到的越少」—— 那是一段**完全冗余**
      的动作区间：两个不同的 k 给出同一个增益。留着它，稳态能量这个
      二次目标就有两个解，训练在对称分岔点上随机落点
      （实验 11 实测：k 在 0.1~1.9 之间来回跳）。

      限制在**增益单调**的那一段之后，目标对 k 是一一的，唯一可解。

    ★ 为什么用「重参数化」而不是「夹紧」（min(k, k*)）
      夹紧会让 z 在钳位区**完全没有梯度**：k_eff ≡ k*，dk_eff/dz = 0，
      于是网络既学不到「该减小 k」，也没有任何力量把它推出来 ——
      策略会永远卡在钳位边界上（实测：260 拍的训练里 z 一步没动）。
      重参数化让 dk/dz 处处非零，收敛路径不会断。

    参数
    ----------
    z : tensor，形状 (...,)
    light : float 或与 z 前导维一致的 tensor

    返回
    -------
    与 z 同形状的 tensor
    """
    import torch

    z = torch.as_tensor(z, dtype=torch.float32)
    u = 0.5 + 0.5 * z / (1.0 + z.abs())
    if mode in ("fixed", "fixed_excess"):
        hi = torch.as_tensor(
            breath_useful_max(float(light_ref), k_breath, c, mode),
            dtype=z.dtype, device=z.device)
    elif torch.is_tensor(light):
        hi = breath_useful_max_tensor(light.to(z.device), k_breath, c, mode,
                                      light_ref)
    else:
        hi = torch.as_tensor(breath_useful_max(float(light), k_breath, c, mode,
                                               light_ref),
                             dtype=z.dtype, device=z.device)
    return BREATH_MIN + (hi - BREATH_MIN) * u


def breath_shallow_k(light: float, gamma: float, energy_cap: float, dt: float,
                     center: float = None, mode: str = "step",
                     move_cost: float = 0.0) -> float:
    """稳态能量停在中心时对应的**浅解**呼吸深度（闭式解）。

        K·L·k − C·k² = g*   →   k = [K·L − √((K·L)² − 4C·g*)] / (2C)

    判别式 < 0（光太暗、目标在物理上不可达）时，最优解就是最大增益点 k*。

    ★ 这个函数只用于**测试和作图**（给出理论预测 k(light) 曲线），
      不参与训练 —— 训练仍然只是把 E∞ 往中心拽，路径要网络自己学。
    """
    g_star = breath_optimal_gain(gamma, energy_cap, dt, center, mode) \
        + float(move_cost)
    a = BREATH_K * float(light)
    disc = a * a - 4.0 * BREATH_C * g_star
    if disc <= 0.0:
        return a / (2.0 * BREATH_C)
    return (a - math.sqrt(disc)) / (2.0 * BREATH_C)



# ══════════════════════════════════════════════════════════════════════
# 体重
# ══════════════════════════════════════════════════════════════════════

def mass_of(grid: int) -> float:
    """体重 = 矩阵的乘积 = grid²。

    128×128 → 16,384。将来长到 129×129 → 16,641。
    """
    return float(grid) * float(grid)


def energy_capacity(mass: float) -> float:
    """能量容量 E₀ ∝ 体重。长大之后能量预算跟着涨。"""
    return ENERGY_PER_MASS * float(mass)


# ══════════════════════════════════════════════════════════════════════
# 系数
# ══════════════════════════════════════════════════════════════════════

def mass_coef(energy: float, mass: float) -> float:
    """质量系数 = √(E/m) = 场的均方根幅度。"""
    if mass <= 0 or energy <= 0:
        return 0.0
    return math.sqrt(float(energy) / float(mass))


def phys_x(energy_norm: float) -> float:
    """把归一化能量映射到物理法则的自变量 x。

    energy_norm = E / E₀，生存带内 ∈ [0.2, 3.0] → x ∈ [−6, +6]
    """
    return X_MAX * (float(energy_norm) - BAND_CENTER) / BAND_HALF


def phys_coef(energy_norm: float) -> float:
    """物理法则系数 f(x) = (1 + sin(x²)) / (1 + x²)。

    ∈ [0, 1]，x=0 时 = 1，|x|→∞ 时 → 0。永远不为负。
    """
    x = phys_x(energy_norm)
    x2 = x * x
    if x2 < 1e-12:
        return 1.0
    return (1.0 + math.sin(x2)) / (1.0 + x2)


def speed_scale(energy: float, mass: float, energy_cap: float) -> float:
    """速度缩放 = 质量系数 × 物理法则系数，裁到 [0, 1]。"""
    if energy_cap <= 0:
        return 0.0
    s = mass_coef(energy, mass)
    f = phys_coef(energy / energy_cap)
    return max(0.0, min(1.0, s * f))


def speed_scale_tensor(energy, mass: float, energy_cap: float):
    """**可微版** speed_scale —— 写场头的训练目标。

    ★ 为什么写场头该用这个当目标：
      写场头不在动作分布里，拿不到策略梯度（∂logπ/∂write_head = 0），
      必须给它一个可微的目标。之前用的是「最大化 ΔE」，等于告诉它
      「拼命充能」—— 结果每代写得更猛、每代死得更快（实测 90 拍 → 22 拍）。

      改用「最大化速度能力」之后：
        - 目标不是我编的常数，而是**物理法则曲线的峰值**（≈ 17480，1.75×E₀）
        - 两侧都有梯度（低了正、高了负），是真优化问题
        - 峰值离上边界 3.00 还有 1.25× 余量，不会自爆

    参数
    ----------
    energy : torch.Tensor
        写入之后的场能量，必须带梯度。
    """
    import torch

    e = torch.clamp(energy, min=1e-9)
    s = torch.sqrt(e / float(mass))
    x = X_MAX * (e / float(energy_cap) - BAND_CENTER) / BAND_HALF
    f = (1.0 + torch.sin(x * x)) / (1.0 + x * x)
    return s * f



# ══════════════════════════════════════════════════════════════════════
# 从「力/意愿」到「世界速度」
# ══════════════════════════════════════════════════════════════════════

def force_to_velocity(force, energy: float, mass: float, energy_cap: float):
    """把 T 输出的力矢量转成世界速度矢量。

    参数
    ----------
    force : 长度为 2 的序列（或 numpy 数组）
        力 / 意愿矢量（x, z），被 tanh 压在 [−1,1]²。

    返回
    -------
    (velocity_world, v_norm)
        velocity_world : (2,) 世界单位/秒
        v_norm         : 归一化速度 ∈ [0,1]
    """
    import numpy as np

    f = np.asarray(force, dtype=np.float64).reshape(2)
    mag = float(np.linalg.norm(f))
    if mag < 1e-9:
        return np.zeros(2, dtype=np.float64), 0.0

    scale = speed_scale(energy, mass, energy_cap)
    v_norm = min(1.0, mag * scale)          # 意愿 × 两个系数，再裁到 ≤1
    direction = f / mag
    return direction * v_norm * V_WORLD, v_norm


# ══════════════════════════════════════════════════════════════════════
# 运动耗能
# ══════════════════════════════════════════════════════════════════════

def move_energy_cost(v_norm: float, mass: float) -> float:
    """运动一拍烧掉的能量 = K_MOVE · ½ m v²。

    标定：中央满速（v_norm=1）跑一拍 ≈ 1.5% 的 16000 = 240。
    """
    return K_MOVE * 0.5 * float(mass) * float(v_norm) ** 2


# ══════════════════════════════════════════════════════════════════════
# 生死
# ══════════════════════════════════════════════════════════════════════

def death_cause(energy: float, energy_cap: float) -> str:
    """'low' / 'high' / ''（还活着）。"""
    if energy_cap <= 0:
        return "low"
    r = float(energy) / float(energy_cap)
    if r < LOWER:
        return "low"
    if r > UPPER:
        return "high"
    return ""


def describe_bounds(energy_cap: float) -> dict:
    """返回生存带的绝对数值，方便日志/诊断。"""
    return {
        "energy_cap": energy_cap,
        "lower": energy_cap * LOWER,
        "upper": energy_cap * UPPER,
        "center": energy_cap * BAND_CENTER,
    }
