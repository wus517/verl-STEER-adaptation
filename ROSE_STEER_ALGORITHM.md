# ROSE 与 STEER 的四象限加权算法

在 ROSE 的策略更新中，对每个回答 token 计算 STEER 的有符号分数，再根据 ROSE 优势的正负和分数的正负设置权重。Ⅰ沿用 STEER 的指数降权。Ⅱ和Ⅳ给过小的增熵项助推，给过大的增熵项降权，中间区间保留原力度。Ⅲ使用独立权重，默认恢复原 ROSE 的更新力度。

| 象限 | ROSE 优势 | STEER 估计方向 | 策略项权重 |
|---|---|---|---|
| Ⅰ | 正 | 降熵 | 原 STEER 指数降权 |
| Ⅱ | 正 | 增熵 | 幅度过小则助推，适中则保持，过大则抑制 |
| Ⅲ | 负 | 降熵 | 独立可调，默认 `1.0` |
| Ⅳ | 负 | 增熵 | 与Ⅱ共用一条权重曲线 |

权重接在 actor 的逐 token PPO 策略损失上。输入来自同一批 ROSE 树状采样数据，包括最终优势、采样策略的 log probability，以及当前 actor 前向得到的 log probability 和词表熵。

## 1. 从 ROSE 接入最终优势

对一道题，ROSE 用经过节点的回答奖励均值估计节点价值，再用两端的价值差给片段赋值。

$$
\widehat V(b)=\frac{1}{|\mathcal R(b)|}\sum_{o\in\mathcal R(b)}R(o),
\qquad
A^{\mathrm{tree}}_{i,t}=\widehat V(b_{\mathrm{end}})-\widehat V(b_{\mathrm{start}}).
$$

- $b$ 是树上的一个节点，对应一个生成前缀。
- $\mathcal R(b)$ 是这组采样中经过节点 $b$ 的完整回答，$|\mathcal R(b)|$ 是回答数量。
- $o$ 是集合中的一条回答，$R(o)$ 是该回答的验证奖励，答对为 1，答错为 0。
- $\widehat V(b)$ 是节点的经验价值。
- $i$ 是回答编号，$t$ 是该回答中的 token 位置。
- $b_{\mathrm{start}}$、$b_{\mathrm{end}}$ 是包含位置 $t$ 的片段的起点和终点。片段覆盖起点之后直到终点的 token。
- $A^{\mathrm{tree}}_{i,t}$ 是长度校准前的片段优势。

对较长的正确回答，ROSE 在它与组内最短正确回答的分叉点之后应用长度校准。

$$
A_{i,t}=A^{\mathrm{tree}}_{i,t}-\left|A^{\mathrm{tree}}_{i,t}\right|
\left[1-\left(\frac{T_{\mathrm{short}}-b_c}{T_i-b_c}\right)^{\alpha_{\mathrm{len}}}\right].
$$

- $A_{i,t}$ 是最终进入 PPO 损失与 STEER 分数计算的同一份优势。
- $T_{\mathrm{short}}$ 是组内最短正确回答的 token 数，$T_i$ 是当前回答的 token 数。
- $b_c$ 是两条回答在树上分叉前共享的前缀长度。两个长度分别减去 $b_c$，得到各自的分叉后续写长度。
- $\alpha_{\mathrm{len}}$ 是 ROSE 的长度校准指数。

其余 token 直接使用片段优势。代码接入时直接读取现有 ROSE 计算好的最终 `advantages` 张量。共享前缀按现有 ROSE 的叶子展开方式进入损失，各副本使用各自对应的最终优势。

## 2. 计算有符号分数

对每个仍有策略梯度的回答 token，计算 STEER 第 3 节式 (7) 中期望内部的有符号分数，省略同一更新批次共用的学习率和损失分母。

$$
x_{i,t}=-\rho_{i,t}A_{i,t}(1-p_{i,t})(\log p_{i,t}+H_{i,t}),
\qquad
\rho_{i,t}=\exp(\ell_{i,t}-\ell^{\mathrm{old}}_{i,t}),
\qquad
p_{i,t}=\exp(\ell_{i,t}).
$$

$$
H_{i,t}=-\sum_{v\in\mathcal V}\pi_\theta(v\mid s_{i,t})\log\pi_\theta(v\mid s_{i,t}).
$$

- $s_{i,t}$ 是问题和第 $i$ 条回答在位置 $t$ 之前的完整前缀。
- $a_{i,t}$ 是位置 $t$ 实际采到的 token。
- $\pi_\theta$ 是当前 actor 策略，$\theta$ 是模型参数。
- $\ell_{i,t}=\log\pi_\theta(a_{i,t}\mid s_{i,t})$ 是当前 actor 对采样 token 的 log probability。
- $\ell^{\mathrm{old}}_{i,t}$ 是采样策略对同一前缀和 token 的 log probability，在该批 rollout 的 PPO 更新期间固定。
- $p_{i,t}$ 是当前 actor 对采样 token 的概率，$\rho_{i,t}$ 是当前概率与旧概率之比。
- $H_{i,t}$ 是当前 actor 在同一前缀下完整词表分布的 Shannon 熵。$\mathcal V$ 是词表，$v$ 是词表中的候选 token。$\log$ 使用自然对数，$\exp$ 是自然指数函数。
- $x_{i,t}$ 是用来选象限和比较幅度的逐 token 分数。正值表示 STEER 估计的增熵方向，负值表示估计的降熵方向。

用同一次 actor 前向计算当前 log probability 和熵，两者采用相同的概率分布设置，并对齐到预测位置 $t$ 的 logits。理论中的整状态熵变化涉及动作期望。本算法使用上式的逐 token 分数设置权重。

参与计算的 token 由 ROSE 的 response mask 和策略裁剪共同决定。在标准 PPO 裁剪下，正优势且概率比超过上限，或负优势且概率比低于下限时，该项进入裁剪平坦区。使用训练实现实际生效的裁剪区间和判定。把平坦区、padding 和零优势位置的分数设为 0。

## 3. 在同一次优化器更新内归一化

$$
M=\max_{(i,t)\in\mathcal B}|x_{i,t}|,
\qquad
z_{i,t}=\frac{|x_{i,t}|}{M}\quad(M>0).
$$

- $\mathcal B$ 是本次 actor 优化器更新所用的有效回答 token 集合，覆盖参加该更新的全部 microbatch 及数据并行设备。
- $M$ 是这个集合中分数绝对值的最大值。四个象限共用这个最大值。
- $z_{i,t}$ 是归一化后的绝对幅度，取值范围为 $[0,1]$。分数的符号仍从 $x_{i,t}$ 读取。

有效集合为空或所有分数为零时，取 $M=0$，权重全为 1。每次优化器更新重新计算一份 $M$。当一个 PPO mini-batch 被拆成多个 microbatch 做梯度累积时，先对各 microbatch 的最大值取最大，再在数据并行组内做 `MAX` 归约，最后用同一个结果构造各 microbatch 的权重。

比例阈值针对这一批的相对幅度。例如 $z=0.30$ 表示分数幅度达到本批最大幅度的 30%。当同一批的全部分数同时乘以同一个正数时，各自的 $z$ 保持相同。

## 4. 计算四象限权重

| 配置 | 建议值 | 含义 | 取值范围 |
|---|---:|---|---|
| `steer_alpha` | $-\log(0.7)$，约 0.356675 | Ⅰ的 STEER 指数降权强度。对应的最低权重为 0.7 | 大于或等于 0 |
| `entropy_band_half_width` | `0.20` | Ⅱ和Ⅳ中间区间的半宽，中心为 0.5 | 大于或等于 0，小于 0.5 |
| `entropy_weight_strength` | `0.10` | Ⅱ和Ⅳ的最大助推与抑制幅度 | 大于或等于 0，小于 1 |
| `quadrant_3_weight` | `1.00` | Ⅲ的策略项权重 | 大于 0 |

令 $d$ 表示 `entropy_band_half_width`，$\gamma$ 表示 `entropy_weight_strength`。Ⅱ和Ⅳ使用下列对称阈值和权重曲线。

$$
z_{\mathrm{low}}=0.5-d,
\qquad
z_{\mathrm{high}}=0.5+d,
$$

$$
\lambda_+(z)=
\begin{cases}
1+\gamma\dfrac{z_{\mathrm{low}}-z}{z_{\mathrm{low}}}, & 0<z<z_{\mathrm{low}},\\[6pt]
1, & z_{\mathrm{low}}\le z\le z_{\mathrm{high}},\\[4pt]
1-\gamma\dfrac{z-z_{\mathrm{high}}}{1-z_{\mathrm{high}}}, & z_{\mathrm{high}}<z\le1.
\end{cases}
$$

- $z_{\mathrm{low}}$ 和 $z_{\mathrm{high}}$ 分别是中间区间的下边界和上边界，边界本身的权重均为 1。
- $z$ 是当前 token 的归一化分数幅度，即 $z_{i,t}$。
- $\lambda_+(z)$ 是正分数 token 使用的权重。正分数趋近零时权重趋近 $1+\gamma$，幅度达到本批最大值时权重为 $1-\gamma$。恰好零分的项由下方规则统一设为 1。

默认参数下，区间为 $[0.30,0.70]$。例如 $z=0.10$ 与 $z=0.90$ 的权重分别为约 1.0667 和 0.9333，离中心等远的两处权重偏离 1 的幅度相同。最大助推与抑制幅度由同一个 $\gamma$ 控制。

$$
\lambda_{i,t}=
\begin{cases}
\exp(-\alpha_{\mathrm{STEER}}z_{i,t}), & A_{i,t}>0,\;x_{i,t}<0,\\
\lambda_+(z_{i,t}), & x_{i,t}>0,\\
\lambda_3, & A_{i,t}<0,\;x_{i,t}<0,\\
1, & x_{i,t}=0.
\end{cases}
$$

- $\lambda_{i,t}$ 是当前 token 的最终权重。
- $\alpha_{\mathrm{STEER}}$ 对应 `steer_alpha`。Ⅰ的指数映射采用 STEER 第 5 节式 (9)。
- $\lambda_3$ 对应 `quadrant_3_weight`。`0.95` 保留原 ROSE 策略项的 95%，`1.00` 恢复全部力度，`1.05` 将该项加强 5%。相对原 STEER 的增幅由这个值和该项原本的 STEER 权重共同决定。

## 5. 接到 actor 策略损失

$$
L_{i,t}^{\mathrm{PPO}}=-\min\!\left(
\rho_{i,t}A_{i,t},
\operatorname{clip}(\rho_{i,t},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})A_{i,t}
\right),
$$

$$
\mathcal L=\operatorname{Reduce}_{\mathrm{ROSE}}\!\left(
\left\{\operatorname{stopgrad}(\lambda_{i,t})L_{i,t}^{\mathrm{PPO}}\right\}_{(i,t)\in\mathcal T}
\right)+\beta\,\mathcal K_{\mathrm{ROSE}}.
$$

- $L_{i,t}^{\mathrm{PPO}}$ 是原 ROSE 的逐 token 策略损失。$\min$ 取括号内两项的较小值，前面的负号把最大化策略目标写成最小化损失。
- $\epsilon_{\mathrm{low}}$、$\epsilon_{\mathrm{high}}$ 是 PPO 概率比的下侧、上侧裁剪幅度。采用对称裁剪时，两值相等。$\operatorname{clip}$ 将概率比限制在给定区间内。
- $\mathcal T$ 是 response mask 选中的全部损失位置，策略裁剪仍由 PPO 损失本身处理。$\mathcal B$ 用于计算权重的分数和最大值，$\mathcal T$ 用于汇总原损失。
- $\operatorname{stopgrad}$ 表示把算出的权重视为反向传播中的常数。当前 actor 的 log probability 通过原 PPO 策略损失参与反向传播。
- $\operatorname{Reduce}_{\mathrm{ROSE}}$ 表示使用现有 ROSE 的 mask、求和与归一化方式。
- $\mathcal K_{\mathrm{ROSE}}$ 是现有 ROSE 聚合后的参考模型 KL 项，$\beta$ 是该项系数。权重乘在 PPO 策略项上，KL 项按原配置单独加入。
- $\mathcal L$ 是 actor 最终最小化的损失。

## 6. PyTorch 接入参考

下列两个函数分别计算分数和权重，输入的逐 token 张量形状一致。`active` 由原策略损失的梯度条件和 response mask 得到。`batch_max` 由调用者在本次优化器更新的全部 microbatch 上汇总，形状是标量张量。

```python
import math

import torch


@torch.no_grad()
def steer_signed_score(advantages, current_logp, entropy, ppo_ratio, active):
    """给当前 mini-batch 的有效、未进入裁剪平坦区的 token 计算分数。"""
    active = active.bool() & (advantages != 0)
    score = torch.zeros_like(current_logp, dtype=torch.float32)
    logp = current_logp[active].float()
    # 用 -expm1(logp) 计算 1 - exp(logp)，保留接近概率 1 时的精度。
    one_minus_p = -torch.expm1(logp)
    score[active] = (
        -ppo_ratio[active].float()
        * advantages[active].float()
        * one_minus_p
        * (logp + entropy[active].float())
    )
    return score


@torch.no_grad()
def quadrant_weights(
    score,
    advantages,
    batch_max,
    steer_alpha=-math.log(0.7),
    entropy_band_half_width=0.20,
    entropy_weight_strength=0.10,
    quadrant_3_weight=1.0,
):
    """使用该次优化器更新共用的最大值计算各 token 权重。"""
    denominator = torch.where(
        batch_max > 0, batch_max, torch.ones_like(batch_max)
    )
    z = (score.abs() / denominator).clamp(max=1.0)
    weights = torch.ones_like(score)

    quadrant_1 = (advantages > 0) & (score < 0)
    quadrant_3 = (advantages < 0) & (score < 0)
    increasing = score > 0  # 第二和第四象限

    low = 0.5 - entropy_band_half_width
    high = 0.5 + entropy_band_half_width
    small = increasing & (z < low)
    large = increasing & (z > high)

    weights[quadrant_1] = torch.exp(-steer_alpha * z[quadrant_1])
    weights[quadrant_3] = quadrant_3_weight
    weights[small] = 1.0 + entropy_weight_strength * (low - z[small]) / low
    weights[large] = 1.0 - entropy_weight_strength * (z[large] - high) / (1.0 - high)
    return weights
```

`ppo_ratio` 取策略损失在未裁剪分支上使用的概率比。标准 PPO 的 `active` 可由原 response mask 与两条平坦区条件组合得到。若训练框架对概率比另有数值裁剪或额外裁剪分支，`active` 也由实际策略损失的梯度条件给出。初始化配置时校验第 4 节中的参数范围。

一次 actor 优化器更新的调用顺序如下。

1. 读取该 mini-batch 的 ROSE 优势、旧 log probability 和 response mask。用当前 actor 计算 log probability、熵和 PPO 概率比。
2. 为各 microbatch 构造 `active` 并计算分数，汇总所有有效 token 的绝对分数最大值。单设备先合并本地 microbatch 的最大值，多设备再在数据并行组中取 `MAX`。空分片贡献 0。
3. 用同一个 `batch_max` 计算各 microbatch 权重，将 `weights * policy_loss_per_token` 传给原 ROSE 的损失汇总逻辑，再加上参考模型 KL 项。
4. 完成该 mini-batch 的反向传播和梯度累积后，执行一次 `optimizer.step()`。

实现可缓存同次前向得到的策略损失张量与分数，等最大值就绪再反传。显存需要分开处理时，可先用 `no_grad` 前向收集分数和最大值，再逐个 microbatch 做有梯度的前向与反传。两遍使用相同的 actor 参数、输入和概率分布设置。若前向含随机算子，两遍复用对应的随机状态。

## 7. 实现核对用例

取默认参数，并假设本次更新内绝对分数的最大值为 1.0。

| ROSE 优势符号 | 有符号分数 | 归一化幅度 | 象限 | 期望权重 |
|---|---:|---:|---|---:|
| 正 | −0.60 | 0.60 | Ⅰ | 0.807344 |
| 正 | +0.10 | 0.10 | Ⅱ | 1.066667 |
| 正 | +0.30 | 0.30 | Ⅱ | 1.000000 |
| 正 | +0.50 | 0.50 | Ⅱ | 1.000000 |
| 负 | −0.90 | 0.90 | Ⅲ | 1.000000 |
| 负 | +0.70 | 0.70 | Ⅳ | 1.000000 |
| 负 | +0.90 | 0.90 | Ⅳ | 0.933333 |
| 正 | +1.00 | 1.00 | Ⅱ | 0.900000 |
| 任意 | 0.00 | 0.00 | 零分 | 1.000000 |

单 token 算例。设当前词表概率为 `(0.7, 0.2, 0.1)`，实际采到第三个 token，旧策略对它的概率也是 `0.1`，最终优势为 `+0.20`。

1. 当前分布的 Shannon 熵约为 `0.801819`，概率比为 `1.0`，该项处于 PPO 的可更新区域。
2. 代入第 2 节，得到正分数 `+0.270138`，归入第二象限。
3. 若同批最大绝对分数为该分数的五倍，归一化幅度为 `0.20`，权重为 `1.033333`。
4. 原 PPO 策略损失为 `−0.20`，加权后约为 `−0.206667`。该项的策略梯度放大约 `3.33%`。

还可核对四个批次边界。

- **全零和空有效集合**。零分的权重全部为 1，分母分支返回有限结果。
- **跨 microbatch 的最大值**。把表中 `+1.00` 和 `+0.10` 放到不同 microbatch，后者仍得到 `1.066667`。
- **裁剪平坦区**。放入正优势、概率比已越过 PPO 上限的 token，其分数为 0，批次最大值由其他有效项决定。
- **原 ROSE 等价路径**。将 `steer_alpha` 和 `entropy_weight_strength` 设为 `0`，`quadrant_3_weight` 设为 `1`，全部权重为 1。策略损失及梯度与同一批数据上的原 ROSE 一致。

训练日志可按象限记录权重均值、助推和抑制的 token 比例，以及收到助推或抑制的正分数总量。通过 ROSE 树已有的分叉位置记录通向正确叶子的Ⅱ象限入口所获权重，可以直接看到难题中已采到的有效分支如何进入训练。

## 公式来源

- ROSE 的经验节点价值、片段优势和长度校准见 §3.2 式 (6)–(8)，策略损失见 §3.3 式 (9)–(10)。[ROSE 论文](paper_reading/ROSE.pdf)
- STEER 的有符号熵变化分析与四象限见 §3 式 (7)–(8) 和图 2，第一象限采用 §5 式 (9) 的指数权重。第二、第四象限的对称阈值和第三象限的独立权重由本方案定义。[STEER 论文](paper_reading/STEER.pdf)
