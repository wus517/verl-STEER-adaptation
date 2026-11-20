# STEER 四象限加权算法

本算法基于 GRPO＋STEER。使用同一个有符号分数识别四个象限，再按下表生成逐 token 权重。权重为 `1.0` 时，该 token 保留原 GRPO 策略项的力度。

| 象限 | GRPO 优势 | STEER 估计的熵变化方向 | 权重规则 |
|---|---|---|---|
| Ⅰ | 正 | 降熵 | 沿用 STEER 的指数降权 |
| Ⅱ | 正 | 增熵 | 幅度过小则助推，适中则维持原力度，过大则抑制 |
| Ⅲ | 负 | 降熵 | 采用独立常数，默认 `1.0` |
| Ⅳ | 负 | 增熵 | 与Ⅱ共用同一条权重曲线 |

## 1. 输入与接入位置

直接读取 actor loss 已有的逐 token 张量。各张量按回答和 token 位置对齐，通常形状为 `[回答数, 回复长度]`。

| 输入 | 含义 |
|---|---|
| `advantages` | GRPO 已算好的逐 token 优势。GRPO 的结果奖励版本按同题回答计算组内相对优势，再把每条回答的优势分配给该回答的有效 token。标准差归一化沿用训练配置。 |
| `log_prob` | 当前 actor 对实际采样 token 的 log probability。 |
| `old_log_prob` | 采集该批回答时使用的旧策略对同一 token 的 log probability。 |
| `entropys` | 每个上下文下整个下一 token 分布的 Shannon 熵，具体张量名沿用现有实现。 |
| `response_mask` | 回答位置的 mask，值为 `1` 的位置参与策略损失。 |

熵与 log probability 按预测同一 token 的位置对齐。按论文公式实现时，两者由当前 actor 在同一前缀下的同一分布计算。基于仓库已有训练配方做对照时，两种权重模式读取相同的熵来源。公开 actor 支持从 `data["entropys"]` 读取已有熵。选择即时计算时，在 actor 前向中显式开启 `calculate_entropy`。四象限权重需要熵张量，熵正则项的系数独立配置，可以取 `0`。[公开 actor 的调用位置](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/workers/actor/dp_actor.py#L502-L551)

## 2. 保留 STEER 分数的正负号

每项用 $(i,t)$ 标识，$i$ 是回答编号，$t$ 是回答中的 token 位置。实际实现可直接复用现有 STEER 的概率计算，把取绝对值前的分数写成以下符号约定。

$$
x_{i,t}
= -\frac{A_{i,t}}{p^{\mathrm{old}}_{i,t}}
\,p_{i,t}(1-p_{i,t})\bigl(\log p_{i,t}+H_{i,t}\bigr).
$$

- $x_{i,t}$ 是 STEER 的有符号逐 token 分数。正值表示该项在 STEER 分析下的增熵方向，负值表示降熵方向。
- $A_{i,t}$ 是现有 GRPO 输出给策略损失的优势。
- $p_{i,t}$ 是当前 actor 在该前缀下赋予实际采样 token 的概率，$p^{\mathrm{old}}_{i,t}$ 是采样策略赋予同一 token 的概率。
- $H_{i,t}$ 是该前缀下完整下一 token 分布的 Shannon 熵，$\log$ 是自然对数。

设 $v$ 是词表中的候选 token，$\mathcal V$ 是词表，$\pi_\theta(v\mid s_{i,t})$ 是当前 actor 在上下文 $s_{i,t}$ 下生成 $v$ 的概率，$\theta$ 是模型参数。上下文 $s_{i,t}$ 由输入问题和第 $t$ 个 token 之前的回答前缀组成。Shannon 熵的计算式为：

$$
H_{i,t}=-\sum_{v\in\mathcal V}\pi_\theta(v\mid s_{i,t})\log\pi_\theta(v\mid s_{i,t}).
$$

上式采用 STEER 式 (7) 期望内部的逐 token 表达式，权重计算沿用公开仓库在 PPO 裁剪之前取分数的顺序。分类时保留原符号，归一化幅度时使用绝对值。公开实现的 `asymmetric` 分支已使用这个符号方向，`symmetric` 分支取绝对值后的幅度与之相同。[STEER 论文 §3 式 (7)–(8) 与图 2](paper_reading/STEER.pdf) [公开分数实现](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L619-L655)

## 3. 复用 STEER 的归一化分母

$$
M=\max_{(i,t)\in\mathcal B}|x_{i,t}|,
\qquad
D=\max(M,d_{\mathrm{floor}}),
\qquad
z_{i,t}=\frac{|x_{i,t}|}{D}\quad(D>0).
$$

- $\mathcal B$ 是现有 STEER 单次权重计算使用的有效 token 集合。
- $M$ 是该集合内最大的绝对分数。集合为空时取 `0`。
- $d_{\mathrm{floor}}$ 是分母下限，$D$ 是实际归一化分母。
- $z_{i,t}$ 是归一化幅度，取值范围为 $[0,1]$。四个象限共用 $D$。

本文以公开仓库的 **`symmetric`＋指数映射**为代码基线。该路径以 `response_mask` 选择有效集合，在当前 actor microbatch 内求最大值，并设置 `d_floor = 0.02`。权重在 PPO 裁剪之前计算分数统计，随后乘到裁剪完成的策略项上。四象限模式使用同样的 mask、计算范围、概率截断和分母下限，因此Ⅰ可以与原指数权重逐项对齐。[公开指数映射](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L655-L705) [actor 的 microbatch 调用](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/workers/actor/dp_actor.py#L426-L458)

使用论文式 (9) 的纯最大值作为现有基线时，取 $d_{\mathrm{floor}}=0$，同时继承该基线的有效集合和统计范围。若基线按论文式 (7) 将策略梯度已进入裁剪平坦区的分数设为零，在取最大值前应用同样的处理。若 $D=0$，有效位置权重统一取 `1.0`。例如，公开仓库中若 $M=0.01$，分母仍是 $0.02$，该调用内最大的 $z$ 是 `0.50`，增熵大幅抑制区在这一调用中为空。[STEER 论文 §3 式 (7) 与 §5 式 (9)](paper_reading/STEER.pdf)

## 4. 四象限权重

### 第二、第四象限的对称调节

$$
z_{\mathrm{low}}=0.5-d,
\qquad
z_{\mathrm{high}}=0.5+d.
$$

$$
\lambda_+(z)=
\begin{cases}
1+\gamma\dfrac{z_{\mathrm{low}}-z}{z_{\mathrm{low}}}, & 0<z<z_{\mathrm{low}},\\[6pt]
1, & z_{\mathrm{low}}\le z\le z_{\mathrm{high}},\\[4pt]
1-\gamma\dfrac{z-z_{\mathrm{high}}}{1-z_{\mathrm{high}}}, & z_{\mathrm{high}}<z\le1.
\end{cases}
$$

- $z$ 是当前 token 的归一化分数幅度 $z_{i,t}$。这条曲线用于 $x_{i,t}>0$ 的项。
- $d$ 是中间区间的半宽，建议值为 `0.20`。下边界 $z_{\mathrm{low}}$ 为 `0.30`，上边界 $z_{\mathrm{high}}$ 为 `0.70`，两个边界的权重均为 `1.0`。
- $\gamma$ 是两侧各自偏离 `1.0` 的幅度上限，建议值为 `0.10`。
- $\lambda_+(z)$ 是增熵项的权重。低幅度侧从接近 $1+\gamma$ 逐渐回到 `1.0`，高幅度侧从 `1.0` 逐渐降到 $1-\gamma$。正分数趋近零时取曲线的右极限，恰好零分时走下方的零分规则。

低幅度侧和高幅度侧关于 `0.5` 对称。例如 `0.10` 和 `0.90` 对应 `1.066667` 和 `0.933333`。这种对称性描述的是**策略项的相对缩放比例**。其绝对调整量还会随原策略项的大小变化。

### 完整映射

$$
\lambda_{i,t}=
\begin{cases}
w_{\min}^{\,z_{i,t}}, & A_{i,t}>0,\;x_{i,t}<0,\\
\lambda_+(z_{i,t}), & x_{i,t}>0,\\
w_3, & A_{i,t}<0,\;x_{i,t}<0,\\
1, & x_{i,t}=0.
\end{cases}
$$

- $\lambda_{i,t}$ 是当前有效 token 的权重。
- $w_{\min}$ 是原 STEER 指数模式的最小权重。$w_{\min}^{z_{i,t}}$ 等于 $\exp(-\alpha z_{i,t})$，其中 $\alpha=-\log w_{\min}$，$\exp$ 是自然指数函数。两种写法对应相同的指数映射。
- $w_3$ 是第三象限的独立权重。`0.95` 保留原 GRPO 策略项的 95%，`1.0` 恢复原 GRPO 的完整力度，`1.05` 将该项加强 5%。与原 STEER 的差别由 $w_3$ 和该项原本的指数权重比较得出。

| 配置项 | 示例值 | 合法范围 | 用途 |
|---|---:|---|---|
| `token_weight_min` | `0.70` | 大于 0，小于或等于 1 | 原有 STEER 参数，对应 $w_{\min}$。Ⅰ沿用现有指数基线的值。 |
| `entropy_band_half_width` | `0.20` | 大于或等于 0，小于 0.5 | 控制Ⅱ和Ⅳ的中间区间半宽 $d$。 |
| `entropy_weight_strength` | `0.10` | 大于或等于 0，小于 1 | 控制Ⅱ和Ⅳ两侧的幅度上限 $\gamma$。 |
| `quadrant_3_weight` | `1.00` | 有限且大于 0 | 第三象限权重 $w_3$。 |

`response_mask` 为 `0` 的位置在参考代码中返回权重 `0`，对应公开权重函数的返回形式。有效位置的权重按照上表计算。Ⅱ和Ⅳ权重可以超过 `1.0`，上限由 `entropy_weight_strength` 给出。`token_weight_min` 只约束Ⅰ的指数映射。

## 5. 加权策略损失

$$
\mathcal L_{\mathrm{policy}}=\operatorname{Reduce}_{\mathrm{base}}\!\left(
\bigl\{\operatorname{stopgrad}(\lambda_{i,t})\,L^{\mathrm{clip}}_{i,t}\bigr\},
\mathrm{response\_mask}\right).
$$

- $L^{\mathrm{clip}}_{i,t}$ 是现有策略损失已经完成 PPO 裁剪后的逐 token 损失，公开实现中的变量名为 `pg_losses`。若使用 dual-clip，也在这个变量生成后应用权重。
- $\operatorname{stopgrad}$ 表示反向传播把权重视为常数。PyTorch 中通过 `torch.no_grad()` 计算分数和权重。
- $\operatorname{Reduce}_{\mathrm{base}}$ 是 actor 现有的 `agg_loss`，按原 `response_mask` 和 `loss_agg_mode` 汇总。
- $\mathcal L_{\mathrm{policy}}$ 是加权后的策略损失。训练配置中的 KL 项和熵正则项在 actor 中按原入口另行加入。

公开实现的接入点是 `compute_policy_loss_with_entropy` 内的 `compute_token_weights` 调用。让新的权重模式返回 `token_weights`，即可复用 `weighted_pg_losses = pg_losses * token_weights` 与后续统计和汇总。actor 的配置读取处把三个新增参数传入策略损失，再传到权重函数。[公开策略损失接入点](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L783-L830)

## 6. PyTorch 参考实现

以下函数可直接作为新模式的权重函数。它使用公开仓库指数模式的概率截断、mask 和分母下限。配置加载时按上表校验参数。`denominator_floor` 取现有基线的值，公开仓库示例为 `0.02`。

```python
import torch


@torch.no_grad()
def compute_quadrant_steer_weights(
    advantages,
    entropys,
    old_log_prob,
    log_prob,
    response_mask,
    *,
    token_weight_min=0.70,
    entropy_band_half_width=0.20,
    entropy_weight_strength=0.10,
    quadrant_3_weight=1.00,
    denominator_floor=0.02,
):
    """在现有 STEER 的一次权重计算范围内返回逐 token 权重。"""
    valid = response_mask.bool()
    weights = torch.zeros_like(log_prob, dtype=torch.float32)
    if not valid.any():
        return weights

    # 公式和计算精度沿用基线，保留分数的正负号。
    advantage = advantages[valid]
    probability = log_prob[valid].exp().clamp(min=1e-8, max=1.0 - 1e-8)
    old_probability = old_log_prob[valid].exp().clamp(min=1e-8, max=1.0)
    probability_factor = (
        probability * (1.0 - probability)
        * (probability.log() + entropys[valid])
    )
    score = -(advantage / old_probability) * probability_factor
    score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))

    magnitude = score.abs()
    denominator = torch.maximum(magnitude.max(), score.new_tensor(denominator_floor))
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    z = magnitude / denominator

    low = 0.5 - entropy_band_half_width
    high = 0.5 + entropy_band_half_width
    increasing = score > 0
    quadrant_1 = (advantage > 0) & (score < 0)
    quadrant_3 = (advantage < 0) & (score < 0)
    boost = increasing & (z < low)
    suppress = increasing & (z > high)

    valid_weights = torch.ones_like(score, dtype=weights.dtype)
    # 与公开基线使用相同的运算顺序和夹取区间。
    exponent_scale = -score.new_tensor(token_weight_min).log() / denominator
    valid_weights[quadrant_1] = (
        torch.exp(-exponent_scale * magnitude[quadrant_1])
        .clamp(min=token_weight_min, max=1.0)
        .to(weights.dtype)
    )
    valid_weights[quadrant_3] = quadrant_3_weight
    valid_weights[boost] = (
        1.0 + entropy_weight_strength * (low - z[boost]) / low
    ).to(weights.dtype)
    valid_weights[suppress] = (
        1.0 - entropy_weight_strength * (z[suppress] - high) / (1.0 - high)
    ).to(weights.dtype)
    weights[valid] = valid_weights
    return weights
```

例如，可在已有 `entropy_control_mode` 中新增 `quadrant` 分支，并为该模式指定指数基线。以下 YAML 使用公开仓库现有的策略损失配置层级，三个新增参数由该分支读取。

```yaml
policy_loss:
  loss_mode: entropy_control
  entropy_control_mode: quadrant
  linear: false
  token_weight_min: 0.70
  entropy_band_half_width: 0.20
  entropy_weight_strength: 0.10
  quadrant_3_weight: 1.00
```

训练调用顺序为先得到 GRPO `advantages`，再取得 STEER 所用的 log probability 和熵，随后计算 `token_weights`。用它乘裁剪后的 `pg_losses`，再执行现有的损失汇总和反向传播。用于切回原 STEER 的对照配置为 `entropy_control_mode: symmetric`、`linear: false`，其余已有训练配置与新模式一致。

## 7. 实现核对用例

取默认配置，假设当前有效集合的最大绝对分数为 `1.00`。下表中的分数是第 2 节公式的结果，归一化后的幅度等于分数绝对值。

| GRPO 优势 | 有符号分数 | 象限 | 期望权重 |
|---|---:|---|---:|
| 正 | −0.60 | Ⅰ | 0.807344 |
| 正 | +0.10 | Ⅱ | 1.066667 |
| 正 | +0.30 | Ⅱ，下边界 | 1.000000 |
| 正 | +0.50 | Ⅱ，中间区间 | 1.000000 |
| 负 | −0.90 | Ⅲ | 1.000000 |
| 负 | +0.70 | Ⅳ，上边界 | 1.000000 |
| 负 | +0.90 | Ⅳ | 0.933333 |
| 正 | +1.00 | Ⅱ | 0.900000 |
| 零 | 0.00 | 零分 | 1.000000 |

可用一个三 token 词表核对符号。当前分布为 `(0.7, 0.2, 0.1)`，熵约为 `0.801819`。实际采到第三个 token，旧策略对它的概率也是 `0.1`。取 GRPO 优势 `+0.20`，所得分数约为 `+0.270138`，对应Ⅱ。若该 token 的归一化幅度为 `0.20`，权重为 `1.033333`。当前与旧策略的概率比为 `1.0`，标准 PPO 的原策略损失为 `−0.20`，加权后为 `−0.206667`。

- **Ⅰ的等价检查**。用同一组 FP32 输入分别运行原 `symmetric` 指数模式和新模式，比较Ⅰ象限的权重。分别覆盖最大分数高于和低于 `0.02` 的批次，两个模式在Ⅰ象限的权重逐项相等。
- **Ⅲ的配置检查**。将 `quadrant_3_weight` 从 `1.00` 改到 `0.95`，表中Ⅲ的权重变为 `0.95`，其他象限的期望值照表读取。
- **零分与空 mask**。全零分的有效位置返回 `1.0`。全空 mask 返回同形状的全零张量。
- **mask 与分母**。把一个绝对分数大于 `1.0` 的位置设为 padding 后，其权重为 `0`，其余位置仍使用 `1.0` 作为最大值。
- **原 GRPO 等价检查**。设置 `token_weight_min=1.0`、`entropy_weight_strength=0.0`、`quadrant_3_weight=1.0`，所有有效位置权重为 `1.0`。在相同 actor 输入和训练配置下，策略损失及对应梯度与原 GRPO 一致。
- **停止梯度检查**。权重张量的 `requires_grad` 为 `False`。加权后的策略损失仍通过原 PPO 项对当前 actor 参数求梯度。
