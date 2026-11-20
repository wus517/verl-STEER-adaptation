# STEER 四象限加权：平方曲线助推Ⅱ与Ⅳ

在 GRPO＋STEER 的逐 token 策略损失中，Ⅰ使用原 STEER 的指数权重，Ⅲ使用独立可调的权重。Ⅱ和Ⅳ共用一条平方曲线。**当正分数有大小差异时，最小项默认助推 20%，最大项助推 0.1%，中间项随幅度变大逐渐减小助推比例。**

| 象限 | GRPO 优势 | STEER 估计方向 | 策略项权重 |
|---|---|---|---|
| Ⅰ | 正 | 降熵 | 原 STEER 指数降权 |
| Ⅱ | 正 | 增熵 | 助推正向强化，幅度越小，加成比例越高 |
| Ⅲ | 负 | 降熵 | 独立权重，默认 `1.0`，对应原 GRPO 力度 |
| Ⅳ | 负 | 增熵 | 助推负向惩罚，幅度越小，加成比例越高 |

## 1. 输入与分数

权重函数接收现有 actor 的逐 token 张量。各输入按回答及 token 位置对齐，通常形状为 `[回答数, 回复长度]`。

| 输入 | 含义 |
|---|---|
| `advantages` | GRPO 计算的逐 token 优势。同题回答先计算组内相对优势，再分配到各回答的有效 token。 |
| `log_prob` | 当前 actor 赋予实际采样 token 的对数概率。 |
| `old_log_prob` | 采样策略赋予同一上下文、同一 token 的对数概率。 |
| `entropys` | 该位置的完整词表 Shannon 熵，张量名沿用公开实现。即时计算时，由当前 actor 在预测该 token 的同一分布上计算。 |
| `response_mask` | 回答位置的掩码，值为 `1` 的位置参加分数统计和策略损失汇总。 |

对回答 $i$ 中位置 $t$ 的 token 计算有符号分数。

$$
x_{i,t}=-\frac{A_{i,t}}{p^{\mathrm{old}}_{i,t}}\,p_{i,t}(1-p_{i,t})\bigl(\log p_{i,t}+H_{i,t}\bigr).
$$

- $i$ 是回答编号，$t$ 是该回答的 token 位置。
- $A_{i,t}$ 是 `advantages` 在该位置的值。
- $p_{i,t}$、$p^{\mathrm{old}}_{i,t}$ 是 `log_prob`、`old_log_prob` 取自然指数后的概率。参考代码采用公开实现的概率截断。
- $H_{i,t}$ 是 `entropys` 在该位置的值，$\log$ 是自然对数。
- $x_{i,t}$ 是 STEER 的有符号逐 token 分数。正值和负值分别对应估计的增熵和降熵方向。

有符号分数采用 STEER 式 (7) 中期望内的逐 token 表达形式。本文按公开仓库的 `symmetric` 指数路径在 PPO 裁剪之前统计分数。现有权重函数对分数取绝对值的位置，同时提供有符号值用于分组。以仓库训练配方做对照时，两种模式读取同一来源的熵。采用即时计算时，在 actor 前向中开启 `calculate_entropy`。熵正则项系数可以独立设为 `0`。[STEER 论文 §3 式 (7)–(8) 与图 2](paper_reading/STEER.pdf) [公开分数计算](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L619-L655)

Ⅰ沿用原 STEER 的归一化。Ⅱ和Ⅳ合在一起，用同一次权重调用中的正分数最小值和最大值归一化。

$$
M=\max_{(i,t)\in\mathcal B}|x_{i,t}|,\qquad
D=\max(M,d_{\mathrm{floor}}),\qquad
z_{i,t}=\frac{|x_{i,t}|}{D}\quad(D>0).
$$

- $\mathcal B$ 是本次权重函数收到的有效回答 token 集合。公开实现以当前 actor microbatch 的 `response_mask` 取集合。
- $M$ 是集合中的最大绝对分数。有效集合为空时，权重函数直接返回全零张量。
- $d_{\mathrm{floor}}$ 是分母下限，$D$ 是归一化分母。公开指数路径取 `0.02`。采用论文式 (9) 的纯最大值路径时，取 `0` 并沿用该路径的统计范围。
- $z_{i,t}$ 是该位置的归一化幅度，范围为 $[0,1]$。$D=0$ 时全部有效分数为零，有效位置权重统一取 `1.0`。

Ⅰ沿用已有指数路径的分数精度、mask、归一化范围与分母下限。同一组输入上的Ⅰ权重因此可直接与原 STEER 对照。[公开指数映射](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L655-L705)

## 2. 权重公式与参数

取Ⅱ和Ⅳ中全部有效正分数的最小值和最大值。

$$
x_{\min}=\min_{(i,t)\in\mathcal B,\;x_{i,t}>0} x_{i,t},
\qquad
x_{\max}=\max_{(i,t)\in\mathcal B,\;x_{i,t}>0} x_{i,t}.
$$

$$
u_{i,t}=\begin{cases}
\dfrac{x_{i,t}-x_{\min}}{x_{\max}-x_{\min}}, & x_{\max}>x_{\min},\\[6pt]
1, & x_{\max}=x_{\min}.
\end{cases}
\qquad (x_{i,t}>0)
$$

- $x_{\min}$、$x_{\max}$ 是同一次权重调用中正分数的最小值和最大值，统计集合为 $\mathcal B$ 中的有效正分数项。
- $u_{i,t}$ 是当前正分数 $x_{i,t}$ 的归一化幅度，范围为 $[0,1]$。正分数有大小差异时，最小项对应 `0`，最大项对应 `1`，并列项使用相同幅度。
- 正分数只有一个或全部相等时，统一取 $u_{i,t}=1$，使用 `0.1%` 的助推。正分数集合为空时，直接执行Ⅰ、Ⅲ和零分项的规则。

Ⅱ和Ⅳ的权重从低幅度端的最大助推，逐渐下降到高幅度端的最小助推。

$$
\lambda_+(u)=1+0.001+(\gamma-0.001)(1-u)^2,\qquad 0\le u\le1.
$$

- $u$ 是该正分数 token 的归一化幅度 $u_{i,t}$。
- $\lambda_+(u)$ 是正分数项的策略权重。
- $\gamma$ 是最小正分数项的相对加成。$u=0$ 时权重为 $1+\gamma$。
- `0.001` 是最大正分数项固定保留的相对加成。$u=1$ 时权重为 `1.001`。

默认 $\gamma=0.20$，相对加成从 `20%` 降到 `0.1%`。平方曲线在低幅度一端下降较快，在高幅度一端逐渐变平。加成指该策略项的梯度相对原 GRPO 多乘上的比例，即权重减去 `1.0`。

有效位置使用以下完整映射。

$$
\lambda_{i,t}=\begin{cases}
w_{\min}^{\,z_{i,t}}, & A_{i,t}>0,\;x_{i,t}<0,\\
\lambda_+(u_{i,t}), & x_{i,t}>0,\\
w_3, & A_{i,t}<0,\;x_{i,t}<0,\\
1, & x_{i,t}=0.
\end{cases}
$$

- $\lambda_{i,t}$ 是该位置最终乘到策略损失上的权重。
- $w_{\min}$ 是原 STEER 的 `token_weight_min`。幂函数写法等价于对归一化幅度使用原指数降权。
- $w_3$ 是第三象限的独立权重。`0.95` 保留 GRPO 更新力度的 95%，`1.0` 恢复完整力度，`1.05` 加强该项更新 5%。

| 配置项 | 默认值或示例 | 范围与用途 |
|---|---:|---|
| `token_weight_min` | `0.70` | 大于 0、小于或等于 1。Ⅰ读取原指数基线的值。 |
| `entropy_boost_strength` | `0.20` | 有限数，取值大于或等于 `0.001`，对应 $\gamma$。取 `0.001` 时，所有正分数项统一助推 `0.1%`。 |
| `quadrant_3_weight` | `1.0` | 有限正数，对应 $w_3$。 |

## 3. 接入策略损失

在 `compute_policy_loss_with_entropy` 中按模式选择权重函数，得到与 `pg_losses` 同形状的 `token_weights`。`pg_losses` 是现有 PPO 或 dual-clip 算出的逐 token 策略损失。`torch.no_grad()` 让权重作为常数参与反向传播，actor 参数通过原策略损失求梯度。

```python
weighted_pg_losses = pg_losses * token_weights
pg_loss = agg_loss(
    loss_mat=weighted_pg_losses,
    loss_mask=response_mask,
    loss_agg_mode=loss_agg_mode,
)
```

`agg_loss` 使用现有的 `response_mask` 和 `loss_agg_mode` 汇总逐 token 损失。KL 项和熵正则项由 actor 按训练配置在策略项汇总后加入。[公开策略损失接入点](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L783-L830)

## 4. PyTorch 参考实现

以下函数按公开仓库的指数路径计算分数和Ⅰ象限权重。`denominator_floor` 对应现有归一化分母下限。函数返回 FP32 权重，padding 权重为 `0`。配置加载时按第 2 节校验参数。

```python
import torch


@torch.no_grad()
def compute_quadratic_steer_weights(
    advantages,
    entropys,
    old_log_prob,
    log_prob,
    response_mask,
    *,
    token_weight_min=0.70,
    entropy_boost_strength=0.20,
    quadrant_3_weight=1.0,
    denominator_floor=0.02,
):
    valid = response_mask.bool()
    weights = torch.zeros_like(log_prob, dtype=torch.float32)
    if not valid.any():
        return weights

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

    quadrant_1 = (advantage > 0) & (score < 0)
    quadrant_3 = (advantage < 0) & (score < 0)
    increasing = score > 0  # Ⅱ和Ⅳ共用平方曲线。
    valid_weights = torch.ones_like(score, dtype=weights.dtype)

    exponent_scale = -score.new_tensor(token_weight_min).log() / denominator
    valid_weights[quadrant_1] = (
        torch.exp(-exponent_scale * magnitude[quadrant_1])
        .clamp(min=token_weight_min, max=1.0)
        .to(weights.dtype)
    )
    valid_weights[quadrant_3] = quadrant_3_weight

    positive_scores = score[increasing]
    if positive_scores.numel() > 0:
        min_score, max_score = torch.aminmax(positive_scores)
        score_span = max_score - min_score
        safe_span = torch.where(score_span > 0, score_span, torch.ones_like(score_span))
        relative_increase = (positive_scores - min_score) / safe_span
        relative_increase = torch.where(
            score_span > 0, relative_increase, torch.ones_like(relative_increase)
        )
        valid_weights[increasing] = (
            1.001 + (entropy_boost_strength - 0.001) * (1.0 - relative_increase.float()).square()
        )
    weights[valid] = valid_weights
    return weights
```

可以在现有 `entropy_control_mode` 中注册一个独立模式。下例用 `quadrant_quadratic` 作为模式名，将助推强度和第三象限权重从 actor 配置传入本函数。`linear: false` 对应原指数基线。

```yaml
policy_loss:
  loss_mode: entropy_control
  entropy_control_mode: quadrant_quadratic
  linear: false
  token_weight_min: 0.70
  entropy_boost_strength: 0.20
  quadrant_3_weight: 1.0
```

## 5. 实现核对用例

取默认配置，下表中最大绝对分数为 `1.00`，最小正分数为 `0.10`，最大正分数为 `1.00`。

| 优势符号 | 有符号分数 | 象限 | 正分数归一化幅度 $u$ | 期望权重 |
|---|---:|---|---:|---:|
| 正 | −0.60 | Ⅰ | — | 0.807344 |
| 正 | +0.10 | Ⅱ | 0.00 | 1.200000 |
| 正 | +0.325 | Ⅱ | 0.25 | 1.112938 |
| 正 | +0.55 | Ⅱ | 0.50 | 1.050750 |
| 负 | −0.90 | Ⅲ | — | 1.000000 |
| 负 | +0.775 | Ⅳ | 0.75 | 1.013438 |
| 负 | +1.00 | Ⅳ | 1.00 | 1.001000 |
| 零 | 0.00 | 零分 | — | 1.000000 |

单 token 算例。下一 token 概率分布取 `(0.7, 0.2, 0.1)`，熵约为 `0.801819`。实际采到第三个 token，旧策略对它的概率同样为 `0.1`。当优势为 `+0.20`，分数约为 `+0.270138`，归入Ⅱ。若该分数正好位于批内正分数最小值和最大值的中间，归一化幅度为 `0.50`，权重为 `1.05075`。当前与旧策略的概率比为 `1.0`，原逐 token 策略损失为 `−0.20`，加权后为 `−0.21015`。

- **助推端点**。取 `entropy_boost_strength=0.20`。最小和最大正分数有差异时，两端权重分别为 `1.20` 和 `1.001`。
- **两种归一化**。输入分数 `−2.00`、`+0.002`、`+0.01` 时，Ⅰ的归一化分母为 `2.00`。两个正分数分别对应幅度 `0` 和 `1`，权重分别为 `1.20` 和 `1.001`。
- **正分数全相等**。只有一个正分数或所有正分数相等时，每个正分数的权重为 `1.001`。
- **Ⅰ的等价检查**。用同一组 FP32 输入运行原 `symmetric` 指数模式和本模式，分别覆盖最大分数高于和低于 `0.02` 的批次，比较Ⅰ的权重。
- **Ⅲ的配置检查**。把 `quadrant_3_weight` 改为 `0.95`，表中Ⅲ的期望权重变为 `0.95`。
- **零分与 padding**。有效位置的零分权重为 `1.0`。padding 权重为 `0`，并从分母统计中排除。全空 mask 返回全零张量。
- **梯度检查**。权重的 `requires_grad` 为 `False`，原策略损失产生的该项梯度按权重比例缩放。
