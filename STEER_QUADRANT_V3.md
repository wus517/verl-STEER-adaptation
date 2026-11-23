# STEER 第三版：Ⅱ象限递减助推

第三版在 GRPO＋STEER 的逐 token 策略损失上加权。Ⅰ和Ⅲ使用原 STEER 的指数降权，Ⅱ使用倒数曲线助推，Ⅳ的权重取 `1.0`，对应原 GRPO 的更新力度。TensorBoard 记录各象限的活跃比例、Ⅱ的助推力度和裁剪比例。

## 1. 用优势的正负和概率分界划分象限

对每个有效回答 token，用当前策略对该 token 的概率 $p$ 与 $e^{-H}$ 比较。$H$ 是该位置的完整词表 Shannon 熵，$e$ 是自然指数的底数。实现时计算 $b=\log p+H$，比较 $b$ 与零。$\log$ 是自然对数，$b>0$ 对应 $p>e^{-H}$，$b<0$ 对应 $p<e^{-H}$。

| 象限 | GRPO 优势 $A$ | 概率条件 | 策略权重 |
|---|---|---|---|
| Ⅰ | 正 | $b>0$ | 原 STEER 指数权重 |
| Ⅱ | 正 | $b<0$ | 原始 STEER 分数越大，额外助推比例越小 |
| Ⅲ | 负 | $b<0$ | 原 STEER 指数权重 |
| Ⅳ | 负 | $b>0$ | `1.0` |
| 分界项 | $A=0$ 或 $b=0$ | — | `1.0` |

$A$ 是原 GRPO 给该 token 分配的优势。分类读取原 STEER 分数所用的同一份熵和概率，概率截断也沿用原实现。`response_mask` 为 `1` 的回答位置进入分类和统计，padding 的权重取 `0`。

## 2. 计算原 STEER 分数和四象限权重

对每个有效 token，沿用 STEER 官方代码的逐 token 分数，保留原符号。

$$
x=-\frac{A}{p_{\mathrm{old}}}\,p(1-p)(\log p+H).
$$

- $x$ 是原始、尚未归一化的 STEER 逐 token 分数。正分数对应原分析中的增熵方向，负分数对应降熵方向。
- $p$ 和 $p_{\mathrm{old}}$ 分别是当前策略和采样策略对同一前缀下同一 token 的概率。
- $A$、$H$ 和 $\log p+H$ 与第 1 节使用的量相同。

论文式 (7) 的 $\Omega(s)$ 是对同一上下文 $s$ 下候选动作取期望得到的量，$s$ 指提示词和已生成前缀。这里使用公开实现中对应的逐 token 分数 $x$。公开指数版在 PPO 裁剪前计算这些分数，并以本地 actor microbatch 为统计范围。[STEER 论文 §3 式 (7)–(8)、图 2 与 §5 式 (9)](paper_reading/STEER.pdf) · [公开分数实现](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L619-L660)

### Ⅰ、Ⅲ的原 STEER 权重

$$
D=\max\!\left(0.02,\max_{j\in\mathcal B}|x_j|\right),
\qquad
\lambda_{\mathrm{I,III}}(x)=w_{\min}^{\,|x|/D}.
$$

- $\mathcal B$ 是当前权重函数调用中的全部有效 token，$j$ 是其中一个 token 的编号，$x_j$ 是它的原始分数。
- $D$ 是原公开指数版的分母，等于全体有效 token 的最大绝对分数和 `0.02` 中的较大值。先计算该分母，再给各象限赋权。
- $w_{\min}$ 是现有配置 `token_weight_min`，取值大于 `0` 且小于或等于 `1`。
- $\lambda_{\mathrm{I,III}}(x)$ 是Ⅰ和Ⅲ使用的逐 token 策略权重。

当前配置 `microbatch=1` 时，$D$ 从当前卡这一条回答中取值。四个象限的有效位置都参与原分母统计。用一条回答举例，分数为 `[-0.08, +0.04, +0.01]` 时，原分母为 `0.08`。TensorBoard 的跨卡统计在权重计算之后执行。[公开归一化与指数映射](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L698-L705) · [actor 的 microbatch 调用](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/workers/actor/dp_actor.py#L448-L458)

### Ⅱ的递减助推

$$
\lambda_{\mathrm{II}}(x)=1+\frac{\gamma}{1+x/\tau},\qquad x>0.
$$

- $x$ 是第 2 节算出的原始正分数，直接代入公式。
- $\lambda_{\mathrm{II}}(x)$ 是Ⅱ的策略权重。它减去 `1` 得到相对原 PPO 策略项的助推比例。
- $\gamma$ 是助推比例的上限，建议起始值为 `0.20`。正分数接近零时，助推接近 `20%`。`0` 可用来关闭Ⅱ的额外助推。
- $\tau$ 是在训练配置中固定的正分数尺度。当 $x=\tau$ 时，助推恰好为 $\gamma/2$。示例值为 `0.02`，与Ⅰ、Ⅲ分母中的固定下限是两个独立的量。

固定 $\gamma>0$ 和 $\tau>0$ 后，任意正分数都获得正助推。分数增大时助推比例连续下降。同样大的正分数在不同回答上得到同样的Ⅱ象限权重。

| 原始Ⅱ分数 $x$ | 权重，取 $\gamma=0.20$、$\tau=0.02$ | 助推比例 |
|---:|---:|---:|
| `0.005` | `1.16` | `16%` |
| `0.020` | `1.10` | `10%` |
| `0.080` | `1.04` | `4%` |

## 3. 接到 actor 策略损失

在 `compute_policy_loss_with_entropy` 里计算 `token_weights`，并乘到原裁剪后的逐 token 策略损失 `pg_losses` 上。权重在 `torch.no_grad()` 里计算，反向传播仍通过 `pg_losses` 更新当前策略。原有 `agg_loss`、梯度累积和正则项沿用 actor 的配置。[公开策略损失接入位置](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L783-L830)

下面的参考函数返回权重、原始分数及四个象限的掩码，便于策略损失和日志共用同一套判定。四个掩码按Ⅰ、Ⅱ、Ⅲ、Ⅳ顺序排列。将训练配置中的 `token_weight_min`、`q2_boost_strength` 和 `q2_boost_scale` 传给函数。加载配置时检查助推强度为有限非负数，尺度为有限正数。

```python
import torch


@torch.no_grad()
def compute_steer_v3_weights(
    advantages, entropys, old_log_prob, log_prob, response_mask,
    *, token_weight_min, q2_boost_strength=0.20, q2_boost_scale=0.02,
):
    valid = response_mask.bool()
    weights = torch.zeros_like(log_prob, dtype=torch.float32)
    scores = torch.zeros_like(log_prob)
    quadrants = tuple(torch.zeros_like(valid) for _ in range(4))
    if not valid.any():
        return weights, scores, quadrants

    advantage = advantages[valid]
    probability = log_prob[valid].exp().clamp(min=1e-8, max=1.0 - 1e-8)
    old_probability = old_log_prob[valid].exp().clamp(min=1e-8, max=1.0)
    boundary = probability.log() + entropys[valid]
    probability_factor = probability * (1.0 - probability) * boundary
    score = -(advantage / old_probability) * probability_factor
    finite = torch.isfinite(score)
    score = torch.where(finite, score, torch.zeros_like(score))
    scores = torch.zeros_like(log_prob, dtype=score.dtype)
    scores[valid] = score

    high_probability = (boundary > 0) & finite
    low_probability = (boundary < 0) & finite
    positive_advantage = advantage > 0
    negative_advantage = advantage < 0
    q1 = positive_advantage & high_probability
    q2 = positive_advantage & low_probability
    q3 = negative_advantage & low_probability
    q4 = negative_advantage & high_probability
    for full_mask, local_mask in zip(quadrants, (q1, q2, q3, q4)):
        full_mask[valid] = local_mask

    magnitude = score.abs()
    denominator = torch.maximum(magnitude.max(), score.new_tensor(0.02))
    exponent_scale = -score.new_tensor(token_weight_min).log() / denominator
    valid_weights = torch.ones_like(score, dtype=weights.dtype)
    original_steer = q1 | q3
    valid_weights[original_steer] = (
        torch.exp(-exponent_scale * magnitude[original_steer])
        .clamp(min=token_weight_min, max=1.0)
        .to(weights.dtype)
    )
    valid_weights[q2] = 1.0 + q2_boost_strength / (
        1.0 + magnitude[q2].float() / q2_boost_scale
    )
    weights[valid] = valid_weights  # Ⅳ和分界项沿用初始化的 1.0。
    return weights, scores, quadrants
```

actor 侧可以新增 `entropy_control_mode: quadrant_v3`，为该模式接入上面的函数。对照运行可沿用 `symmetric` 和 `linear: false` 的原指数配置。第三版与原 STEER 对照时，两边读取相同的熵张量和 `token_weight_min`。

```yaml
policy_loss:
  loss_mode: entropy_control
  entropy_control_mode: quadrant_v3
  linear: false
  q2_boost_strength: 0.20
  q2_boost_scale: 0.02
```

`token_weight_min` 读取当前训练配方已有的值。两个 `q2_` 参数由 actor 传到策略损失，再传入权重函数。

## 4. TensorBoard 只记录 STEER 指标

在 `pg_losses` 生成后，结合原 PPO 裁剪条件构造指标。**活跃 token** 指原 PPO 策略项对当前采样 token 的对数概率仍有梯度的有效位置。沿用原损失的裁剪判定，包含负优势的 dual-clip 分支。若原实现对新旧对数概率差做数值截断，活跃掩码也按该截断的梯度范围处理。Ⅱ为正优势，PPO 策略项进入裁剪平坦区对应原 `clipped_by_high`。策略损失相等的裁剪边界沿用现有算子的梯度约定。

| TensorBoard 名称 | 每卡累加的分子 | 每卡累加的分母 |
|---|---|---|
| `steer/q1_active_frac` 至 `steer/q4_active_frac` | 相应象限的活跃 token 数 | 全部活跃回答 token 数，含仍有梯度的分界项 |
| `steer/q2_boost_mean` | Ⅱ活跃 token 的助推比例之和，即逐项权重减 `1` 后求和 | Ⅱ活跃 token 数 |
| `steer/q2_effective_boost` | Ⅱ活跃 token 的助推比例乘以原 PPO 梯度系数绝对值后求和 | 同一批Ⅱ活跃 token 的原 PPO 梯度系数绝对值之和 |
| `steer/q2_clip_frac` | Ⅱ有效 token 中 `clipped_by_high` 为真的数量 | Ⅱ有效 token 数，包含已裁剪项 |
| `steer/q2_at_least_half_boost_frac` | Ⅱ活跃 token 中原始分数小于或等于 `q2_boost_scale` 的数量 | Ⅱ活跃 token 数 |

这里的**原 PPO 梯度系数绝对值**指加权前，策略损失对当前采样 token 的对数概率求导所得系数的绝对值。对活跃的Ⅱ，计算时取 `abs(advantage) * ratio`，再乘原损失汇总和梯度累积对该 token 使用的系数。`ratio` 是原 PPO 损失中的新旧概率比。这个指标量化的是逐 token 策略梯度系数的相对增强，参数梯度仍由所有 token 的向量贡献汇总。

当 $\gamma=0.20$ 时，最后一项统计**获得至少 10% 助推的Ⅱ活跃 token 占比**。例如 100 个Ⅱ活跃 token 中 30 个的权重达到或超过 `1.10`，则记 `0.30`。实现上比较原始分数与 $\tau$ 即可。$\gamma=0$ 时跳过该指标。

### 日志汇总与开销

1. 每个 microbatch 在 GPU 上累加各指标的分子和分母。使用已有的 `advantages`、PPO `ratio`、裁剪掩码、分数和权重。
2. 到配置的日志步后，将这些标量打包，所有数据并行卡做一次 `all_reduce(SUM)`。跨卡归约用来汇总日志，Ⅰ、Ⅲ的权重在每次本地调用时已经算好。
3. 主卡用全局分子除以全局分母后写入 TensorBoard。某项分母为零时，该日志步跳过该比例。可以同时写 `steer/q2_active_count`，该值复用已汇总的Ⅱ活跃计数。
4. 写入后清零累加器。PyTorch 中把已汇总的标量批量转到 CPU 后写日志，GPU 累加过程覆盖多个 microbatch。

### 日志算例

假设一个日志窗口里，Ⅱ有 4 个有效 token，其中 1 个被裁剪，另外 3 个活跃项对应 `16%`、`10%`、`4%` 的助推。为便于演算，取三者原梯度系数绝对值为 `1`、`3`、`6`。

| 指标 | 写入值 |
|---|---:|
| `steer/q2_active_count` | `3` |
| `steer/q2_boost_mean` | `0.10`，三项助推比例的平均值 |
| `steer/q2_effective_boost` | `0.07`，把 `16%`、`10%`、`4%` 分别按 `1`、`3`、`6` 加权平均 |
| `steer/q2_clip_frac` | `0.25`，四项中一项进入 PPO 裁剪平坦区 |
| `steer/q2_at_least_half_boost_frac` | 约 `0.667`，三项中两项至少助推 `10%` |

## 5. 核对用例

- 对同一输入比较原 STEER 指数版和第三版，单独核对Ⅰ和Ⅲ的权重。分别覆盖最大绝对分数低于和高于 `0.02` 的情况。
- 对Ⅱ核对上面的三组分数和权重。保持其中某项的原始分数相同，改变其他象限的分数，再核对该Ⅱ项的权重。
- 核对Ⅳ和分界项权重为 `1`，padding 权重为 `0`。核对Ⅱ已裁剪项乘权重后仍处于同一裁剪分支，策略梯度为零。
- 将日志算例拆到不同 microbatch 和数据并行卡，汇总后的指标与同一张量直接计算的结果一致。
