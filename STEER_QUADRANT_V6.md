# STEER V6 Ⅱ象限高熵 token 递减助推方案

## 1. V6 与 V5 的区别

V6 沿用 V5 的高熵筛选。先用本地 actor microbatch 内全部有效回答 token 的条件熵确定前 20% 的分界值，再筛出达到该阈值的Ⅱ象限位置。V5 对这些位置统一使用权重 `1.0`，V6 按入选位置的原始 STEER 分数幅度给递减助推。

| 位置 | V5 权重 | V6 权重 |
|---|---|---|
| Ⅰ、Ⅲ、Ⅳ象限 | 原始 STEER | 原始 STEER |
| Ⅱ象限，条件熵达到阈值 | `1.0` | 分数幅度越大，权重越接近 `1.0` |
| Ⅱ象限，条件熵低于阈值 | 原始 STEER | 原始 STEER |
| 零优势、象限分界和 padding | 原始 STEER 返回值 | 原始 STEER 返回值 |

权重 `1.0` 保留原 PPO 裁剪后的策略更新力度。V6 的助推比例以这个力度为基准，权重 `1.20` 表示该位置的策略损失和对应梯度系数乘以 `1.20`。

## 2. 沿用 V5 的筛选与原始权重

每次策略损失调用按以下顺序执行。

1. **划分象限。** 复用 V5 的四象限掩码。Ⅱ象限要求原 GRPO 优势为正，且当前采样 token 的截断后概率低于 STEER 的概率分界。用当前对数概率对应的截断后概率和缓存的旧策略条件熵完成判断。
2. **计算基线权重。** 直接沿用 `compute_token_weights` 的对称指数路径，使用完整的 `response_mask`。原分母是当前本地 microbatch 全部有效 token 的最大绝对 STEER 分数与 `0.02` 中的较大值。入选位置也参与这次计算。
3. **确定入选位置。** 复用 `select_q2_high_entropy_tokens`。对全部有效回答位置的有限条件熵，取最高 20% 的最小值作为阈值，取值数量向上取整。Ⅱ象限中条件熵大于或等于阈值的位置进入 `q2_release_mask`。零优势和概率分界上的有效位置参与定阈值，边界同熵的Ⅱ象限位置按同一规则入选。
4. **覆盖入选权重。** 仅在入选集合内部对原始 STEER 分数幅度做最小–最大归一化，按第 3 节计算最终权重。其余位置直接使用第 2 步得到的值。

每卡 `microbatch=1` 时，条件熵阈值从这条回答的全部有效 token 中确定，递减曲线从这条回答的入选Ⅱ象限位置中确定。各卡在本地执行筛选与加权。[V5 阈值与筛选规则](/Users/lhx/VSCodeFiles/RL_discuss/STEER_QUADRANT_V5.md:70)

## 3. V6 的递减曲线

### 3.1 分数及归一化范围

沿用 V5 的记号。对某个位置，带符号的原始 STEER 分数和对应的绝对幅度为

$$
x=-\frac{A}{p_{\mathrm{old}}}\,p(1-p)(\log p+H),
\qquad
\omega=|x|.
$$

- $x$ 是归一化前的逐 token STEER 分数，$\omega$ 是它的绝对值。对Ⅱ象限，使用这个幅度衡量该位置的估计增熵强度。
- $A$ 是该 token 的 GRPO 优势。
- $p$ 和 $p_{\mathrm{old}}$ 分别是当前策略和旧策略对同一前缀下同一已采样 token 的概率，均沿用原 STEER 的概率截断。$1-p$ 是当前策略给其余词表 token 的总概率。
- $H$ 是原 STEER 使用的缓存条件熵，表示该位置完整词表的分布熵。$\log$ 是自然对数，$\log p+H$ 是预测该 token 时的概率分界量。

**条件熵决定入选，分数幅度决定入选后的助推大小。** 分数归一化使用当前 microbatch 中入选且分数幅度有限的Ⅱ象限位置。原 STEER 权重仍使用第 2 节的原始分母。

令 $\omega_{\min}$、$\omega_{\max}$ 分别为这个集合的最小和最大分数幅度。对集合中的当前 token，计算

$$
u=
\begin{cases}
\dfrac{\omega-\omega_{\min}}{\omega_{\max}-\omega_{\min}},
& \omega_{\max}>\omega_{\min},\\[6pt]
0, & \omega_{\max}=\omega_{\min}.
\end{cases}
$$

- $u$ 是当前 token 在入选集合内的归一化分数幅度，范围为 `[0, 1]`。
- $\omega$ 是当前 token 的原始分数幅度。
- $\omega-\omega_{\min}$ 是当前分数超过组内最小值的部分，$\omega_{\max}-\omega_{\min}$ 是组内的分数跨度。
- 只有一个有效分数或所有有效分数相同时，取 $u=0$，这些位置的权重为最大助推 $1+\gamma$。

### 3.2 平方递减权重

沿用 V2 的平方曲线和两端权重。

$$
\lambda_+(u)=1+0.001+(\gamma-0.001)(1-u)^2.
$$

- $\lambda_+(u)$ 是入选 token 最终乘入裁剪后策略损失的权重。
- $1$ 是原 PPO 裁剪后的完整更新力度，`0.001` 是启用助推时保留的最低额外比例，即 `0.1%`。
- $\gamma$ 是最大额外助推比例，V6 实验取 `0.20`。$\gamma-0.001$ 是最大和最小助推之间的差。
- $u$ 是上节的归一化幅度。$(1-u)^2$ 随幅度上升从 `1` 降到 `0`，高幅度端的权重变化逐渐变小。

例如，三个已入选位置的分数幅度分别为 `0.005`、`0.025`、`0.045`。

| 原始分数幅度 | 归一化幅度 | V5 权重 | V6 权重（最大助推 20%） |
|---:|---:|---:|---:|
| `0.005` | `0.0` | `1.0` | `1.20000` |
| `0.025` | `0.5` | `1.0` | `1.05075` |
| `0.045` | `1.0` | `1.0` | `1.00100` |

该曲线按当前入选集合的相对分数分配权重。同样的原始分数幅度在不同 microbatch 中，可能因入选集合的上下界不同而得到不同权重。[V2 平方曲线](/Users/lhx/VSCodeFiles/RL_discuss/STEER_QUADRANT_RECIPROCAL_BOOST.md:75)

## 4. 配置与代码接入

### 4.1 配置

沿用 `q2_high_entropy_ratio` 控制 V5 的熵阈值。新增 `q2_high_entropy_boost_strength`，表示本方案的最大助推比例。代码默认值为 `0.0`，表示入选位置沿用 V5 的 `1.0`。启用助推时校验该值为有限数且大于或等于 `0.001`。两组继续使用同样的对称指数 STEER 配置。

| 配置项 | V5 对照 | V6 |
|---|---:|---:|
| `q2_high_entropy_ratio` | `0.20` | `0.20` |
| `q2_high_entropy_boost_strength` | `0.0` | `0.20` |
| `steer_quadrant_logging` | `true` | `true` |

关闭熵筛选时，完整使用原 STEER 权重。筛选开启而助推强度为零时，按 V5 将入选位置设为 `1.0`。

### 4.2 参考权重覆盖

在 `compute_policy_loss_with_entropy` 内复用 V5 的基线权重和筛选掩码。V6 可从原权重函数取出有限性处理前的逐 token `metric`，取其绝对值作为 `score_magnitude`。对称路径中，该绝对值等于上文的 $\omega$。实现可通过可选返回值把它传给策略损失，原权重计算仍走现有算子顺序和截断。基线配置继续使用原有返回路径。[原权重及 V5 接入](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/core_algos.py:1398)

下面的参考函数以布尔掩码和固定形状归约求上下界。`release_mask` 是 V5 的入选掩码，`steer_weights` 是覆盖前的原始权重，`boost_strength` 是已校验的最大助推比例。曲线和归一化在 FP32 中计算。分数幅度出现非有限值的入选位置使用 V5 权重 `1.0`，且从归一化统计中排除。入选集合为空时，输出等于原始权重。

```python
import torch


@torch.no_grad()
def apply_q2_high_entropy_boost(
    steer_weights, release_mask, score_magnitude, boost_strength
):
    if boost_strength == 0.0:
        return steer_weights.masked_fill(release_mask, 1.0)
    if steer_weights.numel() == 0:
        return steer_weights

    magnitude = score_magnitude.float()
    eligible = release_mask & torch.isfinite(magnitude)
    minimum = magnitude.masked_fill(~eligible, torch.inf).amin()
    maximum = magnitude.masked_fill(~eligible, -torch.inf).amax()
    has_scores = eligible.any()
    minimum = torch.where(has_scores, minimum, 0.0)
    maximum = torch.where(has_scores, maximum, 0.0)

    span = maximum - minimum
    safe_span = torch.where(span > 0, span, 1.0)
    values = torch.where(eligible, magnitude, minimum)
    normalized = ((values - minimum) / safe_span).clamp(0.0, 1.0)
    normalized = torch.where(span > 0, normalized, 0.0)

    boost = 0.001 + (boost_strength - 0.001) * (1.0 - normalized).square()
    boosted_weights = 1.0 + torch.where(eligible, boost, 0.0)
    return torch.where(release_mask, boosted_weights.to(steer_weights.dtype), steer_weights)
```

随后使用现有 `agg_steer_loss(losses * weights, response_mask, loss_agg_mode)`。`losses` 是原 PPO 裁剪和 dual-clip 后的逐 token 策略损失，`weights` 是覆盖后的权重，`response_mask` 和 `loss_agg_mode` 沿用 V5。已进入 PPO 裁剪零梯度区的入选位置仍为零梯度。各卡的筛选与加权均在原梯度同步之前完成。

## 5. TensorBoard

**保留全部原有训练指标、V4 四象限指标和 V5 放行指标**，在同一个 metrics 字典追加 V6 的助推统计。V5 口径中的“放行”统一指达到原入选条件的位置，入选及活跃的定义继续沿用 V5。

| 新增指标 | 计算口径 |
|---|---|
| `steer/q2_release_boost_mean` | 入选且活跃位置的最终权重减 `1.0` 后求平均，记录这些位置相对 V5 的实际助推 |

原 `steer/q2_release_frac` 和 `steer/q2_release_active_frac` 继续报告入选范围，`steer/q2_release_base_weight_mean` 继续读取覆盖前的原始 STEER 权重，`steer/q2_weight_mean` 读取实际乘入损失的权重。新增助推指标复用 V5 的入选活跃计数，将助推之和与现有统计打包，在日志步做一次数据并行归约，再除以全局入选活跃计数。分母为零时跳过该指标。V5 对照的助推强度为零，有入选活跃位置时该值为零。

## 6. 对照与验收

- **V5 与 V6 对照。** 保持两组模型起点、训练数据、rollout、microbatch、损失汇总和 STEER 配置一致，采用相同的 `0.20` 熵筛选比例，比较两种入选权重。
- **筛选一致性。** 对同一份固定输入，核对两组熵阈值、`q2_release_mask`、基线权重逐项相同。加入高熵非Ⅱ象限 token 提高阈值的用例，核对它参与定阈值并沿用原权重。
- **权重边界。** 验证分数跨度存在时得到上表三个权重，单项或同分时得到 $1+\gamma$。覆盖空集合、非有限分数，以及未入选位置保持原权重的情况。
- **损失与梯度。** 最大助推比例为零时，V6 与 V5 的权重、损失和逐 token 对数概率梯度逐项一致。启用助推后，核对入选且活跃位置的梯度按最终权重放大，已裁剪位置保持零梯度。
- **日志。** 回读 TensorBoard event，核对原有指标和各版本 STEER 指标同时写入；将多个 microbatch 的入选活跃数和助推之和汇总，与集中计算的均值核对。
