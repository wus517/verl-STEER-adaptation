# STEER V5 Ⅱ象限高熵 token 放行方案

## 1. 实验设计

V5 以 V4 的对称指数版 STEER 为对照。在每次 actor 策略损失计算中，先沿用 V4 对有效回答 token 划分象限，并为整个本地 microbatch 计算原始 STEER 权重。随后用当前 microbatch 全部有效回答 token 的条件熵确定前 20% 的分界值，将Ⅱ象限中达到这个分界值的位置设为 `1.0`。其余位置沿用原始 STEER 权重。

这里的放行是指保留该 token 在原 PPO 裁剪后的策略更新力度。对照运行使用 V4，使两组共用同一套象限划分、STEER 权重计算和日志统计。实验比较的是在Ⅱ象限中优先放行高熵位置之后的训练效果。

| 位置 | V5 最终权重 |
|---|---|
| Ⅰ、Ⅲ、Ⅳ象限 | 原始 STEER 权重 |
| Ⅱ象限，条件熵达到本地 microbatch 全体有效回答 token 的前 20% 分界值 | `1.0` |
| Ⅱ象限，其余位置 | 原始 STEER 权重 |
| 零优势、概率分界及 V4 的其他未分类有效位置 | 原始 STEER 权重 |
| Padding | 原始 STEER 返回的 `0` |

## 2. 数据来源与象限划分

| 输入 | 含义 |
|---|---|
| `advantages` | 原 GRPO 给各回答 token 分配的优势 |
| `entropys` | 原 STEER 使用的逐位置、完整词表条件熵。当前适配代码读取缓存的旧策略熵 |
| `log_prob` | 当前 actor 对同一位置已采样 token 的对数概率 |
| `old_log_prob` | 旧策略对同一位置已采样 token 的对数概率 |
| `response_mask` | 有效回答位置为 `1`，其余位置为 `0` |

复用 `compute_steer_v4_weights` 返回的四象限掩码。对当前 token，令 $A$ 为原 GRPO 优势，$p$ 为原 STEER 截断后的当前采样 token 概率，$H$ 为该位置传入的条件熵。概率分界量 $b=\log p+H$，其中 $\log$ 是自然对数。$b<0$ 对应 $p<e^{-H}$，$e$ 是自然指数的底数。

| 象限 | 优势 | 概率分界量 |
|---|---|---|
| Ⅰ | $A>0$ | $b>0$ |
| Ⅱ | $A>0$ | $b<0$ |
| Ⅲ | $A<0$ | $b<0$ |
| Ⅳ | $A<0$ | $b>0$ |

有效性、有限性检查和边界位置处理均沿用 V4。象限划分与阈值计算读取同一份 `entropys`。象限划分结合当前概率和优势，放行判断比较Ⅱ象限各位置的条件熵与全体有效回答 token 的熵阈值。

筛选发生在当前本地 microbatch 进入 actor 损失计算时。使用已经与回答位置对齐的熵缓存和本次前向计算得到的 `log_prob`。每卡 `microbatch=1` 时，本卡用这一条回答的有效 token 计算熵阈值，各数据并行卡独立计算。

## 3. 原始权重与Ⅱ象限筛选

### 3.1 原始 STEER 权重

分数、分母和权重记号与 V4 一致。

$$
x=-\frac{A}{p_{\mathrm{old}}}\,p(1-p)(\log p+H).
$$

- $x$ 是当前 token 归一化前的带符号 STEER 分数。正值对应增熵方向，负值对应降熵方向。
- $A$、$p$、$H$ 和 $\log p+H$ 沿用第 2 节的定义。
- $p_{\mathrm{old}}$ 是旧策略对同一前缀下同一已采样 token 的概率，沿用原 STEER 的概率截断。
- $1-p$ 是当前策略给词表中其余 token 的总概率。

$$
D=\max\!\left(0.02,\max_{j\in\mathcal B}|x_j|\right),
\qquad
\lambda(x)=w_{\min}^{\,|x|/D}.
$$

- $\mathcal B$ 是当前本地 actor microbatch 的全部有效回答 token 集合。
- $j$ 是集合中的一个 token 位置，$x_j$ 是该位置的原始分数，$|x_j|$ 是分数的绝对值。
- $D$ 是全体有效 token 的最大绝对分数与固定下限 `0.02` 中的较大值。
- $w_{\min}$ 是原配置 `token_weight_min`，沿用 V4 对照组的值。
- $|x|/D$ 是当前 token 的分数绝对值除以共享分母。
- $\lambda(x)$ 是覆盖前该 token 的原 STEER 权重。

实际权重以原始 `compute_token_weights` 返回值为准，复用其对称指数路径的运算顺序和浮点截断。**先以完整 `response_mask` 算原权重，再覆盖入选位置。** Ⅱ象限入选和未入选的位置都参与原分母计算，和其他象限共用同一个分母。放行操作完成后直接使用最终权重。

### 3.2 用全部有效回答 token 定阈值

1. 用 `response_mask` 取当前 microbatch 内条件熵有限的有效回答 token。四个象限、零优势和概率分界上的有效位置都可以参与计算阈值。Padding 与非有限熵位置排除。
2. 将有效熵的数量乘 `q2_high_entropy_ratio`（V5 训练取 `0.20`），再向上取整。用 `torch.topk` 取得最高的这部分熵值，其中的最小值就是本次熵阈值。
3. 用 `q2` 与“条件熵大于或等于阈值”的掩码取交集，得到 `q2_release_mask`，将这些位置的权重设为 `1.0`。

阈值在 PPO 裁剪前计算。有效熵为空、Ⅱ象限为空，或Ⅱ象限全部低于阈值时，放行掩码全为假。比例为 `0` 时关闭放行。阈值相同的Ⅱ象限位置都按同一规则放行，边界有并列时达到阈值的位置可以超过全体的 20%。

例如，一条回答有 10 个有效 token，最高的两个熵为 `2.0` 和 `1.6`，本次阈值就是 `1.6`。如果两个Ⅱ象限 token 的熵分别为 `1.2` 和 `1.6`，只放行后一个。此时Ⅱ象限的实际放行比例为 50%。如果这两个位置的熵都低于 `1.6`，本次Ⅱ象限的放行比例为 0%。

动态阈值参考 *Beyond the 80/20 Rule* 第 5.1 节。V5 沿用作者代码的本地 microbatch 统计范围，用排名分界值筛选Ⅱ象限。损失继续使用原 `response_mask` 汇总。[论文](</Users/lhx/VSCodeFiles/RL_discuss/paper_reading/Beyond the 80:20 Rule.pdf>) · [作者 top-k 实现](https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR/blob/6f5dda6f70dbee207bea28705e5f96ce12eb98df/verl/trainer/ppo/core_algos.py#L53-L87)

## 4. 代码接入

### 4.1 配置

沿用 `PolicyLossConfig` 和 Hydra 中的 `q2_high_entropy_ratio` 配置名。它表示计算Ⅱ象限放行阈值时采用的全体有效回答 token 比例，代码默认值为 `0.0`，V5 启动配置显式设为 `0.20`。检查取值为有限数且在闭区间 `[0, 1]` 内。启用放行时使用 `loss_mode: entropy_control`、`entropy_control_mode: symmetric` 和 `linear: false`。放行由比例参数控制，`steer_quadrant_logging` 控制日志，V4 和 V5 实验配置均开启日志。

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: entropy_control
      entropy_control_mode: symmetric
      linear: false
      steer_quadrant_logging: true
      q2_high_entropy_ratio: 0.20
```

### 4.2 参考实现

```python
import math

import torch


@torch.no_grad()
def select_q2_high_entropy_tokens(entropys, q2, response_mask, top_ratio):
    if top_ratio == 0.0:
        return torch.zeros_like(q2)

    valid = response_mask.bool() & torch.isfinite(entropys)
    valid_entropies = entropys[valid]
    if valid_entropies.numel() == 0:
        return torch.zeros_like(q2)

    top_count = math.ceil(valid_entropies.numel() * top_ratio)
    threshold = valid_entropies.topk(top_count, sorted=False).values.amin()
    return q2 & valid & (entropys >= threshold)
```

`q2` 是 V4 返回的Ⅱ象限布尔掩码，`top_ratio` 是已经过配置校验的比例。筛选直接用布尔掩码完成，返回值与 `q2` 同形状。

在 `compute_policy_loss_with_entropy` 中，满足“已开启象限日志或放行比例大于零”时，调用 `compute_steer_v4_weights` 获取完整的基线权重 `steer_weights`、`quadrants` 和 `other_mask`。其余调用走原 STEER 路径。在已取得象限掩码的路径按以下方式生成最终权重。

```python
q2 = quadrants[1]
q2_release_mask = select_q2_high_entropy_tokens(
    entropys, q2, response_mask, policy.q2_high_entropy_ratio
)
weights = (
    steer_weights.masked_fill(q2_release_mask, 1.0)
    if policy.q2_high_entropy_ratio > 0.0
    else steer_weights
)
```

原 PPO 概率比、裁剪和 dual-clip 分支照常生成逐 token 损失 `losses`。使用现有 `agg_steer_loss(losses * weights, response_mask, loss_agg_mode)`，其中 `loss_agg_mode` 是原损失汇总配置。`steer_weights` 由原函数在 `torch.no_grad()` 下计算，`masked_fill` 产生最终权重，梯度通过原逐 token 损失向 actor 传播。一个入选位置若已进入 PPO 裁剪的零梯度区，其策略梯度仍为零。策略损失的汇总范围、梯度累积、同步和其余训练流程沿用 V4。

## 5. TensorBoard

**V5 及后续所有版本均保留 TensorBoard 原有训练指标和 STEER 指标。** V5 同时写入训练框架原有指标、V4 的四象限统计，以及下表新增的 V5 指标。沿用现有 logger，把 STEER 字典追加到完整训练 metrics 后统一写入。V4 的权重均值统计读取本次实际乘入损失的最终权重。

| 新增指标 | 统计口径 |
|---|---|
| `steer/q2_release_frac` | Ⅱ象限入选 token 数除以全部Ⅱ象限有效 token 数。观察在线筛选的实际比例 |
| `steer/q2_release_active_frac` | 入选且活跃的Ⅱ象限 token 数除以全部Ⅱ象限活跃 token 数。观察当前有策略梯度的位置中有多少被放行 |
| `steer/q2_release_base_weight_mean` | 入选且活跃的Ⅱ象限 token 的原 STEER 权重之和，除以入选且活跃的Ⅱ象限 token 数。该均值距离 `1.0` 越远，恢复的权重越多 |

活跃指原 PPO 策略项对当前采样 token 的对数概率有有限且非零的导数，沿用 V4 的裁剪判定。阈值用全体有效回答 token 的有限熵计算，放行比例继续以Ⅱ象限的计数为分母。复用 V4 的设备端累加与日志步数据并行汇总，将新增分子和分母随原统计一起归约，再计算全局比例与均值。分母为零时跳过对应指标。

## 6. 对照与验收

- **对照配置**。V4 取 `q2_high_entropy_ratio: 0.0`，V5 取 `q2_high_entropy_ratio: 0.20`。两组使用相同起始模型、训练数据和随机种子、rollout 参数、优势计算、actor batch 与优化器配置。分别记录实验名和 TensorBoard 目录。
- **权重一致性**。对固定输入核对 V5 覆盖前的权重与 V4 逐项相同。最终权重在入选位置为 `1.0`，其余位置与 V4 逐项相同。加入一个入选Ⅱ象限位置恰好提供最大绝对 STEER 分数的用例，核对其他位置的权重。
- **筛选边界**。覆盖有效熵为空、Ⅱ象限为空、Ⅱ象限全部低于阈值、全体计数向上取整、阈值并列、高熵 padding 与非有限熵排除，以及其他象限或零优势位置提高阈值的情况。核对比例设为 `0.0` 时权重、策略损失和对数概率梯度与 V4 逐项一致。
- **梯度校验**。固定同一批输入，对比逐 token 策略梯度。入选且活跃的位置与单位权重下的 PPO 策略梯度一致，已裁剪为零梯度的入选位置仍为零。其余位置的逐 token 策略梯度保持 V4 的数值。
- **日志验收**。读取训练 event 文件，核对原有指标、V4 四象限指标和 V5 新增指标共同写入。用候选数量不同的 microbatch 核对汇总结果，确认比例使用汇总后的计数。
