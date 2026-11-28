# STEER V8 分熵权重实现方案

以 `steer_quadrant_v7` 的 `38bc132` 为代码起点，拟在 `steer_quadrant_v8` 实现。V8 复用 V7 的象限选择、条件熵缓存和双阈值筛选，给已启用象限的低熵 token 使用统一的指数映射下限，给高熵 token 使用各象限配置的下限与恢复率。[V7 策略损失入口](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/core_algos.py:1604) · [V7 方案](/Users/lhx/VSCodeFiles/RL_discuss/STEER_QUADRANT_V7.md)

## 1. 象限和高低熵划分

1. 复用 `compute_steer_v4_weights` 中的象限判定，按 GRPO 优势的正负和当前采样 token 的概率相对于 STEER 概率分界的位置，生成Ⅰ到Ⅳ的四个掩码。概率分界和筛选都使用现有的缓存旧策略条件熵 `entropys`。
2. 用 `select_high_entropy_tokens` 在当前本地 actor microbatch 的**全部有效回答位置**中取有限条件熵。按 `steer_recovery_top_ratio`（默认 `0.20`）选最高的一部分熵，数量向上取整，这部分的最小熵就是本次分界。零优势和概率分界上的有效位置也参与计算。每卡 `microbatch=1` 时，分界来自本卡当前这条回答。
3. 同时满足**条件熵达到排名分界**和**条件熵严格大于 `steer_recovery_min_entropy`**的位置归为高熵。绝对熵门槛默认 `0.5`。其余有限熵的有效位置归为低熵。与排名分界同熵的位置用同一个判断，恰好等于绝对门槛的位置归为低熵。
4. 用 `steer_recovery_quadrants` 决定哪些象限应用 V8。未启用象限、零优势、象限分界及其他未归类位置使用原 STEER 返回值。非有限熵与 padding 延续原有处理。全体有效熵为空时，两个分支均为空，直接使用原 STEER 权重。

| 所在位置 | 指数映射的下限 | 映射后的处理 |
|---|---:|---|
| 已启用象限的低熵 token | 全象限共用 `0.50`，可配置 | 直接使用映射结果 |
| 已启用象限的高熵 token | 各象限独立配置，默认均为 `0.70` | 按各象限恢复率向 `1.0` 恢复，默认均为 `0.35` |
| 未启用象限与其他有效位置 | 原 `token_weight_min`，现有训练脚本为 `0.70` | 使用原 STEER 权重 |
| Padding | 原 STEER 返回的 `0` | 使用原 STEER 权重 |

排名阈值超过 `0.5` 时，例如为 `0.80`，已启用象限内熵为 `0.70` 的位置仍走低熵分支。如果本地 microbatch 的全部有效熵都小于或等于 `0.5`，已启用象限的有限熵位置全走低熵分支。[现有阈值函数](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/core_algos.py:1516)

## 2. 指数映射与权重恢复

### 2.1 所有位置共用原 STEER 分母

$$
D=\max\!\left(0.02,\max_{j\in\mathcal B}|x_j|\right),
\qquad
z_t=\frac{|x_t|}{D}.
$$

- $\mathcal B$ 是当前本地 actor microbatch 中全部有效回答 token 的位置集合，$j$ 是其中任一位置，$t$ 是正在计算权重的位置。
- $x_j$ 和 $x_t$ 是对应位置归一化前的 STEER 逐 token 分数，竖线表示绝对值。分数计算和非有限分数处理沿用原实现。
- $D$ 是共享分母，等于所有有效分数绝对值的最大值与固定下限 `0.02` 中的较大值。
- $z_t$ 是位置 $t$ 的归一化分数幅度，取值在 `[0, 1]`。

所有有效位置都参与原分母计算。低熵、高熵和象限配置只决定该位置采用哪个映射下限与恢复率。[原分数和分母](/Users/lhx/VSCodeFiles/RL_discuss/STEER_QUADRANT_V4.md:62)

### 2.2 先用相应下限映射，再恢复高熵权重

$$
\lambda_t=m_t^{\,z_t},
\qquad
\lambda'_t=\lambda_t+p_t(1-\lambda_t).
$$

- $m_t$ 是位置 $t$ 的**恢复前映射下限**，按第 1 节表格选择。它是指数映射的底数。
- $z_t$ 是上节定义的原 STEER 归一化分数幅度，$\lambda_t$ 是指数映射得出的恢复前权重。
- $p_t$ 是该位置的恢复比例，和采样概率是两个量。已启用象限的高熵位置取所属象限的配置值，其余位置以零表示直接使用映射结果。
- $1$ 表示原 PPO 裁剪后的完整更新力度，$1-\lambda_t$ 是从映射结果到 `1.0` 的差额。$p_t(1-\lambda_t)$ 是补回的权重，$\lambda'_t$ 是最终乘入策略损失的权重。

默认高熵映射下限为 `0.70`，恢复率为 `0.35`，所以恢复后的理论下限为 **`0.805`**。实际权重取决于该位置的归一化分数幅度。对于同一象限的高熵 token，恢复前的权重越低，补回的权重越多。当恢复率小于 `1.0` 时，同组 token 保留原来的权重顺序。

| 归一化分数幅度 | 原 STEER（下限 `0.70`） | V8 低熵最终权重 | V8 高熵恢复前 | V8 高熵最终权重 |
|---:|---:|---:|---:|---:|
| `0.0` | `1.0000` | `1.0000` | `1.0000` | `1.0000` |
| `0.5` | `0.8367` | `0.7071` | `0.8367` | `0.8938` |
| `1.0` | `0.7000` | `0.5000` | `0.7000` | `0.8050` |

算例中的低熵和高熵位置均位于已启用象限。调整某一象限的高熵参数时，其他象限仍按自己的配置计算。

## 3. 配置与启动

在 `PolicyLossConfig` 和 actor Hydra 配置里增加四个字段。用 V8 开关区分 V7 的单一恢复率路径和 V8 的分熵路径，两条路径共用现有象限列表和熵阈值配置。

| 配置字段 | 通用默认值 | V8 脚本设置 | 含义 |
|---|---|---|---|
| `steer_entropy_weighting` | `false` | `true` | 启用 V8 分熵权重 |
| `steer_low_entropy_weight_min` | `0.50` | `0.50` | 所有已启用象限共用的低熵映射下限 |
| `steer_high_entropy_weight_min_by_quadrant` | `[0.70, 0.70, 0.70, 0.70]` | 同通用默认值 | Ⅰ到Ⅳ的高熵恢复前映射下限 |
| `steer_recovery_rate_by_quadrant` | `[0.35, 0.35, 0.35, 0.35]` | 同通用默认值 | Ⅰ到Ⅳ的高熵恢复率 |
| `steer_recovery_quadrants` | `[]` | `[1, 2]` | 沿用 V7 的启用象限列表 |
| `steer_recovery_top_ratio` | `0.20` | `0.20` | 沿用 V7 的排名比例 |
| `steer_recovery_min_entropy` | `0.50` | `0.50` | 沿用 V7 的绝对熵门槛 |
| `token_weight_min` | 沿用原配置 | `0.70` | 原 STEER 权重路径使用的映射下限 |

两组象限参数列表固定按Ⅰ到Ⅳ排列。例如启用 `[2, 4]` 时，Ⅱ和Ⅳ分别读取每个列表的第 2 和第 4 项。各下限取有限值且大于零、小于或等于一，恢复率取零到一之间的有限值，包含两端。V7 的标量 `steer_recovery_rate` 继续供 V7 路径使用。

V8 采用 `loss_mode=entropy_control`、`entropy_control_mode=symmetric`、`linear=false`。脚本将 V5/V6 的 `q2_high_entropy_ratio` 和 `q2_high_entropy_boost_strength` 设为 `0.0`。现有配置校验继续检查旧Ⅱ象限加权与恢复象限列表的互斥关系。

拟从已提交的 V7 datav8 脚本派生 `run_steer_quadrant_v8_datav8.sh`，沿用其训练和评估配置。脚本开放上述 V8 参数，所有 Hydra 列表 override 作为单个带引号的 shell 参数传入。V8 的实验名、checkpoint、rollout 日志和 TensorBoard 路径使用 V8 前缀，源码预检读取 V8 的入口。[V7 脚本的参数传递](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/run_steer_quadrant_v7_datav8.sh:263)

## 4. 接入策略损失

### 4.1 生成逐位置下限

在已有象限判定之后、指数映射之前计算掩码和参数。下面的参考函数复用仓库现有的 `select_high_entropy_tokens`。

```python
import torch


@torch.no_grad()
def build_v8_weight_parameters(entropys, response_mask, quadrants, policy):
    ranked_high = select_high_entropy_tokens(
        entropys, response_mask, policy.steer_recovery_top_ratio
    )
    finite_entropy = response_mask.bool() & torch.isfinite(entropys)
    high_entropy = ranked_high & (entropys > policy.steer_recovery_min_entropy)
    low_entropy = finite_entropy & ~high_entropy

    weight_min = torch.full_like(entropys, policy.token_weight_min, dtype=torch.float32)
    recovery_rate = torch.zeros_like(weight_min)
    recovery_mask = torch.zeros_like(response_mask, dtype=torch.bool)
    for quadrant in policy.steer_recovery_quadrants:
        index = quadrant - 1
        high = quadrants[index] & high_entropy
        low = quadrants[index] & low_entropy
        weight_min.masked_fill_(low, policy.steer_low_entropy_weight_min)
        weight_min.masked_fill_(high, policy.steer_high_entropy_weight_min_by_quadrant[index])
        recovery_rate.masked_fill_(high, policy.steer_recovery_rate_by_quadrant[index])
        recovery_mask |= high
    return weight_min, recovery_rate, recovery_mask, low_entropy
```

`weight_min` 是逐位置映射下限，`recovery_rate` 是逐位置恢复率，`recovery_mask` 标出已启用象限的高熵位置，`low_entropy` 用于分组统计。所有张量与 `response_mask` 同形状。`quadrants` 依次保存Ⅰ至Ⅳ的掩码。以上操作可在设备上用布尔掩码完成。

### 4.2 复用原分数与指数映射

1. 在 `compute_token_weights` 中为对称指数路径增加可选的逐位置映射下限输入。沿用完整 `response_mask` 计算 `metric` 和分母一次，并保留原标量下限路径的结果。
2. V8 启用且象限列表非空时，将第 4.1 节的 `weight_min` 传入。只用同一个分母替换相应位置的指数映射和下限截断。实现沿用原 `exp` 与 `log` 运算顺序、计算 dtype 和上限 `1.0`。下限与原配置相同的位置直接读取原标量路径的结果，确保未启用象限逐项使用原 STEER 权重。原分数与分母的计算不按象限重复执行。
3. 取得恢复前权重 `pre_recovery_weights` 后，计算 `adjusted = pre_recovery_weights + recovery_rate * (1.0 - pre_recovery_weights)`。用 `torch.where(recovery_mask, adjusted, pre_recovery_weights)` 生成最终权重。
4. 在 `compute_policy_loss_with_entropy` 中，使用最终权重乘原 PPO 裁剪及 dual-clip 后的逐 token 损失，调用现有 `agg_steer_loss`。沿用原有效位置掩码、损失汇总方式和梯度累积。掩码、映射和恢复均在 `torch.no_grad()` 下执行。

当象限列表为 `[]` 时，策略损失直接使用原始 STEER 权重，可作对照。**将恢复率设为零只关闭高熵恢复，V8 的映射下限仍然生效。** 将启用象限的两类下限都设为原 `token_weight_min`、恢复率设为 V7 的标量恢复率时，同一输入上的 V8 权重应与 V7 逐项一致。[原始权重函数](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/core_algos.py:1399)

## 5. TensorBoard

**保留原有的训练指标及 STEER 指标，在同一日志字典中追加 V8 分组统计。** 原有各象限权重均值读取最终权重。V7 的恢复增量指标继续记录「最终权重减恢复前权重」，V8 中的恢复前权重使用按高熵下限映射后的值。Ⅱ象限现有的入选活跃位置平均基础权重也读取这个恢复前值。

| 新增指标 | 口径 |
|---|---|
| `steer/qN_low_entropy_weight_mean` | 已启用象限的低熵有效位置最终权重均值 |
| `steer/qN_high_entropy_weight_mean` | 已启用象限的高熵有效位置最终权重均值 |

`qN` 对应 `q1` 至 `q4`。高熵位置的入选比例、活跃入选比例和平均恢复增量直接复用 V7 指标。活跃位置沿用 V7 定义。分组权重和与计数随原统计向量做一次 FP32 数据并行求和，再计算均值。某组计数为零时跳过该组均值。同一次运行的统计向量长度保持固定。[V7 指标归约](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/steer_metrics.py:78)

## 6. 验收

- **分组边界。** 核对最高熵低于绝对门槛、恰好等于绝对门槛、多个位置等于排名分界、全 padding、非有限熵等情况。以同一输入切换 `[1, 2]` 和 `[1, 2, 3, 4]`，两组的熵阈值和分数分母一致。
- **权重与梯度。** 对照第 2 节算例。用不同的Ⅱ、Ⅳ高熵下限和恢复率核对列表下标。空象限列表时检查原 STEER 权重、损失和对数概率梯度逐项一致。恢复率取零时检查高熵权重等于其恢复前权重，低熵仍走自定义指数映射。两类下限与原 STEER 相同且恢复率与 V7 相同时，检查 V8 与 V7 的权重及梯度一致。核对原 PPO 裁剪为零梯度的位置仍为零。
- **入口与日志。** 检查脚本语法、Hydra 列表解析和 V8 参数进入实际 actor 损失。读回 TensorBoard event 核对原有指标、STEER 指标及新增组均值。在 NPU 上核对一次 actor 更新和 HCCL FP32 日志归约。
