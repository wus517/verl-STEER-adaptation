# STEER V7 高熵 token 权重恢复实现方案

代码以 V6 的 `516dae0` 为起点，目标分支为 `steer_quadrant_v7`。复用现有 actor 策略损失、象限掩码、条件熵缓存和 FP32 指标归约。本方案约定在原始对称指数 STEER 权重上恢复，V6 的平方曲线继续由其原配置单独启用。[当前策略损失入口](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/core_algos.py:1565)

## 1. 双阈值和象限选择

每次本地 actor microbatch 进入策略损失计算时，按以下规则生成恢复掩码。

1. **划分象限。** 复用 `compute_steer_v4_weights` 返回的Ⅰ、Ⅱ、Ⅲ、Ⅳ掩码。沿用当前采样 token 概率、原 GRPO 优势和缓存的旧策略条件熵。零优势和概率分界上的位置沿用现有归类。
2. **计算前 20% 的熵阈值。** 从 `response_mask` 标出的全部有效回答位置取有限的条件熵。将位置数乘 `0.20` 后向上取整，取对应数量的最高熵值，其中最小的熵是本次阈值。各象限和未归入四个象限的有效位置都参与定阈值。每卡 `microbatch=1` 时，阈值来自本卡当前这一条回答。
3. **应用双阈值。** 位置的条件熵须达到前 20% 的阈值，且严格大于 `0.5`。熵恰好等于排名阈值时沿用 V5 的处理，熵恰好等于 `0.5` 时排除。全体有效熵为空时，恢复掩码全为假。
4. **应用象限配置。** 将通过双阈值的掩码与所选象限的并集取交集。象限配置只改变哪些位置有资格恢复权重，每个 microbatch 仍共用一个由全部有效回答 token 决定的熵阈值。

| 当前 microbatch 的前 20% 熵阈值 | 当前 token 条件熵 | 所属象限已选时的结果 |
|---:|---:|---|
| `0.30` | `0.50` | 使用原权重 |
| `0.30` | `0.51` | 恢复权重 |
| `0.80` | `0.79` | 使用原权重 |
| `0.80` | `0.80` | 恢复权重 |

同熵位置按同一阈值判断。实际恢复比例取决于双阈值和象限掩码的交集，可以为零。[V5 前 20% 阈值的计算方式](/Users/lhx/VSCodeFiles/RL_discuss/STEER_QUADRANT_V5.md:70)

## 2. 权重计算

先用完整的 `response_mask` 计算原始 STEER 权重。原 STEER 归一化分母取本地 microbatch 全部有效 token 的最大绝对分数与 `0.02` 中的较大值。入选位置也参与原分母计算。[STEER 原分数、分母及权重](/Users/lhx/VSCodeFiles/RL_discuss/STEER_QUADRANT_V4.md:62)

对恢复掩码为真的位置应用

$$
\lambda'_t=\lambda_t+\rho(1-\lambda_t),\qquad 0\le\rho\le1.
$$

- $t$ 是当前回答 token 的位置，$\lambda_t$ 是原始对称指数 STEER 的逐 token 权重，$\lambda'_t$ 是最终乘入该位置策略损失的权重。
- $1$ 对应原 PPO 裁剪后的完整更新力度，$1-\lambda_t$ 是原 STEER 扣掉的权重。
- $\rho$ 是恢复比例，$\rho(1-\lambda_t)$ 是补回的权重。所有启用的象限共用一个恢复比例。

| 原始 STEER 权重 | 恢复比例 | 权重增量 | 最终权重 |
|---:|---:|---:|---:|
| `0.70` | `0.25` | `0.075` | `0.775` |
| `0.80` | `0.25` | `0.050` | `0.850` |
| `0.90` | `0.25` | `0.025` | `0.925` |
| `1.00` | `0.25` | `0.000` | `1.000` |

原权重越低，补回越多。恢复比例小于 `1` 时，入选 token 之间保留原权重排序。恢复比例取 `0` 时所有权重等于基线，取 `1` 时入选权重为 `1.0`。Ⅰ、Ⅲ的恢复增加倾向于降熵的更新力度，Ⅱ、Ⅳ的恢复增加倾向于增熵的更新力度，这里的方向沿用 STEER 的逐 token 分析。

## 3. 配置和启动脚本

在 `PolicyLossConfig` 和 actor Hydra 配置中加入以下字段。通用配置中象限列表默认为 `[]`，使已有运行配方保留原行为。V7 脚本显式设置 `[1, 2]`。

| 字段 | 通用默认值 | V7 脚本默认值 | 含义 |
|---|---|---|---|
| `steer_recovery_quadrants` | `[]` | `[1, 2]` | 允许恢复的象限编号，接受由整数 `1` 至 `4` 组成的无重复列表 |
| `steer_recovery_top_ratio` | `0.20` | `0.20` | 用于计算全体有效回答 token 排名阈值的比例，取有限值且大于零、小于或等于一 |
| `steer_recovery_min_entropy` | `0.5` | `0.5` | 条件熵的绝对门槛，取有限非负数，比较使用严格大于 |
| `steer_recovery_rate` | `0.25` | `0.25` | 原 STEER 扣减的恢复比例，取零到一之间的有限值，包含两端 |

V7 脚本从 V6 的 datav8 脚本派生，建议命名为 `run_steer_quadrant_v7_datav8.sh`。启用恢复时使用 `loss_mode=entropy_control`、`entropy_control_mode=symmetric` 和 `linear=false`。V7 脚本显式将 `q2_high_entropy_ratio` 和 `q2_high_entropy_boost_strength` 设为 `0.0`。配置校验对同时选择 V7 恢复象限和启用旧Ⅱ象限加权的组合报错，避免同一次损失调用叠加两套权重。原有 V5、V6 配方继续读取各自的字段。

启动脚本用 `STEER_RECOVERY_QUADRANTS` 接收列表，并把完整的 Hydra override 作为一个带引号的 shell 参数传入。三种配置使用不同实验名和输出目录，以便和同起点的对照并排查看。下列命令对应拟新增的脚本。

```bash
EXPERIMENT_NAME=steer_v7_q12 STEER_RECOVERY_QUADRANTS='[1,2]' bash run_steer_quadrant_v7_datav8.sh
EXPERIMENT_NAME=steer_v7_all STEER_RECOVERY_QUADRANTS='[1,2,3,4]' bash run_steer_quadrant_v7_datav8.sh
EXPERIMENT_NAME=steer_v7_control STEER_RECOVERY_QUADRANTS='[]' bash run_steer_quadrant_v7_datav8.sh
```

其中 `[2]` 可以单独启用Ⅱ象限。V7 脚本沿用 `steer_quadrant_logging=true`，恢复条件由象限配置决定，日志开关独立控制统计。[V6 脚本的配置传递方式](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/run_steer_quadrant_v6_datav8.sh:264)

## 4. 接入现有策略损失

将现有 `select_q2_high_entropy_tokens` 中计算排名阈值的部分提取为通用函数，返回全体有效位置的高熵布尔掩码。旧Ⅱ象限选择函数与这个掩码取交集，V7 再加上绝对门槛和所选象限。参考逻辑如下。

```python
import math

import torch


@torch.no_grad()
def select_high_entropy_tokens(entropys, response_mask, top_ratio):
    valid = response_mask.bool() & torch.isfinite(entropys)
    if top_ratio == 0.0:
        return torch.zeros_like(valid)

    valid_entropies = entropys[valid]
    if valid_entropies.numel() == 0:
        return torch.zeros_like(valid)

    top_count = math.ceil(valid_entropies.numel() * top_ratio)
    threshold = valid_entropies.topk(top_count, sorted=False).values.amin()
    return valid & (entropys >= threshold)


@torch.no_grad()
def apply_steer_recovery(steer_weights, entropys, response_mask, quadrants, policy):
    selected_quadrants = torch.zeros_like(response_mask, dtype=torch.bool)
    for quadrant in policy.steer_recovery_quadrants:
        selected_quadrants |= quadrants[quadrant - 1]

    high_entropy = select_high_entropy_tokens(
        entropys, response_mask, policy.steer_recovery_top_ratio
    )
    recovery_mask = selected_quadrants & high_entropy & (entropys > policy.steer_recovery_min_entropy)
    adjusted_weights = steer_weights + policy.steer_recovery_rate * (1.0 - steer_weights)
    weights = torch.where(recovery_mask, adjusted_weights, steer_weights)
    return weights, recovery_mask
```

`entropys` 是与回答位置对齐的缓存条件熵，`response_mask` 是有效回答掩码，`quadrants` 按Ⅰ至Ⅳ顺序存放布尔掩码，`policy` 是校验后的策略损失配置。`steer_weights` 是原始权重，`weights` 是恢复后实际乘入损失的权重。掩码保持与权重同形状。

在 `compute_policy_loss_with_entropy` 内，象限日志开启、V7 选择了象限或旧版Ⅱ象限加权开启时，复用 `compute_steer_v4_weights`。V7 的路径使用原权重和象限掩码，调用上面的覆盖函数一次。空象限列表的 V7 对照走原权重路径，同时可收集四象限指标。V7 的权重恢复只读取原权重张量，额外的分数幅度返回值继续由旧 V6 平方曲线路径按需请求。

最终沿用 `agg_steer_loss(losses * weights, response_mask, loss_agg_mode)`。`losses` 是原 PPO 裁剪和 dual-clip 后的逐 token 损失，`loss_agg_mode` 是现有的损失汇总方式。梯度经过原策略损失，权重和掩码在 `torch.no_grad()` 下计算。处于 PPO 裁剪零梯度区的位置乘上恢复权重后仍是零梯度。恢复使用本地 microbatch，跨卡通信仅用于现有训练同步和日志汇总。[原策略损失与指标接入](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/core_algos.py:1565)

## 5. TensorBoard 指标

**保留全部原有训练指标和现有 STEER 指标，新增项追加到同一个 metrics 字典。** 四象限权重均值继续读取实际乘入损失的最终权重。原 V5/V6 配方保留各自的统计。V7 用Ⅱ象限恢复掩码计算现有Ⅱ象限放行指标。`steer/q2_release_boost_mean` 沿用入选活跃位置最终权重减 `1.0` 的均值，V7 下该值表示相对完整更新力度的差，范围小于或等于零。相对原 STEER 补回多少由下表的增量指标记录。

| 新增指标 | 口径 |
|---|---|
| `steer/qN_recovery_frac` | 该象限通过双阈值并符合象限配置的位置数，除以该象限全部有效 token 数 |
| `steer/qN_recovery_active_frac` | 该象限入选且活跃的位置数，除以该象限全部活跃 token 数 |
| `steer/qN_recovery_delta_mean` | 该象限入选且活跃位置的最终权重与原 STEER 权重之差的平均值 |

`qN` 对应 `q1` 至 `q4`。活跃位置沿用 V6 的判定，即裁剪后的原 PPO 策略项对当前采样 token 的对数概率具有有限、非零的导数。象限配置未选中时，该象限的恢复计数为零。各项分母为零时跳过对应比例或均值。

复用 `SteerV4Metrics` 的统计和输出路径，共用原来的数据并行 `all_reduce(SUM)`。把每个象限的恢复计数、活跃恢复计数和权重增量和拼入现有统计向量，向量长度在同一次运行中固定，某个 microbatch 全未入选时填零。**沿用 V6 的 HCCL 修复，权重求和、日志统计打包和跨卡归约统一使用 FP32**。跨卡求和后再计算比例和均值。[现有统计及归约实现](/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/steer_metrics.py:79)

## 6. 实现验收

- **阈值。** 用固定输入分别覆盖排名阈值低于和高于 `0.5` 的情况，核对双阈值边界、同熵、padding、非有限熵和有效位置为空的处理。用同一份输入比较 `[1, 2]` 与 `[1, 2, 3, 4]`，两组使用相同的排名阈值。
- **权重。** 对照第二节的算例。固定输入下，`[1, 2, 3, 4]` 和 `[1, 2]` 对Ⅰ、Ⅱ给出逐项相同的权重，前者额外恢复满足双阈值的Ⅲ、Ⅳ位置。把提供原 STEER 最大绝对分数的位置设为入选项，核对其余位置继续使用同一原分母。
- **梯度与兼容。** 空列表或恢复比例为零时，对照原 STEER 的权重、损失和对数概率梯度。启用恢复时，核对活跃位置按最终权重缩放，裁剪后的零梯度位置仍为零。复跑原 V5/V6 的选择及权重用例。
- **脚本与日志。** 核对 Bash 语法和 Hydra 解析出的列表值。读取 TensorBoard event，确认原有训练指标、STEER 指标和按象限追加的指标共同写入。在 NPU 上核对一次实际 actor 更新与 HCCL 日志归约。

对照运行使用相同模型起点、数据、随机种子、rollout、microbatch、损失汇总和原 STEER 参数。`[]` 与 `[1, 2]` 比较Ⅰ、Ⅱ高熵恢复的效果，`[1, 2]` 与 `[1, 2, 3, 4]` 比较额外加入Ⅲ、Ⅳ的效果。
