# STEER V4 四象限等价对照方案

## 1. 实验目标与版本定义

V4 在原始 STEER 的权重作用于策略损失之前，按优势正负和概率分界划分四个象限。四个象限均使用原始 STEER 算出的逐 token 权重，完整复用其分数计算、归一化、指数映射和策略损失接入方式。

本方案的对照基线采用此前使用的对称指数版 STEER，配置为 `entropy_control_mode: symmetric`、`linear: false`。在相同输入、配置和数值精度下，V4 与该基线的逐 token 权重、策略损失及梯度逐项一致。

实验用于检查四象限划分及其接入过程是否影响了训练计算。V4 的四象限统计可以直接用于观察原始 STEER 在各类 token 上的处理情况。

### TensorBoard 的持续要求

**V4 及后续所有版本均记录训练框架原有指标和相应版本的 STEER 指标。**

训练框架生成的 loss、KL、entropy、clip fraction、学习率、梯度范数、奖励、吞吐和耗时等指标，连同原有其他指标，一起进入现有日志流程。新增的四象限指标使用 `steer/` 前缀，追加到同一个 metrics 字典，再交给现有 logger 写入 TensorBoard。

## 2. 输入与象限划分

### 2.1 输入张量

| 输入 | 含义 |
|---|---|
| `advantages` | 原训练流程计算并分配到回答 token 的 GRPO 优势 |
| `entropys` | 原 STEER 权重函数实际读取的逐位置完整词表 Shannon 熵 |
| `old_log_prob` | 原 STEER 使用的旧策略对已采样 token 的对数概率 |
| `log_prob` | 当前 actor 对同一已采样 token 的对数概率 |
| `response_mask` | 有效回答 token 的掩码，回答位置为 `1`，其余位置为 `0` |

熵张量沿用现有 STEER 的来源。当前适配实现传入的是缓存的旧策略熵，V4 分类和权重计算读取这一份相同的张量。

### 2.2 概率分界

对有效回答位置计算

$$
b_t=\log p_t+H_t.
$$

- $t$ 表示当前 microbatch 中的一个回答 token 位置。
- $p_t$ 是当前 actor 对该位置已采样 token 的概率，沿用原 STEER 的概率截断结果。
- $H_t$ 是该位置传入原 STEER 的熵。
- $\log$ 是自然对数。
- $b_t$ 是用于比较概率和熵分界的数值。

$b_t>0$ 对应 $p_t>e^{-H_t}$，$b_t<0$ 对应 $p_t<e^{-H_t}$。这里 $e$ 是自然指数的底数，$e^{-H_t}$ 是 STEER 四象限分析中的概率分界。

实现中直接比较 `probability.log() + entropys` 与零。`probability` 使用与原权重函数相同的 `exp`、`clamp` 和输入精度。

### 2.3 四象限及权重

| 象限 | 优势条件 | 概率条件 | V4 权重 |
|---|---|---|---|
| Ⅰ | $A_t>0$ | $b_t>0$ | 该 token 的原始 STEER 权重 |
| Ⅱ | $A_t>0$ | $b_t<0$ | 该 token 的原始 STEER 权重 |
| Ⅲ | $A_t<0$ | $b_t<0$ | 该 token 的原始 STEER 权重 |
| Ⅳ | $A_t<0$ | $b_t>0$ | 该 token 的原始 STEER 权重 |

$A_t$ 是该 token 的原 GRPO 优势。四个象限掩码互斥。

优势为零、恰好处于概率分界，以及分类所需数值经过有限性检查后落在四象限以外的位置，进入 `other_mask`。这些位置同样使用原始 STEER 返回的权重。有限输入下，优势为零或 $b_t=0$ 的位置，其原始 STEER 分数为零，权重为 `1`。padding 权重由原函数处理为 `0`。

## 3. 原始 STEER 的分数与权重

### 3.1 逐 token 分数

逐 token 分数沿用 V3 的记号和符号约定。

$$
x=-\frac{A}{p_{\mathrm{old}}}\,p(1-p)(\log p+H).
$$

- $x$ 是当前 token 归一化前的带符号 STEER 分数。正值对应增熵方向，负值对应降熵方向。
- $A$ 是该 token 的原 GRPO 优势。
- $p$ 和 $p_{\mathrm{old}}$ 分别是当前 actor 和旧策略对同一前缀下同一已采样 token 的概率，$1-p$ 是当前 actor 对其余 token 的总概率。
- $H$ 是原权重函数使用的熵，$\log$ 是自然对数，$\log p+H$ 是第 2 节的概率分界量，省略位置下标。

分数的绝对值 $|x|$ 对应公开对称版代码取绝对值后的 `metric`。论文中的状态级期望量 $\Omega(s)$ 使用论文自身的定义，$s$ 表示提示词和已生成前缀。[公开 STEER 分数计算](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L619-L660)

原函数对当前概率使用 `[1e-8, 1.0-1e-8]` 的截断，对旧概率使用 `[1e-8, 1.0]` 的截断。指数、乘除、对数、有限性处理和类型转换均由该函数执行。

### 3.2 归一化与指数映射

$$
D=\max\!\left(0.02,\max_{j\in\mathcal B}|x_j|\right),
\qquad
\lambda(x)=w_{\min}^{\,|x|/D}.
$$

- $\mathcal B$ 是当前本地 actor microbatch 内全部有效回答 token 的集合。
- $j$ 是该集合中一个 token 的位置，$x_j$ 是它的原始分数，$|x_j|$ 是分数绝对值。
- $\max_{j\in\mathcal B}|x_j|$ 是该集合的最大绝对分数。
- $D$ 是原 STEER 的归一化分母，固定下限为 `0.02`。
- $w_{\min}$ 是原配置 `token_weight_min`，取值大于 `0` 且小于或等于 `1`。
- $|x|/D$ 是当前 token 的分数绝对值除以共享分母。
- $\lambda(x)$ 是根据该 token 的分数 $x$ 得到、最终乘入策略损失的权重。

分母和权重的数学写法与 [V3 文档](STEER_QUADRANT_V3.md#2-计算原-steer-分数和四象限权重) 的Ⅰ、Ⅲ相同，V4 对四个象限统一使用这个权重。当前 microbatch 的全部有效 token 共用一个分母。实现以现有 `compute_token_weights` 的返回值为准，直接沿用原函数的 `exp` 运算顺序和 `clamp` 截断。[公开 STEER 对称指数映射](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L698-L710)

当前设置每卡 `microbatch=1` 时，每次分母取当前卡这一条回答内的最大绝对分数与 `0.02` 中的较大值。16 张数据并行卡分别执行各自的权重计算，日志阶段再汇总四象限统计。

### 3.3 数值算例

假设一个本地 microbatch 内，四个象限各有一个有效 token，已经算出的绝对分数如下，且 `token_weight_min=0.70`。

| 象限 | 绝对分数 | 共享分母 | 原始 STEER 权重 | V4 权重 |
|---|---:|---:|---:|---:|
| Ⅰ | `0.005` | `0.040` | 约 `0.9564` | 约 `0.9564` |
| Ⅱ | `0.010` | `0.040` | 约 `0.9147` | 约 `0.9147` |
| Ⅲ | `0.020` | `0.040` | 约 `0.8367` | 约 `0.8367` |
| Ⅳ | `0.040` | `0.040` | `0.7000` | `0.7000` |

四个位置的差异来自各自的原始分数。象限标签用于统计该位置属于哪一类。

## 4. 实现结构

### 4.1 配置入口

建议在 `PolicyLossConfig` 增加布尔开关 `steer_quadrant_logging`，默认值为 `false`。V4 开启该开关，并继续走原始 `symmetric` 权重路径。

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: entropy_control
      entropy_control_mode: symmetric
      linear: false
      steer_quadrant_logging: true

trainer:
  logger: [console, tensorboard]
```

`token_weight_min`、`token_weight_max` 读取原始 STEER 对照运行的实际值。`steer_quadrant_logging` 控制象限分类及新增统计的接入，四个象限的权重由原函数统一返回。

对照配置如下。

| 配置 | 原始 STEER 对照 | V4 |
|---|---|---|
| `loss_mode` | `entropy_control` | `entropy_control` |
| `entropy_control_mode` | `symmetric` | `symmetric` |
| `linear` | `false` | `false` |
| `token_weight_min` | 原训练配方的值 | 与对照相同 |
| `token_weight_max` | 原训练配方的值 | 与对照相同 |
| `steer_quadrant_logging` | `false` | `true` |

### 4.2 参考封装

下面的函数放在现有 `compute_token_weights` 可见的作用域内。它先生成象限掩码，再调用一次原始 STEER 权重函数，返回权重和日志所需的掩码。

```python
import torch


@torch.no_grad()
def compute_steer_v4_weights(
    advantages,
    entropys,
    old_log_prob,
    log_prob,
    response_mask,
    *,
    token_weight_min,
    token_weight_max,
):
    valid = response_mask.bool()
    probability = log_prob.exp().clamp(min=1e-8, max=1.0 - 1e-8)
    boundary = probability.log() + entropys
    classifiable = (
        valid
        & torch.isfinite(advantages)
        & torch.isfinite(boundary)
    )

    positive_advantage = advantages > 0
    negative_advantage = advantages < 0
    high_probability = boundary > 0
    low_probability = boundary < 0

    q1 = classifiable & positive_advantage & high_probability
    q2 = classifiable & positive_advantage & low_probability
    q3 = classifiable & negative_advantage & low_probability
    q4 = classifiable & negative_advantage & high_probability
    quadrants = (q1, q2, q3, q4)
    other_mask = valid & ~(q1 | q2 | q3 | q4)

    weights = compute_token_weights(
        advantages=advantages,
        entropys=entropys,
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        response_mask=response_mask,
        token_weight_min=token_weight_min,
        token_weight_max=token_weight_max,
        linear=False,
        mode="symmetric",
    )
    return weights, quadrants, other_mask
```

`weights` 是原始函数的直接返回值。`quadrants` 依次保存Ⅰ、Ⅱ、Ⅲ、Ⅳ的布尔掩码，`other_mask` 保存其余有效位置。权重在 `torch.no_grad()` 下计算，原策略损失承担反向传播。

### 4.3 接入策略损失

在现有 `compute_policy_loss_with_entropy` 中按以下顺序接入。

1. 从 actor 接收原有的优势、熵、旧对数概率、当前对数概率和回答掩码。
2. V4 开关开启时调用上述封装，取得四象限掩码和原始权重。对照路径直接调用原始 `compute_token_weights`。
3. 沿用现有 PPO 概率比、概率比截断、正负优势分支及负优势 dual-clip 分支，得到逐 token 策略损失。
4. 用原始权重乘逐 token 策略损失，再调用现有 `agg_steer_loss`。
5. 读取本次实际使用的权重、象限掩码和裁剪中间量，累加新增 STEER 指标。
6. 继续执行现有梯度累积、梯度同步、优化器更新及日志流程。

权重乘入处的代码为

```python
weighted_losses = losses * weights
loss = agg_steer_loss(weighted_losses, response_mask, loss_agg_mode)
```

`losses` 是原 PPO 裁剪和 dual-clip 处理后的逐 token 策略损失，`weights` 是完整本地 microbatch 的原 STEER 权重，`loss_agg_mode` 是原训练配置的损失汇总方式。最终的 `loss` 接到现有 actor 反向传播路径。

原始分数和权重按原函数在 PPO 裁剪前的路径计算，裁剪后的逐 token 损失再乘上权重。分类、原始权重计算和日志均读取同一次训练调用的张量。

## 5. 追加的 TensorBoard 指标

### 5.1 统计范围

**有效 token** 指 `response_mask` 为 `1` 的位置。

**活跃 token** 指加权前的原 PPO 策略损失对当前已采样 token 的对数概率具有非零导数的有效位置。这个判定包含现有概率比裁剪、负优势 dual-clip，以及对数概率差截断的梯度行为。裁剪边界和两个分支相等处，采用现有 PyTorch 算子的实际导数。

活跃判定可以复用已有 `steer_metrics.py` 中的原 PPO 梯度系数计算部分，再把统计输出扩展到四个象限。用于判定的导数系数在权重乘入之前计算。

### 5.2 指标定义

表中的 `qN` 分别替换为 `q1`、`q2`、`q3`、`q4`。

| TensorBoard 名称 | 统计定义 | 用途 |
|---|---|---|
| `steer/qN_frac` | 该象限有效 token 数除以全部有效 token 数 | 观察四象限的 token 分布 |
| `steer/qN_active_frac` | 该象限活跃 token 数除以全部活跃 token 数 | 观察原 PPO 实际更新位置的象限分布 |
| `steer/qN_weight_mean` | 该象限全部有效 token 的原 STEER 权重之和除以该象限有效 token 数 | 观察各象限的平均权重 |
| `steer/qN_inactive_frac` | 该象限有效 token 中原 PPO 策略项导数为零的位置数，除以该象限有效 token 数 | 观察该象限落入零导数位置的比例 |
| `steer/other_frac` | `other_mask` 中有效 token 数除以全部有效 token 数 | 补全四象限之外的位置统计 |
| `steer/other_active_frac` | `other_mask` 中活跃 token 数除以全部活跃 token 数 | 补全活跃位置统计 |
| `steer/weight_mean` | 全部有效 token 的原 STEER 权重均值 | 观察整体权重水平 |
| `steer/valid_count` | 日志窗口内全部有效 token 数 | 给出有效位置统计规模 |
| `steer/active_count` | 日志窗口内全部活跃 token 数 | 给出活跃位置统计规模 |

某个象限的平均权重覆盖该象限全部有效 token，与原分母覆盖的有效位置范围相配合。活跃比例则反映原 PPO 策略项实际产生梯度的位置。

当全部有效 token 数大于零时，四个 `qN_frac` 加上 `other_frac` 的和为 `1`。当全部活跃 token 数大于零时，四个 `qN_active_frac` 加上 `other_active_frac` 的和为 `1`。

某项分母为零时，该次日志跳过该项比例或均值，计数指标记录实际的零值。`other_mask` 覆盖分类所需数值未通过有限性检查的位置时，活跃比例按原 PPO 导数的有限值判定进行统计。

### 5.3 累加与跨卡汇总

1. 每个 microbatch 在 GPU 上累加有效计数、活跃计数和权重总和，累加张量使用 `float64`，便于覆盖较长日志窗口。
2. 在现有训练日志步，将这些分子、分母打包成一个张量，在 actor 的数据并行组内执行一次 `all_reduce(SUM)`。TP、SP 等模型并行配置下，按实际数据分片选择参与统计的 rank，使每份训练数据贡献一次计数。
3. 汇总后批量转到 CPU，以全局总和计算比例和均值。
4. 将结果追加到现有 metrics 字典，继续交给现有日志流程。写入后清零累加器。

跨 microbatch、跨卡的均值通过“先汇总分子分母，再相除”计算。现有训练窗口中的每次策略损失调用都贡献本次计数，重复 PPO epoch 对应各自实际发生的训练调用。

这一过程复用训练已有的张量和裁剪中间量。新增工作为布尔掩码、标量累加和日志步的一次数据并行归约。

### 5.4 与现有日志合并

实现时在已有汇总位置追加指标。

```python
metrics.update(steer_metrics)
logger.log(data=metrics, step=global_step)
```

`metrics` 是原训练流程已经生成的完整指标字典，`steer_metrics` 是新增 STEER 统计字典，`global_step` 是现有训练步数。新增指标名称与现有名称分别核对，日志合并结果同时包含两者。

TensorBoard adapter 遍历传入的完整 metrics 字典，逐项调用 `add_scalar`。训练框架原有的 STEER 权重均值、最小值和最大值等指标也按原流程写入。

## 6. 原始 STEER 与 V4 的对照运行

### 6.1 对照条件

| 项目 | 两组配置 |
|---|---|
| 初始模型与恢复状态 | 使用相同模型权重、优化器状态及恢复步数 |
| 数据 | 使用相同训练集、数据顺序和随机种子 |
| rollout | 使用相同采样参数、每题回答数及回答长度 |
| actor | 使用相同学习率、PPO epoch、裁剪参数和正则系数 |
| 优势 | 使用相同 GRPO 优势计算和处理方式 |
| 权重输入 | 使用相同熵缓存、旧对数概率和当前对数概率来源 |
| 损失汇总 | 使用相同 `loss_agg_mode` 和梯度累积缩放 |
| 并行和精度 | 使用相同数据并行规模、microbatch、动态批处理配置、精度和梯度同步设置 |
| 日志 | 两组均写入完整原有指标，V4 追加四象限统计 |

分别使用独立的实验名和 TensorBoard 目录，便于在同一界面叠加曲线。

### 6.2 固定输入核对

固定同一份输入张量，分别调用原始 STEER 和 V4。

| 核对对象 | 通过标准 |
|---|---|
| 象限覆盖 | 四个掩码互斥，与 `other_mask` 合起来覆盖全部有效位置 |
| 逐 token 权重 | 相同 dtype、相同形状，各位置与原始 STEER 完全相等 |
| 策略损失 | 使用同一汇总路径后，标量与原始 STEER 完全相等 |
| 对数概率梯度 | 对两份相同的叶子 `log_prob` 分别反向传播，梯度完全相等 |
| 一次优化器更新 | 恢复同一初始状态和随机状态，在同一执行环境的确定性设置下，更新后的参数一致 |

权重、损失和对数概率梯度检查在同一进程内进行。两条路径调用同一个权重函数和同一套损失算子，采用逐项精确相等检查。

输入用例覆盖四象限、零优势、概率分界、padding、单个有效 token，以及最大绝对分数处于 `0.02` 下方、等于 `0.02` 和高于 `0.02` 的情况。裁剪用例覆盖正优势上界、负优势下界、dual-clip 和各分支相等处。全部为 padding 的权重调用，与原函数的全零返回值对照。

优化器更新检查同时恢复模型状态、优化器状态和随机数状态，并使用固定的同一批回答。这样可以逐层检查从权重到最终参数更新的接入过程。

### 6.3 TensorBoard 核对

1. 在一次日志写入中同时提供已有训练指标和新增 `steer/` 指标，读取 event 文件确认两类 tag 及其值均已写入。
2. 对真实训练的首个日志步，核对原有 loss、KL、entropy、奖励、学习率等实际生成的 tag，并核对新增的四象限 tag。
3. 使用固定统计样例，核对四象限比例、剩余位置比例和各象限平均权重。
4. 将同一统计样例拆到多个 microbatch、多个数据并行 rank，再与集中计算的分子分母进行核对。
5. 核对某个象限为空、全部有效优势为零，以及日志累加器清零后的计数行为。

长程训练沿用用户现有的训练效果和测试效果评价流程。固定输入检查用于验证计算等价，原有训练曲线用于比较两组实际运行过程。

## 7. 实现交付项

- 为 `PolicyLossConfig` 增加 `steer_quadrant_logging` 布尔开关，并接到实际 actor 策略损失入口。
- 增加四象限分类封装，完整调用现有 `compute_token_weights`，返回其原始权重及分类掩码。
- 在权重乘入策略损失前生成象限掩码，随后沿用完整张量上的原加权和汇总路径。
- 在现有 STEER 统计模块中增加四象限计数、权重总和及活跃统计，通过数据并行日志汇总后追加到 metrics。
- 为原始 STEER 和 V4 分别提供对应配置，记录两组运行使用的代码提交、模型起点和完整训练配置。
- 完成权重、损失、梯度、一次参数更新和 TensorBoard tag 的核对，形成可复查的结果。

## 8. 实现参考位置

- 原始公开实现的分数、权重和策略损失接入位于 `STEER-main/verl/trainer/ppo/core_algos.py`。[公开源码](https://github.com/zz-haooo/STEER/blob/56576b78727f09e7ce233fdbb7d6135358934333/verl/trainer/ppo/core_algos.py#L580-L847)
- 当前适配版的权重函数和策略损失入口位于 `/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/core_algos.py`。
- actor 策略损失配置位于 `/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/workers/config/actor.py`。
- v3 分支中的统计模块位于 `/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/trainer/ppo/steer_metrics.py`，可复用其原 PPO 活跃判定和日志累加结构。
- TensorBoard 写入入口位于 `/Users/lhx/VSCodeFiles/codex_work/verl090_steer/verl-0.9.0/verl/utils/tracking.py`。
