# STEER 算法、旧实现与 verl 0.9.0 适配笔记

> 记录日期：2026-11-04。当前阶段：STEER 算法核心已适配到 verl 0.9.0，CPU 验证通过；实际 NPU 训练尚未验证。
> 目的：后续会话先读本文，再沿文中的函数定位继续工作，避免把论文定义、旧实现行为和适配建议混在一起。
> 第 2–9 节保留初始调研与历史建议；当前实现和验证状态见第 11 节。用户最新决定：**有差异的地方以旧版为准**；该决定覆盖历史建议中的 FP32 强转、全局 loss 归一化、当前熵正则和 rollout IS 扩展。仅适配算法，不处理实验配置、启动脚本或奖励函数；后续训练目标为 NPU + Qwen3.5-9B。

## 1. 结论与后续工作入口

STEER 是在现有策略梯度算法上增加 **token 级、停止梯度的自适应损失权重**。先根据优势、概率和条件熵计算熵变化代理量，再减小熵变化幅度较大 token 的更新贡献。主实验使用绝对值和指数映射，同时抑制过大的熵增与熵减。GRPO 的采样、奖励和组内优势计算本身不需要重写。

移植到当前目录中的 verl 0.9.0，沿 **默认 V1 同步 trainer + unified model engine** 接入，核心改动位于：

1. `verl/workers/config/actor.py` 和 `verl/trainer/config/actor/actor.yaml`：增加有类型的 STEER 配置。
2. `verl/trainer/ppo/core_algos.py`：权重函数与注册的 `entropy_control` policy loss。
3. `verl/workers/utils/losses.py::ppo_loss`：提供缓存旧熵，按旧版局部聚合及梯度累积规则缩放 STEER loss；不叠加 rollout correction。
4. 按熵来源处理 trainer 数据路径：V1 已存储旧熵 `entropy`；V0 兼容路径会删除 `entropys`，需要有条件保留。
5. 增加算法数值和接口验证。训练 recipe、奖励适配入口按用户最新指示不纳入本次改动。

**推荐顺序：先迁移旧实现的有效核心并建立数值对照，再单独评估更接近论文公式的修改。** 特别是旧熵改为当前熵、加入 clipping 指示器、改变最大值归一化范围，都会改变结果，不能作为“仅 API 迁移”悄悄混入。

## 2. 材料与版本边界

| 材料 | 本地位置 | 核实结果 |
| --- | --- | --- |
| 论文 | [STEER.pdf](STEER.pdf) | 28 页；arXiv:2510.10150v4，2026-04-29；题名 *Rethinking Entropy Interventions in RLVR: An Entropy Change Perspective* |
| 作者旧代码 | [STEER-main](STEER-main) | `verl/version/version` 为 `0.4.1.dev`；README 安装指向 `v0.4.1.x` |
| 目标代码 | [verl-0.9.0](verl-0.9.0) | `verl/version/version` 为 `0.9.0`；本文以这个本地快照为准 |
| 仓库规则 | [verl-0.9.0/AGENTS.md](verl-0.9.0/AGENTS.md) | 已阅读；后续 Python 环境管理按其要求使用 uv；本任务没有发起上游 PR |

论文重点：§2–5、§6、附录 C、D、E、F、G。对 PDF 第 3、5、6 页进行了渲染检查，以核实熵公式、clipping 定义和指数映射。附录 G 是理论假设与推导；附录 E 是训练/评估条件；附录 F 是线性/二值映射消融。本文不把论文的全部实验复现视作第一阶段适配范围。

源码行号是阅读快照的导航点，后续编辑后可能移动，**以函数名为准**。两个目录都包含顶层 `verl` 包，运行时必须确认导入的是目标目录，不能让旧脚本的 `PYTHONPATH` 抢先导入旧版。

## 3. 算法细节：从 GRPO 到 STEER

### 3.1 基本符号与 GRPO

- 输入问题为 `q`，同一问题采样 `G` 条回答 `o_i`。
- 第 `t` 个 token 的状态 `s=(q,o_i,<t)`，动作为实际采样 token `a=o_i,t`。
- `p=πθ(a|s)`：当前 actor 在实际 token 上的概率；`p_old=π_old(a|s)`：更新前固定策略的概率。
- `r=p/p_old`：PPO importance ratio；它不是最大概率 token 的概率比。
- `Hθ(s)=-Σ_v πθ(v|s) log πθ(v|s)`：完整词表分布的条件熵。
- `M_i,t`：回答有效位置的 mask；padding 和不参与训练的位置不应进入权重归一化。
- `L=Σ_i |o_i|`：论文中的 token 数归一化；不要与回答最大长度混淆。

论文式 (1) 的组内优势为：

```text
A_i = (R_i - mean_group(R)) / std_group(R)
A_i,t = A_i                  # 广播到回答的有效 token
```

旧版与新版默认 GRPO 都把 token reward 沿回答求和，然后按同一 prompt 的 `uid` 分组，计算均值和标准差，最后乘 response mask。实现带 `epsilon=1e-6`，可通过 `norm_adv_by_std_in_grpo` 关闭标准差归一化。同组全对或全错时通常没有组内相对优势。不要为了 STEER 去修改这部分逻辑。

论文式 (2) 的 clipped surrogate（最大化形式）为：

```text
J = E[ Σ_i,t min(r_i,t A_i,t,
                 clip(r_i,t, 1-ε_low, 1+ε_high) A_i,t) / L ]
```

源码以最小化负 loss 实现，并额外使用 dual-clip PPO，详见第 5 节。

### 3.2 熵变化的一阶近似

论文式 (6)：

```text
I_clip(s,a) = 0   if A>0 and r>1+ε_high
            = 0   if A<0 and r<1-ε_low
            = 1   otherwise
```

论文式 (7) / 附录式 (15)：

```text
Ω(s) = -(η/L) E_{a~πθ(.|s)} [
           (I_clip(s,a) A(s,a) / π_old(a|s))
           · p(1-p) · (log p + Hθ(s))
       ]
ΔH(s) ≈ Ω(s), remainder O(η²)
```

论文式 (8) 把概率与熵的联合影响记为：

```text
δ(p,H) = -p(1-p)(log p + H)
```

四类因素为 clipping、优势及旧概率、当前概率、当前条件熵。必须区分 **状态上对动作的期望 Ω(s)** 与旧代码对实际采样 token 计算的 **代理量**；源码没有逐状态枚举所有动作的优势，也没有精确计算上述期望。

理论边界：附录 G 假设独立 logit，并忽略未采样动作 logit 的更新；LLM 共享参数、Adam、梯度裁剪及不同 token 的耦合不满足简单的逐 logit 更新模型。因此这里是指导重加权的近似，不能解释成实际每步熵变化的硬约束或严格保证。

### 3.3 熵变化方向的四象限

`log p + H=0` 对应 `p=exp(-H)`。所谓高/低概率是相对于当前条件熵的阈值，不能固定写成 0.5。

| 优势 | 相对概率 | δ 符号 | 预测方向 | 论文解释 |
| --- | --- | --- | --- | --- |
| `A>0` | `p>exp(-H)` | 负 | 熵减 | I：强化已高概率的正确行为，exploitation |
| `A>0` | `p<exp(-H)` | 正 | 熵增 | II：强化罕见正确行为，exploration |
| `A<0` | `p<exp(-H)` | 正 | 熵减 | III：压低低概率错误行为，suppression |
| `A<0` | `p>exp(-H)` | 负 | 熵增 | IV：纠正高概率错误，error correction |

这也解释了为什么“高熵 token 一律给更大优势”可能放大已有熵下降，以及为什么只降低正样本权重没有覆盖全部熵减来源。

### 3.4 主方法：停止梯度的指数权重

论文式 (9)：

```text
m(s) = |Ω(s)|
α = -log(λ_min)
λ(s) = exp(-α · m(s) / max_{s'∈B} m(s'))
```

有效权重在 `[λ_min,1]`；变化幅度越大，权重越低。使用绝对值意味着熵增和熵减都可能被削弱。论文主实验 `λ_min=0.7`；不是直接最大化熵，也不设定目标熵值。

权重作为常数乘入每个 token 的策略 loss：

```text
loss = aggregate(M · stop_gradient(λ) · token_policy_loss)
```

停止梯度很关键：若让梯度穿过概率、熵和归一化极值，会多出代理量的导数，改变“调整原策略梯度大小”的方法。

聚合的分母仍是原 mask 下的有效 token 数，**不是 `Σ Mλ`**；权重不应重新归一化为均值 1，否则会抵消整体衰减并改变算法。

### 3.5 消融和扩展

- 附录 F.1：线性映射将变化量映射到 `[λ_min,λ_max]`，论文取 `λ_min=0.7, λ_max=1.2`。因此线性分支可以放大部分 token。
- 二值映射：最高 20% 的变化量 token 用 `λ_min`，其余用 1；论文表现较差，旧核心没有实现这一分支，第一阶段不必新增。
- 旧仓库额外提供 `asymmetric`；它是方向性扩展，不是论文主实验默认。线性与指数分支的实际行为还不同，见第 4.3 节。
- 论文报告对 RLOO/OPO 也有效，但第一阶段应先跑通同步、单轮数学任务的 GRPO，不能仅凭一个共用 loss 就宣称已验证全部算法/后端。

## 4. 旧版代码：权重到底怎样算

### 4.1 主要源码定位

| 功能 | 源码与阅读行号 |
| --- | --- |
| 策略 loss 注册 | [core_algos.py](STEER-main/verl/trainer/ppo/core_algos.py)，`register_policy_loss/get_policy_loss_fn`，52 起 |
| GRPO advantage | 同文件 `compute_grpo_outcome_advantage`，220–273 |
| 权重 | 同文件 `compute_token_weights`，580–713 |
| STEER loss | 同文件 `compute_policy_loss_with_entropy`，717–851 |
| 普通 PPO 对照 | 同文件 `compute_policy_loss`，854–937 |
| 熵与 logprob 前向 | [dp_actor.py](STEER-main/verl/workers/actor/dp_actor.py)，`_forward_micro_batch`，82 起 |
| actor 更新及 STEER 调用 | 同文件 `update_policy`，399 起；510–551 是熵来源和 loss 分支 |
| 旧熵缓存到训练 batch | [ray_trainer.py](STEER-main/verl/trainer/ppo/ray_trainer.py)，1197–1220 |
| worker 返回旧熵 | [fsdp_workers.py](STEER-main/verl/workers/fsdp_workers.py)，`compute_log_prob`，703 起 |
| recipe | [run_exp.sh](STEER-main/run/run_exp.sh)、[run_linear.sh](STEER-main/run/run_linear.sh)、[run_asymmetric.sh](STEER-main/run/run_asymmetric.sh) |

### 4.2 `compute_token_weights` 的精确行为

输入 `advantages/entropys/old_log_prob/log_prob/response_mask` 都是 `[B,T_response]`。

整个函数在 `torch.no_grad()` 中：

```text
p = clamp(exp(log_prob), 1e-8, 1-1e-8)
p_old = clamp(exp(old_log_prob), 1e-8, 1)
f = p(1-p)(log(p) + entropys)
u = (advantages / p_old) · f

symmetric:  metric = abs(u)
asymmetric: metric = -u          # Ω>0 表示熵增，Ω<0 表示熵减
```

注意代码变量 `x_one_minus_x_squared` 名称有误导，实际是 `p*(1-p)`，**没有额外平方**；注释里的 `A / old_log_prob` 实际除的是 `p_old`，不是 logprob。

后续处理：

1. 非有限 metric 替换成 0，并打印警告。
2. 仅在 `response_mask.bool()` 为真的位置计算最小/最大值。
3. 没有有效 token 时，权重函数直接返回全零。
4. 最终 padding 位置权重置零。

对称线性分支，令 `m=abs(u)`：

```text
λ = λ_max - (m-m_min)(λ_max-λ_min)/(m_max-m_min)
```

如果 `m_max==m_min`，全部有效位置用 `(λ_min+λ_max)/2`，再截断到给定范围。即使全部优势为零，也不是返回全 1；不过此时策略梯度本身为零。

对称指数分支：

```text
D = max(m_max, 0.02)
k = -log(λ_min)/D
λ = clamp(exp(-k·m), λ_min, 1)
```

该分支没有使用 `λ_max`。当 `m_max<0.02` 时，实际最小权重大于 `λ_min`，所以不保证每次都有 token 恰好取下界。这里的 `0.02` 是旧实现额外的数值/尺度选择，论文式 (9) 没有这个固定阈值。

### 4.3 非对称分支：不能只看 README 描述

非对称指数映射：

```text
signed_metric = -u
D = max(max(abs(signed_metric)), 0.02)
k = -log(λ_min)/D
λ = clamp(exp(k·signed_metric), λ_min, λ_max)
```

若 `λ_max=1`，预测熵增 token 保持 1，预测熵减 token 被衰减；若上界大于 1，也可能放大熵增 token。

非对称**线性**映射则是：

```text
λ = λ_min + (signed_metric-m_min)(λ_max-λ_min)/(m_max-m_min)
```

它并不保证所有熵增 token 权重都为 1，即使 `λ_max=1` 也是如此：只有最大 signed metric 取上界；Ω=0 也不一定映射为 1。因此不能把 README 的“另一方向保持 1”概括套到所有分支。

`run_asymmetric.sh` 实际采用线性、`0.9–1.0`、clip high/low 均 `0.2`；README 推荐的是另一组 `0.7–1.0`。旧函数没有校验 mode 的合法值，错误字符串可能落入不预期分支；迁移时应显式校验。

## 5. 旧版损失、数据流与附带改动

### 5.1 加权 dual-clip PPO

`compute_policy_loss_with_entropy` 先计算上述权重，再复用普通 PPO 的 token loss：

```text
d = clamp(log_prob-old_log_prob, -20, 20)
r = exp(d)
l1 = -A*r
l2 = -A*clip(r, 1-ε_low, 1+ε_high)
l_clip = max(l1,l2)
l_dual = min(-A*c, l_clip)       # c=clip_ratio_c > 1
l = where(A<0, l_dual, l_clip)
pg_loss = agg_loss(l * λ, response_mask, loss_agg_mode)
```

返回 9 项：loss、clipfrac、approx KL、dual-clip fraction、clip high fraction、clip low fraction、两个 clipping mask，以及含权重均值/最小/最大值的 stats dict。

PPO 的 clipping 分支会阻断相应 token 的梯度；**权重代理量本身没有乘 `I_clip`**。因此已 clip 的 token 即使没有有效梯度，仍可能通过 micro-batch 的最大 metric 影响其他 token 的权重。这与“理论考虑 clipping”不是完全相同的实现。

dual-clip 对负优势且高 ratio 的 token 也可能使梯度不活跃，不能用论文式 (6) 的两个条件完整描述旧损失的全部活跃区间。

### 5.2 一步训练的数据流

```text
prompt batch -> 为每个 prompt 建 uid -> 每个 prompt 采样 n 条回答
 -> verifier 给 outcome reward（放到最后一个有效回答 token）
 -> 更新前 actor 前向，得到 old_log_probs、entropys、max_prob_log_probs
 -> trainer 缓存 entropys 和 max_prob_log_probs 到 batch
 -> GRPO 按 uid 计算组内优势，广播到有效回答 token
 -> 每个 mini-batch / micro-batch：
      当前 actor 前向，计算当前 log_prob
      用缓存的旧 entropys 和当前 log_prob 计算权重
      权重乘 token PPO loss -> 聚合 -> backward
 -> 每个 mini-batch optimizer.step -> 同步 actor 权重到 rollout
```

旧熵在一次 rollout batch 内固定。actor 经过不同 mini-batch 更新后，`log_prob` 已随 θ 改变，而 `entropys` 仍来自更新前策略。这是混用 `pθ` 和 `H_old` 的工程近似，不是论文中的全当前策略 `pθ,Hθ`。

旧 actor 的 STEER 正常路径直接使用缓存熵；它的缺失熵 fallback 在 `entropy_coeff=0` 时不会计算熵，而且字段选择本身要求 `entropys`。因此不是可靠的“缺失时自动重算”机制。迁移后缺失必需张量应明确报错。

若同时开启 entropy regularization，缓存旧熵也不能提供有效的当前策略熵梯度；旧主脚本 `entropy_coeff=0` 避开了这个问题。按用户“以旧版为准”的决定，迁移后保留此行为，不能自行换成可微当前熵。

### 5.3 归一化范围与 batch 单位

旧权重函数是在 `update_policy` 的 micro-batch 循环内调用，因此 min/max 是 **单 DP rank、单 micro-batch 的有效回答 token** 范围，没有跨 rank all-reduce，也没有跨整个 optimizer mini-batch 统一最大值。

这意味着改变设备数、micro-batch size、动态分批或样本重排，可能改变权重本身。旧版局部 token mean 与新版全局 token mean 在变长回答上也会产生不同梯度；当前 STEER 同时保留旧权重范围和旧 loss 缩放规则。

旧脚本 `ppo_mini_batch_size=32` 的单位也要注意：旧 FSDP worker 会乘 `rollout.n` 再除 DP size；新版 trainer 同样先乘 `rollout.n`。当 `n=8, DP=8, SP=1` 时：

```text
rollout batch: 512 prompts × 8 = 4096 responses
optimizer mini-batch: 32 prompts × 8 = 256 responses（全局）
每 rank: 256 / 8 = 32 responses
micro_batch_size_per_gpu=8 -> 每 rank 每 mini-batch 4 个 micro-batch
```

不能把配置中的 32 误解成全局只有 32 条回答。

### 5.4 哪些旧改动不是 STEER 必需

- `max_prob_log_probs` 是词表最大概率 token 的 logprob，用于 clipping 分析；核心权重函数不读取它。
- `all_clip_positions`、clip token 概率比统计、额外绘图/日志不是 STEER 的必需输入。
- 旧 actor/worker 的返回值为这些统计做了扩展；不应把 9 元组或完整 actor 文件照搬到新版。
- `save_best_only/delete_old_best_checkpoint/save_after/best_metric_key` 是旧 trainer 的 checkpoint 定制，不属于算法。新版未发现同名行为实现。
- 旧 Megatron actor 没有同等的 `entropy_control` 接入逻辑，不能将旧仓库称为所有后端均支持的实现。

## 6. 论文与旧实现的差异清单

| 项目 | 论文 / 主实验 | 本地旧代码 | 适配时的处理建议 |
| --- | --- | --- | --- |
| 熵来源 | 当前策略 `Hθ` | 通常缓存 `H_old` | 第一阶段保留旧行为；当前熵方案另做实验 |
| Ω 的定义 | 状态上动作期望 | 采样 token 的代理量 | 文档和指标不要声称精确 ΔH |
| clipping | 理论显式有 `I_clip` | 权重无 `I_clip`，PPO loss 自身 clipping | 先保持，后续单独对照 |
| 公共系数 | 有 `η/L` | 省略 | 纯最大值比值中可抵消；有 0.02 floor 时尺度不再完全不变 |
| 最大值范围 | `max_{s∈B}`，未把工程分片方式写清 | rank 本地 micro-batch | 先明确记录局部行为，另评估全局范围 |
| 指数映射 | 除 batch max | 除 `max(batch max,0.02)` | 显式保留兼容阈值，不能静默删除 |
| 主实验下界 | 0.7 | exp / linear 脚本 0.8 | 提供清晰命名的配置，论文配置显式设 0.7 |
| 上界 | 指数 1；线性消融 1.2 | 指数忽略 `token_weight_max` | 校验/说明分支参数的意义 |
| 默认参数 | 不同于 helper 默认 | helper/loss 默认 0.8/1.2/linear；actor fallback 0.95/1.05 | 建立单一配置来源，recipe 全部显式指定 |
| clip high | 标准 GRPO 主方法；clip-high baseline 0.28 | exp / linear 脚本使用 0.28 | 保留旧 recipe，并标明是组合配置；论文基线条件另设 |
| 训练步数 | 最多 200 rollout steps | 脚本 10 epochs | 显式设置 `trainer.total_training_steps=200` 才与论文步数对齐 |
| 非对称控制 | 非主实验默认 | README/代码额外提供 | 后续可选，不作为默认论文实现 |
| 验证采样 | temperature 1，top-p 0.7 | 旧默认同此；n=1 | 新版默认不同，须显式覆盖 |

论文的 `λ_min=0.7` 是论文报告值，旧脚本 `0.8` 是当前源码事实；不能把二者视为相同实验。理论近似也不承诺真实更新的熵变化落入某个硬区间。

## 7. 奖励、数据与评估：另一个适配重点

### 7.1 本地数据清单与尚未核实项

[STEER-main/datasets](STEER-main/datasets) 实际包含：`DAPO-Math-17k.parquet`、`aime24.parquet`、`aime25.parquet`、`amc23.parquet`、`math500.parquet`、`minerva_math.parquet`、`olympiadbench.parquet`、`gsm8k_test.parquet`、`omni.parquet`。

本次环境没有 pyarrow/pandas，未读取 Parquet schema、实际行数和 `data_source` 分布。论文附录 E 表 8 的 17,398 个训练样本等是论文统计，不能充当对本地文件的检查结果。第一阶段 recipe 开始前必须检查 `prompt/reward_model.ground_truth/data_source/extra_info`，以及 uid 分组、模板、截断、EOS 和 response mask。

论文还新增了 coding 实验，但本地目录主要提供数学任务数据与脚本，不具备论文全部 coding 实验材料。

### 7.2 verifier 差异已确认

旧 [reward_score/__init__.py](STEER-main/verl/utils/reward_score/__init__.py) 有额外数据源分派：

- `math_dapo/deepscaler/simplelr_qwen/math_train` 等走 `math_dapo.compute_score`。
- `math_dapo_boxed/math500/olympiadbench/aime_2024_dapo_boxed/aime_2025_dapo_boxed/amc2023_dapo_boxed/minerva_math/...` 走 `multi_datasets_eval`，默认 `eval_method="both"`。
- `multi_datasets_eval.compute_score_both` 取 DAPO 与 Qwen 判定的 **逻辑或**；任一正确就记正确，返回 `score/acc/pred` 等。
- 依赖同目录 `experience_maker.py`、`qwen_math_eval_toolkit/`，并有旧的 `sys.path` 修改，后续应整理为可导入的独立 reward 模块。

旧 `math_dapo.compute_score` 默认 `strict_box_verify=True`；新版默认是 `False`。两边都返回正确 `+1`、错误 `-1`，但默认答案提取/验证路径不同，训练奖励可能变化。新版默认分派也没有完整保留上述自定义评估数据源。

建议使用新版 `reward.custom_reward_function.path/name` 接入独立 STEER reward wrapper，显式保留需要的分派与 strict-box 规则，复用新版 reward loop；不要全量替换新版奖励基础设施。先读取真实 `data_source` 再确定迁移哪些分派。

### 7.3 实验条件

论文数学训练条件：Qwen2.5-Math-7B 等，DAPO-Math-17k，512 prompts，8 rollouts/prompt，mini-batch 配置 32，lr `1e-6`、无 warmup，输入/输出最大长度 `1024/3072`，训练 temperature/top-p 为 `1/1`；无 KL loss、无 reward KL、无 entropy bonus、无 dynamic sampling；最多 200 rollout steps。主实验两次独立运行取平均。

旧 exp recipe：8 张 H20 配置、rollout TP=4、GPU memory utilization=0.6、micro-batch=8、dual-clip c=10、clip low/high=`0.2/0.28`、指数权重下界 0.8。极端场景脚本只把 clip high/low 改为 `5/0.99`。

论文评估 temperature/top-p=`1/0.7`；AIME24/AIME25/AMC23 报 avg@32，其余数学集 avg@1。旧 `eval.sh` 设置 `val_kwargs.n=1`，但实际读取 Parquet 后确认 AIME24、AIME25、AMC23 每题已重复 32 次（分别为 960/960/1280 行，对应 30/30/40 个唯一 prompt），因而数据层已经提供每题 32 次采样预算，不能再无条件把 n 调到 32。旧脚本还包含 GSM8K/Omni 共八个测试集。选择最佳 checkpoint 与每次验证都使用多少采样也要分清。

**新版 rollout 验证默认是 temperature=0、top-p=1、do_sample=False、n=1**，与旧默认不同。只迁移 loss 而不覆盖这些设置，结果不可直接比较。avg@32 是每题多次采样的平均准确率，不是 pass@32。

## 8. verl 0.9.0 的真实接入路径

### 8.1 默认 V1 与兼容 V0 必须区分

[main_ppo.py](verl-0.9.0/verl/trainer/main_ppo.py) 根据 `trainer.use_v1` 分流；[ppo_trainer.yaml](verl-0.9.0/verl/trainer/config/ppo_trainer.yaml) 默认为 `true`，`trainer.v1.trainer_mode=sync`。

默认路径：

```text
main_ppo.main
 -> TaskRunnerV1 -> TransferQueue 初始化
 -> ppo/v1/trainer_sync.py::PPOTrainerSync
 -> ppo/v1/trainer_base.py::PPOTrainer
 -> ActorRolloutRefWorker（workers/engine_workers.py）
 -> TrainingWorker.train_mini_batch -> engine.train_batch
 -> workers/utils/losses.py::ppo_loss
 -> core_algos.get_policy_loss_fn(loss_mode)
```

V1 `_step_once`（536 起）依次采样、奖励、balance、old logprob、可选 ref/critic、advantage、update。同步 trainer 在采样结束 sleep rollout replicas，step 结束同步新权重。第一阶段保留同步模式，不引入异步陈旧样本语义。

`trainer.use_v1=false` 才走 `main_ppo_v0.py -> ray_trainer.py::RayPPOTrainer`；它仍使用新版 unified engine，**并不是回到旧 `DataParallelPPOActor`**。目标目录中不存在旧版 `verl/workers/actor/dp_actor.py`。

### 8.2 V1 的熵已经缓存，但格式不同

[v1/trainer_base.py](verl-0.9.0/verl/trainer/ppo/v1/trainer_base.py)：

- `_compute_old_log_prob`，1479–1537：请求 `calculate_entropy=True`，读取模型输出 `entropy/log_probs`；通过 `response_from_nested` 截取回答位置，写回 TQ 为 `entropy/old_log_probs`。
- 此时 `data["entropy"]` 是 **更新前的、response-only nested tensor**，不是当前模型输出的全序列 entropy。
- `_update_actor`，1672–1711：设置 mini-batch、epochs、temperature 等 metadata，再调用统一 worker。是否计算当前熵由 `actor.calculate_entropy or entropy_coeff!=0` 决定。
- TQ bridge 位于 [transferqueue_utils.py](verl-0.9.0/verl/utils/transferqueue_utils.py)，`tqbridge`、`_async_meta_to_realdata`、`async_kv_batch_meta2batch_meta`。metadata 的 `fields` 若有限制，会筛掉未列出的字段；需要在真实集成验证中确认缓存熵始终随更新 batch 到达 loss。

迁移旧行为时，应从 `data["entropy"]` 读取缓存熵并用所选 response 字段一起 `to_padded_tensor()`；不要对它再次调用 `no_padding_2_padding`，否则会重复切片/左移。

### 8.3 V0 的熵被主动删除

[ray_trainer.py](verl-0.9.0/verl/trainer/ppo/ray_trainer.py)：

- `_compute_old_log_prob`，1265–1300：返回 `old_log_probs/entropys`，均为回答位置 padded 张量。
- `fit`，1561–1585：记录熵指标后 `old_log_prob.batch.pop("entropys")`，之后才 union 到 batch。
- `_update_actor`，1302–1351：转 TensorDict、转换输入 padding 格式，再调用统一 worker。

若第一阶段也支持 V0，需要仅在 STEER 模式保留旧熵；loss bridge 可把 V1 的 `entropy` 和 V0 的 `entropys` 转成同一个局部变量传入，避免改变所有算法的字段约定。

### 8.4 `ppo_loss` 是统一损失桥

[workers/utils/losses.py](verl-0.9.0/verl/workers/utils/losses.py)，57–144：

1. 当前 `model_output["log_probs"]` 是模型预测位置输出，用 `no_padding_2_padding` 转成 `[B,T_response]`，包含预测对齐所需的一次左移。
2. 可选当前 `model_output["entropy"]` 走同样转换。
3. 从 data 中只选择 `response_mask/old_log_probs/advantages`，以及可选 `rollout_is_weights/ref_log_prob`；**现有字段列表不包含旧熵**。
4. 当前注册 loss 调用契约：

   ```python
   pg_loss, pg_metrics = policy_loss_fn(
       old_log_prob=old_log_prob,
       log_prob=log_prob,
       advantages=advantages,
       response_mask=response_mask,
       loss_agg_mode=loss_agg_mode,
       config=config,
       rollout_is_weights=rollout_is_weights,
   )
   ```

5. 额外添加 entropy regularization 和 KL loss，再返回 `(loss, metrics)`。

所以只在 `core_algos.py` 注册旧函数不够：旧函数参数多一个熵张量、没有新版 config/rollout_is_weights 契约，返回值也不兼容。建议只对 `entropy_control` 分支增加 entropy kwargs，其他注册 loss 不受影响；避免无差别向所有函数传入新参数。

### 8.5 新版全局 loss 缩放与旧版的区别

新版 [core_algos.py](verl-0.9.0/verl/trainer/ppo/core_algos.py)：`compute_policy_loss_vanilla` 1286–1376 与旧 PPO 主体相同，但支持 `rollout_is_weights`，返回 `(pg_loss, pg_metrics)`，并调用：

```python
agg_loss(..., **config.global_batch_info)
```

`global_batch_info` 包含 `dp_size/batch_num_tokens/global_batch_size/loss_scale_factor`。FSDP engine 的 `forward_backward_batch`（700 起）在分 micro-batch 前跨 DP 汇总有效 token 数。token-mean 情况：

```text
loss_micro = sum(M·l·λ [·rollout_is_weights]) / global_valid_tokens · dp_size
```

以上是 0.9.0 原生 loss 的语义。**当前 STEER 按用户决定不采用该分母**：恢复旧 actor 的局部 loss、静态累积除数和动态样本数比例，然后由 FSDP 对 DP 梯度平均。其他算法保持 0.9.0 原生聚合。不使用 `sum(Mλ)` 作分母。

旧代码没有新版 rollout IS/RS；当前 STEER 显式拒绝这些组合和 bypass，避免改变旧算法。将来只有用户明确要求研究扩展时再另行实现。

### 8.6 配置、引擎与依赖

- [PolicyLossConfig](verl-0.9.0/verl/workers/config/actor.py) 79 起目前没有 STEER 字段；YAML 使用 `_target_`，worker 经 Hydra 转 dataclass。只用旧脚本的 `+token_weight_min` 不能替代增加 dataclass 字段，构造阶段可能失败。
- `ActorConfig.calculate_entropy` 已存在，默认 false。若以后选择当前熵模式，需要自动保证 actor forward 请求熵，不能依赖非零 `entropy_coeff` 才计算。
- [engine_workers.py](verl-0.9.0/verl/workers/engine_workers.py) 637 附近为 actor 绑定 `partial(ppo_loss, config=actor_config)`，通常无需修改整个 worker。
- [engine/fsdp/transformer_impl.py](verl-0.9.0/verl/workers/engine/fsdp/transformer_impl.py) 已支持当前 entropy；不要新增一套完整 logits 传输。当前熵会增加词表归约/图存储开销，旧熵路径可复用 inference 已完成的计算。
- 模型输出与 response data 的格式转换分别参考 [padding.py](verl-0.9.0/verl/workers/utils/padding.py) 的 `response_from_nested/no_padding_2_padding/response_to_nested`；全序列输出与 response-only 缓存绝不能共用一次不加区分的转换。
- 旧 requirements 为 `tensordict<=0.6.2`；目标 setup 为 `>=0.8.0,<=0.10.0,!=0.9.0`，其中 V0 `DataProto.to_tensordict` 又明确要求至少 0.10。应选兼容实际路径的依赖，不能复用旧环境。
- 目标 setup 的 vLLM extra 要求 `vllm>=0.18.0`，旧 requirements 仅注释 `vllm==0.8.4`；目标默认 rollout.mode 为 async serving。环境应按本地 0.9.0 文档和后端要求重新配，不是复制旧依赖锁定。

## 9. 初始分阶段适配计划（当前进度见第 11 节）

> 本节为历史计划。用户随后要求算法差异以旧版为准，以下关于 FP32、空 mask 安全化、全局梯度不变性及 IS 的建议均未作为最终语义；最终验收标准见第 11 节。阶段 C 的实验工作仍不在范围内。

### 阶段 A：先固定第一版语义

建议第一版目标：**V1 sync + FSDP + vLLM + 单轮数学 GRPO；对称指数 STEER；兼容旧 token proxy；复用缓存旧熵；局部 micro-batch 归一化；保留 0.02 floor。**

使用相同实现提供两套明确命名的 recipe 参数：

- 旧 recipe 对照：`λ_min=0.8`、clip high=0.28、c=10，便于检查迁移前后差异。
- 论文设置参考：`λ_min=0.7`、明确采样/验证/200 步条件；需如实标注仍沿用旧实现的代理量和缓存熵近似，不称为逐项严格公式复现。

线性分支可在同一 helper 中保留并验证；非对称分支若保留，沿用原行为并清楚区分 linear/exp。暂不新增二值映射、异步训练、GSPO 混合、蒸馏或全局极值归一化。

### 阶段 B：实现配置与数学核心

预计修改目标目录下：

| 文件 | 具体工作 | 验收点 |
| --- | --- | --- |
| `verl/workers/config/actor.py` | 给 PolicyLossConfig 增加 min/max、linear、entropy_control_mode 等字段，统一默认来源 | 无未知字段；合法范围/模式校验清楚 |
| `verl/trainer/config/actor/actor.yaml` | 暴露对应配置，注释遵守现有格式规则 | Hydra 组合并实例化成功 |
| `verl/trainer/ppo/core_algos.py` | 迁移 `compute_token_weights`；注册新契约的 entropy_control loss | 权重停止梯度；PPO/dual-clip、IS、global aggregation 正确 |
| `verl/workers/utils/losses.py` | 条件读取缓存旧熵；传给 STEER；不改其他 loss 的调用 | V1 response nested / V0 padded 正确；缺失熵清晰报错 |
| `verl/trainer/ppo/v1/trainer_base.py` | 检查 entropy 从 TQ 经字段筛选到 worker 的可达性；只在确有必要时修改 | 缓存熵没有丢失或被当前模型输出误覆盖 |
| `verl/trainer/ppo/ray_trainer.py` | 若支持 V0，STEER 模式保留 entropys | 非 STEER 路径原行为保持 |

旧函数遇到全 mask 时，helper 虽返回零，loss 后续 `min/max` 对空张量仍报错；新版必须让空位置统计安全、loss 可导且无 NaN。非有限输入策略、FP32 中间计算、mask 在归约前的处理应明确，避免照搬“先产生 Inf 再乘 0”。

权重相关中间量建议用 FP32；旧 clamp 上界 `1-1e-8` 在低精度下未必可表示。若数值处理使正常有限输入的结果发生变化，应在兼容测试中定量记录。`0<λ_min<=1`；线性上下界合法；非对称指数还需确保中性点/上界语义合理。不要在普通训练中把警告后的静默 fallback 当成成功。

指标返回新版 dict；不要返回 clipping 大张量或 Python `.item()` 多次同步的研究分析数据。至少记录权重 mean/min/max、预测熵变化幅度、有效 token 数和异常计数。当前 `ppo_loss` 把所有 pg_metrics 包成 `Metric(..., MEAN)`，所以真正跨 DP 的 min/max 和 token 加权均值需要额外设计/验证，不能把“micro-batch 最小值的平均”标成全局最小值。

### 阶段 C：recipe、奖励与环境（用户已明确暂不处理）

1. 在目标目录新增独立 STEER example/recipe，不改写普通 GRPO 默认 loss。
2. 显式固定 `trainer.use_v1=true`、`trainer.v1.trainer_mode=sync`、GRPO、token-mean、关闭 KL/entropy bonus/动态筛组、bypass=false。
3. 显式设置训练和验证采样参数，区分 smoke test、旧 recipe 对照与论文评估预算。
4. 检查 Parquet schema/data_source，提供独立 verifier wrapper；小样本对照旧版正确/错误/无 box/等价表达式等结果。
5. 从目标环境启动，记录 `verl.__file__`、版本、有效配置和模型/tokenizer 路径。验证 Ray worker 与 driver 都导入目标包。
6. 按实际 GPU/显存决定 TP、DP、micro-batch；本地脚本的 8×H20 不是未知机器可以直接使用的配置。
7. 使用原生 checkpoint 机制；需要复现最佳 checkpoint 选择时再明确增加规则。后续要断点续训时须保存模型、optimizer 和训练状态，旧脚本的仅 `hf_model` 不能被当成完整恢复方案。

### 阶段 D：分层验证

以下是原始验收计划，已完成的部分见第 11 节：

- **数学对照**：用固定张量比较旧 helper 与迁移 helper 的权重、loss 和对 logprob 的梯度。覆盖两个 A 符号、clipping/dual-clip、padding、常量 metric、零 A、极小概率、线性/指数、需要保留的非对称模式。
- **退化对照**：所有有效权重为 1 时，新 loss 与 0.9.0 vanilla 的 loss/梯度一致；不要只比较返回 scalar。
- **mask/边界**：全 mask、混合有效长度、padding 极端值、非法配置和缺失熵；不出现空 min/max、除零或 NaN。
- **梯度隔离**：权重无梯度，旧熵无需梯度；当有 entropy bonus 时，正则项仍使用当前可微熵。
- **聚合**：固定权重后验证不同 micro-batch 拆分和模拟 DP 缩放与一次全局计算一致；单独展示局部 min/max 重新计算导致的预期权重差异，不能混为 loss 聚合 bug。
- **接口**：Hydra -> dataclass -> 注册表 -> ppo_loss 跑通；V1 nested entropy 与 token 输出只左移一次；V0 若声明支持也做测试；IS 权重与 STEER 权重乘法正确。
- **端到端 smoke**：小模型/小数据运行数步，观察 reward/advantage/entropy/weight/clip/grad norm，完成保存与验证；确认没有跑回旧包。
- **训练对照**：同模型、数据、seed、verifier、采样条件运行 vanilla 与 STEER；先看训练信号与熵曲线，再扩展到约 200 步和多 seed。小规模 smoke 不能证明论文性能。

可复用的目标测试位置：

- `tests/trainer/ppo/test_core_algos_on_cpu.py`
- `tests/trainer/ppo/test_dynamic_policy_losses_on_cpu.py`
- `tests/trainer/ppo/test_loss_aggregation_on_cpu.py`
- `tests/workers/config/test_actor_config_on_cpu.py`
- `tests/models/test_engine.py`（后端集成参考）

新增测试应验证 STEER 的实际契约与迁移语义，不需要重写整套既有 PPO 测试。

### 阶段 E：迁移稳定后再研究的变化

1. 改用当前 `Hθ`，与缓存 `H_old` 做独立对照；保证 actor 计算熵且仅权重路径 detach。
2. 在 proxy 中加入实际梯度活跃 mask，并考虑 dual-clip；验证它对最大值归一化的影响。
3. 明确论文 `B` 的工程含义，评估全 mini-batch / DP 全局最大值；不能简单在任意局部 loss 内加 collective 就宣称支持，需考虑 micro-batch 调度、SP/PP 和同步范围。
4. 对比是否保留 0.02 floor；这与省略 `η/L` 的尺度选择有关。
5. 验证 Megatron/其他 engine、RLOO/OPO 以及异步 rollout；每项单独声明支持范围。

## 10. 恢复工作清单与当前限制

后续继续工作时按顺序执行：

1. 读本文第 1、6、8、9 节，检查工作区是否已有新改动。
2. 阅读第 11 节，沿已完成的旧实现兼容方案继续；有差异以旧版为准，不要从头移植或擅自替换成当前熵、全局 token mean、FP32 proxy 或 rollout IS 版本。
3. 用户明确限定当前只做算法适配；不新增实验配置、启动脚本、数据集奖励函数或依赖安装脚本。
4. 后续若用户要求运行验证，再在 NPU + Qwen3.5-9B 上确认训练链路。
5. 将实际修改、测试结果、语义取舍更新回本文。

最初调研时默认 Python 缺少训练依赖；本轮已从本机只读缓存加载实际 CPU PyTorch、TensorDict、Hydra 完成数值对照，并用 PyArrow 核实 Parquet。没有创建或修改训练环境，没有运行分布式/NPU 训练。完整 Ray/TransferQueue 运行、NPU 算子兼容性与训练性能仍需目标机器验证。

修改前阅读快照指纹（SHA-256；不是当前文件指纹）：

```text
STEER.pdf
50199fa221ed9defc230e350ee60edebd848e7d6c56b5c0fb500c8289ea4073f
STEER-main/verl/trainer/ppo/core_algos.py
ca0ab2f064f22a4fc8ecbb138ea3a44d8b0dfb050b580d2b3bf7a3c047902d2e
STEER-main/verl/workers/actor/dp_actor.py
d028fc67a98910bfa3278051d2c04af7ffc0bcc9cdfadc32f2c5228bc3ee0626
verl-0.9.0/verl/trainer/ppo/core_algos.py
0ec87b90cba24fa0aa959adff573efb2cc49434d705621cf546be231e56a8932
verl-0.9.0/verl/workers/utils/losses.py
c6dfeb10900054e96b432633b975f3e3a535d3358a4ad13eeda9f730c5942d4a
```

## 11. 当前适配：以旧版为准（2026-11-04）

### 11.1 用户决定与范围

用户在差异审查后明确决定：**“有差异的地方以旧版为准。”** 以 `STEER-main` 的数学和 actor 更新行为作为本次适配依据，论文用于理解和解释旧实现的近似。旧源码不修改。只适配 `verl-0.9.0` 的算法路径，不处理实验 recipe、启动脚本、奖励迁移或环境安装。

原先第一版迁移采用的 FP32 proxy、全局 token/sequence 分母、当前熵正则、附加 rollout IS 以及全空 mask 安全返回，均已撤回。先前的 21 项测试把新版 `agg_loss` 注入旧函数，不能证明完整旧版聚合一致；当前测试从旧源码独立提取 `agg_loss` 和旧 masked reductions，不再借用新版归约。

### 11.2 已固定的语义

| 项目 | 当前 STEER 行为 |
| --- | --- |
| 熵来源 | 更新前缓存的 `H_old`；当前采样 token 概率随 actor 更新变化 |
| proxy | 按旧源码运算顺序计算，不补论文的动作期望、η/L 或 clipping 指示器 |
| dtype | 保留输入 dtype 的指数、clamp、log 等计算；不强制中间量 FP32；输出权重仍是旧版的 FP32 |
| 非有限 proxy | 旧版警告后逐元素置零，再做绝对值及映射；不改成异常退出 |
| 归一化范围 | 单 DP rank、单 micro-batch 有效 token 的 min/max；不跨卡或跨 micro-batch 汇总 |
| 映射 | symmetric/asymmetric × linear/exponential 四分支；保留指数分母 floor=0.02 及原算术顺序 |
| actor 默认参数 | `token_weight_min=0.95, token_weight_max=1.05, linear=true, entropy_control_mode=symmetric`，匹配旧 actor fallback |
| 独立 helper 默认 | `0.8/1.2/linear=true`，保留旧 helper 默认；旧 exp recipe 的 `0.8/1.0/false` 不能冒充 actor 默认 |
| PPO | 旧 clipped + dual-clip 计算，权重乘最终 token loss；不除以权重和 |
| 静态梯度累积 | 每个 micro-batch 的局部 loss 除以 `local_mini_batch_size // ppo_micro_batch_size_per_gpu` |
| 动态梯度累积 | 每个局部 loss 乘 `len(micro_batch) / local_mini_batch_size` |
| DP | FSDP 对各 rank 累计梯度做平均；STEER 不使用新版全局 token 分母 |
| 熵正则 | 非零 entropy_coeff 时也用 detached 缓存旧熵，影响 loss 数值但无当前熵梯度；保留旧行为 |
| KL 正则 | 旧 KL 公式和局部聚合，随后应用相同累积缩放；不采用新版 `k3+` 等 straight-through 变体 |
| rollout correction | 拒绝 IS/RS/bypass，不悄悄乘额外权重或筛掉旧算法原有样本 |
| 边界情况 | helper 对全空 mask 返回零；loss 后续 min/max 仍会失败。全 BF16 或全 FP64 输入写入 FP32 权重的 dtype 错误也按旧版保留 |

四种旧聚合精确为（`M` 是 response mask，`L` 是 padded response 宽度）：

```text
token-mean:               sum(loss*M) / (sum(M)+1e-8)
seq-mean-token-sum:       mean_batch(sum_token(loss*M))
seq-mean-token-mean:      mean_batch(sum_token(loss*M)/sum_token(M))
seq-mean-token-sum-norm:  sum(loss*M) / L
```

最后一种在旧代码中**没有除 batch size**，当前不自行补上。V1 nested 张量局部转 padded 后可能缩短宽度，因此 loss bridge 恢复到 `rollout.response_length`，以保留旧归一化 horizon。`loss_scale_factor` 和 `global_batch_info` 不替换上述旧分母。

变长例子：两个 micro-batch 各一条回答，长度为 1 和 3，单位权重且无 clipping。旧 token-mean 对四个有效 log_prob 的梯度为 `[-1/2, -1/6, -1/6, -1/6]`；全局 token mean 为四个 `-1/4`。当前已通过实际 FSDP engine 的分批和 backward 循环验证前者；模型前向与分布式 collective 在该 CPU 测试中使用替身。

### 11.3 文件与调用链

| 文件 | 改动 |
| --- | --- |
| `verl-0.9.0/verl/workers/config/actor.py`、`verl/trainer/config/actor/actor.yaml` | 有类型 STEER 字段和旧 actor 默认；有限值/范围/模式校验；vanilla 仍是普通默认 loss |
| `verl-0.9.0/verl/trainer/ppo/core_algos.py` | 旧 `compute_token_weights`、独立 `agg_steer_loss`、注册 `entropy_control`；返回新版 `(loss, metrics)` 接口 |
| `verl-0.9.0/verl/workers/engine_workers.py` | actor 更新传递 STEER 标记、原 response horizon 并关闭当前熵计算；TrainingWorker 在真正的 rank-local optimizer mini-batch 记录样本数 |
| `verl-0.9.0/verl/workers/utils/losses.py` | V1/V0 缓存旧熵对齐；恢复 response horizon；旧局部聚合和累积缩放；旧熵及 KL 正则；不改其他 loss 的数学规则 |
| `verl-0.9.0/verl/trainer/ppo/ray_trainer.py` | V0 保留 `entropys`；STEER 缓存 old/ref log_prob 和熵时不额外 `.float()`，其他算法保持 FP32 转换 |
| `verl-0.9.0/verl/utils/config.py` | 拒绝 bypass、rollout IS/RS、V1 非同步、蒸馏、非 fsdp/fsdp2 和旧版不支持的 token-sum |
| `verl-0.9.0/tests/trainer/ppo/test_steer_on_cpu.py` | 旧源码独立参考、数值/梯度、loss bridge、真实 worker mini-batch 元数据及引擎 backward 测试 |

算法入口仍为 `actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control`。四个算法参数位于 `actor_rollout_ref.actor.policy_loss` 下。没有新增训练实验配置。

V1 链路：`_compute_old_log_prob` → response-only nested `entropy` → TransferQueue → `ActorRolloutRefWorker.update_actor` → `TrainingWorker.train_mini_batch` 记录本地样本数 → FSDP micro-batch → `ppo_loss` → 旧 STEER 数学 → 累积缩放 → backward。

V1 原本已经缓存完整词表条件熵；cached entropy 不再次左移。模型 log_prob 经 `no_padding_2_padding` 左移一次。V0 使用 `entropys`，缺失旧熵明确报错。

TransferQueue 字段可达性已静态追踪：本机缓存 `transferqueue==0.1.8` 的后续 put 合并 metadata 字段；写入 advantages/returns 不会删除旧 entropy。这不是 Ray/TransferQueue 分布式服务的运行验证。

额外指标为 `actor/token_weights_microbatch_mean` 和 `actor/token_weights_min/max`。mean 是 micro-batch 均值聚合；min/max 分别用 MIN/MAX；不称为全局 token 加权均值。loss 指标包含旧累积缩放，按 micro-batch SUM 和 DP mean 汇总，因此日志组织与旧研究脚本不保证逐项相同。没有搬运旧 clipping 大张量调试返回值。

### 11.4 验证结果与限制

本轮 STEER 专项 **56 项通过**，覆盖：

- 四映射 × 四聚合，和独立旧源码权重、loss 严格相等及梯度对照；输入精度和旧异常行为。
- 变长回答经 `ppo_loss`、静态/动态 micro-batch、模拟 DP=1/2 平均，含旧熵正则及 KL 后的累计 loss/梯度对照。
- V1 nested / V0 padded 缓存熵、变长 prompt/response 对齐、旧熵与当前熵的梯度隔离、response horizon。
- 实际 TrainingWorker mini-batch 循环中的本地样本数，actor 更新的元数据，以及实际 FSDP engine 拆分和反向传播循环。
- V0 old/ref 概率和熵缓存的 BF16 保留；非 STEER 继续转 FP32；Hydra、dataclass、注册和不支持组合。

原有四组算法/配置回归 **40 项通过**，FSDP engine 单进程测试 **5 项通过**。额外双进程 FSDP 原有测试运行失败：本机 Gloo 报 `Cannot resolve 127.0.0.1 to a (local) address`，未能验证真实跨进程通信。没有改测试掩盖此限制。修改文件 Ruff 检查、格式检查与 `git diff --check` 通过。

正常依赖环境下，从 `verl-0.9.0` 执行：

```bash
python -m pytest -q tests/trainer/ppo/test_steer_on_cpu.py
python -m pytest -q tests/trainer/ppo/test_core_algos_on_cpu.py tests/trainer/ppo/test_dynamic_policy_losses_on_cpu.py tests/trainer/ppo/test_loss_aggregation_on_cpu.py tests/workers/config/test_actor_config_on_cpu.py
python -m pytest -q tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py
```

本机没有可用 uv；之前联网安装未成功且审批服务返回错误，未安装包或创建环境。本轮沿用 `/tmp/steer-test-runner.py`，从本机只读缓存加载 Python 3.12、PyTorch 2.14.1 CPU、TensorDict 0.10.0、Hydra 等依赖，关闭无关 pytest plugin 自动加载。离线 runner 已补 main guard，以允许 multiprocessing spawn；随后双进程测试停在上述 Gloo 初始化错误。后续环境管理仍按 AGENTS 使用 uv。

**未验证**：NPU 实际前向/反向、Qwen3.5-9B 加载、真实 DP/SP、完整 Ray/TransferQueue、checkpoint、训练收益。当前仅对相同输入张量与相同 micro-batch 划分验证旧数学一致；新版模型内核、动态分组和运行环境可能产生不同概率、熵或分片，不能据此承诺完整训练逐位一致。当前接入范围为同步 FSDP/FSDP2，未宣称支持 NPU MindSpeed/Megatron 引擎。
