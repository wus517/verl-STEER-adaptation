# STEER 四象限适配过程记录

> 本文记录原分段四象限模式。`steer_quadrant_algorithm_v2` 新增Ⅱ/Ⅳ平方助推独立模式，规格见 [STEER_QUADRANT_RECIPROCAL_BOOST.md](STEER_QUADRANT_RECIPROCAL_BOOST.md)，最新实现与验证见 [STEER_QUADRATIC_IMPLEMENTATION_LOG.md](STEER_QUADRATIC_IMPLEMENTATION_LOG.md)。原模式和原脚本继续保留。

## 当前状态与授权边界

- 分支：`steer_quadrant_algorithm`。
- 状态：**四象限算法及服务器脚本已实现，本地 CPU 验证完成；NPU 实机训练待验证。**
- 授权：准备阶段只建立过程记录；后续用户明确要求开始执行、完成算法，并基于 `run_steer_4096.sh` 新建服务器脚本。
- 准备开始时工作区干净；具体源码、配置、测试和脚本改动见文末交付清单。
- 算法规格来源：`STEER_QUADRANT_ALGORITHM.md`。现有旧版迁移背景见 `STEER_ADAPTATION_NOTES.md` 第 11 节。
- 用户最新明确确认：**继续使用 `D=max(M, 0.02)`**，不因上一轮疑问改成纯最大值。
- 新规格只覆盖四象限模式的权重行为；未指定改变的训练行为沿用当前 STEER 基线。
- 后续每完成一阶段，更新本文件的状态、变更文件、测试结果和未决项；不要因上下文压缩重复实现或遗漏用户约束。

## 已确认的算法规则

有符号逐 token 分数：

```text
p = exp(log_prob)，沿用现有概率 clamp
p_old = exp(old_log_prob)，沿用现有概率 clamp
x = -(A/p_old) * [p*(1-p)*(log(p)+H)]
M = max_valid(abs(x))
D = max(M, 0.02)
z = abs(x)/D
```

- `x>0` 为估计增熵，`x<0` 为估计降熵；不是直接按 token 概率阈值划象限。
- 保留现有采样 token proxy；不自行补论文的 clipping 指示器、动作期望或公共系数。
- `M` 在单 DP rank、单次 actor micro-batch 的所有有效 token 中计算。**四个象限共用分母**，不分别求最大值，不跨卡汇总。
- 计算精度和概率截断沿用当前基线；非有限 score 按文档参考实现置零。

| 条件 | 象限 | 权重 |
| --- | --- | --- |
| `A>0, x<0` | Ⅰ | 原 symmetric 指数 STEER，`w_min^z`，按原指数运算顺序实现 |
| `A>0, x>0` | Ⅱ | 使用下述增熵曲线 |
| `A<0, x<0` | Ⅲ | 常数 `w3` |
| `A<0, x>0` | Ⅳ | 与Ⅱ共用同一条增熵曲线 |
| `x=0` | 零分 | 有效位置为 `1.0` |
| `response_mask=0` | padding | `0.0` |

增熵曲线：

```text
low  = 0.5-d
high = 0.5+d
0<z<low:       lambda = 1 + gamma*(low-z)/low
low<=z<=high:  lambda = 1
high<z<=1:     lambda = 1 - gamma*(z-high)/(1-high)
```

推荐显式参数为 `w_min=0.70, d=0.20, gamma=0.10, w3=1.00`。
`0.30/0.70` 是 **z 的分段边界**；`0.9` 和趋近 `1.1` 是该参数下的**输出权重**。零分单独取 1；`z→0+` 时增熵权重趋近 1.1。
Ⅱ/Ⅳ可以大于 1，不能统一 clamp 到 `[w_min,1]`，不能被原 `token_weight_max` 截断。

参数名和范围：

| 字段 | 推荐值 | 校验 |
| --- | --- | --- |
| `token_weight_min` | `0.70` | 有限，`0<w_min<=1` |
| `entropy_band_half_width` | `0.20` | 有限，`0<=d<0.5` |
| `entropy_weight_strength` | `0.10` | 有限，`0<=gamma<1` |
| `quadrant_3_weight` | `1.00` | 有限，`w3>0` |

## 实现边界与既有路径

- 保留 `algorithm.adv_estimator=grpo`，不重写优势估计。
- 入口采用文档给出的 `loss_mode=entropy_control`、`entropy_control_mode=quadrant`、`linear=false`。
- 新增独立四象限 helper，在 `compute_policy_loss_with_entropy` 的权重选择处接入，复用裁剪完成后的 token PPO/dual-clip loss。
- 权重在 `torch.no_grad()` 下计算；梯度只经过被加权的原策略项。不按权重和重新归一化。
- 默认沿用当前分支的缓存旧熵，与原 STEER 对照使用同一来源。当前 worker 明确关闭更新前向的当前熵，loss bridge 读取 V1 `entropy` / V0 `entropys`。不擅自增加即时熵模式。
- `old_log_prob` 在当前基线是训练端更新前重算的 proximal old actor 概率；不把它偷偷换成 vLLM `rollout_log_probs`。文档“采样策略”称谓不能导致改动该数据流。
- 保留旧版 `agg_steer_loss`、静态/动态梯度累积缩放、response horizon、DP 平均、KL/熵正则入口和 IS/RS/bypass 限制。
- 原 `symmetric/asymmetric` 行为与现有默认参数不得因新模式变化。推荐四象限参数通过新模式配置显式选择，不改旧 actor/helper 默认映射。
- quadrant 与 `linear=true` 不应悄悄退回线性映射；实现时使用明确配置校验或清晰分支契约，确保它始终按本文指数基线执行。
- 文档的“单位权重退化 GRPO”要求相同归约与累积规则。不能拿旧 STEER 局部 mean 与 0.9.0 vanilla 全局 token mean 直接比较并声称等价。
- helper 全空 mask 返回零是明确验收项；当前完整 loss 的 min/max 在全空 mask 下会报错。不得将 helper 支持夸大为完整训练链路已支持，也不顺带修复旧模式边界行为。
- 训练目标仍为 NPU + Qwen3.5-9B；本机 CPU 验证不能替代服务器验证。保持设备通用张量运算。

## 实施顺序（已完成）

1. 重查分支和工作区，保留用户新增改动；读此记录及算法规格，确认适用 `AGENTS.md`。
2. 增加 quadrant helper、模式分派和三个参数的传递。
3. 在 `PolicyLossConfig` 和 actor YAML 暴露字段、范围校验及模式契约；保持旧模式默认。
4. 添加独立算法参考与真实 policy-loss 调用验证；检查 V1/V0 缓存熵路径可直接复用。
5. 如提供训练入口，优先新增独立 quadrant 脚本或明确模式覆盖，保留原 STEER 对照脚本；不重调数据、奖励、batch、学习率和并行配置。具体文件在实施时记录。
6. 跑针对性测试及相关回归、Ruff、Bash/Hydra 检查；记录通过项、失败原因与验证限度。
7. 更新本文件的完成清单与结果，汇报实际改动；未经请求不提交或推送。

实际核心改动涉及：

- `verl-0.9.0/verl/trainer/ppo/core_algos.py`
- `verl-0.9.0/verl/workers/config/actor.py`
- `verl-0.9.0/verl/trainer/config/actor/actor.yaml`
- 四象限专项 CPU 测试与必要的训练入口文件
- 只有证据表明需要时才调整 loss bridge / worker，避免扩大改动范围。

## 验收清单

- [x] 文档第 7 节数值表、四个象限、符号示例。
- [x] Ⅰ与原 symmetric 指数模式在相同 FP32 输入下逐项对齐，覆盖 `M>0.02` 和 `M<0.02`。
- [x] Ⅱ/Ⅳ共用曲线；`z=0.3/0.7` 权重为 1；`z=0.1/0.9` 为约 `1.066667/0.933333`；`z=1` 为 0.9。
- [x] Ⅲ独立参数只改变Ⅲ，仍参与共享分母统计。
- [x] `d=0`、`gamma=0`、`w_min=1`、非法参数和非有限参数校验。
- [x] 零优势、`x=0`、全零 score、非有限 score、padding 和 helper 全空 mask。
- [x] padding 极值不影响分母；不同象限中的极值共同决定分母；floor 生效时不强求 z 最大值为 1。
- [x] 全单位权重与相同归约的未加权 clipped/dual-clip 策略 loss 和梯度一致。
- [x] 权重不求导；策略 loss 可反传；四象限 helper 的显式 FP32 输出与基线输入精度关系有验证。
- [x] 四种已有 loss 聚合、静态/动态 micro-batch 累积及缓存熵对齐不回归。
- [x] 原 STEER symmetric/asymmetric 回归通过；Hydra → dataclass → registry → loss 参数可达。

## 已知验证环境与历史问题

- 环境管理遵循 `verl-0.9.0/AGENTS.md`，使用 uv；不自行联网安装或改服务器依赖。
- 历史 CPU 验证使用 `/tmp/steer-test-runner.py` 加载本机只读缓存依赖；临时文件可能已失效，执行前检查。历史数字不是本轮结果。
- 当前引擎新增的 `grad_sync_every_micro_batch` 是用户已有优化，必须保留。此前两个测试替身缺 `engine_config` 导致报错；实际正常引擎构造会设置该属性。实施时区分既有测试夹具问题和新算法问题，不擅自回退优化。
- 真实双进程 FSDP 测试此前受本机 Gloo 回环地址解析失败阻塞；NPU、真实跨卡通信及训练效果未在本机验证。

## 执行记录

### 准备阶段

- 已通读算法规格，对照当前权重函数、策略损失、配置入口和缓存熵路径。
- 已记录用户确认的 `D=max(M,0.02)` 和 z 边界/权重范围的区别。
- 只创建本过程文档；未修改 Python/YAML/shell 算法或训练入口，未运行算法测试。
- 准备阶段已结束，收到用户明确开始指令。

### 实施启动

- 已重查分支与工作区；仅本准备文档为未跟踪文件，无用户未提交的源码改动。
- 当前训练基准为 `run_steer_4096.sh`（1024 prompt / 4096 response，NPU FSDP2，每 micro-batch 同步，KL 系数为 0）；新脚本命名为 `run_steer_quadrant_4096.sh`。
- 不改原服务器脚本，保留其数据、奖励和训练优化；仅新脚本增加模式参数、独立输出目录和必要的源码加载检查。
- 本机现在可找到 uv，但默认全局缓存不可写；后续使用临时缓存/环境或只读依赖加载进行离线验证，不重试历史失效临时文件。

### 核心与训练入口接入

- 新增 `compute_quadrant_steer_weights`，严格按规格保留 signed score、共享 `max(M,0.02)`、Ⅰ指数、Ⅲ常数、Ⅱ/Ⅳ分段曲线；非有限分数置零，显式 FP32 输出和 no_grad。
- 通过 `compute_policy_loss_with_entropy` 按模式分派；旧 helper 未改。三个新参数已加入 `PolicyLossConfig` / actor YAML，校验有限值与合法范围；quadrant + linear=true 明确报错。
- 未修改 worker、loss bridge、旧熵来源、归约或梯度累积实现。
- 新建 `run_steer_quadrant_4096.sh`，增加四象限参数、独立输出目录、默认日志文件名，以及所有节点启动 Ray 前的源码/参数检查。原 `run_steer_4096.sh` 未改。
- 修正两处既有 CPU 测试替身：补 engine_config，不修改训练引擎；扩展为覆盖逐 micro-batch 同步开关两种取值及 quadrant 单位权重。
- 新增 `test_steer_quadrant_on_cpu.py`：首轮 63 项通过，已继续加入服务器脚本完整 Hydra 参数和对照差异验证；完整回归运行中。
- 离线环境由 uv 创建于 `/tmp/steer-quadrant-venv`，使用 `/tmp/steer-quadrant-runner.py` 加载只读缓存依赖。实际 torch=2.14.1 CPU、TensorDict=0.10.0、Transformers=5.10.4；未安装包、未改变服务器环境。

### 验证结果与交付

- 最终合并执行完整 CPU 回归：**173 passed、1 deselected**（15.61 秒），四象限专项共 68 项。此前分批结果为 170 项回归和 3 项新增脚本预检查通过。
- 独立标量参考、文档 PyTorch 参考实现、FP32/BF16/FP64、实际 loss/梯度均已核对。文档参考代码直接提取执行，避免只测试实现自身。
- V1 nested/V0 padded 数据、固定/动态累积、模拟 DP=1/2、缓存旧熵、熵正则及 KL 接入均有验证；这些测试不代表真实多卡通信已经验证。
- 新旧脚本通过真实 shell 展开训练命令后比较：除模式和三个新增算法参数外，其余训练参数一致；输出目录与项目名独立。脚本测试仅执行被抽出的命令或预检查片段，未启动 Ray、安装服务器依赖或训练。
- Hydra 全量覆盖、环境变量覆盖及 symmetric 对照切换检查通过。所有节点的源码/参数检查位于 Ray 启动之前。
- Ruff 格式与静态检查、`bash -n` 和 `git diff --check` 通过。
- 未运行真实分布式用例 `distributed_accumulation`：此前本机 Gloo 回环地址解析失败；本轮明确排除。**没有 NPU 实机训练结果，也未验证服务器当前依赖组合与 Qwen3.5-9B 的实际运行。**

文件清单：

| 文件 | 改动 |
| --- | --- |
| `verl-0.9.0/verl/trainer/ppo/core_algos.py` | 新增四象限 helper，策略 loss 按模式分派；旧权重函数不变 |
| `verl-0.9.0/verl/workers/config/actor.py` | 三个字段、quadrant 模式与参数范围校验 |
| `verl-0.9.0/verl/trainer/config/actor/actor.yaml` | 暴露三个参数，保留旧模式默认值 |
| `verl-0.9.0/tests/trainer/ppo/test_steer_quadrant_on_cpu.py` | 四象限、真实 loss 接口、脚本及配置专项测试 |
| `verl-0.9.0/tests/trainer/ppo/test_steer_on_cpu.py` | 既有测试替身补齐 engine_config，覆盖同步开关及 quadrant 单位权重 |
| `verl-0.9.0/tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py` | 同样补齐测试替身，覆盖两种同步取值；无引擎实现改动 |
| `run_steer_quadrant_4096.sh` | 从 4096 基准新建的 NPU 服务器脚本 |
| 本文及 `STEER_ADAPTATION_NOTES.md` 入口提示 | 保存规格、范围、验证和使用方式 |

最终核对：原 `run_steer_4096.sh`、算法规格、worker、FSDP 引擎、loss bridge 未修改；GRPO 优势估计、局部归约、梯度累积缩放及用户的 `grad_sync_every_micro_batch` 优化均保留。未提交、未推送。

复现本机验证（临时环境存在时，从 `verl-0.9.0` 执行）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /tmp/steer-quadrant-venv/bin/python /tmp/steer-quadrant-runner.py -q \
  tests/trainer/ppo/test_steer_quadrant_on_cpu.py \
  tests/trainer/ppo/test_steer_on_cpu.py \
  tests/trainer/ppo/test_core_algos_on_cpu.py \
  tests/trainer/ppo/test_dynamic_policy_losses_on_cpu.py \
  tests/trainer/ppo/test_loss_aggregation_on_cpu.py \
  tests/workers/config/test_actor_config_on_cpu.py \
  tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py \
  -k 'not distributed_accumulation' --tb=short
```

临时 runner 只负责本机缓存依赖选择，不属于训练代码，不应复制到服务器。服务器可在自身可用环境内直接对上述路径运行 pytest。

## 服务器脚本使用

把本分支的适配源码及新脚本同步到所有训练节点，使用与原脚本相同的平台环境变量与挂载路径：

```bash
bash run_steer_quadrant_4096.sh
```

默认选择 `quadrant`、`linear=false`、`token_weight_min=0.7`、`d=0.2`、`gamma=0.1`、`w3=1.0`。`token_weight_max=1.0` 是沿用的旧配置字段，不截断Ⅱ/Ⅳ权重。

需要覆盖新参数时：

```bash
STEER_ENTROPY_BAND_HALF_WIDTH=0.20 \
STEER_ENTROPY_WEIGHT_STRENGTH=0.10 \
STEER_QUADRANT_3_WEIGHT=1.00 \
bash run_steer_quadrant_4096.sh
```

使用同一脚本切回原 symmetric 指数 STEER 对照：

```bash
STEER_ENTROPY_CONTROL_MODE=symmetric bash run_steer_quadrant_4096.sh
```

脚本从 `/cache/algorithm/verl-0.9.0` 或 `/cache/algorithm` 查找目标源码，设置 `PYTHONPATH` 后输出实际导入文件路径；其他部署布局可用 `VERL_CODE_DIR` 显式指定。保留 Qwen3.5-9B、NPU FSDP2、1024/4096 长度、5120 token 上限和原训练优化。输出目录默认使用 `steer_quadrant`，不会沿用旧对照的默认输出目录。
