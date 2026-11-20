# STEER 平方助推适配过程记录

## 当前状态

- 分支：`steer_quadrant_algorithm_v2`；用户已授权修改代码并从 `run_steer_quadrant_4096.sh` 新建服务器脚本。
- 状态：**算法、配置、服务器脚本及本地验证已完成；NPU实机训练尚未验证。** 开始时仅用户新规格 `STEER_QUADRANT_RECIPROCAL_BOOST.md` 未跟踪；没有其他未提交改动。
- 权威规格：[STEER_QUADRANT_RECIPROCAL_BOOST.md](STEER_QUADRANT_RECIPROCAL_BOOST.md)。文件名含 reciprocal，实际公式为平方曲线。
- 保留旧 `quadrant` 模式和原脚本，新增 `quadrant_quadratic`；不提交、不推送。

## 必须保留的规则

1. 有符号分数、输入精度、概率 clamp、非有限 score 置零沿用基线。
2. Ⅰ保持指数权重及运算顺序，分母仍为本 rank 当前 microbatch 全部有效 token 的 `D=max(max(abs(x)),0.02)`，包含其他象限的极值。
3. Ⅱ和Ⅳ合并，仅对有效 `x>0` 取 min/max；`u=(x-min)/(max-min)`。一个正分数或全部相等时 `u=1`，没有正分数时跳过。
4. 新曲线 `1.001+(gamma-0.001)*(1-u)^2`，默认 gamma=0.20；最小正分数权重 1.20，最大 1.001，中点 1.05075。没有原 0.3/0.7 中性区或高分抑制。
5. 新字段 `entropy_boost_strength`：有限且 >=0.001，不增加文档未规定的上限；gamma=0.001 时所有正分数仍为 1.001，不能声称退化为全单位权重。
6. Ⅲ独立常数 `quadrant_3_weight`，默认1，有限正数。零分有效位置为1、padding为0、helper全空mask返回0。
7. no_grad，输出FP32；只加权裁剪后的策略项。维持旧熵来源、old_log_prob数据流、GRPO优势、归约、累积、KL/熵项及用户NPU训练优化。

## 执行与验收计划

- [x] 新 helper、loss 分派、配置/YAML 字段与模式校验。
- [x] 文档数值表、三token例、两种归一化、单正分数/相等/空、mask/非有限值。
- [x] Ⅰ与 symmetric 指数逐项对齐（floor上下），Ⅲ独立，FP32/BF16/FP64文档参考一致。
- [x] 独立标量参考的loss及梯度、四种归约、V0/V1缓存熵及固定/动态累积。
- [x] 新脚本 `run_steer_quadrant_quadratic_4096.sh`；继承原参数，改模式/助推字段及独立输出目录；所有节点Ray前校验实际新字段。
- [x] 旧模式回归、Hydra实际参数展开、新脚本预检查、Ruff/Bash/diff检查。
- [x] 更新最终结果与使用方法，明确CPU和NPU验证范围。

## 环境与注意事项

- 已读 `verl-0.9.0/AGENTS.md`；复用此前uv离线临时环境及只读缓存 runner，不安装依赖、不改服务器环境。
- 历史测试结果不代替本轮结果。真实多进程FSDP测试此前受本机Gloo回环地址解析限制，本机无NPU。
- 完整loss的空mask min/max是既有边界；本次只保证权重helper的空mask规则。

## 实施记录

- 已增加 `compute_quadratic_steer_weights`，在同一 `entropy_control` loss 内独立按 `quadrant_quadratic` 模式分派；旧两个helper未修改。
- 新增dataclass/YAML字段 `entropy_boost_strength=0.20`；两个四象限模式共享 `linear=false` 和w3校验，旧分段字段只在旧quadrant模式校验。
- 为NPU使用常见的独立 `min()` / `max()` 规约，代替参考代码中的 `aminmax`，数值与参考代码一致；未改变统计范围或精度。
- 新脚本 `run_steer_quadrant_quadratic_4096.sh` 已创建且可执行，原脚本未改；保留原NPU/Qwen3.5-9B/4096参数、预检查、Ray拓扑及优化，仅替换新模式/参数并独立默认输出目录。
- 复用现有四象限测试文件，增加独立标量参考的平方公式，并扩展实际loss bridge的V0/V1、四种归约、固定/动态累积、模拟DP测试到新模式。
- 首轮专项 **143 passed**；随后补充Ⅰ在其他象限极值决定分母时的精确对照，最终回归结果见下节。

## 最终验证结果

- 完整回归 **250 passed、1 deselected、12 warnings**，耗时16.36秒。其中四象限测试文件共145项（原68项加77项新模式/集成用例）。
- 文档数值表与第三象限独立参数：`test_quadratic_document_examples_and_third_quadrant`。
- 全集合分母与正分数min-max互相独立、mask排除：`test_quadratic_two_normalizations_and_mask`；Ⅰ精确复用旧指数：`test_quadrant_one_matches_baseline_exactly_with_shared_denominator`（含两个模式、floor上下、Ⅳ提供最大值）。
- 单正分数、全相等、并列极值、无正分数：`test_quadratic_equal_single_and_no_positive`。
- 参数0.001/0.2/0.8/2.0与独立标量参考对齐，无擅自增加上限：`test_quadratic_independent_reference_and_parameter_range`；非法/非有限参数被配置拒绝。
- 从用户规格直接提取参考函数执行，FP32/BF16/FP64逐项相等且no_grad：`test_quadratic_document_reference_and_no_grad`。参考代码的可选floor在生产代码中固定为用户确认的0.02。
- 零分、非有限分数、空mask和三token符号/loss算例有专门测试。
- 四种归约下的clipped/dual-clip loss及梯度与独立参考一致；最小合法助推参数仍保留1.001，不将其误认作单位权重。
- 实际 `ppo_loss`：V1 nested、V0 padded、固定/动态microbatch、模拟DP=1/2、缓存旧熵、KL/熵正则及response horizon均通过；自定义助推0.37和w3=0.95确实传入权重。
- 新脚本真实shell展开的训练参数与旧脚本比较，仅模式、移除的两个旧曲线字段、新助推字段有差异；项目名/输出目录在比较中统一，实际默认目录独立。
- 新脚本完整Hydra→dataclass→registry→loss、环境覆盖和symmetric对照通过；抽取Python预检查验证有效/无效配置及Ray前执行顺序。测试未运行完整服务器脚本、Ray、安装命令或训练。
- Ruff检查、格式检查、`bash -n`、可执行权限与`git diff --check`全部通过。
- 对比HEAD的AST确认 `compute_token_weights`、`compute_quadrant_steer_weights`、`agg_steer_loss` 不变；原两个服务器脚本、worker、loss bridge及FSDP引擎无diff。
- 未运行的1项是 `distributed_accumulation` 真实通信测试（本机历史Gloo回环地址限制）。本轮无NPU，不声称已验证服务器依赖、Qwen3.5-9B实际前向、真实跨卡通信或训练效果。

本机复现命令，从 `verl-0.9.0` 执行：

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

该临时runner只负责加载本机只读缓存依赖，不是服务器训练所需文件；服务器可在自身环境直接运行pytest。

## 交付文件与使用方法

| 文件 | 用途 |
| --- | --- |
| `verl-0.9.0/verl/trainer/ppo/core_algos.py` | 新平方helper和loss模式分派 |
| `verl-0.9.0/verl/workers/config/actor.py` | 新字段、模式、参数校验 |
| `verl-0.9.0/verl/trainer/config/actor/actor.yaml` | 暴露新字段，保留旧默认模式 |
| `verl-0.9.0/tests/trainer/ppo/test_steer_quadrant_on_cpu.py` | 新模式及原模式对照/集成测试 |
| `run_steer_quadrant_quadratic_4096.sh` | 基于原四象限脚本的新训练入口 |
| 本文及旧过程记录的入口提示 | 避免把旧分段曲线与新平方助推混淆 |

将本分支适配源码部署至所有训练节点，保留原平台环境变量与数据/模型挂载后运行：

```bash
bash run_steer_quadrant_quadratic_4096.sh
```

脚本默认模式与参数：

```yaml
algorithm.adv_estimator: grpo
actor_rollout_ref.actor.policy_loss:
  loss_mode: entropy_control
  entropy_control_mode: quadrant_quadratic
  linear: false
  token_weight_min: 0.70
  entropy_boost_strength: 0.20
  quadrant_3_weight: 1.00
```

调整新助推强度或第三象限：

```bash
STEER_ENTROPY_BOOST_STRENGTH=0.30 STEER_QUADRANT_3_WEIGHT=0.95 \
  bash run_steer_quadrant_quadratic_4096.sh
```

旧分段四象限对照仍用 `bash run_steer_quadrant_4096.sh`；原symmetric指数对照可在新脚本使用 `STEER_ENTROPY_CONTROL_MODE=symmetric`。新脚本不转发旧分段曲线两个参数，调整旧曲线请用旧脚本。

新脚本保留NPU FSDP2、Qwen3.5-9B、1024/4096长度、5120 token上限、batch128/mini32/rollout8、每microbatch梯度同步、offload/prefetch设置、缓存旧熵和原KL配置。`token_weight_max=1.0`不限制新Ⅱ/Ⅳ的助推权重。日志、TensorBoard、rollout与checkpoint默认路径使用 `steer_quadrant_quadratic`。

源码查找和所有节点预检查继承原脚本；其他部署布局可显式设 `VERL_CODE_DIR`。本轮没有提交或推送。
