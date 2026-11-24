# STEER Quadrant V4 实现记录

## 目标

按 `STEER_QUADRANT_V4.md` 在当前 `steer_quadrant_v4` 分支实现四象限统计。训练权重、策略损失、梯度与原始 STEER 对照完全一致。另基于 `run_steer_datav8.sh` 派生 V4 启动脚本。

## 当前分支和用户已有工作

- 当前分支 `steer_quadrant_v4`，基线提交 `d68646a`。
- 初始状态存在用户修改 `.DS_Store`，以及未跟踪的 `STEER_QUADRANT_V4.md`、`run_steer_datav8.sh`。这些文件不作为实现目标编辑。
- V4 文档引用的 V3 文档、V3 测试和 `steer_metrics.py` 在此分支均不存在。V4 统计根据当前 `core_algos.py` 的 PPO 算子重新实现并用 autograd 校验。

## 固定实现语义

- 新配置 `policy_loss.steer_quadrant_logging` 默认 `false`。启用要求 `loss_mode=entropy_control`、`entropy_control_mode=symmetric`、`linear=false`。
- V4 helper 依同一输入张量生成 mutually exclusive Q1–Q4 和 `other_mask`，然后直接调用原 `compute_token_weights`，逐位复用它返回的 weights。分类只记录指标，不反馈权重或 loss。
- 活跃位置按照未加权原 PPO/dual-clip 策略项相对当前 log-prob 的真实 PyTorch 梯度判断，含差值截断、ratio clamp、maximum/minimum 相等边界。裁剪仅用于活跃统计，不参与权重计算。
- 每次 actor 更新累加 microbatch 统计为 GPU float64 标量，更新末尾在 actor DP 组做一次 SUM，转 CPU 后生成 `steer/*` 指标并合入原 actor metrics。TensorBoard adapter 不过滤指标。
- V1 同步 FSDP/FSDP2 为目标训练入口，启动脚本基于用户的 `run_steer_datav8.sh` 派生并隔离 experiment、log、TensorBoard、rollout 与 checkpoint 目录。

## 清单

- [x] 配置字段、校验与 Hydra 默认值。
- [x] V4 权重/象限封装和原损失接入。
- [x] 活跃梯度统计、float64 窗口累加、DP 汇总和原指标合并。
- [x] 数学、mask、边界、原始权重/loss/gradient 等价测试。
- [x] V0/V1 trainer 指标命名路径测试，原 TensorBoard 指标保留测试。
- [x] V4 脚本、语法、Hydra 参数及源加载检查。
- [x] Ruff、diff 检查、完整结果与已知限制。

## 进度

2026-11-24 已直接取回 V3 分支 `steer_metrics.py` 和 `test_steer_v3_on_cpu.py` 作为活跃梯度/分布式统计/日志测试基准。V4 重用并扩展其 clamp 边界自动求导检测思路，将 q2 专用统计改为四象限与 other 的 counts/weight sums。

已实现配置、V4 权重/象限适配、float64 统计累加、actor DP 汇总、V1/V0 trainer key 传递，以及基于 datav8 的 V4 脚本。新脚本已做 `bash -n`。用捕获到的脚本参数执行真实 Hydra compose 和 `validate_config`，96 个训练 override 全部通过，源码预检也通过。

V4 专项及既有 STEER/策略损失/配置/Tracking/FSDP CPU 回归组合测试现为 **138 项通过**。其中含真实双进程 V4 DP 统计归约和既有双进程 FSDP 梯度同步测试。V4 覆盖原权重分母下限、四象限及分界/padding、float32/float64/BF16、原始 loss/gradient 和一次 AdamW 更新对照、裁剪边界 autograd、microbatch/DP 汇总、V0/V1 trainer tag、TensorBoard 指标保留和 Hydra 参数。原 STEER 权重 helper 对 FP64/BF16 输入会在赋值时抛出 dtype mismatch，V4 直接调用原 helper并原样保留此行为。

完整回归命令（在 `verl-0.9.0` 子目录运行）。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/trainer/ppo/test_steer_v4_on_cpu.py \
  tests/trainer/ppo/test_steer_on_cpu.py \
  tests/trainer/ppo/test_core_algos_on_cpu.py \
  tests/trainer/ppo/test_dynamic_policy_losses_on_cpu.py \
  tests/trainer/ppo/test_loss_aggregation_on_cpu.py \
  tests/workers/config/test_actor_config_on_cpu.py \
  tests/utils/test_tracking_on_cpu.py \
  tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py
```

本机命令使用 `/tmp/steer-v3-venv/bin/python`。完整运行在沙箱外获准执行，`138 passed`。沙箱内单独运行 socket 测试曾报 `uv_bind: operation not permitted`，获准后两项真实双进程测试均通过。

脚本 `run_steer_quadrant_v4_datav8.sh` 对照原 datav8 配方，默认 V4 logging 开启、对称指数映射、下界/上界 `0.7/1.0`。实验名、项目名、TensorBoard、rollout、checkpoint 和日志目录均使用独立的 `steer_quadrant_v4` 路径。启动脚本语法通过；捕获实际命令展开后的 96 个训练 override 交给 Hydra、dataclass 和 `validate_config`，全部通过；脚本 V4 源码自检独立通过。

Ruff lint、格式、Python 编译和 `git diff --check` 均通过。NPU/HCCL、目标 Qwen3.5 模型加载和长程训练尚未在本机执行。
