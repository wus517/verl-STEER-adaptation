# STEER Quadrant V5 实现记录

## 范围与决定

当前分支 `steer_quadrant_v5` 从 V4 的 `d384415` 创建。以 `STEER_QUADRANT_V5.md` 为规格，从 `run_steer_datav8.sh` 派生 `run_steer_quadrant_v5_datav8.sh`。

- `q2_high_entropy_ratio` 默认 `0.0`，V5 脚本设为 `0.20`。放行可独立于 `steer_quadrant_logging` 启用。
- 在同一 actor microbatch 内先用完整 `response_mask` 计算原权重。用全部有效回答 token 的有限熵确定前 20% 的分界值，再将达到该值的Ⅱ象限位置权重改为 `1.0`。阈值并列的Ⅱ象限位置均放行。
- 策略损失继续使用现有裁剪、dual-clip、原回答掩码及损失聚合。用 V4 活跃判定生成三项新增日志，V4 权重指标读取实际乘入 loss 的权重。每次日志刷新沿用一次 DP SUM。
- V4 对照用比例 `0.0`，保留 V4 日志集合。放行比例大于零且开启日志时，追加分母非零的 V5 指标。两种配置均保留原 TensorBoard 指标。

## 实施清单

- [x] 配置校验、Ⅱ象限选择、策略损失接入。
- [x] V4 和 V5 指标合并及 DP 汇总接入。
- [x] 固定输入测试、实际梯度、TensorBoard 事件回读。
- [x] 双进程汇总。
- [x] 新脚本，校验 V4 对照与 V5 的 Hydra 参数。
- [x] HCCL 兼容性修复，指标 DP 归约向量使用 FP32。

## 进度

已核对分支、文档、V4 的权重及日志入口。开始接入配置和数学逻辑。

已接入原策略损失和 V4 统计窗口。配置为非零时保留一份覆盖前权重给新指标，现有权重统计使用覆盖后的权重。日志关闭时仍执行放行。开始用固定样例核对最大分数、裁剪梯度和跨 microbatch 聚合。

156 项单进程 CPU 测试通过，涵盖 V5、V4、原 STEER、PPO、配置、Tracking 和 FSDP。另有 3 项真实双进程 Gloo 测试通过，覆盖 V5 与 V4 的 DP 日志归约和既有 FSDP 梯度同步。V5 的 TensorBoard 用例通过真实 `ppo_loss` 和原写入器生成事件，回读全部标签及其数值。

新脚本两种运行模式均已通过 97 个 Hydra override、actor 配置实例化及训练配置校验。与原 datav8 的参数作逐键对照，差异为两个新功能开关及实验输出位置。已检查源码预检和 Bash 语法。按仓库 pre-commit 生成规则同步四份只读参考 YAML，补齐原有 STEER 字段与 FSDP 已存在的默认值。

## 实验入口

```bash
bash run_steer_quadrant_v5_datav8.sh

EXPERIMENT_NAME=steer_v4_control STEER_Q2_HIGH_ENTROPY_RATIO=0.0 \
  bash run_steer_quadrant_v5_datav8.sh
```

脚本使用独立的 V5 输出路径，并在每个节点启动 Ray 前验证加载的 verl 含 V5 选择函数。可用 `VERL_CODE_DIR` 指定部署的 `verl-0.9.0` 源码目录。正式实验的两组需使用同一初始模型和恢复状态，两组的实验名应各自独立。

## 验证范围

CPU 回归在 `verl-0.9.0` 下执行，Python 为 `/tmp/steer-v3-venv/bin/python`，使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。测试文件为 `tests/trainer/ppo/test_steer{,_v4,_v5}_on_cpu.py`、`tests/trainer/ppo/test_{core_algos,dynamic_policy_losses,loss_aggregation}_on_cpu.py`、`tests/workers/config/test_actor_config_on_cpu.py`、`tests/utils/test_tracking_on_cpu.py` 及 `tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py`。单进程批次用 `-k 'not two_process and not distributed'`，另在允许本地进程通信的环境中执行排除的 3 项。

NPU/HCCL、目标模型加载、真实训练首步及长程曲线尚待集群环境验证。

## HCCL dtype 修复

`steer_metrics.py` 原先把整组计数和权重和提升为 FP64，HCCL 不支持 Double all-reduce。现在权重和及归约向量使用 FP32，与 V3 的 DP 归约 dtype 一致。V4/V5 单进程 STEER 测试 50 项通过，V5 专项测试 19 项通过，并断言统计向量为 FP32。本机双进程 Gloo 用例在进程组初始化阶段因沙箱无法解析或绑定本地接口而失败，未执行到归约。HCCL 集群验证仍待执行。
