# STEER Quadrant V7 实现记录

## 范围与约定

在 `steer_quadrant_v7` 的 `516dae0` 上实现 `STEER_QUADRANT_V7.md`，从 `run_steer_quadrant_v6_datav8.sh` 派生 V7 脚本。

- 用本地 actor microbatch 的全部有效且有限的旧策略熵取排名阈值，再应用严格熵下限及象限选择。排名阈值包含并列项。
- 使用完整回答掩码计算原 STEER 权重及分母。入选权重补回 `rho * (1 - weight)`。空象限列表关闭恢复，恢复比例为零时权重等于原 STEER，仍可记录入选统计。
- V5/V6 的Ⅱ象限加权由 `q2_high_entropy_ratio > 0` 启用，与 V7 恢复互斥。V7 脚本显式将旧比例及助推强度均设为零。
- 保留 loss 裁剪、原训练指标和旧配方的 STEER 日志。V7 的Ⅱ象限恢复继续填充原放行指标，新增各象限恢复统计，与原统计共用一次 FP32 DP SUM。

## 进度

已核对基线、方案文档和旧策略熵缓存、actor loss、配置与指标路径。V7 权重路径、Hydra 配置、FP32 指标汇总及 V7 datav8 脚本已落地。

## 验证结果

- 206 项单进程 CPU 回归通过。V7 覆盖排名阈值和绝对门槛的边界、同熵、padding、非有限熵、原 STEER 分母、恢复比例的两个端点、与原 loss 和梯度对照、PPO 裁剪及真实 TensorBoard 事件回读。原 V5/V6 的选择、权重及日志回归也通过。
- 5 项真实双进程 Gloo 测试通过，覆盖 V4–V7 指标和 FSDP 梯度同步。V7 的 33 项 FP32 统计在无入选的 microbatch 仍保持向量形状。跨卡测试用两个 rank 上不同的活跃入选数和权重增量核对按 token 汇总。
- 核对 V7 默认、全象限、仅Ⅱ象限、原 STEER 对照和单步验收五种脚本配置。每组 103 个 Hydra override 均通过 Bash 展开、Hydra compose、actor 配置实例化和训练配置校验。可选的 `TOTAL_TRAINING_STEPS` 默认解析为 `null`，常规运行的步数仍由训练轮数推导。源码预检、Bash 语法、Ruff lint/format 和 diff 检查通过。
- 当前主机为 macOS ARM64，缺少 Ascend 设备和 `torch_npu`。方案要求的首次 NPU actor 更新和 HCCL 日志归约仍需在集群验证。

## 启动

默认 `[1,2]`，熵排名比例 `0.20`，严格条件熵下限 `0.5`，恢复比例 `0.25`。切换象限配置时使用独立实验名和输出目录。

```bash
bash run_steer_quadrant_v7_datav8.sh
EXPERIMENT_NAME=steer_v7_all STEER_RECOVERY_QUADRANTS='[1,2,3,4]' bash run_steer_quadrant_v7_datav8.sh
EXPERIMENT_NAME=steer_v7_q2 STEER_RECOVERY_QUADRANTS='[2]' bash run_steer_quadrant_v7_datav8.sh
EXPERIMENT_NAME=steer_v7_control STEER_RECOVERY_QUADRANTS='[]' bash run_steer_quadrant_v7_datav8.sh
```

## Ascend 单步验收

在目标集群用新的实验名执行一次完整训练 step，沿用 datav8 的模型、数据和并行参数。以下命令跳过保存和评估。

```bash
EXPERIMENT_NAME=steer_v7_hccl_smoke TOTAL_TRAINING_STEPS=1 SAVE_FREQ=-1 VAL_FREQ=-1 \
  bash run_steer_quadrant_v7_datav8.sh
```

以 step 1 的 `actor/pg_loss`、`steer/valid_count` 和分母非零的象限恢复指标为验收证据。V1 trainer 在 actor 更新和指标归约返回后才写入这些指标。单步运行用于验证执行链路。
