# STEER Quadrant V6 实现记录

## 范围

从 `steer_quadrant_v6` 的 `3d76267` 实施 `STEER_QUADRANT_V6.md`，最后从 `run_steer_datav8.sh` 派生 V6 脚本。

- 复用 V5 的本地 microbatch 阈值。全部有效且熵有限的回答位置参与定阈值，达到阈值的Ⅱ象限位置入选。
- 从原 STEER 权重函数的有限性处理前取得原始分数幅度。只在入选且分数有限的集合中按 FP32 做 min–max 归一化。用户确认单项或同分时取 `u=0`，即最大助推 `1+gamma`。
- 未入选位置使用原 STEER 权重。入选但原始分数非有限时用 `1.0`。筛选比例为零时沿用原 STEER，助推强度为零时沿用 V5 的入选权重 `1.0`。
- 策略损失沿用 V5 的裁剪、dual-clip 和归约。训练、V4 和 V5 指标持续写入 TensorBoard，新增 `steer/q2_release_boost_mean` 使用同一个 DP 归约窗口。

## 进度

已核对基线并同步 V6 文档中的单项及同分规则。配置、权重覆盖和日志已接入。现有权重函数仅在 V6 实际启用时额外返回未经有限性清洗的分数幅度，V5 对照沿用原来的权重返回路径。

已基于 `run_steer_datav8.sh` 派生 `run_steer_quadrant_v6_datav8.sh`，默认熵比例和最大助推均为 `0.20`。使用独立的实验输出路径，并复用 V5 的源码定位和节点启动预检。

## 验证结果

- 178 项单进程 CPU 测试通过，覆盖原始分数、平方曲线、单项及同分、空集合、非有限分数、全体有效 token 熵阈值、裁剪梯度和真实 TensorBoard 事件回读。
- 4 项双进程 Gloo 测试通过，覆盖 V4、V5、V6 指标归约和 FSDP 梯度同步。V6 日志用两个 microbatch 共 3 个活跃入选位置验证按 token 加权的助推均值 `(2 × 0.20 + 0.001) / 3`。
- 分别核对 V6 与 V5 对照各 98 个 Hydra 启动参数，并与原 datav8 脚本的训练参数逐键比对。源码预检、Bash 语法、Ruff 检查和 `git diff --check` 均通过。
- NPU/HCCL 实际训练尚未执行。

## HCCL dtype 修复

原统计向量及权重和使用 FP64，HCCL 不支持 Double all-reduce。现将四象限权重和、V5 释放基准权重和、V6 助推和及最终 DP 归约向量统一改为 FP32，并在 V4/V5 测试中断言归约 dtype。修复后 V4、V5、V6 单进程 CPU 测试 73 项通过，3 项双进程 Gloo 归约测试通过。HCCL 集群验证仍待执行。

## 启动

默认 V6 实验使用 `bash run_steer_quadrant_v6_datav8.sh`。V5 对照沿用同一熵比例，使用独立实验名与零助推强度。

```bash
EXPERIMENT_NAME=qwen3_5_9b_steer_quadrant_v5_control \
STEER_Q2_HIGH_ENTROPY_BOOST_STRENGTH=0.0 \
bash run_steer_quadrant_v6_datav8.sh
```
