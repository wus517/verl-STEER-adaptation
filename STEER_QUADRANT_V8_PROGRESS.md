# STEER Quadrant V8 实施记录

## 范围与约定

在 `steer_quadrant_v8` 的 `27b4430` 上实施 `STEER_QUADRANT_V8.md`。全 padding 的 loss 和权重日志沿用原处理。

- 复用 V7 的熵排名、严格绝对门槛和象限判定。已启用象限中的低熵位置共用映射下限，各象限的高熵位置分别配置映射下限与恢复率。
- 在原 STEER 权重函数内用完整回答掩码计算分数与分母，按位置选择指数映射下限。未变更下限的位置使用原标量映射结果。最终权重乘原 PPO 策略损失，权重计算在 `no_grad` 内。
- 既有 STEER 指标沿用 V7 的分母。V8 的恢复增量以高熵映射结果为基准，高低熵组均值追加到既有 FP32 DP SUM。
- 根据最新指令从 `run_steer_quadrant_v7_datav8_res8192_overlong.sh` 派生新脚本。该文件虽带 `overlong` 后缀，实际使用 `naive` 评分且未配置 DAPO 长度惩罚。训练、评估与评分配置按该文件继承，V8 默认象限按方案取 `[1,2]`。

## 进度

已接入逐位置映射下限、V8 loss 分支、配置和日志，并派生 `run_steer_quadrant_v8_datav8_res8192_overlong.sh`。

## 验证

- 227 项单进程 CPU 回归通过，覆盖 V8 两个熵边界、并列熵、非有限熵、文档权重算例、不同象限参数、完整分母、V7 与原 STEER 对照、PPO 梯度及真实 TensorBoard 事件回读。全 padding 用例检查 V8 权重函数的掩码与原返回值，loss 空权重日志照旧。旧配方回归通过。
- 6 项真实双进程 Gloo 测试通过。V8 两个 rank 的低熵组计数和平均权重不同，其中一个 rank 的高熵组为空。测试核对 49 项 FP32 向量及跨 rank 的按 token 加权均值。
- V8 默认、全象限、Ⅱ和Ⅳ自定义、原 STEER 与 V7 对照五种脚本配置均通过 Bash 参数展开、Hydra 解析和 actor 配置校验，每组 106 项 override。新旧脚本用相同象限列表时，已有 override 的差异仅为实验名、项目名及输出路径。Bash 语法、V8 源码预检、Ruff 和 diff 检查通过。
- 本次验收范围为本地回归与启动配置校验。按用户确认，Ascend 集群首步验收不属于本次交付范围。

## 启动

默认使用 8192 token 回答上限、Ⅰ和Ⅱ象限、高熵下限 `0.70`、低熵下限 `0.50` 和高熵恢复率 `0.35`。

```bash
bash run_steer_quadrant_v8_datav8_res8192_overlong.sh
EXPERIMENT_NAME=steer_v8_all STEER_RECOVERY_QUADRANTS='[1,2,3,4]' bash run_steer_quadrant_v8_datav8_res8192_overlong.sh
```

设置 `STEER_RECOVERY_QUADRANTS='[]'` 得到原 STEER 对照。设置 `STEER_ENTROPY_WEIGHTING=false` 并保留非空象限列表时，使用 V7 的标量恢复率。
