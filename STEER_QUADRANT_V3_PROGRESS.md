# STEER Quadrant V3 实现记录

## 目标与依据

按 `STEER_QUADRANT_V3.md` 适配 `verl-0.9.0`，并从 `run_steer_4096.sh` 派生 `run_steer_quadrant_v3_4096.sh`。当前分支为 `steer_quadrant_v3`。

## 已核对的接口决定

- 原始参照为 `STEER-main`。Ⅰ、Ⅲ采用其 symmetric exponential 映射，分母为当前 DP rank 的当前 actor microbatch 内所有有效回答 token 的最大绝对分数与 `0.02` 的较大值。
- 新模式为 `policy_loss.entropy_control_mode=quadrant_v3`，Ⅱ参数默认 `q2_boost_strength=0.20`、`q2_boost_scale=0.02`，指数下限由配方的 `token_weight_min` 提供。V3 模式直接采用文档的固定映射，原 `linear` 和 `token_weight_max` 参数供原 STEER 模式使用。
- 读取现有缓存旧熵，当前概率来自 actor 前向。参考函数保留输入精度运算，输出权重为 FP32。
- PPO 裁剪后的逐 token loss 乘 detached 权重，采用已有 STEER 局部 loss 聚合与梯度累积。
- 活跃判定按现有 maximum/minimum/clamp 的真实导数，覆盖 dual-clip、数值截断及相等边界。effective boost 包含局部 loss 聚合和梯度累积系数。
- 设备端累计日志窗口的计数与加权和，日志步一次 DP SUM 后批量转 CPU。默认每训练步记录，支持配置间隔及末步刷新。
- V3 将 `steer/*` 标量加入 TensorBoard，原有训练标量继续写入。
- 适配现有同步 FSDP/FSDP2 的 V1 与 V0 缓存格式。

## 实现与核对清单

- [x] 配置字段、有效值检查、Hydra 路由。
- [x] 权重 helper 与 actor 策略损失。
- [x] 精确活跃判定与含聚合系数的统计，已完成自动求导对照。
- [x] worker 日志累加、DP SUM、日志窗口刷新、trainer 标签传递。
- [x] TensorBoard 保留原指标并加入 STEER 指标。
- [x] 原代码Ⅰ/Ⅲ对照、Ⅱ算例、分界与 padding、裁剪边界梯度测试。
- [x] 变长 microbatch、模拟 DP 汇总、缓存熵接口、旧适配回归。
- [x] 新 4096 启动脚本、语法及参数检查。
- [x] 完成审查与环境限制记录。

## 当前进度

2026-11-23 已实现配置、数学核心、设备累加器、worker 与 V0/V1 trainer 接入、TensorBoard 指标追加，并交付 `run_steer_quadrant_v3_4096.sh`。

CPU 环境由 uv 在 `/tmp/steer-v3-venv` 创建，Python 3.12.14，以 `.pth` 加载本机只读缓存依赖。缓存的 pydantic/core 版本冲突已通过选取匹配版本解决。PyTorch 为 2.14.1，TensorDict 为 0.10.0。

自动求导对照发现本机 PyTorch clamp 在端点的导数为零。为跟随部署版本的实际算子语义，日志代码对每个 device/dtype 用两个独立标量执行一次端点导数标定，并缓存设备张量。逐 microbatch 使用该结果和原 PPO 分支张量计算统计。该标定与模型计算图独立。

旧 STEER 及 FSDP 同步测试的 FSDPEngine 替身缺少后续引入的 `engine_config.grad_sync_every_micro_batch` 字段，已补齐测试替身。第一轮测试暴露的新测试配置冻结字段写入和 Hydra 必需参数问题已修正。

首轮完整验证为 138 项通过，含 42 项 V3、56 项旧 STEER、40 项原算法与配置回归。TensorBoard 测试实际读回 event 文件。原先 V3 专属的标签过滤已移除，当前回归断言同时读回 `steer/q2_active_count`、`actor/pg_loss` 和 `val/accuracy`。DP 窗口测试覆盖分子分母求和和单次 SUM 调用，真实跨进程归约待检查。

## 最终验证

本轮本地可运行集合为 **151 项通过、2 项跨进程测试未纳入该通过集合**。其中 V3 单进程 45 项、旧 STEER 及新增 V3 引擎分批反向对照 57 项、原算法和 actor 配置 40 项、Tracking 回归 4 项、FSDP 单进程回归 5 项。

从 `verl-0.9.0` 执行的测试命令如下，当前环境用 `/tmp/steer-v3-venv/bin/python` 代替 `python`，并设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。

```bash
python -m pytest -q \
  tests/trainer/ppo/test_steer_v3_on_cpu.py \
  tests/trainer/ppo/test_steer_on_cpu.py \
  tests/trainer/ppo/test_core_algos_on_cpu.py \
  tests/trainer/ppo/test_dynamic_policy_losses_on_cpu.py \
  tests/trainer/ppo/test_loss_aggregation_on_cpu.py \
  tests/workers/config/test_actor_config_on_cpu.py \
  tests/utils/test_tracking_on_cpu.py \
  tests/workers/test_fsdp_gradient_accumulation_sync_on_cpu.py \
  -k 'not distributed'
```

覆盖的核对项包括以下内容。

- 直接从 `STEER-main` 独立提取指数函数，Ⅰ、Ⅲ逐位对照，覆盖分母上下界。
- Ⅱ三组权重、跨回答原始分数独立性、关闭助推、FP32/FP64/BF16、零优势、概率分界、非有限分数、padding 和空 helper 输入。
- 四种 loss 聚合下，裁剪内外与相等边界的活跃计数和梯度系数均与实际 autograd 比较，包含正优势 PPO、负优势 dual-clip 和 log-ratio 数值截断。
- V0 padded/V1 nested 缓存熵，经真实 `ppo_loss`、静态/动态累积和变长 microbatch 的 effective boost 与实际累积梯度对照。
- 原 FSDP 引擎的 microbatch 拆分及 backward 循环对 V3 执行验证，模型前向和 collective 使用 CPU 替身。
- 日志算例拆到不同 microbatch 和模拟 DP rank，计数、助推、梯度加权和、裁剪、半助推占比一致。窗口间隔、末步刷新、分母为零跳过、gamma 为零跳过半助推比例均通过。
- V0/V1 trainer 方法实际执行，检查日志步元数据和顶层 `steer/*` 标签。TensorBoard event 文件读回校验原有 `actor/*`、`val/*` 与新增 `steer/*` 标签同时写入。
- 修改的 11 个 Python 文件通过 Ruff lint 和格式检查，`git diff --check` 通过。

新脚本通过 `bash -n`。用隔离的命令捕获器展开脚本变量后，将全部 **98 个训练参数** 交给真实 Hydra compose、Actor dataclass 和 `validate_config` 检查，结果通过。已核对 response horizon 为 4096、microbatch 为 1、模式为 quadrant_v3、下限为 0.7、助推参数为 0.20/0.02。脚本中的 V3 源码加载检查也单独执行通过。

## 脚本入口和参数

```bash
bash run_steer_quadrant_v3_4096.sh
```

默认 `STEER_Q2_BOOST_STRENGTH=0.20`、`STEER_Q2_BOOST_SCALE=0.02`、`STEER_LOG_INTERVAL=1`、`STEER_TOKEN_WEIGHT_MIN=0.7`。实验名、TensorBoard、rollout、checkpoint 和日志目录采用 `steer_quadrant_v3`。脚本在每个节点启动 Ray 前解析 `VERL_CODE_DIR`、设置 `PYTHONPATH` 并检查 V3 接口。

## 环境验证限制

真实双进程 FSDP 测试在沙箱中失败于 Gloo 的本地 socket 绑定，错误为 `uv_bind: operation not permitted`。使用 `GLOO_SOCKET_IFNAME=lo0` 已排除接口选择问题。申请在沙箱外运行 V3 双进程日志算例与 FSDP 测试时，自动审批服务返回 `Upstream request failed`，审批未能完成，沙箱外命令未执行。

新增 `test_distributed_statistics_match_document_example` 保留了真实两进程 DP SUM 算例，运行需要允许本机进程间通信的环境。当前跨卡证据为模拟 DP 测试和 collective 调用核对。

本机没有 NPU/GPU，完整 Ray/TransferQueue、Qwen3.5-9B 训练与 NPU/HCCL 执行尚待部署环境验证。日志累加器为运行时内存状态，恢复 checkpoint 后从新的日志窗口开始累计。
