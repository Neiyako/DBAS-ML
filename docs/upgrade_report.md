# v2.0 升级交付报告（M1–M7）

> 本报告汇总本轮升级的**实际交付**：架构、审计、差异、实测、模型状态、测试、
> 遗留问题与下一轮路线。所有数字来自本机真实运行，未运行的项明确标注 `NOT_RUN`。
>
> 交付时间：2026-11-07 ｜ 目标机器：Apple Silicon Mac（**无 NVIDIA GPU**）

---

## 0. 验收结论速览

| 验收维度 | 结论 | 关键证据 |
|---|---|---|
| 工程（可安装/CLI/配置/异常/测试） | **PASS** | 全量 `pytest` 退出码 0；`cloudwater` 11 组命令真实可用；配置校验拦截非法配置 |
| GPU（CUDA 训练/AMP/显存/回退/续训） | **CONDITIONAL_PASS** | 设备抽象/AMP/显存/累积/续训**代码完成并经 CPU+MPS 实测**；RTX 5060 的 CUDA 实测为 **NOT_RUN（本机无 NVIDIA GPU）** |
| 数据（NetCDF 按需/分块/缓存/单位/划分） | **PASS** | 流式分批、分级缓存（命中/失效/清理实测）、坐标/时间/层质量检查、溯源透传 |
| 模型（MLP/RF/LightGBM/CNN/U-Net/ConvLSTM/多任务） | **PASS** | 8 个模型全部注册；前向/梯度/存取/最小训练测试通过 |
| 科研（真仿可区分/混合/时空泛化/误差/物理约束/不确定性） | **CONDITIONAL_PASS** | 真仿溯源+混合采样+场景分组+共形区间+物理校验器已实现；**跨域泛化实验矩阵未在真实 ERA5 上运行（NOT_RUN）** |

> 之所以 GPU 与科研两项是 CONDITIONAL_PASS 而不是 PASS：
> **本机没有 NVIDIA 显卡、也没有真实 ERA5 数据**。代码路径与等价路径已测，
> 但「RTX 5060 上的实测数字」和「真实数据的科研结论」不能由本机产生，
> 按诚实性约定记为 NOT_RUN，而不是拿 CPU 数字冒充 GPU 结果。

---

## 1. 完整架构图

```
CLI (cloudwater): doctor | gpu info|benchmark | models list | data validate |
                  cache info|clear | inspect | train | resume | evaluate | compare | report | synth
                                    │
        ┌───────────────────────────▼────────────────────────────┐
        │ experiments/  编排 + 生命周期 + 溯源                    │
        │  manager(状态机/唯一ID)  provenance(git/数据指纹/环境)  │
        │  runner(端到端)          tracking(产物/对比/拒绝覆盖)   │
        └───┬──────────────┬──────────────┬──────────────┬───────┘
            │              │              │              │
      ┌─────▼─────┐  ┌─────▼──────┐  ┌────▼───────┐  ┌───▼────────┐
      │ data/     │  │ models/    │  │ training/  │  │ evaluation/│
      │ 缓存/流式 │  │ 8 个模型   │  │ Trainer    │  │ 指标/泛化  │
      │ 质量/溯源 │  │ 统一契约   │  │ RuntimePlan│  │ 共形区间   │
      └─────┬─────┘  └─────┬──────┘  │ Checkpoint │  └───┬────────┘
            │              │         └────┬───────┘      │
      ┌─────▼──────────────▼──────────────▼──────────────▼───────┐
      │ physics/  约束接口·单位·积分·校验器·兼容桥               │
      └─────┬────────────────────────────────────────────────────┘
      ┌─────▼────────────────────────────────────────────────────┐
      │ hardware/  设备探测·精度策略·显存监控·性能剖析           │
      └─────┬────────────────────────────────────────────────────┘
      ┌─────▼────────────────────────────────────────────────────┐
      │ utils/  配置·校验·异常·结构化日志·可复现                  │
      └──────────────────────────────────────────────────────────┘
```

依赖方向严格自上而下；`models/`、`training/` 从不 import `cli`/`experiments`。

---

## 2. 工程审计与升级报告

见 [`docs/engineering_audit.md`](engineering_audit.md)。摘要：

- 审计覆盖 15 个方面，产出 **3 个 BLOCKER + 17 个 MAJOR + 8 个 MINOR** 缺陷（均带 `file:line`）；
- 3 个 BLOCKER：
  1. 真实 NetCDF 无来源契约时被标 `unknown` → 混合采样把训练集清空 → scaler 拟合出 NaN；
  2. 断点续训状态不连续（缺 scheduler/scaler/RNG/历史/早停）；
  3. 显存/内存零约束零观测（无 AMP、无累积、无峰值记录、无 OOM 处理）。
- 三者均已修复，并有测试守护（见 §7）。

---

## 3. 新旧架构差异

| 维度 | v1.0 | v2.0 |
|---|---|---|
| 设备 | `resolve_device` 字符串，仅 CPU/MPS | `hardware/` 抽象：GPU 名/CUDA 构建/显存/算力/多卡/回退原因 |
| 精度 | 无 | fp32/fp16/bf16 + auto（CUDA 优先 bf16，fp16 自动 GradScaler） |
| 训练循环 | 全同步、无累积 | AMP + 梯度累积（含尾部 flush）+ 裁剪 + channels_last + compile + 梯度检查点 |
| 显存 | 不可知 | 峰值显存/RSS 记录、OOM 捕获并给出可执行建议 |
| 检查点 | model+optimizer+epoch | + scheduler/scaler/global_step/RNG/历史/早停/运行时计划 |
| 续训 | epoch 前推，轨迹断裂 | 状态连续（曲线不截断、学习率不重启、随机流不重复） |
| 数据读取 | 整载选区、体积守卫有死代码 | 流式分批（受 batch_samples 约束）+ 分级缓存 + 失效/清理/重建 |
| 数据质量 | 6 项浅检查 | + 时间连续性/经纬度/气压层/缺测分布/物理量程 |
| 溯源 | 只有路径 | 每样本 source/source_origin/dataset/scene_id + 文件指纹 + git commit |
| 真仿融合 | 按标签过滤（会筛空） | `MixedSampler` 四策略 + 场景分组防泄漏 + 不可解析时保持全量 |
| 模型 | 3 个（MLP/RF/LGBM） | 8 个（+CNN/U-Net/ConvLSTM/MultiTask/XGBoost） |
| 物理约束 | 2 个（其一构造即抛），作用在 z-score | 独立 physics 包：单位感知、硬/软分离、4 个可用 + 4 个预留、作用在物理单位 |
| 不确定性 | 只有区间指标，生成是 stub | 分割共形区间 + CRPS + 覆盖率/宽度/CWC + 可靠性报告 |
| 实验管理 | 目录 + metrics.json | 状态机 + 唯一 ID + 失败/中断记录 + 原子建目录 + 拒绝覆盖 |
| 性能基准 | 无 | benchmark 模块：吞吐/三段耗时/峰值显存 + 按实测硬件给建议 |
| CLI | 5 条 | 11 组（doctor/gpu/models/data/cache/resume/compare/benchmark 等） |
| 异常/日志/配置 | ValueError + 文本日志 + 无校验 | 统一异常体系（兼容旧类型）+ JSON 日志 + 配置校验（错误/警告分级） |

---

## 4. GPU 训练实测结果

### 4.1 本机（Apple MPS，无 CUDA）

| 项目 | 结果 |
|---|---|
| 设备探测 | `device_type=mps`，`torch_cuda_version=None`，无 `nvidia-smi` |
| 训练设备 | MPS（`runtime_plan.device=mps`） |
| 精度 | `fp16`（MPS auto 策略） |
| 端到端训练 | 86 epoch（早停于 100 中的第 86），best 恢复，测试 RMSE 7.29e-06 |
| 峰值内存 | 0.336 GiB（8192 样本基准） |
| 吞吐 | **2358 samples/s**（MLP 64 特征，batch 256，2 epoch 平均） |
| 三段耗时 | 数据加载 2.11 s / 计算 4.70 s（数据占比 31%） |
| 峰值显存 | 0.008 GiB（MPS 驱动统计） |

### 4.2 RTX 5060 / CUDA —— `NOT_RUN`

| 项目 | 状态 | 说明 |
|---|---|---|
| RTX 5060 被正确识别 | **NOT_RUN** | 本机无 NVIDIA GPU |
| CUDA 训练真实运行 | **NOT_RUN** | 同上 |
| AMP（bf16/fp16）在 CUDA 上 | **NOT_RUN** | 代码路径与策略已测（CPU bf16 真实训练通过） |
| CUDA 显存峰值记录 | **NOT_RUN** | `MemoryMonitor` CUDA 分支有 `@pytest.mark.gpu` 用例，无卡自动 skip |
| OOM 捕获与建议 | **NOT_RUN** | 分支已实现并带 `pytest.mark.gpu` 守护 |

到货后按 [`docs/gpu_training.md`](gpu_training.md) §7 的清单跑 6 条命令即可补齐这五项。

---

## 5. 各模型实现状态

| 模型 | 状态 | 参数量（默认配置） | 测试覆盖 |
|---|---|---|---|
| `mlp` | ✅ 实现 | 依 hidden_layers | 前向/梯度/存取/训练 |
| `random_forest` | ✅ 实现 | 依 n_estimators | 同上 + 特征重要性 |
| `lightgbm` | ✅ 实现 | 依 n_estimators | 同上（可选依赖） |
| `xgboost` | ✅ 实现 | 依 n_estimators | 同上（**本机未装 xgboost，测试 skip**） |
| `cnn` | ✅ 实现 | `[B,C,H,W]→[B,T]` | 前向/梯度/存取/训练 |
| `unet` | ✅ 实现 | 编码-解码 + skip | 同上 + 尺寸整除校验 |
| `convlstm` | ✅ 实现 | ConvLSTMCell 堆叠 | 同上 + `predict_field` 多步形状 |
| `multitask` | ✅ 实现 | 共享骨干 + 逐任务头 | 同上 + 任务权重生效 |

> 8/8 已注册且可训练；无占位实现残留。

---

## 6. 自动化测试结果

```
$ python -m pytest -q
300 passed, 6 skipped        # 退出码 0
```

- **skipped 全是环境依赖**：CUDA 专属用例（本机无卡）、xgboost 未安装；
- 测试分层：单元（指标/配置/异常/硬件/物理）、集成（数据/模型/训练）、
  端到端（pipeline/CLI）、GPU（自动 skip）；
- `pytest` 标记：`gpu` / `integration` / `pipeline` / `perf` / `slow` 已在 pyproject 声明；
- **真实数据加载测试**：NetCDF fixture 用 xarray 真实写盘 → 真实读取，不是 mock；
- 新增测试文件：`test_hardware.py`、`test_runtime.py`、`test_config_schema.py`、
  `test_training_v2.py`、`test_models_v2.py`、`test_data_v2.py`、`test_physics.py`、
  `test_uncertainty.py`、`test_data_sources.py`。

---

## 7. 未解决问题（诚实清单）

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | **CUDA 实跑数字缺失** | 无法证明 5060 上的真实吞吐/显存 | 在 GPU 机器上跑 §4.2 清单 |
| 2 | **真实 ERA5 未接入** | 科研结论（反演精度）不能给出 | 接入真实数据后跑 A~E 配比与跨域实验 |
| 3 | ~~训练侧尚未流式~~ | **已解决**：`data/stream.py` + `Trainer.fit_stream` + `MLP.fit_stream`，真实 NetCDF 端到端训练通过 | — |
| 4 | ~~physics_report 判 FAIL（负 clwc/ciwc）~~ | **已解决**：新增 `MLP(output_activation=softplus)` 硬约束（`configs/experiments/mlp_nonneg.yaml`），`negative_values` 由 FAIL→PASS；代价 R² 0.976→0.744（显式取舍） | — |
| 5 | ~~region 分组恒 skipped~~ | **已解决**：`configs/training.yaml` 增加 `evaluation.region_boxes`，区域指标已产出 | — |
| 6 | `docs/data_schema.md` / `docs/experiment_protocol.md` / `docs/model_design.md` 仍是 v1.0 文本 | 与 v2.0 细节有出入 | 建议下轮统一重写（本轮已新增 7 篇 v2.0 文档） |
| 7 | mypy 仅配置基线：`mypy src/cloudwater_ml` 当前 47 条告警，**均为 `nn.Module` 动态属性/`Optional` 收窄的误报**（已逐条人工核对，运行时行为由测试覆盖），未做全量收敛 | 类型回归无法自动拦截 | 下轮分批加 `cast`/`Protocol` 收敛并进 CI |
| 8 | 缓存不跨机器迁移 | 换机需重建（只影响速度） | 可接受，文档已说明 |

---

## 8. 后续科研实验路线

1. **接入真实 ERA5**：`clwc/ciwc/cwp` 单位口径确认 → 用 `physics.utils` 校验 →
   在 `data/schema.py` 登记单位 → 开启 `nonnegativity_softplus` 硬约束。
2. **A~E 配比实验**：真实数据到位后跑 `mix.real_fraction` 五档，
   用 `cloudwater compare` 出对比表，报告**实际配比**而非请求配比。
3. **时空泛化**：`time` / `year` / `region` 三种划分 + `evaluation.region_boxes`，
   报告跨年/跨季/跨区域误差，避免随机划分的自相关泄漏。
4. **物理一致性闭环**：CWP 柱积分一致性（需 ρ 廓线或 dp/g）从预留转为启用，
   用 `PhysicsValidator` 做数值一致性报告。
5. **不确定性进论文**：共形区间（验证集校准）给出覆盖率/宽度/CRPS，
   与点预测指标并列报告。
6. **U-Net / ConvLSTM 空间场实验**：先把 NetCDF 网格重采样到固定 H×W，
   再接 `predict_field` 出空间分布图。

---

## 9. 推荐的下一轮开发任务

| 优先级 | 任务 | 理由 |
|---|---|---|
| ~~P0~~ | ~~训练侧 IterableDataset 流式接入~~ | **本轮已完成**（`data/stream.py`、`Trainer.fit_stream`、7 个真实 NetCDF 测试） |
| P0 | 在 RTX 5060 机器完成 §4.2 验收 | 补齐 GPU 实测数字 |
| ~~P1~~ | ~~`evaluation.region_boxes`~~ | **本轮已完成**（配置已补，区域指标已产出） |
| ~~P1~~ | ~~非负硬约束~~ | **本轮已完成**（softplus 输出头 + 持久化 + 3 个测试） |
| P2 | mypy 全量收敛 + pre-commit 钩子 | 防类型回归 |
| P2 | 重写 v1.0 遗留三篇文档 | 文档一致性 |
| P3 | 深度集成（Deep Ensemble）训练编排 | 不确定性方法从 1 种扩到 3 种 |

---

## 10. 边界声明

- 本轮**没有**伪造任何 Benchmark 数字、科研结论或 GPU 实测；
- 合成数据只用于工程链路验证，`source` 恒为 `simulated`；
- 原始科研数据目录 `data/raw` 全程只读（缓存有硬保护，测试断言其不被写入）；
- 已初始化 git 仓库并提交基线，使 `provenance.git` 能真正记录 commit（此前无 `.git`）。
