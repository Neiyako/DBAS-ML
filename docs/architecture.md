# 架构说明 · cloudwater-ml v2.0

> 本文只描述**代码里真实存在**的结构。每条结论都给文件出处。
> 快照：2026-11-07（v2.0 升级后）。

---

## 1. 分层图

```
┌──────────────────────────────────────────────────────────────────────────┐
│ cli.py · typer 应用 `cloudwater`                                          │
│   doctor | gpu info | gpu benchmark | models list | data validate          │
│   inspect | train | resume | evaluate | compare | report | synth          │
└───────────────┬──────────────────────────────────────────────────────────┘
                │  CLI 只做「解析参数 → 调用下层 → 打印」，不写业务逻辑
┌───────────────▼──────────────────────────────────────────────────────────┐
│ experiments/   编排与生命周期                                             │
│   manager.py       状态机 + 唯一实验 ID + 失败/中断记录                   │
│   provenance.py    git commit / 数据指纹 / 环境快照                        │
│   runner.py        配置→数据→质量→划分→mix→scaler→训练→物理校验→指标      │
│   tracking.py      实验目录(原子创建/拒绝覆盖) / 快照 / report.md / 对比   │
│   registry.py      A~E 真实/仿真配比登记（仅配置层）                      │
└──┬────────────┬──────────────┬──────────────┬──────────────┬─────────────┘
   │            │              │              │              │
┌──▼─────────┐ ┌▼──────────┐ ┌▼───────────┐ ┌▼────────────┐ ┌▼────────────┐
│ data/      │ │ models/   │ │ training/  │ │ evaluation/ │ │ benchmark/  │
│ schema     │ │ base(契约)│ │ trainer    │ │ metrics     │ │ 吞吐/瓶颈   │
│ adapters   │ │ registry  │ │ runtime    │ │ generalization│ │ 配置建议   │
│ (netcdf/   │ │ mlp/rf/   │ │ losses     │ │ robustness  │ └─────────────┘
│  array)    │ │  lgbm/    │ │ checkpoint │ │ uncertainty │
│ cache(分级)│ │  cnn/unet/│ │ optimizer  │ │ (共形区间)  │
│ preprocess │ │  convlstm/│ │ scheduler  │ └─────────────┘
│ splitting  │ │  multitask│ │ callbacks  │
│ validation │ │  xgboost  │ └─────────────┘
│ sources    │ └───────────┘
│ (真/仿融合)│
└──┬─────────┘
┌──▼──────────────────────────────────────────────────────────────────────┐
│ physics/   物理约束体系（单位感知、硬/软分离、校验器）                    │
│   base/composite/constraints/units/integrals/transforms/validator/bridge  │
└──┬──────────────────────────────────────────────────────────────────────┘
┌──▼──────────────────────────────────────────────────────────────────────┐
│ hardware/  设备·精度·显存·剖析（CUDA / MPS / CPU）                       │
│   device.py（探测） precision.py（AMP） memory.py（显存） profiling.py   │
└──┬──────────────────────────────────────────────────────────────────────┘
┌──▼──────────────────────────────────────────────────────────────────────┐
│ utils/  config(继承) config_schema(校验) exceptions(异常) logging(结构化)│
│         reproducibility(种子/RNG) device(兼容入口)                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 依赖方向

- **严格自上而下**：`data/` 不 import `models/`；`evaluation/` 不 import `models/`
  （`evaluation/robustness.py` 的 `predict_fn` 由调用方注入）；
  `models/` 与 `training/` **从不** import `cli` 或 `experiments`。
- 唯一有意的反向引用：`training/trainer.py` **延迟 import** `models.base.TrainingResult`，
  注释写明「避免包循环依赖」。
- `physics/bridge.py` 用**延迟 import** 访问 `training.losses`，
  避免 physics → training 的导入期硬依赖。

### 三条工程纪律

1. **torch 与 LightGBM 不同进程共存**（macOS 实测同进程会 `Abort trap: 6`）：
   `models/base.py` 的 `_LAZY_MODULES` 惰性 import 规避；因此
   `experiments/runner.py` **不在模块级 import torch**（物理校验器等 torch 相关代码
   都放在 `if model.is_torch:` 分支内）。
2. **原始数据只读**：数据层没有任何写 `data/raw/` 的路径；缓存目录独立。
3. **失败显式化**：非法配置、未知模型、未实现约束、设备不可用、OOM 都有明确异常，
   不静默降级。

---

## 2. 模块职责表

| 层 | 文件 | 职责 | 关键出口 |
|---|---|---|---|
| hardware | `hardware/device.py` | 运行时探测 GPU/CUDA/MPS/显存/算力 | `detect_hardware`、`resolve_torch_device`、`device_report` |
| hardware | `hardware/precision.py` | fp32/fp16/bf16 策略 + GradScaler | `resolve_precision`、`autocast_context`、`make_grad_scaler` |
| hardware | `hardware/memory.py` | 显存/内存快照与峰值、训练前余量检查 | `MemoryMonitor`、`current_memory` |
| hardware | `hardware/profiling.py` | 三段耗时（数据/传输/计算） | `Profiler`、`sample_gpu_utilization` |
| training | `training/runtime.py` | 配置 → 一次性解析成执行计划 | `resolve_runtime`、`RuntimePlan`、`DataLoaderOptions` |
| training | `training/trainer.py` | 训练循环：AMP/累积/裁剪/channels_last/OOM/计时/检查点 | `Trainer` |
| training | `training/checkpoint.py` | 完整训练状态快照与恢复 | `save_checkpoint`、`load_checkpoint`、`CheckpointManager` |
| training | `training/losses.py` | 监督损失 + 物理约束合成（含物理单位还原） | `get_loss`、`build_constraints`、`CompositeLoss` |
| physics | `physics/base.py` | 约束接口、软/硬分离 | `PhysicsConstraint`、`SoftConstraint`、`HardConstraint` |
| physics | `physics/constraints.py` | 可用约束 + 预留约束（构造即抛） | `build_constraint`、`RESERVED_CONSTRAINTS` |
| physics | `physics/units.py` | 单位解析/比较/换算/口径族 | `parse_unit`、`units_compatible`、`convert` |
| physics | `physics/integrals.py` | CWP 柱积分（kg/kg 与 kg/m³ 两口径） | `cwp_integral`、`water_content_convention` |
| physics | `physics/validator.py` | 物理一致性校验（四态输出） | `PhysicsValidator`、`format_report_markdown` |
| physics | `physics/bridge.py` | 新接口 → 旧 `penalty()` 兼容桥 | `adapt_for_legacy`、`to_legacy_list` |
| data | `data/schema.py` | `SampleBundle`（X/Y/meta/units 透传） | `SampleBundle`、`spatial_shape_guard` |
| data | `data/adapters/netcdf_adapter.py` | 范围筛选→抽样→体积守卫→按需/分批读取 | `NetcdfAdapter` |
| data | `data/cache.py` | 分级缓存（元数据/预处理/索引/内存）+ 失效与版本校验 | `DataCache` |
| data | `data/preprocessing.py` | scaler + 拟合证据 + 非有限值策略 | `fit_scalers`、`Preprocessors` |
| data | `data/splitting.py` | 5 种划分 + 三集互斥强校验 | `split_indices` |
| data | `data/validation.py` | 质量检查（默认只读报告） | `run_quality_checks`、`apply_cleaning` |
| data | `data/sources.py` | 真/仿来源追溯 + 混合采样 + 场景分组 | `build_provenance`、`MixedSampler`、`scene_ids` |
| models | `models/base.py` | `BaseModel` 契约 + 注册表 + 惰性加载 + 存取模板方法 | `build_model`、`load_model`、`register_model` |
| models | `models/{mlp,random_forest,lightgbm,xgboost}.py` | 表格类基线 | `@register_model` |
| models | `models/{cnn,unet,convlstm,multitask}.py` | 深度模型（4D/5D 张量契约） | `@register_model` |
| experiments | `experiments/manager.py` | 生命周期状态机 + 唯一 ID | `ExperimentState`、`transition`、`find_experiment` |
| experiments | `experiments/provenance.py` | git/环境/数据指纹 | `collect_provenance`、`data_fingerprint` |
| experiments | `experiments/runner.py` | 端到端编排 | `run_experiment`、`evaluate_saved_experiment` |
| evaluation | `evaluation/uncertainty.py` | 共形区间 + CRPS + 可靠性 | `estimate_uncertainty`、`conformal_interval`、`crps` |
| evaluation | `evaluation/metrics.py` | 指标唯一实现处（MAE/MSE/RMSE/R²/Bias/NSE/Pearson） | `regression_metrics` |
| benchmark | `benchmark/runner.py` | 吞吐/瓶颈/峰值显存 + 硬件推导建议 | `benchmark_training`、`recommend_config` |
| utils | `utils/exceptions.py` | 统一异常体系（兼容 v1.0 的 ValueError/RuntimeError） | `CloudWaterError` 及其子类 |
| utils | `utils/config_schema.py` | 配置校验（错误 vs 警告分级） | `validate_config`、`describe_config` |

---

## 3. 关键数据流

```
YAML ──load_config──► 展开配置（extends 深合并）
  │
  ├─validate_config────────► 非法配置立刻失败（带问题清单）
  │
  ├─resolve_runtime────────► RuntimePlan{device, precision, dataloader, accum}
  │        │                        （旧键兼容：具体值优先，新段优先）
  │        ▼
  ├─load_dataset───────────► SampleBundle（NetCDF 按需 / 分块 / 缓存）
  │        ▼
  ├─run_quality_checks─────► 质量报告（默认只读）
  │        ▼
  ├─split_indices──────────► train/val/test（互斥强校验，防时空泄漏）
  │        ▼
  ├─MixedSampler/apply_mix► 训练集配比（测试集永不参与；标签不可解析时不筛空）
  │        ▼
  ├─fit_scalers────────────► 只在 train 上拟合（带拟合证据）
  │        ▼
  ├─Trainer.fit────────────► AMP → 累积 → 裁剪 → 检查点（含 RNG/历史/早停状态）
  │        ▼
  ├─PhysicsValidator───────► 物理一致性报告（四态）
  │        ▼
  ├─evaluate_predictions───► 指标（物理单位）+ 分组泛化 + 鲁棒性 + 共形区间
  │        ▼
  └─metrics.json / report.md / state.json（可复现全量记录）
```

---

## 4. 已知限制（写进代码，不是口号）

- **本机无 CUDA**：开发机为 Apple Silicon，CUDA 路径通过 CPU/MPS 等价路径 +
  单元测试覆盖；真实 CUDA 数值结果需在目标机器上实测（见 `docs/gpu_training.md` §7）。
- 训练输入仍是 numpy 数组（`Trainer.fit` 收 `[n, f]` 矩阵），
  **TB 级 NetCDF 的端到端流式训练尚未接通**（数据层已支持分批读取，
  训练侧的 `IterableDataset` 接入是下一步）。
- 物理约束中除非负性与单位一致性外，其余为**预留**（构造即抛）。
- 不确定性目前实现了**分割共形**区间；MC Dropout / 深度集成需要调用方提供采样源。
