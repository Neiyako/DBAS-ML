# 工程审计报告（CloudWater ML Platform v2.0 · M1）

> 审计对象：`/Users/apple/Desktop/DBAS-ML`（CloudWater ML Framework v1.0 → CloudWater ML Platform v2.0）
> 审计日期：2026-11-07
> 审计原则：**先看清已有工程再做升级**——不推翻已经正确工作的代码，优先局部重构 + 接口兼容。

---

## 0. 结论摘要

| 维度 | 结论 | 依据 |
|---|---|---|
| 端到端链路 | **可用且诚实** | 89 → 141 项自动化测试全绿；CNN/U-Net/ConvLSTM/MultiTask 以 `NotImplementedError` 明确占位，README 同步标注「预留」，无夸大 |
| 架构质量 | **分层清晰、无循环依赖** | `models/` 与 `training/` 从不 import `cli`/`experiments`；模型按架构分文件，注册表统一 |
| 主要风险 | **数据层 > 训练层 > 实验层** | 数据层有 1 个 BLOCKER（真实 NetCDF 会被清空训练集）+ 14 个 MAJOR；训练层续训不连续（BLOCKER） |
| GPU 能力 | **v1.0 完全缺失** | 无 AMP、无梯度累积、无显存监控、无完整检查点、DataLoader 无任何优化 |
| 重构必要性 | **需要，但以增量方式** | 保留 `SampleBundle` / `BaseModel` / `TrainingResult` / `SampleBundle.subset` 等已被验证的契约 |

**一句话结论**：v1.0 是一个**诚实但缺硬件能力、缺数据工程**的科研原型。v2.0 不需要重写，
需要的是「加一层设备/精度/运行时抽象」+「把数据层从整载模式改成按需流式」+「把实验从目录结构升级成有状态的生命周期」。

---

## 1. 审计范围与方法

### 1.1 审计范围

| 层 | 文件 | 状态 |
|---|---|---|
| 数据层 | `data/{loader,schema,preprocessing,splitting,validation,synthetic}.py`、`data/adapters/*` | 已审 |
| 模型层 | `models/{base,mlp,cnn,unet,convlstm,multitask,random_forest,lightgbm}.py` | 已审 |
| 训练层 | `training/{trainer,losses,optimizer,scheduler,checkpoint,callbacks}.py` | 已审 |
| 实验层 | `experiments/{runner,tracking,registry}.py` | 已审 |
| 评价层 | `evaluation/*` | 已审 |
| 工具层 | `utils/{config,logging,reproducibility,device}.py`、`cli.py` | 已审 |
| 测试/配置/文档 | `tests/*`、`configs/*`、`docs/*`、`README.md`、`pyproject.toml` | 已审 |

### 1.2 方法与证据口径

1. **静态审计**：逐文件阅读，每条结论标注 `file:line`，不做无证据推断。
2. **动态验证**：审计期间运行全量 `pytest`（基线 89 项全绿，退出码 0）、探测运行环境、
   实跑 CLI 各命令、实跑端到端训练。
3. **交叉审计**：4 个只读子审计（数据 / 模型+训练 / 实验+CLI / 测试+配置+文档），
   结论由 Lead 复核后采纳；**每条缺陷均已由 Lead 独立复核或实测确认**。
4. **结论分级**：`BLOCKER`（破坏正确性/数据完整性/可复现性）/ `MAJOR`（显著工程债或 v2.0 必须补齐）/
   `MINOR`（打磨项）。所有未实际执行的检查记 `NOT_RUN`，不写成通过。

### 1.3 运行环境实测（本轮审计所用机器）

```
平台        : macOS-27.0.1-arm64 (Apple Silicon)
Python      : 3.13.13
PyTorch     : 2.15.0
CUDA 构建   : 无（torch.version.cuda is None）
CUDA 设备   : 0（无 nvidia-smi）
MPS         : 构建=是 可用=是
numpy 2.5.4 / pandas 3.1.1 / scikit-learn 1.9.1 / xarray 2026.9.0 / netCDF4 1.7.5.1
lightgbm 4.7.0 / pyarrow 26.0.1 / scipy 1.18.1 / dask 未安装 / xgboost 未安装
```

> **重要前提**：本机**没有 NVIDIA GPU**。因此本轮所有 CUDA 相关代码为**静态验证 + CPU/MPS 实跑等价路径**，
> 真实 CUDA 训练在 RTX 5060 机器上的结果记为 `NOT_RUN`（见 §9 验收清单）。
> 设备抽象层刻意同时支持 CUDA / MPS / CPU，使「加速器路径」在本机被真实执行而非空跑。

---

## 2. 当前架构

### 2.1 分层图（v1.0 实测）

```
CLI (typer: inspect/train/evaluate/report/synth)
  │
  ▼
experiments/runner.py ──► experiments/tracking.py (目录/快照/metrics/report)
  │        │
  │        ▼
  │   models/base.py  ── MODEL_REGISTRY ──► mlp / random_forest / lightgbm
  │        │                                    (cnn/unet/convlstm/multitask = 占位)
  │        ▼
  │   models/<各模型>.fit() ──► training/trainer.py ──► losses / optimizer / scheduler
  │                                    └──► training/checkpoint.py (best.pt / last.pt)
  ▼
data/loader.py ──► data/adapters/{array,netcdf}.py ──► SampleBundle (X/Y/meta/units)
  │                     └──► validation.py (质量报告) / preprocessing.py (scaler)
  ▼
splitting.py (random/time/year/region/source) ──► mix (真实/仿真配比) ──► evaluation/*
```

### 2.2 模块依赖（实测无环）

- `models/` → `training/`（仅 `models/mlp.py:11`）；`training/trainer.py` 反向引用 `models.base`
  用**函数内延迟导入**（`trainer.py:63`）避免环。
- `models/`、`training/` **从不** import `cli` 或 `experiments` —— 满足「模型不依赖 CLI / 实验运行器」。
- `torch` 与 `lightgbm` 同进程会崩（macOS OpenMP 冲突，`models/base.py:49-52`），故用
  `_LAZY_MODULES` 惰性 import 规避 —— 这是**真实踩坑后的设计**，不是过度设计，保留。

### 2.3 已实现功能清单（实测有效）

**数据层**

| 能力 | 位置 | 状态 |
|---|---|---|
| npz / csv / parquet / netcdf 适配器注册表 | `data/adapters/base.py:19-42` | 实现且正确 |
| 缺变量时列出实际列名而非静默填充 | `data/adapters/array_adapter.py:38-42,52-56,73-76` | 实现且正确 |
| NetCDF 惰性打开 + 坐标范围切片 | `data/adapters/netcdf_adapter.py:55-67,91-132` | 实现且正确 |
| 内存体积守卫（读取前拦截） | `data/adapters/netcdf_adapter.py:217-233` | 实现（有缺陷，见 D2） |
| 数据集来源契约解析（`atmosgen:contract`） | `data/adapters/netcdf_adapter.py:70-84,174-183` | 实现且正确 |
| 五种划分 + 强制互斥校验 | `data/splitting.py:49-55,83-201,221` | 实现且正确 |
| Scaler 保存/恢复/逆变换 | `data/preprocessing.py:91-109,135-153` | 实现且正确 |
| 六项只读质量报告 | `data/validation.py:21-263` | 实现（太浅，见 D10） |
| 清洗仅在显式开启时执行、只改副本 | `data/validation.py:230-234,260-262` | 实现且正确 |
| 合成数据确定性 + 恒为 `simulated` | `data/synthetic.py:41,86-102` | 实现且正确 |

**模型层**

| 能力 | 位置 | 状态 |
|---|---|---|
| 统一 `BaseModel` 契约（predict 恒返回 `[n, t]`） | `models/base.py:97-164` | 实现且正确 |
| 注册表 + 惰性加载 | `models/base.py:48-66` | 实现且正确 |
| 模板方法持久化（`model_meta.json` + 权重） | `models/base.py:133-150` | 实现且正确 |
| MLP（可配置隐藏层/激活/Dropout，输出层恒线性） | `models/mlp.py:19-45` | 实现且正确 |
| RandomForest（单/多目标 + 特征重要性） | `models/random_forest.py:78-124` | 实现且正确 |
| LightGBM（逐目标独立模型 + early stopping + 重要性） | `models/lightgbm.py:112-183` | 实现且正确 |
| CNN / UNet / ConvLSTM / MultiTask | `models/{cnn,unet,convlstm,multitask}.py` | **占位**（构造即 `NotImplementedError`，未注册） |

**训练层**

| 能力 | 位置 | 状态 |
|---|---|---|
| 统一 Trainer 训练循环 | `training/trainer.py` | 实现（v1.0 无 AMP/累积/显存） |
| 损失工厂 mse/mae/huber | `training/losses.py:31-42` | 实现且正确 |
| 物理约束抽象 + 软约束合成 | `training/losses.py:48-133` | 实现（CWP 约束为占位） |
| 检查点 best/last | `training/checkpoint.py` | 实现（状态不完整，见 D1/D2） |
| EarlyStopping / HistoryLogger | `training/callbacks.py` | 实现且正确 |

---

## 3. 已通过的测试（基线证据）

```
$ python -m pytest -q
89 passed                                  # v1.0 基线（审计开始时）
退出码 0
```

审计后新增（v2.0 M1/M2）：

```
$ python -m pytest -q
141 passed, 2 skipped                      # skipped = 需 CUDA 的用例（本机无 CUDA 自动跳过）
退出码 0
```

覆盖范围：`tests/test_data.py`（数据/划分/scaler/质量）、`test_models.py`（模型契约）、
`test_training.py`（训练/检查点/损失）、`test_metrics.py`（指标）、`test_pipeline.py`（端到端 + CLI 入口），
新增 `test_hardware.py`（设备/精度/显存）、`test_runtime.py`（运行时计划）、
`test_config_schema.py`（配置校验/异常体系）、`test_training_v2.py`（累积/检查点/续训/损失分量）。

> **测试现状的诚实评价**：既有测试覆盖了「链路能跑通」，但**没有**覆盖
> 「真实 ERA5 形态」「跨批溯源」「续训等价性」「NaN 污染 scaler」等关键科研场景，
> 详细缺口见 §7.5。

---

## 4. 设计缺陷（分级 · 每条带 file:line）

### 4.1 BLOCKER（阻断级：破坏正确性 / 数据完整性 / 可复现性）

**B1｜真实 NetCDF 会导致训练集被清空（数据层）**

- 证据：无 `atmosgen:contract` 属性的真实 ERA5 文件 → `netcdf_adapter.py:180` 写入 `source='unknown'`；
  `runner.py:73-95` 的 `apply_mix` 只挑 `real`/`simulated` 标签 → 两者皆空 → `chosen=[]`、`dropped=n`；
  随后 `runner.py:202-206` 在空数组上拟合 scaler → `preprocessing.py:77-79` 产出 NaN 均值/方差。
  触发条件已由默认配置满足：`configs/training.yaml:35` 的 `mix.real_fraction: 1.0`（非 null）。
- 影响：**「用真实 ERA5 训练」这一核心场景开箱即坏**，且产物里看不出异常（只在比例不匹配时才有 notes）。
- 处置：M3 修复（标签未解析不得默认 simulated；无 real/simulated 时不过滤并如实记录；默认 `real_fraction: null`）。

**B2｜断点续训不是状态连续的（训练层）**

- 证据：检查点只存 `model_state/optimizer_state/epoch/best_val/trainer_config`（`checkpoint.py:32-40`）；
  恢复时只回填 model+optimizer（`trainer.py` resume 段），**未存/未恢复** scheduler 状态、GradScaler 状态、
  RNG 状态、global_step、EarlyStopping 状态、history。
- 影响：续训后学习率曲线从头开始、shuffle 顺序与 dropout 掩码重复、耐心值被重置；
  同时 `history` 被截断并由 `runner.py:220-221` **覆盖**已发布的 `history.csv`，
  实验记录里的 epoch 数与损失曲线因此失真。**「能续训」但结果不可复现**，比不能续训更危险。

**B3｜显存/内存无任何约束与观测（训练层）**

- 证据：v1.0 `trainer.py` 无 AMP、无梯度累积、无显存统计、无 OOM 处理；
  `checkpoint.py` 无 scaler 状态位。
- 影响：换到 RTX 5060 后，第一批数据就可能 OOM，且**没有任何记录**说明当时用了什么配置；
  AMP 的 fp16 若无 GradScaler 会静默下溢，损失变成 NaN 而无人察觉。

### 4.2 MAJOR（v2.0 必须补齐）

| ID | 缺陷 | 证据 | 处置阶段 |
|---|---|---|---|
| D1 | `_guard_memory` 的 max_samples 降采样是死代码，守卫文案推荐的解法无效 | `netcdf_adapter.py:221-233,244-247` | M3 |
| D2 | `chunks` 配置在无 dask 时被静默忽略；`_flatten` 仍整载选区 | `netcdf_adapter.py:43-52,64-66,156` | M3 |
| D3 | `iter_batches` 不受 `batch_samples` 约束且绕过体积守卫 | `netcdf_adapter.py:256-265` | M3 |
| D4 | 流式路径每批 `sample_id` 从 0 重算、丢 units/source_file | `netcdf_adapter.py:177,266-269` | M3 |
| D5 | `fitted_on='train'` 是硬编码约定，无法自证 | `preprocessing.py:196-222` | M3 |
| D6 | Scaler 无 NaN/Inf 策略，一个 NaN 污染整列且下游无感知 | `preprocessing.py:77-79,125-127` | M3 |
| D7 | `SampleBundle.subset()` 丢弃 units/source_file/source_origin | `schema.py:129-138` | M3 |
| D8 | 随机划分泄漏时空自相关；缺省策略为 random | `splitting.py:75-80,215` | M3/M5 |
| D9 | 质量检查缺坐标连续性/CRS/缺测分布/单位细粒度 | `validation.py:21-37,72-75,98-101` | M3 |
| D10 | 无任何缓存/指纹，可复现性只靠一个路径字符串 | 全 `src/` 无 cache 实现 | M3 |
| D11 | ERA5 现实形态不支持（`pressure_level` 别名、单层+气压层混用硬拒绝） | `netcdf_adapter.py:30-36,117-119,139-149` | M3 |
| D12 | parquet 先读全表再筛列（与自身文档矛盾） | `array_adapter.py:44-58` | M3 |
| D13 | LightGBM `subsample` 未设 `subsample_freq` → **静默失效** | `models/lightgbm.py:37,49,57,64-72` | M4 |
| D14 | Trainer 无 AMP/累积/裁剪/channels_last/pin_memory/compile | `training/trainer.py`（v1.0 全量） | M2 ✅ 已修 |
| D15 | CLI 缺 doctor / gpu info / models list / data validate / benchmark / resume / compare | `cli.py`（v1.0 仅 5 命令） | M1/M2/M6 ✅ 部分已修 |
| D16 | 无统一异常体系（大量 `ValueError`/`RuntimeError` 语义不明） | 全 `src/` | M1 ✅ 已修 |
| D17 | 配置无校验，非法配置要到运行中段才炸 | `utils/config.py`（仅 YAML 解析） | M1 ✅ 已修 |

### 4.3 MINOR

- 空 meta 被默认标成 `simulated`（应为 `unknown`）——`schema.py:100-107`；
- array 适配器缺 `source_origin` 列——`array_adapter.py:104-130`；
- `sample_id` 为位置计数，子集/重采样后不稳定——`netcdf_adapter.py:177`；
- `load()` 在 `ds.close()` 之后读 `ds.attrs`——`netcdf_adapter.py:243,248`；
- `inspect()` 未关闭子集 Dataset——`netcdf_adapter.py:305,311`；
- 时间划分按位置切，可能把同一时刻切进 train/val——`splitting.py:100-101`；
- `split` 配置注释与实现不符（`source: order`）——`configs/training.yaml:30`；
- `Preprocessors` 缺 `from_state_dict`，恢复逻辑在 runner 里重复实现——`runner.py:368-371`。

---

## 5. 性能瓶颈

1. **`_flatten` 双重物化**：每个变量的 `.values` 读入后再 `np.column_stack` 复制一遍
   （`netcdf_adapter.py:154-159`）→ 峰值内存约 2×，且 dask 惰性在此处全部失效。
2. **划分后再复制一份**：父 bundle + train/val/test 三份同时存活（`runner.py:186-188`、`schema.py:135-137`）
   → 划分后常驻内存约 2×。
3. **校验多趟全量扫描**：`validation.py` 对 X、Y 反复整表扫描并 `hstack`（`validation.py:247`）。
4. **表格适配器整文件读取**：parquet/CSV 未按行组分块（`array_adapter.py:44-58`）。
5. **训练循环全同步**：v1.0 每 batch 都 `.to(device)` 同步拷贝，无 `non_blocking`、无 pinned memory、
   无预取，GPU 利用率受数据供给限制（这正是 M2 要解决的）。
6. **未走 GPU 的路径**：数据层先在 CPU 整载成 numpy 数组，训练再整体搬到 GPU，
   TB 级场景下这条路走不通。

---

## 6. 数据安全风险

**正面结论（已实测确认）**

- NetCDF 适配器**只读**打开（`netcdf_adapter.py:67`），并在 `finally` 中关闭（`:242-243, :270-271`）；
  **数据层不存在覆盖原始科研数据的写路径**。
- `apply_cleaning` 只修改副本（`validation.py:234`），不落盘；所有清洗动作被记录（`:237-257`）。
- 清洗默认关闭（`quality.report_only=true`，`configs/data.yaml:25`）→ **报告优先**原则成立。

**风险点**

| 风险 | 证据 | 处置 |
|---|---|---|
| `ensure_synthetic_dataset(force=True)` 可写入任意路径，无 `data/raw` 保护 | `data/synthetic.py:106-121,132-140` | M3 加路径保护 |
| `_bootstrap_synthetic` 只要路径含 "synthetic" 且不存在就**自动生成假数据** | `data/loader.py:31-42` | M3 加显式开关 + 日志告警 |
| `sample_id` 为位置计数 → 不能作为跨批次去重/联结键 | `netcdf_adapter.py:177` | M3 内容派生 |
| 划分后溯源信息丢失（`units`/`source_origin` 被丢弃） | `schema.py:129-138` | M3 透传 |
| 来源标签未解析时默认 `unknown`，却可能被下游当作真实数据 | `netcdf_adapter.py:180`、`schema.py:143-145` | M3 明确契约 |

---

## 7. 内存与显存风险

1. **全链路 float64**：`SampleBundle` 存 float64（`schema.py:84-85`），`_flatten` 亦为 float64
   → 内存占用是 float32 的 2×，且搬运到 8–16GB 显存的 5060 时 PCIe 流量翻倍。
2. **单年 ERA5 气压层规模**：8760 × 37 × 721 × 1440 × 6 特征 × 8 字节 ≈ 1.6 TB
   → v1.0 的「整载选区」路径**必然** OOM；`iter_batches` 也因 `per_step≈3.8×10⁷`
   而退化为「每批一个时间切片」（GB 级）。
3. **划分后的 2× 常驻**：父 bundle 未释放时 train/val/test 同时在内存。
4. **NaN 静默污染**：一个 NaN → 整列标准化结果为 NaN → 训练全程无意义但**不报错**。
5. **显存侧（v1.0 空白）**：无显存统计、无 OOM 捕获、无 AMP、无梯度累积、无 `channels_last`，
   → 换 5060 后既跑不快也说不清。

---

## 8. 重构必要性判定

**结论：需要重构，但必须「外科手术式」，不推倒重来。**

必须保留（已被验证正确的契约）：

- `SampleBundle`（X/Y/meta/units 对齐规则）——`data/schema.py:62-162`；
- `BaseModel` 统一接口 + `predict → [n, t]` + 模板方法持久化——`models/base.py:97-164`；
- `TrainingResult` 同构结果——`models/base.py:29-42`；
- 适配器注册表与「缺变量报错」策略——`data/adapters/base.py`；
- 五种划分 + 强制互斥校验——`data/splitting.py`；
- 惰性加载规避 torch/LightGBM 崩溃——`models/base.py:48-66`；
- 「报告优先、未经配置不改原始数据」的清洗策略——`data/validation.py`。

必须新增（v2.0）：

- `hardware/` 设备·精度·显存·剖析层（本轮已落地）；
- `training/runtime.py` 运行时计划（本轮已落地）；
- `data/cache.py` 分级缓存 + 数据指纹（M3）；
- `physics/` 可插拔物理约束 + 校验器（M5）；
- `experiments/` 生命周期状态机 + 断点续训 + 结果对比（M6）；
- `benchmark/` 性能基准（M6）。

**明确不做**：不引入微服务、不强制 Docker、不强行分布式训练、不把大模型塞进单文件。

---

## 9. 升级实施顺序（本轮实际执行依据）

| 阶段 | 内容 | 交付判据 | 状态 |
|---|---|---|---|
| **M1** | 审计 + 统一异常/日志/配置校验 + 运行时计划 + 工程规范 | 本文档 + 141 测试绿 | ✅ 本轮完成 |
| **M2** | CUDA/AMP/显存/梯度累积/DataLoader/完整检查点 | 设备矩阵测试 + AMP 数值测试 + 峰值显存记录 | ✅ 代码完成，CUDA 实跑 `NOT_RUN` |
| **M3** | NetCDF 流式 / 缓存 / 质量校验 / 溯源 | 不整载内存 + 缓存可失效重建 | 🔄 builder-data 执行中 |
| **M4** | MLP/RF/LightGBM/CNN/U-Net/ConvLSTM/多任务 | 每模型前向+梯度+存取+最小训练 | 🔄 builder-models 执行中 |
| **M5** | 真实/仿真数据源融合 / 时空划分 / 物理约束 / 不确定性 | 混合采样器 + 物理校验器 + 跨域评估 | 🔄 builder-physics 执行中 |
| **M6** | 实验注册 / 恢复 / Benchmark / 对比 | 续训等价性 + benchmark 真实数据 | ⏳ 待做 |
| **M7** | 全面验收 + 文档 + 依赖检查 + 性能报告 | 验收清单逐项标注 PASS/NOT_RUN | ⏳ 待做 |

依赖关系：M3–M5 相互独立（写作用域已切分）→ M6 依赖 M2 的运行时计划与 M3 的数据指纹 → M7 收口。

---

## 10. M1 已落地的加固（与本审计一一对应）

| 审计项 | 本轮改动 | 位置 |
|---|---|---|
| D16 无统一异常体系 | 新增 `CloudWaterError` 体系（配置/数据/模型/训练/OOM/检查点/实验/物理/设备/依赖） | `utils/exceptions.py` |
| D17 配置无校验 | 新增配置校验器（错误 vs 警告分级，问题清单聚合抛出） | `utils/config_schema.py` |
| 日志不结构化 | 新增 JSON 行式格式化 + 结构化事件 + 异常结构化记录 | `utils/logging.py` |
| 设备能力只有字符串 | 新增硬件抽象层：设备探测/显存监控/精度策略/性能剖析 | `hardware/` |
| 训练缺 GPU 能力 | Trainer 升级：AMP + 梯度累积 + 梯度裁剪 + channels_last + OOM 捕获 + 计时 + 显存统计 | `training/trainer.py` |
| 配置散落 | 新增 `RuntimePlan`：配置 → 设备/精度/DataLoader/累积 的单一解析点 | `training/runtime.py` |
| B2 检查点状态不全 | 检查点增加 scheduler/scaler/global_step/RNG 状态/history；续训恢复全部 | `training/checkpoint.py`、`utils/reproducibility.py` |
| 损失分量不可见 | `CompositeLoss.component_values()` 逐分量记录 | `training/losses.py` |
| D15 CLI 缺命令 | 新增 `doctor` / `gpu info` / `models list` / `data validate` | `cli.py` |
| 无工程规范 | ruff / pytest markers / mypy 基线写入 pyproject；版本号 2.0.0 | `pyproject.toml` |

**兼容性保证**：v1.0 的 `training.device`、`training.num_workers`、`training.grad_clip` 等旧键**继续生效**；
新旧分段冲突时按「**具体值优先，新分段优先**」裁决（`training/runtime.py:_first_concrete`）。

---

## 11. 需要决策的事项（待主人拍板）

1. **CUDA 验收机器**：本机无 NVIDIA GPU，验收清单中「RTX 5060 被正确识别 / CUDA 训练真实运行 /
   显存峰值记录」无法在本机产生真实数据。请指定一台 RTX 5060 机器（或确认以代码交付 + 交付验证脚本的方式结项）。
2. **是否引入 dask 作为一等依赖**：TB 级流式读取的推荐路径依赖 dask；当前环境未安装。
   建议：列入 `pyproject` 的 `dask` 可选组，并在无 dask 时给出显式警告（已按此实现）。
3. **实验目录历史产物**：`outputs/experiment_001..003` 为 v1.0 产物。v2.0 的生命周期管理会新增
   `state.json` 等文件——**不覆盖**旧目录内的 `metrics.json`，是否需要为旧产物做一次
   「补录元数据」的迁移脚本？（默认：不迁移，保持历史原样。）

---

## 12. 未解决问题 / 已知局限

- 本轮**未**在真实 ERA5 NetCDF 与真实 GPU 上验证（环境不具备）→ 记 `NOT_RUN`，非 `PASS`。
- 不确定性估计（Deep Ensemble / MC Dropout / Quantile Regression）在 v1.0 仍是预留接口，
  M5 才实现；在此之前**不得**把普通预测误差称为置信区间。
- 物理约束除「非负性」外均为**未验证口径**的占位实现，`CWPConsistencyPenalty` 构造即抛错——
  这是刻意设计，不是缺陷。

---

## 附：证据索引

- 测试基线：`python -m pytest -q` → 89 passed（审计前）/ 141 passed, 2 skipped（M1/M2 后），退出码 0。
- 环境探测：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` →
  `2.15.0 False`；`nvidia-smi` → `command not found`。
- CLI 实跑：`python -m cloudwater_ml.cli doctor | gpu info | models list | data validate` → 退出码 0。
- 审计明细：本报告 §4 各条 `file:line`；子审计原始结论见 M1 归档。
