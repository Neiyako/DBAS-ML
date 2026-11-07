# CloudWater ML Platform v2.0

> 云水资源科研机器学习平台：CLWC / CIWC 反演、CWP 预测、时空场重建、仿真-真实数据融合训练。
>
> 面向大气科学与云水资源综合评价的科研实验平台：从**数据**（TB 级 ERA5 NetCDF）到
> **GPU 训练**（CUDA / RTX 50 系）到**可复现实验管理**，一条链路走通。

```text
数据生成/接入 → Dataset → DataLoader → Model → Loss(+物理约束) → Backprop
   → Validation → Checkpoint → Metrics → Uncertainty → Visualization → Experiment Record
```

---

## 1. 三十秒上手

```bash
pip install -e ".[dev]"
cloudwater doctor                       # 环境体检（依赖/设备/目录/模型）
cloudwater train --config configs/experiments/mlp_baseline.yaml
cloudwater report --experiment experiment_001
```

> 本仓库自带**合成**测试数据（`data/synthetic/cloudwater_synth.npz`），用于跑通工程链路。
> 合成数据 `source` 恒为 `simulated`，**不得当作真实观测使用**。
> 仓库历史产物 `outputs/experiment_001..003` 是基于该合成数据的基线运行（见 `outputs/*/metrics.json`）。

---

## 2. 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # 开发：pytest / ruff / mypy / lightgbm / pyarrow
pip install -e ".[lgbm,parquet]"   # 按需追加
pip install -e ".[dask]"           # TB 级 NetCDF 分块读取（可选）
```

| 可选组 | 内容 | 什么时候需要 |
|---|---|---|
| `lgbm` | LightGBM | 树模型基线 |
| `xgboost` | XGBoost | 树模型基线 |
| `parquet` | pyarrow | Parquet 数据源 |
| `dask` | dask | TB 级 NetCDF 分块/惰性读取 |
| `dev` | pytest / ruff / mypy | 开发与 CI |

> **macOS / iCloud 提示（已实测）**：放在 iCloud 同步目录（如 `Desktop`）里的项目，
> iCloud 会反复把 `.venv/lib/*/site-packages/*.pth` 打上 `hidden` 标志，
> Python 的 `site` 模块会跳过 hidden 的 `.pth` → `cloudwater` 控制台脚本报
> `ModuleNotFoundError: No module named 'cloudwater_ml'`。
>
> 三种解法（任选其一）：
> 1. **推荐**：用仓库自带启动器 `./scripts/cloudwater ...`（不依赖 `.pth`）；
> 2. `PYTHONPATH=src cloudwater ...`；
> 3. 临时清标志 `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`
>    （iCloud 可能会再次设置，属环境问题）。
>
> pytest 不受影响（`pyproject.toml` 已设 `pythonpath = ["src"]`）。

---

## 3. CUDA 环境配置

| 显卡 | 需要的 PyTorch 构建 |
|---|---|
| RTX 50 系（Blackwell, sm_120） | **cu128**（`torch.version.cuda >= 12.8`） |
| RTX 30/40 系 | cu118 / cu121 |
| Apple Silicon | 默认 CPU / MPS |

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
cloudwater gpu info      # 确认 CUDA 可用、型号、算力、显存
```

平台**不写死**任何显卡型号或显存容量：一切以 `cloudwater gpu info` 的**实测值**为准。
详见 [`docs/gpu_training.md`](docs/gpu_training.md)。

---

## 4. GPU / 设备检测

```bash
cloudwater gpu info          # 人类可读
cloudwater gpu info --json   # 结构化
cloudwater doctor            # 依赖 + 设备 + 目录 + 已注册模型
```

自动检测项：GPU 名称、CUDA 版本、PyTorch CUDA 构建、可用/总显存、计算能力、
多卡数量、MPS 可用性、cuDNN 版本。**CUDA 不可用时给出明确原因并回退 CPU**（不静默）。

---

## 5. 数据接入

### 5.1 支持的格式

| 适配器 | 适用 | 说明 |
|---|---|---|
| `npz` / `csv` / `parquet` | 中小规模表格 | parquet 走**列投影**读取 |
| `netcdf` | ERA5 等 | 按时间/空间/气压层/变量**按需读取**，可分块、可流式 |

### 5.2 NetCDF 按需读取

```yaml
data:
  adapter: netcdf
  path: /data/era5/era5_2015.nc
  features: [t, q, rh, u, v, w]
  targets: [clwc, ciwc]
  netcdf:
    time_range: ["2015-01-01", "2015-12-31"]
    latitude_range: [20.0, 50.0]
    longitude_range: [90.0, 130.0]
    level_range: [500.0, 900.0]
    chunks: { time: 16 }
```

- **不会默认把整个数据集载入 RAM**；
- 体积守卫会在读取前拦截，并给出**真正有效**的降级建议；
- 缓存分级（文件元数据 / 预处理结果 / 索引 / 内存），**可失效、可重建、可追溯**，
  且**永不写入 `data/raw/`**。

### 5.3 数据质量与溯源

```bash
cloudwater inspect --config configs/data.yaml        # 结构/变量/维度/单位（只读）
cloudwater data validate --config configs/data.yaml   # 质量报告
```

检查项：NaN/Inf、异常值、变量范围、经纬度、时间连续性、单位、维度、缺测分布。
**报告优先**：未经配置不修改任何原始科研数据。

详见 [`docs/data_engine.md`](docs/data_engine.md)。

---

## 6. 模型训练

### 6.1 已注册模型

```bash
cloudwater models list     # 从注册表读取，状态不会过期
```

| 模型 | 类型 | 输入契约 |
|---|---|---|
| `mlp` | 深度学习 | `[B, F]` → `[B, T]` |
| `random_forest` | 传统 ML | `[B, F]` → `[B, T]` |
| `lightgbm` | 传统 ML | `[B, F]` → `[B, T]` |
| `cnn` | 深度学习 | `[B, C, H, W]` → `[B, T]` |
| `unet` | 深度学习 | `[B, C, H, W]` → `[B, out, H, W]` |
| `convlstm` | 深度学习 | `[B, T, C, H, W]` → `[B, T_out, C_out, H, W]` |
| `multitask` | 深度学习 | 共享骨干 + 逐任务头（逐任务损失权重） |
| `xgboost` | 传统 ML | `[B, F]` → `[B, T]` |

### 6.2 训练配置

```yaml
training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  optimizer: adamw
  loss: mse
  scheduler: cosine
  gradient_accumulation_steps: 1   # 等效 batch = batch_size × 本值
  gradient_clip_norm: 1.0
  seed: 42
  early_stopping: { enabled: true, patience: 15 }

hardware:
  device: auto          # auto | cpu | cuda | mps
  precision: auto       # auto | fp32 | fp16 | bf16
  channels_last: auto   # 只对 4D 卷积模型自动启用
  compile: false
  grad_checkpointing: false

dataloader:
  num_workers: auto
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

能力：混合精度（AMP + GradScaler）、梯度累积（含尾部 flush）、梯度裁剪、
channels_last、torch.compile、梯度检查点、显存峰值统计、OOM 捕获、
数据/传输/计算三段耗时剖析、逐损失分量记录。

### 6.3 物理约束

```yaml
physics:
  constraints: [nonnegativity]
  nonnegativity_weight: 0.1
```

物理约束**作用在物理单位上**（不是 z-score）；单位口径未定的约束（如
CWP 柱积分一致性）在实现前**构造即报错**，不会把未验证的物理假设写进损失。
详见 [`docs/physics_constraints.md`](docs/physics_constraints.md)。

---

## 7. 模型扩展

新增模型三步：

```python
# src/cloudwater_ml/models/my_model.py
from .base import BaseModel, TrainingResult, register_model

@register_model("my_model", kind="深度学习")
class MyModel(BaseModel):
    is_torch = True

    def fit(self, X, Y, *, X_val=None, Y_val=None, config=None, out_dir=None):
        ...
    def predict(self, X) -> np.ndarray:      # 必须返回 [n_samples, n_targets]
        ...
    def _save_weights(self, path): ...
    @classmethod
    def _load_from(cls, path, meta): ...
```

```python
# src/cloudwater_ml/models/__init__.py
register_lazy_model("my_model", "cloudwater_ml.models.my_model")
```

契约：**不依赖 CLI / 实验运行器**；`predict` 恒返回 `[n_samples, n_targets]`；
每个模型独立文件。详见 [`docs/model_registry.md`](docs/model_registry.md)。

---

## 8. 实验复现

每次实验自动记录：git commit（+dirty 标记）、数据文件指纹、展开配置、
随机种子、依赖版本、GPU/CUDA/PyTorch 版本、运行时计划、耗时、峰值显存、全部指标。

```bash
cloudwater report --experiment experiment_001        # 单实验报告
cloudwater report                                   # 全部实验对比
cloudwater compare --experiments exp1 exp2          # 指定对比
```

---

## 9. 断点续训

```bash
cloudwater resume --experiment experiment_001
```

- 恢复模型 / 优化器 / 调度器 / GradScaler / 随机数状态 / 历史曲线 / 早停状态；
- **新建实验目录**，不覆盖原结果；新实验通过 `parent_run_id` 关联原实验。

---

## 10. 性能测试

```bash
cloudwater gpu benchmark --samples 8192 --epochs 2 --out outputs/benchmark.md
```

输出：samples/s、epoch 耗时、**数据加载 / 设备传输 / 计算**三段耗时、
峰值显存与内存、瓶颈判定，以及**按本机实测硬件**推导的配置建议。
详见 [`docs/benchmark.md`](docs/benchmark.md)。

---

## 11. 不确定性

```yaml
evaluation:
  uncertainty:
    enabled: true
    method: conformal     # 分割共形区间（名义覆盖率 1-alpha）
    alpha: 0.1
```

用**验证集做校准集**生成预测区间，并给出实际覆盖率、平均宽度与 CRPS。
**普通模型的预测误差不会被称作置信区间**；没有区间来源时接口直接报错。

---

## 12. 故障排查

| 症状 | 处理 |
|---|---|
| `import cloudwater_ml` 失败 / `cloudwater` 命令报 ModuleNotFound | 用 `./scripts/cloudwater ...` 或 `PYTHONPATH=src`（iCloud 把 `.pth` 标成 hidden，环境问题） |
| `配置指定 cuda，但当前环境无可用 CUDA 设备` | 装 CUDA 版 torch，或用 `device: auto` |
| `no kernel image is available` | Blackwell 需 cu128 构建 |
| 显存 OOM | 降 `batch_size` / 开 `gradient_accumulation_steps` / 开 `grad_checkpointing` / 降精度 |
| `subset 结果为空：来源标签未解析` | 数据没声明来源：设 `data.source_label` 或把 `mix.real_fraction` 置 null |
| `实验目录已存在且非空` | 换 `--name`；确需覆盖用 `--force`（旧目录会备份） |
| 物理约束没效果 | `nonnegativity_weight` 是否 > 0（0 = 关闭，框架会告警） |
| 训练慢、GPU 占用低 | 看 `runtime_plan.profiling`：`data_load` 占比大就调 workers/prefetch |
| 指标里没有 NSE | `evaluation.metrics` 里显式列出 `nse` |

---

## 13. 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 架构分层、依赖方向、模块职责 |
| [`docs/engineering_audit.md`](docs/engineering_audit.md) | **工程审计报告**（现状/缺陷/瓶颈/风险/顺序） |
| [`docs/upgrade_report.md`](docs/upgrade_report.md) | **v2.0 升级交付报告**（架构图/差异/实测/模型状态/遗留） |
| [`docs/gpu_training.md`](docs/gpu_training.md) | GPU 训练：设备/精度/累积/显存/续训 |
| [`docs/data_engine.md`](docs/data_engine.md) | 数据引擎：流式读取/缓存/质量/溯源 |
| [`docs/model_registry.md`](docs/model_registry.md) | 模型注册表与扩展规范 |
| [`docs/physics_constraints.md`](docs/physics_constraints.md) | 物理约束体系与单位口径 |
| [`docs/experiment_management.md`](docs/experiment_management.md) | 实验生命周期、状态机、对比 |
| [`docs/benchmark.md`](docs/benchmark.md) | 性能基准与瓶颈归因 |
| [`docs/reproducibility.md`](docs/reproducibility.md) | 可复现性：代码/数据/随机性/环境 |
| [`docs/deployment.md`](docs/deployment.md) | 部署、目录约定、批处理、备份迁移 |
| [`docs/data_schema.md`](docs/data_schema.md) | 数据契约（v1.0） |
| [`docs/experiment_protocol.md`](docs/experiment_protocol.md) | 实验协议（v1.0） |

---

## 14. 边界与诚实性约定

- **不伪造**：未实现的写成「未实现」并让接口报错；不把占位符包装成已实现；
- **不伪造科研结论**：合成数据只用于工程链路验证，不得写成观测结论；
- **不擅自改数据**：原始科研数据只读；清洗需显式开启且全部留痕；
- **不静默降级**：设备回退、精度降级、配比未达成、缓存失效都记录原因；
- **不覆盖结果**：同名实验目录默认拒绝覆盖；续训新建实验。

## 15. 许可

MIT。
