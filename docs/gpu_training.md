# GPU 训练指南（CloudWater ML Platform v2.0）

> 适用范围：CUDA / MPS / CPU 三种设备；本文以 CUDA（含 RTX 50 系 / Blackwell）为主。
> 所有结论都标注了代码位置，**实测与推断分开写**。

---

## 1. 设备与能力探测

```bash
cloudwater gpu info        # 人类可读
cloudwater gpu info --json # 结构化（写进实验记录用）
cloudwater doctor          # 依赖 + 设备 + 目录 + 已注册模型
```

探测结果来自 `cloudwater_ml.hardware.device`：

| 字段 | 来源 | 说明 |
|---|---|---|
| `device_type` | `detect_hardware()` | 实际使用的设备（cpu / cuda / mps） |
| `fallback` / `fallback_reason` | 同上 | 是否降级、**为什么**降级（不静默） |
| `torch_version` / `torch_cuda_version` | `torch.__version__` / `torch.version.cuda` | 非 CUDA 构建时后者为 `None` |
| `cuda_device_count` | `torch.cuda.device_count()` | 多卡可见数 |
| `gpu_devices[].name` | `torch.cuda.get_device_properties(i).name` | **运行时实测**，不写死型号 |
| `gpu_devices[].compute_capability` | `major.minor` | 如 `12.0`（Blackwell） |
| `gpu_devices[].total_memory_bytes` / `free_memory_bytes` | `props.total_memory` / `torch.cuda.mem_get_info` | 总量与当前空闲 |
| `gpu_devices[].multi_processor_count` | `props.multi_processor_count` | SM 数 |
| `gpu_devices[].supports_bfloat16` | `torch.cuda.is_bf16_supported()` | bf16 可用性 |
| `cudnn_version` | `torch.backends.cudnn.version()` | cuDNN 版本 |
| `host_memory_bytes` | `psutil` / `sysctl hw.memsize` / `/proc/meminfo` | 拿不到就是 `None`（不猜） |

**回退策略**（`hardware/device.py:293-330`）：

- `device: auto` → `cuda` → `mps` → `cpu`；
- 显式 `device: cuda` 但不可用 → **不静默降级**，而是记录 `fallback_reason` 并回退 CPU
  （原因写进实验 metrics，避免「以为在 GPU 上训练」）；
- 任何降级都会产生一条 warning，进入实验 `notes`。

### RTX 5060（Blackwell, sm_120）注意事项

- 需要 **CUDA 12.8+ 构建**的 PyTorch（`torch.version.cuda >= "12.8"`）；
  低于该版本会出现 `no kernel image is available`。
- 本框架在探测到 `compute_capability >= 12.0` 且 PyTorch CUDA 构建 < 12.8 时会**主动告警**
  （`hardware/device.py:335-349`），但**不阻断**——真实能力以 PyTorch 构建为准。
- **不写死显存容量**：所有显存判断都用运行时探测值（见 §6 建议生成）。

---

## 2. 混合精度（AMP）

精度策略在 `cloudwater_ml.hardware.precision`：

| 模式 | dtype | GradScaler | 说明 |
|---|---|---|---|
| `fp32` | float32 | 否 | 默认精度，数值最稳 |
| `fp16` | float16 | **CUDA 上启用** | 需要 loss scaling 防梯度下溢 |
| `bf16` | bfloat16 | 否 | 指数位宽同 fp32，无需 loss scaling |
| `auto` | — | 按硬件 | CUDA 且支持 bf16 → bf16；否则 CUDA → fp16；MPS → fp16；CPU → fp32 |

```yaml
hardware:
  precision: auto      # auto | fp32 | fp16 | bf16
```

要点：

- `auto` 在 CUDA 上**优先 bf16**（无需 GradScaler，数值更稳）；
- CPU 上 `fp16` 会被**拒绝**（torch CPU autocast 只支持 bfloat16），错误信息明确；
- CUDA 上 bf16 不被支持时显式 `precision: bf16` 会**报错**并提示改用 fp16/fp32；
- **不强制所有模型用混合精度**：`fp32` 永远可用，逐实验配置；
- **验证与指标计算默认 fp32**（`training/trainer.py` 中验证路径不进入 autocast），
  避免用混合精度累加出的指标做论文级结论。

---

## 3. 运行时计划（RuntimePlan）

配置 → 一次解析 → 一份可追溯的执行计划：`cloudwater_ml/training/runtime.py`。

```yaml
hardware:
  device: auto
  precision: auto
  compile: false
  channels_last: auto
  grad_checkpointing: false

training:
  batch_size: 256
  gradient_accumulation_steps: 1
  gradient_clip_norm: 1.0

dataloader:
  num_workers: auto
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

解析结果写入实验 `metrics.json` 的 `runtime_plan` 段：

```json
{"device": "cuda", "precision": {"mode": "bf16", "scaler_enabled": false},
 "batch_size": 256, "effective_batch_size": 1024,
 "channels_last": false, "compile": false,
 "dataloader": {"num_workers": 4, "pin_memory": true, "persistent_workers": true,
                "prefetch_factor": 2, "non_blocking": true}}
```

### 旧键兼容（v1.0 配置继续有效）

| 新键 | 旧键 | 裁决规则 |
|---|---|---|
| `hardware.device` | `training.device` | **具体值优先**，新段在前 |
| `dataloader.num_workers` | `training.num_workers` | 同上 |
| `training.gradient_clip_norm` | `training.grad_clip` | 同上 |

「auto」不算具体值：两边都是 auto 时才走自动推断
（`training/runtime.py:_first_concrete`）。

### CPU 回退策略

| 项 | CPU / MPS 上的行为 |
|---|---|
| `pin_memory` | 强制关闭（PyTorch 只在 CUDA 支持 pinned memory） |
| `non_blocking` | 强制关闭（仅 CUDA 有意义） |
| `persistent_workers` / `prefetch_factor` | `num_workers=0` 时自动失效（否则 PyTorch 直接抛 `ValueError`） |
| `num_workers: auto` | 样本 < 2000 取 0（多进程启动开销大于收益），否则 `min(8, CPU核数/2)` |
| `precision: auto` | CPU → fp32 |
| `channels_last: auto` | 只对 4D 卷积类模型（cnn / unet / resnet / convnext）启用 |

---

## 4. 训练循环能力

`cloudwater_ml/training/trainer.py`：

| 能力 | 开关 | 说明 |
|---|---|---|
| 混合精度 | `hardware.precision` | autocast + GradScaler（仅 CUDA+fp16） |
| 梯度累积 | `training.gradient_accumulation_steps` | 等效 batch = batch × 累积步；**尾部会 flush**（不静默丢弃样本） |
| 梯度裁剪 | `training.gradient_clip_norm` | AMP 下先 `unscale_` 再裁剪 |
| channels_last | `hardware.channels_last` | 只对 4D 模型有意义 |
| torch.compile | `hardware.compile` | 默认关；首次编译有额外耗时，收益需实测 |
| 梯度检查点 | `hardware.grad_checkpointing` | 以重算换显存（大模型） |
| 显存监控 | 自动 | 峰值显存 / 峰值 RSS 写入实验记录 |
| OOM 捕获 | 自动 | 抛 `OutOfMemoryError`，附带实际配置与建议 |
| 性能剖析 | 自动 | 数据加载 / 设备传输 / 计算 三段耗时 |
| 损失分量 | 自动 | 监督项与各物理约束分项记录 |

### 梯度累积的尾部处理

当 `batch 数 % 累积步 ≠ 0` 时，最后不足一个完整组的梯度会被**按实际比例修正后走一次优化器步进**
（`trainer.py` 尾部 flush 分支），并在日志里说明修正系数——
**不会出现「配置 256×4、实际少算一批」这种静默不一致**。

---

## 5. 检查点与断点续训

检查点（`training/checkpoint.py`）保存：

| 内容 | 键 |
|---|---|
| 模型权重 | `model_state`（**始终是未编译的原始模块**，避免 `_orig_mod.` 前缀导致跨配置续训失败） |
| 优化器状态 | `optimizer_state` |
| 学习率调度器状态 | `scheduler_state` |
| GradScaler 状态 | `scaler_state` |
| 轮次 / 全局步数 | `epoch` / `global_step` |
| 随机数状态 | `rng_state`（python / numpy / torch / cuda） |
| 历史记录 | `history`（续训后曲线**连续**，不被截断） |
| 早停状态 | `early_stopping_state`（best / best_epoch / wait） |
| 配置与运行时计划 | `trainer_config` / `runtime_plan` |

```bash
cloudwater resume --experiment exp_20261107_210405_a1b2c3
```

- 续训**新建实验目录**（新实验 ID），**不覆盖原实验结果**；
- 原实验与新实验通过 `state.json` 的 `parent_run_id` / `resumed_from` 关联。

### 确定性说明

- `set_seed(seed, deterministic=True)` 会调用 `torch.use_deterministic_algorithms(True, warn_only=True)`；
- **GPU 上仍有非确定性算子**（如 `index_put_` 原子累加、某些 cuDNN 卷积反向），
  `warn_only=True` 只会告警而不会强制报错——**位级一致不是承诺**；
- 想要「尽量可复现」：固定 seed + 固定数据划分 + 固定 `num_workers`；
  多 worker 下的 shuffle 顺序由主进程 RNG 决定，与 worker 数无关。

---

## 6. 按硬件给配置建议

```bash
cloudwater gpu benchmark --samples 8192 --batch-size 256
```

建议由 `benchmark/runner.py:recommend_config` 生成，**依据运行时探测到的可用显存**：

- 可用显存 × 0.6 作为可用预算，按特征数估算每样本激活占用 → 给出起步 `batch_size`；
- 显存偏紧时建议 `gradient_accumulation_steps=4` 维持等效批量；
- 有 bf16 能力 → `precision: bf16`；否则 `fp16`（CUDA 自动带 GradScaler）；
- CPU 环境建议 `fp32` + 少量 worker（用于链路验证，不用于最终实验）。

> **不写死任何型号的显存**：同一「RTX 5060」在不同批次上容量并不相同，
> 建议永远以 `cloudwater gpu info` 的实测值为准。

---

## 7. 在 RTX 5060 上验收清单

在带 NVIDIA GPU 的机器上执行：

```bash
pip install -e ".[dev]"
cloudwater doctor                       # 应显示 CUDA 可用、无 fallback
cloudwater gpu info                     # 应显示型号/sm_120/显存
python -m pytest -m gpu -q             # GPU 专属用例（无 CUDA 时自动 skip）
python -m pytest -q                    # 全量
cloudwater gpu benchmark --out outputs/benchmark.md
cloudwater train --config configs/experiments/mlp_baseline.yaml
```

需要核对的项目：

| 项 | 期望 |
|---|---|
| `torch.version.cuda` | ≥ 12.8（Blackwell） |
| `gpu info` 的 `compute_capability` | `12.0` |
| `runtime_plan.precision.mode` | `bf16`（支持时） |
| `runtime_plan.dataloader.pin_memory` | `true` |
| `metrics.json` 的 `experiment.peak_vram_bytes` | 非空且为合理值 |
| `metrics.json` 的 `runtime_plan.profiling.stages` | 含 `data_load` / `transfer` / `compute` |
| 续训后 `history.csv` | 连续（1..N，不截断） |

> 本仓库当前开发机为 **Apple Silicon Mac（无 CUDA）**，
> 因此上表的 CUDA 实测项在本轮记为 `NOT_RUN`，
> 代码路径已通过 CPU/MPS 等价路径 + 单元测试覆盖。

---

## 8. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `no kernel image is available` | PyTorch CUDA 构建与显卡算力不匹配（Blackwell 需 cu128） | 安装 cu128 构建 |
| `配置指定 cuda，但当前环境无可用 CUDA 设备` | 非 CUDA 构建或驱动缺失 | `pip install torch --index-url .../cu128`；或改 `device: auto` |
| 显存 OOM | batch 过大 | 减小 `batch_size` 或开 `gradient_accumulation_steps`；开 `grad_checkpointing` |
| 损失变 NaN | fp16 梯度下溢 | 用 `precision: bf16`（自动档已优先 bf16）；fp16 时确认 GradScaler 已启用 |
| `pin_memory` 无效 | 非 CUDA 设备 | 正常：框架已自动关闭并在 notes 里说明 |
| `persistent_workers` 报错 | `num_workers=0` | 框架已自动失效该组合 |
| 训练很慢但 GPU 占用低 | 数据供给瓶颈 | 看 `runtime_plan.profiling`：`data_load` 占比大 → 调 workers/prefetch |
| 续训报 key 不匹配 | 曾用 `compile: true` 保存、旧版本无前缀处理 | v2.0 已统一保存原始模块 state_dict；旧检查点需重新生成 |
