# 部署与运行手册（CloudWater ML Platform v2.0）

> 覆盖：安装、CUDA 环境配置、目录约定、运行入口、故障排查、备份与迁移。

---

## 1. 安装

```bash
git clone <repo> && cd DBAS-ML
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # 开发：含 pytest / ruff / mypy / lightgbm / pyarrow
pip install -e ".[lgbm,parquet]" # 生产：按需追加
pip install -e ".[dask]"         # TB 级 NetCDF 分块读取（可选）
```

入口：`cloudwater`（等价 `python -m cloudwater_ml.cli`）。

> **本机开发环境注意**：macOS 上 iCloud 可能把 `.venv` 内的 `.pth` 标记为 hidden，
> 导致 site 跳过它（`import cloudwater_ml` 失败）。对策：`pyproject.toml` 里
> 已设 `pythonpath = ["src"]`（pytest 生效）；命令行直接使用时
> `export PYTHONPATH=src`。

---

## 2. CUDA 环境

| 目标 | 要求 |
|---|---|
| RTX 50 系（Blackwell, sm_120） | PyTorch **cu128** 构建（`torch.version.cuda >= 12.8`） |
| RTX 30/40 系（sm_86/sm_89） | cu118/cu121 均可 |
| 驱动 | 满足 CUDA runtime 要求的较新驱动 |

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
cloudwater gpu info        # 确认 CUDA 可用、型号、算力、显存
```

平台**不硬编码**任何型号或显存容量：一切以 `cloudwater gpu info` 的实测值为准。

---

## 3. 目录约定

```
DBAS-ML/
├── configs/            # 配置（experiments/*.yaml 为可运行实验）
├── data/
│   ├── raw/            # 原始科研数据（只读；平台不写入此目录）
│   ├── processed/      # 派生数据
│   └── synthetic/      # 合成测试数据（source 恒为 simulated）
├── docs/               # 文档
├── outputs/            # 实验产物（每个实验一个目录）
├── scripts/            # 辅助脚本
├── src/cloudwater_ml/  # 源码
└── tests/              # 测试
```

**硬约束**：

- `data/raw/` 下的原始数据**只读**，平台任何路径都不会写入该目录；
- 缓存写在 `data/cache/`（或 `.cache/`），**不与原始数据混放**；
- `outputs/` 中同名实验目录**不会被静默覆盖**。

---

## 4. 常用命令

```bash
cloudwater doctor                        # 环境体检
cloudwater gpu info                      # GPU/CUDA/MPS 详情
cloudwater gpu benchmark --out outputs/benchmark.md
cloudwater models list                   # 模型注册表
cloudwater inspect --config configs/data.yaml
cloudwater data validate --config configs/data.yaml --out outputs/quality.md
cloudwater train --config configs/experiments/mlp_baseline.yaml
cloudwater resume --experiment experiment_001
cloudwater evaluate --experiment experiment_001
cloudwater compare --experiments experiment_001 experiment_002
cloudwater report --experiment experiment_001
```

---

## 5. 批处理与调度

平台本身**不含**分布式训练或作业调度（刻意不做）。在集群上建议：

```bash
for cfg in configs/experiments/*.yaml; do
  cloudwater train --config "$cfg" --name "$(basename "$cfg" .yaml)" || echo "FAILED: $cfg"
done
```

- 每个实验独立目录，互不覆盖；
- 失败实验在 `state.json` 里留有原因，便于批量排查；
- 需要 GPU 排队时用外部调度器（SLURM/PBS），平台只负责单作业内的资源使用。

---

## 6. 备份与迁移

要长期保存的只有两样：

1. `outputs/<exp>/`（`metrics.json` + `state.json` + `config*.yaml` + `model/`）；
2. 数据文件本身（平台不复制、不修改原始数据）。

迁移到新机器时：

```bash
# 1) 拷贝代码与配置
rsync -a DBAS-ML/ new-host:~/DBAS-ML/
# 2) 拷贝数据（或挂载共享存储），保持路径一致
# 3) 校验环境
cloudwater doctor
# 4) 用原配置重跑并对比
cloudwater train --config outputs/experiment_001/config.yaml --name rerun_001
cloudwater compare --experiments experiment_001 rerun_001
```

**缓存不随迁移**（可重建）：删掉 `data/cache/` 不影响结果，只影响速度。

---

## 7. 监控与告警要点

一次训练应能在 `metrics.json` 里看到：

- `experiment.peak_vram_bytes`（显存峰值）
- `experiment.duration_sec`（耗时）
- `runtime_plan.profiling.stages`（数据/传输/计算三段耗时）
- `train.extra.loss_components`（损失分量）
- `state.json` 的 `state`（生命周期）

GPU 利用率可用 `nvidia-smi dmon` 或 `cloudwater gpu benchmark` 观察；
平台不在训练循环里轮询 `nvidia-smi`（避免干扰计时）。

---

## 8. 故障排查速查

| 症状 | 处理 |
|---|---|
| `No module named cloudwater_ml` | `export PYTHONPATH=src`（或 `pip install -e .`） |
| `配置指定 cuda，但当前环境无可用 CUDA 设备` | 装 CUDA 版 torch，或改 `device: auto` |
| `no kernel image is available` | Blackwell 需 cu128 构建 |
| 显存 OOM | 降 `batch_size` / 开 `gradient_accumulation_steps` / 开 `grad_checkpointing` / 降精度 |
| `实验目录已存在且非空` | 换 `--name`，或显式 `--force`（旧目录会被备份） |
| 训练中断后想续跑 | `cloudwater resume --experiment <id>` |
| 指标里没有 NSE | `evaluation.metrics` 里显式列出 `nse`（默认只输出 mae/mse/rmse/r2/bias） |
| 物理约束没起作用 | 检查 `physics.nonnegativity_weight` 是否 > 0（0 等于关闭，框架会告警） |
| `quality.report_only` 下脏数据未处理 | 默认只报告不改数据；确需清洗时显式设 `report_only: false` |

---

## 9. 版本与升级

- 包版本：`cloudwater_ml.__version__`（与 `pyproject.toml` 同步）；
- 升级前先跑 `python -m pytest -q`（含 GPU 标记的用例在无 GPU 机器上自动跳过）；
- 重大版本升级后请重跑一遍基准，比较 `samples_per_second` 与显存峰值，
  确认没有性能回退。
