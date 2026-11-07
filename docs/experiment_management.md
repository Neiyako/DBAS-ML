# 实验管理与可复现性（CloudWater ML Platform v2.0）

> 目标：任何一次实验都能回答四个问题——
> **用什么数据、什么代码、什么配置、跑出了什么**；并且**中断了能接着跑，跑完了能对比**。

---

## 1. 生命周期状态机

```
CREATED ──► PREPARING ──► RUNNING ──► VALIDATING ──► COMPLETED
    │           │            │             │
    └───────────┴────────────┴─────────────┴──► FAILED / INTERRUPTED
```

实现在 `cloudwater_ml/experiments/manager.py`：

- 状态写在实验目录的 `state.json`（**权威状态文件**），与 `metrics.json` 并存；
- 非法迁移会抛 `ExperimentStateError`（例如把已 COMPLETED 的实验改回 RUNNING）；
- 失败/中断会记录异常类型与消息，**不会**留下一个「看起来正常」的半成品目录；
- `FAILED` / `INTERRUPTED` 可以重新发起（记为新的 `run_id`），历史记录不丢。

```bash
cat outputs/experiment_001/state.json
```

---

## 2. 实验 ID 与不覆盖保证

- 每个实验有全局唯一 ID：`exp_<YYYYmmdd_HHMMSS>_<6位哈希>`（哈希含随机串，同秒创建也不会撞名）；
- 自动编号目录用 `mkdir(exist_ok=False)` **原子创建**，并发跑多个实验不会算出同一个编号；
- **同名目录一律拒绝覆盖**；只有显式 `--force` 才允许，且旧目录会先被
  重命名为 `<name>.backup-<时间戳>`（**不删除**原始结果）。

```bash
cloudwater train --config configs/experiments/mlp_baseline.yaml --name my_exp
# → FileExistsError: 实验目录已存在且非空 …（拒绝覆盖）
```

---

## 3. 每次实验记录什么

`metrics.json` + `state.json` 合计记录：

| 类别 | 字段 |
|---|---|
| 身份 | `experiment_id`、`run_id`、`name`、`description`、`parent_run_id` |
| 模型 | 模型名、**完整超参数**、结构信息 |
| 配置 | 配置路径、展开后的 `config.yaml` 快照、含注释的 `config.source.yaml` |
| 数据 | 路径、适配器、特征/目标、来源分布、**文件指纹**（size/mtime/sha256 片段）、质量报告 |
| 划分 | 策略、比例、各集样本数、丢弃数、划分备注 |
| 混合 | 请求的真实/仿真配比、**实际配比**、丢弃数 |
| 归一化 | scaler 方法、**拟合证据**（n_samples / 特征名 / 拟合时间 / 训练数组指纹） |
| 随机性 | `seed`、各 RNG 状态（随检查点保存） |
| 环境 | Python / 平台 / 各依赖版本 / PyTorch / CUDA 构建 / cuDNN / GPU 型号 / 显存 / 算力 |
| 代码 | git commit、分支、**是否 dirty**、脏文件数 |
| 训练 | 最优轮次、最优损失、是否早停、轮次数、最终损失、**损失分量** |
| 资源 | 耗时、**峰值显存**、峰值内存、三段耗时剖析、运行时计划 |
| 评价 | 总体与逐目标指标、分组泛化、鲁棒性、不确定性状态 |
| 状态 | 状态、开始/结束时间、失败原因 |

---

## 4. 断点续训

```bash
cloudwater resume --experiment exp_20261107_210405_a1b2c3
# 也可用目录名：cloudwater resume --experiment experiment_001
```

行为：

1. 读取原实验的 `config.yaml` 与 `model/last.pt`；
2. **新建一个实验目录**（新 ID），**不覆盖原实验**；
3. 新实验的 `state.json` 记录 `parent_run_id` 与 `resumed_from`；
4. 恢复内容包括：模型权重、优化器、调度器、GradScaler、RNG 状态、global_step、
   **历史曲线**、**早停状态**——因此续训后的曲线是连续的，不是从断点重新开始。

检查点内容见 `docs/gpu_training.md` §5。

---

## 5. 实验对比

```bash
cloudwater compare --experiments experiment_001 experiment_002
cloudwater compare --experiments exp_a exp_b --json
```

对比表字段：实验名、模型、划分策略、训练样本数、测试集 overall 指标
（MAE / MSE / RMSE / R² / Bias / NSE / Pearson）。

```bash
cloudwater report                      # 全部实验横向对比
cloudwater report --experiment exp_x   # 单个实验的 Markdown 报告
```

---

## 6. 可复现性

### 6.1 随机性来源与控制

| 来源 | 控制方式 |
|---|---|
| Python `random` | `set_seed` |
| NumPy | `set_seed`（`np.random.seed`） |
| PyTorch CPU | `set_seed`（`torch.manual_seed`） |
| PyTorch CUDA | `torch.cuda.manual_seed_all` |
| DataLoader shuffle | 主进程 RNG 决定（与 worker 数无关） |
| 模型初始化 | `set_seed` 在建模前调用 |
| 训练中断续跑 | 检查点保存/恢复 RNG 状态 |

### 6.2 确定性模式与其边界

```yaml
training:
  deterministic: true
```

- 会调用 `torch.use_deterministic_algorithms(True, warn_only=True)`；
- **GPU 上仍有非确定性算子**（原子加、某些 cuDNN 卷积反向），
  `warn_only=True` 只告警不报错 → **平台不承诺位级一致**，只承诺「同种子 + 同数据 + 同配置」下结果统计一致。

### 6.3 复现清单（论文/评审用）

一次实验要复现，需要这六项（都已落盘）：

1. git commit（+ dirty 标记）
2. 数据文件指纹（path / size / mtime / sha256 片段）
3. 展开后的配置快照
4. 随机种子
5. 运行时计划（设备 / 精度 / batch / worker / 累积）
6. 依赖版本（Python / torch / xarray / netCDF4 / …）

---

## 7. 数据来源与仿真数据纪律

- 每条样本都带 `source`（real / simulated）、`source_origin`（来源判定依据）、
  `dataset`（数据集标识）与 `sample_id`；
- **仿真数据不得被标成真实观测**：来源标签由数据契约或显式配置决定，
  未解析时一律 `unknown`，**不默认 simulated**；
- 混合配比只作用于**训练集**，测试集永不参与配比；
- 请求配比无法满足时**如实记录实际配比并给出 notes**，不伪造标签；
- 划分策略必须能防时空泄漏（`time` / `year` / `region` / `source`），
  `random` 会引入自相关泄漏，仅用于链路验证。

---

## 8. 目录产物

```
outputs/experiment_001/
├── config.yaml            # 展开后的配置快照
├── config.source.yaml     # 原始配置（含注释）
├── state.json             # 生命周期状态（权威）
├── metrics.json           # 全部指标 + 溯源 + 运行时计划
├── evaluation.json        # 重新评估的结果（不覆盖 metrics.json）
├── history.csv            # 逐 epoch 损失（续训后连续）
├── preprocessors.json     # 归一化参数 + 拟合证据
├── model/                 # best.pt / last.pt / model.pt / model_meta.json
├── logs/                  # train.log / quality_report.md / history.json
└── plots/                 # 损失曲线、预测-真值、残差分布、特征重要性、空间分布
```
