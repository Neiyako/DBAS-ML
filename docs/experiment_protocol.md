# 实验协议 · cloudwater-ml

> 出处标注 `路径:行号`。本协议只写**已实现或已登记**的内容；未运行的实验一律标
> 「配置已预留，未运行」。快照时间 2026-11-07 20:15（runner/models 由队友并行维护，
> 行号以复核当刻代码为准）。

## 1. 数据划分的五种策略与防泄漏论证

统一入口 `split_indices(bundle, split_cfg)`（`src/cloudwater_ml/data/splitting.py:213`），
策略表 `_STRATEGIES = ("random","time","year","region","source")`（:23），
配置键 `split.strategy`（`configs/training.yaml:23`）。

**通用保证**：任何策略跑完都要过 `SplitResult.check_disjoint()`（:49-55），
三集两两相交即抛 `划分泄漏：train 与 val 存在 N 个重叠样本`；`split_indices` 内统一调用（:221）。
另外样本数 <10 直接拒绝划分（:218-219），train/test 为空也拒绝（:222-223）。

| 策略 | 语义 | 关键参数 | 防泄漏论证 | 出处 |
|---|---|---|---|---|
| `random` | 固定 seed 置换后按位置三段切分 | `split.seed`（默认 42）、`split.ratios` | 同分布假设下的基线；**不防时间泄漏**，只用于对照，故 `random_forest.yaml` 才用它 | `_split_random:75-80` |
| `time` | 按时间排序后切整块；给 `train_end`/`val_end` 时按时间边界切 | `split.params.train_end`、`val_end`；未给则按升序位置切 | 训练集只含 `t <= train_end` 的样本，验证在 `(train_end, val_end]`，测试 `> val_end` —— **未来样本永不进训练集**，时间自相关不会跨集合泄漏 | `_split_time:83-106`（边界判断 :95-97） |
| `year` | 整年归属同一子集 | `split.params.train_years/val_years/test_years`；缺省按 ratios 切年份（至少 3 个年份，:125-126） | 同一时刻的所有样本必然落在同一年内，杜绝「相邻时刻分属训练/测试」的泄漏；年与年之间无样本重叠 | `_split_year:109-141` |
| `region` | 空间盒整块归属 | `split.params.boxes = {train/val/test: [lon0,lon1,lat0,lat1]}` | **盒子必须互斥**：任意两盒重叠到样本即抛错（:161-167）；不落在任何盒子内的样本**显式丢弃并计数**（`dropped`，:168-178），不静默 | `_split_region:144-179` |
| `source` | 按 `real`/`simulated` 域标签整块归属（域迁移实验） | `split.params.train_sources/val_sources/test_sources` | 训练与测试来自不同数据域，度量跨域泛化；未列入配置的来源显式丢弃计数 | `_split_source:182-201` |

`split.ratios` 必须是 3 个非负数且和为 1.0，否则报错（`splitting.py:58-64`）。
time/year 依赖 `meta.time`（或 `meta.year`）列，缺失时报错并提示改用其它策略
（:84-85,:113-115）；region 需要 `meta.latitude/longitude`（:150-152）；
source 需要三个 params 键齐全（:185-188）。

三份实验配置实际使用：
`configs/experiments/mlp_baseline.yaml:18-22` = `time`（train_end 2017-12-31、val_end 2018-12-31）；
`random_forest.yaml:19` = `random`；`lightgbm.yaml:21` = `year`（ratios 0.6/0.2/0.2）。

## 2. train / val / test 的用途分工

| 集合 | 用途 | 允许 | 禁止 | 代码依据 |
|---|---|---|---|---|
| train | 拟合模型参数 + 拟合归一化参数 + 训练集内真实/仿真配比 | 训练、早停以外的一切拟合 | — | `runner.py:155`（建模型）、`:202`（scaler）、`:217`（fit） |
| val | 早停/最佳 epoch 选择（`training.early_stopping`，`configs/training.yaml:11-14`） | 只读，只用于监控 | **不参与梯度更新**；不参与 mix 配比 | `runner.py:211-212` 把 val 传给 `fit(X_val=,Y_val=)` |
| test | 最终指标，写进 `metrics.json.test` | 训练结束后一次性评价 | **不参与调参/早停/配比/归一化拟合** | `runner.py:188`（取子集）、`:242-248`：test 只在训练完成后用于 `evaluate_predictions` |

补充约束：

- `mix` **只作用于训练集**（`runner.py:45` 模块注释、:191「测试集不参与」）；
- scaler 只看 train（见 §3）；
- 重新评估已保存模型时，重载的是同一份 config 快照与同一套划分，
  并且**不覆盖原 metrics.json**，结果写 `evaluation.json`
  （`runner.py:340-385`，docstring :344-346）。

## 3. scaler 只在训练集拟合的实现证据

`src/cloudwater_ml/data/preprocessing.py`：

- 模块铁律写在 docstring `:2-5`：「归一化参数只能从训练集拟合，再应用到验证集与测试集」；
- `fit_scalers(X_train, Y_train, ...)`（`:196-222`）**签名只接受训练集数组**，
  没有「整数据集」的入口；`:219-220` 是仅有的两处 `fit` 调用；
- 拟合后写 `notes`：`"scalers 仅在训练集上拟合（fitted_on=train）"`（`:221`），
  并把 `fitted_on="train"` 固化进 `Preprocessors`（`:175`）；
- 调用点 `runner.py:202-207` 传的是 `train.X, train.Y`（划分在 `runner.py:185` 之后，
  所以喂进去的确实是训练 split 的子集，还可能再被 mix 抽样）；
- 未 fit 就 transform 直接抛错（`StandardScaler._check`，`:82-85`）；
- `state_dict()` 落盘到 `preprocessors.json`（`runner.py:236`），
  重新评估时按 `x.method`/`y.method` 还原（`runner.py:369-371`）；
- 测试：`tests/test_data.py:199 test_scaler_fit_on_train_only`、
  `:223 test_scaler_refuses_unfitted`。

`scale: auto` 的语义（`preprocessing.py:210-217`）：`is_torch=True` 才缩放，
树模型（`is_torch=False`）不缩放并改用 `NoScaler`，同时在 notes 里记录该决策。

## 4. A~E 真实/仿真配比实验矩阵（配置已预留，未运行）

登记处 `src/cloudwater_ml/experiments/registry.py:20-26`：

```python
MIX_RECIPES = {"A": {"real_fraction": 1.00}, "B": {"real_fraction": 0.75},
               "C": {"real_fraction": 0.50}, "D": {"real_fraction": 0.25},
               "E": {"real_fraction": 0.00}}
```

| 实验 | 训练集 real : sim | `mix.real_fraction` | registry 可运行标记 | 当前状态 |
|---|---|---|---|---|
| A | 100 : 0 | 1.00 | `runnable=False`（`registry.py:61`） | **配置已预留，未运行**（需真实数据） |
| B | 75 : 25 | 0.75 | `runnable=False`（:63） | **配置已预留，未运行**（需真实数据） |
| C | 50 : 50 | 0.50 | `runnable=False`（:64） | **配置已预留，未运行**（需真实数据） |
| D | 25 : 75 | 0.25 | `runnable=False`（:65） | **配置已预留，未运行**（需真实数据） |
| E | 0 : 100 | 0.00 | `runnable=True`（:66，纯仿真可用合成数据） | **配置已预留，未运行**（可用合成数据跑，但本仓当前未产出任何实验目录） |

执行语义 `apply_mix(train_bundle, real_fraction, seed=42)`（`runner.py:47-114`）：

- `real_fraction=None` → 不抽样，但仍回填真实/仿真计数（`runner.py:63-73`）；
- 目标比例按 `round(fraction * n)` 算名额，某来源不足时把名额转给另一来源（:82-87）；
- 抽样后**如实记录实际比例**：`actual_real_fraction`、`n_real`、`n_simulated`、
  `n_dropped`（:106-113）；实际比例偏离请求时在 notes 里写明
  「不调整、不伪造标签（如需请先接入真实数据）」（:101-105）；
- 结果进 `metrics.json` 的 `mix` 键（`runner.py:314`），并渲染进 report
  （`tracking.py:149-157`）。

**当前状态声明**：仓库内 `outputs/` 只有 `.gitkeep`，没有任何 `experiment_*/metrics.json`；
`data/raw/`、`data/processed/` 为空，没有真实 ERA5 数据。
因此上表五组实验**全部未运行**，本仓也不提供任何 A~E 的对比结论。

## 5. 复现清单（一次实验要留什么）

`metrics.json`（写入 `runner.py:289-332`）里对应的字段：

| 复现要素 | 字段路径 | 出处 |
|---|---|---|
| 随机种子 | `metrics.json → experiment.seed`（取自 `training.seed`，默认 42） | `runner.py:152,:296` |
| 配置快照 | 实验目录 `config.yaml`（展开后）+ `config.source.yaml`（原始含注释）；`experiment.config_path` 记录原始路径 | `tracking.py:106-116`、`runner.py:295` |
| 环境信息 | `environment.python / platform / cloudwater_ml / numpy / pandas / scikit_learn / torch / lightgbm`（缺失写 `not-installed`） | `runner.py:118-138` |
| 数据血统 | `data.path / adapter / features / source_counts / n_samples_total / quality_report / quality_passed` | `runner.py:303-311` |
| 划分 | `split.strategy / n_train / n_val / n_test / n_dropped / notes` | `splitting.py:39-47`、`runner.py:313` |
| 配比 | `mix.applied / requested / actual_real_fraction / n_real / n_simulated / n_dropped / notes` | `runner.py:58-114,:314` |
| 归一化参数 | `preprocessing`（= `preprocessors.json` 内容，含 `fitted_on`、`notes`） | `runner.py:315`、`preprocessing.py:187-193` |
| 模型与超参 | `experiment.model`、`experiment.model_params` | `runner.py:293-294` |
| 时间与耗时 | `experiment.started_at / finished_at / duration_sec` | `runner.py:298-300` |
| 训练过程 | `train.best_epoch / best_val_loss / stopped_early / n_epochs / final_train_loss / extra` + `history.csv` | `runner.py:316-323`、`tracking.py:98-103` |
| 结果 | `test.n_samples / metrics.overall / metrics.per_target / subgroups` | `runner.py:324-328` |
| 诚实标注 | `notes[]`（质量警告、mix 偏差、预留项说明） | `runner.py:331` |

复现步骤：`pip install -e .` 同一环境 → 用实验目录里的 `config.source.yaml`
重跑 `cloudwater train -c <该文件> --name <新名字>` → 逐位可复现性由
`set_seed`（`utils/reproducibility.py:12`，覆盖 random/numpy/PYTHONHASHSEED/torch）保证；
需要更严格的 torch 确定性算法时传 `set_seed(seed, deterministic=True)`（:12-34）。
非 torch 模型用 `set_seed(seed, with_torch=False)` —— 完全不 import torch
（`reproducibility.py:16-24`），这是避免 torch/LightGBM OpenMP 冲突的关键
（`runner.py:153-156`）。

## 6. 泛化、鲁棒性、不确定性协议

### 6.1 泛化分组（已实现）

配置 `evaluation.subgroups: [year, season, region]`（`configs/training.yaml:35`），
实现 `evaluation/generalization.py:104`：

- `year`：`meta.year` 优先，缺了从 `meta.time` 推（:40-45）；
- `season`：气象季节 DJF/MAM/JJA/SON（北半球月映射表 :26-31）；
- `region`：需 `evaluation.region_boxes = {"名称": [lon0,lon1,lat0,lat1]}`；
  **未配置就返回 `{"skipped": "no region_boxes configured"}`，不虚构区域**
  （:161-163）；
- 任一维度缺列/失败 → 该维度 `{"skipped": "<原因>"}`，不影响整体指标
  （`evaluator.py:99-102`）。

### 6.2 鲁棒性（噪声注入，已实现）

配置 `evaluation.robustness.noise_levels: [0.0, 0.05, 0.1]`
（`configs/training.yaml:36-37`），实现 `evaluation/robustness.py:26`：

- **噪声尺度 = `level × 该特征在 X 上的标准差`**（逐特征，`feature_std = Xa.std(axis=0)`，
  ddof=0；零方差特征 scale=0 → 无噪声）——`robustness.py:66-67,:75`；
- 噪声为 `N(0,1) × scale`，`np.random.default_rng(seed)` 固定种子可复现（:65,:75）；
- `level=0.0` 直接用原始 X 作基线（:72-73）；
- 返回 `{str(level): regression_metrics(...)}`，单水平失败记 `{"error": ...}`
  不崩主流程（:84-87）；
- runner 只在 `noise_levels` 非空时执行（`runner.py:251-256`），
  结果写 `metrics.json.robustness`（:329）并渲染成表（`tracking.py:186-196`）。

### 6.3 不确定性（预留，NOT_RUN）

- `evaluation.uncertainty.enabled` 默认 `false`（`configs/training.yaml:38-39`）；
- **即使打开**，runner 也只写 `{"status": "reserved", "note": "Prediction Interval /
  Coverage / CRPS 生成方法为第二阶段预留，本轮 NOT_RUN"}`，并在 notes 里声明
  「未产出数值（不伪装）」——`runner.py:258-265`；
- 可用的部分只有 `interval_metrics(y_true, y_lo, y_hi, ...)`：给定区间算
  `coverage` 与 `interval_width`（`uncertainty.py:45-90`）；
- `predict_interval()` 与 `crps()` 显式 `NotImplementedError`（`uncertainty.py:93-100`）；
- 报告里渲染为 `_预留未实现_`（`tracking.py:198-201`）。

## 7. 指标口径（评价的唯一标准）

`evaluation/metrics.py:35-62` 的 `METRIC_DEFINITIONS` 登记五项：MAE / MSE / RMSE / Bias /
R²，公式与单位见该表；`overall` 是**全部目标展平后的 micro 汇总**，
跨目标量纲差异大时（如 `clwc[kg kg-1]` 与 `cwp[kg m-2]` 混用）以 `per_target` 为准
（模块 docstring `:11-14`）。

R² 保护：`SS_tot < 1e-12`（绝对阈值，`_R2_SS_TOT_EPS`，:32）时 `r2=None` 并附
`r2_note="target 近似常数，R² 不可解释"`，同时 logger warning（:110-119）；
`report.md` 里也会标注（`tracking.py:174-175`）。指标输出一律 JSON 安全
（非有限值降为 `None`，`metrics.py:77-83`）。
