# 模型设计 · cloudwater-ml

> 出处标注 `路径:行号`，参数默认值一律**从代码抄录**（未跑过的实验不写结果）。
> 快照时间 2026-11-07 20:18（builder-models 已交付并仍在并行维护 `models/` 与 `training/`）。

## 1. `BaseModel` 契约（所有模型必须实现）

定义 `src/cloudwater_ml/models/base.py:97`：

| 成员 | 签名 | 说明 | 行号 |
|---|---|---|---|
| `name` | `str` 类属性 | 注册名，由 `@register_model(...)` 写入 | `base.py:100`、写入处 `:69-77` |
| `is_torch` | `bool` 类属性 | `True`=走 Trainer/需 device；`False`=sklearn 风格，runner 强制 cpu 且 auto 不缩放 | `base.py:101`、`runner.py:213,:297`、`preprocessing.py:210-212` |
| `__init__(**params)` | | `self.params` 存全部构造参数；`feature_names/target_names/is_fitted` 初始化 | `base.py:103-107` |
| `fit` | `fit(X, Y, *, X_val=None, Y_val=None, config=None, out_dir=None) -> TrainingResult` | 抽象方法；`config` 是完整实验配置（取 `training` 段），`out_dir` 是实验 `model/` 目录 | `base.py:110-121` |
| `predict` | `predict(X) -> np.ndarray` | 抽象方法；**恒返回二维 `[n_samples, n_targets]`**（单目标列数 1） | `base.py:124-126`、docstring `:7` |
| `feature_importance` | `-> dict[str, float] \| None` | 默认返回 `None`；**不适用时不得编造**（MLP 即返回 None） | `base.py:128-130` |
| `save` | `save(path)` | 模板方法：写 `model_meta.json` 再调子类 `_save_weights` | `base.py:133-150` |
| `_save_weights` | `_save_weights(path: Path)` | 抽象；落权重文件（model.pt / model.joblib / model_target_*.txt） | `base.py:152-154` |
| `_load_from` | `@classmethod _load_from(path, meta) -> BaseModel` | 抽象；子类还原实例 | `base.py:156-159` |
| `load` | `BaseModel.load(path)` | 等价模块级 `load_model(path)` | `base.py:161-164` |

`model_meta.json` 内容（`base.py:137-146`）：`format_version=1`、`name`、`class`、
`params`、`feature_names`、`target_names`、`is_torch`、`is_fitted`。
`load_model(path)` 按 `name` 查注册表还原，未注册会列出当前已注册名单
（`base.py:167-185`，:176-179）。

### `TrainingResult`（`base.py:30-42`）

| 字段 | 类型 | 含义 |
|---|---|---|
| `history` | `list[dict]` | 每 epoch 一行，约定键 `epoch / train_loss / val_loss`（sklearn 基线为单行 `epoch=0`，见 `base.py:34` 注释） |
| `best_epoch` | `int \| None` | 最佳 epoch |
| `best_val_loss` | `float \| None` | 最佳验证损失（无 val 时 None） |
| `stopped_early` | `bool` | 是否早停触发 |
| `extra` | `dict` | 各模型自由附加信息（会进 `metrics.json.train.extra`，`runner.py:322`） |
| `final_train_loss` | property | `history[-1]["train_loss"]`（`base.py:40-42`） |

### 注册表：**惰性注册**（不是 import 即注册）

| 组件 | 作用 | 行号 |
|---|---|---|
| `MODEL_REGISTRY` | `name -> 类`，真正注册发生在此 | `base.py:48` |
| `_LAZY_MODULES` | `name -> 模块路径`，首次使用才 import | `base.py:49-53` |
| `register_lazy_model(name, module)` | 登记惰性模型 | `base.py:56-58` |
| `_ensure_loaded(name)` | `build_model/load_model` 先触发 import | `base.py:61-66` |
| `@register_model("名字")` | 装饰器，把类写进 `MODEL_REGISTRY` | `base.py:69-77` |
| `build_model(name, params)` | `_ensure_loaded` 后查表构造；未注册报错并点名预留模型 | `base.py:84-91` |
| `available_models()` | `sorted(set(REGISTRY) \| set(LAZY))` | `base.py:80-81` |

三行惰性登记在 `models/__init__.py:36-38`（mlp / random_forest / lightgbm）；
四个预留模型的**类**经 PEP 562 `__getattr__` 惰性暴露（`models/__init__.py:41-55`），
但**不注册**，所以 `model.name: cnn` 仍会报未注册错（见 §4）。

> 为什么惰性：macOS 上 torch 与 LightGBM 的 OpenMP 运行库同进程会崩
> （`Abort trap: 6`，`models/base.py:50-52`、`models/__init__.py:6-8`）；
> 惰性加载 + `set_seed(..., with_torch=False)`（`reproducibility.py:16-24`，
> runner 在 `runner.py:153-156` 先建模型再种种子）保证 LightGBM 实验完全不加载 torch。

## 2. 三个已实现模型

### 2.1 MLP（PyTorch）

`src/cloudwater_ml/models/mlp.py`，注册名 `mlp`（`mlp.py:48`），`is_torch = True`（:59）。

网络结构 `MLPNet`（:19-45）：`Linear → 激活 → [Dropout]` 重复 `hidden_layers` 次，
**输出层恒为线性**（:41 注释「输出层恒为线性」）——防止 ReLU 把非负量截断成
结构化偏差，也保证负值惩罚项有梯度可言（见 `docs/physics_constraints.md` §4-4）。

| 参数 | 代码默认值 | 配置示例（`configs/experiments/mlp_baseline.yaml:11-14`） | 出处 |
|---|---|---|---|
| `hidden_layers` | `[128, 64]` | `[128, 64]` | `mlp.py:63,:67` |
| `activation` | `"relu"` | `relu`（可选 relu/gelu/tanh，:16,:31-32） | `mlp.py:64` |
| `dropout` | `0.0` | `0.1` | `mlp.py:65` |

行为要点：

- 输入/输出维度在 `fit` 时从 X/Y 自动取：`n_in = X.shape[1]`、`n_out = Y.shape[1]`
  （:90），**多目标 = 单头 n_targets 线性单元**（类 docstring :50）；
- 数据转 `float32`（:86-87）；
- 训练交给 `Trainer(self.net, training_cfg)`（:102-108）：`training_cfg` 取
  `config["training"]` 段，并**把顶层 `physics` 段并进去**（:103-106，
  注释「physics 段位于完整配置顶层，须显式并入才能到达 Trainer → build_constraints」）；
- `predict` 未训练即报错（:114-115），输出 reshape 成 `[n, n_out]`（:123）；
- 权重存 `model.pt`（`state_dict` + `structure` 元数据，:126-141），
  加载用 `weights_only=True`（:145）；
- **不实现 `feature_importance`** → 继承基类返回 `None`（`base.py:128-130`），
  因此 MLP 实验不会产 `feature_importance.png`（`runner.py:272-279` 只在有值时出图）。

### 2.2 Random Forest（sklearn）

`src/cloudwater_ml/models/random_forest.py`，注册名 `random_forest`（:17），`is_torch = False`（:29）。

| 参数 | 代码默认值 | 配置值（`configs/experiments/random_forest.yaml:11-16`） | 透传目标 |
|---|---|---|---|
| `n_estimators` | `200` | 200 | `RandomForestRegressor` |
| `max_depth` | `None` | `null` | 同上 |
| `min_samples_leaf` | `2` | 2 | 同上 |
| `n_jobs` | `-1` | -1 | 同上（多目标时也用于 `MultiOutputRegressor`，:80） |
| `random_state` | `42` | 42 | 同上 |

（默认值出处 `random_forest.py:33-37`，透传 :48-52。docstring 还提示：受限环境下
 `n_jobs=-1` 可能段错误，此时显式传 `n_jobs=1`，`random_forest.py:25-26`。）

- **多目标策略：`MultiOutputRegressor(rf)`**（单目标直接用 `rf`，:78-83），
  即逐目标各训一片森林；
- 数据转 `float64`（:68-69）；`fit` 忽略 `X_val` 并在 `extra` 里**如实记录**
  `"x_val_ignored": True` 与说明（:86-94）；
- `history` 为单行 `{"epoch": 0, "train_loss": 训练集MSE, "val_loss": None}`，
  注释明确「sklearn 无逐 epoch 训练循环…不编造曲线」（:86,:96）；
- `feature_importance`：多目标取各森林 `feature_importances_` 的**平均**（:112-122）；
- 权重存 `model.joblib`（:127-130），加载时只回填已知参数键（:133-140）。

### 2.3 LightGBM（可选依赖）

`src/cloudwater_ml/models/lightgbm.py`，注册名 `lightgbm`（:29），`is_torch = False`（:40）。
缺依赖时**模块仍可 import**，只在构造/`fit` 时抛 `RuntimeError` 并提示
`pip install 'cloudwater-ml[lgbm]'`（:13-26）。

| 参数 | 代码默认值 | 配置值（`configs/experiments/lightgbm.yaml:11-17`） |
|---|---|---|
| `n_estimators` | `300` | 300 |
| `learning_rate` | `0.05` | 0.05 |
| `num_leaves` | `31` | 31 |
| `subsample` | `0.8` | **0.9（配置覆盖代码默认）** |
| `colsample_bytree` | `0.8` | **0.9（配置覆盖代码默认）** |
| `random_state` | `42` | 42 |
| `verbose` | `-1` | 未在配置中出现 → 用代码默认 |
| `importance_type` | `"gain"` | 未在配置中出现 → 用代码默认 |

（默认值出处 `lightgbm.py:43-52`，透传 :62-72。）

- **多目标策略：逐目标独立 `LGBMRegressor`**，`self.models` 列表按目标顺序存放，
  `predict` 逐列 `column_stack`（:110-124,:153-160）；
- 早停：仅当有 `X_val` 且 `training.early_stopping.enabled`（默认 True）且
  `patience > 0` 时启用，patience 缺省取 **50**（配置给 15，`lightgbm.py:100`），
  用 `lgb.early_stopping`；lightgbm≥4.3 走 `eval_X/eval_y`，旧版回退 `eval_set`
  （:98-121）；
- `history` 单行，`val_loss` = 各目标验证 MSE 均值；`extra` 含
  `early_stopping`、`best_iterations`、`val_mse`、`importance_type`（:126-150）；
- `feature_importance`：各目标 `booster_.feature_importance(importance_type=gain)`
  的平均（:163-174）；
- 权重逐目标存 `model_target_0.txt …`（booster 文本，:177-183），加载时按编号排序
  还原并回填 `n_features_in_`、`fitted_`（:186-201）。

### 2.4 三者对比一览

| 维度 | mlp | random_forest | lightgbm |
|---|---|---|---|
| 框架 | PyTorch | sklearn | LightGBM |
| `is_torch` | `True` | `False` | `False` |
| 归一化（`preprocessing.scale=auto`） | 缩放 | 不缩放（NoScaler） | 不缩放 |
| device | `resolve_device`（auto→cuda/mps/cpu） | 强制 `cpu (sklearn-style)` | 同左 |
| 多目标 | 单头线性输出 `n_out=n_targets` | `MultiOutputRegressor` | 逐目标独立模型 |
| 训练循环 | Trainer 逐 epoch + 早停 + checkpoint | 单次 `fit`，忽略 val | 逐目标 `fit`，可选 lgb 早停 |
| `history` | 每 epoch 一行 | 单行 epoch=0 | 单行 epoch=0 |
| `feature_importance` | `None`（不画图） | 平均 impurity importance | 平均 gain importance |
| 权重文件 | `model.pt` | `model.joblib` | `model_target_*.txt` |
| 是否 import torch | 是 | 否（惰性加载，见 §1） | 否（惰性加载，见 §1） |

## 3. Trainer（PyTorch 模型的公共训练循环）

`src/cloudwater_ml/training/trainer.py:29`，`Trainer(model, training_cfg, device=None)`。

- 读取键（docstring :35-38）：`epochs / batch_size / learning_rate / weight_decay /
  optimizer / loss / huber_delta / scheduler / early_stopping / seed / device /
  num_workers / grad_clip / resume`，可选 `physics` 段；
- `fit(X_train, Y_train, X_val, Y_val, out_dir)` 流程（:54-190）：`set_seed`（:66）→
  DataLoader 分批（:78-94）→ `get_loss + build_constraints + CompositeLoss`（:97-101）→
  `build_optimizer`（:102）→ `build_scheduler`（:103）→ 可选 resume（:110-117）→
  逐 epoch：训练/验证 loss → 调度 step → 有 val 监控 val 否则监控 train（:146）→
  保存 `best.pt`/`last.pt`（:152-155）→ 早停判断（:165-171）；
- 收尾**恢复 best 权重**再返回（:173-177），与 sklearn「最优模型」语义对齐；
- `extra` 返回 `device / epochs_run / monitor / restored_best`（:184-189）；
- 支持 `training.resume` 指定 checkpoint 续训（:110-117，
  `load_checkpoint` 见 `training/checkpoint.py:45`）；
- `losses = {"X": xb, "Y": yb}` 传给 criterion（:215），正是 `PhysicsConstraint.penalty`
  拿真值的通道。

配套：`get_loss` mse/mae/huber（`training/losses.py:31-42`）、
`build_optimizer` adam/adamw/sgd（`training/optimizer.py:11-33`）、
`build_scheduler` null/cosine/step（`training/scheduler.py:11-33`）、
`EarlyStopping(patience=10, min_delta=0.0)`（`training/callbacks.py:19`，
配置给 15 / 0.0001）、`CheckpointManager` 管 `best.pt`/`last.pt`
（`training/checkpoint.py:69`，payload 键见 `checkpoint.py:3-4`）。

> 物理约束的可达性：`trainer.py:98` 用 `cfg.get("physics")` 取约束，
> 而 `models/mlp.py:103-106` 会把顶层 `physics` 段并入 `training_cfg` ——
> **当前链路可达**，默认仍是 `constraints: []`（全关）。详见
> `docs/physics_constraints.md` §2.3。

## 4. 四个预留模型（**未实现，不得当作已实现**）

| 模型 | 文件 | 注册状态 | 构造行为 | 预期用途（代码 docstring 原文口径） |
|---|---|---|---|---|
| CNN | `models/cnn.py:16` | **未注册** | `__init__` 即 `NotImplementedError`（:32-34） | 1D/2D 卷积，网格化输入的云水资源反演（局地特征→CLWC/CIWC/CWP），输入 `[batch, channels, length]` 或 `[batch, channels, H, W]`（:17-25） |
| UNet | `models/unet.py:16` | **未注册** | 同上（:31-33） | 编码-解码+跳连，云水资源二维场的空间分布重建/降尺度，输入 `[batch, channels, H, W]`（:17-24） |
| ConvLSTM | `models/convlstm.py:16` | **未注册** | 同上（:32-34，raise 在 :34） | 卷积+LSTM，云水资源场时空演变的多步预报，输入 `[batch, time, channels, H, W]`，用 `spatial_shape_guard(kind="temporal")` 校验（:17-25） |
| MultiTask | `models/multitask.py:16` | **未注册** | 同上（:31-33） | 共享骨干+逐任务头，CLWC/CIWC/CWP 联合反演，可配逐任务损失权重（:17-24） |

四个文件的 `fit/predict/_save_weights/_load_from` 全部 `raise NotImplementedError`
（各自 :36-47）。在实现前：

- `model.name: cnn` 调 `build_model` 会得到
  `未知模型 'cnn'，已注册: [...]。预留模型(cnn/unet/convlstm/multitask)需第二阶段实现后注册。`
  （`base.py:84-91`）——**明确报错，不伪装**；
- `cloudwater_ml.models.CNN` 这个**类**可以取到（`models/__init__.py:41-55` 惰性暴露），
  但**一实例化就 NotImplementedError**，两条路径都不产生"已实现"的假象；
- 它们各自 docstring 里的「建议参数」是**设计建议**，不是已生效的实现参数。

## 5. 如何添加一个新模型（步骤示例）

以新增 `gradient_boosting` 为例：

1. **建文件** `src/cloudwater_ml/models/gradient_boosting.py`：

```python
from .base import BaseModel, TrainingResult, register_model

@register_model("gradient_boosting")          # ① 注册名
class GradientBoostingModel(BaseModel):
    is_torch = False                          # ② sklearn 风格 → 不缩放、cpu

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, Y, *, X_val=None, Y_val=None, config=None, out_dir=None,
            feature_names=None, target_names=None) -> TrainingResult: ...
    def predict(self, X):                      # ③ 必须返回 [n, n_targets]
        ...
    def feature_importance(self):              # ④ 没有就返回 None，不编数
        return None
    def _save_weights(self, path): ...         # ⑤ 权重文件
    @classmethod
    def _load_from(cls, path, meta): ...       # ⑥ 还原实例
```

2. **在 `models/__init__.py` 登记惰性映射**（现有三行见 `:36-38`）：

```python
register_lazy_model("gradient_boosting", "cloudwater_ml.models.gradient_boosting")
```

3. **配置启用**（`configs/experiments/*.yaml`）：

```yaml
model:
  name: gradient_boosting
  params: { n_estimators: 300 }
```

4. **跑通**：`cloudwater train -c configs/experiments/<新文件>.yaml`；
   `available_models()`（`base.py:80`）能列出它，`model/` 里能被
   `load_model` 还原（`runner.py:363` 的 evaluate 链路会用到）。

约束清单：`predict` 形状必须 `[n, t]`（`runner.py:242-244` 还会兜底 reshape）；
`fit` 不得自行做归一化（scaler 由 runner 统一拟合与反变换，`runner.py:202-240`）；
不要在模型里读写 `outputs/` 之外的路径；`history` 至少要有 `epoch/train_loss` 键
（`tracking.py:98-103` 直接 `pd.DataFrame(history)` 落成 `history.csv`）；
非 torch 模型**不要在模块顶层 import torch**（惰性加载的意义所在，见 §1）。
