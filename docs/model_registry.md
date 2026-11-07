# 模型注册表与扩展规范（CloudWater ML Platform v2.0）

> 目标：**统一接口**、**可独立注册**、**可加载复现**。模型不依赖 CLI，也不依赖实验运行器。

---

## 1. 统一契约

所有模型实现 `BaseModel`（`src/cloudwater_ml/models/base.py`）：

| 方法 | 契约 | 说明 |
|---|---|---|
| `fit(X, Y, *, X_val, Y_val, config, out_dir)` | 返回 `TrainingResult` | 训练入口 |
| `predict(X)` | **恒返回 `[n_samples, n_targets]`** | 单目标列数为 1；任何形状偏离都算违约 |
| `feature_importance()` | `dict[name, float]` 或 `None` | 不适用返回 `None`（不得编造） |
| `save(path)` / `load_model(path)` | 目录互操作 | 模板方法：`model_meta.json` + 子类权重 |
| `name` / `is_torch` | 类属性 | `is_torch=True` 走 GPU 训练引擎 |

`TrainingResult`（`history` / `best_epoch` / `best_val_loss` / `stopped_early` / `extra`）
是训练层与实验层之间的唯一结果契约。

---

## 2. 注册表

```python
@register_model("cnn", kind="深度学习")     # 装饰器注册，可带元数据
class CNN(BaseModel): ...
```

```python
# models/__init__.py —— 惰性映射（首次使用时才 import）
register_lazy_model("cnn", "cloudwater_ml.models.cnn")
```

### 为什么惰性加载

macOS 上**同一进程先 import torch 再 fit LightGBM 会 `Abort trap: 6`**
（OpenMP 运行库冲突）。惰性加载让 LightGBM 实验完全不触碰 torch。

因此有两条硬规则：

1. 树模型模块（`lightgbm.py` / `xgboost.py` / `random_forest.py`）**不得 import torch**；
2. `experiments/runner.py` **不得在模块级 import torch**（相关代码放在 `if model.is_torch:` 内）。

### 注册表查询

```bash
cloudwater models list          # 读注册表元数据，状态随代码自动更新
cloudwater models list --json
```

```python
from cloudwater_ml.models import available_models, model_metadata
available_models()   # ['cnn', 'convlstm', 'lightgbm', 'mlp', 'multitask', 'random_forest', 'unet', 'xgboost']
model_metadata()     # {name: {kind, status, is_torch, module, summary}}
```

---

## 3. 模型清单与输入契约

| 模型 | 类型 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| `mlp` | torch | `[B, F]` | `[B, T]` | 可配隐藏层/激活/Dropout，输出层恒线性 |
| `random_forest` | sklearn | `[B, F]` | `[B, T]` | 单/多目标，含特征重要性 |
| `lightgbm` | LightGBM | `[B, F]` | `[B, T]` | 逐目标独立模型 + early stopping |
| `xgboost` | XGBoost | `[B, F]` | `[B, T]` | 逐目标独立模型（可选依赖） |
| `cnn` | torch | `[B, C, H, W]` | `[B, T]` | 2D 卷积特征提取 + 池化读出 |
| `unet` | torch | `[B, C, H, W]` | `[B, out, H, W]` | 编码-解码 + skip connections |
| `convlstm` | torch | `[B, T, C, H, W]` | `[B, T_out, C_out, H, W]`（`predict_field`） | 时空预测；`predict` 池化为 `[B, n_targets]` |
| `multitask` | torch | `[B, F]` | `[B, T]` | 共享骨干 + 逐任务头，**逐任务损失权重** |

> 空间模型的 `predict` 与 `predict_field` 分工：契约要求 `predict` 返回二维
> `[n, n_targets]`（供指标/实验层统一消费），完整场通过 `predict_field` 取得。
> 标量读出用**可学习**的线性适配器（`out_channels ≠ n_targets` 时启用），
> 训练与推理使用同一读出路径（避免训练/推理口径不一致）。

---

## 4. 新增模型（三步）

```python
# ① src/cloudwater_ml/models/my_model.py
from pathlib import Path
from typing import Any
import numpy as np
from .base import BaseModel, TrainingResult, register_model

@register_model("my_model", kind="深度学习")
class MyModel(BaseModel):
    is_torch = True

    def __init__(self, hidden: int = 64) -> None:
        super().__init__(hidden=hidden)
        self.net = None

    def fit(self, X, Y, *, X_val=None, Y_val=None, config=None, out_dir=None) -> TrainingResult:
        ...  # 构造网络 → 训练 → self.is_fitted = True
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...  # 必须返回 [n_samples, n_targets]

    def _save_weights(self, path: Path) -> None:
        ...  # 存权重 + 结构

    @classmethod
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> "MyModel":
        ...  # 按 meta 还原结构并存回权重
```

```python
# ② src/cloudwater_ml/models/__init__.py
register_lazy_model("my_model", "cloudwater_ml.models.my_model")
```

```python
# ③ tests/test_my_model.py —— 至少四个测试
#   1) 前向形状  2) 梯度存在  3) save→load→predict 一致  4) 最小训练损失下降
```

---

## 5. 每个模型必须满足的测试

| 测试 | 断言 |
|---|---|
| 前向形状 | 输出形状与文档契约一致 |
| 梯度 | `loss.backward()` 后参数有非 None 梯度 |
| 存取往返 | `save` → `load_model` → `predict` 与原模型一致 |
| 最小训练 | 若干 epoch 后训练损失下降 |
| 注册 | `build_model(name)` 可构造；未知模型名报清晰错误 |

`tests/test_models_v2.py` 覆盖了上述项；`tests/test_models.py` 守住注册表契约。

---

## 6. 模型持久化格式

```
model/
├── model_meta.json     # format_version / name / class / params / feature_names / target_names / is_torch
├── model.pt            # torch 模型：state_dict + 结构信息
├── model.pkl           # sklearn 基线
└── model_target_*.txt  # LightGBM / XGBoost 逐目标模型
```

`load_model(path)` 依 `model_meta.json` 里的 `name` 走注册表还原，
因此 `cloudwater evaluate --experiment <id>` 能恢复任意已保存模型。

---

## 7. 参数与可复现

- `model.params` 会写进 `metrics.json`，是复现的关键信息；
- 参数必须是 JSON 可序列化的（列表/数字/字符串/布尔）；
- 未生效的参数属于**实现缺陷**（例：v1.0 的 LightGBM `subsample` 因缺 `subsample_freq`
  而静默失效——v2.0 已修）。
