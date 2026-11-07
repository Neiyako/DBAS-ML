"""模型层：统一接口 + **惰性加载**。

惰性加载有两个实际收益：

1. 轻量：只 import 真正用到的模型模块；
2. **避免 torch 与 LightGBM 的 OpenMP 运行库冲突**：macOS 实测「先 import torch
   再 fit LightGBM」会触发 ``Abort trap: 6``（原生崩溃），因此跑 LightGBM 实验时
   本包完全不加载 torch（``build_model("lightgbm")`` 只 import models.lightgbm）。

模型契约见 :mod:`cloudwater_ml.models.base`：

- 已实现并注册：mlp / random_forest / lightgbm / cnn / unet / convlstm /
  multitask / xgboost（lightgbm 与 xgboost 为可选依赖，模块导入不因缺依赖失败，
  只在构造时抛 RuntimeError 并给出安装提示）；
- 场模型（cnn / unet / convlstm）为 PyTorch 实现，共享
  :class:`cloudwater_ml.models.cnn.TensorTrainer` 的张量级训练循环；
- 树模型模块（lightgbm / xgboost）**不 import torch**，torch 模型模块
  **不 import lightgbm / xgboost**，两类运行库不会同进程共存。

新增模型：在对应模块里用 ``@register_model("name")`` 注册，并在下方登记惰性映射。
"""
from __future__ import annotations

import importlib
from typing import Any

from .base import (
    MODEL_REGISTRY,
    BaseModel,
    TrainingResult,
    available_models,
    build_model,
    load_model,
    register_lazy_model,
    register_model,
)

# 惰性映射：模型名 -> 模块路径（首次使用时才 import）
register_lazy_model("mlp", "cloudwater_ml.models.mlp")
register_lazy_model("random_forest", "cloudwater_ml.models.random_forest")
register_lazy_model("lightgbm", "cloudwater_ml.models.lightgbm")
# v2.0：场模型 / 时空模型（PyTorch）
register_lazy_model("cnn", "cloudwater_ml.models.cnn")
register_lazy_model("unet", "cloudwater_ml.models.unet")
register_lazy_model("convlstm", "cloudwater_ml.models.convlstm")
register_lazy_model("multitask", "cloudwater_ml.models.multitask")
# v2.0：XGBoost 基线（可选依赖，模块 import 不依赖 xgboost 是否安装）
register_lazy_model("xgboost", "cloudwater_ml.models.xgboost")

# 类对象同样惰性暴露，避免为拿类而加载 torch
_LAZY_CLASSES = {
    "CNN": "cloudwater_ml.models.cnn",
    "UNet": "cloudwater_ml.models.unet",
    "ConvLSTM": "cloudwater_ml.models.convlstm",
    "MultiTask": "cloudwater_ml.models.multitask",
    "XGBoostModel": "cloudwater_ml.models.xgboost",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LAZY_CLASSES:
        module = importlib.import_module(_LAZY_CLASSES[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MODEL_REGISTRY",
    "BaseModel",
    "TrainingResult",
    "available_models",
    "build_model",
    "load_model",
    "register_lazy_model",
    "register_model",
    "CNN",
    "UNet",
    "ConvLSTM",
    "MultiTask",
    "XGBoostModel",
]
