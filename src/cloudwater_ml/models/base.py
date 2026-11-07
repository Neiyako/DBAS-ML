"""模型统一接口与注册表（所有模型的契约）。

设计要点：

- 无论 PyTorch 模型还是 sklearn/LightGBM 基线，都实现同一套
  :class:`BaseModel`：``fit`` / ``predict`` / ``feature_importance`` / ``save`` / ``load``；
- ``predict`` 恒返回 ``[n_samples, n_targets]`` 二维数组（单目标列数为 1）；
- ``save`` 采用模板方法：公共元数据（模型名、类、参数、特征/目标名）写
  ``model_meta.json``，权重由子类 ``_save_weights`` 落盘；``load_model`` 依据
  注册表还原，保证 evaluate/report 能恢复任意已保存模型；
- 未来模型（CNN/U-Net/ConvLSTM/MultiTask）只需注册并实现本接口即可接入全链路。
"""
from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BaseModel",
    "TrainingResult",
    "ModelInfo",
    "MODEL_METADATA",
    "register_model",
    "build_model",
    "load_model",
    "available_models",
    "model_metadata",
]

# ----------------------------------------------------------------------
# 训练结果契约（Trainer 与 sklearn fit 都产出同构结果）
# ----------------------------------------------------------------------
@dataclass
class TrainingResult:
    """统一训练结果：每个 epoch 一行历史 + 最佳点 + 提前停止标记。"""

    history: list[dict[str, Any]] = field(default_factory=list)
    # history 行约定键：epoch, train_loss, val_loss（sklearn 基线可为单行 epoch=0）
    best_epoch: int | None = None
    best_val_loss: float | None = None
    stopped_early: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def final_train_loss(self) -> float | None:
        return self.history[-1].get("train_loss") if self.history else None


# ----------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[BaseModel]] = {}
# 惰性模型：name -> 模块路径（首次使用时才 import）
# 原因：macOS 上 torch 与 LightGBM 的 OpenMP 运行库不能在同一进程共存
# （先 import torch 再 fit LightGBM → Abort trap: 6），惰性加载让 LightGBM
# 实验完全不触碰 torch。
_LAZY_MODULES: dict[str, str] = {}


def register_lazy_model(name: str, module: str) -> None:
    """登记惰性模型：``name`` 首次使用时才 import ``module``。"""
    _LAZY_MODULES[name] = module


def _ensure_loaded(name: str) -> None:
    if name in MODEL_REGISTRY:
        return
    module = _LAZY_MODULES.get(name)
    if module:
        importlib.import_module(module)


@dataclass
class ModelInfo:
    """注册表元数据（CLI 展示用，避免在 CLI 里硬编码「已实现」状态）。"""

    name: str
    kind: str = "深度学习"
    status: str = "已实现"
    is_torch: bool = False
    module: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "is_torch": self.is_torch,
            "module": self.module,
            "summary": self.summary,
        }


#: 注册表元数据（name -> ModelInfo）
MODEL_METADATA: dict[str, ModelInfo] = {}


def register_model(name: str, *, kind: str = "深度学习", status: str = "已实现"):
    """模型注册装饰器：``@register_model("mlp")``。

    ``kind`` / ``status`` 是**注册时声明**的元数据，CLI 直接读注册表，
    不再硬编码一张会过期的「已实现」表。
    """

    def deco(cls: type[BaseModel]) -> type[BaseModel]:
        cls.name = name
        cls.kind = kind
        MODEL_REGISTRY[name] = cls
        doc = (cls.__doc__ or "").strip().splitlines()
        MODEL_METADATA[name] = ModelInfo(
            name=name,
            kind=kind,
            status=status,
            is_torch=bool(getattr(cls, "is_torch", False)),
            module=cls.__module__,
            summary=doc[0].strip() if doc else "",
        )
        return cls

    return deco


def model_metadata() -> dict[str, dict[str, Any]]:
    """返回注册表元数据（{name: {...}}）。"""
    return {name: info.to_dict() for name, info in MODEL_METADATA.items()}


def available_models() -> list[str]:
    return sorted(set(MODEL_REGISTRY) | set(_LAZY_MODULES))


def build_model(name: str, params: dict[str, Any] | None = None) -> BaseModel:
    _ensure_loaded(name)
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"未知模型 {name!r}，已注册: {available_models()}。"
            "预留模型(cnn/unet/convlstm/multitask)需第二阶段实现后注册。"
        )
    return MODEL_REGISTRY[name](**(params or {}))


# ----------------------------------------------------------------------
# 基类
# ----------------------------------------------------------------------
class BaseModel(ABC):
    """模型统一接口。子类实现 ``fit``/``predict``/``_save_weights``/``_load_from``。"""

    name: str = "base"
    is_torch: bool = False  # True = PyTorch 模型（走 Trainer），False = sklearn 风格

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.feature_names: list[str] = []
        self.target_names: list[str] = []
        self.is_fitted: bool = False

    # -- 训练 ----------------------------------------------------------
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
        out_dir: str | Path | None = None,
    ) -> TrainingResult:
        """训练。config 为完整实验配置（取 training 段），out_dir 为实验 model/ 目录。"""

    # -- 推理 ----------------------------------------------------------
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 [n_samples, n_targets] 的二维预测（单目标列数为 1）。"""

    def feature_importance(self) -> dict[str, float] | None:
        """特征重要性：{特征名: 重要性}；不适用的模型返回 None（不得编造）。"""
        return None

    # -- 持久化（模板方法） --------------------------------------------
    def save(self, path: str | Path) -> None:
        """保存：model_meta.json + 子类权重。path 为目录。"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "format_version": 1,
            "name": self.name,
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "params": self.params,
            "feature_names": list(self.feature_names),
            "target_names": list(self.target_names),
            "is_torch": bool(self.is_torch),
            "is_fitted": bool(self.is_fitted),
        }
        (path / "model_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._save_weights(path)

    @abstractmethod
    def _save_weights(self, path: Path) -> None:
        """子类落权重文件（如 model.pt / model.pkl / model.txt）。"""

    @classmethod
    @abstractmethod
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> BaseModel:
        """子类从目录还原实例（含权重与 feature/target names）。"""

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        """从保存目录还原模型（等价于模块级 load_model）。"""
        return load_model(path)


def load_model(path: str | Path) -> BaseModel:
    """按 model_meta.json + 注册表还原任意已保存模型。"""
    path = Path(path)
    meta_path = path / "model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"不是有效的模型保存目录（缺 model_meta.json）: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    name = meta.get("name")
    _ensure_loaded(name or "")
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"模型 {name!r} 未注册（当前: {available_models()}），无法加载 {path}"
        )
    model = MODEL_REGISTRY[name]._load_from(path, meta)
    model.feature_names = list(meta.get("feature_names") or [])
    model.target_names = list(meta.get("target_names") or [])
    model.params = dict(meta.get("params") or {})
    model.is_fitted = bool(meta.get("is_fitted", True))
    return model
