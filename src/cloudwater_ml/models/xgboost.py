"""XGBoost 梯度提升基线（可选依赖：pip install 'cloudwater-ml[xgboost]'）。

与 :mod:`cloudwater_ml.models.lightgbm` 同风格：**多目标 = 逐目标独立模型**，
逐目标训练后拼接输出，支持特征重要性与目录级持久化。

可选依赖处理（与 LightGBM 一致）
--------------------------------
模块 import **不因缺依赖而失败**：缺 xgboost 时 ``xgb = None``，只在**构造
模型时**抛 ``RuntimeError`` 并给出安装提示。这样 ``available_models()`` 与
``build_model("xgboost")`` 在未装 xgboost 的环境里依然可用（能拿到清晰报错，
而不是 ImportError 堆栈）。

**依赖隔离**：本模块**不 import torch**（macOS 上 torch 与 LightGBM/XGBoost 的
OpenMP 运行库不能同进程共存，见 :mod:`cloudwater_ml.models.base`）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseModel, TrainingResult, register_model

__all__ = ["XGBoostModel"]

try:  # 缺依赖时本模块仍可导入，只在构造/fit 时抛 RuntimeError
    import xgboost as xgb
except ImportError:  # pragma: no cover - 取决于环境
    xgb = None  # type: ignore[assignment]

_XGB_INSTALL_MSG = (
    "XGBoost 未安装，请 pip install 'cloudwater-ml[xgboost]'（或 pip install xgboost）后重试"
)


def _require_xgb() -> Any:
    """确保 xgboost 可用，否则给出可执行的安装提示。"""
    if xgb is None:
        raise RuntimeError(_XGB_INSTALL_MSG)
    return xgb


@register_model("xgboost", kind="传统机器学习")
class XGBoostModel(BaseModel):
    """逐目标独立 ``XGBRegressor``（多目标=每目标一个模型）。

    Parameters
    ----------
    n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=1.0, reg_lambda=1.0,
    random_state=42, n_jobs=None, verbosity=0, tree_method="hist",
    importance_type="gain"（feature_importance 用的重要性类型，跨目标平均）。
    """

    is_torch = False

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int | None = None,
        verbosity: int = 0,
        tree_method: str = "hist",
        importance_type: str = "gain",
    ) -> None:
        _require_xgb()
        super().__init__(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            min_child_weight=float(min_child_weight),
            reg_lambda=float(reg_lambda),
            random_state=int(random_state),
            n_jobs=None if n_jobs is None else int(n_jobs),
            verbosity=int(verbosity),
            tree_method=str(tree_method),
            importance_type=str(importance_type),
        )
        self.models: list[Any] = []

    def _make(self, early_stopping_rounds: int | None = None) -> Any:
        """构造一个 XGBRegressor（可选开启早停）。"""
        p = self.params
        kwargs: dict[str, Any] = {
            "n_estimators": int(p["n_estimators"]),
            "learning_rate": float(p["learning_rate"]),
            "max_depth": int(p["max_depth"]),
            "subsample": float(p["subsample"]),
            "colsample_bytree": float(p["colsample_bytree"]),
            "min_child_weight": float(p["min_child_weight"]),
            "reg_lambda": float(p["reg_lambda"]),
            "random_state": int(p["random_state"]),
            "n_jobs": p["n_jobs"],
            "verbosity": int(p["verbosity"]),
            "tree_method": str(p["tree_method"]),
        }
        if early_stopping_rounds:
            kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        return _require_xgb().XGBRegressor(**kwargs)

    # -- 训练 ----------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
        out_dir: str | Path | None = None,
        feature_names: list[str] | None = None,
        target_names: list[str] | None = None,
    ) -> TrainingResult:
        """逐目标训练独立模型。

        有验证集且配置允许早停时使用 XGBoost 原生早停
        （``early_stopping_rounds=patience``）。
        """
        _require_xgb()
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n_targets = int(Y.shape[1])
        if feature_names and not self.feature_names:
            self.feature_names = list(feature_names)
        if target_names and not self.target_names:
            self.target_names = list(target_names)

        tr_cfg = (config or {}).get("training") or {}
        es_cfg = tr_cfg.get("early_stopping") or {}
        patience = int(es_cfg.get("patience", 50))
        use_es = (
            X_val is not None and Y_val is not None and len(X_val) > 0
            and bool(es_cfg.get("enabled", True)) and patience > 0
        )
        Xv = np.asarray(X_val, dtype=np.float64) if use_es else None
        Yv = np.asarray(Y_val, dtype=np.float64) if use_es else None
        if Yv is not None and Yv.ndim == 1:
            Yv = Yv.reshape(-1, 1)

        self.models = []
        val_mses: list[float] = []
        for i in range(n_targets):
            y = Y[:, i]
            if use_es:
                m = self._make(early_stopping_rounds=patience)
                try:  # xgboost>=1.6 支持 eval_set + verbose
                    m.fit(X, y, eval_set=[(Xv, Yv[:, i])], verbose=False)
                except TypeError:  # 老版本没有 verbose 参数
                    m.fit(X, y, eval_set=[(Xv, Yv[:, i])])
                val_mses.append(float(np.mean((m.predict(Xv) - Yv[:, i]) ** 2)))
            else:
                m = self._make()
                m.fit(X, y)
            self.models.append(m)

        # sklearn 风格：无逐 epoch 循环 → 单行历史，损失为训练集 MSE（不编造曲线）
        self.is_fitted = True
        pred = self.predict(X)
        train_mse = float(np.mean((pred - Y) ** 2))
        extra: dict[str, Any] = {
            "note": "XGBoost 无逐 epoch 训练循环；history 为单行(epoch=0)，"
                    "train_loss=训练集 MSE，val_loss=早停验证集 MSE(可选)",
            "n_targets": int(n_targets),
            "early_stopping": use_es,
            "importance_type": str(self.params["importance_type"]),
        }
        if use_es:
            extra["best_iterations"] = [
                int(getattr(m, "best_iteration", 0) or 0) for m in self.models
            ]
            extra["val_mse"] = val_mses
        return TrainingResult(
            history=[{
                "epoch": 0,
                "train_loss": train_mse,
                "val_loss": float(np.mean(val_mses)) if val_mses else None,
            }],
            best_epoch=0,
            best_val_loss=float(np.mean(val_mses)) if val_mses else None,
            stopped_early=False,
            extra=extra,
        )

    # -- 推理 ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 ``[n_samples, n_targets]``。"""
        if not self.is_fitted or not self.models:
            raise RuntimeError("xgboost 尚未训练，请先调用 fit()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        cols = [np.asarray(m.predict(X), dtype=np.float64).reshape(-1) for m in self.models]
        return np.column_stack(cols)

    # -- 特征重要性 ----------------------------------------------------
    def feature_importance(self) -> dict[str, float] | None:
        """跨目标平均 importance（默认 gain）。

        XGBoost 的 ``get_score`` 只返回**被使用过**的特征，因此这里按
        ``f0…fN`` 映射回特征下标，未使用的特征补 0，保证返回长度与特征数一致。
        """
        if not self.is_fitted or not self.models:
            return None
        itype = str(self.params.get("importance_type", "gain"))
        n_features = int(self.models[0].get_booster().num_feature())
        per_target: list[np.ndarray] = []
        for m in self.models:
            imp = np.zeros(n_features, dtype=np.float64)
            scores = m.get_booster().get_score(importance_type=itype)
            for key, value in scores.items():
                idx = _feature_index(key)
                if idx is not None and 0 <= idx < n_features:
                    imp[idx] = float(value)
            per_target.append(imp)
        imp = np.mean(per_target, axis=0)
        names = self.feature_names or [f"feature_{i}" for i in range(len(imp))]
        return {str(n): float(v) for n, v in zip(names, imp)}

    # -- 持久化 --------------------------------------------------------
    def _save_weights(self, path: Path) -> None:
        if not self.models:
            raise RuntimeError("xgboost 尚未训练，无可保存的权重")
        for i, m in enumerate(self.models):
            m.get_booster().save_model(str(path / f"model_target_{i}.json"))

    @classmethod
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> "XGBoostModel":
        _require_xgb()
        params = dict(meta.get("params") or {})
        known_keys = (
            "n_estimators", "learning_rate", "max_depth", "subsample",
            "colsample_bytree", "min_child_weight", "reg_lambda",
            "random_state", "n_jobs", "verbosity", "tree_method", "importance_type",
        )
        model = cls(**{k: params[k] for k in known_keys if k in params})
        files = sorted(
            path.glob("model_target_*.json"),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        )
        model.models = []
        for f in files:
            booster = xgb.Booster()
            booster.load_model(str(f))
            est = model._make()
            est._Booster = booster  # 直接挂载 booster（与 LightGBM 的做法同构）
            est.n_features_in_ = booster.num_feature()
            model.models.append(est)
        model.is_fitted = bool(meta.get("is_fitted", True)) and bool(model.models)
        return model


def _feature_index(key: str) -> int | None:
    """把 booster 的特征名（``f0``/``f12``）解析成特征下标。"""
    name = str(key)
    if name.startswith("f") and name[1:].isdigit():
        return int(name[1:])
    return None
