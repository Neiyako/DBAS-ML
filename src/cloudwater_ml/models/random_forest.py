"""随机森林回归基线（sklearn）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from .base import BaseModel, TrainingResult, register_model

__all__ = ["RandomForestModel"]


@register_model("random_forest")
class RandomForestModel(BaseModel):
    """随机森林回归：多目标时逐目标独立森林（MultiOutputRegressor）。

    Parameters
    ----------
    n_estimators / max_depth / min_samples_leaf / n_jobs / random_state
        : 原样透传给 ``RandomForestRegressor``。
        受限环境（joblib 多进程受限的沙箱）下 ``n_jobs=-1`` 可能段错误，
        此时请显式传 ``n_jobs=1``。
    """

    is_torch = False

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, n_jobs=n_jobs,
            random_state=random_state,
        )
        self.model: Any = None

    def _make(self) -> Any:
        p = self.params
        rf = RandomForestRegressor(
            n_estimators=int(p["n_estimators"]), max_depth=p["max_depth"],
            min_samples_leaf=int(p["min_samples_leaf"]), n_jobs=int(p["n_jobs"]),
            random_state=int(p["random_state"]),
        )
        return rf

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
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n_targets = Y.shape[1]
        if feature_names and not self.feature_names:
            self.feature_names = list(feature_names)
        if target_names and not self.target_names:
            self.target_names = list(target_names)

        rf = self._make()
        if n_targets > 1:
            self.model = MultiOutputRegressor(rf, n_jobs=int(self.params["n_jobs"]))
        else:
            self.model = rf
        self.model.fit(X, Y if n_targets > 1 else Y.ravel())
        self.is_fitted = True

        # sklearn 无逐 epoch 训练循环：单行历史，train_loss=训练集 MSE（不编造曲线）
        pred = self.predict(X)
        train_mse = float(np.mean((pred - Y) ** 2))
        extra: dict[str, Any] = {
            "note": "sklearn 无验证循环，X_val 被忽略；history 为单行(epoch=0)，"
                    "train_loss=训练集 MSE，非逐 epoch 损失",
            "n_targets": int(n_targets),
            "x_val_ignored": X_val is not None,
        }
        return TrainingResult(
            history=[{"epoch": 0, "train_loss": train_mse, "val_loss": None}],
            best_epoch=0,
            best_val_loss=None,
            stopped_early=False,
            extra=extra,
        )

    # -- 推理 ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("random_forest 尚未训练，请先调用 fit()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.asarray(self.model.predict(X), dtype=np.float64)
        return out.reshape(len(X), -1)

    # -- 特征重要性 ----------------------------------------------------
    def feature_importance(self) -> dict[str, float] | None:
        if not self.is_fitted or self.model is None:
            return None
        if isinstance(self.model, MultiOutputRegressor):
            imp = np.mean(
                [est.feature_importances_ for est in self.model.estimators_], axis=0
            )
        else:
            imp = np.asarray(self.model.feature_importances_, dtype=np.float64)
        names = self.feature_names or [f"feature_{i}" for i in range(len(imp))]
        return {str(n): float(v) for n, v in zip(names, imp)}

    # -- 持久化 --------------------------------------------------------
    def _save_weights(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("random_forest 尚未训练，无可保存的权重")
        joblib.dump(self.model, path / "model.joblib")

    @classmethod
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> "RandomForestModel":
        params = dict(meta.get("params") or {})
        known = {k: params[k] for k in
                 ("n_estimators", "max_depth", "min_samples_leaf", "n_jobs", "random_state")
                 if k in params}
        model = cls(**known)
        model.model = joblib.load(path / "model.joblib")
        model.is_fitted = bool(meta.get("is_fitted", True))
        return model
