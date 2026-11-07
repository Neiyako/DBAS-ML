"""LightGBM 梯度提升基线（可选依赖：pip install 'cloudwater-ml[lgbm]'）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseModel, TrainingResult, register_model

__all__ = ["LightGBMModel"]

try:  # 缺依赖时本模块仍可导入，只在构造/fit 时抛 RuntimeError
    import lightgbm as lgb
except ImportError:  # pragma: no cover - 取决于环境
    lgb = None  # type: ignore[assignment]

_LGBM_INSTALL_MSG = (
    "LightGBM 未安装，请 pip install 'cloudwater-ml[lgbm]'（或 pip install lightgbm）后重试"
)


def _require_lgbm() -> Any:
    if lgb is None:
        raise RuntimeError(_LGBM_INSTALL_MSG)
    return lgb


@register_model("lightgbm", kind="传统机器学习")
class LightGBMModel(BaseModel):
    """逐目标独立 LGBMRegressor（多目标=每目标一个模型）。

    Parameters
    ----------
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    importance_type="gain"（feature_importance 用 gain，跨目标平均）。
    """

    is_torch = False

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        verbose: int = -1,
        importance_type: str = "gain",
    ) -> None:
        _require_lgbm()
        super().__init__(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, subsample=subsample,
            colsample_bytree=colsample_bytree, random_state=random_state,
            verbose=verbose, importance_type=importance_type,
        )
        self.models: list[Any] = []

    def _make(self) -> Any:
        p = self.params
        return _require_lgbm().LGBMRegressor(
            n_estimators=int(p["n_estimators"]),
            learning_rate=float(p["learning_rate"]),
            num_leaves=int(p["num_leaves"]),
            subsample=float(p["subsample"]),
            colsample_bytree=float(p["colsample_bytree"]),
            random_state=int(p["random_state"]),
            verbose=int(p["verbose"]),
        )

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
        _require_lgbm()
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n_targets = Y.shape[1]
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
            m = self._make()
            y = Y[:, i]
            if use_es:
                callback = lgb.early_stopping(patience, verbose=False)
                try:  # lightgbm>=4.3 推荐 eval_X/eval_y；老版本回退 eval_set
                    m.fit(X, y, eval_X=Xv, eval_y=Yv[:, i], callbacks=[callback])
                except TypeError:
                    m.fit(X, y, eval_set=[(Xv, Yv[:, i])], callbacks=[callback])
                val_mses.append(float(np.mean((m.predict(Xv) - Yv[:, i]) ** 2)))
            else:
                m.fit(X, y)
            self.models.append(m)

        # sklearn 风格：无逐 epoch 循环 → 单行历史，损失为训练集 MSE（不编造曲线）
        self.is_fitted = True
        pred = self.predict(X)
        train_mse = float(np.mean((pred - Y) ** 2))
        extra: dict[str, Any] = {
            "note": "LightGBM 无逐 epoch 训练循环；history 为单行(epoch=0)，"
                    "train_loss=训练集 MSE，val_loss=早停验证集 MSE(可选)",
            "n_targets": int(n_targets),
            "early_stopping": use_es,
            "importance_type": str(self.params["importance_type"]),
        }
        if use_es:
            extra["best_iterations"] = [int(getattr(m, "best_iteration_", 0)) for m in self.models]
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
        if not self.is_fitted or not self.models:
            raise RuntimeError("lightgbm 尚未训练，请先调用 fit()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        cols = [np.asarray(m.predict(X), dtype=np.float64).reshape(-1) for m in self.models]
        return np.column_stack(cols)

    # -- 特征重要性 ----------------------------------------------------
    def feature_importance(self) -> dict[str, float] | None:
        """跨目标平均 importance（importance_type 默认 gain，见 params）。"""
        if not self.is_fitted or not self.models:
            return None
        itype = str(self.params.get("importance_type", "gain"))
        imps = [
            np.asarray(m.booster_.feature_importance(importance_type=itype), dtype=np.float64)
            for m in self.models
        ]
        imp = np.mean(imps, axis=0)
        names = self.feature_names or [f"feature_{i}" for i in range(len(imp))]
        return {str(n): float(v) for n, v in zip(names, imp)}

    # -- 持久化 --------------------------------------------------------
    def _save_weights(self, path: Path) -> None:
        if not self.models:
            raise RuntimeError("lightgbm 尚未训练，无可保存的权重")
        for i, m in enumerate(self.models):
            (path / f"model_target_{i}.txt").write_text(
                m.booster_.model_to_string(), encoding="utf-8"
            )

    @classmethod
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> "LightGBMModel":
        _require_lgbm()
        params = dict(meta.get("params") or {})
        known_keys = ("n_estimators", "learning_rate", "num_leaves", "subsample",
                      "colsample_bytree", "random_state", "verbose", "importance_type")
        model = cls(**{k: params[k] for k in known_keys if k in params})
        files = sorted(path.glob("model_target_*.txt"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        model.models = []
        for f in files:
            est = model._make()
            est._Booster = lgb.Booster(model_str=f.read_text(encoding="utf-8"))
            est.n_features_in_ = est._Booster.num_feature()
            est.fitted_ = True  # sklearn 接口的 fitted 标记（lightgbm 4.x）
            model.models.append(est)
        model.is_fitted = bool(meta.get("is_fitted", True)) and bool(model.models)
        return model
