"""归一化 / 预处理。

铁律：**归一化参数只能从训练集拟合**，再应用到验证集与测试集。
本模块只提供 scaler 与拟合入口；把 fit 用在整个数据集上的调用方式是错误用法，
:func:`fit_scalers` 只接收训练 split 的数组。

v2.0 修正：

- **D6 拟合证据**：`fit_scalers` 记录拟合证据（样本数、特征名、拟合时间、
  schema 版本、调用方声明的数据来源标识、训练数组指纹），并写入
  ``state_dict()``，可用 :func:`verify_fit_evidence` 校验「确实只用了训练集」；
- **D7 非有限值策略**：fit 时检测 NaN/Inf，默认**拒绝**并给出明确的列名与数量；
  可显式配置 ``nan_policy="impute"``（用训练集中位数填补，统计量仍只来自训练集）。
"""
from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .schema import DATA_SCHEMA_VERSION

__all__ = [
    "Scaler",
    "StandardScaler",
    "MinMaxScaler",
    "NoScaler",
    "make_scaler",
    "fit_scalers",
    "Preprocessors",
    "verify_fit_evidence",
    "NAN_POLICY_ERROR",
    "NAN_POLICY_IMPUTE",
]

#: 非有限值策略：拒绝（默认）/ 用训练集中位数填补
NAN_POLICY_ERROR = "error"
NAN_POLICY_IMPUTE = "impute"


def _column_label(names: Sequence[str] | None, i: int) -> str:
    if names is not None and i < len(names):
        return str(names[i])
    return f"col{i}"


def _detect_nonfinite(X: np.ndarray, names: Sequence[str] | None) -> dict[str, dict[str, int]]:
    """统计各列 NaN/Inf 数量（只返回有问题的列）。"""
    detail: dict[str, dict[str, int]] = {}
    for i in range(X.shape[1]):
        col = X[:, i]
        nan = int(np.isnan(col).sum())
        inf = int(np.isinf(col).sum())
        if nan or inf:
            detail[_column_label(names, i)] = {"nan": nan, "inf": inf, "total": nan + inf}
    return detail


def _format_nonfinite(detail: dict[str, dict[str, int]]) -> str:
    return ", ".join(f"{k}(NaN={v['nan']}, Inf={v['inf']})" for k, v in detail.items())


def _fingerprint(arr: np.ndarray) -> str:
    """训练数组内容指纹（sha256 前 16 字节十六进制）——用于校验「只用了训练集」。"""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:32]


class Scaler:
    """逐列归一化基类：fit 只应在训练集上调用一次。

    子类的 ``fit`` 统一支持：

    - ``names``：列名（用于报错时给出「哪一列」）；
    - ``nan_policy``：``"error"``（默认，拒绝 NaN/Inf）或 ``"impute"``（训练集中位数填补）。
    """

    method = "base"

    def fit(self, X: np.ndarray, *, names: Sequence[str] | None = None,
            nan_policy: str = NAN_POLICY_ERROR) -> Scaler:  # pragma: no cover - 抽象
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - 抽象
        raise NotImplementedError

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - 抽象
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def state_dict(self) -> dict[str, Any]:  # pragma: no cover - 抽象
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Scaler:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    def _prepare_fit_matrix(
        self,
        X: np.ndarray,
        names: Sequence[str] | None,
        nan_policy: str,
    ) -> np.ndarray:
        """校验/清洗训练矩阵（唯一的 NaN/Inf 入口）。

        返回清洗后的矩阵；非有限值明细记录在 ``self.nonfinite_report_``。
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.fit_names_ = [str(n) for n in names] if names is not None else None
        self.n_fit_samples_ = int(X.shape[0])
        self.impute_values_ = None
        finite = np.isfinite(X)
        if bool(finite.all()):
            self.nonfinite_report_ = {}
            return X
        self.nonfinite_report_ = _detect_nonfinite(X, names)
        if nan_policy == NAN_POLICY_ERROR:
            raise ValueError(
                "训练数据含非有限值（NaN/Inf），拒绝拟合归一化参数（默认拒绝，不静默丢弃）。"
                f" 列名与数量: {_format_nonfinite(self.nonfinite_report_)}。"
                "处理方式：先修复/剔除缺测样本（quality.cleaning.dropna），"
                "或显式设置 nan_policy='impute'（用训练集中位数填补，统计量仍只来自训练集）。"
            )
        if nan_policy != NAN_POLICY_IMPUTE:
            raise ValueError(f"未知 nan_policy={nan_policy!r}，可选: {[NAN_POLICY_ERROR, NAN_POLICY_IMPUTE]}")
        # impute：NaN 与 ±Inf 均按缺失处理，用**训练集**中位数填补
        X = X.copy()
        X[~finite] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0) if X.size else np.zeros(X.shape[1])
        med = np.where(np.isfinite(med), med, 0.0)
        self.impute_values_ = med
        nan_mask = np.isnan(X)
        if nan_mask.any():
            rows, cols = np.where(nan_mask)
            X[rows, cols] = med[cols]
        return X

    def _apply_impute(self, X: np.ndarray) -> np.ndarray:
        """transform 阶段使用与 fit 相同的填补值（保证训练/推理一致）。"""
        if self.impute_values_ is None:
            return X
        X = np.asarray(X, dtype=np.float64)
        bad = ~np.isfinite(X)
        if not bad.any():
            return X
        X = X.copy()
        X[bad] = np.nan
        nan_mask = np.isnan(X)
        if nan_mask.any():
            rows, cols = np.where(nan_mask)
            X[rows, cols] = np.asarray(self.impute_values_)[cols]
        return X


class NoScaler(Scaler):
    """不做缩放（method=none）。

    没有统计量 → 不会污染列；但仍记录非有限值明细（供报告），不因此拒绝。
    """

    method = "none"

    def __init__(self) -> None:
        self.fit_names_: list[str] | None = None
        self.n_fit_samples_: int = 0
        self.impute_values_: np.ndarray | None = None
        self.nonfinite_report_: dict[str, dict[str, int]] = {}

    def fit(self, X: np.ndarray, *, names: Sequence[str] | None = None,
            nan_policy: str = NAN_POLICY_ERROR) -> NoScaler:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.fit_names_ = [str(n) for n in names] if names is not None else None
        self.n_fit_samples_ = int(X.shape[0])
        self.nonfinite_report_ = _detect_nonfinite(X, names)
        self.impute_values_ = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "fit_names": self.fit_names_,
            "n_fit_samples": self.n_fit_samples_,
            "nonfinite": self.nonfinite_report_,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> NoScaler:
        obj = cls()
        obj.fit_names_ = state.get("fit_names")
        obj.n_fit_samples_ = int(state.get("n_fit_samples") or 0)
        obj.nonfinite_report_ = state.get("nonfinite") or {}
        return obj


class StandardScaler(Scaler):
    """零均值单位方差：mean/std 逐列；std=0 的列按 1.0 处理（避免除零）。"""

    method = "standard"

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.fit_names_: list[str] | None = None
        self.n_fit_samples_: int = 0
        self.impute_values_: np.ndarray | None = None
        self.nonfinite_report_: dict[str, dict[str, int]] = {}

    def fit(self, X: np.ndarray, *, names: Sequence[str] | None = None,
            nan_policy: str = NAN_POLICY_ERROR) -> StandardScaler:
        X = self._prepare_fit_matrix(X, names, nan_policy)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        return self

    def _check(self, X: np.ndarray) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler 未 fit 就被调用（归一化参数只能来自训练集拟合）")

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check(X)
        X = self._apply_impute(np.asarray(X, dtype=np.float64))
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check(X)
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.mean_

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
            "fit_names": self.fit_names_,
            "n_fit_samples": self.n_fit_samples_,
            "impute_values": None if self.impute_values_ is None else list(map(float, self.impute_values_)),
            "nonfinite": self.nonfinite_report_,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> StandardScaler:
        obj = cls()
        if state.get("mean") is not None:
            obj.mean_ = np.asarray(state["mean"], dtype=np.float64)
            obj.scale_ = np.asarray(state["scale"], dtype=np.float64)
        obj.fit_names_ = state.get("fit_names")
        obj.n_fit_samples_ = int(state.get("n_fit_samples") or 0)
        imp = state.get("impute_values")
        obj.impute_values_ = None if imp is None else np.asarray(imp, dtype=np.float64)
        obj.nonfinite_report_ = state.get("nonfinite") or {}
        return obj


class MinMaxScaler(Scaler):
    """线性缩放到 [0,1]（method=minmax）；常数列缩放为 0。"""

    method = "minmax"

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None
        self.fit_names_: list[str] | None = None
        self.n_fit_samples_: int = 0
        self.impute_values_: np.ndarray | None = None
        self.nonfinite_report_: dict[str, dict[str, int]] = {}

    def fit(self, X: np.ndarray, *, names: Sequence[str] | None = None,
            nan_policy: str = NAN_POLICY_ERROR) -> MinMaxScaler:
        X = self._prepare_fit_matrix(X, names, nan_policy)
        self.min_ = X.min(axis=0)
        span = X.max(axis=0) - self.min_
        self.range_ = np.where(span < 1e-12, 1.0, span)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("MinMaxScaler 未 fit")
        X = self._apply_impute(np.asarray(X, dtype=np.float64))
        return (X - self.min_) / self.range_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("MinMaxScaler 未 fit")
        X = np.asarray(X, dtype=np.float64)
        return X * self.range_ + self.min_

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "min": None if self.min_ is None else self.min_.tolist(),
            "range": None if self.range_ is None else self.range_.tolist(),
            "fit_names": self.fit_names_,
            "n_fit_samples": self.n_fit_samples_,
            "impute_values": None if self.impute_values_ is None else list(map(float, self.impute_values_)),
            "nonfinite": self.nonfinite_report_,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> MinMaxScaler:
        obj = cls()
        if state.get("min") is not None:
            obj.min_ = np.asarray(state["min"], dtype=np.float64)
            obj.range_ = np.asarray(state["range"], dtype=np.float64)
        obj.fit_names_ = state.get("fit_names")
        obj.n_fit_samples_ = int(state.get("n_fit_samples") or 0)
        imp = state.get("impute_values")
        obj.impute_values_ = None if imp is None else np.asarray(imp, dtype=np.float64)
        obj.nonfinite_report_ = state.get("nonfinite") or {}
        return obj


_SCALERS: dict[str, type[Scaler]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "none": NoScaler,
}


def make_scaler(method: str = "standard") -> Scaler:
    if method not in _SCALERS:
        raise ValueError(f"未知归一化方法 {method!r}，可选: {sorted(_SCALERS)}")
    return _SCALERS[method]()


@dataclass
class Preprocessors:
    """训练集拟合的 X/Y 两个 scaler + 拟合来源说明（进实验快照，保证可复现）。"""

    x: Scaler
    y: Scaler
    fitted_on: str = "train"  # 只允许 train
    notes: list[str] = field(default_factory=list)
    fit_evidence: dict[str, Any] = field(default_factory=dict)  # D6 拟合证据（可校验）
    schema_version: str = DATA_SCHEMA_VERSION

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return self.x.transform(X)

    def transform_y(self, Y: np.ndarray) -> np.ndarray:
        return self.y.transform(Y)

    def inverse_y(self, Y: np.ndarray) -> np.ndarray:
        return self.y.inverse_transform(Y)

    def state_dict(self) -> dict[str, Any]:
        return {
            "fitted_on": self.fitted_on,
            "schema_version": self.schema_version,
            "x": self.x.state_dict(),
            "y": self.y.state_dict(),
            "notes": self.notes,
            "fit_evidence": self.fit_evidence,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> Preprocessors:
        """从 state_dict 恢复（证据一并恢复，便于事后校验）。"""
        x = make_scaler((state.get("x") or {}).get("method", "standard")).from_state_dict(state.get("x") or {})
        y = make_scaler((state.get("y") or {}).get("method", "standard")).from_state_dict(state.get("y") or {})
        return cls(
            x=x,
            y=y,
            fitted_on=state.get("fitted_on", "train"),
            notes=list(state.get("notes") or []),
            fit_evidence=dict(state.get("fit_evidence") or {}),
            schema_version=state.get("schema_version", DATA_SCHEMA_VERSION),
        )

    def verify_evidence(self) -> dict[str, Any]:
        """校验拟合证据（自证「只用了训练集」）。"""
        return verify_fit_evidence(self.state_dict())


def verify_fit_evidence(state: dict[str, Any]) -> dict[str, Any]:
    """校验 scaler 状态里的拟合证据是否自洽。

    返回 ``{"valid": bool, "problems": [...], "evidence": {...}}``：

    - ``fitted_on`` 必须是 ``train``；
    - ``schema_version`` 必须与当前一致；
    - ``n_samples`` 必须为正；
    - scaler 参数个数必须与证据里的特征名数量一致；
    - 必须带拟合时间与训练数组指纹（指纹缺失 → 无法证明只用了训练集）。
    """
    problems: list[str] = []
    evidence = dict((state or {}).get("fit_evidence") or {})
    if not evidence:
        return {"valid": False, "problems": ["缺少 fit_evidence（无法证明归一化参数只来自训练集）"],
                "evidence": {}}
    if evidence.get("fitted_on") != "train":
        problems.append(f"fitted_on 必须是 'train'，实际 {evidence.get('fitted_on')!r}")
    if evidence.get("schema_version") != DATA_SCHEMA_VERSION:
        problems.append(
            f"schema_version 不一致：证据 {evidence.get('schema_version')!r} != 当前 {DATA_SCHEMA_VERSION!r}"
        )
    n = evidence.get("n_samples")
    if not isinstance(n, int) or n <= 0:
        problems.append(f"n_samples 无效: {n!r}")
    if not evidence.get("fitted_at"):
        problems.append("缺少拟合时间 fitted_at")
    if not evidence.get("x_fingerprint") or not evidence.get("y_fingerprint"):
        problems.append("缺少训练数组指纹（x_fingerprint / y_fingerprint）")
    feat = evidence.get("feature_names")
    x_state = (state or {}).get("x") or {}
    scale = x_state.get("scale")
    if isinstance(feat, list) and isinstance(scale, list) and len(feat) != len(scale):
        problems.append(f"特征名数量 {len(feat)} 与 scaler 参数数量 {len(scale)} 不一致")
    return {"valid": not problems, "problems": problems, "evidence": evidence}


def fit_scalers(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    scale: str | bool = "auto",
    method: str = "standard",
    is_torch: bool = True,
    source: str = "train",
    feature_names: Sequence[str] | None = None,
    target_names: Sequence[str] | None = None,
    nan_policy: str = NAN_POLICY_ERROR,
) -> Preprocessors:
    """只用训练集拟合归一化参数。

    scale: ``auto``（torch 模型缩放，树模型不缩放——树模型对尺度不敏感）|
    ``true`` | ``false``；method: standard|minmax|none。

    source: 调用方声明的数据来源标识（默认 ``"train"``）；写入拟合证据，
    便于事后核对「这些参数确实来自训练集」。

    feature_names / target_names: 列名（写入证据 + NaN 报错时给出具体列名）。

    nan_policy: ``"error"``（默认，含 NaN/Inf 直接报错并列出列名与数量）|
    ``"impute"``（用训练集中位数填补；统计量仍只来自训练集）。
    """
    notes: list[str] = []
    if scale == "auto":
        apply = bool(is_torch)
        notes.append(f"scale=auto → is_torch={is_torch}，应用缩放={apply}")
    else:
        apply = bool(scale)
        notes.append(f"scale={scale}")
    if not apply:
        method = "none"
    Xa = np.asarray(X_train, dtype=np.float64)
    Ya = np.asarray(Y_train, dtype=np.float64)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Ya.ndim == 1:
        Ya = Ya.reshape(-1, 1)
    if Xa.shape[0] != Ya.shape[0]:
        raise ValueError(f"X_train/Y_train 样本数不一致: {Xa.shape[0]} vs {Ya.shape[0]}")

    x_s, y_s = make_scaler(method), make_scaler(method)
    x_s.fit(Xa, names=feature_names, nan_policy=nan_policy)
    y_s.fit(Ya, names=target_names, nan_policy=nan_policy)
    notes.append("scalers 仅在训练集上拟合（fitted_on=train）")

    evidence: dict[str, Any] = {
        "fitted_on": "train",
        "declared_source": str(source),
        "n_samples": int(Xa.shape[0]),
        "n_samples_x": int(Xa.shape[0]),
        "n_samples_y": int(Ya.shape[0]),
        "n_features": int(Xa.shape[1]),
        "n_targets": int(Ya.shape[1]),
        "feature_names": [str(n) for n in (feature_names if feature_names is not None else [])],
        "target_names": [str(n) for n in (target_names if target_names is not None else [])],
        "method": method,
        "scale": bool(apply),
        "nan_policy": nan_policy,
        "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": DATA_SCHEMA_VERSION,
        "x_fingerprint": _fingerprint(Xa),
        "y_fingerprint": _fingerprint(Ya),
        "nonfinite": {
            "X": dict(getattr(x_s, "nonfinite_report_", {}) or {}),
            "Y": dict(getattr(y_s, "nonfinite_report_", {}) or {}),
        },
    }
    if evidence["nonfinite"]["X"] or evidence["nonfinite"]["Y"]:
        notes.append(
            "训练集含非有限值，已按 nan_policy="
            f"{nan_policy} 处理（明细见 fit_evidence.nonfinite）"
        )
    notes.append(
        f"拟合证据: n_samples={evidence['n_samples']}, source={evidence['declared_source']!r}, "
        f"schema={evidence['schema_version']}, fitted_at={evidence['fitted_at']}"
    )
    return Preprocessors(
        x=x_s, y=y_s, fitted_on="train", notes=notes,
        fit_evidence=evidence, schema_version=DATA_SCHEMA_VERSION,
    )
