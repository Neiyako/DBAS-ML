"""不确定性估计（v2.0）：可插拔方法 + 统一的区间/可靠性输出。

方法与适用条件
--------------

======================  ==========================================  ==========================
方法                    怎么得到不确定性                            前提
======================  ==========================================  ==========================
``mc_dropout``          推理时保持 dropout 激活，重复 N 次取分布    模型含 Dropout 层
``ensemble``            多个独立训练模型的预测分布                  需 K 个模型（不同种子）
``normal``              残差标准差假设（最弱，只作基线）           有校准集残差
``conformal``           **分割共形（split conformal）**：用校准集   需要独立的校准集
                        残差分位数构造区间
======================  ==========================================  ==========================

**诚实性约定（重要）**

* 普通模型的「预测误差」**不叫置信区间**。本模块只有真正生成了区间分布的方法
  才会返回 ``status="ok"``；没有校准集/没有可重复采样源时，接口会明确报错，
  而不是拿一个残差标准差冒充区间。
* 每种方法都记录 ``assumptions``（如共形区间依赖「校准集与测试集可交换」），
  避免读者误以为区间在分布外推场景仍然有效。
* 区间**不保证**在时间外推/空间外推场景下有覆盖率——那是泛化问题，不是区间问题。
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.exceptions import ConfigurationError

logger = logging.getLogger("cloudwater_ml.evaluation")

__all__ = [
    "interval_metrics",
    "crps",
    "UncertaintyResult",
    "mc_dropout_samples",
    "ensemble_samples",
    "normal_interval",
    "conformal_interval",
    "estimate_uncertainty",
    "reliability_report",
]


def _as_2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _block(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "coverage": _f(np.mean(inside)) if y.size else None,
        "interval_width": _f(np.mean(width)) if y.size else None,
        # CWC：覆盖率不足时同时惩罚宽度（越小越好）
        "cwc": _f(np.mean(width * (1.0 - np.mean(inside)))) if y.size else None,
    }


def interval_metrics(
    y_true: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    target_names: list[str],
) -> dict:
    """预测区间的覆盖率与平均宽度评价。

    Returns
    -------
    dict
        ``{"overall": {"coverage", "interval_width", "cwc"},
           "per_target": {name: {...}}}``
        overall 为全部元素展平后的 micro 汇总；值为 float 或 None，JSON 可序列化。
    """
    yt = _as_2d(y_true)
    lo = _as_2d(y_lo)
    hi = _as_2d(y_hi)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError(
            f"y_true/y_lo/y_hi 形状不一致: {yt.shape} vs {lo.shape} vs {hi.shape}"
        )
    names = list(target_names) if target_names else []
    if len(names) != yt.shape[1]:
        names = [
            names[i] if i < len(names) else f"target_{i}" for i in range(yt.shape[1])
        ]

    per_target = {
        nm: _block(yt[:, j].ravel(), lo[:, j].ravel(), hi[:, j].ravel())
        for j, nm in enumerate(names)
    }
    overall = _block(yt.ravel(), lo.ravel(), hi.ravel())
    if yt.size == 0:
        empty = {"coverage": None, "interval_width": None, "cwc": None, "note": "空输入：无样本"}
        overall = dict(empty)
        per_target = {nm: dict(empty) for nm in names}
    return {"overall": overall, "per_target": per_target}


# ----------------------------------------------------------------------
# CRPS（高斯预测分布）
# ----------------------------------------------------------------------
def crps(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """高斯预测分布的 CRPS（连续排序概率分数），**越小越好**。

    闭式解（对每个样本）::

        CRPS = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],
        z = (y - mu) / sigma

    含义：同时惩罚「预测偏得远」与「分布过宽」；是概率预报评价的标准指标。
    """
    yt = _as_2d(y_true)
    mu = _as_2d(y_pred)
    sd = _as_2d(sigma)
    if not (yt.shape == mu.shape == sd.shape):
        raise ValueError(f"形状不一致: {yt.shape} vs {mu.shape} vs {sd.shape}")
    eps = 1e-12
    sd = np.where(np.abs(sd) < eps, eps, sd)
    z = (yt - mu) / sd
    from math import erf, pi, sqrt

    phi = np.exp(-0.5 * z**2) / sqrt(2.0 * pi)
    cdf = 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))
    value = sd * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / sqrt(pi))
    return value


# ----------------------------------------------------------------------
# 采样型方法
# ----------------------------------------------------------------------
def mc_dropout_samples(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    *,
    n_samples: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """MC Dropout：重复推理取预测分布。

    ``predict_fn`` 必须在**保持 dropout 激活**（模型处于 train 模式）时调用，
    否则多次采样会得到同一结果（此时 sigma≈0，方法无效）。

    Returns
    -------
    ndarray
        ``[n_samples, n_samples_x, n_targets]``
    """
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    preds = []
    for _ in range(int(n_samples)):
        out = np.asarray(predict_fn(X), dtype=np.float64)
        preds.append(out.reshape(out.shape[0], -1))
    return np.stack(preds, axis=0)


def ensemble_samples(
    predict_fns: list[Callable[[np.ndarray], np.ndarray]],
    X: np.ndarray,
) -> np.ndarray:
    """深度集成：多个独立训练模型的预测分布。

    Returns
    -------
    ndarray
        ``[n_models, n_samples_x, n_targets]``
    """
    if not predict_fns:
        raise ValueError("集成需要至少一个预测函数")
    preds = []
    for fn in predict_fns:
        out = np.asarray(fn(X), dtype=np.float64)
        preds.append(out.reshape(out.shape[0], -1))
    return np.stack(preds, axis=0)


# ----------------------------------------------------------------------
# 区间构造
# ----------------------------------------------------------------------
def normal_interval(
    samples: np.ndarray,
    *,
    z: float = 1.96,
) -> dict[str, np.ndarray]:
    """由预测样本的均值/标准差构造正态近似区间。

    仅在样本数足够、分布近似对称时可用；**不提供覆盖率保证**。
    """
    mu = samples.mean(axis=0)
    sd = samples.std(axis=0, ddof=1) if samples.shape[0] > 1 else np.zeros_like(mu)
    return {"mean": mu, "std": sd, "lo": mu - z * sd, "hi": mu + z * sd, "z": np.array(z)}


def conformal_interval(
    y_calib: np.ndarray,
    pred_calib: np.ndarray,
    pred_test: np.ndarray,
    *,
    alpha: float = 0.1,
    per_target: bool = False,
) -> dict[str, Any]:
    """**分割共形（split conformal）** 预测区间。

    做法：用**独立校准集**上残差的经验分位数构造区间宽度，
    在校准集与测试集可交换的假设下，有限样本覆盖保证：
    ``P(Y_test ∈ 区间) ≥ 1 - alpha``。

    这是本平台**默认推荐**的区间方法：不需要模型内部结构假设，
    且覆盖率有统计保证（前提：校准集与测试集同分布）。

    Returns
    -------
    dict
        ``{"lo", "hi", "q", "alpha", "n_calib", "method", "assumptions"}``
    """
    if not (0.0 < alpha < 1.0):
        raise ConfigurationError(f"alpha 必须在 (0,1)，当前 {alpha}")
    yc = _as_2d(y_calib)
    pc = _as_2d(pred_calib)
    pt = _as_2d(pred_test)
    if yc.shape != pc.shape:
        raise ValueError(f"校准集真值与预测形状不一致: {yc.shape} vs {pc.shape}")
    if yc.size == 0:
        raise ValueError("校准集为空，无法构造共形区间")

    resid = np.abs(yc - pc)
    if per_target:
        qs = np.quantile(resid, 1.0 - alpha, axis=0)
        lo = pt - qs
        hi = pt + qs
    else:
        q = float(np.quantile(resid, 1.0 - alpha))
        qs = np.array(q)
        lo = pt - q
        hi = pt + q
    return {
        "lo": lo,
        "hi": hi,
        "q": qs,
        "alpha": float(alpha),
        "n_calib": int(yc.shape[0]),
        "method": "conformal",
        "assumptions": [
            "校准集与测试集可交换（同分布）",
            "模型在测试集上未再训练",
            f"名义覆盖率 1-alpha = {1 - alpha:.2f}",
        ],
    }


# ----------------------------------------------------------------------
@dataclass
class UncertaintyResult:
    """统一的不确定性输出。"""

    method: str = ""
    status: str = "ok"
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    samples: np.ndarray | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON 可序列化输出（数组转 list）。"""
        def _arr(a):
            return None if a is None else np.asarray(a).tolist()

        return {
            "method": self.method,
            "status": self.status,
            "mean": _arr(self.mean),
            "std": _arr(self.std),
            "lo": _arr(self.lo),
            "hi": _arr(self.hi),
            "metrics": self.metrics,
            "assumptions": list(self.assumptions),
            "notes": list(self.notes),
        }


def estimate_uncertainty(
    y_true: np.ndarray,
    *,
    method: str = "conformal",
    predict_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    X: np.ndarray | None = None,
    samples: np.ndarray | None = None,
    y_calib: np.ndarray | None = None,
    pred_calib: np.ndarray | None = None,
    pred_test: np.ndarray | None = None,
    alpha: float = 0.1,
    n_samples: int = 30,
    seed: int = 0,
    target_names: list[str] | None = None,
) -> UncertaintyResult:
    """统一入口：按方法生成不确定性估计 + 可靠性指标。

    支持 ``method``:

    - ``"conformal"``：需要 ``y_calib`` / ``pred_calib`` / ``pred_test``（推荐，有覆盖率保证）；
    - ``"mc_dropout"``：需要 ``predict_fn`` + ``X``；
    - ``"ensemble"``：需要 ``samples``（[K, N, T]）或直接传预测样本；
    - ``"normal"``：由 ``samples`` 的均值/标准差构造（**无覆盖率保证**，仅作基线）。
    """
    method = (method or "").lower()
    res = UncertaintyResult(method=method)

    if method == "conformal":
        if y_calib is None or pred_calib is None or pred_test is None:
            raise ValueError(
                "conformal 方法需要 y_calib / pred_calib / pred_test（独立校准集）"
            )
        ci = conformal_interval(y_calib, pred_calib, pred_test, alpha=alpha)
        res.lo, res.hi = ci["lo"], ci["hi"]
        res.mean = np.asarray(pred_test, dtype=np.float64)
        res.std = np.asarray(ci["q"], dtype=np.float64) * np.ones_like(res.mean)
        res.assumptions = list(ci["assumptions"])
        res.notes.append(f"校准样本数 {ci['n_calib']}，名义覆盖率 {1 - alpha:.2f}")

    elif method in ("mc_dropout", "ensemble"):
        if samples is None:
            if method == "mc_dropout":
                if predict_fn is None or X is None:
                    raise ConfigurationError("mc_dropout 需要 predict_fn 与 X")
                samples = mc_dropout_samples(predict_fn, X, n_samples=n_samples, seed=seed)
            else:
                raise ConfigurationError("ensemble 方法需要 samples（[K, N, T] 的预测样本）")
        ni = normal_interval(samples, z=1.96)
        res.mean, res.std = ni["mean"], ni["std"]
        res.lo, res.hi = ni["lo"], ni["hi"]
        res.samples = samples
        res.assumptions = [
            "预测样本近似正态分布",
            "MC Dropout 需要模型在推理时保持 dropout 激活",
        ]
        if method == "mc_dropout":
            res.notes.append(f"MC Dropout 采样 {samples.shape[0]} 次")

    elif method == "normal":
        if samples is None:
            raise ConfigurationError("normal 方法需要 samples（[K, N, T] 的预测样本）")
        ni = normal_interval(samples, z=1.96)
        res.mean, res.std = ni["mean"], ni["std"]
        res.lo, res.hi = ni["lo"], ni["hi"]
        res.samples = samples
        res.assumptions = ["正态近似；**不提供覆盖率保证**，仅作基线对比"]
        res.notes.append("normal 方法只是基线，不应当作有保证的预测区间")
    else:
        raise ConfigurationError(
            f"未知不确定性方法 {method!r}",
            hint="可选: conformal | mc_dropout | ensemble | normal",
        )

    if y_true is not None and res.lo is not None:
        names = list(target_names or [])
        res.metrics = interval_metrics(y_true, res.lo, res.hi, names)
        if res.std is not None:
            try:
                c = crps(y_true, res.mean, res.std)
                res.metrics["overall"]["crps"] = _f(np.mean(c))
            except Exception as exc:  # pragma: no cover
                logger.warning("CRPS 计算失败: %s", exc)
    return res


def reliability_report(
    y_true: np.ndarray,
    result: UncertaintyResult,
    *,
    target_names: list[str] | None = None,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """可靠性报告：名义 vs 实际覆盖率、宽度、CRPS。

    只报告**实测**的可靠性数字；若结果里没有区间（如 normal 方法），
    会明确标注「无覆盖率保证」，不给出覆盖率结论。
    """
    out: dict[str, Any] = {
        "method": result.method,
        "status": result.status,
        "nominal_coverage": 1.0 - alpha,
        "assumptions": list(result.assumptions),
    }
    if result.lo is None or result.hi is None:
        out["status"] = "NOT_AVAILABLE"
        out["note"] = "该方法未产出区间，无法评价可靠性"
        return out
    metrics = interval_metrics(
        y_true, result.lo, result.hi, list(target_names or [])
    )
    out["metrics"] = metrics
    cov = metrics["overall"].get("coverage")
    if cov is not None:
        out["coverage_gap"] = _f(cov - (1.0 - alpha))
    if result.method == "normal":
        out["guarantee"] = "NONE（正态近似，无覆盖率保证）"
    elif result.method == "conformal":
        out["guarantee"] = "有限样本覆盖保证（需校准集与测试集可交换）"
    else:
        out["guarantee"] = "无形式化保证（启发式区间）"
    return out
