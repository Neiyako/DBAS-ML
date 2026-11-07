"""回归评价指标（评价口径的唯一实现处）。

口径（与 configs/training.yaml ``evaluation.metrics`` 键名一致）：

- MAE  = mean(|e|)，e = pred − true，单位与目标一致；
- MSE  = mean(e²)，单位 = 目标单位²；
- RMSE = √MSE，单位与目标一致；
- Bias = mean(pred − true)，**>0 表示模型高估**，单位与目标一致；
- R²   = 1 − SS_res / SS_tot，无量纲；SS_tot = Σ(y − ȳ)²。

汇总方式（overall 的定义，二选一已定）：
    **overall = 把所有目标列展平后按全部元素（micro）计算一次**；
    per_target = 按列各算一次。跨目标量纲差异大时（例如 clwc[kg kg-1]
    与 cwp[kg m-2] 混用）overall 会被大数值目标主导，此时以 per_target 为准。

R² 保护：目标近似常数（SS_tot < 1e-12，绝对阈值）时 r2=None 并附
``r2_note``，同时 logger warning —— 常数目标的 R² 无解释意义。

所有返回值均为 JSON 可序列化（python float / None；非有限值一律降为 None）。
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("cloudwater_ml.evaluation")

__all__ = ["METRIC_DEFINITIONS", "regression_metrics"]

#: R² 保护的绝对阈值：SS_tot 低于它视为目标近似常数
_R2_SS_TOT_EPS = 1e-12

# 指标口径登记（不编造：公式即代码实现，单位随目标变量）
METRIC_DEFINITIONS: dict[str, dict] = {
    "mae": {
        "formula": "MAE = mean(|y_pred - y_true|)",
        "units": "与目标变量单位一致（如 kg kg-1）",
        "applicability": "所有回归目标；对量纲/尺度敏感，跨目标不可直接比较",
    },
    "mse": {
        "formula": "MSE = mean((y_pred - y_true)^2)",
        "units": "目标变量单位的平方（如 kg^2 kg^-2）",
        "applicability": "所有回归目标；放大离群误差，通常配合 RMSE 解读",
    },
    "rmse": {
        "formula": "RMSE = sqrt(MSE)",
        "units": "与目标变量单位一致（如 kg kg-1）",
        "applicability": "所有回归目标；与 MAE 差距大提示误差分布有重尾",
    },
    "r2": {
        "formula": "R2 = 1 - SS_res / SS_tot, SS_res = sum((y_true - y_pred)^2), "
        "SS_tot = sum((y_true - mean(y_true))^2)",
        "units": "无量纲",
        "applicability": "目标需有真实变异；目标近似常数（SS_tot < 1e-12）时不可解释，返回 None",
    },
    "bias": {
        "formula": "Bias = mean(y_pred - y_true)",
        "units": "与目标变量单位一致（如 kg kg-1）",
        "applicability": "所有回归目标；>0 = 系统性高估，<0 = 系统性低估",
    },
    "nse": {
        "formula": "NSE = 1 - SS_res / SS_tot, SS_res = sum((y_true - y_pred)^2), "
        "SS_tot = sum((y_true - mean(y_true))^2)",
        "units": "无量纲",
        "applicability": "水文/资源领域常用效率系数；取观测均值作基准，"
        "与 r2 同式（数值一致），保留是为口径对齐而非新信息",
    },
    "pearson": {
        "formula": "r = cov(y_true, y_pred) / (std(y_true) * std(y_pred))",
        "units": "无量纲",
        "applicability": "刻画线性相关强度（与 R² 不同：R² 含偏差项，r 只看相关性）",
    },
}


def _as_2d(a: np.ndarray) -> np.ndarray:
    """统一为二维 [n_samples, n_targets]。"""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _f(x: float) -> float | None:
    """numpy/标量 → JSON 安全 float；NaN/inf → None。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _names(target_names: list[str] | None, n_cols: int) -> list[str]:
    names = list(target_names) if target_names else []
    if len(names) != n_cols:
        logger.warning(
            "target_names 数量 %d 与目标列数 %d 不符，缺失列按 target_i 命名",
            len(names),
            n_cols,
        )
        names = [names[i] if i < len(names) else f"target_{i}" for i in range(n_cols)]
    return names


def _block(y: np.ndarray, p: np.ndarray, name: str) -> dict:
    """对一维 (y, p) 计算五项指标 + R² 保护，返回 JSON 安全 dict。"""
    err = p - y
    n = y.size
    out: dict = {
        "mae": _f(np.mean(np.abs(err))),
        "mse": _f(np.mean(err**2)),
        "rmse": _f(np.sqrt(np.mean(err**2))),
        "bias": _f(np.mean(err)),
    }
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < _R2_SS_TOT_EPS:
        out["r2"] = None
        out["r2_note"] = "target 近似常数，R² 不可解释"
        out["nse"] = None
        out["nse_note"] = "target 近似常数，NSE 不可解释"
        logger.warning(
            "R² 不可解释：目标 %r 近似常数（SS_tot=%g < %g，n=%d）",
            name,
            ss_tot,
            _R2_SS_TOT_EPS,
            n,
        )
    else:
        out["r2"] = _f(1.0 - ss_res / ss_tot)
        # NSE：观测均值基准（与 R² 同式，保留是为水文口径对齐）
        out["nse"] = _f(1.0 - ss_res / ss_tot)
    # Pearson r：只看线性相关强度
    if n >= 2:
        y_std = float(np.std(y))
        p_std = float(np.std(p))
        if y_std > 0 and p_std > 0:
            out["pearson"] = _f(float(np.mean((y - np.mean(y)) * (p - np.mean(p))) / (y_std * p_std)))
        else:
            out["pearson"] = None
            out["pearson_note"] = "目标或预测近似常数，相关系数不可解释"
    else:
        out["pearson"] = None
    return out


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict:
    """计算整体与逐目标的回归指标。

    Parameters
    ----------
    y_true, y_pred : ndarray
        一维 [n] 或二维 [n, t]；二维时按列（目标）分别计算。
    target_names : list[str]
        与目标列数一致的名称。

    Returns
    -------
    dict
        ``{"overall": {mae, mse, rmse, r2, bias[, r2_note]},
           "per_target": {name: {同五项}}}``
        overall 为**全部元素展平后的 micro 汇总**（见模块 docstring）；
        值均为 float 或 None（R² 保护/空输入/非有限值时为 None），
        可直接 ``json.dumps``。

    Raises
    ------
    ValueError
        y_true / y_pred 形状不一致时。
    """
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true 与 y_pred 形状不一致: {yt.shape} vs {yp.shape}")
    names = _names(target_names, yt.shape[1])

    per_target: dict[str, dict] = {}
    for j, nm in enumerate(names):
        per_target[nm] = _block(yt[:, j].ravel(), yp[:, j].ravel(), nm)

    overall = _block(yt.ravel(), yp.ravel(), "overall")
    if yt.size == 0:
        # 空输入：全部降为 None 并附 note（不编数）
        overall = dict.fromkeys(("mae", "mse", "rmse", "r2", "bias", "nse", "pearson"))
        overall["note"] = "空输入：无样本可评价"
        per_target = {
            nm: {**dict.fromkeys(("mae", "mse", "rmse", "r2", "bias", "nse", "pearson")),
                 "note": "空输入：无样本可评价"}
            for nm in names
        }
    return {"overall": overall, "per_target": per_target}
