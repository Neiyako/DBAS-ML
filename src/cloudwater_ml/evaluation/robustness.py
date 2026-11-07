"""噪声鲁棒性评估：对输入特征加高斯噪声后重算预测指标。

噪声口径（与 configs/training.yaml ``evaluation.robustness.noise_levels`` 对应）：

- 噪声尺度 = ``level × 该特征在 X 上的标准差``（逐特征，std 沿样本轴 axis=0）；
- 噪声为高斯 ``N(0, 1) × scale``，用 ``np.random.default_rng(seed)`` 固定种子，
  可复现；``level=0.0`` 时**直接用原始 X**（不加任何噪声，也不消耗随机数语义外的扰动）；
- ``predict_fn(X) -> Yhat`` 由调用方（runner）注入，本模块不依赖 models 包。

返回键为 ``str(level)``，如 ``{"0.0": ..., "0.05": ..., "0.1": ...}``，
每个值是 :func:`regression_metrics` 的完整输出。
"""
from __future__ import annotations

import logging

import numpy as np

from .metrics import regression_metrics

logger = logging.getLogger("cloudwater_ml.evaluation")

__all__ = ["noise_robustness"]


def noise_robustness(
    predict_fn,
    X: np.ndarray,
    Y: np.ndarray,
    target_names: list[str],
    noise_levels: list[float],
    seed: int = 42,
) -> dict:
    """逐噪声水平计算预测指标，衡量模型对输入扰动的鲁棒性。

    Parameters
    ----------
    predict_fn : callable
        ``predict_fn(X_noisy) -> Y_pred``，形状与 Y 一致（runner 传 model.predict）。
    X : ndarray [n, f]
        输入特征；逐特征 std 由它估计。
    Y : ndarray [n, t]
        真值。
    target_names : list[str]
    noise_levels : list[float]
        相对噪声强度；0.0 表示原始数据基线。
    seed : int
        高斯噪声随机种子（level=0 不使用）。

    Returns
    -------
    dict
        ``{str(level): regression_metrics(...)}``；predict_fn 抛错时该水平
        记 ``{"error": ...}`` 而不向上抛（不崩训练主流程）。
    """
    Xa = np.asarray(X, dtype=np.float64)
    Ya = np.asarray(Y, dtype=np.float64)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if Ya.ndim == 1:
        Ya = Ya.reshape(-1, 1)
    if Xa.shape[0] != Ya.shape[0]:
        raise ValueError(f"X/Y 样本数不一致: {Xa.shape[0]} vs {Ya.shape[0]}")

    rng = np.random.default_rng(seed)
    # 逐特征标准差（population std, ddof=0）；零方差特征 scale=0 → 噪声自然为 0
    feature_std = Xa.std(axis=0)

    out: dict[str, dict] = {}
    for level in noise_levels:
        key = str(level)
        if float(level) == 0.0:
            X_in = Xa  # level=0：原始数据
        else:
            X_in = Xa + rng.standard_normal(Xa.shape) * (float(level) * feature_std)
        try:
            y_pred = np.asarray(predict_fn(X_in), dtype=np.float64)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            if y_pred.shape != Ya.shape:
                raise ValueError(
                    f"predict_fn 输出形状 {y_pred.shape} 与 Y 形状 {Ya.shape} 不一致"
                )
            out[key] = regression_metrics(Ya, y_pred, target_names)
        except Exception as exc:  # noqa: BLE001 —— 容错：单水平失败不崩整体
            logger.warning("noise level=%s 下预测失败: %s", key, exc)
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
