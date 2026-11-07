"""预测对比图与残差分布图（每目标一个子图）。

约定：Agg 后端、savefig(dpi=150)、不调用 plt.show()、返回 Path；
异常一律降为 None。数值量级小（如 1e-5）时坐标轴用科学计数显示。
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("cloudwater_ml.visualization")

__all__ = [
    "plot_prediction_vs_truth",
    "plot_residual_distribution",
    "plot_feature_importance",
]


def _as_2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _names(target_names, n_cols: int) -> list[str]:
    names = list(target_names) if target_names else []
    return [names[i] if i < len(names) else f"target_{i}" for i in range(n_cols)]


def _sci(ax) -> None:
    """小量级数值用科学计数显示（1e-5 级目标可读）。"""
    ax.ticklabel_format(style="sci", axis="both", useOffset=False)


def plot_prediction_vs_truth(
    y_true,
    y_pred,
    target_names: list[str],
    out_path,
) -> Path | None:
    """预测值 vs 真值散点图，每目标一个子图 + 1:1 参考线。

    Returns
    -------
    Path | None
        保存路径；形状不一致/输入不可用/绘图失败时返回 None。
    """
    try:
        yt = _as_2d(y_true)
        yp = _as_2d(y_pred)
        if yt.shape != yp.shape:
            logger.warning("plot_prediction_vs_truth 形状不一致: %s vs %s", yt.shape, yp.shape)
            return None
        names = _names(target_names, yt.shape[1])
        n = yt.shape[1]
        ncol = min(n, 3)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.2 * nrow), squeeze=False)
        for j in range(n):
            ax = axes[j // ncol][j % ncol]
            a, b = yt[:, j], yp[:, j]
            ax.scatter(a, b, s=10, alpha=0.55, edgecolors="none")
            lo = float(min(np.nanmin(a), np.nanmin(b)))
            hi = float(max(np.nanmax(a), np.nanmax(b)))
            pad = (hi - lo) * 0.05 or 1e-12
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    "r--", lw=1.2, label="1:1")
            ax.set_xlabel("truth")
            ax.set_ylabel("prediction")
            ax.set_title(str(names[j]))
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
            try:
                _sci(ax)
            except Exception:  # noqa: BLE001 —— 科学计数设置失败不致命
                pass
        for j in range(n, nrow * ncol):  # 空白子图关掉
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Prediction vs Truth")
        fig.tight_layout()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_prediction_vs_truth 失败: %s", exc)
        plt.close("all")
        return None


def plot_residual_distribution(
    y_true,
    y_pred,
    target_names: list[str],
    out_path,
) -> Path | None:
    """残差（pred − truth）直方图，每目标一个子图。

    Returns
    -------
    Path | None
        保存路径；输入不可用/绘图失败时返回 None。
    """
    try:
        yt = _as_2d(y_true)
        yp = _as_2d(y_pred)
        if yt.shape != yp.shape:
            logger.warning("plot_residual_distribution 形状不一致: %s vs %s", yt.shape, yp.shape)
            return None
        names = _names(target_names, yt.shape[1])
        n = yt.shape[1]
        ncol = min(n, 3)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.0 * nrow), squeeze=False)
        for j in range(n):
            ax = axes[j // ncol][j % ncol]
            resid = (yp[:, j] - yt[:, j]).ravel()
            resid = resid[np.isfinite(resid)]
            if resid.size == 0:
                ax.set_title(f"{names[j]} (no valid residual)")
                continue
            ax.hist(resid, bins=min(40, max(10, int(np.sqrt(resid.size)))),
                    color="steelblue", edgecolor="white", alpha=0.9)
            ax.axvline(0.0, color="red", ls="--", lw=1.2)
            ax.axvline(float(np.mean(resid)), color="orange", ls=":", lw=1.4,
                       label=f"mean={np.mean(resid):.3e}")
            ax.set_xlabel("residual (pred − truth)")
            ax.set_ylabel("count")
            ax.set_title(str(names[j]))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            try:
                _sci(ax)
            except Exception:  # noqa: BLE001
                pass
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Residual Distribution")
        fig.tight_layout()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_residual_distribution 失败: %s", exc)
        plt.close("all")
        return None


def plot_feature_importance(
    importance: dict[str, float] | None,
    out_path,
    title: str = "Feature Importance",
) -> Path | None:
    """特征重要性水平条形图（按重要性降序）。

    Parameters
    ----------
    importance : dict[str, float] | None
        特征名 → 重要性（如 permutation importance / model feature_importances_）。
        None 或空 dict → 返回 None（不画空图）。
    out_path : str | Path

    Returns
    -------
    Path | None
        保存路径；无有效数据或绘图失败时返回 None。
    """
    try:
        if not importance:
            return None
        pairs = [
            (str(k), float(v))
            for k, v in importance.items()
            if isinstance(v, (int, float)) and np.isfinite(float(v))
        ]
        if not pairs:
            logger.warning("plot_feature_importance: 无有效数值，返回 None")
            return None
        pairs.sort(key=lambda kv: kv[1])
        labels = [k for k, _ in pairs]
        vals = [v for _, v in pairs]

        fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.38 * len(labels) + 1.8)))
        ax.barh(labels, vals, color="teal", alpha=0.85)
        ax.set_xlabel("importance")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_feature_importance 失败: %s", exc)
        plt.close("all")
        return None
