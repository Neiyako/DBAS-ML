"""空间分布图：经纬度散点 + colorbar（不依赖 cartopy）。

约定：Agg 后端、savefig(dpi=150)、不调用 plt.show()、返回 Path；
经纬度缺失/对不上/绘图失败时返回 None，不抛异常。
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("cloudwater_ml.visualization")

__all__ = ["plot_scatter_map"]


def plot_scatter_map(
    latitude,
    longitude,
    values,
    out_path,
    title: str | None = None,
) -> Path | None:
    """经纬度散点图（colorbar 表示 values 量级）。

    Parameters
    ----------
    latitude, longitude : array-like | None
        与 values 等长的坐标；None / 缺列（None 输入）直接返回 None。
    values : array-like
        每个点的标量值（如预测均值、RMSE、偏差）。
    out_path : str | Path
    title : str | None

    Returns
    -------
    Path | None
        保存路径；坐标缺失/长度不匹配/无有限值/绘图失败时返回 None。
    """
    try:
        if latitude is None or longitude is None or values is None:
            logger.warning("plot_scatter_map: 经纬度或 values 缺失，返回 None")
            return None
        lat = np.asarray(latitude, dtype=np.float64).ravel()
        lon = np.asarray(longitude, dtype=np.float64).ravel()
        val = np.asarray(values, dtype=np.float64).ravel()
        if not (lat.size == lon.size == val.size):
            logger.warning(
                "plot_scatter_map 长度不一致: lat=%d lon=%d values=%d",
                lat.size, lon.size, val.size,
            )
            return None
        ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val)
        if not ok.any():
            logger.warning("plot_scatter_map: 无任何有限样本点，返回 None")
            return None
        lat, lon, val = lat[ok], lon[ok], val[ok]

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        sc = ax.scatter(lon, lat, c=val, s=14, cmap="viridis", alpha=0.85)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("value")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(title if title else "Spatial distribution")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    except Exception as exc:  # noqa: BLE001 —— 容错
        logger.warning("plot_scatter_map 失败: %s", exc)
        plt.close("all")
        return None
