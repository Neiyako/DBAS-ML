"""泛化分析：按 季节 / 年份 / 区域 分组计算回归指标。

分组口径（不编造数据，缺列/缺配置一律 skipped + note）：

- **year**：``meta["year"]``；缺失时从 ``meta["time"]`` 推 ``dt.year``；
- **season**：由 ``meta["time"]`` 推气象季节 DJF/MAM/JJA/SON
  （北半球：12/1/2=DJF，3/4/5=MAM，6/7/8=JJA，9/10/11=SON）；
- **region**：``region_boxes = {"名称": [lon0, lon1, lat0, lat1]}``，
  闭区间矩形筛选 ``lon0<=lon<=lon1 且 lat0<=lat<=lat1``；
  未提供 region_boxes 时该组返回 ``{"skipped": "no region_boxes configured"}``
  （不虚构区域）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .metrics import regression_metrics

logger = logging.getLogger("cloudwater_ml.evaluation")

__all__ = ["subgroup_metrics", "meteorological_season"]

_SEASON_BY_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def meteorological_season(times: pd.Series) -> pd.Series:
    """时间列 → 气象季节标签（DJF/MAM/JJA/SON）；非法/缺失时间为 NaN。"""
    ts = pd.to_datetime(times, errors="coerce")
    return ts.dt.month.map(_SEASON_BY_MONTH)


def _year_key(meta: pd.DataFrame) -> pd.Series | None:
    if "year" in meta.columns:
        return meta["year"]
    if "time" in meta.columns:
        return pd.to_datetime(meta["time"], errors="coerce").dt.year
    return None


def _time_key(meta: pd.DataFrame) -> pd.Series | None:
    if "time" in meta.columns:
        return pd.to_datetime(meta["time"], errors="coerce")
    return None


def _region_masks(meta: pd.DataFrame, region_boxes: dict) -> dict[str, np.ndarray] | None:
    if "longitude" not in meta.columns or "latitude" not in meta.columns:
        logger.warning("region 分组需要 meta 的 longitude/latitude 列，缺失，跳过")
        return None
    lon = pd.to_numeric(meta["longitude"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(meta["latitude"], errors="coerce").to_numpy(dtype=float)
    masks: dict[str, np.ndarray] = {}
    for name, box in region_boxes.items():
        try:
            lon0, lon1, lat0, lat1 = (float(v) for v in box)
        except (TypeError, ValueError):
            logger.warning("region_boxes[%r]=%r 不是 [lon0, lon1, lat0, lat1]，跳过该区域", name, box)
            continue
        lo, hi = (lon0, lon1) if lon0 <= lon1 else (lon1, lon0)
        la, lb = (lat0, lat1) if lat0 <= lat1 else (lat1, lat0)
        masks[str(name)] = (
            np.isfinite(lon) & np.isfinite(lat)
            & (lon >= lo) & (lon <= hi) & (lat >= la) & (lat <= lb)
        )
    return masks


def _group(
    keys: pd.Series | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    skip_note: str,
) -> dict:
    """按 keys 分组算指标；keys 为 None 时整组 skipped。"""
    if keys is None:
        return {"skipped": skip_note}
    out: dict[str, dict] = {}
    valid = pd.notna(keys)
    if not bool(valid.any()):
        return {"skipped": f"{skip_note}（键列存在但全为空）"}
    for k in pd.unique(keys[valid]):
        idx = np.flatnonzero((keys == k).to_numpy() & valid.to_numpy())
        if idx.size == 0:
            continue
        block = regression_metrics(y_true[idx], y_pred[idx], target_names)
        # 仅当整块无有效样本（mae=None，见 metrics 空输入 note）才跳过；
        # 单项 R² 因常数目标为 None 不影响该组其余指标。
        if block["overall"].get("mae") is None:
            out[str(k)] = {"skipped": "组内无有效样本"}
        else:
            out[str(k)] = block
    return out


def subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    target_names: list[str],
    subgroups: list[str] = ("year", "season", "region"),
    region_boxes: dict | None = None,
) -> dict:
    """按 year / season / region 分组的回归指标（泛化分析）。

    Parameters
    ----------
    y_true, y_pred : ndarray
        [n, t]（或一维），行与 ``meta`` 一一对应。
    meta : DataFrame
        至少含 ``time``；region 还需 ``longitude``/``latitude``。
    target_names : list[str]
    subgroups : list[str]
        请求的分组维度子集；未知维度名记录到该组的 skipped note。
    region_boxes : dict | None
        ``{"名称": [lon0, lon1, lat0, lat1]}``；None 时 region 组 skipped。

    Returns
    -------
    dict
        ``{"year": {"2015": {metrics...}}, "season": {...}, "region": {...}}``；
        每组的值为 :func:`regression_metrics` 的输出；某维度缺列/缺配置时
        该维度值为 ``{"skipped": "<note>"}``。不编造任何数值。
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true 与 y_pred 形状不一致: {yt.shape} vs {yp.shape}")
    if meta is None or len(meta) == 0:
        return {g: {"skipped": "meta 为空，无法分组"} for g in subgroups}
    if len(meta) != yt.shape[0]:
        raise ValueError(f"meta 行数 {len(meta)} 与样本数 {yt.shape[0]} 不一致")

    result: dict[str, dict] = {}
    for group in subgroups:
        if group == "year":
            result["year"] = _group(
                _year_key(meta), yt, yp, target_names,
                "meta 无 year 列且无 time 列可推年份",
            )
        elif group == "season":
            t = _time_key(meta)
            keys = meteorological_season(t) if t is not None else None
            result["season"] = _group(
                keys, yt, yp, target_names,
                "meta 无 time 列，无法推气象季节 DJF/MAM/JJA/SON",
            )
        elif group == "region":
            if region_boxes is None or not region_boxes:
                result["region"] = {"skipped": "no region_boxes configured"}
                continue
            masks = _region_masks(meta, region_boxes)
            if masks is None:
                result["region"] = {"skipped": "meta 缺 longitude/latitude 列"}
                continue
            region_out: dict[str, dict] = {}
            for name, m in masks.items():
                if not m.any():
                    region_out[name] = {"skipped": "组内无样本（盒子未命中任何点）"}
                    continue
                region_out[name] = regression_metrics(yt[m], yp[m], target_names)
            result["region"] = region_out
        else:
            result[group] = {"skipped": f"未知分组维度 {group!r}"}
            logger.warning("未知分组维度 %r，已跳过", group)
    return result
