"""评价编排入口：按 configs/training.yaml ``evaluation`` 段组织一次完整评价。

容错约定：cfg 缺 ``evaluation`` 段、段内键缺失、region_boxes 未配置时
一律取默认值，不抛异常；metrics 过滤只保留配置里点名的指标
（``r2_note`` 随 r2 保留）。
"""
from __future__ import annotations

import logging

from .generalization import subgroup_metrics
from .metrics import METRIC_DEFINITIONS, regression_metrics

logger = logging.getLogger("cloudwater_ml.evaluation")

__all__ = ["evaluate_predictions"]

_DEFAULT_METRICS = tuple(METRIC_DEFINITIONS.keys())
_DEFAULT_SUBGROUPS = ("year", "season", "region")


def _evaluation_section(cfg: dict | None) -> dict:
    """取 evaluation 段：容错多种传入形态。"""
    if not isinstance(cfg, dict):
        return {}
    ev = cfg.get("evaluation")
    if isinstance(ev, dict):
        return ev
    # 允许直接把 evaluation 段当 cfg 传入（含其任一已知键）
    if any(k in cfg for k in ("metrics", "subgroups", "robustness", "uncertainty")):
        return cfg
    return {}


def _filter_metrics(block: dict, wanted: list[str]) -> dict:
    """只保留配置点名的指标键（r2_note 随 r2 保留，note 不过滤）。"""
    keep = set(wanted) | {"note"}
    if "r2" in wanted:
        keep.add("r2_note")
    return {k: v for k, v in block.items() if k in keep}


def evaluate_predictions(
    y_true,
    y_pred,
    meta,
    target_names: list[str],
    cfg: dict,
) -> dict:
    """一次完整评价：整体/逐目标指标 + 分组泛化指标。

    Parameters
    ----------
    y_true, y_pred : ndarray
        [n, t]（或一维），行与 meta 对齐。
    meta : pd.DataFrame
        含 time（season/year 推导）与 longitude/latitude（region 分组）。
    target_names : list[str]
    cfg : dict
        完整训练配置（取 ``cfg["evaluation"]``）；也可直接传 evaluation 段。
        相关键：
        ``metrics``（过滤输出指标）、``subgroups``（分组维度）、
        ``region_boxes``（区域盒子，可选）。

    Returns
    -------
    dict
        ``{"metrics": regression_metrics(...),
           "subgroups": subgroup_metrics(...)}``
    """
    ev = _evaluation_section(cfg)
    wanted = ev.get("metrics")
    if not wanted:
        wanted = list(_DEFAULT_METRICS)
    wanted = [str(m) for m in wanted]
    unknown = [m for m in wanted if m not in METRIC_DEFINITIONS]
    if unknown:
        # 未知指标名必须**报错**而不是「警告后忽略」：
        # 静默丢弃会让配置写错的人以为指标已算出来（实测踩过：nse 配了却没产出）。
        from ..utils.exceptions import ConfigurationError

        raise ConfigurationError(
            f"evaluation.metrics 含未知指标: {unknown}",
            hint=f"可用指标: {sorted(METRIC_DEFINITIONS)}",
            context={"requested": wanted},
        )

    subgroups = ev.get("subgroups")
    if not subgroups:
        subgroups = list(_DEFAULT_SUBGROUPS)
    region_boxes = ev.get("region_boxes")

    raw = regression_metrics(y_true, y_pred, target_names)
    metrics = {
        "overall": _filter_metrics(raw["overall"], wanted),
        "per_target": {
            nm: _filter_metrics(block, wanted) for nm, block in raw["per_target"].items()
        },
    }
    try:
        subs = subgroup_metrics(
            y_true, y_pred, meta, target_names,
            subgroups=[str(g) for g in subgroups],
            region_boxes=region_boxes,
        )
    except Exception as exc:  # noqa: BLE001 —— 分组失败不拖垮整体指标
        logger.warning("subgroup_metrics 失败: %s", exc)
        subs = {str(g): {"skipped": f"subgroup_metrics 失败: {type(exc).__name__}: {exc}"}
                for g in subgroups}

    return {"metrics": metrics, "subgroups": subs}
