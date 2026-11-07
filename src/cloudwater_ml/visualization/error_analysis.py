"""误差分析图：按子组（year/season/region）的 RMSE 分组柱状图。

约定：Agg 后端、savefig(dpi=150)、不调用 plt.show()、返回 Path；
输入为空/结构不可用时返回 None，不抛异常。
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("cloudwater_ml.visualization")

__all__ = ["plot_subgroup_rmse"]


def _group_value(metrics) -> tuple[dict[str, float], list[str]]:
    """从一个分组的指标块提取 {系列名: rmse}（兼容两种结构）。

    结构 A（平铺，任务示例）: ``{"rmse": v}`` 或 ``{"overall": {...}}``；
    结构 B（regression_metrics 输出）: ``{"overall": {...}, "per_target": {...}}``
    → 优先 per_target 逐目标系列，否则 overall 单系列。
    返回 (series, notes)；skipped/无有效 rmse 时 series 为空。
    """
    notes: list[str] = []
    if not isinstance(metrics, dict):
        return {}, ["非 dict 指标块"]
    if "skipped" in metrics:
        return {}, [str(metrics["skipped"])]
    if isinstance(metrics.get("per_target"), dict):
        series = {}
        for nm, blk in metrics["per_target"].items():
            if isinstance(blk, dict) and isinstance(blk.get("rmse"), (int, float)):
                series[str(nm)] = float(blk["rmse"])
        if series:
            return series, notes
    # 平铺结构：直接 rmse，或 overall.rmse
    if isinstance(metrics.get("rmse"), (int, float)):
        return {"rmse": float(metrics["rmse"])}, notes
    ov = metrics.get("overall")
    if isinstance(ov, dict) and isinstance(ov.get("rmse"), (int, float)):
        return {"overall": float(ov["rmse"])}, notes
    return {}, ["未找到有效 rmse（可能为 skipped 或空组）"]


def plot_subgroup_rmse(
    subgroup_metrics: dict,
    out_path,
    title: str = "RMSE by Subgroup",
) -> Path | None:
    """把 ``subgroup_metrics`` 的分组 RMSE 画成分组柱状图。

    Parameters
    ----------
    subgroup_metrics : dict
        ``{"year": {"2015": {"overall": {...}, "per_target": {...}}, ...},
           "season": {...}, "region": {...}}``
        也兼容平铺 ``{"year": {"2015": {"rmse": v}}}``。
        skipped 的分组不画（不编数）。
    out_path : str | Path
    title : str

    Returns
    -------
    Path | None
        保存路径；无任何有效分组或绘图失败时返回 None。
    """
    try:
        if not isinstance(subgroup_metrics, dict) or not subgroup_metrics:
            logger.warning("subgroup_metrics 为空，跳过 RMSE 分组图")
            return None

        # 收集 (维度, 组名, {系列: rmse})
        dims: list[str] = []
        groups: list[str] = []
        values: list[dict[str, float]] = []
        skipped_notes: list[str] = []
        for dim, block in subgroup_metrics.items():
            if not isinstance(block, dict):
                skipped_notes.append(f"{dim}: 非 dict")
                continue
            if "skipped" in block and len(block) == 1:  # 维度级 skipped（如缺 region_boxes）
                skipped_notes.append(f"{dim}: {block['skipped']}")
                continue
            for gname, m in block.items():
                series, notes = _group_value(m)
                if not series:
                    skipped_notes.append(f"{dim}/{gname}: {'; '.join(notes) or 'skipped'}")
                    continue
                dims.append(str(dim))
                groups.append(f"{dim}={gname}")
                values.append(series)
        if not values:
            logger.warning("无有效分组可画 RMSE（跳过注记: %s）", skipped_notes[:5])
            return None
        if skipped_notes:
            logger.info("plot_subgroup_rmse 跳过 %d 组: %s", len(skipped_notes),
                        skipped_notes[:5])

        series_names: list[str] = []
        for s in values:
            for k in s:
                if k not in series_names:
                    series_names.append(k)

        x = np.arange(len(groups))
        n_series = len(series_names)
        width = 0.8 / max(n_series, 1)
        fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(groups) + 3), 4.8))
        for i, sname in enumerate(series_names):
            ys = [v.get(sname, np.nan) for v in values]
            ax.bar(x + i * width, ys, width=width, label=sname)
        ax.set_xticks(x + width * (n_series - 1) / 2)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        if n_series > 1 or series_names != ["rmse"]:
            ax.legend(fontsize=8)
        fig.tight_layout()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    except Exception as exc:  # noqa: BLE001 —— 容错
        logger.warning("plot_subgroup_rmse 失败: %s", exc)
        plt.close("all")
        return None
