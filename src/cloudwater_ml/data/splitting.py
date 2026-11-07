"""数据划分：random / time / year / region / source（防时空泄漏的关键实现）。

原则：

- Train / Validation / Test 三集互斥（:meth:`SplitResult.check_disjoint` 强校验）；
- time/year 划分整块归属，杜绝「未来样本进训练集」的时间泄漏；
- region 划分按空间盒整块归属，越界样本被显式丢弃并记录（不静默）；
- source 划分按 real/simulated 标签整块归属，用于域迁移类实验；
- 归一化参数在划分之后**只从 train 拟合**（见 preprocessing）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .schema import SampleBundle

__all__ = ["SplitResult", "split_indices", "available_strategies"]

_STRATEGIES = ("random", "time", "year", "region", "source")


def available_strategies() -> list[str]:
    return list(_STRATEGIES)


@dataclass
class SplitResult:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    strategy: str
    notes: list[str] = field(default_factory=list)
    dropped: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "n_train": int(self.train.size),
            "n_val": int(self.val.size),
            "n_test": int(self.test.size),
            "n_dropped": int(self.dropped),
            "notes": list(self.notes),
        }

    def check_disjoint(self) -> None:
        """强校验三集两两不相交（防止划分实现引入泄漏）。"""
        sets = {k: set(map(int, getattr(self, k))) for k in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            inter = sets[a] & sets[b]
            if inter:
                raise ValueError(f"划分泄漏：{a} 与 {b} 存在 {len(inter)} 个重叠样本")


def _ratios(split_cfg: dict[str, Any]) -> np.ndarray:
    r = np.asarray(split_cfg.get("ratios") or [0.7, 0.15, 0.15], dtype=np.float64)
    if r.size != 3 or np.any(r < 0):
        raise ValueError(f"split.ratios 必须是 3 个非负数 [train, val, test]，得到 {r.tolist()}")
    if abs(float(r.sum()) - 1.0) > 1e-6:
        raise ValueError(f"split.ratios 之和应为 1.0，得到 {float(r.sum())}")
    return r


def _positional_split(n: int, r: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按给定顺序（order）做位置三段切分（order 已按时间升序即为时序划分）。"""
    c1 = int(round(r[0] * n))
    c2 = c1 + int(round(r[1] * n))
    c1, c2 = max(c1, 0), min(max(c2, c1), n)
    return order[:c1], order[c1:c2], order[c2:]


def _split_random(bundle: SampleBundle, split_cfg: dict[str, Any]) -> SplitResult:
    n = bundle.n_samples
    rng = np.random.default_rng(int(split_cfg.get("seed", 42)))
    order = rng.permutation(n)
    tr, va, te = _positional_split(n, _ratios(split_cfg), order)
    return SplitResult(tr, va, te, "random", notes=[f"seed={split_cfg.get('seed', 42)}"])


def _split_time(bundle: SampleBundle, split_cfg: dict[str, Any]) -> SplitResult:
    if "time" not in bundle.meta.columns:
        raise ValueError("time 划分需要 meta.time 列（数据源没有时间戳时改用 random/region/source）")
    t = pd.to_datetime(bundle.meta["time"])
    if t.isna().any():
        raise ValueError("meta.time 存在无法解析的时间戳（NaT），先修复数据源")
    params = split_cfg.get("params") or {}
    notes: list[str] = []
    train_end, val_end = params.get("train_end"), params.get("val_end")
    if train_end and val_end:
        train_end = pd.Timestamp(train_end)
        val_end = pd.Timestamp(val_end)
        tr = np.flatnonzero((t <= train_end).to_numpy())
        va = np.flatnonzero(((t > train_end) & (t <= val_end)).to_numpy())
        te = np.flatnonzero((t > val_end).to_numpy())
        notes.append(f"train_end={train_end}, val_end={val_end}")
    else:
        order = np.argsort(t.to_numpy(), kind="stable")
        tr, va, te = _positional_split(bundle.n_samples, _ratios(split_cfg), order)
        notes.append("未给 train_end/val_end → 按时间升序位置切分（ratios）")
    for name, arr in (("train", tr), ("val", va), ("test", te)):
        if arr.size == 0:
            raise ValueError(f"time 划分后 {name} 为空，调整 split.params 时间边界或 ratios")
    return SplitResult(tr, va, te, "time", notes=notes)


def _split_year(bundle: SampleBundle, split_cfg: dict[str, Any]) -> SplitResult:
    if "year" in bundle.meta.columns:
        years = bundle.meta["year"].astype(int).to_numpy()
    elif "time" in bundle.meta.columns:
        years = pd.to_datetime(bundle.meta["time"]).dt.year.to_numpy()
    else:
        raise ValueError("year 划分需要 meta.year 或 meta.time 列")
    params = split_cfg.get("params") or {}
    uniq = np.unique(years)
    if params.get("train_years") and params.get("test_years") is not None:
        tr_y = set(map(int, params.get("train_years") or []))
        va_y = set(map(int, params.get("val_years") or []))
        te_y = set(map(int, params.get("test_years") or []))
    else:
        r = _ratios(split_cfg)
        k = len(uniq)
        if k < 3:
            raise ValueError(f"year 划分至少需要 3 个不同年份，数据只有 {k} 个: {uniq.tolist()}")
        c1 = min(max(int(round(r[0] * k)), 1), k - 2)
        c2 = min(c1 + max(int(round(r[1] * k)), 1), k - 1)
        tr_y, va_y, te_y = set(uniq[:c1]), set(uniq[c1:c2]), set(uniq[c2:])
    tr = np.flatnonzero(np.isin(years, sorted(tr_y)))
    va = np.flatnonzero(np.isin(years, sorted(va_y)))
    te = np.flatnonzero(np.isin(years, sorted(te_y)))
    notes = [
        f"train_years={sorted(int(y) for y in tr_y)}",
        f"val_years={sorted(int(y) for y in va_y)}",
        f"test_years={sorted(int(y) for y in te_y)}",
    ]
    for name, arr in (("train", tr), ("val", va), ("test", te)):
        if arr.size == 0:
            raise ValueError(f"year 划分后 {name} 为空（{notes}）")
    return SplitResult(tr, va, te, "year", notes=notes)


def _split_region(bundle: SampleBundle, split_cfg: dict[str, Any]) -> SplitResult:
    boxes = (split_cfg.get("params") or {}).get("boxes")
    if not boxes or not all(k in boxes for k in ("train", "val", "test")):
        raise ValueError(
            "region 划分需要 split.params.boxes = {train: [lon0,lon1,lat0,lat1], val: ..., test: ...}"
        )
    need = {"latitude", "longitude"}
    if not need.issubset(bundle.meta.columns):
        raise ValueError(f"region 划分需要 meta 列 {need}，实际: {list(bundle.meta.columns)}")
    lat = bundle.meta["latitude"].to_numpy(dtype=float)
    lon = bundle.meta["longitude"].to_numpy(dtype=float)
    masks = {}
    for key in ("train", "val", "test"):
        lon0, lon1, lat0, lat1 = map(float, boxes[key])
        masks[key] = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    # 先匹配者得；被任何盒子之外的样本显式丢弃
    claimed = masks["train"] | masks["val"] | masks["test"]
    overlap = (
        (masks["train"] & masks["val"])
        | (masks["train"] & masks["test"])
        | (masks["val"] & masks["test"])
    )
    if overlap.any():
        raise ValueError(f"region 盒子互相重叠，涉及 {int(overlap.sum())} 个样本；boxes 必须互斥")
    dropped = int((~claimed).sum())
    result = SplitResult(
        np.flatnonzero(masks["train"]),
        np.flatnonzero(masks["val"]),
        np.flatnonzero(masks["test"]),
        "region",
        notes=[f"boxes={boxes}"],
        dropped=dropped,
    )
    if dropped:
        result.notes.append(f"{dropped} 个样本不在任何盒子内，已丢弃（显式记录）")
    return result


def _split_source(bundle: SampleBundle, split_cfg: dict[str, Any]) -> SplitResult:
    params = split_cfg.get("params") or {}
    keys = ("train_sources", "val_sources", "test_sources")
    if not all(params.get(k) for k in keys):
        raise ValueError(
            "source 划分需要 split.params = {train_sources: [...], val_sources: [...], test_sources: [...]}"
        )
    src = bundle.source_labels.to_numpy()
    tr = np.flatnonzero(np.isin(src, list(params["train_sources"])))
    va = np.flatnonzero(np.isin(src, list(params["val_sources"])))
    te = np.flatnonzero(np.isin(src, list(params["test_sources"])))
    claimed = np.isin(src, list(params["train_sources"] + params["val_sources"] + params["test_sources"]))
    result = SplitResult(
        tr, va, te, "source",
        notes=[f"train_sources={params['train_sources']} val={params['val_sources']} test={params['test_sources']}"],
        dropped=int((~claimed).sum()),
    )
    if result.dropped:
        result.notes.append(f"{result.dropped} 个样本来源不在配置内，已丢弃（显式记录）")
    return result


_DISPATCH = {
    "random": _split_random,
    "time": _split_time,
    "year": _split_year,
    "region": _split_region,
    "source": _split_source,
}


def split_indices(bundle: SampleBundle, split_cfg: dict[str, Any]) -> SplitResult:
    """按 split.strategy 划分并强校验互斥性。"""
    strategy = (split_cfg or {}).get("strategy", "random")
    if strategy not in _DISPATCH:
        raise ValueError(f"未知划分策略 {strategy!r}，可选: {available_strategies()}")
    if bundle.n_samples < 10:
        raise ValueError(f"样本数过少（{bundle.n_samples}），不足以划分 train/val/test")
    result = _DISPATCH[strategy](bundle, split_cfg)
    result.check_disjoint()
    if result.train.size == 0 or result.test.size == 0:
        raise ValueError(f"划分结果不完整: {result.as_dict()}")
    return result
