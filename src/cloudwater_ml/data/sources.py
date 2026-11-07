"""真实 / 仿真数据源融合（v2.0）：来源追溯、混合采样、防泄漏分组。

设计要点
--------

1. **来源必须可追溯**：每条样本都能回答「来自哪个数据集、是真观测还是仿真、
   怎么生成的、生成参数是什么、原始文件/契约是谁」。
2. **仿真不得冒充真实**：来源标签只认「数据契约声明」或「显式配置」，
   解析不出来就是 ``unknown``——**绝不默认成 simulated**（v1.0 的坑）。
3. **同场景样本不得跨集**：来自同一原始场景（如同一 ERA5 时次 + 同一气压层
   切片）的高度相关样本会被打上同一个 ``scene_id``，划分时按组绑定，
   避免训练集里出现测试集场景的「近邻」。
4. **混合只作用于训练集**：测试集永远保持原样，配比实验才有意义。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from ..utils.exceptions import ConfigurationError

__all__ = [
    "SourceKind",
    "SampleProvenance",
    "build_provenance",
    "scene_ids",
    "MixedSampler",
    "domain_of_sample",
]


class SourceKind(str, Enum):
    """样本来源类别。"""

    REAL = "real"
    SIMULATED = "simulated"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, value: Any) -> SourceKind:
        """把任意来源字符串归一化；**无法判定就是 UNKNOWN**（不猜）。"""
        if value is None:
            return cls.UNKNOWN
        text = str(value).strip().lower()
        if text in ("real", "observed", "observation", "satellite", "gauge", "真实", "观测"):
            return cls.REAL
        if text in ("simulated", "synthetic", "simulation", "reanalysis_like", "仿真", "合成"):
            return cls.SIMULATED
        return cls.UNKNOWN


@dataclass
class SampleProvenance:
    """单条样本的来源档案。"""

    sample_id: str = ""
    source: str = SourceKind.UNKNOWN.value
    dataset_id: str = ""
    source_origin: str = ""
    generation_method: str = ""
    generation_config: dict[str, Any] = field(default_factory=dict)
    time: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    level: float | None = None
    scene_id: str = ""
    parent_scene_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "dataset_id": self.dataset_id,
            "source_origin": self.source_origin,
            "generation_method": self.generation_method,
            "generation_config": dict(self.generation_config),
            "time": self.time,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "level": self.level,
            "scene_id": self.scene_id,
            "parent_scene_id": self.parent_scene_id,
        }


# ----------------------------------------------------------------------
def _meta(bundle: Any) -> pd.DataFrame:
    meta = getattr(bundle, "meta", None)
    if meta is None:
        return pd.DataFrame()
    return meta


def _scene_key(row: pd.Series, *, level_block: float | None = None) -> str:
    """构造「原始场景」标识：数据集 + 时间 + 层 + 空间块。

    同一场景的不同变体（不同噪声强度、不同仿真条件）会落在同一个 scene_id 上，
    划分时必须整组进出同一集合。
    """
    dataset = str(row.get("dataset", "") or "")
    time = str(row.get("time", "") or "")
    level = row.get("level", None)
    if level_block is not None and level is not None and not pd.isna(level):
        import contextlib

        with contextlib.suppress(TypeError, ValueError):
            level = float(np.floor(float(level) / level_block) * level_block)
    lat = row.get("latitude", None)
    lon = row.get("longitude", None)
    return f"{dataset}|{time}|{level}|{lat}|{lon}"


def build_provenance(
    bundle: Any,
    *,
    level_block: float | None = None,
    generation_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """为数据 bundle 的每条样本生成来源档案表。

    输出列：``sample_id / source / dataset_id / source_origin / generation_method /
    generation_config / time / latitude / longitude / level / scene_id``。

    ``generation_config`` 优先取 meta 里已有的列（如 ``generation_config``、
    ``sim_config``），否则用传入的兜底值。
    """
    meta = _meta(bundle)
    n = int(getattr(bundle, "n_samples", len(meta)))
    if meta.empty:
        meta = pd.DataFrame(index=range(n))

    records: list[SampleProvenance] = []
    for i in range(n):
        row = meta.iloc[i] if i < len(meta) else pd.Series(dtype=object)
        sample_id = str(row.get("sample_id", f"idx_{i}") or f"idx_{i}")
        source = SourceKind.normalize(row.get("source")).value
        dataset_id = str(row.get("dataset", "") or "")
        origin = str(row.get("source_origin", "") or "")
        method = str(row.get("generation_method", "") or "")
        cfg_raw = row.get("generation_config", None)
        cfg: dict[str, Any] = {}
        if isinstance(cfg_raw, dict):
            cfg = dict(cfg_raw)
        elif isinstance(cfg_raw, str) and cfg_raw.strip():
            try:
                import json

                parsed = json.loads(cfg_raw)
                cfg = parsed if isinstance(parsed, dict) else {"raw": cfg_raw}
            except Exception:
                cfg = {"raw": cfg_raw}
        if not cfg and generation_config:
            cfg = dict(generation_config)
        prov = SampleProvenance(
            sample_id=sample_id,
            source=source,
            dataset_id=dataset_id,
            source_origin=origin,
            generation_method=method,
            generation_config=cfg,
            time=None if pd.isna(row.get("time", None)) else str(row.get("time")),
            latitude=_to_float(row.get("latitude", None)),
            longitude=_to_float(row.get("longitude", None)),
            level=_to_float(row.get("level", None)),
        )
        prov.scene_id = _scene_key(row, level_block=level_block)
        records.append(prov)

    df = pd.DataFrame([r.to_dict() for r in records])
    if not df.empty:
        df["scene_id"] = [
            hashlib.sha1(s.encode("utf-8")).hexdigest()[:12] for s in df["scene_id"]
        ]
    return df


def _to_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def scene_ids(bundle: Any, *, level_block: float | None = None) -> np.ndarray:
    """返回每个样本所属的 ``scene_id``（用于分组划分防泄漏）。"""
    prov = build_provenance(bundle, level_block=level_block)
    if prov.empty:
        return np.array([], dtype=object)
    return prov["scene_id"].to_numpy()


def domain_of_sample(source: Any) -> str:
    """样本所属的**评测域**：``simulated`` / ``real`` / ``unknown``。

    用于区分「仿真域内测试」「真实域测试」「跨域泛化测试」。
    """
    return SourceKind.normalize(source).value


# ----------------------------------------------------------------------
class MixedSampler:
    """真实 / 仿真混合采样器（**只应作用于训练集**）。

    采样策略
    --------

    ``ratio``
        按真实数据比例抽样（不足时靠拢并**如实记录**实际比例，不伪造标签）。
    ``source``
        按数据源抽样（每个来源各取指定数量）。
    ``weighted``
        按来源权重抽样（权重归一化后分配名额）。
    ``stratified``
        分层抽样：先按 ``stratify_by``（如年份/季节/区域）分层，
        再在层内按比例抽样，保持各层分布不被破坏。
    """

    def __init__(
        self,
        bundle: Any,
        *,
        seed: int = 42,
        level_block: float | None = None,
    ) -> None:
        self.bundle = bundle
        self.seed = int(seed)
        self.provenance = build_provenance(bundle, level_block=level_block)

    # -- 基础信息 ----------------------------------------------------
    @property
    def source_labels(self) -> np.ndarray:
        if self.provenance.empty:
            return np.array([], dtype=object)
        return self.provenance["source"].to_numpy()

    def counts(self) -> dict[str, int]:
        labels = self.source_labels
        return {
            "real": int((labels == SourceKind.REAL.value).sum()),
            "simulated": int((labels == SourceKind.SIMULATED.value).sum()),
            "unknown": int((labels == SourceKind.UNKNOWN.value).sum()),
        }

    # -- 采样 --------------------------------------------------------
    def sample(
        self,
        *,
        real_fraction: float | None = None,
        strategy: str = "ratio",
        source_counts: dict[str, int] | None = None,
        weights: dict[str, float] | None = None,
        stratify_by: str | None = None,
        keep_scenes_intact: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        """执行采样，返回 ``(子集 bundle, 采样信息)``。

        ``keep_scenes_intact=True`` 时，被选中的样本会**连同同场景的其它变体**
        一起纳入（避免同场景样本被拆散造成半泄漏）。
        """
        labels = self.source_labels
        n = int(getattr(self.bundle, "n_samples", len(labels)))
        info: dict[str, Any] = {
            "strategy": strategy,
            "requested_real_fraction": real_fraction,
            "n_original": n,
            "notes": [],
        }
        if n == 0:
            info.update({
                "n_selected": 0,
                "actual_real_fraction": None,
                "counts": {"real": 0, "simulated": 0, "unknown": 0},
                "n_real": 0, "n_simulated": 0, "n_unknown": 0, "n_dropped": 0,
            })
            return self.bundle, info

        strategy = strategy.lower()
        rng = np.random.default_rng(self.seed)

        if strategy == "ratio":
            if real_fraction is None:
                chosen = np.arange(n)
            else:
                if not 0.0 <= float(real_fraction) <= 1.0:
                    raise ConfigurationError(
                        f"real_fraction 必须在 [0,1]，当前 {real_fraction}"
                    )
                real_idx = np.flatnonzero(labels == SourceKind.REAL.value)
                sim_idx = np.flatnonzero(labels == SourceKind.SIMULATED.value)
                if real_idx.size == 0 and sim_idx.size == 0:
                    # 来源标签完全无法解析时**保持全部样本**：
                    # 筛空会让 scaler 在空数组上拟合出 NaN，整场训练静默报废。
                    chosen = np.arange(n)
                    info["notes"].append(
                        f"来源标签中既无 real 也无 simulated（全部 unknown），"
                        f"已保持全部 {n} 个样本不变（不筛空、不伪造标签）"
                    )
                    subset = self.bundle.subset(chosen)
                    info.update({
                        "n_selected": int(chosen.size),
                        "n_dropped": 0,
                        "actual_real_fraction": None,
                        "counts": {"real": 0, "simulated": 0, "unknown": int(chosen.size)},
                        "n_real": 0, "n_simulated": 0, "n_unknown": int(chosen.size),
                    })
                    return subset, info
                want_real = int(round(float(real_fraction) * n))
                want_sim = n - want_real
                take_real = min(want_real, len(real_idx))
                take_sim = min(want_sim, len(sim_idx))
                # 名额不足时在两类之间转移（不丢样本、不伪造标签）
                if take_real < want_real:
                    take_sim = min(len(sim_idx), want_sim + (want_real - take_real))
                if take_sim < want_sim:
                    take_real = min(len(real_idx), want_real + (want_sim - take_sim))
                sel_real = rng.choice(real_idx, take_real, replace=False) if take_real else np.array([], int)
                sel_sim = rng.choice(sim_idx, take_sim, replace=False) if take_sim else np.array([], int)
                chosen = np.sort(np.concatenate([sel_real, sel_sim]))
                if n - chosen.size:
                    info["notes"].append(
                        f"为接近配比丢弃 {n - chosen.size} 个样本"
                        f"（real 可用 {len(real_idx)}，sim 可用 {len(sim_idx)}）"
                    )
        elif strategy == "source":
            spec = source_counts or {}
            parts = []
            for kind, count in spec.items():
                idx = np.flatnonzero(labels == SourceKind.normalize(kind).value)
                take = min(int(count), len(idx))
                parts.append(rng.choice(idx, take, replace=False) if take else np.array([], int))
            chosen = np.sort(np.concatenate(parts)) if parts else np.arange(0)
        elif strategy == "weighted":
            if not weights:
                raise ConfigurationError(
                    "weighted 策略需要 weights（如 {'real': 0.7, 'simulated': 0.3}）"
                )
            total = float(sum(max(0.0, float(v)) for v in weights.values()))
            if total <= 0:
                raise ConfigurationError("weights 之和必须为正")
            parts = []
            for kind, w in weights.items():
                idx = np.flatnonzero(labels == SourceKind.normalize(kind).value)
                quota = int(round(float(w) / total * n))
                take = min(quota, len(idx))
                parts.append(rng.choice(idx, take, replace=False) if take else np.array([], int))
            chosen = np.sort(np.concatenate(parts)) if parts else np.arange(0)
        elif strategy == "stratified":
            if not stratify_by:
                raise ConfigurationError(
                    "stratified 策略需要 stratify_by（如 'year' / 'season' / 'region'）"
                )
            meta = _meta(self.bundle)
            if stratify_by not in meta.columns:
                raise ConfigurationError(
                    f"meta 里没有分层字段 {stratify_by!r}（可用: {list(meta.columns)}）"
                )
            chosen_parts: list[np.ndarray] = []
            groups = meta[stratify_by].astype(str).to_numpy()
            for value in np.unique(groups):
                idx = np.flatnonzero(groups == value)
                if real_fraction is None:
                    chosen_parts.append(idx)
                    continue
                real_mask = labels[idx] == SourceKind.REAL.value
                real_idx = idx[real_mask]
                other_idx = idx[~real_mask]
                want_real = int(round(float(real_fraction) * idx.size))
                take_real = min(want_real, len(real_idx))
                take_other = min(idx.size - take_real, len(other_idx))
                sel_real = rng.choice(real_idx, take_real, replace=False) if take_real else np.array([], int)
                sel_other = rng.choice(other_idx, take_other, replace=False) if take_other else np.array([], int)
                chosen_parts.append(np.concatenate([sel_real, sel_other]))
            chosen = np.sort(np.concatenate(chosen_parts)) if chosen_parts else np.arange(0)
        else:
            raise ConfigurationError(
                f"未知采样策略 {strategy!r}",
                hint="可选: ratio | source | weighted | stratified",
            )

        if keep_scenes_intact and not self.provenance.empty:
            scenes = self.provenance["scene_id"].to_numpy()
            sel_scenes = set(scenes[chosen])
            chosen = np.flatnonzero(np.isin(scenes, list(sel_scenes)))
            info["notes"].append(
                f"按 scene_id 整组纳入：{len(sel_scenes)} 个场景 → {chosen.size} 个样本"
            )

        subset = self.bundle.subset(chosen)
        sel_labels = labels[chosen] if chosen.size else np.array([], dtype=object)
        actual = (
            float((sel_labels == SourceKind.REAL.value).mean()) if sel_labels.size else None
        )
        counts = {
            "real": int((sel_labels == SourceKind.REAL.value).sum()),
            "simulated": int((sel_labels == SourceKind.SIMULATED.value).sum()),
            "unknown": int((sel_labels == SourceKind.UNKNOWN.value).sum()),
        }
        info.update({
            "n_selected": int(chosen.size),
            "n_dropped": int(n - chosen.size),
            "actual_real_fraction": actual,
            "counts": counts,
            # 与 runner.apply_mix 的字段口径保持一致
            "n_real": counts["real"],
            "n_simulated": counts["simulated"],
            "n_unknown": counts["unknown"],
        })
        if real_fraction is not None and actual is not None and abs(actual - float(real_fraction)) > 1e-9:
            info["notes"].append(
                    f"请求 real_fraction={real_fraction}，实际 {actual:.3f}"
                    " —— 来源标签分布不满足请求，不调整、不伪造标签"
                )
        return subset, info
