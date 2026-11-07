"""统一加载入口：配置 → SampleBundle（含合成数据自举、分级缓存、分批读取）。

缓存策略（v2.0）
----------------

``load_dataset`` 会在**预处理缓存**里按「文件版本 + 数据处理规格」缓存整份
``SampleBundle``：

* key = 文件 size/mtime（可选内容哈希） + adapter/features/targets/范围/抽样；
* 文件一变 → key 变 → 旧缓存自动失效（不需要手工清理）；
* 缓存默认位于 ``.cache/cloudwater_ml``（**在 data/ 之外**），
  且 ``DataCache`` 硬性拒绝写入 ``data/raw``；
* 任何缓存读写失败都**不阻断**数据加载（降级为直读并告警）。

关闭方式：``data.cache.enabled: false``。
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .adapters import available_adapters, get_adapter
from .schema import SampleBundle
from .synthetic import ensure_synthetic_dataset

__all__ = [
    "load_dataset",
    "inspect_data",
    "iter_batches",
    "resolve_data_cfg",
    "clear_cache",
    "cache_stats",
]

_logger = get_logger("data.loader")


def resolve_data_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """兼容两种入参：完整实验配置（取 data 段）或纯 data 配置。"""
    if isinstance(cfg, dict) and "data" in cfg and isinstance(cfg["data"], dict):
        return cfg["data"]
    return cfg


def _validate(data_cfg: dict[str, Any]) -> None:
    adapter = data_cfg.get("adapter")
    if not adapter:
        raise ValueError("缺少 data.adapter（可选: " + " | ".join(available_adapters()) + "）")
    if not data_cfg.get("features"):
        raise ValueError("缺少 data.features（输入特征列表，按数据源实际变量配置）")
    if not data_cfg.get("targets"):
        raise ValueError("缺少 data.targets（预测目标列表，按数据源实际变量配置）")


def _bootstrap_synthetic(data_cfg: dict[str, Any]) -> None:
    """合成测试数据不存在时按配置自举生成（仅限 data/synthetic 路径）。"""
    path = Path(data_cfg.get("path") or "")
    if path.suffix == ".npz" and not path.exists() and "synthetic" in path.as_posix():
        synth = data_cfg.get("synthetic") or {}
        ensure_synthetic_dataset(
            path,
            n_samples=int(synth.get("n_samples", 5000)),
            seed=int(synth.get("seed", 42)),
            start_year=int(synth.get("start_year", 2015)),
            end_year=int(synth.get("end_year", 2019)),
        )


# ----------------------------------------------------------------------
# 缓存
# ----------------------------------------------------------------------
def _make_cache(data_cfg: dict[str, Any]) -> Any | None:
    """按配置构造缓存；关闭或不可用时返回 None（降级为直读）。"""
    cfg = data_cfg.get("cache") or {}
    if cfg.get("enabled", True) is False:
        return None
    try:
        from .cache import DataCache

        return DataCache(
            root=cfg.get("root"),
            hash_files=bool(cfg.get("hash_files", False)),
        )
    except Exception as exc:  # 缓存不可用不应阻断数据加载
        _logger.warning("缓存不可用（降级为直读）: %s", exc)
        return None


def _bundle_key(cache: Any, data_cfg: dict[str, Any]) -> str | None:
    """数据版本化 key：文件版本 + 处理规格。"""
    path = data_cfg.get("path")
    if not path:
        return None
    spec = {
        k: data_cfg.get(k)
        for k in (
            "adapter", "features", "targets", "meta_columns",
            "sampling", "netcdf", "source_label", "csv",
        )
    }
    try:
        return cache.versioned_key(
            path, "bundle", json.dumps(spec, sort_keys=True, default=str)
        )
    except FileNotFoundError:
        return None
    except Exception as exc:
        _logger.debug("构造缓存 key 失败: %s", exc)
        return None


def _load_cached_bundle(cache: Any, key: str) -> SampleBundle | None:
    """从缓存还原 SampleBundle；任何异常都当作 miss。"""
    try:
        meta = cache.cached_preprocessed(f"{key}::meta")
        arrays = cache.cached_preprocessed(f"{key}::arrays")
        if not isinstance(meta, dict) or not isinstance(arrays, dict):
            return None
        return SampleBundle(
            X=np.asarray(arrays["X"]),
            Y=np.asarray(arrays["Y"]),
            feature_names=list(meta["feature_names"]),
            target_names=list(meta["target_names"]),
            meta=pd.DataFrame(meta.get("meta") or []),
            units=dict(meta.get("units") or {}),
            source_file=str(meta.get("source_file") or ""),
            source_origin=str(meta.get("source_origin") or ""),
        )
    except Exception as exc:  # 缓存损坏 → miss，不阻断
        _logger.warning("读取数据缓存失败（回退直读）: %s", exc)
        return None


def _store_cached_bundle(cache: Any, key: str, bundle: SampleBundle) -> None:
    """写入缓存；失败只告警。"""
    try:
        cache.store_preprocessed(
            f"{key}::arrays",
            {"X": np.asarray(bundle.X), "Y": np.asarray(bundle.Y)},
            fmt="npz",
            source=bundle.source_file or None,
        )
        cache.store_preprocessed(
            f"{key}::meta",
            {
                "feature_names": list(bundle.feature_names),
                "target_names": list(bundle.target_names),
                "meta": bundle.meta.to_dict(orient="records"),
                "units": dict(bundle.units or {}),
                "source_file": bundle.source_file,
                "source_origin": bundle.source_origin,
                "n_samples": int(bundle.n_samples),
            },
            fmt="json",
            source=bundle.source_file or None,
        )
    except Exception as exc:
        _logger.warning("写入数据缓存失败（不影响本次运行）: %s", exc)


# ----------------------------------------------------------------------
def load_dataset(cfg: dict[str, Any]) -> SampleBundle:
    """按配置选择适配器加载数据。NetCDF 走按需筛选+体积守卫，绝不默认整载入。

    命中预处理缓存时直接返回缓存副本；否则读原始数据并回填缓存。
    """
    data_cfg = resolve_data_cfg(cfg)
    _validate(data_cfg)
    _bootstrap_synthetic(data_cfg)

    cache = _make_cache(data_cfg)
    key = _bundle_key(cache, data_cfg) if cache is not None else None
    if cache is not None and key:
        cached = _load_cached_bundle(cache, key)
        if cached is not None:
            _logger.info("数据命中预处理缓存（%d 样本）", cached.n_samples)
            return cached

    bundle = get_adapter(data_cfg["adapter"]).load(data_cfg)
    # 追溯：数据来源文件 + 配置路径
    if not bundle.source_file:
        bundle.source_file = str(data_cfg.get("path") or "")

    if cache is not None and key:
        _store_cached_bundle(cache, key, bundle)
    return bundle


def inspect_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """只读检查：结构/变量/维度/单位/来源契约（不载入变量数值）。"""
    data_cfg = resolve_data_cfg(cfg)
    if not data_cfg.get("adapter"):
        raise ValueError("缺少 data.adapter")
    _bootstrap_synthetic(data_cfg)

    cache = _make_cache(data_cfg)
    path = data_cfg.get("path")
    if cache is not None and path:
        cached = cache.cached_metadata(path, "inspect")
        if cached is not None:
            info = dict(cached)
            info["cache"] = "metadata_hit"
        else:
            info = get_adapter(data_cfg["adapter"]).inspect(data_cfg)
            try:
                cache.store_metadata(path, "inspect", info)
            except Exception as exc:  # pragma: no cover
                _logger.debug("写入元数据缓存失败: %s", exc)
    else:
        info = get_adapter(data_cfg["adapter"]).inspect(data_cfg)

    info["requested_features"] = list(data_cfg.get("features") or [])
    info["requested_targets"] = list(data_cfg.get("targets") or [])
    info["source_label_override"] = data_cfg.get("source_label")
    return info


def iter_batches(cfg: dict[str, Any], batch_samples: int = 100_000) -> Iterator[SampleBundle]:
    """分批读取入口（目前 netcdf 支持；其余适配器抛清晰错误）。"""
    data_cfg = resolve_data_cfg(cfg)
    _validate(data_cfg)
    yield from get_adapter(data_cfg["adapter"]).iter_batches(data_cfg, batch_samples)


# ----------------------------------------------------------------------
# 缓存运维（可清理 / 可重建 / 可追溯）
# ----------------------------------------------------------------------
def cache_stats(data_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """缓存统计：命中/未命中/条目数/占用空间。"""
    cache = _make_cache(resolve_data_cfg(data_cfg) if data_cfg else {})
    if cache is None:
        return {"enabled": False, "reason": "缓存已关闭或不可用"}
    return {"enabled": True, **cache.stats()}


def clear_cache(
    data_cfg: dict[str, Any] | None = None,
    *,
    level: str | None = None,
) -> dict[str, Any]:
    """清理缓存（可重建）：删除指定级别或全部缓存条目。**不会碰 data/raw**。"""
    cache = _make_cache(resolve_data_cfg(data_cfg) if data_cfg else {})
    if cache is None:
        return {"enabled": False, "removed": 0}
    removed = cache.clear(level)
    return {"enabled": True, "level": level or "all", "removed": int(removed)}
