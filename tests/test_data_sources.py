"""真实/仿真数据源融合测试：来源追溯、混合采样、防泄漏分组。

对应审计里的 BLOCKER B1：来源标签未解析时**不得**把训练集筛空。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cloudwater_ml.data.schema import SampleBundle
from cloudwater_ml.data.sources import (
    MixedSampler,
    SourceKind,
    build_provenance,
    domain_of_sample,
    scene_ids,
)


def _bundle(n=100, sources=None):
    if sources is None:
        n_sim = n // 2
        n_real = n // 3
        sources = ["simulated"] * n_sim + ["real"] * n_real + ["unknown"] * (n - n_sim - n_real)
    sources = list(sources)
    assert len(sources) == n, "sources 长度必须与 n 一致"
    X = np.arange(n * 3, dtype=float).reshape(n, 3)
    Y = np.arange(n * 2, dtype=float).reshape(n, 2)
    meta = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "source": sources,
        "dataset": ["synth"] * (n // 2) + ["era5"] * (n - n // 2),
        "source_origin": ["dataset_contract"] * n,
        "time": [f"2015-01-{i % 28 + 1:02d}" for i in range(n)],
        "latitude": [30.0 + i * 0.01 for i in range(n)],
        "longitude": [100.0 + i * 0.01 for i in range(n)],
        "level": [500.0] * n,
    })
    return SampleBundle(
        X=X, Y=Y, meta=meta,
        feature_names=["a", "b", "c"], target_names=["clwc", "ciwc"],
    )


# ----------------------------------------------------------------------
def test_source_kind_normalization():
    assert SourceKind.normalize("real") is SourceKind.REAL
    assert SourceKind.normalize("simulated") is SourceKind.SIMULATED
    assert SourceKind.normalize("仿真") is SourceKind.SIMULATED
    # 关键：无法判定必须落到 unknown，**不能默认 simulated**
    assert SourceKind.normalize("mystery") is SourceKind.UNKNOWN
    assert SourceKind.normalize(None) is SourceKind.UNKNOWN
    assert SourceKind.normalize("") is SourceKind.UNKNOWN


def test_domain_of_sample():
    assert domain_of_sample("real") == "real"
    assert domain_of_sample("simulated") == "simulated"
    assert domain_of_sample("nonsense") == "unknown"


def test_provenance_contains_required_fields():
    prov = build_provenance(_bundle())
    for col in ("sample_id", "source", "dataset_id", "source_origin",
                "generation_method", "generation_config", "time",
                "latitude", "longitude", "level", "scene_id"):
        assert col in prov.columns, f"缺少溯源字段 {col}"
    assert len(prov) == 100


def test_provenance_scene_id_groups_same_scene():
    b = _bundle()
    prov = build_provenance(b)
    # 同 dataset + 同 time + 同 level + 同经纬 → 同一 scene；这里时间不同 → scene 不同
    assert prov["scene_id"].nunique() > 1


def test_scene_ids_returns_one_per_sample():
    b = _bundle()
    ids = scene_ids(b)
    assert len(ids) == b.n_samples


# ----------------------------------------------------------------------
def test_ratio_sampling_records_actual_fraction():
    b = _bundle()
    sampler = MixedSampler(b, seed=1)
    counts = sampler.counts()
    sub, info = sampler.sample(real_fraction=0.5)
    n_real = counts["real"]
    n_sim = counts["simulated"]
    # 目标 real 份额超过可用量时，名额在两类之间转移（不丢样本、不伪造标签）
    assert info["n_real"] == n_real
    assert info["n_simulated"] == n_sim
    assert info["n_selected"] == n_real + n_sim
    assert info["actual_real_fraction"] == pytest.approx(n_real / (n_real + n_sim))
    assert any("real_fraction" in n for n in info["notes"]), "配比未达成必须留痕"


def test_sampling_never_empties_when_labels_unknown():
    """BLOCKER B1：全是 unknown 时不能把训练集筛空。"""
    b = _bundle(n=40, sources=["unknown"] * 40)
    sampler = MixedSampler(b, seed=1)
    sub, info = sampler.sample(real_fraction=1.0)
    assert info["n_selected"] == 40
    assert info["n_dropped"] == 0
    assert sub.n_samples == 40


def test_weighted_strategy():
    b = _bundle()
    sampler = MixedSampler(b, seed=2)
    sub, info = sampler.sample(
        strategy="weighted", weights={"real": 0.3, "simulated": 0.7}
    )
    assert info["n_selected"] > 0
    assert set(info["counts"]) == {"real", "simulated", "unknown"}


def test_source_strategy():
    b = _bundle()
    sampler = MixedSampler(b, seed=3)
    sub, info = sampler.sample(strategy="source", source_counts={"real": 10})
    assert info["n_real"] == 10


def test_stratified_strategy_keeps_layers():
    b = _bundle()
    sampler = MixedSampler(b, seed=4)
    sub, info = sampler.sample(
        real_fraction=0.5, strategy="stratified", stratify_by="dataset"
    )
    assert info["n_selected"] > 0


def test_stratified_requires_field():
    b = _bundle()
    sampler = MixedSampler(b, seed=5)
    with pytest.raises(ValueError):
        sampler.sample(strategy="stratified", real_fraction=0.5, stratify_by="nope")


def test_unknown_strategy_raises():
    b = _bundle()
    sampler = MixedSampler(b, seed=6)
    with pytest.raises(ValueError):
        sampler.sample(strategy="magic")


def test_keep_scenes_intact_expands_selection():
    b = _bundle(n=20)
    sampler = MixedSampler(b, seed=7)
    sub, info = sampler.sample(real_fraction=1.0, keep_scenes_intact=True)
    assert info["n_selected"] >= 1
    assert any("scene_id" in n for n in info["notes"])


# ----------------------------------------------------------------------
# 数据缓存（loader 集成）
# ----------------------------------------------------------------------
def test_loader_cache_roundtrip_and_clear(tmp_path, monkeypatch):
    """缓存命中返回等价 bundle；clear 后能重建；任何情况下都不写 data/raw。"""
    import json as _json

    import numpy as np

    from cloudwater_ml.data.cache import DataCache
    from cloudwater_ml.data.loader import _load_cached_bundle, _store_cached_bundle

    cache = DataCache(root=tmp_path / "cache")
    bundle = _bundle(n=30)
    path = tmp_path / "toy.npz"
    np.savez(path, X=bundle.X, Y=bundle.Y)
    key = cache.versioned_key(path, "bundle", _json.dumps({"adapter": "npz"}, sort_keys=True))
    _store_cached_bundle(cache, key, bundle)
    restored = _load_cached_bundle(cache, key)
    assert restored is not None
    assert restored.n_samples == bundle.n_samples
    assert list(restored.feature_names) == list(bundle.feature_names)
    assert np.allclose(restored.X, bundle.X)
    assert restored.units == bundle.units or restored.units == {}

    # 文件变化 → key 变化 → 旧缓存不再命中（自动失效）
    path.write_bytes(path.read_bytes() + b"x")
    key2 = cache.versioned_key(path, "bundle", _json.dumps({"adapter": "npz"}, sort_keys=True))
    assert key2 != key
    assert _load_cached_bundle(cache, key2) is None

    # clear 可清理、可重建
    assert cache.clear() >= 1
    assert _load_cached_bundle(cache, key) is None


def test_cache_refuses_to_write_into_data_raw():
    """缓存根目录不得落在 data/raw 内（硬保护）。"""
    import pytest as _pytest

    from cloudwater_ml.data.cache import DataCache

    with _pytest.raises((ValueError, OSError)):
        DataCache(root="data/raw/cache")
