"""数据层 v2.0 回归测试：覆盖审计缺陷 D1/D2/D3/D4/D5/D6/D7/D8/D10/D11/D12/D13。

原则：**真实读真实文件** —— 用 tmp_path 造 NetCDF / parquet fixture（含
time/lat/lon/level 维、单位属性、_FillValue、NaN 区域），不做 mock 数据源。

每条测试注明对应的缺陷编号，便于回溯。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cloudwater_ml.data import inspect_data, iter_batches, load_dataset, run_quality_checks
from cloudwater_ml.data.adapters import array_adapter
from cloudwater_ml.data.cache import DataCache, compute_file_version, make_key
from cloudwater_ml.data.preprocessing import (
    NAN_POLICY_IMPUTE,
    StandardScaler,
    fit_scalers,
    verify_fit_evidence,
)
from cloudwater_ml.data.schema import SOURCE_UNKNOWN, SampleBundle

FEATURES = ["t", "q"]
TARGETS = ["clwc"]


# ----------------------------------------------------------------------
# 真实文件 fixture
# ----------------------------------------------------------------------
def _make_nc(
    path: Path,
    *,
    n_time: int = 6,
    n_lat: int = 4,
    n_lon: int = 5,
    n_level: int = 3,
    level_name: str = "level",
    contract: dict | None = None,
    with_nan: bool = True,
    freq: str = "6h",
) -> Path:
    """写一个真实 NetCDF：time/lat/lon/level 维 + 单位属性 + _FillValue + NaN 区域。"""
    rng = np.random.default_rng(11)
    times = pd.date_range("2020-01-01", periods=n_time, freq=freq)
    lat = np.linspace(28.0, 38.0, n_lat)
    lon = np.linspace(78.0, 98.0, n_lon)
    lev = np.array([1000.0, 850.0, 700.0, 500.0], dtype=float)[:n_level]
    dims = ((("time", level_name, "latitude", "longitude")) if n_level
            else ("time", "latitude", "longitude"))
    shape = (n_time, n_level, n_lat, n_lon) if n_level else (n_time, n_lat, n_lon)
    coords: dict[str, object] = {"time": times, "latitude": lat, "longitude": lon}
    if n_level:
        coords[level_name] = lev
    ds = xr.Dataset(coords=coords)
    for name, unit, base in (("t", "K", 250.0), ("q", "kg kg-1", 5e-3), ("clwc", "kg kg-1", 1e-4)):
        arr = (rng.normal(base, abs(base) * 0.1 + 1e-3, shape)).astype("float32")
        ds[name] = (dims, arr, {"units": unit, "_FillValue": np.float32(-9999.0), "long_name": name})
    if with_nan:
        # 缺测区域（读回来是 NaN —— _FillValue → NaN）
        ds["clwc"].values[0, 0, 0, 0] = np.nan
        ds["t"].values[1, 0, 0, 0] = np.nan
    if n_level:
        ds[level_name].attrs["units"] = "hPa"
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["units"] = "degrees_east"
    if contract is not None:
        ds.attrs["atmosgen:contract"] = json.dumps(contract)
    ds.to_netcdf(path)
    ds.close()
    return path


def _nc_cfg(path: Path, **over) -> dict:
    cfg: dict = {
        "adapter": "netcdf",
        "path": str(path),
        "features": list(FEATURES),
        "targets": list(TARGETS),
        "netcdf": {},
        "sampling": {},
    }
    cfg.update(over)
    return cfg


# ----------------------------------------------------------------------
# D1：来源标签未解析不得默认 simulated
# ----------------------------------------------------------------------
def test_d1_unresolved_source_is_unknown_not_simulated(tmp_path):
    """无 atmosgen 契约的真实 NetCDF → source='unknown'（绝不 simulated）。"""
    p = _make_nc(tmp_path / "real.nc")  # 不写契约
    b = load_dataset(_nc_cfg(p))
    assert b.source_counts() == {SOURCE_UNKNOWN: b.n_samples}
    assert b.meta["source_origin"].unique().tolist() == ["unresolved"]
    assert b.source_origin == "unresolved"
    assert "simulated" not in b.source_counts()


def test_d1_contract_and_config_override_source(tmp_path):
    p = _make_nc(tmp_path / "sim.nc", contract={"source": {"kind": "simulated"}})
    b = load_dataset(_nc_cfg(p))
    assert b.source_counts() == {"simulated": b.n_samples}
    assert b.source_origin == "dataset_contract"

    b2 = load_dataset(_nc_cfg(p, source_label="real"))
    assert b2.source_counts() == {"real": b2.n_samples}
    assert b2.source_origin == "config"


def test_d1_empty_meta_defaults_to_unknown_not_simulated():
    """空 meta 的 bundle 来源标签必须是 unknown（v1 默认为 simulated）。"""
    b = SampleBundle(X=np.zeros((3, 2)), Y=np.zeros((3, 1)), feature_names=["a", "b"],
                     target_names=["c"])
    assert b.source_counts() == {SOURCE_UNKNOWN: 3}
    assert b.source_resolution()["resolved"] is False


def test_d1_subset_empty_raises_actionable_error(tmp_path):
    """按 real/simulated 过滤会清空训练集 → 必须响亮失败而不是产出 NaN scaler。"""
    p = _make_nc(tmp_path / "u.nc")
    b = load_dataset(_nc_cfg(p))
    with pytest.raises(ValueError) as ei:
        b.subset(np.array([], dtype=int))
    msg = str(ei.value)
    assert "source_label" in msg and "real_fraction" in msg


# ----------------------------------------------------------------------
# D2：max_samples 必须在体积估算之前生效；守卫返回降采样后的 Dataset
# ----------------------------------------------------------------------
def test_d2_max_samples_relieves_volume_guard(tmp_path):
    p = _make_nc(tmp_path / "big.nc", n_time=10, n_lat=5, n_lon=6, n_level=4)
    small_gb = 5e-5  # ≈ 53 KB
    # 有 max_samples → 降采样后体积落在阈值内 → 不再误报
    b = load_dataset(_nc_cfg(p, netcdf={"max_in_memory_gb": small_gb},
                             sampling={"max_samples": 50}))
    assert 0 < b.n_samples <= 50
    # 无 max_samples → 守卫必须拦下（证明 max_samples 真的作用在估算之前）
    with pytest.raises(ValueError, match="max_in_memory_gb"):
        load_dataset(_nc_cfg(p, netcdf={"max_in_memory_gb": small_gb}, sampling={}))


def test_d2_guard_message_recommends_only_effective_measures(tmp_path):
    p = _make_nc(tmp_path / "big2.nc", n_time=8, n_lat=5, n_lon=6, n_level=4)
    with pytest.raises(ValueError) as ei:
        load_dataset(_nc_cfg(p, netcdf={"max_in_memory_gb": 1e-6}, sampling={}))
    msg = str(ei.value)
    assert "stride" in msg and "time_range" in msg and "chunks" in msg
    # v1 的死代码建议（max_samples 不解除守卫）不得再出现
    assert "max_samples" not in msg


def test_d2_max_samples_applied_to_batches_consistently(tmp_path):
    """max_samples 在分批路径同样生效，且与整体加载的样本数一致。"""
    p = _make_nc(tmp_path / "b.nc", n_time=8, n_lat=4, n_lon=5, n_level=3)
    cfg = _nc_cfg(p, sampling={"max_samples": 40})
    whole = load_dataset(cfg)
    parts = list(iter_batches(cfg, batch_samples=1000))
    assert sum(x.n_samples for x in parts) == whole.n_samples
    assert whole.n_samples <= 40


# ----------------------------------------------------------------------
# D3：dask 缺失必须显式告警（不得静默忽略 chunks）
# ----------------------------------------------------------------------
def test_d3_missing_dask_warns_explicitly(tmp_path):
    p = _make_nc(tmp_path / "d.nc")
    cfg = _nc_cfg(p, netcdf={"chunks": {"time": 2}})
    with pytest.warns(RuntimeWarning, match="dask"):
        b = load_dataset(cfg)
    assert b.n_samples > 0  # 回退路径仍可用（不静默失败）

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = inspect_data(cfg)
    assert info["chunks"]["dask_available"] is False
    assert info["chunks"]["dask_enabled"] is False
    assert info["chunks"]["requested_chunks"] == {"time": 2}
    assert "netCDF4" in info["chunks"]["note"]


def test_d3_no_chunks_configured_no_warning(tmp_path):
    p = _make_nc(tmp_path / "d2.nc")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        load_dataset(_nc_cfg(p))
    assert not [w for w in rec if "dask" in str(w.message)]


# ----------------------------------------------------------------------
# D4/D5：分批受 batch_samples 约束 + 跨批 sample_id 唯一 + 溯源字段齐全
# ----------------------------------------------------------------------
def test_d4_batches_respect_batch_samples(tmp_path):
    p = _make_nc(tmp_path / "b.nc", n_time=8, n_lat=5, n_lon=6, n_level=3)
    cfg = _nc_cfg(p)
    parts = list(iter_batches(cfg, batch_samples=100))
    assert len(parts) > 1
    assert all(x.n_samples <= 100 for x in parts)
    whole = load_dataset(cfg)
    assert sum(x.n_samples for x in parts) == whole.n_samples
    np.testing.assert_allclose(np.vstack([x.X for x in parts]), whole.X)


def test_d4_batches_split_inside_step_when_step_too_large(tmp_path):
    """单步样本数 > batch_samples 时必须在步内继续切（气压层/网格场景）。"""
    p = _make_nc(tmp_path / "s.nc", n_time=2, n_lat=30, n_lon=30, n_level=1)
    parts = list(iter_batches(_nc_cfg(p), batch_samples=250))
    assert len(parts) > 2
    assert all(x.n_samples <= 250 for x in parts)
    assert sum(x.n_samples for x in parts) == 2 * 30 * 30


def test_d5_sample_ids_unique_across_batches_and_stable(tmp_path):
    p = _make_nc(tmp_path / "u.nc", n_time=8, n_lat=4, n_lon=5, n_level=3)
    cfg = _nc_cfg(p)
    whole = load_dataset(cfg)
    parts = list(iter_batches(cfg, batch_samples=77))
    ids = [sid for part in parts for sid in part.meta["sample_id"]]
    assert len(ids) == len(set(ids)), "跨批 sample_id 必须唯一"
    assert ids == list(whole.meta["sample_id"]), "sample_id 必须稳定（与整体加载一致）"
    assert all(str(s).startswith(p.stem) for s in ids)


def test_d5_batches_carry_units_source_file_and_origin(tmp_path):
    p = _make_nc(tmp_path / "u2.nc")
    for part in iter_batches(_nc_cfg(p), batch_samples=50):
        assert part.units.get("t") == "K"
        assert part.source_file == str(p)
        assert part.source_origin == "unresolved"
        assert part.meta["source_origin"].unique().tolist() == ["unresolved"]


# ----------------------------------------------------------------------
# D8：subset 必须透传 units / source_file / source_origin
# ----------------------------------------------------------------------
def test_d8_subset_keeps_traceability(tmp_path):
    p = _make_nc(tmp_path / "s.nc")
    b = load_dataset(_nc_cfg(p))
    sub = b.subset([0, 5, 9])
    assert sub.units == b.units
    assert sub.source_file == b.source_file
    assert sub.source_origin == b.source_origin
    assert sub.meta["source_origin"].tolist() == b.meta["source_origin"].iloc[[0, 5, 9]].tolist()
    assert sub.n_samples == 3


# ----------------------------------------------------------------------
# D7：scaler 的 NaN/Inf 策略（默认拒绝 + 明确列名与数量）
# ----------------------------------------------------------------------
def test_d7_scaler_rejects_nan_with_column_names_and_counts():
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0], [6.0, np.inf]])
    Y = np.zeros((4, 1))
    with pytest.raises(ValueError) as ei:
        fit_scalers(X, Y, scale=True, method="standard", feature_names=["t", "q"])
    msg = str(ei.value)
    assert "t" in msg and "q" in msg
    assert "NaN=1" in msg and "Inf=1" in msg


def test_d7_scaler_impute_policy_uses_train_statistics_only():
    X = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, 30.0], [5.0, 40.0]])
    Y = np.array([[0.0], [1.0], [2.0], [3.0]])
    pre = fit_scalers(X, Y, scale=True, method="standard", nan_policy=NAN_POLICY_IMPUTE,
                      feature_names=["a", "b"])
    # 统计量只来自训练集：a 列 NaN 用训练集中位数 3.0 填补 → 均值 (1+3+3+5)/4
    assert pre.x.mean_[0] == pytest.approx((1.0 + 3.0 + 3.0 + 5.0) / 4)
    assert pre.fit_evidence["nonfinite"]["X"]["a"]["nan"] == 1
    z = pre.transform_x(np.array([[np.nan, 10.0]]))
    assert np.isfinite(z).all()


def test_d7_scaler_state_records_nonfinite_report():
    X = np.array([[1.0, 2.0], [np.nan, 3.0]])
    pre = fit_scalers(X, np.zeros((2, 1)), scale=True, method="standard", nan_policy="impute",
                      feature_names=["t", "q"])
    assert pre.x.state_dict()["nonfinite"]["t"]["nan"] == 1


# ----------------------------------------------------------------------
# D6：拟合证据可校验
# ----------------------------------------------------------------------
def test_d6_fit_evidence_recorded_and_verifiable():
    X = np.arange(40, dtype=float).reshape(20, 2)
    Y = np.arange(20, dtype=float).reshape(20, 1)
    pre = fit_scalers(X, Y, scale=True, method="standard", source="train_split_2020",
                      feature_names=["t", "q"], target_names=["clwc"])
    ev = pre.state_dict()["fit_evidence"]
    assert ev["n_samples"] == 20
    assert ev["feature_names"] == ["t", "q"]
    assert ev["target_names"] == ["clwc"]
    assert ev["fitted_at"]
    assert ev["schema_version"] == pre.schema_version
    assert ev["declared_source"] == "train_split_2020"
    assert ev["fitted_on"] == "train"
    assert ev["x_fingerprint"] and ev["y_fingerprint"]
    check = verify_fit_evidence(pre.state_dict())
    assert check["valid"] is True, check["problems"]


def test_d6_evidence_detects_tampered_state():
    X = np.random.default_rng(0).normal(size=(10, 2))
    pre = fit_scalers(X, np.zeros((10, 1)), scale=True, method="standard")
    state = pre.state_dict()
    state["fit_evidence"]["fitted_on"] = "all_data"
    state["fit_evidence"]["schema_version"] = "0.0"
    bad = verify_fit_evidence(state)
    assert bad["valid"] is False
    assert any("train" in p for p in bad["problems"])
    assert any("schema_version" in p for p in bad["problems"])
    # 缺证据也必须判为不可信
    assert verify_fit_evidence({"x": {}})["valid"] is False


# ----------------------------------------------------------------------
# D11：分级缓存（命中 / 失效 / 重建 / 版本校验 / 绝不写 data/raw）
# ----------------------------------------------------------------------
def test_d11_cache_hit_miss_invalidate_rebuild(tmp_path):
    src = tmp_path / "src.nc"
    _make_nc(src)
    cache = DataCache(root=tmp_path / ".cache" / "cloudwater_ml")
    assert cache.cached_metadata(src, "inspect") is None
    cache.store_metadata(src, "inspect", {"dims": {"time": 6}})
    hit = cache.cached_metadata(src, "inspect")
    assert hit == {"dims": {"time": 6}}
    stats = cache.stats()
    assert stats["hits"] >= 1

    # 源文件变化 → 版本变化 → 缓存失效（miss）
    _make_nc(src, n_time=7)
    assert cache.cached_metadata(src, "inspect") is None
    # 重建 → 再次命中
    cache.store_metadata(src, "inspect", {"dims": {"time": 7}})
    assert cache.cached_metadata(src, "inspect") == {"dims": {"time": 7}}


def test_d11_cache_version_mismatch_invalidates_entry(tmp_path):
    src = tmp_path / "v.bin"
    src.write_bytes(b"v1")
    cache = DataCache(root=tmp_path / ".cache")
    key = cache.versioned_key(src, "metadata", "k")
    cache.put("metadata", key, {"v": 1}, version=compute_file_version(src), source=str(src))
    assert cache.get("metadata", key, version=compute_file_version(src)) == {"v": 1}
    src.write_bytes(b"v2-longer")  # 内容与 mtime 都变了
    assert cache.get("metadata", key, version=compute_file_version(src)) is None


def test_d11_cache_refuses_to_write_into_raw(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="raw"):
        DataCache(root=raw_dir / "cache")
    with pytest.raises(ValueError, match="raw"):
        DataCache(root=tmp_path / "data" / "raw")


def test_d11_cache_never_writes_into_data_raw(tmp_path, monkeypatch):
    """真正写一次缓存，确认 data/raw 目录里没有任何新文件。"""
    repo_raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    before = sorted(p.name for p in repo_raw.iterdir()) if repo_raw.exists() else []
    work = tmp_path / "work"
    (work / "data" / "raw").mkdir(parents=True)
    src = work / "d.nc"
    _make_nc(src)
    cache = DataCache(root=work / ".cache" / "cloudwater_ml")
    cache.store_metadata(src, "inspect", {"ok": True})
    cache.store_index(src, "rows", {"ranges": [[0, 10]]})
    cache.store_preprocessed("pre-key", {"mean": 1.0})
    assert (work / "data" / "raw").exists()
    assert sorted(p.name for p in (work / "data" / "raw").iterdir()) == []
    after = sorted(p.name for p in repo_raw.iterdir()) if repo_raw.exists() else []
    assert after == before, "缓存不得写入 data/raw"
    # 缓存根目录在 data/ 之外
    assert "data" not in Path(cache.root).parts


def test_d11_cache_levels_and_clear(tmp_path):
    cache = DataCache(root=tmp_path / ".cache")
    cache.put("raw", make_key("i"), {"rows": 10})
    cache.put("metadata", make_key("m"), {"dims": {}})
    cache.put("preprocessing", make_key("p"), {"scaled": True})
    cache.put("memory", make_key("mem"), np.zeros(4))
    cache.put("gpu_batch", make_key("gpu"), {"batch": 1})
    stats = cache.stats()
    assert stats["per_level"]["raw"] == 1
    assert stats["per_level"]["memory"] == 1
    assert cache.get("memory", make_key("mem")) is not None
    cache.clear("metadata")
    assert cache.get("metadata", make_key("m")) is None
    assert cache.get("raw", make_key("i")) is not None
    cache.clear()
    assert cache.get("raw", make_key("i")) is None
    assert cache.stats()["per_level"]["memory"] == 0


def test_d11_cache_disabled_is_noop(tmp_path):
    cache = DataCache(root=tmp_path / ".cache", enabled=False)
    cache.put("metadata", "k", {"v": 1})
    assert cache.get("metadata", "k") is None
    assert not (tmp_path / ".cache").exists()


# ----------------------------------------------------------------------
# D10 / D12：质量检查深化 + 气压层对齐
# ----------------------------------------------------------------------
def test_d10_time_gap_detected(tmp_path):
    """时间不连续（缺测）必须被报出。"""
    p = tmp_path / "gap.nc"
    times = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 06:00", "2020-01-01 12:00",
                            "2020-01-05 00:00"])
    lat = np.linspace(28, 30, 2)
    lon = np.linspace(78, 80, 2)
    ds = xr.Dataset(
        {"t": (("time", "latitude", "longitude"), np.zeros((4, 2, 2), dtype="float32")),
         "clwc": (("time", "latitude", "longitude"), np.zeros((4, 2, 2), dtype="float32"))},
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    ds["t"].attrs["units"] = "K"
    ds.to_netcdf(p)
    b = load_dataset(_nc_cfg(p, features=["t"], targets=["clwc"]))
    rep = run_quality_checks(b, {"quality": {"checks": ["time"]}})
    t = rep["checks"]["time"]
    assert t["n_gaps"] >= 1
    assert t["expected_interval"] == "0 days 06:00:00"
    assert t["passed"] is False
    assert any("时间" in w for w in rep["warnings"])


def test_d10_latitude_out_of_range_detected(tmp_path):
    b = SampleBundle(
        X=np.zeros((4, 1)), Y=np.zeros((4, 1)), feature_names=["t"], target_names=["clwc"],
        meta=pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(4)],
            "latitude": [28.0, 30.0, 95.0, 31.0],   # 95 越界
            "longitude": [78.0, 80.0, 82.0, 84.0],
            "time": pd.date_range("2020-01-01", periods=4, freq="6h"),
            "source": SOURCE_UNKNOWN,
        }),
    )
    rep = run_quality_checks(b, {"quality": {"checks": ["geo"]}})
    geo = rep["checks"]["geo"]
    assert geo["latitude"]["out_of_range"] == 1
    assert geo["passed"] is False


def test_d10_longitude_zero_360_convention_reported(tmp_path):
    b = SampleBundle(
        X=np.zeros((3, 1)), Y=np.zeros((3, 1)), feature_names=["t"], target_names=["clwc"],
        meta=pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(3)],
            "latitude": [28.0, 29.0, 30.0],
            "longitude": [10.0, 200.0, 350.0],   # 0-360 约定
            "time": pd.date_range("2020-01-01", periods=3, freq="6h"),
            "source": SOURCE_UNKNOWN,
        }),
    )
    rep = run_quality_checks(b, {"quality": {"checks": ["geo"]}})
    assert rep["checks"]["geo"]["longitude"]["convention"] == "0-360"


def test_d10_missing_distribution_by_level_lat_band_season(tmp_path):
    p = _make_nc(tmp_path / "m.nc", n_time=4, n_lat=4, n_lon=4, n_level=3)
    b = load_dataset(_nc_cfg(p))
    rep = run_quality_checks(b, {"quality": {"checks": ["missing"]}})
    dist = rep["checks"]["missing"]["distribution"]
    assert "by_level" in dist and "by_latitude_band" in dist and "by_season" in dist
    assert dist["n_missing_samples"] >= 1
    assert sum(v["missing"] for v in dist["by_level"].values()) == dist["n_missing_samples"]


def test_d10_x_side_extreme_and_physical_bounds(tmp_path):
    X = np.zeros((20, 2))
    X[-1, 0] = 1000.0          # X 侧离群点（|z| > 4 需要足够样本量）
    Y = np.zeros((20, 1))
    Y[0, 0] = -1.0              # clwc 出现物理上不可能的负值
    b = SampleBundle(X=X, Y=Y, feature_names=["t", "q"], target_names=["clwc"],
                     units={"t": "K", "q": "kg kg-1", "clwc": "kg kg-1"})
    rep = run_quality_checks(b, {"quality": {"checks": ["extreme", "range"]}})
    assert "X" in rep["checks"]["extreme"]
    assert rep["checks"]["extreme"]["total_extreme"] >= 1
    assert rep["checks"]["extreme"]["X"]["per_column"]["X.t"]["n_extreme"] >= 1
    rng = rep["checks"]["range"]["per_variable"]["clwc"]
    assert rng["violations"] == 1
    assert rng["range_source"] == "physical_default"


def test_d10_source_labels_check_reports_unresolved(tmp_path):
    p = _make_nc(tmp_path / "s.nc")
    b = load_dataset(_nc_cfg(p))
    rep = run_quality_checks(b, {"quality": {"checks": ["source_labels"]}})
    blk = rep["checks"]["source_labels"]
    assert blk["n_unknown"] == b.n_samples
    assert blk["resolved"] is False
    assert blk["passed"] is False


def test_d12_pressure_level_alias_supported(tmp_path):
    """ERA5 常见坐标名 pressure_level 必须被识别为 level。"""
    p = _make_nc(tmp_path / "plv.nc", level_name="pressure_level", n_level=3)
    b = load_dataset(_nc_cfg(p))
    assert "level" in b.meta.columns
    assert sorted(b.meta["level"].unique().tolist()) == [700.0, 850.0, 1000.0]
    assert b.units.get("level") == "hPa"


def test_d12_level_range_filter_works_with_alias(tmp_path):
    p = _make_nc(tmp_path / "plv2.nc", level_name="pressure_level", n_level=4)
    b = load_dataset(_nc_cfg(p, netcdf={"level_range": [800, 900]}))
    assert sorted(b.meta["level"].unique().tolist()) == [850.0]


def test_d12_mixed_single_level_and_pressure_level(tmp_path):
    """单层变量 + 气压层变量：默认广播；level_alignment=error 时给清晰错误。"""
    rng = np.random.default_rng(3)
    times = pd.date_range("2020-01-01", periods=3, freq="6h")
    lat = np.linspace(28, 30, 3)
    lon = np.linspace(78, 80, 3)
    lev = np.array([1000.0, 850.0])
    ds = xr.Dataset(
        {
            "t": (("time", "level", "latitude", "longitude"),
                  rng.normal(250, 2, (3, 2, 3, 3)).astype("float32")),
            "sst": (("time", "latitude", "longitude"),
                    rng.normal(290, 1, (3, 3, 3)).astype("float32")),   # 单层变量
            "clwc": (("time", "level", "latitude", "longitude"),
                     rng.normal(1e-4, 1e-5, (3, 2, 3, 3)).astype("float32")),
        },
        coords={"time": times, "latitude": lat, "longitude": lon, "level": lev},
    )
    ds["t"].attrs["units"] = "K"
    ds["sst"].attrs["units"] = "K"
    path = tmp_path / "mixed.nc"
    ds.to_netcdf(path)

    cfg = _nc_cfg(path, features=["t", "sst"], targets=["clwc"])
    b = load_dataset(cfg)
    assert b.n_samples == 3 * 2 * 3 * 3          # 单层特征广播到各层
    assert b.X.shape[1] == 2
    # 单层变量的取值在每个气压层上重复一次（广播语义）
    _vals, counts = np.unique(b.X[:, 1], return_counts=True)
    assert set(counts.tolist()) == {2}

    with pytest.raises(ValueError, match="level_alignment"):
        load_dataset(_nc_cfg(path, features=["t", "sst"], targets=["clwc"],
                             netcdf={"level_alignment": "error"}))


def test_d12_incompatible_dims_still_rejected(tmp_path):
    """与气压层无关的维度不一致仍然明确报错（不静默对齐）。"""
    p = tmp_path / "bad.nc"
    ds = xr.Dataset(
        {"t": (("time", "latitude", "longitude"), np.zeros((3, 2, 2), dtype="float32")),
         "q": (("latitude", "longitude"), np.zeros((2, 2), dtype="float32")),
         "clwc": (("time", "latitude", "longitude"), np.zeros((3, 2, 2), dtype="float32"))},
        coords={"time": pd.date_range("2020-01-01", periods=3),
                "latitude": np.linspace(28, 30, 2), "longitude": np.linspace(78, 80, 2)},
    )
    ds.to_netcdf(p)
    with pytest.raises(ValueError, match="维度不一致"):
        load_dataset(_nc_cfg(p, features=["t", "q"], targets=["clwc"]))


# ----------------------------------------------------------------------
# D13：parquet 列投影 + 行组迭代
# ----------------------------------------------------------------------
def _make_parquet(path: Path, n: int = 200) -> Path:
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "t": rng.normal(250, 5, n),
        "q": rng.normal(5e-3, 1e-3, n),
        "clwc": rng.normal(1e-4, 1e-5, n),
        "unused_a": rng.normal(0, 1, n),
        "unused_b": rng.normal(0, 1, n),
        "time": pd.date_range("2020-01-01", periods=n, freq="6h"),
        "latitude": rng.uniform(28, 38, n),
        "longitude": rng.uniform(78, 98, n),
    })
    df.to_parquet(path, index=False)
    return path


def test_d13_parquet_reads_only_requested_columns(tmp_path, monkeypatch):
    p = _make_parquet(tmp_path / "p.parquet")
    captured: list[dict] = []
    original = array_adapter.pd.read_parquet

    def spy(*args, **kwargs):
        captured.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(array_adapter.pd, "read_parquet", spy)
    b = load_dataset({
        "adapter": "parquet", "path": str(p),
        "features": ["t", "q"], "targets": ["clwc"],
        "meta_columns": ["time", "latitude", "longitude"],
    })
    assert b.n_samples == 200
    assert captured, "read_parquet 必须带 columns= 投影"
    cols = list(captured[0]["columns"])
    assert "unused_a" not in cols and "unused_b" not in cols
    assert set(cols) == {"t", "q", "clwc", "time", "latitude", "longitude"}


def test_d13_parquet_row_group_iteration(tmp_path):
    p = _make_parquet(tmp_path / "p2.parquet", n=120)
    parts = list(iter_batches({
        "adapter": "parquet", "path": str(p),
        "features": ["t", "q"], "targets": ["clwc"],
        "meta_columns": ["time", "latitude", "longitude"],
    }, batch_samples=50))
    assert len(parts) >= 2
    assert all(x.n_samples <= 50 for x in parts)
    assert sum(x.n_samples for x in parts) == 120
    ids = [sid for part in parts for sid in part.meta["sample_id"]]
    assert len(ids) == len(set(ids))


def test_d13_parquet_missing_column_error(tmp_path):
    p = _make_parquet(tmp_path / "p3.parquet", n=10)
    with pytest.raises(ValueError, match="缺少配置要求的变量"):
        load_dataset({"adapter": "parquet", "path": str(p), "features": ["nope"],
                      "targets": ["clwc"]})


# ----------------------------------------------------------------------
# D14 / D15：array 适配器来源出处 + 内容派生 sample_id
# ----------------------------------------------------------------------
def test_d15_array_adapter_source_origin_and_content_ids(tmp_path):
    df = pd.DataFrame({
        "t": np.linspace(250, 260, 10),
        "q": np.linspace(1e-3, 2e-3, 10),
        "clwc": np.linspace(1e-4, 2e-4, 10),
        "time": pd.date_range("2020-01-01", periods=10, freq="6h"),
    })
    p = tmp_path / "a.csv"
    df.to_csv(p, index=False)
    cfg = {"adapter": "csv", "path": str(p), "features": ["t", "q"], "targets": ["clwc"],
           "meta_columns": ["time"]}
    b = load_dataset(cfg)
    assert b.source_counts() == {SOURCE_UNKNOWN: 10}
    assert b.source_origin == "unresolved"
    assert b.meta["source_origin"].unique().tolist() == ["unresolved"]
    # 内容派生：不是位置计数（首行 id 不是 "0" 之类）
    assert not any(str(s).isdigit() for s in b.meta["sample_id"])
    # 抽样后仍然稳定（同样内容 → 同样 id）
    b2 = load_dataset({**cfg, "sampling": {"stride": 2}})
    assert list(b2.meta["sample_id"]) == [s for i, s in enumerate(b.meta["sample_id"]) if i % 2 == 0]


def test_d15_array_adapter_config_source_label(tmp_path):
    df = pd.DataFrame({"t": [1.0, 2.0], "clwc": [0.1, 0.2]})
    p = tmp_path / "b.csv"
    df.to_csv(p, index=False)
    b = load_dataset({"adapter": "csv", "path": str(p), "features": ["t"], "targets": ["clwc"],
                      "source_label": "real"})
    assert b.source_counts() == {"real": 2}
    assert b.source_origin == "config"


# ----------------------------------------------------------------------
# 回归：scaler 状态往返 + 旧行为不破
# ----------------------------------------------------------------------
def test_scaler_state_roundtrip_with_evidence():
    X = np.random.default_rng(1).normal(size=(30, 2))
    Y = np.random.default_rng(2).normal(size=(30, 1))
    pre = fit_scalers(X, Y, scale=True, method="standard", feature_names=["a", "b"])
    s = StandardScaler.from_state_dict(pre.x.state_dict())
    np.testing.assert_allclose(s.transform(X), pre.transform_x(X))
    assert verify_fit_evidence(pre.state_dict())["valid"] is True
