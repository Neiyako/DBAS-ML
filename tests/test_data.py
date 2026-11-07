"""数据层测试：加载、维度与类型、划分防泄漏、归一化、质量检查、NetCDF 按需读取。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cloudwater_ml.data import (
    apply_cleaning,
    ensure_synthetic_dataset,
    fit_scalers,
    format_quality_report,
    inspect_data,
    iter_batches,
    load_dataset,
    make_synthetic_bundle,
    run_quality_checks,
    save_synthetic_npz,
    split_indices,
)
from cloudwater_ml.data.preprocessing import StandardScaler
from cloudwater_ml.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["t", "q", "rh", "u", "v", "w"]
TARGETS = ["clwc", "ciwc"]


@pytest.fixture(scope="module")
def bundle():
    return make_synthetic_bundle(n_samples=500, seed=7)


# ---------------------------------------------------------------- 合成数据
def test_synthetic_shapes_and_dtypes(bundle):
    assert bundle.X.shape == (500, 6)
    assert bundle.Y.shape == (500, 2)
    assert bundle.X.dtype == np.float64 and bundle.Y.dtype == np.float64
    assert bundle.feature_names == FEATURES
    assert bundle.target_names == TARGETS


def test_synthetic_reproducible():
    a = make_synthetic_bundle(n_samples=200, seed=5)
    b = make_synthetic_bundle(n_samples=200, seed=5)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.Y, b.Y)
    pd.testing.assert_frame_equal(a.meta, b.meta)


def test_synthetic_source_always_simulated(bundle):
    """铁律：合成数据绝不标成真实观测。"""
    assert bundle.source_counts() == {"simulated": 500}
    assert set(bundle.meta["dataset"]) == {"synthetic_test_data"}


def test_synthetic_targets_nonnegative(bundle):
    assert (bundle.Y >= 0).all()


def test_meta_traceability_columns(bundle):
    for col in ("sample_id", "time", "latitude", "longitude", "source", "dataset", "year"):
        assert col in bundle.meta.columns
    assert len(bundle.meta) == bundle.n_samples  # 行对齐
    assert pd.api.types.is_datetime64_any_dtype(bundle.meta["time"])


def test_bundle_subset_keeps_alignment(bundle):
    idx = [3, 10, 42, 42]
    sub = bundle.subset(idx)
    np.testing.assert_array_equal(sub.X, bundle.X[idx])
    np.testing.assert_array_equal(sub.Y, bundle.Y[idx])
    assert list(sub.meta["sample_id"]) == list(bundle.meta.iloc[idx]["sample_id"])


def test_spatial_shape_guard():
    from cloudwater_ml.data.schema import spatial_shape_guard

    spatial_shape_guard((2, 3, 4, 4), "spatial")
    with pytest.raises(ValueError):
        spatial_shape_guard((2, 3, 4), "spatial")
    with pytest.raises(ValueError):
        spatial_shape_guard((2, 3, 4, 4), "temporal")


# ---------------------------------------------------------------- 加载与 IO
def test_npz_roundtrip(tmp_path, bundle):
    path = tmp_path / "b.npz"
    save_synthetic_npz(bundle, path)
    loaded = load_dataset({
        "adapter": "npz", "path": str(path),
        "features": FEATURES, "targets": TARGETS,
        "meta_columns": ["sample_id", "time", "latitude", "longitude", "source", "dataset"],
    })
    np.testing.assert_allclose(loaded.X, bundle.X)
    np.testing.assert_allclose(loaded.Y, bundle.Y)
    assert loaded.source_counts() == {"simulated": 500}


def test_csv_load_and_missing_variable_error(tmp_path, bundle):
    df = pd.DataFrame(bundle.X, columns=FEATURES)
    df[TARGETS] = bundle.Y
    for c in ("time", "latitude", "longitude", "source", "dataset"):
        df[c] = bundle.meta[c].to_numpy()
    path = tmp_path / "b.csv"
    df.to_csv(path, index=False)
    base = {"adapter": "csv", "path": str(path), "features": FEATURES, "targets": TARGETS,
            "meta_columns": ["time", "latitude", "longitude", "source", "dataset"]}
    loaded = load_dataset(base)
    assert loaded.n_samples == 500
    with pytest.raises(ValueError, match="缺少配置要求的变量"):
        load_dataset({**base, "features": ["t", "not_a_var"]})


def test_inspect_json_serializable(bundle, tmp_path):
    path = tmp_path / "i.npz"
    save_synthetic_npz(bundle, path)
    info = inspect_data({"adapter": "npz", "path": str(path),
                         "features": FEATURES, "targets": TARGETS})
    json.dumps(info, ensure_ascii=False, default=str)
    assert info["exists"] and info["size_bytes"] > 0


def test_auto_bootstrap_synthetic(tmp_path):
    """data/synthetic 下缺失文件时自举生成（可复现、带 simulated 标签）。"""
    p = tmp_path / "synthetic" / "x.npz"
    b = load_dataset({"adapter": "npz", "path": str(p), "features": ["t"], "targets": ["clwc"],
                      "meta_columns": ["time", "source"], "synthetic": {"n_samples": 50, "seed": 3}})
    assert p.exists()
    assert b.n_samples == 50 and b.source_counts() == {"simulated": 50}


def test_ensure_synthetic_dataset_idempotent(tmp_path):
    p = tmp_path / "synthetic" / "y.npz"
    ensure_synthetic_dataset(p, n_samples=30, seed=1)
    mtime = p.stat().st_mtime_ns
    ensure_synthetic_dataset(p, n_samples=30, seed=1)  # 已存在不重建
    assert p.stat().st_mtime_ns == mtime


# ---------------------------------------------------------------- 划分与防泄漏
def _time_cfg(train_end, val_end):
    return {"strategy": "time", "ratios": [0.7, 0.15, 0.15],
            "params": {"train_end": train_end, "val_end": val_end}, "seed": 42}


def test_split_time_no_future_leakage(bundle):
    sp = split_indices(bundle, _time_cfg("2017-12-31T23:59:59", "2018-12-31T23:59:59"))
    t = pd.to_datetime(bundle.meta["time"])
    assert t.iloc[sp.train].max() <= pd.Timestamp("2017-12-31T23:59:59")
    assert t.iloc[sp.val].max() <= pd.Timestamp("2018-12-31T23:59:59")
    assert t.iloc[sp.val].min() > pd.Timestamp("2017-12-31T23:59:59")
    assert t.iloc[sp.test].min() > pd.Timestamp("2018-12-31T23:59:59")
    sp.check_disjoint()  # 两两互斥（内部还会再校验一次）


def test_split_year_whole_years(bundle):
    sp = split_indices(bundle, {"strategy": "year", "ratios": [0.6, 0.2, 0.2], "params": {}, "seed": 1})
    years = bundle.meta["year"].to_numpy()
    owner: dict[int, str] = {}
    for name in ("train", "val", "test"):
        for y in years[getattr(sp, name)]:
            assert owner.setdefault(int(y), name) == name, f"年份 {y} 跨集泄漏"
    sp.check_disjoint()


def test_split_random_deterministic_and_disjoint(bundle):
    a = split_indices(bundle, {"strategy": "random", "ratios": [0.7, 0.15, 0.15], "seed": 42})
    b = split_indices(bundle, {"strategy": "random", "ratios": [0.7, 0.15, 0.15], "seed": 42})
    np.testing.assert_array_equal(a.train, b.train)
    a.check_disjoint()
    assert a.train.size + a.val.size + a.test.size == bundle.n_samples


def test_split_region_boxes(bundle):
    boxes = {"train": [78, 90, 28, 38], "val": [90, 97, 28, 38], "test": [97, 103, 28, 38]}
    sp = split_indices(bundle, {"strategy": "region", "params": {"boxes": boxes}})
    sp.check_disjoint()
    lon = bundle.meta["longitude"].to_numpy()
    assert (lon[sp.train] <= 90).all()
    assert sp.as_dict()["n_train"] + sp.as_dict()["n_val"] + sp.as_dict()["n_test"] + sp.dropped == bundle.n_samples


def test_split_source_requires_params(bundle):
    with pytest.raises(ValueError, match="train_sources"):
        split_indices(bundle, {"strategy": "source", "params": {}})


def test_split_unknown_strategy(bundle):
    with pytest.raises(ValueError, match="未知划分策略"):
        split_indices(bundle, {"strategy": "kmeans"})


# ---------------------------------------------------------------- 归一化
def test_scaler_fit_on_train_only(bundle):
    sp = split_indices(bundle, _time_cfg("2017-12-31T23:59:59", "2018-12-31T23:59:59"))
    Xtr = bundle.X[sp.train]
    pre = fit_scalers(Xtr, bundle.Y[sp.train], scale=True, method="standard")
    np.testing.assert_allclose(pre.x.mean_, Xtr.mean(axis=0), rtol=1e-12)
    assert pre.fitted_on == "train"
    # train 变换后近零均值单位方差；val/test 不重拟合
    Z = pre.transform_x(Xtr)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-10)
    Zv = pre.transform_x(bundle.X[sp.val])
    assert not np.allclose(Zv.mean(axis=0), 0, atol=1e-10)  # val 用 train 参数
    back = pre.inverse_y(pre.transform_y(bundle.Y[sp.test]))
    np.testing.assert_allclose(back, bundle.Y[sp.test], rtol=1e-10)


def test_scaler_constant_column_and_state_roundtrip():
    X = np.hstack([np.ones((20, 1)), np.arange(20).reshape(-1, 1).astype(float)])
    s = StandardScaler().fit(X)
    Z = s.transform(X)
    assert np.isfinite(Z).all() and Z[:, 0].std() == 0
    s2 = StandardScaler.from_state_dict(s.state_dict())
    np.testing.assert_allclose(s2.transform(X), Z)


def test_scaler_refuses_unfitted():
    with pytest.raises(RuntimeError, match="未 fit"):
        StandardScaler().transform(np.ones((3, 2)))


# ---------------------------------------------------------------- 质量检查
def test_quality_report_structure_and_readonly(bundle):
    data_cfg = {"quality": {"report_only": True,
                            "checks": ["missing", "nan_inf", "extreme", "range", "dimensions", "units"],
                            "ranges": {"clwc": [0.0, 0.01]}}}
    before = bundle.X.copy()
    rep = run_quality_checks(bundle, data_cfg)
    np.testing.assert_array_equal(bundle.X, before)  # 报告不改数据
    assert set(rep["checks"]) == {"missing", "nan_inf", "extreme", "range", "dimensions", "units"}
    assert rep["summary"]["report_only"] is True
    assert rep["summary"]["source_counts"] == {"simulated": 500}
    md = format_quality_report(rep)
    assert "数据质量报告" in md and "样本数" in md


def test_quality_detects_nan_and_range_violations(bundle):
    X = bundle.X.copy()
    X[0, 0] = np.nan
    from cloudwater_ml.data import SampleBundle

    dirty = SampleBundle(X=X, Y=bundle.Y.copy(), feature_names=FEATURES,
                         target_names=TARGETS, meta=bundle.meta.copy())
    rep = run_quality_checks(dirty, {"quality": {"report_only": True, "checks": ["missing", "nan_inf", "range"],
                                                 "ranges": {"t": [100.0, 300.0]}}})
    assert rep["checks"]["nan_inf"]["x_nan"] == 1
    assert rep["checks"]["nan_inf"]["passed"] is False
    assert rep["summary"]["passed"] is False


def test_cleaning_refused_by_default(bundle):
    with pytest.raises(RuntimeError, match="report_only"):
        apply_cleaning(bundle, {"quality": {"report_only": True}})


def test_cleaning_explicit_dropna(bundle):
    X = bundle.X.copy()
    X[5, 2] = np.nan
    from cloudwater_ml.data import SampleBundle

    dirty = SampleBundle(X=X, Y=bundle.Y.copy(), feature_names=FEATURES,
                         target_names=TARGETS, meta=bundle.meta.copy())
    cleaned, actions = apply_cleaning(dirty, {"quality": {"report_only": False,
                                                          "cleaning": {"dropna": True}}})
    assert cleaned.n_samples == 499
    assert actions and actions[0]["action"] == "dropna" and actions[0]["dropped"] == 1
    assert np.isfinite(cleaned.X).all()


# ---------------------------------------------------------------- NetCDF
def _tiny_nc(path: Path) -> None:
    b = make_synthetic_bundle(n_samples=300, seed=3)
    ds = xr.Dataset(
        {**{v: (("time", "latitude", "longitude"), b.X[:, i].reshape(10, 6, 5))
            for i, v in enumerate(FEATURES)},
         **{v: (("time", "latitude", "longitude"), b.Y[:, i].reshape(10, 6, 5))
            for i, v in enumerate(TARGETS)}},
        coords={"time": pd.date_range("2015-01-01", periods=10),
                "latitude": np.linspace(28, 38, 6),
                "longitude": np.linspace(78, 103, 5)},
    )
    ds.attrs["atmosgen:contract"] = json.dumps({"source": {"kind": "simulated"}})
    ds["t"].attrs["units"] = "K"
    ds.to_netcdf(path)


def _nc_cfg(path: Path) -> dict:
    return {"adapter": "netcdf", "path": str(path), "features": FEATURES, "targets": TARGETS,
            "netcdf": {"time_range": ["2015-01-02", "2015-01-09"],
                       "latitude_range": [30, 37], "longitude_range": [80, 100], "chunks": {}},
            "sampling": {"stride": 1, "max_samples": None}}


def test_netcdf_subset_and_contract_label(tmp_path):
    p = tmp_path / "tiny.nc"
    _tiny_nc(p)
    info = inspect_data(_nc_cfg(p))
    assert info["source_kind"] == "simulated"
    assert info["estimated_samples"] == 96  # 8 时间 × 4 纬度 × 3 经度
    assert info["data_vars"]["t"]["units"] == "K"

    b = load_dataset(_nc_cfg(p))
    assert b.n_samples == 96
    assert b.meta["source"].unique().tolist() == ["simulated"]
    assert b.meta["source_origin"].unique().tolist() == ["dataset_contract"]  # 契约来源
    assert b.units.get("t") == "K"
    assert "year" in b.meta.columns
    t = pd.to_datetime(b.meta["time"])
    assert t.min() >= pd.Timestamp("2015-01-02") and t.max() <= pd.Timestamp("2015-01-09")


def test_netcdf_iter_batches_equals_load(tmp_path):
    p = tmp_path / "tiny.nc"
    _tiny_nc(p)
    whole = load_dataset(_nc_cfg(p))
    parts = list(iter_batches(_nc_cfg(p), batch_samples=40))
    assert all(x.n_samples <= 40 for x in parts)
    cat_X = np.vstack([x.X for x in parts])
    np.testing.assert_allclose(cat_X, whole.X)


def test_netcdf_memory_guard(tmp_path):
    p = tmp_path / "tiny.nc"
    _tiny_nc(p)
    cfg = _nc_cfg(p)
    with pytest.raises(ValueError, match="max_in_memory_gb"):
        load_dataset({**cfg, "netcdf": {**cfg["netcdf"], "max_in_memory_gb": 1e-6}})


def test_netcdf_missing_variable_error(tmp_path):
    p = tmp_path / "tiny.nc"
    _tiny_nc(p)
    cfg = _nc_cfg(p)
    with pytest.raises(ValueError, match="缺少配置要求的变量"):
        load_dataset({**cfg, "features": ["t", "no_such_var"]})


def test_netcdf_disjoint_variables_dims_rejected(tmp_path):
    """v1 限制：变量维度集必须一致（诚实报错，不静默对齐）。"""
    p = tmp_path / "mixed.nc"
    b = make_synthetic_bundle(n_samples=60, seed=4)
    ds = xr.Dataset(
        {"t": (("time", "latitude", "longitude"), b.X[:, 0].reshape(4, 3, 5)),
         "q": (("latitude", "longitude"), b.X[:15, 1].reshape(3, 5)),
         "clwc": (("time", "latitude", "longitude"), b.Y[:60, 0].reshape(4, 3, 5))},
        coords={"time": pd.date_range("2015-01-01", periods=4),
                "latitude": np.linspace(28, 38, 3), "longitude": np.linspace(78, 103, 5)},
    )
    ds.to_netcdf(p)
    cfg = {"adapter": "netcdf", "path": str(p), "features": ["t", "q"], "targets": ["clwc"],
           "netcdf": {}, "sampling": {}}
    with pytest.raises(ValueError, match="维度不一致"):
        load_dataset(cfg)


# ---------------------------------------------------------------- 配置
def test_config_extends_resolution():
    cfg = load_config(ROOT / "configs/experiments/mlp_baseline.yaml")
    assert cfg["model"]["name"] == "mlp"          # 来自实验文件
    assert cfg["data"]["adapter"] == "npz"          # 来自 ../data.yaml
    assert cfg["training"]["optimizer"] == "adamw"  # 来自 ../training.yaml
    assert cfg["split"]["strategy"] == "time"       # 实验覆盖
    assert Path(cfg["_config_path"]).name == "mlp_baseline.yaml"


def test_config_cycle_detection(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text("extends: [b.yaml]\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: [a.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="环"):
        load_config(a)
