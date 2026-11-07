"""流式训练测试：真实 NetCDF 按批读取 → Trainer.fit_stream → 损失下降。

这里**不 mock**：用 xarray 真实写一个 NetCDF 文件，走真实的
`iter_batches` → `NetCDFStreamDataset` → `fit_stream` 全链路。
"""
from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from cloudwater_ml.data.stream import NetCDFStreamDataset, shard_time_range
from cloudwater_ml.models.mlp import MLP
from cloudwater_ml.utils.exceptions import ConfigurationError


def _write_nc(path, n_time=48, n_lat=6, n_lon=5, seed=0):
    """写一个真实的小型 NetCDF（time/lat/lon + 3 特征 + 2 目标，clwc/ciwc 非负）。"""
    rng = np.random.default_rng(seed)
    times = np.array(
        [np.datetime64("2015-01-01") + np.timedelta64(i, "h") for i in range(n_time)]
    )
    lat = np.linspace(30.0, 35.0, n_lat)
    lon = np.linspace(100.0, 104.0, n_lon)
    t = rng.normal(280, 5, (n_time, n_lat, n_lon))
    q = np.abs(rng.normal(0.01, 0.002, (n_time, n_lat, n_lon)))
    rh = np.clip(rng.normal(60, 10, (n_time, n_lat, n_lon)), 0, 100)
    clwc = np.clip(0.5 * q + 0.001 * (t - 280) + rng.normal(0, 1e-4, t.shape), 0, None)
    ciwc = np.clip(0.2 * q + 0.002 * (rh - 60) + rng.normal(0, 1e-4, t.shape), 0, None)
    ds = xr.Dataset(
        {
            "t": (("time", "lat", "lon"), t),
            "q": (("time", "lat", "lon"), q),
            "rh": (("time", "lat", "lon"), rh),
            "clwc": (("time", "lat", "lon"), clwc),
            "ciwc": (("time", "lat", "lon"), ciwc),
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    ds["t"].attrs["units"] = "K"
    ds["q"].attrs["units"] = "kg kg-1"
    ds["clwc"].attrs["units"] = "kg kg-1"
    ds["ciwc"].attrs["units"] = "kg kg-1"
    ds.to_netcdf(path)
    return path


def _cfg(path):
    return {
        "data": {
            "adapter": "netcdf",
            "path": str(path),
            "features": ["t", "q", "rh"],
            "targets": ["clwc", "ciwc"],
            "netcdf": {"time_range": ["2015-01-01", "2015-01-02"]},
        }
    }


# ----------------------------------------------------------------------
def test_netcdf_stream_dataset_yields_batches(tmp_path):
    nc = _write_nc(tmp_path / "toy.nc")
    ds = NetCDFStreamDataset(_cfg(nc), batch_samples=50)
    batches = list(ds)
    assert batches, "流式数据源没有产出任何批次"
    total = sum(b[0].shape[0] for b in batches)
    # time_range 是闭区间：2015-01-01..2015-01-02 覆盖 25 个逐小时时次
    assert total == 25 * 6 * 5
    # 全部样本都应被读到（不丢不重）
    assert total > 0 and len(batches) >= (total // 50)
    x, y = batches[0]
    assert x.shape[1] == 3 and y.shape[1] == 2
    assert x.dtype.is_floating_point and y.dtype.is_floating_point
    # 每批样本数必须受 batch_samples 约束
    assert all(b[0].shape[0] <= 50 for b in batches)


def test_stream_dataset_requires_features_and_targets():
    with pytest.raises(ConfigurationError):
        NetCDFStreamDataset({"data": {"adapter": "netcdf", "path": "x.nc"}})


def test_shard_time_range_splits_evenly():
    segs = shard_time_range(["2015-01-01", "2015-01-11"], 4)
    assert len(segs) == 4
    assert segs[0][0].startswith("2015-01-01")
    assert segs[-1][1].startswith("2015-01-11")
    # 相邻段首尾相接
    for a, b in zip(segs, segs[1:]):
        assert a[1] == b[0]


def test_shard_time_range_rejects_bad_input():
    with pytest.raises(ConfigurationError):
        shard_time_range(["not-a-date", "2015-01-01"], 2)
    with pytest.raises(ConfigurationError):
        shard_time_range(["2015-02-01", "2015-01-01"], 2)


# ----------------------------------------------------------------------
def test_streaming_training_decreases_loss(tmp_path):
    """真实流式训练：NetCDF → 分批 → MLP.fit_stream，损失必须下降。"""
    nc = _write_nc(tmp_path / "toy.nc")
    cfg = _cfg(nc)
    ds = NetCDFStreamDataset(cfg, batch_samples=100)

    model = MLP(hidden_layers=[32, 16], activation="relu", dropout=0.0)
    cfg["training"] = {"epochs": 12, "batch_size": 32, "learning_rate": 1e-2, "seed": 7}
    result = model.fit_stream(ds, n_features=3, n_targets=2, config=cfg)

    assert result.history, "没有训练历史"
    first = result.history[0]["train_loss"]
    last = result.history[-1]["train_loss"]
    assert last < first, f"流式训练损失未下降: {first} -> {last}"
    assert result.extra["runtime_plan"]["device"]
    assert result.extra["n_batches_last_epoch"] > 0


def test_streaming_prediction_shape(tmp_path):
    nc = _write_nc(tmp_path / "toy.nc")
    cfg = _cfg(nc)
    cfg["training"] = {"epochs": 2, "batch_size": 64, "learning_rate": 1e-2, "seed": 3}
    ds = NetCDFStreamDataset(cfg, batch_samples=100)
    model = MLP(hidden_layers=[8], activation="relu", dropout=0.0)
    model.fit_stream(ds, n_features=3, n_targets=2, config=cfg)
    pred = model.predict(np.random.rand(7, 3).astype(np.float32))
    assert pred.shape == (7, 2)


def test_streaming_and_in_memory_use_same_loop(tmp_path):
    """流式与内存模式共用 _fit_core：都应产出 runtime_plan / 检查点字段。"""
    nc = _write_nc(tmp_path / "toy.nc")
    cfg = _cfg(nc)
    cfg["training"] = {"epochs": 1, "batch_size": 64, "learning_rate": 1e-2, "seed": 1}
    ds = NetCDFStreamDataset(cfg, batch_samples=100)
    model = MLP(hidden_layers=[8], dropout=0.0)
    stream_res = model.fit_stream(ds, n_features=3, n_targets=2, config=cfg, out_dir=tmp_path / "ck")

    X = np.random.rand(50, 3).astype(np.float32)
    Y = np.random.rand(50, 2).astype(np.float32)
    mem_res = MLP(hidden_layers=[8], dropout=0.0).fit(X, Y, config=cfg)

    for key in ("runtime_plan", "memory", "profiling", "global_step"):
        assert key in stream_res.extra, f"流式结果缺少 {key}"
        assert key in mem_res.extra, f"内存结果缺少 {key}"
    assert (tmp_path / "ck" / "last.pt").exists()
