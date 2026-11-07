"""指标测试：已知值核算、R² 常数保护、Bias 符号、分组泛化、噪声鲁棒性、区间指标。"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from cloudwater_ml.evaluation import (
    METRIC_DEFINITIONS,
    evaluate_predictions,
    interval_metrics,
    noise_robustness,
    regression_metrics,
    subgroup_metrics,
)

TARGETS = ["clwc", "ciwc"]


def _rand(rng, *shape):
    """numpy 2.x：Generator.random 只收 size=；用位置形状包装。"""
    return rng.random(shape)


def test_known_values_manual():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 4.0], [5.0, 4.0]])
    m = regression_metrics(y_true, y_pred, TARGETS)
    # 目标 clwc: 误差 [0, +2] → mae=1, mse=2, rmse=√2, bias=+1
    t0 = m["per_target"]["clwc"]
    assert t0["mae"] == pytest.approx(1.0)
    assert t0["mse"] == pytest.approx(2.0)
    assert t0["rmse"] == pytest.approx(math.sqrt(2.0))
    assert t0["bias"] == pytest.approx(1.0)          # bias = mean(pred-true)
    # R² 手工核算: SS_tot=2, SS_res=4 → 1-4/2 = -1
    assert t0["r2"] == pytest.approx(-1.0)
    # 目标 ciwc: 误差 [+2, 0] → 同量级
    assert m["per_target"]["ciwc"]["mae"] == pytest.approx(1.0)
    # overall 为全部元素展平 micro 汇总（文档口径）
    assert set(m["overall"]) >= {"mae", "mse", "rmse", "r2", "bias"}


def test_perfect_prediction():
    y = np.linspace(0, 1, 50).reshape(-1, 1)
    m = regression_metrics(y, y.copy(), ["x"])
    assert m["overall"]["mae"] == pytest.approx(0.0)
    assert m["overall"]["bias"] == pytest.approx(0.0)
    assert m["overall"]["r2"] == pytest.approx(1.0)


def test_bias_sign_convention():
    """pred > true → bias > 0（高估为正）。"""
    y = np.full((10, 1), 2.0)
    pred = y + 0.5
    m = regression_metrics(y, pred, ["x"])
    assert m["overall"]["bias"] == pytest.approx(0.5)


def test_r2_constant_target_guard():
    """目标近似常数时 R² 置 None 并带说明，不输出误导数值。"""
    y = np.full((30, 1), 3.0)
    pred = y + np.random.default_rng(0).normal(0, 0.01, (30, 1))
    m = regression_metrics(y, pred, ["const"])
    assert m["overall"]["r2"] is None
    assert m["overall"].get("r2_note")
    json.dumps(m, ensure_ascii=False)  # JSON 可序列化（None 合法）


def test_metric_definitions_documented():
    for name in ("mae", "mse", "rmse", "r2", "bias"):
        assert name in METRIC_DEFINITIONS
        entry = METRIC_DEFINITIONS[name]
        for key in ("formula", "units", "applicability"):
            assert key in entry and entry[key], f"{name} 缺 {key}"


def test_json_serializable_output():
    rng = np.random.default_rng(1)
    y = _rand(rng, 40, 2) * 1e-5
    m = regression_metrics(y, y + 1e-7, TARGETS)
    json.dumps(m, ensure_ascii=False)


# ---------------------------------------------------------------- 分组泛化
def _meta(n=240):
    times = pd.date_range("2015-11-01", periods=n, freq="20h")  # 跨年+跨季
    return pd.DataFrame({
        "time": times,
        "year": times.year.to_numpy(),
        "latitude": np.linspace(28, 38, n),
        "longitude": np.linspace(78, 103, n),
    })


def test_subgroup_year_and_season():
    rng = np.random.default_rng(2)
    n = 240
    y = _rand(rng, n, 2) * 1e-5
    g = subgroup_metrics(y, y + 1e-6, _meta(n), TARGETS, subgroups=("year", "season"))
    assert set(g["year"]) == {"2015", "2016"}
    assert set(g["season"]) <= {"DJF", "MAM", "JJA", "SON"} and len(g["season"]) >= 2
    for block in g["year"].values():
        assert "rmse" in block or "overall" in block


def test_subgroup_region_skipped_without_boxes():
    rng = np.random.default_rng(3)
    n = 60
    y = _rand(rng, n, 1) * 1e-5
    g = subgroup_metrics(y, y, _meta(n), ["clwc"], subgroups=("region",))
    # 未配置区域盒 → 显式 skipped，不编造分区
    assert "region" in g
    assert "skipped" in str(g["region"]).lower() or g["region"] == {} or "skipped" in g["region"]


def test_subgroup_region_with_boxes():
    rng = np.random.default_rng(4)
    n = 120
    y = _rand(rng, n, 1) * 1e-5
    boxes = {"西部": [78, 90, 28, 38], "东部": [90, 103, 28, 38]}
    g = subgroup_metrics(y, y, _meta(n), ["clwc"],
                         subgroups=("region",), region_boxes=boxes)
    assert set(g["region"]) == {"西部", "东部"}


# ---------------------------------------------------------------- 鲁棒性
def test_noise_robustness_levels_and_reproducibility():
    rng = np.random.default_rng(5)
    X = _rand(rng, 200, 4) * np.array([10, 1e-3, 20, 5])
    Y = _rand(rng, 200, 2) * 1e-5

    def model(X_):  # 恒等映射型假模型：指标随噪声单调不劣化也可，只要结构正确
        return np.hstack([X_[:, :1], X_[:, 2:3]])[:, :2] * 1e-7

    r1 = noise_robustness(model, X, Y, TARGETS, [0.0, 0.1], seed=42)
    r2 = noise_robustness(model, X, Y, TARGETS, [0.0, 0.1], seed=42)
    assert set(r1) == {"0.0", "0.1"}
    assert r1["0.0"]["overall"]["rmse"] == r2["0.0"]["overall"]["rmse"]
    assert r1["0.1"]["overall"]["rmse"] == r2["0.1"]["overall"]["rmse"]
    # 有噪声应不优于无噪声（本构造下噪声直接进入特征）
    assert r1["0.1"]["overall"]["rmse"] >= r1["0.0"]["overall"]["rmse"]


# ---------------------------------------------------------------- 不确定性（预留）
def test_interval_metrics_math():
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    lo = y - 0.5
    hi = y + 1.5
    m = interval_metrics(y, lo, hi, ["x"])
    assert m["overall"]["coverage"] == pytest.approx(1.0)
    assert m["overall"]["interval_width"] == pytest.approx(2.0)


def test_interval_metrics_coverage_partial():
    y = np.array([[0.0], [0.0], [0.0], [0.0]])
    lo = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    hi = np.array([[1.0], [2.0], [1.0], [2.0]])
    m = interval_metrics(y, lo, hi, ["x"])  # 覆盖 2/4
    assert m["overall"]["coverage"] == pytest.approx(0.5)


def test_predict_interval_is_implemented_via_conformal():
    """v2.0：区间生成已实现（分割共形），不再是无条件 NotImplementedError。

    v1.0 的 `predict_interval` 占位接口已被 `estimate_uncertainty` 取代；
    这里守住「接口可用 + 缺前置条件时明确报错」的新契约。
    """
    from cloudwater_ml.evaluation.uncertainty import estimate_uncertainty

    # 缺少校准集 → 明确报错（而不是悄悄给一个没有覆盖率保证的区间）
    with pytest.raises(ValueError):
        estimate_uncertainty(np.zeros((3, 1)), method="conformal", pred_test=np.zeros((3, 1)))


def test_crps_is_implemented():
    """v2.0：CRPS 已实现（高斯预测分布的闭式解）。"""
    from cloudwater_ml.evaluation.uncertainty import crps

    y = np.array([[0.0], [1.0]])
    sigma = np.full_like(y, 0.1)
    good = float(np.mean(crps(y, y, sigma)))
    bad = float(np.mean(crps(y, y + 1.0, sigma)))
    assert good < bad


# ---------------------------------------------------------------- 编排入口
def test_evaluate_predictions_with_config():
    rng = np.random.default_rng(6)
    n = 100
    y = _rand(rng, n, 2) * 1e-5
    pred = y + rng.normal(0, 1e-6, (n, 2))
    meta = _meta(n).iloc[:n].reset_index(drop=True)
    cfg = {"evaluation": {"metrics": ["mae", "rmse"], "subgroups": ["year"]}}
    out = evaluate_predictions(y, pred, meta, TARGETS, cfg)
    assert set(out["metrics"]["overall"].keys()) >= {"mae", "rmse"}
    assert "r2" not in out["metrics"]["overall"]  # 按配置过滤
    assert "year" in out["subgroups"]
