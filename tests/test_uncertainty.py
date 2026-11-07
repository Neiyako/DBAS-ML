"""不确定性估计测试：共形区间 / CRPS / 可靠性报告。

要点：区间必须有**真实来源**，不能拿普通残差冒充置信区间。
"""
from __future__ import annotations

import numpy as np
import pytest

from cloudwater_ml.evaluation.uncertainty import (
    conformal_interval,
    crps,
    ensemble_samples,
    estimate_uncertainty,
    interval_metrics,
    mc_dropout_samples,
    normal_interval,
    reliability_report,
)


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=(n, 2))
    pred = y + rng.normal(scale=0.1, size=y.shape)
    return y, pred


# ----------------------------------------------------------------------
def test_conformal_interval_uses_calibration_residuals():
    y_cal, pred_cal = _data(300, seed=1)
    pred_test = np.zeros((10, 2))
    ci = conformal_interval(y_cal, pred_cal, pred_test, alpha=0.1)
    assert ci["method"] == "conformal"
    assert ci["n_calib"] == 300
    assert np.all(ci["hi"] >= ci["lo"])
    # 区间宽度应与校准残差的 90% 分位数一致
    q = np.quantile(np.abs(y_cal - pred_cal), 0.9)
    assert np.allclose(ci["lo"], -q)


def test_conformal_interval_coverage_is_reasonable():
    y_cal, pred_cal = _data(500, seed=2)
    y_test, pred_test = _data(400, seed=3)
    u = estimate_uncertainty(
        y_test, method="conformal", y_calib=y_cal, pred_calib=pred_cal,
        pred_test=pred_test, alpha=0.1, target_names=["clwc", "ciwc"],
    )
    cov = u.metrics["overall"]["coverage"]
    assert cov is not None and cov >= 0.8, "共形区间覆盖率明显偏低"
    assert u.metrics["overall"]["interval_width"] > 0


def test_conformal_requires_calibration_data():
    with pytest.raises(ValueError):
        estimate_uncertainty(np.zeros((3, 1)), method="conformal", pred_test=np.zeros((3, 1)))


def test_alpha_must_be_valid():
    y_cal, pred_cal = _data(50)
    with pytest.raises(ValueError):
        conformal_interval(y_cal, pred_cal, np.zeros((2, 2)), alpha=1.5)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        estimate_uncertainty(np.zeros((3, 1)), method="telepathy")


# ----------------------------------------------------------------------
def test_mc_dropout_samples_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))

    def predict_fn(x):
        return rng.normal(size=(x.shape[0], 2))

    s = mc_dropout_samples(predict_fn, X, n_samples=7, seed=0)
    assert s.shape == (7, 20, 2)


def test_ensemble_samples_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 3))
    fns = [lambda x: rng.normal(size=(x.shape[0], 2)) for _ in range(4)]
    s = ensemble_samples(fns, X)
    assert s.shape == (4, 15, 2)


def test_normal_interval_has_no_guarantee_flag():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(5, 10, 2))
    ni = normal_interval(s)
    assert np.all(ni["hi"] >= ni["lo"])
    u = estimate_uncertainty(rng.normal(size=(10, 2)), method="normal", samples=s)
    rep = reliability_report(rng.normal(size=(10, 2)), u)
    assert "NONE" in rep["guarantee"], "normal 方法必须声明无覆盖率保证"


# ----------------------------------------------------------------------
def test_crps_is_smaller_for_better_predictions():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(100, 1))
    sigma = np.full_like(y, 0.1)
    good = float(np.mean(crps(y, y, sigma)))
    bad = float(np.mean(crps(y, y + 1.0, sigma)))
    assert good < bad


def test_crps_penalises_overly_wide_distribution():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(100, 1))
    narrow = float(np.mean(crps(y, y, np.full_like(y, 0.2))))
    wide = float(np.mean(crps(y, y, np.full_like(y, 5.0))))
    assert narrow < wide


# ----------------------------------------------------------------------
def test_interval_metrics_coverage_and_width():
    y = np.array([[0.0], [1.0], [2.0]])
    lo = np.array([[-0.5], [0.5], [1.5]])
    hi = np.array([[0.5], [1.5], [2.5]])
    m = interval_metrics(y, lo, hi, ["clwc"])
    assert m["overall"]["coverage"] == 1.0
    assert m["overall"]["interval_width"] == pytest.approx(1.0)


def test_interval_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        interval_metrics(np.zeros((3, 1)), np.zeros((2, 1)), np.ones((3, 1)), ["a"])


def test_uncertainty_result_is_json_serializable():
    import json

    y_cal, pred_cal = _data(100)
    y_test, pred_test = _data(50)
    u = estimate_uncertainty(
        y_test, method="conformal", y_calib=y_cal, pred_calib=pred_cal, pred_test=pred_test
    )
    payload = json.dumps(u.to_dict(), ensure_ascii=False)
    assert "conformal" in payload
