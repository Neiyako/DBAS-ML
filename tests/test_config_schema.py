"""配置校验与统一异常体系测试。"""
from __future__ import annotations

import pytest

from cloudwater_ml.utils.config import load_config
from cloudwater_ml.utils.config_schema import describe_config, validate_config
from cloudwater_ml.utils.exceptions import (
    CloudWaterError,
    ConfigurationError,
    DataError,
    DataValidationError,
    DeviceError,
    ExperimentStateError,
    ModelNotRegisteredError,
    OutOfMemoryError,
    exception_to_dict,
)


# ----------------------------------------------------------------------
# 异常体系
# ----------------------------------------------------------------------
def test_all_framework_errors_share_base_class():
    for exc_cls in (
        ConfigurationError,
        DataError,
        DataValidationError,
        DeviceError,
        ModelNotRegisteredError,
        OutOfMemoryError,
        ExperimentStateError,
    ):
        assert issubclass(exc_cls, CloudWaterError)


def test_configuration_error_exit_code_is_two():
    assert ConfigurationError("x").exit_code == 2


def test_error_carries_context_and_hint():
    err = OutOfMemoryError("显存不足", hint="减小 batch_size", context={"batch_size": 256})
    d = err.to_dict()
    assert d["error_type"] == "OutOfMemoryError"
    assert d["hint"] == "减小 batch_size"
    assert d["context"]["batch_size"] == 256


def test_exception_to_dict_handles_unknown_exception():
    d = exception_to_dict(ValueError("普通异常"))
    assert d["error_type"] == "ValueError"
    assert d["exit_code"] == 1


# ----------------------------------------------------------------------
# 配置校验
# ----------------------------------------------------------------------
def test_shipped_experiment_configs_pass_validation():
    """仓库自带的所有实验配置都必须能通过校验（防止文档与代码漂移）。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "configs" / "experiments").glob("*.yaml"))
    assert configs, "找不到实验配置"
    for cfg_path in configs:
        cfg = load_config(cfg_path)
        warnings = validate_config(cfg)
        assert isinstance(warnings, list)


def test_base_configs_pass_validation():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("training.yaml", "data.yaml", "model.yaml"):
        cfg = load_config(root / "configs" / name)
        validate_config(cfg)


def test_invalid_adapter_is_rejected():
    with pytest.raises(ConfigurationError) as exc:
        validate_config({"data": {"adapter": "hdf5", "features": ["a"], "targets": ["b"]}})
    assert "adapter" in str(exc.value)


def test_missing_features_is_rejected():
    with pytest.raises(ConfigurationError):
        validate_config({"data": {"adapter": "npz", "targets": ["clwc"]}})


def test_non_positive_epochs_rejected():
    with pytest.raises(ConfigurationError):
        validate_config({"training": {"epochs": 0}})


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ConfigurationError) as exc:
        validate_config({"split": {"strategy": "time", "ratios": [0.7, 0.15, 0.1]}})
    assert "ratios" in str(exc.value)


def test_mix_fraction_out_of_range_rejected():
    with pytest.raises(ConfigurationError):
        validate_config({"mix": {"real_fraction": 1.5}})


def test_unknown_top_level_section_is_warning_not_error():
    warnings = validate_config({"data": {"adapter": "npz", "features": ["a"], "targets": ["b"]},
                                "totally_unknown": {"x": 1}})
    assert any("totally_unknown" in w for w in warnings)


def test_invalid_precision_rejected():
    with pytest.raises(ConfigurationError):
        validate_config({"hardware": {"precision": "int4"}})


def test_invalid_constraint_rejected():
    with pytest.raises(ConfigurationError):
        validate_config({"physics": {"constraints": ["teleport"]}})


def test_describe_config_returns_summary():
    cfg = {"model": {"name": "mlp"}, "training": {"epochs": 5, "batch_size": 8}}
    d = describe_config(cfg)
    assert d["model"] == "mlp"
    assert d["epochs"] == 5
