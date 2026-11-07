"""训练运行时计划测试：配置 → 设备/精度/DataLoader/累积策略。"""
from __future__ import annotations

import pytest

from cloudwater_ml.training.runtime import resolve_dataloader_options, resolve_runtime
from cloudwater_ml.utils.exceptions import ConfigurationError


def _cfg(**overrides):
    base = {
        "training": {"epochs": 3, "batch_size": 64, "learning_rate": 0.001, "seed": 1},
        "hardware": {"device": "cpu", "precision": "fp32"},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


# ----------------------------------------------------------------------
def test_resolve_runtime_legacy_device_key_still_works():
    """v1.0 的 training.device 必须继续生效（向后兼容）。"""
    cfg = {"training": {"device": "cpu", "batch_size": 32}}
    plan = resolve_runtime(cfg)
    assert str(plan.device) == "cpu"


def test_resolve_runtime_hardware_section_wins_over_legacy():
    cfg = {"training": {"device": "cpu"}, "hardware": {"device": "cpu", "precision": "fp32"}}
    plan = resolve_runtime(cfg)
    assert str(plan.device) == "cpu"
    assert plan.precision.mode == "fp32"


def test_gradient_accumulation_scales_effective_batch():
    cfg = _cfg(training={"batch_size": 32, "gradient_accumulation_steps": 4})
    plan = resolve_runtime(cfg)
    assert plan.gradient_accumulation_steps == 4
    assert plan.effective_batch_size == 128


def test_invalid_accumulation_steps_rejected():
    cfg = _cfg(training={"gradient_accumulation_steps": 0})
    with pytest.raises(ConfigurationError):
        resolve_runtime(cfg)


def test_gradient_clip_norm_reads_new_and_legacy_keys():
    plan = resolve_runtime(_cfg(training={"gradient_clip_norm": 1.5}))
    assert plan.gradient_clip_norm == 1.5
    plan = resolve_runtime({"training": {"grad_clip": 2.0}})
    assert plan.gradient_clip_norm == 2.0


def test_invalid_clip_rejected():
    with pytest.raises(ConfigurationError):
        resolve_runtime(_cfg(training={"gradient_clip_norm": -1}))


def test_unknown_device_rejected():
    with pytest.raises(ConfigurationError):
        resolve_runtime(_cfg(hardware={"device": "tpu"}))


def test_unknown_precision_rejected():
    with pytest.raises(ConfigurationError):
        resolve_runtime(_cfg(hardware={"precision": "int8"}))


def test_channels_last_auto_depends_on_model():
    assert resolve_runtime(_cfg(), model_name="unet").channels_last is True
    assert resolve_runtime(_cfg(), model_name="mlp").channels_last is False
    assert resolve_runtime(_cfg(), model_name="mlp", supports_channels_last=True).channels_last is True


def test_channels_last_explicit_override():
    assert resolve_runtime(_cfg(hardware={"channels_last": True}), model_name="mlp").channels_last is True
    assert resolve_runtime(_cfg(hardware={"channels_last": False}), model_name="unet").channels_last is False


def test_compile_flag_is_boolean():
    assert resolve_runtime(_cfg(hardware={"compile": True})).compile is True
    assert resolve_runtime(_cfg(hardware={"compile": False})).compile is False


# ----------------------------------------------------------------------
# DataLoader 选项
# ----------------------------------------------------------------------
def test_dataloader_workers_zero_disables_persistent_and_prefetch():
    opts = resolve_dataloader_options({"dataloader": {"num_workers": 0, "persistent_workers": True, "prefetch_factor": 4}})
    assert opts.num_workers == 0
    assert opts.persistent_workers is False
    assert opts.prefetch_factor is None
    kwargs = opts.loader_kwargs()
    assert "persistent_workers" not in kwargs and "prefetch_factor" not in kwargs


def test_dataloader_workers_positive_keeps_prefetch():
    opts = resolve_dataloader_options({"dataloader": {"num_workers": 2, "prefetch_factor": 3}})
    assert opts.num_workers == 2
    assert opts.prefetch_factor == 3
    assert opts.loader_kwargs()["prefetch_factor"] == 3


def test_dataloader_auto_workers_scales_with_data_size():
    small = resolve_dataloader_options({}, n_samples=100)
    assert small.num_workers == 0
    assert any("样本" in n for n in small.notes)


def test_pin_memory_only_on_cuda():
    cpu = resolve_dataloader_options({}, device_type="cpu")
    cuda = resolve_dataloader_options({}, device_type="cuda")
    assert cpu.pin_memory is False
    assert cuda.pin_memory is True


def test_non_blocking_disabled_off_cuda():
    opts = resolve_dataloader_options({}, device_type="mps")
    assert opts.non_blocking is False


def test_legacy_num_workers_is_used_when_new_section_absent():
    opts = resolve_dataloader_options({}, legacy_training_cfg={"num_workers": 3})
    assert opts.num_workers == 3


def test_invalid_num_workers_rejected():
    with pytest.raises(ConfigurationError):
        resolve_dataloader_options({"dataloader": {"num_workers": -1}})
