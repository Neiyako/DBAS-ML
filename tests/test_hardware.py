"""硬件抽象层测试：设备探测、精度策略、显存监控。

分层原则：**GPU 专属断言在无 CUDA 环境下自动跳过**，但设备矩阵与精度
决策逻辑必须在任何环境下都被真正执行（用 CPU/MPS 跑真实 autocast）。
"""
from __future__ import annotations

import pytest

from cloudwater_ml.hardware import (
    MemoryMonitor,
    autocast_context,
    current_memory,
    detect_hardware,
    device_report,
    make_grad_scaler,
    resolve_precision,
    resolve_torch_device,
)
from cloudwater_ml.utils.exceptions import ConfigurationError, DeviceError

cuda_only = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="需要 CUDA 设备",
)


# ----------------------------------------------------------------------
# 设备探测
# ----------------------------------------------------------------------
def test_detect_hardware_returns_valid_device_type():
    info = detect_hardware("auto")
    assert info.device_type in ("cpu", "cuda", "mps")
    assert info.requested == "auto"
    assert isinstance(info.to_dict(), dict)


def test_detect_hardware_rejects_unknown_spec():
    with pytest.raises(DeviceError) as exc:
        detect_hardware("tpu")
    assert "tpu" in str(exc.value)


def test_detect_hardware_explicit_cpu_never_falls_back():
    info = detect_hardware("cpu")
    assert info.device_type == "cpu"
    assert info.fallback is False


def test_resolve_torch_device_returns_torch_device():
    import torch

    dev = resolve_torch_device("auto")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda", "mps")


def test_device_report_contains_sections():
    text = device_report(detect_hardware("auto"))
    assert "运行环境" in text and "设备" in text


def test_cuda_unavailable_gives_reason_not_silent_cpu():
    """显式要求 cuda 但不可用时必须给出可读原因（而不是静默降级）。"""
    import torch

    if torch.cuda.is_available():
        pytest.skip("本机有 CUDA")
    info = detect_hardware("cuda")
    assert info.device_type == "cpu"
    assert info.fallback is True
    assert info.fallback_reason
    assert "CUDA" in info.fallback_reason


# ----------------------------------------------------------------------
# 精度策略
# ----------------------------------------------------------------------
def test_precision_auto_cuda_prefers_bf16_when_supported():
    p = resolve_precision("auto", device_type="cuda", supports_bfloat16=True)
    assert p.mode == "bf16"
    assert p.autocast_enabled is True
    assert p.scaler_enabled is False  # bf16 无需 loss scaling


def test_precision_auto_cuda_falls_back_to_fp16_with_scaler():
    p = resolve_precision("auto", device_type="cuda", supports_bfloat16=False)
    assert p.mode == "fp16"
    assert p.scaler_enabled is True


def test_precision_auto_cpu_is_fp32():
    p = resolve_precision("auto", device_type="cpu", supports_bfloat16=False)
    assert p.mode == "fp32"
    assert p.autocast_enabled is False
    assert p.scaler_enabled is False


def test_precision_auto_mps_uses_fp16():
    p = resolve_precision("auto", device_type="mps", supports_bfloat16=False)
    assert p.mode == "fp16"
    assert p.autocast_enabled is True


def test_precision_fp16_on_cpu_is_rejected():
    """CPU 无 float16 autocast：必须显式报错而不是悄悄 fallback。"""
    with pytest.raises(ConfigurationError):
        resolve_precision("fp16", device_type="cpu")


def test_precision_bf16_on_unsupported_cuda_is_rejected():
    with pytest.raises(ConfigurationError):
        resolve_precision("bf16", device_type="cuda", supports_bfloat16=False)


def test_precision_unknown_mode_rejected():
    with pytest.raises(ConfigurationError):
        resolve_precision("int4", device_type="cpu")


def test_grad_scaler_only_for_cuda_fp16():
    assert make_grad_scaler(resolve_precision("auto", device_type="cpu")) is None
    assert make_grad_scaler(resolve_precision("auto", device_type="cuda", supports_bfloat16=True)) is None
    scaler = make_grad_scaler(resolve_precision("fp16", device_type="cuda"))
    assert scaler is not None


def test_autocast_fp32_is_a_noop_context():
    import torch

    p = resolve_precision("fp32", device_type="cpu")
    with autocast_context(p):
        out = torch.randn(4, 4) @ torch.randn(4, 4)
    assert out.dtype == torch.float32


def test_autocast_bf16_on_cpu_actually_runs():
    """CPU 上真实跑一次 bf16 autocast（torch CPU 支持 bfloat16）。"""
    import torch

    p = resolve_precision("bf16", device_type="cpu")
    with autocast_context(p):
        out = torch.randn(8, 8) @ torch.randn(8, 8)
    assert out.dtype in (torch.bfloat16, torch.float32)


# ----------------------------------------------------------------------
# 内存监控
# ----------------------------------------------------------------------
def test_current_memory_returns_snapshot():
    snap = current_memory("cpu")
    assert snap.device_type == "cpu"
    assert snap.to_dict()["device_type"] == "cpu"


def test_memory_monitor_tracks_peak():
    mon = MemoryMonitor("cpu")
    mon.start()
    data = [bytearray(1024 * 1024) for _ in range(8)]  # 真实占用一点内存
    snap = mon.stop()
    assert snap.host_rss_bytes is not None and snap.host_rss_bytes > 0
    assert mon.to_dict()["device_type"] == "cpu"
    del data


def test_memory_check_headroom_does_not_raise_on_cpu():
    from cloudwater_ml.hardware.memory import MemoryMonitor as MM

    snap = MM.check_headroom("cpu", required_bytes=10**12)
    assert snap.device_type == "cpu"


@cuda_only
@pytest.mark.gpu
def test_cuda_memory_peak_is_recorded():
    import torch

    mon = MemoryMonitor("cuda")
    mon.start()
    x = torch.randn(1024, 1024, device="cuda")
    del x
    snap = mon.stop()
    assert snap.peak_allocated_bytes is not None
    assert mon.peak_vram_bytes is not None


@cuda_only
@pytest.mark.gpu
def test_cuda_check_headroom_raises_with_actionable_hint():
    from cloudwater_ml.hardware.memory import MemoryMonitor as MM

    with pytest.raises(Exception) as exc:
        MM.check_headroom("cuda", required_bytes=10**15, context={"batch_size": 256})
    assert "建议" in str(exc.value)
