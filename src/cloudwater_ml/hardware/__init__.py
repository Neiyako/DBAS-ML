"""硬件抽象层（v2.0）：设备探测、显存监控、精度策略、性能剖析。

对外统一入口：``from cloudwater_ml.hardware import detect_hardware, resolve_precision``。
本包**不依赖 CUDA**即可导入与使用（CPU 环境可完整跑通单元测试）。
"""
from __future__ import annotations

from .device import (
    DeviceInfo,
    GPUDeviceInfo,
    detect_hardware,
    device_report,
    resolve_torch_device,
)
from .memory import MemoryMonitor, MemorySnapshot, current_memory
from .precision import PrecisionPolicy, autocast_context, make_grad_scaler, resolve_precision
from .profiling import EpochStats, Profiler, StageTimer, sample_gpu_utilization

__all__ = [
    "DeviceInfo",
    "GPUDeviceInfo",
    "detect_hardware",
    "device_report",
    "resolve_torch_device",
    "MemoryMonitor",
    "MemorySnapshot",
    "current_memory",
    "PrecisionPolicy",
    "autocast_context",
    "make_grad_scaler",
    "resolve_precision",
    "EpochStats",
    "Profiler",
    "StageTimer",
    "sample_gpu_utilization",
]
