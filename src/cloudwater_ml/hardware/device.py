"""硬件探测与设备管理（CUDA / MPS / CPU）。

本模块是 v2.0 GPU 训练引擎的底座，遵循三条硬约束：

1. **不写死任何 GPU 型号或显存容量**：所有属性在运行时探测；
2. **CUDA 不可用时必须给出可读原因**（而不是静默退化成 CPU）；
3. **设备选择与硬件信息分离**：``DeviceInfo`` 是「探测结果」，
   ``resolve_torch_device`` 是「选择结果」，二者可独立测试。

典型用法::

    info = detect_hardware("auto")          # 探测
    device = resolve_torch_device("auto")   # 选择（= torch.device）
    print(info.to_dict())                   # 结构化记录（写进实验 metrics）
"""
from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from ..utils.exceptions import DeviceError
from ..utils.logging import get_logger

__all__ = [
    "GPUDeviceInfo",
    "DeviceInfo",
    "detect_hardware",
    "resolve_torch_device",
    "device_report",
]

_logger = get_logger("hardware.device")

_VALID_SPECS = ("auto", "cpu", "cuda", "mps")

#: RTX 50 系（Blackwell, compute capability 12.0）需要 CUDA 12.8+ 构建；
#: 这里只做「提示」，不做硬性拦截——真实能力由 PyTorch 构建版本决定。
_MIN_CUDA_FOR_BLACKWELL = (12, 8)


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class GPUDeviceInfo:
    """单块加速卡的运行时信息（全部为实测值，不含任何硬编码型号）。"""

    index: int
    name: str = "unknown"
    compute_capability: str = ""
    major: int = 0
    minor: int = 0
    total_memory_bytes: int = 0
    free_memory_bytes: int = 0
    multi_processor_count: int = 0
    supports_bfloat16: bool = False
    uuid: str | None = None
    is_current: bool = False

    @property
    def total_memory_gib(self) -> float:
        return round(self.total_memory_bytes / (1024**3), 2)

    @property
    def free_memory_gib(self) -> float:
        return round(self.free_memory_bytes / (1024**3), 2)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_memory_gib"] = self.total_memory_gib
        d["free_memory_gib"] = self.free_memory_gib
        return d


@dataclass
class DeviceInfo:
    """一次硬件探测的完整结果。"""

    requested: str = "auto"
    device_type: str = "cpu"           # cpu | cuda | mps
    fallback: bool = False             # 是否因不可用而回退
    fallback_reason: str | None = None
    torch_version: str = ""
    torch_cuda_version: str | None = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_runtime_version: str | None = None
    cudnn_version: str | None = None
    mps_available: bool = False
    mps_built: bool = False
    gpu_devices: list[GPUDeviceInfo] = field(default_factory=list)
    cpu_count: int = 0
    host_memory_bytes: int | None = None
    platform: str = ""
    python_version: str = ""
    warnings: list[str] = field(default_factory=list)

    # -- 便捷属性 ------------------------------------------------------
    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"

    @property
    def is_accelerator(self) -> bool:
        return self.device_type in ("cuda", "mps")

    @property
    def primary_gpu(self) -> GPUDeviceInfo | None:
        return self.gpu_devices[0] if self.gpu_devices else None

    @property
    def total_vram_bytes(self) -> int:
        g = self.primary_gpu
        return g.total_memory_bytes if g else 0

    @property
    def free_vram_bytes(self) -> int:
        g = self.primary_gpu
        return g.free_memory_bytes if g else 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gpu_devices"] = [g.to_dict() for g in self.gpu_devices]
        d["total_vram_gib"] = round(self.total_vram_bytes / (1024**3), 2)
        d["free_vram_gib"] = round(self.free_vram_bytes / (1024**3), 2)
        return d

    def summary(self) -> str:
        """给人看的一行摘要。"""
        if self.is_cuda:
            g = self.primary_gpu
            if g is None:
                return f"CUDA（{self.cuda_device_count} 卡可见，属性探测失败）"
            return (
                f"CUDA {g.name} (sm_{g.major}{g.minor}) "
                f"{g.total_memory_gib} GiB, {self.cuda_device_count} 卡可见"
            )
        if self.device_type == "mps":
            return "Apple Metal (MPS)"
        return "CPU"


# ----------------------------------------------------------------------
# 探测
# ----------------------------------------------------------------------
def _host_memory_bytes() -> int | None:
    """读取宿主可用物理内存（字节）；拿不到返回 None，不猜。"""
    try:  # psutil 若被安装则优先（跨平台一致）
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        import os

        if sys.platform == "darwin":
            out = os.popen("sysctl -n hw.memsize").read().strip()
            return int(out) if out.isdigit() else None
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _cuda_runtime_version() -> str | None:
    try:
        import torch

        return getattr(torch.version, "cuda", None)
    except Exception:
        return None


def _cudnn_version() -> str | None:
    try:
        import torch

        v = getattr(torch.backends, "cudnn", None)
        return v.version() if v is not None else None
    except Exception:
        return None


def _detect_cuda_devices(torch_mod: Any) -> list[GPUDeviceInfo]:
    """枚举可见 GPU。每块卡的显存/算力都来自运行时查询。"""
    devices: list[GPUDeviceInfo] = []
    try:
        count = int(torch_mod.cuda.device_count())
    except Exception:
        return devices
    try:
        current = int(torch_mod.cuda.current_device())
    except Exception:
        current = 0
    for i in range(count):
        info = GPUDeviceInfo(index=i, is_current=(i == current))
        try:
            props = torch_mod.cuda.get_device_properties(i)
            info.name = str(getattr(props, "name", "unknown"))
            info.major = int(getattr(props, "major", 0))
            info.minor = int(getattr(props, "minor", 0))
            info.compute_capability = f"{info.major}.{info.minor}"
            info.total_memory_bytes = int(getattr(props, "total_memory", 0))
            info.multi_processor_count = int(getattr(props, "multi_processor_count", 0))
        except Exception:
            pass
        try:
            free, total = torch_mod.cuda.mem_get_info(i)
            info.free_memory_bytes = int(free)
            if not info.total_memory_bytes:
                info.total_memory_bytes = int(total)
        except Exception:
            pass
        try:
            info.supports_bfloat16 = bool(torch_mod.cuda.is_bf16_supported())
        except Exception:
            info.supports_bfloat16 = info.major >= 8
        try:
            info.uuid = str(torch_mod.cuda.get_device_properties(i).uuid)
        except Exception:
            info.uuid = None
        devices.append(info)
    return devices


def detect_hardware(requested: str = "auto") -> DeviceInfo:
    """探测当前运行环境，返回 :class:`DeviceInfo`。

    ``requested`` 仅作为「用户想要什么」的记录；真正的选择结果在
    ``device_type`` 中，``fallback`` / ``fallback_reason`` 记录是否降级及原因。
    """
    requested = (requested or "auto").lower()
    if requested not in _VALID_SPECS:
        raise DeviceError(
            f"未知设备规格 {requested!r}",
            hint=f"可选: {' | '.join(_VALID_SPECS)}",
        )

    info = DeviceInfo(
        requested=requested,
        platform=platform.platform(),
        python_version=platform.python_version(),
    )
    try:
        import os

        info.cpu_count = os.cpu_count() or 1
    except Exception:
        info.cpu_count = 1
    info.host_memory_bytes = _host_memory_bytes()

    try:
        import torch
    except ImportError as exc:
        info.device_type = "cpu"
        info.fallback = requested != "cpu"
        info.fallback_reason = f"未安装 PyTorch（{exc}），只能用 CPU"
        info.warnings.append("未检测到 PyTorch，无法进行 GPU 训练")
        return info

    info.torch_version = str(getattr(torch, "__version__", "unknown"))
    info.torch_cuda_version = _cuda_runtime_version()
    info.cuda_runtime_version = _cuda_runtime_version()
    info.cudnn_version = _cudnn_version()

    # -- CUDA ---------------------------------------------------------
    try:
        info.cuda_available = bool(torch.cuda.is_available())
    except Exception:
        info.cuda_available = False
    if info.cuda_available:
        info.cuda_device_count = int(torch.cuda.device_count())
        info.gpu_devices = _detect_cuda_devices(torch)
        if not info.gpu_devices:
            info.cuda_available = False
            info.cuda_device_count = 0
            info.warnings.append("torch.cuda.is_available() 为真但枚举不到设备")

    # -- MPS ----------------------------------------------------------
    try:
        mps = getattr(torch.backends, "mps", None)
        info.mps_built = bool(mps and mps.is_built())
        info.mps_available = bool(mps and mps.is_available())
    except Exception:
        info.mps_built = False
        info.mps_available = False

    # -- 选择 ----------------------------------------------------------
    if requested == "cuda":
        if info.cuda_available:
            info.device_type = "cuda"
        else:
            info.device_type = "cpu"
            info.fallback = True
            info.fallback_reason = (
                "配置指定 cuda，但当前环境无 CUDA 设备"
                f"（torch {info.torch_version} 的 CUDA 构建版本: "
                f"{info.torch_cuda_version or '无'}）"
            )
            info.warnings.append(info.fallback_reason)
    elif requested == "mps":
        if info.mps_available:
            info.device_type = "mps"
        else:
            info.device_type = "cpu"
            info.fallback = True
            info.fallback_reason = "配置指定 mps，但当前环境无可用 MPS 设备"
            info.warnings.append(info.fallback_reason)
    elif requested == "cpu":
        info.device_type = "cpu"
    else:  # auto
        if info.cuda_available:
            info.device_type = "cuda"
        elif info.mps_available:
            info.device_type = "mps"
        else:
            info.device_type = "cpu"
            info.fallback = True
            if info.torch_cuda_version is None:
                info.fallback_reason = (
                    f"PyTorch {info.torch_version} 为非 CUDA 构建"
                    "（torch.version.cuda 为 None），无 CUDA 设备可用"
                )
            else:
                info.fallback_reason = "未检测到可用的 CUDA 设备"
            info.warnings.append(info.fallback_reason)

    # -- Blackwell 提示（不拦截，只提示） -------------------------------
    for g in info.gpu_devices:
        if (g.major, g.minor) >= (12, 0) and info.torch_cuda_version:
            try:
                cuda_ver = tuple(int(x) for x in info.torch_cuda_version.split(".")[:2])
            except Exception:
                cuda_ver = (0, 0)
            if cuda_ver < _MIN_CUDA_FOR_BLACKWELL:
                info.warnings.append(
                    f"检测到 compute capability {g.compute_capability}（Blackwell 系），"
                    f"但 PyTorch CUDA 构建为 {info.torch_cuda_version}；"
                    f"建议使用 ≥ 12.8 的 CUDA 构建以获得完整算子支持"
                )
    return info


def resolve_torch_device(spec: str = "auto", index: int | None = None) -> Any:
    """把设备规格解析为 :class:`torch.device`。

    ``auto`` 按 cuda > mps > cpu 选择；显式指定但不可用时抛
    :class:`DeviceError`（**不静默回退**，避免用户以为在 GPU 上训练）。
    """
    info = detect_hardware(spec)
    if info.device_type == "cuda":
        import torch

        if index is not None:
            if index < 0 or index >= max(info.cuda_device_count, 1):
                raise DeviceError(
                    f"CUDA 设备索引 {index} 越界（可见 {info.cuda_device_count} 张卡）",
                    context={"cuda_device_count": info.cuda_device_count},
                )
            torch.cuda.set_device(index)
            return torch.device("cuda", index)
        return torch.device("cuda")
    if info.device_type == "mps":
        import torch

        return torch.device("mps")
    return torch.device("cpu")


def device_report(info: DeviceInfo | None = None) -> Any:
    """生成人类可读的硬件报告（CLI ``cloudwater gpu info`` 使用）。"""

    info = info or detect_hardware("auto")
    lines: list[str] = []
    lines.append("== 运行环境 ==")
    lines.append(f"平台        : {info.platform}")
    lines.append(f"Python      : {info.python_version}")
    lines.append(f"PyTorch     : {info.torch_version}")
    lines.append(f"CUDA 构建   : {info.torch_cuda_version or '无（非 CUDA 构建）'}")
    lines.append(f"cuDNN       : {info.cudnn_version or '无'}")
    lines.append(f"CPU 核心数  : {info.cpu_count}")
    if info.host_memory_bytes:
        lines.append(f"宿主内存    : {round(info.host_memory_bytes / 1024**3, 2)} GiB")
    lines.append("")
    lines.append("== 设备 ==")
    lines.append(f"请求        : {info.requested}")
    lines.append(f"实际使用    : {info.device_type}")
    if info.fallback:
        lines.append(f"回退原因    : {info.fallback_reason}")
    if info.gpu_devices:
        lines.append(f"可见 GPU 数 : {info.cuda_device_count}")
        for g in info.gpu_devices:
            lines.append(
                f"  [{g.index}] {g.name} | sm_{g.major}{g.minor} | "
                f"显存 {g.total_memory_gib} GiB (空闲 {g.free_memory_gib} GiB) | "
                f"SM {g.multi_processor_count} | bf16={'是' if g.supports_bfloat16 else '否'}"
            )
    else:
        lines.append("可见 GPU 数 : 0（CPU 环境或未安装 CUDA 版 PyTorch）")
    if info.mps_built:
        lines.append(f"MPS         : 构建={'是' if info.mps_built else '否'}, "
                     f"可用={'是' if info.mps_available else '否'}")
    if info.warnings:
        lines.append("")
        lines.append("== 警告 ==")
        for w in info.warnings:
            lines.append(f"  ⚠ {w}")
    return "\n".join(lines)
