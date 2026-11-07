"""设备检测：不强制 CUDA，CPU 环境可用。"""
from __future__ import annotations

__all__ = ["resolve_device", "available_devices"]

_VALID = ("cpu", "mps", "cuda")


def resolve_device(spec: str = "auto") -> str:
    """把配置里的 device 解析为实际可用设备。

    - ``auto``：cuda > mps > cpu，自动探测；
    - 其它取值：显式指定，但会先校验可用性，不可用则回退 cpu 并非静默失败。
    """
    spec = (spec or "auto").lower()
    if spec != "auto":
        if spec not in _VALID:
            raise ValueError(f"未知设备 {spec!r}，可选: auto | {' | '.join(_VALID)}")
        if spec == "cuda" and not _cuda_ok():
            raise RuntimeError("配置指定 cuda，但当前环境无可用 CUDA 设备")
        if spec == "mps" and not _mps_ok():
            raise RuntimeError("配置指定 mps，但当前环境无可用 MPS 设备")
        return spec

    if _cuda_ok():  # pragma: no cover - CPU 环境不走
        return "cuda"
    if _mps_ok():  # pragma: no cover - CI 无 MPS 时走 cpu 分支
        return "mps"
    return "cpu"


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _mps_ok() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        return False


def available_devices() -> list[str]:
    """列出当前可用设备（用于报告与调试）。"""
    out = ["cpu"]
    if _mps_ok():  # pragma: no cover
        out.append("mps")
    if _cuda_ok():  # pragma: no cover
        out.append("cuda")
    return out
