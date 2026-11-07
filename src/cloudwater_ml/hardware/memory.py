"""显存 / 内存监控（CUDA、MPS、CPU RSS 统一口径）。

要点
----

* **不用「清空 CUDA cache」当万能药**：OOM 的正确处理是记录实际配置 + 给出
  可执行建议（减小 batch、启用梯度累积、梯度检查点、换精度），而不是偷偷改
  配置让实验与记录不一致。
* 峰值显存可被准确记录：CUDA 用 ``max_memory_allocated`` / ``max_memory_reserved``；
  MPS 用 ``torch.mps.current_allocated_memory``；CPU 用 ``resource`` 的 RSS 峰值。
* 不依赖 psutil：缺失时 CPU 内存降级为 ``None`` 而不是编造数值。
"""
from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from typing import Any

from ..utils.exceptions import OutOfMemoryError
from ..utils.logging import get_logger

__all__ = ["MemorySnapshot", "MemoryMonitor", "current_memory"]

_logger = get_logger("hardware.memory")


def _rss_bytes() -> int | None:
    """当前进程 RSS（字节）。macOS 返回字节，Linux 返回 KB，已归一化。"""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


@dataclass
class MemorySnapshot:
    """某一时刻的内存状态（单位：字节）。"""

    device_type: str = "cpu"
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None
    host_rss_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        def gib(v: int | None) -> float | None:
            return None if v is None else round(v / (1024**3), 4)

        return {
            "device_type": self.device_type,
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
            "host_rss_bytes": self.host_rss_bytes,
            "allocated_gib": gib(self.allocated_bytes),
            "peak_allocated_gib": gib(self.peak_allocated_bytes),
            "peak_reserved_gib": gib(self.peak_reserved_bytes),
            "host_rss_gib": gib(self.host_rss_bytes),
        }


def current_memory(device_type: str = "cpu") -> MemorySnapshot:
    """取当前内存快照（不抛异常；拿不到的字段为 None）。"""
    snap = MemorySnapshot(device_type=device_type)
    snap.host_rss_bytes = _rss_bytes()
    if device_type == "cuda":
        try:
            import torch

            snap.allocated_bytes = int(torch.cuda.memory_allocated())
            snap.reserved_bytes = int(torch.cuda.memory_reserved())
            snap.peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
            snap.peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
            free, total = torch.cuda.mem_get_info()
            snap.free_bytes = int(free)
            snap.total_bytes = int(total)
        except Exception as exc:  # pragma: no cover - 非 CUDA 环境
            _logger.debug("读取 CUDA 内存失败: %s", exc)
    elif device_type == "mps":
        try:
            import torch

            mps = getattr(torch, "mps", None)
            if mps is not None:
                alloc = getattr(mps, "current_allocated_memory", None)
                driver = getattr(mps, "driver_allocated_memory", None)
                if callable(alloc):
                    snap.allocated_bytes = int(alloc())
                if callable(driver):
                    snap.reserved_bytes = int(driver())
                peak = getattr(mps, "current_allocated_memory", None)
                if callable(peak):
                    snap.peak_allocated_bytes = int(peak())
        except Exception as exc:  # pragma: no cover
            _logger.debug("读取 MPS 内存失败: %s", exc)
    return snap


class MemoryMonitor:
    """训练期内存监控器。

    用法::

        mon = MemoryMonitor("cuda")
        mon.start()            # 记录基线
        ...                    # 训练
        snap = mon.stop()      # 峰值快照
    """

    def __init__(self, device_type: str = "cpu") -> None:
        self.device_type = device_type
        self.started_at: MemorySnapshot | None = None
        self._last: MemorySnapshot | None = None

    def start(self) -> MemorySnapshot:
        """记录起点快照（并重置 CUDA 峰值计数，保证峰值是本次训练的）。"""
        if self.device_type == "cuda":
            try:
                import torch

                torch.cuda.reset_peak_memory_stats()
            except Exception:  # pragma: no cover
                pass
        self.started_at = current_memory(self.device_type)
        self._last = self.started_at
        return self.started_at

    def sample(self) -> MemorySnapshot:
        """取当前快照（训练循环里按需调用，开销极小）。"""
        self._last = current_memory(self.device_type)
        return self._last

    def stop(self) -> MemorySnapshot:
        """结束并返回含峰值的快照。"""
        self._last = current_memory(self.device_type)
        return self._last

    @property
    def peak_vram_bytes(self) -> int | None:
        """峰值显存（CUDA reserved / MPS allocated；CPU 返回 None）。"""
        snap = self._last or current_memory(self.device_type)
        if self.device_type == "cuda":
            return snap.peak_reserved_bytes if snap.peak_reserved_bytes is not None else snap.peak_allocated_bytes
        if self.device_type == "mps":
            return snap.reserved_bytes
        return None

    @property
    def peak_host_rss_bytes(self) -> int | None:
        """宿主进程内存峰值。"""
        return (self._last or current_memory(self.device_type)).host_rss_bytes

    def to_dict(self) -> dict[str, Any]:
        """结构化输出（写进实验 metrics.json）。"""
        snap = self._last or current_memory(self.device_type)
        d = snap.to_dict()
        d["peak_vram_bytes"] = self.peak_vram_bytes
        d["peak_host_rss_bytes"] = self.peak_host_rss_bytes
        return d

    # ------------------------------------------------------------------
    @staticmethod
    def check_headroom(
        device_type: str,
        required_bytes: int,
        *,
        context: dict[str, Any] | None = None,
    ) -> MemorySnapshot:
        """训练前显存检查：可用显存不足时抛 :class:`OutOfMemoryError`。

        **不做自动降级**：抛出的异常里带上实际配置与可执行建议，由使用者决定
        是否修改配置（避免「配置写 256、实际跑 64」这种不可追溯的降级）。
        """
        snap = current_memory(device_type)
        info = dict(context or {})
        if device_type == "cuda" and snap.free_bytes is not None:
            info.setdefault("free_vram_gib", round(snap.free_bytes / 1024**3, 2))
            info.setdefault("total_vram_gib", round((snap.total_bytes or 0) / 1024**3, 2))
            if snap.free_bytes < required_bytes:
                raise OutOfMemoryError(
                    f"显存不足：需要约 {required_bytes / 1024**3:.2f} GiB，"
                    f"当前可用 {snap.free_bytes / 1024**3:.2f} GiB",
                    hint=(
                        "可尝试：1) 减小 batch_size；2) 启用 gradient_accumulation_steps "
                        "以等效大批量；3) 降低输入分辨率/时间步；4) 对大模型启用梯度检查点；"
                        "5) 关闭其它占用显存的进程"
                    ),
                    context=info,
                )
        return snap
