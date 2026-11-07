"""训练性能剖析：数据加载 / 计算 / 传输 三段耗时与吞吐统计。

用途
----

* 判断瓶颈在「喂数据」还是在「算模型」——这决定该调 DataLoader 还是调模型；
* 为 M6 Benchmark 提供统一口径的采样指标（samples/s、epoch time、峰值显存）；
* 全程只做被动计时，不改变训练语义（计时用 ``time.perf_counter``）。

GPU 利用率
----------

通过 ``nvidia-smi`` 采样获取（仅在存在 CUDA 设备时尝试）。取不到就返回
``None``，**不伪造**利用率数字。
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ..utils.logging import get_logger

__all__ = ["StageTimer", "Profiler", "EpochStats", "sample_gpu_utilization"]

_logger = get_logger("hardware.profiling")


@dataclass
class StageTimer:
    """单阶段累计计时。"""

    name: str
    total_seconds: float = 0.0
    count: int = 0

    @property
    def average_seconds(self) -> float:
        return self.total_seconds / self.count if self.count else 0.0


class Profiler:
    """分阶段计时器（线程安全不保证，单线程训练足够）。"""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.timers: dict[str, StageTimer] = {}
        self._counters: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """计时一个阶段：``with profiler.stage("data_load"): ...``"""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            timer = self.timers.setdefault(name, StageTimer(name))
            timer.total_seconds += dt
            timer.count += 1

    def count(self, name: str, n: int = 1) -> None:
        """累加计数器（如样本数、batch 数）。"""
        self._counters[name] = self._counters.get(name, 0) + int(n)

    def counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def reset(self) -> None:
        self.timers.clear()
        self._counters.clear()

    def stage_accumulate(self, name: str, seconds: float) -> None:
        """累加一个阶段耗时（用于手工计时的场景，如「取下一个 batch」的等待）。"""
        if not self.enabled:
            return
        timer = self.timers.setdefault(name, StageTimer(name))
        timer.total_seconds += float(seconds)
        timer.count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": {
                name: {
                    "total_seconds": round(t.total_seconds, 6),
                    "count": t.count,
                    "average_seconds": round(t.average_seconds, 6),
                }
                for name, t in self.timers.items()
            },
            "counters": dict(self._counters),
        }


@dataclass
class EpochStats:
    """单个 epoch 的性能统计。"""

    epoch: int = 0
    n_samples: int = 0
    n_batches: int = 0
    epoch_seconds: float = 0.0
    data_load_seconds: float = 0.0
    compute_seconds: float = 0.0
    transfer_seconds: float = 0.0
    train_loss: float | None = None
    val_loss: float | None = None

    @property
    def samples_per_second(self) -> float:
        return self.n_samples / self.epoch_seconds if self.epoch_seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "n_samples": self.n_samples,
            "n_batches": self.n_batches,
            "epoch_seconds": round(self.epoch_seconds, 6),
            "data_load_seconds": round(self.data_load_seconds, 6),
            "compute_seconds": round(self.compute_seconds, 6),
            "transfer_seconds": round(self.transfer_seconds, 6),
            "samples_per_second": round(self.samples_per_second, 3),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }


def sample_gpu_utilization(device_index: int = 0) -> dict[str, Any] | None:
    """用 ``nvidia-smi`` 采样一次 GPU 利用率与显存。

    不可用（无 CUDA / 无 nvidia-smi / 解析失败）时返回 ``None``，不编造。
    """
    cmd = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=10, check=False
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    line = out.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        util, used, total = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    return {
        "utilization_percent": util,
        "memory_used_mib": used,
        "memory_total_mib": total,
    }
