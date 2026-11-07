"""流式训练数据源（v2.0）：把 NetCDF 分批读取包装成 torch IterableDataset。

为什么需要它
------------

v1.0/v2.0 早期版本的训练入口 `Trainer.fit(X, Y, ...)` 要求**整份数据已在内存里**
（numpy 矩阵）。TB 级 ERA5 走不通这条路，而数据层早已具备按批读取能力
（`data/loader.iter_batches`）。本模块把两者接起来：

```
NetCDF ──iter_batches(batch_samples)──► SampleBundle ──► (x, y) float32 张量 ──► Trainer.fit_stream
```

分片策略（`num_workers > 1` 时）
--------------------------------

* ``shard_by="time"``（**默认**）：把 ``netcdf.time_range`` 均分成 N 段，每个 worker
  只读自己那一段 —— **I/O 不重复**，适合 ERA5 这种时间维数据；
  若配置里没有 ``time_range``，退化为按批序号取模，并**显式告警** I/O 会被放大。
* ``shard_by="batch"``：按批序号取模（简单、但每个 worker 都会遍历文件）。

诚实性：本模块不做随机 shuffle（流式下需要 shuffle buffer，成本高且会破坏顺序读取）；
需要打乱时请由调用方提供 shuffle buffer 或改用内存模式。
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..utils.exceptions import ConfigurationError
from ..utils.logging import get_logger

__all__ = ["NetCDFStreamDataset", "shard_time_range"]

_logger = get_logger("data.stream")


def shard_time_range(time_range: Any, n_shards: int) -> list[list[str]]:
    """把 ``[start, end]`` 时间范围均分成 ``n_shards`` 段（闭区间、可用字符串日期）。

    Returns
    -------
    list[list[str]]
        长度为 ``n_shards`` 的 ``[[start_i, end_i], ...]``；无法解析时抛
        :class:`ConfigurationError`（不静默产生错误分片）。
    """
    n_shards = max(1, int(n_shards))
    if n_shards == 1:
        return [list(time_range)]
    if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
        raise ConfigurationError(
            f"time_range 必须是 [start, end]，当前 {time_range!r}"
        )
    import pandas as pd

    try:
        start = pd.Timestamp(time_range[0])
        end = pd.Timestamp(time_range[1])
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(
            f"time_range 无法解析为时间: {time_range!r}",
            hint="使用 'YYYY-MM-DD' 或 ISO 时间字符串",
        ) from exc
    if end < start:
        raise ConfigurationError(f"time_range 的结束早于开始: {time_range!r}")
    total = (end - start).total_seconds()
    step = total / n_shards
    segments: list[list[str]] = []
    for i in range(n_shards):
        seg_start = start + pd.Timedelta(seconds=step * i)
        seg_end = end if i == n_shards - 1 else start + pd.Timedelta(seconds=step * (i + 1))
        segments.append([seg_start.isoformat(), seg_end.isoformat()])
    return segments


class NetCDFStreamDataset(IterableDataset):
    """把 ``data.iter_batches`` 包装成 ``IterableDataset``，逐批产出 ``(x, y)``。

    Parameters
    ----------
    cfg : dict
        完整实验配置或 ``data`` 段。
    batch_samples : int
        每批样本数上限（交给适配器，适配器保证 ≤ 该值）。
    shard_by : str
        ``"time"``（默认，按时间范围分片）或 ``"batch"``（按批序号取模）。
    shuffle_buffer : int
        0 = 不打乱（顺序读取，I/O 友好）；>0 = 维护一个缓冲区并随机取出。

    每批产出的张量形状：``x [B, F]``、``y [B, T]``，dtype=float32。
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        batch_samples: int = 100_000,
        shard_by: str = "time",
        shuffle_buffer: int = 0,
    ) -> None:
        from .loader import resolve_data_cfg

        self.cfg = dict(cfg)
        self.data_cfg = dict(resolve_data_cfg(cfg))
        self.batch_samples = max(1, int(batch_samples))
        if shard_by not in ("time", "batch"):
            raise ConfigurationError(
                f"未知 shard_by={shard_by!r}", hint="可选: time | batch"
            )
        self.shard_by = shard_by
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.feature_names: list[str] = list(self.data_cfg.get("features") or [])
        self.target_names: list[str] = list(self.data_cfg.get("targets") or [])
        if not self.feature_names or not self.target_names:
            raise ConfigurationError(
                "流式数据源需要 data.features 与 data.targets（决定张量列数）"
            )

    # ------------------------------------------------------------------
    def _worker_cfg(self) -> list[dict[str, Any]]:
        """返回本 worker 要读的配置列表（分片后通常只有 1 份）。"""
        info = get_worker_info()
        n_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        if n_workers <= 1:
            return [self.data_cfg]

        if self.shard_by == "time":
            netcdf = self.data_cfg.get("netcdf") or {}
            time_range = netcdf.get("time_range")
            if time_range:
                segments = shard_time_range(time_range, n_workers)
                cfg = dict(self.data_cfg)
                netcdf_cfg = dict(netcdf)
                netcdf_cfg["time_range"] = segments[worker_id]
                cfg["netcdf"] = netcdf_cfg
                return [cfg]
            _logger.warning(
                "shard_by=time 但配置没有 netcdf.time_range：退化为按批序号取模，"
                "每个 worker 都会遍历文件（I/O 放大 %d 倍）；建议配置 time_range 或 num_workers=0",
                n_workers,
            )
        # batch 分片：返回全量配置，由 __iter__ 按批序号取模
        return [self.data_cfg]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        from .loader import iter_batches

        info = get_worker_info()
        n_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        use_modulo = n_workers > 1 and self.shard_by == "batch"
        if n_workers > 1 and self.shard_by == "time":
            netcdf = self.data_cfg.get("netcdf") or {}
            use_modulo = not bool(netcdf.get("time_range"))

        buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        rng = np.random.default_rng(1234 + worker_id)
        idx = 0
        for cfg in self._worker_cfg():
            for bundle in iter_batches(cfg, self.batch_samples):
                if use_modulo and (idx % n_workers) != worker_id:
                    idx += 1
                    continue
                idx += 1
                x = torch.from_numpy(np.ascontiguousarray(bundle.X, dtype=np.float32))
                y = torch.from_numpy(np.ascontiguousarray(bundle.Y, dtype=np.float32))
                if y.ndim == 1:
                    y = y.reshape(-1, 1)
                if self.shuffle_buffer <= 1:
                    yield x, y
                    continue
                buffer.append((x, y))
                if len(buffer) >= self.shuffle_buffer:
                    j = int(rng.integers(0, len(buffer)))
                    buffer[j], buffer[-1] = buffer[-1], buffer[j]
                    yield buffer.pop()
        # 冲刷缓冲区
        while buffer:
            j = int(rng.integers(0, len(buffer)))
            buffer[j], buffer[-1] = buffer[-1], buffer[j]
            yield buffer.pop()
