"""数据适配器统一接口：NumPy/CSV/Parquet/NetCDF/Xarray 走同一协议。

设计要点：

- 适配器只负责「把数据源变成 :class:`SampleBundle`」，不负责划分/归一化/清洗；
- NetCDF 适配器必须支持按需加载、分块、时间/空间范围筛选与分批读取，
  禁止默认把整个文件读进内存（见 ``data.netcdf.max_in_memory_gb`` 保护）；
- ``inspect`` 只读元数据/结构，供 ``cloudwater inspect`` 与质量检查使用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from ..schema import SampleBundle

__all__ = ["DataAdapter", "register_adapter", "get_adapter", "available_adapters", "ADAPTERS"]

ADAPTERS: dict[str, type[DataAdapter]] = {}


def register_adapter(name: str):
    """适配器注册装饰器：新增数据源格式时只需新增一个子类。"""

    def deco(cls: type[DataAdapter]) -> type[DataAdapter]:
        cls.name = name
        ADAPTERS[name] = cls
        return cls

    return deco


def get_adapter(name: str) -> DataAdapter:
    if name not in ADAPTERS:
        raise KeyError(
            f"未知数据适配器 {name!r}，可用: {sorted(ADAPTERS)}（新增格式请实现 DataAdapter 子类并注册）"
        )
    return ADAPTERS[name]()


def available_adapters() -> list[str]:
    return sorted(ADAPTERS)


class DataAdapter(ABC):
    """数据适配器基类。"""

    name: str = "base"

    @abstractmethod
    def load(self, data_cfg: dict[str, Any]) -> SampleBundle:
        """按配置读取数据为统一 SampleBundle（应施加范围筛选与抽样，控制内存）。"""

    def inspect(self, data_cfg: dict[str, Any]) -> dict[str, Any]:
        """只读检查：结构、变量、维度、单位、来源契约 —— 不载入数据值。"""
        raise NotImplementedError(f"{self.name} 适配器未实现 inspect")

    def iter_batches(self, data_cfg: dict[str, Any], batch_samples: int) -> Iterator[SampleBundle]:
        """分批读取：按样本批次产出 SampleBundle（大 NetCDF 的低内存入口）。"""
        raise NotImplementedError(
            f"{self.name} 适配器不支持分批读取（目前仅 netcdf 支持 iter_batches）"
        )
