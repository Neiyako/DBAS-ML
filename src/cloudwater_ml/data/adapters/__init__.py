"""数据适配器包：导入即完成注册。"""
from .array_adapter import ArrayAdapter
from .base import ADAPTERS, DataAdapter, available_adapters, get_adapter, register_adapter
from .netcdf_adapter import NetcdfAdapter

__all__ = [
    "ADAPTERS",
    "ArrayAdapter",
    "DataAdapter",
    "NetcdfAdapter",
    "available_adapters",
    "get_adapter",
    "register_adapter",
]
