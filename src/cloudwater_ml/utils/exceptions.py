"""统一异常体系（v2.0）。

设计原则
--------

1. **单一基类**：所有框架内异常继承 :class:`CloudWaterError`，调用方可以
   ``except CloudWaterError`` 一次性捕获，避免 ``except Exception`` 吞掉真实问题。
2. **语义分层**：按子系统划分（配置 / 数据 / 模型 / 训练 / 检查点 / 实验 / 物理 /
   设备 / 依赖），既便于定位，又保留 ``except DataError`` 这类粗粒度捕获能力。
3. **可结构化**：每个异常携带 ``context`` 字典，可直接转成 JSON 记入日志或
   实验记录；``exit_code`` 供 CLI 区分「用户配置错误(2)」与「运行失败(1)」。
4. **不吞异常**：任何真实故障都必须带上「发生了什么 / 为什么 / 怎么修」三要素。

本模块只依赖标准库，保证在未安装 torch / xarray 的最小环境中也能导入。
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "CloudWaterError",
    "ConfigurationError",
    "DependencyError",
    "DeviceError",
    "DataError",
    "DataNotFoundError",
    "DataValidationError",
    "CacheError",
    "ModelError",
    "ModelNotRegisteredError",
    "TrainingError",
    "OutOfMemoryError",
    "CheckpointError",
    "ExperimentError",
    "ExperimentStateError",
    "PhysicsConstraintError",
    "UncertaintyNotSupportedError",
    "exception_to_dict",
]


class CloudWaterError(Exception):
    """框架异常基类。

    Parameters
    ----------
    message : str
        面向人的错误描述，必须说明「哪里错了」。
    hint : str | None
        面向人的修复建议（可选，但强烈建议给出）。
    context : dict | None
        结构化上下文（路径、配置键、实验 id、GPU 信息等），会被写入日志。
    """

    #: CLI 退出码：2 = 用户可自行修正的配置/用法错误，1 = 运行期失败。
    exit_code: int = 1

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.message = str(message)
        self.hint = hint
        self.context: dict[str, Any] = dict(context or {})
        super().__init__(self.message)

    def __str__(self) -> str:  # pragma: no cover - 展示逻辑
        base = self.message
        if self.hint:
            return f"{base}\n  建议: {self.hint}"
        return base

    def to_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的结构化记录。"""
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "hint": self.hint,
            "context": self.context,
            "exit_code": self.exit_code,
        }


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
class ConfigurationError(CloudWaterError, ValueError):
    """配置缺失、类型错误或取值非法。

    同时继承 :class:`ValueError`：**保持与 v1.0 的兼容性**
    （旧代码里 ``except ValueError`` 仍能捕获配置错误）。
    """

    exit_code = 2


# ----------------------------------------------------------------------
# 依赖
# ----------------------------------------------------------------------
class DependencyError(CloudWaterError):
    """缺少可选依赖（如 lightgbm / dask / pyarrow），或版本不满足。"""

    exit_code = 2


# ----------------------------------------------------------------------
# 设备 / 硬件
# ----------------------------------------------------------------------
class DeviceError(CloudWaterError, RuntimeError):
    """设备选择或硬件探测失败（CUDA 不可用、显存不足等）。

    同时继承 :class:`RuntimeError`（与 v1.0 的 ``RuntimeError`` 行为保持一致）。
    """


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
class DataError(CloudWaterError, ValueError):
    """数据读取、解析或契约违反（兼容 v1.0 的 ``except ValueError``）。"""


class DataNotFoundError(DataError):
    """数据文件 / 变量 / 维度不存在。"""


class DataValidationError(DataError):
    """数据质量检查未通过（NaN/Inf/越界/单位/坐标等）。"""


class CacheError(DataError):
    """缓存写入、读取、失效或版本校验失败。"""


# ----------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------
class ModelError(CloudWaterError, ValueError):
    """模型构造、参数或契约违反（兼容 v1.0 的 ``except ValueError``）。"""


class ModelNotRegisteredError(ModelError):
    """请求了未注册 / 未实现的模型名。"""

    exit_code = 2


# ----------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------
class TrainingError(CloudWaterError, RuntimeError):
    """训练过程失败（兼容 v1.0 的 ``except RuntimeError``）。"""


class OutOfMemoryError(TrainingError):
    """显存 / 内存不足。``context`` 记录实际 batch size、精度、模型与建议方案。"""


# ----------------------------------------------------------------------
# 检查点
# ----------------------------------------------------------------------
class CheckpointError(CloudWaterError):
    """检查点保存、读取或恢复失败（损坏、版本不兼容、状态缺失）。"""


# ----------------------------------------------------------------------
# 实验
# ----------------------------------------------------------------------
class ExperimentError(CloudWaterError):
    """实验编排 / 执行失败。"""


class ExperimentStateError(ExperimentError):
    """实验状态机非法迁移（如 CREATED → COMPLETED 跳步、重复完成）。"""


# ----------------------------------------------------------------------
# 物理 / 科学
# ----------------------------------------------------------------------
class PhysicsConstraintError(CloudWaterError):
    """物理约束定义、单位口径或组合方式非法。"""


class UncertaintyNotSupportedError(CloudWaterError):
    """请求了尚未实现的不确定性估计方法（拒绝伪装成已实现）。"""

    exit_code = 2


def exception_to_dict(exc: BaseException) -> dict[str, Any]:
    """把任意异常转成结构化字典（未知异常也保证字段齐全）。"""
    if isinstance(exc, CloudWaterError):
        return exc.to_dict()
    return {
        "error_type": type(exc).__name__,
        "message": str(exc),
        "hint": None,
        "context": {},
        "exit_code": 1,
    }
