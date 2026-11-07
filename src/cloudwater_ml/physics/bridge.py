"""与既有 ``training/losses.py`` 的兼容桥。

既有接口（``cloudwater_ml.training.losses``）：

- ``PhysicsConstraint.penalty(outputs, batch)`` —— 抽象方法，返回标量惩罚；
- ``CompositeLoss`` 依赖约束实例的 ``.name`` 与 ``.weight``，
  并在 ``component_values`` 里按 ``.name`` 逐项记录。

本包的新接口是 ``compute(prediction, batch)``。本模块提供：

- :func:`adapt_for_legacy`：把新约束包装成**旧基类的真实子类**，
  可直接传给 ``training.losses.CompositeLoss``；
- :func:`to_legacy_list`：批量包装。

本模块**不修改** ``training/losses.py``：旧实现保持原样
（其 ``CWPConsistencyPenalty`` 仍然构造即抛）。

旧基类在**首次调用时**才导入（延迟导入），避免 ``physics`` 包对
``training`` 包产生导入期的硬依赖，也避免将来出现循环导入。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor

from .base import PhysicsConstraint

__all__ = ["adapt_for_legacy", "to_legacy_list", "legacy_base_class"]

_LEGACY_BASE: type | None = None
_ADAPTER_CLASSES: dict[type, type] = {}


def legacy_base_class() -> type:
    """返回旧接口基类 ``training.losses.PhysicsConstraint``。

    若 ``training`` 包不可导入（例如精简部署），退化为一个**接口等价**的
    本地最小基类（同名方法 ``penalty(outputs, batch)``、属性 ``name``/
    ``weight``），保证兼容桥在任何环境下都能用。
    """
    global _LEGACY_BASE
    if _LEGACY_BASE is None:
        try:  # pragma: no cover - 取决于部署环境
            from cloudwater_ml.training.losses import PhysicsConstraint as base
        except Exception:  # pragma: no cover - 退化路径

            class base(ABC):  # type: ignore[no-redef]
                """与 ``training.losses.PhysicsConstraint`` 接口等价的最小基类。"""

                name: str = "base"

                def __init__(self, weight: float = 1.0) -> None:
                    self.weight = float(weight)

                @abstractmethod
                def penalty(self, outputs: Tensor, batch: Mapping[str, Any]) -> Tensor:
                    """对一个 batch 的预测施加惩罚（越大越违例）。"""

            _LEGACY_BASE = base
        else:
            _LEGACY_BASE = base
    return _LEGACY_BASE


def _adapter_class(base: type) -> type:
    """构造（或复用）适配器类：``base`` 的子类，实现旧接口 ``penalty``。"""
    cached = _ADAPTER_CLASSES.get(base)
    if cached is not None:
        return cached

    class LegacyConstraintAdapter(base):  # type: ignore[misc, valid-type]
        """把 ``physics`` 包的新约束适配成旧 ``penalty(outputs, batch)`` 接口。

        - ``.name`` / ``.weight`` 透传，满足 ``CompositeLoss`` 的取用方式；
        - ``penalty`` 转发到新接口 ``compute``；
        - ``requirements`` / ``expects_units`` 透传，便于在旧框架里仍然
          看到「这个约束还缺什么前置条件」。
        """

        def __init__(self, inner: PhysicsConstraint) -> None:
            super().__init__(weight=float(inner.weight))
            self.inner = inner
            self.name = inner.name
            self.enabled = inner.enabled

        def penalty(self, outputs: Tensor, batch: Mapping[str, Any]) -> Tensor:
            """转发到新接口 ``compute``。"""
            return self.inner.compute(outputs, batch)

        def compute(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
            """新接口直通（便于把适配器当成新约束继续用）。"""
            return self.inner.compute(prediction, batch)

        @property
        def active(self) -> bool:
            return bool(self.enabled) and self.weight != 0.0

        def requirements(self) -> list[str]:
            """透传内层约束的前置条件。"""
            return list(type(self.inner).requirements())

        def expects_units(self) -> dict[str, str]:
            """透传内层约束的单位口径声明。"""
            return self.inner.expects_units()

        def describe(self) -> dict[str, Any]:
            """透传内层约束的元信息（标注为 legacy 适配器）。"""
            info = self.inner.describe()
            info["adapter"] = "legacy"
            return info

    LegacyConstraintAdapter.__name__ = "LegacyConstraintAdapter"
    LegacyConstraintAdapter.__qualname__ = "LegacyConstraintAdapter"
    _ADAPTER_CLASSES[base] = LegacyConstraintAdapter
    return LegacyConstraintAdapter


def adapt_for_legacy(constraint: PhysicsConstraint) -> Any:
    """把新约束包装成旧接口实例（可直接传给 ``training.losses.CompositeLoss``）。

    Parameters
    ----------
    constraint : PhysicsConstraint
        ``physics`` 包中的约束实例。

    Returns
    -------
    Any
        旧基类（``training.losses.PhysicsConstraint``）的子类实例。
    """
    if not isinstance(constraint, PhysicsConstraint):
        raise TypeError(f"期望 PhysicsConstraint 实例，收到 {type(constraint)!r}")
    adapter_cls = _adapter_class(legacy_base_class())
    return adapter_cls(constraint)


def to_legacy_list(constraints: Sequence[PhysicsConstraint]) -> list[Any]:
    """批量包装为旧接口实例列表。"""
    return [adapt_for_legacy(c) for c in constraints]
