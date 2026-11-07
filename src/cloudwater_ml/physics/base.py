"""物理约束的统一接口（可插拔）。

统一接口
--------
``compute(prediction, batch) -> Tensor``：返回标量惩罚（**越大越违例**）。
每个约束都带：

- ``name``：名称（用于逐分量记录与日志）；
- ``weight``：权重 λ（**默认 0**，即默认关闭）；
- ``enabled``：开关；
- ``expects_units()``：期望的输入单位口径；
- ``requirements()``：启用该约束所需的前置条件。

硬约束 / 软约束
---------------
- **软约束**（:class:`SoftConstraint`）：惩罚进损失，
  ``L_total = L_data + Σ λ_i · L_physics,i``。
- **硬约束**（:class:`HardConstraint`）：以**投影 / 重参数化**方式保证满足
  （如 clamp 截断、softplus 变换），**不进损失**（``compute`` 恒返回 0）。

默认权重为 0 —— 未验证的物理假设默认关闭，必须显式打开。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from torch import Tensor

__all__ = [
    "ConstraintKind",
    "PhysicsConstraint",
    "SoftConstraint",
    "HardConstraint",
]


class ConstraintKind(StrEnum):
    """约束类型：硬约束（投影生效）或软约束（进损失）。"""

    HARD = "hard"
    SOFT = "soft"


class PhysicsConstraint(ABC):
    """物理约束基类。

    子类必须实现 :meth:`compute`，返回标量惩罚（越大越违例）。
    """

    #: 约束名（逐分量记录 / 日志的 key）
    name: str = "base"
    #: 约束类型
    kind: ConstraintKind = ConstraintKind.SOFT

    def __init__(self, weight: float = 0.0, enabled: bool = True) -> None:
        """
        Parameters
        ----------
        weight : float
            权重 λ。默认 0 —— 未验证的物理假设默认关闭。
        enabled : bool
            开关。关闭后不参与损失，也不进入逐分量记录。
        """
        self.weight = float(weight)
        self.enabled = bool(enabled)

    # ------------------------------------------------------------------
    @abstractmethod
    def compute(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """对一个 batch 的预测返回标量惩罚（越大越违例）。

        Parameters
        ----------
        prediction : Tensor
            模型预测（形状由调用方约定，如 ``[batch, targets]``）。
        batch : Mapping[str, Any]
            批次数据。除张量外可携带 ``units``（变量 → 单位）与
            ``target_names``（目标变量名），供单位感知检查使用。
        """

    # ------------------------------------------------------------------
    # 元信息
    # ------------------------------------------------------------------
    @classmethod
    def requirements(cls) -> list[str]:
        """启用该约束所需的前置条件（缺一不可）。

        预留约束在此列出「还需要哪些输入（气压层坐标 / 空气密度廓线 / 单位
        口径等）才可启用」；可用约束通常返回空列表。
        """
        return []

    @classmethod
    def required_units(cls) -> dict[str, str]:
        """该约束需要的输入单位（类级，可在构造前查询）。"""
        return {}

    def expects_units(self) -> dict[str, str]:
        """本实例期望的输入单位（变量 → 单位字符串）。"""
        return dict(type(self).required_units())

    def unit_of(self, variable: str) -> str | None:
        """取某个输入变量的期望单位（未声明则 ``None``）。"""
        return self.expects_units().get(variable)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """是否参与损失（启用且权重非 0）。"""
        return bool(self.enabled) and self.weight != 0.0

    def penalty(self, outputs: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """兼容既有 ``training/losses.py`` 的接口别名（见 ``physics/bridge.py``）。"""
        return self.compute(outputs, batch)

    def __call__(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        return self.compute(prediction, batch)

    def describe(self) -> dict[str, Any]:
        """约束元信息（写日志 / 实验记录用）。"""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "weight": self.weight,
            "enabled": self.enabled,
            "expects_units": self.expects_units(),
            "requirements": list(type(self).requirements()),
        }


class SoftConstraint(PhysicsConstraint):
    """软约束：惩罚进入损失（``L_data + Σ λ_i · L_i``）。"""

    kind = ConstraintKind.SOFT


class HardConstraint(PhysicsConstraint):
    """硬约束：以投影 / 重参数化方式保证满足，**不进损失**。

    子类实现 :meth:`project`；:meth:`compute` 恒返回 0 标量。
    """

    kind = ConstraintKind.HARD

    def project(self, prediction: Tensor) -> Tensor:
        """返回满足该硬约束的预测（投影 / 重参数化）。"""
        raise NotImplementedError

    def violation(self, prediction: Tensor) -> Tensor:
        """投影**前**的违例量（诊断用，不进损失）。"""
        raise NotImplementedError

    def compute(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """硬约束不进损失：恒返回 0 标量。"""
        return prediction.new_zeros(())
