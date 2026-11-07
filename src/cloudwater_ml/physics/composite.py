"""物理约束组合：``L_total = L_data + Σ λ_i · L_physics,i``。

约定
----
- λ_i **独立可配**：每个约束自带 ``weight``，默认 0（= 关闭）。
  未验证的物理假设不会因为「组合器存在」而被动生效。
- **硬约束不进损失**：硬约束以投影 / 重参数化方式生效
  （:meth:`PhysicsLoss.project`），其 ``compute`` 恒为 0。
- **逐分量记录**：:meth:`PhysicsLoss.component_values` 给出监督项与每个
  物理项的独立数值（与 ``training/losses.py::CompositeLoss.component_values``
  的做法一致，便于日志与实验记录逐项诊断）。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from .base import HardConstraint, PhysicsConstraint

__all__ = ["PhysicsLoss"]


class PhysicsLoss(nn.Module):
    """监督损失 + Σ λ_i · 物理惩罚（硬约束以投影生效，不进损失）。

    Parameters
    ----------
    supervised : nn.Module
        监督损失模块（如 ``nn.MSELoss()``）。
    constraints : Sequence[PhysicsConstraint]
        软约束列表（每个约束自带独立权重 λ_i）。
    hard_constraints : Sequence[HardConstraint]
        硬约束列表（投影 / 重参数化）。
    apply_projection : bool
        为 True 时，``forward`` 先对预测做硬约束投影再算损失
        （默认 False —— 投影通常加在模型输出侧，损失侧保持原样）。
    """

    def __init__(
        self,
        supervised: nn.Module,
        constraints: Sequence[PhysicsConstraint] | None = None,
        hard_constraints: Sequence[HardConstraint] | None = None,
        apply_projection: bool = False,
    ) -> None:
        super().__init__()
        self.supervised = supervised
        self.constraints: list[PhysicsConstraint] = list(constraints or [])
        self.hard_constraints: list[HardConstraint] = list(hard_constraints or [])
        self.apply_projection = bool(apply_projection)

    # ------------------------------------------------------------------
    @staticmethod
    def _target(batch: Mapping[str, Any] | Tensor) -> Tensor:
        """取监督目标（batch 为字典时取 ``Y``，否则视为目标本身）。"""
        if isinstance(batch, Mapping):
            return batch.get("Y", batch)  # type: ignore[return-value]
        return batch

    # ------------------------------------------------------------------
    def project(self, outputs: Tensor) -> Tensor:
        """依次施加硬约束投影，返回满足硬约束的预测。"""
        x = outputs
        for hard in self.hard_constraints:
            if not hard.enabled:
                continue
            x = hard.project(x)
        return x

    def forward(self, outputs: Tensor, batch: Mapping[str, Any] | Tensor) -> Tensor:
        """返回 ``L_data + Σ_{启用且 λ_i≠0} λ_i · L_physics,i``。"""
        target = self._target(batch)
        x = self.project(outputs) if self.apply_projection else outputs
        loss = self.supervised(x, target)
        for constraint in self.constraints:
            if not constraint.enabled or constraint.weight == 0.0:
                continue
            loss = loss + constraint.weight * constraint.compute(x, batch)  # type: ignore[arg-type]
        return loss

    # ------------------------------------------------------------------
    def component_values(self, outputs: Tensor, batch: Mapping[str, Any] | Tensor) -> dict[str, float]:
        """各损失分量的独立数值（未加权原始量纲，便于逐项诊断）。

        - ``supervised``：监督损失；
        - ``<约束名>``：每个**已启用**软约束的惩罚值（权重为 0 也照样记录，
          便于观察「打开某个约束会带来多大的量级」）；
        - 硬约束不在此处出现（它们不进损失），其投影前违例量见
          :meth:`hard_violation_values`。
        """
        target = self._target(batch)
        values: dict[str, float] = {
            "supervised": float(self.supervised(outputs, target).detach())
        }
        for constraint in self.constraints:
            if not constraint.enabled:
                continue
            values[constraint.name] = float(constraint.compute(outputs, batch).detach())  # type: ignore[arg-type]
        return values

    def hard_violation_values(self, outputs: Tensor) -> dict[str, float]:
        """硬约束投影**前**的违例量（诊断用，不进损失）。"""
        values: dict[str, float] = {}
        for hard in self.hard_constraints:
            if not hard.enabled:
                continue
            values[hard.name] = float(hard.violation(outputs).detach())
        return values

    # ------------------------------------------------------------------
    @property
    def active_constraints(self) -> list[PhysicsConstraint]:
        """参与损失的约束（启用且权重非 0）。"""
        return [c for c in self.constraints if c.active]

    def describe(self) -> list[dict[str, Any]]:
        """约束元信息列表（写日志 / 实验记录用）。"""
        return [c.describe() for c in self.constraints] + [
            h.describe() for h in self.hard_constraints
        ]
