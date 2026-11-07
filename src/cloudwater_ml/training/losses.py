"""损失函数与物理约束（统一接口）。

设计要点：

- :func:`get_loss` 按配置名返回 torch 模块（mse / mae / huber）；
- 物理约束统一为 :class:`PhysicsConstraint`，``penalty(outputs, batch)`` 里
  ``batch`` 至少含 ``Y``（真值），供需要真值的约束使用；
- :class:`CompositeLoss` = 监督损失 + Σ weight × penalty；constraints 为空时
  退化为纯监督（默认配置即如此——未验证的物理假设不强制写入模型）；
- :class:`CWPConsistencyPenalty` 构造即抛 ``NotImplementedError``：CLWC/CIWC
  为 kg/kg，须结合空气密度换算后才能积分成 CWP（kg/m²），单位口径未定前不启用。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn

from ..utils.exceptions import PhysicsConstraintError

__all__ = [
    "get_loss",
    "PhysicsConstraint",
    "NonNegativityPenalty",
    "CWPConsistencyPenalty",
    "build_constraints",
    "CompositeLoss",
]


def get_loss(name: str, cfg: dict[str, Any] | None = None) -> nn.Module:
    """按名称构造监督损失：``mse`` | ``mae`` | ``huber``（huber_delta 可配）。"""
    cfg = cfg or {}
    key = str(name or "mse").lower()
    if key == "mse":
        return nn.MSELoss()
    if key == "mae":
        return nn.L1Loss()
    if key == "huber":
        delta = float(cfg.get("huber_delta", 1.0))
        return nn.HuberLoss(delta=delta)
    raise ValueError(f"未知损失 {name!r}，可选: mse | mae | huber")


# ----------------------------------------------------------------------
# 物理约束
# ----------------------------------------------------------------------
class PhysicsConstraint(ABC):
    """物理约束基类：``penalty(outputs, batch)`` 返回标量惩罚项。"""

    name: str = "base"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    @abstractmethod
    def penalty(self, outputs: Tensor, batch: dict[str, Tensor]) -> Tensor:
        """对一个 batch 的预测施加惩罚（越大越违例）。"""


class NonNegativityPenalty(PhysicsConstraint):
    """非负约束：对负预测惩罚 ``mean(relu(-y))``（CLWC/CIWC/CWP 等非负量）。"""

    name = "nonnegativity"

    def penalty(self, outputs: Tensor, batch: dict[str, Tensor]) -> Tensor:
        return torch.relu(-outputs).mean()


class CWPConsistencyPenalty(PhysicsConstraint):
    """CWP 与柱积分一致性惩罚 —— **预留，构造即抛 NotImplementedError**。

    原因：CLWC/CIWC 单位为 kg/kg，须结合空气密度换算后才能积分成 CWP
    （kg/m²）；单位口径未定、换算路径未验证前不启用，防止把未验证的物理
    假设写进损失。
    """

    name = "cwp_consistency"

    def __init__(self, weight: float = 1.0) -> None:
        raise NotImplementedError(
            "CWPConsistencyPenalty 为第二阶段预留：单位口径未定"
            "（CLWC 为 kg/kg 时须结合空气密度换算后才能积分成 CWP kg/m²），"
            "未验证前不启用。"
        )

    def penalty(self, outputs: Tensor, batch: dict[str, Tensor]) -> Tensor:  # pragma: no cover - 构造即抛
        raise NotImplementedError("CWPConsistencyPenalty 未实现")


def build_constraints(physics_cfg: dict[str, Any] | None) -> list[PhysicsConstraint]:
    """按 ``physics`` 配置构造约束列表；接受 physics 段或含 physics 键的完整配置。

    **优先使用** ``cloudwater_ml.physics`` 的新约束体系（单位感知、硬/软分离、
    预留约束构造即抛）；新体系不可用时回退到本模块的旧实现（向后兼容）。

    配置口径：

    - ``constraints``：约束名列表（见 :func:`cloudwater_ml.physics.build_constraint`）；
    - ``weights``：逐约束权重映射 ``{名称: λ}``（推荐）；
    - ``nonnegativity_weight``：旧键，非负性权重（默认 0.0 = 关闭）。
    """
    if not physics_cfg:
        return []
    cfg = physics_cfg
    if "constraints" not in cfg and isinstance(cfg.get("physics"), dict):
        cfg = cfg["physics"]  # 传入的是完整配置而非 physics 段
    names = list(cfg.get("constraints") or [])
    if not names:
        return []
    weights = dict(cfg.get("weights") or {})
    default_weight = float(cfg.get("nonnegativity_weight", 0.0) or 0.0)

    out: list[PhysicsConstraint] = []
    try:
        from ..physics import build_constraint, to_legacy_list

        built = []
        for raw in names:
            key = str(raw).lower()
            lam = float(weights.get(key, weights.get(raw, default_weight)))
            built.append(build_constraint(key, weight=lam))
        out.extend(to_legacy_list(built))
    except ImportError:
        # 新体系不可用 → 回退旧实现（保证框架仍可运行）
        out = []
        for raw in names:
            key = str(raw).lower()
            lam = float(weights.get(key, weights.get(raw, default_weight)))
            if key in ("nonnegativity", "nonneg", "non_negative"):
                out.append(NonNegativityPenalty(lam))
            elif key in ("cwp_consistency", "cwp"):
                out.append(CWPConsistencyPenalty(lam))  # 构造即抛（预留）
            else:
                raise ValueError(
                    f"未知物理约束 {raw!r}，可选: nonnegativity | cwp_consistency"
                ) from None
    if out and all(c.weight == 0.0 for c in out):
        # 权重为 0 ⇒ 约束完全不起作用。必须显式提示，否则等于「开了个寂寞」
        import logging

        logging.getLogger("cloudwater_ml.training.losses").warning(
            "物理约束已列出但权重全为 0，实际不会对损失产生任何影响：%s",
            [c.name for c in out],
        )
    return out


class CompositeLoss(nn.Module):
    """监督损失 + Σ weight × penalty；constraints 为空即纯监督损失。"""

    def __init__(
        self,
        supervised: nn.Module,
        constraints: list[PhysicsConstraint] | None = None,
        physical_transform: Any | None = None,
    ) -> None:
        super().__init__()
        self.supervised = supervised
        self.constraints: list[PhysicsConstraint] = list(constraints or [])
        #: 可选：把「标准化空间」的预测还原到物理单位后再算物理约束。
        #: 不给的话，约束会作用在 z-score 上（例如把「低于均值」误判成「负云水」），
        #: 那是**科学上错误**的。传入的 callable 接受/返回 torch.Tensor。
        self.physical_transform = physical_transform

    def _physical(self, outputs: Tensor) -> Tensor:
        if self.physical_transform is None:
            return outputs
        try:
            return self.physical_transform(outputs)
        except Exception as exc:  # pragma: no cover - 变换失败必须显式报错
            raise PhysicsConstraintError(
                f"物理单位还原失败，无法计算物理约束: {exc}",
                hint="检查 training 配置里的物理变换是否与目标的 scaler 匹配",
            ) from exc

    def forward(self, outputs: Tensor, batch: dict[str, Tensor]) -> Tensor:
        target = batch["Y"] if isinstance(batch, dict) else batch  # type: ignore[assignment]
        loss = self.supervised(outputs, target)
        if self.constraints:
            phys = self._physical(outputs)
            for c in self.constraints:
                loss = loss + c.weight * c.penalty(phys, batch)
        return loss

    def component_tensors(self, outputs: Tensor, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """返回各损失分量的 **设备张量**（未加权原始量纲）。

        放在设备上返回，是为了让 Trainer 一个 epoch 只做一次 host 同步
        （每 batch 都 ``.item()`` 会让 GPU 管线被反复打断）。
        """
        with torch.no_grad():
            target = batch["Y"] if isinstance(batch, dict) else batch  # type: ignore[assignment]
            out: dict[str, Tensor] = {"supervised": self.supervised(outputs, target).detach()}
            if self.constraints:
                phys = self._physical(outputs)
                for c in self.constraints:
                    out[c.name] = c.penalty(phys, batch).detach()
        return out

    def component_values(self, outputs: Tensor, batch: dict[str, Tensor]) -> dict[str, float]:
        """返回各损失分量的数值（未加权的原始量纲，便于逐项诊断）。

        - ``supervised``：监督损失；
        - ``<约束名>``：每个物理约束的惩罚值（作用在**物理单位**上）。

        用于「记录每个损失分量」的要求：训练日志与实验记录里能看到
        监督项与各物理项各占多少，而不是只有一个总数。
        """
        return {k: float(v) for k, v in self.component_tensors(outputs, batch).items()}
