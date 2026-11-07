"""物理约束实现（可用两级：可真实计算 / 预留未实现）。

**可用（可真实计算）**

- :class:`NonNegativityConstraint`（软）：CLWC/CIWC/CWP 物理上非负，
  惩罚 ``mean(relu(-y))``；
- :class:`UnitConsistencyConstraint`（软）：核对输入声明的单位与登记口径是否
  相符；不符直接抛 :class:`~cloudwater_ml.physics.errors.UnitMismatchError`
  （不折算成惩罚值 —— 单位搞错属于配置/数据错误，必须显式失败）；
- :class:`NonNegativityClamp` / :class:`NonNegativitySoftplus`（硬）：以截断
  投影 / softplus 重参数化保证非负，**不进损失**。

**预留（构造即抛 NotImplementedError）**

- :class:`CWPIntegralConsistencyConstraint`
- :class:`WaterBudgetConstraint`
- :class:`VerticalProfileStructureConstraint`
- :class:`AtmosphericWaterCycleConstraint`

预留项的 :meth:`requirements` 列出「还需要哪些输入才可启用」，
:meth:`required_units` 列出期望的单位口径。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor

from .base import HardConstraint, PhysicsConstraint, SoftConstraint
from .errors import ReservedConstraintError, UnitMismatchError
from .transforms import clamp_projection, softplus
from .units import (
    assert_unit_family,
    batch_units,
    classify_unit_family,
    expected_families_for,
    schema_units,
)

__all__ = [
    "check_variable_unit",
    "NonNegativityConstraint",
    "UnitConsistencyConstraint",
    "NonNegativityClamp",
    "NonNegativitySoftplus",
    "CWPIntegralConsistencyConstraint",
    "WaterBudgetConstraint",
    "VerticalProfileStructureConstraint",
    "AtmosphericWaterCycleConstraint",
    "RESERVED_CONSTRAINTS",
    "build_constraint",
]


def check_variable_unit(
    variable: str,
    actual: str,
    expected: str | None = None,
) -> None:
    """校验「变量声明的单位」与「期望口径」是否相符。

    允许的口径族取自 ``data/schema.py`` 的登记：
    ``clwc``/``ciwc`` 允许「比含水量（无量纲比值）」与「质量密度」两种表达，
    ``cwp`` 只允许面质量积分 ``kg/m2``。

    Raises
    ------
    UnitMismatchError
        口径不符（例如把 ``kg/m2`` 当作云水含量单位）。
    """
    allowed = expected_families_for(variable)
    if expected is not None:
        allowed.add(classify_unit_family(expected))
    if not allowed:
        return
    assert_unit_family(actual, allowed, context=f"变量 {variable!r}")


def _target_names(batch: Mapping[str, Any] | None) -> list[str]:
    """从 batch 取目标变量名列表（没有则空列表）。"""
    if not isinstance(batch, Mapping):
        return []
    names = batch.get("target_names")
    if not names:
        return []
    return [str(n) for n in names]


def _select_variable(prediction: Tensor, batch: Mapping[str, Any], variable: str) -> Tensor:
    """按目标变量名取出预测中对应的列（取不到则退回整体）。"""
    names = _target_names(batch)
    if variable in names:
        idx = names.index(variable)
        if prediction.ndim >= 2 and idx < prediction.shape[-1]:
            return prediction[..., idx]
    return prediction


class NonNegativityConstraint(SoftConstraint):
    """非负约束（软惩罚）：``mean(relu(-y))``。

    CLWC / CIWC / CWP 是非负量（云水含量与云水路径不可能为负）。
    这是**可证明**的物理性质，因此同时提供硬约束版本
    （:class:`NonNegativityClamp` / :class:`NonNegativitySoftplus`）。

    单位感知：若 batch 声明了该变量的单位，会核对其口径是否属于
    「云水含量」族（``kg/kg`` 或 ``kg/m3``）；口径不符直接抛错。
    ``require_declared_unit=True`` 时，未声明单位也算错误
    （口径不明时不得默认启用）。
    """

    name = "nonnegativity"

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = True,
        variable: str = "clwc",
        require_declared_unit: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        weight : float
            权重 λ（默认 0 = 关闭）。
        enabled : bool
            开关。
        variable : str
            目标变量名（如 ``clwc`` / ``ciwc`` / ``cwp``）；batch 提供
            ``target_names`` 时只惩罚该变量对应的列。
        require_declared_unit : bool
            是否要求 batch 必须声明该变量的单位。
        """
        super().__init__(weight=weight, enabled=enabled)
        self.variable = variable
        self.require_declared_unit = bool(require_declared_unit)

    def expects_units(self) -> dict[str, str]:
        """期望单位（登记口径；实际允许同族的两种表达，见 :func:`check_variable_unit`）。"""
        expected = schema_units(self.variable)
        return {self.variable: expected} if expected else {}

    def compute(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """返回 ``mean(relu(-y))``（满足非负时为 0）。"""
        self._check_units(batch)
        x = _select_variable(prediction, batch, self.variable)
        return torch.relu(-x).mean()

    def _check_units(self, batch: Mapping[str, Any]) -> None:
        units = batch_units(batch)
        actual = units.get(self.variable)
        if actual is None:
            if self.require_declared_unit:
                raise UnitMismatchError(
                    f"变量 {self.variable!r} 未声明单位 —— 单位感知的物理约束要求"
                    "显式声明单位口径，无法确认时不得默认启用"
                )
            return
        check_variable_unit(self.variable, actual, schema_units(self.variable))


class UnitConsistencyConstraint(SoftConstraint):
    """单位一致性检查（可用）：核对 batch 声明的单位与登记口径是否相符。

    相符 → 返回 0 标量；不符 → 抛 :class:`UnitMismatchError`。
    刻意**不**把「单位搞错了」折算成惩罚值：那属于配置/数据层面的错误，
    混进损失只会掩盖问题。
    """

    name = "unit_consistency"

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = True,
        variables: Iterable[str] | None = None,
        expected_units: Mapping[str, str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        weight : float
            权重 λ（默认 0 = 关闭）。
        enabled : bool
            开关。
        variables : Iterable[str] | None
            只检查这些变量（默认检查 batch 中声明了单位的全部变量）。
        expected_units : Mapping[str, str] | None
            覆盖登记口径的期望单位（变量 → 单位）。
        """
        super().__init__(weight=weight, enabled=enabled)
        self.variables = list(variables) if variables is not None else None
        self.expected_units = dict(expected_units or {})

    def expects_units(self) -> dict[str, str]:
        if self.variables is not None:
            return {v: self.expected_units.get(v) or schema_units(v) or "" for v in self.variables}
        return dict(self.expected_units)

    def compute(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """单位全部相符时返回 0 标量；否则抛错。"""
        units = batch_units(batch)
        targets = self.variables if self.variables is not None else list(units.keys())
        checked = 0
        for variable in targets:
            actual = units.get(variable)
            if actual is None:
                continue
            expected = self.expected_units.get(variable) or schema_units(variable)
            check_variable_unit(variable, actual, expected)
            checked += 1
        if checked == 0:
            # 没有可比对的单位 → 不做任何判断（不报错，也不谎称通过）
            return prediction.new_zeros(())
        return prediction.new_zeros(())


class NonNegativityClamp(HardConstraint):
    """硬约束（截断投影）：把负预测截断到 ``min_value``，**不进损失**。

    投影由调用方在模型输出侧施加（见
    :meth:`cloudwater_ml.physics.composite.PhysicsLoss.project`）。
    """

    name = "nonnegativity_clamp"

    def __init__(
        self,
        enabled: bool = True,
        min_value: float = 0.0,
        variable: str = "clwc",
        weight: float = 0.0,
    ) -> None:
        super().__init__(weight=weight, enabled=enabled)
        self.min_value = float(min_value)
        self.variable = variable

    def expects_units(self) -> dict[str, str]:
        expected = schema_units(self.variable)
        return {self.variable: expected} if expected else {}

    def project(self, prediction: Tensor) -> Tensor:
        """返回 ``clamp(prediction, min=min_value)``。"""
        return clamp_projection(prediction, self.min_value)

    def violation(self, prediction: Tensor) -> Tensor:
        """投影前的违例量 ``mean(relu(min_value - y))``（诊断用）。"""
        return torch.relu(self.min_value - prediction).mean()


class NonNegativitySoftplus(HardConstraint):
    """硬约束（softplus 重参数化）：输出恒 ≥ 0，**不进损失**。

    适合直接作为模型输出层：网络输出 raw 值，再经 softplus 变成非负量。
    """

    name = "nonnegativity_softplus"

    def __init__(
        self,
        enabled: bool = True,
        beta: float = 1.0,
        threshold: float = 20.0,
        variable: str = "clwc",
        weight: float = 0.0,
    ) -> None:
        super().__init__(weight=weight, enabled=enabled)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.variable = variable

    def expects_units(self) -> dict[str, str]:
        expected = schema_units(self.variable)
        return {self.variable: expected} if expected else {}

    def project(self, prediction: Tensor) -> Tensor:
        """返回 ``softplus(prediction)``。"""
        return softplus(prediction, beta=self.beta, threshold=self.threshold)

    def violation(self, prediction: Tensor) -> Tensor:
        """投影前的违例量 ``mean(relu(-y))``（诊断用）。"""
        return torch.relu(-prediction).mean()


# ----------------------------------------------------------------------
# 预留约束：构造即抛 NotImplementedError
# ----------------------------------------------------------------------
class CWPIntegralConsistencyConstraint(SoftConstraint):
    """CWP 与柱积分一致性 —— **预留，构造即抛 NotImplementedError**。

    为什么预留
    ----------
    柱积分公式取决于云水含量的单位口径，两种口径的结果**不同**：

    - 比含水量口径（``kg/kg``，``data/schema.py`` 当前登记口径）::

          CWP = ∫ ρ_air(z) · CLWC(z) · dz      [kg/m2]

    - 质量密度口径（``kg/m3``）::

          CWP = ∫ CLWC(z) · dz                 [kg/m2]

    用错口径不会在数值上明显报错，只会整体偏一个 ρ_air 量级（约 1 kg/m3
    量级）的因子。因此在下列前置条件确认前，本约束**不允许**进入损失。

    公式层面可显式计算的积分工具见
    :func:`cloudwater_ml.physics.integrals.cwp_integral`（要求调用方显式
    声明口径并提供该口径所需的全部输入）；启用所需前置条件见
    :meth:`requirements`。
    """

    name = "cwp_integral_consistency"

    def __init__(self, weight: float = 0.0, enabled: bool = True, **_: Any) -> None:
        raise ReservedConstraintError(
            "CWPIntegralConsistencyConstraint 为预留约束：单位口径与前置输入未确认前"
            "不启用（防止把未验证的物理假设写进损失）。所需前置条件: "
            + "；".join(type(self).requirements())
        )

    @classmethod
    def requirements(cls) -> list[str]:
        """启用所需的前置条件（缺一不可）。"""
        return [
            "CLWC/CIWC 的单位口径（比含水量 kg/kg 或质量密度 kg/m3）需人工确认",
            "气压层坐标或各层几何厚度 Δz（m）",
            "空气密度廓线 ρ_air(z)（kg/m3）——比含水量口径下必需",
            "CWP 的相态口径（仅液水 / 液水+冰水）需确认",
            "积分离散化方式（梯形 / 中点 / 辛普森）与残差容差",
            "柱体边界（单柱 / 网格单元平均）与地形高度处理方式",
        ]

    @classmethod
    def required_units(cls) -> dict[str, str]:
        """期望的输入单位口径。"""
        return {
            "clwc": "kg kg-1 或 kg m-3",
            "ciwc": "kg kg-1 或 kg m-3",
            "cwp": "kg m-2",
            "air_density": "kg m-3",
            "layer_thickness": "m",
        }

    def compute(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """未实现（构造即抛，此处仅为满足抽象接口）。"""
        raise NotImplementedError("CWPIntegralConsistencyConstraint 未实现")


class WaterBudgetConstraint(SoftConstraint):
    """水量守恒 —— **预留，构造即抛 NotImplementedError**。

    大气（柱）水量守恒的闭合方程涉及多个通量项与储量项，**具体形式与
    符号约定必须先有文献/数据依据**；在依据确认前，本约束不进入损失，
    也不在代码里写死任何一条闭合式（避免把未验证的物理结论固化成假设）。
    """

    name = "water_budget"

    def __init__(self, weight: float = 0.0, enabled: bool = True, **_: Any) -> None:
        raise ReservedConstraintError(
            "WaterBudgetConstraint 为预留约束：闭合方程的通量项与符号约定未验证前"
            "不启用。所需前置条件: " + "；".join(type(self).requirements())
        )

    @classmethod
    def requirements(cls) -> list[str]:
        """启用所需的前置条件（缺一不可）。"""
        return [
            "守恒方程的具体形式与符号约定（需文献/数据依据，本模块不预设）",
            "研究域/柱体的储水量定义（水汽 + 云水 + 降水）与时间步长",
            "降水率 P 的单位口径（mm/s 或 kg m-2 s-1）",
            "蒸发率 E 的单位口径",
            "水汽辐合/辐散通量（水平 + 垂直）",
            "侧边界通量（开边界 / 闭合边界）",
            "闭合残差容差与检验的时间窗",
        ]

    @classmethod
    def required_units(cls) -> dict[str, str]:
        """期望的输入单位口径。"""
        return {
            "precipitation_rate": "kg m-2 s-1",
            "evaporation_rate": "kg m-2 s-1",
            "moisture_flux": "kg m-2 s-1",
            "storage": "kg m-2",
        }

    def compute(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """未实现（构造即抛，此处仅为满足抽象接口）。"""
        raise NotImplementedError("WaterBudgetConstraint 未实现")


class VerticalProfileStructureConstraint(SoftConstraint):
    """垂直廓线结构 —— **预留，构造即抛 NotImplementedError**。

    云水含量的垂直分布受热力学与微物理约束（如云底/云顶位置、云量与含水
    量的关系）。这些**具体判据**依赖数据源与分辨率，需先确认判据形式，
    因此本模块不预设任何判据公式。
    """

    name = "vertical_profile_structure"

    def __init__(self, weight: float = 0.0, enabled: bool = True, **_: Any) -> None:
        raise ReservedConstraintError(
            "VerticalProfileStructureConstraint 为预留约束：廓线结构判据未确认前不启用。"
            "所需前置条件: " + "；".join(type(self).requirements())
        )

    @classmethod
    def requirements(cls) -> list[str]:
        """启用所需的前置条件（缺一不可）。"""
        return [
            "气压层坐标（hPa）与各层厚度",
            "温度廓线与露点廓线（用于 LCL 判定）",
            "对流层顶高度的诊断方法",
            "云量/云分数廓线（0-1）及其单位口径",
            "廓线结构判据的具体形式（需依据确认，本模块不预设）",
        ]

    @classmethod
    def required_units(cls) -> dict[str, str]:
        """期望的输入单位口径。"""
        return {
            "pressure": "hPa",
            "temperature": "K",
            "cloud_fraction": "1",
            "layer_thickness": "m",
        }

    def compute(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """未实现（构造即抛，此处仅为满足抽象接口）。"""
        raise NotImplementedError("VerticalProfileStructureConstraint 未实现")


class AtmosphericWaterCycleConstraint(SoftConstraint):
    """大气水循环 —— **预留，构造即抛 NotImplementedError**。

    水循环相态转换（升华/凝华/冻结/融化）与水分输送构成闭合循环，但**各项
    的观测口径与归并方式**必须先确认；在确认前本约束不启用，代码中也不
    写死任何收支关系。
    """

    name = "atmospheric_water_cycle"

    def __init__(self, weight: float = 0.0, enabled: bool = True, **_: Any) -> None:
        raise ReservedConstraintError(
            "AtmosphericWaterCycleConstraint 为预留约束：相态转换收支口径未确认前"
            "不启用。所需前置条件: " + "；".join(type(self).requirements())
        )

    @classmethod
    def requirements(cls) -> list[str]:
        """启用所需的前置条件（缺一不可）。"""
        return [
            "相态转换收支项（升华/凝华/冻结/融化）的观测或再分析来源与口径",
            "水汽输送通量与水汽储量的时间序列",
            "降水/蒸发/蒸腾的来源划分口径",
            "循环闭合检验的时间窗与残差容差",
            "各收支项的单位口径与符号约定",
        ]

    @classmethod
    def required_units(cls) -> dict[str, str]:
        """期望的输入单位口径。"""
        return {
            "phase_conversion_rate": "kg m-2 s-1",
            "water_vapour_flux": "kg m-2 s-1",
            "water_vapour_storage": "kg m-2",
            "precipitation_rate": "kg m-2 s-1",
        }

    def compute(self, prediction: Tensor, batch: Mapping[str, Any]) -> Tensor:
        """未实现（构造即抛，此处仅为满足抽象接口）。"""
        raise NotImplementedError("AtmosphericWaterCycleConstraint 未实现")


#: 所有预留约束（构造即抛）
RESERVED_CONSTRAINTS: tuple[type[PhysicsConstraint], ...] = (
    CWPIntegralConsistencyConstraint,
    WaterBudgetConstraint,
    VerticalProfileStructureConstraint,
    AtmosphericWaterCycleConstraint,
)


def build_constraint(name: str, **kwargs: Any) -> PhysicsConstraint:
    """按名称构造约束。

    预留约束在这里同样会抛错（它们本就未实现）—— 早失败好过把未验证的
    物理假设写进损失。
    """
    key = str(name).lower()
    if key in ("nonnegativity", "nonneg", "non_negative"):
        return NonNegativityConstraint(**kwargs)
    if key in ("unit_consistency", "units"):
        return UnitConsistencyConstraint(**kwargs)
    if key in ("nonnegativity_clamp", "clamp"):
        return NonNegativityClamp(**kwargs)
    if key in ("nonnegativity_softplus", "softplus"):
        return NonNegativitySoftplus(**kwargs)
    if key in ("cwp_integral_consistency", "cwp_consistency", "cwp"):
        return CWPIntegralConsistencyConstraint(**kwargs)
    if key in ("water_budget",):
        return WaterBudgetConstraint(**kwargs)
    if key in ("vertical_profile_structure", "vertical_profile"):
        return VerticalProfileStructureConstraint(**kwargs)
    if key in ("atmospheric_water_cycle", "water_cycle"):
        return AtmosphericWaterCycleConstraint(**kwargs)
    raise ValueError(f"未知物理约束 {name!r}")
