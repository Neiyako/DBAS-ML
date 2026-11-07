"""M5 物理约束体系（CloudWater ML Platform v2.0）。

设计原则
--------
1. **未验证的物理假设默认关闭**：约束权重默认 0（= 关闭）；预留约束
   **构造即抛** ``NotImplementedError``，防止把未验证的物理假设写进损失。
2. **单位敏感**：单位口径不符时**报错**，绝不默默算错；云水含量的两种口径
   （比含水量 ``kg/kg`` 与质量密度 ``kg/m3``）在积分公式里被显式区分。
3. **硬约束 / 软约束分离**：硬约束以投影或重参数化生效，不进损失；
   软约束以权重 λ 进入 ``L_total = L_data + Σ λ_i · L_physics,i``。
4. **逐分量记录**：组合器给出监督项与每个约束的独立数值，便于日志与实验
   记录逐项诊断。
5. **不改动既有代码**：与 ``training/losses.py`` 通过 ``physics/bridge.py``
   做兼容（旧文件保持原样）。
"""
from __future__ import annotations

from .base import ConstraintKind, HardConstraint, PhysicsConstraint, SoftConstraint
from .bridge import adapt_for_legacy, legacy_base_class, to_legacy_list
from .composite import PhysicsLoss
from .constraints import (
    RESERVED_CONSTRAINTS,
    AtmosphericWaterCycleConstraint,
    CWPIntegralConsistencyConstraint,
    NonNegativityClamp,
    NonNegativityConstraint,
    NonNegativitySoftplus,
    UnitConsistencyConstraint,
    VerticalProfileStructureConstraint,
    WaterBudgetConstraint,
    build_constraint,
    check_variable_unit,
)
from .errors import PhysicsNotReadyError, ReservedConstraintError, UnitMismatchError
from .integrals import (
    CONVENTION_MASS_DENSITY,
    CONVENTION_MASS_FRACTION,
    convert_tensor,
    cwp_integral,
    cwp_residual,
    water_content_convention,
)
from .transforms import clamp_projection, softplus
from .units import (
    FAMILY_DIMENSIONLESS,
    FAMILY_MASS_DENSITY,
    FAMILY_PATH_MASS,
    WATER_CONTENT_FAMILIES,
    Unit,
    assert_unit_family,
    batch_units,
    classify_unit_family,
    convert,
    expected_families_for,
    format_unit,
    is_mass_density,
    is_mass_fraction,
    parse_unit,
    require_compatible,
    schema_units,
    units_compatible,
)
from .validator import (
    STATUS_FAIL,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_IMPLEMENTED,
    STATUS_PASS,
    CheckResult,
    PhysicsValidator,
    format_report_markdown,
)

__all__ = [
    # 基础接口
    "PhysicsConstraint",
    "SoftConstraint",
    "HardConstraint",
    "ConstraintKind",
    # 约束实现
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
    "check_variable_unit",
    # 组合
    "PhysicsLoss",
    # 兼容桥
    "adapt_for_legacy",
    "to_legacy_list",
    "legacy_base_class",
    # 单位
    "Unit",
    "parse_unit",
    "format_unit",
    "units_compatible",
    "require_compatible",
    "convert",
    "convert_tensor",
    "classify_unit_family",
    "assert_unit_family",
    "batch_units",
    "schema_units",
    "expected_families_for",
    "is_mass_fraction",
    "is_mass_density",
    "FAMILY_DIMENSIONLESS",
    "FAMILY_MASS_DENSITY",
    "FAMILY_PATH_MASS",
    "WATER_CONTENT_FAMILIES",
    # 积分
    "cwp_integral",
    "cwp_residual",
    "water_content_convention",
    "CONVENTION_MASS_FRACTION",
    "CONVENTION_MASS_DENSITY",
    # 变换
    "softplus",
    "clamp_projection",
    # 验证器
    "PhysicsValidator",
    "CheckResult",
    "format_report_markdown",
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_NOT_APPLICABLE",
    "STATUS_NOT_IMPLEMENTED",
    # 异常
    "UnitMismatchError",
    "PhysicsNotReadyError",
    "ReservedConstraintError",
]
