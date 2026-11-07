"""M5 物理约束体系测试。

覆盖验收点：

1. 非负约束在负预测上产生正惩罚、满足约束时为 0；
2. 权重与组合的总损失 = 监督 + Σ λ_i 项（数值断言）；
3. 硬约束与软约束的区分行为（硬约束不进入损失，以投影形式生效）；
4. 预留约束**构造即报错**（``NotImplementedError``）——防止有人误以为已实现；
5. 单位感知：单位不符时报错；``kg/kg`` 与 ``kg/m3`` 两种口径的积分结果不同
   （并写清哪个口径对本仓库数据成立）；
6. Validator 报告对未实现项输出 ``NOT_IMPLEMENTED`` 而非 ``PASS``。
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from cloudwater_ml.physics import (
    RESERVED_CONSTRAINTS,
    AtmosphericWaterCycleConstraint,
    CWPIntegralConsistencyConstraint,
    NonNegativityClamp,
    NonNegativityConstraint,
    NonNegativitySoftplus,
    PhysicsLoss,
    PhysicsValidator,
    UnitConsistencyConstraint,
    VerticalProfileStructureConstraint,
    WaterBudgetConstraint,
    adapt_for_legacy,
    classify_unit_family,
    convert,
    cwp_integral,
    is_mass_fraction,
    parse_unit,
    units_compatible,
)
from cloudwater_ml.physics.errors import (
    PhysicsNotReadyError,
    ReservedConstraintError,
    UnitMismatchError,
)
from cloudwater_ml.physics.integrals import (
    CONVENTION_MASS_DENSITY,
    CONVENTION_MASS_FRACTION,
    water_content_convention,
)

# ----------------------------------------------------------------------
# 公共样本
# ----------------------------------------------------------------------
OUTPUTS = torch.tensor([[-2.0, 1.0]])  # 第 0 列负值，第 1 列非负
TARGET = torch.tensor([[0.0, 1.0]])
MSE = float(nn.MSELoss()(OUTPUTS, TARGET))  # = 2.0
PENALTY = float(torch.relu(-OUTPUTS).mean())  # = 1.0


# ----------------------------------------------------------------------
# 1. 非负约束
# ----------------------------------------------------------------------
def test_nonnegativity_penalty_is_positive_on_negative_prediction():
    """负预测 → 正惩罚。"""
    penalty = NonNegativityConstraint(weight=1.0).compute(OUTPUTS, {"Y": TARGET})
    assert float(penalty) > 0.0
    assert float(penalty) == pytest.approx(PENALTY)


def test_nonnegativity_penalty_is_zero_when_satisfied():
    """非负预测 → 惩罚为 0。"""
    outputs = torch.tensor([[0.0, 1.0], [2.5, 3.0]])
    penalty = NonNegativityConstraint(weight=1.0).compute(outputs, {"Y": outputs})
    assert float(penalty) == pytest.approx(0.0)


def test_nonnegativity_selects_target_column():
    """batch 提供 target_names 时只惩罚对应变量列。"""
    batch = {"Y": TARGET, "target_names": ["clwc", "ciwc"]}
    penalty = NonNegativityConstraint(weight=1.0, variable="clwc").compute(OUTPUTS, batch)
    # 只看 clwc 列（值为 -2.0）→ mean(relu(2.0)) = 2.0
    assert float(penalty) == pytest.approx(2.0)

    penalty_ciwc = NonNegativityConstraint(weight=1.0, variable="ciwc").compute(OUTPUTS, batch)
    assert float(penalty_ciwc) == pytest.approx(0.0)


def test_nonnegativity_default_weight_is_zero():
    """默认权重 0 —— 未验证的物理假设默认关闭。"""
    assert NonNegativityConstraint().weight == 0.0
    assert NonNegativityConstraint().active is False
    assert NonNegativityConstraint(weight=0.5).active is True


# ----------------------------------------------------------------------
# 2. 权重与组合：L_total = L_data + Σ λ_i · L_physics,i
# ----------------------------------------------------------------------
def test_total_loss_equals_supervised_plus_weighted_penalty():
    """数值断言：总损失 = 监督 + λ·惩罚。"""
    loss_fn = PhysicsLoss(nn.MSELoss(), [NonNegativityConstraint(weight=0.5)])
    total = float(loss_fn(OUTPUTS, {"Y": TARGET}))
    assert total == pytest.approx(MSE + 0.5 * PENALTY)
    assert total == pytest.approx(2.0 + 0.5 * 1.0)


def test_weight_zero_constraint_does_not_change_loss():
    """λ=0 → 关闭，损失只剩监督项。"""
    loss_fn = PhysicsLoss(nn.MSELoss(), [NonNegativityConstraint(weight=0.0)])
    assert float(loss_fn(OUTPUTS, {"Y": TARGET})) == pytest.approx(MSE)


def test_disabled_constraint_does_not_change_loss():
    """enabled=False → 关闭。"""
    loss_fn = PhysicsLoss(nn.MSELoss(), [NonNegativityConstraint(weight=1.0, enabled=False)])
    assert float(loss_fn(OUTPUTS, {"Y": TARGET})) == pytest.approx(MSE)


def test_multiple_constraints_weights_are_independent():
    """多个约束的 λ 相互独立，逐项相加。"""
    c1 = NonNegativityConstraint(weight=0.5, variable="clwc")
    c2 = NonNegativityConstraint(weight=0.25, variable="ciwc")
    loss_fn = PhysicsLoss(nn.MSELoss(), [c1, c2])
    batch = {"Y": TARGET, "target_names": ["clwc", "ciwc"]}
    total = float(loss_fn(OUTPUTS, batch))
    assert total == pytest.approx(MSE + 0.5 * 2.0 + 0.25 * 0.0)


def test_component_values_records_each_component():
    """逐分量记录：监督项与每个约束的独立数值。"""
    loss_fn = PhysicsLoss(nn.MSELoss(), [NonNegativityConstraint(weight=0.5)])
    values = loss_fn.component_values(OUTPUTS, {"Y": TARGET})
    assert values["supervised"] == pytest.approx(MSE)
    assert values["nonnegativity"] == pytest.approx(PENALTY)


# ----------------------------------------------------------------------
# 3. 硬约束 vs 软约束
# ----------------------------------------------------------------------
def test_hard_constraint_does_not_enter_loss():
    """硬约束不进损失：总损失与没有硬约束时完全一致。"""
    soft_only = PhysicsLoss(nn.MSELoss(), [NonNegativityConstraint(weight=0.5)])
    with_hard = PhysicsLoss(
        nn.MSELoss(),
        [NonNegativityConstraint(weight=0.5)],
        hard_constraints=[NonNegativityClamp()],
    )
    batch = {"Y": TARGET}
    assert float(with_hard(OUTPUTS, batch)) == pytest.approx(float(soft_only(OUTPUTS, batch)))
    # 硬约束的 compute 恒为 0
    assert float(NonNegativityClamp().compute(OUTPUTS, batch)) == pytest.approx(0.0)


def test_hard_constraint_projection_makes_values_nonnegative():
    """硬约束以投影形式生效（clamp 截断到 0）。"""
    hard = NonNegativityClamp()
    projected = hard.project(OUTPUTS)
    assert float(projected.min()) >= 0.0
    assert torch.allclose(projected, torch.tensor([[0.0, 1.0]]))
    # 投影前的违例量可用于诊断，但不计进损失
    assert float(hard.violation(OUTPUTS)) == pytest.approx(PENALTY)


def test_softplus_projection_is_nonnegative_and_monotone():
    """softplus 重参数化：输出恒 ≥ 0 且单调。"""
    hard = NonNegativitySoftplus(beta=1.0)
    projected = hard.project(OUTPUTS)
    assert float(projected.min()) >= 0.0
    assert float(projected[0, 1]) > float(projected[0, 0])


def test_physics_loss_project_applies_hard_constraints():
    """组合器的 project() 依次施加硬约束。"""
    loss_fn = PhysicsLoss(nn.MSELoss(), hard_constraints=[NonNegativityClamp()])
    projected = loss_fn.project(OUTPUTS)
    assert torch.allclose(projected, torch.tensor([[0.0, 1.0]]))


# ----------------------------------------------------------------------
# 4. 预留约束：构造即报错
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls",
    [
        CWPIntegralConsistencyConstraint,
        WaterBudgetConstraint,
        VerticalProfileStructureConstraint,
        AtmosphericWaterCycleConstraint,
    ],
)
def test_reserved_constraints_raise_on_construction(cls):
    """预留约束构造即抛 NotImplementedError（防止误以为已实现）。"""
    with pytest.raises(NotImplementedError):
        cls()
    # 本模块的错误类型同时是物理约束错误，便于统一捕获
    with pytest.raises(ReservedConstraintError):
        cls()
    with pytest.raises(PhysicsNotReadyError):
        cls()


def test_reserved_constraints_expose_requirements_without_instantiation():
    """requirements() 可在构造前查询（类方法）：列出还缺哪些输入才可启用。"""
    for cls in RESERVED_CONSTRAINTS:
        requirements = cls.requirements()
        assert isinstance(requirements, list) and requirements
        assert all(isinstance(r, str) and r for r in requirements)
        units = cls.required_units()
        assert isinstance(units, dict) and units


def test_cwp_reserved_constraint_lists_required_inputs():
    """CWP 一致性约束明确列出：气压层坐标 / 空气密度廓线 / 单位口径。"""
    requirements = " ".join(CWPIntegralConsistencyConstraint.requirements())
    assert "kg/kg" in requirements and "kg/m3" in requirements
    assert "空气密度" in requirements
    assert "气压层" in requirements


# ----------------------------------------------------------------------
# 5. 单位感知
# ----------------------------------------------------------------------
def test_unit_parsing_and_conversion():
    """单位解析与确定无疑的换算。"""
    assert parse_unit("kg kg-1").canonical == "1"
    assert parse_unit("kg/kg").scale == pytest.approx(1.0)
    assert parse_unit("g/kg").scale == pytest.approx(1e-3)
    assert parse_unit("kg m-3").canonical == "kg m-3"
    assert parse_unit("kg m-2").canonical == "kg m-2"
    assert convert(1.0, "hPa", "Pa") == pytest.approx(100.0)
    assert convert(1.0, "kg/kg", "g/kg") == pytest.approx(1000.0)
    # 量纲不同的两个单位不可换算
    assert units_compatible("kg/kg", "g/kg")
    assert not units_compatible("kg/kg", "kg m-3")
    with pytest.raises(UnitMismatchError):
        convert(1.0, "kg/kg", "kg m-3")


def test_mass_fraction_vs_mass_density_classification():
    """kg/kg 是质量分数，kg/m3 是质量密度，% 不是质量分数。"""
    assert is_mass_fraction("kg kg-1")
    assert is_mass_fraction("kg/kg")
    assert not is_mass_fraction("%")
    assert classify_unit_family("kg m-3") == "mass_density"
    assert classify_unit_family("kg m-2") == "path_mass"
    assert water_content_convention("kg/kg") == CONVENTION_MASS_FRACTION
    assert water_content_convention("kg m-3") == CONVENTION_MASS_DENSITY
    with pytest.raises(UnitMismatchError):
        water_content_convention("kg m-2")


def test_constraint_raises_on_unit_mismatch():
    """单位口径不符 → 报错（不默默算错）。"""
    batch = {"Y": TARGET, "units": {"clwc": "kg m-2"}}  # 把 CWP 的单位错安到 clwc 上
    constraint = NonNegativityConstraint(weight=1.0, variable="clwc")
    with pytest.raises(UnitMismatchError):
        constraint.compute(OUTPUTS, batch)


def test_constraint_accepts_both_water_content_conventions():
    """kg/kg 与 kg/m3 都是合法的云水含量口径，不报错。"""
    for unit in ("kg kg-1", "kg m-3"):
        batch = {"Y": TARGET, "units": {"clwc": unit}}
        penalty = NonNegativityConstraint(weight=1.0, variable="clwc").compute(OUTPUTS, batch)
        assert float(penalty) == pytest.approx(PENALTY)


def test_constraint_requires_declared_unit_when_asked():
    """require_declared_unit=True 时，未声明单位也算错误。"""
    constraint = NonNegativityConstraint(weight=1.0, require_declared_unit=True)
    with pytest.raises(UnitMismatchError):
        constraint.compute(OUTPUTS, {"Y": TARGET})


def test_unit_consistency_constraint_passes_or_raises():
    """单位一致性检查：相符返回 0，不符抛错。"""
    ok = UnitConsistencyConstraint(weight=1.0).compute(
        OUTPUTS, {"Y": TARGET, "units": {"clwc": "kg kg-1", "cwp": "kg m-2"}}
    )
    assert float(ok) == pytest.approx(0.0)
    with pytest.raises(UnitMismatchError):
        UnitConsistencyConstraint(weight=1.0).compute(
            OUTPUTS, {"Y": TARGET, "units": {"cwp": "kg kg-1"}}
        )


def test_cwp_integral_two_conventions_give_different_results():
    """kg/kg 与 kg/m3 两种口径的积分结果不同。

    数值例子（2 层，各 100 m；ρ_air = 1.2 / 1.1 kg/m3）：

    - 口径 A（比含水量 kg/kg，本仓库 ``data/schema.py`` 登记的 clwc 口径）：
      ``CWP = ∫ ρ_air·CLWC dz = 1.2·1e-3·100 + 1.1·2e-3·100 = 0.34 kg/m2``；
    - 口径 B（质量密度 kg/m3）：``CWP = ∫ CLWC dz = 1e-3·100 + 2e-3·100 = 0.30 kg/m2``。

    哪个是对的取决于数据本身的单位登记：对**本仓库**数据（clwc 登记为 kg/kg）
    口径 A 正确；只有当数据确实以 kg/m3 存储时口径 B 才成立，此时再乘空气
    密度属于重复缩放（会高估约 ρ_air 倍）。
    """
    clwc = torch.tensor([1e-3, 2e-3])
    dz = torch.tensor([100.0, 100.0])
    rho = torch.tensor([1.2, 1.1])

    cwp_fraction = cwp_integral(
        clwc, units="kg kg-1", layer_thickness=dz, density_profile=rho
    )
    assert float(cwp_fraction) == pytest.approx(0.34, abs=1e-6)

    cwp_density = cwp_integral(clwc, units="kg m-3", layer_thickness=dz)
    assert float(cwp_density) == pytest.approx(0.30, abs=1e-6)

    # 用错口径不会报错，只会整体偏一个 ρ_air 量级
    assert not torch.allclose(cwp_fraction, cwp_density)


def test_cwp_integral_requires_density_for_mass_fraction():
    """比含水量口径缺空气密度廓线 → 报错（NotImplementedError 子类）。"""
    with pytest.raises(PhysicsNotReadyError):
        cwp_integral(torch.tensor([1e-3]), units="kg kg-1", layer_thickness=torch.tensor([100.0]))
    with pytest.raises(NotImplementedError):
        cwp_integral(torch.tensor([1e-3]), units="kg kg-1", layer_thickness=torch.tensor([100.0]))


def test_cwp_integral_rejects_density_profile_in_density_convention():
    """质量密度口径下再给空气密度 → 报错（防止重复乘 ρ）。"""
    with pytest.raises(UnitMismatchError):
        cwp_integral(
            torch.tensor([1e-3]),
            units="kg m-3",
            layer_thickness=torch.tensor([100.0]),
            density_profile=torch.tensor([1.2]),
        )


def test_cwp_integral_requires_layer_thickness():
    """缺层厚 → 报错（没有层厚无法积分）。"""
    with pytest.raises(PhysicsNotReadyError):
        cwp_integral(
            torch.tensor([1e-3]),
            units="kg kg-1",
            density_profile=torch.tensor([1.2]),
        )


# ----------------------------------------------------------------------
# 6. Validator：PASS / FAIL / NOT_APPLICABLE / NOT_IMPLEMENTED
# ----------------------------------------------------------------------
def _statuses(report: dict) -> dict[str, str]:
    return {check["name"]: check["status"] for check in report["checks"]}


def test_validator_marks_reserved_checks_not_implemented():
    """未实现的检查必须记 NOT_IMPLEMENTED，不能记 PASS。"""
    report = PhysicsValidator().validate(OUTPUTS, {"Y": TARGET})
    statuses = _statuses(report)
    assert statuses["reserved_constraint:cwp_integral_consistency"] == "NOT_IMPLEMENTED"
    assert statuses["reserved_constraint:water_budget"] == "NOT_IMPLEMENTED"
    assert statuses["reserved_constraint:vertical_profile_structure"] == "NOT_IMPLEMENTED"
    assert statuses["reserved_constraint:atmospheric_water_cycle"] == "NOT_IMPLEMENTED"
    assert statuses["water_budget_residual"] == "NOT_IMPLEMENTED"
    assert statuses["cwp_vertical_integration"] == "NOT_IMPLEMENTED"
    # 存在未实现项 → 总体状态不得宣称 PASS
    assert report["summary"]["status"] == "NOT_IMPLEMENTED"


def test_validator_detects_negative_cloud_water():
    """负云水含量 → FAIL。"""
    prediction = torch.tensor([[-0.5, 1.0]])
    batch = {"Y": TARGET, "target_names": ["clwc", "ciwc"], "units": {"clwc": "kg kg-1"}}
    report = PhysicsValidator().validate(prediction, batch)
    statuses = _statuses(report)
    assert statuses["negative_values"] == "FAIL"
    assert report["summary"]["status"] == "FAIL"
    failures = [c for c in report["checks"] if c["status"] == "FAIL"]
    assert any("负" in c["message"] for c in failures)


def test_validator_units_check_not_applicable_without_units():
    """没有声明单位 → NOT_APPLICABLE（不是 PASS，也不是 FAIL）。"""
    report = PhysicsValidator().validate(
        torch.tensor([[1.0]]), {"Y": TARGET, "target_names": ["clwc"]}
    )
    statuses = _statuses(report)
    assert statuses["variable_units"] == "NOT_APPLICABLE"
    assert statuses["negative_values"] == "PASS"


def test_validator_units_check_fails_on_mismatch():
    """单位与登记口径不符 → FAIL。"""
    prediction = torch.tensor([[1.0]])
    batch = {"Y": TARGET, "target_names": ["cwp"], "units": {"cwp": "kg kg-1"}}
    report = PhysicsValidator().validate(prediction, batch)
    assert _statuses(report)["variable_units"] == "FAIL"


def test_validator_runs_integration_check_when_inputs_given():
    """显式提供积分输入时，垂直积分一致性才真正计算（不是 NOT_IMPLEMENTED）。

    例子：2 层各 100 m，ρ = 1.2 / 1.1 kg/m3，clwc = 1e-3 / 2e-3 kg/kg
    → CWP = 0.34 kg/m2，真值取 0.34 → 残差 0 → PASS。
    """
    prediction = torch.tensor([[1e-3, 0.34]])
    batch = {
        "Y": torch.tensor([[1e-3, 0.34]]),
        "target_names": ["clwc", "cwp"],
        "units": {"clwc": "kg kg-1", "cwp": "kg m-2"},
        "clwc_profile": torch.tensor([[1e-3, 2e-3]]),
        "layer_thickness": torch.tensor([100.0, 100.0]),
        "air_density": torch.tensor([1.2, 1.1]),
    }
    report = PhysicsValidator().validate(prediction, batch)
    statuses = _statuses(report)
    assert statuses["cwp_vertical_integration"] == "PASS"
    check = next(c for c in report["checks"] if c["name"] == "cwp_vertical_integration")
    assert check["details"]["max_abs_residual"] == pytest.approx(0.0, abs=1e-6)


def test_validator_markdown_contains_statuses():
    """Markdown 报告逐项列出状态。"""
    validator = PhysicsValidator()
    report = validator.validate(OUTPUTS, {"Y": TARGET})
    markdown = validator.to_markdown(report)
    assert "# 物理一致性报告" in markdown
    assert "NOT_IMPLEMENTED" in markdown
    assert "cwp_vertical_integration" in markdown
    assert "总体状态" in markdown


# ----------------------------------------------------------------------
# 7. 与既有 training/losses.py 的兼容桥
# ----------------------------------------------------------------------
def test_legacy_bridge_adapter_is_subclass_of_legacy_constraint():
    """适配器是旧基类的真实子类，可直接传给 training.losses.CompositeLoss。"""
    from cloudwater_ml.training.losses import PhysicsConstraint as LegacyConstraint

    constraint = NonNegativityConstraint(weight=0.5)
    adapter = adapt_for_legacy(constraint)
    assert isinstance(adapter, LegacyConstraint)
    assert adapter.name == "nonnegativity"
    assert adapter.weight == pytest.approx(0.5)
    assert float(adapter.penalty(OUTPUTS, {"Y": TARGET})) == pytest.approx(PENALTY)


def test_legacy_bridge_works_inside_legacy_composite_loss():
    """旧 CompositeLoss 能直接吃适配器，总损失数值一致。"""
    from cloudwater_ml.training.losses import CompositeLoss as LegacyCompositeLoss

    adapter = adapt_for_legacy(NonNegativityConstraint(weight=0.5))
    legacy = LegacyCompositeLoss(nn.MSELoss(), [adapter])
    batch = {"Y": TARGET}
    assert float(legacy(OUTPUTS, batch)) == pytest.approx(MSE + 0.5 * PENALTY)
    values = legacy.component_values(OUTPUTS, batch)
    assert values["supervised"] == pytest.approx(MSE)
    assert values["nonnegativity"] == pytest.approx(PENALTY)


def test_legacy_losses_file_untouched_still_raises():
    """既有 training/losses.py 的预留约束仍然构造即抛（本包没有改动它）。"""
    from cloudwater_ml.training.losses import CWPConsistencyPenalty

    with pytest.raises(NotImplementedError):
        CWPConsistencyPenalty()
