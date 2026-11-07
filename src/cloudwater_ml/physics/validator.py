"""物理一致性验证器（独立模块）。

产出**物理一致性报告**：dict + 可格式化 Markdown。

状态四态
--------
``PASS`` / ``FAIL`` / ``NOT_APPLICABLE`` / ``NOT_IMPLEMENTED``。

硬规则：**未实现的检查一律记 ``NOT_IMPLEMENTED``，绝不记 ``PASS``**；
只要报告里存在 ``NOT_IMPLEMENTED`` 项，总体状态就不是 ``PASS``。

检查项
------
1. ``negative_values``：非负变量（CLWC/CIWC/CWP）的预测是否出现负值；
2. ``physical_range``：预测是否落在**由单位口径定义**的合法区间内
   （比含水量 < 1、其余非负）；无依据的上界只记录、不判 FAIL；
3. ``cwp_vertical_integration``：垂直积分一致性 —— **只在调用方显式提供
   积分所需输入（口径、层厚、空气密度廓线、CWP 真值）时才计算**，
   否则记 ``NOT_IMPLEMENTED`` 并列出缺什么；
4. ``variable_units``：变量声明的单位与登记口径是否一致；
5. ``water_budget_residual``：水量守恒残差 —— 预留约束，恒记
   ``NOT_IMPLEMENTED``；
6. ``reserved_constraint:<name>``：各预留约束的启用前置条件（恒
   ``NOT_IMPLEMENTED``）。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from .base import PhysicsConstraint
from .constraints import RESERVED_CONSTRAINTS, check_variable_unit
from .errors import UnitMismatchError
from .integrals import convert_tensor, cwp_integral
from .units import (
    batch_units,
    classify_unit_family,
    is_mass_fraction,
    schema_units,
)

__all__ = [
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_NOT_APPLICABLE",
    "STATUS_NOT_IMPLEMENTED",
    "CheckResult",
    "PhysicsValidator",
    "format_report_markdown",
]

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
STATUS_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"

#: 登记为物理非负的变量（data/schema.py::KNOWN_VARIABLES）
_DEFAULT_NONNEGATIVE = ("clwc", "ciwc", "cwp")

#: 比含水量的定义性上界（水的质量分数不能 ≥ 1）
_MASS_FRACTION_UPPER = 1.0


@dataclass
class CheckResult:
    """单项检查结果。

    Attributes
    ----------
    name : str
        检查项名称。
    status : str
        ``PASS`` / ``FAIL`` / ``NOT_APPLICABLE`` / ``NOT_IMPLEMENTED``。
    message : str
        面向人的说明。
    details : dict[str, Any]
        结构化明细（数值、缺失项等）。
    """

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的字典。"""
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": dict(self.details),
        }


def _target_names(batch: Mapping[str, Any] | None) -> list[str]:
    """取 batch 的目标变量名列表。"""
    if not isinstance(batch, Mapping):
        return []
    names = batch.get("target_names")
    if not names:
        return []
    return [str(n) for n in names]


def _columns(prediction: Tensor, names: Sequence[str]) -> dict[str, Tensor]:
    """把预测张量按目标变量名拆列。"""
    if not names:
        return {}
    if prediction.ndim == 1:
        return {names[0]: prediction}
    cols: dict[str, Tensor] = {}
    for idx, name in enumerate(names):
        if idx < prediction.shape[-1]:
            cols[name] = prediction[..., idx]
    return cols


class PhysicsValidator:
    """物理一致性验证器。

    Parameters
    ----------
    expected_units : Mapping[str, str] | None
        覆盖登记口径的期望单位（变量 → 单位）。
    nonnegative_variables : Iterable[str] | None
        需要非负的变量（默认取 ``data/schema.py`` 登记的非负变量：
        ``clwc`` / ``ciwc`` / ``cwp``）。
    range_bounds : Mapping[str, tuple[float, float]] | None
        额外的数值范围约束（变量 → ``(下界, 上界)``），作用于**原始数值**
        （不做单位换算）。
    integration_tolerance : float
        垂直积分一致性的残差容差（kg/m2）。
    reserved_constraints : Sequence[type[PhysicsConstraint]] | None
        需要在报告里列出「预留」状态的约束类（默认
        :data:`cloudwater_ml.physics.constraints.RESERVED_CONSTRAINTS`）。
    """

    def __init__(
        self,
        *,
        expected_units: Mapping[str, str] | None = None,
        nonnegative_variables: Iterable[str] | None = None,
        range_bounds: Mapping[str, tuple[float, float]] | None = None,
        integration_tolerance: float = 1.0e-6,
        reserved_constraints: Sequence[type[PhysicsConstraint]] | None = None,
    ) -> None:
        self.expected_units = dict(expected_units or {})
        self.nonnegative_variables = tuple(
            nonnegative_variables if nonnegative_variables is not None else _DEFAULT_NONNEGATIVE
        )
        self.range_bounds = {k: tuple(v) for k, v in (range_bounds or {}).items()}
        self.integration_tolerance = float(integration_tolerance)
        self.reserved_constraints = tuple(
            reserved_constraints if reserved_constraints is not None else RESERVED_CONSTRAINTS
        )

    # ------------------------------------------------------------------
    def validate(
        self,
        prediction: Tensor,
        batch: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行全部检查，返回物理一致性报告（dict）。

        Parameters
        ----------
        prediction : Tensor
            模型预测（``[batch, targets]`` 或 ``[batch]``）。
        batch : Mapping[str, Any] | None
            批次数据；可含 ``Y`` / ``target_names`` / ``units`` /
            ``clwc_profile`` / ``layer_thickness`` / ``air_density`` /
            ``cwp_target``。
        """
        batch = batch if isinstance(batch, Mapping) else {}
        names = _target_names(batch)
        units = batch_units(batch)
        columns = _columns(prediction, names)
        checks: list[CheckResult] = [
            self._check_negative_values(columns, names),
            self._check_physical_range(columns, units),
            self._check_vertical_integration(batch, names),
            self._check_variable_units(units),
            self._check_water_budget(),
        ]
        checks.extend(self._check_reserved_constraints())
        return self._build_report(checks, names, units)

    # ------------------------------------------------------------------
    # 具体检查
    # ------------------------------------------------------------------
    def _check_negative_values(
        self,
        columns: Mapping[str, Tensor],
        names: Sequence[str],
    ) -> CheckResult:
        """检查 1：非负变量是否出现负预测。"""
        if not names:
            return CheckResult(
                name="negative_values",
                status=STATUS_NOT_APPLICABLE,
                message="batch 未提供 target_names，无法定位非负变量对应的预测列",
                details={"nonnegative_variables": list(self.nonnegative_variables)},
            )
        checked: list[str] = []
        violations: dict[str, Any] = {}
        for name in names:
            if name not in self.nonnegative_variables:
                continue
            column = columns.get(name)
            if column is None:
                continue
            checked.append(name)
            negatives = (column < 0)
            n_neg = int(negatives.sum().item())
            if n_neg:
                violations[name] = {
                    "n_negative": n_neg,
                    "min_value": float(column.min().item()),
                }
        if not checked:
            return CheckResult(
                name="negative_values",
                status=STATUS_NOT_APPLICABLE,
                message="预测中没有需要非负检查的变量",
                details={"nonnegative_variables": list(self.nonnegative_variables)},
            )
        if violations:
            return CheckResult(
                name="negative_values",
                status=STATUS_FAIL,
                message=f"出现负云水含量: {violations}",
                details={"checked": checked, "violations": violations},
            )
        return CheckResult(
            name="negative_values",
            status=STATUS_PASS,
            message=f"非负变量均未出现负值: {checked}",
            details={"checked": checked},
        )

    def _check_physical_range(
        self,
        columns: Mapping[str, Tensor],
        units: Mapping[str, str],
    ) -> CheckResult:
        """检查 2：预测是否落在由单位口径定义的合法区间内。

        只使用**定义性**边界：比含水量 < 1（水的质量分数不能 ≥ 1）、
        其余非负量 ≥ 0。没有依据的上界（如云水含量的经验上限）只记录，
        不判 FAIL —— 本模块不臆造物理阈值。
        """
        findings: list[dict[str, Any]] = []
        notes: list[str] = []
        applied = 0
        for name, column in columns.items():
            unit = units.get(name)
            if unit is None:
                continue
            family = classify_unit_family(unit)
            low, high = self.range_bounds.get(name, (None, None))
            if low is not None and float(column.min().item()) < float(low):
                findings.append(
                    {
                        "variable": name,
                        "rule": f"用户设定下界 {low}",
                        "min_value": float(column.min().item()),
                    }
                )
            if high is not None and float(column.max().item()) > float(high):
                findings.append(
                    {
                        "variable": name,
                        "rule": f"用户设定上界 {high}",
                        "max_value": float(column.max().item()),
                    }
                )
            if is_mass_fraction(unit):
                applied += 1
                # 换算到 kg/kg 后再比较（% 口径需先除以 100）
                values = convert_tensor(column, unit, "kg kg-1")
                max_value = float(values.max().item())
                if max_value >= _MASS_FRACTION_UPPER:
                    findings.append(
                        {
                            "variable": name,
                            "rule": f"比含水量必须 < {_MASS_FRACTION_UPPER} kg/kg",
                            "max_value": max_value,
                        }
                    )
            elif family in ("mass_density", "path_mass"):
                applied += 1
                min_value = float(column.min().item())
                if min_value < 0.0:
                    findings.append(
                        {
                            "variable": name,
                            "rule": "非负量必须 ≥ 0",
                            "min_value": min_value,
                        }
                    )
                notes.append(
                    f"{name}: 质量类量无依据的理论上界（需数据统计/文献确认），本项只检查非负"
                )
        if applied == 0 and not findings:
            return CheckResult(
                name="physical_range",
                status=STATUS_NOT_APPLICABLE,
                message="没有可用于范围检查的变量单位（无法确定适用边界）",
                details={"notes": notes},
            )
        if findings:
            return CheckResult(
                name="physical_range",
                status=STATUS_FAIL,
                message=f"预测超出定义性物理范围: {findings}",
                details={"violations": findings, "notes": notes},
            )
        return CheckResult(
            name="physical_range",
            status=STATUS_PASS,
            message="预测落在定义性物理范围内",
            details={"notes": notes},
        )

    def _check_vertical_integration(
        self,
        batch: Mapping[str, Any],
        names: Sequence[str],
    ) -> CheckResult:
        """检查 3：CWP 与柱积分一致性（缺输入 → NOT_IMPLEMENTED）。

        只有当调用方**显式提供**积分所需输入时才计算：

        - ``units`` 中的 clwc 口径与 cwp 单位；
        - ``clwc_profile``（廓线）、``layer_thickness``（层厚）；
        - ``air_density``（空气密度廓线，比含水量口径下必需）；
        - CWP 真值（``cwp_target`` 或 ``Y`` 中 ``cwp`` 列）。
        """
        units = batch_units(batch)
        missing: list[str] = []
        clwc_units = units.get("clwc")
        if not clwc_units:
            missing.append("units['clwc']（云水含量的单位口径）")
        profile = batch.get("clwc_profile")
        if profile is None:
            missing.append("batch['clwc_profile']（云水含量廓线）")
        thickness = batch.get("layer_thickness")
        if thickness is None:
            missing.append("batch['layer_thickness']（各层厚度，m）")
        density = batch.get("air_density")
        if density is None and clwc_units and is_mass_fraction(clwc_units):
            missing.append("batch['air_density']（空气密度廓线，kg/m3；比含水量口径必需）")
        cwp_target = batch.get("cwp_target")
        if cwp_target is None:
            y = batch.get("Y")
            if y is not None and "cwp" in names:
                idx = list(names).index("cwp")
                if y.ndim >= 2 and idx < y.shape[-1]:
                    cwp_target = y[..., idx]
            if cwp_target is None:
                missing.append("CWP 真值（batch['cwp_target'] 或 Y 中的 cwp 列）")
        if missing:
            return CheckResult(
                name="cwp_vertical_integration",
                status=STATUS_NOT_IMPLEMENTED,
                message=(
                    "CWP 垂直积分一致性未执行：缺少积分所需输入（"
                    + "；".join(missing)
                    + "）。对应约束类 CWPIntegralConsistencyConstraint 仍为预留状态"
                    "（构造即抛 NotImplementedError）"
                ),
                details={"missing_inputs": missing},
            )
        assert clwc_units is not None and profile is not None and thickness is not None
        assert cwp_target is not None
        computed = cwp_integral(
            profile,
            units=clwc_units,
            layer_thickness=thickness,
            density_profile=density,
        )
        residual = cwp_target - computed
        max_abs = float(residual.abs().max().item())
        status = STATUS_PASS if max_abs <= self.integration_tolerance else STATUS_FAIL
        return CheckResult(
            name="cwp_vertical_integration",
            status=status,
            message=(
                f"柱积分残差 max|CWP_真值 − ∫ρ·CLWC dz| = {max_abs:.6g} kg/m2"
                f"（容差 {self.integration_tolerance:g}）"
            ),
            details={
                "max_abs_residual": max_abs,
                "tolerance": self.integration_tolerance,
                "clwc_units": clwc_units,
            },
        )

    def _check_variable_units(self, units: Mapping[str, str]) -> CheckResult:
        """检查 4：变量声明的单位与登记口径是否一致。"""
        if not units:
            return CheckResult(
                name="variable_units",
                status=STATUS_NOT_APPLICABLE,
                message="batch 未声明任何变量单位，无法核对口径",
                details={},
            )
        checked: list[str] = []
        skipped: list[str] = []
        mismatches: list[dict[str, Any]] = []
        for variable, actual in units.items():
            expected = self.expected_units.get(variable) or schema_units(variable)
            if expected is None:
                skipped.append(variable)
                continue
            checked.append(variable)
            try:
                check_variable_unit(variable, actual, expected)
            except UnitMismatchError as exc:
                mismatches.append(
                    {"variable": variable, "actual": actual, "expected": expected, "reason": str(exc)}
                )
        if mismatches:
            return CheckResult(
                name="variable_units",
                status=STATUS_FAIL,
                message=f"变量单位与登记口径不符: {mismatches}",
                details={"mismatches": mismatches, "checked": checked, "skipped": skipped},
            )
        return CheckResult(
            name="variable_units",
            status=STATUS_PASS,
            message=f"变量单位口径一致（已核对 {checked}）",
            details={"checked": checked, "skipped": skipped},
        )

    def _check_water_budget(self) -> CheckResult:
        """检查 5：水量守恒残差 —— 预留约束，恒 NOT_IMPLEMENTED。"""
        from .constraints import WaterBudgetConstraint

        return CheckResult(
            name="water_budget_residual",
            status=STATUS_NOT_IMPLEMENTED,
            message=(
                "水量守恒残差未实现：闭合方程的通量项与符号约定未验证"
                "（约束类 WaterBudgetConstraint 构造即抛 NotImplementedError）"
            ),
            details={"requirements": list(WaterBudgetConstraint.requirements())},
        )

    def _check_reserved_constraints(self) -> list[CheckResult]:
        """检查 6：各预留约束的启用前置条件（恒 NOT_IMPLEMENTED）。"""
        results: list[CheckResult] = []
        for cls in self.reserved_constraints:
            requirements = list(cls.requirements())
            results.append(
                CheckResult(
                    name=f"reserved_constraint:{cls.name}",
                    status=STATUS_NOT_IMPLEMENTED,
                    message=(
                        f"{cls.name} 为预留约束（构造即抛 NotImplementedError）；"
                        f"启用前置条件: " + "；".join(requirements)
                    ),
                    details={"requirements": requirements, "required_units": cls.required_units()},
                )
            )
        return results

    # ------------------------------------------------------------------
    def _build_report(
        self,
        checks: Sequence[CheckResult],
        names: Sequence[str],
        units: Mapping[str, str],
    ) -> dict[str, Any]:
        """汇总报告（总体状态：任一 FAIL → FAIL；有 NOT_IMPLEMENTED → NOT_IMPLEMENTED）。"""
        counts = {
            STATUS_PASS: 0,
            STATUS_FAIL: 0,
            STATUS_NOT_APPLICABLE: 0,
            STATUS_NOT_IMPLEMENTED: 0,
        }
        for check in checks:
            counts[check.status] = counts.get(check.status, 0) + 1
        if counts[STATUS_FAIL]:
            overall = STATUS_FAIL
        elif counts[STATUS_NOT_IMPLEMENTED]:
            overall = STATUS_NOT_IMPLEMENTED
        else:
            overall = STATUS_PASS
        return {
            "summary": {
                "status": overall,
                "n_checks": len(checks),
                "counts": counts,
            },
            "checks": [c.to_dict() for c in checks],
            "metadata": {
                "target_names": list(names),
                "units": dict(units),
                "integration_tolerance": self.integration_tolerance,
            },
        }

    # ------------------------------------------------------------------
    def to_markdown(self, report: Mapping[str, Any]) -> str:
        """把报告格式化为 Markdown（与 :func:`format_report_markdown` 等价）。"""
        return format_report_markdown(report)


def format_report_markdown(report: Mapping[str, Any]) -> str:
    """把物理一致性报告格式化为 Markdown 文本。

    表格逐项列出检查名、状态与说明；状态原样保留
    （``PASS`` / ``FAIL`` / ``NOT_APPLICABLE`` / ``NOT_IMPLEMENTED``）。
    """
    summary = dict(report.get("summary") or {})
    counts = dict(summary.get("counts") or {})
    lines = [
        "# 物理一致性报告",
        "",
        f"- 总体状态：**{summary.get('status', 'UNKNOWN')}**",
        "- 检查项统计："
        + " / ".join(
            f"{key} {counts.get(key, 0)}"
            for key in (STATUS_PASS, STATUS_FAIL, STATUS_NOT_APPLICABLE, STATUS_NOT_IMPLEMENTED)
        ),
        "",
        "| 检查项 | 状态 | 说明 |",
        "| --- | --- | --- |",
    ]
    for check in report.get("checks") or []:
        name = str(check.get("name", "?"))
        status = str(check.get("status", "?"))
        message = str(check.get("message", "")).replace("|", "\\|")
        lines.append(f"| `{name}` | **{status}** | {message} |")
    metadata = dict(report.get("metadata") or {})
    if metadata:
        lines.extend(["", "## 元信息", ""])
        for key, value in metadata.items():
            lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"
