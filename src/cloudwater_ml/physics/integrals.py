"""CWP 柱积分工具（把「单位口径」显式化）。

两种口径的公式（**结果不同，不能混用**）：

- **比含水量口径**（``kg/kg``，``data/schema.py`` 当前登记的 clwc/ciwc 口径）::

      CWP [kg/m2] = Σ_i  ρ_air,i [kg/m3] · CLWC_i [kg/kg] · Δz_i [m]

- **质量密度口径**（``kg/m3``）::

      CWP [kg/m2] = Σ_i  CLWC_i [kg/m3] · Δz_i [m]

两条公式的差别就是空气密度因子 ρ_air（约 1 kg/m3 量级）：比含水量口径下
**必须**乘空气密度，质量密度口径下**不能再乘**。用错口径不会在数值上明显
报错，只会整体偏一个量级因子，因此本模块要求调用方显式声明口径，
缺输入时抛 :class:`PhysicsNotReadyError`（``NotImplementedError`` 子类），
口径与输入矛盾时抛 :class:`UnitMismatchError`。

注意：本模块只做**给定口径下的积分**，不代表该口径已被验证 —— 哪个口径
正确取决于数据本身的单位登记（见 ``data/schema.py::KNOWN_VARIABLES``）。
"""
from __future__ import annotations

from torch import Tensor

from .errors import PhysicsNotReadyError, UnitMismatchError
from .units import (
    Unit,
    is_mass_density,
    is_mass_fraction,
    parse_unit,
    require_compatible,
)

__all__ = [
    "CONVENTION_MASS_FRACTION",
    "CONVENTION_MASS_DENSITY",
    "water_content_convention",
    "convert_tensor",
    "cwp_integral",
    "cwp_residual",
]

#: 比含水量口径（kg/kg）
CONVENTION_MASS_FRACTION = "mass_fraction"
#: 质量密度口径（kg/m3）
CONVENTION_MASS_DENSITY = "mass_density"

_AIR_DENSITY_UNIT = "kg m-3"
_LENGTH_UNIT = "m"


def convert_tensor(value: Tensor, from_unit: str | Unit, to_unit: str | Unit) -> Tensor:
    """同量纲张量换算（比例因子相除；量纲不同直接报错）。"""
    uf, ut = parse_unit(from_unit), parse_unit(to_unit)
    require_compatible(uf, ut, context="单位换算")
    return value * (uf.scale / ut.scale)


def water_content_convention(units: str | Unit) -> str:
    """判断云水含量单位属于哪种口径。

    Returns
    -------
    str
        ``"mass_fraction"``（比含重量，比值）或 ``"mass_density"``（质量密度）。

    Raises
    ------
    UnitMismatchError
        单位既不是比含水量也不是质量密度（例如 ``kg/m2``、``K``）。
    """
    if is_mass_density(units):
        return CONVENTION_MASS_DENSITY
    if is_mass_fraction(units):
        return CONVENTION_MASS_FRACTION
    raise UnitMismatchError(
        f"单位 {parse_unit(units).text!r} 不是云水含量的合法口径"
        "（应为比含水量 kg/kg 或质量密度 kg/m3）"
    )


def cwp_integral(
    clwc_profile: Tensor,
    *,
    units: str,
    layer_thickness: Tensor | None = None,
    density_profile: Tensor | None = None,
    air_density_units: str = _AIR_DENSITY_UNIT,
    thickness_units: str = _LENGTH_UNIT,
) -> Tensor:
    """按**声明口径**把云水含量廓线积分成 CWP（kg/m2）。

    Parameters
    ----------
    clwc_profile : Tensor
        云水含量廓线，形状 ``[..., n_levels]``（最后一维为垂直层）。
    units : str
        ``clwc_profile`` 的单位口径，必须是 ``kg/kg``（比含水量）或
        ``kg/m3``（质量密度）。
    layer_thickness : Tensor | None
        各层几何厚度 ``Δz``（单位见 ``thickness_units``），形状 ``[n_levels]``。
    density_profile : Tensor | None
        空气密度廓线 ``ρ_air(z)``（单位见 ``air_density_units``），
        形状 ``[n_levels]``。**比含重量口径下必需**。
    air_density_units : str
        空气密度单位，默认 ``kg m-3``。
    thickness_units : str
        层厚单位，默认 ``m``。

    Returns
    -------
    Tensor
        CWP（kg/m2），形状为 ``clwc_profile`` 去掉最后一维。

    Raises
    ------
    PhysicsNotReadyError
        缺该口径所需的输入（层厚 / 空气密度廓线）。
    UnitMismatchError
        单位口径不合法，或口径与给定输入矛盾（如质量密度口径下又给了密度廓线）。
    """
    convention = water_content_convention(units)
    if layer_thickness is None:
        raise PhysicsNotReadyError(
            "缺层厚 Δz（m）：CWP 是垂直积分，缺层厚无法积分；"
            "所需前置条件见 CWPIntegralConsistencyConstraint.requirements()"
        )
    dz = convert_tensor(layer_thickness, thickness_units, _LENGTH_UNIT)
    if dz.shape[-1] != clwc_profile.shape[-1]:
        raise ValueError(
            f"层厚层数 {dz.shape[-1]} 与廓线层数 {clwc_profile.shape[-1]} 不一致"
        )
    if convention == CONVENTION_MASS_DENSITY:
        if density_profile is not None:
            raise UnitMismatchError(
                "质量密度口径（kg/m3）下不能再乘空气密度 —— 重复乘会把 CWP "
                "整体缩放约 1/ρ_air 量级"
            )
        integrand = clwc_profile
    else:
        if density_profile is None:
            raise PhysicsNotReadyError(
                "比含水量口径（kg/kg）缺空气密度廓线 ρ_air(z)（kg/m3）："
                "CWP = ∫ ρ_air·CLWC·dz，缺 ρ_air 会把 CWP 算小约 1/ρ_air 量级；"
                "所需前置条件见 CWPIntegralConsistencyConstraint.requirements()"
            )
        require_compatible(air_density_units, _AIR_DENSITY_UNIT, context="空气密度单位")
        rho = convert_tensor(density_profile, air_density_units, _AIR_DENSITY_UNIT)
        if rho.shape[-1] != clwc_profile.shape[-1]:
            raise ValueError(
                f"空气密度层数 {rho.shape[-1]} 与廓线层数 {clwc_profile.shape[-1]} 不一致"
            )
        integrand = clwc_profile * rho
    return (integrand * dz).sum(dim=-1)


def cwp_residual(
    cwp_target: Tensor,
    clwc_profile: Tensor,
    **kwargs: object,
) -> Tensor:
    """CWP 残差 = 目标 CWP − 廓线积分 CWP（单位 kg/m2）。

    供物理一致性检查用；``kwargs`` 原样传给 :func:`cwp_integral`。
    """
    computed = cwp_integral(clwc_profile, **kwargs)  # type: ignore[arg-type]
    return cwp_target - computed
