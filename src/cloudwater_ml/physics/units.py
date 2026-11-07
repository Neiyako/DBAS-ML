"""单位解析、比较与换算（物理约束的单位感知基础设施）。

与 ``cloudwater_ml.data.schema.KNOWN_VARIABLES`` 的登记口径对齐：

======================  ==============  ==========================
变量                    登记单位         口径
======================  ==============  ==========================
``clwc`` / ``ciwc``     ``kg kg-1``      比含水量（质量分数）
``cwp``                 ``kg m-2``       云水路径（面质量积分）
``q``                   ``kg kg-1``      比湿
``p``                   ``hPa``          气压
``w`` / ``omega``       ``Pa s-1``       垂直速度
``t``                   ``K``            温度
======================  ==============  ==========================

原则
----
1. **只实现确定无疑的换算**：质量/长度/时间/温度的十进制前缀，以及同量纲
   之间的线性换算（``hPa → Pa``、``kg/kg → g/kg``）。
2. **量纲不同绝不隐式换算**：``kg/kg`` 与 ``kg/m3`` 量纲不同，
   :func:`require_compatible` 会抛 :class:`UnitMismatchError`；两者之间的
   换算必须显式提供空气密度（见 :mod:`cloudwater_ml.physics.integrals`）。
3. **未知单位报错**，不做任何猜测。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from cloudwater_ml.data.schema import KNOWN_VARIABLES

from .errors import UnitMismatchError

__all__ = [
    "Unit",
    "parse_unit",
    "format_unit",
    "units_compatible",
    "require_compatible",
    "convert",
    "classify_unit_family",
    "assert_unit_family",
    "batch_units",
    "schema_units",
    "expected_families_for",
    "is_mass_fraction",
    "is_mass_density",
    "FAMILY_DIMENSIONLESS",
    "FAMILY_MASS",
    "FAMILY_MASS_DENSITY",
    "FAMILY_PATH_MASS",
    "FAMILY_PRESSURE",
    "FAMILY_PRESSURE_RATE",
    "FAMILY_LENGTH",
    "FAMILY_TIME",
    "FAMILY_TEMPERATURE",
    "FAMILY_SPEED",
    "FAMILY_OTHER",
    "WATER_CONTENT_FAMILIES",
    "FAMILY_LABELS",
    "KNOWN_UNITS",
]

# ----------------------------------------------------------------------
# 单位定义表（量纲按 SI 基本量 kg / m / s / K 展开）
# ----------------------------------------------------------------------
#: 单位名 → (量纲指数, 换算到 SI 基本单位的比例因子)
_UNIT_DEFS: dict[str, tuple[dict[str, float], float]] = {
    "kg": ({"kg": 1.0}, 1.0),
    "g": ({"kg": 1.0}, 1.0e-3),
    "m": ({"m": 1.0}, 1.0),
    "s": ({"s": 1.0}, 1.0),
    "K": ({"K": 1.0}, 1.0),
    "Pa": ({"kg": 1.0, "m": -1.0, "s": -2.0}, 1.0),
    "hPa": ({"kg": 1.0, "m": -1.0, "s": -2.0}, 1.0e2),
    "%": ({}, 1.0e-2),
    "1": ({}, 1.0),
}

#: 小写别名 → 标准单位名
_ALIASES: dict[str, str] = {
    "hpa": "hPa",
    "hectopascal": "hPa",
    "mbar": "hPa",
    "mb": "hPa",
    "pa": "Pa",
    "pascal": "Pa",
    "kg": "kg",
    "kilogram": "kg",
    "g": "g",
    "gram": "g",
    "gramme": "g",
    "m": "m",
    "meter": "m",
    "metre": "m",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "k": "K",
    "kelvin": "K",
    "%": "%",
    "percent": "%",
    "pct": "%",
    "1": "1",
    "": "1",
    "-": "1",
    "none": "1",
    "dimensionless": "1",
    "unitless": "1",
}

#: 质量类基本单位（用于判定「质量分数」形态）
_MASS_BASES = frozenset({"kg", "g"})

#: 量纲输出顺序
_SI_ORDER = ("kg", "m", "s", "K")

#: 单个 token 的语法：名字（字母/百分号）+ 可选指数（``m-3`` / ``m3`` / ``m**-3``）
_TOKEN_RE = re.compile(r"^(?P<name>[A-Za-z%]+)(?:\*\*|\^)?(?P<exp>[+-]?\d+)?$")

#: 已知单位名（供报错提示）
KNOWN_UNITS: tuple[str, ...] = tuple(sorted(_UNIT_DEFS))

# ----------------------------------------------------------------------
# 量纲族（family）——用于「口径是否对得上」的粗粒度判断
# ----------------------------------------------------------------------
FAMILY_DIMENSIONLESS = "dimensionless"
FAMILY_MASS = "mass"
FAMILY_MASS_DENSITY = "mass_density"
FAMILY_PATH_MASS = "path_mass"
FAMILY_PRESSURE = "pressure"
FAMILY_PRESSURE_RATE = "pressure_rate"
FAMILY_LENGTH = "length"
FAMILY_TIME = "time"
FAMILY_TEMPERATURE = "temperature"
FAMILY_SPEED = "speed"
FAMILY_OTHER = "other"

#: 族 → 中文说明（错误信息用）
FAMILY_LABELS: dict[str, str] = {
    FAMILY_DIMENSIONLESS: "无量纲比值（如 kg/kg、%）",
    FAMILY_MASS: "质量（如 kg）",
    FAMILY_MASS_DENSITY: "质量密度（如 kg/m3）",
    FAMILY_PATH_MASS: "面质量积分（如 kg/m2）",
    FAMILY_PRESSURE: "压强（如 Pa、hPa）",
    FAMILY_PRESSURE_RATE: "压强随时间变化率（如 Pa/s）",
    FAMILY_LENGTH: "长度（如 m）",
    FAMILY_TIME: "时间（如 s）",
    FAMILY_TEMPERATURE: "温度（如 K）",
    FAMILY_SPEED: "速度（如 m/s）",
    FAMILY_OTHER: "其它量纲",
}

#: 云水含量的两种合法口径：比含水量（无量纲比值）或质量密度
WATER_CONTENT_FAMILIES: frozenset[str] = frozenset(
    {FAMILY_DIMENSIONLESS, FAMILY_MASS_DENSITY}
)


@dataclass(frozen=True)
class Unit:
    """解析后的单位。

    Attributes
    ----------
    text : str
        原始单位字符串（保留，用于报错展示）。
    dimension : dict[str, float]
        SI 基本量（kg / m / s / K）的指数，缺省为 0。
    scale : float
        比例因子：``1 个该单位 = scale 个 SI 基本单位``（如 hPa 的 scale=100）。
    terms : tuple[tuple[str, float], ...]
        构成该单位的「基本单位名 × 指数」列表（保留结构信息，
        便于区分 ``kg/kg``（质量分数）与 ``%``（真无量纲））。
    """

    text: str
    dimension: Mapping[str, float]
    scale: float
    terms: tuple[tuple[str, float], ...] = ()

    @property
    def canonical(self) -> str:
        """规范化单位文本（按量纲排序，如 ``kg kg-1``、``kg m-2``）。

        注意：规范化结果只保留**量纲**，不含比例因子（``hPa`` 与 ``Pa``
        的规范化结果相同），换算请用 :func:`convert`。
        """
        if not self.dimension:
            return "1"
        parts = []
        for key in _SI_ORDER:
            exp = self.dimension.get(key, 0.0)
            if abs(exp) > 1e-12:
                parts.append(key if abs(exp - 1.0) < 1e-12 else f"{key}{exp:g}")
        for key in sorted(k for k in self.dimension if k not in _SI_ORDER):
            exp = self.dimension[key]
            if abs(exp) > 1e-12:
                parts.append(f"{key}{exp:g}")
        return " ".join(parts)

    @property
    def is_dimensionless(self) -> bool:
        """量纲是否为零（无量纲）。"""
        return not self.dimension

    def to_si(self, value: float) -> float:
        """数值换算到 SI 基本单位。"""
        return float(value) * self.scale

    def from_si(self, value: float) -> float:
        """数值从 SI 基本单位换算回该单位。"""
        return float(value) / self.scale

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return self.canonical


def _parse_tokens(text: str) -> list[tuple[str, float]]:
    """把一段单位文本拆成 ``[(基本单位名, 指数), ...]``。"""
    out: list[tuple[str, float]] = []
    for token in text.split():
        match = _TOKEN_RE.match(token)
        if match is None:
            raise ValueError(f"无法解析单位片段 {token!r}（来自 {text!r}）")
        name = match.group("name")
        raw_exp = match.group("exp")
        exponent = float(raw_exp) if raw_exp not in (None, "") else 1.0
        base = _ALIASES.get(name.lower(), name)
        if base not in _UNIT_DEFS:
            raise ValueError(
                f"未知单位 {name!r}（已知单位: {', '.join(KNOWN_UNITS)}）"
            )
        out.append((base, exponent))
    return out


def parse_unit(unit: str | Unit) -> Unit:
    """解析单位字符串，返回 :class:`Unit`。

    支持的写法：``kg kg-1`` / ``kg/kg`` / ``g kg-1`` / ``kg m-3`` / ``kg/m3`` /
    ``kg m-2`` / ``m s-1`` / ``Pa s-1`` / ``hPa`` / ``K`` / ``%`` / ``1``。

    Raises
    ------
    ValueError
        单位为空、含未知单位名或片段语法不合法。
    """
    if isinstance(unit, Unit):
        return unit
    raw = str(unit).strip()
    if not raw:
        raise ValueError("单位字符串为空")
    text = raw.replace("·", " ").replace("*", " ")
    if "/" in text:
        parts = [p.strip() for p in text.split("/") if p.strip()]
        if len(parts) < 2:
            raise ValueError(f"单位 {raw!r} 的 '/' 用法不完整")
        terms = _parse_tokens(parts[0])
        for denom in parts[1:]:
            for name, exponent in _parse_tokens(denom):
                terms.append((name, -exponent))
    else:
        terms = _parse_tokens(text)
    if not terms:
        raise ValueError(f"单位 {raw!r} 未包含任何可识别的单位")
    dimension: dict[str, float] = {}
    scale = 1.0
    for name, exponent in terms:
        dim, factor = _UNIT_DEFS[name]
        for key, value in dim.items():
            dimension[key] = dimension.get(key, 0.0) + value * exponent
        scale *= factor**exponent
    dimension = {k: v for k, v in dimension.items() if abs(v) > 1e-12}
    return Unit(text=raw, dimension=dimension, scale=scale, terms=tuple(terms))


def format_unit(unit: str | Unit) -> str:
    """返回规范化单位文本（按量纲排序）。"""
    return parse_unit(unit).canonical


def units_compatible(a: str | Unit, b: str | Unit) -> bool:
    """两个单位是否量纲相同（可线性换算）。"""
    return parse_unit(a).dimension == parse_unit(b).dimension


def require_compatible(
    a: str | Unit,
    b: str | Unit,
    *,
    context: str = "",
) -> None:
    """要求两个单位量纲一致，否则抛 :class:`UnitMismatchError`。"""
    ua, ub = parse_unit(a), parse_unit(b)
    if ua.dimension != ub.dimension:
        prefix = f"{context}：" if context else ""
        raise UnitMismatchError(
            f"{prefix}单位量纲不一致 —— {ua.text!r} ({ua.canonical}) vs "
            f"{ub.text!r} ({ub.canonical})；量纲不同的量之间不做隐式换算"
        )


def convert(value: float, from_unit: str | Unit, to_unit: str | Unit) -> float:
    """同量纲线性换算（如 ``1 hPa → 100 Pa``、``1 kg/kg → 1000 g/kg``）。

    Raises
    ------
    UnitMismatchError
        量纲不同（例如 ``kg/kg → kg/m3``，必须显式引入空气密度）。
    """
    uf, ut = parse_unit(from_unit), parse_unit(to_unit)
    require_compatible(uf, ut, context="单位换算")
    return uf.to_si(value) / ut.scale


def classify_unit_family(unit: str | Unit) -> str:
    """判断单位所属的「口径族」（如 ``mass_density`` / ``path_mass``）。"""
    dims = parse_unit(unit).dimension
    if not dims:
        return FAMILY_DIMENSIONLESS
    if dims == {"kg": 1.0}:
        return FAMILY_MASS
    if dims == {"kg": 1.0, "m": -3.0}:
        return FAMILY_MASS_DENSITY
    if dims == {"kg": 1.0, "m": -2.0}:
        return FAMILY_PATH_MASS
    if dims == {"kg": 1.0, "m": -1.0, "s": -2.0}:
        return FAMILY_PRESSURE
    if dims == {"kg": 1.0, "m": -1.0, "s": -3.0}:
        return FAMILY_PRESSURE_RATE
    if dims == {"m": 1.0}:
        return FAMILY_LENGTH
    if dims == {"s": 1.0}:
        return FAMILY_TIME
    if dims == {"K": 1.0}:
        return FAMILY_TEMPERATURE
    if dims == {"m": 1.0, "s": -1.0}:
        return FAMILY_SPEED
    return FAMILY_OTHER


def assert_unit_family(
    unit: str | Unit,
    allowed: Iterable[str],
    *,
    context: str = "",
) -> str:
    """校验单位口径族是否在允许集合内，否则抛 :class:`UnitMismatchError`。

    Returns
    -------
    str
        实际命中的族名。
    """
    family = classify_unit_family(unit)
    allowed_set = set(allowed)
    if family not in allowed_set:
        prefix = f"{context}：" if context else ""
        allowed_desc = " | ".join(
            FAMILY_LABELS.get(f, f) for f in sorted(allowed_set)
        )
        raise UnitMismatchError(
            f"{prefix}单位 {parse_unit(unit).text!r} 的口径为 "
            f"{FAMILY_LABELS.get(family, family)}，与期望不符；"
            f"允许的口径: {allowed_desc}"
        )
    return family


def batch_units(batch: Mapping[str, Any] | None) -> dict[str, str]:
    """从 batch 中取出「变量 → 单位」映射（缺失时返回空字典）。

    约定：batch 可携带 ``units`` 键（形如 ``{"clwc": "kg kg-1"}``），
    与 ``data/schema.py::SampleBundle.units`` 的约定一致。
    """
    if not isinstance(batch, Mapping):
        return {}
    raw = batch.get("units") or {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v is not None}


def schema_units(variable: str) -> str | None:
    """取 ``data/schema.py`` 中登记的单位（未知变量返回 ``None``）。"""
    spec = KNOWN_VARIABLES.get(variable)
    return spec.units if spec is not None else None


def expected_families_for(variable: str) -> set[str]:
    """某变量被允许的单位口径族集合。

    - ``clwc`` / ``ciwc``：允许「比含水量（无量纲比值）」与「质量密度」两种
      口径（两者都是物理上成立的云水含量表达）。
    - 其它变量：取 ``data/schema.py`` 登记单位的族（单一）。
    - 未知变量：空集合（不做判断，由调用方决定是否报错）。
    """
    if variable in ("clwc", "ciwc"):
        return set(WATER_CONTENT_FAMILIES)
    spec = KNOWN_VARIABLES.get(variable)
    if spec is None:
        return set()
    return {classify_unit_family(spec.units)}


def is_mass_fraction(unit: str | Unit) -> bool:
    """是否「质量/质量」型比值（如 ``kg/kg``、``g/kg``）。

    与真无量纲（``%``、``1``）区分：只有含质量且质量指数抵消的单位才算比含水量。
    """
    u = parse_unit(unit)
    if u.dimension:
        return False
    mass_exp = sum(exp for name, exp in u.terms if name in _MASS_BASES)
    has_mass = any(name in _MASS_BASES for name, _ in u.terms)
    return has_mass and abs(mass_exp) < 1e-12


def is_mass_density(unit: str | Unit) -> bool:
    """是否质量密度（``kg/m3`` 口径）。"""
    return classify_unit_family(unit) == FAMILY_MASS_DENSITY
