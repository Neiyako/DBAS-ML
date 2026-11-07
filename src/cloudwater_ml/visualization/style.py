"""绘图设计系统：统一配色、字体、坐标轴风格与量级换算。

约定：
- 只提供「样式层」与纯函数工具，不画任何图，便于被各绘图函数复用；
- 所有函数幂等、可重复调用，不修改全局 rcParams（除非显式调用
  :func:`apply_style`）；
- 数值量级极小（如云水含量 1e-5 kg kg⁻¹）时用 :func:`scaled_values`
  换算成「×10⁻⁵ kg kg⁻¹」的可读刻度，而不是让坐标轴显示成一串 0.0000x。

配色取 Okabe–Ito 色盲友好板；顺序色标用 viridis（感知均匀、色盲安全）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 pyproject 前
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "PALETTE",
    "DENSITY_CMAP",
    "DIVERGING_CMAP",
    "INK",
    "apply_style",
    "despine",
    "scaled_values",
    "save_figure",
    "add_value_axis",
    "stat_box",
]

# ---------------------------------------------------------------- 颜色板 ----

#: Okabe–Ito 色盲友好定性色板（8 色）
PALETTE: dict[str, str] = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "ink": "#1B2430",
}

#: 正文/次要文字色
INK = "#1B2430"
MUTED = "#66707C"
GRID = "#E4E8ED"

#: 密度/标量场顺序色标（感知均匀、色盲安全）
DENSITY_CMAP = "viridis"
#: 偏差（有正负）发散色标
DIVERGING_CMAP = "RdBu_r"

#: 字体回退链：拉丁字体优先，缺字时自动回退到中文字体。
#: **必须写成 font.family 的显式列表**——只有这样 matplotlib 才会启用逐字形
#: 回退（写成 font.family="sans-serif" + font.sans-serif=列表不会回退，
#: 中文会缺字变方框）。
_FONT_STACK = [
    "Helvetica Neue",
    "PingFang SC",
    "Arial",
    "Heiti SC",
    "DejaVu Sans",
]

_STYLE_APPLIED = False


def apply_style() -> None:
    """应用全局 rcParams（幂等）。"""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B7BFC9",
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK,
            "axes.labelsize": 11.5,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": "#4A5560",
            "ytick.color": "#4A5560",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "font.size": 11,
            "font.family": list(_FONT_STACK),
            "font.sans-serif": list(_FONT_STACK),
            "axes.unicode_minus": False,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
        }
    )
    _STYLE_APPLIED = True


def despine(ax, keep: tuple[str, ...] = ("left", "bottom")) -> None:
    """去掉多余边框；``keep`` 里的轴保留。"""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# ------------------------------------------------------------ 量级换算 ----


@dataclass(frozen=True)
class ValueScale:
    """量级换算结果。

    Attributes
    ----------
    factor : float
        原始值 → 可读值的除数（原始值 / factor 得到可读值）。
    exponent : int
        10 的指数（本项目典型为 -5）。
    """

    factor: float
    exponent: int

    @property
    def label(self) -> str:
        """轴标签用的量级后缀，如 ``×10⁻⁵``。"""
        if self.exponent == 0:
            return ""
        sup = str(self.exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        return f"×10{sup}"


def scaled_values(values, *, unit: str = "", decimals: int = 2) -> tuple[np.ndarray, ValueScale]:
    """把极小量级的数值换算成可读刻度。

    Parameters
    ----------
    values : array_like
        原始数值。
    unit : str
        单位后缀（放在轴标签里）。
    decimals : int
        换算后的小数位数（仅用于决定指数，不影响返回值精度）。

    Returns
    -------
    (scaled, scale) : tuple[np.ndarray, ValueScale]
        ``scaled`` 为换算后的数值；``scale.label`` 为 ``×10ⁿ`` 文本。

    Examples
    --------
    >>> import numpy as np
    >>> s, sc = scaled_values(np.array([1e-5, 2e-5]))
    >>> float(s[0]), sc.exponent
    (1.0, -5)
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 0.0
    if not math.isfinite(vmax) or vmax == 0.0:
        return arr.copy(), ValueScale(factor=1.0, exponent=0)
    exponent = int(math.floor(math.log10(vmax)))
    factor = 10.0**exponent
    return arr / factor, ValueScale(factor=factor, exponent=exponent)


def add_value_axis(
    ax,
    values,
    *,
    unit: str = "",
    label: str | None = None,
    decimals: int = 2,
    axis: str = "y",
) -> ValueScale:
    """给坐标轴套上「×10ⁿ 量级」刻度并写好标签。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    values : array_like
        该轴对应的原始数值（用于推断量级）。
    unit : str
        单位文本，如 ``kg kg⁻¹``。
    label : str | None
        变量名；为空则只写单位。
    decimals : int
        刻度小数位。
    axis : str
        ``"x"`` 或 ``"y"``。

    Returns
    -------
    ValueScale
        实际使用的换算（含单位后缀的轴标签已写入）。
    """
    _, scale = scaled_values(values, decimals=decimals)
    suffix = " ".join(x for x in (label, unit) if x)
    text = f"{suffix} ({scale.label} {unit})" if scale.label else suffix
    if scale.label:
        text = f"{suffix} [{scale.label} {unit}]" if unit else f"{suffix} [{scale.label}]"
    if axis == "x":
        ax.set_xlabel(text)
    else:
        ax.set_ylabel(text)
    return scale


def stat_box(
    ax,
    lines: list[str],
    *,
    loc: str = "upper left",
    fontsize: float = 9.5,
    alpha: float = 0.92,
) -> None:
    """在坐标轴上叠加一个半透明统计信息框。"""
    if not lines:
        return
    ax.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=INK,
        alpha=alpha,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#C9D0D8",
            "linewidth": 0.8,
            "alpha": alpha,
        },
        zorder=6,
    )


def _format_value(v: float, scale: ValueScale, decimals: int = 3) -> str:
    if v is None or not math.isfinite(float(v)):
        return "n/a"
    return f"{float(v) / scale.factor:.{decimals}f}"


def save_figure(fig, out_path, *, dpi: int = 200, close: bool = True) -> Path:
    """保存图片并返回路径（不调用 ``plt.show``）。"""
    apply_style()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    if close:
        plt.close(fig)
    return out


def percent_axis(ax, axis: str = "y") -> None:
    """把坐标轴改成百分比刻度。"""
    fmt = mticker.PercentFormatter(xmax=1.0, decimals=0)
    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.yaxis.set_major_formatter(fmt)
