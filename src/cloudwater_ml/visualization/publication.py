"""出版级实验图：密度散点、训练曲线、残差诊断、分组泛化、空间场、鲁棒性、量级-误差。

与旧版绘图模块的差别：
- **密度散点**取代密集散点（hexbin + 对数密度色标），数据点不再糊成一团；
- 极小量级统一换算成 ``×10⁻⁵ kg kg⁻¹`` 可读刻度，不再显示一串 0.0000x；
- 每张图带标题 / 副标题 / 脚注（含数据来源与口径），可独立当报告插图；
- 统一的色盲友好配色、去边框、留白与字号层级。

约定：Agg 后端、savefig(dpi=200)、不调用 ``plt.show()``、返回 Path；
输入不可用/绘图失败一律返回 None，不抛异常（与既有绘图模块一致）。
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from .style import (
    DENSITY_CMAP,
    DIVERGING_CMAP,
    INK,
    MUTED,
    PALETTE,
    apply_style,
    save_figure,
    scaled_values,
    stat_box,
)

logger = logging.getLogger("cloudwater_ml.visualization")

__all__ = [
    "plot_parity_density",
    "plot_training_history",
    "plot_residual_diagnostics",
    "plot_subgroup_performance",
    "plot_spatial_maps",
    "plot_robustness",
    "plot_error_by_magnitude",
]

_TARGET_COLORS = [PALETTE["blue"], PALETTE["vermillion"], PALETTE["green"], PALETTE["purple"]]


# ------------------------------------------------------------------ 工具 ----


def _as_2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _names(target_names, n_cols: int) -> list[str]:
    names = list(target_names or [])
    return [names[i] if i < len(names) else f"target_{i}" for i in range(n_cols)]


def _wrap_text(text: str, max_em: float) -> list[str]:
    """按 em 宽度折行（中文按 1 em、拉丁按 0.55 em 估算）。

    脚注若不折行会超出画布右边界，把 ``savefig(bbox="tight")`` 的输出
    撑出大片空白——所以这里统一折行。
    """
    lines: list[str] = []
    cur = ""
    cur_w = 0.0
    for ch in text:
        w = 1.0 if ord(ch) > 0x2E80 else 0.55
        if cur_w + w > max_em and cur:
            lines.append(cur)
            cur, cur_w = "", 0.0
        cur += ch
        cur_w += w
    if cur:
        lines.append(cur)
    return lines or [""]


def _header(fig, title: str, subtitle: str = "", note: str = "") -> None:
    """左上角标题 + 副标题 + 底部脚注（脚注按画布宽度折行）。"""
    apply_style()
    fig.text(0.0, 1.0, title, ha="left", va="bottom", fontsize=15.5,
             fontweight="semibold", color=INK)
    if subtitle:
        fig.text(0.0, 0.948, subtitle, ha="left", va="bottom", fontsize=10.5, color=MUTED)
    if note:
        fig_w_in = float(fig.get_size_inches()[0])
        max_em = max(fig_w_in * 72.0 / 8.8 * 0.96, 20.0)
        for i, line in enumerate(_wrap_text(note, max_em)):
            fig.text(0.0, -0.02 - 0.028 * i, line, ha="left", va="top",
                     fontsize=8.8, color=MUTED)


def _regression_stats(y: np.ndarray, p: np.ndarray, scale) -> dict[str, float]:
    err = p - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }
    if y.size > 1 and np.std(y) > 0 and np.std(p) > 0:
        out["pearson"] = float(np.corrcoef(y, p)[0, 1])
    else:
        out["pearson"] = float("nan")
    out["scale"] = scale
    return out


def _fmt(v: float, factor: float, decimals: int = 3) -> str:
    if v is None or not math.isfinite(float(v)):
        return "n/a"
    return f"{float(v) / factor:.{decimals}f}"


#: 标准正态分位数的有理逼近系数（Acklam 近似，|误差| < 1.15e-9）
#: ——平台不依赖 scipy，故自带实现，避免为了画一张 Q-Q 图引入额外依赖。
_NORM_PPF_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
               1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_NORM_PPF_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
               6.680131188771972e+01, -1.328068155288572e+01]
_NORM_PPF_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
               -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_NORM_PPF_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
               3.754408661907416e+00]


def _norm_ppf(p) -> np.ndarray:
    """标准正态分位函数（不依赖 scipy）。

    Parameters
    ----------
    p : array_like
        概率，取值 (0, 1)。

    Returns
    -------
    np.ndarray
        对应的分位数。
    """
    q = np.asarray(p, dtype=np.float64)
    out = np.empty_like(q)
    left = q < 0.02425
    right = q > 1.0 - 0.02425
    mid = ~(left | right)

    if left.any():
        u = np.sqrt(-2.0 * np.log(q[left]))
        out[left] = (((((_NORM_PPF_C[0] * u + _NORM_PPF_C[1]) * u + _NORM_PPF_C[2]) * u
                       + _NORM_PPF_C[3]) * u + _NORM_PPF_C[4]) * u + _NORM_PPF_C[5]) / (
            (((_NORM_PPF_D[0] * u + _NORM_PPF_D[1]) * u + _NORM_PPF_D[2]) * u
             + _NORM_PPF_D[3]) * u + 1.0)
    if right.any():
        u = np.sqrt(-2.0 * np.log(1.0 - q[right]))
        out[right] = -(((((_NORM_PPF_C[0] * u + _NORM_PPF_C[1]) * u + _NORM_PPF_C[2]) * u
                        + _NORM_PPF_C[3]) * u + _NORM_PPF_C[4]) * u + _NORM_PPF_C[5]) / (
            (((_NORM_PPF_D[0] * u + _NORM_PPF_D[1]) * u + _NORM_PPF_D[2]) * u
             + _NORM_PPF_D[3]) * u + 1.0)
    if mid.any():
        u = q[mid] - 0.5
        r = u * u
        out[mid] = (((((_NORM_PPF_A[0] * r + _NORM_PPF_A[1]) * r + _NORM_PPF_A[2]) * r
                      + _NORM_PPF_A[3]) * r + _NORM_PPF_A[4]) * r + _NORM_PPF_A[5]) * u / (
            ((((_NORM_PPF_B[0] * r + _NORM_PPF_B[1]) * r + _NORM_PPF_B[2]) * r
              + _NORM_PPF_B[3]) * r + _NORM_PPF_B[4]) * r + 1.0)
    return out


# ------------------------------------------------------------- 密度散点 ----


def plot_parity_density(
    y_true,
    y_pred,
    target_names: list[str],
    out_path,
    *,
    title: str = "预测值 vs 真值（密度散点）",
    subtitle: str = "",
    note: str = "",
    unit: str = "kg kg⁻¹",
) -> Path | None:
    """预测 vs 真值的**密度散点图**（hexbin，对数密度色标）。

    相比逐点 ``scatter``，密集数据点会聚合成可读的二维密度场，
    既保留分布信息又不糊成一片。

    Returns
    -------
    Path | None
        保存路径；形状不一致/输入不可用/绘图失败时返回 None。
    """
    try:
        apply_style()
        yt = _as_2d(y_true)
        yp = _as_2d(y_pred)
        if yt.shape != yp.shape or yt.size == 0:
            logger.warning("plot_parity_density 形状不一致或为空: %s vs %s", yt.shape, yp.shape)
            return None
        names = _names(target_names, yt.shape[1])
        n = yt.shape[1]
        fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.4), squeeze=False)
        hm = None
        # 全局量程（两个目标**共享坐标轴**，才能直接比较拟合斜率）
        glob = np.concatenate([yt.ravel(), yp.ravel()])
        glob = glob[np.isfinite(glob)]
        gscale = scaled_values(glob)[1]
        glo = float(glob.min()) / gscale.factor
        ghi = float(glob.max()) / gscale.factor
        gpad = (ghi - glo) * 0.04 or 1e-12

        for j in range(n):
            ax = axes[0][j]
            a, b = yt[:, j], yp[:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            if a.size == 0:
                ax.set_visible(False)
                continue
            sa, scale_a = scaled_values(a)
            sb, scale_b = scaled_values(b)
            factor = scale_a.factor
            hm = ax.hexbin(
                sa, sb, gridsize=42, mincnt=1, bins="log",
                cmap=DENSITY_CMAP, linewidths=0.0, rasterized=True,
            )
            ax.plot([glo - gpad, ghi + gpad], [glo - gpad, ghi + gpad], color=INK, lw=1.1,
                    ls="-", zorder=4, label="1:1")
            if a.size > 2:
                k, c = np.polyfit(sa, sb, 1)
                xs = np.array([glo, ghi])
                ax.plot(xs, k * xs + c, color=PALETTE["vermillion"], lw=1.5, ls="--",
                        zorder=5, label=f"OLS  k={k:.3f}")
            st = _regression_stats(a, b, factor)
            stat_box(ax, [
                f"$R^2$ = {st['r2']:.4f}",
                f"RMSE = {_fmt(st['rmse'], factor)}",
                f"MAE  = {_fmt(st['mae'], factor)}",
                f"Bias = {_fmt(st['bias'], factor)}",
                f"r    = {st['pearson']:.4f}",
                f"n = {a.size}",
            ])
            ax.set_xlabel(f"真值 [{scale_a.label} {unit}]")
            ax.set_ylabel(f"预测值 [{scale_b.label} {unit}]")
            ax.set_title(names[j])
            ax.set_xlim(glo - gpad, ghi + gpad)
            ax.set_ylim(glo - gpad, ghi + gpad)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, color="#EDF1F5", linewidth=0.8)
        if hm is not None:
            cbar = fig.colorbar(hm, ax=axes[0].tolist(), fraction=0.030, pad=0.015)
            cbar.ax.set_title("样本密度（对数刻度）", fontsize=9.5, color=INK, pad=8)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(labelsize=9.5, color="#C9D0D8")
            cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
            cbar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        _header(fig, title, subtitle, note)
        fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        return Path(out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_parity_density 失败: %s", exc)
        plt.close("all")
        return None


# ------------------------------------------------------------- 训练曲线 ----


def plot_training_history(
    history,
    out_path,
    *,
    title: str = "训练 / 验证损失",
    subtitle: str = "",
    note: str = "",
) -> Path | None:
    """训练-验证损失曲线 + 泛化间隙（log 纵轴、标注最佳轮次）。

    Parameters
    ----------
    history : list[dict] | dict
        逐 epoch 记录，至少含 ``train_loss`` / ``val_loss``。
    out_path : str | Path
    """
    try:
        apply_style()
        rows = history
        if isinstance(rows, dict):
            rows = rows.get("history", [])
        rows = [r for r in (rows or []) if isinstance(r, dict)]
        if not rows:
            logger.warning("plot_training_history: 无历史记录")
            return None
        ep = np.array([float(r.get("epoch", i + 1)) for i, r in enumerate(rows)])
        tr = np.array([float(r.get("train_loss", np.nan)) for r in rows])
        va = np.array([float(r.get("val_loss", np.nan)) for r in rows])
        if not np.isfinite(tr).any():
            logger.warning("plot_training_history: train_loss 全为非有限值")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
        ax = axes[0]
        ax.semilogy(ep, tr, color=PALETTE["blue"], lw=1.8, label="训练损失")
        if np.isfinite(va).any():
            ax.semilogy(ep, va, color=PALETTE["vermillion"], lw=1.8, label="验证损失")
            k = int(np.nanargmin(va))
            ax.axvline(ep[k], color=MUTED, ls=":", lw=1.2, zorder=1)
            ax.scatter([ep[k]], [va[k]], s=48, facecolor="white",
                       edgecolor=PALETTE["vermillion"], linewidth=1.6, zorder=6)
            # 标注放在**中上部空白区**（曲线在此已收敛到底部），避免压住曲线
            ax.annotate(
                f"最佳 epoch {int(ep[k])}\nval = {va[k]:.4g}",
                xy=(ep[k], va[k]), xycoords="data",
                xytext=(0.56, 0.80), textcoords="axes fraction",
                fontsize=9.5, color=INK, ha="left", va="top",
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#C9D0D8",
                      "linewidth": 0.8, "pad": 3.2},
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.9,
                            "shrinkA": 2, "shrinkB": 4},
                zorder=6,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("损失（归一化尺度）")
        ax.set_title("收敛过程")
        # 图例放左下（起始段曲线在上方，左下为空），避开最佳 epoch 竖线与标记
        ax.legend(loc="lower left", fontsize=9.5)

        ax2 = axes[1]
        if np.isfinite(va).any():
            gap = va - tr
            ax2.plot(ep, gap, color=PALETTE["purple"], lw=1.7, label="验证 − 训练")
            ax2.fill_between(ep, 0, gap, color=PALETTE["purple"], alpha=0.12, linewidth=0)
            k = int(np.nanargmin(va))
            ax2.axvline(ep[k], color=MUTED, ls=":", lw=1.2)
            # 标注固定在轴内顶部（不会溢出框外，也不会压曲线）
            ax2.annotate(f"最佳 epoch {int(ep[k])}", xy=(ep[k], 1.0),
                         xycoords=("data", "axes fraction"),
                         xytext=(-6, -14), textcoords="offset points",
                         fontsize=9.5, color=MUTED, ha="right", va="top")
            # 泛化间隙跨两个数量级（初期未收敛 ~-0.26，后期 ~-0.01）→ symlog 才看得见结构
            ax2.set_yscale("symlog", linthresh=1e-2, linscale=0.6)
            ax2.set_ylim(bottom=min(0.0, float(np.nanmin(gap)) * 1.15))
            ax2.axhline(0.0, color="#C9D0D8", lw=0.9)
        else:
            ax2.plot(ep, tr, color=PALETTE["blue"], lw=1.7)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("泛化间隙（symlog 刻度）")
        ax2.set_title("泛化间隙（过拟合监控）")
        ax2.legend(loc="lower left", fontsize=9.5)

        _header(fig, title, subtitle, note)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return save_figure(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_training_history 失败: %s", exc)
        plt.close("all")
        return None


# ------------------------------------------------------------- 残差诊断 ----


def plot_residual_diagnostics(
    y_true,
    y_pred,
    target_names: list[str],
    out_path,
    *,
    title: str = "残差诊断",
    subtitle: str = "",
    note: str = "",
    unit: str = "kg kg⁻¹",
) -> Path | None:
    """四联残差诊断：直方图+正态拟合 / Q-Q / 残差-预测散点 / 分位箱误差。"""
    try:
        apply_style()
        yt = _as_2d(y_true)
        yp = _as_2d(y_pred)
        if yt.shape != yp.shape or yt.size == 0:
            logger.warning("plot_residual_diagnostics 形状不一致或为空")
            return None
        names = _names(target_names, yt.shape[1])
        n = yt.shape[1]
        fig, axes = plt.subplots(n, 4, figsize=(15.6, 3.9 * n), squeeze=False)
        for j in range(n):
            y, p = yt[:, j], yp[:, j]
            ok = np.isfinite(y) & np.isfinite(p)
            y, p = y[ok], p[ok]
            if y.size == 0:
                continue
            r = p - y
            _, scale = scaled_values(np.concatenate([y, p]))
            factor = scale.factor
            color = _TARGET_COLORS[j % len(_TARGET_COLORS)]

            # (1) 直方图 + 正态拟合（密度与直方图**同一坐标**：PDF 需乘 factor 换算）
            ax = axes[j][0]
            rs = r / factor
            ax.hist(rs, bins="auto", color=color, alpha=0.72,
                    edgecolor="white", linewidth=0.6, density=True, label="经验分布")
            mu, sd = float(np.mean(r)), float(np.std(r))
            xs = np.linspace(r.min(), r.max(), 300)
            pdf = np.exp(-0.5 * ((xs - mu) / (sd or 1.0)) ** 2) / (sd * math.sqrt(2 * math.pi))
            ax.plot(xs / factor, pdf * factor, color=PALETTE["vermillion"], lw=1.7,
                    label="正态拟合")
            ax.axvline(0.0, color=INK, lw=1.0, ls="--", label="零偏差")
            ax.set_xlabel(f"残差 (预测 − 真值) [{scale.label} {unit}]")
            ax.set_ylabel("概率密度")
            ax.set_title(f"{names[j]} · 残差分布")
            ax.legend(loc="upper right", fontsize=8.8)

            # (2) Q-Q（正态参考）
            ax = axes[j][1]
            obs = np.sort(r / factor)
            p_i = (np.arange(1, obs.size + 1) - 0.5) / obs.size
            theo = _norm_ppf(p_i)
            ax.scatter(theo, obs, s=12, alpha=0.55, color=color, linewidths=0,
                       rasterized=True)
            if obs.size > 2:
                k, c = np.polyfit(theo, obs, 1)
                lim = [float(theo.min()), float(theo.max())]
                ax.plot(lim, [k * v + c for v in lim], color=INK, lw=1.2, ls="--",
                        label="正态参考")
            ax.set_xlabel("理论分位数（标准正态）")
            ax.set_ylabel("样本分位数")
            ax.set_title(f"{names[j]} · Q-Q 图")
            ax.legend(loc="upper left")

            # (3) 残差 vs 预测
            ax = axes[j][2]
            ax.scatter(p / factor, r / factor, s=12, alpha=0.4, color=color,
                       linewidths=0, rasterized=True, label="样本")
            nb = 12
            if p.size > nb:
                edges = np.quantile(p, np.linspace(0, 1, nb + 1))
                idx = np.clip(np.digitize(p, edges[1:-1]), 0, nb - 1)
                for b in range(nb):
                    m = idx == b
                    if m.sum() >= 3:
                        ax.plot(np.array([edges[b], edges[b + 1]]) / factor,
                                np.array([np.median(r[m])] * 2) / factor,
                                color=PALETTE["vermillion"], lw=1.8)
                ax.plot([], [], color=PALETTE["vermillion"], lw=1.8,
                        label="分箱残差中位数")
            ax.axhline(0.0, color=INK, lw=1.0, ls="--", label="零偏差")
            ax.set_xlabel(f"预测值 [{scale.label} {unit}]")
            ax.set_ylabel(f"残差 [{scale.label} {unit}]")
            ax.set_title(f"{names[j]} · 残差 vs 预测")
            ax.legend(loc="upper right", fontsize=8.8)

            # (4) 分位分箱误差（异方差诊断）
            ax = axes[j][3]
            if p.size >= 20:
                qs = np.quantile(p, np.linspace(0, 1, 11))
                centers, rmses = [], []
                for b in range(10):
                    m = (p >= qs[b]) & (p <= qs[b + 1] if b == 9 else p < qs[b + 1])
                    if m.sum() >= 3:
                        centers.append(float(np.mean(p[m])))
                        rmses.append(float(np.sqrt(np.mean(r[m] ** 2))))
                ax.plot(np.array(centers) / factor, np.array(rmses) / factor,
                        color=color, marker="o", lw=1.8)
                ax.set_xlabel(f"预测值分箱均值 [{scale.label} {unit}]")
                ax.set_ylabel(f"箱内 RMSE [{scale.label} {unit}]")
            ax.set_title(f"{names[j]} · 误差 vs 预测量级")

        # 同一列跨行统一量程（避免上下行刻度不同造成误读）
        if n > 1:
            for col in (1, 3):
                lo = min(float(axes[k][col].get_ylim()[0]) for k in range(n))
                hi = max(float(axes[k][col].get_ylim()[1]) for k in range(n))
                for k in range(n):
                    axes[k][col].set_ylim(lo, hi)
            lo = max(float(axes[k][1].get_xlim()[0]) for k in range(n))
            hi = min(float(axes[k][1].get_xlim()[1]) for k in range(n))
            for k in range(n):
                axes[k][1].set_xlim(lo, hi)

        _header(fig, title, subtitle, note)
        fig.tight_layout(rect=(0, 0, 1, 0.93 if n > 1 else 0.90))
        return save_figure(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_residual_diagnostics 失败: %s", exc)
        plt.close("all")
        return None


# ------------------------------------------------------------- 分组泛化 ----


def _collect_series(block: dict) -> dict[str, float]:
    """从分组指标块里取 {系列: rmse} 与 {系列: r2}。"""
    rmse: dict[str, float] = {}
    r2: dict[str, float] = {}
    if not isinstance(block, dict):
        return {"rmse": rmse, "r2": r2}
    src = block.get("per_target") if isinstance(block.get("per_target"), dict) else None
    if src:
        for nm, blk in src.items():
            if not isinstance(blk, dict):
                continue
            if isinstance(blk.get("rmse"), (int, float)):
                rmse[str(nm)] = float(blk["rmse"])
            if isinstance(blk.get("r2"), (int, float)):
                r2[str(nm)] = float(blk["r2"])
        return {"rmse": rmse, "r2": r2}
    ov = block.get("overall") if isinstance(block.get("overall"), dict) else block
    if isinstance(ov.get("rmse"), (int, float)):
        rmse["overall"] = float(ov["rmse"])
    if isinstance(ov.get("r2"), (int, float)):
        r2["overall"] = float(ov["r2"])
    return {"rmse": rmse, "r2": r2}


def plot_subgroup_performance(
    subgroups: dict[str, Any],
    out_path,
    *,
    title: str = "分组泛化性能",
    subtitle: str = "",
    note: str = "",
    unit: str = "kg kg⁻¹",
    dim_labels: dict[str, str] | None = None,
) -> Path | None:
    """按 year / season / region 分组的 RMSE 与 R² 对比（分组柱状图）。"""
    try:
        apply_style()
        if not isinstance(subgroups, dict) or not subgroups:
            logger.warning("plot_subgroup_performance: 空分组指标")
            return None
        labels = {"year": "年份", "season": "季节", "region": "区域"}
        labels.update(dim_labels or {})

        dims: list[tuple[str, list[str], list[dict]]] = []
        for dim, block in subgroups.items():
            if not isinstance(block, dict) or "skipped" in block:
                continue
            names: list[str] = []
            rows: list[dict] = []
            for gname, blk in block.items():
                ser = _collect_series(blk)
                if not ser["rmse"] and not ser["r2"]:
                    continue
                names.append(str(gname))
                rows.append(ser)
            if len(names) >= 2:
                dims.append((str(dim), names, rows))
        if not dims:
            logger.warning("plot_subgroup_performance: 无可画分组（每维至少需要 2 组）")
            return None

        rows_all = [r for _, _, rows in dims for r in rows]
        fig, axes = plt.subplots(
            len(dims), 2, figsize=(13.0, 4.0 * len(dims)), squeeze=False,
        )
        # 系列顺序：按目标在分组指标里出现的先后（与 target_names 一致），不排序
        series: list[str] = []
        for r in rows_all:
            for k in r["rmse"]:
                if k not in series:
                    series.append(k)
        for i, (dim, names, rows) in enumerate(dims):
            x = np.arange(len(names))
            width = 0.78 / max(len(series), 1)

            ax = axes[i][0]
            all_rmse = [v for r in rows for v in r["rmse"].values()]
            if all_rmse:
                _, scale = scaled_values(all_rmse)
                factor = scale.factor
                for k, s in enumerate(series):
                    ys = [r["rmse"].get(s, np.nan) / factor for r in rows]
                    ax.bar(x + k * width - 0.39 + width / 2, ys, width=width * 0.92,
                           label=s, color=_TARGET_COLORS[k % len(_TARGET_COLORS)],
                           alpha=0.9, edgecolor="white", linewidth=0.8)
                    # 柱顶标数值（零基线柱状图 + 精确读数）
                    for xi, yv in zip(x, ys):
                        if np.isfinite(yv):
                            ax.text(xi + k * width - 0.39 + width / 2, yv,
                                    f"{yv:.3f}", ha="center", va="bottom",
                                    fontsize=7.8, color=INK)
                ax.set_ylim(0, max(all_rmse) / factor * 1.18)
                ax.set_ylabel(f"RMSE [{scale.label} {unit}]")
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            ax.set_xlim(-0.55, len(names) - 0.45)   # 与右侧点棒图对齐
            ax.set_title(f"{labels.get(dim, dim)} · RMSE")

            # R² 用**棒棒糖图**而非截断基线柱状图：
            # R² 全都在 0.96–0.98，零基线柱会挤成一条线，
            # 截断基线柱又会夸大差异（审稿人最忌讳的可视化误导）。
            ax2 = axes[i][1]
            all_r2 = [v for r in rows for v in r["r2"].values()]
            if all_r2:
                for k, s in enumerate(series):
                    ys = [r["r2"].get(s, np.nan) for r in rows]
                    xs = x + k * width - 0.39 + width / 2
                    color = _TARGET_COLORS[k % len(_TARGET_COLORS)]
                    base = min(all_r2) - 0.02
                    for xi, yv in zip(xs, ys):
                        if not np.isfinite(yv):
                            continue
                        ax2.plot([xi, xi], [base, yv], color=color, lw=1.6, alpha=0.55,
                                 solid_capstyle="round")
                        ax2.scatter([xi], [yv], s=62, color=color, zorder=6,
                                    edgecolor="white", linewidth=1.2)
                        ax2.annotate(f"{yv:.3f}", xy=(xi, yv), xytext=(0, 8),
                                     textcoords="offset points", ha="center",
                                     fontsize=8.4, color=INK)
                lo = min(all_r2)
                hi = max(all_r2)
                pad = max((hi - lo) * 0.25, 0.004)
                ax2.set_ylim(lo - pad, hi + pad * 1.6)
                ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
                ax2.set_ylabel("R²（点棒图 · 纵轴已截断，数值以标注为准）")
            ax2.set_xticks(x)
            ax2.set_xticklabels(names)
            ax2.set_xlim(-0.55, len(names) - 0.45)  # 与左侧柱状图对齐
            ax2.set_title(f"{labels.get(dim, dim)} · R²")

        # 图级统一图例（副标题下方），避免每个子图各放一份压住标题
        handles = [
            plt.Line2D([], [], marker="s", ls="", ms=9,
                       color=_TARGET_COLORS[k % len(_TARGET_COLORS)], label=s)
            for k, s in enumerate(series)
        ]
        if handles:
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.928),
                       ncols=len(handles), fontsize=9.5, frameon=False)
        _header(fig, title, subtitle, note)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        return save_figure(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_subgroup_performance 失败: %s", exc)
        plt.close("all")
        return None


# --------------------------------------------------------------- 空间场 ----


def plot_spatial_maps(
    longitude,
    latitude,
    y_true,
    y_pred,
    target_names: list[str],
    out_path,
    *,
    region_boxes: dict[str, list[float]] | None = None,
    title: str = "空间分布：真值 / 预测 / 偏差（按目标分行）",
    subtitle: str = "",
    note: str = "",
    unit: str = "kg kg⁻¹",
) -> Path | None:
    """真值-预测-偏差三联空间图（hexbin 均值聚合 + 区域盒标注）。"""
    try:
        apply_style()
        lon = np.asarray(longitude, dtype=np.float64).ravel()
        lat = np.asarray(latitude, dtype=np.float64).ravel()
        yt = _as_2d(y_true)
        yp = _as_2d(y_pred)
        if yt.shape != yp.shape or lon.size != lat.size or lon.size != yt.shape[0]:
            logger.warning("plot_spatial_maps 输入形状不匹配")
            return None
        names = _names(target_names, yt.shape[1])
        ok = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(yt).all(1) & np.isfinite(yp).all(1)
        lon, lat = lon[ok], lat[ok]
        yt, yp = yt[ok], yp[ok]
        if lon.size == 0:
            logger.warning("plot_spatial_maps: 无有效样本")
            return None

        # 经度跨度 / 纬度跨度 → 决定色块数量与画布比例（避免出现大片空白）
        lon_span = float(lon.max() - lon.min()) or 1.0
        lat_span = float(lat.max() - lat.min()) or 1.0
        mean_lat = float(np.mean(lat))
        # 等距圆柱投影下的经度压缩：1° 经度 ≈ cos(纬度) × 1° 纬度
        aspect = 1.0 / max(math.cos(math.radians(mean_lat)), 0.2)
        panel_h = 2.0 + 2.4 / max(lon_span * aspect / max(lat_span, 1e-6), 0.6)
        n_t = yt.shape[1]
        # 每个目标一行（真值 / 预测 / 偏差），避免"只画了第一个目标"的歧义
        fig, axes = plt.subplots(n_t, 3, figsize=(15.6, panel_h * n_t + 0.9),
                                 squeeze=False)

        # 网格按数据长宽比铺开：每个色块聚合若干样本，避免"满屏碎点"
        nx = 34
        ny = max(6, int(round(nx * lat_span / (lon_span * aspect))))
        pad_x = lon_span * 0.02
        pad_y = lat_span * 0.04
        for j in range(n_t):
            truth, pred = yt[:, j], yp[:, j]
            allv = np.concatenate([truth, pred])
            _, scale = scaled_values(allv)
            factor = scale.factor
            vmin = float(allv.min()) / factor
            vmax = float(allv.max()) / factor
            err = (pred - truth) / factor
            emax = float(np.percentile(np.abs(err), 98)) or 1e-12
            panels = [
                (truth / factor, f"{names[j]} · 真值", DENSITY_CMAP, vmin, vmax),
                (pred / factor, f"{names[j]} · 预测", DENSITY_CMAP, vmin, vmax),
                (err, f"{names[j]} · 偏差（预测 − 真值）", DIVERGING_CMAP, -emax, emax),
            ]
            for ax, (vals, lab, cmap, lo, hi) in zip(axes[j], panels):
                hb = ax.hexbin(lon, lat, C=vals, gridsize=(nx, ny), reduce_C_function=np.mean,
                               cmap=cmap, mincnt=1,
                               extent=(lon.min(), lon.max(), lat.min(), lat.max()),
                               linewidths=0.0, rasterized=True)
                cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
                cb.outline.set_visible(False)
                cb.ax.tick_params(labelsize=9, color="#C9D0D8")
                # 中文色标标签水平置顶（竖排 90° 的中文难以阅读）
                cb.ax.set_title(f"{lab.split('· ')[-1]} [{scale.label} {unit}]",
                                fontsize=9.2, color=INK, pad=7)
                for rname, box in (region_boxes or {}).items():
                    try:
                        lon0, lon1, lat0, lat1 = [float(v) for v in box]
                    except Exception:  # noqa: BLE001
                        continue
                    ax.add_patch(plt.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0,
                                               fill=False, edgecolor=INK, lw=1.2, ls="--",
                                               zorder=5))
                    # 区域名一律放到**坐标区顶部（数据域之外）**，再用细引线连回区域盒，
                    # 保证任何区域名都不会压住数据格点
                    x_mid = lon0 + (lon1 - lon0) * 0.5
                    ax.set_xlim(lon.min() - pad_x, lon.max() + pad_x)
                    ax.set_ylim(lat.min() - pad_y, lat.max() + pad_y * 3.2)
                    ax.plot([x_mid, x_mid], [lat1, ax.get_ylim()[1]],
                            color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=4)
                    ax.annotate(
                        rname, xy=(x_mid, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 5), textcoords="offset points",
                        fontsize=8.6, color=INK, ha="center", va="bottom", zorder=7,
                        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none",
                              "pad": 1.2},
                    )
                ax.set_xlabel("经度 (°E)")
                ax.set_ylabel("纬度 (°N)")
                # 标题上移，给「坐标区顶部之外」的区域名标注让位
                ax.set_title(lab, pad=23)
                ax.set_aspect(aspect, adjustable="box")
                ax.grid(True, color="#EDF1F5", linewidth=0.8)

        _header(fig, title, subtitle, note)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        return save_figure(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_spatial_maps 失败: %s", exc)
        plt.close("all")
        return None


# --------------------------------------------------------------- 鲁棒性 ----


def plot_robustness(
    robustness: dict[str, Any],
    out_path,
    *,
    title: str = "噪声鲁棒性",
    subtitle: str = "",
    note: str = "",
    unit: str = "kg kg⁻¹",
) -> Path | None:
    """不同噪声强度下的 RMSE / R² 退化曲线。"""
    try:
        apply_style()
        if not isinstance(robustness, dict) or not robustness:
            logger.warning("plot_robustness: 空数据")
            return None
        levels = sorted(float(k) for k in robustness)
        rmse_overall, r2_overall = [], []
        per_target: dict[str, list[float]] = {}
        for lv in levels:
            blk = robustness.get(str(lv), robustness.get(lv))
            ser = _collect_series(blk) if isinstance(blk, dict) else {"rmse": {}, "r2": {}}
            ov = blk.get("overall", {}) if isinstance(blk, dict) else {}
            rmse_overall.append(float(ov.get("rmse", np.nan)) if isinstance(ov, dict) else np.nan)
            r2_overall.append(float(ov.get("r2", np.nan)) if isinstance(ov, dict) else np.nan)
            for nm, v in ser["rmse"].items():
                per_target.setdefault(nm, []).append(v)
        if not np.isfinite(np.asarray(rmse_overall, dtype=float)).any():
            logger.warning("plot_robustness: 无有效 RMSE")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
        ax = axes[0]
        rmse_all = np.asarray(rmse_overall, dtype=float)
        _, scale = scaled_values(rmse_all[np.isfinite(rmse_all)])
        factor = scale.factor
        # 「整体」用中性色，避免与逐目标同色（黑白打印也要能区分）
        ax.plot(levels, rmse_all / factor, marker="o", color=MUTED, lw=2.4,
                label="整体", zorder=5)
        for k, (nm, vals) in enumerate(per_target.items()):
            v = np.asarray(vals, dtype=float)
            ax.plot(levels, v / factor, marker="s", lw=1.6, ls="--",
                    color=_TARGET_COLORS[k % len(_TARGET_COLORS)], label=nm, ms=5)
        ax.set_xlabel("输入噪声水平（相对标准差）")
        ax.set_ylabel(f"RMSE [{scale.label} {unit}]")
        ax.set_title("误差随噪声增大")
        ax.legend(loc="upper left")
        lo = float(np.nanmin(rmse_all)) / factor
        hi = float(np.nanmax(rmse_all)) / factor
        # y 轴范围必须覆盖**所有**系列（整体 + 各目标），否则逐目标曲线会被裁掉
        all_series = list(rmse_all[np.isfinite(rmse_all)] / factor)
        for vals in per_target.values():
            vv = np.asarray(vals, dtype=float)
            all_series.extend((vv[np.isfinite(vv)] / factor).tolist())
        if all_series:
            lo_a, hi_a = min(all_series), max(all_series)
            pad_a = max((hi_a - lo_a) * 0.12, 1e-9)
            ax.set_ylim(lo_a - pad_a, hi_a + pad_a * 1.6)

        ax2 = axes[1]
        r2_arr = np.asarray(r2_overall, dtype=float)
        ax2.plot(levels, r2_arr, marker="o", color=MUTED, lw=2.4, label="整体 R²",
                 zorder=5)
        for xi, yi in zip(levels, r2_overall):
            if math.isfinite(yi):
                # 首个点向右偏移，避开 y 轴刻度标签
                dx = 11 if xi == levels[0] else 0
                ax2.annotate(f"{yi:.4f}", xy=(xi, yi), xytext=(dx, 10),
                             textcoords="offset points",
                             ha="left" if dx else "center", fontsize=9, color=INK)
        ax2.set_xlabel("输入噪声水平（相对标准差）")
        ax2.set_ylabel("R²")
        ax2.set_title("拟合优度随噪声增大")
        finite = r2_arr[np.isfinite(r2_arr)]
        if finite.size:
            lo2 = float(finite.min())
            ax2.set_ylim(lo2 - (1.0 - lo2) * 0.35, 1.0)

        _header(fig, title, subtitle, note)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        return save_figure(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_robustness 失败: %s", exc)
        plt.close("all")
        return None


# --------------------------------------------------------- 量级-误差曲线 ----


def plot_error_by_magnitude(
    y_true,
    y_pred,
    target_names: list[str],
    out_path,
    *,
    title: str = "误差随真值量级的变化",
    subtitle: str = "",
    note: str = "",
    unit: str = "kg kg⁻¹",
) -> Path | None:
    """按真值分位分箱的 RMSE 曲线（判断误差是否随量级放大）。"""
    try:
        apply_style()
        yt = _as_2d(y_true)
        yp = _as_2d(y_pred)
        if yt.shape != yp.shape or yt.size == 0:
            return None
        names = _names(target_names, yt.shape[1])
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        allv = np.concatenate([yt.ravel(), yp.ravel()])
        _, scale = scaled_values(allv)
        factor = scale.factor
        for j in range(yt.shape[1]):
            y, p = yt[:, j], yp[:, j]
            ok = np.isfinite(y) & np.isfinite(p)
            y, p = y[ok], p[ok]
            if y.size < 20:
                continue
            r = p - y
            qs = np.quantile(y, np.linspace(0, 1, 11))
            cx, cy = [], []
            for b in range(10):
                m = (y >= qs[b]) & (y <= qs[b + 1] if b == 9 else y < qs[b + 1])
                if m.sum() >= 3:
                    cx.append(float(np.mean(y[m])) / factor)
                    cy.append(float(np.sqrt(np.mean(r[m] ** 2))) / factor)
            ax.plot(cx, cy, marker="o", lw=1.9,
                    color=_TARGET_COLORS[j % len(_TARGET_COLORS)], label=names[j])
        ax.set_xlabel(f"真值分箱均值 [{scale.label} {unit}]")
        ax.set_ylabel(f"箱内 RMSE [{scale.label} {unit}]")
        ax.set_title("分箱 RMSE")
        ax.legend(loc="upper left")
        _header(fig, title, subtitle, note)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        return save_figure(fig, out_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("plot_error_by_magnitude 失败: %s", exc)
        plt.close("all")
        return None
