"""可视化（Agg 后端、无 plt.show()、异常降 None、返回 Path）。

两套图层：

- **基础图**（``prediction_plots`` / ``training_curves`` / ``error_analysis`` / ``spatial_maps``）
  ——训练主流程每次自动产出的速览图，接口保持不变；
- **出版级图**（``publication``）——报告用插图：密度散点、残差诊断、分组泛化、
  空间三联、鲁棒性、量级-误差；配色/字体/量级刻度统一由 ``style`` 提供。

所有绘图函数都遵守同一契约：失败返回 None 而不是抛异常，保证训练主流程
不因出图失败而中断。
"""
from __future__ import annotations

from .error_analysis import plot_subgroup_rmse
from .prediction_plots import (
    plot_feature_importance,
    plot_prediction_vs_truth,
    plot_residual_distribution,
)
from .publication import (
    plot_error_by_magnitude,
    plot_parity_density,
    plot_residual_diagnostics,
    plot_robustness,
    plot_spatial_maps,
    plot_subgroup_performance,
    plot_training_history,
)
from .spatial_maps import plot_scatter_map
from .style import PALETTE, apply_style
from .training_curves import plot_loss_curve

__all__ = [
    "plot_loss_curve",
    "plot_prediction_vs_truth",
    "plot_residual_distribution",
    "plot_feature_importance",
    "plot_subgroup_rmse",
    "plot_scatter_map",
    # 出版级图
    "plot_parity_density",
    "plot_training_history",
    "plot_residual_diagnostics",
    "plot_subgroup_performance",
    "plot_spatial_maps",
    "plot_robustness",
    "plot_error_by_magnitude",
    # 设计系统
    "apply_style",
    "PALETTE",
]
