"""可视化（Agg 后端、无 plt.show()、savefig dpi=150、异常降 None、返回 Path）。

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
from .spatial_maps import plot_scatter_map
from .training_curves import plot_loss_curve

__all__ = [
    "plot_loss_curve",
    "plot_prediction_vs_truth",
    "plot_residual_distribution",
    "plot_feature_importance",
    "plot_subgroup_rmse",
    "plot_scatter_map",
]
