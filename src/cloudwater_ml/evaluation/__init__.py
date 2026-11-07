"""评价与泛化分析（metrics / subgroups / robustness / uncertainty / 编排入口）。

口径单一来源：:mod:`cloudwater_ml.evaluation.metrics` 的 ``METRIC_DEFINITIONS``
（公式、单位、适用条件），不得另立口径。
"""
from __future__ import annotations

from .evaluator import evaluate_predictions
from .generalization import subgroup_metrics
from .metrics import METRIC_DEFINITIONS, regression_metrics
from .robustness import noise_robustness
from .uncertainty import interval_metrics

__all__ = [
    "METRIC_DEFINITIONS",
    "regression_metrics",
    "subgroup_metrics",
    "noise_robustness",
    "interval_metrics",
    "evaluate_predictions",
]
