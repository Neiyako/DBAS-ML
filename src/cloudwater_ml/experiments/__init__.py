"""实验层：编排、配方注册、结果追踪。"""
from .registry import MIX_RECIPES, available_experiments, list_experiment_configs, mix_recipe
from .tracking import (
    DEFAULT_OUTPUTS,
    build_comparison_table,
    create_experiment_dir,
    load_metrics,
    write_report,
)

__all__ = [
    "run_experiment",
    "evaluate_saved_experiment",
    "MIX_RECIPES",
    "available_experiments",
    "list_experiment_configs",
    "mix_recipe",
    "DEFAULT_OUTPUTS",
    "build_comparison_table",
    "create_experiment_dir",
    "load_metrics",
    "write_report",
]


def __getattr__(name: str):
    # runner 依赖 models/evaluation/visualization（队友并行开发中），延迟导入避免环
    if name in ("run_experiment", "evaluate_saved_experiment"):
        from .runner import evaluate_saved_experiment, run_experiment

        return {"run_experiment": run_experiment,
                "evaluate_saved_experiment": evaluate_saved_experiment}[name]
    raise AttributeError(name)
