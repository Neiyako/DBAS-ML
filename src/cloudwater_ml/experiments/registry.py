"""实验配方注册表：命名实验入口与 A~E 真实/仿真配比登记。

这里只登记**实验设计**（配置层），不代表任何实验已运行或已有结论；
A~E 五组配比是预留实验矩阵，须接入真实数据后才可执行。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "MIX_RECIPES",
    "register_experiment",
    "available_experiments",
    "mix_recipe",
    "list_experiment_configs",
]

# 任务书 §8 的实验矩阵：训练集中真实/仿真数据配比（仅作用于训练集）
MIX_RECIPES: dict[str, dict[str, float]] = {
    "A": {"real_fraction": 1.00},
    "B": {"real_fraction": 0.75},
    "C": {"real_fraction": 0.50},
    "D": {"real_fraction": 0.25},
    "E": {"real_fraction": 0.00},
}

EXPERIMENT_REGISTRY: dict[str, dict[str, Any]] = {}


def register_experiment(name: str, description: str, *, runnable: bool = True,
                        default_config: str | None = None) -> None:
    EXPERIMENT_REGISTRY[name] = {
        "description": description,
        "runnable": runnable,
        "default_config": default_config,
    }


def available_experiments() -> list[str]:
    return sorted(EXPERIMENT_REGISTRY)


def mix_recipe(name: str) -> dict[str, float]:
    name = name.upper()
    if name not in MIX_RECIPES:
        raise KeyError(f"未知配比实验 {name!r}，可选: {sorted(MIX_RECIPES)}")
    return dict(MIX_RECIPES[name])


def list_experiment_configs(configs_dir: str | Path = "configs/experiments") -> dict[str, Path]:
    """枚举 configs/experiments/*.yaml（CLI 用于列出可运行实验）。"""
    root = Path(configs_dir)
    if not root.exists():
        return {}
    return {p.stem: p for p in sorted(root.glob("*.yaml"))}


# 内置配方登记（A~E 无真实数据时 runnable=False，防止误跑出“结论”）
register_experiment(
    "mix_A", "真实:仿真 = 100%:0%（需真实数据）", runnable=False
)
register_experiment("mix_B", "真实:仿真 = 75%:25%（需真实数据）", runnable=False)
register_experiment("mix_C", "真实:仿真 = 50%:50%（需真实数据）", runnable=False)
register_experiment("mix_D", "真实:仿真 = 25%:75%（需真实数据）", runnable=False)
register_experiment("mix_E", "真实:仿真 = 0%:100%（纯仿真）", runnable=True)
