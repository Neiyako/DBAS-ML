"""学习率调度工厂：``null`` | ``cosine`` | ``step``。"""
from __future__ import annotations

from typing import Any

import torch

__all__ = ["build_scheduler"]


def build_scheduler(
    name: str | None, optimizer: torch.optim.Optimizer, cfg: dict[str, Any] | None = None
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """按配置构造学习率调度器；``None``/空串/``null`` 返回 None（不调度）。

    cfg 读取：``epochs``（cosine 的 T_max、step 的默认 step_size 依据）、
    ``eta_min``（cosine）、``step_size``/``gamma``（step）。
    """
    cfg = cfg or {}
    # 兼容两种传入：training 段本身，或含 {"training": {...}} 的完整配置
    if isinstance(cfg.get("training"), dict):
        cfg = cfg["training"]
    key = None if name in (None, "", "null", "none") else str(name).lower()
    if key is None:
        return None
    epochs = max(1, int(cfg.get("epochs", 100)))

    if key == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(cfg.get("eta_min", 0.0))
        )
    if key == "step":
        step_size = int(cfg.get("step_size", max(1, epochs // 3)))
        gamma = float(cfg.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"未知调度器 {name!r}，可选: null | cosine | step")
