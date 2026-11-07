"""优化器工厂：``adam`` | ``adamw`` | ``sgd``（momentum 可配）。"""
from __future__ import annotations

from typing import Any

import torch

__all__ = ["build_optimizer"]


def build_optimizer(name: str, model: Any, cfg: dict[str, Any] | None = None) -> torch.optim.Optimizer:
    """按配置构造优化器。

    Parameters
    ----------
    name : adam | adamw | sgd
    model : torch.nn.Module（或任意带 ``parameters()`` 的对象）
    cfg : 至少读 ``learning_rate``、``weight_decay``；sgd 另读 ``momentum``
    """
    cfg = cfg or {}
    lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0) or 0.0)
    params = model.parameters() if hasattr(model, "parameters") else model
    key = str(name or "adamw").lower()

    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"未知优化器 {name!r}，可选: adam | adamw | sgd")
