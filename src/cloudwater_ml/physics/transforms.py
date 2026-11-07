"""硬约束用的输出变换（投影 / 重参数化）。

这些变换只保证**数值性质**（如非负），不代表任何未验证的物理假设；
是否采用由调用方显式决定。
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["softplus", "clamp_projection"]


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """数值稳定的 softplus：``log(1 + exp(beta·x)) / beta``（恒 ≥ 0）。

    当 ``beta·x`` 超过 ``threshold`` 时直接返回 ``x``（此时
    ``log1p(exp(beta·x))/beta ≈ x``，避免指数溢出）。

    Parameters
    ----------
    x : Tensor
        任意形状张量。
    beta : float
        陡度系数，必须为正。
    threshold : float
        切换阈值（作用在 ``beta·x`` 上）。
    """
    if beta <= 0:
        raise ValueError(f"beta 必须为正，收到 {beta!r}")
    z = x * beta
    safe = torch.clamp(z, max=threshold)
    out = torch.log1p(torch.exp(safe)) / beta
    return torch.where(z > threshold, x, out)


def clamp_projection(x: Tensor, min_value: float = 0.0) -> Tensor:
    """把小于 ``min_value`` 的分量截断到 ``min_value``（投影到可行集）。"""
    return torch.clamp(x, min=min_value)
