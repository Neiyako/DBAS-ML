"""随机种子与可复现性。"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

__all__ = ["set_seed", "get_rng", "capture_rng_state", "restore_rng_state"]


def set_seed(seed: int = 42, *, deterministic: bool = False, with_torch: bool = True) -> None:
    """为 random / numpy / torch（若已安装）统一种子。

    deterministic=True 时额外要求 torch 确定性算法（可能变慢，仅在需要严格复现时开）。
    with_torch=False 时完全不 import torch —— 跑 LightGBM 实验时必须这样做
    （torch 与 LightGBM 的 OpenMP 运行库同进程会崩，见 models/base.py 说明）。
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if not with_torch:
        return
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CPU 环境不走
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass  # 纯 sklearn 实验无 torch 也可复现


def get_rng(seed: int = 42) -> np.random.Generator:
    """返回独立的 NumPy 随机生成器（不影响全局状态）。"""
    return np.random.default_rng(int(seed))


# ----------------------------------------------------------------------
# 随机数状态快照（断点续训的连续性依赖它）
# ----------------------------------------------------------------------
def capture_rng_state() -> dict[str, Any]:
    """捕获当前全部随机数状态（python / numpy / torch / cuda）。

    用于检查点：恢复训练时若能还原 RNG 状态，数据 shuffle 顺序与 dropout
    掩码才能真正连续（否则「续训」只是「换个初始点重跑」）。
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": None,
        "torch_cuda": None,
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():  # pragma: no cover - CPU 环境不走
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:  # pragma: no cover
        pass
    return state


def restore_rng_state(state: dict[str, Any] | None) -> bool:
    """恢复 :func:`capture_rng_state` 捕获的状态；返回是否真的恢复了。"""
    if not state:
        return False
    try:
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
    except Exception:
        return False
    try:
        import torch

        if state.get("torch") is not None:
            # 检查点可能以 map_location=<device> 读回；RNG 状态必须是 CPU 上的 ByteTensor
            torch.set_rng_state(state["torch"].cpu())
        if state.get("torch_cuda") and torch.cuda.is_available():  # pragma: no cover
            torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    except ImportError:  # pragma: no cover
        pass
    return True
