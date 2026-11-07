"""检查点：保存/恢复训练状态，CheckpointManager 管理 best.pt / last.pt。

checkpoint 内容（键）：``model_state``、``optimizer_state``、``scheduler_state``、
``scaler_state``（GradScaler）、``epoch``、``global_step``、``best_val_loss``、
``rng_state``（python/numpy/torch/cuda 随机数状态）、``history``（逐 epoch 损失
记录）、``trainer_config``（Trainer 配置快照）、``extra``。

v2.0 起检查点是**可续训状态**的完整快照：恢复后学习率调度、损失缩放、
数据顺序与 dropout 随机流都能连续（见 ``utils/reproducibility.capture_rng_state``）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..utils import get_logger

__all__ = ["save_checkpoint", "load_checkpoint", "CheckpointManager"]

_logger = get_logger("training.checkpoint")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    best_val: float | None = None,
    extra: dict[str, Any] | None = None,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    global_step: int = 0,
    rng_state: dict[str, Any] | None = None,
) -> Path:
    """把完整训练状态写到 ``path``（单文件 .pt）。

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler.LRScheduler | None
        学习率调度器（保存 ``state_dict``）。
    scaler : torch.amp.GradScaler | None
        混合精度的损失缩放器（保存 ``state_dict``）。
    global_step : int
        全局优化步数（跨 epoch 累计）。
    rng_state : dict | None
        随机数状态快照（python/numpy/torch/cuda）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(extra or {})
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": None if best_val is None else float(best_val),
        "best_val_loss": None if best_val is None else float(best_val),
        "rng_state": rng_state,
        "history": extra.pop("history", None),
        "early_stopping_state": extra.pop("early_stopping_state", None),
        "trainer_config": extra.pop("trainer_config", None),
        "extra": extra,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """读取 checkpoint（map_location 默认 cpu 兼容跨设备），可顺带回填 model/optimizer。

    返回 payload dict；``model``/``optimizer`` 为 None 时只读不回填。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # 老版本 torch 无 weights_only 参数
        payload = torch.load(path, map_location=map_location)
    if model is not None and payload.get("model_state") is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


class CheckpointManager:
    """管理 ``dir/best.pt`` 与 ``dir/last.pt``；``dir=None`` 时只在内存里保留最新状态。"""

    def __init__(self, dir: str | Path | None = None) -> None:
        self.dir = Path(dir) if dir is not None else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, dict[str, Any]] = {}

    # -- 路径 ----------------------------------------------------------
    @property
    def best_path(self) -> Path | None:
        return self.dir / "best.pt" if self.dir else None

    @property
    def last_path(self) -> Path | None:
        return self.dir / "last.pt" if self.dir else None

    # -- 写 ------------------------------------------------------------
    def save(
        self,
        kind: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        best_val: float | None = None,
        extra: dict[str, Any] | None = None,
        *,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        global_step: int = 0,
        rng_state: dict[str, Any] | None = None,
    ) -> Path | None:
        """kind: ``best`` | ``last``。有目录则落盘，同时更新内存镜像。"""
        if kind not in ("best", "last"):
            raise ValueError(f"未知 checkpoint 类型 {kind!r}，可选: best | last")
        payload_extra = dict(extra or {})
        if self.dir is None:
            # 仅内存：手工构造 payload（复用 save_checkpoint 的字段口径）
            payload_extra.setdefault("trainer_config", None)
            self._memory[kind] = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_val": None if best_val is None else float(best_val),
                "best_val_loss": None if best_val is None else float(best_val),
                "rng_state": rng_state,
                "history": payload_extra.pop("history", None),
                "early_stopping_state": payload_extra.pop("early_stopping_state", None),
                "trainer_config": payload_extra.pop("trainer_config", None),
                "extra": payload_extra,
            }
            return None
        target = self.best_path if kind == "best" else self.last_path
        save_checkpoint(
            target,
            model,
            optimizer,
            epoch,
            best_val,
            payload_extra,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            rng_state=rng_state,
        )
        return target

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        best_val: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        """保存 best.pt（便捷方法，等价 save('best', ...)）。"""
        return self.save("best", model, optimizer, epoch, best_val, extra)

    def save_last(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        best_val: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        """保存 last.pt（便捷方法，等价 save('last', ...)）。"""
        return self.save("last", model, optimizer, epoch, best_val, extra)

    # -- 读 ------------------------------------------------------------
    def load(
        self,
        kind: str,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any] | None:
        """读取 best/last；目录模式优先读文件，否则回退内存镜像。"""
        if self.dir is not None:
            target = self.best_path if kind == "best" else self.last_path
            if target is not None and target.exists():
                return load_checkpoint(target, model, optimizer)
        payload = self._memory.get(kind)
        if payload is None:
            return None
        if model is not None:
            model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        return payload

    def best_val(self) -> float | None:
        """当前 best 的验证损失（无记录返回 None）。"""
        payload = self.load("best")
        return payload.get("best_val_loss") if payload else None
