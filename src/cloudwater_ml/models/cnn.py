"""CNN 二维卷积回归模型 + torch 张量模型通用训练循环。

本模块提供两样东西：

1. :class:`CNNNet` / :class:`CNN` —— 二维卷积网络，用于云水场的局地特征提取
   （输入 ``[B, C, H, W]`` → 输出 ``[B, n_targets]``）；
2. :class:`TensorTrainer` —— **CNN / U-Net / ConvLSTM / MultiTask 共用**的张量级
   训练循环。

为什么共享训练循环放在本模块
----------------------------
v2.0 的 :class:`cloudwater_ml.training.Trainer` 固定走二维 ``[B, F]`` 的
TensorDataset 路径，而场模型（CNN/U-Net/ConvLSTM）的输入是 4D/5D 张量，
MultiTask 又需要「逐任务加权损失」——两者都无法直接套用 Trainer。
:class:`TensorTrainer` 保持 Trainer 的产出契约（同构 :class:`TrainingResult`、
同样的设备/精度/梯度累积/裁剪/调度/早停/检查点逻辑），但把「前向」与「损失」
交给调用方注入，于是四种模型共用一份经过验证的训练逻辑，而不是各写一遍。

**依赖隔离**：本模块只依赖 torch，**不 import lightgbm / xgboost**
（macOS 上 torch 与 LightGBM 的 OpenMP 运行库不能同进程共存，见
:mod:`cloudwater_ml.models.base` 的说明）。
"""
from __future__ import annotations

import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ..hardware import autocast_context, make_grad_scaler
from ..training import (
    CheckpointManager,
    EarlyStopping,
    HistoryLogger,
    build_optimizer,
    build_scheduler,
    get_loss,
)
from ..training.runtime import resolve_runtime
from ..utils import capture_rng_state, get_logger, set_seed
from .base import BaseModel, TrainingResult, register_model

__all__ = [
    "CNN",
    "CNNNet",
    "TensorTrainer",
    "infer_grid_shape",
    "to_field_tensor",
]

_logger = get_logger("models.cnn")

#: 支持的激活函数（与 MLP 一致的命名风格，额外给出 leaky_relu / silu）
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


# ----------------------------------------------------------------------
# 形状工具
# ----------------------------------------------------------------------
def _norm2d(channels: int) -> nn.Module:
    """归一化层（GroupNorm）。

    刻意不用 BatchNorm：BatchNorm 在 batch=1 的尾批次会直接抛
    ``Expected more than 1 value per channel``，且其统计量依赖 batch 构成；
    场模型常在小样本 / CPU 上训练，GroupNorm 与 batch 大小无关，更稳。
    """
    groups = 1
    for g in range(min(8, int(channels)), 0, -1):
        if int(channels) % g == 0:
            groups = g
            break
    return nn.GroupNorm(groups, int(channels))


def infer_grid_shape(
    n_features: int, grid_shape: tuple[int, int] | None = None
) -> tuple[int, int]:
    """把「特征数」映射成二维网格 ``(H, W)``。

    - 指定 ``grid_shape`` 时严格校验 ``H * W == n_features``；
    - 否则自动挑选**最接近正方形**的因子对（如 12 → 3×4，16 → 4×4）。

    Raises
    ------
    ValueError
        特征数非正，或 ``grid_shape`` 与特征数不匹配。
    """
    n = int(n_features)
    if n <= 0:
        raise ValueError(f"特征数必须为正，当前 {n}")
    if grid_shape is not None:
        h, w = int(grid_shape[0]), int(grid_shape[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"grid_shape 必须为正整数，当前 {grid_shape}")
        if h * w != n:
            raise ValueError(
                f"grid_shape={grid_shape} 与特征数 {n} 不匹配（要求 H*W={n}）"
            )
        return h, w
    best: tuple[tuple[int, int], tuple[int, int]] | None = None
    for h in range(1, math.isqrt(n) + 1):
        if n % h != 0:
            continue
        w = n // h
        key = (abs(w - h), -h)  # 先取接近正方形，再取 H<=W
        if best is None or key < best[0]:
            best = (key, (h, w))
    if best is None:  # pragma: no cover - n>=1 时必然有因子 1
        raise ValueError(f"无法为特征数 {n} 推断二维网格形状")
    return best[1]


def to_field_tensor(
    X: np.ndarray,
    grid_shape: tuple[int, int] | None = None,
    *,
    name: str = "X",
) -> np.ndarray:
    """把输入整理成 ``[n, C, H, W]`` 的 float32 数组。

    - 4D 输入 ``[n, C, H, W]`` 原样使用；
    - 2D 输入 ``[n, F]`` 视为「每个样本一张单通道图」，按 :func:`infer_grid_shape`
      重排成 ``[n, 1, H, W]``（平台 runner 传进来的就是二维特征矩阵）。

    Raises
    ------
    ValueError
        维度不支持（既不是 2D 也不是 4D）。
    """
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 2:
        h, w = infer_grid_shape(arr.shape[1], grid_shape)
        return arr.reshape(arr.shape[0], 1, h, w)
    raise ValueError(
        f"{name} 维度不支持: {arr.shape}；场模型需要 [n, C, H, W] 或 [n, F]"
    )


# ----------------------------------------------------------------------
# 通用张量训练循环
# ----------------------------------------------------------------------
class TensorTrainer:
    """通用 torch 训练循环（张量级输入 + 可注入损失）。

    与 :class:`cloudwater_ml.training.Trainer` 的差异只有两点，其余行为同构：

    1. 输入可以是任意形状张量（4D 场 / 5D 时空序列），不做 ``[B, F]`` 假设；
    2. 前向与损失由调用方注入（``forward_fn`` / ``loss_fn``），因此
       MultiTask 的「逐任务加权损失」能真正生效，而不是装饰性参数。

    Parameters
    ----------
    module : nn.Module
        被训练的模块（参数、优化器、检查点都以它为准）。
    forward_fn : Callable[[Tensor], Tensor] | None
        前向函数，默认直接用 ``module``。
    loss_fn : Callable[[Tensor, Tensor], Tensor]
        损失函数 ``loss_fn(outputs, targets)``，必须返回标量张量。
    training_cfg : dict | None
        实验配置的 ``training`` 段（epochs / batch_size / learning_rate /
        optimizer / scheduler / early_stopping / seed / gradient_accumulation_steps …）。
    config : dict | None
        完整实验配置（用于解析 hardware / dataloader 段）。
    model_name : str | None
        模型名（channels_last 自动判定用）。
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        forward_fn: Callable[[Tensor], Tensor] | None = None,
        training_cfg: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.module = module
        self.forward_fn: Callable[[Tensor], Tensor] = forward_fn or module
        self.loss_fn = loss_fn
        self.cfg = dict(training_cfg or {})
        self.full_config = dict(config or {})
        self.model_name = model_name
        self.device: torch.device = torch.device("cpu")
        self._global_step = 0

    # -- 计划 / 数据 ---------------------------------------------------
    def _build_plan(self, n_samples: int, batch_size: int):
        """解析运行时计划（设备 / 精度 / DataLoader / 累积 / 裁剪）。"""
        cfg_for_plan = dict(self.full_config) if self.full_config else {}
        if not cfg_for_plan:
            cfg_for_plan = {"training": dict(self.cfg)}
        else:
            cfg_for_plan["training"] = {
                **dict(self.cfg),
                **dict(cfg_for_plan.get("training") or {}),
            }
        plan = resolve_runtime(
            cfg_for_plan,
            model_name=self.model_name or type(self.module).__name__.lower(),
            n_samples=int(n_samples),
        )
        plan.batch_size = int(batch_size)
        return plan

    def _loader(self, X: np.ndarray, Y: np.ndarray, batch_size: int, plan, *, shuffle: bool):
        """构造 DataLoader（TensorDataset，形状无关）。"""
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        kwargs = plan.dataloader.loader_kwargs()
        kwargs["shuffle"] = shuffle
        kwargs["drop_last"] = plan.dataloader.drop_last and shuffle
        return DataLoader(ds, batch_size=int(batch_size), **kwargs)

    # -- 训练 ----------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        out_dir: str | Path | None = None,
    ) -> TrainingResult:
        """训练并返回与 Trainer 同构的 :class:`TrainingResult`。"""
        cfg = self.cfg
        set_seed(
            int(cfg.get("seed", 42)),
            deterministic=bool(cfg.get("deterministic", False)),
        )
        epochs = int(cfg.get("epochs", 50))
        batch_size = int(cfg.get("batch_size", 64) or 64)

        plan = self._build_plan(len(X_train), batch_size)
        self.device = plan.device
        _logger.info(
            "运行时计划: device=%s precision=%s accum=%d batch=%d workers=%d",
            plan.device, plan.precision.mode, plan.gradient_accumulation_steps,
            plan.batch_size, plan.dataloader.num_workers,
        )

        X_train = np.asarray(X_train, dtype=np.float32)
        Y_train = np.asarray(Y_train, dtype=np.float32)
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)
        if len(X_train) != len(Y_train):
            raise ValueError(
                f"训练样本数不一致：X {len(X_train)} 行 vs Y {len(Y_train)} 行"
            )
        train_loader = self._loader(X_train, Y_train, batch_size, plan, shuffle=True)
        val_loader = None
        if X_val is not None and Y_val is not None and len(X_val) > 0:
            Xv = np.asarray(X_val, dtype=np.float32)
            Yv = np.asarray(Y_val, dtype=np.float32)
            if Yv.ndim == 1:
                Yv = Yv.reshape(-1, 1)
            val_loader = self._loader(Xv, Yv, batch_size, plan, shuffle=False)

        optimizer = build_optimizer(str(cfg.get("optimizer", "adamw")), self.module, cfg)
        scheduler = build_scheduler(cfg.get("scheduler"), optimizer, cfg)
        scaler = make_grad_scaler(plan.precision)

        self.module.to(self.device)
        if plan.channels_last:
            try:
                self.module.to(memory_format=torch.channels_last)
            except Exception as exc:  # pragma: no cover - 非 4D 模型属正常
                _logger.info("channels_last 未生效: %s", exc)

        es_cfg = cfg.get("early_stopping") or {}
        es_enabled = bool(es_cfg.get("enabled", True))
        early = EarlyStopping(
            patience=int(es_cfg.get("patience", 10)),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
        )
        history = HistoryLogger()
        ckpt = CheckpointManager(out_dir)

        best_state: dict[str, Tensor] | None = None
        best_val: float | None = None
        best_epoch: int | None = None
        stopped_early = False

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, optimizer, scaler, plan, train=True)
            val_loss = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, optimizer, scaler, plan, train=False)
            if scheduler is not None:
                scheduler.step()

            monitored = float(val_loss if val_loss is not None else train_loss)
            history.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            improved = best_val is None or monitored < best_val
            if improved:
                best_val = monitored
                best_epoch = epoch
                best_state = copy.deepcopy(self.module.state_dict())
                ckpt.save(
                    "best", self.module, optimizer, epoch, best_val,
                    {"trainer_config": cfg, "runtime_plan": plan.to_dict()},
                    scheduler=scheduler, scaler=scaler,
                    global_step=self._global_step, rng_state=capture_rng_state(),
                )
            ckpt.save(
                "last", self.module, optimizer, epoch, best_val,
                {"trainer_config": cfg, "runtime_plan": plan.to_dict()},
                scheduler=scheduler, scaler=scaler,
                global_step=self._global_step, rng_state=capture_rng_state(),
            )
            _logger.info(
                "epoch %d/%d | train=%.6f | val=%s | %.1fs%s",
                epoch, epochs, train_loss,
                f"{val_loss:.6f}" if val_loss is not None else "-",
                time.time() - t0, " *" if improved else "",
            )
            if es_enabled and early.step(monitored, epoch):
                stopped_early = True
                break

        if best_state is not None:
            self.module.load_state_dict(best_state)

        return TrainingResult(
            history=history.as_dicts(),
            best_epoch=best_epoch,
            best_val_loss=best_val,
            stopped_early=stopped_early,
            extra={
                "device": str(self.device),
                "epochs_run": history.rows[-1]["epoch"] if history.rows else 0,
                "monitor": "val_loss" if val_loader is not None else "train_loss",
                "restored_best": best_state is not None,
                "runtime_plan": plan.to_dict(),
                "global_step": self._global_step,
                "trainer": "TensorTrainer(通用张量循环)",
            },
        )

    # -- 内部 ----------------------------------------------------------
    def _run_epoch(self, loader, optimizer, scaler, plan, *, train: bool) -> float:
        """跑一个 epoch，返回按样本加权的平均损失。"""
        if train:
            self.module.train()
        else:
            self.module.eval()
        accum = plan.gradient_accumulation_steps if train else 1
        total, n = 0.0, 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for step, (xb, yb) in enumerate(loader):
                xb = xb.to(self.device, non_blocking=plan.dataloader.non_blocking)
                yb = yb.to(self.device, non_blocking=plan.dataloader.non_blocking)
                if train and plan.channels_last and xb.ndim == 4:
                    xb = xb.contiguous(memory_format=torch.channels_last)
                amp_ctx = autocast_context(plan.precision, self.device) if train else nullcontext()
                with amp_ctx:
                    out = self.forward_fn(xb)
                    loss = self.loss_fn(out, yb)
                    loss_value = float(loss.detach())
                    if train:
                        loss = loss / accum
                if train:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if (step + 1) % accum == 0:
                        self._apply_gradients(optimizer, scaler, plan)
                bs = int(xb.shape[0])
                total += loss_value * bs
                n += bs
        return total / max(n, 1)

    def _apply_gradients(self, optimizer, scaler, plan) -> None:
        """裁剪 + 一步优化（AMP 下先 unscale_ 再裁剪）。"""
        if plan.gradient_clip_norm:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.module.parameters(), float(plan.gradient_clip_norm)
            )
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        self._global_step += 1
        optimizer.zero_grad(set_to_none=True)


# ----------------------------------------------------------------------
# CNN
# ----------------------------------------------------------------------
class CNNNet(nn.Module):
    """二维卷积特征提取 + 全局平均池化 + 线性回归头。

    结构：``[Conv2d → Norm → Act → (Dropout) → MaxPool] * len(channels)``
    → ``AdaptiveAvgPool2d(1)`` → ``Flatten`` → ``Dropout`` → ``Linear(n_targets)``。

    输出层恒为线性（与 MLP 一致，不默认 ReLU），因此可以输出负值。
    """

    def __init__(
        self,
        in_channels: int,
        n_targets: int,
        channels: list[int] | tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        key = str(activation).lower()
        if key not in _ACTIVATIONS:
            raise ValueError(f"未知激活 {activation!r}，可选: {sorted(_ACTIVATIONS)}")
        act_cls = _ACTIVATIONS[key]
        chans = [int(c) for c in channels]
        if not chans:
            raise ValueError("channels 不能为空")
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"kernel_size 必须为正奇数（padding 对称），当前 {kernel_size}")

        blocks: list[nn.Module] = []
        prev = int(in_channels)
        for i, ch in enumerate(chans):
            blocks.append(nn.Conv2d(prev, ch, k, padding=k // 2))
            blocks.append(_norm2d(ch))
            blocks.append(act_cls())
            if dropout and float(dropout) > 0:
                blocks.append(nn.Dropout2d(float(dropout)))
            if i < len(chans) - 1:
                blocks.append(nn.MaxPool2d(2))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(float(dropout)) if dropout and float(dropout) > 0 else nn.Identity()
        self.head = nn.Linear(prev, int(n_targets))

    def forward(self, x: Tensor) -> Tensor:
        """``[B, C, H, W]`` → ``[B, n_targets]``。"""
        h = self.features(x)
        h = self.pool(h).flatten(1)
        h = self.drop(h)
        return self.head(h)


@register_model("cnn")
class CNN(BaseModel):
    """二维卷积回归模型（云水场 → CLWC/CIWC/CWP）。

    Parameters
    ----------
    in_channels : int | None
        输入通道数；``None`` 表示在 ``fit`` 时按数据推断（4D 输入取 ``X.shape[1]``，
        2D 输入取 1）。
    channels : list[int]
        各卷积块通道数，默认 ``[32, 64, 128]``。
    kernel_size : int
        卷积核大小，默认 3（必须为正奇数）。
    dropout : float
        卷积块内 Dropout2d 与池化后 Dropout 概率（0 关闭）。
    activation : str
        ``relu`` | ``gelu`` | ``tanh`` | ``leaky_relu`` | ``silu``。
    grid_shape : tuple[int, int] | None
        2D 输入 ``[n, F]`` 时的网格形状；``None`` 自动取最接近正方形的因子对。

    Notes
    -----
    - 输入约定：``[n, C, H, W]``（直接使用）或 ``[n, F]``（自动重排为单通道图）；
    - ``fit`` 收到的 X 已是 runner 标准化后的量纲，**模型内部不再做归一化**
      （与 MLP 完全一致），``predict`` 直接返回同一量纲；
    - ``predict`` 恒返回 ``[n_samples, n_targets]``。
    """

    is_torch = True

    def __init__(
        self,
        in_channels: int | None = None,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
        grid_shape: tuple[int, int] | None = None,
    ) -> None:
        chans = [32, 64, 128] if channels is None else [int(c) for c in channels]
        super().__init__(
            in_channels=None if in_channels is None else int(in_channels),
            channels=chans,
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            activation=str(activation),
            grid_shape=None if grid_shape is None else tuple(int(v) for v in grid_shape),
        )
        self.net: CNNNet | None = None
        self.n_in: int | None = None
        self.n_out: int | None = None

    # -- 训练 ----------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
        out_dir: str | Path | None = None,
        feature_names: list[str] | None = None,
        target_names: list[str] | None = None,
    ) -> TrainingResult:
        """训练 CNN。

        ``X`` 支持 ``[n, C, H, W]`` 或 ``[n, F]``；``Y`` 为 ``[n, n_targets]``。
        """
        Xf = to_field_tensor(X, self.params.get("grid_shape"), name="X")
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        in_ch = int(self.params["in_channels"] or Xf.shape[1])
        if in_ch != int(Xf.shape[1]):
            raise ValueError(
                f"in_channels={in_ch} 与实际输入通道数 {Xf.shape[1]} 不一致"
                f"（输入形状 {Xf.shape}）"
            )
        self.n_in, self.n_out = int(Xf.shape[1]), int(Y.shape[1])
        if feature_names and not self.feature_names:
            self.feature_names = list(feature_names)
        if target_names and not self.target_names:
            self.target_names = list(target_names)

        self.net = CNNNet(
            self.n_in,
            self.n_out,
            channels=list(self.params["channels"]),
            kernel_size=int(self.params["kernel_size"]),
            dropout=float(self.params["dropout"]),
            activation=str(self.params["activation"]),
        )
        training_cfg = dict((config or {}).get("training") or {})
        trainer = TensorTrainer(
            self.net,
            loss_fn=self._make_loss_fn(training_cfg),
            training_cfg=training_cfg,
            config=config,
            model_name=self.name,
        )
        Xv = to_field_tensor(X_val, self.params.get("grid_shape"), name="X_val") if X_val is not None else None
        result = trainer.fit(Xf, Y, X_val=Xv, Y_val=Y_val, out_dir=out_dir)
        self.is_fitted = True
        return result

    def _make_loss_fn(self, training_cfg: dict[str, Any]):
        """构造监督损失（mse / mae / huber）。"""
        criterion = get_loss(str(training_cfg.get("loss", "mse")), training_cfg)

        def _loss(out: Tensor, y: Tensor) -> Tensor:
            return criterion(out, y)

        return _loss

    # -- 推理 ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 ``[n_samples, n_targets]`` 的预测。"""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("CNN 尚未训练，请先调用 fit()")
        Xf = to_field_tensor(X, self.params.get("grid_shape"), name="X")
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(Xf).to(device)).cpu().numpy()
        return np.asarray(out, dtype=np.float64).reshape(len(Xf), -1)

    # -- 特征重要性 ----------------------------------------------------
    def feature_importance(self) -> dict[str, float] | None:
        """CNN 无内置特征重要性（不编造），返回 None。"""
        return None

    # -- 持久化 --------------------------------------------------------
    def _save_weights(self, path: Path) -> None:
        if self.net is None:
            raise RuntimeError("CNN 尚未训练，无可保存的权重")
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "structure": {
                    "n_in": self.n_in,
                    "n_out": self.n_out,
                    "in_channels": self.params["in_channels"],
                    "channels": list(self.params["channels"]),
                    "kernel_size": int(self.params["kernel_size"]),
                    "dropout": float(self.params["dropout"]),
                    "activation": str(self.params["activation"]),
                    "grid_shape": self.params["grid_shape"],
                },
            },
            path / "model.pt",
        )

    @classmethod
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> "CNN":
        blob = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        st = blob["structure"]
        model = cls(
            in_channels=st["in_channels"],
            channels=list(st["channels"]),
            kernel_size=int(st["kernel_size"]),
            dropout=float(st["dropout"]),
            activation=str(st["activation"]),
            grid_shape=st.get("grid_shape"),
        )
        model.n_in, model.n_out = int(st["n_in"]), int(st["n_out"])
        model.net = CNNNet(
            model.n_in,
            model.n_out,
            channels=list(st["channels"]),
            kernel_size=int(st["kernel_size"]),
            dropout=float(st["dropout"]),
            activation=str(st["activation"]),
        )
        model.net.load_state_dict(blob["state_dict"])
        model.is_fitted = bool(meta.get("is_fitted", True))
        return model
