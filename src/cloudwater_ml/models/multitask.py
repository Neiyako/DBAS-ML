"""MultiTask 多任务共享骨干网络（CLWC / CIWC / CWP 联合反演）。

结构
----
- **共享编码器**：``Linear → Act → Dropout`` 逐层堆叠（默认 ``[128, 64]``），
  输出共享表征；
- **多任务头**：每个任务一个 ``Linear(hidden, 1)``，各输出 1 个标量。

逐任务损失权重
--------------
``task_weights`` **真正参与损失计算**：总损失为

    L = Σ_i w_i · L_i(pred_i, y_i) / Σ_i w_i

（``normalize_weights=True`` 时除以权重和，使损失量级不随权重绝对值漂移；
关闭时按权重和直接加权）。权重为 0 的任务其损失项恒为 0 —— 对应任务头
**不接收梯度**，这是权重生效的可观测证据（见测试）。

**依赖隔离**：本模块只依赖 torch，不 import lightgbm / xgboost。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .base import BaseModel, TrainingResult, register_model
from .cnn import TensorTrainer, get_loss

__all__ = ["MultiTask", "MultiTaskNet"]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


class MultiTaskNet(nn.Module):
    """共享编码器 + 逐任务线性头。

    Parameters
    ----------
    n_in : int
        输入特征数。
    n_tasks : int
        任务数（= 目标列数）。
    hidden_layers : list[int]
        共享骨干宽度，默认 ``[128, 64]``。
    dropout : float
        骨干 Dropout 概率（0 关闭）。
    activation : str
        ``relu`` | ``gelu`` | ``tanh`` | ``leaky_relu`` | ``silu``。
    """

    def __init__(
        self,
        n_in: int,
        n_tasks: int,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        key = str(activation).lower()
        if key not in _ACTIVATIONS:
            raise ValueError(f"未知激活 {activation!r}，可选: {sorted(_ACTIVATIONS)}")
        act_cls = _ACTIVATIONS[key]
        hidden = [128, 64] if hidden_layers is None else [int(h) for h in hidden_layers]
        if not hidden:
            raise ValueError("hidden_layers 不能为空")
        layers: list[nn.Module] = []
        prev = int(n_in)
        for h in hidden:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(act_cls())
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(int(n_tasks))])
        self.n_tasks = int(n_tasks)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, F]`` → ``[B, n_tasks]``（每任务 1 个标量）。"""
        h = self.encoder(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


@register_model("multitask")
class MultiTask(BaseModel):
    """共享骨干 + 多任务头的联合反演模型。

    Parameters
    ----------
    hidden_layers : list[int]
        共享骨干宽度，默认 ``[128, 64]``。
    task_names : list[str] | None
        任务名列表；``None`` 时取 ``target_names``（再退化为 ``task_0…``）。
        长度必须与目标列数一致，否则报错。
    task_weights : list[float] | None
        逐任务损失权重；``None`` 表示等权。长度必须与任务数一致，否则报错。
    dropout : float
        骨干 Dropout 概率（0 关闭）。
    activation : str
        编码器激活函数，默认 ``relu``。
    normalize_weights : bool
        是否把加权损失除以权重和（默认 True，保持损失量级稳定）。

    Notes
    -----
    - ``predict`` 恒返回 ``[n_samples, n_tasks]``；
    - 模型内部**不做归一化**（与 MLP 一致，输入已是 runner 标准化后的量纲）。
    """

    is_torch = True

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        task_names: list[str] | None = None,
        task_weights: list[float] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_weights: bool = True,
    ) -> None:
        super().__init__(
            hidden_layers=[128, 64] if hidden_layers is None else [int(h) for h in hidden_layers],
            task_names=None if task_names is None else [str(t) for t in task_names],
            task_weights=None if task_weights is None else [float(w) for w in task_weights],
            dropout=float(dropout),
            activation=str(activation),
            normalize_weights=bool(normalize_weights),
        )
        self.net: MultiTaskNet | None = None
        self.n_in: int | None = None
        self.n_out: int | None = None
        self.task_names_resolved: list[str] = []
        self.task_weights_resolved: list[float] = []

    # -- 权重 ----------------------------------------------------------
    def _resolve_tasks(self, n_tasks: int) -> tuple[list[str], list[float]]:
        """确定任务名与任务权重（长度必须与目标列数一致）。"""
        names = self.params.get("task_names")
        if names:
            names = [str(t) for t in names]
            if len(names) != n_tasks:
                raise ValueError(
                    f"task_names 长度 {len(names)} 与目标列数 {n_tasks} 不一致"
                )
        else:
            names = list(self.target_names) if len(self.target_names) == n_tasks else [
                f"task_{i}" for i in range(n_tasks)
            ]
        weights = self.params.get("task_weights")
        if weights:
            weights = [float(w) for w in weights]
            if len(weights) != n_tasks:
                raise ValueError(
                    f"task_weights 长度 {len(weights)} 与任务数 {n_tasks} 不一致"
                )
            if any(w < 0 for w in weights):
                raise ValueError(f"task_weights 不能为负：{weights}")
            if sum(weights) <= 0:
                raise ValueError("task_weights 之和必须 > 0")
        else:
            weights = [1.0] * n_tasks
        return names, weights

    # -- 损失 ----------------------------------------------------------
    def task_losses(self, predictions: Tensor, targets: Tensor) -> list[Tensor]:
        """返回逐任务损失列表（未加权原始量纲）。"""
        criterion = getattr(self, "_criterion", None) or get_loss("mse", {})
        return [criterion(predictions[:, i], targets[:, i]) for i in range(predictions.shape[1])]

    def weighted_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """按 ``task_weights`` 加权的总损失（训练真正使用的那个损失）。"""
        per_task = self.task_losses(predictions, targets)
        weights = list(self.task_weights_resolved) or [1.0] * len(per_task)
        total = predictions.new_zeros(())
        for w, loss in zip(weights, per_task):
            total = total + float(w) * loss
        if self.params.get("normalize_weights", True):
            total = total / max(sum(weights), 1e-12)
        return total

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
        """训练多任务模型。``X`` 为 ``[n, F]``（>2D 会展平）；``Y`` 为 ``[n, n_tasks]``。"""
        Xa = np.asarray(X, dtype=np.float32)
        if Xa.ndim > 2:
            Xa = Xa.reshape(len(Xa), -1)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if Y.ndim != 2:
            raise ValueError(f"目标维度不支持: {Y.shape}；需要 [n, n_tasks]")
        self.n_in, self.n_out = int(Xa.shape[1]), int(Y.shape[1])
        if feature_names and not self.feature_names:
            self.feature_names = list(feature_names)
        if target_names and not self.target_names:
            self.target_names = list(target_names)
        self.task_names_resolved, self.task_weights_resolved = self._resolve_tasks(self.n_out)

        self.net = MultiTaskNet(
            self.n_in,
            self.n_out,
            hidden_layers=list(self.params["hidden_layers"]),
            dropout=float(self.params["dropout"]),
            activation=str(self.params["activation"]),
        )
        training_cfg = dict((config or {}).get("training") or {})
        criterion = get_loss(str(training_cfg.get("loss", "mse")), training_cfg)
        self._criterion = criterion  # 供 task_losses / weighted_loss 诊断复用

        def _loss(out: Tensor, y: Tensor) -> Tensor:
            total = out.new_zeros(())
            for i, w in enumerate(self.task_weights_resolved):
                total = total + float(w) * criterion(out[:, i], y[:, i])
            if self.params.get("normalize_weights", True):
                total = total / max(sum(self.task_weights_resolved), 1e-12)
            return total

        trainer = TensorTrainer(
            self.net,
            loss_fn=_loss,
            training_cfg=training_cfg,
            config=config,
            model_name=self.name,
        )
        Xv = None
        if X_val is not None:
            Xv = np.asarray(X_val, dtype=np.float32)
            if Xv.ndim > 2:
                Xv = Xv.reshape(len(Xv), -1)
        result = trainer.fit(Xa, Y, X_val=Xv, Y_val=Y_val, out_dir=out_dir)
        result.extra["task_names"] = list(self.task_names_resolved)
        result.extra["task_weights"] = list(self.task_weights_resolved)
        result.extra["normalize_weights"] = bool(self.params.get("normalize_weights", True))
        self.is_fitted = True
        return result

    # -- 推理 ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 ``[n_samples, n_tasks]``。"""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("MultiTask 尚未训练，请先调用 fit()")
        Xa = np.asarray(X, dtype=np.float32)
        if Xa.ndim > 2:
            Xa = Xa.reshape(len(Xa), -1)
        if Xa.ndim == 1:
            Xa = Xa.reshape(-1, 1)
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(Xa).to(device)).cpu().numpy()
        return np.asarray(out, dtype=np.float64).reshape(len(Xa), -1)

    # -- 特征重要性 ----------------------------------------------------
    def feature_importance(self) -> dict[str, float] | None:
        """MultiTask 无内置特征重要性（不编造），返回 None。"""
        return None

    # -- 持久化 --------------------------------------------------------
    def _save_weights(self, path: Path) -> None:
        if self.net is None:
            raise RuntimeError("MultiTask 尚未训练，无可保存的权重")
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "structure": {
                    "n_in": self.n_in,
                    "n_out": self.n_out,
                    "hidden_layers": list(self.params["hidden_layers"]),
                    "dropout": float(self.params["dropout"]),
                    "activation": str(self.params["activation"]),
                    "normalize_weights": bool(self.params.get("normalize_weights", True)),
                },
                "task_names": list(self.task_names_resolved),
                "task_weights": list(self.task_weights_resolved),
            },
            path / "model.pt",
        )

    @classmethod
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> MultiTask:
        blob = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        st = blob["structure"]
        model = cls(
            hidden_layers=list(st["hidden_layers"]),
            task_names=blob.get("task_names"),
            task_weights=blob.get("task_weights"),
            dropout=float(st["dropout"]),
            activation=str(st["activation"]),
            normalize_weights=bool(st.get("normalize_weights", True)),
        )
        model.n_in, model.n_out = int(st["n_in"]), int(st["n_out"])
        model.task_names_resolved = list(blob.get("task_names") or [])
        model.task_weights_resolved = list(blob.get("task_weights") or [])
        model.net = MultiTaskNet(
            model.n_in,
            model.n_out,
            hidden_layers=list(st["hidden_layers"]),
            dropout=float(st["dropout"]),
            activation=str(st["activation"]),
        )
        model.net.load_state_dict(blob["state_dict"])
        model.is_fitted = bool(meta.get("is_fitted", True))
        return model
