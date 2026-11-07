"""MLP 多层感知机（PyTorch，基线模型）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..training import Trainer
from .base import BaseModel, TrainingResult, register_model

__all__ = ["MLP", "MLPNet"]

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


class MLPNet(nn.Module):
    """纯 torch 网络：隐藏层(Linear→Act→Dropout)* + 线性输出头（禁 ReLU）。"""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_layers: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"未知激活 {activation!r}，可选: {sorted(_ACTIVATIONS)}")
        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden_layers:
            layers += [nn.Linear(prev, int(h)), act_cls()]
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, n_out))  # 输出层恒为线性
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("mlp")
class MLP(BaseModel):
    """多层感知机回归：输入/输出维度在 fit 时从 X/Y 自动适配，多目标=单头 n_targets 线性单元。

    Parameters
    ----------
    hidden_layers : 隐藏层宽度列表，默认 [128, 64]。
    activation : relu | gelu | tanh。
    dropout : 隐藏层 Dropout 概率（0 关闭）。
    """

    is_torch = True

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        hidden_layers = [128, 64] if hidden_layers is None else list(hidden_layers)
        super().__init__(hidden_layers=list(hidden_layers), activation=activation, dropout=dropout)
        self.net: MLPNet | None = None
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
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        self.n_in, self.n_out = int(X.shape[1]), int(Y.shape[1])
        if feature_names and not self.feature_names:
            self.feature_names = list(feature_names)
        if target_names and not self.target_names:
            self.target_names = list(target_names)

        self.net = MLPNet(
            self.n_in, self.n_out,
            hidden_layers=list(self.params["hidden_layers"]),
            activation=str(self.params["activation"]),
            dropout=float(self.params["dropout"]),
        )
        training_cfg = dict((config or {}).get("training") or {})
        # physics 段位于完整配置顶层，须显式并入才能到达 Trainer → build_constraints
        physics_cfg = (config or {}).get("physics")
        if isinstance(physics_cfg, dict):
            training_cfg.setdefault("physics", physics_cfg)
        trainer = Trainer(self.net, training_cfg, config=config, model_name=self.name)
        result = trainer.fit(X, Y, X_val=X_val, Y_val=Y_val, out_dir=out_dir)
        self.is_fitted = True
        return result

    # -- 推理 ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.net is None:
            raise RuntimeError("MLP 尚未训练，请先调用 fit()")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(X).to(device)).cpu().numpy()
        return out.reshape(len(X), -1)

    # -- 持久化 --------------------------------------------------------
    def _save_weights(self, path: Path) -> None:
        if self.net is None:
            raise RuntimeError("MLP 尚未训练，无可保存的权重")
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "structure": {
                    "n_in": self.n_in,
                    "n_out": self.n_out,
                    "hidden_layers": list(self.params["hidden_layers"]),
                    "activation": str(self.params["activation"]),
                    "dropout": float(self.params["dropout"]),
                },
            },
            path / "model.pt",
        )

    @classmethod
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> MLP:
        blob = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        st = blob["structure"]
        model = cls(
            hidden_layers=st["hidden_layers"],
            activation=st["activation"],
            dropout=st["dropout"],
        )
        model.n_in, model.n_out = int(st["n_in"]), int(st["n_out"])
        model.net = MLPNet(
            model.n_in, model.n_out,
            hidden_layers=st["hidden_layers"],
            activation=st["activation"],
            dropout=st["dropout"],
        )
        model.net.load_state_dict(blob["state_dict"])
        model.is_fitted = bool(meta.get("is_fitted", True))
        return model
