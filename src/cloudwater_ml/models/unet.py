"""U-Net 编码-解码网络（云水二维场重建 / 反演）。

结构：``[B, C, H, W]`` → ``[B, out_channels, H, W]``（输出保持输入空间尺寸）。

要点
----
- 编码器逐级下采样（2×），解码器逐级上采样并与编码器同层特征 **skip 连接**；
- 输入空间尺寸必须能被 ``2 ** depth`` 整除，否则**明确报错**（不静默裁剪，
  避免张量错位导致的「看起来能跑但结果错位」）；
- 上采样方式二选一：``bilinear=True`` 用双线性插值 + 卷积，
  ``bilinear=False`` 用转置卷积；
- 与平台契约一致：``predict`` 恒返回 ``[n_samples, n_targets]`` 二维数组
  （标量读出 = 预测场的逐通道空间平均）；完整预测场用
  :meth:`UNet.predict_field` 取回。

**依赖隔离**：本模块只依赖 torch，不 import lightgbm / xgboost。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .base import BaseModel, TrainingResult, register_model
from .cnn import TensorTrainer, get_loss, to_field_tensor

__all__ = ["UNet", "UNetNet", "DoubleConv"]


def _norm2d(channels: int) -> nn.Module:
    """GroupNorm（与 CNN 一致：不受 batch=1 尾批次限制，小样本训练更稳）。"""
    groups = 1
    for g in range(min(8, int(channels)), 0, -1):
        if int(channels) % g == 0:
            groups = g
            break
    return nn.GroupNorm(groups, int(channels))


class DoubleConv(nn.Module):
    """``(Conv3×3 → Norm → ReLU) × 2``：U-Net 的基本卷积单元。"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(int(in_channels), int(out_channels), 3, padding=1, bias=False),
            _norm2d(int(out_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_channels), int(out_channels), 3, padding=1, bias=False),
            _norm2d(int(out_channels)),
            nn.ReLU(inplace=True),
        ]
        if dropout and float(dropout) > 0:
            layers.append(nn.Dropout2d(float(dropout)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNetNet(nn.Module):
    """U-Net 主干：输出与输入同空间尺寸的 ``out_channels`` 张量。

    Parameters
    ----------
    in_channels : int
        输入通道数。
    out_channels : int
        输出通道数（目标场数，如 CWP 1 场）。
    base_channels : int
        编码器首层通道数，默认 32。
    depth : int
        下采样层数，默认 3（输入 H、W 须能被 ``2 ** depth`` 整除）。
    bilinear : bool
        True=双线性插值上采样；False=转置卷积上采样。
    dropout : float
        卷积块后的 Dropout2d 概率（0 关闭）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 3,
        bilinear: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        depth = int(depth)
        if depth < 1:
            raise ValueError(f"depth 必须 ≥ 1，当前 {depth}")
        self.depth = depth
        self.bilinear = bool(bilinear)
        base = int(base_channels)
        if base < 1:
            raise ValueError(f"base_channels 必须 ≥ 1，当前 {base_channels}")

        self.inc = DoubleConv(int(in_channels), base, dropout)
        self.downs = nn.ModuleList()
        ch = base
        for _ in range(depth):
            self.downs.append(DoubleConv(ch, ch * 2, dropout))
            ch *= 2
        self.bottleneck = DoubleConv(ch, ch, dropout)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for _ in range(depth):
            # 上采样后通道减半，与对应 skip 层的通道数一致
            self.up_convs.append(
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
                if not bilinear
                else nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1)
            )
            self.ups.append(DoubleConv(ch, ch // 2, dropout))
            ch //= 2
        self.outc = nn.Conv2d(base, int(out_channels), kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, C, H, W]`` → ``[B, out_channels, H, W]``。"""
        skips: list[Tensor] = []
        h = self.inc(x)
        for down in self.downs:
            skips.append(h)
            h = down(nn.functional.max_pool2d(h, 2))
        h = self.bottleneck(h)
        for up_conv, up in zip(self.up_convs, self.ups):
            if self.bilinear:
                h = torch.nn.functional.interpolate(
                    h, scale_factor=2.0, mode="bilinear", align_corners=False
                )
            h = up_conv(h)
            skip = skips.pop()
            h = up(torch.cat([skip, h], dim=1))
        return self.outc(h)


@register_model("unet")
class UNet(BaseModel):
    """U-Net 云水场反演模型。

    Parameters
    ----------
    in_channels : int | None
        输入通道数；``None`` 表示在 ``fit`` 时按数据推断。
    out_channels : int | None
        输出通道数；``None`` 表示在 ``fit`` 时按目标推断（2D 目标 → 目标列数；
        4D 场目标 → 目标场通道数）。
    base_channels : int
        编码器首层通道数，默认 32。
    depth : int
        下采样层数，默认 3。
    bilinear : bool
        上采样方式，默认 True（双线性）。
    dropout : float
        Dropout2d 概率，默认 0。
    grid_shape : tuple[int, int] | None
        2D 输入 ``[n, F]`` 的网格形状（None=自动推断）。

    Notes
    -----
    - **空间尺寸硬校验**：``H % 2**depth != 0`` 或 ``W % 2**depth != 0`` 时
      直接抛 ``ValueError``（不静默裁剪）；
    - 目标可以是标量 ``Y [n, n_targets]``（对预测场做逐通道空间平均后监督），
      也可以是整场 ``Y [n, C, H, W]``（逐像素监督）；
    - ``predict`` 恒返回 ``[n_samples, n_targets]``。
    """

    is_torch = True

    def __init__(
        self,
        in_channels: int | None = None,
        out_channels: int | None = None,
        base_channels: int = 32,
        depth: int = 3,
        bilinear: bool = True,
        dropout: float = 0.0,
        grid_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            in_channels=None if in_channels is None else int(in_channels),
            out_channels=None if out_channels is None else int(out_channels),
            base_channels=int(base_channels),
            depth=int(depth),
            bilinear=bool(bilinear),
            dropout=float(dropout),
            grid_shape=None if grid_shape is None else tuple(int(v) for v in grid_shape),
        )
        self.net: UNetNet | None = None
        self.adapter: nn.Linear | None = None
        self.n_in: int | None = None
        self.n_out: int | None = None
        self.field_mode: bool = False  # True=整场监督（Y 为 4D）

    # -- 尺寸校验 ------------------------------------------------------
    def _check_spatial(self, H: int, W: int) -> None:
        """检查输入空间尺寸能否被 ``2 ** depth`` 整除（不满足就明确报错）。"""
        factor = 2 ** int(self.params["depth"])
        bad = [n for n, v in (("H", H), ("W", W)) if int(v) % factor != 0]
        if bad:
            raise ValueError(
                f"UNet 输入空间尺寸必须能被 2**depth={factor} 整除"
                f"（depth={self.params['depth']}），当前 {'/'.join(bad)}="
                f"{H if 'H' in bad else ''}{'x' if len(bad) > 1 else ''}"
                f"{W if 'W' in bad else ''}；"
                f"请裁剪/补齐到 {factor} 的整数倍（不静默裁剪，避免特征错位）"
            )

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
        """训练 U-Net。

        ``X`` 支持 ``[n, C, H, W]`` 或 ``[n, F]``；``Y`` 支持 ``[n, n_targets]``
        （标量监督）或 ``[n, C, H, W]``（整场监督）。
        """
        Xf = to_field_tensor(X, self.params.get("grid_shape"), name="X")
        H, W = int(Xf.shape[2]), int(Xf.shape[3])
        self._check_spatial(H, W)

        Y = np.asarray(Y, dtype=np.float32)
        self.field_mode = Y.ndim == 4
        if self.field_mode:
            if Y.shape[2] != H or Y.shape[3] != W:
                raise ValueError(
                    f"整场目标的空间尺寸 {Y.shape[2]}x{Y.shape[3]} 与输入 {H}x{W} 不一致"
                )
            n_targets = int(Y.shape[1])
        elif Y.ndim == 1:
            Y = Y.reshape(-1, 1)
            n_targets = 1
        elif Y.ndim == 2:
            n_targets = int(Y.shape[1])
        else:
            raise ValueError(f"目标维度不支持: {Y.shape}；需要 [n, T] 或 [n, C, H, W]")

        in_ch = int(self.params["in_channels"] or Xf.shape[1])
        if in_ch != int(Xf.shape[1]):
            raise ValueError(
                f"in_channels={in_ch} 与实际输入通道数 {Xf.shape[1]} 不一致（输入形状 {Xf.shape}）"
            )
        out_ch = int(self.params["out_channels"] or n_targets)
        if self.field_mode and out_ch != n_targets:
            raise ValueError(
                f"整场监督时 out_channels 必须等于目标场通道数 {n_targets}，当前 {out_ch}"
            )

        self.n_in, self.n_out = int(Xf.shape[1]), int(n_targets)
        if feature_names and not self.feature_names:
            self.feature_names = list(feature_names)
        if target_names and not self.target_names:
            self.target_names = list(target_names)

        self.net = UNetNet(
            self.n_in,
            out_ch,
            base_channels=int(self.params["base_channels"]),
            depth=int(self.params["depth"]),
            bilinear=bool(self.params["bilinear"]),
            dropout=float(self.params["dropout"]),
        )
        # 标量模式下：预测场 → 逐通道空间平均 →（必要时）线性适配到 n_targets
        self.adapter = (
            nn.Linear(out_ch, n_targets) if (not self.field_mode and out_ch != n_targets) else None
        )

        training_cfg = dict((config or {}).get("training") or {})
        criterion = get_loss(str(training_cfg.get("loss", "mse")), training_cfg)
        if self.field_mode:
            def _forward(x: Tensor) -> Tensor:
                return self.net(x)

            def _loss(out: Tensor, y: Tensor) -> Tensor:
                return criterion(out, y)
        else:
            def _forward(x: Tensor) -> Tensor:
                pooled = self.net(x).mean(dim=(2, 3))
                return pooled if self.adapter is None else self.adapter(pooled)

            def _loss(out: Tensor, y: Tensor) -> Tensor:
                return criterion(out, y)

        trainer = TensorTrainer(
            self.net,
            loss_fn=_loss,
            forward_fn=_forward,
            training_cfg=training_cfg,
            config=config,
            model_name=self.name,
        )
        Xv = (
            to_field_tensor(X_val, self.params.get("grid_shape"), name="X_val")
            if X_val is not None
            else None
        )
        result = trainer.fit(Xf, Y, X_val=Xv, Y_val=Y_val, out_dir=out_dir)
        self.is_fitted = True
        return result

    # -- 推理 ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 ``[n_samples, n_targets]``（预测场逐通道的空间平均）。"""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("UNet 尚未训练，请先调用 fit()")
        field = self.predict_field(X)
        pooled = field.mean(axis=(2, 3))
        if self.adapter is not None:
            pooled = pooled @ self.adapter.weight.detach().cpu().numpy().T + (
                self.adapter.bias.detach().cpu().numpy()
            )
        return np.asarray(pooled, dtype=np.float64).reshape(len(field), -1)

    def predict_field(self, X: np.ndarray) -> np.ndarray:
        """返回完整预测场 ``[n, out_channels, H, W]``（U-Net 的原生输出）。"""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("UNet 尚未训练，请先调用 fit()")
        Xf = to_field_tensor(X, self.params.get("grid_shape"), name="X")
        self._check_spatial(int(Xf.shape[2]), int(Xf.shape[3]))
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(Xf).to(device)).cpu().numpy()
        return np.asarray(out, dtype=np.float64)

    # -- 特征重要性 ----------------------------------------------------
    def feature_importance(self) -> dict[str, float] | None:
        """U-Net 无内置特征重要性（不编造），返回 None。"""
        return None

    # -- 持久化 --------------------------------------------------------
    def _save_weights(self, path: Path) -> None:
        if self.net is None:
            raise RuntimeError("UNet 尚未训练，无可保存的权重")
        payload: dict[str, Any] = {
            "state_dict": self.net.state_dict(),
            "structure": {
                "n_in": self.n_in,
                "n_out": self.n_out,
                "in_channels": self.params["in_channels"],
                "out_channels": self.params["out_channels"],
                "base_channels": int(self.params["base_channels"]),
                "depth": int(self.params["depth"]),
                "bilinear": bool(self.params["bilinear"]),
                "dropout": float(self.params["dropout"]),
                "grid_shape": self.params["grid_shape"],
                "field_mode": bool(self.field_mode),
            },
        }
        if self.adapter is not None:
            payload["adapter"] = {
                "state_dict": self.adapter.state_dict(),
                "in_features": int(self.adapter.in_features),
                "out_features": int(self.adapter.out_features),
            }
        torch.save(payload, path / "model.pt")

    @classmethod
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> UNet:
        blob = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        st = blob["structure"]
        model = cls(
            in_channels=st["in_channels"],
            out_channels=st["out_channels"],
            base_channels=int(st["base_channels"]),
            depth=int(st["depth"]),
            bilinear=bool(st["bilinear"]),
            dropout=float(st["dropout"]),
            grid_shape=st.get("grid_shape"),
        )
        model.n_in, model.n_out = int(st["n_in"]), int(st["n_out"])
        model.field_mode = bool(st.get("field_mode", False))
        model.net = UNetNet(
            model.n_in,
            int(st["out_channels"] or model.n_out),
            base_channels=int(st["base_channels"]),
            depth=int(st["depth"]),
            bilinear=bool(st["bilinear"]),
            dropout=float(st["dropout"]),
        )
        model.net.load_state_dict(blob["state_dict"])
        if blob.get("adapter"):
            ad = blob["adapter"]
            model.adapter = nn.Linear(int(ad["in_features"]), int(ad["out_features"]))
            model.adapter.load_state_dict(ad["state_dict"])
        model.is_fitted = bool(meta.get("is_fitted", True))
        return model
