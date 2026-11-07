"""ConvLSTM 时空预测网络（云水场的时空演变预报）。

结构与做法
----------
- :class:`ConvLSTMCell` —— 卷积版 LSTM 单元：把 LSTM 的四组门控（输入门 i、
  遗忘门 f、输出门 o、候选 g）里的「逐元素乘法」换成「逐元素乘法 + 2D 卷积」，
  即论文中的卷积记忆单元形式（Zaichuk & Wolff, 2017 的门控公式）；
- :class:`ConvLSTMNet` —— 多层 ConvLSTM 堆叠：
  1. **输入段**：把 ``seq_len`` 帧依次喂进多层单元，得到隐状态；
  2. **预报段（多步预测）**：用**自回归（autoregressive）**方式外推
     ``pred_len`` 步——每一步先由 ``pred_head``（1×1 卷积）把隐状态投成
     ``out_channels`` 的预测场，再经 ``feedback``（kernel_size 卷积）把该场
     投影回 ``in_channels`` 作为下一步的输入，隐状态同时继续向后传递；
     也就是「循环 + 末步特征外推」，而不是一次性回归出 pred_len 帧。
- 输出 ``[B, pred_len, out_channels, H, W]``；平台契约要求 ``predict`` 返回
  ``[n_samples, n_targets]``，因此标量读出 = 预测场**逐通道空间平均**
  （可选 ``readout="last"`` 只取最后一步）；完整预测场用
  :meth:`ConvLSTM.predict_field` 取回。

**依赖隔离**：本模块只依赖 torch，不 import lightgbm / xgboost。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .base import BaseModel, TrainingResult, register_model
from .cnn import TensorTrainer, get_loss, infer_grid_shape

__all__ = ["ConvLSTM", "ConvLSTMNet", "ConvLSTMCell"]


class ConvLSTMCell(nn.Module):
    """卷积 LSTM 单元（卷积记忆单元的门控实现）。

    记号：``X_t`` 输入帧，``H_{t-1} / C_{t-1}`` 上一时刻的隐/细胞状态，
    ``⊙`` 逐元素乘，``⊛`` 保持空间尺寸的 2D 卷积。

        i_t = σ(W_xi ⊛ X_t + W_hi ⊛ H_{t-1} + b_i)
        f_t = σ(W_xf ⊛ X_t + W_hf ⊛ H_{t-1} + b_f)
        o_t = σ(W_xo ⊛ X_t + W_ho ⊛ H_{t-1} + b_o)
        g_t = tanh(W_xg ⊛ X_t + W_hg ⊛ H_{t-1} + b_g)
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
        H_t = o_t ⊙ tanh(C_t)

    其中输入门/遗忘门/输出门用同一组卷积权重合并计算（输出通道切 4 份），
    隐状态到门控的卷积用 ``hidden_channels → 4 * hidden_channels``，
    这样参数量与标准卷积 LSTM 一致。
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"kernel_size 必须为正奇数（padding 对称），当前 {kernel_size}")
        self.hidden_channels = int(hidden_channels)
        padding = k // 2
        self.conv_x = nn.Conv2d(int(in_channels), 4 * self.hidden_channels, k, padding=padding)
        self.conv_h = nn.Conv2d(self.hidden_channels, 4 * self.hidden_channels, k, padding=padding)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        """单步更新：``x [B, C, H, W]``，``h/c [B, hidden, H, W]`` → 新 ``(h, c)``。"""
        gates = self.conv_x(x) + self.conv_h(h)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMNet(nn.Module):
    """多层 ConvLSTM + 自回归多步预报头。

    Parameters
    ----------
    in_channels : int
        每帧的通道数。
    hidden_channels : int
        隐状态通道数，默认 16。
    num_layers : int
        堆叠层数，默认 1。
    kernel_size : int
        卷积核，默认 3。
    out_channels : int
        每步预测场的通道数（= 目标场数）。
    pred_len : int
        外推步数，默认 1。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        num_layers: int = 1,
        kernel_size: int = 3,
        out_channels: int = 1,
        pred_len: int = 1,
    ) -> None:
        super().__init__()
        layers = int(num_layers)
        if layers < 1:
            raise ValueError(f"num_layers 必须 ≥ 1，当前 {num_layers}")
        cells: list[ConvLSTMCell] = []
        for i in range(layers):
            cin = int(in_channels) if i == 0 else int(hidden_channels)
            cells.append(ConvLSTMCell(cin, int(hidden_channels), kernel_size))
        self.cells = nn.ModuleList(cells)
        self.hidden_channels = int(hidden_channels)
        self.pred_len = int(pred_len)
        self.pred_head = nn.Conv2d(self.hidden_channels, int(out_channels), kernel_size=1)
        # 反馈通道：把预测场投回输入通道空间，供下一步自回归使用
        self.feedback = nn.Conv2d(
            int(out_channels), int(in_channels), int(kernel_size), padding=int(kernel_size) // 2
        )

    def forward(self, x: Tensor) -> Tensor:
        """``[B, T, C, H, W]`` → ``[B, pred_len, out_channels, H, W]``。"""
        if x.ndim != 5:
            raise ValueError(f"ConvLSTM 输入应为 [B, T, C, H, W]，实际 {tuple(x.shape)}")
        b, t, _, h, w = x.shape
        hs = [x.new_zeros(b, self.hidden_channels, h, w) for _ in self.cells]
        cs = [x.new_zeros(b, self.hidden_channels, h, w) for _ in self.cells]
        # -- 输入段：逐帧推进隐状态 --------------------------------------
        for step in range(t):
            xt = x[:, step]
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(xt, hs[i], cs[i])
                xt = hs[i]
        # -- 预报段：自回归外推 ------------------------------------------
        preds: list[Tensor] = []
        for _ in range(self.pred_len):
            field = self.pred_head(hs[-1])
            preds.append(field)
            xt = self.feedback(field)
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(xt, hs[i], cs[i])
                xt = hs[i]
        return torch.stack(preds, dim=1)


@register_model("convlstm")
class ConvLSTM(BaseModel):
    """ConvLSTM 时空预测模型。

    Parameters
    ----------
    in_channels : int | None
        每帧通道数；``None`` 表示在 ``fit`` 时按数据推断。
    hidden_channels : int
        隐状态通道数，默认 16。
    num_layers : int
        堆叠层数，默认 1。
    kernel_size : int
        卷积核，默认 3（正奇数）。
    seq_len : int
        输入时间步长，默认 3（用于把 2D 输入 ``[n, F]`` 重排成序列）。
    pred_len : int
        外推步数，默认 1。
    out_channels : int | None
        每步预测场的通道数；``None`` 表示按目标数推断。
    readout : str
        标量读出方式：``mean``（对 pred_len 步与空间同时平均，默认）或
        ``last``（只取最后一步的空间平均）。
    grid_shape : tuple[int, int] | None
        2D 输入 ``[n, F]`` 的空间网格形状（None=自动推断）。

    Notes
    -----
    - 输入约定：``[n, T, C, H, W]``（直接使用）、``[n, C, H, W]``（单帧序列，
      T=1）、``[n, F]``（按 ``seq_len × in_channels × H × W`` 重排）；
    - ``predict`` 恒返回 ``[n_samples, n_targets]``；完整预报场用
      :meth:`predict_field`。
    """

    is_torch = True

    def __init__(
        self,
        in_channels: int | None = None,
        hidden_channels: int = 16,
        num_layers: int = 1,
        kernel_size: int = 3,
        seq_len: int = 3,
        pred_len: int = 1,
        out_channels: int | None = None,
        readout: str = "mean",
        grid_shape: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            in_channels=None if in_channels is None else int(in_channels),
            hidden_channels=int(hidden_channels),
            num_layers=int(num_layers),
            kernel_size=int(kernel_size),
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            out_channels=None if out_channels is None else int(out_channels),
            readout=str(readout),
            grid_shape=None if grid_shape is None else tuple(int(v) for v in grid_shape),
        )
        self.net: ConvLSTMNet | None = None
        self.adapter: nn.Linear | None = None
        self.n_in: int | None = None
        self.n_out: int | None = None

    # -- 输入整理 ------------------------------------------------------
    def _to_sequence(self, X: np.ndarray, *, name: str = "X") -> np.ndarray:
        """把输入整理成 ``[n, T, C, H, W]``。"""
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 5:
            return arr
        if arr.ndim == 4:
            return arr[:, None, ...]  # [n, C, H, W] → 单帧序列
        if arr.ndim == 2:
            n, f = arr.shape
            seq = int(self.params["seq_len"])
            if seq < 1:
                raise ValueError(f"seq_len 必须 ≥ 1，当前 {seq}")
            if f % seq != 0:
                raise ValueError(
                    f"特征数 {f} 不能被 seq_len={seq} 整除，无法重排为时空序列"
                )
            per_frame = f // seq
            in_ch = int(self.params["in_channels"] or 1)
            if per_frame % in_ch != 0:
                raise ValueError(
                    f"每帧元素数 {per_frame} 不能被 in_channels={in_ch} 整除"
                )
            hw = per_frame // in_ch
            h, w = infer_grid_shape(hw, self.params.get("grid_shape"))
            return arr.reshape(n, seq, in_ch, h, w)
        raise ValueError(
            f"{name} 维度不支持: {arr.shape}；ConvLSTM 需要 [n, T, C, H, W] / [n, C, H, W] / [n, F]"
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
        """训练 ConvLSTM。``X`` 见类文档的输入约定；``Y`` 为 ``[n, n_targets]``。"""
        Xs = self._to_sequence(X, name="X")
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n_targets = int(Y.shape[1])
        in_ch = int(self.params["in_channels"] or Xs.shape[2])
        if in_ch != int(Xs.shape[2]):
            raise ValueError(
                f"in_channels={in_ch} 与实际每帧通道数 {Xs.shape[2]} 不一致（输入 {Xs.shape}）"
            )
        out_ch = int(self.params["out_channels"] or n_targets)
        self.n_in, self.n_out = in_ch, n_targets
        if feature_names and not self.feature_names:
            self.feature_names = list(feature_names)
        if target_names and not self.target_names:
            self.target_names = list(target_names)

        self.net = ConvLSTMNet(
            in_channels=in_ch,
            hidden_channels=int(self.params["hidden_channels"]),
            num_layers=int(self.params["num_layers"]),
            kernel_size=int(self.params["kernel_size"]),
            out_channels=out_ch,
            pred_len=int(self.params["pred_len"]),
        )
        self.adapter = nn.Linear(out_ch, n_targets) if out_ch != n_targets else None

        training_cfg = dict((config or {}).get("training") or {})
        criterion = get_loss(str(training_cfg.get("loss", "mse")), training_cfg)

        def _forward(x: Tensor) -> Tensor:
            field = self.net(x)  # [B, pred_len, out_ch, H, W]
            # 空间维是 (3, 4)=(H, W)；时间维是 1
            spatial = field.mean(dim=(3, 4))  # [B, pred_len, out_ch]
            if str(self.params["readout"]) == "last":
                pooled = spatial[:, -1]
            else:
                pooled = spatial.mean(dim=1)  # [B, out_ch]
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
        Xv = self._to_sequence(X_val, name="X_val") if X_val is not None else None
        result = trainer.fit(Xs, Y, X_val=Xv, Y_val=Y_val, out_dir=out_dir)
        self.is_fitted = True
        return result

    # -- 推理 ----------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回 ``[n_samples, n_targets]``（预测场的标量读出）。"""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("ConvLSTM 尚未训练，请先调用 fit()")
        field = self.predict_field(X)
        # field: [n, pred_len, out_channels, H, W]
        # 与训练时的 _forward 保持一致：只对 时间/空间 求均值，**保留通道维**
        if str(self.params["readout"]) == "last":
            pooled = field[:, -1].mean(axis=(2, 3))      # -> [n, C]
        else:
            pooled = field.mean(axis=(1, 3, 4))           # -> [n, C]
        if self.adapter is not None:
            w = self.adapter.weight.detach().cpu().numpy()
            b = self.adapter.bias.detach().cpu().numpy()
            pooled = pooled @ w.T + b
        return np.asarray(pooled, dtype=np.float64).reshape(len(field), -1)

    def predict_field(self, X: np.ndarray) -> np.ndarray:
        """返回完整预报场 ``[n, pred_len, out_channels, H, W]``。"""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("ConvLSTM 尚未训练，请先调用 fit()")
        Xs = self._to_sequence(X, name="X")
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.from_numpy(Xs).to(device)).cpu().numpy()
        return np.asarray(out, dtype=np.float64)

    # -- 特征重要性 ----------------------------------------------------
    def feature_importance(self) -> dict[str, float] | None:
        """ConvLSTM 无内置特征重要性（不编造），返回 None。"""
        return None

    # -- 持久化 --------------------------------------------------------
    def _save_weights(self, path: Path) -> None:
        if self.net is None:
            raise RuntimeError("ConvLSTM 尚未训练，无可保存的权重")
        payload: dict[str, Any] = {
            "state_dict": self.net.state_dict(),
            "structure": {
                "n_in": self.n_in,
                "n_out": self.n_out,
                "in_channels": self.params["in_channels"],
                "hidden_channels": int(self.params["hidden_channels"]),
                "num_layers": int(self.params["num_layers"]),
                "kernel_size": int(self.params["kernel_size"]),
                "seq_len": int(self.params["seq_len"]),
                "pred_len": int(self.params["pred_len"]),
                "out_channels": self.params["out_channels"],
                "readout": str(self.params["readout"]),
                "grid_shape": self.params["grid_shape"],
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
    def _load_from(cls, path: Path, meta: dict[str, Any]) -> ConvLSTM:
        blob = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        st = blob["structure"]
        model = cls(
            in_channels=st["in_channels"],
            hidden_channels=int(st["hidden_channels"]),
            num_layers=int(st["num_layers"]),
            kernel_size=int(st["kernel_size"]),
            seq_len=int(st["seq_len"]),
            pred_len=int(st["pred_len"]),
            out_channels=st["out_channels"],
            readout=str(st["readout"]),
            grid_shape=st.get("grid_shape"),
        )
        model.n_in, model.n_out = int(st["n_in"]), int(st["n_out"])
        out_ch = int(st["out_channels"] or model.n_out)
        model.net = ConvLSTMNet(
            in_channels=int(st["in_channels"] or model.n_in),
            hidden_channels=int(st["hidden_channels"]),
            num_layers=int(st["num_layers"]),
            kernel_size=int(st["kernel_size"]),
            out_channels=out_ch,
            pred_len=int(st["pred_len"]),
        )
        model.net.load_state_dict(blob["state_dict"])
        if blob.get("adapter"):
            ad = blob["adapter"]
            model.adapter = nn.Linear(int(ad["in_features"]), int(ad["out_features"]))
            model.adapter.load_state_dict(ad["state_dict"])
        model.is_fitted = bool(meta.get("is_fitted", True))
        return model
