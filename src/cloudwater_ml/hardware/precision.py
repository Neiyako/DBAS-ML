"""混合精度（AMP）策略。

数值精度约定
------------

=========  ==================  ============================
模式       dtype               适用
=========  ==================  ============================
``fp32``   torch.float32       CPU / MPS / 不确定的数值实验
``fp16``   torch.float16      CUDA（配 GradScaler）、MPS
``bf16``   torch.bfloat16     CUDA compute capability ≥ 8.0
=========  ==================  ============================

* **auto 的默认顺序**：CUDA 且支持 bf16 → bf16；否则 CUDA → fp16；
  MPS → fp16（并给出提示）；CPU → fp32。
* **bf16 不需要 GradScaler**（与 fp32 相同的指数位宽），fp16 在 CUDA 上需要。
* **不强制所有模型用混合精度**：策略由配置决定，``fp32`` 永远可用；
  验证与指标计算默认在**关闭 autocast** 的环境下进行（见 trainer）。
* CPU 上强制 fp16 会被拒绝（torch CPU autocast 只支持 bfloat16），避免
  「以为在加速、实际全靠 fallback 算子」的情况。
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ..utils.exceptions import ConfigurationError
from ..utils.logging import get_logger

__all__ = ["PrecisionPolicy", "resolve_precision", "autocast_context", "make_grad_scaler"]

_logger = get_logger("hardware.precision")

_VALID_MODES = ("auto", "fp32", "fp16", "bf16")


@dataclass
class PrecisionPolicy:
    """一次精度决策的结果（可写入实验 metrics）。"""

    requested: str = "auto"
    mode: str = "fp32"            # fp32 | fp16 | bf16
    device_type: str = "cpu"
    autocast_enabled: bool = False
    scaler_enabled: bool = False
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def torch_dtype(self) -> Any:
        import torch

        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.mode]

    @property
    def is_mixed(self) -> bool:
        return self.mode != "fp32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "mode": self.mode,
            "device_type": self.device_type,
            "autocast_enabled": self.autocast_enabled,
            "scaler_enabled": self.scaler_enabled,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


def resolve_precision(
    requested: str = "auto",
    *,
    device_type: str = "cpu",
    supports_bfloat16: bool | None = None,
) -> PrecisionPolicy:
    """按配置与硬件能力决定实际使用的精度。

    Parameters
    ----------
    requested : str
        ``auto`` | ``fp32`` | ``fp16`` | ``bf16``。
    device_type : str
        ``cpu`` | ``cuda`` | ``mps``。
    supports_bfloat16 : bool | None
        硬件是否支持 bf16；``None`` 表示未知（按不支持处理，保守）。
    """
    requested = (requested or "auto").lower()
    if requested not in _VALID_MODES:
        raise ConfigurationError(
            f"未知精度模式 {requested!r}",
            hint=f"可选: {' | '.join(_VALID_MODES)}",
        )

    policy = PrecisionPolicy(requested=requested, device_type=device_type)
    supports_bf16 = bool(supports_bfloat16)

    if requested == "fp32":
        policy.mode = "fp32"
        policy.autocast_enabled = False
        policy.scaler_enabled = False
        policy.reason = "显式指定 fp32"
        return policy

    if requested == "bf16":
        if device_type == "cuda" and not supports_bf16:
            raise ConfigurationError(
                "当前 CUDA 设备不支持 bfloat16（需要 compute capability ≥ 8.0）",
                hint="改用 precision: fp16 或 fp32；auto 模式会自动降级为 fp16",
                context={"device_type": device_type},
            )
        if device_type == "mps":
            policy.warnings.append(
                "MPS 上 bf16 仅部分算子支持，已按 fp16 执行并保留 autocast 关闭的兜底路径"
            )
            policy.mode = "fp16"
        else:
            policy.mode = "bf16"
        policy.autocast_enabled = policy.mode != "fp32"
        policy.scaler_enabled = False  # bf16 指数位宽同 fp32，无需 loss scaling
        policy.reason = f"显式指定 {requested}"
        return policy

    if requested == "fp16":
        if device_type == "cpu":
            raise ConfigurationError(
                "CPU 不支持 float16 autocast（torch CPU autocast 仅支持 bfloat16）",
                hint="使用 precision: bf16 或 fp32",
                context={"device_type": device_type},
            )
        policy.mode = "fp16"
        policy.autocast_enabled = True
        policy.scaler_enabled = device_type == "cuda"
        policy.reason = "显式指定 fp16"
        if device_type == "mps":
            policy.warnings.append("MPS 无 GradScaler，fp16 梯度下溢风险由监控指标人工关注")
        return policy

    # ---- auto ------------------------------------------------------
    if device_type == "cuda":
        if supports_bf16:
            policy.mode = "bf16"
            policy.autocast_enabled = True
            policy.scaler_enabled = False
            policy.reason = "CUDA 设备支持 bfloat16，优先使用 bf16（无需 loss scaling）"
        else:
            policy.mode = "fp16"
            policy.autocast_enabled = True
            policy.scaler_enabled = True
            policy.reason = "CUDA 设备不支持 bfloat16，降级为 fp16（启用 GradScaler）"
    elif device_type == "mps":
        policy.mode = "fp16"
        policy.autocast_enabled = True
        policy.scaler_enabled = False
        policy.reason = "Apple MPS 使用 fp16 autocast"
        policy.warnings.append("MPS 精度策略未经本项目数值验证，请在论文实验中核对 loss 曲线")
    else:
        policy.mode = "fp32"
        policy.autocast_enabled = False
        policy.scaler_enabled = False
        policy.reason = "CPU 环境默认 fp32（CPU 混合精度收益有限且可能更慢）"
    return policy


@contextlib.contextmanager
def autocast_context(policy: PrecisionPolicy, device: Any = None) -> Iterator[None]:
    """按策略进入 ``torch.autocast``；fp32 时是空上下文（完全等价普通训练）。"""
    if not policy.autocast_enabled or policy.mode == "fp32":
        yield
        return
    import torch

    device_type = policy.device_type
    if device is not None and getattr(device, "type", None):
        device_type = str(device.type)
    if device_type not in ("cuda", "cpu", "mps"):
        yield
        return
    with torch.autocast(device_type=device_type, dtype=policy.torch_dtype, enabled=True):
        yield


def make_grad_scaler(policy: PrecisionPolicy) -> Any | None:
    """构造 GradScaler；仅 CUDA + fp16 需要，返回 ``None`` 表示不使用。"""
    if not policy.scaler_enabled or policy.device_type != "cuda" or policy.mode != "fp16":
        return None
    import torch

    try:
        return torch.amp.GradScaler(device="cuda", enabled=True)
    except TypeError:  # 兼容旧版 torch（无 device 参数）
        return torch.cuda.amp.GradScaler(enabled=True)
