"""训练运行时计划（RuntimePlan）：把配置解析成一份**可追溯、可复现**的执行计划。

为什么要单独一层
----------------

Trainer 里散落着「读配置 → 猜硬件能力 → 决定退化策略」的逻辑时，
同一份配置在不同机器上会跑出不同结果，而且实验记录里说不清当时到底用了什么。
本模块把这件事集中成一次解析，产出一个 :class:`RuntimePlan`：

* 设备（含 CUDA 不可用的原因与回退记录）；
* 精度模式（fp32 / fp16 / bf16 + 是否启用 GradScaler）；
* 梯度累积、梯度裁剪、channels_last、torch.compile；
* DataLoader 选项（workers / pin_memory / persistent_workers / prefetch_factor）。

配置兼容性
----------

v2.0 引入新分段，同时**完整保留 v1.0 的旧键**：

===============  ==========================  ==========================
新分段/键        v1.0 旧键（仍生效）        说明
===============  ==========================  ==========================
``hardware``     ``training.device``        新分段优先
``dataloader``   ``training.num_workers``   同上
``training.gradient_accumulation_steps``  —            新增
``training.gradient_clip_norm``  ``training.grad_clip``  同上
``hardware.precision``  —            新增
``hardware.channels_last``  —      新增
===============  ==========================  ==========================
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ..hardware import (
    DeviceInfo,
    PrecisionPolicy,
    detect_hardware,
    resolve_precision,
)
from ..utils.exceptions import ConfigurationError
from ..utils.logging import get_logger

__all__ = ["DataLoaderOptions", "RuntimePlan", "resolve_runtime", "resolve_dataloader_options"]

_logger = get_logger("training.runtime")

_VALID_DEVICES = ("auto", "cpu", "cuda", "mps")
_VALID_PRECISIONS = ("auto", "fp32", "fp16", "bf16")


@dataclass
class DataLoaderOptions:
    """DataLoader 选项（已按硬件能力与 CPU 回退策略修正过）。"""

    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    drop_last: bool = False
    shuffle: bool = True
    non_blocking: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
            "non_blocking": self.non_blocking,
            "notes": list(self.notes),
        }

    def loader_kwargs(self) -> dict[str, Any]:
        """转成 ``torch.utils.data.DataLoader`` 的 kwargs（剔除非法组合）。"""
        kwargs: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs


@dataclass
class RuntimePlan:
    """一次运行的执行计划（写进实验 metrics，保证事后可追溯）。"""

    device: Any                      # torch.device
    device_info: DeviceInfo
    precision: PrecisionPolicy
    batch_size: int = 256
    channels_last: bool = False
    compile: bool = False
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = None
    grad_checkpointing: bool = False
    dataloader: DataLoaderOptions = field(default_factory=DataLoaderOptions)
    notes: list[str] = field(default_factory=list)

    @property
    def device_type(self) -> str:
        return str(self.device.type)

    @property
    def effective_batch_size(self) -> int:
        """梯度累积后的等效批大小。"""
        return self.batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "device_info": self.device_info.to_dict(),
            "precision": self.precision.to_dict(),
            "channels_last": self.channels_last,
            "compile": self.compile,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
            "grad_checkpointing": self.grad_checkpointing,
            "batch_size": self.batch_size,
            "effective_batch_size": self.effective_batch_size,
            "dataloader": self.dataloader.to_dict(),
            "notes": list(self.notes),
        }


# ----------------------------------------------------------------------
def _auto_num_workers(n_samples: int | None) -> tuple[int, list[str]]:
    """按 CPU 核数与数据量决定 worker 数（小数据不启动进程，避免启动开销）。"""
    notes: list[str] = []
    cpu = os.cpu_count() or 1
    if n_samples is not None and n_samples < 2000:
        notes.append(
            f"数据集仅 {n_samples} 样本，num_workers 自动取 0（多进程启动开销大于收益）"
        )
        return 0, notes
    workers = max(0, min(8, cpu // 2))
    notes.append(f"num_workers 自动取 {workers}（CPU 核数 {cpu}，上限 8）")
    return workers, notes


def _first_concrete(*values: Any) -> Any:
    """返回第一个「非 auto 且非 None」的具体值；都没有则返回 None（走自动推断）。

    用于 v1.0 旧键与 v2.0 新分段并存时的取值裁决：**具体值优先**，
    避免基线配置里的 auto 覆盖掉实验配置里写死的值。
    """
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v.strip().lower() == "auto":
            continue
        return v
    return None


def resolve_dataloader_options(
    cfg: dict[str, Any],
    *,
    device_type: str = "cpu",
    n_samples: int | None = None,
    legacy_training_cfg: dict[str, Any] | None = None,
) -> DataLoaderOptions:
    """解析 DataLoader 选项；所有取值都有 CPU 回退。

    兼容规则（**具体值优先，新分段优先于旧键**）：

    * ``num_workers``：依次看 ``dataloader.num_workers``、``training.num_workers``；
      取第一个**非 auto** 的具体值；两者都是 auto/缺省时按 CPU 核数与数据量推断。
      这样「实验配置里写死 ``training.num_workers: 4``」不会被基线里的 auto 覆盖，
      反之新的 ``dataloader.num_workers`` 也能覆盖旧的同名旧键。
    * ``persistent_workers`` / ``prefetch_factor`` 在 ``num_workers == 0`` 时强制失效
      （PyTorch 会直接抛 ``ValueError``）。
    * ``pin_memory`` 在 CPU/MPS 上默认关闭（无意义且浪费内存），CUDA 上默认开启。
    * ``non_blocking`` 仅在 CUDA 上启用（与 pin_memory 配合才有意义）。
    """
    opts = DataLoaderOptions()
    dl_cfg = dict(cfg.get("dataloader") or {})
    legacy = dict(legacy_training_cfg or {})

    raw_workers = _first_concrete(
        dl_cfg.get("num_workers"),
        legacy.get("num_workers"),
    )
    if raw_workers is None:
        workers, notes = _auto_num_workers(n_samples)
        opts.notes.extend(notes)
    else:
        try:
            workers = int(raw_workers)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"dataloader.num_workers 非法: {raw_workers!r}",
                hint="使用非负整数或 'auto'",
            ) from exc
        if workers < 0:
            raise ConfigurationError("dataloader.num_workers 不能为负数")
    opts.num_workers = workers

    raw_pin = dl_cfg.get("pin_memory")
    if raw_pin is None:
        opts.pin_memory = device_type == "cuda"
        if device_type != "cuda":
            opts.notes.append("非 CUDA 设备，pin_memory 自动关闭")
    else:
        opts.pin_memory = bool(raw_pin)
        if opts.pin_memory and device_type != "cuda":
            # PyTorch 只在 CUDA 下支持 pinned memory；MPS/CPU 上强制关闭并留痕
            opts.pin_memory = False
            opts.notes.append(
                f"pin_memory 仅在 CUDA 下有效（当前设备 {device_type}），已自动关闭"
            )

    opts.persistent_workers = bool(dl_cfg.get("persistent_workers", workers > 0))
    if opts.persistent_workers and workers == 0:
        opts.persistent_workers = False
        opts.notes.append("num_workers=0 时 persistent_workers 强制关闭（PyTorch 限制）")

    raw_pf = dl_cfg.get("prefetch_factor")
    if raw_pf is not None:
        try:
            pf = int(raw_pf)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"dataloader.prefetch_factor 非法: {raw_pf!r}", hint="使用正整数"
            ) from exc
        if pf < 1:
            raise ConfigurationError("dataloader.prefetch_factor 必须 ≥ 1")
        if workers == 0:
            opts.notes.append("num_workers=0 时 prefetch_factor 失效（PyTorch 限制）")
        else:
            opts.prefetch_factor = pf
    elif workers > 0:
        opts.prefetch_factor = 2
        opts.notes.append("prefetch_factor 未配置，自动取 2")

    opts.drop_last = bool(dl_cfg.get("drop_last", False))
    opts.non_blocking = bool(dl_cfg.get("non_blocking", device_type == "cuda"))
    if opts.non_blocking and device_type != "cuda":
        opts.non_blocking = False
        opts.notes.append("non_blocking 仅在 CUDA 上有意义，已关闭")
    return opts


# ----------------------------------------------------------------------
def resolve_runtime(
    cfg: dict[str, Any],
    *,
    model_name: str | None = None,
    supports_channels_last: bool | None = None,
    n_samples: int | None = None,
) -> RuntimePlan:
    """解析整份配置 → :class:`RuntimePlan`。

    Parameters
    ----------
    cfg : dict
        完整实验配置（或至少含 ``training`` / ``hardware`` / ``dataloader`` 段）。
    model_name : str | None
        模型名（用于 channels_last 的 auto 判定）。
    supports_channels_last : bool | None
        模型是否支持 channels_last（4D 卷积类为 True；显式给出时优先）。
    """
    cfg = cfg or {}
    training_cfg = dict(cfg.get("training") or {})
    hw_cfg = dict(cfg.get("hardware") or {})

    # -- 设备 ----------------------------------------------------------
    # 具体值优先：hardware.device > training.device > auto
    _raw_device = _first_concrete(hw_cfg.get("device"), training_cfg.get("device"))
    device_spec = str(_raw_device if _raw_device is not None else "auto").lower()
    if device_spec not in _VALID_DEVICES:
        raise ConfigurationError(
            f"未知设备规格 {device_spec!r}",
            hint=f"可选: {' | '.join(_VALID_DEVICES)}",
        )
    device_info = detect_hardware(device_spec)
    if device_info.device_type == "cuda":
        import torch

        device = torch.device("cuda")
    elif device_info.device_type == "mps":
        import torch

        device = torch.device("mps")
    else:
        import torch

        device = torch.device("cpu")

    # -- 精度 ----------------------------------------------------------
    precision_spec = str(hw_cfg.get("precision", "auto") or "auto").lower()
    if precision_spec not in _VALID_PRECISIONS:
        raise ConfigurationError(
            f"未知精度模式 {precision_spec!r}",
            hint=f"可选: {' | '.join(_VALID_PRECISIONS)}",
        )
    supports_bf16 = any(g.supports_bfloat16 for g in device_info.gpu_devices) or None
    precision = resolve_precision(
        precision_spec,
        device_type=device_info.device_type,
        supports_bfloat16=bool(supports_bf16),
    )

    # -- 梯度累积 / 裁剪 ----------------------------------------------
    raw_accum = training_cfg.get("gradient_accumulation_steps")
    accum = 1 if raw_accum is None else int(raw_accum)
    if accum < 1:
        raise ConfigurationError(
            f"training.gradient_accumulation_steps 必须 ≥ 1，当前 {accum}"
        )
    raw_clip = training_cfg.get("gradient_clip_norm", training_cfg.get("grad_clip"))
    clip = None if raw_clip in (None, "", "null") else float(raw_clip)
    if clip is not None and clip <= 0:
        raise ConfigurationError(f"梯度裁剪范数必须为正数，当前 {clip}")

    # -- channels_last --------------------------------------------------
    raw_cl = hw_cfg.get("channels_last", "auto")
    if isinstance(raw_cl, bool):
        channels_last = raw_cl
    else:
        cl_spec = str(raw_cl if raw_cl is not None else "auto").lower()
        if cl_spec in ("true", "1", "yes", "on"):
            channels_last = True
        elif cl_spec in ("false", "0", "no", "off"):
            channels_last = False
        else:
            if supports_channels_last is not None:
                channels_last = bool(supports_channels_last)
            else:
                # auto 判定：只有 4D 卷积类模型值得用 channels_last；
                # 小 MLP / 树模型加了只会增加复杂度而无收益。
                channels_last = str(model_name or "").lower() in (
                    "cnn",
                    "unet",
                    "resnet",
                    "convnext",
                )
    if channels_last and device_info.device_type == "cpu":
        _logger.info("channels_last 在 CPU 上收益有限，仍按配置启用")

    # -- torch.compile --------------------------------------------------
    compile_spec = hw_cfg.get("compile", False)
    compile_enabled = bool(compile_spec) and not isinstance(compile_spec, str)
    if isinstance(compile_spec, str):
        compile_enabled = compile_spec.lower() in ("true", "1", "yes", "on")
    if compile_enabled:
        try:
            import torch  # noqa: F401

            if not hasattr(torch, "compile"):
                compile_enabled = False
                _logger.warning("当前 PyTorch 不支持 torch.compile，已忽略该配置")
        except ImportError:  # pragma: no cover
            compile_enabled = False

    # -- DataLoader ------------------------------------------------------
    dl_opts = resolve_dataloader_options(
        cfg,
        device_type=device_info.device_type,
        n_samples=n_samples,
        legacy_training_cfg=training_cfg,
    )

    raw_bs = training_cfg.get("batch_size", 256)
    batch_size = 256 if raw_bs is None else int(raw_bs)
    if batch_size < 1:
        raise ConfigurationError(f"training.batch_size 必须 ≥ 1，当前 {batch_size}")

    plan = RuntimePlan(
        device=device,
        device_info=device_info,
        precision=precision,
        batch_size=batch_size,
        channels_last=channels_last,
        compile=compile_enabled,
        gradient_accumulation_steps=accum,
        gradient_clip_norm=clip,
        grad_checkpointing=bool(hw_cfg.get("grad_checkpointing", False)),
        dataloader=dl_opts,
    )
    plan.notes.extend(device_info.warnings)
    plan.notes.extend(precision.warnings)
    return plan
