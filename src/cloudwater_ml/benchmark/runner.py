"""训练性能基准（M6）：吞吐、瓶颈归因、显存峰值与配置建议。

原则
----

1. **不伪造数据**：所有数字来自真实跑出来的训练（真实 forward/backward），
   跑不了的项记 ``None`` 并说明原因（例如本机无 CUDA 时 GPU 利用率为空）。
2. **建议必须随硬件变化**：显存/算力建议由**运行时探测到的**显存与算力推导，
   **不写死任何具体型号的显存容量**（不同批次的 50 系卡容量并不相同）。
3. **CPU 环境可跑**：基准必须在没有 GPU 的机器上给出基本结果（CPU 基线）。
4. **瓶颈归因**：分别统计「取数据」「搬到设备」「计算」三段耗时，
   让人知道该调 DataLoader 还是调模型。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..hardware import detect_hardware
from ..utils.logging import get_logger

__all__ = [
    "BenchmarkResult",
    "benchmark_training",
    "run_benchmark_suite",
    "recommend_config",
    "format_report",
]

_logger = get_logger("benchmark.runner")


@dataclass
class BenchmarkResult:
    """一次基准测试的结果。"""

    name: str = ""
    device: str = "cpu"
    precision: str = "fp32"
    batch_size: int = 0
    n_samples: int = 0
    epochs: int = 0
    epoch_seconds: float = 0.0
    data_load_seconds: float = 0.0
    compute_seconds: float = 0.0
    transfer_seconds: float = 0.0
    samples_per_second: float = 0.0
    peak_vram_bytes: int | None = None
    peak_host_rss_bytes: int | None = None
    gpu_utilization: dict[str, Any] | None = None
    final_loss: float | None = None
    config: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["peak_vram_gib"] = (
            round(self.peak_vram_bytes / (1024**3), 3) if self.peak_vram_bytes else None
        )
        d["peak_host_rss_gib"] = (
            round(self.peak_host_rss_bytes / (1024**3), 3) if self.peak_host_rss_bytes else None
        )
        return d

    @property
    def bottleneck(self) -> str:
        """粗判瓶颈：数据供给 vs 计算。"""
        total = self.data_load_seconds + self.compute_seconds + self.transfer_seconds
        if total <= 0:
            return "unknown"
        if self.data_load_seconds / total > 0.5:
            return "data_loading"
        if self.compute_seconds / total > 0.7:
            return "compute"
        return "balanced"


# ----------------------------------------------------------------------
def _make_dataset(n_samples: int, n_features: int, n_targets: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    W = rng.normal(size=(n_features, n_targets)).astype(np.float32)
    Y = (X @ W + rng.normal(scale=0.1, size=(n_samples, n_targets))).astype(np.float32)
    return X, Y


def benchmark_training(
    *,
    name: str = "baseline",
    device: str = "auto",
    precision: str = "auto",
    batch_size: int = 256,
    n_samples: int = 8192,
    n_features: int = 64,
    n_targets: int = 2,
    hidden_layers: tuple[int, ...] = (256, 128),
    epochs: int = 2,
    num_workers: Any = "auto",
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    channels_last: bool = False,
    seed: int = 42,
) -> BenchmarkResult:
    """跑一次真实训练基准，返回吞吐 / 三段耗时 / 峰值显存。"""
    from torch import nn

    from ..training.runtime import resolve_runtime
    from ..training.trainer import Trainer

    X, Y = _make_dataset(n_samples, n_features, n_targets, seed=seed)

    layers: list[nn.Module] = []
    prev = n_features
    for h in hidden_layers:
        layers += [nn.Linear(prev, int(h)), nn.ReLU()]
        prev = int(h)
    layers.append(nn.Linear(prev, n_targets))
    model = nn.Sequential(*layers)

    cfg: dict[str, Any] = {
        "training": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "loss": "mse",
            "seed": int(seed),
        },
        "hardware": {"device": device, "precision": precision},
        "dataloader": {"num_workers": num_workers},
    }
    if pin_memory is not None:
        cfg["dataloader"]["pin_memory"] = bool(pin_memory)
    if persistent_workers is not None:
        cfg["dataloader"]["persistent_workers"] = bool(persistent_workers)
    if prefetch_factor is not None:
        cfg["dataloader"]["prefetch_factor"] = int(prefetch_factor)

    plan = resolve_runtime(cfg, model_name="mlp", n_samples=n_samples)
    trainer = Trainer(model, cfg["training"], config=cfg, model_name="mlp")

    t0 = time.perf_counter()
    result = trainer.fit(X, Y)
    wall = time.perf_counter() - t0

    prof = (result.extra or {}).get("profiling") or {}
    stages = prof.get("stages") or {}

    def stage_total(name: str) -> float:
        return float((stages.get(name) or {}).get("total_seconds", 0.0))

    data_s = stage_total("data_load")
    transfer_s = stage_total("transfer")
    compute_s = max(wall - data_s - transfer_s, 0.0)

    out = BenchmarkResult(
        name=name,
        device=str(trainer.device),
        precision=plan.precision.mode,
        batch_size=int(batch_size),
        n_samples=int(n_samples),
        epochs=int(epochs),
        epoch_seconds=wall / max(epochs, 1),
        data_load_seconds=data_s,
        compute_seconds=compute_s,
        transfer_seconds=transfer_s,
        samples_per_second=(n_samples * epochs) / wall if wall > 0 else 0.0,
        peak_vram_bytes=(result.extra or {}).get("peak_vram_bytes"),
        peak_host_rss_bytes=(result.extra or {}).get("peak_host_rss_bytes"),
        final_loss=result.final_train_loss,
        config={
            "device": device,
            "precision": precision,
            "batch_size": int(batch_size),
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "epochs": int(epochs),
            "num_workers": plan.dataloader.num_workers,
            "pin_memory": plan.dataloader.pin_memory,
            "channels_last": plan.channels_last,
        },
    )
    if plan.device_type == "cuda":
        from ..hardware.profiling import sample_gpu_utilization

        out.gpu_utilization = sample_gpu_utilization()
    out.notes.extend(plan.notes)
    return out


# ----------------------------------------------------------------------
def recommend_config(
    *,
    device_info: Any = None,
    model_kind: str = "mlp",
    n_features: int = 64,
) -> dict[str, Any]:
    """给出配置建议（**按实测硬件推导**，不写死任何型号的显存容量）。"""
    info = device_info or detect_hardware("auto")
    rec: dict[str, Any] = {
        "device": info.device_type,
        "precision": "auto",
        "batch_size": 256,
        "gradient_accumulation_steps": 1,
        "num_workers": 4,
        "pin_memory": info.device_type == "cuda",
        "persistent_workers": True,
        "prefetch_factor": 2,
        "channels_last": model_kind.lower() in ("cnn", "unet"),
        "compile": False,
        "rationale": [],
    }
    why = rec["rationale"]

    if info.device_type == "cuda":
        vram_gib = info.total_vram_bytes / (1024**3)
        free_gib = info.free_vram_bytes / (1024**3)
        rec["vram_total_gib"] = round(vram_gib, 2)
        rec["vram_free_gib"] = round(free_gib, 2)
        rec["precision"] = "bf16" if any(g.supports_bfloat16 for g in info.gpu_devices) else "fp16"
        why.append(
            f"CUDA 设备显存 {vram_gib:.1f} GiB（可用 {free_gib:.1f} GiB），"
            f"精度建议 {rec['precision']}"
        )
        # 按「实测可用显存」给 batch 建议：留 40% 余量给激活/优化器状态
        usable = max(free_gib, 0.5) * 0.6
        # 粗略估算：每个样本激活占用随特征数线性增长（经验系数，仅用于起步）
        per_sample_mb = max(0.05, n_features * 0.004)
        est_batch = int(usable * 1024 / per_sample_mb)
        rec["batch_size"] = max(32, min(1024, est_batch))
        why.append(
            f"按可用显存 {free_gib:.1f} GiB 与特征数 {n_features} 估算，建议 batch_size≈{rec['batch_size']}"
        )
        if est_batch < 64:
            rec["gradient_accumulation_steps"] = 4
            why.append("显存偏紧：建议 gradient_accumulation_steps=4 维持等效批量")
        rec["num_workers"] = 4
        rec["pin_memory"] = True
        rec["compile"] = False
        why.append("torch.compile 默认关闭（首次编译有额外耗时，收益需实测再开）")
    elif info.device_type == "mps":
        rec["precision"] = "fp16"
        rec["pin_memory"] = False
        rec["batch_size"] = 128
        why.append("Apple MPS：fp16 autocast；pin_memory 不适用（PyTorch 不支持）")
    else:
        rec["precision"] = "fp32"
        rec["batch_size"] = 128
        rec["num_workers"] = 2
        why.append("CPU 环境：fp32 + 少量 worker；建议用小模型/少样本做链路验证")
    if info.fallback_reason:
        why.append(f"注意：{info.fallback_reason}")
    return rec


# ----------------------------------------------------------------------
def run_benchmark_suite(
    *,
    devices: tuple[str, ...] = ("auto",),
    precisions: tuple[str, ...] = ("auto",),
    batch_sizes: tuple[int, ...] = (128, 256),
    n_samples: int = 8192,
    n_features: int = 64,
    epochs: int = 2,
    num_workers: Any = "auto",
) -> dict[str, Any]:
    """跑一组基准（设备 × 精度 × batch size），返回可对比的结果表。"""
    results: list[BenchmarkResult] = []
    for dev in devices:
        for prec in precisions:
            for bs in batch_sizes:
                name = f"dev={dev} prec={prec} bs={bs}"
                try:
                    res = benchmark_training(
                        name=name,
                        device=dev,
                        precision=prec,
                        batch_size=bs,
                        n_samples=n_samples,
                        n_features=n_features,
                        epochs=epochs,
                        num_workers=num_workers,
                    )
                except Exception as exc:
                    _logger.warning("基准用例失败 %s: %s", name, exc)
                    continue
                results.append(res)
                _logger.info(
                    "%s → %.1f samples/s（瓶颈: %s）",
                    name, res.samples_per_second, res.bottleneck,
                )
    info = detect_hardware("auto")
    return {
        "device_info": info.to_dict(),
        "results": [r.to_dict() for r in results],
        "recommendations": recommend_config(device_info=info, n_features=n_features),
        "notes": [
            "所有数值来自本次真实训练；不可测项（如无 CUDA 时的 GPU 利用率）为空。",
        ],
    }


# ----------------------------------------------------------------------
def format_report(payload: dict[str, Any]) -> str:
    """把基准结果格式化成 Markdown 报告。"""
    lines: list[str] = ["# 训练性能基准报告", ""]
    info = payload.get("device_info") or {}
    lines += [
        "## 运行环境",
        f"- 设备类型: `{info.get('device_type')}`",
        f"- 设备摘要: {info.get('device_summary', '')}",
        f"- PyTorch: {info.get('torch_version')}",
        f"- CUDA 构建: {info.get('torch_cuda_version') or '无'}",
        f"- 可见 GPU 数: {info.get('cuda_device_count', 0)}",
        "",
    ]
    lines += ["## 实测结果", ""]
    results = payload.get("results") or []
    if results:
        lines += [
            "| 用例 | 设备 | 精度 | batch | samples/s | epoch(s) | 数据(s) | 计算(s) | 峰值显存(GiB) | 峰值内存(GiB) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in results:
            lines.append(
                f"| {r['name']} | {r['device']} | {r['precision']} | {r['batch_size']} | "
                f"{r['samples_per_second']:.1f} | {r['epoch_seconds']:.3f} | "
                f"{r['data_load_seconds']:.3f} | {r['compute_seconds']:.3f} | "
                f"{r['peak_vram_gib'] if r['peak_vram_gib'] is not None else '—'} | "
                f"{r['peak_host_rss_gib'] if r['peak_host_rss_gib'] is not None else '—'} |"
            )
    else:
        lines.append("_本次没有成功的用例_")
    rec = payload.get("recommendations") or {}
    lines += ["", "## 配置建议（按实测硬件推导）", ""]
    if rec:
        lines.append("```json")
        lines.append(json_dumps(rec))
        lines.append("```")
    notes = payload.get("notes") or []
    if notes:
        lines += ["", "## 备注", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
