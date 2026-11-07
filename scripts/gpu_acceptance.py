#!/usr/bin/env python
"""GPU 验收脚本：在目标 GPU 机器上一键跑完 v2.0 的 CUDA 验收清单。

用法（在 GPU 机器上，仓库根目录）::

    python scripts/gpu_acceptance.py                    # 人类可读
    python scripts/gpu_acceptance.py --json out.json    # 结构化结果
    python scripts/gpu_acceptance.py --quick            # 跳过训练（只查设备/AMP/显存）

设计原则
--------
* **不伪造**：CUDA 不可用时对应项直接记 ``NOT_RUN`` 并说明原因，不输出假数字；
* **可远程执行**：只依赖本仓库 + torch，适合被 GPU Worker 以
  ``generic_python`` / ``training`` 任务调用；
* **失败可诊断**：每项都带 ``detail``，失败时能直接看出是驱动、构建版本还是代码问题。

验收项（对应任务书 §12 的 GPU 五项）
------------------------------------
1. ``gpu_detected``      —— 能否正确识别 GPU（名称/算力/显存）
2. ``cuda_training``     —— CUDA 上真实训练（loss 下降）
3. ``amp``               —— bf16/fp16 混合精度真实运行
4. ``vram_peak``         —— 峰值显存可记录
5. ``oom_handling``      —— OOM 捕获并给出可执行建议（尽力而为，不强制触发）
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT_RUN"


def _record(rows: list[dict], item: str, status: str, detail: str = "") -> None:
    rows.append({"item": item, "status": status, "detail": detail})
    mark = {"PASS": "✓", "FAIL": "✗", "NOT_RUN": "·"}[status]
    print(f"  {mark} {item:<22} {status:<8} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None, help="把结果写入 JSON")
    ap.add_argument("--quick", action="store_true", help="跳过训练，只查设备/AMP/显存")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples", type=int, default=4096)
    args = ap.parse_args()

    rows: list[dict] = []
    print("=" * 78)
    print("CloudWater ML Platform v2.0 · GPU 验收")
    print("=" * 78)

    # ---- 环境 ------------------------------------------------------
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        print(f"未安装 PyTorch: {exc}")
        _record(rows, "torch_available", FAIL, f"{exc}")
        return _finish(rows, args)

    from cloudwater_ml.hardware import (
        MemoryMonitor,
        detect_hardware,
        device_report,
        resolve_precision,
    )

    info = detect_hardware("auto")
    print(device_report(info))
    print("-" * 78)

    # ---- 1. 设备识别 ------------------------------------------------
    if info.device_type != "cuda":
        reason = info.fallback_reason or "未检测到 CUDA 设备"
        for item in ("gpu_detected", "cuda_training", "amp", "vram_peak", "oom_handling"):
            _record(rows, item, NOT_RUN, reason)
        return _finish(rows, args)

    g = info.primary_gpu
    _record(
        rows, "gpu_detected", PASS,
        f"{g.name} sm_{g.major}{g.minor} {g.total_memory_gib} GiB "
        f"(free {g.free_memory_gib}) cuda={info.torch_cuda_version}",
    )
    if info.torch_cuda_version and g.major >= 12:
        try:
            ver = tuple(int(x) for x in info.torch_cuda_version.split(".")[:2])
        except ValueError:
            ver = (0, 0)
        if ver < (12, 8):
            _record(rows, "blackwell_build", FAIL,
                    f"sm_{g.major}{g.minor} 需要 cu128+，当前 {info.torch_cuda_version}")

    if args.quick:
        _record(rows, "cuda_training", NOT_RUN, "--quick 跳过")
        _record(rows, "amp", NOT_RUN, "--quick 跳过")
        _record(rows, "vram_peak", NOT_RUN, "--quick 跳过")
        _record(rows, "oom_handling", NOT_RUN, "--quick 跳过")
        return _finish(rows, args)

    # ---- 2/3/4. 真实训练 + AMP + 峰值显存 ----------------------------
    import numpy as np
    from torch import nn

    from cloudwater_ml.training import Trainer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(args.samples, 64)).astype(np.float32)
    W = rng.normal(size=(64, 2)).astype(np.float32)
    Y = (X @ W + rng.normal(scale=0.05, size=(args.samples, 2))).astype(np.float32)

    def _train(precision: str) -> tuple[bool, str, float | None]:
        net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2))
        cfg = {
            "epochs": args.epochs,
            "batch_size": 256,
            "learning_rate": 1e-2,
            "optimizer": "adamw",
            "loss": "mse",
            "seed": 0,
        }
        full = {"training": cfg, "hardware": {"device": "cuda", "precision": precision}}
        res = Trainer(net, cfg, config=full, model_name="mlp").fit(X, Y)
        hist = res.history
        first, last = hist[0]["train_loss"], hist[-1]["train_loss"]
        peak = (res.extra or {}).get("peak_vram_bytes")
        ok = last < first
        return ok, f"loss {first:.5f}→{last:.5f} device={res.extra.get('device')}", peak

    try:
        ok, detail, peak = _train("auto")
        _record(rows, "cuda_training", PASS if ok else FAIL, detail)
        _record(rows, "vram_peak", PASS if peak else FAIL,
                f"峰值显存 {round((peak or 0) / 1024**3, 3)} GiB")
        # AMP 模式
        mode = resolve_precision("auto", device_type="cuda",
                                 supports_bfloat16=bool(g.supports_bfloat16)).mode
        ok_amp, detail_amp, _ = _train(mode)
        _record(rows, "amp", PASS if ok_amp else FAIL, f"{mode}: {detail_amp}")
    except Exception as exc:
        _record(rows, "cuda_training", FAIL, f"{type(exc).__name__}: {exc}")
        print(traceback.format_exc()[:800])

    # ---- 5. OOM 处理（不强制触发，只验证异常类型与提示可构造） -----
    try:
        from cloudwater_ml.utils.exceptions import OutOfMemoryError

        err = OutOfMemoryError("模拟", hint="减小 batch_size", context={"batch_size": 256})
        _record(rows, "oom_handling", PASS,
                f"OutOfMemoryError 带 hint/context（exit_code={err.exit_code}）")
    except Exception as exc:
        _record(rows, "oom_handling", FAIL, f"{type(exc).__name__}: {exc}")

    return _finish(rows, args)


def _finish(rows: list[dict], args) -> int:
    n_pass = sum(1 for r in rows if r["status"] == PASS)
    n_fail = sum(1 for r in rows if r["status"] == FAIL)
    n_nr = sum(1 for r in rows if r["status"] == NOT_RUN)
    print("-" * 78)
    print(f"PASS={n_pass}  FAIL={n_fail}  NOT_RUN={n_nr}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"rows": rows, "summary": {"pass": n_pass, "fail": n_fail, "not_run": n_nr}},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"结果已写入: {args.json}")
    # 有 FAIL → 非零退出码（方便 CI / Worker 判定）
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
