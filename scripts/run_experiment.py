#!/usr/bin/env python3
"""运行实验（等价于 ``cloudwater train``）。

用法::

    python scripts/run_experiment.py --config configs/experiments/mlp_baseline.yaml
    python scripts/run_experiment.py --config ... --name experiment_001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 未安装时兜底

from cloudwater_ml.experiments.runner import run_experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="运行一次完整实验")
    ap.add_argument("--config", required=True, help="实验配置 YAML")
    ap.add_argument("--outputs", default="outputs", help="实验输出根目录")
    ap.add_argument("--name", default=None, help="指定实验目录名（默认自动 experiment_XXX）")
    args = ap.parse_args()

    try:
        result = run_experiment(args.config, outputs_root=args.outputs, output_name=args.name)
    except Exception as exc:
        print(f"实验失败: {exc}", file=sys.stderr)
        return 1
    m = result["metrics"]
    overall = ((m.get("test") or {}).get("metrics") or {}).get("overall") or {}
    print(f"✓ 实验完成: {result['experiment']} → {result['dir']}")
    print(f"  测试集 overall: {overall}")
    print(f"  报告: {Path(result['dir']) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
