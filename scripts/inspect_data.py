#!/usr/bin/env python3
"""检查数据（等价于 ``cloudwater inspect``，便于未安装 CLI 时直接运行）。

用法::

    python scripts/inspect_data.py --config configs/data.yaml [--quality] [--out logs/quality.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 未安装时兜底

from cloudwater_ml.data import (  # noqa: E402
    format_quality_report,
    inspect_data,
    load_dataset,
    run_quality_checks,
)
from cloudwater_ml.utils.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="数据结构与质量检查（只读）")
    ap.add_argument("--config", default="configs/data.yaml", help="数据配置 YAML")
    ap.add_argument("--quality", action="store_true",
                    help="加载数据值执行质量检查（netcdf 默认跳过，加本开关强制执行）")
    ap.add_argument("--out", default=None, help="质量报告 Markdown 输出路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", cfg)
    info = inspect_data(cfg)
    print("== 数据结构 ==")
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))

    if args.quality or data_cfg.get("adapter") != "netcdf":
        try:
            bundle = load_dataset(cfg)
            report = run_quality_checks(bundle, data_cfg)
        except Exception as exc:
            print(f"== 数据质量 == NOT_RUN: {exc}")
            return 0
        print("== 数据质量 ==")
        s = report["summary"]
        print(f"样本 {s['n_samples']} · 来源 {s['source_counts']} · "
              f"通过 {'是' if s['passed'] else '否'}（警告 {s['n_warnings']} 条）")
        for w in report.get("warnings", []):
            print(f"  - ⚠ {w}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(format_quality_report(report), encoding="utf-8")
            print(f"质量报告已写入: {args.out}")
    else:
        print("== 数据质量 == SKIPPED（netcdf 默认不加载数值；--quality 显式执行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
