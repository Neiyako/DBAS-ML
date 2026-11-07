#!/usr/bin/env python3
"""从**已完成的实验目录**重绘出版级插图，并在出图前核对指标一致性。

为什么需要这个脚本：
- 实验在别的机器/别的环境跑完时，结果包里往往**只有指标、没有图**
  （或带着默认样式的图）；本脚本用「结果包里的模型 + 归一化参数 + 数据」
  在本地精确复现测试集预测，**先核对指标与 ``metrics.json`` 完全一致**，
  再据此出图——图上的每个数字都能回指到权威指标文件。
- 出图不依赖训练过程，可在任何时候对历史实验重新渲染。

用法::

    python scripts/make_experiment_figures.py --experiment outputs/experiment_003

产出（默认写回 ``<experiment>/plots/``）::

    parity_density.png        密度散点（预测 vs 真值）
    training_history.png      训练/验证曲线 + 泛化间隙
    residual_diagnostics.png  残差四联诊断
    subgroup_performance.png  季节/区域分组泛化
    spatial_maps.png          真值-预测-偏差三联空间图
    robustness.png            噪声鲁棒性
    error_by_magnitude.png    误差随量级变化
    figures_manifest.json     每张图的标题/口径/关键数字
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cloudwater_ml.data.loader import load_dataset  # noqa: E402
from cloudwater_ml.data.splitting import split_indices  # noqa: E402
from cloudwater_ml.models.base import load_model  # noqa: E402
from cloudwater_ml.visualization.publication import (  # noqa: E402
    plot_error_by_magnitude,
    plot_parity_density,
    plot_residual_diagnostics,
    plot_robustness,
    plot_spatial_maps,
    plot_subgroup_performance,
    plot_training_history,
)

#: 指标核对容差（相对误差）
VERIFY_RTOL = 1e-6
#: **抵消型统计量**（bias = 有符号误差的均值）的额外绝对容差，按该指标的
#: RMSE 量级折算：bias 是两个同量级数相减的结果，float32 推理 + 输入
#: 1e-14 级浮点差异会把它的相对误差放大两个数量级，因此对 bias 用
#: 「相对于误差尺度」的容差判定（默认取 RMSE 的 0.1%）。
VERIFY_CANCEL_ATOL_RATIO = 1e-3


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_history(exp_dir: Path) -> list[dict[str, Any]]:
    csv_path = exp_dir / "history.csv"
    if csv_path.exists():
        import csv

        rows: list[dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                row: dict[str, Any] = {}
                for k, v in rec.items():
                    if k == "epoch":
                        row[k] = int(float(v))
                    else:
                        row[k] = float(v)
                rows.append(row)
        if rows:
            return rows
    hist_json = exp_dir / "logs" / "history.json"
    if hist_json.exists():
        payload = _read_json(hist_json)
        if isinstance(payload, list):
            return [dict(r) for r in payload]
        if isinstance(payload, dict) and isinstance(payload.get("history"), list):
            return [dict(r) for r in payload["history"]]
    return []


def _scalers(pre: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = pre["x"]
    y = pre["y"]
    return (
        np.asarray(x["mean"], dtype=np.float64),
        np.asarray(x["scale"], dtype=np.float64),
        np.asarray(y["mean"], dtype=np.float64),
        np.asarray(y["scale"], dtype=np.float64),
    )


def _reproduce_test_predictions(
    exp_dir: Path, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """复现测试集逐样本预测（与结果包模型/归一化参数一致）。"""
    bundle = load_dataset(cfg)
    splits = split_indices(bundle, cfg.get("split") or {})
    test = bundle.subset(splits.test)

    x_mean, x_scale, y_mean, y_scale = _scalers(_read_json(exp_dir / "preprocessors.json"))
    model = load_model(exp_dir / "model")
    model.feature_names = list(bundle.feature_names)
    model.target_names = list(bundle.target_names)

    x_scaled = (np.asarray(test.X, dtype=np.float64) - x_mean) / x_scale
    y_pred_scaled = np.asarray(model.predict(x_scaled), dtype=np.float64)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_pred = y_pred_scaled * y_scale + y_mean

    lon = np.asarray(test.meta["longitude"], dtype=np.float64)
    lat = np.asarray(test.meta["latitude"], dtype=np.float64)
    return np.asarray(test.Y, dtype=np.float64), y_pred, lon, lat, list(bundle.target_names)


def _metric_block(metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics.get("test") or {}
    if "metrics" in test:
        return test["metrics"]
    return test


def _verify(metrics: dict[str, Any], y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
    """把复现结果与 ``metrics.json`` 逐项比对，返回人类可读的差异行。"""
    problems: list[str] = []
    block = _metric_block(metrics)
    overall = block.get("overall") or {}
    per_target = block.get("per_target") or {}

    def _cmp(name: str, got: float, want: float | None, *, atol: float = 0.0) -> None:
        """比对单项指标：相对容差 + 可选绝对容差（用于抵消型统计量）。"""
        if want is None:
            return
        denom = abs(want) if abs(want) > 0 else 1.0
        rel = abs(got - want) / denom
        if rel > VERIFY_RTOL and abs(got - want) > atol:
            problems.append(
                f"{name}: 复现 {got:.6e} vs 权威 {want:.6e}"
                f"（相对差 {rel:.2e} > {VERIFY_RTOL:.0e}，绝对差 {abs(got - want):.2e}"
                f" > 容差 {atol:.2e}）"
            )

    err = (y_pred - y_true).ravel()
    rmse_overall = float(np.sqrt(np.mean(err**2)))
    _cmp("overall.mae", float(np.mean(np.abs(err))), overall.get("mae"))
    _cmp("overall.rmse", rmse_overall, overall.get("rmse"))
    _cmp("overall.bias", float(np.mean(err)), overall.get("bias"),
         atol=rmse_overall * VERIFY_CANCEL_ATOL_RATIO)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true.ravel() - np.mean(y_true.ravel())) ** 2))
    if ss_tot > 0:
        _cmp("overall.r2", 1.0 - ss_res / ss_tot, overall.get("r2"))

    for j, (nm, blk) in enumerate(per_target.items()):
        if j >= y_true.shape[1] or not isinstance(blk, dict):
            continue
        e = (y_pred[:, j] - y_true[:, j]).ravel()
        rmse_t = float(np.sqrt(np.mean(e**2)))
        _cmp(f"{nm}.mae", float(np.mean(np.abs(e))), blk.get("mae"))
        _cmp(f"{nm}.rmse", rmse_t, blk.get("rmse"))
        _cmp(f"{nm}.bias", float(np.mean(e)), blk.get("bias"),
             atol=rmse_t * VERIFY_CANCEL_ATOL_RATIO)
    return problems


def _read_region_boxes(cfg: dict[str, Any]) -> dict[str, list[float]]:
    ev = cfg.get("evaluation") or {}
    boxes = ev.get("region_boxes") or {}
    return {str(k): list(v) for k, v in boxes.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="重绘实验出版级插图（带指标一致性核对）")
    ap.add_argument("--experiment", default="outputs/experiment_003", help="实验目录")
    ap.add_argument("--out", default=None, help="出图目录（默认 <experiment>/plots）")
    ap.add_argument("--config", default=None, help="配置（默认 <experiment>/config.yaml）")
    ap.add_argument("--note", default=None, help="脚注：数据来源与口径")
    ap.add_argument("--allow-mismatch", action="store_true", help="指标对不上也继续出图（默认中止）")
    args = ap.parse_args()

    exp_dir = Path(args.experiment).resolve()
    if not exp_dir.is_dir():
        print(f"[FAIL] 实验目录不存在: {exp_dir}")
        return 2
    cfg_path = Path(args.config) if args.config else exp_dir / "config.yaml"
    cfg = _read_json(cfg_path) if cfg_path.suffix == ".json" else _yaml_load(cfg_path)
    out_dir = Path(args.out) if args.out else exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = exp_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    exp_meta = metrics.get("experiment") or {}
    exp_name = exp_meta.get("name") or exp_dir.name
    device = exp_meta.get("device", "cpu")
    model_name = exp_meta.get("model", "?")

    print(f"[1/5] 复现测试集预测（{exp_name} · {model_name} · {device}）")
    y_true, y_pred, lon, lat, target_names = _reproduce_test_predictions(exp_dir, cfg)

    if metrics:
        problems = _verify(metrics, y_true, y_pred)
        if problems:
            print("[FAIL] 复现指标与 metrics.json 不一致：")
            for p in problems:
                print("   -", p)
            if not args.allow_mismatch:
                print("[ABORT] 默认中止出图（--allow-mismatch 可强制继续）")
                return 1
        else:
            print(f"      指标核对通过（相对容差 {VERIFY_RTOL:.0e}）")
    else:
        print("      [WARN] 未找到 metrics.json，跳过指标核对")

    note = args.note or (
        f"数据：{cfg.get('data', {}).get('path', 'n/a')}（合成测试数据，source=simulated）"
        f" · 划分：{cfg.get('split', {}).get('strategy', '?')}"
        f" · 实验：{exp_name} · 数值全部来自本次真实运行的 metrics.json"
    )
    subtitle = (
        f"模型 {model_name} · 设备 {device} · 测试集 n={y_true.shape[0]}"
        f" · 目标：{', '.join(target_names)}"
    )

    manifest: dict[str, Any] = {
        "experiment": exp_name,
        "experiment_id": exp_meta.get("experiment_id"),
        "model": model_name,
        "device": device,
        "test_n": int(y_true.shape[0]),
        "targets": target_names,
        "note": note,
        "figures": {},
    }
    figures: list[tuple[str, Any]] = []

    print("[2/5] 密度散点（预测 vs 真值）")
    figures.append(("parity_density.png", plot_parity_density(
        y_true, y_pred, target_names, out_dir / "parity_density.png",
        subtitle=subtitle, note=note)))

    print("[3/5] 训练曲线 + 残差诊断")
    history = _read_history(exp_dir)
    figures.append(("training_history.png", plot_training_history(
        history, out_dir / "training_history.png",
        subtitle=f"训练轮数 {len(history)} · 早停 patience="
                 f"{(cfg.get('training') or {}).get('early_stopping', {}).get('patience', '?')}",
        note=note)))
    figures.append(("residual_diagnostics.png", plot_residual_diagnostics(
        y_true, y_pred, target_names, out_dir / "residual_diagnostics.png",
        subtitle=subtitle, note=note)))

    print("[4/5] 分组泛化 + 空间场")
    subgroups = (metrics.get("test") or {}).get("subgroups") or _metric_block(metrics).get("subgroups") or {}
    figures.append(("subgroup_performance.png", plot_subgroup_performance(
        subgroups, out_dir / "subgroup_performance.png",
        subtitle=subtitle, note=note)))
    figures.append(("spatial_maps.png", plot_spatial_maps(
        lon, lat, y_true, y_pred, target_names, out_dir / "spatial_maps.png",
        region_boxes=_read_region_boxes(cfg),
        subtitle=subtitle, note=note)))

    print("[5/5] 鲁棒性 + 量级-误差")
    robust = metrics.get("robustness") or {}
    figures.append(("robustness.png", plot_robustness(
        robust, out_dir / "robustness.png", subtitle=subtitle, note=note)))
    figures.append(("error_by_magnitude.png", plot_error_by_magnitude(
        y_true, y_pred, target_names, out_dir / "error_by_magnitude.png",
        subtitle=subtitle, note=note)))

    ok = 0
    for name, path in figures:
        if path is None:
            manifest["figures"][name] = {"status": "FAILED"}
            print(f"      [FAIL] {name}")
        else:
            manifest["figures"][name] = {"status": "ok", "path": str(path)}
            ok += 1
            print(f"      [ok] {name}")
    manifest["figures_generated"] = ok
    manifest["figures_total"] = len(figures)
    (out_dir / "figures_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] {ok}/{len(figures)} 张图写入 {out_dir}")
    return 0 if ok == len(figures) else 1


def _yaml_load(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    raise SystemExit(main())
