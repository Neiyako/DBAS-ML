"""实验目录管理与结果追踪：统一的实验产物格式（各模型横向可比）。"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.config import save_config

__all__ = [
    "DEFAULT_OUTPUTS",
    "create_experiment_dir",
    "write_json",
    "write_history",
    "save_config_snapshot",
    "write_report",
    "build_comparison_table",
    "load_metrics",
]

DEFAULT_OUTPUTS = "outputs"


def _json_default(obj: Any) -> Any:
    """JSON 序列化兜底：numpy / pandas / Path → 可序列化类型。"""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"无法 JSON 序列化: {type(obj)}")


def create_experiment_dir(
    outputs_root: str | Path = DEFAULT_OUTPUTS,
    name: str | None = None,
    *,
    exist_ok: bool = False,
    force_overwrite: bool = False,
    max_attempts: int = 1000,
) -> Path:
    """创建 outputs/experiment_XXX 目录结构（model/ logs/ plots/）。

    - name=None → 自动递增 experiment_XXX；**目录创建是原子的**
      （``mkdir(exist_ok=False)``），并发跑多个实验不会算出同一个编号；
    - name 显式给定且目录已存在且非空 → 默认报错，**拒绝静默覆盖已有实验产物**；
    - ``force_overwrite=True`` 时才允许覆盖，且旧目录会先被重命名备份为
      ``<name>.backup-<时间戳>``（不删除原始结果）。
    """
    root = Path(outputs_root)
    root.mkdir(parents=True, exist_ok=True)

    if name:
        exp = root / name
        if exp.exists() and any(exp.iterdir()):
            if not force_overwrite:
                raise FileExistsError(
                    f"实验目录已存在且非空: {exp}。"
                    "拒绝覆盖已有实验结果；如确需覆盖请显式传 force_overwrite（旧目录会被备份）。"
                )
            backup = exp.with_name(f"{exp.name}.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            shutil.move(str(exp), str(backup))
        exp.mkdir(parents=True, exist_ok=True)
    else:
        used = []
        for p in root.iterdir():
            m = re.fullmatch(r"experiment_(\d+)", p.name)
            if m:
                used.append(int(m.group(1)))
        nxt = max(used, default=0) + 1
        exp = root / f"experiment_{nxt:03d}"
        # 原子创建：并发时若被别人抢先占号，自增重试（最多 max_attempts 次）
        for _ in range(max_attempts):
            try:
                exp.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                nxt += 1
                exp = root / f"experiment_{nxt:03d}"
        else:  # pragma: no cover - 极端并发
            raise FileExistsError(
                f"无法在 {root} 下分配实验目录（尝试 {max_attempts} 次）"
            )
    for sub in ("model", "logs", "plots"):
        (exp / sub).mkdir(parents=True, exist_ok=True)
    return exp


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return path


def load_metrics(exp_dir: str | Path) -> dict[str, Any]:
    path = Path(exp_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"实验目录缺少 metrics.json: {exp_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_history(history: list[dict[str, Any]] | pd.DataFrame, exp_dir: str | Path) -> Path:
    """训练历史 → history.csv（每 epoch 一行；sklearn 基线为单行）。"""
    df = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    path = Path(exp_dir) / "history.csv"
    df.to_csv(path, index=False)
    return path


def save_config_snapshot(
    cfg: dict[str, Any], exp_dir: str | Path, source_config: str | Path | None = None
) -> list[Path]:
    """保存展开后的 config.yaml + 原始配置文件副本（含注释，保证可追溯复现）。"""
    exp_dir = Path(exp_dir)
    paths = [save_config(cfg, exp_dir / "config.yaml")]
    if source_config and Path(source_config).exists():
        dst = exp_dir / "config.source.yaml"
        shutil.copyfile(source_config, dst)
        paths.append(dst)
    return paths


def write_report(exp_dir: str | Path, metrics: dict[str, Any]) -> Path:
    """由 metrics.json 生成 report.md（区分事实与预留，不写虚构结论）。"""
    exp_dir = Path(exp_dir)
    m = metrics
    lines: list[str] = [f"# 实验报告 · {m.get('experiment_name', exp_dir.name)}", ""]

    def sec(title: str) -> None:
        lines.extend(["", f"## {title}", ""])

    sec("基本信息")
    info = m.get("experiment", {})
    lines += [
        f"- 模型: `{info.get('model')}`   任务: 回归（{'、'.join(m.get('targets', []))}）",
        f"- 配置: `{info.get('config_path')}`   seed: {info.get('seed')}   device: {info.get('device')}",
        f"- 运行时间: {info.get('finished_at')}   耗时: {info.get('duration_sec')} s",
        f"- 环境: {m.get('environment', {}).get('python', '')} / "
        f"cloudwater-ml {m.get('environment', {}).get('cloudwater_ml', '')}",
    ]

    sec("数据与划分")
    d = m.get("data", {})
    s = m.get("split", {})
    lines += [
        f"- 数据: `{d.get('path')}`（适配器 {d.get('adapter')}，来源分布 {d.get('source_counts')}）",
        f"- 划分策略: `{s.get('strategy')}`  train/val/test = "
        f"{s.get('n_train')}/{s.get('n_val')}/{s.get('n_test')}"
        + (f"（丢弃 {s.get('n_dropped')}）" if s.get("n_dropped") else ""),
    ]
    for n in s.get("notes", []):
        lines.append(f"  - {n}")
    mix = m.get("mix") or {}
    if mix.get("applied"):
        lines.append(
            f"- 训练集真实/仿真配比: 请求 real_fraction={mix.get('requested')}，"
            f"实际 real={mix.get('actual_real_fraction', 0.0):.3f}（real={mix.get('n_real')}, "
            f"sim={mix.get('n_simulated')}）"
        )
        for n in mix.get("notes", []):
            lines.append(f"  - ⚠ {n}")
    else:
        lines.append(
            f"- 训练集真实/仿真配比: 未启用混合（mix.real_fraction=null）；"
            f"训练集来源 real={mix.get('n_real', 0)} / simulated={mix.get('n_simulated', 0)}"
        )

    sec("测试集指标（原单位）")
    per = (m.get("test", {}).get("metrics", {}) or {}).get("per_target", {})
    overall = (m.get("test", {}).get("metrics", {}) or {}).get("overall", {})
    if per:
        names = list(next(iter(per.values())).keys())
        lines.append("| 目标 | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for tgt, vals in per.items():
            cells = []
            for k in names:
                v = vals.get(k)
                cells.append("—" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)))
            lines.append(f"| {tgt} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"**Overall**: {json.dumps(overall, ensure_ascii=False)}")
        if any(v.get("r2") is None for v in per.values()):
            lines.append("> 注：部分目标 R²=None（目标近似常数，R² 不可解释，见 metrics.json）。")
    else:
        lines.append("_无测试集指标_")

    sub = m.get("test", {}).get("subgroups", {}) or {}
    if sub:
        sec("泛化分组误差（跨年份 / 跨季节 / 跨区域）")
        lines.append("```json")
        lines.append(json.dumps(sub, ensure_ascii=False, indent=2, default=str))
        lines.append("```")

    rob = m.get("robustness", {}) or {}
    if rob:
        sec("鲁棒性（不同噪声强度）")
        lines.append("| noise_level | rmse | mae | r2 |")
        lines.append("|---|---|---|---|")
        for lvl, vals in rob.items():
            if isinstance(vals, dict) and "overall" in vals:
                o = vals["overall"]
                lines.append(
                    f"| {lvl} | {o.get('rmse')} | {o.get('mae')} | {o.get('r2')} |"
                )

    unc = m.get("uncertainty", {}) or {}
    if unc.get("status") == "reserved":
        sec("不确定性")
        lines.append(f"_预留未实现_: {unc.get('note')}")

    notes = m.get("notes", []) or []
    if notes:
        sec("运行备注（含警告，不隐瞒）")
        lines += [f"- {n}" for n in notes]

    sec("产物清单")
    lines += [
        "- `config.yaml`（展开后配置快照）/ `config.source.yaml`（原始配置）",
        "- `metrics.json` / `history.csv` / `report.md`",
        "- `model/`（最佳与最终权重 + model_meta.json）/ `preprocessors.json`（仅 train 拟合的归一化参数）",
        "- `logs/train.log` / `plots/*.png`",
        "",
        "> 本报告由框架自动生成，数值全部来自本次真实运行；未运行的项目标注为预留/NOT_RUN，不含虚构结果。",
    ]
    path = exp_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_comparison_table(outputs_root: str | Path = DEFAULT_OUTPUTS) -> pd.DataFrame:
    """汇总 outputs/*/metrics.json → 横向对比表（统一实验结果格式）。"""
    rows: list[dict[str, Any]] = []
    root = Path(outputs_root)
    if not root.exists():
        return pd.DataFrame()
    for exp in sorted(root.iterdir()):
        mj = exp / "metrics.json"
        if not mj.exists():
            continue
        try:
            m = json.loads(mj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        overall = (m.get("test", {}).get("metrics", {}) or {}).get("overall", {}) or {}
        row = {
            "experiment": exp.name,
            "model": (m.get("experiment") or {}).get("model"),
            "split": (m.get("split") or {}).get("strategy"),
            "n_train": (m.get("split") or {}).get("n_train"),
            **{f"test_{k}": v for k, v in overall.items()},
        }
        rows.append(row)
    return pd.DataFrame(rows)
