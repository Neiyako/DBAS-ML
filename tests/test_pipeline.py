"""端到端测试：CLI 命令、完整实验产物、评估与报告、错误路径。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from cloudwater_ml.cli import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()

MINI_CONFIG = """
extends: [{{root}}/configs/data.yaml, {{root}}/configs/training.yaml]

experiment:
  name: pipeline_test
  description: 端到端冒烟（小规模，仅验证工程链路）

data:
  sampling:
    max_samples: 600

model:
  name: mlp
  params:
    hidden_layers: [16]
    dropout: 0.0

training:
  epochs: 2
  early_stopping:
    enabled: false
    patience: 5

split:
  strategy: time
  ratios: [0.7, 0.15, 0.15]
  params:
    train_end: "2017-12-31T23:59:59"
    val_end: "2018-12-31T23:59:59"
  seed: 42

evaluation:
  metrics: [mae, rmse]
  subgroups: [year]
  robustness:
    noise_levels: [0.0, 0.05]
"""


@pytest.fixture()
def mini_config(tmp_path: Path) -> Path:
    p = tmp_path / "mini.yaml"
    p.write_text(MINI_CONFIG.replace("{{root}}", str(ROOT)), encoding="utf-8")
    return p


def test_cli_help_lists_all_commands():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0, r.output
    for cmd in ("inspect", "train", "evaluate", "report", "synth"):
        assert cmd in r.stdout


def test_cli_inspect_synthetic_data():
    r = runner.invoke(app, ["inspect", "--config", str(ROOT / "configs/data.yaml"), "--json"])
    assert r.exit_code == 0, r.stdout + str(r.exception)
    start = r.stdout.find("{")
    payload = json.loads(r.stdout[start:])
    assert payload["structure"]["adapter"] == "npz"
    assert payload["structure"]["requested_features"]
    assert payload["quality"]["summary"]["n_samples"] >= 100
    assert payload["quality"]["summary"]["source_counts"].get("simulated", 0) > 0


def test_cli_synth_generates_reproducible_file(tmp_path: Path):
    out = tmp_path / "syn" / "s.npz"
    r = runner.invoke(app, ["synth", "--out", str(out), "--n", "120", "--seed", "7"])
    assert r.exit_code == 0, r.stdout
    assert out.exists()
    import numpy as np

    with np.load(out, allow_pickle=False) as z:
        assert z["feature__t"].shape == (120,)
        assert set(z["meta__source"].tolist()) == {"simulated"}


def _artifacts_ok(exp: Path) -> None:
    required = [
        "config.yaml", "config.source.yaml", "metrics.json", "history.csv",
        "report.md", "preprocessors.json", "model/model_meta.json",
        "logs/quality_report.md", "logs/train.log",
        "plots/loss_curve.png", "plots/prediction_vs_truth.png",
        "plots/residual_distribution.png",
    ]
    missing = [rel for rel in required if not (exp / rel).exists()]
    assert not missing, f"缺少实验产物: {missing}"


def test_end_to_end_train_evaluate_report(mini_config: Path, tmp_path: Path):
    outputs = tmp_path / "outputs"

    # ---- train ----
    r = runner.invoke(app, ["train", "--config", str(mini_config), "--outputs", str(outputs)])
    assert r.exit_code == 0, r.stdout + str(r.exception)
    exp = outputs / "experiment_001"
    assert exp.exists()
    _artifacts_ok(exp)

    metrics = json.loads((exp / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["experiment"]["model"] == "mlp"
    assert metrics["split"]["strategy"] == "time"
    assert metrics["split"]["n_train"] > metrics["split"]["n_val"] > 0
    overall = metrics["test"]["metrics"]["overall"]
    assert {"mae", "rmse"} <= set(overall) and "r2" not in overall  # 按配置过滤
    assert metrics["robustness"] and {"0.0", "0.05"} <= set(metrics["robustness"])
    assert metrics["preprocessing"]["fitted_on"] == "train"
    assert metrics["train"]["n_epochs"] == 2
    assert metrics["uncertainty"]["status"] in ("disabled", "reserved")

    hist = pd.read_csv(exp / "history.csv")
    assert len(hist) == 2 and {"epoch", "train_loss", "val_loss"} <= set(hist.columns)

    report = (exp / "report.md").read_text(encoding="utf-8")
    assert "实验报告" in report and "测试集指标" in report
    quality = (exp / "logs/quality_report.md").read_text(encoding="utf-8")
    assert "数据质量报告" in quality

    # ---- evaluate（同一模型重算，应与训练时指标一致）----
    r2 = runner.invoke(app, ["evaluate", "--experiment", exp.name, "--outputs", str(outputs)])
    assert r2.exit_code == 0, r2.stdout + str(r2.exception)
    ev = json.loads((exp / "evaluation.json").read_text(encoding="utf-8"))
    assert ev["n_test"] == metrics["test"]["n_samples"]
    assert ev["metrics"]["overall"]["mae"] == pytest.approx(overall["mae"], rel=1e-6)

    # ---- report（单实验 + 横向对比）----
    r3 = runner.invoke(app, ["report", "--experiment", exp.name, "--outputs", str(outputs)])
    assert r3.exit_code == 0 and "实验报告" in r3.stdout
    r4 = runner.invoke(app, ["report", "--outputs", str(outputs)])
    assert r4.exit_code == 0 and "实验横向对比" in r4.stdout and exp.name in r4.stdout

    # ---- 拒绝覆盖已有实验 ----
    r5 = runner.invoke(app, ["train", "--config", str(mini_config),
                             "--outputs", str(outputs), "--name", exp.name])
    assert r5.exit_code == 1
    combined = r5.stdout + (getattr(r5, "stderr", "") or "")
    assert "已存在" in combined


def test_cli_train_missing_config_clean_error(tmp_path: Path):
    r = runner.invoke(app, ["train", "--config", str(tmp_path / "nope.yaml"),
                            "--outputs", str(tmp_path / "out")])
    assert r.exit_code == 1
    combined = r.stdout + (getattr(r, "stderr", "") or "")
    assert "错误" in combined and ("不存在" in combined or "No such file" in combined)


def test_script_entry_points(tmp_path: Path):
    """scripts/*.py 与 CLI 等价可用（subprocess 级验证）。"""
    import subprocess
    import sys

    out = tmp_path / "outputs"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_experiment.py"),
         "--config", str(tmp_path / "mini2.yaml"), "--outputs", str(out)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr[-300:]
    assert "实验失败" in proc.stderr and "mini2.yaml" in proc.stderr
    # 正常路径
    (tmp_path / "mini2.yaml").write_text(
        MINI_CONFIG.replace("{{root}}", str(ROOT)), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_experiment.py"),
         "--config", str(tmp_path / "mini2.yaml"), "--outputs", str(out)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert (out / "experiment_001" / "metrics.json").exists()

    proc2 = subprocess.run(
        [sys.executable, str(ROOT / "scripts/inspect_data.py"),
         "--config", str(ROOT / "configs/data.yaml")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc2.returncode == 0, proc2.stderr[-400:]
    assert "数据结构" in proc2.stdout
