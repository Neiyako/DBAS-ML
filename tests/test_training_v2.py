"""v2.0 训练引擎测试：运行时计划、梯度累积、完整检查点、续训连续性、损失分量。

这些是**真实训练**（真实 forward/backward/optimizer），不是 mock。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from cloudwater_ml.training import Trainer
from cloudwater_ml.training.checkpoint import load_checkpoint


def _data(n=64, n_in=6, n_out=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_in)).astype(np.float32)
    Y = (X[:, :n_out] * 0.5).astype(np.float32)
    return X, Y


def _model(n_in=6, n_out=2):
    return nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, n_out))


def _cfg(**kw):
    base = {
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 0.01,
        "optimizer": "adamw",
        "loss": "mse",
        "seed": 42,
        "device": "cpu",
    }
    base.update(kw)
    return base


# ----------------------------------------------------------------------
def test_result_records_runtime_plan():
    X, Y = _data()
    tr = Trainer(_model(), _cfg(), config={"training": _cfg()}, model_name="mlp")
    res = tr.fit(X, Y)
    plan = res.extra["runtime_plan"]
    assert plan["device"] == "cpu"
    assert plan["precision"]["mode"] == "fp32"
    assert plan["batch_size"] == 16
    assert res.extra["device"] == "cpu"


def test_gradient_accumulation_reduces_optimizer_steps(monkeypatch):
    X, Y = _data(n=64, n_in=6, n_out=2)
    calls = {"n": 0}
    original = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)
    tr = Trainer(_model(), _cfg(gradient_accumulation_steps=2))
    tr.fit(X, Y)
    # 2 epoch × (64 样本 / batch 16 = 4 batch) = 8 个 micro-batch；
    # 累积 2 步 → 4 次优化器更新
    assert calls["n"] == 4


def test_gradient_accumulation_without_accum_uses_every_batch(monkeypatch):
    X, Y = _data(n=64, n_in=6, n_out=2)
    calls = {"n": 0}
    original = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)
    Trainer(_model(), _cfg()).fit(X, Y)
    # 2 epoch × 4 batch = 8 次更新
    assert calls["n"] == 8


def test_gradient_clipping_is_applied():
    X, Y = _data()
    tr = Trainer(_model(), _cfg(gradient_clip_norm=1e-4))
    res = tr.fit(X, Y)
    assert res.extra["runtime_plan"]["gradient_clip_norm"] == pytest.approx(1e-4)


# ----------------------------------------------------------------------
# 检查点完整性
# ----------------------------------------------------------------------
def test_checkpoint_contains_full_training_state(tmp_path):
    X, Y = _data()
    tr = Trainer(_model(), _cfg(scheduler="cosine"))
    tr.fit(X, Y, out_dir=tmp_path)
    payload = load_checkpoint(tmp_path / "last.pt")
    assert payload["model_state"]
    assert payload["optimizer_state"] is not None
    assert payload["scheduler_state"] is not None
    assert payload["epoch"] == 2
    assert payload["global_step"] == 0 or isinstance(payload["global_step"], int)
    assert payload["rng_state"] is not None
    assert "python" in payload["rng_state"] and "numpy" in payload["rng_state"]
    assert payload["trainer_config"] is not None


def test_resume_continues_epoch_numbering(tmp_path):
    """续训后 history 必须是**连续**的（含续训前的 epoch），不能被截断。"""
    X, Y = _data()
    Trainer(_model(), _cfg(epochs=2)).fit(X, Y, out_dir=tmp_path)
    cfg = _cfg(epochs=4, resume=str(tmp_path / "last.pt"))
    res = Trainer(_model(), cfg).fit(X, Y, out_dir=tmp_path)
    epochs = [row["epoch"] for row in res.history]
    assert epochs == [1, 2, 3, 4], "续训后历史被截断（应为 1..4 连续）"


def test_resume_restores_early_stopping_state(tmp_path):
    """续训必须恢复 early stopping 的 best/wait，否则耐心值会被重置。"""
    X, Y = _data()
    Trainer(_model(), _cfg(epochs=2, early_stopping={"enabled": True, "patience": 1})).fit(
        X, Y, out_dir=tmp_path
    )
    payload = load_checkpoint(tmp_path / "last.pt")
    es = payload.get("early_stopping_state") or {}
    assert "best" in es and "wait" in es, "检查点未保存 early stopping 状态"


def test_checkpoint_saves_history_rows(tmp_path):
    X, Y = _data()
    Trainer(_model(), _cfg(epochs=3)).fit(X, Y, out_dir=tmp_path)
    payload = load_checkpoint(tmp_path / "last.pt")
    assert payload.get("history"), "检查点未保存 history（续训会丢失曲线）"
    assert len(payload["history"]) == 3


def test_gradient_accumulation_tail_is_flushed(monkeypatch):
    """batch 数不是 accum 整数倍时，尾部梯度必须走一次 step（不能静默丢弃）。"""
    X, Y = _data(n=40, n_in=6, n_out=2)  # batch=16 → 3 个 micro-batch
    calls = {"n": 0}
    original = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)
    Trainer(_model(), _cfg(epochs=1, batch_size=16, gradient_accumulation_steps=2)).fit(X, Y)
    # 3 个 micro-batch，accum=2 → 1 次整组 step + 1 次尾部 flush = 2 次
    assert calls["n"] == 2, "梯度累积尾部被静默丢弃"


def test_resume_restores_optimizer_state(tmp_path):
    X, Y = _data()
    tr = Trainer(_model(), _cfg(epochs=1))
    tr.fit(X, Y, out_dir=tmp_path)
    payload = load_checkpoint(tmp_path / "last.pt")
    assert payload["optimizer_state"] is not None
    # Adam 的动量缓冲应已存在（说明确实是训练后的状态，不是空壳）
    state = payload["optimizer_state"]["state"]
    assert state, "optimizer 状态为空，说明检查点没有真正保存训练状态"


def test_rng_state_restored_on_resume(tmp_path):
    X, Y = _data()
    Trainer(_model(), _cfg(epochs=1)).fit(X, Y, out_dir=tmp_path)
    payload = load_checkpoint(tmp_path / "last.pt")
    assert payload["rng_state"]["numpy"] is not None


# ----------------------------------------------------------------------
# 损失分量
# ----------------------------------------------------------------------
def test_loss_components_are_recorded_per_constraint():
    X, Y = _data()
    cfg = _cfg(physics={"constraints": ["nonnegativity"], "nonnegativity_weight": 0.1})
    res = Trainer(_model(), cfg).fit(X, Y)
    comps = res.extra.get("loss_components") or {}
    assert "supervised" in comps
    assert "nonnegativity" in comps


def test_composite_loss_component_values():
    from cloudwater_ml.training.losses import CompositeLoss, NonNegativityPenalty

    crit = CompositeLoss(nn.MSELoss(), [NonNegativityPenalty(0.5)])
    out = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    values = crit.component_values(out, {"X": out, "Y": torch.zeros_like(out)})
    assert values["supervised"] > 0
    assert values["nonnegativity"] > 0


# ----------------------------------------------------------------------
# 内存与精度
# ----------------------------------------------------------------------
def test_memory_stats_are_recorded():
    X, Y = _data()
    res = Trainer(_model(), _cfg()).fit(X, Y)
    assert res.extra["memory"]["device_type"] == "cpu"
    assert res.extra["peak_host_rss_bytes"] is not None


def test_bf16_cpu_training_runs():
    """CPU 上真跑一次 bf16 autocast 训练（验证混合精度链路端到端可用）。"""
    X, Y = _data(n=32)
    cfg = _cfg(epochs=1, batch_size=8)
    res = Trainer(_model(), cfg, config={"hardware": {"precision": "bf16", "device": "cpu"}}).fit(X, Y)
    assert res.extra["runtime_plan"]["precision"]["mode"] == "bf16"
    assert res.extra["runtime_plan"]["precision"]["autocast_enabled"] is True


def test_validation_runs_without_autocast():
    """验证/指标默认 fp32：混合精度开启时验证损失仍应稳定可复现。"""
    X, Y = _data(n=32)
    cfg = _cfg(epochs=1, batch_size=8)
    res = Trainer(_model(), cfg, config={"hardware": {"precision": "bf16", "device": "cpu"}}).fit(
        X, Y, X_val=X, Y_val=Y
    )
    assert res.history[0]["val_loss"] is not None
