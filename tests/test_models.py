"""模型接口测试：注册表、MLP 前向与线性输出头、RF/LGBM 多目标、预留模型诚实报错、保存恢复。"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from cloudwater_ml.data import make_synthetic_bundle
from cloudwater_ml.models import available_models, build_model, load_model
from cloudwater_ml.models.base import MODEL_REGISTRY, BaseModel, TrainingResult

TRAIN_CFG = {
    "training": {
        "epochs": 3,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "loss": "mse",
        "seed": 42,
        "device": "auto",
        "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.0},
    }
}


@pytest.fixture(scope="module")
def data():
    b = make_synthetic_bundle(n_samples=300, seed=11)
    return b.X, b.Y, b.feature_names, b.target_names


def _fit(model, X, Y, features, targets, cfg=None, out_dir=None) -> TrainingResult:
    model.feature_names = list(features)
    model.target_names = list(targets)
    return model.fit(X, Y, X_val=X[:50], Y_val=Y[:50], config=cfg or TRAIN_CFG, out_dir=out_dir)


# ---------------------------------------------------------------- 注册表
def test_required_models_registered():
    for name in ("mlp", "random_forest", "lightgbm"):
        assert name in available_models(), f"{name} 未注册"


def test_unknown_model_lists_available():
    with pytest.raises(KeyError, match="已注册"):
        build_model("transformer")


# ---------------------------------------------------------------- MLP
def test_mlp_forward_shapes_single_and_multi_target(data):
    X, Y, features, targets = data
    mlp = build_model("mlp", {"hidden_layers": [32, 16], "dropout": 0.1})
    r = _fit(mlp, X, Y, features, targets)
    assert isinstance(r, TrainingResult)
    assert len(r.history) >= 1 and r.history[0]["train_loss"] >= 0
    pred = mlp.predict(X)
    assert pred.shape == (300, 2)
    assert np.isfinite(pred).all()

    single = build_model("mlp", {"hidden_layers": [16]})
    _fit(single, X, Y[:, :1], features, ["clwc"])
    assert single.predict(X).shape == (300, 1)


def test_mlp_history_matches_config_epochs(data):
    X, Y, features, targets = data
    mlp = build_model("mlp", {"hidden_layers": [8]})
    r = _fit(mlp, X, Y, features, targets,
             cfg={"training": {**TRAIN_CFG["training"], "epochs": 2,
                               "early_stopping": {"enabled": False, "patience": 99}}})
    assert len(r.history) == 2  # 早停关闭时与 epochs 一致
    assert "val_loss" in r.history[0]


def test_mlp_output_layer_is_linear_not_relu(data):
    """输出层非 ReLU 的行为学验证：可输出负值预测。"""
    X, _, features, _ = data
    # 构造带大幅负值的目标（标准化后的 t 中心化即含负值）
    Yneg = (X[:, 0] - X[:, 0].mean()) / (X[:, 0].std() + 1e-12)
    Yneg = Yneg.reshape(-1, 1)
    mlp = build_model("mlp", {"hidden_layers": [32], "activation": "relu", "dropout": 0.0})
    cfg = {"training": {**TRAIN_CFG["training"], "epochs": 60, "learning_rate": 0.01,
                        "early_stopping": {"enabled": False, "patience": 999}}}
    mlp.feature_names, mlp.target_names = features, ["y_neg"]
    mlp.fit(X[:240], Yneg[:240], X_val=X[240:], Y_val=Yneg[240:], config=cfg)
    pred = mlp.predict(X[240:])
    assert pred.min() < 0, "ReLU 输出头永远非负——违反『输出层不默认 ReLU』要求"


def test_mlp_activation_configurable(data):
    X, Y, features, targets = data
    for act in ("gelu", "tanh"):
        m = build_model("mlp", {"hidden_layers": [8], "activation": act})
        _fit(m, X, Y, features, targets,
             cfg={"training": {**TRAIN_CFG["training"], "epochs": 1}})
        assert np.isfinite(m.predict(X[:10])).all()


# ---------------------------------------------------------------- Random Forest
def test_random_forest_multitarget_and_importance(data):
    X, Y, features, targets = data
    rf = build_model("random_forest", {"n_estimators": 15, "random_state": 0, "n_jobs": 1})
    r = _fit(rf, X, Y, features, targets)
    pred = rf.predict(X)
    assert pred.shape == (300, 2)
    imp = rf.feature_importance()
    assert imp is not None and set(imp) == set(features)
    imp_vals = [float(v) for v in imp.values()]
    assert all(np.isfinite(v) and v >= 0 for v in imp_vals)
    # sklearn 没有逐 epoch 循环：历史只能是单行，不许编造
    assert len(r.history) == 1


def test_random_forest_single_target(data):
    X, Y, features, _ = data
    rf = build_model("random_forest", {"n_estimators": 10, "n_jobs": 1})
    _fit(rf, X, Y[:, :1], features, ["clwc"])
    assert rf.predict(X).shape == (300, 1)


# ---------------------------------------------------------------- LightGBM
# LightGBM 必须在**独立子进程**里跑：macOS 上 torch 与 LightGBM 的 OpenMP
# 运行库同进程共存会 Abort trap: 6（本文件前面的 MLP 用例已加载 torch）。
LGBM_SCRIPT = """
import numpy as np
from cloudwater_ml.data import make_synthetic_bundle
from cloudwater_ml.models import build_model

b = make_synthetic_bundle(n_samples=300, seed=11)
m = build_model("lightgbm", {"n_estimators": 20, "learning_rate": 0.1})
m.feature_names, m.target_names = b.feature_names, b.target_names
r = m.fit(b.X, b.Y, X_val=b.X[:50], Y_val=b.Y[:50],
          config={"training": {"epochs": 1, "batch_size": 64, "learning_rate": 0.1,
                               "optimizer": "adamw", "loss": "mse", "seed": 42,
                               "device": "auto",
                               "early_stopping": {"enabled": False, "patience": 5}}})
pred = m.predict(b.X)
imp = m.feature_importance()
assert pred.shape == (300, 2), pred.shape
assert np.isfinite(pred).all()
assert imp is not None and set(imp) == set(b.feature_names), imp
assert len(r.history) >= 1
print("LGBM_SUBPROC_OK")
"""


def test_lightgbm_multitarget_independent_models():
    """子进程隔离验证：多目标=逐目标独立模型，输出特征重要性。"""
    import subprocess
    import sys

    import os

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}  # 绕过 .pth 可能被 hidden 的问题
    proc = subprocess.run(
        [sys.executable, "-c", LGBM_SCRIPT],
        capture_output=True, text=True, cwd=str(root), env=env,
    )
    assert "LGBM_SUBPROC_OK" in proc.stdout, (
        f"stdout={proc.stdout[-400:]} stderr={proc.stderr[-600:]}"
    )


def test_lightgbm_missing_dependency_hint_present():
    """依赖缺失时必须给出清晰安装提示（源码级检查，避免真卸载重装）。"""
    src = (Path(__file__).resolve().parents[1]
           / "src/cloudwater_ml/models/lightgbm.py").read_text(encoding="utf-8")
    assert "[lgbm]" in src, "缺少 pip install 'cloudwater-ml[lgbm]' 提示"
    assert "lightgbm" in src.lower()


# ---------------------------------------------------------------- 模型注册与契约
def _load_all_registered_models() -> None:
    """导入全部惰性模型模块，让注册表完整（与 CLI 的 models list 同一口径）。"""
    from cloudwater_ml.models.base import _LAZY_MODULES

    for module in list(_LAZY_MODULES.values()):
        try:
            importlib.import_module(module)
        except Exception:  # 缺可选依赖（如 xgboost）不应让本用例崩溃
            pass


def test_registered_models_are_constructible():
    """v2.0：已实现的模型必须已注册且可构造（不再有「预留」占位）。"""
    from cloudwater_ml.models import available_models

    _load_all_registered_models()
    for name in ("mlp", "random_forest", "lightgbm", "cnn", "unet", "convlstm", "multitask"):
        assert name in available_models(), f"{name} 未注册"
        assert name in MODEL_REGISTRY, f"{name} 不在注册表中（惰性模块可能未导入）"


def test_unknown_model_raises_clear_error():
    """未注册模型名必须给出清晰错误（不静默返回空壳）。"""
    with pytest.raises(KeyError) as exc:
        build_model("definitely_not_a_model")
    assert "未注册" in str(exc.value) or "未知模型" in str(exc.value)


def test_no_silent_stubs_in_registered_models():
    """注册了的模型不得在构造时抛 NotImplementedError（即不得伪装已实现）。"""
    _load_all_registered_models()
    for name in ("cnn", "unet", "convlstm", "multitask"):
        if name not in MODEL_REGISTRY:
            pytest.skip(f"{name} 尚未实现（第二阶段任务）")
        cls = MODEL_REGISTRY[name]
        try:
            cls()
        except NotImplementedError as exc:
            pytest.fail(f"{name} 仍是占位实现: {exc}")
        except Exception:
            pass  # 构造函数要求必填参数是允许的（不是占位符）



# ---------------------------------------------------------------- 持久化
def test_save_load_roundtrip(tmp_path, data):
    X, Y, features, targets = data
    mlp = build_model("mlp", {"hidden_layers": [24, 12], "dropout": 0.0})
    _fit(mlp, X, Y, features, targets,
         cfg={"training": {**TRAIN_CFG["training"], "epochs": 2}})
    path = tmp_path / "model"
    mlp.save(path)
    assert (path / "model_meta.json").exists()

    restored = load_model(path)
    assert restored.name == "mlp"
    assert restored.feature_names == features
    assert restored.target_names == targets
    np.testing.assert_allclose(restored.predict(X), mlp.predict(X), rtol=1e-5, atol=1e-8)


def test_save_load_random_forest(tmp_path, data):
    X, Y, features, targets = data
    rf = build_model("random_forest", {"n_estimators": 8})
    _fit(rf, X, Y, features, targets)
    rf.save(tmp_path / "rf")
    restored = load_model(tmp_path / "rf")
    np.testing.assert_allclose(restored.predict(X[:20]), rf.predict(X[:20]))


def test_load_rejects_unknown_meta(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "model_meta.json").write_text(json.dumps({"name": "nope", "params": {}}))
    with pytest.raises(KeyError, match="未注册"):
        load_model(d)
