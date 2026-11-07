"""v2.0 场模型 / 时空模型 / 多任务模型 / XGBoost 基线的契约测试。

每个模型都覆盖四项：**前向形状**、**梯度可回传**、**保存-加载往返一致**、
**最小训练确实降低损失**。所有用例都是真实计算（无 mock、无桩函数）。

- GPU 相关用例统一用 ``@pytest.mark.gpu`` 标记，本机无 CUDA 时自动 skip；
- xgboost 是可选依赖：未安装时相关用例 skip 并给出明确原因（不联网安装）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cloudwater_ml.models.base import TrainingResult, load_model
from cloudwater_ml.models.cnn import CNN, CNNNet
from cloudwater_ml.models.convlstm import ConvLSTM, ConvLSTMCell, ConvLSTMNet
from cloudwater_ml.models.multitask import MultiTask, MultiTaskNet
from cloudwater_ml.models.unet import UNet, UNetNet


# ----------------------------------------------------------------------
# 通用夹具与工具
# ----------------------------------------------------------------------
def _cfg(epochs: int = 8, lr: float = 2e-2, batch_size: int = 8, device: str = "cpu") -> dict:
    """最小训练配置（关早停，便于断言 history 长度与损失下降）。"""
    return {
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "loss": "mse",
            "seed": 42,
            "device": device,
            "early_stopping": {"enabled": False, "patience": 99, "min_delta": 0.0},
        }
    }


def _assert_loss_decreased(result: TrainingResult, ratio: float = 0.95) -> None:
    """断言训练确实把损失降下来了（不是装饰性的 history）。"""
    losses = [float(row["train_loss"]) for row in result.history]
    assert len(losses) >= 2, f"history 至少要有 2 个 epoch，实际 {len(losses)}"
    assert all(np.isfinite(losses)), f"损失出现非有限值: {losses}"
    assert losses[-1] < losses[0] * ratio, (
        f"损失未明显下降: 首 {losses[0]:.6f} → 末 {losses[-1]:.6f}（阈值 {ratio}）"
    )


def _has_grad(module: torch.nn.Module) -> bool:
    """是否至少有一个参数拿到了非 None 且非零的梯度。"""
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    return bool(grads) and any(bool(torch.any(g != 0)) for g in grads)


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20261107)


# ----------------------------------------------------------------------
# CNN
# ----------------------------------------------------------------------
def test_cnn_net_forward_shape():
    """[B, C, H, W] → [B, n_targets]。"""
    net = CNNNet(in_channels=3, n_targets=2, channels=[8, 16])
    out = net(torch.randn(4, 3, 12, 10))
    assert out.shape == (4, 2)


def test_cnn_net_output_is_linear_can_be_negative():
    """输出层非 ReLU：可以给出负值预测。"""
    net = CNNNet(in_channels=1, n_targets=1, channels=[4])
    with torch.no_grad():
        net.head.bias.fill_(-1.0)
    out = net(torch.randn(2, 1, 6, 6))
    assert bool((out < 0).all())


def test_cnn_gradients_flow():
    """loss.backward() 后必须有非 None 梯度。"""
    net = CNNNet(in_channels=2, n_targets=1, channels=[4, 8])
    out = net(torch.randn(3, 2, 8, 8))
    out.pow(2).mean().backward()
    assert _has_grad(net)


def test_cnn_predict_is_2d_and_finite(rng):
    """predict 恒返回 [n_samples, n_targets] 二维数组。"""
    X = rng.normal(size=(20, 2, 8, 8)).astype(np.float32)
    Y = rng.normal(size=(20, 3)).astype(np.float32)
    model = CNN(channels=[8, 16])
    model.fit(X, Y, config=_cfg(epochs=2))
    pred = model.predict(X)
    assert pred.ndim == 2 and pred.shape == (20, 3)
    assert np.isfinite(pred).all()


def test_cnn_accepts_2d_input_as_single_channel_grid(rng):
    """平台 runner 传进来的是二维特征矩阵 → 自动重排为单通道网格。"""
    X = rng.normal(size=(16, 12)).astype(np.float32)
    Y = rng.normal(size=(16, 1)).astype(np.float32)
    model = CNN(channels=[8])
    model.fit(X, Y, config=_cfg(epochs=2))
    assert model.predict(X).shape == (16, 1)
    assert model.n_in == 1


def test_cnn_training_decreases_loss(rng):
    """最小训练：损失确实下降。"""
    X = rng.normal(size=(48, 2, 8, 8)).astype(np.float32)
    Y = (X[:, 0].mean(axis=(1, 2)) * 2.0).reshape(-1, 1).astype(np.float32)
    model = CNN(channels=[8, 16], dropout=0.0)
    result = model.fit(X, Y, config=_cfg(epochs=12, lr=5e-2))
    _assert_loss_decreased(result)
    assert len(result.history) == 12


def test_cnn_save_load_roundtrip(tmp_path, rng):
    """save → load_model → predict 结果一致。"""
    X = rng.normal(size=(12, 2, 8, 8)).astype(np.float32)
    Y = rng.normal(size=(12, 1)).astype(np.float32)
    model = CNN(channels=[8])
    model.fit(X, Y, config=_cfg(epochs=2))
    model.save(tmp_path / "cnn")
    restored = load_model(tmp_path / "cnn")
    assert restored.name == "cnn"
    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=1e-5, atol=1e-6)


def test_cnn_feature_importance_is_none(rng):
    """不适用时必须返回 None（不得编造）。"""
    X = rng.normal(size=(8, 1, 8, 8)).astype(np.float32)
    Y = rng.normal(size=(8, 1)).astype(np.float32)
    model = CNN(channels=[4])
    model.fit(X, Y, config=_cfg(epochs=1))
    assert model.feature_importance() is None


# ----------------------------------------------------------------------
# UNet
# ----------------------------------------------------------------------
def test_unet_net_forward_keeps_spatial_size():
    """[B, C, H, W] → [B, out_channels, H, W]，空间尺寸不变。"""
    net = UNetNet(in_channels=2, out_channels=1, base_channels=8, depth=3)
    out = net(torch.randn(2, 2, 16, 16))
    assert out.shape == (2, 1, 16, 16)


def test_unet_gradients_flow():
    net = UNetNet(in_channels=1, out_channels=2, base_channels=8, depth=2)
    out = net(torch.randn(2, 1, 8, 8))
    out.pow(2).mean().backward()
    assert _has_grad(net)


def test_unet_rejects_non_divisible_spatial_size(rng):
    """H/W 不能被 2**depth 整除时必须明确报错（不静默裁剪）。"""
    X = rng.normal(size=(4, 1, 10, 10)).astype(np.float32)
    Y = rng.normal(size=(4, 1, 10, 10)).astype(np.float32)
    model = UNet(base_channels=8, depth=3)
    with pytest.raises(ValueError, match="整除"):
        model.fit(X, Y, config=_cfg(epochs=1))


def test_unet_field_training_and_roundtrip(tmp_path, rng):
    """整场监督：训练损失下降 + 保存加载一致。"""
    X = rng.normal(size=(24, 2, 16, 16)).astype(np.float32)
    Y = X.mean(axis=1, keepdims=True)
    model = UNet(base_channels=8, depth=3)
    result = model.fit(X, Y, config=_cfg(epochs=8, lr=5e-2, batch_size=4))
    _assert_loss_decreased(result)
    field = model.predict_field(X)
    assert field.shape == (24, 1, 16, 16)
    assert model.predict(X).shape == (24, 1)  # 契约：标量读出

    model.save(tmp_path / "unet")
    restored = load_model(tmp_path / "unet")
    np.testing.assert_allclose(restored.predict_field(X), field, rtol=1e-5, atol=1e-6)


def test_unet_scalar_target_mode(rng):
    """标量目标：对预测场做空间平均后监督，predict 仍返回二维。"""
    X = rng.normal(size=(20, 1, 8, 8)).astype(np.float32)
    Y = X.mean(axis=(1, 2, 3)).reshape(-1, 1).astype(np.float32)
    model = UNet(base_channels=8, depth=2)
    result = model.fit(X, Y, config=_cfg(epochs=6, lr=5e-2))
    _assert_loss_decreased(result)
    pred = model.predict(X)
    assert pred.shape == (20, 1)
    assert np.isfinite(pred).all()


def test_unet_bilinear_false_variant(rng):
    """转置卷积上采样分支同样可用。"""
    X = rng.normal(size=(8, 1, 8, 8)).astype(np.float32)
    Y = X.mean(axis=(1, 2, 3)).reshape(-1, 1).astype(np.float32)
    model = UNet(base_channels=8, depth=2, bilinear=False)
    model.fit(X, Y, config=_cfg(epochs=2))
    assert model.predict(X).shape == (8, 1)


# ----------------------------------------------------------------------
# ConvLSTM
# ----------------------------------------------------------------------
def test_convlstm_cell_zero_input_shape_and_spatial_uniformity():
    """零输入 + 零状态：形状正确，且输出在空间上均匀（无输入信息 → 只由偏置决定）。

    注意：**带偏置的 ConvLSTM 单元在零输入下输出不是 0**
    （i = sigmoid(b_i) ≠ 0, g = tanh(b_g) ≠ 0），这是数学事实而非缺陷；
    真正的「零输入零状态 → 零输出」只在偏置被置零时成立（见后面的断言）。
    """
    cell = ConvLSTMCell(in_channels=1, hidden_channels=4, kernel_size=3)
    x = torch.zeros(2, 1, 6, 6)
    h = torch.zeros(2, 4, 6, 6)
    c = torch.zeros(2, 4, 6, 6)
    h_new, c_new = cell(x, h, c)
    assert h_new.shape == (2, 4, 6, 6) and c_new.shape == (2, 4, 6, 6)
    # 空间均匀：零输入下每个空间位置的输出相同
    assert torch.allclose(h_new, h_new[:, :, :1, :1].expand_as(h_new), atol=1e-6)
    # 隐状态有界（h = o ⊙ tanh(c)）
    assert float(h_new.detach().abs().max()) <= 1.0

    # 偏置归零后：零输入 + 零状态 → 零输出
    with torch.no_grad():
        for prm in cell.parameters():
            if prm.dim() == 1:  # bias 向量
                prm.zero_()
    h0, c0 = cell(x, h, c)
    assert torch.allclose(h0, torch.zeros_like(h0), atol=1e-6)
    assert torch.allclose(c0, torch.zeros_like(c0), atol=1e-6)


def test_convlstm_cell_output_is_bounded():
    """h = o ⊙ tanh(c) ⇒ 隐状态恒在 (-1, 1)。"""
    cell = ConvLSTMCell(in_channels=2, hidden_channels=4, kernel_size=3)
    h = torch.zeros(1, 4, 5, 5)
    c = torch.zeros(1, 4, 5, 5)
    for _ in range(3):
        h, c = cell(torch.randn(1, 2, 5, 5), h, c)
    assert float(h.abs().max()) <= 1.0


def test_convlstm_net_forward_shape():
    """[B, T, C, H, W] → [B, pred_len, out_channels, H, W]。"""
    net = ConvLSTMNet(in_channels=1, hidden_channels=4, num_layers=1,
                       kernel_size=3, out_channels=2, pred_len=3)
    out = net(torch.randn(2, 4, 1, 8, 8))
    assert out.shape == (2, 3, 2, 8, 8)


def test_convlstm_gradients_flow():
    net = ConvLSTMNet(in_channels=1, hidden_channels=4, num_layers=1,
                       kernel_size=3, out_channels=1, pred_len=2)
    out = net(torch.randn(2, 3, 1, 6, 6))
    out.pow(2).mean().backward()
    assert _has_grad(net)


def test_convlstm_training_decreases_loss(rng):
    """最小训练：损失下降。"""
    X = rng.normal(size=(32, 3, 1, 8, 8)).astype(np.float32)
    Y = X[:, -1].mean(axis=(1, 2, 3)).reshape(-1, 1).astype(np.float32)
    model = ConvLSTM(in_channels=1, hidden_channels=8, seq_len=3, pred_len=1)
    result = model.fit(X, Y, config=_cfg(epochs=10, lr=5e-2, batch_size=8))
    _assert_loss_decreased(result)
    assert model.predict(X).shape == (32, 1)


def test_convlstm_multistep_shape_and_roundtrip(tmp_path, rng):
    """多步自回归：pred_len=2 → 预报场有 2 步；保存加载一致。"""
    X = rng.normal(size=(16, 3, 1, 8, 8)).astype(np.float32)
    Y = rng.normal(size=(16, 1)).astype(np.float32)
    model = ConvLSTM(in_channels=1, hidden_channels=8, seq_len=3, pred_len=2)
    model.fit(X, Y, config=_cfg(epochs=2))
    field = model.predict_field(X)
    assert field.shape == (16, 2, 1, 8, 8)
    model.save(tmp_path / "convlstm")
    restored = load_model(tmp_path / "convlstm")
    np.testing.assert_allclose(restored.predict_field(X), field, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=1e-5, atol=1e-6)


def test_convlstm_rejects_bad_2d_reshape(rng):
    """2D 输入无法按 seq_len 重排时必须明确报错。"""
    X = rng.normal(size=(8, 10)).astype(np.float32)
    Y = rng.normal(size=(8, 1)).astype(np.float32)
    model = ConvLSTM(in_channels=1, seq_len=3)
    with pytest.raises(ValueError, match="整除"):
        model.fit(X, Y, config=_cfg(epochs=1))


# ----------------------------------------------------------------------
# MultiTask
# ----------------------------------------------------------------------
def test_multitask_net_forward_shape():
    """[B, F] → [B, n_tasks]，每任务 1 个标量。"""
    net = MultiTaskNet(n_in=5, n_tasks=3, hidden_layers=[16, 8])
    out = net(torch.randn(4, 5))
    assert out.shape == (4, 3)


def test_multitask_gradients_flow():
    net = MultiTaskNet(n_in=4, n_tasks=2, hidden_layers=[8])
    out = net(torch.randn(3, 4))
    out.pow(2).mean().backward()
    assert _has_grad(net)


def test_multitask_training_decreases_loss(rng):
    X = rng.normal(size=(48, 6)).astype(np.float32)
    Y = np.column_stack([X[:, 0] * 2 + X[:, 1], X[:, 2] - X[:, 3]]).astype(np.float32)
    model = MultiTask(hidden_layers=[32, 16], task_names=["clwc", "ciwc"])
    result = model.fit(X, Y, config=_cfg(epochs=12, lr=5e-2, batch_size=8))
    _assert_loss_decreased(result)
    pred = model.predict(X)
    assert pred.shape == (48, 2)
    assert np.isfinite(pred).all()


def test_multitask_weights_are_actually_used(rng):
    """逐任务权重必须真正进入损失（不是装饰参数）。"""
    X = rng.normal(size=(16, 4)).astype(np.float32)
    Y = rng.normal(size=(16, 2)).astype(np.float32)
    model = MultiTask(hidden_layers=[8], task_weights=[1.0, 0.0])
    model.fit(X, Y, config=_cfg(epochs=2))
    assert model.task_weights_resolved == [1.0, 0.0]

    pred = torch.tensor([[1.0, 10.0]])
    tgt = torch.tensor([[0.0, 0.0]])
    # 权重 [1, 0]：只有任务 0 贡献 → loss = 1.0；任务 1 的 100 被完全屏蔽
    assert float(model.weighted_loss(pred, tgt)) == pytest.approx(1.0, rel=1e-5)

    model2 = MultiTask(hidden_layers=[8], task_weights=[0.0, 1.0])
    model2.fit(X, Y, config=_cfg(epochs=2))
    assert float(model2.weighted_loss(pred, tgt)) == pytest.approx(100.0, rel=1e-5)


def test_multitask_zero_weight_task_gets_no_gradient(rng):
    """权重为 0 的任务：其任务头拿不到梯度（权重生效的可观测证据）。"""
    X = rng.normal(size=(16, 4)).astype(np.float32)
    Y = rng.normal(size=(16, 2)).astype(np.float32)
    model = MultiTask(hidden_layers=[8], task_weights=[1.0, 0.0])
    model.fit(X, Y, config=_cfg(epochs=2))
    out = model.net(torch.from_numpy(X))
    loss = model.weighted_loss(out, torch.from_numpy(Y))
    loss.backward()
    assert model.net.heads[0].weight.grad is not None
    g1 = model.net.heads[1].weight.grad
    assert g1 is None or bool(torch.all(g1 == 0)), "权重为 0 的任务仍收到梯度——权重没真正生效"


def test_multitask_task_weights_length_mismatch(rng):
    X = rng.normal(size=(8, 4)).astype(np.float32)
    Y = rng.normal(size=(8, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="task_weights"):
        MultiTask(hidden_layers=[8], task_weights=[1.0]).fit(X, Y, config=_cfg(epochs=1))


def test_multitask_save_load_roundtrip(tmp_path, rng):
    X = rng.normal(size=(12, 4)).astype(np.float32)
    Y = rng.normal(size=(12, 2)).astype(np.float32)
    model = MultiTask(hidden_layers=[16], task_names=["clwc", "ciwc"], task_weights=[1.0, 2.0])
    model.fit(X, Y, config=_cfg(epochs=2))
    model.save(tmp_path / "mt")
    restored = load_model(tmp_path / "mt")
    assert restored.task_weights_resolved == [1.0, 2.0]
    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# XGBoost（可选依赖）
# ----------------------------------------------------------------------
_XGBOOST_AVAILABLE = True
try:  # pragma: no cover - 取决于环境
    import xgboost  # noqa: F401
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False

_XGB_SKIP_REASON = (
    "xgboost 未安装（可选依赖，pip install 'cloudwater-ml[xgboost]'）；"
    "本用例需要真实训练 xgboost 模型，故跳过"
)


def test_xgboost_module_imports_without_dependency():
    """缺依赖时模块仍可导入（构造时才报错并给安装提示）。"""
    src = (Path(__file__).resolve().parents[1]
           / "src/cloudwater_ml/models/xgboost.py").read_text(encoding="utf-8")
    assert "[xgboost]" in src, "缺少 pip install 'cloudwater-ml[xgboost]' 安装提示"
    # 只检查真正的 import 语句（文档里提到 "不 import torch" 属于说明文字）
    import_lines = [
        ln.strip() for ln in src.splitlines()
        if ln.strip().startswith(("import ", "from "))
    ]
    assert not any(
        ln == "import torch" or ln.startswith("import torch.") or ln.startswith("from torch")
        for ln in import_lines
    ), f"树模型模块不得 import torch（OpenMP 冲突）: {import_lines}"


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason=_XGB_SKIP_REASON)
def test_xgboost_multitarget_and_importance(rng):
    """逐目标独立模型 + 特征重要性。"""
    from cloudwater_ml.models.xgboost import XGBoostModel

    X = rng.normal(size=(60, 5)).astype(np.float64)
    Y = np.column_stack([X[:, 0] * 2, X[:, 1] - X[:, 2]]).astype(np.float64)
    model = XGBoostModel(n_estimators=20, learning_rate=0.2, n_jobs=1)
    model.feature_names = [f"f{i}" for i in range(5)]
    result = model.fit(X, Y, config=_cfg(epochs=1))
    assert isinstance(result, TrainingResult)
    pred = model.predict(X)
    assert pred.shape == (60, 2)
    imp = model.feature_importance()
    assert imp is not None and set(imp) == set(model.feature_names)


@pytest.mark.skipif(not _XGBOOST_AVAILABLE, reason=_XGB_SKIP_REASON)
def test_xgboost_save_load_roundtrip(tmp_path, rng):
    from cloudwater_ml.models.xgboost import XGBoostModel

    X = rng.normal(size=(40, 4)).astype(np.float64)
    Y = X[:, :2].astype(np.float64)
    model = XGBoostModel(n_estimators=10, n_jobs=1)
    model.fit(X, Y, config=_cfg(epochs=1))
    model.save(tmp_path / "xgb")
    restored = load_model(tmp_path / "xgb")
    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=1e-5, atol=1e-6)


def test_xgboost_missing_dependency_raises_clear_error():
    """未安装 xgboost 时构造必须给出清晰安装提示（而不是静默空壳）。"""
    from cloudwater_ml.models.xgboost import XGBoostModel

    if _XGBOOST_AVAILABLE:  # pragma: no cover - 装了依赖就不测这条
        pytest.skip("xgboost 已安装，跳过缺依赖报错用例")
    with pytest.raises(RuntimeError, match="xgboost"):
        XGBoostModel()


# ----------------------------------------------------------------------
# GPU 用例（无 CUDA 自动跳过）
# ----------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境无 CUDA 设备")
def test_cnn_cuda_training(rng):
    X = rng.normal(size=(32, 2, 16, 16)).astype(np.float32)
    Y = X[:, 0].mean(axis=(1, 2)).reshape(-1, 1).astype(np.float32)
    model = CNN(channels=[8, 16])
    result = model.fit(X, Y, config=_cfg(epochs=4, device="cuda"))
    _assert_loss_decreased(result)
    assert model.predict(X).shape == (32, 1)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境无 CUDA 设备")
def test_unet_cuda_training(rng):
    X = rng.normal(size=(16, 1, 16, 16)).astype(np.float32)
    Y = X.mean(axis=1, keepdims=True)
    model = UNet(base_channels=8, depth=3)
    result = model.fit(X, Y, config=_cfg(epochs=4, device="cuda", batch_size=4))
    _assert_loss_decreased(result)
    assert model.predict_field(X).shape == (16, 1, 16, 16)
