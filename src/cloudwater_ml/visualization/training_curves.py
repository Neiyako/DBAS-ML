"""训练曲线图：train/val loss 随 epoch 变化。

约定：Agg 后端（无显示）、savefig(dpi=150)、不调用 plt.show()、
返回保存路径 Path；任何异常降为 None，绝不让训练主流程崩。
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger("cloudwater_ml.visualization")

__all__ = ["plot_loss_curve"]


def _as_frame(history: list[dict] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        df = history.copy()
    else:
        df = pd.DataFrame(list(history))
    # 列名容错：epoch / train_loss / val_loss（缺失 val 即只画 train）
    df.columns = [str(c).strip() for c in df.columns]
    return df


def plot_loss_curve(
    history: list[dict] | pd.DataFrame,
    out_path,
) -> Path | None:
    """绘制训练/验证损失曲线。

    Parameters
    ----------
    history : list[dict] | DataFrame
        每行含 ``epoch`` / ``train_loss`` / ``val_loss``（val 可缺省）。
    out_path : str | Path
        PNG 输出路径（父目录会自动创建）。

    Returns
    -------
    Path | None
        保存路径；输入不可用或绘图失败时返回 None（不抛异常）。
    """
    try:
        df = _as_frame(history)
        if df.empty or "train_loss" not in df.columns:
            logger.warning("history 为空或缺 train_loss 列，跳过 loss 曲线")
            return None
        x = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, pd.to_numeric(df["train_loss"], errors="coerce"),
                marker="o", ms=3, lw=1.6, label="train_loss")
        if "val_loss" in df.columns:
            val = pd.to_numeric(df["val_loss"], errors="coerce")
            if val.notna().any():
                ax.plot(x, val, marker="s", ms=3, lw=1.6, label="val_loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training / Validation Loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    except Exception as exc:  # noqa: BLE001 —— 容错：绘图失败不崩 runner
        logger.warning("plot_loss_curve 失败: %s", exc)
        plt.close("all")
        return None
