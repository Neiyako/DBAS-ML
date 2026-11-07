"""训练回调：EarlyStopping 与 HistoryLogger。"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

__all__ = ["EarlyStopping", "HistoryLogger"]


class EarlyStopping:
    """监控标量（通常是 val_loss），超过 patience 未改善即请求停止。

    - ``min_delta``：改善的最小变化量（低于该幅度不算改善）；
    - ``step(val, epoch=None) -> bool``：更新 best/wait 并返回 ``should_stop``；
    - ``best`` / ``best_epoch``：历史最优点记录。
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.wait: int = 0
        self.should_stop: bool = False

    def step(self, val: float, epoch: int | None = None) -> bool:
        val = float(val)
        improved = self.best is None or val < (self.best - self.min_delta)
        if improved:
            self.best = val
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self) -> None:
        self.best = None
        self.best_epoch = None
        self.wait = 0
        self.should_stop = False


class HistoryLogger:
    """逐 epoch 历史：``rows`` 列表 + ``to_csv(path)`` + ``as_dicts()``。"""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def log(self, **row: Any) -> None:
        self.rows.append(dict(row))

    def add(self, row: dict[str, Any]) -> None:
        """按 dict 追加一行（与 log 等价，兼容 ``add({...})`` 调用习惯）。"""
        self.rows.append(dict(row))

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.rows]

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("", encoding="utf-8")
            return path
        # 列并集按首次出现顺序，保证不同 epoch 键集合不一致时也能落盘
        fields: list[str] = []
        for row in self.rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        return path
