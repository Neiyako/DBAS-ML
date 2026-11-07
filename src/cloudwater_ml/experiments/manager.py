"""实验生命周期管理（v2.0）：唯一 ID、状态机、失败记录、断点续训入口。

状态机
------

::

    CREATED ──► PREPARING ──► RUNNING ──► VALIDATING ──► COMPLETED
        │           │            │             │
        └───────────┴────────────┴─────────────┴──► FAILED / INTERRUPTED

规则：

* **非法迁移直接报错**（:class:`ExperimentStateError`），避免出现「已完成」的实验
  又被改回「运行中」这类说不清的状态；
* **每次尝试（run）有独立 run_id**，同一实验目录下的重跑不会覆盖上一次的记录；
* ``state.json`` 是实验的**权威状态文件**，与 ``metrics.json`` 并存；
  失败实验同样留下 ``state.json``（含异常类型与消息），便于排查。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..utils.exceptions import ExperimentStateError
from ..utils.logging import get_logger

__all__ = [
    "ExperimentState",
    "ExperimentRecord",
    "new_experiment_id",
    "create_record",
    "save_state",
    "load_state",
    "transition",
    "list_experiments",
    "find_experiment",
]

_logger = get_logger("experiments.manager")

STATE_FILE = "state.json"


class ExperimentState(str, Enum):
    """实验状态。"""

    CREATED = "CREATED"
    PREPARING = "PREPARING"
    RUNNING = "RUNNING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


#: 允许的状态迁移
_ALLOWED: dict[ExperimentState, set[ExperimentState]] = {
    ExperimentState.CREATED: {ExperimentState.PREPARING, ExperimentState.FAILED},
    ExperimentState.PREPARING: {ExperimentState.RUNNING, ExperimentState.FAILED},
    ExperimentState.RUNNING: {
        ExperimentState.VALIDATING,
        ExperimentState.FAILED,
        ExperimentState.INTERRUPTED,
    },
    ExperimentState.VALIDATING: {ExperimentState.COMPLETED, ExperimentState.FAILED},
    ExperimentState.COMPLETED: set(),
    ExperimentState.FAILED: {ExperimentState.PREPARING},   # 允许失败后重新发起（记为新 run）
    ExperimentState.INTERRUPTED: {ExperimentState.PREPARING},
}

#: 终态（不可再迁移）
TERMINAL_STATES = {ExperimentState.COMPLETED, ExperimentState.FAILED, ExperimentState.INTERRUPTED}


@dataclass
class ExperimentRecord:
    """一次实验（及其运行尝试）的完整状态记录。"""

    experiment_id: str = ""
    name: str = ""
    run_id: str = ""
    state: str = ExperimentState.CREATED.value
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    config_path: str | None = None
    seed: int | None = None
    model: str | None = None
    parent_run_id: str | None = None
    device: str | None = None
    precision: str | None = None
    peak_vram_bytes: int | None = None
    peak_host_rss_bytes: int | None = None
    error: dict[str, Any] | None = None
    metrics_summary: dict[str, Any] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def state_enum(self) -> ExperimentState:
        try:
            return ExperimentState(self.state)
        except ValueError:  # pragma: no cover - 手工改坏的文件
            raise ExperimentStateError(
                f"未知的实验状态: {self.state!r}",
                hint=f"合法状态: {[s.value for s in ExperimentState]}",
            ) from None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


# ----------------------------------------------------------------------
def new_experiment_id(
    *,
    name: str | None = None,
    config_path: str | Path | None = None,
    seed: int | None = None,
    timestamp: datetime | None = None,
) -> str:
    """生成全局唯一实验 ID。

    形式：``exp_YYYYmmdd_HHMMSS_<6位哈希>``。哈希取自 (名字, 配置路径, seed, 时间戳)，
    保证同一秒内创建的多个实验也不会撞名。
    """
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    payload = "|".join([str(name or ""), str(config_path or ""), str(seed or ""), ts, uuid.uuid4().hex])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:6]
    return f"exp_{ts}_{digest}"


def create_record(
    exp_dir: str | Path,
    *,
    experiment_id: str | None = None,
    config_path: str | Path | None = None,
    seed: int | None = None,
    model: str | None = None,
    provenance: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
) -> ExperimentRecord:
    """为实验目录创建初始记录（CREATED）。"""
    exp_dir = Path(exp_dir)
    now = datetime.now().isoformat(timespec="seconds")
    record = ExperimentRecord(
        experiment_id=experiment_id or new_experiment_id(
            name=exp_dir.name, config_path=config_path, seed=seed
        ),
        name=exp_dir.name,
        run_id=uuid.uuid4().hex[:12],
        state=ExperimentState.CREATED.value,
        created_at=now,
        config_path=str(config_path) if config_path else None,
        seed=seed,
        model=model,
        parent_run_id=parent_run_id,
        provenance=provenance or {},
    )
    save_state(exp_dir, record)
    return record


def save_state(exp_dir: str | Path, record: ExperimentRecord) -> Path:
    """写入 ``state.json``（原子替换，避免读到半个文件）。"""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / STATE_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def load_state(exp_dir: str | Path) -> ExperimentRecord | None:
    """读取 ``state.json``；不存在返回 None（兼容 v1.0 实验目录）。"""
    path = Path(exp_dir) / STATE_FILE
    if not path.exists():
        return None
    try:
        return ExperimentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError) as exc:
        _logger.warning("读取实验状态失败 %s: %s", path, exc)
        return None


def transition(
    record: ExperimentRecord,
    new_state: ExperimentState | str,
    *,
    exp_dir: str | Path | None = None,
    error: dict[str, Any] | None = None,
    **updates: Any,
) -> ExperimentRecord:
    """状态迁移（校验合法性）并落盘。

    Raises
    ------
    ExperimentStateError
        非法迁移（例如 COMPLETED → RUNNING）。
    """
    target = ExperimentState(new_state) if not isinstance(new_state, ExperimentState) else new_state
    current = record.state_enum
    if target != current and target not in _ALLOWED.get(current, set()):
        raise ExperimentStateError(
            f"非法状态迁移: {current.value} → {target.value}",
            hint=f"{current.value} 允许迁移到: "
            f"{sorted(s.value for s in _ALLOWED.get(current, set())) or '（终态，不可再迁移）'}",
            context={"experiment_id": record.experiment_id, "from": current.value, "to": target.value},
        )
    for key, value in updates.items():
        if hasattr(record, key):
            setattr(record, key, value)
    record.state = target.value
    if target is ExperimentState.RUNNING and not record.started_at:
        record.started_at = datetime.now().isoformat(timespec="seconds")
    if target in TERMINAL_STATES:
        record.finished_at = datetime.now().isoformat(timespec="seconds")
    if error is not None:
        record.error = error
    if exp_dir is not None:
        save_state(exp_dir, record)
    return record


def list_experiments(outputs_root: str | Path = "outputs") -> list[ExperimentRecord]:
    """列出 outputs/ 下的全部实验（按目录名排序）。"""
    root = Path(outputs_root)
    if not root.exists():
        return []
    records: list[ExperimentRecord] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        rec = load_state(d)
        if rec is not None:
            records.append(rec)
    return records


def find_experiment(
    ident: str | Path,
    outputs_root: str | Path = "outputs",
) -> tuple[Path, ExperimentRecord | None]:
    """按 **实验 ID / 目录名 / 路径** 查找实验。

    Returns
    -------
    (目录, 记录)
        记录可能为 None（v1.0 旧实验目录没有 state.json）。
    """
    ident_str = str(ident)
    candidates: list[Path] = []
    p = Path(ident_str)
    if p.is_dir():
        candidates.append(p)
    else:
        candidates.append(Path(outputs_root) / ident_str)
    for rec in list_experiments(outputs_root):
        if rec.experiment_id == ident_str and Path(outputs_root) / rec.name not in candidates:
            candidates.append(Path(outputs_root) / rec.name)
    for cand in candidates:
        if cand.is_dir():
            return cand, load_state(cand)
    raise FileNotFoundError(
        f"找不到实验: {ident_str}",
        hint=f"在 {outputs_root}/ 下用 `cloudwater report` 查看全部实验",
    )
