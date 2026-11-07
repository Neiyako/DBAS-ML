"""统一日志：控制台 + 可选文件，支持结构化（JSON）输出。

v2.0 变化
---------

1. **结构化字段**：任何日志调用都可带 ``extra={"字段": 值}``；JSON 模式下
   这些字段进入结构化记录，便于实验日志被机器解析（而不是靠正则抠文本）。
2. **JSON 模式**：`setup_logging(json_format=True)` 时输出行式 JSON，
   字段固定为 ``ts / level / logger / message / fields``。
3. **异常结构化**：:func:`log_exception` 把异常按
   :func:`cloudwater_ml.utils.exceptions.exception_to_dict` 的口径记录，
   含 ``hint`` 与 ``context``，便于事后定位。
4. 保持 v1.0 行为兼容：默认仍是人类可读格式，``get_logger`` 语义不变。
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["setup_logging", "get_logger", "log_event", "log_exception", "JsonFormatter"]

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False

#: 标准 logging 的 LogRecord 自带属性（不出现在 fields 里）
_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    """行式 JSON 格式化器：一条日志 = 一个 JSON 对象。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC)
            .astimezone()
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        }
        if fields:
            payload["fields"] = fields
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(
    level: str = "INFO",
    logfile: str | Path | None = None,
    *,
    json_format: bool = False,
) -> None:
    """配置根日志。重复调用会先清理旧 handler，避免重复输出。

    Parameters
    ----------
    level : str
        日志级别（DEBUG / INFO / WARNING / ERROR）。
    logfile : str | Path | None
        额外的日志文件（实验 ``logs/`` 目录）。
    json_format : bool
        True → 行式 JSON（便于机器解析）；False → 人类可读格式。
    """
    global _configured
    root = logging.getLogger("cloudwater_ml")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    if json_format:
        fmt = JsonFormatter()
    else:
        fmt = logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False
    _configured = True


def get_logger(name: str = "cloudwater_ml") -> logging.Logger:
    """获取框架内 logger（未初始化时自动初始化）。"""
    if not _configured:
        setup_logging()
    if name == "cloudwater_ml" or name.startswith("cloudwater_ml."):
        return logging.getLogger(name)
    return logging.getLogger(f"cloudwater_ml.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """记录一条结构化事件。

    Examples
    --------
    >>> log_event(log, "epoch_end", epoch=3, loss=0.12, device="cuda")
    """
    logger.log(level, event, extra={"event": event, **fields})


def log_exception(logger: logging.Logger, exc: BaseException, message: str | None = None) -> dict[str, Any]:
    """结构化记录一个异常（返回同一份字典，便于同时写进实验记录）。"""
    from .exceptions import exception_to_dict

    payload = exception_to_dict(exc)
    logger.error(
        message or f"{payload['error_type']}: {payload['message']}",
        extra={"event": "exception", **payload},
    )
    return payload
