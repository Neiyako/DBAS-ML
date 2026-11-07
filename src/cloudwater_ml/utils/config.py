"""YAML 配置加载、extends 继承解析与深度合并。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_yaml", "deep_merge", "load_config", "save_config", "get_nested"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射(dict): {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并：override 覆盖 base；两侧同名 dict 递归合并。"""
    out: dict[str, Any] = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """加载 YAML 实验配置，解析 ``extends`` 继承。

    ``extends`` 是相对当前文件的路径列表，按顺序加载并深合并，后写覆盖先写；
    最后合并本文件自身的键。合并结果附带 ``_config_path`` 供快照与追溯。
    """
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"配置继承形成环: {path}")
    _seen = _seen | {path}

    raw = load_yaml(path)
    extends = raw.pop("extends", None) or []
    if isinstance(extends, str):
        extends = [extends]

    merged: dict[str, Any] = {}
    for rel in extends:
        merged = deep_merge(merged, load_config(path.parent / rel, _seen=_seen))
    merged = deep_merge(merged, raw)
    merged["_config_path"] = str(path)
    return merged


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)


def get_nested(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """按 ``a.b.c`` 取嵌套键，缺失返回 default。"""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
