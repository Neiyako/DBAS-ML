"""实验溯源信息采集（v2.0）：环境、代码、数据、硬件。

设计原则：**只记录实测到的事实**，拿不到就写 ``None`` 并注明原因，
绝不用默认值冒充实测值（例如「CUDA 版本」在没有 CUDA 时必须为空，而不是 "12.4"）。
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "git_info",
    "environment_info",
    "data_fingerprint",
    "ProvenanceBundle",
    "collect_provenance",
]


# ----------------------------------------------------------------------
# 代码版本
# ----------------------------------------------------------------------
def git_info(repo_root: str | Path | None = None) -> dict[str, Any]:
    """采集 git commit / 分支 / 是否脏。

    非 git 仓库或 git 不可用时返回 ``{"available": False, "reason": ...}``。
    """
    root = Path(repo_root or Path(__file__).resolve().parents[3])
    info: dict[str, Any] = {"available": False}
    try:
        def _run(*args: str) -> str:
            out = subprocess.run(  # noqa: S603
                ["git", *args], cwd=str(root), capture_output=True, text=True,
                timeout=15, check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else ""

        commit = _run("rev-parse", "HEAD")
        if not commit:
            info["reason"] = "不是 git 仓库或 git 不可用"
            return info
        info["available"] = True
        info["commit"] = commit
        info["short_commit"] = commit[:8]
        info["branch"] = _run("rev-parse", "--abbrev-ref", "HEAD")
        status = _run("status", "--porcelain")
        info["dirty"] = bool(status)
        info["dirty_files"] = len(status.splitlines()) if status else 0
        info["commit_time"] = _run("show", "-s", "--format=%cI", "HEAD")
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        info["reason"] = f"git 调用失败: {exc}"
    return info


# ----------------------------------------------------------------------
# 运行环境
# ----------------------------------------------------------------------
def environment_info() -> dict[str, Any]:
    """采集 Python / 平台 / 关键依赖版本 / 硬件信息。"""
    from importlib.metadata import PackageNotFoundError, version

    from ..hardware import detect_hardware

    packages = [
        "cloudwater-ml", "numpy", "pandas", "scikit-learn", "torch", "xarray",
        "netCDF4", "matplotlib", "lightgbm", "xgboost", "pyarrow", "dask", "scipy",
    ]
    deps: dict[str, Any] = {}
    for name in packages:
        try:
            deps[name] = version(name)
        except PackageNotFoundError:
            deps[name] = None
    hw = detect_hardware("auto")
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": deps,
        "device": hw.to_dict(),
        "device_summary": hw.summary(),
    }


# ----------------------------------------------------------------------
# 数据指纹
# ----------------------------------------------------------------------
def data_fingerprint(
    path: str | Path | None,
    *,
    max_hash_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """为数据文件生成**只读**指纹：路径 / 大小 / mtime / 内容哈希。

    大文件只对**首尾各 4 MiB** 取哈希（记录 ``hashed_bytes`` 与
    ``partial=True``），避免为了指纹把 TB 级文件读一遍。
    """
    if not path:
        return {"available": False, "reason": "未记录数据路径"}
    p = Path(path)
    if not p.exists():
        return {"available": False, "reason": f"文件不存在: {p}"}
    try:
        stat = p.stat()
    except OSError as exc:
        return {"available": False, "reason": f"无法读取文件属性: {exc}"}

    info: dict[str, Any] = {
        "available": True,
        "path": str(p.resolve()),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
    }
    if p.is_file():
        digest = hashlib.sha256()
        hashed = 0
        try:
            with p.open("rb") as f:
                if stat.st_size <= max_hash_bytes:
                    while chunk := f.read(1024 * 1024):
                        digest.update(chunk)
                        hashed += len(chunk)
                    info["partial"] = False
                else:
                    head = f.read(max_hash_bytes // 2)
                    digest.update(head)
                    hashed += len(head)
                    f.seek(-(max_hash_bytes // 2), os.SEEK_END)
                    tail = f.read(max_hash_bytes // 2)
                    digest.update(tail)
                    hashed += len(tail)
                    info["partial"] = True
            info["sha256"] = digest.hexdigest()
            info["hashed_bytes"] = hashed
        except OSError as exc:
            info["hash_error"] = str(exc)
    else:
        info["kind"] = "directory"
    return info


# ----------------------------------------------------------------------
@dataclass
class ProvenanceBundle:
    """一次采集到的溯源信息集合。"""

    git: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    config_path: str | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "git": self.git,
            "environment": self.environment,
            "data_fingerprint": self.data,
            "config_path": self.config_path,
            "seed": self.seed,
        }


def collect_provenance(
    *,
    config_path: str | Path | None = None,
    data_path: str | Path | None = None,
    seed: int | None = None,
    repo_root: str | Path | None = None,
) -> ProvenanceBundle:
    """一次性采集 git / 环境 / 数据指纹。"""
    return ProvenanceBundle(
        git=git_info(repo_root),
        environment=environment_info(),
        data=data_fingerprint(data_path),
        config_path=str(config_path) if config_path else None,
        seed=seed,
    )
