"""分级数据缓存（Raw → Metadata → Preprocessing → Memory → GPU Batch）。

目标场景是 TB 级 ERA5：同一份原始数据被反复「读 header → 算统计量 → 拟合归一化 →
分批喂训练」时，缓存把重复 IO 压掉，同时**保证原始科研数据一个字节都不被改动**。

分级说明
--------

===========  ==========================================================
级别          内容
===========  ==========================================================
raw          原始数据的**清单/索引**（行区间、变量清单）——只存索引，
             **绝不复制原始数据本体**，也不写回 data/raw
metadata     文件元数据（维度、变量、单位、来源契约、inspect 结果）
preprocessing 预处理结果（缩放后矩阵、统计量缓存等）
memory       进程内内存缓存（可选，按字节预算做简单 LRU）
gpu_batch    批次缓存（进程内，供训练侧复用已就绪的 batch）
===========  ==========================================================

铁律
----

- **缓存目录默认 ``<项目根>/.cache/cloudwater_ml/``**，位于 ``data/`` 之外；
- **任何写缓存的路径都要过保护**：路径中出现 ``raw`` 目录层级（或落在
  ``forbidden_roots`` 内）一律拒绝写入 —— 缓存绝不能覆盖科研原始数据；
- 每条缓存带 **key + 数据版本**（文件 size / mtime_ns / 可选 sha256），
  源文件一变 → 版本变 → key 变 → 旧条目自动失效（并可显式 invalidate）；
- 缓存**可清理、可重建、可追溯**（条目内记录来源路径、版本、创建时间、命中次数）。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "CACHE_LEVELS",
    "DISK_LEVELS",
    "MEMORY_LEVELS",
    "DataCache",
    "FileVersion",
    "compute_file_version",
    "make_key",
    "DEFAULT_CACHE_ROOT",
]

#: 缓存级别（Raw → Metadata → Preprocessing → Memory → GPU Batch）
CACHE_LEVELS = ("raw", "metadata", "preprocessing", "memory", "gpu_batch")
#: 落盘级别
DISK_LEVELS = ("raw", "metadata", "preprocessing")
#: 仅进程内级别
MEMORY_LEVELS = ("memory", "gpu_batch")

#: 默认缓存根目录（项目根下的 .cache/，位于 data/ 之外）
DEFAULT_CACHE_ROOT = Path(".cache") / "cloudwater_ml"

#: 默认禁止写入的目录名（相对路径任一层级命中即拒绝）
_FORBIDDEN_COMPONENTS = ("raw",)


def make_key(*parts: Any) -> str:
    """稳定 key：把任意可序列化部件拼成 sha1 十六进制（前 16 位足够区分）。"""
    h = hashlib.sha1()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class FileVersion:
    """文件版本（数据版本校验用）。

    默认用 ``size + mtime_ns``（便宜）；``sha256`` 可选（内容级校验）。
    """

    size: int
    mtime_ns: int
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"size": int(self.size), "mtime_ns": int(self.mtime_ns), "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> FileVersion:
        data = data or {}
        return cls(
            size=int(data.get("size", 0)),
            mtime_ns=int(data.get("mtime_ns", 0)),
            sha256=data.get("sha256"),
        )

    def token(self) -> str:
        """版本短标记（拼进 key → 源文件一变 key 就变）。"""
        base = f"{self.size}-{self.mtime_ns}"
        if self.sha256:
            base += f"-{self.sha256[:12]}"
        return hashlib.sha1(base.encode()).hexdigest()[:12]

    def matches(self, other: FileVersion | None) -> bool:
        if other is None:
            return False
        if self.sha256 and other.sha256:
            return self.sha256 == other.sha256
        return self.size == other.size and self.mtime_ns == other.mtime_ns


def compute_file_version(path: str | Path, *, with_hash: bool = False,
                         chunk_size: int = 1 << 20) -> FileVersion:
    """计算文件版本（size / mtime_ns / 可选 sha256）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在，无法计算版本: {p}")
    st = p.stat()
    sha: str | None = None
    if with_hash:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                h.update(chunk)
        sha = h.hexdigest()
    return FileVersion(size=int(st.st_size), mtime_ns=int(st.st_mtime_ns), sha256=sha)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _assert_not_raw(path: Path, forbidden_roots: Iterable[Path] = ()) -> None:
    """硬保护：拒绝把**缓存根目录**放进原始数据目录。"""
    resolved = Path(path).resolve()
    parts = [p.lower() for p in resolved.parts]
    for comp in _FORBIDDEN_COMPONENTS:
        if comp in parts:
            raise ValueError(
                f"拒绝在原始数据目录中写入缓存: {resolved}"
                "（路径含 'raw' 层级；缓存绝不能写入或覆盖 data/raw 下的科研原始数据）"
            )
    for root in forbidden_roots:
        try:
            resolved.relative_to(Path(root).resolve())
        except ValueError:
            continue
        raise ValueError(
            f"拒绝在原始数据目录中写入缓存: {resolved}（位于受保护目录 {root} 内）"
        )


class DataCache:
    """分级缓存管理器（磁盘三级 + 进程内两级）。

    Parameters
    ----------
    root : 缓存根目录；默认 ``<cwd>/.cache/cloudwater_ml``（位于 data/ 之外）
    enabled : 关闭后所有写操作变成 no-op（读也直接 miss），便于随时停用
    hash_files : 是否计算文件 sha256（内容级版本校验；TB 级文件慎开）
    memory_limit_bytes : 进程内缓存字节预算（超出按插入顺序淘汰）
    forbidden_roots : 额外禁止写入的目录（默认再加 ``<cwd>/data/raw``）
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        enabled: bool = True,
        hash_files: bool = False,
        memory_limit_bytes: int = 256 * 1024 * 1024,
        forbidden_roots: Iterable[str | Path] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.hash_files = bool(hash_files)
        self.memory_limit_bytes = int(memory_limit_bytes)
        base = Path(root) if root is not None else DEFAULT_CACHE_ROOT
        if forbidden_roots is None:
            forbidden = [Path("data") / "raw"]
        else:
            forbidden = [Path(r) for r in forbidden_roots]
        # 缓存根本身不得落在原始数据目录里（硬保护）
        _assert_not_raw(base, forbidden_roots=forbidden)
        self.root = Path(base)
        self._forbidden_roots = forbidden
        self._mem: dict[tuple[str, str], Any] = {}
        self._mem_bytes = 0
        self._hits = 0
        self._misses = 0
        self._writes = 0

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def _ensure_dir(self, path: Path) -> Path:
        """建目录前先确认**在缓存根目录之内**（缓存根本身已受 raw 保护）。"""
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError:
            raise ValueError(
                f"拒绝写入缓存根目录之外的路径: {resolved}（缓存根 {self.root}）"
            ) from None
        path.mkdir(parents=True, exist_ok=True)
        return path

    def file_version(self, path: str | Path, *, with_hash: bool | None = None) -> FileVersion:
        """文件版本（size/mtime/hash）——缓存失效的判据。"""
        use_hash = self.hash_files if with_hash is None else bool(with_hash)
        return compute_file_version(path, with_hash=use_hash)

    def versioned_key(self, path: str | Path, *parts: Any, with_hash: bool | None = None) -> str:
        """带版本的 key：文件一变 → key 变 → 旧缓存自动失效。"""
        ver = self.file_version(path, with_hash=with_hash)
        return make_key(Path(path).resolve().name, ver.token(), *parts)

    def _entry_paths(self, level: str, key: str) -> tuple[Path, Path]:
        base = self.root / level / key[:2]
        return base / f"{key}.json", base / f"{key}.npy"

    # ------------------------------------------------------------------
    # 磁盘级别：raw / metadata / preprocessing
    # ------------------------------------------------------------------
    def put(
        self,
        level: str,
        key: str,
        payload: Any,
        *,
        fmt: str = "json",
        version: FileVersion | None = None,
        source: str | Path | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """写入缓存条目，返回 key。

        ``fmt='json'``：payload 必须可 JSON 序列化；
        ``fmt='npy'``：payload 为 ndarray（另存 .npy）；
        ``fmt='npz'``：payload 为 {name: ndarray} 字典（另存 .npz）。
        """
        if not self.enabled:
            return key
        if level in MEMORY_LEVELS:
            self._mem_put(level, key, payload)
            return key
        if level not in DISK_LEVELS:
            raise ValueError(f"level={level!r} 不是落盘级别，可选: {list(DISK_LEVELS)}")
        json_path, arr_path = self._entry_paths(level, key)
        self._ensure_dir(json_path.parent)
        envelope: dict[str, Any] = {
            "key": key,
            "level": level,
            "created_at": _now(),
            "version": version.to_dict() if version else None,
            "source": str(source) if source is not None else None,
            "meta": dict(meta or {}),
        }
        if fmt == "json":
            envelope["payload"] = payload
            envelope["format"] = "json"
            json_path.write_text(json.dumps(envelope, ensure_ascii=False, default=str), encoding="utf-8")
        elif fmt in ("npy", "npz"):
            envelope["format"] = fmt
            if fmt == "npy":
                np.save(arr_path, np.asarray(payload))
                envelope["payload_file"] = arr_path.name
            else:
                # np.savez 会自己追加 .npz 后缀：必须显式用 .npz 路径，
                # 否则写入 {key}.npy.npz、读取却找 {key}.npy → 永远 miss。
                npz_path = arr_path.with_suffix(".npz")
                np.savez(npz_path, **{str(k): np.asarray(v) for k, v in dict(payload).items()})
                envelope["payload_file"] = npz_path.name
            json_path.write_text(json.dumps(envelope, ensure_ascii=False, default=str), encoding="utf-8")
        else:
            raise ValueError(f"未知 fmt={fmt!r}，可选: json/npy/npz")
        self._writes += 1
        self._index_add(source, level, key)
        return key

    def get(self, level: str, key: str, *, version: FileVersion | None = None) -> Any | None:
        """读缓存条目；版本不一致 → 视为 miss（并清掉过期条目）。"""
        if not self.enabled:
            return None
        if level in MEMORY_LEVELS:
            return self._mem_get(level, key)
        json_path, arr_path = self._entry_paths(level, key)
        if not json_path.exists():
            self._misses += 1
            return None
        try:
            envelope = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._misses += 1
            return None
        stored = FileVersion.from_dict(envelope.get("version"))
        if version is not None and not version.matches(stored):
            # 版本不一致 → 失效（源文件已变）
            self._remove_entry(level, key)
            self._misses += 1
            return None
        fmt = envelope.get("format", "json")
        if fmt == "json":
            self._hits += 1
            return envelope.get("payload")
        if fmt == "npy":
            if not arr_path.exists():
                self._misses += 1
                return None
            self._hits += 1
            return np.load(arr_path)
        if fmt == "npz":
            # 以 envelope 里记录的实文件名称为准（np.savez 会改后缀）
            name = envelope.get("payload_file")
            npz_path = json_path.parent / name if name else arr_path.with_suffix(".npz")
            if not npz_path.exists():
                self._misses += 1
                return None
            self._hits += 1
            with np.load(npz_path) as z:
                return {k: z[k] for k in z.files}
        self._misses += 1
        return None

    def has(self, level: str, key: str, *, version: FileVersion | None = None) -> bool:
        return self.get(level, key, version=version) is not None

    def _remove_entry(self, level: str, key: str) -> None:
        json_path, arr_path = self._entry_paths(level, key)
        for p in (json_path, arr_path, arr_path.with_suffix(".npz")):
            try:
                if p.exists():
                    p.unlink()
            except OSError:  # pragma: no cover - 清理失败不影响主流程
                pass

    # ------------------------------------------------------------------
    # 进程内级别：memory / gpu_batch
    # ------------------------------------------------------------------
    def _mem_put(self, level: str, key: str, payload: Any) -> None:
        mk = (level, key)
        if mk in self._mem:
            self._mem_bytes -= self._mem_size(self._mem[mk])
            del self._mem[mk]
        self._mem[mk] = payload
        self._mem_bytes += self._mem_size(payload)
        while self._mem_bytes > self.memory_limit_bytes and len(self._mem) > 1:
            oldest = next(iter(self._mem))
            old = self._mem.pop(oldest)
            self._mem_bytes -= self._mem_size(old)
        self._writes += 1

    def _mem_get(self, level: str, key: str) -> Any | None:
        if (level, key) in self._mem:
            self._hits += 1
            return self._mem[(level, key)]
        self._misses += 1
        return None

    @staticmethod
    def _mem_size(payload: Any) -> int:
        if isinstance(payload, np.ndarray):
            return int(payload.nbytes)
        if isinstance(payload, dict):
            return int(sum(DataCache._mem_size(v) for v in payload.values()))
        return 0

    # ------------------------------------------------------------------
    # 索引（来源 → 条目），用于按文件失效
    # ------------------------------------------------------------------
    @property
    def _index_path(self) -> Path:
        return self.root / "_index.json"

    def _load_index(self) -> dict[str, list[list[str]]]:
        if not self._index_path.exists():
            return {}
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _index_add(self, source: str | Path | None, level: str, key: str) -> None:
        if source is None:
            return
        idx = self._load_index()
        k = str(source)
        entries = idx.setdefault(k, [])
        if [level, key] not in entries:
            entries.append([level, key])
        self._ensure_dir(self._index_path.parent)
        self._index_path.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # 业务便捷 API
    # ------------------------------------------------------------------
    def cached_metadata(self, path: str | Path, kind: str) -> dict[str, Any] | None:
        """取文件元数据缓存（按文件版本失效）。"""
        if not self.enabled:
            return None
        try:
            ver = self.file_version(path)
        except FileNotFoundError:
            return None
        key = self.versioned_key(path, "metadata", kind)
        payload = self.get("metadata", key, version=ver)
        return payload if isinstance(payload, dict) else None

    def store_metadata(self, path: str | Path, kind: str, payload: dict[str, Any]) -> str:
        """存文件元数据缓存。"""
        ver = self.file_version(path)
        key = self.versioned_key(path, "metadata", kind)
        return self.put("metadata", key, payload, version=ver, source=str(path),
                        meta={"kind": kind, "path": str(path)})

    def cached_index(self, path: str | Path, name: str) -> Any | None:
        """取索引缓存（raw 级别：原始数据的行区间/变量清单，不含数据本体）。"""
        if not self.enabled:
            return None
        try:
            ver = self.file_version(path)
        except FileNotFoundError:
            return None
        key = self.versioned_key(path, "index", name)
        return self.get("raw", key, version=ver)

    def store_index(self, path: str | Path, name: str, payload: Any) -> str:
        """存索引缓存（raw 级别）。"""
        ver = self.file_version(path)
        key = self.versioned_key(path, "index", name)
        return self.put("raw", key, payload, version=ver, source=str(path),
                        meta={"kind": "index", "name": name, "path": str(path)})

    def cached_preprocessed(self, key: str, *, version: FileVersion | None = None) -> Any | None:
        """取预处理结果缓存。"""
        return self.get("preprocessing", key, version=version)

    def store_preprocessed(self, key: str, payload: Any, *, fmt: str = "json",
                           version: FileVersion | None = None, source: str | Path | None = None) -> str:
        """存预处理结果缓存。"""
        return self.put("preprocessing", key, payload, fmt=fmt, version=version, source=source)

    # ------------------------------------------------------------------
    # 失效 / 清理 / 统计
    # ------------------------------------------------------------------
    def invalidate(self, *, level: str | None = None, key: str | None = None,
                   source: str | Path | None = None) -> int:
        """失效缓存条目（按级别 / key / 来源文件）。返回删除条数。"""
        removed = 0
        if source is not None:
            idx = self._load_index()
            for level_name, k in idx.pop(str(source), []):
                self._remove_entry(level_name, k)
                removed += 1
            if self._index_path.exists():
                self._ensure_dir(self._index_path.parent)
                self._index_path.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
        if level is not None and key is not None:
            self._remove_entry(level, key)
            removed += 1
        if level is not None and key is None:
            removed += self.clear(level)
        for lv in MEMORY_LEVELS:
            if level is None or level == lv:
                for mk in [m for m in self._mem if m[0] == lv]:
                    self._mem_bytes -= self._mem_size(self._mem.pop(mk))
                    removed += 1
        return removed

    def invalidate_file(self, path: str | Path) -> int:
        """源文件变化（或强制重建）时调用：清掉该文件相关的全部缓存条目。"""
        return self.invalidate(source=str(path))

    def clear(self, level: str | None = None) -> int:
        """清理缓存（全部或某一级）。返回清理条目数。"""
        removed = 0
        levels = DISK_LEVELS if level is None else (level,)
        for lv in levels:
            if lv not in DISK_LEVELS:
                continue
            d = self.root / lv
            if d.exists():
                removed += sum(1 for _ in d.rglob("*") if _.is_file())
                shutil.rmtree(d, ignore_errors=True)
        if level is None or level in MEMORY_LEVELS:
            removed += len(self._mem)
            self._mem.clear()
            self._mem_bytes = 0
        if level is None and self._index_path.exists():
            try:
                self._index_path.unlink()
            except OSError:  # pragma: no cover
                pass
        return removed

    def stats(self) -> dict[str, Any]:
        """缓存统计（命中/未命中/写入/各级条目数）。"""
        per_level: dict[str, int] = {}
        for lv in DISK_LEVELS:
            d = self.root / lv
            per_level[lv] = sum(1 for p in d.rglob("*.json")) if d.exists() else 0
        for lv in MEMORY_LEVELS:
            per_level[lv] = sum(1 for m in self._mem if m[0] == lv)
        return {
            "enabled": self.enabled,
            "root": str(self.root),
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._writes,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1), 4),
            "per_level": per_level,
            "memory_bytes": int(self._mem_bytes),
            "memory_limit_bytes": int(self.memory_limit_bytes),
        }
