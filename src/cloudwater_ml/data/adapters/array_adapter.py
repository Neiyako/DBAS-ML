"""数组/表格适配器：npz / csv / parquet → SampleBundle。

约定：

- npz 键名前缀 ``feature__*`` / ``target__*`` / ``meta__*``（也接受裸变量名键）；
- csv/parquet 按 ``data.features`` / ``data.targets`` / ``data.meta_columns`` 选列，
  只读需要的列，缺失变量直接报错并列出可用列（不猜测、不填充）；
- ``sampling.stride`` / ``sampling.max_samples`` 控制抽样。

v2.0 修正：

- **D13 parquet 投影式读取**：用 ``columns=`` 只读需要的列（大文件走行组迭代路径），
  不再「先读全表再筛列」；
- **D15 来源溯源**：``source_origin``（config / column / unresolved）写入 meta，
  ``sample_id`` 内容派生（不再是位置计数）；
- 补齐 ``iter_batches``（csv 分块读 / parquet 行组迭代），供大批量训练使用。
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..schema import SOURCE_UNKNOWN, SampleBundle, content_sample_ids
from .base import DataAdapter, register_adapter

__all__ = ["ArrayAdapter"]


def _adapter_of(data_cfg: dict[str, Any], path: Path) -> str:
    adapter = data_cfg.get("adapter") or path.suffix.lstrip(".")
    return str(adapter)


def _parquet_available_columns(path: Path) -> list[str]:
    """读 parquet schema（只读元数据，不读数据行）。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "读取 parquet 需要 pyarrow，请先安装：pip install 'cloudwater-ml[parquet]'"
        ) from exc
    return list(pq.ParquetFile(path).schema.names)


def _read_frame(data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    path = Path(data_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    adapter = _adapter_of(data_cfg, path)
    features = list(data_cfg.get("features") or [])
    targets = list(data_cfg.get("targets") or [])
    meta_cols = list(data_cfg.get("meta_columns") or [])
    wanted = features + targets

    if adapter == "npz" or path.suffix == ".npz":
        frame = _frame_from_npz(path, wanted, meta_cols)
    elif adapter == "csv" or path.suffix == ".csv":
        head = pd.read_csv(path, nrows=0)
        missing = [c for c in wanted if c not in head.columns]
        if missing:
            raise ValueError(
                f"{path} 缺少配置要求的变量 {missing}；文件实际列: {list(head.columns)}"
            )
        usecols = wanted + [c for c in meta_cols if c in head.columns]
        frame = pd.read_csv(path, usecols=usecols)  # 列投影
    elif adapter == "parquet" or path.suffix in (".parquet", ".pq"):
        available = _parquet_available_columns(path)
        missing = [c for c in wanted if c not in available]
        if missing:
            raise ValueError(
                f"{path} 缺少配置要求的变量 {missing}；文件实际列: {available}"
            )
        keep = wanted + [c for c in meta_cols if c in available]
        try:
            # 投影式读取：只读需要的列（不再先读全表）
            frame = pd.read_parquet(path, columns=keep)
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise RuntimeError(
                "读取 parquet 需要 pyarrow，请先安装：pip install 'cloudwater-ml[parquet]'"
            ) from exc
    else:
        raise ValueError(f"ArrayAdapter 不支持的格式: {adapter}（支持 npz / csv / parquet）")
    return frame, path


def _frame_from_npz(path: Path, wanted: list[str], meta_cols: list[str]) -> pd.DataFrame:
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        for name in wanted:
            for key in (f"feature__{name}", f"target__{name}", name):
                if key in keys:
                    arrays[name] = z[key]
                    break
            else:
                raise ValueError(
                    f"{path} 中找不到变量 {name!r}；可用键: {sorted(keys)}"
                )
        for col in meta_cols:
            for key in (f"meta__{col}", col):
                if key in keys:
                    try:
                        arrays[col] = z[key]
                    except ValueError as exc:
                        raise ValueError(
                            f"npz 元数据键 {key!r} 是 object 数组（读取需要 pickle）。"
                            "出于安全默认 allow_pickle=False；请重新生成该 npz（cloudwater synth）"
                            f"或改用 csv/parquet。原错误: {exc}"
                        ) from None
                    break
    return pd.DataFrame(arrays)


def _apply_sampling(frame: pd.DataFrame, data_cfg: dict[str, Any]) -> pd.DataFrame:
    sampling = data_cfg.get("sampling") or {}
    stride = int(sampling.get("stride") or 1)
    max_samples = sampling.get("max_samples")
    if stride > 1:
        frame = frame.iloc[::stride].reset_index(drop=True)
    if max_samples and len(frame) > int(max_samples):
        idx = np.linspace(0, len(frame) - 1, int(max_samples)).astype(int)
        frame = frame.iloc[idx].reset_index(drop=True)
    return frame


def _to_bundle(
    frame: pd.DataFrame,
    data_cfg: dict[str, Any],
    source_path: Path,
    seen: dict[str, int] | None = None,
) -> SampleBundle:
    features = list(data_cfg.get("features") or [])
    targets = list(data_cfg.get("targets") or [])
    meta_cols = [c for c in (data_cfg.get("meta_columns") or []) if c in frame.columns]

    meta = frame[meta_cols].copy().reset_index(drop=True)
    if "time" in meta.columns:
        try:
            meta["time"] = pd.to_datetime(meta["time"])
        except (TypeError, ValueError):  # 时间列解析失败保留原样，交给质量检查报
            pass

    override = data_cfg.get("source_label")
    if override:
        meta["source"] = override  # 显式配置覆盖（可追溯：报告里会注明）
        origin = "config"
    elif "source" in meta.columns:
        origin = "column"
    else:
        meta["source"] = SOURCE_UNKNOWN
        origin = "unresolved"
    meta["source_origin"] = origin
    if "dataset" not in meta.columns:
        meta["dataset"] = source_path.stem

    if "sample_id" not in meta.columns:
        # 内容派生（不用位置计数）：抽样后仍稳定；seen 跨批共享 → 跨批唯一
        key_cols = [c for c in ("time", "level", "latitude", "longitude") if c in meta.columns]
        payload = meta[key_cols] if key_cols else frame[features + targets]
        meta.insert(0, "sample_id", content_sample_ids(payload, tag=source_path.stem, seen=seen))

    bundle = SampleBundle(
        X=frame[features].to_numpy(dtype=np.float64),
        Y=frame[targets].to_numpy(dtype=np.float64),
        feature_names=features,
        target_names=targets,
        meta=meta,
        units={},
        source_file=str(source_path),
        source_origin=origin,
    )
    return bundle


def _iter_parquet_frames(path: Path, columns: list[str], batch_rows: int) -> Iterator[pd.DataFrame]:
    """parquet 行组迭代（不整表载入内存）。"""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas()


def _iter_csv_frames(path: Path, columns: list[str], batch_rows: int) -> Iterator[pd.DataFrame]:
    """csv 分块读。"""
    yield from pd.read_csv(path, usecols=columns, chunksize=batch_rows)


@register_adapter("npz")
@register_adapter("csv")
@register_adapter("parquet")
class ArrayAdapter(DataAdapter):
    """NumPy npz / CSV / Parquet 适配器（中小规模数据全载入内存）。"""

    name = "array"

    def load(self, data_cfg: dict[str, Any]) -> SampleBundle:
        frame, path = _read_frame(data_cfg)
        frame = _apply_sampling(frame, data_cfg)
        return _to_bundle(frame, data_cfg, path)

    def iter_batches(self, data_cfg: dict[str, Any], batch_samples: int) -> Iterator[SampleBundle]:
        """分批读取：parquet 走行组迭代 / csv 走分块读 / npz 一次读完。

        每批的样本数 ≤ ``batch_samples``；sample_id 内容派生且跨批唯一。
        """
        path = Path(data_cfg["path"])
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        adapter = _adapter_of(data_cfg, path)
        features = list(data_cfg.get("features") or [])
        targets = list(data_cfg.get("targets") or [])
        meta_cols = list(data_cfg.get("meta_columns") or [])
        wanted = features + targets
        batch_samples = max(1, int(batch_samples))
        stride = int((data_cfg.get("sampling") or {}).get("stride") or 1)
        max_samples = (data_cfg.get("sampling") or {}).get("max_samples")
        seen: dict[str, int] = {}

        if adapter == "npz" or path.suffix == ".npz":
            frame = _apply_sampling(_frame_from_npz(path, wanted, meta_cols), data_cfg)
            yield _to_bundle(frame, data_cfg, path)
            return

        if adapter == "csv" or path.suffix == ".csv":
            head = pd.read_csv(path, nrows=0)
            missing = [c for c in wanted if c not in head.columns]
            if missing:
                raise ValueError(
                    f"{path} 缺少配置要求的变量 {missing}；文件实际列: {list(head.columns)}"
                )
            cols = wanted + [c for c in meta_cols if c in head.columns]
            kept = 0
            for chunk in _iter_csv_frames(path, cols, batch_samples):
                if stride > 1:
                    chunk = chunk.iloc[::stride].reset_index(drop=True)
                if max_samples and kept + len(chunk) > int(max_samples):
                    chunk = chunk.iloc[: int(max_samples) - kept].reset_index(drop=True)
                kept += len(chunk)
                yield _to_bundle(chunk, data_cfg, path, seen)
                if max_samples and kept >= int(max_samples):
                    break
            return

        if adapter == "parquet" or path.suffix in (".parquet", ".pq"):
            available = _parquet_available_columns(path)
            missing = [c for c in wanted if c not in available]
            if missing:
                raise ValueError(
                    f"{path} 缺少配置要求的变量 {missing}；文件实际列: {available}"
                )
            cols = wanted + [c for c in meta_cols if c in available]
            kept = 0
            for chunk in _iter_parquet_frames(path, cols, batch_samples):
                if stride > 1:
                    chunk = chunk.iloc[::stride].reset_index(drop=True)
                if max_samples and kept + len(chunk) > int(max_samples):
                    chunk = chunk.iloc[: int(max_samples) - kept].reset_index(drop=True)
                kept += len(chunk)
                yield _to_bundle(chunk, data_cfg, path, seen)
                if max_samples and kept >= int(max_samples):
                    break
            return

        raise ValueError(f"ArrayAdapter 不支持的格式: {adapter}（支持 npz / csv / parquet）")

    def inspect(self, data_cfg: dict[str, Any]) -> dict[str, Any]:
        path = Path(data_cfg["path"])
        info: dict[str, Any] = {
            "adapter": data_cfg.get("adapter"),
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "features": list(data_cfg.get("features") or []),
            "targets": list(data_cfg.get("targets") or []),
        }
        if not path.exists():
            return info
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                info["keys"] = sorted(z.files)
                info["shapes"] = {k: list(z[k].shape) for k in z.files}
        elif path.suffix in (".parquet", ".pq"):
            try:
                info["columns"] = _parquet_available_columns(path)
                info["projection_read"] = True
            except RuntimeError as exc:
                info["columns"] = None
                info["error"] = str(exc)
        else:
            head = pd.read_csv(path, nrows=0)
            info["columns"] = list(head.columns)
        return info
