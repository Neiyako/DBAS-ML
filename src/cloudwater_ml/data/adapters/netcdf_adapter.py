"""NetCDF / Xarray 适配器：按需加载、分块、范围筛选、分批读取。

铁律（面向 TB 级 ERA5）：

- **禁止默认整载入内存**：先做时间/空间/层次范围筛选与 stride 抽样，
  再按 ``sampling.max_samples`` 降采样，**最后**才估算体积；
  超过 ``data.netcdf.max_in_memory_gb``（默认 2 GB）直接拒绝加载并给出
  **真正有效**的降载建议（stride / 收窄 *_range / 减小 chunks）；
- ``iter_batches`` 分批读取真正受 ``batch_samples`` 约束，并复用同一套体积守卫；
- **惰性读取**：有 dask 时走 xarray 分块（按块 compute，不整载）；没有 dask 时
  回退 netCDF4 原生按需读取，并**显式告警**（绝不静默忽略 chunks 配置）；
- ``inspect`` 只读 header/坐标元数据与契约（如 AtmosGen 的 ``atmosgen:contract``），
  不读变量数值；
- 来源标签优先级：配置 ``data.source_label`` > 数据集内契约 ``source.kind`` > **unknown**，
  未解析时**绝不**默认成 simulated（否则真实观测会被误当仿真数据）。

v2.0 修正清单：D1 来源标签 / D2 max_samples 生效顺序 / D3 dask 缺失告警 +
惰性路径 / D4 分批约束与守卫 / D5 跨批唯一 sample_id 与溯源字段 /
D12 气压层别名与单层-气压层对齐策略。
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..schema import SOURCE_UNKNOWN, SampleBundle, content_sample_ids
from .base import DataAdapter, register_adapter

__all__ = ["NetcdfAdapter", "read_source_kind"]

# 坐标名别名 → 统一 meta 列名
_ALIASES = {
    "lat": "latitude",
    "lon": "longitude",
    "lev": "level",
    "plev": "level",
    "pressure_level": "level",          # ERA5 常用坐标名（D12）
    "pressure_level_value": "level",
    "isobaricInhPa": "level",
    "valid_time": "time",
}
# meta 列名 → 展示顺序无关，这里仅登记已知坐标
_KNOWN_META = ("time", "level", "latitude", "longitude")

_DEFAULT_MAX_GB = 2.0
#: 单次读入内存的块大小（样本数）——惰性路径按块读取，避免整载
_DEFAULT_BLOCK_SAMPLES = 65536


def _dask_available() -> bool:
    """dask 是否可用（只做导入探测，不产生任何数据读取）。"""
    try:  # pragma: no cover - 取决于环境
        import dask  # noqa: F401

        return True
    except ImportError:
        return False


def _resolve_chunks(nc_cfg: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """返回 ``(chunks, dask_enabled)``。

    dask 缺失且配置了 chunks → **显式告警**（不静默忽略）：
    回退到 netCDF4 原生按需读取（仍不整载，但失去惰性分块计算能力）。
    """
    chunks = nc_cfg.get("chunks") or {}
    if not chunks:
        return {}, True
    if _dask_available():
        return dict(chunks), True
    warnings.warn(
        f"配置了 netcdf.chunks={dict(chunks)} 但当前环境未安装 dask，已回退到 netCDF4 原生按需读取"
        "（不会整载整个文件，但失去惰性分块计算能力）。"
        "安装：pip install 'cloudwater-ml[dask]'",
        RuntimeWarning,
        stacklevel=3,
    )
    return {}, False


def _open_dataset(data_cfg: dict[str, Any]) -> tuple[xr.Dataset, dict[str, Any]]:
    """打开 NetCDF（只读）。返回 (dataset, chunks_info)。"""
    path = Path(data_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"NetCDF 文件不存在: {path}")
    nc_cfg = data_cfg.get("netcdf") or {}
    engine = nc_cfg.get("engine", "auto")
    kwargs: dict[str, Any] = {}
    if engine and engine != "auto":
        kwargs["engine"] = engine
    chunks, dask_enabled = _resolve_chunks(nc_cfg)
    if chunks:
        kwargs["chunks"] = chunks
    ds = xr.open_dataset(path, **kwargs)
    chunks_info = {
        "requested_chunks": dict(nc_cfg.get("chunks") or {}),
        "effective_chunks": dict(chunks),
        "dask_available": _dask_available(),
        "dask_enabled": bool(chunks) and dask_enabled,
        "lazy": bool(chunks),
        "note": ("dask 分块生效（按块惰性 compute）" if chunks
                 else "未启用 dask 分块：netCDF4 原生按需读取（按块读，不整载）"),
    }
    return ds, chunks_info


def read_source_kind(ds: xr.Dataset) -> str | None:
    """从数据集全局属性解析来源（AtmosGen 契约 JSON 或常见 attr）。

    解析不出时返回 ``None`` —— 调用方必须记 ``unknown``，**不得**默认 simulated。
    """
    raw = ds.attrs.get("atmosgen:contract")
    if raw:
        try:
            kind = (json.loads(raw).get("source") or {}).get("kind")
            if kind in ("real", "simulated"):
                return str(kind)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    for key in ("source_kind", "data_source_kind", "source"):
        val = ds.attrs.get(key)
        if isinstance(val, str) and val in ("real", "simulated"):
            return str(val)
    return None


def _norm_name(name: str) -> str:
    return _ALIASES.get(name, name)


def _range_slice(coord: xr.DataArray, lo: Any, hi: Any, name: str) -> slice:
    """坐标范围 → 基于位置的切片（不依赖坐标升/降序，不读变量数据）。"""
    vals = coord.values
    if np.issubdtype(vals.dtype, np.datetime64):
        mask = (vals >= np.datetime64(lo)) & (vals <= np.datetime64(hi))
    else:
        mask = (vals >= float(lo)) & (vals <= float(hi))
    idx = np.flatnonzero(np.asarray(mask))
    if idx.size == 0:
        raise ValueError(f"范围筛选后 {name} 无坐标点: [{lo}, {hi}]（检查 data.netcdf.{name}_range）")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _apply_subsets(ds: xr.Dataset, data_cfg: dict[str, Any]) -> xr.Dataset:
    nc_cfg = data_cfg.get("netcdf") or {}
    coord_of = {"time": "time", "latitude": "latitude", "longitude": "longitude", "level": "level"}
    for key, (lo, hi) in {
        "time_range": ("time", None),
        "latitude_range": ("latitude", None),
        "longitude_range": ("longitude", None),
        "level_range": ("level", None),
    }.items():
        rng = nc_cfg.get(key)
        if not rng:
            continue
        want = coord_of[key.replace("_range", "")]
        actual = next((c for c in ds.coords if _norm_name(c) == want), None)
        if actual is None:
            raise ValueError(f"配置了 {key}，但数据集没有坐标 {want!r}；实际坐标: {list(ds.coords)}")
        ds = ds.isel({actual: _range_slice(ds.coords[actual], rng[0], rng[1], want)})

    # 空间/时间抽样步长（level 不抽样，保持垂直结构完整）
    stride = int((data_cfg.get("sampling") or {}).get("stride") or 1)
    if stride > 1:
        steps = {
            c: stride
            for c in ds.dims
            if _norm_name(c) in ("time", "latitude", "longitude")
        }
        if steps:
            ds = ds.isel({d: slice(None, None, s) for d, s in steps.items()})
    return ds


def _dim_order(ds: xr.Dataset, first_var: str) -> tuple[str, ...]:
    return tuple(ds[first_var].dims)


def _common_dims(ds: xr.Dataset, names: list[str]) -> tuple[str, ...]:
    dims = _dim_order(ds, names[0])
    dim_sets = {n: set(ds[n].dims) for n in names}
    for n, s in dim_sets.items():
        if s != set(dims):
            detail = {k: list(v) for k, v in dim_sets.items()}
            raise ValueError(
                "所选变量维度不一致，无法对齐成 [samples, features]（v1 限制：各变量须共享同一维度集，"
                f"见 docs/data_schema.md）: {detail}"
            )
    return dims


# ----------------------------------------------------------------------
# D12：气压层别名 + 单层/气压层对齐策略
# ----------------------------------------------------------------------
def _canonical_dim_order(ds: xr.Dataset, names: list[str], level_dim: str | None) -> tuple[str, ...]:
    """统一维度顺序：time → level → 其余（按首次出现顺序）。"""
    base = list(_dim_order(ds, names[0]))
    all_dims: list[str] = []
    for n in names:
        for d in ds[n].dims:
            if d not in all_dims:
                all_dims.append(d)
    if level_dim and level_dim not in base:
        base.insert(base.index("time") + 1 if "time" in base else 0, level_dim)
    for d in all_dims:
        if d not in base:
            base.append(d)
    return tuple(base)


def _select_level(ds: xr.Dataset, level_dim: str, data_cfg: dict[str, Any]) -> float:
    """level_alignment='select'：确定唯一目标气压层。"""
    nc_cfg = data_cfg.get("netcdf") or {}
    vals = np.asarray(ds.coords[level_dim].values, dtype=float)
    target_level = nc_cfg.get("target_level")
    if target_level is not None:
        sel = float(target_level)
        if not np.any(np.isclose(vals, sel)):
            raise ValueError(
                f"netcdf.target_level={sel} 不在数据集气压层 {vals.tolist()} 中，无法按层选择"
            )
        return sel
    rng = nc_cfg.get("level_range")
    if rng:
        sel_vals = vals[(vals >= float(rng[0])) & (vals <= float(rng[1]))]
        if sel_vals.size != 1:
            raise ValueError(
                f"level_alignment='select' 需要唯一气压层，但 netcdf.level_range={rng} 命中 "
                f"{sel_vals.size} 层: {sel_vals.tolist()}；请改用 level_range=[hPa, hPa] 或 target_level"
            )
        return float(sel_vals[0])
    raise ValueError(
        "level_alignment='select' 需要指定唯一气压层：请设置 netcdf.target_level=850 "
        "或 netcdf.level_range=[850, 850]"
    )


def _resolve_alignment(
    ds: xr.Dataset,
    names: list[str],
    data_cfg: dict[str, Any],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """确定统一维度顺序 + 对齐策略（D12）。

    - 所有变量维度集一致 → 直接对齐（v1 行为）；
    - **仅**在「是否含气压层维」上不一致 → 按 ``netcdf.level_alignment`` 处理：
      ``broadcast``（单层变量广播到各层，默认）/ ``select``（按层选择特征）/
      ``error``（明确报错）；
    - 其他不一致 → 明确报错（保持「维度不一致」提示）。
    """
    sets = {n: tuple(ds[n].dims) for n in names}
    distinct = {frozenset(s) for s in sets.values()}
    if len(distinct) == 1:
        dims = tuple(sets[names[0]])
        return dims, {"strategy": "identical", "level_dim": None, "broadcast_vars": [],
                      "selected_level": None, "dims": list(dims)}

    # 非气压层维度必须完全一致（只允许「是否含气压层维」不同）
    non_level = {n: frozenset(d for d in s if _norm_name(d) != "level") for n, s in sets.items()}
    if len(set(non_level.values())) > 1:
        detail = {k: list(v) for k, v in sets.items()}
        raise ValueError(
            "所选变量维度不一致，无法对齐成 [samples, features]（v1 限制：各变量须共享同一维度集，"
            f"见 docs/data_schema.md）: {detail}"
        )

    level_dims = {d for s in sets.values() for d in s if _norm_name(d) == "level"}
    if len(level_dims) > 1:
        raise ValueError(f"数据集存在多个气压层维 {sorted(level_dims)}，无法对齐成统一维度")
    level_dim = next(iter(level_dims))
    strategy = str((data_cfg.get("netcdf") or {}).get("level_alignment", "broadcast")).lower()
    if strategy not in ("broadcast", "select", "error"):
        raise ValueError(
            f"未知 netcdf.level_alignment={strategy!r}，可选: broadcast / select / error"
        )
    targets = list(data_cfg.get("targets") or [])
    target_has_level = any(_norm_name(d) == "level" for t in targets for d in sets.get(t, ()))
    if strategy == "error":
        raise ValueError(
            f"检测到单层变量与气压层变量混用（气压层维 {level_dim!r}），"
            "而 netcdf.level_alignment='error'。可选策略："
            "broadcast（单层特征广播到各层，默认）/ select（按层选择特征，需 netcdf.target_level）"
        )
    selected_level: float | None = None
    if strategy == "select" and not target_has_level:
        selected_level = _select_level(ds, level_dim, data_cfg)
    dims = _canonical_dim_order(ds, names, level_dim)
    if selected_level is not None:
        dims = tuple(d for d in dims if d != level_dim)
    broadcast_vars = [n for n in names if level_dim not in sets[n]]
    return dims, {
        "strategy": strategy,
        "level_dim": level_dim,
        "broadcast_vars": broadcast_vars,
        "selected_level": selected_level,
        "dims": list(dims),
    }


def _var_values(ds: xr.Dataset, name: str, dims: tuple[str, ...]) -> np.ndarray:
    """变量取值 → 广播到统一维度 ``dims``（缺失维按 1 广播，单层→各层）。"""
    var = ds[name]
    missing = [d for d in dims if d not in var.dims]
    if missing:
        var = var.expand_dims(dict.fromkeys(missing, 1))
    arr = np.asarray(var.transpose(*dims).values, dtype=np.float64)
    target = tuple(int(ds.sizes[d]) for d in dims)
    if arr.shape != target:
        arr = np.broadcast_to(arr, target)
    return arr


def _flatten_block(
    ds: xr.Dataset,
    features: list[str],
    targets: list[str],
    dims: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """把一个数据块摊平成 ``[samples, features]`` + meta（不整载：按块调用）。"""

    def cols(names: list[str]) -> np.ndarray:
        if not names:
            return np.empty((0, 0), dtype=np.float64)
        return np.column_stack([_var_values(ds, n, dims).ravel() for n in names])

    X, Y = cols(features), cols(targets)

    meta: dict[str, np.ndarray] = {}
    axes = []
    for d in dims:
        if d in ds.coords:
            axes.append(np.asarray(ds.coords[d].values))
        else:
            axes.append(np.arange(int(ds.sizes[d])))
    grids = np.meshgrid(*axes, indexing="ij") if axes else []
    for dim, grid in zip(dims, grids):
        col = _norm_name(dim)
        if col in _KNOWN_META:
            meta[col] = np.asarray(grid).ravel()
    if "time" in meta:
        t = pd.DatetimeIndex(meta["time"].astype("datetime64[ns]"))
        meta["time"] = t
        meta["year"] = t.year.to_numpy()
    return X, Y, pd.DataFrame(meta)


def _iter_slices(sizes: list[int], max_block: int) -> Iterator[tuple[slice, ...]]:
    """按维度顺序切块（保持行序不变），每块样本数 ≤ ``max_block``。

    维度按「外层 → 内层」嵌套：某维之后的所有维（tail）已经超过上限时，
    该维步长退化为 1 并继续向内递归 —— 保证**任何**维度组合下每块都不超限。
    """
    if not sizes:
        yield ()
        return
    n0 = int(sizes[0])
    tail = int(np.prod(sizes[1:])) if len(sizes) > 1 else 1
    step = max(1, int(max_block) // max(tail, 1))
    for start in range(0, n0, step):
        if tail <= max_block:
            yield (slice(start, start + step),)
        else:
            for rest in _iter_slices(sizes[1:], max_block):
                yield (slice(start, start + 1),) + rest


def _iter_blocks(
    ds: xr.Dataset,
    dims: tuple[str, ...],
    max_block_samples: int,
) -> Iterator[tuple[xr.Dataset, int, int]]:
    """沿维度顺序切块（保持行序不变），每块样本数 ≤ ``max_block_samples``。

    产出 ``(block_ds, offset, n_rows)``；``offset`` 是该块在全局样本序列中的起点。
    """
    sizes = [int(ds.sizes[d]) for d in dims]
    if not sizes:
        return
    offset = 0
    for sl in _iter_slices(sizes, max(1, int(max_block_samples))):
        block = ds.isel(dict(zip(dims, sl)))
        rows = int(np.prod([int(block.sizes[d]) for d in dims]))
        yield block, offset, rows
        offset += rows


def _attach_source_and_ids(
    meta: pd.DataFrame,
    source: str,
    source_origin: str,
    path: Path,
    seen: dict[str, int],
    value_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """写入来源标签、来源出处与**内容派生**的 sample_id（跨批唯一且稳定）。"""
    meta = meta.copy()
    key_cols = [c for c in ("time", "level", "latitude", "longitude") if c in meta.columns]
    if key_cols:
        ids = content_sample_ids(meta, tag=path.stem, cols=key_cols, seen=seen)
    elif value_frame is not None and len(value_frame):
        ids = content_sample_ids(value_frame, tag=path.stem, seen=seen)
    else:
        ids = content_sample_ids(meta, tag=path.stem, seen=seen)
    meta.insert(0, "sample_id", ids)
    meta["source"] = source
    meta["source_origin"] = source_origin
    meta["dataset"] = path.stem
    return meta


def _nbytes_estimate(ds: xr.Dataset, names: list[str], dims: tuple[str, ...] | None = None) -> int:
    """选中变量 + meta 的内存占用估算（字节）。"""
    total = 0
    if dims is None:
        dims = _dim_order(ds, names[0])
    n_cells = int(np.prod([int(ds.sizes[d]) for d in dims])) if dims else 0
    for n in names:
        itemsize = np.dtype(ds[n].dtype).itemsize
        total += int(np.prod(ds[n].shape)) * itemsize if dims is None else n_cells * itemsize
    # meta 开销：坐标列（time 8B + 其他 8B）+ 若干字符串列（按 16 字节估）
    n_meta_cols = sum(1 for d in dims if _norm_name(d) in _KNOWN_META) + 3
    total += n_cells * n_meta_cols * 16
    return total


def _max_samples_plan(ds: xr.Dataset, dims: tuple[str, ...], max_samples: int | None) -> dict[str, int]:
    """按样本上限算出各维抽样步长（保证样本数 ≤ 上限）。"""
    if not max_samples or max_samples <= 0:
        return {}
    sizes = {d: int(ds.sizes[d]) for d in dims}
    total = int(np.prod(list(sizes.values()))) if sizes else 0
    if total <= int(max_samples):
        return {}
    strides = dict.fromkeys(dims, 1)

    def kept() -> int:
        return int(np.prod([max(1, ceil(sizes[d] / strides[d])) for d in dims]))

    while kept() > int(max_samples):
        target = max(dims, key=lambda d: ceil(sizes[d] / strides[d]))
        strides[target] *= 2
        if all(strides[d] >= sizes[d] for d in dims):
            break
    return {d: s for d, s in strides.items() if s > 1}


@register_adapter("netcdf")
class NetcdfAdapter(DataAdapter):
    """NetCDF 适配器：范围筛选 → 抽样 → max_samples 降采样 → 体积守卫 → 按需加载。"""

    name = "netcdf"

    # ------------------------------------------------------------------
    def _prepare(
        self, data_cfg: dict[str, Any]
    ) -> tuple[xr.Dataset, list[str], list[str], tuple[str, ...], dict[str, Any]]:
        """打开 + 筛选 + 对齐，返回 (ds, features, targets, dims, align_info)。"""
        ds, chunks_info = _open_dataset(data_cfg)
        try:
            features = list(data_cfg.get("features") or [])
            targets = list(data_cfg.get("targets") or [])
            available = set(ds.data_vars)
            missing = [v for v in features + targets if v not in available]
            if missing:
                raise ValueError(
                    f"数据集缺少配置要求的变量 {missing}；实际数据变量: {sorted(available)}"
                )
            ds = _apply_subsets(ds, data_cfg)
            dims, align_info = _resolve_alignment(ds, features + targets, data_cfg)
            if align_info.get("selected_level") is not None:
                level_dim = align_info["level_dim"]
                ds = ds.sel({level_dim: align_info["selected_level"]})
            align_info["chunks"] = chunks_info
            return ds, features, targets, dims, align_info
        except Exception:
            ds.close()
            raise

    # ------------------------------------------------------------------
    def _apply_max_samples(
        self, ds: xr.Dataset, dims: tuple[str, ...], data_cfg: dict[str, Any]
    ) -> xr.Dataset:
        """按 ``sampling.max_samples`` 降采样（**体积估算之前**生效，D2）。"""
        max_samples = (data_cfg.get("sampling") or {}).get("max_samples")
        if not max_samples:
            return ds
        plan = _max_samples_plan(ds, dims, int(max_samples))
        if not plan:
            return ds
        return ds.isel({d: slice(None, None, s) for d, s in plan.items()})

    def _guard_memory(self, ds: xr.Dataset, names: list[str], data_cfg: dict[str, Any],
                      dims: tuple[str, ...] | None = None) -> xr.Dataset:
        """体积守卫（整体加载路径）。

        顺序：**先**按 max_samples 降采样，**再**估算体积；返回**降采样后的 Dataset**
        （调用方必须使用返回值，否则降采样会被丢弃 —— 这正是 v1 的死代码缺陷）。
        """
        if dims is None:
            dims = _common_dims(ds, names)
        ds = self._apply_max_samples(ds, dims, data_cfg)
        nbytes = _nbytes_estimate(ds, names, dims)
        max_gb = float((data_cfg.get("netcdf") or {}).get("max_in_memory_gb", _DEFAULT_MAX_GB))
        if nbytes > max_gb * (1 << 30):
            raise ValueError(
                f"按当前筛选预计加载 {nbytes / (1 << 30):.2f} GB，超过 data.netcdf.max_in_memory_gb="
                f"{max_gb} GB。有效降载手段：增大 data.sampling.stride（时间/空间抽样步长）、"
                "收窄 data.netcdf.time_range / latitude_range / longitude_range / level_range、"
                "减小 data.netcdf.chunks（dask 分块大小）；或改用 iter_batches 分批读取。"
            )
        return ds

    def _guard_batch(self, nbytes: int, data_cfg: dict[str, Any], max_block_samples: int) -> None:
        """分批路径的体积守卫（与整体守卫同一阈值，单批体积）。"""
        max_gb = float((data_cfg.get("netcdf") or {}).get("max_in_memory_gb", _DEFAULT_MAX_GB))
        if nbytes > max_gb * (1 << 30):
            raise ValueError(
                f"单批预计占用 {nbytes / (1 << 30):.2f} GB，超过 data.netcdf.max_in_memory_gb="
                f"{max_gb} GB（batch_samples={max_block_samples}）。有效降载手段："
                "减小 batch_samples、增大 data.sampling.stride、"
                "收窄 data.netcdf.time_range / latitude_range / longitude_range / level_range、"
                "减小 data.netcdf.chunks（dask 分块大小）。"
            )

    # ------------------------------------------------------------------
    def _source_info(self, ds: xr.Dataset, data_cfg: dict[str, Any]) -> tuple[str, str]:
        """来源标签 + 来源出处（未解析 → unknown，绝不默认 simulated）。"""
        override = data_cfg.get("source_label")
        if override:
            return str(override), "config"
        kind = read_source_kind(ds)
        if kind:
            return kind, "dataset_contract"
        return SOURCE_UNKNOWN, "unresolved"

    def _units(self, ds: xr.Dataset, features: list[str], targets: list[str]) -> dict[str, str]:
        """变量单位 + 坐标单位（level → hPa 等，供质量检查）。"""
        units: dict[str, str] = {}
        for n in features + targets:
            u = (ds[n].attrs or {}).get("units")
            if u:
                units[str(n)] = str(u)
        for c in ds.coords:
            u = (ds.coords[c].attrs or {}).get("units")
            if u:
                units[_norm_name(str(c))] = str(u)
        return units

    # ------------------------------------------------------------------
    def load(self, data_cfg: dict[str, Any]) -> SampleBundle:
        ds, features, targets, dims, align_info = self._prepare(data_cfg)
        try:
            source, source_origin = self._source_info(ds, data_cfg)
            units = self._units(ds, features, targets)
            ds = self._guard_memory(ds, features + targets, data_cfg, dims)
            sizes = [int(ds.sizes[d]) for d in dims]
            n_samples = int(np.prod(sizes)) if sizes else 0
            X = np.empty((n_samples, len(features)), dtype=np.float64)
            Y = np.empty((n_samples, len(targets)), dtype=np.float64)
            metas: list[pd.DataFrame] = []
            seen: dict[str, int] = {}
            block_samples = int((data_cfg.get("sampling") or {}).get("block_samples")
                                or _DEFAULT_BLOCK_SAMPLES)
            offset = 0
            for block, off, rows in _iter_blocks(ds, dims, block_samples):
                xb, yb, mb = _flatten_block(block, features, targets, dims)
                X[off:off + rows] = xb
                Y[off:off + rows] = yb
                metas.append(mb)
                offset += rows
            meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
        finally:
            ds.close()
        path = Path(data_cfg["path"])
        value_frame = pd.DataFrame(np.hstack([X, Y])) if X.size else None
        meta = _attach_source_and_ids(meta, source, source_origin, path, seen, value_frame)
        bundle = SampleBundle(
            X=X, Y=Y, feature_names=features, target_names=targets, meta=meta,
            units=units, source_file=str(path), source_origin=source_origin,
        )
        return bundle

    def iter_batches(self, data_cfg: dict[str, Any], batch_samples: int) -> Iterator[SampleBundle]:
        """沿维度顺序分批产出 SampleBundle：每批样本数 ≤ ``batch_samples``（D4）。

        每批都带 units / source_file / source_origin，且 sample_id 跨批唯一（D5）。
        """
        batch_samples = max(1, int(batch_samples))
        ds, features, targets, dims, align_info = self._prepare(data_cfg)
        try:
            source, source_origin = self._source_info(ds, data_cfg)
            units = self._units(ds, features, targets)
            # max_samples 降采样同样在分批路径生效（与整体加载一致）
            ds = self._apply_max_samples(ds, dims, data_cfg)
            n_vars = len(features) + len(targets)
            per_sample_bytes = n_vars * 8 + 8 * (sum(1 for d in dims if _norm_name(d) in _KNOWN_META) + 3)
            # 体积守卫：单批体积必须落在阈值内（复用同一阈值与同一套建议）
            self._guard_batch(batch_samples * per_sample_bytes, data_cfg, batch_samples)
            seen: dict[str, int] = {}
            path = Path(data_cfg["path"])
            for block, _off, rows in _iter_blocks(ds, dims, batch_samples):
                X, Y, meta = _flatten_block(block, features, targets, dims)
                value_frame = pd.DataFrame(np.hstack([X, Y])) if X.size else None
                meta = _attach_source_and_ids(meta, source, source_origin, path, seen, value_frame)
                yield SampleBundle(
                    X=X, Y=Y, feature_names=features, target_names=targets, meta=meta,
                    units=units, source_file=str(path), source_origin=source_origin,
                )
        finally:
            ds.close()

    def inspect(self, data_cfg: dict[str, Any]) -> dict[str, Any]:
        """只读 header：维度、坐标、变量与单位、契约来源、预计样本量（不读数值）。"""
        path = Path(data_cfg["path"])
        ds, chunks_info = _open_dataset(data_cfg)
        try:
            features = list(data_cfg.get("features") or [])
            targets = list(data_cfg.get("targets") or [])
            missing = [v for v in features + targets if v not in ds.data_vars]
            info: dict[str, Any] = {
                "adapter": "netcdf",
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "dims": dict(ds.sizes),
                "coords": {c: str(ds.coords[c].dtype) for c in ds.coords},
                "coord_units": {_norm_name(str(c)): (ds.coords[c].attrs or {}).get("units")
                                for c in ds.coords},
                "data_vars": {
                    n: {
                        "dims": list(ds[n].dims),
                        "dtype": str(ds[n].dtype),
                        "units": (ds[n].attrs or {}).get("units"),
                        "long_name": (ds[n].attrs or {}).get("long_name"),
                    }
                    for n in ds.data_vars
                },
                "source_kind": read_source_kind(ds),
                "missing_variables": missing,
                "chunks": chunks_info,
                "global_attrs": {
                    k: str(v)[:200] for k, v in list(ds.attrs.items())[:20]
                    if k != "atmosgen:contract"
                },
            }
            present = [v for v in features + targets if v in ds.data_vars]
            if present:
                sub = _apply_subsets(ds, data_cfg)
                try:
                    dims, align_info = _resolve_alignment(sub, present, data_cfg)
                    info["selected_dims"] = list(dims)
                    info["level_alignment"] = align_info
                    info["estimated_samples"] = int(np.prod([sub.sizes[d] for d in dims]))
                    info["estimated_bytes"] = _nbytes_estimate(sub, present, dims)
                except ValueError as exc:
                    info["alignment_error"] = str(exc)
            return info
        finally:
            ds.close()
