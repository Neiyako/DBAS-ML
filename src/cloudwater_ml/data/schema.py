"""数据标准结构与变量登记（schema）。

统一约定（详见 docs/data_schema.md）：

- 表格样本：``X: [samples, features]``、``Y: [samples, targets]``；
- 时空预留：``X_spatial: [batch, channels, height, width]``、
  ``X_temporal: [batch, time, channels, height, width]``；
- 元数据 ``meta`` 与样本一一对应：时间戳、经纬度、层次、样本 ID、
  数据来源（real / simulated / unknown）、数据集名 —— 保证样本可追溯到原始数据。

铁律（v2.0 修正）：

- **来源标签未解析时一律记为 ``unknown``**，绝不默认 ``simulated``
  （否则真实观测会被当成仿真数据，混合配比与域迁移结论全错）；
- ``subset`` 必须透传 ``units`` / ``source_file`` / ``source_origin``
  （划分后仍可溯源到原始文件与契约）；
- ``sample_id`` **内容派生**（坐标/取值哈希），同一份内容恒得同一 ID，
  跨抽样、跨批次稳定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "KNOWN_VARIABLES",
    "VariableSpec",
    "SampleBundle",
    "SOURCE_REAL",
    "SOURCE_SIMULATED",
    "SOURCE_UNKNOWN",
    "DATA_SCHEMA_VERSION",
    "content_sample_ids",
]

SOURCE_REAL = "real"
SOURCE_SIMULATED = "simulated"
SOURCE_UNKNOWN = "unknown"

#: 数据结构版本（写入 scaler 拟合证据 / 缓存条目，用于版本校验）
DATA_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class VariableSpec:
    """变量登记：含义、单位、物理约束提示。单位不硬编码到算法，仅作校验参考。"""

    name: str
    description: str
    units: str
    role: str = "feature"  # feature | target | derived
    nonnegative: bool = False  # 物理非负提示（供物理约束/范围检查参考）


def _v(name: str, desc: str, units: str, role: str = "feature", nonneg: bool = False) -> VariableSpec:
    return VariableSpec(name=name, description=desc, units=units, role=role, nonnegative=nonneg)


# 预留输入变量全集（数据源不必同时具备全部变量，缺失即缺省，不做填充）
KNOWN_VARIABLES: dict[str, VariableSpec] = {
    spec.name: spec
    for spec in [
        _v("t", "温度", "K"),
        _v("q", "比湿", "kg kg-1"),
        _v("rh", "相对湿度", "%"),
        _v("u", "东西向风速", "m s-1"),
        _v("v", "南北向风速", "m s-1"),
        _v("w", "垂直速度 (omega)", "Pa s-1"),
        _v("omega", "垂直运动 (omega 别名)", "Pa s-1"),
        _v("p", "气压", "hPa"),
        _v("z", "高度位势/几何高度", "m"),
        _v("clwc", "云液态水含量", "kg kg-1", role="target", nonneg=True),
        _v("ciwc", "云冰水含量", "kg kg-1", role="target", nonneg=True),
        _v("cwp", "云水路径", "kg m-2", role="target", nonneg=True),
    ]
}


def content_sample_ids(
    frame: pd.DataFrame,
    tag: str = "s",
    cols: list[str] | None = None,
    seen: dict[str, int] | None = None,
) -> list[str]:
    """内容派生的样本 ID 列表。

    同一份内容（坐标 / 取值）恒得同一 ID —— 抽样（stride / max_samples）、
    分批读取、重复加载都保持稳定；内容完全重复的样本按出现次序追加后缀，
    保证列表内唯一。

    Parameters
    ----------
    frame : 用于派生的数据表（通常是 meta 的坐标列或特征+目标矩阵）
    tag : ID 前缀（一般用文件名或数据集名，便于溯源）
    cols : 参与哈希的列名；``None`` 表示全部列
    seen : 外部计数器（跨批次共享 → 保证**跨批唯一**）；``None`` 表示用局部计数器
    """
    n = len(frame)
    if n == 0:
        return []
    use = [c for c in (cols if cols is not None else list(frame.columns)) if c in frame.columns]
    if use:
        hashes = pd.util.hash_pandas_object(frame[use], index=False).to_numpy(dtype=np.uint64)
    else:
        hashes = np.zeros(n, dtype=np.uint64)
    counter = seen if seen is not None else {}
    out: list[str] = []
    for h in hashes:
        base = f"{tag}-{int(h):016x}"
        k = counter.get(base, 0)
        counter[base] = k + 1
        out.append(base if k == 0 else f"{base}-{k}")
    return out


@dataclass
class SampleBundle:
    """统一数据载体：特征 + 目标 + 元数据 + 来源标签。

    Parameters
    ----------
    X : [n_samples, n_features]
    Y : [n_samples, n_targets]（单目标也是二维，列数为 1）
    meta : DataFrame，行与 X/Y 对齐，至少可含
        sample_id / time / latitude / longitude / level / source / dataset
    feature_names / target_names : 列名，顺序与 X/Y 列一致
    units : 变量单位字典（来自数据元数据，供质量检查与溯源）
    source_file : 追溯：样本来自哪个数据文件
    source_origin : 追溯：来源标签怎么来的（config / dataset_contract / column / unresolved）
    """

    X: np.ndarray
    Y: np.ndarray
    feature_names: list[str]
    target_names: list[str]
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    units: dict[str, str] = field(default_factory=dict)  # 变量单位（源自数据元数据，供质量检查）
    source_file: str = ""  # 追溯：样本来自哪个数据文件
    source_origin: str = ""  # 追溯：来源标签来源（config/dataset_contract/column/unresolved）

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.Y = np.asarray(self.Y, dtype=np.float64)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if self.Y.ndim == 1:
            self.Y = self.Y.reshape(-1, 1)
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"X/Y 样本数不一致: {self.X.shape[0]} vs {self.Y.shape[0]}")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X 列数 {self.X.shape[1]} 与 feature_names 数 {len(self.feature_names)} 不符"
            )
        if self.Y.shape[1] != len(self.target_names):
            raise ValueError(
                f"Y 列数 {self.Y.shape[1]} 与 target_names 数 {len(self.target_names)} 不符"
            )
        if len(self.meta) == 0:
            # 空 meta：来源标签记 unknown（绝不默认 simulated）
            payload = pd.DataFrame(np.hstack([self.X, self.Y])) if self.X.size else pd.DataFrame()
            self.meta = pd.DataFrame(
                {
                    "sample_id": content_sample_ids(payload, tag="s"),
                    "source": SOURCE_UNKNOWN,
                    "source_origin": "default_unknown",
                    "dataset": "unknown",
                }
            )
        if len(self.meta) != self.n_samples:
            raise ValueError(
                f"meta 行数 {len(self.meta)} 与样本数 {self.n_samples} 不一致（追溯关系被破坏）"
            )
        if "sample_id" not in self.meta.columns:
            # 内容派生（不用位置计数）：抽样/分批后 ID 保持稳定
            self.meta = self.meta.copy()
            payload = pd.DataFrame(np.hstack([self.X, self.Y])) if self.X.size else pd.DataFrame()
            self.meta.insert(0, "sample_id", content_sample_ids(payload, tag="s"))

    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_targets(self) -> int:
        return int(self.Y.shape[1])

    def subset(self, indices: np.ndarray | list[int]) -> SampleBundle:
        """按索引取子集，保持 X/Y/meta 三者对齐（划分与混合比例的基础操作）。

        **透传** ``units`` / ``source_file`` / ``source_origin``，保证划分后
        仍能溯源到原始文件与来源判定依据。
        """
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            raise ValueError(
                "subset 结果为空：所选索引没有任何样本。若来源标签未解析"
                "（meta.source='unknown'），不要按 real/simulated 过滤——"
                "先显式设置 data.source_label（real|simulated），或把 mix.real_fraction 置为 null。"
            )
        sub = SampleBundle(
            X=self.X[idx],
            Y=self.Y[idx],
            feature_names=list(self.feature_names),
            target_names=list(self.target_names),
            meta=self.meta.iloc[idx].reset_index(drop=True),
            units=dict(self.units or {}),
            source_file=self.source_file,
            source_origin=self.source_origin,
        )
        return sub

    @property
    def source_labels(self) -> pd.Series:
        """来源标签（real / simulated / unknown）；缺失列按 unknown 处理并保持显式。"""
        if "source" in self.meta.columns:
            return self.meta["source"].astype(str)
        return pd.Series([SOURCE_UNKNOWN] * self.n_samples, index=self.meta.index)

    def source_counts(self) -> dict[str, int]:
        return {str(k): int(v) for k, v in self.source_labels.value_counts().items()}

    def source_resolution(self) -> dict[str, Any]:
        """来源标签解析情况（用于如实记录 unresoved 来源，见质量检查 source_labels）。

        ``resolved`` 表示至少有一个样本带 real/simulated 明确标签；
        若 real 与 simulated 都不存在，任何按来源比例的混合都会把训练集清空。
        """
        counts = self.source_counts()
        n_real = counts.get(SOURCE_REAL, 0)
        n_sim = counts.get(SOURCE_SIMULATED, 0)
        return {
            "counts": counts,
            "n_real": n_real,
            "n_simulated": n_sim,
            "n_unknown": counts.get(SOURCE_UNKNOWN, 0),
            "n_other": int(sum(v for k, v in counts.items() if k not in (SOURCE_REAL, SOURCE_SIMULATED, SOURCE_UNKNOWN))),
            "resolved": bool(n_real or n_sim),
            "mixable": bool(n_real or n_sim),
            "source_origin": self.source_origin or self._meta_source_origin(),
        }

    def _meta_source_origin(self) -> str:
        if "source_origin" in self.meta.columns and len(self.meta):
            vals = self.meta["source_origin"].astype(str).unique().tolist()
            return "|".join(vals)
        return ""

    def describe(self) -> dict[str, Any]:
        """给 inspect / 报告用的结构摘要（不读科研数值本身）。"""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_targets": self.n_targets,
            "feature_names": list(self.feature_names),
            "target_names": list(self.target_names),
            "meta_columns": list(self.meta.columns),
            "source_counts": self.source_counts(),
            "source_origin": self.source_origin or self._meta_source_origin(),
            "units": dict(self.units or {}),
            "source_file": self.source_file,
            "shape_X": list(self.X.shape),
            "shape_Y": list(self.Y.shape),
        }


def validate_bundle_dims(bundle: SampleBundle) -> list[str]:
    """维度与类型检查，返回问题列表（空列表 = 通过）。"""
    problems: list[str] = []
    if bundle.X.ndim != 2:
        problems.append(f"X 应为二维 [samples, features]，实际 {bundle.X.ndim} 维")
    if bundle.Y.ndim != 2:
        problems.append(f"Y 应为二维 [samples, targets]，实际 {bundle.Y.ndim} 维")
    if not np.issubdtype(bundle.X.dtype, np.floating):
        problems.append(f"X dtype 应为浮点，实际 {bundle.X.dtype}")
    if not np.issubdtype(bundle.Y.dtype, np.floating):
        problems.append(f"Y dtype 应为浮点，实际 {bundle.Y.dtype}")
    if bundle.n_samples == 0:
        problems.append("样本数为 0")
    return problems


def spatial_shape_guard(shape: tuple[int, ...], kind: str = "spatial") -> None:
    """时空张量形状校验（第二阶段 CNN/U-Net/ConvLSTM 预留接口共用）。

    kind='spatial' 期望 [batch, channels, height, width]；
    kind='temporal' 期望 [batch, time, channels, height, width]。
    """
    expected = 4 if kind == "spatial" else 5
    if len(shape) != expected:
        raise ValueError(
            f"{kind} 张量应为 {expected} 维 {'[batch, channels, height, width]' if kind == 'spatial' else '[batch, time, channels, height, width]'}，实际 {shape}"
        )
