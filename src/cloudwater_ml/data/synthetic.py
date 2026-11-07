"""合成测试数据生成（仅用于软件链路验证，不代表真实大气物理模拟）。

生成 ~5000 条样本：输入特征 t/q/rh/u/v/w，目标 clwc/ciwc 由固定的平滑非线性
函数 + 噪声构造（非负截断），并附带时间戳 / 经纬度 / 样本 ID / 来源标签。

铁律：本数据 source 恒为 ``simulated``，绝不标成 real（禁止把仿真数据误认为真实观测）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import SOURCE_SIMULATED, SampleBundle

__all__ = ["make_synthetic_bundle", "save_synthetic_npz", "ensure_synthetic_dataset"]

DEFAULT_FEATURES = ["t", "q", "rh", "u", "v", "w"]
DEFAULT_TARGETS = ["clwc", "ciwc"]
_SUPPORTED = set(DEFAULT_FEATURES) | set(DEFAULT_TARGETS)


def make_synthetic_bundle(
    n_samples: int = 5000,
    seed: int = 42,
    start_year: int = 2015,
    end_year: int = 2019,
    features: list[str] | None = None,
    targets: list[str] | None = None,
) -> SampleBundle:
    """构造可重复的合成样本（固定 seed 下逐位一致）。"""
    features = list(features or DEFAULT_FEATURES)
    targets = list(targets or DEFAULT_TARGETS)
    unknown = [v for v in features + targets if v not in _SUPPORTED]
    if unknown:
        raise ValueError(
            f"合成生成器仅支持内置变量 {sorted(_SUPPORTED)}，收到未知变量: {unknown}"
        )

    rng = np.random.default_rng(seed)
    n = int(n_samples)

    # --- 输入特征（量级参考对流层常见值，纯工程测试用） ---
    t = rng.normal(240.0, 12.0, n)  # K
    q = np.clip(rng.normal(2.0e-3, 1.5e-3, n), 1e-6, None)  # kg kg-1
    rh = np.clip(rng.normal(65.0, 18.0, n), 0.0, 100.0)  # %
    u = rng.normal(0.0, 6.0, n)  # m s-1
    v = rng.normal(0.0, 6.0, n)  # m s-1
    w = rng.normal(0.0, 0.2, n)  # Pa s-1（仅测试用，不取物理符号约定）

    # --- 目标：平滑非线性 + 噪声，非负截断（软件测试构造式，非物理模型） ---
    rh_c = (rh - 65.0) / 20.0
    w_up = -w / 0.2
    clwc = np.clip(
        8.0e-5 * np.clip(rh_c, 0.0, None) ** 1.5
        + 3.0e-5 * np.clip(w_up, 0.0, None)
        + 1.0e-5 * (q / 2.0e-3)
        + rng.normal(0.0, 8.0e-6, n),
        0.0,
        None,
    )
    ciwc = np.clip(
        5.0e-5 * np.clip(-(t - 240.0) / 12.0, 0.0, None)  # 更冷 → 冰相更多（测试构造）
        + 2.0e-5 * np.clip(w_up, 0.0, None)
        + 5.0e-6 * (q / 2.0e-3)
        + rng.normal(0.0, 6.0e-6, n),
        0.0,
        None,
    )

    col_map = {"t": t, "q": q, "rh": rh, "u": u, "v": v, "w": w, "clwc": clwc, "ciwc": ciwc}
    X = np.column_stack([col_map[f] for f in features]).astype(np.float64)
    Y = np.column_stack([col_map[name] for name in targets]).astype(np.float64)

    # --- 元数据：跨 5 个年份的时间戳 + 高原经纬度盒（time/year 划分可测） ---
    years = rng.integers(start_year, end_year + 1, size=n)
    doy = rng.integers(0, 365, size=n)
    hours = rng.integers(0, 24, size=n)
    epoch = pd.to_datetime([f"{int(y)}-01-01" for y in years])
    times = epoch + pd.to_timedelta(doy, unit="D") + pd.to_timedelta(hours, unit="h")
    lat = rng.uniform(28.0, 38.0, n)
    lon = rng.uniform(78.0, 103.0, n)

    # 统一按时间升序重排所有列，保证 X/Y/meta 逐行对齐
    order = np.argsort(times.values, kind="stable")
    X, Y = X[order], Y[order]
    times = pd.DatetimeIndex(times[order])
    years = years[order]
    lat, lon = lat[order], lon[order]

    meta = pd.DataFrame(
        {
            "sample_id": [f"syn-{seed}-{i:06d}" for i in range(n)],
            "time": times,
            "latitude": np.round(lat, 4),
            "longitude": np.round(lon, 4),
            "source": SOURCE_SIMULATED,  # 恒为 simulated
            "dataset": "synthetic_test_data",  # 独立数据集标签
            "year": years.astype(int),
        }
    )
    return SampleBundle(X=X, Y=Y, feature_names=features, target_names=targets, meta=meta)


def save_synthetic_npz(bundle: SampleBundle, path: str | Path) -> Path:
    """把 bundle 存成 npz（feature__/target__/meta__ 前缀约定），供 adapter: npz 读取。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for i, name in enumerate(bundle.feature_names):
        arrays[f"feature__{name}"] = bundle.X[:, i]
    for i, name in enumerate(bundle.target_names):
        arrays[f"target__{name}"] = bundle.Y[:, i]
    for col in bundle.meta.columns:
        arr = bundle.meta[col].to_numpy()
        if arr.dtype == object:  # 字符串列转定长 Unicode，避免 object 数组（加载需 pickle）
            arr = arr.astype(str)
        arrays[f"meta__{col}"] = arr
    np.savez_compressed(path, **arrays)
    return path


def ensure_synthetic_dataset(
    path: str | Path = "data/synthetic/cloudwater_synth.npz",
    *,
    n_samples: int = 5000,
    seed: int = 42,
    start_year: int = 2015,
    end_year: int = 2019,
    force: bool = False,
) -> Path:
    """确保合成数据文件存在（不存在则按固定参数生成；force=True 强制重建）。"""
    path = Path(path)
    if path.exists() and not force:
        return path
    bundle = make_synthetic_bundle(
        n_samples=n_samples, seed=seed, start_year=start_year, end_year=end_year
    )
    save_synthetic_npz(bundle, path)
    return path
