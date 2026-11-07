"""数据质量检查：缺失 / NaN-Inf / 极端值 / 范围 / 维度 / 单位坐标元数据。

默认 ``quality.report_only=true``：**只生成报告，不修改科研数据**。
自动清洗必须显式配置（``report_only: false`` + ``cleaning: {...}``），
每次清洗动作会被记录进报告 ``cleaning.applied``，可追踪、可复核。

v2.0 深化（D10）：

- ``missing``：除逐列统计外，增加**缺测分布**（按气压层 / 纬度带 / 季节）；
- ``extreme``：同时检查 **X 侧**异常值（原来只看 Y）；
- ``range``：接入**默认物理边界**（clwc / ciwc / cwp 非负 + 宽松上界），
  配置的 ``quality.ranges`` 优先；
- 新增可选检查名：``time``（时间单调性 / 重复 / 缺测连续性）、
  ``geo``（经纬度范围 / 间隔 / 0-360 约定）、``levels``（气压层顺序与单位）、
  ``source_labels``（来源标签解析情况，未解析会如实报出）。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    KNOWN_VARIABLES,
    SampleBundle,
    validate_bundle_dims,
)

__all__ = [
    "run_quality_checks",
    "format_quality_report",
    "apply_cleaning",
    "DEFAULT_CHECKS",
    "PHYSICAL_BOUNDS",
]

_Z_THRESHOLD = 4.0

#: 默认检查项（保持既有行为；新检查名为可选，需在 quality.checks 中显式启用）
DEFAULT_CHECKS = ["missing", "nan_inf", "extreme", "range", "dimensions", "units"]

#: 默认物理边界（单位与 KNOWN_VARIABLES 一致；上界取宽松值，只抓「物理上不可能」）
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "clwc": (0.0, 0.1),     # kg kg-1（云液态水含量上界 ~0.05，取 0.1 宽松）
    "ciwc": (0.0, 0.1),     # kg kg-1
    "cwp": (0.0, 500.0),    # kg m-2
}


def _norm_unit(u: Any) -> str:
    """单位写法归一（仅用于比较，不做单位换算）。"""
    if u is None:
        return ""
    s = str(u).strip().lower().replace("**", "").replace("·", " ")
    s = s.replace("/kg", " kg-1").replace("kg-1", " kg-1")
    s = s.replace("/", " ").replace(" ", "")
    return s


def _physical_ranges(bundle: SampleBundle) -> tuple[dict[str, list[float]], list[str]]:
    """默认物理边界（仅在单位与登记表一致时套用）。"""
    ranges: dict[str, list[float]] = {}
    skipped: list[str] = []
    provided = dict(bundle.units or {})
    for name, (lo, hi) in PHYSICAL_BOUNDS.items():
        if name not in bundle.target_names:
            continue
        unit = provided.get(name)
        if unit and _norm_unit(unit) != _norm_unit(KNOWN_VARIABLES[name].units):
            skipped.append(f"{name}(单位 {unit!r} 与登记 {KNOWN_VARIABLES[name].units!r} 不一致，未套用默认边界)")
            continue
        ranges[name] = [float(lo), float(hi)]
    return ranges, skipped


def _col_stats(arr: np.ndarray, names: list[str], prefix: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(names):
        col = arr[:, i]
        nan = int(np.isnan(col).sum())
        inf = int(np.isinf(col).sum())
        finite = col[np.isfinite(col)]
        out[f"{prefix}{name}"] = {
            "nan": nan,
            "inf": inf,
            "missing_fraction": round((nan + inf) / max(len(col), 1), 6),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
            "std": float(finite.std()) if finite.size else None,
        }
    return out


def _extreme_check(arr: np.ndarray, names: list[str], prefix: str) -> dict[str, Any]:
    per: dict[str, Any] = {}
    total = 0
    for i, name in enumerate(names):
        col = arr[:, i]
        finite = col[np.isfinite(col)]
        if finite.size < 2:
            continue
        mean, std = float(finite.mean()), float(finite.std())
        if std < 1e-12:
            per[f"{prefix}{name}"] = {"n_extreme": 0, "note": "常数列，z-score 不适用"}
            continue
        z = np.abs((finite - mean) / std)
        n_ext = int((z > _Z_THRESHOLD).sum())
        total += n_ext
        per[f"{prefix}{name}"] = {
            "n_extreme": n_ext,
            "fraction": round(n_ext / int(finite.size), 6),
            "z_threshold": _Z_THRESHOLD,
        }
    return {"method": f"|z|>{_Z_THRESHOLD}（基于均值/标准差，仅报告不裁剪）",
            "per_column": per, "total_extreme": total}


def _range_check(bundle: SampleBundle, ranges: dict[str, list[float]]) -> dict[str, Any]:
    per: dict[str, Any] = {}
    passed = True
    cols: dict[str, np.ndarray] = {}
    for i, n in enumerate(bundle.feature_names):
        cols[n] = bundle.X[:, i]
    for i, n in enumerate(bundle.target_names):
        cols[n] = bundle.Y[:, i]
    phys, skipped_phys = _physical_ranges(bundle)
    merged: dict[str, list[float]] = {k: list(map(float, v)) for k, v in phys.items()}
    merged.update({k: list(map(float, v)) for k, v in (ranges or {}).items()})
    for name, (lo, hi) in merged.items():
        if name not in cols:
            per[name] = {"skipped": "该变量不在本数据集的选择变量内"}
            continue
        col = cols[name][np.isfinite(cols[name])]
        bad = int(((col < float(lo)) | (col > float(hi))).sum())
        per[name] = {
            "configured_range": [float(lo), float(hi)],
            "range_source": "config" if name in (ranges or {}) else "physical_default",
            "violations": bad,
            "fraction": round(bad / max(col.size, 1), 6),
        }
        if bad:
            passed = False
    return {"per_variable": per, "passed": passed,
            "physical_defaults": phys,
            "physical_defaults_skipped": skipped_phys,
            "note": "范围检查只报告；裁剪/剔除须显式配置 quality.cleaning"}


def _units_check(bundle: SampleBundle) -> dict[str, Any]:
    expected = {n: KNOWN_VARIABLES[n].units for n in bundle.feature_names + bundle.target_names
                if n in KNOWN_VARIABLES}
    provided = dict(bundle.units or {})
    mismatches, unknown = [], []
    for name, exp in expected.items():
        got = provided.get(name)
        if not got:
            unknown.append(name)
        elif str(got).replace("**", "").replace("^", "") != exp.replace(" ", ""):
            # 单位写法差异只提示（Pa vs hPa 属实质差异，人工核对）
            if str(got).split()[0] != exp.split()[0]:
                mismatches.append({"variable": name, "expected_by_registry": exp, "found": got})
    return {
        "registry_expected": expected,
        "provided_by_data": provided,
        "mismatches": mismatches,
        "units_unknown": unknown,
        "passed": not mismatches,
    }


# ----------------------------------------------------------------------
# v2.0 新增检查
# ----------------------------------------------------------------------
def _modal_delta(deltas: pd.Series) -> Any:
    """最常见的时间间隔（众数）——用于识别「缺测造成的空洞」。"""
    if deltas.empty:
        return None
    counts = deltas.value_counts()
    return counts.index[0]


def _time_check(bundle: SampleBundle) -> dict[str, Any]:
    """时间单调性 / 重复 / 缺测连续性。

    网格数据里同一时刻对应多个网格点（时间戳重复是**正常**的），因此：
    - 「间隔 / 空洞」在**唯一时刻序列**上计算；
    - 「重复」只报告数量，真正的异常判据是「每个时刻的样本数不一致」。
    """
    if "time" not in bundle.meta.columns:
        return {"skipped": "meta 无 time 列", "passed": True, "problems": []}
    raw = bundle.meta["time"]
    t = pd.to_datetime(raw, errors="coerce")
    nat = int(t.isna().sum())
    valid = t.dropna()
    out: dict[str, Any] = {
        "n_total": int(len(t)),
        "n_nat": nat,
        "is_monotonic_increasing": bool(valid.is_monotonic_increasing) if len(valid) else True,
        "n_duplicates": int(valid.duplicated().sum()),
        "min": str(valid.min()) if len(valid) else None,
        "max": str(valid.max()) if len(valid) else None,
    }
    if len(valid):
        counts = valid.value_counts()
        out["n_unique_timestamps"] = int(counts.size)
        out["samples_per_timestamp"] = {
            "min": int(counts.min()), "max": int(counts.max()),
            "uniform": bool(counts.nunique() == 1),
        }
        uniq = counts.index.sort_values()
        deltas = pd.Series(uniq).diff().dropna()
        expected = _modal_delta(deltas)
        out["expected_interval"] = str(expected) if expected is not None else None
        if expected is not None and expected != pd.Timedelta(0):
            gaps = deltas[deltas > expected]
            out["n_gaps"] = int(len(gaps))
            out["gap_examples"] = [str(x) for x in gaps.head(5)]
            out["max_gap"] = str(deltas.max()) if len(deltas) else None
        else:
            out["n_gaps"] = 0
            out["gap_examples"] = []
            out["max_gap"] = None
        out["interval_stats"] = {
            "min": str(deltas.min()) if len(deltas) else None,
            "median": str(deltas.median()) if len(deltas) else None,
            "max": str(deltas.max()) if len(deltas) else None,
        }
    dup_values = valid[valid.duplicated(keep=False)]
    out["duplicate_examples"] = [str(x) for x in dup_values.head(5)]
    problems: list[str] = []
    if nat:
        problems.append(f"{nat} 个时间戳无法解析（NaT）")
    if not out["is_monotonic_increasing"]:
        problems.append("时间戳非单调递增（存在乱序样本）")
    if not out.get("samples_per_timestamp", {}).get("uniform", True):
        problems.append(
            "每个时刻的样本数不一致（min={min}, max={max}）——采样不均匀".format(
                **out["samples_per_timestamp"])
        )
    if out.get("n_gaps"):
        problems.append(f"{out['n_gaps']} 处时间间隔大于期望间隔 {out.get('expected_interval')}（疑似缺测）")
    out["problems"] = problems
    out["passed"] = not problems
    return out


def _coord_report(values: np.ndarray, lo: float, hi: float, label: str) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out: dict[str, Any] = {
        "n": int(v.size),
        "min": float(v.min()) if v.size else None,
        "max": float(v.max()) if v.size else None,
        "valid_range": [lo, hi],
        "out_of_range": int(((v < lo) | (v > hi)).sum()),
    }
    if v.size:
        uniq = np.unique(v)
        out["n_unique"] = int(uniq.size)
        out["n_duplicates"] = int(v.size - uniq.size)
        diffs = np.diff(uniq) if uniq.size > 1 else np.array([])
        if diffs.size:
            vals, counts = np.unique(diffs, return_counts=True)
            expected = float(vals[int(np.argmax(counts))])
            out["expected_spacing"] = expected
            out["n_irregular"] = int((np.abs(diffs - expected) > 1e-9).sum())
        else:
            out["expected_spacing"] = None
            out["n_irregular"] = 0
    return out


def _geo_check(bundle: SampleBundle) -> dict[str, Any]:
    """经纬度范围 / 间隔 / 0-360 经度约定。"""
    out: dict[str, Any] = {}
    problems: list[str] = []
    if "latitude" in bundle.meta.columns:
        lat = bundle.meta["latitude"].to_numpy(dtype=float)
        rep = _coord_report(lat, -90.0, 90.0, "latitude")
        out["latitude"] = rep
        if rep["out_of_range"]:
            problems.append(f"{rep['out_of_range']} 个纬度越界（应在 [-90, 90]）")
        if rep.get("n_duplicates"):
            problems.append(f"纬度有 {rep['n_duplicates']} 个重复值")
    else:
        out["latitude"] = {"skipped": "meta 无 latitude 列"}
    if "longitude" in bundle.meta.columns:
        lon = bundle.meta["longitude"].to_numpy(dtype=float)
        rep = _coord_report(lon, -180.0, 360.0, "longitude")
        finite = lon[np.isfinite(lon)]
        in_036 = bool(finite.size and ((finite >= 0) & (finite <= 360)).all())
        in_pm180 = bool(finite.size and ((finite >= -180) & (finite <= 180)).all())
        rep["convention"] = "0-360" if in_036 else ("-180-180" if in_pm180 else "mixed")
        rep["out_of_range_pm180"] = int(((finite < -180) | (finite > 180)).sum()) if finite.size else 0
        out["longitude"] = rep
        if rep["convention"] == "mixed":
            problems.append("经度混用 0-360 与 -180-180 两种约定（需统一）")
        if rep["out_of_range_pm180"]:
            problems.append(f"{rep['out_of_range_pm180']} 个经度超出 [-180, 180]（若采用 0-360 约定可忽略）")
        if rep.get("n_duplicates"):
            problems.append(f"经度有 {rep['n_duplicates']} 个重复值")
    else:
        out["longitude"] = {"skipped": "meta 无 longitude 列"}
    out["problems"] = problems
    out["passed"] = not problems
    return out


def _level_check(bundle: SampleBundle) -> dict[str, Any]:
    """气压层顺序 / 单位 / 重复。"""
    if "level" not in bundle.meta.columns:
        return {"skipped": "meta 无 level 列（非气压层数据）", "passed": True, "problems": []}
    lv = pd.to_numeric(bundle.meta["level"], errors="coerce")
    n_nan = int(lv.isna().sum())
    valid = lv.dropna()
    uniq = valid.unique()
    out: dict[str, Any] = {
        "n_levels": int(uniq.size),
        "levels": [float(x) for x in uniq[:50]],
        "n_nan": n_nan,
        "n_duplicates": int(valid.duplicated().sum()),
        "order": "unknown",
        "unit": str((bundle.units or {}).get("level") or ""),
    }
    problems: list[str] = []
    if uniq.size:
        ascending = bool(np.all(np.diff(uniq) > 0))
        descending = bool(np.all(np.diff(uniq) < 0))
        out["order"] = "ascending" if ascending else ("descending" if descending else "unordered")
        if out["order"] == "unordered":
            problems.append("气压层顺序非单调（ascending/descending），请核对 level 坐标")
    unit = out["unit"].strip().lower()
    if unit:
        ok = unit in ("hpa", "millibar", "mb", "hpa)", "millibars")
        out["unit_ok"] = ok
        if not ok:
            problems.append(f"气压层单位为 {out['unit']!r}，期望 hPa（未自动换算）")
    else:
        out["unit_ok"] = None
    if out["n_duplicates"]:
        problems.append(f"气压层有 {out['n_duplicates']} 个重复值")
    if n_nan:
        problems.append(f"{n_nan} 个气压层值无法解析")
    out["problems"] = problems
    out["passed"] = not problems
    return out


def _source_check(bundle: SampleBundle) -> dict[str, Any]:
    """来源标签解析情况（未解析必须如实报出，见 D1）。"""
    res = bundle.source_resolution()
    out = {
        "counts": res["counts"],
        "n_real": res["n_real"],
        "n_simulated": res["n_simulated"],
        "n_unknown": res["n_unknown"],
        "n_other": res["n_other"],
        "resolved": res["resolved"],
        "source_origin": res["source_origin"],
        "source_file": bundle.source_file,
    }
    problems: list[str] = []
    if res["n_unknown"]:
        problems.append(
            f"{res['n_unknown']} 个样本来源未解析（unknown）——按 real/simulated 过滤会把训练集清空；"
            "请显式设置 data.source_label（real|simulated）或把 mix.real_fraction 置为 null"
        )
    if res["n_other"]:
        problems.append(f"{res['n_other']} 个样本来源标签既非 real 也非 simulated 也非 unknown")
    out["problems"] = problems
    out["passed"] = not problems
    return out


def _season_of(t: pd.Series) -> pd.Series:
    """月份 → 季节标签（DJF/MAM/JJA/SON）。"""
    month = t.dt.month
    return month.map(lambda m: "DJF" if m in (12, 1, 2) else "MAM" if m in (3, 4, 5)
                      else "JJA" if m in (6, 7, 8) else "SON")


def _missing_distribution(bundle: SampleBundle) -> dict[str, Any]:
    """缺测分布：按气压层 / 纬度带 / 季节。"""
    n = bundle.n_samples
    if n == 0:
        return {"skipped": "无样本"}
    bad = ~np.isfinite(bundle.X).all(axis=1)
    bad |= ~np.isfinite(bundle.Y).all(axis=1)
    frame = pd.DataFrame({"missing": bad})
    out: dict[str, Any] = {"n_missing_samples": int(bad.sum()),
                            "missing_sample_fraction": round(float(bad.mean()), 6)}
    if "level" in bundle.meta.columns:
        frame["level"] = bundle.meta["level"].to_numpy()
        g = frame.groupby("level")["missing"].agg(["sum", "count"])
        out["by_level"] = {str(k): {"missing": int(r["sum"]), "n": int(r["count"]),
                                    "fraction": round(float(r["sum"]) / max(int(r["count"]), 1), 6)}
                           for k, r in g.iterrows()}
    if "latitude" in bundle.meta.columns:
        lat = pd.to_numeric(bundle.meta["latitude"], errors="coerce")
        frame["lat_band"] = np.floor(lat / 10.0) * 10.0
        g = frame.dropna(subset=["lat_band"]).groupby("lat_band")["missing"].agg(["sum", "count"])
        out["by_latitude_band"] = {f"{k:g}": {"missing": int(r["sum"]), "n": int(r["count"]),
                                              "fraction": round(float(r["sum"]) / max(int(r["count"]), 1), 6)}
                                   for k, r in g.iterrows()}
    if "time" in bundle.meta.columns:
        t = pd.to_datetime(bundle.meta["time"], errors="coerce")
        frame["season"] = _season_of(t).to_numpy()
        g = frame.dropna(subset=["season"]).groupby("season")["missing"].agg(["sum", "count"])
        out["by_season"] = {str(k): {"missing": int(r["sum"]), "n": int(r["count"]),
                                     "fraction": round(float(r["sum"]) / max(int(r["count"]), 1), 6)}
                            for k, r in g.iterrows()}
    return out


def run_quality_checks(bundle: SampleBundle, data_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行质量检查，返回 JSON 友好报告（默认不改数据）。

    可选检查名（需在 ``quality.checks`` 中显式启用）：
    ``time`` / ``geo`` / ``levels`` / ``source_labels``。
    """
    data_cfg = data_cfg or {}
    q_cfg = data_cfg.get("quality") or {}
    checks_wanted = set(q_cfg.get("checks") or DEFAULT_CHECKS)
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    if "missing" in checks_wanted:
        stats = {}
        stats.update(_col_stats(bundle.X, bundle.feature_names, "X."))
        stats.update(_col_stats(bundle.Y, bundle.target_names, "Y."))
        if "time" in bundle.meta.columns:
            nat = int(pd.to_datetime(bundle.meta["time"], errors="coerce").isna().sum())
            stats["meta.time"] = {"nat": nat, "missing_fraction": round(nat / max(bundle.n_samples, 1), 6)}
        total_cells = bundle.X.size + bundle.Y.size
        total_bad = sum(v.get("nan", 0) + v.get("inf", 0) for v in stats.values()
                        if isinstance(v, dict))
        checks["missing"] = {
            "per_column": stats,
            "total_missing_fraction": round(total_bad / max(total_cells, 1), 6),
            "distribution": _missing_distribution(bundle),
        }
        if total_bad:
            warnings.append(f"存在缺失值/非有限值：{total_bad} 个单元（默认不插补）")

    if "nan_inf" in checks_wanted:
        item = {
            "x_nan": int(np.isnan(bundle.X).sum()), "x_inf": int(np.isinf(bundle.X).sum()),
            "y_nan": int(np.isnan(bundle.Y).sum()), "y_inf": int(np.isinf(bundle.Y).sum()),
        }
        item["passed"] = all(v == 0 for v in item.values() if isinstance(v, int))
        checks["nan_inf"] = item
        if not item["passed"]:
            warnings.append(f"NaN/Inf 检查未通过: {item}")

    if "extreme" in checks_wanted:
        extreme = {"Y": _extreme_check(bundle.Y, bundle.target_names, "Y.")}
        extreme["X"] = _extreme_check(bundle.X, bundle.feature_names, "X.")
        extreme["total_extreme"] = int(extreme["Y"]["total_extreme"] + extreme["X"]["total_extreme"])
        checks["extreme"] = extreme

    if "range" in checks_wanted:
        checks["range"] = _range_check(bundle, q_cfg.get("ranges") or {})
        if not checks["range"]["passed"]:
            warnings.append("配置的变量范围被突破（见 checks.range）")
        if checks["range"].get("physical_defaults_skipped"):
            warnings.append("；".join(checks["range"]["physical_defaults_skipped"]))

    if "dimensions" in checks_wanted:
        problems = validate_bundle_dims(bundle)
        checks["dimensions"] = {
            "shape_X": list(bundle.X.shape), "shape_Y": list(bundle.Y.shape),
            "dtype_X": str(bundle.X.dtype), "dtype_Y": str(bundle.Y.dtype),
            "n_meta_columns": int(len(bundle.meta.columns)),
            "problems": problems, "passed": not problems,
        }
        if problems:
            warnings.append(f"维度检查问题: {problems}")

    if "units" in checks_wanted:
        checks["units"] = _units_check(bundle)
        if not checks["units"]["passed"]:
            warnings.append("单位与登记表不一致（见 checks.units.mismatches），未自动换算")
        if checks["units"]["units_unknown"]:
            warnings.append(
                f"以下变量数据源未提供单位，无法校验: {checks['units']['units_unknown']}"
            )

    # ---- v2.0 可选深化检查 ----
    if "time" in checks_wanted:
        checks["time"] = _time_check(bundle)
        if not checks["time"].get("passed", True):
            warnings.append("时间轴检查未通过: " + "；".join(checks["time"].get("problems", [])))

    if "geo" in checks_wanted:
        checks["geo"] = _geo_check(bundle)
        if not checks["geo"].get("passed", True):
            warnings.append("经纬度检查未通过: " + "；".join(checks["geo"].get("problems", [])))

    if "levels" in checks_wanted:
        checks["levels"] = _level_check(bundle)
        if not checks["levels"].get("passed", True):
            warnings.append("气压层检查未通过: " + "；".join(checks["levels"].get("problems", [])))

    if "source_labels" in checks_wanted:
        checks["source_labels"] = _source_check(bundle)
        if not checks["source_labels"].get("passed", True):
            warnings.append("来源标签未解析: " + "；".join(checks["source_labels"].get("problems", [])))

    passed = not warnings
    return {
        "summary": {
            "n_samples": bundle.n_samples,
            "n_features": bundle.n_features,
            "n_targets": bundle.n_targets,
            "source_counts": bundle.source_counts(),
            "source_file": bundle.source_file,
            "source_origin": bundle.source_origin or bundle._meta_source_origin(),
            "report_only": bool(q_cfg.get("report_only", True)),
            "passed": passed,
            "n_warnings": len(warnings),
        },
        "checks": checks,
        "cleaning": {"report_only": bool(q_cfg.get("report_only", True)), "applied": []},
        "warnings": warnings,
    }


def format_quality_report(report: dict[str, Any]) -> str:
    """报告 → Markdown（给 inspect 输出与实验目录留档）。"""
    s = report["summary"]
    lines = [
        "# 数据质量报告",
        "",
        f"- 样本数: {s['n_samples']}  特征: {s['n_features']}  目标: {s['n_targets']}",
        f"- 来源分布: {s['source_counts']}",
        f"- 数据文件: {s.get('source_file') or '—'}",
        f"- 模式: {'只读报告（不修改数据）' if s['report_only'] else '允许配置的清洗'}",
        f"- 结论: {'通过' if s['passed'] else '存在 ' + str(s['n_warnings']) + ' 条警告'}",
        "",
    ]
    for name, block in report.get("checks", {}).items():
        lines.append(f"## {name}")
        lines.append("```json")
        import json

        lines.append(json.dumps(block, ensure_ascii=False, indent=2, default=str))
        lines.append("```")
        lines.append("")
    if report.get("warnings"):
        lines.append("## 警告")
        lines.extend(f"- {w}" for w in report["warnings"])
    if report.get("cleaning", {}).get("applied"):
        lines.append("## 已应用清洗（可追踪）")
        lines.extend(f"- {a}" for a in report["cleaning"]["applied"])
    return "\n".join(lines) + "\n"


def apply_cleaning(bundle: SampleBundle, data_cfg: dict[str, Any]) -> tuple[SampleBundle, list[dict[str, Any]]]:
    """显式清洗（仅当 report_only=false 时由 runner 调用）。

    支持动作：``dropna``（剔除含 NaN 行）、``clip``（把配置范围外的值裁到边界）。
    每个动作记录为 dict，随报告落盘，可追踪。
    """
    q_cfg = data_cfg.get("quality") or {}
    if q_cfg.get("report_only", True):
        raise RuntimeError("quality.report_only=true（默认）：拒绝清洗。显式配置 report_only=false 才允许。")
    cleaning = q_cfg.get("cleaning") or {}
    actions: list[dict[str, Any]] = []
    X, Y = bundle.X.copy(), bundle.Y.copy()
    keep = np.ones(bundle.n_samples, dtype=bool)

    if cleaning.get("dropna"):
        before = int(keep.sum())
        keep &= np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        actions.append({"action": "dropna", "dropped": before - int(keep.sum())})
        X, Y = X[keep], Y[keep]

    if cleaning.get("clip"):
        ranges = q_cfg.get("ranges") or {}
        cols = {**{n: i for i, n in enumerate(bundle.feature_names)},
                **{n: i + bundle.n_features for i, n in enumerate(bundle.target_names)}}
        flat = np.hstack([X, Y]) if X.size else np.empty((0, bundle.n_features + bundle.n_targets))
        for name, (lo, hi) in ranges.items():
            if name not in cols or flat.size == 0:
                continue
            j = cols[name]
            before = int(((flat[:, j] < float(lo)) | (flat[:, j] > float(hi))).sum())
            flat[:, j] = np.clip(flat[:, j], float(lo), float(hi))
            if before:
                actions.append({"action": "clip", "variable": name, "clipped": before,
                                "range": [float(lo), float(hi)]})
        X, Y = flat[:, :bundle.n_features], flat[:, bundle.n_features:]

    meta = bundle.meta.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    cleaned = SampleBundle(X=X, Y=Y, feature_names=bundle.feature_names,
                           target_names=bundle.target_names, meta=meta,
                           units=bundle.units, source_file=bundle.source_file,
                           source_origin=bundle.source_origin)
    return cleaned, actions
