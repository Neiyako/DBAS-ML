"""实验编排器：配置 → 数据 → 质量报告 → 划分 → 混合 → 归一化 → 训练 → 评价 → 产物。

一次 :func:`run_experiment` 产出统一结构的 outputs/experiment_XXX/：

``config.yaml / config.source.yaml / metrics.json / history.csv / report.md /
preprocessors.json / quality_report(在 logs/) / model/ / logs/ / plots/``
"""
from __future__ import annotations

import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..data import (
    apply_cleaning,
    fit_scalers,
    format_quality_report,
    load_dataset,
    run_quality_checks,
    split_indices,
)
from ..models import build_model
from ..utils.config import load_config
from ..utils.device import resolve_device
from ..utils.logging import get_logger, setup_logging
from ..utils.reproducibility import set_seed
from .manager import ExperimentState, create_record, transition
from .provenance import collect_provenance
from .tracking import (
    create_experiment_dir,
    save_config_snapshot,
    write_history,
    write_json,
    write_report,
)

__all__ = ["run_experiment", "apply_mix", "evaluate_saved_experiment"]

#: 模块级 logger（失败/中断状态记录、异常告警用）
_logger = get_logger("experiments.runner")


# ----------------------------------------------------------------------
# 仿真/真实混合（仅作用于训练集；测试集永不参与配比）
# ----------------------------------------------------------------------
def apply_mix(
    train_bundle,
    real_fraction: float | None,
    *,
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """按请求的真实数据比例抽样训练集。

    - ``real_fraction=None`` → 不做任何抽样；
    - 可用样本不足以达到请求比例时，尽量靠拢并**如实记录**实际比例（不伪造来源）。
    """
    info: dict[str, Any] = {
        "applied": real_fraction is not None,
        "requested": real_fraction,
        "notes": [],
    }
    if real_fraction is None:
        labels0 = train_bundle.source_labels.to_numpy()
        info.update({
            "n_original": train_bundle.n_samples,
            "n_real": int((labels0 == "real").sum()),
            "n_simulated": int((labels0 == "simulated").sum()),
            "actual_real_fraction": float((labels0 == "real").mean()) if train_bundle.n_samples else 0.0,
            "n_selected": train_bundle.n_samples,
            "n_dropped": 0,
        })
        return train_bundle, info

    labels = train_bundle.source_labels.to_numpy()
    real_idx = np.flatnonzero(labels == "real")
    sim_idx = np.flatnonzero(labels == "simulated")
    n = train_bundle.n_samples

    # 关键修复（审计 B1）：来源标签既非 real 也非 simulated（例如真实 ERA5
    # 没有来源契约时被标成 unknown）时，**不得**把训练集筛空——
    # 那会让 scaler 在空数组上拟合出 NaN，整场训练静默报废。
    if real_idx.size == 0 and sim_idx.size == 0:
        info.update({
            "n_original": n,
            "n_selected": n,
            "n_real": 0,
            "n_simulated": 0,
            "n_unknown": int(n),
            "actual_real_fraction": None,
            "n_dropped": 0,
        })
        info["notes"].append(
            f"来源标签中既无 real 也无 simulated（全部为 unknown），"
            f"已**保持全部 {n} 个样本不变**（不筛空、不伪造标签）；"
            f"如需按配比混合，请为数据显式声明来源标签或提供来源契约"
        )
        return train_bundle, info

    want_real = int(round(float(real_fraction) * n))
    want_sim = n - want_real

    take_real = min(want_real, len(real_idx))
    take_sim = min(want_sim, len(sim_idx))
    if take_real < want_real:  # real 不足 → 名额转给 sim
        take_sim = min(len(sim_idx), want_sim + (want_real - take_real))
    if take_sim < want_sim:  # sim 不足 → 名额转给 real
        take_real = min(len(real_idx), want_real + (want_sim - take_sim))

    rng = np.random.default_rng(seed)
    sel_real = rng.choice(real_idx, size=take_real, replace=False) if take_real else np.array([], int)
    sel_sim = rng.choice(sim_idx, size=take_sim, replace=False) if take_sim else np.array([], int)
    chosen = np.sort(np.concatenate([sel_real, sel_sim]))
    dropped = n - chosen.size
    if dropped:
        info["notes"].append(
            f"为接近配比丢弃 {dropped} 个训练样本（real 可用 {len(real_idx)}，sim 可用 {len(sim_idx)}）"
        )

    sub = train_bundle.subset(chosen)
    actual = float((sub.source_labels == "real").mean()) if sub.n_samples else 0.0
    if abs(actual - float(real_fraction)) > 1e-9:
        info["notes"].append(
            f"请求 real_fraction={real_fraction}，实际 {actual:.3f} —— 来源标签分布不满足请求，"
            "不调整、不伪造标签（如需请先接入真实数据）"
        )
    info.update({
        "n_original": n,
        "n_selected": int(chosen.size),
        "n_real": int((sub.source_labels == "real").sum()),
        "n_simulated": int((sub.source_labels == "simulated").sum()),
        "n_unknown": int((sub.source_labels == "unknown").sum()),
        "actual_real_fraction": actual,
        "n_dropped": int(dropped),
    })
    return sub, info


# ----------------------------------------------------------------------
def _torch_inverse_y(pre: Any) -> Any | None:
    """构造 **torch 版** Y 反变换（标准化空间 → 物理单位）。

    为什么必须有它：物理约束（如「云水含量非负」）必须作用在**物理量**上。
    若直接作用在 z-score 上，「低于均值」会被误判成「负云水」——那是科学错误，
    不是风格问题。

    只在 torch 模型实验里构造（sklearn/LightGBM 实验不 import torch，
    见 models/base.py 关于 OpenMP 冲突的说明）。
    """
    try:
        import torch

        state = (pre.state_dict() or {}).get("y") or {}
        method = state.get("method")
        if method == "standard" and state.get("mean") is not None:
            mean = torch.tensor(state["mean"], dtype=torch.float32)
            scale = torch.tensor(state["scale"], dtype=torch.float32)

            def _inv(y: Any) -> Any:
                return y * scale.to(y.device, y.dtype) + mean.to(y.device, y.dtype)

            return _inv
        if method == "minmax" and state.get("min") is not None:
            lo = torch.tensor(state["min"], dtype=torch.float32)
            span = torch.tensor(state["range"], dtype=torch.float32)

            def _inv(y: Any) -> Any:
                return y * span.to(y.device, y.dtype) + lo.to(y.device, y.dtype)

            return _inv
    except Exception as exc:  # pragma: no cover
        _logger.warning("构造物理单位反变换失败（物理约束将作用在标准化空间）: %s", exc)
    return None


# ----------------------------------------------------------------------
def _environment() -> dict[str, str]:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cloudwater_ml": _pkg_version("cloudwater_ml"),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        # 发行名是 scikit-learn（sklearn 只是 import 名）；两个都试，避免谎报未安装
        "scikit_learn": _pkg_version("scikit-learn"),
        "torch": _pkg_version("torch"),
        "lightgbm": _pkg_version("lightgbm"),
        "xarray": _pkg_version("xarray"),
        "netCDF4": _pkg_version("netCDF4"),
        "dask": _pkg_version("dask"),
    }
    return env


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not-installed"


def run_experiment(
    config_path: str | Path,
    outputs_root: str | Path = "outputs",
    output_name: str | None = None,
    *,
    allow_existing: bool = False,
    force_overwrite: bool = False,
) -> dict[str, Any]:
    """执行一次完整实验，返回 ``{"experiment", "dir", "metrics"}``。

    失败或被中断时，会把实验状态写成 ``FAILED`` / ``INTERRUPTED``（含异常信息），
    **不会**留下一个看起来正常但其实没跑完的实验目录。
    """
    holder: dict[str, Any] = {}
    try:
        return _run_experiment_impl(
            config_path,
            outputs_root,
            output_name,
            allow_existing=allow_existing,
            force_overwrite=force_overwrite,
            holder=holder,
        )
    except KeyboardInterrupt as exc:
        _mark_failure(holder, ExperimentState.INTERRUPTED, exc)
        raise
    except Exception as exc:
        _mark_failure(holder, ExperimentState.FAILED, exc)
        raise


def _mark_failure(holder: dict[str, Any], state: ExperimentState, exc: BaseException) -> None:
    """把失败/中断写入实验状态文件（尽力而为，不掩盖原始异常）。"""
    record = holder.get("record")
    exp_dir = holder.get("exp_dir")
    if record is None or exp_dir is None:
        return
    try:
        transition(
            record,
            state,
            exp_dir=exp_dir,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
    except Exception as inner:  # pragma: no cover - 状态写入失败不应掩盖原异常
        _logger.warning("写入实验失败状态时出错: %s", inner)


def _run_experiment_impl(
    config_path: str | Path,
    outputs_root: str | Path,
    output_name: str | None,
    *,
    allow_existing: bool,
    force_overwrite: bool = False,
    holder: dict[str, Any],
) -> dict[str, Any]:
    """实验执行体（生命周期状态在 run_experiment 的包装层统一收口）。"""
    t0 = time.time()
    started_at = datetime.now().isoformat(timespec="seconds")
    cfg = load_config(config_path)
    # 配置校验前置：非法配置在加载数据/建模之前就失败，避免跑一半才发现
    from ..utils.config_schema import validate_config

    config_warnings = validate_config(cfg)
    training_cfg = cfg.get("training") or {}
    seed = int(training_cfg.get("seed", 42))
    # 先建模型：据此决定是否加载 torch（LightGBM 实验必须不加载，见 models/base.py）
    model_cfg = cfg.get("model") or {}
    model = build_model(model_cfg.get("name"), model_cfg.get("params") or {})
    set_seed(seed, with_torch=model.is_torch)

    exp_cfg = cfg.get("experiment") or {}
    name = output_name or exp_cfg.get("output_name") or None
    exp_dir = create_experiment_dir(
        outputs_root, name, exist_ok=allow_existing, force_overwrite=force_overwrite
    )
    provenance = collect_provenance(
        config_path=config_path,
        data_path=(cfg.get("data") or {}).get("path"),
        seed=seed,
        repo_root=Path(__file__).resolve().parents[3],
    )
    record = create_record(
        exp_dir,
        config_path=config_path,
        seed=seed,
        model=model.name,
        provenance=provenance.to_dict(),
    )
    holder["record"] = record
    holder["exp_dir"] = exp_dir
    transition(record, ExperimentState.PREPARING, exp_dir=exp_dir)
    setup_logging("INFO", exp_dir / "logs/train.log")
    log = get_logger("runner")
    save_config_snapshot(cfg, exp_dir, source_config=config_path)
    notes: list[str] = []
    log.info("实验目录: %s", exp_dir)
    for w in config_warnings:
        log.warning("配置提示: %s", w)
        notes.append(f"配置提示: {w}")

    # ---- 数据 + 质量报告（默认只读） ----
    bundle = load_dataset(cfg)
    data_cfg = cfg.get("data") or {}
    quality = run_quality_checks(bundle, data_cfg)
    report_only = bool((data_cfg.get("quality") or {}).get("report_only", True))
    if not report_only:
        bundle, actions = apply_cleaning(bundle, data_cfg)
        quality = run_quality_checks(bundle, data_cfg)
        quality["cleaning"]["applied"] = actions
        if actions:
            notes.append(f"已应用显式清洗: {actions}")
    quality_path = exp_dir / "logs/quality_report.md"
    quality_path.write_text(format_quality_report(quality), encoding="utf-8")
    if quality["warnings"]:
        notes.extend(f"质量检查: {w}" for w in quality["warnings"])
    log.info("数据加载完成: %d 样本, 来源 %s", bundle.n_samples, bundle.source_counts())

    # ---- 划分（互斥强校验在 split_indices 内） ----
    splits = split_indices(bundle, cfg.get("split") or {})
    train = bundle.subset(splits.train)
    val = bundle.subset(splits.val)
    test = bundle.subset(splits.test)
    log.info("划分: %s", splits.as_dict())

    # ---- 训练集混合配比（测试集不参与） ----
    mix_cfg = cfg.get("mix") or {}
    rf = mix_cfg.get("real_fraction")
    strategy = str(mix_cfg.get("strategy", "ratio")).lower()
    if strategy != "ratio":
        # 更复杂的混合策略（按来源/按权重/分层）走统一采样器
        from ..data.sources import MixedSampler

        sampler = MixedSampler(train, seed=seed)
        train, mix_info = sampler.sample(
            real_fraction=rf,
            strategy=strategy,
            source_counts=mix_cfg.get("source_counts"),
            weights=mix_cfg.get("weights"),
            stratify_by=mix_cfg.get("stratify_by"),
            keep_scenes_intact=bool(mix_cfg.get("keep_scenes_intact", False)),
        )
        mix_info.setdefault("notes", [])
    else:
        train, mix_info = apply_mix(train, rf, seed=seed)
    if mix_info.get("notes"):
        notes.extend(f"mix: {n}" for n in mix_info["notes"])
        log.warning("mix 备注: %s", mix_info["notes"])

    # ---- 归一化（scaler 只在 train 上拟合；模型已在函数开头构建） ----
    model.feature_names = list(bundle.feature_names)
    model.target_names = list(bundle.target_names)
    pre_cfg = cfg.get("preprocessing") or {}
    pre = fit_scalers(
        train.X, train.Y,
        scale=pre_cfg.get("scale", "auto"),
        method=pre_cfg.get("method", "standard"),
        is_torch=model.is_torch,
    )
    notes.extend(pre.notes)

    # 物理约束必须作用在物理单位上 → 注入 Y 的反标准化变换（仅 torch 模型）
    if model.is_torch:
        _inv = _torch_inverse_y(pre)
        if _inv is not None:
            cfg["_physical_transform"] = _inv
        else:
            notes.append("未能构造物理单位反变换：物理约束（若启用）将作用在标准化空间")

    Xtr, Ytr = pre.transform_x(train.X), pre.transform_y(train.Y)
    Xva = pre.transform_x(val.X) if val.n_samples else None
    Yva = pre.transform_y(val.Y) if val.n_samples else None
    device = resolve_device(training_cfg.get("device", "auto")) if model.is_torch else "cpu"

    # ---- 训练 ----
    transition(record, ExperimentState.RUNNING, exp_dir=exp_dir, device=device)
    log.info("开始训练: model=%s device=%s", model.name, device)
    result = model.fit(Xtr, Ytr, X_val=Xva, Y_val=Yva, config=cfg, out_dir=exp_dir / "model")
    model.feature_names = list(bundle.feature_names)  # 防 fit 内部清空
    model.target_names = list(bundle.target_names)
    write_history(result.history, exp_dir)
    write_json(result.history, exp_dir / "logs/history.json")

    from ..visualization import (  # 延迟导入：训练完成后出图
        plot_feature_importance,
        plot_loss_curve,
        plot_prediction_vs_truth,
        plot_residual_distribution,
        plot_scatter_map,
        plot_subgroup_rmse,
    )

    plot_loss_curve(result.history, exp_dir / "plots/loss_curve.png")

    # ---- 保存模型与归一化参数 ----
    model.save(exp_dir / "model")
    write_json(pre.state_dict(), exp_dir / "preprocessors.json")

    # ---- 测试集评价（指标一律换算回原单位） ----
    transition(record, ExperimentState.VALIDATING, exp_dir=exp_dir)
    def predict_raw(X_raw: np.ndarray) -> np.ndarray:
        return pre.inverse_y(np.asarray(model.predict(pre.transform_x(X_raw)), dtype=np.float64))

    y_pred = predict_raw(test.X)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    from ..evaluation import evaluate_predictions, noise_robustness

    ev = evaluate_predictions(test.Y, y_pred, test.meta, list(bundle.target_names), cfg)

    # ---- 物理一致性报告（独立于指标） -------------------------------
    physics_report: dict[str, Any] | None = None
    if model.is_torch:
        try:
            import torch

            from ..physics.validator import PhysicsValidator, format_report_markdown

            validator = PhysicsValidator(expected_units=dict(bundle.units or {}))
            physics_report = validator.validate(
                torch.from_numpy(np.asarray(y_pred, dtype=np.float32)),
                {
                    "Y": torch.from_numpy(np.asarray(test.Y, dtype=np.float32)),
                    "target_names": list(bundle.target_names),
                    "units": dict(bundle.units or {}),
                },
            )
            report_md = exp_dir / "logs" / "physics_report.md"
            report_md.parent.mkdir(parents=True, exist_ok=True)
            report_md.write_text(format_report_markdown(physics_report), encoding="utf-8")
            notes.append(
                f"物理一致性报告: {physics_report.get('summary', {}).get('status', 'UNKNOWN')}"
                f"（{report_md.name}）"
            )
        except Exception as exc:
            notes.append(f"物理一致性校验未完成: {type(exc).__name__}: {exc}")
    else:
        notes.append("物理一致性校验仅对 PyTorch 模型执行（sklearn 风格基线未产出该报告）")

    eval_cfg = cfg.get("evaluation") or {}
    noise_levels = list(((eval_cfg.get("robustness") or {}).get("noise_levels")) or [])
    robust: dict[str, Any] = {}
    if noise_levels:
        robust = noise_robustness(
            predict_raw, test.X, test.Y, list(bundle.target_names), noise_levels, seed=seed
        )

    # ---- 不确定性（区间 + 可靠性） -----------------------------------
    unc_cfg = eval_cfg.get("uncertainty") or {}
    if unc_cfg.get("enabled"):
        from ..evaluation.uncertainty import estimate_uncertainty, reliability_report

        method = str(unc_cfg.get("method", "conformal")).lower()
        alpha = float(unc_cfg.get("alpha", 0.1))
        names = list(bundle.target_names)
        if method != "conformal":
            uncertainty = {
                "status": "NOT_IMPLEMENTED",
                "method": method,
                "note": "本轮仅实现 conformal（分割共形）区间；其它方法需自行提供采样源",
            }
            notes.append(f"uncertainty.method={method} 尚未实现，未产出区间（不伪装）")
        elif val.n_samples < 10:
            uncertainty = {
                "status": "NOT_AVAILABLE",
                "method": method,
                "note": f"校准集（验证集）样本数 {val.n_samples} 太少，无法构造可靠区间",
            }
            notes.append("uncertainty: 验证集样本不足，未产出区间")
        else:
            # 用**验证集**做校准集（测试集不参与任何拟合/校准）
            pred_val = pre.inverse_y(
                np.asarray(model.predict(pre.transform_x(val.X)), dtype=np.float64)
            )
            if pred_val.ndim == 1:
                pred_val = pred_val.reshape(-1, 1)
            u = estimate_uncertainty(
                test.Y,
                method="conformal",
                y_calib=val.Y,
                pred_calib=pred_val,
                pred_test=y_pred,
                alpha=alpha,
                target_names=names,
            )
            uncertainty = u.to_dict()
            uncertainty["reliability"] = reliability_report(
                test.Y, u, target_names=names, alpha=alpha
            )
            notes.append(
                f"uncertainty: 共形区间（名义覆盖率 {1 - alpha:.2f}，"
                f"校准样本 {val.n_samples}）"
            )
    else:
        uncertainty = {"status": "disabled"}

    # ---- 图 ----
    plot_prediction_vs_truth(test.Y, y_pred, list(bundle.target_names),
                             exp_dir / "plots/prediction_vs_truth.png")
    plot_residual_distribution(test.Y, y_pred, list(bundle.target_names),
                               exp_dir / "plots/residual_distribution.png")
    importance = None
    try:
        importance = model.feature_importance()
    except Exception as exc:  # 重要性失败不阻断实验
        notes.append(f"feature_importance 失败（不影响指标）: {exc}")
    if importance:
        plot_feature_importance(importance, exp_dir / "plots/feature_importance.png")
        write_json(importance, exp_dir / "logs/feature_importance.json")
    if ev.get("subgroups"):
        plot_subgroup_rmse(ev["subgroups"], exp_dir / "plots/subgroup_rmse.png")
    if {"latitude", "longitude"}.issubset(test.meta.columns):
        plot_scatter_map(test.meta["latitude"].to_numpy(), test.meta["longitude"].to_numpy(),
                         y_pred[:, 0], exp_dir / "plots/spatial_map_pred.png",
                         title=f"predicted {bundle.target_names[0]}")

    # ---- 汇总落盘 ----
    duration = round(time.time() - t0, 2)
    metrics: dict[str, Any] = {
        "experiment": {
            "experiment_id": record.experiment_id,
            "run_id": record.run_id,
            "name": exp_dir.name,
            "description": exp_cfg.get("description"),
            "model": model.name,
            "model_params": model.params,
            "config_path": str(Path(config_path).resolve()),
            "seed": seed,
            "device": device if model.is_torch else "cpu (sklearn-style)",
            "started_at": started_at,
            "duration_sec": duration,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "peak_vram_bytes": (result.extra or {}).get("peak_vram_bytes"),
            "peak_host_rss_bytes": (result.extra or {}).get("peak_host_rss_bytes"),
        },
        "environment": _environment(),
        "provenance": provenance.to_dict(),
        "runtime_plan": (result.extra or {}).get("runtime_plan"),
        "data": {
            "path": data_cfg.get("path"),
            "adapter": data_cfg.get("adapter"),
            "features": list(bundle.feature_names),
            "source_counts": bundle.source_counts(),
            "n_samples_total": bundle.n_samples,
            "quality_report": str(quality_path),
            "quality_passed": quality["summary"]["passed"],
        },
        "targets": list(bundle.target_names),
        "split": splits.as_dict(),
        "mix": mix_info,
        "preprocessing": pre.state_dict(),
        "train": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "stopped_early": result.stopped_early,
            "n_epochs": len(result.history),
            "final_train_loss": result.final_train_loss,
            "extra": result.extra,
        },
        "test": {
            "n_samples": int(test.n_samples),
            "metrics": ev.get("metrics"),
            "subgroups": ev.get("subgroups"),
        },
        "robustness": robust,
        "uncertainty": uncertainty,
        "physics_report": physics_report,
        "notes": notes,
    }
    write_json(metrics, exp_dir / "metrics.json")
    write_report(exp_dir, metrics)
    overall = ((metrics.get("test") or {}).get("metrics") or {}).get("overall") or {}
    transition(
        record,
        ExperimentState.COMPLETED,
        exp_dir=exp_dir,
        duration_sec=duration,
        peak_vram_bytes=(result.extra or {}).get("peak_vram_bytes"),
        peak_host_rss_bytes=(result.extra or {}).get("peak_host_rss_bytes"),
        metrics_summary=overall,
        device=device if model.is_torch else "cpu (sklearn-style)",
    )
    log.info("实验完成: %s（%.1fs）→ %s", exp_dir.name, duration, exp_dir)
    return {
        "experiment": exp_dir.name,
        "experiment_id": record.experiment_id,
        "dir": str(exp_dir),
        "metrics": metrics,
    }


# ----------------------------------------------------------------------
def evaluate_saved_experiment(
    experiment: str | Path,
    outputs_root: str | Path = "outputs",
) -> dict[str, Any]:
    """评估已保存模型：读 config 快照重载数据与划分，重算测试集指标。

    不重新训练、不改写 metrics.json（结果写 evaluation.json），保证原始记录不被覆盖。
    """
    exp_dir = Path(experiment)
    if not exp_dir.exists():
        exp_dir = Path(outputs_root) / str(experiment)
    if not exp_dir.exists():
        raise FileNotFoundError(f"找不到实验目录: {experiment}")

    cfg = load_config(exp_dir / "config.yaml")
    from ..data.preprocessing import MinMaxScaler, NoScaler, StandardScaler
    from ..evaluation import evaluate_predictions
    from ..models import load_model

    bundle = load_dataset(cfg)
    splits = split_indices(bundle, cfg.get("split") or {})
    test = bundle.subset(splits.test)

    model = load_model(exp_dir / "model")
    pre_path = exp_dir / "preprocessors.json"
    if pre_path.exists():
        import json

        state = json.loads(pre_path.read_text(encoding="utf-8"))
        x_cls = {"standard": StandardScaler, "minmax": MinMaxScaler, "none": NoScaler}
        x_scaler = x_cls.get((state.get("x") or {}).get("method", "none"), NoScaler).from_state_dict(state["x"])
        y_scaler = x_cls.get((state.get("y") or {}).get("method", "none"), NoScaler).from_state_dict(state["y"])
    else:
        x_scaler = y_scaler = NoScaler()

    y_pred = np.asarray(model.predict(x_scaler.transform(test.X)), dtype=np.float64)
    y_pred = y_scaler.inverse_transform(y_pred)
    ev = evaluate_predictions(test.Y, y_pred, test.meta, list(bundle.target_names), cfg)
    payload = {
        "experiment": exp_dir.name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_test": int(test.n_samples),
        **ev,
    }
    write_json(payload, exp_dir / "evaluation.json")
    return payload
