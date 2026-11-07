"""配置校验（v2.0）：把「配置写错」挡在训练开始之前。

设计要点
--------

* **错误 vs 警告分级**：取值非法（不存在的 adapter、比例和 ≠ 1、epochs ≤ 0）
  是 *错误*，直接抛 :class:`ConfigurationError` 并列出全部问题；可疑但可用
  （文件不存在但可自举、未知顶层键）是 *警告*，记录进实验 notes。
* **不猜、不改**：校验器只报告问题，**不修改**用户配置。
* **向后兼容**：v1.0 的配置段全部继续有效；v2.0 新增分段（``hardware`` /
  ``dataloader``）出现时才启用对应规则。

对外入口::

    problems = validate_config(cfg)          # 抛异常或返回警告列表
    report = describe_config(cfg)             # 人类可读的结构摘要
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError

__all__ = ["validate_config", "describe_config", "KNOWN_SECTIONS"]

#: 已知顶层配置段（前缀下划线的是框架内部键，忽略）
KNOWN_SECTIONS = {
    "data",
    "model",
    "training",
    "preprocessing",
    "split",
    "mix",
    "evaluation",
    "physics",
    "experiment",
    "hardware",      # v2.0
    "dataloader",    # v2.0
    "benchmark",     # v2.0
}

_VALID_ADAPTERS = ("npz", "csv", "parquet", "netcdf")
_VALID_OPTIMIZERS = ("adam", "adamw", "sgd")
_VALID_LOSSES = ("mse", "mae", "huber")
_VALID_DEVICES = ("auto", "cpu", "cuda", "mps")
_VALID_PRECISIONS = ("auto", "fp32", "fp16", "bf16")
_VALID_SPLIT_STRATEGIES = ("random", "time", "year", "region", "source")
#: 物理约束名（与 cloudwater_ml.physics.build_constraint 的可识别名一致）
_VALID_CONSTRAINTS = (
    "nonnegativity",
    "nonnegativity_clamp",
    "nonnegativity_softplus",
    "unit_consistency",
    "cwp_consistency",
    "cwp_integral_consistency",
    "water_budget",
    "vertical_profile_structure",
    "atmospheric_water_cycle",
)
#: 别名 → 规范名
_CONSTRAINT_ALIASES = {
    "nonneg": "nonnegativity",
    "non_negative": "nonnegativity",
    "clamp": "nonnegativity_clamp",
    "softplus": "nonnegativity_softplus",
    "units": "unit_consistency",
    "cwp": "cwp_consistency",
    "vertical_profile": "vertical_profile_structure",
    "water_cycle": "atmospheric_water_cycle",
}
_VALID_SCALERS = ("standard", "minmax", "none")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _check_positive(
    problems: list[str], section: str, key: str, value: Any, *, allow_zero: bool = False
) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        problems.append(f"{section}.{key} 必须是数字，当前为 {value!r}")
        return
    if v < 0 or (v == 0 and not allow_zero):
        problems.append(f"{section}.{key} 必须为正数，当前为 {value!r}")


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """校验整份实验配置。

    Returns
    -------
    list[str]
        警告列表（不致命但值得记录）。

    Raises
    ------
    ConfigurationError
        存在致命配置错误时抛出，``context["problems"]`` 携带全部问题清单。
    """
    problems: list[str] = []
    warnings: list[str] = []

    if not isinstance(cfg, dict):
        raise ConfigurationError(f"配置必须是映射(dict)，当前为 {type(cfg).__name__}")

    # -- 顶层段 ------------------------------------------------------
    unknown = [
        k for k in cfg if not k.startswith("_") and k not in KNOWN_SECTIONS
    ]
    if unknown:
        warnings.append(f"未知顶层配置段（已忽略）: {', '.join(sorted(unknown))}")

    # -- data --------------------------------------------------------
    data = cfg.get("data") or {}
    if data:
        adapter = str(data.get("adapter") or "").lower()
        if not adapter:
            problems.append("缺少 data.adapter（可选: " + " | ".join(_VALID_ADAPTERS) + "）")
        elif adapter not in _VALID_ADAPTERS:
            problems.append(
                f"data.adapter 非法: {adapter!r}（可选: " + " | ".join(_VALID_ADAPTERS) + "）"
            )
        feats = _as_list(data.get("features"))
        if not feats:
            problems.append("缺少 data.features（输入特征列表）")
        elif not all(isinstance(f, str) for f in feats):
            problems.append("data.features 必须是字符串列表")
        tgts = _as_list(data.get("targets"))
        if not tgts:
            problems.append("缺少 data.targets（预测目标列表）")
        elif not all(isinstance(t, str) for t in tgts):
            problems.append("data.targets 必须是字符串列表")
        path = data.get("path")
        if path:
            p = Path(str(path))
            if not p.exists() and "synthetic" not in p.as_posix():
                warnings.append(f"data.path 不存在: {p}")
        src_label = data.get("source_label")
        if src_label is not None and str(src_label).lower() not in ("real", "simulated"):
            problems.append(f"data.source_label 必须是 real 或 simulated，当前 {src_label!r}")
        nc = data.get("netcdf") or {}
        for rng_key in ("time_range", "latitude_range", "longitude_range", "level_range"):
            val = nc.get(rng_key)
            if val is not None:
                if not isinstance(val, (list, tuple)) or len(val) != 2:
                    problems.append(f"data.netcdf.{rng_key} 必须是 [min, max] 两元素序列")
                elif val[0] > val[1]:
                    problems.append(f"data.netcdf.{rng_key} 的下限大于上限: {val!r}")
        chunks = nc.get("chunks")
        if chunks is not None and not isinstance(chunks, dict):
            problems.append("data.netcdf.chunks 必须是映射（如 {time: 16}）")

    # -- model --------------------------------------------------------
    model = cfg.get("model") or {}
    if model:
        name = model.get("name")
        if not name:
            problems.append("缺少 model.name")
        params = model.get("params") or {}
        if not isinstance(params, dict):
            problems.append("model.params 必须是映射")

    # -- training -----------------------------------------------------
    training = cfg.get("training") or {}
    if training:
        if "epochs" in training:
            _check_positive(problems, "training", "epochs", training["epochs"])
        if "batch_size" in training:
            _check_positive(problems, "training", "batch_size", training["batch_size"])
        if "learning_rate" in training:
            _check_positive(problems, "training", "learning_rate", training["learning_rate"])
        if "weight_decay" in training:
            _check_positive(problems, "training", "weight_decay", training["weight_decay"], allow_zero=True)
        opt = training.get("optimizer")
        if opt and str(opt).lower() not in _VALID_OPTIMIZERS:
            problems.append(
                f"training.optimizer 非法: {opt!r}（可选: " + " | ".join(_VALID_OPTIMIZERS) + "）"
            )
        loss = training.get("loss")
        if loss and str(loss).lower() not in _VALID_LOSSES:
            problems.append(
                f"training.loss 非法: {loss!r}（可选: " + " | ".join(_VALID_LOSSES) + "）"
            )
        dev = training.get("device")
        if dev and str(dev).lower() not in _VALID_DEVICES:
            problems.append(
                f"training.device 非法: {dev!r}（可选: " + " | ".join(_VALID_DEVICES) + "）"
            )
        if "num_workers" in training and training["num_workers"] is not None:
            nw = training["num_workers"]
            if not (isinstance(nw, str) and nw.strip().lower() == "auto"):
                try:
                    if int(nw) < 0:
                        problems.append("training.num_workers 不能为负数")
                except (TypeError, ValueError):
                    problems.append(f"training.num_workers 非法: {nw!r}（可用整数或 'auto'）")
        accum = training.get("gradient_accumulation_steps")
        if accum is not None:
            try:
                if int(accum) < 1:
                    problems.append("training.gradient_accumulation_steps 必须 ≥ 1")
            except (TypeError, ValueError):
                problems.append(f"training.gradient_accumulation_steps 非法: {accum!r}")
        clip = training.get("gradient_clip_norm", training.get("grad_clip"))
        if clip not in (None, "", "null"):
            try:
                if float(clip) <= 0:
                    problems.append("training.gradient_clip_norm 必须为正数")
            except (TypeError, ValueError):
                problems.append(f"training.gradient_clip_norm 非法: {clip!r}")
        sched = training.get("scheduler")
        if isinstance(sched, str) and sched.lower() not in ("null", "none", "cosine", "step"):
            problems.append(f"training.scheduler 非法: {sched!r}（可选: null | cosine | step）")

    # -- hardware / dataloader (v2.0) ----------------------------------
    hw = cfg.get("hardware") or {}
    if hw:
        dev = hw.get("device")
        if dev and str(dev).lower() not in _VALID_DEVICES:
            problems.append(
                f"hardware.device 非法: {dev!r}（可选: " + " | ".join(_VALID_DEVICES) + "）"
            )
        prec = hw.get("precision")
        if prec and str(prec).lower() not in _VALID_PRECISIONS:
            problems.append(
                f"hardware.precision 非法: {prec!r}（可选: " + " | ".join(_VALID_PRECISIONS) + "）"
            )
        cl = hw.get("channels_last")
        if cl is not None and not isinstance(cl, (bool, str)):
            problems.append("hardware.channels_last 必须是 bool 或 'auto'")
    dl = cfg.get("dataloader") or {}
    if dl:
        nw = dl.get("num_workers")
        if nw is not None and not (isinstance(nw, str) and nw.lower() == "auto"):
            try:
                if int(nw) < 0:
                    problems.append("dataloader.num_workers 不能为负数")
            except (TypeError, ValueError):
                problems.append(f"dataloader.num_workers 非法: {nw!r}")
        pf = dl.get("prefetch_factor")
        if pf is not None:
            try:
                if int(pf) < 1:
                    problems.append("dataloader.prefetch_factor 必须 ≥ 1")
            except (TypeError, ValueError):
                problems.append(f"dataloader.prefetch_factor 非法: {pf!r}")

    # -- preprocessing -------------------------------------------------
    pre = cfg.get("preprocessing") or {}
    if pre:
        method = pre.get("method")
        if method and str(method).lower() not in _VALID_SCALERS:
            problems.append(
                f"preprocessing.method 非法: {method!r}（可选: " + " | ".join(_VALID_SCALERS) + "）"
            )

    # -- split ----------------------------------------------------------
    split = cfg.get("split") or {}
    if split:
        strat = split.get("strategy")
        if strat and str(strat).lower() not in _VALID_SPLIT_STRATEGIES:
            problems.append(
                f"split.strategy 非法: {strat!r}（可选: " + " | ".join(_VALID_SPLIT_STRATEGIES) + "）"
            )
        ratios = split.get("ratios")
        if ratios is not None:
            vals = _as_list(ratios)
            if len(vals) != 3:
                problems.append(f"split.ratios 必须是 3 个数字（train/val/test），当前 {vals!r}")
            else:
                try:
                    total = float(sum(float(v) for v in vals))
                except (TypeError, ValueError):
                    problems.append(f"split.ratios 含非数字: {vals!r}")
                else:
                    if abs(total - 1.0) > 1e-6:
                        problems.append(f"split.ratios 之和必须为 1.0，当前 {total}")
                    if any(float(v) <= 0 for v in vals):
                        problems.append(f"split.ratios 各分量必须为正数: {vals!r}")

    # -- mix -------------------------------------------------------------
    mix = cfg.get("mix") or {}
    if mix and mix.get("real_fraction") is not None:
        try:
            rf = float(mix["real_fraction"])
        except (TypeError, ValueError):
            problems.append(f"mix.real_fraction 非法: {mix['real_fraction']!r}")
        else:
            if not (0.0 <= rf <= 1.0):
                problems.append(f"mix.real_fraction 必须在 [0, 1]，当前 {rf}")

    # -- evaluation ------------------------------------------------------
    ev = cfg.get("evaluation") or {}
    if ev:
        metrics = _as_list(ev.get("metrics"))
        known_metrics = {"mae", "mse", "rmse", "r2", "bias", "nse", "mape", "pearson", "spearman"}
        unknown = [m for m in metrics if str(m).lower() not in known_metrics]
        if unknown:
            warnings.append(f"evaluation.metrics 含未知指标: {', '.join(map(str, unknown))}")
        noise = (ev.get("robustness") or {}).get("noise_levels")
        if noise is not None:
            try:
                if any(float(x) < 0 for x in _as_list(noise)):
                    problems.append("evaluation.robustness.noise_levels 不能为负")
            except (TypeError, ValueError):
                problems.append("evaluation.robustness.noise_levels 含非数字")

    # -- physics ---------------------------------------------------------
    phys = cfg.get("physics") or {}
    if phys:
        cons = _as_list(phys.get("constraints"))
        for c in cons:
            key = str(c).lower()
            canonical = _CONSTRAINT_ALIASES.get(key, key)
            if canonical not in _VALID_CONSTRAINTS:
                problems.append(
                    f"physics.constraints 含未知约束: {c!r}"
                    f"（可选: " + " | ".join(_VALID_CONSTRAINTS) + "）"
                )
        w = phys.get("nonnegativity_weight")
        if w is not None:
            try:
                if float(w) < 0:
                    problems.append("physics.nonnegativity_weight 不能为负")
            except (TypeError, ValueError):
                problems.append(f"physics.nonnegativity_weight 非法: {w!r}")
        weights = phys.get("weights") or {}
        if not isinstance(weights, dict):
            problems.append("physics.weights 必须是映射（{约束名: 权重}）")
        else:
            for name, val in weights.items():
                try:
                    if float(val) < 0:
                        problems.append(f"physics.weights[{name!r}] 不能为负")
                except (TypeError, ValueError):
                    problems.append(f"physics.weights[{name!r}] 非法: {val!r}")

    # -- 汇总 -------------------------------------------------------------
    if problems:
        raise ConfigurationError(
            "配置校验未通过：" + "；".join(problems),
            hint="修正上述问题后重试（配置不会被自动改写）",
            context={"problems": problems, "warnings": warnings},
        )
    return warnings


def describe_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """结构化摘要（写进实验记录 / CLI 展示）。"""
    training = cfg.get("training") or {}
    hardware = cfg.get("hardware") or {}
    data = cfg.get("data") or {}
    split = cfg.get("split") or {}
    model = cfg.get("model") or {}
    return {
        "model": model.get("name"),
        "model_params": model.get("params") or {},
        "adapter": data.get("adapter"),
        "data_path": data.get("path"),
        "features": list(data.get("features") or []),
        "targets": list(data.get("targets") or []),
        "epochs": training.get("epochs"),
        "batch_size": training.get("batch_size"),
        "learning_rate": training.get("learning_rate"),
        "optimizer": training.get("optimizer"),
        "loss": training.get("loss"),
        "seed": training.get("seed"),
        "device_requested": hardware.get("device", training.get("device", "auto")),
        "precision_requested": hardware.get("precision", "auto"),
        "split_strategy": split.get("strategy"),
        "split_ratios": split.get("ratios"),
        "mix_real_fraction": (cfg.get("mix") or {}).get("real_fraction"),
    }
