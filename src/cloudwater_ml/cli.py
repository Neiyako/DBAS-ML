"""cloudwater 命令行入口。

命令一览::

    cloudwater doctor                      # 环境体检（依赖/设备/目录/配置）
    cloudwater gpu info                    # GPU/CUDA/MPS 详情
    cloudwater models list                 # 模型注册表
    cloudwater inspect  --config configs/data.yaml          # 检查数据（只读）
    cloudwater data validate --config configs/data.yaml     # 数据质量报告
    cloudwater train    --config configs/experiments/mlp_baseline.yaml
    cloudwater evaluate --experiment experiment_001         # 评估已保存模型
    cloudwater report   --experiment experiment_001         # 查看实验报告
    cloudwater synth    --out data/synthetic/cloudwater_synth.npz   # 生成合成测试数据
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    name="cloudwater",
    help="Cloud Water ML Framework — 云水资源机器学习实验框架 CLI",
    no_args_is_help=True,
    add_completion=False,
)

# 子命令组
gpu_app = typer.Typer(help="GPU / 加速器信息与基准", no_args_is_help=True)
data_app = typer.Typer(help="数据检查与质量报告", no_args_is_help=True)
models_app = typer.Typer(help="模型注册表", no_args_is_help=True)
cache_app = typer.Typer(help="数据缓存运维（可清理/可重建/可追溯）", no_args_is_help=True)
app.add_typer(gpu_app, name="gpu")
app.add_typer(data_app, name="data")
app.add_typer(models_app, name="models")
app.add_typer(cache_app, name="cache")


def _fail(msg: str, code: int = 1) -> None:
    typer.secho(f"错误: {msg}", fg="red", err=True)
    raise typer.Exit(code=code)


def _handle(exc: Exception, debug: bool = False) -> None:
    """统一异常出口：框架异常按 exit_code 退出并给出 hint / context。

    - 配置/用法类错误 → 退出码 2；
    - 运行期失败 → 退出码 1；
    - 非框架异常在 ``--debug`` 下保留原始 traceback，便于定位。
    """
    from cloudwater_ml.utils.exceptions import CloudWaterError, exception_to_dict

    if isinstance(exc, CloudWaterError):
        payload = exception_to_dict(exc)
        typer.secho(f"错误: {payload['message']}", fg="red", err=True)
        if payload.get("hint"):
            typer.secho(f"  建议: {payload['hint']}", fg="yellow", err=True)
        ctx = payload.get("context") or {}
        if ctx:
            typer.secho(f"  上下文: {ctx}", fg="yellow", err=True)
        raise typer.Exit(code=int(payload.get("exit_code", 1)))
    if debug:
        raise exc
    typer.secho(f"错误: {type(exc).__name__}: {exc}", fg="red", err=True)
    raise typer.Exit(code=1)


def _load_cfg(config: Path) -> dict[str, Any]:
    from cloudwater_ml.utils.config import load_config

    if not Path(config).exists():
        _fail(f"配置文件不存在: {config}")
    try:
        return load_config(config)
    except Exception as exc:  # 配置错误给出干净提示
        _fail(f"配置解析失败: {exc}")
    return {}


@app.command()
def inspect(
    config: Path = typer.Option(..., "--config", "-c", help="数据配置 YAML（如 configs/data.yaml）"),
    quality: bool | None = typer.Option(
        None, "--quality/--no-quality",
        help="是否加载数据值做质量检查；默认 npz/csv 执行、netcdf 跳过（大文件按需读取保护）",
    ),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出"),
    out: Path | None = typer.Option(None, "--out", help="把质量报告 Markdown 写入该文件"),
) -> None:
    """检查数据：结构 / 变量 / 维度 / 单位 / 来源契约 + 数据质量报告（只读，不修改数据）。"""
    from cloudwater_ml.data import (
        format_quality_report,
        inspect_data,
        load_dataset,
        run_quality_checks,
    )

    cfg = _load_cfg(config)
    try:
        info = inspect_data(cfg)
    except Exception as exc:
        _fail(f"数据检查失败: {exc}")
        return

    data_cfg = cfg.get("data") if "data" in cfg else cfg
    adapter = (data_cfg or {}).get("adapter")
    want_quality = quality if quality is not None else adapter != "netcdf"
    quality_block: dict[str, Any] | None = None
    if want_quality:
        try:
            bundle = load_dataset(cfg)
            quality_block = run_quality_checks(bundle, data_cfg or {})
        except Exception as exc:  # 大文件守卫等情况：如实说明未执行
            quality_block = {"status": "NOT_RUN", "reason": str(exc)}
    else:
        quality_block = {
            "status": "SKIPPED",
            "reason": "netcdf 默认不加载数值（按需读取）；加 --quality 显式执行（受内存守卫保护）",
        }

    payload = {"structure": info, "quality": quality_block}
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        typer.echo("== 数据结构（只读） ==")
        typer.echo(json.dumps(info, ensure_ascii=False, indent=2, default=str))
        typer.echo("\n== 数据质量 ==")
        if quality_block and "summary" in quality_block:
            s = quality_block["summary"]
            typer.echo(f"样本 {s['n_samples']} · 来源 {s['source_counts']} · "
                       f"通过 {'是' if s['passed'] else '否'}（警告 {s['n_warnings']} 条）")
            for w in quality_block.get("warnings", []):
                typer.echo(f"  - ⚠ {w}")
        else:
            typer.echo(f"  {quality_block.get('status')}: {quality_block.get('reason')}")
    if out and quality_block and "summary" in quality_block:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(format_quality_report(quality_block), encoding="utf-8")
        typer.echo(f"质量报告已写入: {out}")


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", help="实验配置 YAML"),
    outputs: Path = typer.Option("outputs", "--outputs", help="实验输出根目录"),
    name: str | None = typer.Option(None, "--name", help="指定实验目录名（默认自动 experiment_XXX）"),
) -> None:
    """运行一次实验：数据 → 划分 → 训练 → 评价 → outputs/experiment_XXX。"""
    from cloudwater_ml.experiments.runner import run_experiment

    if not Path(config).exists():
        _fail(f"配置文件不存在: {config}")
    try:
        result = run_experiment(config, outputs_root=outputs, output_name=name)
    except Exception as exc:
        _fail(f"实验失败: {exc}")
        return
    m = result["metrics"]
    overall = ((m.get("test") or {}).get("metrics") or {}).get("overall") or {}
    typer.secho(f"✓ 实验完成: {result['experiment']} → {result['dir']}", fg="green")
    typer.echo(f"  测试集: {overall}")
    typer.echo(f"  报告: {Path(result['dir']) / 'report.md'}")


@app.command()
def evaluate(
    experiment: str = typer.Option(..., "--experiment", "-e", help="实验名或目录（如 experiment_001）"),
    outputs: Path = typer.Option("outputs", "--outputs", help="实验输出根目录"),
) -> None:
    """评估已保存模型（不重训，结果写 evaluation.json，不覆盖原 metrics.json）。"""
    from cloudwater_ml.experiments.runner import evaluate_saved_experiment

    try:
        payload = evaluate_saved_experiment(experiment, outputs_root=outputs)
    except Exception as exc:
        _fail(f"评估失败: {exc}")
        return
    typer.secho(f"✓ 评估完成: {payload['experiment']}（测试样本 {payload['n_test']}）", fg="green")
    typer.echo(json.dumps(payload.get("metrics"), ensure_ascii=False, indent=2, default=str))


@app.command()
def report(
    experiment: str | None = typer.Option(
        None, "--experiment", "-e", help="实验名或目录；省略则输出全部实验横向对比表"
    ),
    outputs: Path = typer.Option("outputs", "--outputs", help="实验输出根目录"),
) -> None:
    """查看实验报告；不带 --experiment 时汇总 outputs/ 下全部实验的指标对比表。"""
    from cloudwater_ml.experiments.tracking import build_comparison_table

    if experiment is None:
        table = build_comparison_table(outputs)
        if table.empty:
            typer.echo(f"{outputs}/ 下暂无可对比的实验（缺少 metrics.json）")
            return
        typer.echo("== 实验横向对比（测试集 overall 指标）==")
        typer.echo(table.to_string(index=False))
        return
    exp_dir = Path(experiment)
    if not exp_dir.exists():
        exp_dir = Path(outputs) / experiment
    md = exp_dir / "report.md"
    if not md.exists():
        _fail(f"找不到报告: {md}")
    typer.echo(md.read_text(encoding="utf-8"))


@app.command()
def synth(
    out: Path = typer.Option("data/synthetic/cloudwater_synth.npz", "--out", help="输出 npz 路径"),
    n: int = typer.Option(5000, "--n", help="样本数"),
    seed: int = typer.Option(42, "--seed", help="随机种子（可复现）"),
    force: bool = typer.Option(False, "--force", help="已存在则重建"),
) -> None:
    """生成合成测试数据（source 恒为 simulated，仅供工程链路验证，非真实观测）。"""
    from cloudwater_ml.data import ensure_synthetic_dataset

    path = ensure_synthetic_dataset(out, n_samples=n, seed=seed, force=force)
    typer.secho(f"✓ 合成数据: {path}（n={n}, seed={seed}, source=simulated）", fg="green")


# ----------------------------------------------------------------------
# doctor：环境体检
# ----------------------------------------------------------------------
@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出"),
    check_data: bool = typer.Option(True, "--data/--no-data", help="是否检查数据目录与配置"),
) -> None:
    """环境体检：依赖版本、计算设备、数据目录、实验配置、已注册模型。"""
    from importlib.metadata import PackageNotFoundError, version

    from cloudwater_ml.hardware import detect_hardware

    packages = [
        "numpy", "pandas", "scikit-learn", "torch", "xarray", "netCDF4",
        "matplotlib", "typer", "PyYAML", "lightgbm", "pyarrow", "dask", "scipy",
    ]
    deps: dict[str, Any] = {}
    missing: list[str] = []
    for name in packages:
        try:
            deps[name] = version(name)
        except PackageNotFoundError:
            deps[name] = None
            missing.append(name)

    info = detect_hardware("auto")
    root = Path.cwd()
    data_dirs = {
        "data/raw": (root / "data" / "raw"),
        "data/processed": (root / "data" / "processed"),
        "data/synthetic": (root / "data" / "synthetic"),
        "outputs": (root / "outputs"),
    }
    dir_status = {
        name: {
            "exists": p.exists(),
            "writable": p.exists() and _is_writable(p),
        }
        for name, p in data_dirs.items()
    }
    configs = sorted((root / "configs" / "experiments").glob("*.yaml")) if check_data else []
    try:
        from cloudwater_ml.models import available_models

        models = available_models()
    except Exception:  # pragma: no cover
        models = []

    payload = {
        "python": __import__("sys").version.split()[0],
        "packages": deps,
        "missing_optional": missing,
        "device": info.to_dict(),
        "device_summary": info.summary(),
        "directories": dir_status,
        "experiment_configs": [str(p) for p in configs],
        "registered_models": models,
    }
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return

    typer.echo("== 依赖 ==")
    for name, ver in deps.items():
        mark = "✓" if ver else "·"
        typer.echo(f"  {mark} {name:14s} {ver or '未安装（可选）'}")
    typer.echo("")
    typer.echo("== 计算设备 ==")
    typer.echo(f"  {info.summary()}")
    if info.fallback_reason:
        typer.secho(f"  回退原因: {info.fallback_reason}", fg="yellow")
    for w in info.warnings:
        typer.secho(f"  ⚠ {w}", fg="yellow")
    typer.echo("")
    typer.echo("== 目录 ==")
    for name, st in dir_status.items():
        state = "可写" if st["writable"] else ("存在但不可写" if st["exists"] else "不存在")
        typer.echo(f"  {name:16s} {state}")
    typer.echo("")
    typer.echo(f"== 已注册模型 ({len(models)}) ==")
    for m in models:
        typer.echo(f"  · {m}")
    if configs:
        typer.echo("")
        typer.echo(f"== 实验配置 ({len(configs)}) ==")
        for c in configs:
            typer.echo(f"  · {c}")


def _is_writable(path: Path) -> bool:
    import os

    return os.access(path, os.W_OK)


# ----------------------------------------------------------------------
# gpu info
# ----------------------------------------------------------------------
@gpu_app.command("info")
def gpu_info(
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """显示 GPU / CUDA / MPS 的运行时详情（名称、显存、算力、驱动版本）。"""
    from cloudwater_ml.hardware import detect_hardware, device_report

    info = detect_hardware("auto")
    if as_json:
        typer.echo(json.dumps(info.to_dict(), ensure_ascii=False, indent=2, default=str))
        return
    typer.echo(device_report(info))


# ----------------------------------------------------------------------
# models list
# ----------------------------------------------------------------------
@models_app.command("list")
def models_list(
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """列出模型注册表中的可用模型与实现状态。"""
    import importlib

    from cloudwater_ml.models import available_models
    from cloudwater_ml.models.base import _LAZY_MODULES, MODEL_METADATA

    # 导入全部惰性模块，让注册表元数据完整（避免 CLI 里硬编码一张会过期的状态表）
    import_errors: dict[str, str] = {}
    for name, module in list(_LAZY_MODULES.items()):
        try:
            importlib.import_module(module)
        except Exception as exc:
            import_errors[name] = f"{type(exc).__name__}: {exc}"

    names = available_models()
    rows = []
    for name in names:
        meta = MODEL_METADATA.get(name)
        rows.append({
            "name": name,
            "type": meta.kind if meta else "-",
            "status": meta.status if meta else "已注册",
            "summary": meta.summary if meta else "",
            "import_error": import_errors.get(name),
        })
    if as_json:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    if not rows:
        typer.echo("（注册表为空）")
        return
    width = max(len(r["name"]) for r in rows)
    typer.echo(f"{'模型'.ljust(width)}  {'类型'.ljust(14)}  状态")
    for r in rows:
        line = f"{r['name'].ljust(width)}  {r['type'].ljust(14)}  {r['status']}"
        typer.echo(line)
        if r.get("import_error"):
            typer.secho(f"{' ' * width}  └ 导入失败: {r['import_error']}", fg="yellow")


# ----------------------------------------------------------------------
# data validate
# ----------------------------------------------------------------------
@data_app.command("validate")
def data_validate(
    config: Path = typer.Option(..., "--config", "-c", help="数据配置 YAML"),
    out: Path | None = typer.Option(None, "--out", help="把质量报告 Markdown 写入该文件"),
    strict: bool = typer.Option(
        False, "--strict", help="质量检查未通过时以非零退出码结束"
    ),
) -> None:
    """生成数据质量报告（只读，不修改科研数据）。"""
    from cloudwater_ml.data import format_quality_report, load_dataset, run_quality_checks

    cfg = _load_cfg(config)
    data_cfg = cfg.get("data") if "data" in cfg else cfg
    try:
        bundle = load_dataset(cfg)
        report = run_quality_checks(bundle, data_cfg or {})
    except Exception as exc:
        _fail(f"数据质量检查失败: {exc}")
        return
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(format_quality_report(report), encoding="utf-8")
        typer.secho(f"✓ 质量报告已写入: {out}", fg="green")
    summary = report.get("summary") or {}
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    for w in report.get("warnings", []):
        typer.secho(f"  ⚠ {w}", fg="yellow")
    if strict and not summary.get("passed", False):
        raise typer.Exit(code=1)


# ----------------------------------------------------------------------
# resume：断点续训
# ----------------------------------------------------------------------
@app.command()
def resume(
    experiment: str = typer.Option(..., "--experiment", "-e", help="实验 ID 或目录名"),
    outputs: Path = typer.Option("outputs", "--outputs", help="实验输出根目录"),
    epochs: int | None = typer.Option(None, "--epochs", help="续训到的总 epoch 数（默认沿用原配置）"),
    debug: bool = typer.Option(False, "--debug", help="保留原始异常堆栈"),
) -> None:
    """从已有实验的检查点继续训练。

    **不会覆盖原实验结果**：续训会新建一个实验目录（新的实验 ID），
    并在 ``state.json`` 里记录 ``resumed_from`` 与原 ``run_id``。
    """
    from cloudwater_ml.experiments.manager import find_experiment
    from cloudwater_ml.experiments.runner import run_experiment
    from cloudwater_ml.utils.config import load_config, save_config

    try:
        exp_dir, record = find_experiment(experiment, outputs_root=outputs)
        cfg = load_config(exp_dir / "config.yaml")
        ckpt = exp_dir / "model" / "last.pt"
        if not ckpt.exists():
            _fail(f"实验 {exp_dir.name} 没有可用的检查点: {ckpt}")
        cfg.setdefault("training", {})
        cfg["training"]["resume"] = str(ckpt)
        if epochs is not None:
            cfg["training"]["epochs"] = int(epochs)
        # 新实验：继承原配置 + 续训检查点
        staging = outputs / f".resume-{exp_dir.name}.yaml"
        staging.parent.mkdir(parents=True, exist_ok=True)
        save_config(cfg, staging)
        try:
            result = run_experiment(staging, outputs_root=outputs)
        finally:
            staging.unlink(missing_ok=True)
        typer.secho(
            f"✓ 续训完成: {result['experiment']}（源实验 {exp_dir.name}，检查点 {ckpt.name}）",
            fg="green",
        )
        typer.echo(f"  报告: {Path(result['dir']) / 'report.md'}")
    except typer.Exit:
        raise
    except Exception as exc:
        _handle(exc, debug=debug)


# ----------------------------------------------------------------------
# compare：实验对比
# ----------------------------------------------------------------------
@app.command()
def compare(
    experiments: list[str] = typer.Option(..., "--experiments", "-e", help="要对比的实验 ID/目录名（可多个）"),
    outputs: Path = typer.Option("outputs", "--outputs", help="实验输出根目录"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """对比多个实验（模型、划分、测试集指标、耗时、峰值显存）。"""
    from cloudwater_ml.experiments.tracking import build_comparison_table

    try:
        table = build_comparison_table(outputs)
        wanted = {str(e) for e in experiments}
        if table.empty:
            _fail(f"{outputs}/ 下暂无可对比的实验（缺少 metrics.json）")
        mask = table["experiment"].isin(wanted)
        selected = table[mask]
        if selected.empty:
            _fail(
                f"指定的实验不在 {outputs}/ 中: {', '.join(sorted(wanted))}",
                code=2,
            )
        if as_json:
            typer.echo(selected.to_json(orient="records", force_ascii=False, indent=2))
            return
        typer.echo("== 实验对比（测试集 overall 指标）==")
        typer.echo(selected.to_string(index=False))
    except typer.Exit:
        raise
    except Exception as exc:
        _handle(exc)


# ----------------------------------------------------------------------
# benchmark gpu：性能基准
# ----------------------------------------------------------------------
@gpu_app.command("benchmark")
def gpu_benchmark(
    n_samples: int = typer.Option(8192, "--samples", help="基准样本数"),
    n_features: int = typer.Option(64, "--features", help="特征维度"),
    epochs: int = typer.Option(2, "--epochs", help="基准 epoch 数"),
    batch_size: int = typer.Option(256, "--batch-size", help="批大小"),
    device: str = typer.Option("auto", "--device", help="auto | cpu | cuda | mps"),
    precision: str = typer.Option("auto", "--precision", help="auto | fp32 | fp16 | bf16"),
    workers: str = typer.Option("auto", "--workers", help="DataLoader worker 数（auto 或整数）"),
    out: Path | None = typer.Option(None, "--out", help="把 Markdown 报告写入该文件"),
) -> None:
    """训练性能基准：吞吐、三段耗时、峰值显存 + 按实测硬件给出的配置建议。"""
    from cloudwater_ml.benchmark.runner import benchmark_training, format_report, recommend_config
    from cloudwater_ml.hardware import detect_hardware

    workers_value: Any = workers
    if workers.strip().lower() != "auto":
        try:
            workers_value = int(workers)
        except ValueError:
            _fail(f"--workers 必须是整数或 auto，当前 {workers!r}", code=2)

    try:
        res = benchmark_training(
            name="cli",
            device=device,
            precision=precision,
            batch_size=batch_size,
            n_samples=n_samples,
            n_features=n_features,
            epochs=epochs,
            num_workers=workers_value,
        )
        info = detect_hardware(device)
        payload = {
            "device_info": info.to_dict(),
            "results": [res.to_dict()],
            "recommendations": recommend_config(device_info=info, n_features=n_features),
            "notes": ["数值来自本次真实训练；不可测项为空。"],
        }
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(format_report(payload), encoding="utf-8")
            typer.secho(f"✓ 基准报告已写入: {out}", fg="green")
        typer.echo(format_report(payload))
    except typer.Exit:
        raise
    except Exception as exc:
        _handle(exc)


# ----------------------------------------------------------------------
# cache info / clear
# ----------------------------------------------------------------------
@cache_app.command("info")
def cache_info(
    config: Path | None = typer.Option(None, "--config", "-c", help="数据配置（读取 cache 段）"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """查看数据缓存统计（命中/未命中/条目/占用）。"""
    from cloudwater_ml.data.loader import cache_stats

    data_cfg = _load_cfg(config).get("data") if config else None
    stats = cache_stats(data_cfg)
    if as_json:
        typer.echo(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
        return
    if not stats.get("enabled"):
        typer.echo(f"缓存未启用: {stats.get('reason')}")
        return
    typer.echo(f"缓存根目录: {stats.get('root')}")
    typer.echo(f"写入次数  : {stats.get('writes')}")
    typer.echo(f"各级条目  : {stats.get('per_level')}")
    typer.echo(f"内存占用  : {stats.get('memory_bytes')} / {stats.get('memory_limit_bytes')} 字节")


@cache_app.command("clear")
def cache_clear(
    config: Path | None = typer.Option(None, "--config", "-c", help="数据配置（读取 cache 段）"),
    level: str | None = typer.Option(None, "--level", help="仅清理某一级（raw/metadata/preprocessing）"),
) -> None:
    """清理数据缓存（可重建）。**绝不触碰 data/raw 下的原始数据**。"""
    from cloudwater_ml.data.loader import clear_cache

    data_cfg = _load_cfg(config).get("data") if config else None
    result = clear_cache(data_cfg, level=level)
    if not result.get("enabled"):
        typer.echo("缓存未启用，无需清理")
        return
    typer.secho(f"✓ 已清理 {result['removed']} 个缓存条目（级别: {result['level']}）", fg="green")
    typer.echo("缓存会在下次加载时自动重建。")


def main() -> None:
    app()


if __name__ == "__main__":
    app()
