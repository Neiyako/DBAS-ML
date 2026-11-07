"""性能基准模块（M6）：训练吞吐、瓶颈归因、显存峰值与配置建议。"""
from .runner import (
    BenchmarkResult,
    benchmark_training,
    format_report,
    recommend_config,
    run_benchmark_suite,
)

__all__ = [
    "BenchmarkResult",
    "benchmark_training",
    "format_report",
    "recommend_config",
    "run_benchmark_suite",
]
