"""数据层：统一结构、适配器、加载、预处理、划分、质量检查、合成数据。"""
from .adapters import available_adapters, get_adapter
from .loader import inspect_data, iter_batches, load_dataset
from .preprocessing import NoScaler, Preprocessors, StandardScaler, fit_scalers, make_scaler
from .schema import (
    KNOWN_VARIABLES,
    SOURCE_REAL,
    SOURCE_SIMULATED,
    SampleBundle,
    validate_bundle_dims,
)
from .splitting import SplitResult, available_strategies, split_indices
from .synthetic import ensure_synthetic_dataset, make_synthetic_bundle, save_synthetic_npz
from .validation import apply_cleaning, format_quality_report, run_quality_checks

__all__ = [
    "KNOWN_VARIABLES",
    "SOURCE_REAL",
    "SOURCE_SIMULATED",
    "SampleBundle",
    "validate_bundle_dims",
    "available_adapters",
    "get_adapter",
    "load_dataset",
    "inspect_data",
    "iter_batches",
    "fit_scalers",
    "make_scaler",
    "Preprocessors",
    "StandardScaler",
    "NoScaler",
    "SplitResult",
    "split_indices",
    "available_strategies",
    "ensure_synthetic_dataset",
    "make_synthetic_bundle",
    "save_synthetic_npz",
    "run_quality_checks",
    "format_quality_report",
    "apply_cleaning",
]
