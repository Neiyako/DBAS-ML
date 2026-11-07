"""训练层：Trainer、损失、优化器、调度、回调、检查点、运行时计划。"""
from .callbacks import EarlyStopping, HistoryLogger
from .checkpoint import CheckpointManager, load_checkpoint, save_checkpoint
from .losses import CompositeLoss, NonNegativityPenalty, build_constraints, get_loss
from .optimizer import build_optimizer
from .runtime import DataLoaderOptions, RuntimePlan, resolve_runtime
from .scheduler import build_scheduler
from .trainer import Trainer

__all__ = [
    "Trainer",
    "get_loss",
    "CompositeLoss",
    "NonNegativityPenalty",
    "build_constraints",
    "build_optimizer",
    "build_scheduler",
    "EarlyStopping",
    "HistoryLogger",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "RuntimePlan",
    "DataLoaderOptions",
    "resolve_runtime",
]
