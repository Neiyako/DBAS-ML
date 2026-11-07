"""统一 Trainer：v2.0 GPU 训练引擎（本包核心训练循环）。

与 v1.0 的差异
--------------

=========================  ==========================================  ==========================
能力                       v1.0                                      v2.0
=========================  ==========================================  ==========================
设备                       resolve_device 字符串                      :class:`RuntimePlan`（含回退原因）
混合精度                   无                                          fp32 / fp16 / bf16 + GradScaler
梯度累积                   无                                          ``gradient_accumulation_steps``
显存监控                   无                                          峰值显存 / 峰值 RSS / OOM 捕获
DataLoader                 仅 num_workers                             pin_memory / persistent_workers /
                                                                  prefetch_factor / non_blocking
channels_last              无                                          按模型自动判定
torch.compile              无                                          可选（默认关）
性能剖析                   无                                          数据加载 / 计算 / 传输 三段耗时
检查点                     model + optimizer + epoch                   + scheduler / scaler / global_step /
                                                                  RNG 状态 / 实验配置
损失分量                   只记录总损失                                逐分量记录（监督 + 各物理约束）
=========================  ==========================================  ==========================

**向后兼容**：``Trainer(model, training_cfg)`` 的调用方式与
``fit(X_train, Y_train, X_val=..., Y_val=..., out_dir=...)`` 的签名保持不变；
新增的 ``hardware`` / ``dataloader`` 配置段为可选项。
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..hardware import MemoryMonitor, Profiler, autocast_context, make_grad_scaler
from ..utils import get_logger, resolve_device, set_seed
from ..utils.exceptions import OutOfMemoryError
from ..utils.reproducibility import capture_rng_state, restore_rng_state
from .callbacks import EarlyStopping, HistoryLogger
from .checkpoint import CheckpointManager, load_checkpoint
from .losses import CompositeLoss, build_constraints, get_loss
from .optimizer import build_optimizer
from .runtime import RuntimePlan, resolve_runtime
from .scheduler import build_scheduler

__all__ = ["Trainer"]

_logger = get_logger("training.trainer")


class Trainer:
    """统一训练器（PyTorch 模型共用）。

    Parameters
    ----------
    model : torch.nn.Module
        输入 ``[batch, features]``（或模型自身的张量契约），输出 ``[batch, targets]``。
    training_cfg : dict
        ``training`` 段（epochs / batch_size / learning_rate / optimizer / loss /
        scheduler / early_stopping / seed / device / num_workers / grad_clip /
        gradient_accumulation_steps / gradient_clip_norm）。
    device : str | None
        显式设备；None 时由配置解析（``hardware.device`` → ``training.device`` → auto）。
    config : dict | None
        **完整实验配置**（可选）。提供时可读取 ``hardware`` / ``dataloader`` /
        ``physics`` 等分段。
    model_name : str | None
        模型名（用于 channels_last 的 auto 判定）。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        training_cfg: dict[str, Any] | None = None,
        device: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model = model
        self.cfg = dict(training_cfg or {})
        self.full_config = dict(config or {})
        self.model_name = model_name
        self.logger = _logger
        self.plan: RuntimePlan | None = None
        self._explicit_device = device
        self._global_step = 0        # 跨 epoch 累计的优化器步数
        self.profiler = Profiler(enabled=True)
        self._last_components: dict[str, float] = {}
        if device:
            # 显式指定设备优先（保持 v1.0 行为：cuda 不可用时明确报错）
            resolved = resolve_device(str(device))
            self.device = torch.device(resolved)
        else:
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        out_dir: str | Path | None = None,
    ) -> Any:
        """训练并返回 ``TrainingResult``（history 每 epoch 一行）。"""
        cfg = self.cfg
        set_seed(int(cfg.get("seed", 42)), deterministic=bool(cfg.get("deterministic", False)))
        batch_size = int(cfg.get("batch_size", 256))

        # -- 运行时计划（设备 / 精度 / DataLoader / 累积） --------------
        plan = self._build_plan(X_train, batch_size)
        self.plan = plan
        self.device = plan.device
        self.logger.info(
            "运行时计划: device=%s precision=%s accum=%d batch=%d workers=%d",
            plan.device,
            plan.precision.mode,
            plan.gradient_accumulation_steps,
            plan.batch_size,
            plan.dataloader.num_workers,
        )
        for note in plan.notes:
            self.logger.warning("运行时提示: %s", note)

        # -- 数据 --------------------------------------------------------
        X_train = np.asarray(X_train, dtype=np.float32)
        Y_train = np.asarray(Y_train, dtype=np.float32)
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)
        train_loader = self._build_loader(
            X_train, Y_train, batch_size, plan, shuffle=True
        )
        val_loader: DataLoader | None = None
        if X_val is not None and Y_val is not None and len(X_val) > 0:
            X_val = np.asarray(X_val, dtype=np.float32)
            Y_val = np.asarray(Y_val, dtype=np.float32)
            if Y_val.ndim == 1:
                Y_val = Y_val.reshape(-1, 1)
            val_loader = self._build_loader(
                X_val, Y_val, batch_size, plan, shuffle=False
            )

        return self._fit_core(
            plan=plan,
            train_loader=train_loader,
            val_loader=val_loader,
            out_dir=out_dir,
        )

    def _fit_core(
        self,
        *,
        plan: RuntimePlan,
        train_loader: Any,
        val_loader: Any | None,
        out_dir: str | Path | None,
    ) -> Any:
        """训练核心（内存模式与流式模式共用）。

        调用方负责：设备计划、DataLoader 构造、随机种子。
        本方法负责：损失/约束、优化器/调度、AMP、梯度累积、检查点、
        早停、显存统计与结果汇总 —— 两条数据路径**共享同一实现**，
        避免出现「流式训练没有 AMP/累积」这类不一致。
        """
        from ..models.base import TrainingResult  # 延迟导入避免包循环依赖

        cfg = self.cfg
        epochs = int(cfg.get("epochs", 100))
        # -- 损失 / 约束 / 优化 / 调度 ------------------------------------
        supervised = get_loss(str(cfg.get("loss", "mse")), cfg)
        constraints = build_constraints(cfg.get("physics"))
        if constraints:
            self.logger.info("启用物理约束: %s", [c.name for c in constraints])
        physical_transform = self.full_config.get("_physical_transform")
        criterion: torch.nn.Module = CompositeLoss(
            supervised, constraints, physical_transform=physical_transform
        ).to(self.device)
        if constraints and physical_transform is None:
            self.logger.warning(
                "未提供物理单位还原变换：物理约束将作用在标准化空间上，"
                "结论不具物理意义（请在实验配置中注入 _physical_transform）"
            )
        optimizer = build_optimizer(str(cfg.get("optimizer", "adamw")), self.model, cfg)
        scheduler = build_scheduler(cfg.get("scheduler"), optimizer, cfg)

        # -- 模型上设备 / channels_last / 梯度检查点 / compile ------------
        self.model.to(self.device)
        raw_model = self.model
        if plan.channels_last:
            self._apply_channels_last()
        if plan.grad_checkpointing:
            self._enable_gradient_checkpointing(raw_model)
        if plan.compile:
            self.model = self._maybe_compile(self.model)
        # 检查点始终保存 **未编译的原始模块** 的 state_dict：
        # torch.compile 包装后键名会带 "_orig_mod." 前缀，导致跨配置续训失败
        self._raw_model = raw_model

        # -- 混合精度 ------------------------------------------------------
        scaler = make_grad_scaler(plan.precision)

        # -- 回调与检查点 --------------------------------------------------
        es_cfg = cfg.get("early_stopping") or {}
        early = EarlyStopping(
            patience=int(es_cfg.get("patience", 10)),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
        )
        es_enabled = bool(es_cfg.get("enabled", True))
        history = HistoryLogger()
        ckpt = CheckpointManager(out_dir)
        # -- resume -------------------------------------------------------
        start_epoch = 1
        best_val: float | None = None
        global_step = self._global_step
        resume = cfg.get("resume")
        if resume:
            payload = load_checkpoint(
                resume, raw_model, optimizer, map_location=str(self.device)
            )
            start_epoch = int(payload.get("epoch", 0)) + 1
            best_val = payload.get("best_val_loss")
            global_step = int(payload.get("global_step", 0) or 0)
            self._global_step = global_step
            if payload.get("scheduler_state") and scheduler is not None:
                try:
                    scheduler.load_state_dict(payload["scheduler_state"])
                except Exception as exc:  # 调度器结构变了也不能中断续训
                    self.logger.warning("恢复 scheduler 状态失败（忽略）: %s", exc)
            if payload.get("scaler_state") and scaler is not None:
                try:
                    scaler.load_state_dict(payload["scaler_state"])
                except Exception as exc:
                    self.logger.warning("恢复 GradScaler 状态失败（忽略）: %s", exc)
            rng_state = payload.get("rng_state")
            if rng_state:
                restore_rng_state(rng_state)
            es_state = payload.get("early_stopping_state") or {}
            if es_state:
                early.best = es_state.get("best")
                early.best_epoch = es_state.get("best_epoch")
                early.wait = int(es_state.get("wait", 0) or 0)
            resumed_history = payload.get("history") or []
            if resumed_history:
                for row in resumed_history:
                    history.add(row)
                self.logger.info("已恢复历史记录 %d 行", len(resumed_history))
            self.logger.info(
                "从 %s 恢复训练: epoch=%s, best_val=%s, global_step=%s",
                resume, payload.get("epoch"), best_val, global_step,
            )

        monitor_is_val = val_loader is not None
        best_epoch: int | None = None
        stopped_early = False
        self.profiler = Profiler(enabled=True)
        profiler = self.profiler
        monitor = MemoryMonitor(plan.device_type)
        monitor.start()

        def _improved(v: float) -> bool:
            return best_val is None or v < best_val

        # -- 训练循环 ------------------------------------------------------
        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss, train_components, n_batches = self._run_epoch(
                train_loader, criterion, optimizer, scaler, plan, train=True
            )
            val_loss = None
            val_components: dict[str, float] = {}
            if val_loader is not None:
                val_loss, val_components, _ = self._run_epoch(
                    val_loader, criterion, None, None, plan, train=False
                )
            if scheduler is not None:
                scheduler.step()

            monitored = val_loss if monitor_is_val else train_loss
            history.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            improved = _improved(float(monitored))
            if improved:
                best_val = float(monitored)
                best_epoch = epoch
                ckpt.save(
                    "best", raw_model, optimizer, epoch, best_val,
                    {
                        "trainer_config": cfg,
                        "runtime_plan": plan.to_dict(),
                        "loss_components": train_components,
                        "history": history.as_dicts(),
                        "early_stopping_state": {
                            "best": early.best,
                            "best_epoch": early.best_epoch,
                            "wait": early.wait,
                        },
                    },
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=self._global_step,
                    rng_state=capture_rng_state(),
                )
            ckpt.save(
                "last", raw_model, optimizer, epoch, best_val,
                {
                    "trainer_config": cfg,
                    "runtime_plan": plan.to_dict(),
                    "loss_components": train_components,
                    "history": history.as_dicts(),
                    "early_stopping_state": {
                        "best": early.best,
                        "best_epoch": early.best_epoch,
                        "wait": early.wait,
                    },
                },
                scheduler=scheduler,
                scaler=scaler,
                global_step=self._global_step,
                rng_state=capture_rng_state(),
            )

            self.logger.info(
                "epoch %d/%d | train=%.6f | val=%s | %.1fs%s",
                epoch, epochs, train_loss,
                f"{val_loss:.6f}" if val_loss is not None else "-",
                time.time() - t0,
                " *" if improved else "",
            )
            if train_components:
                self._last_components = dict(train_components)
                self.logger.info("  损失分量: %s", train_components)

            if es_enabled and early.step(float(monitored), epoch):
                stopped_early = True
                self.logger.info(
                    "EarlyStopping: %d epoch 未改善（best epoch=%s, best=%s），停止训练",
                    early.patience, early.best_epoch, early.best,
                )
                break

        # -- 收尾：恢复最优点权重（与 sklearn 基线“best model”语义一致） -----
        restored = ckpt.load("best", raw_model, optimizer) if best_val is not None else None
        if restored is not None:
            self.logger.info("已恢复 best 权重（epoch=%s, val=%.6f）",
                             restored.get("epoch"), best_val)
        memory = monitor.stop()

        return TrainingResult(
            history=history.as_dicts(),
            best_epoch=best_epoch,
            best_val_loss=best_val,
            stopped_early=stopped_early,
            extra={
                "device": str(self.device),
                "epochs_run": history.rows[-1]["epoch"] if history.rows else 0,
                "monitor": "val_loss" if monitor_is_val else "train_loss",
                "restored_best": restored is not None,
                "runtime_plan": plan.to_dict(),
                "global_step": self._global_step,
                "loss_components": dict(self._last_components),
                "memory": memory.to_dict(),
                "peak_vram_bytes": monitor.peak_vram_bytes,
                "peak_host_rss_bytes": monitor.peak_host_rss_bytes,
                "profiling": profiler.to_dict(),
                "n_batches_last_epoch": n_batches,
            },
        )

    def fit_stream(
        self,
        source: Any,
        *,
        n_features: int,
        n_targets: int,
        n_samples_hint: int | None = None,
        out_dir: str | Path | None = None,
    ) -> Any:
        """**流式训练**：直接消费分批产出的数据源，不要求整份数据在内存里。

        Parameters
        ----------
        source : Iterable 或 DataLoader
            每个元素是 ``(x, y)`` 张量元组（``x [B,F]``、``y [B,T]``）。
            通常传 ``data.stream.NetCDFStreamDataset``。
        n_features / n_targets : int
            用于推导运行时计划（模型结构由调用方自行构造）。
        n_samples_hint : int | None
            预估总样本数（仅用于 `num_workers` 自动推断，可为 None）。
        out_dir : 检查点目录。

        与 :meth:`fit` 的区别只有「数据怎么来」；训练循环、AMP、梯度累积、
        检查点、早停、显存统计**完全共用** :meth:`_fit_core`。
        """
        from torch.utils.data import DataLoader as _DL

        set_seed(int(self.cfg.get("seed", 42)), deterministic=bool(self.cfg.get("deterministic", False)))

        plan = self._build_plan_shape(
            n_features=n_features,
            n_targets=n_targets,
            batch_size=int(self.cfg.get("batch_size", 256)),
            n_samples=n_samples_hint,
        )
        self.plan = plan
        self.device = plan.device
        self.logger.info(
            "流式训练计划: device=%s precision=%s accum=%d workers=%d",
            plan.device,
            plan.precision.mode,
            plan.gradient_accumulation_steps,
            plan.dataloader.num_workers,
        )

        # batch_size=None：数据源自己产出「一批」，DataLoader 不再切分
        kwargs = plan.dataloader.loader_kwargs()
        kwargs.pop("shuffle", None)
        kwargs.pop("drop_last", None)
        loader = (
            source
            if isinstance(source, _DL)
            else _DL(source, batch_size=None, **{k: v for k, v in kwargs.items() if k != "pin_memory"})
        )
        return self._fit_core(plan=plan, train_loader=loader, val_loader=None, out_dir=out_dir)

    def _build_plan_shape(
        self,
        *,
        n_features: int,
        n_targets: int,
        batch_size: int,
        n_samples: int | None,
    ) -> RuntimePlan:
        """按张量形状（而非具体数组）解析运行时计划，供流式路径使用。"""
        cfg_for_plan = dict(self.full_config) if self.full_config else {}
        if not cfg_for_plan:
            cfg_for_plan = {"training": dict(self.cfg)}
        else:
            cfg_for_plan["training"] = {**dict(self.cfg), **dict(cfg_for_plan.get("training") or {})}
        if self.cfg.get("physics") and "physics" not in cfg_for_plan:
            cfg_for_plan["physics"] = self.cfg["physics"]
        if self._explicit_device:
            cfg_for_plan.setdefault("hardware", {})
            cfg_for_plan["hardware"] = {
                **dict(cfg_for_plan.get("hardware") or {}),
                "device": self._explicit_device,
            }
        plan = resolve_runtime(
            cfg_for_plan,
            model_name=self.model_name or type(self.model).__name__.lower(),
            supports_channels_last=getattr(self.model, "supports_channels_last", None),
            n_samples=n_samples,
        )
        plan.batch_size = int(batch_size)
        plan.n_features = int(n_features)  # type: ignore[attr-defined]
        plan.n_targets = int(n_targets)    # type: ignore[attr-defined]
        return plan

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _build_plan(self, X_train: np.ndarray, batch_size: int) -> RuntimePlan:
        """解析运行时计划（完整配置优先，其次仅 training 段）。"""
        cfg_for_plan = dict(self.full_config) if self.full_config else {}
        if not cfg_for_plan:
            cfg_for_plan = {"training": dict(self.cfg)}
        else:
            cfg_for_plan["training"] = {**dict(self.cfg), **dict(cfg_for_plan.get("training") or {})}
        if self.cfg.get("physics") and "physics" not in cfg_for_plan:
            cfg_for_plan["physics"] = self.cfg["physics"]
        if self._explicit_device:
            cfg_for_plan.setdefault("hardware", {})
            cfg_for_plan["hardware"] = {
                **dict(cfg_for_plan.get("hardware") or {}),
                "device": self._explicit_device,
            }
        plan = resolve_runtime(
            cfg_for_plan,
            model_name=self.model_name or type(self.model).__name__.lower(),
            supports_channels_last=getattr(self.model, "supports_channels_last", None),
            n_samples=int(len(X_train)),
        )
        plan.batch_size = batch_size
        return plan

    def _build_loader(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        plan: RuntimePlan,
        *,
        shuffle: bool,
    ) -> DataLoader:
        """构造 DataLoader（按运行时计划设置 workers / pin_memory / prefetch）。"""
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        kwargs = plan.dataloader.loader_kwargs()
        kwargs["shuffle"] = shuffle
        kwargs["drop_last"] = plan.dataloader.drop_last and shuffle
        return DataLoader(ds, batch_size=batch_size, **kwargs)

    def _apply_channels_last(self) -> None:
        """把 4D 卷积权重转成 channels_last（小 MLP 不受影响，会直接跳过）。"""
        try:
            self.model.to(memory_format=torch.channels_last)
            self.logger.info("已启用 channels_last 内存格式")
        except Exception as exc:  # pragma: no cover - 非 4D 模型会失败，属正常
            self.logger.info("channels_last 未生效（模型非 4D 卷积）: %s", exc)

    def _enable_gradient_checkpointing(self, model: torch.nn.Module) -> None:
        """梯度检查点：用「重算中间激活」换显存（大模型 / 显存紧张时启用）。"""
        try:
            import torch.utils.checkpoint as ckpt

            original_forward = model.forward

            def forward(*args: Any, **kwargs: Any) -> Any:
                if not torch.is_grad_enabled():
                    return original_forward(*args, **kwargs)
                return ckpt.checkpoint(
                    lambda *a: original_forward(*a, **kwargs),
                    *args,
                    use_reentrant=False,
                )

            model.forward = forward  # type: ignore[method-assign]
            self.logger.info("已启用梯度检查点（以重算开销换显存）")
        except Exception as exc:  # pragma: no cover
            self.logger.warning("启用梯度检查点失败（忽略）: %s", exc)

    def _maybe_compile(self, model: torch.nn.Module) -> torch.nn.Module:
        try:
            compiled = torch.compile(model)
            self.logger.info("已启用 torch.compile（首次编译有额外耗时）")
            return compiled
        except Exception as exc:  # pragma: no cover
            self.logger.warning("torch.compile 启用失败，回退到 eager: %s", exc)
            return model

    # ------------------------------------------------------------------
    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: Any,
        plan: RuntimePlan,
        *,
        train: bool,
    ) -> tuple[float, dict[str, float], int]:
        """跑一个 epoch，返回 (按样本加权平均损失, 损失分量, batch 数)。

        计时口径（写进 profiler，供 Benchmark 归因瓶颈）：

        - ``data_load``：从 DataLoader 取下一个 batch 的等待时间
          （含多进程 worker 的排队与传输时间）；
        - ``compute``：前向 + 反向 + 优化器步进；
        - ``transfer``：把 batch 搬到目标设备的时间。
        """
        if train:
            self.model.train()
        else:
            self.model.eval()
        n_batches = 0
        pending = 0            # 已 backward、尚未 step 的 micro-batch 数
        n_seen = 0
        loss_sum = None        # 设备上的损失累加（epoch 末只同步一次）
        comp_sum: dict[str, Any] = {}
        accum = plan.gradient_accumulation_steps if train else 1
        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            iterator = iter(loader)
            step = 0
            while True:
                t_fetch = time.perf_counter()
                try:
                    xb, yb = next(iterator)
                except StopIteration:
                    break
                data_seconds = time.perf_counter() - t_fetch
                self.profiler.stage_accumulate("data_load", data_seconds)

                t_transfer = time.perf_counter()
                xb = xb.to(self.device, non_blocking=plan.dataloader.non_blocking)
                yb = yb.to(self.device, non_blocking=plan.dataloader.non_blocking)
                if plan.device_type == "cuda" and plan.dataloader.non_blocking:
                    # pinned memory + non_blocking：真正完成搬运需要同步
                    torch.cuda.current_stream().synchronize()
                transfer_seconds = time.perf_counter() - t_transfer
                self.profiler.stage_accumulate("transfer", transfer_seconds)

                if train and plan.channels_last and xb.ndim == 4:
                    xb = xb.contiguous(memory_format=torch.channels_last)

                if train:
                    # 混合精度：训练走 autocast；验证/指标默认 fp32（数值更稳）
                    amp_ctx = autocast_context(plan.precision, self.device)
                else:
                    amp_ctx = nullcontext()

                with amp_ctx:
                    out = self.model(xb)
                    loss = criterion(out, {"X": xb, "Y": yb})
                    raw_loss = loss.detach()  # 未除以累积步的原始损失
                    if train:
                        loss = loss / accum

                bs = xb.shape[0]
                loss_sum = raw_loss * bs if loss_sum is None else loss_sum + raw_loss * bs
                n_seen += bs
                n_batches += 1
                step += 1

                if train:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    pending += 1
                    if pending >= accum:
                        self._apply_gradients(optimizer, scaler, plan)
                        pending = 0
                    # 损失分量：设备上累加，epoch 末统一取回（避免每 batch 同步）
                    if hasattr(criterion, "component_tensors"):
                        for k, v in criterion.component_tensors(out, {"X": xb, "Y": yb}).items():
                            prev = comp_sum.get(k)
                            comp_sum[k] = v * bs if prev is None else prev + v * bs

            # -- 尾部 flush：batch 数不是 accum 整数倍时，剩余梯度必须走一次 step，
            #    否则这批样本被静默丢弃（等效 batch 小于配置值）。
            if train and pending > 0:
                scale = float(accum) / float(pending)
                for prm in self.model.parameters():
                    if prm.grad is not None:
                        prm.grad.mul_(scale)
                self.logger.info(
                    "梯度累积尾部 flush：剩余 %d 个 micro-batch，按 %.3f 修正平均", pending, scale
                )
                self._apply_gradients(optimizer, scaler, plan)
                pending = 0

        total = float(loss_sum) if loss_sum is not None else 0.0
        n = n_seen
        last_components = {
            k: float(v) / max(n, 1) for k, v in comp_sum.items()
        }
        return total / max(n, 1), last_components, n_batches

    def _apply_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        plan: RuntimePlan,
    ) -> None:
        """裁剪 + 一步优化（AMP 下需先 unscale_ 再裁剪）。"""
        try:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if plan.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(plan.gradient_clip_norm)
                )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            self._global_step += 1
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - 需 CUDA
            raise OutOfMemoryError(
                f"显存不足（优化器步进阶段）：{exc}",
                hint=(
                    "可尝试：减小 batch_size、启用 gradient_accumulation_steps、"
                    "降低精度到 fp16/bf16、对大模型启用梯度检查点"
                ),
                context={
                    "batch_size": plan.batch_size,
                    "effective_batch_size": plan.effective_batch_size,
                    "precision": plan.precision.mode,
                    "device": str(plan.device),
                },
            ) from exc
        finally:
            optimizer.zero_grad(set_to_none=True)

