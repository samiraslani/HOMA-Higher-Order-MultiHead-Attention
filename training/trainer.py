"""
Unified training loop for all published tasks.

``Trainer`` supports both per-residue classification (SS3) and global
regression (fluorescence, stability) by accepting task-specific loss and
metric functions.

Checkpointing
-------------
After every epoch the trainer saves a checkpoint to
``{checkpoint_dir}/{attn_name}.pt``.  If a checkpoint already exists,
training resumes from the last saved epoch automatically.
"""

import contextlib
import io
import logging
import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import TrainingConfig
from utils.checkpointing import load_checkpoint, save_checkpoint
from .efficiency import EfficiencyTracker

logger = logging.getLogger(__name__)


class Trainer:
    """Train and validate a ``ProteinTransformer`` for one or more tasks.

    Args:
        config: Training hyperparameters.
        attn_name: Short name used for checkpoint file naming (e.g. the
            attention type string such as ``"homa"``).
        select_by: Which validation signal determines the best checkpoint.
            ``"val_loss"`` minimises validation loss (used for SS3);
            ``"val_metric"`` maximises the primary task metric, i.e. Spearman ρ
            (used for fluorescence and stability).
    """

    def __init__(
        self,
        config: TrainingConfig,
        attn_name: str = "model",
        select_by: str = "val_loss",
        metric_label: str = "metric",
    ) -> None:
        if select_by not in ("val_loss", "val_metric"):
            raise ValueError("select_by must be 'val_loss' or 'val_metric'")
        self.config = config
        self.attn_name = attn_name
        self.select_by = select_by
        self.metric_label = metric_label
        self.device = torch.device(
            config.device
            if config.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._checkpoint_path = os.path.join(
            config.checkpoint_dir, f"{attn_name}.pt"
        )
        self._best_checkpoint_path = os.path.join(
            config.checkpoint_dir, f"{attn_name}_best.pt"
        )

    # ------------------------------------------------------------------
    # Internal scheduler builder
    # ------------------------------------------------------------------

    def _build_scheduler(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
        """Return a per-step LambdaLR scheduler, or ``None`` if not needed."""
        scheduler_type = self.config.lr_scheduler
        if warmup_steps == 0 and scheduler_type == "none":
            return None

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            if scheduler_type == "cosine":
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            if scheduler_type == "linear":
                return max(0.0, 1.0 - progress)
            return 1.0  # "none": flat after warmup

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        metric_fn: Callable,
        is_classification: bool = True,
        track_efficiency: bool = False,
    ) -> Tuple[nn.Module, Dict[str, List]]:
        """Run the full training loop.

        Args:
            model: The model to train (will be moved to ``self.device``).
            train_loader: Training ``DataLoader``.
            val_loader: Validation ``DataLoader``.
            criterion: Loss function.  For SS3 this is
                ``nn.CrossEntropyLoss(ignore_index=-100)``; for regression
                ``nn.MSELoss()``.
            metric_fn: Callable ``(logits, labels) → float`` returning the
                primary scalar metric (accuracy or Spearman ρ).
            is_classification: If ``True``, treats the task as per-residue
                classification (model returns ``(logits, labels)``); if
                ``False``, treats it as global regression (model returns
                scalar predictions, batch has ``"targets"`` key).
            track_efficiency: If ``True``, collect per-step timing and memory
                metrics using :class:`EfficiencyTracker`.

        Returns:
            Tuple ``(trained_model, history)`` where ``history`` is a dict of
            lists keyed by metric name (train_loss, val_loss, val_metric, and
            optionally efficiency keys).
        """
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # Build LR scheduler (needs total steps, so requires loader length)
        total_steps = self.config.epochs * len(train_loader)
        warmup_steps_lr = int(self.config.warmup_ratio * total_steps)
        scheduler = self._build_scheduler(optimizer, warmup_steps_lr, total_steps)

        start_epoch = 0
        history: Dict[str, List] = {
            "train_loss": [],
            "train_metric": [],
            "val_loss": [],
            "val_metric": [],
        }
        if track_efficiency:
            for k in (
                "avg_step_ms_e2e", "tokens_per_sec_e2e",
                "avg_step_ms_compute", "tokens_per_sec_compute",
                "avg_data_ms", "avg_compute_ms",
                "epoch_wall_s", "peak_mem_alloc_gb", "peak_mem_reserved_gb",
            ):
                history[k] = []

        # Resume from checkpoint if one exists
        if os.path.exists(self._checkpoint_path):
            logger.info(
                "Found checkpoint for %s — resuming training.", self.attn_name
            )
            ckpt = load_checkpoint(
                self._checkpoint_path, model, optimizer, device=str(self.device)
            )
            start_epoch = ckpt.get("epoch", -1) + 1
            history = ckpt.get("history", history)
            if scheduler is not None and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # --- Training summary (printed once) ---
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info("Total trainable parameters: %d", total_params)

        sep = "-" * 50
        print(f"\n{sep}")
        print("  Training Setup")
        print(sep)
        if hasattr(model, "attn_cfg"):
            print(f"  Attention type    : {model.attn_cfg.type}")
            print(f"  Rank              : {model.attn_cfg.rank_3d}")
            if model.attn_cfg.type.lower() in ("homa", "blockwise3d"):
                print(f"  Window size       : {model.attn_cfg.window_size}")
                if getattr(model.attn_cfg, "tie_u_to_k", False):
                    print(f"  Tie U -> K        : yes (score = Q·K·K)")
                if getattr(model.attn_cfg, "uniform_pool_3d", False):
                    print(f"  Triadic attention : UNIFORM POOL (ablation, no scores)")
                combine = getattr(model.attn_cfg, "combine", "fusion")
                if combine != "fusion":
                    detail = ("attn_2d + res_3d (no fusion MLP)" if combine == "add"
                              else f"attn_2d + gate * res_3d (gate init "
                                   f"{getattr(model.attn_cfg, 'gate_init', 1.0)})")
                    print(f"  2D/3D combine     : {combine.upper()} (ablation) — {detail}")
        print(f"  Device            : {self.device}")
        print(f"  Epochs            : {self.config.epochs}")
        print(f"  Learning rate     : {self.config.learning_rate}")
        print(f"  LR scheduler      : {self.config.lr_scheduler}"
              + (f"  (warmup {self.config.warmup_ratio:.0%} = {warmup_steps_lr} steps)"
                 if warmup_steps_lr > 0 else ""))
        print(f"  Grad clip         : {self.config.grad_clip if self.config.grad_clip > 0 else 'disabled'}")
        u_lam = getattr(self.config, "u_entropy_lambda", 0.0)
        print(f"  U-entropy penalty : {u_lam if u_lam > 0 else 'disabled'}")
        print(f"  Best-model by     : {self.select_by}")
        print(f"  Trainable params  : {total_params:,}")
        if frozen_params > 0:
            frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
            frozen_roots = sorted({n.rsplit(".", 1)[0] for n in frozen_names})
            print(f"  Frozen params     : {frozen_params:,}")
            print(f"  Frozen modules    : {', '.join(frozen_roots)}")
        print(f"{sep}\n")

        # Initialise best-model tracking.
        # val_loss is minimised; val_metric is maximised.
        best_val_loss = float("inf")
        best_val_metric = -float("inf")
        best_epoch = -1

        tracker = (
            EfficiencyTracker(
                device=str(self.device),
                warmup_steps=self.config.warmup_steps,
            )
            if track_efficiency
            else None
        )

        for epoch in range(start_epoch, self.config.epochs):
            train_loss, train_metric = self._train_epoch(
                model, train_loader, optimizer, criterion,
                is_classification, tracker, metric_fn, scheduler,
            )
            val_loss, val_metric = self._validate(
                model, val_loader, criterion, metric_fn, is_classification
            )

            history["train_loss"].append(train_loss)
            history["train_metric"].append(train_metric)
            history["val_loss"].append(val_loss)
            history["val_metric"].append(val_metric)

            # --- best-model check ---
            is_best = (
                val_loss < best_val_loss
                if self.select_by == "val_loss"
                else val_metric > best_val_metric
            )
            if is_best:
                best_val_loss = val_loss
                best_val_metric = val_metric
                best_epoch = epoch
                save_checkpoint(
                    self._best_checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    extra={
                        "history": history,
                        "val_loss": val_loss,
                        "val_metric": val_metric,
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    },
                )
                logger.info("New best model saved at epoch %d.", epoch + 1)

            if tracker is not None:
                eff = tracker.end_epoch()
                for k, v in eff.items():
                    history[k].append(v)
                self._log_efficiency(epoch, train_loss, train_metric, val_loss, val_metric, eff,
                                     is_best=is_best)
            else:
                best_marker = "  ← best" if is_best else ""
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train {self.metric_label}: {train_metric:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val {self.metric_label}: {val_metric:.4f}"
                    f"{best_marker}"
                )

            # Rolling checkpoint (for resume)
            save_checkpoint(
                self._checkpoint_path,
                model,
                optimizer,
                epoch,
                extra={
                    "history": history,
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                },
            )

        # Load the best weights back before returning.
        if os.path.exists(self._best_checkpoint_path):
            best_ckpt = torch.load(
                self._best_checkpoint_path,
                map_location=str(self.device),
                weights_only=False,
            )
            model.load_state_dict(best_ckpt["model_state_dict"])
            print(
                f"\nLoaded best model from epoch {best_epoch + 1} "
                f"(val_loss={best_val_loss:.4f}, val_metric={best_val_metric:.4f})"
            )

        history["best_epoch"] = best_epoch
        return model, history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        loader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        is_classification: bool,
        tracker: Optional[EfficiencyTracker],
        metric_fn: Callable,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    ) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        if tracker is not None:
            tracker.start_epoch()

        pbar = tqdm(loader, desc="Training", leave=False)
        prev_t = None
        for step, batch in enumerate(pbar):
            if tracker is not None:
                if prev_t is None:
                    # first batch: no data-loading gap measurement
                    import time
                    prev_t = time.perf_counter()
                tracker.record_batch_start()
                tracker.record_compute_start()

            if is_classification:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits, labels = model(input_ids, labels)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
                all_preds.append(logits.detach().cpu())
                all_targets.append(labels.detach().cpu())
            else:
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                preds = model(input_ids)
                loss = criterion(preds.squeeze(-1), targets)
                all_preds.append(preds.detach().cpu())
                all_targets.append(targets.detach().cpu())

            loss.backward()
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if tracker is not None:
                n_tokens = int(input_ids.numel())
                tracker.record_compute_end(n_tokens)

        train_metric = metric_fn(torch.cat(all_preds), torch.cat(all_targets))
        return total_loss / max(len(loader), 1), train_metric

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        loader,
        criterion: nn.Module,
        metric_fn: Callable,
        is_classification: bool,
    ) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        for batch in loader:
            if is_classification:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits, labels = model(input_ids, labels)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
                all_preds.append(logits.cpu())
                all_targets.append(labels.cpu())
            else:
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)
                preds = model(input_ids)
                loss = criterion(preds.squeeze(-1), targets)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)

        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)

        metric = metric_fn(preds_cat, targets_cat)

        return avg_loss, metric

    def _log_efficiency(
        self,
        epoch: int,
        train_loss: float,
        train_metric: float,
        val_loss: float,
        val_metric: float,
        eff: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        best_marker = "  ← best" if is_best else ""
        print(
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | Train {self.metric_label}: {train_metric:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val {self.metric_label}: {val_metric:.4f}{best_marker}\n"
            f"  End-to-end: {eff['avg_step_ms_e2e']:.2f} ms/step | "
            f"{eff['tokens_per_sec_e2e']:,.0f} tok/s\n"
            f"  Compute:    {eff['avg_step_ms_compute']:.2f} ms/step | "
            f"{eff['tokens_per_sec_compute']:,.0f} tok/s\n"
            f"  Memory:     alloc {eff['peak_mem_alloc_gb']:.2f} GB | "
            f"reserved {eff['peak_mem_reserved_gb']:.2f} GB\n"
            f"  Timing:     data {eff['avg_data_ms']:.2f} ms | "
            f"compute {eff['avg_compute_ms']:.2f} ms | "
            f"epoch {eff['epoch_wall_s']:.1f} s"
        )
