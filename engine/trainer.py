"""Training / validation loops with early stopping + LR scheduling."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.ema import ModelEMA
from utils.config import TrainConfig


@dataclass
class BestMetrics:
    best_val_acc: float = 0.0
    best_epoch: int = -1
    stopped_at: int = -1


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
    epochs: int,
):
    if cfg.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience
        )
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs * steps_per_epoch)
        )
    if cfg.lr_scheduler == "warmup_cosine":
        total = max(1, epochs * steps_per_epoch)
        warmup = max(1, int(total * max(0.0, cfg.warmup_ratio)))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    scheduler=None,
    per_step_sched: bool = False,
    ema: ModelEMA | None = None,
    label_smoothing: float = 0.0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        # Apply binary label smoothing on the soft targets only (train-time);
        # validation/accuracy still uses the original {0,1} labels.
        if label_smoothing > 0.0:
            y_soft = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            y_soft = y
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y_soft)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if per_step_sched and scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == y.long()).sum().item()
            total_n += y.size(0)
            total_loss += loss.item() * y.size(0)
    return total_loss / max(1, total_n), total_correct / max(1, total_n)


@torch.no_grad()
def valid_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        logits = model(x)
        loss = criterion(logits, y)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        total_correct += (preds == y.long()).sum().item()
        total_n += y.size(0)
        total_loss += loss.item() * y.size(0)
    return total_loss / max(1, total_n), total_correct / max(1, total_n)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    ckpt_path: str | Path,
    logger: logging.Logger,
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    tag: str = "init",
) -> BestMetrics:
    epochs = epochs_override if epochs_override is not None else cfg.epochs
    lr = lr_override if lr_override is not None else cfg.lr
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=cfg.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, cfg, len(train_loader), epochs)
    per_step_sched = cfg.lr_scheduler in ("cosine", "warmup_cosine")

    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema_decay and cfg.ema_decay > 0 else None

    best = BestMetrics()
    patience = 0

    logger.info(
        f"[{tag}] start training: epochs={epochs}, lr={lr:.2e}, steps/epoch={len(train_loader)}, "
        f"ema={'on' if ema else 'off'}, "
        f"params_trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Shadow model used for EMA-based validation; same architecture as `model`.
    shadow_model = None
    if ema is not None:
        import copy
        shadow_model = copy.deepcopy(model).to(device)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            cfg.grad_clip, scheduler=scheduler, per_step_sched=per_step_sched, ema=ema,
            label_smoothing=cfg.label_smoothing,
        )
        val_loss, val_acc = valid_one_epoch(model, val_loader, criterion, device)
        ema_val_acc = None
        if ema is not None:
            ema.copy_to(shadow_model)
            _, ema_val_acc = valid_one_epoch(shadow_model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        ema_str = f" | ema val acc {ema_val_acc*100:.2f}" if ema_val_acc is not None else ""
        logger.info(
            f"[{tag}] epoch {epoch:02d}/{epochs} | lr {lr_now:.2e} | "
            f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f} | "
            f"val loss {val_loss:.4f} acc {val_acc*100:.2f}{ema_str}"
        )

        if cfg.lr_scheduler == "plateau" and scheduler is not None:
            scheduler.step(val_acc)

        # Pick the better of EMA and raw weights for the checkpoint. EMA needs
        # a few epochs to warm up (shadow starts at random init), so during
        # `ema_warmup_epochs` we always trust the raw weights.
        use_ema = (
            ema_val_acc is not None
            and epoch > cfg.ema_warmup_epochs
            and ema_val_acc >= val_acc
        )
        if use_ema:
            eff_val = ema_val_acc
            eff_state = shadow_model.state_dict()
        else:
            eff_val = val_acc
            eff_state = model.state_dict()

        if eff_val > best.best_val_acc:
            best.best_val_acc = eff_val
            best.best_epoch = epoch
            patience = 0
            torch.save({"model_state": eff_state, "val_acc": eff_val}, str(ckpt_path))
            logger.info(f"[{tag}]   ↘ new best val acc {eff_val*100:.2f}, ckpt saved")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                best.stopped_at = epoch
                logger.info(f"[{tag}] early stop at epoch {epoch}")
                break

    logger.info(f"[{tag}] best val acc {best.best_val_acc*100:.2f} @ epoch {best.best_epoch}")
    return best
