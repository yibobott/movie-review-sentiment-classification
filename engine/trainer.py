"""Training / validation loops with early stopping + LR scheduling."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.config import TrainConfig


@dataclass
class BestMetrics:
    best_val_acc: float = 0.0
    best_epoch: int = -1
    stopped_at: int = -1


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int):
    if cfg.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience
        )
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.epochs * steps_per_epoch)
        )
    return None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    scheduler=None,
    cosine_step: bool = False,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if cosine_step and scheduler is not None:
            scheduler.step()
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
    tag: str = "init",
) -> BestMetrics:
    epochs = epochs_override if epochs_override is not None else cfg.epochs
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, cfg, len(train_loader))
    cosine_step = cfg.lr_scheduler == "cosine"

    best = BestMetrics()
    patience = 0

    logger.info(f"[{tag}] start training: epochs={epochs}, steps/epoch={len(train_loader)}, "
                f"params_trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            cfg.grad_clip, scheduler=scheduler, cosine_step=cosine_step,
        )
        val_loss, val_acc = valid_one_epoch(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[{tag}] epoch {epoch:02d}/{epochs} | lr {lr_now:.2e} | "
            f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f} | "
            f"val loss {val_loss:.4f} acc {val_acc*100:.2f}"
        )

        if cfg.lr_scheduler == "plateau" and scheduler is not None:
            scheduler.step(val_acc)

        if val_acc > best.best_val_acc:
            best.best_val_acc = val_acc
            best.best_epoch = epoch
            patience = 0
            torch.save({"model_state": model.state_dict(), "val_acc": val_acc},
                       str(ckpt_path))
            logger.info(f"[{tag}]   ↳ new best val acc {val_acc*100:.2f}, ckpt saved")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                best.stopped_at = epoch
                logger.info(f"[{tag}] early stop at epoch {epoch}")
                break

    logger.info(f"[{tag}] best val acc {best.best_val_acc*100:.2f} @ epoch {best.best_epoch}")
    return best
