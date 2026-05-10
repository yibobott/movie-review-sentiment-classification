"""Language-model training loop with perplexity tracking and early stopping.

Notable choices
---------------
* Loss is token-level cross entropy with ``ignore_index=PAD_IDX``; EOS is NOT
  ignored (we want the LM to learn document boundaries — see design §4.2).
* Perplexity is reported as ``exp(mean_loss)``, computed as a token-weighted
  average so chunks of differing effective lengths (after PAD masking) do not
  bias the metric.
* Optional warmup-cosine schedule per step. LSTM-LMs gradient-explode easily,
  so ``grad_clip`` defaults to a strict 0.5.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.config import LMConfig


@dataclass
class LMBest:
    best_val_ppl: float = float("inf")
    best_epoch: int = -1
    stopped_at: int = -1


def _make_warmup_cosine(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * max(0.0, warmup_ratio)))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_ppl(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_idx: int,
) -> float:
    """Token-weighted perplexity over ``loader``."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)  # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        # Count tokens that contributed to the loss (i.e. y != pad).
        n_tokens = int((y != pad_idx).sum().item())
        total_loss += float(loss.item())
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def train_lm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: LMConfig,
    device: torch.device,
    *,
    ckpt_path: str | Path,
    logger: logging.Logger,
    pad_idx: int,
) -> LMBest:
    """Run LM pretraining; saves best (lowest val PPL) checkpoint to ``ckpt_path``."""
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, cfg.epochs * len(train_loader))
    scheduler = _make_warmup_cosine(optimizer, total_steps, cfg.warmup_ratio)

    best = LMBest()
    patience = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[lm] start training: epochs={cfg.epochs}, lr={cfg.lr:.2e}, "
        f"steps/epoch={len(train_loader)}, params_trainable={n_params}, "
        f"bptt={cfg.bptt_len}, batch={cfg.batch_size}"
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)  # (B, T, V)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            n_tokens = int((y != pad_idx).sum().item())
            running_loss += float(loss.item()) * max(1, n_tokens)
            running_tokens += n_tokens

        train_ppl = math.exp(running_loss / max(1, running_tokens))
        val_ppl = evaluate_ppl(model, val_loader, device, pad_idx)
        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(
            f"[lm] epoch {epoch:02d}/{cfg.epochs} | lr {lr_now:.2e} | "
            f"train ppl {train_ppl:.2f} | val ppl {val_ppl:.2f}"
        )

        if val_ppl < best.best_val_ppl:
            best.best_val_ppl = val_ppl
            best.best_epoch = epoch
            torch.save(
                {"model_state": model.state_dict(), "val_ppl": val_ppl, "epoch": epoch},
                str(ckpt_path),
            )
            logger.info(f"[lm]   \u2198 new best val ppl {val_ppl:.2f}, ckpt -> {Path(ckpt_path).name}")
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                best.stopped_at = epoch
                logger.info(f"[lm] early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

    logger.info(f"[lm] best val ppl {best.best_val_ppl:.2f} @ epoch {best.best_epoch}")
    return best
