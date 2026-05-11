"""Optimizer construction (discriminative learning rates)."""
from __future__ import annotations

import logging

import torch

from utils.config import TrainConfig


def build_discriminative_optimizer(
    model: torch.nn.Module,
    cfg_train: TrainConfig,
    logger: logging.Logger,
) -> torch.optim.Optimizer:
    """Build AdamW with three LR groups: embedding / lstm / head.

    The split is purely for fine-tuning *after* LM transfer: pretrained
    parameters get a smaller LR, randomly-initialized head gets the largest.
    """
    head_modules = []
    for name in ("attn", "feat_norm", "classifier"):
        m = getattr(model, name, None)
        if m is not None:
            head_modules.append((name, m))

    embedding_params = list(model.embedding.parameters())
    lstm_params = list(model.lstm.parameters())
    head_params: list[torch.nn.Parameter] = []
    for _, m in head_modules:
        head_params.extend(list(m.parameters()))

    # Sanity: cover every trainable parameter exactly once.
    seen = {id(p) for p in embedding_params + lstm_params + head_params}
    leftover = [p for p in model.parameters() if id(p) not in seen and p.requires_grad]
    if leftover:
        head_params.extend(leftover)
        logger.info(
            f"[disc-lr] {len(leftover)} unrecognized trainable params lumped into head group"
        )

    param_groups = [
        {"params": [p for p in embedding_params if p.requires_grad], "lr": cfg_train.lr_embedding},
        {"params": [p for p in lstm_params      if p.requires_grad], "lr": cfg_train.lr_lstm},
        {"params": [p for p in head_params      if p.requires_grad], "lr": cfg_train.lr_head},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg_train.weight_decay)
    logger.info(
        f"[disc-lr] embedding lr={cfg_train.lr_embedding:.1e} ({sum(p.numel() for p in embedding_params)} params), "
        f"lstm lr={cfg_train.lr_lstm:.1e} ({sum(p.numel() for p in lstm_params)} params), "
        f"head lr={cfg_train.lr_head:.1e} ({sum(p.numel() for p in head_params)} params)"
    )
    return optimizer
