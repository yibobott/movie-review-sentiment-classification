"""ULMFiT-style two-phase fine-tuning: head-only warmup then disc-LR full fine-tune."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from engine.trainer import BestMetrics, raw_ckpt_path, train
from pipeline.optim import build_discriminative_optimizer
from utils.config import Config


def run_initial_phases(
    cfg: Config,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    ckpt: Path,
    logger: logging.Logger,
    lm_loaded: bool,
) -> BestMetrics:
    """Run Phase A (head-only warmup, optional) followed by Phase B (full fine-tune).

    Phase A is active iff: LM was loaded AND disc-LR is on AND
    ``freeze_body_epochs > 0``. Otherwise this is a single Phase B.
    The best model state is loaded into ``model`` before returning.
    """
    use_two_phase = (
        lm_loaded
        and cfg.train.use_discriminative_lr
        and cfg.train.freeze_body_epochs > 0
    )
    if use_two_phase:
        for p in model.embedding.parameters():
            p.requires_grad = False
        for p in model.lstm.parameters():
            p.requires_grad = False
        head_params = [p for p in model.parameters() if p.requires_grad]
        warmup_optim = torch.optim.AdamW(
            head_params,
            lr=cfg.train.lr_head_warmup,
            weight_decay=cfg.train.weight_decay,
        )
        logger.info(
            f"[freeze] phase A: head-only warmup, "
            f"epochs={cfg.train.freeze_body_epochs}, "
            f"lr={cfg.train.lr_head_warmup:.1e}, "
            f"trainable={sum(p.numel() for p in head_params)}"
        )
        train(
            model, train_loader, val_loader, cfg.train, device,
            ckpt_path=ckpt, logger=logger, tag="frozen",
            optimizer=warmup_optim,
            epochs_override=cfg.train.freeze_body_epochs,
        )
        # When freeze_body_epochs <= ema_warmup_epochs, the EMA ckpt is never
        # written (EMA still warming up); fall back to the raw-best ckpt.
        phase_a_ckpt = ckpt if ckpt.exists() else raw_ckpt_path(ckpt)
        state = torch.load(phase_a_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        logger.info(f"[freeze] restored phase A best from {phase_a_ckpt.name}")
        for p in model.parameters():
            p.requires_grad = True
        logger.info("[freeze] phase A done; unfroze all params for phase B")

    # Phase B (or single-phase init when no LM / no freeze)
    init_optimizer = None
    if lm_loaded and cfg.train.use_discriminative_lr:
        init_optimizer = build_discriminative_optimizer(model, cfg.train, logger)
    init_epochs = cfg.train.epochs - (cfg.train.freeze_body_epochs if use_two_phase else 0)
    best = train(
        model, train_loader, val_loader, cfg.train, device,
        ckpt_path=ckpt, logger=logger, tag="init",
        optimizer=init_optimizer,
        epochs_override=init_epochs,
    )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    return best
