"""Self-training loop: iterative pseudo-labeling on the unlabeled pool."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import PseudoLabeledDataset, SenDataset
from engine.inference import predict_probs
from engine.trainer import BestMetrics, promote_round_best, train
from utils.config import Config


def pick_pseudo(
    probs: np.ndarray,
    pos_th: float,
    neg_th: float,
    cap: int,
    balance: bool = True,
):
    """Select high-confidence pseudo-labeled candidates.

    When ``balance`` is True, take the most confident ``cap // 2`` positives
    and the most confident ``cap // 2`` negatives. This prevents the model's
    existing class bias from being amplified during self-training. Otherwise
    take the top ``cap`` overall, ranked by margin away from 0.5.
    Labels are always derived from which threshold side the sample came from,
    not from ``probs >= 0.5`` (so a ``neg_th=0.05`` sample is always labeled 0).
    """
    pos_idx = np.where(probs >= pos_th)[0]
    neg_idx = np.where(probs <= neg_th)[0]
    pos_conf = probs[pos_idx]
    neg_conf = 1.0 - probs[neg_idx]

    if balance:
        n_each = cap // 2
        if len(pos_idx) > n_each:
            pos_idx = pos_idx[np.argsort(-pos_conf)[:n_each]]
        if len(neg_idx) > n_each:
            neg_idx = neg_idx[np.argsort(-neg_conf)[:n_each]]
    else:
        all_idx = np.concatenate([pos_idx, neg_idx])
        is_pos = np.concatenate([
            np.ones(len(pos_idx), dtype=bool),
            np.zeros(len(neg_idx), dtype=bool),
        ])
        all_conf = np.concatenate([pos_conf, neg_conf])
        if len(all_idx) > cap:
            keep = np.argsort(-all_conf)[:cap]
            all_idx = all_idx[keep]
            is_pos = is_pos[keep]
        pos_idx = all_idx[is_pos]
        neg_idx = all_idx[~is_pos]

    idx = np.concatenate([pos_idx, neg_idx])
    labels = np.concatenate([
        np.ones(len(pos_idx), dtype=np.int64),
        np.zeros(len(neg_idx), dtype=np.int64),
    ])
    return idx, labels


def run_self_training(
    cfg: Config,
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_unlabel: torch.Tensor,
    val_loader: DataLoader,
    device: torch.device,
    *,
    run_dir: Path,
    ckpt: Path,
    best: BestMetrics,
    logger: logging.Logger,
) -> tuple[BestMetrics, list[int]]:
    """Iteratively pseudo-label and fine-tune. Updates ``best`` in-place via the
    promotion helper. Returns (best, pseudo_added_per_round).
    """
    pseudo_added: list[int] = []
    st = cfg.self_training
    if not (st.enable and len(X_unlabel) > 0):
        return best, pseudo_added

    remaining = np.arange(len(X_unlabel))
    for r in range(1, st.rounds + 1):
        if len(remaining) == 0:
            break
        un_loader = DataLoader(
            SenDataset(X_unlabel[remaining]),
            batch_size=cfg.inference.batch_size, shuffle=False, num_workers=0,
        )
        probs = predict_probs(model, un_loader, device)
        idx, pseudo_y = pick_pseudo(
            probs, st.pos_threshold, st.neg_threshold, st.max_pseudo_per_round,
            balance=st.balance_pseudo,
        )
        if len(idx) == 0:
            logger.info(f"[self-train r{r}] no confident samples, stop")
            break
        n_pos = int(pseudo_y.sum())
        n_neg = len(idx) - n_pos
        # Imbalance circuit breaker: if one class is empty (or <10% of the other),
        # the model's logit distribution has drifted. Continuing would amplify the
        # bias via a feedback loop (seen in run 20260510_024824 r2 -> pos=0).
        if n_pos == 0 or n_neg == 0 or min(n_pos, n_neg) * 10 < max(n_pos, n_neg):
            logger.info(
                f"[self-train r{r}] pseudo class imbalance (pos={n_pos}, neg={n_neg}); "
                f"stop self-training to avoid feedback loop"
            )
            break
        chosen_global = remaining[idx]
        pseudo_X = X_unlabel[chosen_global]
        pseudo_y_t = torch.from_numpy(pseudo_y)
        logger.info(
            f"[self-train r{r}] added {len(idx)} pseudo "
            f"(pos={n_pos}, neg={n_neg}) from {len(remaining)} candidates"
        )
        pseudo_added.append(int(len(idx)))

        merged = PseudoLabeledDataset(
            X_train, y_train, pseudo_X, pseudo_y_t,
            word_dropout=cfg.train.word_dropout,
        )
        tr_loader = DataLoader(
            merged, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0,
        )

        # Always start each fine-tune round from the current global best,
        # so a regressed previous round cannot poison the next round's init.
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])

        round_ckpt = run_dir / f"ckpt_self_train_r{r}.pt"
        round_best = train(
            model, tr_loader, val_loader, cfg.train, device,
            ckpt_path=round_ckpt, logger=logger,
            epochs_override=st.finetune_epochs,
            lr_override=st.finetune_lr,
            tag=f"self-train-r{r}",
        )
        promote_round_best(
            best, round_best, ckpt, round_ckpt,
            logger=logger, tag=f"self-train-r{r}",
        )
        # Reload global EMA-best so later rounds train from the most stable init.
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state"])

        mask = np.ones(len(remaining), dtype=bool)
        mask[idx] = False
        remaining = remaining[mask]

    return best, pseudo_added
