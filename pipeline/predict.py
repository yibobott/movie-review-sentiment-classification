"""Final inference: load EMA + RAW best, write the three submission CSVs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import SenDataset
from engine.inference import predict_probs, save_predictions
from engine.trainer import raw_ckpt_path


def write_submissions(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    test_ids: Sequence[str],
    *,
    ckpt: Path,
    run_dir: Path,
    inference_batch_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Save three submission files under ``run_dir``:

    * ``predict.csv``     — primary submission, sourced from EMA-best ckpt
    * ``predict_ema.csv`` — exact copy of predict.csv (audit trail)
    * ``predict_raw.csv`` — only if a separate raw-best ckpt was written

    Two ckpts are written during training; we predict on Kaggle with the EMA
    one by default. ``predict_raw.csv`` is for post-hoc comparison only --
    NOT for ensembling. The user submits exactly one CSV per Kaggle slot.
    """
    test_loader = DataLoader(
        SenDataset(X_test), batch_size=inference_batch_size,
        shuffle=False, num_workers=0,
    )

    # EMA-best inference
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    probs_ema = predict_probs(model, test_loader, device)
    save_predictions(test_ids, probs_ema, run_dir / "predict.csv", logger=logger)
    save_predictions(test_ids, probs_ema, run_dir / "predict_ema.csv", logger=logger)
    np.save(run_dir / "test_probs_ema.npy", probs_ema)

    # Raw-best inference (if a separate raw ckpt exists)
    raw_ckpt = raw_ckpt_path(ckpt)
    if raw_ckpt.exists():
        state = torch.load(raw_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        probs_raw = predict_probs(model, test_loader, device)
        save_predictions(test_ids, probs_raw, run_dir / "predict_raw.csv", logger=logger)
        np.save(run_dir / "test_probs_raw.npy", probs_raw)
    else:
        logger.info("raw-best ckpt not found; skipping predict_raw.csv")
