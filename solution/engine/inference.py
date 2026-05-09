"""Inference helpers: probabilities, predictions, predict.csv writer."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class RunSummary:
    best_val_acc: float
    pseudo_added_per_round: List[int]


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def save_predictions(ids: Sequence[str], probs: np.ndarray, out_csv: str | Path,
                     logger: logging.Logger | None = None) -> None:
    labels = (probs >= 0.5).astype(int)
    # Match sample_submission.csv header exactly: "id,label" (singular)
    df = pd.DataFrame({"id": list(ids), "label": labels})
    df.to_csv(out_csv, index=False)
    if logger:
        pos = int(labels.sum())
        logger.info(f"saved predictions -> {out_csv} (n={len(df)}, pos={pos}, neg={len(df) - pos})")
