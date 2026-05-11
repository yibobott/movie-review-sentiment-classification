"""Stratified train/val split + DataLoader builders."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import SenDataset


def stratified_train_val_split(labels: np.ndarray, val_ratio: float, seed: int):
    """Small dependency-free stratified split for binary labels."""
    rng = np.random.RandomState(seed)
    train_parts = []
    val_parts = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_loaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
    word_dropout: float = 0.0,
) -> tuple[DataLoader, DataLoader]:
    tr = DataLoader(
        SenDataset(X_train, y_train, word_dropout=word_dropout),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    va = DataLoader(
        SenDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return tr, va
