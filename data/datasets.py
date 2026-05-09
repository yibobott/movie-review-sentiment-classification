"""Torch datasets for labeled / unlabeled / test splits."""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset


def _apply_word_dropout(x: torch.Tensor, prob: float, pad_idx: int, unk_idx: int) -> torch.Tensor:
    """Randomly replace non-pad tokens with UNK. Cheap augmentation that
    regularises the embedding / LSTM without needing external NLP tools.
    """
    if prob <= 0.0:
        return x
    x = x.clone()
    mask = (x != pad_idx) & (torch.rand(x.shape) < prob)
    x[mask] = unk_idx
    return x


class SenDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        word_dropout: float = 0.0,
        pad_idx: int = 0,
        unk_idx: int = 1,
    ):
        self.X = X
        self.y = y
        self.word_dropout = float(word_dropout)
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.word_dropout > 0.0:
            x = _apply_word_dropout(x, self.word_dropout, self.pad_idx, self.unk_idx)
        if self.y is None:
            return x
        return x, self.y[idx]


class PseudoLabeledDataset(Dataset):
    """Concat of a labeled dataset and a pseudo-labeled tensor pair."""

    def __init__(
        self,
        X_real: torch.Tensor,
        y_real: torch.Tensor,
        X_pseudo: torch.Tensor,
        y_pseudo: torch.Tensor,
        word_dropout: float = 0.0,
        pad_idx: int = 0,
        unk_idx: int = 1,
    ):
        self.X = torch.cat([X_real, X_pseudo], dim=0)
        self.y = torch.cat([y_real, y_pseudo], dim=0)
        self.word_dropout = float(word_dropout)
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.word_dropout > 0.0:
            x = _apply_word_dropout(x, self.word_dropout, self.pad_idx, self.unk_idx)
        return x, self.y[idx]
