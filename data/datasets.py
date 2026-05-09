"""Torch datasets for labeled / unlabeled / test splits."""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset


class SenDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class PseudoLabeledDataset(Dataset):
    """Concat of a labeled dataset and a pseudo-labeled tensor pair."""

    def __init__(self, X_real: torch.Tensor, y_real: torch.Tensor,
                 X_pseudo: torch.Tensor, y_pseudo: torch.Tensor):
        self.X = torch.cat([X_real, X_pseudo], dim=0)
        self.y = torch.cat([y_real, y_pseudo], dim=0)

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
