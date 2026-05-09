"""Exponential moving average of model weights.

EMA produces a single model's weight trajectory averaged over training steps.
The evaluated / saved network is still **one** model, so this is compatible
with the HW4 rule that disallows ensemble methods.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)
