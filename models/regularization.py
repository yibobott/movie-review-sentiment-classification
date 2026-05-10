"""Self-implemented LSTM regularizers.

This module is hand-written (no external packages beyond torch) per HW4-Rules.
It contains two AWD-LSTM-style regularizers commonly applied to LSTM language
models and downstream classifiers:

* ``LockedDropout`` (a.k.a. variational dropout): a dropout mask that is shared
  across the time dimension. Each batch draws ONE mask of shape ``(B, 1, D)``
  and broadcasts it over all time steps. Standard ``nn.Dropout`` in contrast
  draws an independent mask at every time step, which is suboptimal for
  recurrent activations.

* ``WeightDrop``: DropConnect on the recurrent ``weight_hh_l*`` parameters of
  an ``nn.LSTM``. At each forward pass, a fresh dropout mask is applied to the
  raw hidden-to-hidden weight, and the dropped tensor is installed in-place
  on the wrapped module. The "raw" parameter is the trainable storage; the
  in-place dropped tensor is what cuDNN actually uses for the forward pass.

Both modules are no-ops at ``p == 0`` and at eval time (``module.eval()``).
"""
from __future__ import annotations

import warnings
from typing import Iterable, List

import torch
from torch import nn


class LockedDropout(nn.Module):
    """Variational dropout: time-shared mask broadcast over the T dimension.

    Args:
        p: Dropout probability. ``0.0`` makes this a no-op.

    Input/Output: ``(B, T, D)`` tensor (batch_first). The mask has shape
    ``(B, 1, D)`` so all T positions of a given (batch, feature) entry share
    the same Bernoulli draw.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"LockedDropout p must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        # mask shape (B, 1, D), broadcast over time
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.p)
        mask = mask.div_(1.0 - self.p)  # inverted dropout (preserve E[x])
        return x * mask

    def extra_repr(self) -> str:
        return f"p={self.p}"


class WeightDrop(nn.Module):
    """DropConnect on selected parameters of a wrapped module (typically nn.LSTM).

    Mechanism (AWD-LSTM, Merity et al. 2017):
      * For each name in ``weight_names``, the original ``nn.Parameter`` is
        de-registered from the wrapped module and re-registered under
        ``<name>_raw`` (still trainable).
      * Before every forward pass, dropout is applied to ``<name>_raw`` and
        the result is assigned back to ``<name>`` as a plain tensor (not a
        Parameter). This is what cuDNN reads.
      * Gradients flow through the dropped tensor back to ``<name>_raw``.

    Notes:
      * cuDNN's flatten-parameters cache is invalidated on every forward; we
        suppress its UserWarning since this is intentional.
      * State-dict keys for the dropped weights become ``<name>_raw``. The
        weight-transfer code (``utils/weight_transfer.py``) is aware of this
        and will look up the ``_raw`` variant when present.
    """

    def __init__(self, module: nn.Module, weight_names: Iterable[str], p: float = 0.0):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"WeightDrop p must be in [0, 1), got {p}")
        self.module = module
        self.weight_names: List[str] = list(weight_names)
        self.p = float(p)
        self._setup()

    def _setup(self) -> None:
        # Move each target parameter to a "<name>_raw" slot.
        for name in self.weight_names:
            if not hasattr(self.module, name):
                # name not present (e.g. only 1-layer LSTM but user listed l1) — skip
                continue
            w = getattr(self.module, name)
            # remove the original parameter so we can install a dropped tensor
            del self.module._parameters[name]
            self.module.register_parameter(name + "_raw", nn.Parameter(w.data))

    def _set_dropped_weights(self) -> None:
        for name in self.weight_names:
            raw = getattr(self.module, name + "_raw", None)
            if raw is None:
                continue
            if self.training and self.p > 0.0:
                w = nn.functional.dropout(raw, p=self.p, training=True)
            else:
                w = raw
            # install as plain attribute (NOT a Parameter) so cuDNN reads it
            setattr(self.module, name, w)

    def forward(self, *args, **kwargs):
        self._set_dropped_weights()
        # cuDNN warns when LSTM weights aren't contiguous in its preferred
        # layout; with WeightDrop we knowingly reassign weights every step.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        return f"p={self.p}, weights={self.weight_names}"

    # Expose common LSTM attributes so call sites that read e.g.
    # ``classifier.lstm.hidden_size`` keep working when the LSTM is wrapped.
    @property
    def hidden_size(self) -> int:
        return self.module.hidden_size  # type: ignore[attr-defined]

    @property
    def num_layers(self) -> int:
        return self.module.num_layers  # type: ignore[attr-defined]

    @property
    def bidirectional(self) -> bool:
        return self.module.bidirectional  # type: ignore[attr-defined]


def maybe_wrap_lstm_with_weight_drop(lstm: nn.LSTM, p: float) -> nn.Module:
    """Return ``WeightDrop(lstm, [...weight_hh_l*], p)`` if p > 0, else ``lstm``.

    No-op at ``p == 0`` so existing checkpoints / state_dict layouts remain
    untouched. The set of target parameter names is derived from the LSTM's
    ``num_layers`` (forward direction only; backward ``*_reverse`` weights
    are intentionally NOT dropped to keep behavior conservative).
    """
    if p <= 0.0:
        return lstm
    names = [f"weight_hh_l{i}" for i in range(lstm.num_layers)]
    if lstm.bidirectional:
        names += [f"weight_hh_l{i}_reverse" for i in range(lstm.num_layers)]
    return WeightDrop(lstm, names, p=p)
