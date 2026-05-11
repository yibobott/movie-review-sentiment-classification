"""Forward LSTM Language Model used for unsupervised pretraining.

* Unidirectional 2-layer LSTM (autoregressive next-token prediction).
* Tied input embedding <-> output projection via a small ``adapter`` linear
  (since ``embed_dim != hidden_dim``). This both regularizes and saves
  ~22M parameters versus an untied projection.
* Vocab size at LM time is ``V_cls + 1`` — the trailing row is the EOS
  token. Weight transfer to the classifier later slices ``[:V_cls]``.
* Embedding can be warm-started from a Word2Vec matrix; the EOS row keeps
  its random init.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .regularization import LockedDropout, maybe_wrap_lstm_with_weight_drop


class LSTMLanguageModel(nn.Module):
    """Forward (causal) LSTM-LM.

    Forward pass
    ------------
    ``x: (B, T)`` token ids -> ``logits: (B, T, V)`` raw scores. Use with
    ``F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=PAD_IDX)``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.4,
        embed_dropout: float = 0.3,
        tie_weights: bool = True,
        embedding_init: Optional[torch.Tensor] = None,
        pad_idx: int = 0,
        locked_dropout: float = 0.0,
        weight_drop: float = 0.0,
    ) -> None:
        super().__init__()
        # NOTE: ``vocab_size`` should be ``classifier_vocab_size + 1`` (extra row for EOS).
        # The trailing row is the EOS token; classifier transfer slices it off.
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_init is not None:
            # ``embedding_init`` typically has V_cls rows (Word2Vec), which is
            # one less than our V_cls+1 LM vocab. Copy what we have; EOS row
            # stays at default (small random init).
            n_init = int(embedding_init.size(0))
            if n_init > self.embedding.weight.size(0):
                raise ValueError(
                    f"embedding_init has {n_init} rows but LM embedding only has "
                    f"{self.embedding.weight.size(0)}"
                )
            with torch.no_grad():
                self.embedding.weight.data[:n_init].copy_(embedding_init)

        self.embed_dropout = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        # AWD-LSTM-style DropConnect on recurrent weights (no-op at 0).
        self.lstm = maybe_wrap_lstm_with_weight_drop(self.lstm, weight_drop)
        # Variational dropout on LSTM output (time-shared mask). When > 0,
        # this REPLACES output_dropout below; both default to no-op at 0.
        self.locked_dropout = LockedDropout(locked_dropout)
        self.output_dropout = nn.Dropout(dropout)
        # Adapter projects LSTM output (hidden_dim) back to embed_dim so we
        # can tie weights with the input embedding.
        self.adapter = nn.Linear(hidden_dim, embed_dim)
        # Output projection. With weight tying we reuse the embedding matrix
        # transposed; bias is False to avoid an extra parameter set.
        self.proj = nn.Linear(embed_dim, vocab_size, bias=False)
        if tie_weights:
            self.proj.weight = self.embedding.weight  # share storage

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed_dropout(self.embedding(x))
        h, _ = self.lstm(emb)
        h = self.locked_dropout(h)
        h = self.output_dropout(h)
        h = self.adapter(h)
        return self.proj(h)
