"""LSTM-like sentiment classifier (self-designed; single model)."""
from __future__ import annotations

import torch
from torch import nn


class AttentionPool(nn.Module):
    """Additive attention pooling over time with masking."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) with 1 for real tokens
        scores = self.v(torch.tanh(self.proj(x))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (x * w).sum(dim=1)  # (B, D)


class LSTMClassifier(nn.Module):
    """BiLSTM + (attention, max, mean) pooling + MLP head.

    Outputs a raw logit (use BCEWithLogitsLoss); label 1 = positive.
    """

    def __init__(
        self,
        embedding: torch.Tensor,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.4,
        embed_dropout: float = 0.3,
        embed_noise_std: float = 0.0,
        bidirectional: bool = True,
        fix_embedding: bool = False,
        pad_idx: int = 0,
        pool: str = "attn_max_mean",
    ):
        super().__init__()
        vocab_size, embed_dim = embedding.size()
        self.pad_idx = pad_idx
        self.pool_type = pool
        self.embed_noise_std = float(embed_noise_std)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        with torch.no_grad():
            self.embedding.weight.copy_(embedding)
        self.embedding.weight.requires_grad = not fix_embedding

        self.embed_dropout = nn.Dropout(embed_dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attn = AttentionPool(out_dim) if "attn" in pool else None

        n_pools = sum(p in pool for p in ("attn", "max", "mean"))
        fc_in = out_dim * n_pools
        self.feat_norm = nn.LayerNorm(fc_in)
        self.classifier = nn.Sequential(
            nn.Linear(fc_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mask = (inputs != self.pad_idx)  # (B, T)
        x = self.embedding(inputs)
        if self.training and self.embed_noise_std > 0:
            x = x + torch.randn_like(x) * self.embed_noise_std
        x = self.embed_dropout(x)
        x, _ = self.lstm(x)
        # Apply mask so pads don't influence pooling
        mask_f = mask.unsqueeze(-1).float()
        pools = []
        if self.attn is not None:
            pools.append(self.attn(x, mask))
        if "max" in self.pool_type:
            x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pools.append(x_masked.max(dim=1).values)
        if "mean" in self.pool_type:
            lengths = mask_f.sum(dim=1).clamp_min(1.0)
            pools.append((x * mask_f).sum(dim=1) / lengths)
        feat = torch.cat(pools, dim=-1) if len(pools) > 1 else pools[0]
        feat = self.feat_norm(feat)
        logit = self.classifier(feat).squeeze(-1)
        return logit
