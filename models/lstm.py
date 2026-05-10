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


class MultiHeadAttentionPool(nn.Module):
    """Learned-query multi-head attention pooling over time.

    NOT a transformer block: there is no self-attention (Q/K/V from x),
    no feed-forward sublayer, no residual / layernorm stack. Each head
    owns a *learned* query vector that scores time steps via additive
    attention; the head output is the attention-weighted sum over time.
    Heads are concatenated and linearly projected back to ``dim``. This is
    a strict generalization of ``AttentionPool`` (K=1 collapses to it).
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # Per-head additive attention: project x -> head_dim, score with learned query.
        self.proj = nn.Linear(dim, dim)
        # Each head has its own learned query vector of size head_dim.
        self.query = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.1)
        # Output projection back to ``dim`` after head concat.
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = torch.tanh(self.proj(x))                           # (B, T, D)
        h = h.view(B, T, self.n_heads, self.head_dim)          # (B, T, H, d)
        # scores[b, t, k] = <h[b, t, k, :], query[k, :]>
        scores = torch.einsum("bthd,hd->bth", h, self.query)   # (B, T, H)
        scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)         # (B, T, H, 1)
        x_h = x.view(B, T, self.n_heads, self.head_dim)        # (B, T, H, d)
        head_out = (x_h * w).sum(dim=1)                        # (B, H, d)
        concat = head_out.reshape(B, D)                        # (B, D)
        return self.out_proj(concat)


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
        attn_heads: int = 4,
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
        # Pool selection. ``mhattn`` is the multi-head learned-query pool;
        # ``attn`` is the original single-head additive pool. Both can be
        # combined with max/mean by listing them in the pool string.
        use_mhattn = "mhattn" in pool
        use_attn = (not use_mhattn) and ("attn" in pool)
        if use_mhattn:
            self.attn = MultiHeadAttentionPool(out_dim, n_heads=attn_heads)
        elif use_attn:
            self.attn = AttentionPool(out_dim)
        else:
            self.attn = None

        has_attn = use_mhattn or use_attn
        n_pools = (1 if has_attn else 0) + (1 if "max" in pool else 0) + (1 if "mean" in pool else 0)
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
