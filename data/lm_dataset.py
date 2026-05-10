"""LM corpus assembly + BPTT dataset.

Design notes (see LSTM_LM_DESIGN.md §4)
---------------------------------------
* **EOS != PAD**: PAD has ``padding_idx`` semantics (zero embedding, ignored by
  loss). Reusing it as EOS would silently train EOS at zero loss signal. We
  therefore extend the LM vocab by exactly one row at index ``V_cls`` and use
  that as EOS. The classifier never sees this row (transfer slices ``[:V_cls]``).
* **Document-level split before flatten**: Splitting after flatten lets a single
  review straddle train/val, leaking semantic context. We split the document
  list first, then flatten train and val independently.
* **Labeled fully excluded**: classifier val docs are inside the labeled split,
  so we omit *all* labeled docs from the LM corpus. This costs ~0.1-0.2% acc
  but keeps val numbers comparable to historical (no-LM) runs.
* **Discontinuous BPTT**: each chunk starts fresh (no hidden state carry-over
  across batches). Loss is slightly higher but code stays simple.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocess import PAD_IDX, UNK_IDX


def encode_doc(tokens: Sequence[str], word2idx: dict, eos_idx: int) -> List[int]:
    """Encode a single document and append EOS. UNK fallback for OOV."""
    out = [word2idx.get(w, UNK_IDX) for w in tokens]
    out.append(eos_idx)
    return out


def split_docs_train_val(
    docs: Sequence[Sequence[str]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Document-level random split. Returns (train_indices, val_indices)."""
    n = len(docs)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_ratio)))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return train_idx, val_idx


def build_lm_corpus(
    unlabeled_tokens: Sequence[Sequence[str]],
    test_tokens: Sequence[Sequence[str]],
    labeled_tokens: Sequence[Sequence[str]],
    word2idx: dict,
    eos_idx: int,
    val_ratio: float,
    seed: int,
    include_labeled: bool = False,
    include_test: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build flattened LM train/val token sequences.

    Returns (train_seq, val_seq, info_dict).

    The val split is taken from ``unlabeled_tokens`` only (per design §4.3) so
    classifier val docs (inside labeled) are never seen by the LM.
    """
    # 1) Split unlabeled at the document level (val source).
    un_train_idx, un_val_idx = split_docs_train_val(unlabeled_tokens, val_ratio, seed)

    # 2) Encode each split's docs to id streams with trailing EOS.
    def encode_many(docs: Sequence[Sequence[str]]) -> List[int]:
        out: List[int] = []
        for doc in docs:
            out.extend(encode_doc(doc, word2idx, eos_idx))
        return out

    val_doc_lists = [unlabeled_tokens[i] for i in un_val_idx]
    val_seq = encode_many(val_doc_lists)

    train_doc_lists: List[Sequence[str]] = [unlabeled_tokens[i] for i in un_train_idx]
    if include_test:
        train_doc_lists = list(train_doc_lists) + list(test_tokens)
    if include_labeled:
        # Only here for completeness — the simplified design keeps this False.
        train_doc_lists = list(train_doc_lists) + list(labeled_tokens)
    train_seq = encode_many(train_doc_lists)

    info = {
        "n_unlabeled_train_docs": len(un_train_idx),
        "n_unlabeled_val_docs": len(un_val_idx),
        "n_test_docs": len(test_tokens) if include_test else 0,
        "n_labeled_docs": len(labeled_tokens) if include_labeled else 0,
        "n_train_tokens": len(train_seq),
        "n_val_tokens": len(val_seq),
        "eos_idx": eos_idx,
    }
    return (
        np.asarray(train_seq, dtype=np.int64),
        np.asarray(val_seq, dtype=np.int64),
        info,
    )


class LMBPTTDataset(Dataset):
    """Discontinuous BPTT chunks over a single flattened token stream.

    Each item is ``(x, y)`` where ``y`` is ``x`` shifted by one token. Chunks
    starting near the tail of the stream are dropped (so we never produce a
    short chunk that would change the effective batch size).
    """

    def __init__(self, seq: np.ndarray, bptt_len: int):
        if seq.ndim != 1:
            raise ValueError(f"seq must be 1-D, got shape {seq.shape}")
        if bptt_len < 2:
            raise ValueError(f"bptt_len must be >= 2, got {bptt_len}")
        self.seq = seq
        self.bptt_len = bptt_len
        # We need ``bptt_len + 1`` tokens to form (x, y) of length bptt_len.
        self.n_chunks = max(0, (len(seq) - 1) // bptt_len)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.bptt_len
        x = self.seq[start: start + self.bptt_len]
        y = self.seq[start + 1: start + 1 + self.bptt_len]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
