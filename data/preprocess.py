"""Data loading, Word2Vec training, vocabulary / embedding building."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec

from utils.misc import tokenize


PAD_IDX = 0
UNK_IDX = 1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ---------- Raw file loaders ----------

def load_labeled_csv(path: str | Path, lowercase: bool) -> Tuple[List[List[str]], np.ndarray]:
    df = pd.read_csv(path)
    texts = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].to_numpy().astype(np.int64)
    tokens = [tokenize(t, lowercase=lowercase) for t in texts]
    return tokens, labels


def load_unlabeled_csv(path: str | Path, lowercase: bool) -> List[List[str]]:
    df = pd.read_csv(path)
    texts = df.iloc[:, 0].astype(str).tolist()
    return [tokenize(t, lowercase=lowercase) for t in texts]


def load_test_csv(path: str | Path, lowercase: bool) -> Tuple[List[List[str]], List[str]]:
    """Test file format: id,text — return tokens and ids."""
    df = pd.read_csv(path, header=None)
    # First column may be id, second column text
    if df.shape[1] >= 2:
        ids = df.iloc[:, 0].astype(str).tolist()
        texts = df.iloc[:, 1].astype(str).tolist()
    else:
        texts = df.iloc[:, 0].astype(str).tolist()
        ids = [str(i) for i in range(len(texts))]
    # Drop header row if present
    if ids and ids[0].lower() == "id":
        ids = ids[1:]
        texts = texts[1:]
    tokens = [tokenize(t, lowercase=lowercase) for t in texts]
    return tokens, ids


# ---------- Word2Vec ----------

def train_word2vec(
    corpus: Sequence[Sequence[str]],
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    sg: int,
    negative: int,
    epochs: int,
    sample: float = 1e-4,
    logger: Optional[logging.Logger] = None,
) -> Word2Vec:
    if logger:
        logger.info(
            f"Training Word2Vec: size={vector_size}, window={window}, min_count={min_count}, "
            f"sg={sg}, negative={negative}, epochs={epochs}, sample={sample}, "
            f"sentences={len(corpus)}"
        )
    model = Word2Vec(
        sentences=list(corpus),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        epochs=epochs,
        sample=sample,
    )
    return model


# ---------- Vocab / Embedding ----------

class Vocab:
    """Vocab with fixed PAD=0, UNK=1 before Word2Vec words."""

    def __init__(self, w2v: Word2Vec):
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word = [PAD_TOKEN, UNK_TOKEN]
        for w in w2v.wv.key_to_index:
            self.word2idx[w] = len(self.idx2word)
            self.idx2word.append(w)
        self.embedding_matrix = self._build_embedding(w2v)

    def _build_embedding(self, w2v: Word2Vec) -> torch.Tensor:
        dim = w2v.vector_size
        mat = np.zeros((len(self.idx2word), dim), dtype=np.float32)
        # PAD stays zero; UNK random
        rng = np.random.RandomState(0)
        mat[UNK_IDX] = rng.uniform(-0.25, 0.25, size=dim).astype(np.float32)
        for i in range(2, len(self.idx2word)):
            mat[i] = w2v.wv[self.idx2word[i]]
        return torch.from_numpy(mat)

    def __len__(self) -> int:
        return len(self.idx2word)

    def encode(
        self,
        sentences: Sequence[Sequence[str]],
        sen_len: int,
        head_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Encode to fixed length.

        When a review is longer than ``sen_len``, keep ``head_ratio`` portion
        from the start and the remainder from the end. Reviews often put the
        verdict in the last sentences, so preserving the tail helps.
        """
        unk = UNK_IDX
        pad = PAD_IDX
        head_n = max(0, min(sen_len, int(round(sen_len * head_ratio))))
        tail_n = sen_len - head_n
        arr = np.full((len(sentences), sen_len), pad, dtype=np.int64)
        for i, tokens in enumerate(sentences):
            if len(tokens) > sen_len and tail_n > 0:
                kept = list(tokens[:head_n]) + list(tokens[-tail_n:])
            else:
                kept = list(tokens[:sen_len])
            ids = [self.word2idx.get(w, unk) for w in kept]
            arr[i, : len(ids)] = ids
        return torch.from_numpy(arr)
