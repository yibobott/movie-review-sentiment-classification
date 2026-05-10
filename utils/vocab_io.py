"""Vocab persistence and integrity checks (defense against silent w2v re-training).

Why this exists
---------------
Two independent gensim Word2Vec runs over the *same* corpus can produce
different vocabulary orderings due to multi-thread non-determinism. If LM
pretraining and classifier training each retrain w2v, the LM weights would
silently transfer to the wrong rows in the classifier embedding —
"shape OK, semantics misaligned". To guard against this we:

  1. Persist the LM-time ``idx2word`` list verbatim (JSON).
  2. Persist a deterministic md5 of the joined vocab string.
  3. At classifier training time, reload w2v from the *same* file as LM and
     verify both the hash and the full ``idx2word`` list match.

Two layers of defense (hash + full content) make accidental mismatch
detectable even on the astronomically-rare hash collision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence


def vocab_hash(idx2word: Sequence[str]) -> str:
    """Return a deterministic md5 over the ordered vocab list.

    The hash includes the position (via newline separator), so a permutation
    of the same word set produces a different hash.
    """
    h = hashlib.md5()
    for w in idx2word:
        h.update(w.encode("utf-8"))
        h.update(b"\n")  # position-sensitive separator
    return h.hexdigest()


def dump_vocab(idx2word: Sequence[str], path: str | Path) -> str:
    """Persist ``idx2word`` to ``path`` as JSON. Returns the hash for convenience."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(idx2word), f, ensure_ascii=False)
    return vocab_hash(idx2word)


def load_vocab(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def verify_vocab(idx2word: Sequence[str], stored_path: str | Path,
                 expected_hash: str | None = None) -> None:
    """Raise if reloaded vocab disagrees with ``stored_path`` (or ``expected_hash``).

    Performs both checks (hash + full element-wise) — neither alone is enough
    because hashes can collide and subset matches can hide tail differences.
    """
    stored = load_vocab(stored_path)
    if len(stored) != len(idx2word):
        raise ValueError(
            f"vocab size mismatch: stored={len(stored)}, current={len(idx2word)} "
            f"(stored_path={stored_path})"
        )
    # Element-wise compare — single mismatch is fatal.
    for i, (a, b) in enumerate(zip(stored, idx2word)):
        if a != b:
            raise ValueError(
                f"vocab order mismatch at index {i}: stored={a!r}, current={b!r}. "
                f"This typically means w2v was retrained instead of cached. "
                f"Force preprocess.w2v_cache_path to the same w2v.model used by LM."
            )
    # Belt-and-suspenders hash check.
    h_now = vocab_hash(idx2word)
    h_stored = vocab_hash(stored)
    if h_now != h_stored:
        raise ValueError(
            f"vocab hash mismatch despite element-wise equality (this should be unreachable): "
            f"now={h_now}, stored={h_stored}"
        )
    if expected_hash is not None and h_now != expected_hash:
        raise ValueError(
            f"vocab hash {h_now} does not match expected {expected_hash} "
            f"(LM ckpt was trained against a different vocab)"
        )
