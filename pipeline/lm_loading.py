"""LM checkpoint loading: vocab integrity check + weight transfer."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from utils.config import Config
from utils.vocab_io import vocab_hash, verify_vocab
from utils.weight_transfer import transfer_lm_to_classifier


def _load_and_validate_lm_state(
    lm_ckpt_path: Path,
    cfg: Config,
    vocab,
    logger: logging.Logger,
    *,
    role: str,
) -> dict:
    """Load an LM ckpt and assert vocab + arch alignment with the classifier."""
    if not lm_ckpt_path.exists():
        raise FileNotFoundError(f"LM ckpt does not exist: {lm_ckpt_path}")
    lm_run_dir = lm_ckpt_path.parent
    vocab_json = lm_run_dir / "idx2word.json"
    if not vocab_json.exists():
        raise FileNotFoundError(
            f"idx2word.json not found next to LM ckpt ({lm_run_dir}); "
            f"re-run pretrain_lm.py with the current code."
        )
    state = torch.load(lm_ckpt_path, map_location="cpu")
    expected_hash = state.get("vocab_hash")
    verify_vocab(vocab.idx2word, vocab_json, expected_hash=expected_hash)
    logger.info(
        f"[lm-load:{role}] vocab integrity OK (hash={vocab_hash(vocab.idx2word)[:12]}\u2026, "
        f"V_cls={len(vocab)})"
    )
    if state.get("hidden_dim") and state["hidden_dim"] != cfg.model.hidden_dim:
        raise ValueError(
            f"LM hidden_dim={state['hidden_dim']} != classifier hidden_dim="
            f"{cfg.model.hidden_dim}. Re-pretrain or change config."
        )
    if state.get("embed_dim") and state["embed_dim"] != vocab.embedding_matrix.size(1):
        raise ValueError(
            f"LM embed_dim={state['embed_dim']} != classifier embed_dim="
            f"{vocab.embedding_matrix.size(1)}"
        )
    return state


def maybe_load_lm_ckpt(
    cfg: Config,
    vocab,
    model: torch.nn.Module,
    logger: logging.Logger,
    lm_ckpt_path: Optional[Path],
    lm_bw_ckpt_path: Optional[Path] = None,
) -> bool:
    """Validate vocab alignment and transfer LM weights into ``model``.

    ``lm_ckpt_path`` is the FORWARD LM ckpt (from --lm or cfg.lm.ckpt_path).
    ``lm_bw_ckpt_path`` is an optional BACKWARD LM ckpt (from --lm-bw); when
    given, its weights are transferred into the classifier reverse direction.
    Returns True iff at least the forward LM was loaded.
    """
    if lm_ckpt_path is None:
        if lm_bw_ckpt_path is not None:
            raise ValueError("--lm-bw given without --lm; backward LM only is unsupported.")
        return False
    state = _load_and_validate_lm_state(lm_ckpt_path, cfg, vocab, logger, role="fwd")
    if state.get("direction") not in (None, "forward"):
        logger.info(
            f"[lm-load:fwd] WARNING: ckpt direction='{state.get('direction')}' "
            f"(expected 'forward'); proceeding anyway."
        )

    bw_state = None
    if lm_bw_ckpt_path is not None:
        bw_full = _load_and_validate_lm_state(lm_bw_ckpt_path, cfg, vocab, logger, role="bw")
        if bw_full.get("direction") != "backward":
            logger.info(
                f"[lm-load:bw] WARNING: ckpt direction='{bw_full.get('direction')}' "
                f"(expected 'backward'); proceeding anyway."
            )
        bw_state = bw_full["model_state"]

    transfer_lm_to_classifier(
        state["model_state"], model,
        lm_bw_state=bw_state, logger=logger,
    )
    logger.info(
        f"[lm-load] loaded forward LM from {lm_ckpt_path} "
        f"(val_ppl={state.get('val_ppl', 'NA')}, epoch={state.get('epoch', 'NA')})"
    )
    if lm_bw_ckpt_path is not None:
        logger.info(f"[lm-load] loaded backward LM from {lm_bw_ckpt_path}")
    return True
