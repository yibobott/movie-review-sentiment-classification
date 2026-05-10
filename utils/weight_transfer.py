"""LSTM-LM -> classifier BiLSTM weight transfer.

Strategy (see LSTM_LM_DESIGN.md \u00a72.3 and \u00a77)
---------------------------------------------
* **Embedding**: copy ``[:V_cls]`` (drop the trailing EOS row).
* **LSTM layer 0**: shapes match (``input_size = embed_dim`` for both LM and
  BiLSTM forward direction). Direct copy into the forward direction.
* **LSTM layer 1+**: BiLSTM input dim is ``2H`` (concat of fwd+bwd outputs),
  while LM input dim is ``H``. Place LM weight on the first half, zero the
  second half. Backward LSTM contribution to forward layer 1 starts at zero
  and gradually fills in during fine-tuning.
* **Backward direction (``*_reverse``)**: leave at random init (no backward
  LM in v1). Bias and ``weight_hh`` of the reverse direction stay random.
* **Adapter / proj**: not transferred (classifier doesn't need them).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch


def transfer_lm_to_classifier(
    lm_state: Dict[str, torch.Tensor],
    classifier: torch.nn.Module,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[str]]:
    """Copy LM weights into ``classifier`` in-place. Returns audit dict."""
    transferred: List[str] = []
    skipped: List[str] = []

    cs = classifier.state_dict()

    # ---- 1) Embedding: clip the LM EOS row ---------------------------------
    lm_emb = lm_state.get("embedding.weight")
    cls_emb = cs.get("embedding.weight")
    if lm_emb is not None and cls_emb is not None:
        v_cls = cls_emb.shape[0]
        if lm_emb.shape[0] < v_cls:
            skipped.append(
                f"embedding.weight (LM has {lm_emb.shape[0]} rows < classifier {v_cls})"
            )
        elif lm_emb.shape[1] != cls_emb.shape[1]:
            skipped.append(
                f"embedding.weight (embed_dim mismatch {lm_emb.shape[1]} vs {cls_emb.shape[1]})"
            )
        else:
            cls_emb.copy_(lm_emb[:v_cls])
            transferred.append(
                f"embedding.weight (clipped {lm_emb.shape[0]} -> {v_cls})"
            )

    # ---- 2) LSTM forward direction, all layers -----------------------------
    if not hasattr(classifier, "lstm"):
        skipped.append("lstm.* (classifier has no .lstm attribute)")
        classifier.load_state_dict(cs)
        _log(logger, transferred, skipped)
        return {"transferred": transferred, "skipped": skipped}

    H = classifier.lstm.hidden_size
    num_layers = classifier.lstm.num_layers

    for L in range(num_layers):
        # weight_ih_l{L}: input projection, requires special handling at L>0 for BiLSTM.
        lm_ih = lm_state.get(f"lstm.weight_ih_l{L}")
        cls_ih = cs.get(f"lstm.weight_ih_l{L}")
        if lm_ih is not None and cls_ih is not None:
            if lm_ih.shape == cls_ih.shape:
                cls_ih.copy_(lm_ih)
                transferred.append(f"lstm.weight_ih_l{L}")
            elif L > 0 and cls_ih.shape[0] == lm_ih.shape[0] and cls_ih.shape[1] == 2 * lm_ih.shape[1]:
                # Layer L>=1 in BiLSTM has input dim 2H (fwd+bwd of layer L-1).
                # Place LM weight on the "fwd layer L-1 output" half; zero the
                # "bwd layer L-1 output" half so initial behavior matches LM.
                cls_ih.zero_()
                cls_ih[:, : lm_ih.shape[1]].copy_(lm_ih)
                transferred.append(f"lstm.weight_ih_l{L} (zero-padded second half)")
            else:
                skipped.append(
                    f"lstm.weight_ih_l{L} (shape mismatch lm={tuple(lm_ih.shape)} cls={tuple(cls_ih.shape)})"
                )

        # weight_hh / biases: hidden state dim H matches between LM and BiLSTM forward.
        for k in (
            f"lstm.weight_hh_l{L}",
            f"lstm.bias_ih_l{L}",
            f"lstm.bias_hh_l{L}",
        ):
            lm_v = lm_state.get(k)
            cls_v = cs.get(k)
            if lm_v is not None and cls_v is not None:
                if lm_v.shape == cls_v.shape:
                    cls_v.copy_(lm_v)
                    transferred.append(k)
                else:
                    skipped.append(
                        f"{k} (shape mismatch lm={tuple(lm_v.shape)} cls={tuple(cls_v.shape)})"
                    )

    # Backward direction (``*_reverse``) is left at default init in v1 — no backward LM.

    classifier.load_state_dict(cs)
    _log(logger, transferred, skipped)
    return {"transferred": transferred, "skipped": skipped}


def _log(logger: Optional[logging.Logger],
         transferred: List[str], skipped: List[str]) -> None:
    if logger is None:
        return
    logger.info(
        f"[lm-transfer] transferred {len(transferred)}, skipped {len(skipped)}"
    )
    for n in transferred:
        logger.info(f"[lm-transfer]   \u2713 {n}")
    for n in skipped:
        logger.info(f"[lm-transfer]   \u2717 {n}")
