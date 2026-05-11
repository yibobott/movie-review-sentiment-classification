"""LSTM-LM -> classifier BiLSTM weight transfer.

Strategy
--------
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
    lm_bw_state: Optional[Dict[str, torch.Tensor]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List[str]]:
    """Copy LM weights into ``classifier`` in-place. Returns audit dict.

    If ``lm_bw_state`` is provided (a backward LM trained on reversed token
    streams), its LSTM weights are transferred into the classifier's reverse
    direction (``*_reverse`` slots). Embedding is taken only from ``lm_state``
    (forward LM); both LMs share the same vocab so it doesn't matter which
    one we pick.
    """
    transferred: List[str] = []
    skipped: List[str] = []

    cs = classifier.state_dict()

    # When either side (LM ckpt or classifier) wraps its LSTM with WeightDrop,
    # the recurrent weights live under a different key:
    #   * "_raw" suffix on dropped weights (weight_hh_l* -> weight_hh_l*_raw)
    #   * an extra ".module." segment because WeightDrop is a submodule
    #     (lstm.weight_ih_l0 -> lstm.module.weight_ih_l0)
    # To keep the transfer logic below uniform, alias each such key back to
    # its bare classical name. Writes via copy_ on the aliased tensor still
    # mutate the underlying parameter (state_dict tensors share storage).
    def _bare(k: str) -> str:
        # Drop ".module." segment introduced by WeightDrop wrapper.
        k = k.replace(".module.", ".")
        # Drop trailing "_raw" suffix on dropped recurrent weights.
        if k.endswith("_raw"):
            k = k[: -len("_raw")]
        return k

    aliased_in_cs: List[str] = []  # bare names we added to cs (must strip before load)
    for k in list(cs.keys()):
        bare = _bare(k)
        if bare != k and bare not in cs:
            cs[bare] = cs[k]
            aliased_in_cs.append(bare)
    # Same for the LM side (read-only; safe to add bare aliases).
    lm_state = dict(lm_state)  # shallow copy so we don't mutate caller's dict
    for k in list(lm_state.keys()):
        bare = _bare(k)
        if bare != k and bare not in lm_state:
            lm_state[bare] = lm_state[k]
    # And for the optional backward-LM side.
    if lm_bw_state is not None:
        lm_bw_state = dict(lm_bw_state)
        for k in list(lm_bw_state.keys()):
            bare = _bare(k)
            if bare != k and bare not in lm_bw_state:
                lm_bw_state[bare] = lm_bw_state[k]

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

    # ---- 3) Backward direction (``*_reverse``), if backward LM is provided ----
    # The backward LM was trained on token streams reversed at the BPTT level;
    # its weights are uni-directional LSTM weights that, when applied to a
    # reversed input, produce "predict previous token" representations. In a
    # bidirectional classifier, the *_reverse parameters operate on the input
    # processed in reverse temporal order, which is exactly the same semantics.
    # So we copy backward-LM forward-direction weights into classifier reverse
    # slots verbatim (with the same layer-1 zero-padding trick, but on the
    # SECOND H dims because that's where the layer-0 backward output lives in
    # the 2H concatenated layer-1 input).
    if lm_bw_state is not None:
        for L in range(num_layers):
            lm_bw_ih = lm_bw_state.get(f"lstm.weight_ih_l{L}")
            cls_ih_rev = cs.get(f"lstm.weight_ih_l{L}_reverse")
            if lm_bw_ih is not None and cls_ih_rev is not None:
                if lm_bw_ih.shape == cls_ih_rev.shape:
                    cls_ih_rev.copy_(lm_bw_ih)
                    transferred.append(f"lstm.weight_ih_l{L}_reverse [bw-LM]")
                elif (
                    L > 0
                    and cls_ih_rev.shape[0] == lm_bw_ih.shape[0]
                    and cls_ih_rev.shape[1] == 2 * lm_bw_ih.shape[1]
                ):
                    # Layer L>=1 in BiLSTM has input dim 2H = [fwd_l0_out, bwd_l0_out].
                    # Backward LM layer L took H input (its own l0 output, which
                    # corresponds to "bwd_l0_out" in the concat). Place LM weight
                    # on the SECOND half (bwd half); zero the first half.
                    cls_ih_rev.zero_()
                    H_lm = lm_bw_ih.shape[1]
                    cls_ih_rev[:, H_lm:].copy_(lm_bw_ih)
                    transferred.append(
                        f"lstm.weight_ih_l{L}_reverse [bw-LM, zero-padded first half]"
                    )
                else:
                    skipped.append(
                        f"lstm.weight_ih_l{L}_reverse (shape mismatch "
                        f"bw_lm={tuple(lm_bw_ih.shape)} cls={tuple(cls_ih_rev.shape)})"
                    )

            for k_lm, k_cls in (
                (f"lstm.weight_hh_l{L}",  f"lstm.weight_hh_l{L}_reverse"),
                (f"lstm.bias_ih_l{L}",   f"lstm.bias_ih_l{L}_reverse"),
                (f"lstm.bias_hh_l{L}",   f"lstm.bias_hh_l{L}_reverse"),
            ):
                lm_v = lm_bw_state.get(k_lm)
                cls_v = cs.get(k_cls)
                if lm_v is not None and cls_v is not None:
                    if lm_v.shape == cls_v.shape:
                        cls_v.copy_(lm_v)
                        transferred.append(f"{k_cls} [bw-LM]")
                    else:
                        skipped.append(
                            f"{k_cls} (shape mismatch "
                            f"bw_lm={tuple(lm_v.shape)} cls={tuple(cls_v.shape)})"
                        )

    # Strip the bare-name aliases we added earlier so load_state_dict doesn't
    # complain about "unexpected keys". The underlying parameter storage was
    # already mutated in-place via copy_ above, so this is safe.
    for bare in aliased_in_cs:
        cs.pop(bare, None)
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
