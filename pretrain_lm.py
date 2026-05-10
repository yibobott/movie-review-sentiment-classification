"""LSTM Language-Model pretraining entry point.

Pipeline
--------
1. Load labeled / unlabeled / test (same tokenizer as classifier).
2. Train (or load cached) Word2Vec on labeled + unlabeled + test (same as
   classifier so vocab/embedding are exactly aligned downstream).
3. Build Vocab and persist (idx2word.json + vocab_hash.txt + w2v.model)
   so train.py can verify alignment at load time.
4. Build LM corpus: doc-level split (val from unlabeled only), then flatten
   each split independently, append EOS at every doc boundary.
5. Train forward LSTM-LM with cross-entropy + warmup-cosine LR + grad clip.
   Save best (lowest val PPL) checkpoint.

Outputs (under ``results_lm/<timestamp>/``)
--------------------------------------------
* ``w2v.model``       - the gensim Word2Vec model (single source of truth)
* ``idx2word.json``   - exact vocab order, used for cross-run integrity check
* ``vocab_hash.txt``  - md5 of idx2word for quick check
* ``lm_ckpt.pt``      - best LM weights + metadata (vocab_hash, V_cls, config)
* ``lm.log``          - training log
* ``config.yaml``     - the resolved config snapshot
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.config import Config  # noqa: E402
from utils.logger import build_logger  # noqa: E402
from utils.misc import git_sha, pick_device, set_seed, timestamp  # noqa: E402
from utils.vocab_io import dump_vocab, vocab_hash  # noqa: E402
from data.preprocess import (  # noqa: E402
    PAD_IDX, Vocab, load_labeled_csv, load_test_csv, load_unlabeled_csv,
    train_word2vec,
)
from data.lm_dataset import LMBPTTDataset, build_lm_corpus  # noqa: E402
from models.lstm_lm import LSTMLanguageModel  # noqa: E402
from engine.lm_trainer import train_lm  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--tag", default=None, help="optional run name suffix")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if not cfg.lm.enable:
        raise SystemExit(
            "lm.enable is False in config; flip it to true (and remove lm.ckpt_path) "
            "before pretraining."
        )
    set_seed(cfg.lm.seed)

    # ---- Run directory ----
    ts = timestamp()
    run_name = ts if not args.tag else f"{ts}_{args.tag}"
    run_dir = ROOT / "results_lm" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(run_dir, log_name="lm.log")
    logger.info(f"lm run dir: {run_dir}")
    cfg.dump_yaml(run_dir / "config.yaml")
    shutil.copy(args.config, run_dir / "config.source.yaml")
    logger.info(f"git sha: {git_sha(ROOT)}")

    device = pick_device()
    logger.info(f"device: {device}")

    # ---- Load data ----
    data_dir = Path(cfg.data.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    logger.info("loading labeled data ...")
    labeled_tokens, _y = load_labeled_csv(data_dir / cfg.data.train_csv, cfg.preprocess.lowercase)
    logger.info(f"  labeled samples: {len(labeled_tokens)}")

    logger.info("loading unlabeled data ...")
    unlabeled_tokens = load_unlabeled_csv(data_dir / cfg.data.unlabel_csv, cfg.preprocess.lowercase)
    logger.info(f"  unlabeled samples: {len(unlabeled_tokens)}")

    logger.info("loading test data ...")
    test_tokens, _test_ids = load_test_csv(data_dir / cfg.data.test_csv, cfg.preprocess.lowercase)
    logger.info(f"  test samples: {len(test_tokens)}")

    # ---- Word2Vec (same hyperparams as classifier so vocab is identical) ----
    pp = cfg.preprocess
    if pp.w2v_cache_path and Path(pp.w2v_cache_path).exists():
        logger.info(f"loading cached w2v: {pp.w2v_cache_path}")
        w2v = Word2Vec.load(pp.w2v_cache_path)
    else:
        w2v = train_word2vec(
            corpus=labeled_tokens + unlabeled_tokens + test_tokens,
            vector_size=pp.w2v_vector_size, window=pp.w2v_window,
            min_count=pp.min_count, workers=pp.w2v_workers,
            sg=pp.w2v_sg, negative=pp.w2v_negative, epochs=pp.w2v_epochs,
            sample=pp.w2v_sample,
            logger=logger,
        )
        w2v.save(str(run_dir / "w2v.model"))
        logger.info(f"saved w2v -> {run_dir / 'w2v.model'}")

    # ---- Vocab / persistence (single source of truth) ----
    vocab = Vocab(w2v)
    v_cls = len(vocab)
    eos_idx = v_cls  # EOS lives just past the classifier vocab
    lm_vocab_size = v_cls + 1
    logger.info(
        f"vocab size (incl. PAD/UNK): V_cls={v_cls}; LM_vocab=V_cls+1={lm_vocab_size}; "
        f"embed_dim={w2v.vector_size}; EOS_IDX={eos_idx}"
    )

    h = dump_vocab(vocab.idx2word, run_dir / "idx2word.json")
    (run_dir / "vocab_hash.txt").write_text(h, encoding="utf-8")
    logger.info(f"vocab hash: {h}")

    # ---- LM corpus ----
    lc = cfg.lm
    train_seq, val_seq, info = build_lm_corpus(
        unlabeled_tokens=unlabeled_tokens,
        test_tokens=test_tokens,
        labeled_tokens=labeled_tokens,
        word2idx=vocab.word2idx,
        eos_idx=eos_idx,
        val_ratio=lc.val_ratio,
        seed=lc.seed,
        include_labeled=lc.include_labeled_in_train,
        include_test=lc.include_test_in_train,
    )
    logger.info(
        f"LM corpus: train_tokens={info['n_train_tokens']:,}, val_tokens={info['n_val_tokens']:,} | "
        f"un_train_docs={info['n_unlabeled_train_docs']}, un_val_docs={info['n_unlabeled_val_docs']}, "
        f"test_docs={info['n_test_docs']}, labeled_docs={info['n_labeled_docs']}"
    )

    train_ds = LMBPTTDataset(train_seq, lc.bptt_len)
    val_ds = LMBPTTDataset(val_seq, lc.bptt_len)
    logger.info(
        f"LM chunks: train={len(train_ds)}, val={len(val_ds)} | bptt={lc.bptt_len}"
    )
    train_loader = DataLoader(train_ds, batch_size=lc.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=lc.batch_size, shuffle=False, num_workers=0)

    # ---- Model ----
    model = LSTMLanguageModel(
        vocab_size=lm_vocab_size,
        embed_dim=w2v.vector_size,
        hidden_dim=lc.hidden_dim,
        num_layers=lc.num_layers,
        dropout=lc.dropout,
        embed_dropout=lc.embed_dropout,
        tie_weights=lc.tie_weights,
        embedding_init=vocab.embedding_matrix if lc.init_from_w2v else None,
        pad_idx=PAD_IDX,
    ).to(device)

    # ---- Train ----
    ckpt = run_dir / "lm_ckpt.pt"
    best = train_lm(
        model, train_loader, val_loader, lc, device,
        ckpt_path=ckpt, logger=logger, pad_idx=PAD_IDX,
    )

    # Repackage ckpt with vocab metadata so train.py can validate alignment.
    state = torch.load(ckpt, map_location="cpu")
    state.update({
        "vocab_hash": h,
        "v_cls": v_cls,
        "lm_vocab_size": lm_vocab_size,
        "embed_dim": int(w2v.vector_size),
        "hidden_dim": int(lc.hidden_dim),
        "num_layers": int(lc.num_layers),
        "eos_idx": int(eos_idx),
    })
    torch.save(state, ckpt)
    logger.info(f"final lm ckpt -> {ckpt} (val ppl {best.best_val_ppl:.2f})")

    print(f"[lm done] {run_dir}")
    print(f"[next]    1) set preprocess.w2v_cache_path: {run_dir / 'w2v.model'}")
    print(f"[next]    2) set lm.ckpt_path:              {ckpt}")
    print(f"[next]    3) set lm.enable: true")


if __name__ == "__main__":
    main()
