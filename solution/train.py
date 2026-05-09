"""HW4 — improved Text Sentiment Classification pipeline entry point.

Pipeline
--------
1) Load labeled / unlabeled / test; tokenize (lowercase, strip <br />).
2) Train (or load) Word2Vec on labeled + unlabeled + test.
3) Build Vocab (PAD=0, UNK=1) and embedding matrix.
4) Stratified train/val split; build BiLSTM + attention pooling.
5) Train with Adam + BCEWithLogitsLoss + grad clip + early stop.
6) Self-Training rounds: pseudo-label unlabeled w/ high-confidence threshold,
   merge with labeled and fine-tune. Keep the best checkpoint on val.
7) Inference on test.csv -> predict.csv following sample_submission format.
All artifacts land in results/<timestamp>/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.config import Config  # noqa: E402
from utils.logger import build_logger  # noqa: E402
from utils.misc import pick_device, set_seed, timestamp  # noqa: E402
from data.preprocess import (  # noqa: E402
    Vocab, load_labeled_csv, load_test_csv, load_unlabeled_csv, train_word2vec,
)
from data.datasets import PseudoLabeledDataset, SenDataset  # noqa: E402
from models.lstm import LSTMClassifier  # noqa: E402
from engine.trainer import train  # noqa: E402
from engine.inference import predict_probs, save_predictions  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402


def build_loaders(X_train, y_train, X_val, y_val, batch_size):
    tr = DataLoader(SenDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    va = DataLoader(SenDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)
    return tr, va


def pick_pseudo(probs: np.ndarray, pos_th: float, neg_th: float, cap: int):
    pos_mask = probs >= pos_th
    neg_mask = probs <= neg_th
    # rank by confidence (distance from 0.5)
    conf = np.abs(probs - 0.5)
    keep = pos_mask | neg_mask
    idx = np.where(keep)[0]
    if len(idx) > cap:
        order = np.argsort(-conf[idx])
        idx = idx[order[:cap]]
    labels = (probs[idx] >= 0.5).astype(np.int64)
    return idx, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--tag", default=None, help="optional run name suffix")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    set_seed(cfg.seed)

    # ---- Run directory ----
    ts = timestamp()
    run_name = ts if not args.tag else f"{ts}_{args.tag}"
    run_dir = ROOT / cfg.output.result_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(run_dir)
    logger.info(f"run dir: {run_dir}")

    # snapshot config for reproducibility
    cfg.dump_yaml(run_dir / "config.yaml")
    shutil.copy(args.config, run_dir / "config.source.yaml")

    device = pick_device()
    logger.info(f"device: {device}")

    data_dir = Path(cfg.data.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

    # ---- Load data ----
    logger.info("loading labeled data ...")
    train_tokens, y = load_labeled_csv(data_dir / cfg.data.train_csv, cfg.preprocess.lowercase)
    logger.info(f"  labeled samples: {len(train_tokens)}")

    logger.info("loading unlabeled data ...")
    unlabel_tokens = load_unlabeled_csv(data_dir / cfg.data.unlabel_csv, cfg.preprocess.lowercase)
    logger.info(f"  unlabeled samples: {len(unlabel_tokens)}")

    logger.info("loading test data ...")
    test_tokens, test_ids = load_test_csv(data_dir / cfg.data.test_csv, cfg.preprocess.lowercase)
    logger.info(f"  test samples: {len(test_tokens)}")

    lengths = np.array([len(t) for t in train_tokens])
    logger.info(f"token length — mean {lengths.mean():.1f}, median {np.median(lengths):.0f}, "
                f"p90 {np.percentile(lengths, 90):.0f}, p95 {np.percentile(lengths, 95):.0f}, "
                f"max {lengths.max()} | sen_len={cfg.preprocess.sen_len}")

    # ---- Word2Vec ----
    pp = cfg.preprocess
    if pp.w2v_cache_path and Path(pp.w2v_cache_path).exists():
        logger.info(f"loading cached w2v: {pp.w2v_cache_path}")
        w2v = Word2Vec.load(pp.w2v_cache_path)
    else:
        w2v = train_word2vec(
            corpus=train_tokens + unlabel_tokens + test_tokens,
            vector_size=pp.w2v_vector_size, window=pp.w2v_window,
            min_count=pp.min_count, workers=pp.w2v_workers,
            sg=pp.w2v_sg, negative=pp.w2v_negative, epochs=pp.w2v_epochs,
            logger=logger,
        )
        w2v.save(str(run_dir / "w2v.model"))
        logger.info(f"saved w2v -> {run_dir / 'w2v.model'}")

    # ---- Vocab / encode ----
    vocab = Vocab(w2v)
    logger.info(f"vocab size (incl. PAD/UNK): {len(vocab)}; embed_dim={w2v.vector_size}")

    X_all = vocab.encode(train_tokens, pp.sen_len)
    y_all = torch.from_numpy(y.astype(np.int64))
    X_test = vocab.encode(test_tokens, pp.sen_len)
    X_unlabel = vocab.encode(unlabel_tokens, pp.sen_len)

    # ---- Split ----
    tr_idx, va_idx = train_test_split(
        np.arange(len(X_all)), test_size=cfg.train.val_ratio,
        stratify=y_all.numpy(), random_state=cfg.seed,
    )
    X_train, y_train = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[va_idx], y_all[va_idx]
    logger.info(f"train={len(X_train)}, val={len(X_val)} (stratified)")

    train_loader, val_loader = build_loaders(
        X_train, y_train, X_val, y_val, cfg.train.batch_size,
    )

    # ---- Model ----
    mc = cfg.model
    model = LSTMClassifier(
        embedding=vocab.embedding_matrix,
        hidden_dim=mc.hidden_dim, num_layers=mc.num_layers,
        dropout=mc.dropout, bidirectional=mc.bidirectional,
        fix_embedding=mc.fix_embedding, pool=mc.pool,
    ).to(device)

    ckpt = run_dir / "ckpt.pt"
    best = train(model, train_loader, val_loader, cfg.train, device,
                 ckpt_path=ckpt, logger=logger, tag="init")
    # restore best
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])

    pseudo_added = []

    # ---- Self-Training ----
    st = cfg.self_training
    if st.enable and len(X_unlabel) > 0:
        remaining = np.arange(len(X_unlabel))
        for r in range(1, st.rounds + 1):
            if len(remaining) == 0:
                break
            un_loader = DataLoader(
                SenDataset(X_unlabel[remaining]),
                batch_size=cfg.inference.batch_size, shuffle=False, num_workers=0,
            )
            probs = predict_probs(model, un_loader, device)
            idx, pseudo_y = pick_pseudo(
                probs, st.pos_threshold, st.neg_threshold, st.max_pseudo_per_round,
            )
            if len(idx) == 0:
                logger.info(f"[self-train r{r}] no confident samples, stop")
                break
            chosen_global = remaining[idx]
            pseudo_X = X_unlabel[chosen_global]
            pseudo_y_t = torch.from_numpy(pseudo_y)
            logger.info(
                f"[self-train r{r}] added {len(idx)} pseudo "
                f"(pos={int(pseudo_y.sum())}, neg={len(idx) - int(pseudo_y.sum())}) "
                f"from {len(remaining)} candidates"
            )
            pseudo_added.append(int(len(idx)))

            merged = PseudoLabeledDataset(X_train, y_train, pseudo_X, pseudo_y_t)
            tr_loader = DataLoader(merged, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)

            best_r = train(
                model, tr_loader, val_loader, cfg.train, device,
                ckpt_path=ckpt, logger=logger,
                epochs_override=st.finetune_epochs, tag=f"self-train-r{r}",
            )
            # Reload best checkpoint in case this round regressed
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state"])
            best.best_val_acc = max(best.best_val_acc, best_r.best_val_acc)

            mask = np.ones(len(remaining), dtype=bool)
            mask[idx] = False
            remaining = remaining[mask]

    logger.info(f"FINAL best val acc: {best.best_val_acc*100:.2f}")

    # ---- Inference ----
    test_loader = DataLoader(SenDataset(X_test), batch_size=cfg.inference.batch_size,
                             shuffle=False, num_workers=0)
    probs = predict_probs(model, test_loader, device)
    save_predictions(test_ids, probs, run_dir / "predict.csv", logger=logger)

    # Also save raw probabilities for analysis
    np.save(run_dir / "test_probs.npy", probs)

    # Summary
    with open(run_dir / "summary.txt", "w") as f:
        f.write(f"best_val_acc: {best.best_val_acc:.4f}\n")
        f.write(f"best_epoch: {best.best_epoch}\n")
        f.write(f"pseudo_added_per_round: {pseudo_added}\n")
    logger.info(f"done. artifacts at {run_dir}")


if __name__ == "__main__":
    main()
