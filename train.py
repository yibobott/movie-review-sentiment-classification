"""HW4 — text sentiment classification pipeline entry point.

Pipeline
--------
1) Load labeled / unlabeled / test; tokenize (lowercase, strip <br />).
2) Train (or load) Word2Vec on labeled + unlabeled + test.
3) Build Vocab (PAD=0, UNK=1) and embedding matrix.
4) Stratified train/val split; build BiLSTM + attention pooling classifier.
5) Optional: load pretrained LM weights into the BiLSTM body.
6) Two-phase fine-tune (Phase A frozen-body warmup -> Phase B disc-LR).
7) Self-training rounds on the unlabeled set with confidence thresholds.
8) Inference: predict.csv (EMA) + predict_raw.csv. We submit exactly one.

All artifacts land in ``results/<timestamp>[_tag][_acc]/``.
This module is the orchestrator only; the heavy lifting lives in
``pipeline/`` (cli, run_dir, lm_loading, optim, data_split, phases,
self_training, predict).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.preprocess import (  # noqa: E402
    Vocab, load_labeled_csv, load_test_csv, load_unlabeled_csv, train_word2vec,
)
from gensim.models import Word2Vec  # noqa: E402
from models.lstm import LSTMClassifier  # noqa: E402

from pipeline.cli import build_arg_parser, resolve_classifier_lm_paths  # noqa: E402
from pipeline.data_split import build_loaders, stratified_train_val_split  # noqa: E402
from pipeline.lm_loading import maybe_load_lm_ckpt  # noqa: E402
from pipeline.phases import run_initial_phases  # noqa: E402
from pipeline.predict import write_submissions  # noqa: E402
from pipeline.run_dir import finalize_run_dir, setup_run_dir  # noqa: E402
from pipeline.self_training import run_self_training  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.logger import build_logger  # noqa: E402
from utils.misc import git_sha, pick_device, set_seed, timestamp  # noqa: E402


def main() -> None:
    args = build_arg_parser(default_config=ROOT / "config.yaml").parse_args()
    cfg = Config.from_yaml(args.config)

    # Resolve LM ckpt paths (CLI > cfg.lm.ckpt_path > None).
    lm_ckpt_path, lm_bw_ckpt_path = resolve_classifier_lm_paths(
        args, cfg.lm.ckpt_path, root=ROOT,
    )
    set_seed(cfg.seed)

    # ---- Run directory + reproducibility snapshot ----
    ts = timestamp()
    run_name = ts if not args.tag else f"{ts}_{args.tag}"
    resolved_overrides: list[str] = []
    if lm_ckpt_path is not None:
        resolved_overrides.append(f"--lm {lm_ckpt_path}")
    if lm_bw_ckpt_path is not None:
        resolved_overrides.append(f"--lm-bw {lm_bw_ckpt_path}")
    run_dir, logger = setup_run_dir(
        ROOT, cfg.output.result_root, run_name,
        config_path=Path(args.config),
        cli_argv=["train.py", *sys.argv[1:]],
        resolved_overrides=resolved_overrides,
        log_builder=build_logger,
    )
    if args.lm is not None:
        logger.info(f"[cli] --lm {args.lm} -> resolved to {lm_ckpt_path}")
    if args.lm_bw is not None:
        logger.info(f"[cli] --lm-bw {args.lm_bw} -> resolved to {lm_bw_ckpt_path}")
    sha = git_sha(ROOT)
    logger.info(f"git sha: {sha}")

    device = pick_device()
    logger.info(f"device: {device}")

    # ---- Load data ----
    data_dir = Path(cfg.data.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()

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
    logger.info(
        f"token length — mean {lengths.mean():.1f}, median {np.median(lengths):.0f}, "
        f"p90 {np.percentile(lengths, 90):.0f}, p95 {np.percentile(lengths, 95):.0f}, "
        f"max {lengths.max()} | sen_len={cfg.preprocess.sen_len}"
    )

    # ---- Word2Vec (cached or trained from scratch) ----
    pp = cfg.preprocess
    eff_w2v = pp.w2v_cache_path
    if eff_w2v and Path(eff_w2v).exists():
        logger.info(f"loading cached w2v: {eff_w2v}")
        w2v = Word2Vec.load(eff_w2v)
    else:
        w2v = train_word2vec(
            corpus=train_tokens + unlabel_tokens + test_tokens,
            vector_size=pp.w2v_vector_size, window=pp.w2v_window,
            min_count=pp.min_count, workers=pp.w2v_workers,
            sg=pp.w2v_sg, negative=pp.w2v_negative, epochs=pp.w2v_epochs,
            sample=pp.w2v_sample,
            logger=logger,
        )
        w2v.save(str(run_dir / "w2v.model"))
        logger.info(f"saved w2v -> {run_dir / 'w2v.model'}")

    # ---- Vocab / encode ----
    vocab = Vocab(w2v)
    logger.info(f"vocab size (incl. PAD/UNK): {len(vocab)}; embed_dim={w2v.vector_size}")

    X_all = vocab.encode(train_tokens, pp.sen_len, head_ratio=pp.head_ratio)
    y_all = torch.from_numpy(y.astype(np.int64))
    X_test = vocab.encode(test_tokens, pp.sen_len, head_ratio=pp.head_ratio)
    X_unlabel = vocab.encode(unlabel_tokens, pp.sen_len, head_ratio=pp.head_ratio)

    # ---- Split + loaders ----
    tr_idx, va_idx = stratified_train_val_split(y_all.numpy(), cfg.train.val_ratio, cfg.seed)
    X_train, y_train = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[va_idx], y_all[va_idx]
    logger.info(f"train={len(X_train)}, val={len(X_val)} (stratified)")
    train_loader, val_loader = build_loaders(
        X_train, y_train, X_val, y_val, cfg.train.batch_size,
        word_dropout=cfg.train.word_dropout,
    )

    # ---- Model ----
    mc = cfg.model
    model = LSTMClassifier(
        embedding=vocab.embedding_matrix,
        hidden_dim=mc.hidden_dim, num_layers=mc.num_layers,
        dropout=mc.dropout,
        embed_dropout=mc.embed_dropout,
        embed_noise_std=mc.embed_noise_std,
        bidirectional=mc.bidirectional,
        fix_embedding=mc.fix_embedding, pool=mc.pool,
        attn_heads=mc.attn_heads,
        locked_dropout=mc.locked_dropout,
        weight_drop=mc.weight_drop,
    ).to(device)

    # ---- Optional LM weight transfer ----
    lm_loaded = maybe_load_lm_ckpt(
        cfg, vocab, model, logger, lm_ckpt_path,
        lm_bw_ckpt_path=lm_bw_ckpt_path,
    )
    ckpt = run_dir / "ckpt.pt"

    # ---- Training: Phase A (optional frozen warmup) + Phase B ----
    best = run_initial_phases(
        cfg, model, train_loader, val_loader, device,
        ckpt=ckpt, logger=logger, lm_loaded=lm_loaded,
    )

    # ---- Self-training rounds ----
    best, pseudo_added = run_self_training(
        cfg, model, X_train, y_train, X_unlabel, val_loader, device,
        run_dir=run_dir, ckpt=ckpt, best=best, logger=logger,
    )

    logger.info(
        f"FINAL best val acc: {best.best_val_acc * 100:.2f} "
        f"(ema {best.best_ema_acc * 100:.2f}, raw {best.best_raw_acc * 100:.2f})"
    )

    # ---- Inference + submissions ----
    write_submissions(
        model, X_test, test_ids,
        ckpt=ckpt, run_dir=run_dir,
        inference_batch_size=cfg.inference.batch_size,
        device=device, logger=logger,
    )

    # ---- Summary ----
    with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"best_val_acc: {best.best_val_acc:.4f}\n")
        f.write(f"best_epoch: {best.best_epoch}\n")
        f.write(f"best_ema_acc: {best.best_ema_acc:.4f}\n")
        f.write(f"best_ema_epoch: {best.best_ema_epoch}\n")
        f.write(f"best_raw_acc: {best.best_raw_acc:.4f}\n")
        f.write(f"best_raw_epoch: {best.best_raw_epoch}\n")
        f.write(f"pseudo_added_per_round: {pseudo_added}\n")
        f.write(f"git_sha: {sha}\n")
    logger.info(f"done. artifacts at {run_dir}")

    final_run_dir = finalize_run_dir(run_dir, run_name, best.best_val_acc, logger)
    print(f"[done] artifacts at {final_run_dir}")


if __name__ == "__main__":
    main()
