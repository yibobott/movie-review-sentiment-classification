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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.config import Config  # noqa: E402
from utils.logger import build_logger  # noqa: E402
from utils.misc import git_sha, pick_device, set_seed, timestamp  # noqa: E402
from utils.vocab_io import vocab_hash, verify_vocab  # noqa: E402
from utils.weight_transfer import transfer_lm_to_classifier  # noqa: E402
from data.preprocess import (  # noqa: E402
    Vocab, load_labeled_csv, load_test_csv, load_unlabeled_csv, train_word2vec,
)
from data.datasets import PseudoLabeledDataset, SenDataset  # noqa: E402
from models.lstm import LSTMClassifier  # noqa: E402
from engine.trainer import train, raw_ckpt_path  # noqa: E402
from engine.inference import predict_probs, save_predictions  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402


def build_loaders(X_train, y_train, X_val, y_val, batch_size, word_dropout: float = 0.0):
    tr = DataLoader(
        SenDataset(X_train, y_train, word_dropout=word_dropout),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    va = DataLoader(
        SenDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return tr, va


def maybe_load_lm_ckpt(cfg, vocab, model, logger, lm_ckpt_path: Path | None):
    """Validate vocab alignment and transfer LM weights.

    ``lm_ckpt_path`` is the resolved path (from --lm or cfg.lm.ckpt_path).
    If None, returns False (no LM). On any mismatch, raises — never silently
    falls through, since a misaligned LM is worse than no LM.
    """
    if lm_ckpt_path is None:
        return False
    if not lm_ckpt_path.exists():
        raise FileNotFoundError(f"LM ckpt does not exist: {lm_ckpt_path}")

    # Locate the sibling vocab file written by pretrain_lm.py.
    lm_run_dir = lm_ckpt_path.parent
    vocab_json = lm_run_dir / "idx2word.json"
    if not vocab_json.exists():
        raise FileNotFoundError(
            f"idx2word.json not found next to LM ckpt ({lm_run_dir}); "
            f"re-run pretrain_lm.py with the current code."
        )

    # Element-wise vocab match + hash double-check (defense vs silent w2v drift).
    state = torch.load(lm_ckpt_path, map_location="cpu")
    expected_hash = state.get("vocab_hash")
    verify_vocab(vocab.idx2word, vocab_json, expected_hash=expected_hash)
    logger.info(
        f"[lm-load] vocab integrity OK (hash={vocab_hash(vocab.idx2word)[:12]}\u2026, "
        f"V_cls={len(vocab)})"
    )

    # Architecture sanity.
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

    # Transfer weights.
    transfer_lm_to_classifier(state["model_state"], model, logger=logger)
    logger.info(
        f"[lm-load] loaded LM ckpt from {lm_ckpt_path} "
        f"(val_ppl={state.get('val_ppl', 'NA')}, epoch={state.get('epoch', 'NA')})"
    )
    return True


def build_discriminative_optimizer(model, cfg_train, logger):
    """Build AdamW with three LR groups: embedding / lstm / head.

    The split is purely for fine-tuning *after* LM transfer: pretrained
    parameters get a smaller LR, randomly-initialized head gets the largest.
    """
    head_modules = []
    for name in ("attn", "feat_norm", "classifier"):
        m = getattr(model, name, None)
        if m is not None:
            head_modules.append((name, m))

    embedding_params = list(model.embedding.parameters())
    lstm_params = list(model.lstm.parameters())
    head_params = []
    for _, m in head_modules:
        head_params.extend(list(m.parameters()))

    # Sanity: cover every trainable parameter exactly once.
    seen = {id(p) for p in embedding_params + lstm_params + head_params}
    leftover = [p for p in model.parameters() if id(p) not in seen and p.requires_grad]
    if leftover:
        # Fall back to lumping any unrecognized module into the head group.
        head_params.extend(leftover)
        logger.info(
            f"[disc-lr] {len(leftover)} unrecognized trainable params lumped into head group"
        )

    param_groups = [
        {"params": [p for p in embedding_params if p.requires_grad], "lr": cfg_train.lr_embedding},
        {"params": [p for p in lstm_params      if p.requires_grad], "lr": cfg_train.lr_lstm},
        {"params": [p for p in head_params      if p.requires_grad], "lr": cfg_train.lr_head},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg_train.weight_decay)
    logger.info(
        f"[disc-lr] embedding lr={cfg_train.lr_embedding:.1e} ({sum(p.numel() for p in embedding_params)} params), "
        f"lstm lr={cfg_train.lr_lstm:.1e} ({sum(p.numel() for p in lstm_params)} params), "
        f"head lr={cfg_train.lr_head:.1e} ({sum(p.numel() for p in head_params)} params)"
    )
    return optimizer


def stratified_train_val_split(labels: np.ndarray, val_ratio: float, seed: int):
    """Small dependency-free stratified split for binary labels."""
    rng = np.random.RandomState(seed)
    train_parts = []
    val_parts = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def pick_pseudo(
    probs: np.ndarray,
    pos_th: float,
    neg_th: float,
    cap: int,
    balance: bool = True,
):
    """Select high-confidence pseudo-labeled candidates.

    When ``balance`` is True, take the most confident ``cap // 2`` positives
    and the most confident ``cap // 2`` negatives. This prevents the model's
    existing class bias from being amplified during self-training.
    """
    pos_idx = np.where(probs >= pos_th)[0]
    neg_idx = np.where(probs <= neg_th)[0]
    pos_conf = probs[pos_idx]
    neg_conf = 1.0 - probs[neg_idx]

    if balance:
        n_each = cap // 2
        if len(pos_idx) > n_each:
            order = np.argsort(-pos_conf)
            pos_idx = pos_idx[order[:n_each]]
        if len(neg_idx) > n_each:
            order = np.argsort(-neg_conf)
            neg_idx = neg_idx[order[:n_each]]
    else:
        idx = np.concatenate([pos_idx, neg_idx])
        conf = np.concatenate([pos_conf, neg_conf])
        if len(idx) > cap:
            order = np.argsort(-conf)
            idx = idx[order[:cap]]
        labels = (probs[idx] >= 0.5).astype(np.int64)
        return idx, labels

    idx = np.concatenate([pos_idx, neg_idx])
    labels = np.concatenate([
        np.ones(len(pos_idx), dtype=np.int64),
        np.zeros(len(neg_idx), dtype=np.int64),
    ])
    return idx, labels


def _resolve_lm_arg(arg: str | None) -> Path | None:
    """Resolve --lm CLI arg to an absolute lm_ckpt.pt path.

    None        -> None (use config.lm settings as-is).
    'latest'    -> ckpt of the most recent LM run (read from results_lm/LATEST).
    <path>      -> as-is (made absolute against ROOT if relative).
    """
    if arg is None:
        return None
    if arg == "latest":
        latest_file = ROOT / "results_lm" / "LATEST"
        if not latest_file.exists():
            raise SystemExit(
                "--lm latest: results_lm/LATEST not found. Run pretrain_lm.py first."
            )
        run_name = latest_file.read_text(encoding="utf-8").strip()
        candidate = ROOT / "results_lm" / run_name / "lm_ckpt.pt"
        if not candidate.exists():
            raise SystemExit(
                f"--lm latest: resolved to {candidate} but file not found."
            )
        return candidate
    p = Path(arg)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        raise SystemExit(f"--lm: file does not exist: {p}")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--tag", default=None, help="optional run name suffix")
    parser.add_argument(
        "--lm", default=None,
        help="Path to LM ckpt to load. 'latest' = ckpt of most recent LM run "
             "(results_lm/LATEST). Without this flag (and without lm.ckpt_path "
             "in yaml), behavior is the no-LM baseline. Discriminative LR is "
             "controlled by train.use_discriminative_lr in yaml (default true).",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    # CLI args become LOCAL state; cfg is treated as immutable.
    # Resolution order for "load this LM ckpt":
    #   --lm  >  cfg.lm.ckpt_path  >  None (no LM)
    cli_lm_ckpt = _resolve_lm_arg(args.lm)
    if cli_lm_ckpt is not None:
        lm_ckpt_path = cli_lm_ckpt
    elif cfg.lm.ckpt_path:
        p = Path(cfg.lm.ckpt_path)
        lm_ckpt_path = p if p.is_absolute() else (ROOT / p).resolve()
    else:
        lm_ckpt_path = None
    set_seed(cfg.seed)

    # ---- Run directory ----
    ts = timestamp()
    run_name = ts if not args.tag else f"{ts}_{args.tag}"
    run_dir = ROOT / cfg.output.result_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(run_dir)
    logger.info(f"run dir: {run_dir}")
    if args.lm is not None:
        logger.info(f"[cli] --lm {args.lm} -> resolved to {lm_ckpt_path}")

    # Reproducibility snapshot: source yaml verbatim + recorded CLI invocation.
    # No "merged config dump" — cfg is the only source of truth and remains
    # untouched on disk.
    shutil.copy(args.config, run_dir / "config.source.yaml")
    (run_dir / "cli_args.txt").write_text(
        "python train.py " + " ".join(sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    # resolved_overrides.txt records the *absolute* paths after resolving
    # keywords like 'latest'. Use this file (not cli_args.txt) for reproduction.
    resolved_lines = []
    if lm_ckpt_path is not None:
        resolved_lines.append(f"--lm {lm_ckpt_path}")
    (run_dir / "resolved_overrides.txt").write_text(
        "# Reproduce: python train.py --config <path/to/config.source.yaml>"
        + (" " + " ".join(resolved_lines) if resolved_lines else "")
        + "\n",
        encoding="utf-8",
    )
    sha = git_sha(ROOT)
    logger.info(f"git sha: {sha}")

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
    # w2v cache is the single source of truth: preprocess.w2v_cache_path in yaml.
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

    # ---- Split ----
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
    ).to(device)

    # ---- Optional: load LM-pretrained weights before init training ----
    lm_loaded = maybe_load_lm_ckpt(cfg, vocab, model, logger, lm_ckpt_path)
    init_optimizer = None
    if lm_loaded and cfg.train.use_discriminative_lr:
        init_optimizer = build_discriminative_optimizer(model, cfg.train, logger)

    ckpt = run_dir / "ckpt.pt"
    best = train(model, train_loader, val_loader, cfg.train, device,
                 ckpt_path=ckpt, logger=logger, tag="init",
                 optimizer=init_optimizer)
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
                balance=st.balance_pseudo,
            )
            if len(idx) == 0:
                logger.info(f"[self-train r{r}] no confident samples, stop")
                break
            n_pos = int(pseudo_y.sum())
            n_neg = len(idx) - n_pos
            # Imbalance circuit breaker: if one class is empty (or <10% of the other),
            # the model's logit distribution has drifted. Continuing would amplify the
            # bias via a feedback loop (seen in run 20260510_024824 r2 -> pos=0).
            if n_pos == 0 or n_neg == 0 or min(n_pos, n_neg) * 10 < max(n_pos, n_neg):
                logger.info(
                    f"[self-train r{r}] pseudo class imbalance (pos={n_pos}, neg={n_neg}); "
                    f"stop self-training to avoid feedback loop"
                )
                break
            chosen_global = remaining[idx]
            pseudo_X = X_unlabel[chosen_global]
            pseudo_y_t = torch.from_numpy(pseudo_y)
            logger.info(
                f"[self-train r{r}] added {len(idx)} pseudo "
                f"(pos={n_pos}, neg={n_neg}) "
                f"from {len(remaining)} candidates"
            )
            pseudo_added.append(int(len(idx)))

            merged = PseudoLabeledDataset(
                X_train, y_train, pseudo_X, pseudo_y_t,
                word_dropout=cfg.train.word_dropout,
            )
            tr_loader = DataLoader(merged, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0)

            # Always start each fine-tune round from the current global best,
            # so a regressed previous round cannot poison the next round's init.
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state"])

            round_ckpt = run_dir / f"ckpt_self_train_r{r}.pt"
            best_r = train(
                model, tr_loader, val_loader, cfg.train, device,
                ckpt_path=round_ckpt, logger=logger,
                epochs_override=st.finetune_epochs,
                lr_override=st.finetune_lr,
                tag=f"self-train-r{r}",
            )
            # Promote EMA-best and raw-best independently. The two tracks are
            # decided post-hoc on Kaggle, so we should not let one regressed
            # round overwrite a healthier checkpoint on the *other* track.
            round_raw_ckpt = raw_ckpt_path(round_ckpt)
            if best_r.best_ema_acc > best.best_ema_acc:
                shutil.copyfile(round_ckpt, ckpt)
                best.best_ema_acc = best_r.best_ema_acc
                best.best_ema_epoch = best_r.best_ema_epoch
                logger.info(
                    f"[self-train-r{r}] EMA promoted to global best "
                    f"({best.best_ema_acc*100:.2f})"
                )
            else:
                logger.info(
                    f"[self-train-r{r}] EMA kept previous global best "
                    f"({best.best_ema_acc*100:.2f}); round best was "
                    f"{best_r.best_ema_acc*100:.2f}"
                )
            if best_r.best_raw_acc > best.best_raw_acc and round_raw_ckpt.exists():
                shutil.copyfile(round_raw_ckpt, raw_ckpt_path(ckpt))
                best.best_raw_acc = best_r.best_raw_acc
                best.best_raw_epoch = best_r.best_raw_epoch
                logger.info(
                    f"[self-train-r{r}] RAW promoted to global best "
                    f"({best.best_raw_acc*100:.2f})"
                )
            else:
                logger.info(
                    f"[self-train-r{r}] RAW kept previous global best "
                    f"({best.best_raw_acc*100:.2f}); round best was "
                    f"{best_r.best_raw_acc*100:.2f}"
                )
            # Aggregate for legacy summary.
            if best.best_ema_acc >= best.best_raw_acc:
                best.best_val_acc = best.best_ema_acc
                best.best_epoch = best.best_ema_epoch
            else:
                best.best_val_acc = best.best_raw_acc
                best.best_epoch = best.best_raw_epoch
            # Reload global EMA-best so later rounds train from the most stable init.
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state"])

            mask = np.ones(len(remaining), dtype=bool)
            mask[idx] = False
            remaining = remaining[mask]

    logger.info(
        f"FINAL best val acc: {best.best_val_acc*100:.2f} "
        f"(ema {best.best_ema_acc*100:.2f}, raw {best.best_raw_acc*100:.2f})"
    )

    # ---- Inference ----
    # Save TWO submissions: one from EMA-best, one from raw-best. We pick the
    # better one on Kaggle (val acc is biased; we don't know a priori which
    # track generalizes better). predict.csv defaults to EMA.
    test_loader = DataLoader(SenDataset(X_test), batch_size=cfg.inference.batch_size,
                             shuffle=False, num_workers=0)

    # EMA-best inference
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    probs_ema = predict_probs(model, test_loader, device)
    save_predictions(test_ids, probs_ema, run_dir / "predict.csv", logger=logger)
    save_predictions(test_ids, probs_ema, run_dir / "predict_ema.csv", logger=logger)
    np.save(run_dir / "test_probs_ema.npy", probs_ema)

    # Raw-best inference (if a separate raw ckpt exists)
    raw_ckpt = raw_ckpt_path(ckpt)
    if raw_ckpt.exists():
        state = torch.load(raw_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        probs_raw = predict_probs(model, test_loader, device)
        save_predictions(test_ids, probs_raw, run_dir / "predict_raw.csv", logger=logger)
        np.save(run_dir / "test_probs_raw.npy", probs_raw)
    else:
        logger.info("raw-best ckpt not found; skipping predict_raw.csv")

    # Summary
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

    # Rename run dir to include final val acc for at-a-glance comparison.
    # Close the file handler first so Windows does not lock the directory.
    import logging as _logging
    for _h in logger.handlers[:]:
        if isinstance(_h, _logging.FileHandler):
            _h.close()
            logger.removeHandler(_h)
    final_run_dir = run_dir
    try:
        acc_tag = f"{best.best_val_acc*100:.2f}"
        new_run_dir = run_dir.parent / f"{run_name}_{acc_tag}"
        run_dir.rename(new_run_dir)
        final_run_dir = new_run_dir
        print(f"[done] artifacts at {new_run_dir}")
    except Exception as _e:
        print(f"[done] artifacts at {run_dir}  (rename skipped: {_e})")

    # Update marker so ``pretrain_lm.py --w2v latest`` can pick up this run's
    # w2v.model in the future. Best-effort; do not fail the run on IO errors.
    try:
        latest_file = final_run_dir.parent / "LATEST"
        latest_file.write_text(final_run_dir.name, encoding="utf-8")
    except Exception as _e:
        print(f"[warn] could not update {final_run_dir.parent / 'LATEST'}: {_e}")


if __name__ == "__main__":
    main()
