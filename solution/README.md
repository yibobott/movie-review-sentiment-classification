# HW4 Text Sentiment Classification — Improved Pipeline

Single BiLSTM-based model (LSTM-like, self-designed) targeting val acc > 0.903
on the HW4 movie-review sentiment task. Complies with `HW4-Rules.md`:

- ✅ Single LSTM-like model (no transformers / nltk / other pre-trained NLP pkgs)
- ✅ No ensemble (one model, best checkpoint on validation)
- ✅ Uses unlabeled data via **Self-Training** (encouraged by rules)
- ✅ Only stdlib + torch + numpy + pandas + sklearn + gensim + pyyaml

## Structure

```
solution/
├── train.py                   # entry point (train + self-train + predict)
├── config.yaml                # all hyperparameters
├── requirements.txt
├── utils/
│   ├── config.py              # typed Config dataclasses (yaml <-> dataclass)
│   ├── logger.py              # build_logger (stdout + file)
│   └── misc.py                # set_seed, tokenize, pick_device, timestamp
├── models/
│   └── lstm.py                # BiLSTM + attention / max / mean pooling
├── data/
│   ├── preprocess.py          # loaders, Word2Vec (skip-gram), Vocab
│   └── datasets.py            # SenDataset, PseudoLabeledDataset
├── engine/
│   ├── trainer.py             # train loop, early stop, LR sched
│   └── inference.py           # predict_probs, save predict.csv
└── results/<timestamp>/        # per-run outputs
    ├── config.yaml            # resolved config snapshot
    ├── config.source.yaml     # original config used
    ├── train.log              # full log
    ├── w2v.model              # trained Word2Vec
    ├── ckpt.pt                # best checkpoint by val acc
    ├── predict.csv            # kaggle submission
    ├── test_probs.npy         # raw probs for analysis
    └── summary.txt            # final metrics
```

## Usage

```bash
cd solution
pip install -r requirements.txt
python train.py                           # uses config.yaml
python train.py --tag big_hidden          # appends tag to run folder name
python train.py --config my_config.yaml   # custom config
```

Each invocation writes a **new** folder `results/YYYYMMDD_HHMMSS[_tag]/` — easy
to diff hyperparameters across experiments.

## Key improvements over baseline

| Area | Baseline | This version |
|---|---|---|
| Optimizer | SGD lr=1e-3, 2 epoch | Adam lr=1e-3, up to 12 epoch + early stop |
| Loss | BCELoss + Sigmoid in model | **BCEWithLogitsLoss** (numerically stable) |
| Split | first 13k for train (unstratified) | 90/10 **stratified** on 25k |
| sen_len | 30 | **200** (covers p95 of token lengths) |
| Tokenize | raw regex | lowercase + strip `<br />` |
| Embedding | fixed w2v CBOW, vector=250 | **skip-gram + negative**, vector=256, fine-tuned |
| Model | 1-layer unidir LSTM, last step | **2-layer BiLSTM** + attention/max/mean pooling + MLP head |
| Pad handling | last step includes pads | masked pooling, PAD=0 idx |
| Regularization | dropout=0.5 | dropout=0.4 + grad clip 1.0 + weight decay |
| Unlabeled 50k | unused | **Self-training** 2 rounds, conf>=0.9 / <=0.1 |
| Logging | print to stdout | logger to file + stdout + per-run dir |

## Tuning knobs (edit `config.yaml`)

- `preprocess.sen_len` — try 150 / 250
- `preprocess.min_count` — 3 keeps more rare words; 5 is more robust
- `model.hidden_dim` / `num_layers` — 192×2 / 256×2
- `model.fix_embedding` — set `true` if overfitting
- `train.epochs` + `early_stop_patience`
- `self_training.pos_threshold` / `neg_threshold` — higher = cleaner pseudo labels
- `self_training.rounds` — 1–3

## Reproducibility

- Global seed in `config.yaml` (default 42); applied to `random`, `numpy`, `torch`.
- Each run snapshots the config used, alongside logs & checkpoints.
