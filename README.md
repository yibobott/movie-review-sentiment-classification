# HW4 Text Sentiment Classification — Improved Pipeline

Single BiLSTM-based model (LSTM-like, self-designed) targeting val acc > 0.903
on the HW4 movie-review sentiment task. Complies with `HW4-Rules.md`:

- Single LSTM-like model (no transformers / nltk / other pre-trained NLP pkgs)
- No ensemble (one model, best checkpoint on validation)
- Uses unlabeled data via **Self-Training** (encouraged by rules)
- Only stdlib + torch + numpy + pandas + sklearn + gensim + pyyaml

## Structure

```
solution/
├── train.py                   # entry point (train + self-train + predict)
├── config.yaml                # all hyperparameters
├── requirements.txt           # shared non-PyTorch dependencies
├── requirements-mac.txt       # macOS PyTorch + shared dependencies
├── requirements-win-gpu.txt   # Windows NVIDIA CUDA PyTorch + shared dependencies
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

Use the requirements file that matches the machine

### macOS

```bash
pip install -r requirements-mac.txt
```

### Windows with NVIDIA GPU

`requirements-win-gpu.txt` pins the **CUDA 13.0** PyTorch wheel
(`--extra-index-url https://download.pytorch.org/whl/cu130`), the official
recommendation for CUDA driver 13.0.

```powershell
pip install -r requirements-win-gpu.txt
```

If you prefer a different CUDA runtime, edit the `--extra-index-url` line in `requirements-win-gpu.txt`.

Verify that PyTorch can actually launch a kernel on the GPU (this is
stronger than `torch.cuda.is_available()`):

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'); x=torch.randn(2,3).cuda(); print((x@x.T).sum().item())"
```

### Run training

```bash
python train.py                           # uses config.yaml
python train.py --tag big_hidden          # appends tag to run folder name
python train.py --config my_config.yaml   # custom config
```

Each invocation writes a **new** folder `results/YYYYMMDD_HHMMSS[_tag]/` — easy
to diff hyperparameters across experiments.

## Key improvements over baseline

| Area | Baseline | This version |
|---|---|---|
| Optimizer | SGD lr=1e-3, 2 epoch | **AdamW** lr=8e-4, up to 20 epoch + early stop |
| LR schedule | none | **warmup + cosine** (6% warmup) |
| Loss | BCELoss + Sigmoid in model | **BCEWithLogitsLoss** (numerically stable) |
| Split | first 13k for train (unstratified) | 90/10 **stratified** on 25k |
| sen_len | 30 | **400** with **head+tail truncation** (keep 30% head / 70% tail) |
| Tokenize | raw regex | lowercase + strip `<br />`/HTML/URL, preserve contractions (`don't`), collapse `!!!` / `loooove` |
| Embedding | fixed w2v CBOW, vector=250 | **skip-gram + negative + subsampling**, vector=300, fine-tuned |
| Model | 1-layer unidir LSTM, last step | **2-layer BiLSTM** (hidden=256) + attention/max/mean pooling + LayerNorm + GELU MLP head |
| Pad handling | last step includes pads | masked pooling, PAD=0 idx |
| Regularization | dropout=0.5 | dropout=0.5 + **embed dropout 0.3** + **embed Gaussian noise 0.05** + **word dropout 0.10** + grad clip + weight decay |
| Weight averaging | none | **EMA (decay=0.999)** over training steps — single model, ensemble-rule compliant |
| Unlabeled 50k | unused | **Self-training** 2 rounds, conf>=0.95 / <=0.05, cap 15k, finetune lr=3e-4 |
| Logging | print to stdout | logger to file + stdout + per-run dir |

### Why these help

- **Longer head+tail context.** IMDb reviews put the verdict near the end; truncating only head (p90=547 tokens) throws away the punch line.
- **Word dropout & embed noise.** Forces the LSTM to rely on multiple cues instead of memorising rare training tokens — closes the ~7-pt train/val gap we saw at val=90%.
- **EMA weights.** Reduces the stochastic noise in SGD-ish training; the EMA val accuracy is usually the most honest early-stop signal and the best checkpoint.
- **Stricter self-training + lower finetune LR.** The previous run regressed during self-training because finetune restarted Adam at lr=1e-3 on noisy pseudo labels. Stricter threshold + 3e-4 keeps gains.

## Tuning knobs (edit `config.yaml`)

- `preprocess.sen_len` / `head_ratio` — 300/0.3 is faster; 500/0.25 catches even longer reviews
- `model.hidden_dim` — 256 is a sweet spot; 384 helps if you have VRAM
- `train.ema_decay` — 0.995 (faster adapting) / 0.999 (smoother); disable with 0
- `train.word_dropout` — 0.05–0.15
- `self_training.pos_threshold` / `neg_threshold` — higher = cleaner pseudo labels
- `self_training.finetune_lr` — try 1e-4 / 3e-4 / 5e-4

## Reproducibility

- Developed and tested with Python 3.11.15.
- Core non-PyTorch package versions are pinned in `requirements.txt`.
- OS/GPU-specific PyTorch installs are kept in `requirements-mac.txt` and
  `requirements-win-gpu.txt`.
- `gensim` is kept on a conservative NumPy/SciPy stack to avoid Word2Vec C-extension issues.
- Global seed in `config.yaml` (default 42); applied to `random`, `numpy`, `torch`.
- Each run snapshots the config used, alongside logs & checkpoints.
