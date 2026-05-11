# HW4 Text Sentiment Classification — BiLSTM + LM Pretraining + Self-Training

Self-designed single-model pipeline for the HW4 movie-review sentiment task
(25k labeled / 50k unlabeled / 25k test). The final classifier is a 2-layer
BiLSTM with attention pooling, optionally initialized from self-trained
forward / backward LSTM language models (ULMFiT-style), then fine-tuned on
labeled data and refined with multi-round self-training on the unlabeled set.

## Compliance with `HW4 Rules`

- **Single LSTM-like model.** The classifier is one `nn.Module` (a BiLSTM
  with a small attention head). At inference time there is exactly one
  forward pass, one set of weights, and one prediction per example.
- **No external pre-trained models.** Every weight is trained from scratch
  on the assignment data:
  - Word2Vec embeddings: trained on labeled + unlabeled + test text via
    `gensim` (skip-gram + negative sampling). `gensim` is used as a tool to
    *train* w2v from scratch; no pre-trained word vectors are loaded.
  - LSTM language model(s): trained from scratch on unlabeled (+ test) text
    in `pretrain_lm.py`. The LM is ours, then transferred as initialization
    to the classifier's BiLSTM body.
- **No ensemble.** No averaging or voting across multiple models. We do
  maintain an EMA of the classifier's weights during training, but the EMA
  is just a single weight tensor used as a stand-alone checkpoint.
- **No transformers / nltk / external NLP packages.** Dependencies are
  `numpy`, `scipy`, `pandas`, `gensim`, `pyyaml`, `tqdm`, plus `torch`.
- **Self-training** on unlabeled data is explicitly encouraged by the rules.

## File structure

```
movie-review-sentiment-classification/
├── train.py                       # classifier pipeline (train + self-train + predict)
├── pretrain_lm.py                 # LSTM-LM pretraining (--reverse for backward LM)
├── config.yaml                    # all hyperparameters
├── requirements.txt               # numpy / scipy / pandas / gensim / pyyaml / tqdm
├── requirements-mac.txt           # adds the macOS PyTorch wheel
├── requirements-win-gpu.txt       # adds the Windows CUDA PyTorch wheel
│
├── data/
│   ├── preprocess.py              # CSV loaders, tokenizer, Word2Vec, Vocab, head+tail trunc.
│   ├── datasets.py                # SenDataset (with word-dropout), PseudoLabeledDataset
│   └── lm_dataset.py              # LM corpus assembly (doc-level split) + BPTT dataset
│
├── models/
│   ├── lstm.py                    # 2-layer BiLSTM classifier + attention/max/mean pooling
│   ├── lstm_lm.py                 # uni-directional LSTM language model (tied weights)
│   └── regularization.py          # AWD-LSTM-style LockedDropout + WeightDrop (DropConnect)
│
├── engine/
│   ├── trainer.py                 # classifier loop, EMA, early stop, warmup-cosine LR
│   ├── lm_trainer.py              # LM loop, perplexity, grad clip
│   ├── ema.py                     # ModelEMA helper
│   └── inference.py               # predict_probs, predict.csv writer
│
├── utils/
│   ├── config.py                  # typed Config dataclasses (yaml <-> dataclass)
│   ├── logger.py                  # build_logger (stdout + file)
│   ├── misc.py                    # set_seed, pick_device, timestamp, git_sha
│   ├── vocab_io.py                # idx2word.json + vocab hash (LM/classifier alignment)
│   └── weight_transfer.py         # LM -> BiLSTM weight copy (fwd + optional bwd direction)
│
├── results/<timestamp>/           # one folder per classifier run
│   ├── config.source.yaml         # exact config used
│   ├── cli_args.txt               # command-line invocation
│   ├── resolved_overrides.txt     # absolute paths after resolving 'latest'
│   ├── train.log                  # full log
│   ├── ckpt.pt / ckpt.raw.pt      # best EMA / RAW checkpoint by val acc
│   ├── ckpt_self_train_r{1..N}.pt # per-round self-training checkpoints
│   ├── predict.csv                # primary submission (best EMA val acc)
│   ├── predict_ema.csv            # EMA-only submission
│   ├── predict_raw.csv            # RAW-weights submission (often slightly better)
│   └── summary.txt                # final metrics
│
└── results_lm/<timestamp>/        # one folder per LM pretraining run
    ├── lm_ckpt.pt                 # LM weights + vocab metadata + direction tag
    ├── idx2word.json              # exact vocab order (used for cross-run integrity check)
    ├── vocab_hash.txt             # md5 of idx2word
    ├── w2v.model                  # the w2v model (single source of truth for vocab)
    ├── lm.log                     # LM training log
    └── summary.txt                # best val PPL, epoch, etc.
```

`results_lm/LATEST` and `results_lm/LATEST_BW` are simple text files holding
the run name of the most recent forward / backward LM run, used by
`train.py --lm latest` and `train.py --lm-bw latest`.

## Installation

Pick the requirements file that matches your machine:

```bash
# macOS (CPU / MPS):
pip install -r requirements-mac.txt

# Windows + NVIDIA GPU (pinned CUDA wheel):
pip install -r requirements-win-gpu.txt
```

`requirements-win-gpu.txt` pins the CUDA 13.0 PyTorch wheel via
`--extra-index-url https://download.pytorch.org/whl/cu130`. Edit that URL
if your driver needs a different runtime.

Quick GPU sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## End-to-end training

The full pipeline has up to three stages. Stage 1 is required; stages 2+ are
optional and additive.

### Stage 1 — Word2Vec + classifier (no LM)

```bash
python train.py
```

This trains w2v from scratch on labeled+unlabeled+test text, then trains the
BiLSTM classifier with EMA + warmup-cosine LR + early stop, then runs
`self_training.rounds` rounds of self-training on the unlabeled set, and
finally writes `results/<timestamp>/predict.csv`.

### Stage 2 — Forward LSTM language-model pretraining

```bash
python pretrain_lm.py                       # forward LM
python train.py --lm latest                 # classifier with LM init
```

`pretrain_lm.py` reuses the cached w2v from a previous classifier run (see
`preprocess.w2v_cache_path` in `config.yaml`) so the LM and classifier share
the exact same vocabulary. The LM checkpoint and a vocab hash are saved to
`results_lm/<timestamp>/lm_ckpt.pt`; `train.py --lm latest` validates the
hash before transferring weights and refuses to load on any mismatch.

When an LM is loaded, `train.py` switches into ULMFiT-style fine-tuning:

1. **Phase A — head-only warmup.** The embedding and LSTM body are frozen
   for `train.freeze_body_epochs` (default 2) so the randomly-initialized
   attention + classifier head can settle without disturbing the pretrained
   body. Best Phase-A checkpoint is restored before Phase B.
2. **Phase B — discriminative LR fine-tuning.** All parameters unfrozen.
   Three LR groups: `lr_embedding < lr_lstm < lr_head`, so the heavily
   pretrained embedding moves slowly while the head can still adapt.

### Stage 3 — Backward LM (bidirectional pretraining)

```bash
python pretrain_lm.py --reverse             # backward LM (token streams reversed)
python train.py --lm latest --lm-bw latest  # classifier with both LMs
```

`pretrain_lm.py --reverse` trains an *identical* LM but on reversed token
streams, so it learns to predict the previous token. Its weights are
transferred into the classifier's reverse direction (`*_reverse` slots of
the BiLSTM). The classifier itself is unchanged: still a single
`nn.LSTM(bidirectional=True)` with one forward pass at inference.

This is the ULMFiT bidirectional-LM trick (Howard & Ruder 2018). It does
not introduce ensemble at inference: only one classifier is loaded, only
one prediction is produced per example.

## Run-folder hygiene

Each `train.py` invocation writes a new folder `results/<timestamp>[_tag]/`
containing the full source `config.yaml`, the CLI invocation, the resolved
LM paths, the log, and every checkpoint produced along the way. This makes
it trivial to diff hyperparameters across experiments. Same convention for
`pretrain_lm.py` runs under `results_lm/`.

```bash
python train.py --tag h256_emb04         # appends '_h256_emb04' to the run folder
python train.py --config my_config.yaml  # custom config file
```

## Implementation notes

### Tokenization
Lowercases, strips `<br />` and most HTML / URL fragments, preserves
contractions (`don't`, `we'll`), and collapses character-repetition
(`!!!!!` -> `!!`, `loooove` -> `loove`). All in `data/preprocess.py`, no
NLP packages.

### Long-review handling
IMDb reviews are long (mean 260, p95 ~660 tokens) and the verdict tends to
be near the end. We use **head + tail truncation**: keep `head_ratio` of
`sen_len` from the start and the rest from the end. Default `sen_len=400`
with `head_ratio=0.3`.

### Classifier architecture (`models/lstm.py`)
- 2-layer BiLSTM, hidden=256, with optional AWD-LSTM regularizers
  (LockedDropout, WeightDrop).
- Pooling: mean + max + 1-head additive attention (`pool=attn_max_mean`),
  all over the same masked time dimension.
- Head: LayerNorm -> GELU MLP -> linear logit. BCEWithLogitsLoss for
  numerical stability.
- Word dropout on input ids + Gaussian noise on embeddings during training
  for additional regularization.

### Language model (`models/lstm_lm.py`)
- Uni-directional 2-layer LSTM with **tied input/output embedding** via a
  small adapter that projects hidden -> embed dim. EOS lives at index
  `V_cls` (one row past the classifier vocab) so the classifier never
  "sees" it after weight transfer.
- BPTT length 192, batch size 32, 30 epochs, warmup-cosine LR.
- Optional LockedDropout on outputs and DropConnect on `weight_hh_l*`
  (AWD-LSTM regularizers).

### LM-to-classifier transfer (`utils/weight_transfer.py`)
- Embedding is clipped to `V_cls` rows (drops the LM-only EOS row).
- Forward direction: `weight_ih_l0`, `weight_hh_l0`, biases copied directly.
  At layer 1 the BiLSTM input dim is 2H (concat of fwd+bwd of layer 0) but
  the LM input dim is H — we copy the LM weight into the first H columns
  and zero the rest, so the classifier starts as a strict generalisation
  of the LM.
- Backward direction (when `--lm-bw` is provided): the same trick mirrored
  to `*_reverse` slots, with layer-1 zero-padding on the *first* H columns
  (since the layer-0 backward output lives in the second half of the 2H
  concat input).
- Vocab integrity is verified element-wise by an md5 hash before any copy
  happens; the loader raises on any mismatch.

### Self-training loop
After init training, repeatedly:
1. Score the unlabeled pool with the current best classifier.
2. Pick the most-confident pseudo labels above `pos_threshold` and below
   `neg_threshold`, balanced 50/50, capped at `max_pseudo_per_round`.
3. Concatenate pseudo + labeled and fine-tune for `finetune_epochs` with a
   cosine LR. EMA continues across rounds.

The pool shrinks each round (selected items are removed) so later rounds
draw from progressively harder examples. The "global best" checkpoint is
updated only if val accuracy actually improved that round.

### EMA (single-model)
`engine/ema.py` maintains a shadow weight tensor updated each step with
`shadow = decay * shadow + (1 - decay) * w`. The EMA produces a single
checkpoint (`ckpt.pt`); the un-averaged "raw" weights are saved separately
(`ckpt.raw.pt`) so we can pick whichever generalises better at submission
time. This is *not* ensemble — there is still only one model file used at
inference.

## Tuning knobs (edit `config.yaml`)

- `preprocess.sen_len` / `head_ratio` — 400/0.3 is the default; lower
  `sen_len` is faster, higher catches longer reviews
- `model.hidden_dim` / `model.dropout` / `model.embed_dropout` — capacity
  vs regularization trade-off
- `model.locked_dropout` / `model.weight_drop` — AWD-LSTM regularizers; 0
  by default in the classifier (they interact awkwardly with EMA)
- `lm.locked_dropout` / `lm.weight_drop` — *do* enable on the LM (no EMA
  there); we use 0.20 / 0.30 to lower LM perplexity
- `train.freeze_body_epochs` / `lr_head_warmup` — Phase-A schedule
- `train.lr_embedding` / `lr_lstm` / `lr_head` — discriminative LR groups
  for Phase B
- `train.ema_decay` — 0.999 default; 0 disables
- `self_training.rounds` / `pos_threshold` / `neg_threshold` /
  `max_pseudo_per_round` / `finetune_epochs` / `finetune_lr`

## Reproducibility

- Tested on Python 3.11.15 with the requirements pinned in `requirements*.txt`.
- Global seed in `config.yaml` (default 42), applied to `random`, `numpy`,
  and `torch`.
- Every run dumps `config.source.yaml` + `cli_args.txt` +
  `resolved_overrides.txt` next to the checkpoint, so any run can be
  reproduced verbatim with `python train.py --config <run>/config.source.yaml`
  plus the recorded CLI overrides.
- LM and classifier runs cross-validate the vocabulary by an md5 hash of
  the index-to-word list before any weight transfer, so a stale w2v cache
  cannot silently misalign embeddings.
