"""Typed configuration dataclasses for the HW4 pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    data_dir: str
    train_csv: str = "train.csv"
    unlabel_csv: str = "train_unlabel.csv"
    test_csv: str = "test.csv"
    sample_submission: str = "sample_submission.csv"


@dataclass
class OutputConfig:
    result_root: str = "results"


@dataclass
class PreprocessConfig:
    sen_len: int = 200
    lowercase: bool = True
    min_count: int = 3
    head_ratio: float = 1.0              # 1.0 = head-only; <1.0 enables head+tail truncation
    w2v_vector_size: int = 256
    w2v_window: int = 5
    w2v_epochs: int = 10
    w2v_sg: int = 1
    w2v_negative: int = 10
    w2v_workers: int = 8
    w2v_sample: float = 1e-4              # gensim subsampling of frequent words
    w2v_cache_path: Optional[str] = None


@dataclass
class ModelConfig:
    hidden_dim: int = 192
    num_layers: int = 2
    dropout: float = 0.4
    embed_dropout: float = 0.3
    embed_noise_std: float = 0.0
    bidirectional: bool = True
    fix_embedding: bool = False
    pool: str = "attn_max_mean"
    attn_heads: int = 4                       # # heads for mhattn pool; ignored for single-head attn


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    val_ratio: float = 0.1
    early_stop_patience: int = 3
    lr_scheduler: str = "plateau"           # plateau | cosine | warmup_cosine | none
    lr_factor: float = 0.5
    lr_patience: int = 1
    warmup_ratio: float = 0.05
    ema_decay: float = 0.0                   # 0 disables EMA
    ema_warmup_epochs: int = 0               # epochs to skip EMA ckpt selection while shadow warms up
    word_dropout: float = 0.0                # augmentation during training
    label_smoothing: float = 0.0             # BCE target softening epsilon
    # Discriminative LR (only used when LM ckpt is loaded). When False, uses ``lr`` for all params.
    use_discriminative_lr: bool = False
    lr_embedding: float = 1.0e-5             # LM-pretrained: gentle update
    lr_lstm: float = 1.0e-4                  # LM-pretrained: medium update
    lr_head: float = 5.0e-4                  # randomly initialized: stronger update
    # ULMFiT-style gradual unfreezing (active only when LM ckpt loaded + disc-LR on).
    # Phase A: freeze embedding+LSTM, train head only with lr_head_warmup for K epochs.
    # Phase B: unfreeze everything, use disc-LR for the remaining (epochs - K) epochs.
    freeze_body_epochs: int = 0              # 0 disables gradual unfreezing (legacy behavior)
    lr_head_warmup: float = 1.0e-3           # head-only warmup LR during frozen phase


@dataclass
class SelfTrainingConfig:
    enable: bool = True
    rounds: int = 2
    pos_threshold: float = 0.9
    neg_threshold: float = 0.1
    max_pseudo_per_round: int = 30000
    balance_pseudo: bool = True              # take equal #pos and #neg per round
    finetune_epochs: int = 6
    finetune_lr: Optional[float] = None      # defaults to train.lr when None


@dataclass
class LMConfig:
    """LSTM Language Model pretraining configuration.

    See LSTM_LM_DESIGN.md for rationale of each knob.

    Loading semantics: an LM ckpt is loaded iff a path is supplied EITHER via
    ``--lm`` CLI flag (preferred) OR via ``ckpt_path`` here. There is no
    separate "enable" flag — the presence of a path is the switch.
    """
    ckpt_path: Optional[str] = None          # if set, load this LM ckpt at train time
    # Architecture (must align with classifier for clean weight transfer)
    hidden_dim: int = 192
    num_layers: int = 2
    embed_dropout: float = 0.3
    dropout: float = 0.4
    tie_weights: bool = True                 # tie embedding <-> projection (via adapter)
    init_from_w2v: bool = True               # warm-start LM embedding from w2v matrix
    # Data
    bptt_len: int = 128
    val_ratio: float = 0.05                  # fraction of unlabeled docs used as LM val
    val_source: str = "unlabeled_only"       # enforce: never use labeled docs for LM val
    include_labeled_in_train: bool = False   # §4.3 simplified isolation: keep labeled fully out of LM
    include_test_in_train: bool = True       # transductive: test text (no labels) is fine
    # Optimization
    batch_size: int = 64
    epochs: int = 8
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-6
    grad_clip: float = 0.5                   # LSTM-LMs gradient-explode famously easily
    warmup_ratio: float = 0.05
    early_stop_patience: int = 2             # on val perplexity
    seed: int = 42


@dataclass
class InferenceConfig:
    batch_size: int = 256


@dataclass
class Config:
    data: DataConfig
    output: OutputConfig = field(default_factory=OutputConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    self_training: SelfTrainingConfig = field(default_factory=SelfTrainingConfig)
    lm: LMConfig = field(default_factory=LMConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    seed: int = 42

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return Config(
            data=DataConfig(**raw["data"]),
            output=OutputConfig(**raw.get("output", {})),
            preprocess=PreprocessConfig(**raw.get("preprocess", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            self_training=SelfTrainingConfig(**raw.get("self_training", {})),
            lm=LMConfig(**raw.get("lm", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
            seed=raw.get("seed", 42),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def dump_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
