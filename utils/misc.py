"""Miscellaneous helpers: seeding, tokenization, device selection."""
from __future__ import annotations

import os
import random
import re
from datetime import datetime
from typing import List

import numpy as np
import torch

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_BR_RE = re.compile(r"<\s*br\s*/?\s*>", flags=re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    text = _BR_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
