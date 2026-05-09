"""Miscellaneous helpers: seeding, tokenization, device selection."""
from __future__ import annotations

import os
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch

# Keep contractions like don't / it's as single tokens so that negation / possessives
# are preserved. Collapse runs of the same emphatic punctuation (!!! -> !!!) into a
# single token so our vocab doesn't fragment on stylistic variation.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]+|[.,;:]")
_BR_RE = re.compile(r"<\s*br\s*/?\s*>", flags=re.IGNORECASE)
_HTML_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # loooove -> looove


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
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    text = _REPEAT_CHAR_RE.sub(r"\1\1\1", text)
    return _TOKEN_RE.findall(text)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def git_sha(cwd: Path) -> str:
    """Return the current git commit sha, or ``"unknown"`` if not available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown"
