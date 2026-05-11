"""Command-line argument parsing and LM-path resolution for train.py."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser(default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--tag", default=None, help="optional run name suffix")
    parser.add_argument(
        "--lm", default=None,
        help="Path to FORWARD LM ckpt to load. 'latest' = ckpt of most recent "
             "forward-LM run (results_lm/LATEST). Without this flag (and without "
             "lm.ckpt_path in yaml), behavior is the no-LM baseline. Discriminative "
             "LR is controlled by train.use_discriminative_lr in yaml (default true).",
    )
    parser.add_argument(
        "--lm-bw", default=None,
        help="Optional path to a BACKWARD LM ckpt (trained via pretrain_lm.py "
             "--reverse). 'latest' resolves to results_lm/LATEST_BW. When given, "
             "the BiLSTM reverse direction is initialized from this checkpoint "
             "(otherwise: random init). Requires --lm to also be set.",
    )
    return parser


def resolve_lm_arg(
    arg: str | None,
    *,
    root: Path,
    marker: str = "LATEST",
    flag: str = "--lm",
) -> Path | None:
    """Resolve --lm / --lm-bw CLI arg to an absolute lm_ckpt.pt path.

    None         -> None (use config.lm settings as-is).
    'latest'     -> ckpt from results_lm/<marker> (LATEST for fwd, LATEST_BW for bw).
    <path>       -> as-is (made absolute against ``root`` if relative).
    """
    if arg is None:
        return None
    if arg == "latest":
        latest_file = root / "results_lm" / marker
        if not latest_file.exists():
            raise SystemExit(
                f"{flag} latest: results_lm/{marker} not found. "
                f"Run pretrain_lm.py{' --reverse' if marker == 'LATEST_BW' else ''} first."
            )
        run_name = latest_file.read_text(encoding="utf-8").strip()
        candidate = root / "results_lm" / run_name / "lm_ckpt.pt"
        if not candidate.exists():
            raise SystemExit(
                f"{flag} latest: resolved to {candidate} but file not found."
            )
        return candidate
    p = Path(arg)
    if not p.is_absolute():
        p = (root / p).resolve()
    if not p.exists():
        raise SystemExit(f"{flag}: file does not exist: {p}")
    return p


def resolve_classifier_lm_paths(
    args: argparse.Namespace,
    cfg_lm_ckpt_path: str | None,
    *,
    root: Path,
) -> tuple[Path | None, Path | None]:
    """Apply the resolution order: --lm > cfg.lm.ckpt_path > None for forward,
    and --lm-bw > None for backward.
    """
    cli_fwd = resolve_lm_arg(args.lm, root=root, marker="LATEST", flag="--lm")
    if cli_fwd is not None:
        fwd = cli_fwd
    elif cfg_lm_ckpt_path:
        p = Path(cfg_lm_ckpt_path)
        fwd = p if p.is_absolute() else (root / p).resolve()
    else:
        fwd = None
    bw = resolve_lm_arg(args.lm_bw, root=root, marker="LATEST_BW", flag="--lm-bw")
    return fwd, bw
