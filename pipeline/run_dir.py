"""Run-folder lifecycle: create, snapshot config / CLI, finalize with acc tag."""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Sequence


def setup_run_dir(
    root: Path,
    cfg_result_root: str,
    run_name: str,
    *,
    config_path: Path,
    cli_argv: Sequence[str],
    resolved_overrides: Sequence[str],
    log_builder,
) -> tuple[Path, logging.Logger]:
    """Create ``results/<run_name>/`` and write reproducibility artifacts.

    Returns the run directory path and a configured logger. ``log_builder``
    is the function used to wire up file + stdout handlers (passed in to
    avoid hard-coding utils.logger here, keeping the module testable).
    """
    run_dir = root / cfg_result_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = log_builder(run_dir)
    logger.info(f"run dir: {run_dir}")

    shutil.copy(config_path, run_dir / "config.source.yaml")
    (run_dir / "cli_args.txt").write_text(
        "python " + " ".join(cli_argv) + "\n",
        encoding="utf-8",
    )
    (run_dir / "resolved_overrides.txt").write_text(
        "# Reproduce: python train.py --config <path/to/config.source.yaml>"
        + (" " + " ".join(resolved_overrides) if resolved_overrides else "")
        + "\n",
        encoding="utf-8",
    )
    return run_dir, logger


def finalize_run_dir(run_dir: Path, run_name: str, val_acc: float, logger: logging.Logger) -> Path:
    """Append the val-acc tag to the run folder name for at-a-glance comparison.

    Returns the final path (or the original on failure). Closes the logger's
    file handler first so Windows doesn't lock the directory during rename.
    """
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)
    try:
        acc_tag = f"{val_acc * 100:.2f}"
        new_dir = run_dir.parent / f"{run_name}_{acc_tag}"
        run_dir.rename(new_dir)
        return new_dir
    except Exception as e:  # noqa: BLE001
        print(f"[done] artifacts at {run_dir}  (rename skipped: {e})")
        return run_dir
