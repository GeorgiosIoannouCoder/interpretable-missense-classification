"""Logging configuration shared across all scripts.

Logs go to both stderr and a per-run log file under ``logs/`` so that a
dropped SSH session can be diagnosed after the fact.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stderr and (optionally) a file.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.
    log_file : str or Path or None
        If given, a log file path under ``logs/``. Parent directories are
        created automatically.
    level : int
        Logging level, defaults to ``logging.INFO``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
