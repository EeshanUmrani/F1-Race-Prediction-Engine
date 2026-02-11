# utils/logging_utils.py

"""
Lightweight logging utilities for the race-sim-v2 project.

You *can* just use the standard `logging` module everywhere, but this
helper gives you:

    - a single place to configure log format / level,
    - automatic creation of a log directory,
    - a simple `get_logger(__name__)` function.

Usage:

    from utils.logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("Something happened")
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Optional

from config import LOGS_DIR, PROJECT_ROOT


# We keep a simple cache so multiple calls with the same name
# return the same logger instance.
_LOGGER_CACHE: dict[str, Logger] = {}


def _ensure_log_dir(path: Path) -> None:
    """
    Make sure the directory for log files exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def configure_root_logger(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_stdout: bool = True,
    filename: str = "race_sim.log",
) -> None:
    """
    Configure the root logger for the entire project.

    Call this once near the start of main.py *if* you want centralized
    logging. If you never call it, `get_logger` will still work but with
    Python's default basicConfig.

    Args:
        level:
            Logging level (e.g., logging.INFO, logging.DEBUG).
        log_to_file:
            If True, write logs to LOGS_DIR / filename.
        log_to_stdout:
            If True, also log to stdout.
        filename:
            Name of the log file inside LOGS_DIR.
    """
    handlers: list[logging.Handler] = []

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_file:
        _ensure_log_dir(LOGS_DIR)
        file_path = LOGS_DIR / filename
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setFormatter(formatter)
        handlers.append(fh)

    if log_to_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        handlers.append(sh)

    # If root already has handlers, avoid duplicating them
    root = logging.getLogger()
    if root.handlers:
        # Just adjust level if already configured
        root.setLevel(level)
        return

    logging.basicConfig(level=level, handlers=handlers)


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
) -> Logger:
    """
    Get a logger with a given name, configured at a given level.

    If no name is provided, we use the module-level name "__main__".
    The first time this is called, if no handlers exist on the root
    logger, we set up a minimal stdout-only configuration so logs are
    visible.

    Args:
        name:
            Logger name (usually __name__ in the caller).
        level:
            Logging level for this logger.

    Returns:
        logging.Logger instance.
    """
    global _LOGGER_CACHE

    if name is None:
        name = "__main__"

    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # If root logger has no handlers, configure a minimal default
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    _LOGGER_CACHE[name] = logger
    return logger
