"""
logger_config.py
----------------
Centralised logging factory for the MEI + Census pipeline.

Key changes from v1.0
---------------------
- Log path is injected at call-time; no hardcoded Z: drive path.
- Callers (cron_runner) supply the path so the module is portable
  and unit-testable without a network drive.
- RotatingFileHandler is only added when a path is supplied,
  preventing silent failures on machines where the path is unavailable.
- Duplicate-handler guard kept.
- propagate=False kept to avoid double-printing in complex hierarchies.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"


def get_logger(
    name: str = "MEI_Pipeline",
    log_file_path: Optional[str] = None,
    level: int = logging.INFO,
    rotation_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> logging.Logger:
    """
    Return a named logger, configuring handlers only on first call.

    Parameters
    ----------
    name : str
        Logger namespace (e.g. "Main", "FRED_Extract").
    log_file_path : str | None
        Absolute path for the rotating file handler.
        Pass ``None`` (or omit) for console-only logging.
    level : int
        Minimum log level (default INFO).
    rotation_bytes : int
        Max bytes before rotating (default 5 MB).
    backup_count : int
        Number of backup files to retain (default 3).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Idempotent – skip if already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT)

    # Console handler – always present
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Rotating file handler – only when a path is provided
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        try:
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=rotation_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning(
                "Could not open log file '%s': %s — logging to console only.",
                log_file_path,
                exc,
            )

    logger.propagate = False
    return logger
