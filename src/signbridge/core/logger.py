"""Module 17: Logging Subsystem.

Structured, rotating file and color console logging for SignBridge.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: Path = Path("logs"),
    log_level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    """Configures root logger with console and rotating file handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "signbridge.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times
    if root_logger.handlers:
        return

    # 1. Console Stream Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 2. Rotating File Handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Returns a module-specific logger instance."""
    return logging.getLogger(f"signbridge.{name}" if name else "signbridge")
