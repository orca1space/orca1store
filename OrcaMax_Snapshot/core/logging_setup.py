"""Hermes logging setup.

Centralised logging so all modules can log without print() statements.
Logs go to:
  - console (stderr) by default
  - D:\\Hermes\\data\\hermes.log (rotated weekly)
"""
import logging
import logging.handlers
import sys
from pathlib import Path

# Project root
HERMES_ROOT = Path(__file__).parent.parent
LOG_DIR = HERMES_ROOT / "data"
LOG_FILE = LOG_DIR / "hermes.log"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger once, return named logger for this module."""
    global _configured
    logger = logging.getLogger("hermes")

    if not _configured:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False

        # File handler with rotation
        try:
            fh = logging.handlers.RotatingFileHandler(
                LOG_FILE, maxBytes=5_000_000, backupCount=3,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
            logger.addHandler(fh)
        except (OSError, PermissionError):
            # If we can't write the log file, just continue without it
            pass

        # Console handler
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        logger.addHandler(ch)

        _configured = True

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a module."""
    setup_logging()
    return logging.getLogger(f"hermes.{name}")
