"""Logging utilities for nvidia-inst."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOG_DIR = Path("/var/log/nvidia-inst")
LOG_FILE = LOG_DIR / "install.log"


def setup_logging(debug: bool = False, dry_run: bool = False) -> None:
    """Configure logging for the application.

    Args:
        debug: Enable debug-level logging if True.
        dry_run: Skip file logging if True (no root access).
    """
    level = logging.DEBUG if debug else logging.INFO

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    handlers = [console_handler]

    if not dry_run:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                LOG_FILE,
                maxBytes=5_242_880,
                backupCount=5,
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            handlers.append(file_handler)
        except PermissionError:
            pass

    logging.basicConfig(
        level=level,
        handlers=handlers
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Name of the logger (usually __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
