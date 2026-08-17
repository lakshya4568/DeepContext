"""Structured logging configuration for Deep Context Platform."""

import logging
import sys

from deep_context.core.config import settings


def setup_logging(log_level: str | None = None) -> logging.Logger:
    level_name = log_level or settings.log_level
    level = getattr(logging, level_name.upper(), logging.INFO)

    logger = logging.getLogger("deep_context")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
