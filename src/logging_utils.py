"""Console presentation helpers for the pipeline's `__main__` entrypoints.

Stdlib-only (no new dependency): each stage prints a titled banner, its
key metrics aligned in a column, and how long the stage took. This replaces
bare `print(f"key=value")` lines with a consistent, readable format across
every module, while staying print-based per the project's plan (no logging
framework/service — just clearer console formatting).
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Mapping

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


@contextmanager
def stage(logger: logging.Logger, title: str) -> Iterator[None]:
    """Wrap a pipeline stage with a banner and elapsed-time footer."""
    rule = "=" * max(len(title), 8)
    logger.info(rule)
    logger.info(title)
    logger.info(rule)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("-> completed in %.2fs", elapsed)
        logger.info("")


def log_metrics(logger: logging.Logger, metrics: Mapping[str, object]) -> None:
    """Print metrics as an aligned `key : value` block."""
    if not metrics:
        return
    width = max(len(key) for key in metrics)
    for key, value in metrics.items():
        if isinstance(value, float):
            value = f"{value:.4f}"
        logger.info("  %-*s : %s", width, key, value)
