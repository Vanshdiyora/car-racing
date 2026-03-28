"""Structured logging setup for training and racing."""

from __future__ import annotations

import csv
import logging
import pathlib
import sys
from typing import Any


def setup_logger(
    name: str = "car_racing",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """Return a configured logger with console (and optional file) output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        path = pathlib.Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(path), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class CSVLogger:
    """Append-only CSV logger for training metrics."""

    def __init__(self, path: str | pathlib.Path, fieldnames: list[str]) -> None:
        self._path = pathlib.Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames = fieldnames
        self._file = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        # Write header only for new files
        if self._path.stat().st_size == 0:
            self._writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._file.flush()

    def close(self) -> None:
        self._file.close()
