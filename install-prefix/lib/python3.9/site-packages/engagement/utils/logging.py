"""Consistent console/file logging."""

import logging
import os
from pathlib import Path
from typing import Optional, Union


def configure_logging(log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
    level = getattr(logging, os.environ.get("ENGAGEMENT_LOG_LEVEL", "INFO").upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("engagement")

