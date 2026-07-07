"""Image utility helpers."""

import base64
from pathlib import Path

from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger(__name__)


def to_base64(image_path: str) -> str:
    """Convert image file to base64 string.

    image_path may be:
      - an absolute path (legacy local dev)
      - a relative path stored from data/processed/ (e.g. "Col/doc/fig.png")
    """
    try:
        path = Path(image_path)
        if not path.is_absolute():
            # Relative path: resolve from data/processed/
            path = get_settings().data_processed_dir / path
        if not path.exists():
            logger.warning(f"Image not found: {image_path}")
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        logger.warning(f"Image not found: {image_path}")
        return ""
