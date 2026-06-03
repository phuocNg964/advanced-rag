"""
Centralized logging configuration for the project.
Only shows logs from your own modules (src.*), automatically hides all third-party logs.
"""

import logging
import sys

# Your project's module prefix
PROJECT_PREFIX = "src"


class ProjectOnlyFilter(logging.Filter):
    """Only allow logs from project modules."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(PROJECT_PREFIX)


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """
    Configure logging to only show your project's logs.
    All third-party library logs are automatically hidden.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler with project-only filter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ProjectOnlyFilter())
    root_logger.addHandler(console_handler)

    # Add file handler with project-only filter if specified
    if log_file:
        import os

        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ProjectOnlyFilter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module."""
    return logging.getLogger(name)
