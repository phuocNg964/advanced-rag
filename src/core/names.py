import re
from pathlib import Path


COLLECTION_NAME_PATTERN = re.compile(r"^[A-Z][a-zA-Z0-9_]{0,99}$")
MAX_DOCUMENT_FILENAME_LENGTH = 180


class InvalidNameError(ValueError):
    """Raised when user-controlled names are not safe for storage paths."""


def validate_collection_name(name: str) -> str:
    """Return a valid Weaviate collection name or raise InvalidNameError."""
    if not COLLECTION_NAME_PATTERN.fullmatch(name or ""):
        raise InvalidNameError(
            "Collection names must start with a capital letter and contain only "
            "letters, numbers, and underscores."
        )
    return name


def validate_document_filename(filename: str) -> str:
    """Return a safe PDF filename or raise InvalidNameError."""
    normalized = _basename(filename).strip()
    if not normalized:
        raise InvalidNameError("Filename is required.")
    if len(normalized) > MAX_DOCUMENT_FILENAME_LENGTH:
        raise InvalidNameError(
            f"Filename must be {MAX_DOCUMENT_FILENAME_LENGTH} characters or fewer."
        )
    if normalized in {".", ".."} or ".." in Path(normalized).parts:
        raise InvalidNameError("Filename cannot contain path traversal.")
    if any(char in normalized for char in '<>:"/\\|?*'):
        raise InvalidNameError(
            'Filename cannot contain these characters: < > : " / \\ | ? *'
        )
    if any(ord(char) < 32 for char in normalized):
        raise InvalidNameError("Filename cannot contain control characters.")
    if not normalized.lower().endswith(".pdf"):
        raise InvalidNameError("Only PDF files are supported.")
    return normalized


def resolve_child_path(root: Path, *segments: str) -> Path:
    """Resolve a path under root and reject traversal outside that root."""
    root_path = root.resolve()
    target = root_path.joinpath(*segments).resolve()
    if target != root_path and root_path not in target.parents:
        raise InvalidNameError("Path escapes the configured data directory.")
    return target


def _basename(filename: str) -> str:
    normalized = (filename or "").replace("\\", "/")
    if "/" not in normalized:
        return normalized

    prefix, basename = normalized.rsplit("/", 1)
    if prefix.lower() == "c:/fakepath":
        return basename
    raise InvalidNameError("Filename cannot contain path separators.")
