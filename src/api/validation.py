from fastapi import HTTPException

from src.core.names import (
    InvalidNameError,
    validate_collection_name,
    validate_document_filename,
)


def collection_name_or_422(name: str) -> str:
    try:
        return validate_collection_name(name)
    except InvalidNameError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def document_name_or_422(filename: str) -> str:
    try:
        return validate_document_filename(filename)
    except InvalidNameError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
