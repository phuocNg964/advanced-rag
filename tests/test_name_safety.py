import pytest

from src.core.names import (
    InvalidNameError,
    resolve_child_path,
    validate_collection_name,
    validate_document_filename,
)


def test_collection_name_validation_accepts_safe_weaviate_names():
    assert validate_collection_name("ResearchPapers_2026") == "ResearchPapers_2026"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "researchPapers",
        "Research-Papers",
        "Research Papers",
        "A" * 101,
    ],
)
def test_collection_name_validation_rejects_unsafe_names(name):
    with pytest.raises(InvalidNameError):
        validate_collection_name(name)


def test_document_filename_validation_accepts_browser_fakepath_pdf():
    assert validate_document_filename(r"C:\fakepath\Paper 1.pdf") == "Paper 1.pdf"


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "../secret.pdf",
        "bad/name.pdf",
        "bad:name.pdf",
        "notes.txt",
        "bad\x00name.pdf",
    ],
)
def test_document_filename_validation_rejects_unsafe_names(filename):
    with pytest.raises(InvalidNameError):
        validate_document_filename(filename)


def test_resolve_child_path_rejects_paths_outside_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()

    assert resolve_child_path(root, "Collection", "doc.pdf").parent.name == "Collection"

    with pytest.raises(InvalidNameError):
        resolve_child_path(root, "..", "outside.pdf")
