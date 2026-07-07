from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc import (
    DocItemLabel,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TitleItem,
)
from PIL import Image

log = logging.getLogger(__name__)

IMAGE_SCALE = 1.5
ACCELERATOR = AcceleratorDevice.AUTO
NUM_THREADS = 8
NATIVE_TEXT_MIN_CHARS = 50
DOC_TYPE_SAMPLE_PAGES = 3
CHUNK_MAX_TOKENS = 250
CHARS_PER_TOKEN = 4


@dataclass
class ProcessedChunk:
    # Stable ingestion handoff: text, table, and image chunks stay separate.
    content: str
    chunk_type: str  # "text", "table", "image"
    page_number: int | None
    section: str
    image_path: str
    source: str = ""


@dataclass
class ParseResult:
    # Keep Docling's raw result plus lazy normalized chunks.
    filename: str
    source: str
    page_count: int
    doc_type: str
    document: Any
    conversion_result: Any
    parse_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    _elements: list[ProcessedChunk] | None = field(default=None, repr=False)

    @property
    def elements(self) -> list[ProcessedChunk]:
        if self._elements is None:
            self._elements = post_process_items(
                self.document,
                source=self.source or self.filename,
            )
        return self._elements


# Post-processing pipeline
class ApproximateTokenizer(BaseTokenizer):
    """Offline token estimator that avoids Hugging Face/tiktoken downloads."""

    max_tokens: int = CHUNK_MAX_TOKENS
    chars_per_token: int = CHARS_PER_TOKEN

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return (len(text) + self.chars_per_token - 1) // self.chars_per_token

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self.count_tokens


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Use Docling chunking while preserving Markdown table text."""

    def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


def _build_chunker() -> HybridChunker:
    return HybridChunker(
        merge_peers=True,
        tokenizer=ApproximateTokenizer(),
        serializer_provider=MarkdownTableSerializerProvider(),
    )


def _text_chunk_labels() -> set[DocItemLabel]:
    # Exclude floating/caption labels so figures and tables stay separate.
    return MarkdownParams().labels - {
        DocItemLabel.CAPTION,
        DocItemLabel.CHART,
        DocItemLabel.PICTURE,
        DocItemLabel.TABLE,
    }


def _get_page(item: Any) -> int | None:
    # Use the first Docling provenance span as the representative page.
    prov = item.prov[0] if getattr(item, "prov", None) else None
    return prov.page_no if prov else None


def _save_image(img: Image.Image, path: Path) -> None:
    # Store images on disk so the vision summarizer can load them later.
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


def _picture_groups(document: Any) -> list[tuple[PictureItem, str, list[PictureItem]]]:
    # A captioned picture starts a group; following captionless pictures join it.
    groups: list[tuple[PictureItem, str, list[PictureItem]]] = []
    current_anchor: PictureItem | None = None
    current_group: list[PictureItem] | None = None

    for picture in getattr(document, "pictures", []):
        caption = picture.caption_text(document).strip()
        if caption:
            current_anchor = picture
            current_group = [picture]
            groups.append((picture, caption, current_group))
            continue

        if current_anchor is not None and _get_page(picture) == _get_page(current_anchor):
            current_group.append(picture)

    return groups


def _compose_picture_group(
    document: Any,
    group: list[PictureItem],
) -> Image.Image | None:
    # Stack split Docling crops into one figure image for vision ingestion.
    images: list[Image.Image] = []
    for picture in group:
        try:
            image = picture.get_image(document)
        except Exception:
            image = None
        if image is not None:
            images.append(image.convert("RGB"))

    if not images:
        return None
    if len(images) == 1:
        return images[0]

    canvas_width = max(image.width for image in images)
    canvas_height = sum(image.height for image in images)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    y_offset = 0
    for image in images:
        canvas.paste(image, (0, y_offset))
        y_offset += image.height

    return canvas


def _first_page(items: list[Any]) -> int | None:
    # Anchor multi-item chunks to their first visible page.
    pages = [_get_page(item) for item in items]
    return min((page for page in pages if page is not None), default=None)


def _section_from_headings(headings: list[str] | None) -> str:
    # Flatten Docling's heading stack into simple metadata.
    return " / ".join(heading.strip() for heading in headings or [] if heading.strip())


def _sections_for_items(document: Any, item_type: type) -> dict[str, str]:
    """Map item self-refs to the active heading path in document order."""
    sections: dict[str, str] = {}
    heading_by_level: dict[int, str] = {}

    for item, _level in document.iterate_items():
        # Rebuild heading context for floating items handled outside text chunking.
        if isinstance(item, TitleItem):
            heading_by_level = {0: item.text}
            continue

        if isinstance(item, SectionHeaderItem):
            # Replace headings at the same or deeper level for the new section.
            level = item.level
            heading_by_level = {
                key: value for key, value in heading_by_level.items() if key < level
            }
            heading_by_level[level] = item.text
            continue

        if isinstance(item, item_type):
            headings = [
                heading_by_level[key]
                for key in sorted(heading_by_level)
                if heading_by_level[key].strip()
            ]
            sections[item.self_ref] = " / ".join(headings)

    return sections


def _create_image_chunks(
    document: Any,
    output_dir: Path | None,
    source: str = "",
) -> list[ProcessedChunk]:
    # Use Docling caption links instead of nearby-text regex guesses.
    image_sections = _sections_for_items(document, PictureItem)
    chunks: list[ProcessedChunk] = []
    figure_index = 0

    for picture, caption, picture_group in _picture_groups(document):
        figure_index += 1
        image_path = ""
        if output_dir is not None:
            image = _compose_picture_group(document, picture_group)
            if image is not None:
                img_path = output_dir / f"figure-{figure_index}.png"
                _save_image(image, img_path)
                image_path = str(img_path)

        chunks.append(
            ProcessedChunk(
                content=caption,
                chunk_type="image",
                page_number=_get_page(picture),
                section=image_sections.get(picture.self_ref, ""),
                image_path=image_path,
                source=source,
            )
        )

    return chunks


def _create_table_chunks(document: Any, source: str = "") -> list[ProcessedChunk]:
    table_sections = _sections_for_items(document, TableItem)
    serializer = MarkdownTableSerializerProvider().get_serializer(document)
    chunks: list[ProcessedChunk] = []

    for table in getattr(document, "tables", []):
        content = serializer.serialize(item=table).text.strip()
        if not content:
            continue

        chunks.append(
            ProcessedChunk(
                content=content,
                chunk_type="table",
                page_number=_get_page(table),
                section=table_sections.get(table.self_ref, ""),
                image_path="",
                source=source,
            )
        )

    return chunks


def post_process_items(
    document: Any,
    output_dir: Path | None = None,
    source: str = "",
) -> list[ProcessedChunk]:
    """Transform a Docling document into RAG chunks with stable app metadata."""
    if document is None:
        return []

    text_chunker = _build_chunker()
    text_table_chunks: list[ProcessedChunk] = []
    for chunk in text_chunker.chunk(dl_doc=document, labels=_text_chunk_labels()):
        doc_items = chunk.meta.doc_items
        if any(isinstance(item, PictureItem | TableItem) for item in doc_items):
            # Floating items are handled by dedicated table/image passes.
            continue

        content = text_chunker.contextualize(chunk).strip()
        if not content:
            continue

        text_table_chunks.append(
            ProcessedChunk(
                content=content,
                chunk_type="text",
                page_number=_first_page(doc_items),
                section=_section_from_headings(chunk.meta.headings),
                image_path="",
                source=source,
            )
        )

    table_chunks = _create_table_chunks(document, source)
    image_chunks = _create_image_chunks(document, output_dir, source)

    # Keep reading order stable for predictable ingestion chunk IDs.
    all_chunks = text_table_chunks + table_chunks + image_chunks
    all_chunks.sort(key=lambda c: (c.page_number or 0, _type_order(c.chunk_type)))
    return all_chunks


def _type_order(chunk_type: str) -> int:
    """Sort order within same page: text first, then tables, then images."""
    return {"text": 0, "table": 1, "image": 2}.get(chunk_type, 3)


_CONVERTERS: dict[bool, DocumentConverter] = {}
_PARSE_SEMAPHORE = threading.Semaphore(1)


def _count_pages(pdf_path: Path) -> int:
    # Count pages before Docling conversion for API/eval reporting.
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path), strict=False).pages)
    except Exception as exc:
        log.warning("[docling] page count failed for %s: %s", pdf_path, exc)
        return 1


def _detect_doc_type(pdf_path: Path) -> str:
    # Avoid OCR on native PDFs with a cheap PyMuPDF text sample.
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        pages_to_check = min(DOC_TYPE_SAMPLE_PAGES, len(doc))
        chars = sum(
            len((doc[idx].get_text() or "").strip())
            for idx in range(pages_to_check)
        )
        doc.close()
        return "native" if chars / max(pages_to_check, 1) >= NATIVE_TEXT_MIN_CHARS else "scanned"
    except Exception:
        return "scanned"


def _build_converter(do_ocr: bool) -> DocumentConverter:
    # Generate picture images because image chunks call PictureItem.get_image().
    options = PdfPipelineOptions()
    options.images_scale = IMAGE_SCALE
    options.generate_picture_images = True
    options.do_table_structure = True
    options.table_structure_options.mode = TableFormerMode.FAST
    options.do_ocr = do_ocr
    if do_ocr:
        # Vietnamese + English OCR supports scanned local papers.
        options.ocr_options = EasyOcrOptions(lang=["vi", "en"])

    options.accelerator_options = AcceleratorOptions(
        num_threads=NUM_THREADS,
        device=ACCELERATOR,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def get_converter(do_ocr: bool = False) -> DocumentConverter:
    # Reuse one converter per OCR mode because Docling startup is expensive.
    if do_ocr not in _CONVERTERS:
        _CONVERTERS[do_ocr] = _build_converter(do_ocr)
    return _CONVERTERS[do_ocr]


def warmup_docling(do_ocr: bool = False) -> None:
    """Initialize Docling's lazy PDF pipeline before the first real ingestion."""
    from pypdf import PdfWriter

    with tempfile.TemporaryDirectory(prefix="docling-warmup-") as tmp_dir:
        warmup_pdf = Path(tmp_dir) / "warmup.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with warmup_pdf.open("wb") as file:
            writer.write(file)
        get_converter(do_ocr=do_ocr).convert(str(warmup_pdf))


def parse_pdf(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    source: str | None = None,
) -> ParseResult:
    # Public parser entry point for ingestion.py and evals.
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    output_path = Path(output_dir) if output_dir is not None else None
    chunk_source = source or pdf_path.name
    page_count = _count_pages(pdf_path)
    doc_type = _detect_doc_type(pdf_path)

    # One conversion at a time avoids Docling CPU/RAM spikes under concurrent uploads.
    with _PARSE_SEMAPHORE:
        converter = get_converter(do_ocr=doc_type == "scanned")
        started = time.perf_counter()
        result = converter.convert(str(pdf_path))
        parse_seconds = time.perf_counter() - started

        pr = ParseResult(
            filename=pdf_path.name,
            source=chunk_source,
            page_count=page_count,
            doc_type=doc_type,
            document=result.document,
            conversion_result=result,
            parse_seconds=parse_seconds,
        )
        # If output_dir given, eagerly compute elements (saves images to disk)
        if output_path is not None:
            # Eager post-processing makes image files exist before returning.
            pp_start = time.perf_counter()
            pr._elements = post_process_items(
                result.document,
                output_path,
                source=chunk_source,
            )
            pr.postprocess_seconds = time.perf_counter() - pp_start
    return pr
