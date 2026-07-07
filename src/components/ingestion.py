from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import List

from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.base import get_llm
from weaviate.util import generate_uuid5

from src.core.config import Settings, get_settings
from src.core.logger import get_logger
from src.core.weaviate_client import get_weaviate_client
from src.components.docling_parser import parse_pdf
from src.components.image_utils import to_base64
from src.prompts.prompts import IMAGE_SUMMARIZER_PROMPT

from opentelemetry import trace, context

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
IMAGE_SUMMARY_WORKERS = 3


def _to_relative_image_path(image_path: str, processed_base: Path) -> str:
    """Return image_path relative to processed_base (data/processed/).

    Storing a relative path (e.g. "MyCol/doc/figure-1.png") instead of the
    absolute container path makes it portable across Docker and local dev.
    """
    if not image_path:
        return image_path
    try:
        return Path(image_path).resolve().relative_to(
            processed_base.resolve()
        ).as_posix()
    except ValueError:
        # Path is already relative or cannot be made relative; keep as-is.
        return Path(image_path).as_posix()


@dataclass
class IngestionResult:
    chunk_count: int
    warnings: List[str]


class IngestService:
    """
    Service class for document ingestion operations.
    Designed for use in API endpoints.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._summarizer_llm = get_llm("image_summarizer")
        self.warnings: List[str] = []
        self._warnings_lock = threading.Lock()

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True if the exception signals an API rate limit (429)."""
        msg = str(exc).lower()
        return "429" in msg or "rate limit" in msg or "resource_exhausted" in msg

    def _record_warning(self, message: str) -> None:
        """Record a non-fatal ingestion warning for API visibility."""
        logger.warning(message)
        with self._warnings_lock:
            self.warnings.append(message)

    def _summarize_image(self, chunk: dict) -> dict:
        """Summarize an image chunk using LLM vision."""
        with tracer.start_as_current_span("summarize_single_image") as span:
            try:
                image_context = chunk.get("text", "")
                img_path = chunk["metadata"].get("image_path", "unknown")
                span.set_attribute("image_path", img_path)

                system_message = SystemMessage(
                    content=IMAGE_SUMMARIZER_PROMPT.format(image_context=image_context)
                )

                # Process image to base64
                img_base64 = to_base64(img_path)
                if not img_base64:
                    self._record_warning(
                        f"Could not load image {img_path}; using Docling caption/context only."
                    )
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, "Image not found")
                    )
                    return chunk

                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    }
                ]

                # Invoke LLM with automatic retry only on actual 429 / rate-limit errors.
                # Waits 3s then doubles up to 30s per attempt; gives up after 4 retries.
                human_message = HumanMessage(content=content)

                for attempt in Retrying(
                    retry=retry_if_exception(self._is_rate_limit_error),
                    wait=wait_exponential(multiplier=1, min=3, max=30),
                    stop=stop_after_attempt(4),
                    reraise=True,
                ):
                    with attempt:
                        response = self._summarizer_llm.invoke([system_message, human_message])

                summary = response.content.strip()
                chunk["text"] = f"{image_context}\n\n{summary}".strip()
                logger.info(f"Generated summary for image: {img_path}")
                return chunk
            except Exception as e:
                self._record_warning(
                    f"Image summary failed for {img_path}; using Docling caption/context only. Error: {e}"
                )
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                return chunk

    def _parse_pdf(self, file_path: Path, output_dir: Path, source: str):
        with tracer.start_as_current_span("read_pdf") as read_span:
            parse_result = parse_pdf(
                pdf_path=file_path,
                output_dir=output_dir,
                source=source,
            )
            read_span.set_attribute("page_count", parse_result.page_count)
            read_span.set_attribute("doc_type", parse_result.doc_type)
            read_span.set_attribute("parse_seconds", parse_result.parse_seconds)
            read_span.set_attribute(
                "postprocess_seconds",
                parse_result.postprocess_seconds,
            )
            return parse_result

    @staticmethod
    def _chunk_dicts_from_parse_result(
        parse_result,
        processed_base: Path,
        source: str,
        chunk_id_stem: str,
    ) -> list[dict]:
        chunks = []
        for idx, chunk in enumerate(parse_result.elements):
            image_path = _to_relative_image_path(chunk.image_path, processed_base)
            chunk_id = f"{chunk_id_stem}_{idx}"
            chunks.append(
                {
                    "text": chunk.content,
                    "metadata": {
                        "type": chunk.chunk_type,
                        "chunk_id": chunk_id,
                        "source": chunk.source or source,
                        "image_path": image_path,
                        "page_number": chunk.page_number or 0,
                        "section": chunk.section,
                    },
                }
            )
        return chunks

    @staticmethod
    def _image_chunks(chunks: list[dict]) -> list[dict]:
        return [
            chunk
            for chunk in chunks
            if chunk["metadata"].get("type") == "image"
            and chunk["metadata"].get("image_path")
        ]

    def _summarize_image_chunks(self, image_chunks: list[dict]) -> None:
        if not image_chunks:
            return

        with tracer.start_as_current_span("summarize_all_images") as sum_all_span:
            sum_all_span.set_attribute("image_count", len(image_chunks))
            logger.info(
                "Summarizing %s images concurrently (max %s workers)...",
                len(image_chunks),
                IMAGE_SUMMARY_WORKERS,
            )

            current_context = context.get_current()

            def _summarize_with_context(chunk, ctx):
                token = context.attach(ctx)
                try:
                    return self._summarize_image(chunk)
                finally:
                    context.detach(token)

            with ThreadPoolExecutor(max_workers=IMAGE_SUMMARY_WORKERS) as executor:
                futures = [
                    executor.submit(_summarize_with_context, chunk, current_context)
                    for chunk in image_chunks
                ]
                for future in as_completed(futures):
                    future.result()

    @staticmethod
    def _documents_from_chunks(chunks: list[dict]) -> list[Document]:
        return [
            Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"],
            )
            for chunk in chunks
        ]

    @staticmethod
    def _set_document_metrics(span, documents: list[Document]) -> None:
        # Single pass over documents using Counter instead of three separate iterations
        type_counts = Counter(doc.metadata.get("type") for doc in documents)
        num_images = type_counts.get("image", 0)
        num_tables = type_counts.get("table", 0)
        num_texts = len(documents) - num_images - num_tables

        span.set_attribute("final_chunks_total", len(documents))
        span.set_attribute("final_chunks_text", num_texts)
        span.set_attribute("final_chunks_image", num_images)
        span.set_attribute("final_chunks_table", num_tables)

    def preprocess_documents(self, file_name: str, collection_name: str):
        file_path = self.settings.data_raw_dir / collection_name / file_name
        with tracer.start_as_current_span("preprocess_documents") as prep_span:
            try:
                processed_base = self.settings.data_processed_dir
                source = f"{collection_name}/{file_name}" if collection_name else file_name

                document_stem = Path(file_name).stem
                chunk_id_stem = document_stem.replace(" ", "_")
                output_dir = processed_base / collection_name / document_stem

                parse_result = self._parse_pdf(file_path, output_dir, source)
                prep_span.set_attribute("page_count", parse_result.page_count)

                chunks = self._chunk_dicts_from_parse_result(
                    parse_result,
                    processed_base,
                    source,
                    chunk_id_stem,
                )
                del parse_result

                self._summarize_image_chunks(self._image_chunks(chunks))
                documents = self._documents_from_chunks(chunks)
                logger.info(f"Loaded {len(documents)} documents from {file_name}")

                self._set_document_metrics(prep_span, documents)

                return documents
            except Exception as exc:
                prep_span.record_exception(exc)
                prep_span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                logger.exception("Error preprocessing document %s", file_path)
                raise

    def add_documents(
        self,
        collection_name: str,
        chunks: List[Document],
    ):
        """
        Insert document chunks into Weaviate collection.

        Returns:
            None
        """
        with tracer.start_as_current_span("add_documents") as span:
            span.set_attribute("collection", collection_name)
            span.set_attribute("chunk_count", len(chunks))

            collection = get_weaviate_client().collections.get(collection_name)

            try:
                with collection.batch.dynamic() as batch:
                    for doc in chunks:
                        properties = {
                            "text": doc.page_content,
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                            "type": doc.metadata.get("type", ""),
                            "source": doc.metadata.get("source", ""),
                            "image_path": doc.metadata.get("image_path", ""),
                            "page_number": doc.metadata.get("page_number", 0),
                            "section": doc.metadata.get("section", ""),
                        }

                        batch.add_object(
                            properties=properties,
                            uuid=generate_uuid5(
                                f"{properties['source']}:{properties['chunk_id']}"
                            ),
                        )

                failed = len(collection.batch.failed_objects)
                span.set_attribute("failed_objects", failed)

                if failed > 0:
                    message = f"Weaviate batch insert failed for {failed} object(s)."
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, message)
                    )
                    logger.error(message)
                    raise RuntimeError(message)
                else:
                    logger.info(
                        f"Successfully ingested {len(chunks)} chunks into {collection_name}."
                    )
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    def ingest(
        self,
        file_name: str,
        collection_name: str = "",
    ) -> IngestionResult:
        """
        Main ingestion pipeline.
        Returns: IngestionResult with chunk count and non-fatal warnings.
        """
        with tracer.start_as_current_span("ingest_pipeline") as span:
            span.set_attribute("file_name", file_name)
            span.set_attribute("collection_name", collection_name)
            logger.info(f"Starting ingestion for collection: {collection_name}")
            self.warnings.clear()

            try:
                # Step 1: Preprocess documents
                logger.info(f"STEP 1: Loading documents from {file_name}")
                documents = self.preprocess_documents(file_name, collection_name)

                # Step 2: Add documents into Weaviate
                logger.info(
                    f"STEP 2: Adding {len(documents)} documents into Weaviate {collection_name}"
                )
                self.add_documents(collection_name, documents)
                return IngestionResult(
                    chunk_count=len(documents),
                    warnings=list(self.warnings),
                )

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                logger.exception("Ingestion failed")
                raise
