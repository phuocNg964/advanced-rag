from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from pdfminer.pdfpage import PDFPage

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import weaviate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from src.models.base import get_llm
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from weaviate.util import generate_uuid5

from src.core.config import Settings, get_settings
from src.core.logger import get_logger
from src.components.parser import attach_captions, to_base64
from src.prompts.prompts import IMAGE_SUMMARIZER_PROMPT

from opentelemetry import trace, context

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class IngestService:
    """
    Service class for document ingestion operations.
    Designed for use in API endpoints.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: Optional[weaviate.WeaviateClient] = None
        self._summarizer_llm = get_llm("image_summarizer")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def close(self):
        """Close Weaviate client connection."""
        if self._client:
            self._client.close()
            self._client = None

    def _get_client(self) -> weaviate.WeaviateClient:
        """Get or create Weaviate client using config settings."""
        if self._client is None or not self._client.is_connected():
            self._client = weaviate.connect_to_local(
                host=self.settings.weaviate_host,
                port=self.settings.weaviate_http_port,
                grpc_port=self.settings.weaviate_grpc_port,
            )
        return self._client

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True if the exception signals an API rate limit (429)."""
        msg = str(exc).lower()
        return "429" in msg or "rate limit" in msg or "resource_exhausted" in msg

    def _summarize_image(self, chunk: dict) -> dict:
        """Summarize an image chunk using LLM vision."""
        with tracer.start_as_current_span("summarize_single_image") as span:
            try:
                caption = chunk["metadata"].get("caption", "")
                img_path = chunk["metadata"].get("image_path", "unknown")
                span.set_attribute("image_path", img_path)

                system_message = SystemMessage(
                    content=IMAGE_SUMMARIZER_PROMPT.format(caption=caption)
                )

                # Process image to base64
                img_base64 = to_base64(img_path)
                if not img_base64:
                    logger.warning(f"Could not load image {img_path}, skipping summary")
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

                @retry(
                    retry=retry_if_exception(self._is_rate_limit_error),
                    wait=wait_exponential(multiplier=1, min=3, max=30),
                    stop=stop_after_attempt(4),
                    reraise=True,
                )
                def _call_llm():
                    return self._summarizer_llm.invoke([system_message, human_message])

                response = _call_llm()
                summary = response.content.strip()
                chunk["text"] = f"{caption}\n\n{summary}"
                logger.info(f"Generated summary for image: {img_path}")
                return chunk
            except Exception as e:
                logger.error(f"Error generating summaries: {e}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise e

    def preprocess_documents(self, file_name: str, collection_name: str):
        with tracer.start_as_current_span("preprocess_documents") as prep_span:
            try:
                # Document file path (per-collection folder)
                file_path = (
                    self.settings.base_dir
                    / "data"
                    / "raw"
                    / collection_name
                    / file_name
                )

                # Folder store processed images and tables (per-collection folder)
                processed_folder_path = (
                    self.settings.base_dir
                    / "data"
                    / "processed"
                    / collection_name
                    / Path(file_name).stem
                )

                # OOM Protection: Count pages quickly using pdfminer
                try:
                    with open(file_path, "rb") as f:
                        # check_extractable=False ignores DRM restrictions for just counting pages
                        page_count = sum(
                            1 for _ in PDFPage.get_pages(f, check_extractable=False)
                        )
                except Exception as e:
                    logger.warning(
                        f"Could not determine page count for {file_name}: {e}"
                    )
                    page_count = 0

                prep_span.set_attribute("page_count", page_count)

                # Fallback to fast mode (skips heavy Layout AI, drops image/table parsing but saves RAM)
                strategy = "hi_res" if page_count <= 100 else "fast"
                if strategy == "fast":
                    logger.warning(
                        f"[OOM Protection] PDF {file_name} has {page_count} pages (>150). Falling back to 'fast' strategy. Images/Tables will NOT be extracted."
                    )

                with tracer.start_as_current_span("read_pdf") as read_span:
                    # Extract elements from PDF
                    elements = partition_pdf(
                        filename=file_path,
                        strategy=strategy,
                        # hi_res_model_name="yolox_quantized",   # Faster model -> less accuracy extract elements
                        pdf_image_dpi=100,  # Lower resolution (before 200)
                        ocr_mode="individual_blocks",  # Skip full-page OCR (not scanned PDF)
                        extract_image_block_types=["Image", "Table"],
                        extract_image_block_to_payload=False,
                        extract_image_block_output_dir=processed_folder_path,
                    )
                source = f"{collection_name}/{file_name}"
                # Filter insignificant elements and small images (< 10KB) in one pass
                MIN_IMAGE_SIZE = 10 * 1024
                significant_elements = []
                for ele in elements:
                    d = ele.to_dict()
                    etype = d.get("type", "")
                    if etype in ["UncategorizedText", "Header"]:
                        continue
                    # Remove insignificant images and tables by file size (before text check)
                    if etype in ["Image", "Table"] and ele.metadata.image_path:
                        if (
                            Path(ele.metadata.image_path).stat().st_size
                            < MIN_IMAGE_SIZE
                        ):
                            Path(ele.metadata.image_path).unlink(missing_ok=True)
                            continue
                    if etype not in ["Image", "Table"] and len(d.get("text", "")) <= 2:
                        continue
                    significant_elements.append(ele)

                # attach captions (modifies Image/Table text in-place, removes caption elements)
                significant_elements, message = attach_captions(significant_elements)
                logger.info(message)
                # Split into text vs multimodal (also summarize images chunks)
                text_elements = []
                image_chunks = []
                table_chunks = []

                for ele in significant_elements:
                    ele_dict = ele.to_dict()  # single .to_dict() call per element
                    doc_type = ele_dict.get("type", "")
                    ele_dict["metadata"]["source"] = source

                    if doc_type == "Image":
                        image_chunks.append(ele_dict)
                    elif doc_type == "Table":
                        table_chunks.append(ele_dict)
                    else:
                        text_elements.append(ele)

                if image_chunks:
                    with tracer.start_as_current_span(
                        "summarize_all_images"
                    ) as sum_all_span:
                        sum_all_span.set_attribute("image_count", len(image_chunks))
                        logger.info(
                            f"Summarizing {len(image_chunks)} images concurrently (max 5 workers)..."
                        )

                        with ThreadPoolExecutor(max_workers=5) as executor:
                            current_context = context.get_current()

                            def _summarize_with_context(chunk, ctx):
                                token = context.attach(ctx)
                                try:
                                    return self._summarize_image(chunk)
                                finally:
                                    context.detach(token)

                            futures = [
                                executor.submit(_summarize_with_context, chunk, current_context)
                                for chunk in image_chunks
                            ]
                            for future in as_completed(futures):
                                future.result()  # re-raises any exception from _summarize_image

                # Build multimodal documents from images and tables
                multimodal_documents = []
                for ele_dict in image_chunks + table_chunks:
                    document = Document(
                        page_content=ele_dict.get("text", ""),
                        metadata={
                            "type": ele_dict.get("type", ""),
                            "id": ele_dict.get("element_id", ""),
                            "caption": ele_dict["metadata"].get("caption", ""),
                            "source": ele_dict["metadata"].get("source", ""),
                            "image_path": ele_dict["metadata"].get("image_path", ""),
                            "page_number": ele_dict["metadata"].get("page_number", 0),
                        },
                    )
                    multimodal_documents.append(document)
                # Chunking text by title
                text_chunks = chunk_by_title(
                    text_elements,
                    max_characters=10000,
                    combine_text_under_n_chars=500,
                )
                # Turn elements to Documents
                text_documents = []
                for text in text_chunks:

                    text = text.to_dict()
                    text_documents.append(
                        Document(
                            page_content=text.get("text", ""),
                            metadata={
                                "type": text.get("type", ""),
                                "caption": text["metadata"].get("caption", ""),
                                "id": text.get("element_id", ""),
                                "source": source,
                                "image_path": text["metadata"].get("image_path", ""),
                                "page_number": text["metadata"].get("page_number", 0),
                            },
                        )
                    )

                # Split further with RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,  # Target size for each chunk
                    chunk_overlap=150,
                    separators=["\n\n", "\n"],
                )
                text_documents = text_splitter.split_documents(text_documents)

                documents = text_documents + multimodal_documents
                logger.info(f"Loaded {len(documents)} documents from {file_name}")

                num_texts = sum(
                    1
                    for d in documents
                    if d.metadata.get("type") not in ["Image", "Table"]
                )
                num_images = sum(
                    1 for d in documents if d.metadata.get("type") == "Image"
                )
                num_tables = sum(
                    1 for d in documents if d.metadata.get("type") == "Table"
                )

                prep_span.set_attribute("final_chunks_total", len(documents))
                prep_span.set_attribute("final_chunks_text", num_texts)
                prep_span.set_attribute("final_chunks_image", num_images)
                prep_span.set_attribute("final_chunks_table", num_tables)

                return documents
            except Exception as e:
                prep_span.record_exception(e)
                prep_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                logger.error(f"Error executing load_documents for {file_path}: {e}")
                raise e

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

            self._get_client()

            collection = self._client.collections.get(collection_name)

            try:
                with collection.batch.dynamic() as batch:
                    for doc in chunks:
                        properties = {
                            "text": doc.page_content,
                            "chunk_id": doc.metadata.get("id", ""),
                            "type": doc.metadata.get("type", ""),
                            "caption": doc.metadata.get("caption", ""),
                            "source": doc.metadata.get("source", ""),
                            "image_path": doc.metadata.get("image_path", ""),
                            "page_number": doc.metadata.get("page_number", 0),
                        }

                        batch.add_object(
                            properties=properties, uuid=generate_uuid5(doc.page_content)
                        )

                failed = len(collection.batch.failed_objects)
                span.set_attribute("failed_objects", failed)

                if failed > 0:
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR, f"{failed} failed objects")
                    )
                    logger.warning(f"Ingestion finished with {failed} failed objects.")
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
    ):
        """
        Main ingestion pipeline.
        Returns: None
        """
        with tracer.start_as_current_span("ingest_pipeline") as span:
            span.set_attribute("file_name", file_name)
            span.set_attribute("collection_name", collection_name)
            logger.info(f"Starting ingestion for collection: {collection_name}")

            try:
                # Step 1: Preprocess documents
                logger.info(f"STEP 1: Loading documents from {file_name}")
                documents = self.preprocess_documents(str(file_name), collection_name)

                # Step 2: Add documents into Weaviate
                logger.info(
                    f"STEP 2: Adding {len(documents)} documents into Weaviate {collection_name}"
                )
                self.add_documents(collection_name, documents)

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                logger.exception("Ingestion failed")
                raise
