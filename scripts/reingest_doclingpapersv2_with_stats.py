from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "DoclingPapers"
DEFAULT_TARGET_COLLECTION = "DoclingPapersv2"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
READY_STATUSES = {"queued", "running", "processing"}
PHOENIX_BASE_URL = "http://127.0.0.1:6006"
PHOENIX_PROJECT = "agentic-rag"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def request_json(
    method: str,
    url: str,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: int = 30,
):
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    return json.loads(payload.decode("utf-8")) if payload else {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def count_pages(pdf_path: Path) -> int | None:
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path), strict=False).pages)
    except Exception:
        return None


def create_collection(base_url: str, collection: str) -> None:
    body = json.dumps({"name": collection}).encode("utf-8")
    try:
        response = request_json(
            "POST",
            f"{base_url}/collections",
            body=body,
            content_type="application/json",
        )
        print(f"Created collection {collection}: {response.get('message', '')}")
    except RuntimeError as exc:
        if "already exists" in str(exc).lower():
            print(f"Collection {collection} already exists.")
            return
        raise


def list_target_documents(base_url: str, collection: str) -> list[dict]:
    response = request_json("GET", f"{base_url}/collections/{collection}/documents")
    return response.get("documents", [])


def delete_target_documents(base_url: str, collection: str) -> list[str]:
    deleted = []
    for document in list_target_documents(base_url, collection):
        filename = document.get("filename")
        if not filename:
            continue
        encoded_name = quote(filename, safe="")
        request_json(
            "DELETE",
            f"{base_url}/collections/{collection}/documents/{encoded_name}",
            timeout=60,
        )
        deleted.append(filename)
        print(f"Deleted {collection}/{filename}")
    return deleted


def multipart_body(field_name: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----rag-reingest-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/pdf"
    file_bytes = file_path.read_bytes()
    parts = [
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="{file_path.name}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def upload_document(base_url: str, collection: str, file_path: Path) -> str:
    body, content_type = multipart_body("file", file_path)
    response = request_json(
        "POST",
        f"{base_url}/collections/{collection}/documents",
        body=body,
        content_type=content_type,
        timeout=120,
    )
    return response["job_id"]


def wait_for_job(base_url: str, job_id: str, poll_seconds: int) -> dict:
    while True:
        job = request_json("GET", f"{base_url}/collections/jobs/{job_id}")
        status = job["status"]
        message = job.get("message") or ""
        print(f"  {job_id} {status}: {message}")
        if status not in READY_STATUSES:
            return job
        time.sleep(poll_seconds)


def get_chunk_stats(collection: str, source: str) -> dict:
    try:
        import weaviate
        from weaviate.classes.query import Filter
    except Exception as exc:
        return {"error": f"weaviate import failed: {exc}"}

    client = None
    try:
        client = weaviate.connect_to_local(
            host="127.0.0.1",
            port=8080,
            grpc_port=50051,
        )
        weaviate_collection = client.collections.get(collection)
        result = weaviate_collection.query.fetch_objects(
            limit=10000,
            filters=Filter.by_property("source").equal(source),
        )
        counts = Counter(obj.properties.get("type", "") for obj in result.objects)
        return {
            "chunk_count": len(result.objects),
            "text_chunks": counts.get("text", 0),
            "table_chunks": counts.get("table", 0),
            "image_chunks": counts.get("image", 0),
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        if client is not None:
            client.close()


def get_trace_span_stats(
    trace_id: str | None,
    start_time: datetime,
    end_time: datetime,
) -> dict:
    if not trace_id:
        return {"error": "missing trace_id"}

    try:
        from phoenix.client import Client as PhoenixClient
    except Exception as exc:
        return {"error": f"phoenix import failed: {exc}"}

    try:
        client = PhoenixClient(base_url=PHOENIX_BASE_URL)
        df = client.spans.get_spans_dataframe(
            start_time=start_time - timedelta(minutes=2),
            end_time=end_time + timedelta(minutes=2),
            limit=1000,
            project_identifier=PHOENIX_PROJECT,
            timeout=10,
        )
    except Exception as exc:
        return {"error": str(exc)}

    if df.empty or "context.trace_id" not in df.columns:
        return {"error": "no spans returned"}

    trace_df = df[df["context.trace_id"] == trace_id].copy()
    if trace_df.empty:
        return {"error": f"no spans found for trace_id {trace_id}"}

    trace_df["duration_seconds"] = (
        trace_df["end_time"] - trace_df["start_time"]
    ).dt.total_seconds()

    phase_names = {
        "ingest_pipeline",
        "preprocess_documents",
        "read_pdf",
        "summarize_all_images",
        "add_documents",
    }
    phases = {}
    for name in phase_names:
        rows = trace_df[trace_df["name"] == name]
        if not rows.empty:
            phases[name] = round(float(rows["duration_seconds"].max()), 3)

    image_rows = trace_df[trace_df["name"] == "summarize_single_image"]
    if not image_rows.empty:
        phases["summarize_single_image_count"] = int(len(image_rows))
        phases["summarize_single_image_total_seconds"] = round(
            float(image_rows["duration_seconds"].sum()), 3
        )
        phases["summarize_single_image_max_seconds"] = round(
            float(image_rows["duration_seconds"].max()), 3
        )

    return phases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reingest DoclingPapers into DoclingPapersv2 and save latency stats."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target", default=DEFAULT_TARGET_COLLECTION)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Upload over existing documents instead of deleting them first.",
    )
    args = parser.parse_args()

    pdfs = sorted(args.source_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {args.source_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.results_dir / f"reingest_{args.target}_{timestamp}.json"
    result_payload = {
        "metadata": {
            "target_collection": args.target,
            "source_dir": str(args.source_dir),
            "base_url": args.base_url,
            "timestamp": timestamp,
            "poll_seconds": args.poll_seconds,
            "keep_existing": args.keep_existing,
        },
        "deleted_documents": [],
        "documents": [],
        "summary": {},
    }

    request_json("GET", f"{args.base_url}/ready")
    create_collection(args.base_url, args.target)

    if not args.keep_existing:
        result_payload["deleted_documents"] = delete_target_documents(
            args.base_url,
            args.target,
        )
        write_json(output_path, result_payload)

    failures = []
    total_started = time.perf_counter()

    for pdf in pdfs:
        print(f"\nUploading {pdf.name}")
        job_started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        job_id = upload_document(args.base_url, args.target, pdf)
        print(f"Queued {pdf.name}: {job_id}")
        job = wait_for_job(args.base_url, job_id, args.poll_seconds)
        elapsed = time.perf_counter() - started
        job_finished_at = datetime.now(timezone.utc)

        source = f"{args.target}/{pdf.name}"
        chunk_stats = get_chunk_stats(args.target, source)
        span_stats = get_trace_span_stats(
            trace_id=job.get("trace_id"),
            start_time=job_started_at,
            end_time=job_finished_at,
        )
        document_result = {
            "filename": pdf.name,
            "source": source,
            "file_size_bytes": pdf.stat().st_size,
            "page_count": count_pages(pdf),
            "job_id": job_id,
            "status": job["status"],
            "message": job.get("message"),
            "trace_id": job.get("trace_id"),
            "warnings": job.get("warnings", []),
            "latency_seconds": round(elapsed, 3),
            "chunk_stats": chunk_stats,
            "span_stats": span_stats,
        }
        result_payload["documents"].append(document_result)
        write_json(output_path, result_payload)

        if job["status"] != "completed":
            failures.append(document_result)

    total_elapsed = time.perf_counter() - total_started
    completed = [d for d in result_payload["documents"] if d["status"] == "completed"]
    result_payload["summary"] = {
        "document_count": len(result_payload["documents"]),
        "completed_count": len(completed),
        "failed_count": len(failures),
        "total_latency_seconds": round(total_elapsed, 3),
        "total_pages": sum(d["page_count"] or 0 for d in result_payload["documents"]),
        "total_chunks": sum(
            d["chunk_stats"].get("chunk_count", 0)
            for d in result_payload["documents"]
            if isinstance(d.get("chunk_stats"), dict)
        ),
        "latency_seconds_per_page": round(
            total_elapsed
            / max(sum(d["page_count"] or 0 for d in result_payload["documents"]), 1),
            3,
        ),
        "add_documents_total_seconds": round(
            sum(
                d.get("span_stats", {}).get("add_documents", 0)
                for d in result_payload["documents"]
                if isinstance(d.get("span_stats"), dict)
            ),
            3,
        ),
    }
    write_json(output_path, result_payload)

    print(f"\nSaved results: {output_path}")
    if failures:
        print("Reingest finished with failures.")
        return 1

    print(f"Reingest completed: {len(completed)} PDFs into {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
