import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("The smoke test requires requests. Install dependencies first.", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a running RAG deployment.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--collection", default="SmokeTest")
    parser.add_argument("--pdf", type=Path, help="Optional PDF to upload and query.")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    session = requests.Session()

    _check_json(session, "GET", f"{base_url}/health")
    _check_json(session, "GET", f"{base_url}/ready")

    if args.pdf is None:
        print("Smoke test passed: health and readiness checks succeeded.")
        return 0

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    _request(
        session,
        "POST",
        f"{base_url}/collections",
        json={"name": args.collection},
        allowed_status={200, 409},
    )

    with args.pdf.open("rb") as pdf_file:
        response = _check_json(
            session,
            "POST",
            f"{base_url}/collections/{args.collection}/documents",
            files={"file": (args.pdf.name, pdf_file, "application/pdf")},
        )

    job_id = response["job_id"]
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        job = _check_json(session, "GET", f"{base_url}/collections/jobs/{job_id}")
        status = job["status"]
        if status == "completed":
            break
        if status == "failed":
            print(json.dumps(job, indent=2), file=sys.stderr)
            return 1
        time.sleep(3)
    else:
        print(f"Ingestion job timed out after {args.timeout}s: {job_id}", file=sys.stderr)
        return 1

    chat = _check_json(
        session,
        "POST",
        f"{base_url}/collections/{args.collection}/chat",
        json={"message": "Give a one sentence summary of this document."},
    )
    if not chat.get("response"):
        print("Chat response was empty.", file=sys.stderr)
        return 1

    print("Smoke test passed: health, readiness, upload, ingestion, and chat succeeded.")
    return 0


def _check_json(session, method: str, url: str, **kwargs):
    response = _request(session, method, url, **kwargs)
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"{method} {url} did not return JSON") from exc


def _request(session, method: str, url: str, allowed_status={200}, **kwargs):
    response = session.request(method, url, timeout=30, **kwargs)
    if response.status_code not in allowed_status:
        raise RuntimeError(
            f"{method} {url} failed with HTTP {response.status_code}: {response.text}"
        )
    return response


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
