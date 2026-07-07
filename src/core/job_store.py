from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.core.config import Settings, get_settings


class IngestionJobStore:
    """Small Postgres-backed store for upload ingestion job status."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id UUID PRIMARY KEY,
                    status TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    message TEXT,
                    trace_id TEXT,
                    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    completed_at TIMESTAMPTZ
                )
                """
            )

    def create_job(
        self,
        *,
        job_id: str,
        collection_name: str,
        document_name: str,
        status: str = "queued",
        message: str | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, status, collection_name, document_name, message, warnings
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    job_id,
                    status,
                    collection_name,
                    document_name,
                    message,
                    Jsonb(warnings or []),
                ),
            )

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed_fields = {"status", "message", "trace_id", "warnings"}
        updates = []
        params = []

        for field, value in fields.items():
            if field not in allowed_fields:
                raise ValueError(f"Unsupported ingestion job field: {field}")
            updates.append(f"{field} = %s")
            params.append(Jsonb(value) if field == "warnings" else value)

        if not updates:
            return

        updates.append("updated_at = now()")
        if fields.get("status") in {"completed", "failed"}:
            updates.append("completed_at = now()")

        params.append(job_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE ingestion_jobs SET {', '.join(updates)} WHERE job_id = %s",
                params,
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    job_id::text AS job_id,
                    status,
                    collection_name,
                    document_name,
                    message,
                    trace_id,
                    warnings
                FROM ingestion_jobs
                WHERE job_id = %s
                """,
                (job_id,),
            ).fetchone()
        return dict(row) if row else None

    def _connect(self):
        return psycopg.connect(self.settings.pg_url, row_factory=dict_row)


def init_ingestion_job_store(settings: Settings | None = None) -> IngestionJobStore:
    store = IngestionJobStore(settings)
    store.init_schema()
    return store
