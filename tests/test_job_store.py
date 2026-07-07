from types import SimpleNamespace

import pytest

from src.core.job_store import IngestionJobStore


def test_job_store_rejects_unknown_update_fields_without_db_connection():
    store = IngestionJobStore(settings=SimpleNamespace(pg_url="postgresql://unused"))

    with pytest.raises(ValueError, match="Unsupported ingestion job field"):
        store.update_job("00000000-0000-0000-0000-000000000000", unknown=True)
