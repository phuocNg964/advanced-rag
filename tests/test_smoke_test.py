from types import SimpleNamespace

from scripts import smoke_test


class FakeResponse:
    status_code = 200
    text = "{}"

    def __init__(self, payload=None):
        self.payload = payload or {}

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, timeout=30, **kwargs):
        self.calls.append((method, url, timeout, kwargs))
        return FakeResponse({})


def test_smoke_test_health_ready_only(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(smoke_test.requests, "Session", lambda: session)
    monkeypatch.setattr(
        smoke_test.sys,
        "argv",
        ["smoke_test.py", "--base-url", "http://example.test"],
    )

    assert smoke_test.main() == 0
    assert [call[:2] for call in session.calls] == [
        ("GET", "http://example.test/health"),
        ("GET", "http://example.test/ready"),
    ]


def test_smoke_test_missing_pdf_returns_usage_error(monkeypatch, tmp_path):
    session = FakeSession()
    missing_pdf = tmp_path / "missing.pdf"
    monkeypatch.setattr(smoke_test.requests, "Session", lambda: session)
    monkeypatch.setattr(
        smoke_test.sys,
        "argv",
        ["smoke_test.py", "--pdf", str(missing_pdf)],
    )

    assert smoke_test.main() == 2


def test_request_raises_on_unexpected_status():
    response = SimpleNamespace(status_code=503, text="not ready")
    session = SimpleNamespace(request=lambda *args, **kwargs: response)

    try:
        smoke_test._request(session, "GET", "http://example.test/ready")
    except RuntimeError as exc:
        assert "HTTP 503" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")
