from __future__ import annotations

from pathlib import Path

import pytest
import requests

from crypto_market_data import download


class FakeResponse:
    def __init__(
        self,
        *,
        content: bytes = b"",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size: int):
        if self.error is not None:
            raise self.error
        yield self.content


def test_download_once_resumes_partial_file(monkeypatch, tmp_path: Path) -> None:
    temporary = tmp_path / "archive.zip.part"
    temporary.write_bytes(b"first-")
    seen_headers = []

    def fake_get(url, *, headers, stream, timeout):
        seen_headers.append(headers)
        return FakeResponse(
            content=b"second",
            status_code=206,
            headers={"Content-Range": "bytes 6-11/12"},
        )

    monkeypatch.setattr(download.requests, "get", fake_get)

    download._download_once(
        url="https://example.test/archive.zip",
        temporary=temporary,
        timeout_seconds=5,
    )

    assert seen_headers == [{"Range": "bytes=6-"}]
    assert temporary.read_bytes() == b"first-second"


def test_download_once_restarts_when_server_ignores_range(
    monkeypatch, tmp_path: Path
) -> None:
    temporary = tmp_path / "archive.zip.part"
    temporary.write_bytes(b"partial")

    monkeypatch.setattr(
        download.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(content=b"complete", status_code=200),
    )

    download._download_once(
        url="https://example.test/archive.zip",
        temporary=temporary,
        timeout_seconds=5,
    )

    assert temporary.read_bytes() == b"complete"


def test_download_retries_and_preserves_partial(monkeypatch, tmp_path: Path) -> None:
    temporary = tmp_path / "archive.zip.part"
    attempts = []

    def fake_once(*, url, temporary, timeout_seconds):
        attempts.append(temporary.stat().st_size if temporary.exists() else 0)
        if len(attempts) == 1:
            temporary.write_bytes(b"partial")
            raise requests.ConnectionError("interrupted")
        temporary.write_bytes(temporary.read_bytes() + b"-complete")

    monkeypatch.setattr(download, "_download_once", fake_once)
    monkeypatch.setattr(download.time, "sleep", lambda delay: None)

    download._download_with_retries(
        url="https://example.test/archive.zip",
        temporary=temporary,
        timeout_seconds=5,
        max_attempts=2,
    )

    assert attempts == [0, 7]
    assert temporary.read_bytes() == b"partial-complete"


def test_download_rejects_zero_attempts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        download._download_with_retries(
            url="https://example.test/archive.zip",
            temporary=tmp_path / "archive.zip.part",
            timeout_seconds=5,
            max_attempts=0,
        )
