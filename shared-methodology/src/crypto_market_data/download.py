"""Download immutable Binance monthly futures trade archives."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

from .layout import DataLayout
from .months import iter_months


BASE_URL = "https://data.binance.vision/data/futures/um/monthly/trades"
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_MAX_ATTEMPTS = 8
MAX_RETRY_DELAY_SECONDS = 60


def archive_url(symbol: str, month: str) -> str:
    """Return the public Binance URL for one monthly USD-M trade archive."""

    name = f"{symbol}-trades-{month}.zip"
    return f"{BASE_URL}/{symbol}/{name}"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a file SHA-256 without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_once(*, url: str, temporary: Path, timeout_seconds: int) -> None:
    """Download or resume one archive transfer into its temporary file."""

    existing_bytes = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes else {}

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=timeout_seconds,
    ) as response:
        if response.status_code == 416 and existing_bytes:
            # The partial file may already contain the complete response. The
            # checksum below is the authority on whether it is usable.
            return
        response.raise_for_status()

        if existing_bytes and response.status_code == 206:
            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {existing_bytes}-"
            if not content_range.startswith(expected_prefix):
                raise requests.exceptions.RequestException(
                    f"Unexpected Content-Range while resuming: {content_range!r}"
                )
            mode = "ab"
        else:
            # A server is allowed to ignore Range and return the whole file.
            # In that case start clean rather than appending duplicate bytes.
            mode = "wb"

        with temporary.open(mode) as output:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    output.write(chunk)


def _download_with_retries(
    *,
    url: str,
    temporary: Path,
    timeout_seconds: int,
    max_attempts: int,
) -> None:
    """Resume transiently interrupted transfers with bounded backoff."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            _download_once(
                url=url,
                temporary=temporary,
                timeout_seconds=timeout_seconds,
            )
            return
        except requests.exceptions.RequestException:
            if attempt == max_attempts:
                raise
            delay = min(2 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS)
            print(
                f"Download interrupted; retaining {temporary.name} and "
                f"retrying in {delay}s (attempt {attempt + 1}/{max_attempts}).",
                flush=True,
            )
            time.sleep(delay)


def _checksum_with_retries(
    *, url: str, timeout_seconds: int, max_attempts: int
) -> str:
    """Fetch checksum text without letting a transient request stop the run."""

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, timeout=timeout_seconds) as response:
                response.raise_for_status()
                return response.text
        except requests.exceptions.RequestException:
            if attempt == max_attempts:
                raise
            delay = min(2 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS)
            print(
                f"Checksum request interrupted; retrying in {delay}s "
                f"(attempt {attempt + 1}/{max_attempts}).",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("unreachable")


def download_archive(
    *,
    symbol: str,
    month: str,
    destination: Path,
    overwrite: bool = False,
    timeout_seconds: int = 120,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Path:
    """Download one archive resumably and verify Binance's checksum."""

    destination = Path(destination)
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = archive_url(symbol, month)

    _download_with_retries(
        url=url,
        temporary=temporary,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )

    checksum_text = _checksum_with_retries(
        url=f"{url}.CHECKSUM",
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
    )
    expected = checksum_text.split()[0].lower()
    actual = sha256_file(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {destination.name}: "
            f"expected {expected}, received {actual}."
        )

    temporary.replace(destination)
    return destination


def download_range(
    *,
    root: Path,
    symbol: str,
    start_month: str,
    end_month: str,
    overwrite: bool = False,
) -> list[Path]:
    """Download an inclusive month range into the shared source layer."""

    layout = DataLayout(Path(root))
    outputs = []
    for month in iter_months(start_month, end_month):
        destination = layout.source_archive(symbol, month)
        outputs.append(
            download_archive(
                symbol=symbol,
                month=month,
                destination=destination,
                overwrite=overwrite,
            )
        )
    return outputs
