"""Download Binance public aggregate trade data."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests


LOGGER = logging.getLogger(__name__)
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"


def date_range(start_date: str, end_date: str) -> list[date]:
    """Return calendar dates in the inclusive range ``[start_date, end_date]``."""

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if dates.empty:
        raise ValueError("Date range is empty.")
    return [item.date() for item in dates]


def agg_trade_url(symbol: str, trade_date: date) -> str:
    """Build the Binance data URL for one symbol/date aggregate trade file."""

    date_text = trade_date.isoformat()
    filename = f"{symbol}-aggTrades-{date_text}.zip"
    return f"{BASE_URL}/{symbol}/{filename}"


def download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    """Download ``url`` to ``destination`` unless it already exists."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Using existing raw file: %s", destination)
        return destination

    LOGGER.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def download_agg_trades(symbol: str, start_date: str, end_date: str, raw_dir: Path) -> list[Path]:
    """Download daily Binance aggregate trade zip files for a date range."""

    symbol = symbol.upper()
    symbol_dir = raw_dir / symbol
    paths: list[Path] = []
    for trade_date in date_range(start_date, end_date):
        filename = f"{symbol}-aggTrades-{trade_date.isoformat()}.zip"
        paths.append(download_file(agg_trade_url(symbol, trade_date), symbol_dir / filename))
    return paths
