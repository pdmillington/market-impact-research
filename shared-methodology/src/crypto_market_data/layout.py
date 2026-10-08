"""Filesystem layout for shared market data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataLayout:
    """Resolve all paths belonging to one shared-data root."""

    root: Path

    def temporary_root(self) -> Path:
        """Workspace for large transient files on the data volume itself."""

        return self.root / ".tmp" / "crypto-market-data"

    def source_archive(self, symbol: str, month: str) -> Path:
        return (
            self.root
            / "source"
            / "binance"
            / "futures"
            / "um"
            / "trades"
            / symbol
            / f"{symbol}-trades-{month}.zip"
        )

    def source_daily_archive(self, symbol: str, day: str) -> Path:
        """Daily trade archive used to fill gaps in the monthly archive."""

        return (
            self.root
            / "source"
            / "binance"
            / "futures"
            / "um"
            / "trades_daily"
            / symbol
            / f"{symbol}-trades-{day}.zip"
        )

    def daily_supplements(self, symbol: str, month: str) -> list[Path]:
        """Daily archives stored for one month, in date order."""

        folder = self.source_daily_archive(symbol, f"{month}-01").parent
        return sorted(folder.glob(f"{symbol}-trades-{month}-*.zip"))

    def source_metrics_day(self, symbol: str, day: str) -> Path:
        return (
            self.root
            / "source"
            / "binance"
            / "futures"
            / "um"
            / "metrics"
            / symbol
            / f"{symbol}-metrics-{day}.zip"
        )

    def metrics_table(self, symbol: str) -> Path:
        return (
            self.root
            / "canonical"
            / "metrics"
            / f"symbol={symbol}"
            / "metrics_5m.parquet"
        )

    def canonical_month(self, symbol: str, month: str) -> Path:
        year, month_number = month.split("-")
        return (
            self.root
            / "canonical"
            / "fills"
            / f"symbol={symbol}"
            / f"year={year}"
            / f"month={month_number}"
            / "part-000.parquet"
        )

    def event_month(self, symbol: str, month: str) -> Path:
        year, month_number = month.split("-")
        return (
            self.root
            / "events"
            / "timestamp_direction_v1"
            / f"symbol={symbol}"
            / f"year={year}"
            / f"month={month_number}"
            / "part-000.parquet"
        )

    def report_month(self, symbol: str, month: str) -> Path:
        return self.root / "metadata" / symbol / f"{month}.json"
