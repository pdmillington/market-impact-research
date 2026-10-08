"""Repair BTCUSDT months whose monthly trade archive is missing the end of a day.

Found by crypto-impact-study/scripts/trading_gaps_audit.py (2026-10-08): in 12
places the monthly archives stop early on a day and resume at 00:00 UTC, while
exchange trade IDs jump across the gap (trades exist that the file lacks). For 8
of them Binance's daily trade archive is complete
(crypto-impact-study/reports/redo/data_description/daily_archive_gap_check.csv).
The other 4 (2022-10-25, 2023-03-14, 2023-11-21 x2) are missing from the daily
archive too; aggTrades covers them, but it merges fills up to 100 ms apart and
so cannot rebuild timestamp_direction_v1 events (58-65% of the true event count
in a test), so those gaps are left open and flagged downstream.

Steps, per affected month:
1. download the verified daily archive(s) into the source layer;
2. move the month's current canonical fills, events and report to
   shared-data/superseded/<tag>/ (kept, not deleted);
3. re-parse the month; parse_month_archive merges the daily archives (fills whose
   trade ID is absent from the monthly file; any disagreement aborts).

    shared-methodology/.venv/bin/python shared-methodology/scripts/repair_from_daily_archives.py
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.download import download_daily_supplements  # noqa: E402
from crypto_market_data.layout import DataLayout  # noqa: E402
from crypto_market_data.parsing import parse_month_archive  # noqa: E402

SYMBOL = "BTCUSDT"
DAYS = ["2021-02-20", "2021-02-22", "2022-02-14", "2022-09-21", "2022-11-02",
        "2023-02-01", "2023-03-13", "2023-03-15"]
TAG = "2026-10-08_before_daily_supplement"


def main() -> None:
    root = REPO / "shared-data"
    layout = DataLayout(root)
    download_daily_supplements(root=root, symbol=SYMBOL, days=DAYS)
    months = sorted({day[:7] for day in DAYS})
    backup = root / "superseded" / TAG
    for month in months:
        for path in (layout.canonical_month(SYMBOL, month), layout.event_month(SYMBOL, month),
                     layout.report_month(SYMBOL, month)):
            target = backup / path.relative_to(root)
            if path.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(path, target)
        before = json.loads((backup / layout.report_month(SYMBOL, month).relative_to(root)).read_text())
        report = parse_month_archive(root=root, symbol=SYMBOL, month=month, overwrite=True)
        print(f"{month}: fills {before['fill_rows']:,} -> {report.fill_rows:,} "
              f"(+{report.supplement_rows_added:,} from {', '.join(report.supplement_archives)}); "
              f"events {before['event_rows']:,} -> {report.event_rows:,}", flush=True)


if __name__ == "__main__":
    main()
