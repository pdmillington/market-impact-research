"""P-006 monthly data job: download, parse, build bars and quality-check new months.

P-006 (HYPOTHESIS_LEDGER.md) is a prospective, trade-weighted test of C-006 on
BTCUSDT, ETHUSDT and SOLUSDT from 2026-09 onward. Until each scheduled look, only
data-quality checks may run. **This script never computes a price return, a
signal or an edge.** It reads timestamps and counts only.

For each pending month (from 2026-09 to the latest month Binance has published):

1. download the monthly trade archive (checksum-verified) and parse it into
   canonical fills and timestamp_direction_v1 events (shared-methodology; the
   parser validates the month and refuses to write metadata on failure);
2. rebuild the prospective event-bar cache (`impact_research_v2_prospective`,
   200 and 1,000 events) through the latest processed month;
3. run data-quality checks: the parser's validation fields; coverage of the
   month; per-day event counts; the largest gap between events; and bar-cache
   continuity (bars before the new month identical to the test-period cache).

Results are appended to `alpha-search/reports/p006/data_log.jsonl`, and
`status.json` records months processed, flags, and the next look.

Usage:
    python alpha-search/scripts/p006_monthly_data.py            # all pending months
    python alpha-search/scripts/p006_monthly_data.py --dry-run  # report only
    python alpha-search/scripts/p006_monthly_data.py --month 2026-09
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
import requests


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.download import archive_url, download_archive  # noqa: E402
from crypto_market_data.layout import DataLayout  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402
from crypto_market_data.parsing import parse_month_archive  # noqa: E402


SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
FIRST_MONTH = "2026-09"
LOOKS = {1: "2027-08", 2: "2028-08"}            # last month included at each look
CACHE = "impact_research_v2_prospective"
REFERENCE_CACHE = "impact_research_v2_locked"   # verified cache through 2026-08
BAR_SIZES = (200, 1000)
ROOT = REPO / "shared-data"
OUT = REPO / "alpha-search" / "reports" / "p006"
MAX_EVENT_GAP_MINUTES = 10
LOW_DAY_FRACTION = 0.10


# ---------------------------------------------------------------- months and looks


def month_start(month: str) -> datetime:
    return datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)


def next_month(month: str) -> str:
    year, number = map(int, month.split("-"))
    return f"{year + 1}-01" if number == 12 else f"{year}-{number + 1:02d}"


def latest_complete_month(today: date) -> str:
    """The most recent month that has fully ended before `today`."""

    year, number = today.year, today.month
    return f"{year - 1}-12" if number == 1 else f"{year}-{number - 1:02d}"


def look_status(processed_through: str | None) -> dict:
    """Which look, if any, the processed data allow. No edge may be computed otherwise."""

    due = [n for n, last in LOOKS.items() if processed_through is not None and processed_through >= last]
    upcoming = [n for n in LOOKS if n not in due]
    return {"looks_allowed": due, "next_look": upcoming[0] if upcoming else None,
            "next_look_needs_data_through": LOOKS[upcoming[0]] if upcoming else None}


def published(symbol: str, month: str) -> bool:
    try:
        return requests.head(archive_url(symbol, month), timeout=60, allow_redirects=True).status_code == 200
    except requests.exceptions.RequestException:
        return False


def processed(symbol: str, month: str) -> bool:
    return (ROOT / "metadata" / symbol / f"{month}.json").exists()


# ---------------------------------------------------------------- quality checks (no prices)


def event_checks(symbol: str, month: str, report: dict) -> dict:
    """Coverage, per-day counts and gaps from event timestamps only."""

    path = DataLayout(ROOT).event_month(symbol, month)
    times = pl.read_parquet(path, columns=["timestamp_ms"])["timestamp_ms"].to_numpy()
    start = int(month_start(month).timestamp() * 1_000)
    end = int(month_start(next_month(month)).timestamp() * 1_000)
    days = (times - start) // 86_400_000
    counts = np.bincount(days, minlength=(end - start) // 86_400_000)
    gaps = np.diff(np.concatenate([[start], times, [end]]))
    flags = []
    if report.get("fill_count_difference", 0) != 0 or report.get("invalid_fill_rows", 0) != 0:
        flags.append("parser validation differences")
    if (times.min() - start) > 3_600_000 or (end - times.max()) > 3_600_000:
        flags.append("more than 1 h missing at the month start or end")
    low = np.flatnonzero(counts < LOW_DAY_FRACTION * np.median(counts))
    if len(low):
        flags.append(f"days with < {LOW_DAY_FRACTION:.0%} of median events: {[int(d) + 1 for d in low]}")
    if gaps.max() > MAX_EVENT_GAP_MINUTES * 60_000:
        flags.append(f"largest gap between events {gaps.max() / 60_000:.1f} min")
    return {"fill_rows": report.get("fill_rows"), "event_rows": int(len(times)),
            "reused_trade_ids": report.get("reused_trade_ids", 0),
            "exact_duplicate_rows_removed": report.get("exact_duplicate_rows_removed", 0),
            "events_per_day_min": int(counts.min()), "events_per_day_median": float(np.median(counts)),
            "largest_gap_minutes": float(gaps.max() / 60_000), "flags": flags}


def bar_checks(symbol: str, through: str) -> dict:
    """Prospective cache: continuity with the verified cache, ordering, coverage of new months."""

    cache = ROOT / "features" / "alpha" / f"symbol={symbol}" / CACHE
    reference = ROOT / "features" / "alpha" / f"symbol={symbol}" / REFERENCE_CACHE
    boundary = int(month_start(FIRST_MONTH).timestamp() * 1_000) - 86_400_000
    out, flags = {}, []
    for size in BAR_SIZES:
        new = pl.scan_parquet(cache / f"event_{size}" / "part-000.parquet").select("end_time_ms", "end_price", "event_count")
        before_new = new.filter(pl.col("end_time_ms") < boundary).collect()
        before_ref = (pl.scan_parquet(reference / f"event_{size}" / "part-000.parquet")
                      .select("end_time_ms", "end_price", "event_count").filter(pl.col("end_time_ms") < boundary).collect())
        identical = before_new.equals(before_ref)
        stats = new.select(pl.len().alias("bars"), pl.col("end_time_ms").max().alias("last_ms"),
                           (pl.col("end_time_ms").diff() < 0).sum().alias("backwards")).collect().row(0, named=True)
        prospective = new.filter(pl.col("end_time_ms") >= boundary + 86_400_000).select(pl.len()).collect().item()
        out[size] = {"bars": stats["bars"], "prospective_bars": prospective, "backwards": stats["backwards"],
                     "identical_before_2026_09": identical,
                     "last_bar_utc": datetime.fromtimestamp(stats["last_ms"] / 1_000, tz=timezone.utc).isoformat()}
        if not identical:
            flags.append(f"event_{size}: bars before 2026-09 differ from the verified cache")
        if stats["backwards"]:
            flags.append(f"event_{size}: {stats['backwards']} out-of-order bars")
    return {"sizes": out, "flags": flags}


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", help="process only this month (YYYY-MM)")
    parser.add_argument("--dry-run", action="store_true", help="report pending months without downloading")
    parser.add_argument("--skip-bars", action="store_true", help="parse only; do not rebuild the bar cache")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    last = latest_complete_month(datetime.now(timezone.utc).date())
    months = [args.month] if args.month else list(iter_months(FIRST_MONTH, last)) if last >= FIRST_MONTH else []
    if any(m < FIRST_MONTH for m in months):
        raise SystemExit(f"P-006 data start at {FIRST_MONTH}.")
    plan = {s: [m for m in months if not processed(s, m)] for s in SYMBOLS}
    availability = {s: {m: published(s, m) for m in plan[s]} for s in SYMBOLS}
    print(json.dumps({"latest_complete_month": last, "pending": plan, "published": availability}, indent=2))
    if args.dry_run:
        return

    run = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "months": {}, "flags": []}
    for symbol in SYMBOLS:
        for month in plan[symbol]:
            if not availability[symbol][month]:
                print(f"{symbol} {month}: archive not yet published; skipped", flush=True)
                continue
            layout = DataLayout(ROOT)
            download_archive(symbol=symbol, month=month, destination=layout.source_archive(symbol, month))
            report = asdict(parse_month_archive(root=ROOT, symbol=symbol, month=month))
            checks = event_checks(symbol, month, report)
            run["months"].setdefault(symbol, {})[month] = checks
            run["flags"] += [f"{symbol} {month}: {f}" for f in checks["flags"]]
            print(f"{symbol} {month}: {checks['event_rows']:,} events; flags {checks['flags'] or 'none'}", flush=True)

    through = {s: max([m for m in months if processed(s, m)], default=None) for s in SYMBOLS}
    common = min((m for m in through.values() if m), default=None) if all(through.values()) else None
    if common and not args.skip_bars and run["months"]:
        for symbol in SYMBOLS:
            subprocess.run([sys.executable, str(REPO / "alpha-search" / "scripts" / "build_impact_research_bars.py"),
                            "--symbol", symbol, "--end-month", common, "--event-sizes", *map(str, BAR_SIZES),
                            "--time-bars", "--overwrite", "--output-root",
                            str(ROOT / "features" / "alpha" / f"symbol={symbol}" / CACHE)], check=True)
            checks = bar_checks(symbol, common)
            run.setdefault("bars", {})[symbol] = checks
            run["flags"] += [f"{symbol} bars: {f}" for f in checks["flags"]]

    status = {"updated_at": run["run_at"], "processed_through": through, "common_through": common,
              **look_status(common), "flags_this_run": run["flags"],
              "rule": "No signal, return or edge may be computed for P-006 data before a look is allowed (HYPOTHESIS_LEDGER.md, P-006)."}
    with (OUT / "data_log.jsonl").open("a") as stream:
        stream.write(json.dumps(run, default=str) + "\n")
    (OUT / "status.json").write_text(json.dumps(status, indent=2, default=str))
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()
