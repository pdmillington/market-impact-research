"""Command-line interface for shared market-data ingestion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .book_ticker import download_and_parse_month
from .download import download_archive, download_range
from .layout import DataLayout
from .metrics import download_metrics_range, parse_metrics
from .months import iter_months
from .series import DATASETS, backfill_series, download_series_range, parse_series
from .parsing import parse_month_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crypto-market-data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("download", "parse", "download-and-parse"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, required=True)
        child.add_argument("--symbol", default="BTCUSDT")
        child.add_argument("--start-month", required=True)
        child.add_argument("--end-month", required=True)
        child.add_argument("--overwrite", action="store_true")

    metrics = subparsers.add_parser("download-and-parse-metrics")
    metrics.add_argument("--root", type=Path, required=True)
    metrics.add_argument("--symbol", default="BTCUSDT")
    metrics.add_argument("--start-day", required=True)
    metrics.add_argument("--end-day", required=True)
    metrics.add_argument("--overwrite", action="store_true")

    series = subparsers.add_parser("download-and-parse-series")
    series.add_argument("--root", type=Path, required=True)
    series.add_argument("--symbol", default="BTCUSDT")
    series.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    series.add_argument("--start-month", required=True)
    series.add_argument("--end-month", required=True)
    series.add_argument("--overwrite", action="store_true")

    book = subparsers.add_parser("download-and-parse-book-ticker")
    book.add_argument("--root", type=Path, required=True)
    book.add_argument("--symbol", default="BTCUSDT")
    book.add_argument("--start-month", required=True)
    book.add_argument("--end-month", required=True)
    book.add_argument("--overwrite", action="store_true")

    backfill = subparsers.add_parser("backfill-series")
    backfill.add_argument("--root", type=Path, required=True)
    backfill.add_argument("--symbol", default="BTCUSDT")
    backfill.add_argument("--dataset", choices=sorted(DATASETS), required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    status.add_argument("--symbol", default="BTCUSDT")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download":
        paths = download_range(
            root=args.root,
            symbol=args.symbol,
            start_month=args.start_month,
            end_month=args.end_month,
            overwrite=args.overwrite,
        )
        for path in paths:
            print(path)

    if args.command == "download-and-parse":
        layout = DataLayout(args.root)
        for month in iter_months(args.start_month, args.end_month):
            source = download_archive(
                symbol=args.symbol,
                month=month,
                destination=layout.source_archive(args.symbol, month),
                overwrite=args.overwrite,
            )
            print(source)
            report = parse_month_archive(
                root=args.root,
                symbol=args.symbol,
                month=month,
                overwrite=args.overwrite,
            )
            print(json.dumps(asdict(report), indent=2))

    if args.command == "parse":
        for month in iter_months(args.start_month, args.end_month):
            report = parse_month_archive(
                root=args.root,
                symbol=args.symbol,
                month=month,
                overwrite=args.overwrite,
            )
            print(json.dumps(asdict(report), indent=2))

    if args.command == "download-and-parse-metrics":
        paths = download_metrics_range(
            root=args.root,
            symbol=args.symbol,
            start_day=args.start_day,
            end_day=args.end_day,
            overwrite=args.overwrite,
        )
        print(f"{len(paths)} metrics archives present")
        report = parse_metrics(root=args.root, symbol=args.symbol)
        print(json.dumps(asdict(report), indent=2))

    if args.command == "download-and-parse-series":
        paths = download_series_range(
            root=args.root,
            dataset=args.dataset,
            symbol=args.symbol,
            start_month=args.start_month,
            end_month=args.end_month,
            overwrite=args.overwrite,
        )
        print(f"{len(paths)} {args.dataset} archives present")
        report = parse_series(root=args.root, dataset=args.dataset, symbol=args.symbol)
        print(json.dumps(asdict(report), indent=2))

    if args.command == "download-and-parse-book-ticker":
        for month in iter_months(args.start_month, args.end_month):
            report = download_and_parse_month(
                root=args.root, symbol=args.symbol, month=month, overwrite=args.overwrite
            )
            print(json.dumps(asdict(report)), flush=True)

    if args.command == "backfill-series":
        result = backfill_series(root=args.root, dataset=args.dataset, symbol=args.symbol)
        print(json.dumps(result, indent=2))
        report = parse_series(root=args.root, dataset=args.dataset, symbol=args.symbol)
        print(json.dumps(asdict(report), indent=2))

    if args.command == "status":
        report_dir = args.root / "metadata" / args.symbol
        reports = [
            json.loads(path.read_text())
            for path in sorted(report_dir.glob("????-??.json"))
        ]
        if not reports:
            print(json.dumps({"symbol": args.symbol, "months": 0}, indent=2))
            return 0
        summary = {
            "symbol": args.symbol,
            "months": len(reports),
            "first_month": reports[0]["month"],
            "last_month": reports[-1]["month"],
            "fill_rows": sum(report["fill_rows"] for report in reports),
            "event_rows": sum(report["event_rows"] for report in reports),
            "source_bytes": sum(report["source_bytes"] for report in reports),
            "canonical_bytes": sum(report["canonical_bytes"] for report in reports),
            "event_bytes": sum(report["event_bytes"] for report in reports),
        }
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
