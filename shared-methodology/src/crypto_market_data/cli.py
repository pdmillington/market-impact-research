"""Command-line interface for shared market-data ingestion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .download import download_archive, download_range
from .layout import DataLayout
from .months import iter_months
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
