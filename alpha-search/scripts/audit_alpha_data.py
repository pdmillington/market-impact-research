"""Audit one portable shared-data root before alpha experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.bars import EVENT_COLUMNS  # noqa: E402
from alpha_search.research import (  # noqa: E402
    coverage_summary,
    data_fingerprint,
    discover_validated_partitions,
)


def schema_audit(paths: list[Path]) -> dict:
    inspected = []
    for path in paths:
        schema = pl.read_parquet_schema(path)
        names = set(schema)
        inspected.append(
            {
                "path": str(path),
                "columns": sorted(names),
                "missing_required_columns": sorted(EVENT_COLUMNS.difference(names)),
            }
        )
    return {"inspected": inspected, "valid": all(not row["missing_required_columns"] for row in inspected)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data-root", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--event-rule", default="timestamp_direction_v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-contiguous", action="store_true")
    args = parser.parse_args()

    root = args.shared_data_root.expanduser().resolve()
    partitions = discover_validated_partitions(
        root, symbol=args.symbol, event_rule=args.event_rule
    )
    coverage = coverage_summary(partitions)
    schema_paths = []
    if partitions:
        schema_paths = [partitions[0].event_path]
        if len(partitions) > 1:
            schema_paths.append(partitions[-1].event_path)
    source_missing = []
    for partition in partitions:
        source = (
            root / "source" / "binance" / "futures" / "um" / "trades"
            / args.symbol / f"{args.symbol}-trades-{partition.month}.zip"
        )
        if not source.exists():
            source_missing.append(partition.month)
    disk = shutil.disk_usage(root)
    audit = {
        "shared_data_root": str(root),
        "symbol": args.symbol,
        "event_rule": args.event_rule,
        "data_fingerprint": data_fingerprint(partitions),
        "coverage": coverage,
        "contiguous": not coverage["missing_months"],
        "missing_source_archives": source_missing,
        "source_archives_complete": not source_missing,
        "event_schema": schema_audit(schema_paths),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
    text = json.dumps(audit, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)
    if args.require_contiguous and not audit["contiguous"]:
        raise SystemExit(2)
    if not audit["event_schema"]["valid"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
