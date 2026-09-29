"""Derive larger fixed-event impact bars from an aligned smaller-bar cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.bars import (  # noqa: E402
    add_trailing_impact_features,
    coarsen_fixed_event_bars,
)


def add_impact_fields(bars: pl.LazyFrame) -> pl.LazyFrame:
    return bars.with_columns(
        (pl.col("gross_volume") / pl.col("trailing_market_volume"))
        .alias("relative_activity"),
        pl.col("signed_imbalance_ratio").abs().alias("relative_imbalance"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--shared-data-root", type=Path, default=REPO / "shared-data"
    )
    parser.add_argument("--source-size", type=int, default=1_000)
    parser.add_argument(
        "--target-sizes", nargs="+", type=int, default=[2_000, 4_000, 8_000]
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = (
        args.shared_data_root.expanduser().resolve()
        / "features"
        / "alpha"
        / f"symbol={args.symbol}"
        / "impact_research_v2"
    )
    source = root / f"event_{args.source_size}" / "part-000.parquet"
    if not source.exists():
        raise FileNotFoundError(source)
    raw_columns = [
        "bar_number",
        "start_time_ms",
        "end_time_ms",
        "start_price",
        "end_price",
        "high_price",
        "low_price",
        "gross_volume",
        "signed_imbalance",
        "signed_event_count",
        "event_count",
        "fill_count",
        "event_price_level_count_sum",
        "first_trade_id",
        "last_trade_id",
    ]
    for target in sorted(set(args.target_sizes)):
        destination = root / f"event_{target}" / "part-000.parquet"
        if destination.exists() and not args.overwrite:
            print(f"Keeping existing {destination}", flush=True)
            continue
        print(f"Deriving event_{target} from event_{args.source_size}", flush=True)
        bars = pl.scan_parquet(source).select(raw_columns)
        coarser = coarsen_fixed_event_bars(
            bars,
            source_events_per_bar=args.source_size,
            target_events_per_bar=target,
        )
        featured = add_impact_fields(
            add_trailing_impact_features(coarser, window="1d", min_samples=100)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name("part-000.tmp.parquet")
        temporary.unlink(missing_ok=True)
        featured.sink_parquet(temporary, compression="zstd", compression_level=6)
        temporary.replace(destination)
        print(destination, flush=True)

    manifest_path = root / "cache_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["event_bar_sizes"] = sorted(
            int(path.parent.name.split("_", 1)[1])
            for path in root.glob("event_*/part-000.parquet")
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
