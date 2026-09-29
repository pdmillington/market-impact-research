"""Build versioned short-bar caches for impact-function research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.bars import (  # noqa: E402
    add_trailing_impact_features,
    build_fixed_event_bars,
    build_time_bars,
)
from alpha_search.research import (  # noqa: E402
    coverage_summary,
    data_fingerprint,
    discover_validated_partitions,
)


DEFAULT_EVENT_SIZES = (200, 500, 1_000)
DEFAULT_TIME_BARS = ("30s", "1m", "2m", "5m")


def add_impact_v2_fields(bars: pl.LazyFrame) -> pl.LazyFrame:
    """Add the two axes of the proposed dimensionless impact surface."""

    return bars.with_columns(
        (
            pl.col("gross_volume") / pl.col("trailing_market_volume")
        ).alias("relative_activity"),
        pl.col("signed_imbalance_ratio").abs().alias("relative_imbalance"),
    )


def sink_bars(bars: pl.LazyFrame, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        print(f"Keeping existing {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("part-000.tmp.parquet")
    temporary.unlink(missing_ok=True)
    add_impact_v2_fields(
        add_trailing_impact_features(bars, window="1d", min_samples=100)
    ).sink_parquet(temporary, compression="zstd", compression_level=6)
    temporary.replace(destination)
    print(destination, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--shared-data-root", type=Path, default=REPO / "shared-data"
    )
    parser.add_argument("--end-month", default="2026-05")
    parser.add_argument(
        "--event-sizes", nargs="*", type=int, default=list(DEFAULT_EVENT_SIZES)
    )
    parser.add_argument(
        "--time-bars", nargs="*", default=list(DEFAULT_TIME_BARS)
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if any(size <= 0 for size in args.event_sizes):
        raise ValueError("Event-bar sizes must be positive.")
    shared = args.shared_data_root.expanduser().resolve()
    partitions = [
        partition
        for partition in discover_validated_partitions(shared, symbol=args.symbol)
        if partition.month <= args.end_month
    ]
    if not partitions:
        raise FileNotFoundError("No validated partitions are available.")
    paths = [partition.event_path for partition in partitions]
    output_root = args.output_root or (
        shared
        / "features"
        / "alpha"
        / f"symbol={args.symbol}"
        / "impact_research_v2"
    )

    for size in sorted(set(args.event_sizes)):
        print(f"Building {size:,}-event impact bars", flush=True)
        events = pl.scan_parquet(paths)
        bars = build_fixed_event_bars(
            events, events_per_bar=size, assume_ordered=True
        )
        sink_bars(
            bars,
            output_root / f"event_{size}" / "part-000.parquet",
            overwrite=args.overwrite,
        )

    for every in dict.fromkeys(args.time_bars):
        print(f"Building {every} clock-time impact bars", flush=True)
        events = pl.scan_parquet(paths)
        bars = build_time_bars(events, every=every, assume_ordered=True)
        sink_bars(
            bars,
            output_root / f"time_{every}" / "part-000.parquet",
            overwrite=args.overwrite,
        )

    existing_event_sizes = sorted(
        int(path.parent.name.split("_", maxsplit=1)[1])
        for path in output_root.glob("event_*/part-000.parquet")
    )
    existing_time_bars = sorted(
        path.parent.name.split("_", maxsplit=1)[1]
        for path in output_root.glob("time_*/part-000.parquet")
    )
    manifest = {
        "cache_version": "impact_research_v2",
        "symbol": args.symbol,
        "event_rule": "timestamp_direction_v1",
        "shared_data_root": str(shared),
        "data_fingerprint": data_fingerprint(partitions),
        "coverage": coverage_summary(partitions),
        "holdout_months_excluded": ["2026-06", "2026-07", "2026-08"]
        if args.end_month == "2026-05"
        else [],
        "event_bar_sizes": existing_event_sizes,
        "time_bars": existing_time_bars,
        "trailing_window": "1d",
        "impact_axes": {
            "relative_imbalance": "abs(signed_imbalance) / gross_volume",
            "relative_activity": "gross_volume / preceding_24h_gross_volume",
            "normalised_net_flow": "abs(signed_imbalance) / preceding_24h_gross_volume",
            "normalised_response": "side_aligned_bar_return / preceding_24h_realised_volatility",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(output_root / "cache_manifest.json", flush=True)


if __name__ == "__main__":
    main()
