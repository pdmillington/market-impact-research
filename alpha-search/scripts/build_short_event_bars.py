"""Build short fixed-event alpha caches from corrected shared events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.bars import (  # noqa: E402
    add_bar_horizon_targets,
    add_trailing_impact_features,
    build_fixed_event_bars,
)
from alpha_search.screen_features import add_short_bar_screen_features  # noqa: E402
from alpha_search.research import (  # noqa: E402
    coverage_summary,
    data_fingerprint,
    discover_validated_partitions,
)


HORIZONS = {
    50: (240, 320),
    100: (120, 160),
    200: (60, 80),
    500: (24, 32),
    1_000: (12, 16),
}
FUTURE_EVENT_COUNTS = (12_000, 16_000)


def validated_event_paths(shared: Path, symbol: str) -> list[Path]:
    reports = sorted((shared / "metadata" / symbol).glob("????-??.json"))
    paths: list[Path] = []
    for report in reports:
        year, month = report.stem.split("-")
        path = (
            shared
            / "events"
            / "timestamp_direction_v1"
            / f"symbol={symbol}"
            / f"year={year}"
            / f"month={month}"
            / "part-000.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"Validated event partition is missing: {path}")
        paths.append(path)
    if not paths:
        raise FileNotFoundError("No validated corrected-event partitions were found.")
    return paths


def build_size(
    *,
    event_paths: list[Path],
    output_root: Path,
    events_per_bar: int,
    overwrite: bool,
) -> Path:
    if events_per_bar not in HORIZONS:
        raise ValueError(f"No preregistered horizons for {events_per_bar} events.")
    destination = output_root / f"event_{events_per_bar}" / "part-000.parquet"
    if destination.exists() and not overwrite:
        print(f"Keeping existing {destination}", flush=True)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("part-000.tmp.parquet")
    events = pl.scan_parquet(event_paths)
    bars = build_fixed_event_bars(
        events, events_per_bar=events_per_bar, assume_ordered=True
    )
    bars = add_bar_horizon_targets(bars, horizons=HORIZONS[events_per_bar])
    bars = add_trailing_impact_features(bars, window="1d", min_samples=100)
    bars = add_short_bar_screen_features(
        bars,
        events_per_bar=events_per_bar,
        history_event_counts=(1_000, 4_000, 12_000),
    )
    print(
        f"Building {events_per_bar:,}-event bars with horizons {HORIZONS[events_per_bar]}",
        flush=True,
    )
    bars.sink_parquet(temporary, compression="zstd", compression_level=6)
    temporary.replace(destination)
    print(destination, flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--shared-data-root",
        type=Path,
        default=REPO / "shared-data",
        help="Portable shared-data root; may be on an external SSD.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the alpha-cache destination below the shared-data root.",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=sorted(HORIZONS)
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    shared = args.shared_data_root.expanduser().resolve()
    output_root = args.output_root or (
        shared / "features" / "alpha" / f"symbol={args.symbol}"
        / "short_event_screen_v1"
    )
    partitions = discover_validated_partitions(shared, symbol=args.symbol)
    paths = [partition.event_path for partition in partitions]
    fingerprint = data_fingerprint(partitions)
    coverage = coverage_summary(partitions)
    manifest_path = output_root / "cache_manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )
    existing_sizes = {
        int(path.parent.name.split("_", maxsplit=1)[1])
        for path in output_root.glob("event_*/part-000.parquet")
    }
    if existing_sizes and not args.overwrite:
        if existing_manifest is None:
            raise RuntimeError(
                "Existing short-event caches have no data fingerprint. "
                "Rebuild with --overwrite before using them with extended data."
            )
        if existing_manifest.get("data_fingerprint") != fingerprint:
            raise RuntimeError(
                "Existing caches were built from different data coverage. "
                "Rebuild all cached sizes with --overwrite."
            )
    if (
        existing_sizes
        and args.overwrite
        and (
            existing_manifest is None
            or existing_manifest.get("data_fingerprint") != fingerprint
        )
        and not existing_sizes.issubset(set(args.sizes))
    ):
        raise RuntimeError(
            "Data coverage changed, but --sizes does not include every existing "
            f"cache ({sorted(existing_sizes)}). Rebuild all existing sizes together."
        )
    for size in args.sizes:
        build_size(
            event_paths=paths,
            output_root=output_root,
            events_per_bar=size,
            overwrite=args.overwrite,
        )

    all_sizes = sorted(existing_sizes | set(args.sizes))
    config = {
        "symbol": args.symbol,
        "event_rule": "timestamp_direction_v1",
        "event_bar_sizes": all_sizes,
        "future_event_counts": list(FUTURE_EVENT_COUNTS),
        "horizons_by_size": {str(k): list(v) for k, v in HORIZONS.items()},
        "trailing_window": "1d",
        "history_event_counts": [1_000, 4_000, 12_000],
    }
    (output_root / "screen_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    cache_manifest = {
        "cache_version": "short_event_screen_v1",
        "symbol": args.symbol,
        "event_rule": "timestamp_direction_v1",
        "shared_data_root": str(shared),
        "data_fingerprint": fingerprint,
        "coverage": coverage,
        "event_bar_sizes": all_sizes,
        "horizons_by_size": {str(k): list(v) for k, v in HORIZONS.items()},
        "trailing_window": "1d",
        "history_event_counts": [1_000, 4_000, 12_000],
    }
    manifest_path.write_text(json.dumps(cache_manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
