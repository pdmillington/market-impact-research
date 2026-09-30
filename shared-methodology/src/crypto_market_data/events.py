"""Versioned trade-event reconstruction rules."""

from __future__ import annotations

from pathlib import Path

import polars as pl


EVENT_COLUMNS = [
    "timestamp_ms",
    "aggressor_sign",
    "quantity",
    "vwap",
    "first_price",
    "last_price",
    "min_price",
    "max_price",
    "fill_count",
    "price_level_count",
    "first_trade_id",
    "last_trade_id",
]


def timestamp_direction_events(
    lazy_fills: pl.LazyFrame,
    order_by: tuple[str, ...] = ("trade_id",),
) -> pl.LazyFrame:
    """Combine consecutive fills sharing timestamp and aggressor direction.

    Price is deliberately not a grouping key: a single aggressive sweep may execute
    against several price levels. A direction change at the same timestamp starts a
    new event, and a later return to the original direction starts another event.

    Fills are processed in exchange sequence, normally trade-ID order. A month in
    which the exchange reused trade IDs after an outage is processed in
    (timestamp, trade ID) order instead, which restores the true sequence.
    """

    ordered = lazy_fills.sort(list(order_by), maintain_order=True)
    is_new_event = (
        (pl.col("timestamp_ms") != pl.col("timestamp_ms").shift(1))
        | (pl.col("aggressor_sign") != pl.col("aggressor_sign").shift(1))
    ).fill_null(True)

    return (
        ordered.with_columns(
            is_new_event.cast(pl.UInt64).cum_sum().alias("_event_group"),
            (pl.col("price") * pl.col("quantity")).alias("_notional"),
        )
        .group_by("_event_group", maintain_order=True)
        .agg(
            pl.col("timestamp_ms").first(),
            pl.col("aggressor_sign").first(),
            pl.col("quantity").sum(),
            pl.col("_notional").sum().alias("notional"),
            pl.col("price").first().alias("first_price"),
            pl.col("price").last().alias("last_price"),
            pl.col("price").min().alias("min_price"),
            pl.col("price").max().alias("max_price"),
            pl.len().cast(pl.UInt32).alias("fill_count"),
            pl.col("price").n_unique().cast(pl.UInt32).alias("price_level_count"),
            pl.col("trade_id").first().alias("first_trade_id"),
            pl.col("trade_id").last().alias("last_trade_id"),
        )
        .with_columns((pl.col("notional") / pl.col("quantity")).alias("vwap"))
        .select(EVENT_COLUMNS)
    )


def reconstruct_timestamp_direction_events(
    canonical_path: Path,
    output_path: Path,
    order_by: tuple[str, ...] = ("trade_id",),
) -> Path:
    """Write `timestamp_direction_v1` events from canonical fills."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    temporary_path.unlink(missing_ok=True)
    events = timestamp_direction_events(pl.scan_parquet(canonical_path), order_by)
    events.sink_parquet(
        temporary_path,
        compression="zstd",
        compression_level=6,
        statistics=True,
        row_group_size=250_000,
        maintain_order=True,
    )
    temporary_path.replace(output_path)
    return output_path
