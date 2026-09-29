"""Leakage-safe trailing impact observations on a separate decision clock."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


REQUIRED_COLUMNS = {
    "bar_number",
    "start_time_ms",
    "end_time_ms",
    "start_price",
    "end_price",
    "gross_volume",
    "signed_imbalance",
    "trailing_market_volume",
    "trailing_realised_volatility_bps",
}


def resolve_measurement_windows(
    events_per_bar: int,
    requested: Sequence[int | str],
) -> tuple[int, ...]:
    """Resolve ``same_bar`` and validate exact trailing-event windows."""

    if events_per_bar <= 0:
        raise ValueError("events_per_bar must be positive.")
    resolved: list[int] = []
    for value in requested:
        events = events_per_bar if value == "same_bar" else int(value)
        if events < events_per_bar:
            raise ValueError(
                f"Measurement window {events} is shorter than decision bar "
                f"{events_per_bar}."
            )
        if events % events_per_bar:
            raise ValueError(
                f"Measurement window {events} is not an exact multiple of "
                f"decision bar {events_per_bar}."
            )
        if events not in resolved:
            resolved.append(events)
    if not resolved:
        raise ValueError("At least one impact measurement window is required.")
    return tuple(resolved)


def build_trailing_impact_observations(
    bars: pl.LazyFrame | pl.DataFrame,
    *,
    events_per_bar: int,
    measurement_events: int,
) -> pl.LazyFrame:
    """Aggregate a trailing impact window ending at each decision-bar close.

    The output retains the decision-bar clock but replaces the standard impact
    columns with measurements over ``measurement_events``. The volume and
    volatility benchmarks are taken from immediately before the first bar in
    the measurement window, so the window does not enter its own normalisation.

    ``is_non_overlapping_measurement`` marks globally aligned endpoints whose
    measurement windows partition the underlying events without overlap. These
    rows are suitable for calibrating an impact curve; all rows remain available
    for applying that frozen curve as a rolling trading feature.
    """

    if events_per_bar <= 0 or measurement_events <= 0:
        raise ValueError("Event counts must be positive.")
    if measurement_events < events_per_bar:
        raise ValueError("measurement_events cannot be shorter than events_per_bar.")
    if measurement_events % events_per_bar:
        raise ValueError("measurement_events must be a multiple of events_per_bar.")
    lazy = bars.lazy() if isinstance(bars, pl.DataFrame) else bars
    missing = REQUIRED_COLUMNS.difference(lazy.collect_schema().names())
    if missing:
        raise KeyError(f"Missing bar columns: {sorted(missing)}")

    count = measurement_events // events_per_bar
    offset = count - 1
    ordered = (
        lazy.sort("bar_number")
        .with_columns(
            pl.col("signed_imbalance").alias("decision_signed_imbalance"),
            pl.col("start_time_ms").shift(offset).alias("measurement_start_time_ms"),
            pl.col("start_price").shift(offset).alias("measurement_start_price"),
            pl.col("signed_imbalance")
            .rolling_sum(window_size=count, min_samples=count)
            .alias("measurement_signed_imbalance"),
            pl.col("gross_volume")
            .rolling_sum(window_size=count, min_samples=count)
            .alias("measurement_gross_volume"),
            pl.col("trailing_market_volume")
            .shift(offset)
            .alias("measurement_benchmark_volume"),
            pl.col("trailing_realised_volatility_bps")
            .shift(offset)
            .alias("measurement_benchmark_volatility_bps"),
            (
                (pl.col("bar_number") % pl.lit(count)) == pl.lit(offset)
            ).alias("is_non_overlapping_measurement"),
        )
        .with_columns(
            pl.col("measurement_signed_imbalance").alias("signed_imbalance"),
            pl.col("measurement_benchmark_volume").alias("trailing_market_volume"),
            pl.col("measurement_benchmark_volatility_bps")
            .alias("trailing_realised_volatility_bps"),
            (
                (pl.col("end_time_ms") - pl.col("measurement_start_time_ms"))
                / 1_000.0
            ).alias("measurement_duration_seconds"),
            (
                10_000.0
                * (pl.col("end_price") / pl.col("measurement_start_price")).log()
            ).alias("measurement_return_bps"),
        )
        .with_columns(
            (
                pl.col("measurement_signed_imbalance").abs()
                / pl.col("measurement_benchmark_volume")
            ).alias("normalised_imbalance"),
            (
                pl.col("measurement_signed_imbalance").sign()
                * pl.col("measurement_return_bps")
                / pl.col("measurement_benchmark_volatility_bps")
            ).alias("normalised_signed_impact"),
            pl.col("measurement_signed_imbalance").sign().alias("measurement_side"),
            pl.col("decision_signed_imbalance").sign().alias("decision_side"),
            pl.lit(measurement_events).cast(pl.UInt32).alias("measurement_events"),
        )
        .with_columns(
            (pl.col("measurement_side") == pl.col("decision_side"))
            .alias("side_agreement")
        )
    )
    return ordered
