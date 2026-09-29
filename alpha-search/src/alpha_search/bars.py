"""Corrected-data bars and impact-oriented alpha features.

The input is the shared ``timestamp_direction_v1`` event schema. Both bar
constructions use the same events, signing convention, and price fields so
that event-time and clock-time results can be compared without changing the
underlying observations.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


EVENT_COLUMNS = {
    "timestamp_ms",
    "aggressor_sign",
    "quantity",
    "first_price",
    "last_price",
    "min_price",
    "max_price",
    "fill_count",
    "price_level_count",
    "first_trade_id",
    "last_trade_id",
}


def _validate_event_schema(events: pl.LazyFrame) -> None:
    missing = EVENT_COLUMNS.difference(events.collect_schema().names())
    if missing:
        raise KeyError(f"Missing corrected-event columns: {sorted(missing)}")


def _finalise_bars(bars: pl.LazyFrame, *, construction: str) -> pl.LazyFrame:
    """Add common, contemporaneously observable bar features."""

    return (
        bars.sort("bar_number")
        .with_columns(
            pl.lit(construction).alias("construction"),
            ((pl.col("end_time_ms") - pl.col("start_time_ms")) / 1_000.0)
            .alias("duration_seconds"),
            (10_000.0 * (pl.col("end_price") / pl.col("start_price")).log())
            .alias("current_return_bps"),
            (10_000.0 * (pl.col("high_price") / pl.col("low_price")).log())
            .alias("range_bps"),
            pl.col("signed_imbalance").abs().alias("abs_signed_imbalance"),
        )
        .with_columns(
            (pl.col("signed_imbalance") / pl.col("gross_volume"))
            .alias("signed_imbalance_ratio"),
            (pl.col("abs_signed_imbalance") / pl.col("gross_volume"))
            .alias("absolute_imbalance_ratio"),
            (
                pl.col("signed_imbalance").sign()
                * pl.col("current_return_bps")
            ).alias("signed_current_impact_bps"),
        )
    )


def build_fixed_event_bars(
    events: pl.LazyFrame,
    *,
    events_per_bar: int = 4_000,
    drop_incomplete: bool = True,
    assume_ordered: bool = False,
) -> pl.LazyFrame:
    """Build continuous fixed-count bars from reconstructed events.

    The caller should scan all required monthly partitions in one LazyFrame.
    Bar numbering therefore continues across month boundaries rather than
    restarting each month.
    """

    if events_per_bar <= 0:
        raise ValueError("events_per_bar must be positive.")
    _validate_event_schema(events)

    source = events if assume_ordered else events.sort("first_trade_id")
    ordered = (
        source
        .with_row_index("_event_number")
        .with_columns(
            (pl.col("_event_number") // events_per_bar)
            .cast(pl.UInt64)
            .alias("bar_number"),
            (
                pl.col("aggressor_sign").cast(pl.Float64)
                * pl.col("quantity")
            ).alias("_signed_quantity"),
            pl.col("aggressor_sign").cast(pl.Int64).alias("_event_sign"),
        )
    )
    bars = ordered.group_by("bar_number", maintain_order=True).agg(
        pl.col("timestamp_ms").first().alias("start_time_ms"),
        pl.col("timestamp_ms").last().alias("end_time_ms"),
        pl.col("first_price").first().alias("start_price"),
        pl.col("last_price").last().alias("end_price"),
        pl.col("max_price").max().alias("high_price"),
        pl.col("min_price").min().alias("low_price"),
        pl.col("quantity").sum().alias("gross_volume"),
        pl.col("_signed_quantity").sum().alias("signed_imbalance"),
        pl.col("_event_sign").sum().alias("signed_event_count"),
        pl.len().cast(pl.UInt32).alias("event_count"),
        pl.col("fill_count").sum().alias("fill_count"),
        pl.col("price_level_count").sum().alias("event_price_level_count_sum"),
        pl.col("first_trade_id").first(),
        pl.col("last_trade_id").last(),
    )
    if drop_incomplete:
        bars = bars.filter(pl.col("event_count") == events_per_bar)
    return _finalise_bars(bars, construction=f"event_{events_per_bar}")


def coarsen_fixed_event_bars(
    bars: pl.LazyFrame,
    *,
    source_events_per_bar: int,
    target_events_per_bar: int,
) -> pl.LazyFrame:
    """Combine aligned fixed-event bars into a coarser fixed-event clock.

    This is exact when ``target_events_per_bar`` is an integer multiple of the
    source size and the source bars were formed continuously from the same
    ordered event stream. Trailing normalisations are deliberately not carried
    through: callers must recompute them on the coarser return series.
    """

    if source_events_per_bar <= 0 or target_events_per_bar <= 0:
        raise ValueError("Event-bar sizes must be positive.")
    if target_events_per_bar % source_events_per_bar:
        raise ValueError("Target event count must be a multiple of the source count.")
    factor = target_events_per_bar // source_events_per_bar
    if factor < 2:
        raise ValueError("Target event count must be larger than the source count.")
    required = {
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
    }
    missing = required.difference(bars.collect_schema().names())
    if missing:
        raise KeyError(f"Missing fixed-event bar columns: {sorted(missing)}")

    grouped = (
        bars.sort("bar_number")
        .with_columns(
            (pl.col("bar_number") // factor).cast(pl.UInt64).alias("_coarse_number")
        )
        .group_by("_coarse_number", maintain_order=True)
        .agg(
            pl.col("start_time_ms").first(),
            pl.col("end_time_ms").last(),
            pl.col("start_price").first(),
            pl.col("end_price").last(),
            pl.col("high_price").max(),
            pl.col("low_price").min(),
            pl.col("gross_volume").sum(),
            pl.col("signed_imbalance").sum(),
            pl.col("signed_event_count").sum(),
            pl.col("event_count").sum(),
            pl.col("fill_count").sum(),
            pl.col("event_price_level_count_sum").sum(),
            pl.col("first_trade_id").first(),
            pl.col("last_trade_id").last(),
            pl.len().alias("_source_bar_count"),
        )
        .filter(
            (pl.col("_source_bar_count") == factor)
            & (pl.col("event_count") == target_events_per_bar)
        )
        .rename({"_coarse_number": "bar_number"})
        .drop("_source_bar_count")
    )
    return _finalise_bars(grouped, construction=f"event_{target_events_per_bar}")


def build_time_bars(
    events: pl.LazyFrame,
    *,
    every: str = "5m",
    assume_ordered: bool = False,
) -> pl.LazyFrame:
    """Build non-overlapping clock-time bars from reconstructed events."""

    _validate_event_schema(events)
    source = events if assume_ordered else events.sort("first_trade_id")
    timed = source.with_columns(
            pl.from_epoch("timestamp_ms", time_unit="ms").alias("timestamp"),
            (
                pl.col("aggressor_sign").cast(pl.Float64)
                * pl.col("quantity")
            ).alias("_signed_quantity"),
            pl.col("aggressor_sign").cast(pl.Int64).alias("_event_sign"),
    )
    timed = timed.set_sorted("timestamp") if assume_ordered else timed.sort("timestamp")
    bars = (
        timed.group_by_dynamic(
            "timestamp",
            every=every,
            period=every,
            closed="left",
            label="left",
        )
        .agg(
            pl.col("timestamp_ms").first().alias("start_time_ms"),
            pl.col("timestamp_ms").last().alias("end_time_ms"),
            pl.col("first_price").first().alias("start_price"),
            pl.col("last_price").last().alias("end_price"),
            pl.col("max_price").max().alias("high_price"),
            pl.col("min_price").min().alias("low_price"),
            pl.col("quantity").sum().alias("gross_volume"),
            pl.col("_signed_quantity").sum().alias("signed_imbalance"),
            pl.col("_event_sign").sum().alias("signed_event_count"),
            pl.len().cast(pl.UInt32).alias("event_count"),
            pl.col("fill_count").sum().alias("fill_count"),
            pl.col("price_level_count").sum().alias("event_price_level_count_sum"),
            pl.col("first_trade_id").first(),
            pl.col("last_trade_id").last(),
        )
        .filter(pl.col("event_count") > 0)
        .with_row_index("bar_number")
    )
    return _finalise_bars(bars, construction=f"time_{every}")


def add_bar_horizon_targets(
    bars: pl.LazyFrame,
    *,
    horizons: Sequence[int] = (3, 4, 5, 6),
) -> pl.LazyFrame:
    """Add close-to-future-close returns and realized elapsed time.

    Horizon ``h`` means the return from the end of bar ``n`` to the end of
    bar ``n+h``. For fixed-event bars, the accompanying elapsed-minutes
    column makes the variable clock duration explicit. With contiguous
    five-minute bars, horizons 3--6 correspond to 15--30 minutes.
    """

    horizon_values = tuple(int(h) for h in horizons)
    if not horizon_values or any(h <= 0 for h in horizon_values):
        raise ValueError("horizons must contain positive integers.")
    if len(set(horizon_values)) != len(horizon_values):
        raise ValueError("horizons must be unique.")

    schema = set(bars.collect_schema().names())
    required = {"bar_number", "end_time_ms", "end_price"}
    missing = required.difference(schema)
    if missing:
        raise KeyError(f"Missing bar columns: {sorted(missing)}")

    expressions: list[pl.Expr] = []
    for horizon in horizon_values:
        future_price = pl.col("end_price").shift(-horizon)
        future_time = pl.col("end_time_ms").shift(-horizon)
        expressions.extend(
            [
                (10_000.0 * (future_price / pl.col("end_price")).log())
                .alias(f"future_return_{horizon}_bps"),
                ((future_time - pl.col("end_time_ms")) / 60_000.0)
                .alias(f"future_elapsed_{horizon}_minutes"),
            ]
        )
    return bars.sort("bar_number").with_columns(expressions)


def add_trailing_impact_features(
    bars: pl.LazyFrame,
    *,
    window: str = "1d",
    min_samples: int = 100,
) -> pl.LazyFrame:
    """Add strictly backward-looking impact normalisations.

    The same clock-time trailing window is used for event and time bars. The
    current bar is excluded, preventing its volume or return from entering its
    own benchmark.
    """

    if min_samples < 1:
        raise ValueError("min_samples must be at least one.")
    schema = set(bars.collect_schema().names())
    required = {
        "end_time_ms",
        "gross_volume",
        "current_return_bps",
        "signed_imbalance",
        "duration_seconds",
    }
    missing = required.difference(schema)
    if missing:
        raise KeyError(f"Missing bar columns: {sorted(missing)}")

    ordered = (
        bars.sort("end_time_ms")
        .with_columns(
            pl.from_epoch("end_time_ms", time_unit="ms").alias("end_time"),
            pl.col("current_return_bps").pow(2).alias("_squared_return_bps"),
        )
        .with_columns(
            pl.col("gross_volume")
            .rolling_sum_by(
                "end_time",
                window_size=window,
                min_samples=min_samples,
                closed="left",
            )
            .alias("trailing_market_volume"),
            pl.col("_squared_return_bps")
            .rolling_sum_by(
                "end_time",
                window_size=window,
                min_samples=min_samples,
                closed="left",
            )
            .sqrt()
            .alias("trailing_realised_volatility_bps"),
        )
    )
    return (
        ordered.with_columns(
            (
                pl.col("signed_imbalance").abs()
                / pl.col("trailing_market_volume")
            ).alias("normalised_imbalance"),
            (
                pl.col("signed_current_impact_bps")
                / pl.col("trailing_realised_volatility_bps")
            ).alias("normalised_signed_impact"),
            (pl.col("gross_volume") / pl.col("duration_seconds"))
            .alias("flow_rate_btc_per_second"),
        )
        .drop("_squared_return_bps")
    )


def marginal_impact_proxy(
    *,
    normalised_imbalance: pl.Expr,
    trailing_volume: pl.Expr,
    trailing_volatility_bps: pl.Expr,
    log_slope: float,
    imbalance_scale: float,
) -> pl.Expr:
    """Return the fitted marginal response in bps per additional BTC.

    For an impact curve ``a + b*log(1 + x/scale)``, where
    ``x = abs(imbalance)/trailing_volume``, the derivative with respect to
    additional signed BTC is ``volatility*b / ((scale+x)*trailing_volume)``.
    Fit separate buy and sell slopes on training data before using this proxy.
    """

    if imbalance_scale <= 0:
        raise ValueError("imbalance_scale must be positive.")
    return (
        trailing_volatility_bps
        * pl.lit(float(log_slope))
        / (
            (pl.lit(float(imbalance_scale)) + normalised_imbalance)
            * trailing_volume
        )
    )
