"""Leakage-safe, interpretable features for the short-event-bar alpha screen."""

from __future__ import annotations

from collections.abc import Sequence
import math

import polars as pl


BASE_SCREEN_FEATURES: tuple[tuple[str, str], ...] = (
    ("log_abs_signed_imbalance", "imbalance"),
    ("absolute_imbalance_ratio", "imbalance"),
    ("normalised_imbalance", "imbalance"),
    ("log_gross_volume", "volume"),
    ("bar_volume_fraction_24h", "volume"),
    ("log_duration", "activity"),
    ("relative_flow_rate", "activity"),
    ("normalised_range", "volatility"),
    ("log_trailing_volatility", "volatility"),
    ("return_efficiency", "impact"),
    ("normalised_signed_impact", "impact"),
    ("fills_per_event", "microstructure"),
    ("price_levels_per_event", "microstructure"),
    ("abs_event_sign_imbalance", "microstructure"),
)

FITTED_SCREEN_FEATURES: tuple[tuple[str, str], ...] = (
    ("expected_normalised_impact", "fitted_impact"),
    ("impact_residual", "fitted_impact"),
    ("marginal_impact_bps_per_btc", "fitted_impact"),
)


def screen_feature_catalog(
    history_event_counts: Sequence[int] = (1_000, 4_000, 12_000),
) -> tuple[tuple[str, str], ...]:
    """Return the preregistered feature names and their families."""

    histories: list[tuple[str, str]] = []
    for count in history_event_counts:
        histories.extend(
            [
                (f"aligned_pretrend_{count}_events", "pretrend"),
                (f"aligned_flow_persistence_{count}_events", "flow_persistence"),
            ]
        )
    return BASE_SCREEN_FEATURES + tuple(histories) + FITTED_SCREEN_FEATURES


def add_short_bar_screen_features(
    bars: pl.LazyFrame | pl.DataFrame,
    *,
    events_per_bar: int,
    history_event_counts: Sequence[int] = (1_000, 4_000, 12_000),
) -> pl.LazyFrame:
    """Add current and strictly prior event-matched features.

    Pre-trend and flow-persistence windows end at bar ``t-1``.  Their event
    counts are held approximately constant across bar constructions so a
    4,000-event history means 20 bars at size 200, eight at size 500, and four
    at size 1,000.
    """

    if events_per_bar <= 0:
        raise ValueError("events_per_bar must be positive.")
    lazy = bars.lazy() if isinstance(bars, pl.DataFrame) else bars
    required = {
        "end_price",
        "signed_imbalance",
        "abs_signed_imbalance",
        "gross_volume",
        "duration_seconds",
        "range_bps",
        "signed_current_impact_bps",
        "normalised_signed_impact",
        "trailing_market_volume",
        "trailing_realised_volatility_bps",
        "flow_rate_btc_per_second",
        "event_count",
        "fill_count",
        "event_price_level_count_sum",
        "signed_event_count",
    }
    missing = required.difference(lazy.collect_schema().names())
    if missing:
        raise KeyError(f"Missing short-bar feature columns: {sorted(missing)}")

    side = pl.col("signed_imbalance").sign()
    expressions: list[pl.Expr] = [
        pl.col("abs_signed_imbalance").log().alias("log_abs_signed_imbalance"),
        pl.col("gross_volume").log().alias("log_gross_volume"),
        pl.col("duration_seconds").clip(lower_bound=0.001).log().alias("log_duration"),
        (pl.col("gross_volume") / pl.col("trailing_market_volume"))
        .alias("bar_volume_fraction_24h"),
        (
            pl.col("flow_rate_btc_per_second")
            / (pl.col("trailing_market_volume") / 86_400.0)
        ).alias("relative_flow_rate"),
        (
            pl.col("range_bps") / pl.col("trailing_realised_volatility_bps")
        ).alias("normalised_range"),
        pl.col("trailing_realised_volatility_bps")
        .log()
        .alias("log_trailing_volatility"),
        (
            pl.col("signed_current_impact_bps")
            / pl.col("range_bps").clip(lower_bound=1e-12)
        ).alias("return_efficiency"),
        (pl.col("fill_count") / pl.col("event_count")).alias("fills_per_event"),
        (
            pl.col("event_price_level_count_sum") / pl.col("event_count")
        ).alias("price_levels_per_event"),
        (
            pl.col("signed_event_count").abs() / pl.col("event_count")
        ).alias("abs_event_sign_imbalance"),
    ]

    for count in history_event_counts:
        bars_in_history = max(1, math.ceil(count / events_per_bar))
        prior_price = pl.col("end_price").shift(1)
        earlier_price = pl.col("end_price").shift(bars_in_history + 1)
        expressions.extend(
            [
                (
                    side * 10_000.0 * (prior_price / earlier_price).log()
                ).alias(f"aligned_pretrend_{count}_events"),
                (
                    side
                    * pl.col("signed_imbalance")
                    .shift(1)
                    .rolling_sum(window_size=bars_in_history, min_samples=bars_in_history)
                    / pl.col("gross_volume")
                    .shift(1)
                    .rolling_sum(window_size=bars_in_history, min_samples=bars_in_history)
                ).alias(f"aligned_flow_persistence_{count}_events"),
            ]
        )
    return lazy.sort("bar_number").with_columns(expressions)
