"""Run the first event-time versus clock-time impact-alpha comparison."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.impact_model import (  # noqa: E402
    add_asymmetric_impact_features,
    fit_asymmetric_log_impact,
)


REFERENCE_HORIZONS = (3, 4, 5, 6)
NOMINAL_MINUTES = {3: 15, 4: 20, 5: 25, 6: 30}
FEATURES = ("marginal_impact_bps_per_btc", "impact_residual")


def utc_ms(value: str) -> int:
    return int(
        datetime.strptime(value, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def matched_horizon(events_per_bar: int | None, reference: int) -> int:
    if events_per_bar is None:
        return reference
    return max(1, int(reference * 4_000 / events_per_bar + 0.5))


def thresholds(
    frame: pl.LazyFrame,
    feature: str,
    side: int,
) -> tuple[float, float, float, float]:
    values = frame.filter(pl.col("signed_imbalance").sign() == side).select(
        *[
            pl.col(feature).quantile(q).alias(str(q))
            for q in (0.2, 0.4, 0.6, 0.8)
        ]
    ).collect().row(0)
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def quintile(feature: str, cuts: tuple[float, float, float, float]) -> pl.Expr:
    q1, q2, q3, q4 = cuts
    return (
        pl.when(pl.col(feature) <= q1)
        .then(pl.lit("Q1"))
        .when(pl.col(feature) <= q2)
        .then(pl.lit("Q2"))
        .when(pl.col(feature) <= q3)
        .then(pl.lit("Q3"))
        .when(pl.col(feature) <= q4)
        .then(pl.lit("Q4"))
        .otherwise(pl.lit("Q5"))
    )


def mean_and_t(values: list[float]) -> tuple[int, float, float]:
    finite = [value for value in values if math.isfinite(value)]
    n = len(finite)
    if n < 2:
        return n, math.nan, math.nan
    mean = sum(finite) / n
    variance = sum((value - mean) ** 2 for value in finite) / (n - 1)
    t_stat = mean / (math.sqrt(variance) / math.sqrt(n)) if variance > 0 else math.nan
    return n, mean, t_stat


def run(
    *,
    cache_root: Path,
    output_root: Path,
    train_end: int,
    validation_end: int,
) -> None:
    constructions = {
        "time_5m": (cache_root / "time_5m" / "part-000.parquet", None),
        "event_2000": (cache_root / "event_2000" / "part-000.parquet", 2_000),
        "event_4000": (cache_root / "event_4000" / "part-000.parquet", 4_000),
        "event_8000": (cache_root / "event_8000" / "part-000.parquet", 8_000),
    }
    period_filters = {
        "train": lambda frame: frame.filter(pl.col("end_time_ms") < train_end),
        "validation": lambda frame: frame.filter(
            (pl.col("end_time_ms") >= train_end)
            & (pl.col("end_time_ms") < validation_end)
        ),
        "sandbox_test": lambda frame: frame.filter(
            pl.col("end_time_ms") >= validation_end
        ),
    }
    mapping_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []

    for name, (path, events_per_bar) in constructions.items():
        print(f"Analysing {name}", flush=True)
        bars = pl.scan_parquet(path)
        training = period_filters["train"](bars)
        fit = fit_asymmetric_log_impact(training)
        for label, side_fit in (("buy", fit.buy), ("sell", fit.sell)):
            fit_rows.append(
                {
                    "construction": name,
                    "side": label,
                    "observations": side_fit.observations,
                    "imbalance_scale": fit.imbalance_scale,
                    "intercept": side_fit.intercept,
                    "slope": side_fit.slope,
                    "r_squared": side_fit.r_squared,
                }
            )
        enriched = add_asymmetric_impact_features(bars, fit).with_columns(
            pl.from_epoch("end_time_ms", time_unit="ms").dt.date().alias("utc_date")
        )
        training_enriched = period_filters["train"](enriched)
        cuts = {
            (feature, side): thresholds(training_enriched, feature, side)
            for feature in FEATURES
            for side in (-1, 1)
        }

        for reference in REFERENCE_HORIZONS:
            horizon = matched_horizon(events_per_bar, reference)
            elapsed_col = f"future_elapsed_{horizon}_minutes"
            target_col = f"future_return_{horizon}_bps"
            for period, selector in period_filters.items():
                duration = selector(enriched).select(
                    pl.col(elapsed_col).quantile(0.1).alias("p10"),
                    pl.col(elapsed_col).median().alias("median"),
                    pl.col(elapsed_col).quantile(0.9).alias("p90"),
                ).collect().row(0)
                mapping_rows.append(
                    {
                        "construction": name,
                        "reference_4000_horizon": reference,
                        "nominal_time_minutes": NOMINAL_MINUTES[reference],
                        "horizon_bars": horizon,
                        "future_event_count": (
                            None if events_per_bar is None else horizon * events_per_bar
                        ),
                        "period": period,
                        "p10_minutes": duration[0],
                        "median_minutes": duration[1],
                        "p90_minutes": duration[2],
                    }
                )

            for feature in FEATURES:
                for period in ("validation", "sandbox_test"):
                    base = period_filters[period](enriched).filter(
                        pl.col(target_col).is_not_null()
                    )
                    for side in (-1, 1):
                        side_label = "buy" if side == 1 else "sell"
                        sample = base.filter(
                            pl.col("signed_imbalance").sign() == side
                        ).with_columns(
                            quintile(feature, cuts[(feature, side)]).alias("quintile"),
                            (pl.lit(side) * pl.col(target_col)).alias(
                                "aligned_future_return_bps"
                            ),
                        )
                        grouped = sample.group_by("quintile").agg(
                            pl.len().alias("observations"),
                            pl.col(feature).mean().alias("mean_feature"),
                            pl.col("aligned_future_return_bps")
                            .mean()
                            .alias("mean_aligned_return_bps"),
                            pl.col("aligned_future_return_bps")
                            .std()
                            .alias("return_std_bps"),
                        ).collect()
                        for row in grouped.iter_rows(named=True):
                            summary_rows.append(
                                {
                                    "construction": name,
                                    "reference_4000_horizon": reference,
                                    "nominal_time_minutes": NOMINAL_MINUTES[reference],
                                    "horizon_bars": horizon,
                                    "period": period,
                                    "feature": feature,
                                    "side": side_label,
                                    **row,
                                }
                            )
                        daily = (
                            sample.filter(pl.col("quintile").is_in(["Q1", "Q5"]))
                            .group_by("utc_date", "quintile")
                            .agg(
                                pl.col("aligned_future_return_bps")
                                .mean()
                                .alias("daily_mean")
                            )
                            .collect()
                            .pivot(index="utc_date", on="quintile", values="daily_mean")
                            .drop_nulls(["Q1", "Q5"])
                        )
                        differences = (daily["Q5"] - daily["Q1"]).to_list()
                        days, mean_difference, t_stat = mean_and_t(differences)
                        contrast_rows.append(
                            {
                                "construction": name,
                                "reference_4000_horizon": reference,
                                "nominal_time_minutes": NOMINAL_MINUTES[reference],
                                "horizon_bars": horizon,
                                "period": period,
                                "feature": feature,
                                "side": side_label,
                                "days": days,
                                "q5_minus_q1_aligned_return_bps": mean_difference,
                                "day_block_t_stat": t_stat,
                            }
                        )

    output_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "impact_alpha_v1_horizon_mapping.csv": (
            mapping_rows,
            ["construction", "reference_4000_horizon", "period"],
        ),
        "impact_alpha_v1_asymmetric_fits.csv": (
            fit_rows,
            ["construction", "side"],
        ),
        "impact_alpha_v1_quintile_summary.csv": (
            summary_rows,
            [
                "construction",
                "reference_4000_horizon",
                "period",
                "feature",
                "side",
                "quintile",
            ],
        ),
        "impact_alpha_v1_extreme_contrasts.csv": (
            contrast_rows,
            [
                "construction",
                "reference_4000_horizon",
                "period",
                "feature",
                "side",
            ],
        ),
    }
    for filename, (rows, sort_columns) in tables.items():
        pl.DataFrame(rows).sort(sort_columns).write_csv(output_root / filename)
        print(output_root / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="2020-09-01")
    parser.add_argument("--validation-end", default="2021-07-01")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=(
            REPO
            / "shared-data"
            / "features"
            / "alpha"
            / "symbol=BTCUSDT"
            / "impact_alpha_v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "alpha-search" / "reports" / "tables",
    )
    args = parser.parse_args()
    run(
        cache_root=args.cache_root,
        output_root=args.output_root,
        train_end=utc_ms(args.train_end),
        validation_end=utc_ms(args.validation_end),
    )


if __name__ == "__main__":
    main()
