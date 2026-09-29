"""Chronological univariate screen for the preregistered short-bar features."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))

from alpha_search.impact_model import (  # noqa: E402
    add_asymmetric_impact_features,
    fit_asymmetric_log_impact,
)
from alpha_search.screen_features import screen_feature_catalog  # noqa: E402


HORIZONS = {200: (60, 80), 500: (24, 32), 1_000: (12, 16)}
EVENT_COUNTS = {200: {60: 12_000, 80: 16_000},
                500: {24: 12_000, 32: 16_000},
                1_000: {12: 12_000, 16: 16_000}}
PERIODS = ("validation", "later_sandbox")


def utc_ms(value: str) -> int:
    return int(
        datetime.strptime(value, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def normal_p_value(t_stat: float) -> float:
    if not math.isfinite(t_stat):
        return math.nan
    return math.erfc(abs(t_stat) / math.sqrt(2.0))


def grouped_mean(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(keys, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    return unique, sums / counts


def paired_block_differences(
    block: np.ndarray,
    quintile: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    one = quintile == 1
    five = quintile == 5
    block_one, mean_one = grouped_mean(block[one], target[one])
    block_five, mean_five = grouped_mean(block[five], target[five])
    common, i_one, i_five = np.intersect1d(
        block_one, block_five, assume_unique=True, return_indices=True
    )
    if len(common) == 0:
        return np.array([], dtype=float)
    return mean_five[i_five] - mean_one[i_one]


def mean_t(values: np.ndarray) -> tuple[int, float, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return n, math.nan, math.nan
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    t_stat = mean / (std / math.sqrt(n)) if std > 0 else math.nan
    return n, mean, t_stat


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return q.tolist()
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return q.tolist()


def screen_one_size(
    *,
    size: int,
    path: Path,
    train_end: int,
    validation_end: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    print(f"Loading and fitting event_{size}", flush=True)
    base = pl.scan_parquet(path)
    first_ms = int(base.select(pl.col("end_time_ms").min()).collect().item())
    full_history_start = first_ms + 86_400_000
    usable = base.filter(pl.col("end_time_ms") >= full_history_start)
    training = usable.filter(pl.col("end_time_ms") < train_end)
    fit = fit_asymmetric_log_impact(training)
    enriched = add_asymmetric_impact_features(usable, fit)

    catalog = screen_feature_catalog()
    features = [name for name, _ in catalog]
    target_columns = [f"future_return_{h}_bps" for h in HORIZONS[size]]
    elapsed_columns = [f"future_elapsed_{h}_minutes" for h in HORIZONS[size]]
    frame = enriched.select(
        "end_time_ms",
        "signed_imbalance",
        *features,
        *target_columns,
        *elapsed_columns,
    ).collect().rechunk()

    end_ms = frame["end_time_ms"].to_numpy()
    side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
    day = end_ms // 86_400_000
    month = np.array(
        [datetime.fromtimestamp(v / 1_000, tz=timezone.utc).year * 12
         + datetime.fromtimestamp(v / 1_000, tz=timezone.utc).month
         for v in end_ms],
        dtype=np.int32,
    )
    period_masks = {
        "train": end_ms < train_end,
        "validation": (end_ms >= train_end) & (end_ms < validation_end),
        "later_sandbox": end_ms >= validation_end,
    }

    fit_rows = []
    for label, side_fit in (("buy", fit.buy), ("sell", fit.sell)):
        fit_rows.append(
            {
                "construction": f"event_{size}",
                "side": label,
                "observations": side_fit.observations,
                "imbalance_scale": fit.imbalance_scale,
                "intercept": side_fit.intercept,
                "slope": side_fit.slope,
                "r_squared": side_fit.r_squared,
            }
        )

    mapping_rows = []
    for horizon, elapsed_col in zip(HORIZONS[size], elapsed_columns):
        elapsed = frame[elapsed_col].to_numpy()
        for period, mask in period_masks.items():
            values = elapsed[mask & np.isfinite(elapsed)]
            mapping_rows.append(
                {
                    "construction": f"event_{size}",
                    "events_per_bar": size,
                    "horizon_bars": horizon,
                    "future_event_count": EVENT_COUNTS[size][horizon],
                    "period": period,
                    "p10_minutes": float(np.quantile(values, 0.1)),
                    "median_minutes": float(np.median(values)),
                    "p90_minutes": float(np.quantile(values, 0.9)),
                }
            )

    summary_rows: list[dict] = []
    quintile_rows: list[dict] = []
    family_lookup = dict(catalog)
    for horizon, target_col in zip(HORIZONS[size], target_columns):
        raw_target = frame[target_col].to_numpy()
        aligned_target = side * raw_target
        for feature_name in features:
            values = frame[feature_name].to_numpy()
            for side_value, side_label in ((-1, "sell"), (1, "buy")):
                train_mask = (
                    period_masks["train"]
                    & (side == side_value)
                    & np.isfinite(values)
                )
                if train_mask.sum() < 100:
                    continue
                cuts = np.quantile(values[train_mask], (0.2, 0.4, 0.6, 0.8))
                for period in PERIODS:
                    mask = (
                        period_masks[period]
                        & (side == side_value)
                        & np.isfinite(values)
                        & np.isfinite(aligned_target)
                    )
                    if mask.sum() < 100:
                        continue
                    q = np.digitize(values[mask], cuts, right=True) + 1
                    target = aligned_target[mask]
                    q_means = []
                    for q_value in range(1, 6):
                        chosen = q == q_value
                        observations = int(chosen.sum())
                        mean_return = float(target[chosen].mean()) if observations else math.nan
                        q_means.append(mean_return)
                        quintile_rows.append(
                            {
                                "construction": f"event_{size}",
                                "horizon_bars": horizon,
                                "future_event_count": EVENT_COUNTS[size][horizon],
                                "period": period,
                                "feature": feature_name,
                                "feature_family": family_lookup[feature_name],
                                "side": side_label,
                                "quintile": q_value,
                                "observations": observations,
                                "mean_feature": (
                                    float(values[mask][chosen].mean()) if observations else math.nan
                                ),
                                "mean_aligned_return_bps": mean_return,
                            }
                        )

                    daily = paired_block_differences(day[mask], q, target)
                    months = paired_block_differences(month[mask], q, target)
                    days, spread, t_stat = mean_t(daily)
                    month_sign_share = (
                        float(np.mean(np.sign(months) == np.sign(spread)))
                        if len(months) and spread != 0
                        else math.nan
                    )
                    quintile_slope = float(
                        np.polyfit(np.arange(1, 6), np.asarray(q_means), 1)[0]
                    )
                    summary_rows.append(
                        {
                            "construction": f"event_{size}",
                            "events_per_bar": size,
                            "horizon_bars": horizon,
                            "future_event_count": EVENT_COUNTS[size][horizon],
                            "period": period,
                            "feature": feature_name,
                            "feature_family": family_lookup[feature_name],
                            "side": side_label,
                            "observations": int(mask.sum()),
                            "paired_days": days,
                            "q5_minus_q1_aligned_return_bps": spread,
                            "day_block_t_stat": t_stat,
                            "normal_approx_p_value": normal_p_value(t_stat),
                            "quintile_mean_slope_bps": quintile_slope,
                            "monthly_same_sign_share": month_sign_share,
                        }
                    )
    return summary_rows, quintile_rows, mapping_rows, fit_rows


def candidate_ranking(summary: pl.DataFrame) -> pl.DataFrame:
    keys = [
        "construction",
        "events_per_bar",
        "horizon_bars",
        "future_event_count",
        "feature",
        "feature_family",
        "side",
    ]
    validation = summary.filter(pl.col("period") == "validation").select(
        *keys,
        pl.col("q5_minus_q1_aligned_return_bps").alias("validation_spread_bps"),
        pl.col("day_block_t_stat").alias("validation_day_t"),
        pl.col("normal_approx_p_value").alias("validation_p"),
        pl.col("fdr_q_value").alias("validation_fdr_q"),
        pl.col("monthly_same_sign_share").alias("validation_month_sign_share"),
    )
    later = summary.filter(pl.col("period") == "later_sandbox").select(
        *keys,
        pl.col("q5_minus_q1_aligned_return_bps").alias("later_spread_bps"),
        pl.col("day_block_t_stat").alias("later_day_t"),
        pl.col("monthly_same_sign_share").alias("later_month_sign_share"),
    )
    return (
        validation.join(later, on=keys, how="inner")
        .with_columns(
            (
                pl.col("validation_spread_bps").sign()
                == pl.col("later_spread_bps").sign()
            ).alias("same_direction"),
            pl.min_horizontal(
                pl.col("validation_spread_bps").abs(),
                pl.col("later_spread_bps").abs(),
            ).alias("minimum_absolute_spread_bps"),
        )
        .with_columns(
            (
                pl.col("same_direction")
                & (pl.col("validation_fdr_q") <= 0.10)
                & (pl.col("validation_day_t").abs() >= 2.0)
                & (pl.col("later_day_t").abs() >= 1.5)
                & (pl.col("minimum_absolute_spread_bps") >= 1.0)
                & (pl.col("validation_month_sign_share") >= 0.60)
                & (pl.col("later_month_sign_share") >= 0.60)
            ).alias("stability_qualified")
        )
        .sort(
            ["stability_qualified", "minimum_absolute_spread_bps"],
            descending=[True, True],
        )
    )


def cross_scale_ranking(ranking: pl.DataFrame) -> pl.DataFrame:
    """Collapse correlated bar-size cells into cross-scale hypotheses."""

    return (
        ranking.filter(
            pl.col("validation_spread_bps").is_finite()
            & pl.col("later_spread_bps").is_finite()
        )
        .group_by("feature", "feature_family", "side", "future_event_count")
        .agg(
            pl.len().alias("cells"),
            pl.col("construction").n_unique().alias("bar_sizes"),
            pl.col("same_direction").all().alias("each_bar_size_stable"),
            pl.col("validation_spread_bps").sign().n_unique().alias("validation_signs"),
            pl.col("later_spread_bps").sign().n_unique().alias("later_signs"),
            pl.col("validation_spread_bps").median().alias("validation_median_spread_bps"),
            pl.col("later_spread_bps").median().alias("later_median_spread_bps"),
            pl.col("minimum_absolute_spread_bps").min().alias("worst_cell_absolute_spread_bps"),
            pl.col("validation_day_t").abs().min().alias("minimum_validation_abs_day_t"),
            pl.col("later_day_t").abs().min().alias("minimum_later_abs_day_t"),
            pl.col("validation_fdr_q").max().alias("maximum_validation_fdr_q"),
            pl.col("validation_month_sign_share").min().alias("minimum_validation_month_share"),
            pl.col("later_month_sign_share").min().alias("minimum_later_month_share"),
        )
        .with_columns(
            (
                (pl.col("bar_sizes") == 3)
                & pl.col("each_bar_size_stable")
                & (pl.col("validation_signs") == 1)
                & (pl.col("later_signs") == 1)
                & (pl.col("maximum_validation_fdr_q") <= 0.10)
                & (pl.col("worst_cell_absolute_spread_bps") >= 1.0)
                & (pl.col("minimum_validation_abs_day_t") >= 2.0)
                & (pl.col("minimum_later_abs_day_t") >= 1.5)
                & (pl.col("minimum_validation_month_share") >= 0.60)
                & (pl.col("minimum_later_month_share") >= 0.60)
            ).alias("cross_scale_qualified")
        )
        .sort(
            ["cross_scale_qualified", "worst_cell_absolute_spread_bps"],
            descending=[True, True],
        )
    )


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
            / "short_event_screen_v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "alpha-search" / "reports" / "short_event_screen_v1",
    )
    args = parser.parse_args()

    all_summary: list[dict] = []
    all_quintiles: list[dict] = []
    all_mapping: list[dict] = []
    all_fits: list[dict] = []
    for size in sorted(HORIZONS):
        path = args.cache_root / f"event_{size}" / "part-000.parquet"
        summary, quintiles, mapping, fits = screen_one_size(
            size=size,
            path=path,
            train_end=utc_ms(args.train_end),
            validation_end=utc_ms(args.validation_end),
        )
        all_summary.extend(summary)
        all_quintiles.extend(quintiles)
        all_mapping.extend(mapping)
        all_fits.extend(fits)

    summary = pl.DataFrame(all_summary)
    validation_indices = [
        i for i, value in enumerate(summary["period"].to_list())
        if value == "validation"
    ]
    q_values = benjamini_hochberg(
        [summary["normal_approx_p_value"][i] for i in validation_indices]
    )
    fdr = [math.nan] * summary.height
    for index, q_value in zip(validation_indices, q_values):
        fdr[index] = q_value
    summary = summary.with_columns(pl.Series("fdr_q_value", fdr))
    ranking = candidate_ranking(summary)
    cross_scale = cross_scale_ranking(ranking)

    args.output_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "feature_catalog.csv": pl.DataFrame(
            screen_feature_catalog(), schema=["feature", "feature_family"], orient="row"
        ),
        "impact_fits.csv": pl.DataFrame(all_fits),
        "horizon_mapping.csv": pl.DataFrame(all_mapping),
        "feature_summary.csv": summary,
        "quintile_summary.csv": pl.DataFrame(all_quintiles),
        "candidate_ranking.csv": ranking,
        "cross_scale_ranking.csv": cross_scale,
    }
    for filename, table in tables.items():
        path = args.output_root / filename
        table.write_csv(path)
        print(path, flush=True)


if __name__ == "__main__":
    main()
