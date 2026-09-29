"""Does the P&B residual add predictive information beyond its components?

The signed residual is X = R - P, where R is the bar return over trailing
volatility and P = s * pb_activity prediction. Because X is a fixed linear
combination of R and P, the useful questions are:

1. Is the 1:-1 restriction supported, or do R and P deserve separate weights?
2. Does the impact-model component P add out-of-sample information beyond the
   bar return, clock-time pretrend and a naive linear flow term?
3. Does any resulting forecast clear realistic trading costs in its tails?

Every forecast regression is fitted on the preceding 12 months and frozen for
the test month. June--August 2026 are excluded, including as target prices.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
from scipy.stats import rankdata


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "src"))
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

from alpha_search.pb_scaling import predict_pb_scaling  # noqa: E402
from alpha_search.research import add_months  # noqa: E402
from run_rolling_pb_residual_ic import fit_from_row  # noqa: E402


DAY_MS = 86_400_000
FEATURES = ["R", "P", "X", "q", "pre15", "pre60"]


def month_ms(month: str) -> int:
    return int(
        datetime.strptime(f"{month}-01", "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1_000
    )


def month_range(start: str, end: str) -> list[str]:
    values = []
    current = start
    while current <= end:
        values.append(current)
        current = add_months(current, 1)
    return values


def load_price_clock(cache_root: Path, end_ms: int) -> tuple[np.ndarray, np.ndarray]:
    clock = (
        pl.scan_parquet(cache_root / "event_200" / "part-000.parquet")
        .select("end_time_ms", "end_price")
        .filter(pl.col("end_time_ms") < end_ms)
        .sort("end_time_ms")
        .collect()
    )
    return clock["end_time_ms"].to_numpy(), clock["end_price"].to_numpy()


def price_at_or_after(
    clock_time: np.ndarray, clock_price: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    index = np.searchsorted(clock_time, times, side="left")
    valid = index < len(clock_time)
    price = np.full(len(times), np.nan)
    when = np.full(len(times), -1, dtype=np.int64)
    price[valid] = clock_price[index[valid]]
    when[valid] = clock_time[index[valid]]
    return price, when


def price_at_or_before(
    clock_time: np.ndarray, clock_price: np.ndarray, times: np.ndarray
) -> np.ndarray:
    index = np.searchsorted(clock_time, times, side="right") - 1
    price = np.full(len(times), np.nan)
    valid = index >= 0
    price[valid] = clock_price[index[valid]]
    return price


def build_features(
    n_events: int,
    fits: pl.DataFrame,
    config: dict,
    cache_root: Path,
    clock_time: np.ndarray,
    clock_price: np.ndarray,
) -> pl.DataFrame:
    """Signed, decision-time features and immediate/delayed future returns."""

    start_ms = month_ms(config["first_feature_month"])
    end_ms = month_ms(config["data_end_exclusive"][:7])
    frame = (
        pl.scan_parquet(cache_root / f"event_{n_events}" / "part-000.parquet")
        .select(
            "start_time_ms",
            "end_time_ms",
            "start_price",
            "end_price",
            "signed_imbalance",
            "current_return_bps",
            "trailing_realised_volatility_bps",
            "normalised_imbalance",
            "relative_imbalance",
            "relative_activity",
        )
        .filter(
            (pl.col("end_time_ms") >= start_ms) & (pl.col("end_time_ms") < end_ms)
        )
        .sort("end_time_ms")
        .collect()
    )
    time = frame["end_time_ms"].to_numpy()
    start_time = frame["start_time_ms"].to_numpy()
    start_price = frame["start_price"].to_numpy()
    end_price = frame["end_price"].to_numpy()
    side = np.sign(frame["signed_imbalance"].to_numpy()).astype(np.int8)
    sigma = frame["trailing_realised_volatility_bps"].to_numpy()
    z = frame["relative_imbalance"].to_numpy()
    activity = frame["relative_activity"].to_numpy()

    bar_return = frame["current_return_bps"].to_numpy() / sigma
    predicted = np.full(len(time), np.nan)
    month_of_row = np.full(len(time), "", dtype=object)
    usable = (
        np.isfinite(z) & np.isfinite(activity) & (z > 0) & (activity > 0) & (side != 0)
    )
    for month in month_range(config["first_feature_month"], config["last_test_month"]):
        in_month = (time >= month_ms(month)) & (time < month_ms(add_months(month, 1)))
        month_of_row[in_month] = month
        fit_rows = fits.filter(pl.col("calibration_date") == f"{month}-01")
        if fit_rows.height != 1:
            raise ValueError(f"Expected one impact fit for {month}.")
        selected = in_month & usable
        if not selected.any():
            continue
        fit = fit_from_row(fit_rows.row(0, named=True))
        predicted[selected] = side[selected] * predict_pb_scaling(
            fit,
            np.full(selected.sum(), n_events),
            side[selected],
            z[selected],
            activity[selected],
        )

    columns = {
        "month": month_of_row.astype(str),
        "day": time // DAY_MS,
        "time_ms": time,
        "side": side,
        "R": bar_return,
        "P": predicted,
        "X": bar_return - predicted,
        "q": side * frame["normalised_imbalance"].to_numpy(),
    }
    for minutes in config["pretrend_lookbacks_minutes"]:
        past = price_at_or_before(
            clock_time, clock_price, start_time - int(minutes) * 60_000
        )
        columns[f"pre{minutes}"] = 10_000.0 * np.log(start_price / past) / sigma

    delay = int(config["entry_delay_ms"])
    entry_price, entry_time = price_at_or_after(clock_time, clock_price, time + delay)
    columns["entry_delay_seconds"] = np.where(
        entry_time >= 0, (entry_time - time) / 1_000.0, np.nan
    )
    for minutes in config["future_horizons_minutes"]:
        horizon = int(minutes) * 60_000
        exit_now, _ = price_at_or_after(clock_time, clock_price, time + horizon)
        columns[f"F{minutes}"] = 10_000.0 * np.log(exit_now / end_price)
        exit_late, _ = price_at_or_after(
            clock_time, clock_price, np.where(entry_time >= 0, entry_time, 0) + horizon
        )
        delayed = 10_000.0 * np.log(exit_late / entry_price)
        delayed[entry_time < 0] = np.nan
        columns[f"D{minutes}"] = delayed

    output = pl.DataFrame(columns)
    required = FEATURES + [f"F{m}" for m in config["future_horizons_minutes"]]
    return output.filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in required])
        & (pl.col("month") != "")
    )


def ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coefficients


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 20 or np.std(x) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def fama_macbeth(data: pl.DataFrame, config: dict, n_events: int) -> list[dict]:
    """Within-month OLS in raw sigma units; used for signs and the 1:-1 test."""

    low, high = config["winsor_quantiles"]
    rows = []
    for (month,), frame in data.group_by(["month"], maintain_order=True):
        for minutes in config["future_horizons_minutes"]:
            y = frame[f"F{minutes}"].to_numpy()
            y = np.clip(y, *np.quantile(y, [low, high]))
            for model in ("components", "full_impact_flow"):
                names = config["models"][model]
                x = np.column_stack(
                    [np.ones(frame.height)] + [frame[c].to_numpy() for c in names]
                )
                beta = ols(x, y)
                rows.append(
                    {
                        "n_events": n_events,
                        "month": month,
                        "horizon_minutes": int(minutes),
                        "model": model,
                        "observations": frame.height,
                        **{f"b_{c}": float(b) for c, b in zip(names, beta[1:])},
                    }
                )
    return rows


def evaluate_out_of_sample(
    data: pl.DataFrame, config: dict, n_events: int
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    low, high = config["winsor_quantiles"]
    window = int(config["regression_training_months"])
    months = month_range(config["first_test_month"], config["last_test_month"])
    monthly_rows: list[dict] = []
    tail_rows: list[dict] = []
    tail_month_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    for month in months:
        train_months = month_range(add_months(month, -window), add_months(month, -1))
        # Embargo: no training target may end inside the test month.
        embargo_ms = (max(config["future_horizons_minutes"]) + 5) * 60_000
        train = data.filter(
            pl.col("month").is_in(train_months)
            & (pl.col("time_ms") < month_ms(month) - embargo_ms)
        )
        test = data.filter(pl.col("month") == month)
        if train.height < 1_000 or test.height < 20:
            continue
        test_day = test["day"].to_numpy()
        # Feature winsorisation bounds come from the training window only.
        feature_train = train.select(FEATURES).to_numpy()
        bounds = np.quantile(feature_train, [low, high], axis=0)
        clipped_train = dict(
            zip(FEATURES, np.clip(feature_train, bounds[0], bounds[1]).T)
        )
        clipped_test = dict(
            zip(
                FEATURES,
                np.clip(test.select(FEATURES).to_numpy(), bounds[0], bounds[1]).T,
            )
        )
        for minutes in config["future_horizons_minutes"]:
            y_train = train[f"F{minutes}"].to_numpy()
            y_low, y_high = np.quantile(y_train, [low, high])
            y_train = np.clip(y_train, y_low, y_high)
            y_test = test[f"F{minutes}"].to_numpy()
            y_test_w = np.clip(y_test, y_low, y_high)
            delayed = test[f"D{minutes}"].to_numpy()
            for model, names in config["models"].items():
                x_train = np.column_stack([clipped_train[c] for c in names])
                x_test = np.column_stack([clipped_test[c] for c in names])
                centre = x_train.mean(axis=0)
                beta = ols(x_train - centre, y_train - y_train.mean())
                fitted = (x_train - centre) @ beta
                forecast = (x_test - centre) @ beta
                coefficient_rows.append(
                    {
                        "n_events": n_events,
                        "test_month": month,
                        "horizon_minutes": int(minutes),
                        "model": model,
                        **{f"b_{c}": float(b) for c, b in zip(names, beta)},
                    }
                )
                monthly_rows.append(
                    {
                        "n_events": n_events,
                        "test_month": month,
                        "horizon_minutes": int(minutes),
                        "model": model,
                        "observations": len(forecast),
                        "spearman_ic": spearman(forecast, y_test),
                        "pearson_ic_winsorised": float(
                            np.corrcoef(forecast, y_test_w)[0, 1]
                        ),
                        "oos_r2_winsorised": float(
                            1.0
                            - np.sum((y_test_w - forecast) ** 2)
                            / np.sum(y_test_w**2)
                        ),
                    }
                )
                magnitude = np.abs(fitted)
                for fraction in config["tail_fractions"]:
                    threshold = float(np.quantile(magnitude, 1.0 - fraction))
                    chosen = (np.abs(forecast) >= threshold) & np.isfinite(delayed)
                    if not chosen.any():
                        continue
                    direction = np.sign(forecast[chosen])
                    edge_delayed = direction * delayed[chosen]
                    edge_now = direction * y_test[chosen]
                    days, inverse = np.unique(test_day[chosen], return_inverse=True)
                    daily_sum = np.bincount(inverse, weights=edge_delayed)
                    daily_count = np.bincount(inverse)
                    for day, total, count in zip(days, daily_sum, daily_count):
                        tail_rows.append(
                            {
                                "n_events": n_events,
                                "test_month": month,
                                "horizon_minutes": int(minutes),
                                "model": model,
                                "tail_fraction": float(fraction),
                                "day": int(day),
                                "trades": int(count),
                                "sum_edge_delayed_bps": float(total),
                            }
                        )
                    tail_month_rows.append(
                        {
                            "n_events": n_events,
                            "test_month": month,
                            "horizon_minutes": int(minutes),
                            "model": model,
                            "tail_fraction": float(fraction),
                            "trades": int(chosen.sum()),
                            "sum_edge_immediate_bps": float(edge_now.sum()),
                            "sum_edge_delayed_bps": float(edge_delayed.sum()),
                        }
                    )
    return monthly_rows, tail_rows, tail_month_rows, coefficient_rows


def univariate_monthly(data: pl.DataFrame, config: dict, n_events: int) -> list[dict]:
    rows = []
    for (month,), frame in data.group_by(["month"], maintain_order=True):
        for minutes in config["future_horizons_minutes"]:
            y = frame[f"F{minutes}"].to_numpy()
            for feature in FEATURES:
                rows.append(
                    {
                        "n_events": n_events,
                        "month": month,
                        "horizon_minutes": int(minutes),
                        "feature": feature,
                        "spearman_ic": spearman(frame[feature].to_numpy(), y),
                    }
                )
    return rows


def t_stat(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 3 or values.std(ddof=1) == 0:
        return float("nan")
    return float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))


def summarise(
    fm: pl.DataFrame,
    monthly: pl.DataFrame,
    tails: pl.DataFrame,
    tail_months: pl.DataFrame,
    univariate: pl.DataFrame,
    config: dict,
) -> dict[str, pl.DataFrame]:
    keys = ["n_events", "horizon_minutes", "model"]
    fm_rows = []
    for keys_value, frame in fm.group_by(keys, maintain_order=True):
        row = dict(zip(keys, keys_value))
        row["months"] = frame.height
        for column in [c for c in frame.columns if c.startswith("b_")]:
            values = frame[column].to_numpy()
            if np.isnan(values).all():
                continue
            row[f"{column}_mean"] = float(np.nanmean(values))
            row[f"{column}_t"] = t_stat(values)
        if keys_value[2] == "components":
            restriction = frame["b_R"].to_numpy() + frame["b_P"].to_numpy()
            row["b_R_plus_b_P_mean"] = float(restriction.mean())
            row["b_R_plus_b_P_t"] = t_stat(restriction)
        fm_rows.append(row)

    oos_rows = []
    for keys_value, frame in monthly.group_by(keys, maintain_order=True):
        ic = frame["spearman_ic"].to_numpy()
        oos_rows.append(
            {
                **dict(zip(keys, keys_value)),
                "months": frame.height,
                "mean_spearman_ic": float(np.nanmean(ic)),
                "spearman_ic_t": t_stat(ic),
                "positive_month_fraction": float(np.mean(ic > 0)),
                "mean_pearson_ic_winsorised": float(
                    frame["pearson_ic_winsorised"].mean()
                ),
                "mean_oos_r2_winsorised": float(frame["oos_r2_winsorised"].mean()),
            }
        )
    oos = pl.DataFrame(oos_rows).sort(keys)

    comparisons = [
        ("residual", "bar_return", "X versus R alone"),
        ("components", "bar_return", "adding P to R"),
        ("components", "residual", "free R,P weights versus X"),
        ("full_impact", "return_pretrend", "adding P to R + pretrend"),
        ("full_impact_flow", "return_pretrend_flow", "adding P to R + pretrend + q"),
        ("return_pretrend", "pretrend_only", "adding R to pretrend"),
    ]
    increment_rows = []
    for (n_events, minutes), frame in monthly.group_by(
        ["n_events", "horizon_minutes"], maintain_order=True
    ):
        pivot = frame.pivot(on="model", index="test_month", values="spearman_ic")
        for candidate, baseline, label in comparisons:
            difference = (pivot[candidate] - pivot[baseline]).to_numpy()
            increment_rows.append(
                {
                    "n_events": n_events,
                    "horizon_minutes": minutes,
                    "comparison": label,
                    "candidate": candidate,
                    "baseline": baseline,
                    "mean_ic_difference": float(np.nanmean(difference)),
                    "difference_t": t_stat(difference),
                    "candidate_better_month_fraction": float(np.mean(difference > 0)),
                }
            )

    taker = 2.0 * float(config["costs_bps_per_side"]["taker"])
    maker = 2.0 * float(config["costs_bps_per_side"]["maker"])
    tail_keys = keys + ["tail_fraction"]
    tail_summary = []
    month_groups = {
        keys_value: frame
        for keys_value, frame in tail_months.group_by(tail_keys, maintain_order=True)
    }
    for keys_value, frame in tails.group_by(tail_keys, maintain_order=True):
        trades = frame["trades"].to_numpy()
        sums = frame["sum_edge_delayed_bps"].to_numpy()
        daily_mean = sums / trades
        edge = float(sums.sum() / trades.sum())
        by_month = month_groups[keys_value]
        immediate = float(
            by_month["sum_edge_immediate_bps"].sum() / by_month["trades"].sum()
        )
        monthly_edge = (
            by_month["sum_edge_delayed_bps"] / by_month["trades"]
        ).to_numpy()
        # Overlapping tail trades cluster on stress days; measure that directly.
        order = np.argsort(sums)[::-1]
        top_days = order[:10]
        rest_trades = trades.sum() - trades[top_days].sum()
        yearly = (
            by_month.with_columns(pl.col("test_month").str.slice(0, 4).alias("year"))
            .group_by("year")
            .agg(pl.col("sum_edge_delayed_bps").sum() / pl.col("trades").sum())
            .sort("year")
        )
        tail_summary.append(
            {
                **dict(zip(tail_keys, keys_value)),
                "trades": int(trades.sum()),
                "trades_per_day": float(trades.sum() / frame["day"].n_unique()),
                "mean_edge_immediate_bps": float(immediate),
                "mean_edge_delayed_bps": edge,
                "daily_block_t": t_stat(daily_mean),
                "positive_month_fraction": float(np.mean(monthly_edge > 0)),
                "net_taker_bps": edge - taker,
                "net_maker_bps": edge - maker,
                "trading_days": len(trades),
                "best_10_day_edge_share": float(sums[top_days].sum() / sums.sum())
                if sums.sum() != 0
                else float("nan"),
                "edge_excluding_best_10_days_bps": float(
                    (sums.sum() - sums[top_days].sum()) / rest_trades
                )
                if rest_trades > 0
                else float("nan"),
                **{
                    f"edge_{year}_bps": float(value)
                    for year, value in yearly.iter_rows()
                },
            }
        )

    univariate_summary = []
    for keys_value, frame in univariate.group_by(
        ["n_events", "horizon_minutes", "feature"], maintain_order=True
    ):
        for period, chosen in [
            ("2022-01..2022-12", frame.filter(pl.col("month") < "2023-01")),
            ("2023-01..2026-05", frame.filter(pl.col("month") >= "2023-01")),
        ]:
            ic = chosen["spearman_ic"].to_numpy()
            univariate_summary.append(
                {
                    "n_events": keys_value[0],
                    "horizon_minutes": keys_value[1],
                    "feature": keys_value[2],
                    "period": period,
                    "months": len(ic),
                    "mean_spearman_ic": float(np.nanmean(ic)),
                    "spearman_ic_t": t_stat(ic),
                    "negative_month_fraction": float(np.mean(ic < 0)),
                }
            )

    return {
        "fama_macbeth_summary": pl.DataFrame(fm_rows).sort(keys),
        "oos_summary": oos,
        "oos_increments": pl.DataFrame(increment_rows).sort(
            ["n_events", "horizon_minutes", "comparison"]
        ),
        "tail_breakeven": pl.DataFrame(
            tail_summary, infer_schema_length=None
        ).sort(tail_keys),
        "univariate_summary": pl.DataFrame(univariate_summary).sort(
            ["n_events", "horizon_minutes", "feature", "period"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "alpha-search" / "configs" / "residual_decomposition_v1.json",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--event-sizes", type=int, nargs="*")
    parser.add_argument(
        "--resummarise",
        action="store_true",
        help="Rebuild summary tables from saved detailed outputs without re-running.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cache_root = args.cache_root or (
        REPO
        / "shared-data"
        / "features"
        / "alpha"
        / f"symbol={config['symbol']}"
        / config["cache_version"]
    )
    output = args.output_dir or (
        REPO / "alpha-search" / "reports" / config["experiment_id"]
    )
    output.mkdir(parents=True, exist_ok=True)

    if args.resummarise:
        summaries = summarise(
            pl.read_csv(output / "fama_macbeth_monthly.csv", infer_schema_length=None),
            pl.read_csv(output / "oos_monthly.csv"),
            pl.read_parquet(output / "tail_daily.parquet"),
            pl.read_csv(output / "tail_monthly.csv"),
            pl.read_csv(output / "univariate_monthly.csv"),
            config,
        )
        for name, frame in summaries.items():
            frame.write_csv(output / f"{name}.csv")
        print(output, flush=True)
        return

    fits = pl.read_csv(REPO / config["impact_fits"], infer_schema_length=None).filter(
        pl.col("training_window_months") == int(config["impact_training_window_months"])
    )
    end_ms = month_ms(config["data_end_exclusive"][:7])
    clock_time, clock_price = load_price_clock(cache_root, end_ms)

    fm_rows, monthly_rows, tail_rows, tail_month_rows = [], [], [], []
    coefficient_rows, univariate_rows = [], []
    delay_rows = []
    for n_events in args.event_sizes or config["event_sizes"]:
        print(f"Building features for event_{n_events}", flush=True)
        data = build_features(
            n_events, fits, config, cache_root, clock_time, clock_price
        )
        delays = data["entry_delay_seconds"].drop_nans().drop_nulls().to_numpy()
        delay_rows.append(
            {
                "n_events": n_events,
                "observations": data.height,
                "median_entry_delay_seconds": float(np.median(delays)),
                "p90_entry_delay_seconds": float(np.quantile(delays, 0.9)),
            }
        )
        fm_rows.extend(fama_macbeth(data, config, n_events))
        univariate_rows.extend(univariate_monthly(data, config, n_events))
        print(f"Out-of-sample evaluation for event_{n_events}", flush=True)
        monthly, tails, tail_months, coefficients = evaluate_out_of_sample(
            data, config, n_events
        )
        monthly_rows.extend(monthly)
        tail_rows.extend(tails)
        tail_month_rows.extend(tail_months)
        coefficient_rows.extend(coefficients)
        del data

    fm = pl.DataFrame(fm_rows, infer_schema_length=None)
    monthly = pl.DataFrame(monthly_rows)
    tails = pl.DataFrame(tail_rows)
    tail_months = pl.DataFrame(tail_month_rows)
    coefficients = pl.DataFrame(coefficient_rows, infer_schema_length=None)
    univariate = pl.DataFrame(univariate_rows)
    summaries = summarise(fm, monthly, tails, tail_months, univariate, config)

    fm.write_csv(output / "fama_macbeth_monthly.csv")
    monthly.write_csv(output / "oos_monthly.csv")
    coefficients.write_csv(output / "oos_coefficients.csv")
    univariate.write_csv(output / "univariate_monthly.csv")
    tails.write_parquet(output / "tail_daily.parquet", compression="zstd")
    tail_months.write_csv(output / "tail_monthly.csv")
    pl.DataFrame(delay_rows).write_csv(output / "entry_delay.csv")
    for name, frame in summaries.items():
        frame.write_csv(output / f"{name}.csv")
    manifest = {
        "experiment_id": config["experiment_id"],
        "config": str(args.config.resolve()),
        "cache_root": str(cache_root.resolve()),
        "event_sizes": args.event_sizes or config["event_sizes"],
        "holdout_months_excluded": config["holdout_months_excluded"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
