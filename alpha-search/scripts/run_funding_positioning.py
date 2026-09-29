"""Funding, basis and positioning signals on an hourly decision clock.

Builds an hourly panel of decision-time features and forward perp returns
(price-only and funding-carry-inclusive), then applies the same frozen
out-of-sample protocol as the residual decomposition: 12-month rolling OLS,
horizon embargo, monthly test ICs, nested-model increments and costed tails
with stress-day concentration. A separate event study measures returns around
funding settlements.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.months import iter_months  # noqa: E402
from crypto_market_data.series import DATASETS  # noqa: E402
from run_residual_decomposition import (  # noqa: E402
    month_ms,
    month_range,
    ols,
    spearman,
    t_stat,
)
from run_stress_episodes import grid_path  # noqa: E402


HOUR = 3_600
DAY_MS = 86_400_000


# --------------------------------------------------------------------------- data


class MinuteSeries:
    """Dense, forward-filled 1-minute series; value[m] is known at (m + 1) * 60 s."""

    def __init__(self, minutes: np.ndarray, values: np.ndarray, first: int, last: int):
        self.first = first
        dense = np.full(last - first + 1, np.nan)
        keep = (minutes >= first) & (minutes <= last)
        dense[minutes[keep] - first] = values[keep]
        # Forward fill short gaps so a decision always sees the latest known value.
        index = np.where(np.isfinite(dense), np.arange(len(dense)), 0)
        np.maximum.accumulate(index, out=index)
        filled = dense[index]
        filled[: np.argmax(np.isfinite(dense))] = np.nan
        self.values = filled
        valid = np.isfinite(filled)
        self.cumulative = np.concatenate([[0.0], np.cumsum(np.where(valid, filled, 0.0))])
        self.counts = np.concatenate([[0], np.cumsum(valid)])

    def at(self, minute: np.ndarray) -> np.ndarray:
        index = np.asarray(minute) - self.first
        output = np.full(len(index), np.nan)
        ok = (index >= 0) & (index < len(self.values))
        output[ok] = self.values[index[ok]]
        return output

    def mean(self, last_minute: np.ndarray, length: int) -> np.ndarray:
        """Mean over minutes (last - length, last]."""

        upper = np.clip(np.asarray(last_minute) - self.first + 1, 0, len(self.values))
        lower = np.clip(upper - length, 0, len(self.values))
        count = self.counts[upper] - self.counts[lower]
        total = self.cumulative[upper] - self.cumulative[lower]
        return np.where(count >= 0.8 * length, total / np.maximum(count, 1), np.nan)


def perp_minutes(root: Path, config: dict) -> tuple[np.ndarray, np.ndarray]:
    minutes, closes = [], []
    for month in iter_months(config["first_month"], config["last_month"]):
        frame = (
            pl.scan_parquet(grid_path(root, config, month))
            .select("second", "close")
            .group_by((pl.col("second") // 60).alias("minute"))
            .agg(pl.col("close").sort_by("second").last())
            .collect()
        )
        minutes.append(frame["minute"].to_numpy())
        closes.append(frame["close"].to_numpy())
    minute = np.concatenate(minutes)
    order = np.argsort(minute)
    return minute[order], np.concatenate(closes)[order]


def kline_close(root: Path, dataset: str, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    table = pl.read_parquet(
        DATASETS[dataset].table(root, symbol), columns=["open_time_ms", "close"]
    )
    return table["open_time_ms"].to_numpy() // 60_000, table["close"].to_numpy()


def snapshot_index(times: np.ndarray, at: np.ndarray, max_age: int) -> np.ndarray:
    index = np.searchsorted(times, at, side="right") - 1
    fresh = (index >= 0) & (at - times[np.clip(index, 0, None)] <= max_age)
    return np.where(fresh, index, -1)


def take(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    output = np.full(len(index), np.nan)
    ok = index >= 0
    output[ok] = values[index[ok]]
    return output


def rolling_z(values: np.ndarray, window: int, minimum: int) -> np.ndarray:
    series = pl.Series(values)
    mean = series.rolling_mean(window, min_samples=minimum).to_numpy()
    std = series.rolling_std(window, min_samples=minimum).to_numpy()
    return (values - mean) / np.where(std > 0, std, np.nan)


def build_panel(root: Path, config: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    symbol = config["symbol"]
    start = int(datetime.fromisoformat(config["first_decision_day"]).replace(tzinfo=timezone.utc).timestamp())
    end = month_ms(config["data_end_exclusive"][:7]) // 1_000
    first_minute = month_ms(config["first_month"]) // 60_000
    last_minute = end // 60 - 1

    perp = MinuteSeries(*perp_minutes(root, config), first_minute, last_minute)
    spot = MinuteSeries(*kline_close(root, "spot_klines", symbol), first_minute, last_minute)
    premium_minutes, premium_values = kline_close(root, "premium_index", symbol)
    premium = MinuteSeries(premium_minutes, 1e4 * premium_values, first_minute, last_minute)
    basis_raw = 1e4 * np.log(perp.values / spot.values)
    basis = MinuteSeries(
        np.arange(first_minute, last_minute + 1), basis_raw, first_minute, last_minute
    )
    one_minute_return = np.concatenate([[np.nan], 1e4 * np.diff(np.log(perp.values))])
    squared = MinuteSeries(
        np.arange(first_minute, last_minute + 1), one_minute_return**2, first_minute, last_minute
    )

    t = np.arange(start, end, config["decision_interval_minutes"] * 60)
    m = t // 60 - 1  # latest completed minute at the decision time
    price = perp.at(m)

    def past_return(hours: int) -> np.ndarray:
        return 1e4 * np.log(price / perp.at(m - hours * 60))

    columns = {
        "time_s": t,
        "month": [datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m") for v in t],
        "day": t // 86_400,
        "ret_1h": past_return(1),
        "ret_4h": past_return(4),
        "ret_24h": past_return(24),
        "rv_24h": np.sqrt(squared.mean(m, 1_440) * 1_440),
        "premium_1h": premium.mean(m, 60),
        "premium_8h": premium.mean(m, 480),
        "basis_1h": basis.mean(m, 60),
    }
    columns["premium_chg"] = columns["premium_1h"] - premium.mean(m, 1_440)
    columns["basis_chg"] = columns["basis_1h"] - basis.mean(m, 1_440)

    funding = pl.read_parquet(DATASETS["funding"].table(root, symbol)).sort("time_ms")
    settle = funding["time_ms"].to_numpy() // 1_000
    rate = 1e4 * funding["funding_rate"].to_numpy()
    last = np.searchsorted(settle, t, side="right") - 1
    columns["funding_last"] = take(rate, last)
    columns["funding_z30d"] = take(rolling_z(rate, 90, 60), last)

    metrics = (
        pl.read_parquet(root / "canonical" / "metrics" / f"symbol={symbol}" / "metrics_5m.parquet")
        .filter(pl.col("sum_open_interest") > 0)
        .sort("time_ms")
    )
    snap = metrics["time_ms"].to_numpy() // 1_000
    lag = config["metrics_publication_lag_seconds"]
    age = config["metrics_max_age_seconds"]
    oi = metrics["sum_open_interest"].to_numpy()
    now = snapshot_index(snap, t - lag, age)
    hour_ago = snapshot_index(snap, t - lag - HOUR, age)
    day_ago = snapshot_index(snap, t - lag - 24 * HOUR, age)
    columns["oi_chg_1h"] = 100 * (take(oi, now) / take(oi, hour_ago) - 1)
    columns["oi_chg_24h"] = 100 * (take(oi, now) / take(oi, day_ago) - 1)
    columns["oi_x_ret_24h"] = columns["oi_chg_24h"] * columns["ret_24h"] / 100
    for name, column in [
        ("toptrader_ls_z30d", "count_toptrader_long_short_ratio"),
        ("global_ls_z30d", "count_long_short_ratio"),
    ]:
        values = np.log(metrics[column].fill_null(np.nan).to_numpy())
        columns[name] = take(rolling_z(values, 8_640, 2_000), now)
    taker = np.log(metrics["sum_taker_long_short_vol_ratio"].fill_null(np.nan).to_numpy())
    taker_valid = np.isfinite(taker)
    taker_cum = np.concatenate([[0.0], np.cumsum(np.where(taker_valid, taker, 0.0))])
    taker_count = np.concatenate([[0], np.cumsum(taker_valid)])
    upper = np.where(now >= 0, now + 1, 0)
    lower = np.searchsorted(snap, t - lag - HOUR, side="right")
    lower = np.minimum(lower, upper)
    count = taker_count[upper] - taker_count[lower]
    columns["taker_ls_1h"] = np.where(
        count >= 6, (taker_cum[upper] - taker_cum[lower]) / np.maximum(count, 1), np.nan
    )

    entry = t + config["entry_delay_minutes"] * 60
    entry_price = perp.at(entry // 60 - 1)
    for hours in config["horizons_hours"]:
        exit_time = entry + hours * HOUR
        exit_price = np.where(exit_time <= end, perp.at(exit_time // 60 - 1), np.nan)
        price_return = 1e4 * np.log(exit_price / entry_price)
        first = np.searchsorted(settle, entry, side="right")
        stop = np.searchsorted(settle, exit_time, side="right")
        rate_cumulative = np.concatenate([[0.0], np.cumsum(rate)])
        carry = rate_cumulative[stop] - rate_cumulative[first]
        columns[f"F{hours}h"] = price_return
        columns[f"carry{hours}h"] = np.where(np.isfinite(price_return), carry, np.nan)

    panel = pl.DataFrame(columns)

    # Settlement event study: the hour either side of each funding settlement.
    in_range = (settle >= start + 24 * HOUR) & (settle + HOUR < end)
    s = settle[in_range]
    ms = s // 60 - 1
    events = pl.DataFrame(
        {
            "settle_s": s,
            "month": [datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m") for v in s],
            "day": s // 86_400,
            "funding_rate_bps": rate[in_range],
            "premium_7h_before_bps": premium.mean(ms - 60, 420),
            "pre_hour_return_bps": 1e4 * np.log(perp.at(ms) / perp.at(ms - 60)),
            "post_hour_return_bps": 1e4 * np.log(perp.at(ms + 60) / perp.at(ms)),
        }
    )
    return panel, events


# --------------------------------------------------------------------- evaluation


def model_features(config: dict, model: str) -> list[str]:
    return [f for group in config["models"][model] for f in config["feature_groups"][group]]


def all_features(config: dict) -> list[str]:
    return [f for group in config["feature_groups"].values() for f in group]


def evaluate(panel: pl.DataFrame, config: dict) -> dict[str, pl.DataFrame]:
    low, high = config["winsor_quantiles"]
    features = all_features(config)
    window = int(config["regression_training_months"])
    imputed = config.get("impute_zero_when_missing", [])
    usable = panel.with_columns(
        [pl.when(pl.col(f).is_finite()).then(pl.col(f)).otherwise(0.0).alias(f) for f in imputed]
    ).filter(pl.all_horizontal([pl.col(f).is_finite() for f in features]))
    monthly, tails, tail_months, univariate = [], [], [], []

    for (month,), frame in usable.group_by(["month"], maintain_order=True):
        for hours in config["horizons_hours"]:
            y = frame[f"F{hours}h"].to_numpy()
            ok = np.isfinite(y)
            for feature in features:
                univariate.append(
                    {
                        "month": month,
                        "horizon_hours": hours,
                        "feature": feature,
                        "spearman_ic": spearman(frame[feature].to_numpy()[ok], y[ok]),
                    }
                )

    for month in month_range(config["first_test_month"], config["last_test_month"]):
        train_months = month_range(add_month(month, -window), add_month(month, -1))
        test = usable.filter(pl.col("month") == month)
        if test.is_empty():
            continue
        feature_train_all = usable.filter(pl.col("month").is_in(train_months))
        bounds = np.quantile(feature_train_all.select(features).to_numpy(), [low, high], axis=0)
        for hours in config["horizons_hours"]:
            embargo = month_ms(month) // 1_000 - (hours + 1) * HOUR
            train = feature_train_all.filter(
                (pl.col("time_s") < embargo) & pl.col(f"F{hours}h").is_finite()
            )
            test_h = test.filter(pl.col(f"F{hours}h").is_finite())
            if train.height < 1_000 or test_h.height < 20:
                continue
            x_train_all = np.clip(train.select(features).to_numpy(), bounds[0], bounds[1])
            x_test_all = np.clip(test_h.select(features).to_numpy(), bounds[0], bounds[1])
            y_train = train[f"F{hours}h"].to_numpy()
            y_low, y_high = np.quantile(y_train, [low, high])
            y_train = np.clip(y_train, y_low, y_high)
            y_test = test_h[f"F{hours}h"].to_numpy()
            carry = test_h[f"carry{hours}h"].to_numpy()
            days = test_h["day"].to_numpy()
            for model in config["models"]:
                columns = [features.index(f) for f in model_features(config, model)]
                x_train = x_train_all[:, columns]
                x_test = x_test_all[:, columns]
                centre = x_train.mean(axis=0)
                beta = ols(x_train - centre, y_train - y_train.mean())
                fitted = (x_train - centre) @ beta
                forecast = (x_test - centre) @ beta
                monthly.append(
                    {
                        "test_month": month,
                        "horizon_hours": hours,
                        "model": model,
                        "observations": len(forecast),
                        "spearman_ic": spearman(forecast, y_test),
                    }
                )
                for fraction in config["tail_fractions"]:
                    threshold = float(np.quantile(np.abs(fitted), 1.0 - fraction))
                    chosen = np.abs(forecast) >= threshold
                    if not chosen.any():
                        continue
                    direction = np.sign(forecast[chosen])
                    price_edge = direction * y_test[chosen]
                    carry_edge = price_edge - direction * carry[chosen]
                    unique, inverse = np.unique(days[chosen], return_inverse=True)
                    for day, price_sum, carry_sum, count in zip(
                        unique,
                        np.bincount(inverse, weights=price_edge),
                        np.bincount(inverse, weights=carry_edge),
                        np.bincount(inverse),
                    ):
                        tails.append(
                            {
                                "test_month": month,
                                "horizon_hours": hours,
                                "model": model,
                                "tail_fraction": fraction,
                                "day": int(day),
                                "trades": int(count),
                                "sum_price_edge_bps": float(price_sum),
                                "sum_carry_edge_bps": float(carry_sum),
                            }
                        )
    return {
        "univariate_monthly": pl.DataFrame(univariate),
        "oos_monthly": pl.DataFrame(monthly),
        "tail_daily": pl.DataFrame(tails),
    }


def add_month(month: str, offset: int) -> str:
    year, number = map(int, month.split("-"))
    index = year * 12 + number - 1 + offset
    return f"{index // 12}-{index % 12 + 1:02d}"


def summarise(results: dict[str, pl.DataFrame], config: dict) -> dict[str, pl.DataFrame]:
    univariate = results["univariate_monthly"]
    rows = []
    for (hours, feature), frame in univariate.group_by(["horizon_hours", "feature"]):
        for period, chosen in [
            ("2021-02..2022-12", frame.filter(pl.col("month") < config["first_test_month"])),
            ("2023-01..2026-05", frame.filter(pl.col("month") >= config["first_test_month"])),
        ]:
            ic = chosen["spearman_ic"].to_numpy()
            rows.append(
                {
                    "horizon_hours": hours,
                    "feature": feature,
                    "period": period,
                    "months": len(ic),
                    "mean_spearman_ic": float(np.nanmean(ic)),
                    "spearman_ic_t": t_stat(ic),
                    "positive_month_fraction": float(np.mean(ic > 0)),
                }
            )
    univariate_summary = pl.DataFrame(rows).sort("horizon_hours", "feature", "period")

    monthly = results["oos_monthly"]
    oos_rows = []
    for (hours, model), frame in monthly.group_by(["horizon_hours", "model"]):
        ic = frame["spearman_ic"].to_numpy()
        oos_rows.append(
            {
                "horizon_hours": hours,
                "model": model,
                "months": len(ic),
                "mean_spearman_ic": float(np.nanmean(ic)),
                "spearman_ic_t": t_stat(ic),
                "positive_month_fraction": float(np.mean(ic > 0)),
            }
        )
    oos = pl.DataFrame(oos_rows).sort(
        "horizon_hours", "mean_spearman_ic", descending=[False, True]
    )
    increments = []
    for (hours,), frame in monthly.group_by(["horizon_hours"]):
        pivot = frame.pivot(on="model", index="test_month", values="spearman_ic")
        for candidate in ["baseline_funding", "baseline_basis", "baseline_positioning", "all"]:
            difference = (pivot[candidate] - pivot["baseline"]).to_numpy()
            increments.append(
                {
                    "horizon_hours": hours,
                    "candidate": candidate,
                    "mean_ic_gain_over_baseline": float(np.nanmean(difference)),
                    "gain_t": t_stat(difference),
                    "better_month_fraction": float(np.mean(difference > 0)),
                }
            )

    taker = 2.0 * config["costs_bps_per_side"]["taker"]
    tail_rows = []
    daily = results["tail_daily"]
    for keys, frame in daily.group_by(["horizon_hours", "model", "tail_fraction"]):
        trades = frame["trades"].to_numpy()
        price = frame["sum_price_edge_bps"].to_numpy()
        carry = frame["sum_carry_edge_bps"].to_numpy()
        top = np.argsort(carry)[::-1][:10]
        rest = trades.sum() - trades[top].sum()
        years = (
            frame.with_columns(pl.col("test_month").str.slice(0, 4).alias("year"))
            .group_by("year")
            .agg(pl.col("sum_carry_edge_bps").sum() / pl.col("trades").sum())
            .sort("year")
        )
        edge = float(carry.sum() / trades.sum())
        tail_rows.append(
            {
                "horizon_hours": keys[0],
                "model": keys[1],
                "tail_fraction": keys[2],
                "trades": int(trades.sum()),
                "trades_per_day": float(trades.sum() / frame["day"].n_unique()),
                "mean_price_edge_bps": float(price.sum() / trades.sum()),
                "mean_carry_edge_bps": edge,
                "daily_block_t": t_stat(carry / trades),
                "net_taker_bps": edge - taker,
                "best_10_day_share": float(carry[top].sum() / carry.sum()) if carry.sum() else float("nan"),
                "edge_excluding_best_10_days_bps": float((carry.sum() - carry[top].sum()) / rest) if rest else float("nan"),
                **{f"edge_{y}_bps": float(v) for y, v in years.iter_rows()},
            }
        )
    return {
        "univariate_summary": univariate_summary,
        "oos_summary": oos,
        "oos_increments": pl.DataFrame(increments).sort("horizon_hours", "candidate"),
        "tail_breakeven": pl.DataFrame(tail_rows, infer_schema_length=None).sort(
            "horizon_hours", "model", "tail_fraction"
        ),
    }


def settlement_summary(
    events: pl.DataFrame, config: dict
) -> tuple[pl.DataFrame, pl.DataFrame]:
    calibration = events.filter(pl.col("month") < config["first_test_month"])
    cuts = calibration["premium_7h_before_bps"].drop_nans().quantile(1 / 3), calibration[
        "premium_7h_before_bps"
    ].drop_nans().quantile(2 / 3)
    labelled = events.with_columns(
        pl.when(pl.col("premium_7h_before_bps") < cuts[0])
        .then(pl.lit("1 low premium"))
        .when(pl.col("premium_7h_before_bps") < cuts[1])
        .then(pl.lit("2 middle"))
        .otherwise(pl.lit("3 high premium"))
        .alias("predicted_funding_tercile"),
        pl.when(pl.col("month") < config["first_test_month"])
        .then(pl.lit("2021-02..2022-12"))
        .otherwise(pl.lit("2023-01..2026-05"))
        .alias("period"),
    )
    rows = []
    for (period, tercile), frame in labelled.group_by(["period", "predicted_funding_tercile"]):
        row = {"period": period, "predicted_funding_tercile": tercile, "settlements": frame.height}
        for column in ["pre_hour_return_bps", "post_hour_return_bps"]:
            values = frame[column].drop_nans().to_numpy()
            daily = frame.group_by("day").agg(pl.col(column).mean())[column].drop_nans().to_numpy()
            row[f"mean_{column}"] = float(values.mean())
            row[f"{column}_day_t"] = t_stat(daily)
        rows.append(row)
    return pl.DataFrame(rows).sort("period", "predicted_funding_tercile"), labelled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "alpha-search" / "configs" / "funding_positioning_v1.json",
    )
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-panel", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir or (REPO / "alpha-search" / "reports" / config["experiment_id"])
    output.mkdir(parents=True, exist_ok=True)

    panel_path, events_path = output / "hourly_panel.parquet", output / "settlement_events.parquet"
    if args.reuse_panel and panel_path.exists():
        panel, events = pl.read_parquet(panel_path), pl.read_parquet(events_path)
    else:
        panel, events = build_panel(args.shared_data_root, config)
        panel.write_parquet(panel_path, compression="zstd")
    coverage = panel.select(
        pl.len().alias("hours"),
        *[pl.col(f).is_finite().mean().alias(f) for f in all_features(config)],
    )
    coverage.write_csv(output / "feature_coverage.csv")
    print(coverage.transpose(include_header=True), flush=True)

    results = evaluate(panel, config)
    summaries = summarise(results, config)
    settlement, labelled = settlement_summary(events, config)
    labelled.write_parquet(events_path, compression="zstd")
    results["oos_monthly"].write_csv(output / "oos_monthly.csv")
    results["univariate_monthly"].write_csv(output / "univariate_monthly.csv")
    results["tail_daily"].write_parquet(output / "tail_daily.parquet", compression="zstd")
    for name, frame in {**summaries, "settlement_summary": settlement}.items():
        frame.write_csv(output / f"{name}.csv")
    manifest = {
        "experiment_id": config["experiment_id"],
        "config": str(args.config.resolve()),
        "holdout_months_excluded": config["holdout_months_excluded"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
