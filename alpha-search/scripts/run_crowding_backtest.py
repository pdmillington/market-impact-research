"""Implementable hourly backtest of the Step 4 crowding forecasts.

Each hour a frozen OOS forecast becomes a unit signal; the position is the mean of
the last H signals (H staggered tranches, each held H hours), so overlapping
trades net into one position. P&L includes the next-hour perp return, funding
settled while the position is held, and fees on position changes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

from run_funding_positioning import (  # noqa: E402
    HOUR,
    add_month,
    all_features,
    build_panel,
    model_features,
)
from run_residual_decomposition import month_ms, month_range, ols  # noqa: E402


def prepare_panel(panel: pl.DataFrame, config: dict) -> pl.DataFrame:
    imputed = config.get("impute_zero_when_missing", [])
    features = all_features(config)
    return (
        panel.with_columns(
            [pl.when(pl.col(f).is_finite()).then(pl.col(f)).otherwise(0.0).alias(f) for f in imputed]
        )
        .filter(pl.all_horizontal([pl.col(f).is_finite() for f in features]))
        .sort("time_s")
    )


def forecasts(
    usable: pl.DataFrame,
    panel_config: dict,
    backtest: dict,
    *,
    model: str,
    horizon: int,
    window: int,
) -> pl.DataFrame:
    """Hourly OOS forecasts and training thresholds for one model and horizon."""

    low, high = panel_config["winsor_quantiles"]
    features = all_features(panel_config)
    columns = [features.index(f) for f in model_features(panel_config, model)]
    target = f"F{horizon}h"
    rows = []
    for month in month_range(backtest["first_test_month"], backtest["last_test_month"]):
        train_months = month_range(add_month(month, -window), add_month(month, -1))
        embargo = month_ms(month) // 1_000 - (horizon + 1) * HOUR
        train = usable.filter(
            pl.col("month").is_in(train_months)
            & (pl.col("time_s") < embargo)
            & pl.col(target).is_finite()
        )
        test = usable.filter(pl.col("month") == month)
        if train.height < 500 or test.is_empty():
            continue
        x_all = train.select(features).to_numpy()
        bounds = np.quantile(x_all, [low, high], axis=0)
        x_train = np.clip(x_all, bounds[0], bounds[1])[:, columns]
        x_test = np.clip(test.select(features).to_numpy(), bounds[0], bounds[1])[:, columns]
        y = train[target].to_numpy()
        y = np.clip(y, *np.quantile(y, [low, high]))
        centre = x_train.mean(axis=0)
        beta = ols(x_train - centre, y - y.mean())
        threshold = float(
            np.quantile(np.abs((x_train - centre) @ beta), 1.0 - backtest["tail_fraction"])
        )
        rows.append(
            pl.DataFrame(
                {
                    "time_s": test["time_s"],
                    "forecast": (x_test - centre) @ beta,
                    "threshold": np.full(test.height, threshold),
                }
            )
        )
    return pl.concat(rows)


def simulate(
    usable: pl.DataFrame,
    forecast: pl.DataFrame,
    *,
    horizon: int,
    rule: str,
    costs: dict[str, float],
) -> pl.DataFrame:
    """Hourly position and P&L components on a complete hourly grid."""

    start = int(forecast["time_s"].min())
    end = int(forecast["time_s"].max())
    grid = pl.DataFrame({"time_s": np.arange(start, end + HOUR, HOUR)})
    frame = (
        grid.join(forecast, on="time_s", how="left")
        .join(usable.select("time_s", "F1h", "carry1h"), on="time_s", how="left")
        .with_columns(pl.col("F1h").fill_null(0.0).fill_nan(0.0), pl.col("carry1h").fill_null(0.0).fill_nan(0.0))
    )
    f = frame["forecast"].to_numpy()
    threshold = frame["threshold"].to_numpy()
    if rule == "tail":
        signal = np.where(np.abs(f) >= threshold, np.sign(f), 0.0)
    else:
        signal = np.clip(f / threshold, -1.0, 1.0)
    # Hours without a forecast (data gaps) contribute a flat tranche.
    signal = np.where(np.isfinite(signal), signal, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(signal)])
    index = np.arange(1, len(signal) + 1)
    position = (cumulative[index] - cumulative[np.maximum(index - horizon, 0)]) / horizon
    turnover = np.abs(np.diff(np.concatenate([[0.0], position])))
    price = position * frame["F1h"].to_numpy()
    carry = -position * frame["carry1h"].to_numpy()
    output = {
        "time_s": frame["time_s"],
        "position": position,
        "turnover": turnover,
        "price_pnl_bps": price,
        "carry_pnl_bps": carry,
    }
    for name, per_side in costs.items():
        output[f"net_{name}_bps"] = price + carry - per_side * turnover
    return pl.DataFrame(output)


def summarise(hourly: pl.DataFrame, costs: dict[str, float]) -> dict:
    daily = (
        hourly.group_by((pl.col("time_s") // 86_400).alias("day"))
        .agg(pl.all().exclude("time_s", "position").sum(), pl.col("position").abs().mean().alias("gross_exposure"))
        .sort("day")
    )
    days = daily.height
    row = {
        "days": days,
        "mean_abs_position": float(hourly["position"].abs().mean()),
        "turnover_per_day": float(daily["turnover"].mean()),
        "price_pnl_bps_per_year": float(daily["price_pnl_bps"].mean() * 365),
        "carry_pnl_bps_per_year": float(daily["carry_pnl_bps"].mean() * 365),
    }
    for name in costs:
        pnl = daily[f"net_{name}_bps"].to_numpy()
        equity = np.cumsum(pnl)
        drawdown = equity - np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
        top = np.sort(pnl)[::-1][:10]
        row[f"{name}_return_pct_per_year"] = float(pnl.mean() * 365 / 100)
        row[f"{name}_sharpe"] = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(365)) if pnl.std() else float("nan")
        row[f"{name}_max_drawdown_pct"] = float(drawdown.min() / 100)
        row[f"{name}_return_ex_best10_pct_per_year"] = float((pnl.sum() - top.sum()) / days * 365 / 100)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "alpha-search" / "configs" / "crowding_backtest_v1.json",
    )
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    backtest = json.loads(args.config.read_text())
    panel_config = json.loads((REPO / backtest["panel_config"]).read_text())
    output = args.output_dir or (REPO / "alpha-search" / "reports" / backtest["experiment_id"])
    output.mkdir(parents=True, exist_ok=True)

    panel_path = REPO / "alpha-search" / "reports" / backtest["panel_experiment"] / "hourly_panel.parquet"
    if panel_path.exists():
        panel = pl.read_parquet(panel_path)
    else:
        panel, _ = build_panel(args.shared_data_root, panel_config)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(panel_path, compression="zstd")
    usable = prepare_panel(panel, panel_config)
    costs = backtest["costs_bps_per_side"]

    summaries, yearly, hourly_all = [], [], []
    for model, horizon, rule, window in itertools.product(
        backtest["models"],
        backtest["horizons_hours"],
        backtest["position_rules"],
        backtest["training_windows_months"],
    ):
        forecast = forecasts(usable, panel_config, backtest, model=model, horizon=horizon, window=window)
        hourly = simulate(usable, forecast, horizon=horizon, rule=rule, costs=costs)
        key = {"model": model, "horizon_hours": horizon, "position_rule": rule, "training_window_months": window}
        is_primary = key == backtest["primary"]
        recent = summarise(
            hourly.filter(pl.col("time_s") >= month_ms(backtest["recent_start_month"]) // 1_000),
            costs,
        )
        summaries.append(
            {
                **key,
                "primary": is_primary,
                **summarise(hourly, costs),
                "recent_maker_return_pct_per_year": recent["maker_return_pct_per_year"],
                "recent_maker_sharpe": recent["maker_sharpe"],
            }
        )
        years = hourly.with_columns(
            pl.from_epoch("time_s").dt.year().alias("year")
        ).group_by("year")
        for (year,), frame in years:
            yearly.append({**key, "primary": is_primary, "year": year, **summarise(frame, costs)})
        hourly_all.append(hourly.with_columns([pl.lit(v).alias(k) for k, v in key.items()]))
        print(f"{key} done", flush=True)

    # Reference: holding one unit long throughout, paying funding, no trading costs.
    reference = usable.filter(
        (pl.col("time_s") >= month_ms(backtest["first_test_month"]) // 1_000)
    ).select("time_s", "F1h", "carry1h").fill_nan(0.0)
    reference_hourly = pl.DataFrame(
        {
            "time_s": reference["time_s"],
            "position": np.ones(reference.height),
            "turnover": np.zeros(reference.height),
            "price_pnl_bps": reference["F1h"],
            "carry_pnl_bps": -reference["carry1h"],
            **{f"net_{name}_bps": reference["F1h"] - reference["carry1h"] for name in costs},
        }
    )
    summaries.append(
        {"model": "buy_and_hold", "horizon_hours": 0, "position_rule": "long", "training_window_months": 0,
         "primary": False, **summarise(reference_hourly, costs)}
    )

    # Drop-one attribution of the positioning group (a diagnostic, not a candidate search).
    attribution_spec = backtest["attribution"]
    positioning = panel_config["feature_groups"]["positioning"]
    ratio_features = [f for f in positioning if f in attribution_spec["ratio_features"]]
    attribution = []
    for horizon in backtest["horizons_hours"]:
        for dropped in [None, *positioning, "all_long_short_ratios"]:
            if dropped == "all_long_short_ratios":
                kept = [f for f in positioning if f not in ratio_features]
            else:
                kept = [f for f in positioning if f != dropped]
            variant = {
                **panel_config,
                "feature_groups": {**panel_config["feature_groups"], "positioning": kept},
            }
            forecast = forecasts(
                usable, variant, backtest,
                model=attribution_spec["model"], horizon=horizon,
                window=attribution_spec["training_window_months"],
            )
            result = summarise(
                simulate(usable, forecast, horizon=horizon, rule=attribution_spec["position_rule"], costs=costs),
                costs,
            )
            attribution.append(
                {
                    "model": attribution_spec["model"],
                    "horizon_hours": horizon,
                    "dropped_feature": dropped or "none",
                    "maker_return_pct_per_year": result["maker_return_pct_per_year"],
                    "maker_sharpe": result["maker_sharpe"],
                    "taker_sharpe": result["taker_sharpe"],
                }
            )
    pl.DataFrame(attribution).write_csv(output / "positioning_attribution.csv")

    pl.DataFrame(summaries).write_csv(output / "backtest_summary.csv")
    pl.DataFrame(yearly).sort("model", "horizon_hours", "position_rule", "training_window_months", "year").write_csv(
        output / "backtest_by_year.csv"
    )
    pl.concat(hourly_all).write_parquet(output / "backtest_hourly.parquet", compression="zstd")
    manifest = {
        "experiment_id": backtest["experiment_id"],
        "config": str(args.config.resolve()),
        "primary": backtest["primary"],
        "holdout_months_excluded": backtest["holdout_months_excluded"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
