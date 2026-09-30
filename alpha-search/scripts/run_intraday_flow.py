"""H-012: persistent aggressive flow at intraday horizons, with passive entry.

Builds a 5-minute decision panel from the 1-second event grid (trailing-window
flow features, the frozen return benchmark, forward returns and simulated
passive/taker executions), then evaluates the registered cells with the frozen
walk-forward protocol. See HYPOTHESIS_LEDGER.md (H-012) and
FEATURE_CATALOGUE.md ("Trailing-window flow features").
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
from run_residual_decomposition import month_ms, month_range, ols, spearman, t_stat  # noqa: E402
from run_stress_episodes import grid_path, next_month  # noqa: E402
from run_funding_positioning import add_month  # noqa: E402


BENCHMARK = ["ret_5m", "ret_15m", "ret_60m", "rv_24h"]


# ------------------------------------------------------------------------ panel


def minute_table(root: Path, config: dict) -> pl.DataFrame:
    parts = []
    for month in iter_months(config["first_month"], config["last_month"]):
        parts.append(
            pl.scan_parquet(grid_path(root, config, month))
            .select("second", "close", "buy_qty", "sell_qty")
            .group_by((pl.col("second") // 60).alias("minute"))
            .agg(
                pl.col("close").sort_by("second").last(),
                pl.col("buy_qty").sum(),
                pl.col("sell_qty").sum(),
            )
            .collect()
        )
    return pl.concat(parts).sort("minute")


def dense_minutes(table: pl.DataFrame) -> dict[str, np.ndarray]:
    first, last = int(table["minute"].min()), int(table["minute"].max())
    index = table["minute"].to_numpy() - first
    size = last - first + 1
    buy = np.zeros(size)
    sell = np.zeros(size)
    close = np.full(size, np.nan)
    buy[index] = table["buy_qty"].to_numpy()
    sell[index] = table["sell_qty"].to_numpy()
    close[index] = table["close"].to_numpy()
    fill = np.where(np.isfinite(close), np.arange(size), 0)
    np.maximum.accumulate(fill, out=fill)
    close = close[fill]
    net = buy - sell
    return {
        "first": first,
        "close": close,
        "cum_net": np.concatenate([[0.0], np.cumsum(net)]),
        "cum_gross": np.concatenate([[0.0], np.cumsum(buy + sell)]),
        "cum_pos": np.concatenate([[0], np.cumsum(net > 0)]),
        "cum_neg": np.concatenate([[0], np.cumsum(net < 0)]),
        "cum_r2": np.concatenate(
            [[0.0], np.cumsum(np.nan_to_num(np.diff(np.log(close), prepend=np.nan) * 1e4) ** 2)]
        ),
    }


def window_sum(cumulative: np.ndarray, last: np.ndarray, length: int) -> np.ndarray:
    """Sum over minute indices (last - length, last]."""

    upper = last + 1
    lower = np.maximum(upper - length, 0)
    return cumulative[upper] - cumulative[lower]


def rolling_z(values: np.ndarray, window: int, minimum: int) -> np.ndarray:
    """Standardise against strictly earlier values (the current value is excluded)."""

    series = pl.Series(values)
    mean = series.rolling_mean(window, min_samples=minimum).shift(1).to_numpy()
    std = series.rolling_std(window, min_samples=minimum).shift(1).to_numpy()
    return (values - mean) / np.where(std > 0, std, np.nan)


def execution_arrays(root: Path, config: dict, decisions: np.ndarray) -> dict[str, np.ndarray]:
    """Passive fills and taker/passive exit prices from second-level prices."""

    execution = config["execution"]
    timeout = int(execution["passive_timeout_seconds"])
    delay = int(execution["taker_entry_delay_seconds"])
    horizons = [int(h) * 60 for h in config["horizons_minutes"]]
    extra = max(horizons) + timeout + 120
    n = len(decisions)
    output = {
        "entry_price": np.full(n, np.nan),
        "fill_buy_s": np.full(n, np.nan),
        "fill_sell_s": np.full(n, np.nan),
        "taker_entry_price": np.full(n, np.nan),
    }
    for h in horizons:
        for name in ("exit_buy", "exit_sell", "taker_exit"):
            output[f"{name}_{h // 60}m"] = np.full(n, np.nan)

    for month in iter_months(config["first_month"], config["last_month"]):
        start = month_ms(month) // 1_000
        end = month_ms(next_month(month)) // 1_000
        chosen = np.flatnonzero((decisions >= start) & (decisions < end))
        if not len(chosen):
            continue
        frames = [pl.read_parquet(grid_path(root, config, month), columns=["second", "close", "high", "low"])]
        if next_month(month) <= config["last_month"]:
            frames.append(
                pl.scan_parquet(grid_path(root, config, next_month(month)))
                .select("second", "close", "high", "low")
                .filter(pl.col("second") < end + extra)
                .collect()
            )
        grid = pl.concat(frames)
        base = start - 3_600
        size = end + extra - base
        seconds = grid["second"].to_numpy() - base
        keep = (seconds >= 0) & (seconds < size)
        seconds = seconds[keep]
        close = np.full(size, np.nan)
        high = np.full(size, -np.inf)
        low = np.full(size, np.inf)
        close[seconds] = grid["close"].to_numpy()[keep]
        high[seconds] = grid["high"].to_numpy()[keep]
        low[seconds] = grid["low"].to_numpy()[keep]
        populated = np.isfinite(close)
        last_index = np.where(populated, np.arange(size), 0)
        np.maximum.accumulate(last_index, out=last_index)
        close_ff = np.where(np.maximum.accumulate(populated), close[last_index], np.nan)
        data_end = int(seconds.max()) if len(seconds) else -1

        def price_at(offsets: np.ndarray) -> np.ndarray:
            result = np.full(len(offsets), np.nan)
            ok = (offsets >= 0) & (offsets <= data_end)
            result[ok] = close_ff[offsets[ok]]
            return result

        t = decisions[chosen] - base
        # The limit is set at the last trade strictly before the decision time.
        entry = price_at(t - 1)
        output["entry_price"][chosen] = entry
        output["taker_entry_price"][chosen] = price_at(t + delay)
        fill_buy = np.full(len(t), np.nan)
        fill_sell = np.full(len(t), np.nan)
        for k in range(1, timeout + 1):
            index = np.clip(t + k, 0, size - 1)
            new_buy = np.isnan(fill_buy) & (low[index] < entry)
            new_sell = np.isnan(fill_sell) & (high[index] > entry)
            fill_buy[new_buy] = k
            fill_sell[new_sell] = k
        output["fill_buy_s"][chosen] = fill_buy
        output["fill_sell_s"][chosen] = fill_sell
        for h in horizons:
            label = f"{h // 60}m"
            for side, fill in (("buy", fill_buy), ("sell", fill_sell)):
                exit_offsets = np.where(np.isfinite(fill), t + np.nan_to_num(fill).astype(int) + h, -1)
                prices = price_at(exit_offsets)
                prices[~np.isfinite(fill)] = np.nan
                output[f"exit_{side}_{label}"][chosen] = prices
            output[f"taker_exit_{label}"][chosen] = price_at(t + delay + h)
        print(f"executions {month}", flush=True)
    return output


def build_panel(root: Path, config: dict) -> pl.DataFrame:
    minutes = dense_minutes(minute_table(root, config))
    first_minute = minutes["first"]
    start = int(datetime.fromisoformat(config["first_decision_day"]).replace(tzinfo=timezone.utc).timestamp())
    end = month_ms(next_month(config["last_month"])) // 1_000
    step = int(config["decision_interval_seconds"])
    t = np.arange(start, end, step)
    m = t // 60 - 1 - first_minute  # latest completed minute
    close = minutes["close"]

    rv = np.sqrt(window_sum(minutes["cum_r2"], m, int(config["volatility_minutes"])))
    columns: dict[str, object] = {
        "time_s": t,
        "month": [datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m") for v in t],
        "day": t // 86_400,
        "rv_24h": rv,
    }
    for k in (5, 15, 60):
        columns[f"ret_{k}m"] = 1e4 * np.log(close[m] / close[m - k]) / rv

    reference = int(config["reference_days"]) * 1_440
    z_window = int(config["zscore_days"]) * 86_400 // step
    for w in config["windows_minutes"]:
        q = window_sum(minutes["cum_net"], m, w)
        v = window_sum(minutes["cum_gross"], m, w)
        history = window_sum(minutes["cum_gross"], m - w, reference - w)
        v_ref = np.where(m - reference >= 0, history * w / (reference - w), np.nan)
        x = q / v_ref
        z = q / v
        columns[f"x_{w}"] = x
        columns[f"z_{w}"] = z
        columns[f"zx_{w}"] = rolling_z(x, z_window, z_window // 2)
        columns[f"zz_{w}"] = rolling_z(z, z_window, z_window // 2)
        positive = window_sum(minutes["cum_pos"], m, w)
        negative = window_sum(minutes["cum_neg"], m, w)
        columns[f"persist_{w}"] = np.where(q > 0, positive, negative) / w
    for h in config["horizons_minutes"]:
        target = m + h
        valid = target < len(close)
        forward = np.full(len(t), np.nan)
        forward[valid] = 1e4 * np.log(close[target[valid]] / close[m[valid]])
        columns[f"F{h}m"] = forward

    panel = pl.DataFrame(columns)
    execution = execution_arrays(root, config, t)
    return panel.with_columns([pl.Series(k, v) for k, v in execution.items()])


# ------------------------------------------------------------------- evaluation


def registered_cells(config: dict) -> list[dict]:
    primary_w = config["primary"]["window_minutes"]
    primary_h = config["primary"]["horizon_minutes"]
    cells = [
        {"cell": f"flow_W{w}_H{h}", "features": BENCHMARK + [f"zx_{w}"], "horizon": h, "spaced": False}
        for w in config["windows_minutes"]
        for h in config["horizons_minutes"]
    ]
    cells += [
        {"cell": "interaction_primary", "features": BENCHMARK + [f"zx_{primary_w}", f"inter_{primary_w}"],
         "horizon": primary_h, "spaced": False},
        {"cell": "scaling_z_primary", "features": BENCHMARK + [f"zz_{primary_w}"],
         "horizon": primary_h, "spaced": False},
        {"cell": "spaced_primary", "features": BENCHMARK + [f"zx_{primary_w}"],
         "horizon": primary_h, "spaced": True},
    ]
    for cell in cells:
        cell["primary"] = cell["cell"] == f"flow_W{primary_w}_H{primary_h}"
    return cells


def trade_outcomes(test: pl.DataFrame, direction: np.ndarray, horizon: int, config: dict) -> dict[str, np.ndarray]:
    passive = config["execution"]["passive_costs_bps"]
    taker = config["execution"]["taker_costs_bps"]
    entry = test["entry_price"].to_numpy()
    buy_exit = test[f"exit_buy_{horizon}m"].to_numpy()
    sell_exit = test[f"exit_sell_{horizon}m"].to_numpy()
    exit_price = np.where(direction > 0, buy_exit, sell_exit)
    passive_gross = direction * 1e4 * np.log(exit_price / entry)
    taker_gross = direction * 1e4 * np.log(
        test[f"taker_exit_{horizon}m"].to_numpy() / test["taker_entry_price"].to_numpy()
    )
    return {
        "filled": np.isfinite(passive_gross),
        "passive_net": passive_gross - passive["entry_maker"] - passive["exit_taker"],
        "passive_gross": passive_gross,
        "taker_net": taker_gross - taker["entry"] - taker["exit"],
        "unconditional_gross": direction * test[f"F{horizon}m"].to_numpy(),
    }


def evaluate(panel: pl.DataFrame, config: dict) -> dict[str, pl.DataFrame]:
    low, high = config["winsor_quantiles"]
    window = int(config["regression_training_months"])
    w = config["primary"]["window_minutes"]
    panel = panel.with_columns((pl.col(f"zx_{w}") * pl.col(f"persist_{w}")).alias(f"inter_{w}"))
    feature_union = sorted({f for c in registered_cells(config) for f in c["features"]})
    usable = panel.filter(pl.all_horizontal([pl.col(f).is_finite() for f in feature_union]))
    monthly, coefficients, trades = [], [], []
    for month in month_range(config["first_test_month"], config["last_test_month"]):
        train_all = usable.filter(pl.col("month").is_in(month_range(add_month(month, -window), add_month(month, -1))))
        test_all = usable.filter(pl.col("month") == month)
        bounds_all = np.quantile(train_all.select(feature_union).to_numpy(), [low, high], axis=0)
        benchmarks = [{"cell": "benchmark", "features": BENCHMARK, "horizon": h, "spaced": False, "primary": False}
                      for h in config["horizons_minutes"]]
        benchmarks.append({"cell": "benchmark_spaced", "features": BENCHMARK,
                           "horizon": config["primary"]["horizon_minutes"], "spaced": True, "primary": False})
        for cell in benchmarks + registered_cells(config):
            h = cell["horizon"]
            target = f"F{h}m"
            embargo = month_ms(month) // 1_000 - (h + config["embargo_extra_minutes"]) * 60
            train = train_all.filter((pl.col("time_s") < embargo) & pl.col(target).is_finite())
            test = test_all.filter(pl.col(target).is_finite())
            if cell["spaced"]:
                train = train.filter(pl.col("time_s") % (h * 60) == 0)
                test = test.filter(pl.col("time_s") % (h * 60) == 0)
            if train.height < 1_000 or test.height < 50:
                continue
            columns = [feature_union.index(f) for f in cell["features"]]
            x_train = np.clip(train.select(feature_union).to_numpy(), bounds_all[0], bounds_all[1])[:, columns]
            x_test = np.clip(test.select(feature_union).to_numpy(), bounds_all[0], bounds_all[1])[:, columns]
            y = train[target].to_numpy()
            y = np.clip(y, *np.quantile(y, [low, high]))
            centre = x_train.mean(axis=0)
            beta = ols(x_train - centre, y - y.mean())
            forecast = (x_test - centre) @ beta
            fitted = (x_train - centre) @ beta
            monthly.append({"test_month": month, "cell": cell["cell"], "horizon_minutes": h,
                            "observations": len(forecast), "spearman_ic": spearman(forecast, test[target].to_numpy())})
            coefficients.append({"test_month": month, "cell": cell["cell"],
                                 **{f"b_{f}": float(b) for f, b in zip(cell["features"], beta)}})
            if cell["spaced"]:
                continue
            for fraction in config["tail_fractions"]:
                threshold = float(np.quantile(np.abs(fitted), 1.0 - fraction))
                chosen = np.abs(forecast) >= threshold
                if not chosen.any():
                    continue
                selected = test.filter(pl.Series(chosen))
                direction = np.sign(forecast[chosen])
                outcome = trade_outcomes(selected, direction, h, config)
                days = selected["day"].to_numpy()
                unique, inverse = np.unique(days, return_inverse=True)
                filled = outcome["filled"]
                for name in ("passive_net", "taker_net", "unconditional_gross"):
                    values = outcome[name]
                    ok = np.isfinite(values)
                    sums = np.bincount(inverse[ok], weights=values[ok], minlength=len(unique))
                    counts = np.bincount(inverse[ok], minlength=len(unique))
                    for day, total, count in zip(unique, sums, counts):
                        if count:
                            trades.append({"test_month": month, "cell": cell["cell"], "horizon_minutes": h,
                                           "tail_fraction": fraction, "measure": name, "day": int(day),
                                           "trades": int(count), "sum_bps": float(total)})
                trades.append({"test_month": month, "cell": cell["cell"], "horizon_minutes": h,
                               "tail_fraction": fraction, "measure": "fill_rate", "day": -1,
                               "trades": int(len(filled)), "sum_bps": float(filled.sum())})
    return {"monthly": pl.DataFrame(monthly), "coefficients": pl.DataFrame(coefficients, infer_schema_length=None),
            "trades": pl.DataFrame(trades)}


def summarise(results: dict[str, pl.DataFrame], config: dict) -> dict[str, pl.DataFrame]:
    monthly = results["monthly"]
    rows = []
    for (cell, h), frame in monthly.group_by(["cell", "horizon_minutes"]):
        baseline = "benchmark_spaced" if cell == "spaced_primary" else "benchmark"
        base = monthly.filter((pl.col("cell") == baseline) & (pl.col("horizon_minutes") == h))
        joined = frame.join(base.select("test_month", pl.col("spearman_ic").alias("base_ic")), on="test_month")
        delta = (joined["spearman_ic"] - joined["base_ic"]).to_numpy()
        ic = frame["spearman_ic"].to_numpy()
        rows.append({"cell": cell, "horizon_minutes": h, "months": len(ic),
                     "mean_ic": float(np.nanmean(ic)), "ic_t": t_stat(ic),
                     "mean_delta_ic": float(np.nanmean(delta)), "delta_ic_t": t_stat(delta),
                     "delta_positive_month_fraction": float(np.mean(delta > 0))})
    ic_summary = pl.DataFrame(rows).sort("cell", "horizon_minutes")

    coefficient_rows = []
    for cell, frame in results["coefficients"].group_by("cell"):
        flow = [c for c in frame.columns if c.startswith("b_zx_") or c.startswith("b_zz_")]
        for column in flow:
            values = frame[column].drop_nulls().to_numpy()
            if len(values):
                coefficient_rows.append({"cell": cell[0] if isinstance(cell, tuple) else cell, "coefficient": column,
                                         "positive_refit_fraction": float(np.mean(values > 0)),
                                         "mean": float(values.mean())})
    trades = results["trades"]
    rows = []
    for keys, frame in trades.filter(pl.col("measure") != "fill_rate").group_by(
        ["cell", "horizon_minutes", "tail_fraction", "measure"]
    ):
        count = frame["trades"].to_numpy()
        total = frame["sum_bps"].to_numpy()
        top = np.argsort(total)[::-1][:10]
        rest = count.sum() - count[top].sum()
        years = (frame.with_columns(pl.col("test_month").str.slice(0, 4).alias("year"))
                 .group_by("year").agg(pl.col("sum_bps").sum() / pl.col("trades").sum()).sort("year"))
        rows.append({"cell": keys[0], "horizon_minutes": keys[1], "tail_fraction": keys[2], "measure": keys[3],
                     "trades": int(count.sum()), "trades_per_day": float(count.sum() / frame["day"].n_unique()),
                     "mean_bps": float(total.sum() / count.sum()),
                     "daily_block_t": t_stat(total / count),
                     "ordinary_day_mean_bps": float((total.sum() - total[top].sum()) / rest) if rest else float("nan"),
                     "best_10_day_share": float(total[top].sum() / total.sum()) if total.sum() else float("nan"),
                     **{f"mean_{y}_bps": float(v) for y, v in years.iter_rows()}})
    fills = (trades.filter(pl.col("measure") == "fill_rate")
             .group_by(["cell", "horizon_minutes", "tail_fraction"])
             .agg((pl.col("sum_bps").sum() / pl.col("trades").sum()).alias("fill_rate")))
    economics = pl.DataFrame(rows, infer_schema_length=None).join(
        fills, on=["cell", "horizon_minutes", "tail_fraction"], how="left"
    ).sort("cell", "horizon_minutes", "tail_fraction", "measure")
    return {"ic_summary": ic_summary, "coefficient_signs": pl.DataFrame(coefficient_rows),
            "economics": economics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=REPO / "alpha-search" / "configs" / "intraday_flow_v1.json")
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-panel", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir or (REPO / "alpha-search" / "reports" / config["experiment_id"])
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "decision_panel.parquet"
    if args.reuse_panel and panel_path.exists():
        panel = pl.read_parquet(panel_path)
    else:
        panel = build_panel(args.shared_data_root, config)
        panel.write_parquet(panel_path, compression="zstd")
    results = evaluate(panel, config)
    summaries = summarise(results, config)
    results["monthly"].write_csv(output / "oos_monthly.csv")
    results["coefficients"].write_csv(output / "oos_coefficients.csv")
    results["trades"].write_parquet(output / "trades_daily.parquet", compression="zstd")
    for name, frame in summaries.items():
        frame.write_csv(output / f"{name}.csv")
    manifest = {"experiment_id": config["experiment_id"], "hypothesis": config["hypothesis"],
                "config": str(args.config.resolve()),
                "holdout_months_excluded": config["holdout_months_excluded"],
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
