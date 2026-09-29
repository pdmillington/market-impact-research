"""Stress-episode event study on the 1-second event grid.

Pass 1 scores every second by its largest volatility-scaled move over the
trigger windows and keeps candidates above a floor. The episode threshold is
chosen on the calibration years only, candidates are merged into episodes, and
pass 2 measures decision-time features, open interest, reversion paths and
entry-rule outcomes for each episode.

Returns are reported as reversion: positive means price moved back against the
triggering move.
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
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.months import iter_months  # noqa: E402


GRID_COLUMNS = [
    "second",
    "close",
    "high",
    "low",
    "buy_qty",
    "sell_qty",
    "buy_events",
    "sell_events",
    "buy_deep",
    "sell_deep",
    "flips",
    "flip_gap_sum_bps",
]
DAY_SECONDS = 86_400


def month_seconds(month: str) -> int:
    return int(
        datetime.strptime(f"{month}-01", "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def next_month(month: str) -> str:
    year, number = map(int, month.split("-"))
    return f"{year + number // 12}-{number % 12 + 1:02d}"


def previous_month(month: str) -> str:
    year, number = map(int, month.split("-"))
    return f"{year - 1}-12" if number == 1 else f"{year}-{number - 1:02d}"


def grid_path(root: Path, config: dict, month: str) -> Path:
    year, number = month.split("-")
    return (
        root
        / "features"
        / "alpha"
        / f"symbol={config['symbol']}"
        / config["grid_version"]
        / f"year={year}"
        / f"month={number}"
        / "part-000.parquet"
    )


def load_chunk(root: Path, config: dict, month: str, *, before: int, after: int) -> dict:
    """Load a month plus `before` seconds of history and `after` seconds of future."""

    start = month_seconds(month)
    end = month_seconds(next_month(month))
    frames = []
    for candidate, low, high in [
        (previous_month(month), start - before, start),
        (month, start, end),
        (next_month(month), end, end + after),
    ]:
        if not (config["first_month"] <= candidate <= config["last_month"]):
            continue
        path = grid_path(root, config, candidate)
        if not path.exists():
            continue
        frames.append(
            pl.scan_parquet(path)
            .select(GRID_COLUMNS)
            .filter((pl.col("second") >= low) & (pl.col("second") < high))
            .collect()
        )
    frame = pl.concat(frames).sort("second")
    chunk = {name: frame[name].to_numpy() for name in GRID_COLUMNS}
    chunk["month_start"] = start
    chunk["month_end"] = end
    return chunk


def minute_volatility(chunk: dict, lookback: int, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing RMS of 1-minute log returns (bps), excluding the current minute."""

    seconds = chunk["second"]
    minutes = seconds // 60
    first, last = minutes[0], minutes[-1]
    grid = np.arange(first, last + 1)
    close_index = np.searchsorted(seconds, (grid + 1) * 60, side="left") - 1
    minute_close = chunk["close"][close_index]
    returns = np.full(len(grid), np.nan)
    returns[1:] = 10_000.0 * np.diff(np.log(minute_close))
    valid = np.isfinite(returns)
    squared = np.where(valid, returns**2, 0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(squared)])
    counts = np.concatenate([[0], np.cumsum(valid)])
    index = np.arange(len(grid))
    low = np.maximum(index - lookback, 0)
    total = cumulative[index] - cumulative[low]
    count = counts[index] - counts[low]
    sigma = np.where(count >= minimum, np.sqrt(total / np.maximum(count, 1)), np.nan)
    return grid, sigma


def price_at(chunk: dict, times: np.ndarray) -> np.ndarray:
    """Close of the last populated second at or before each time."""

    index = np.searchsorted(chunk["second"], times, side="right") - 1
    output = np.full(len(times), np.nan)
    valid = index >= 0
    output[valid] = chunk["close"][index[valid]]
    # Times beyond the loaded data are unknown rather than stale.
    output[np.asarray(times) > chunk["second"][-1]] = np.nan
    return output


def window_scores(chunk: dict, config: dict) -> dict:
    minutes, sigma = minute_volatility(
        chunk, config["volatility_lookback_minutes"], config["minimum_volatility_minutes"]
    )
    seconds = chunk["second"]
    in_month = (seconds >= chunk["month_start"]) & (seconds < chunk["month_end"])
    t = seconds[in_month]
    sigma_1m = sigma[t // 60 - minutes[0]]
    close = chunk["close"][in_month]
    scores = {}
    for window in config["trigger_windows_seconds"]:
        past = price_at(chunk, t - window)
        scale = sigma_1m * np.sqrt(window / 60.0)
        scores[window] = 10_000.0 * np.log(close / past) / scale
    return {"second": t, "sigma_1m_bps": sigma_1m, "scores": scores}


def candidates_for_month(root: Path, config: dict, month: str) -> pl.DataFrame:
    before = (config["volatility_lookback_minutes"] + 5) * 60
    chunk = load_chunk(root, config, month, before=before, after=0)
    scored = window_scores(chunk, config)
    windows = config["trigger_windows_seconds"]
    stack = np.column_stack([scored["scores"][w] for w in windows])
    magnitude = np.where(np.isfinite(stack), np.abs(stack), -np.inf)
    best = np.argmax(magnitude, axis=1)
    z = magnitude[np.arange(len(best)), best]
    chosen = z >= config["candidate_floor_z"]
    signed_best = stack[np.arange(len(best)), best]
    return pl.DataFrame(
        {
            "second": scored["second"][chosen],
            "z": z[chosen],
            "window_seconds": np.asarray(windows)[best[chosen]],
            "direction": np.sign(signed_best[chosen]).astype(np.int8),
        }
    )


def merge_episodes(candidates: pl.DataFrame, threshold: float, gap: int) -> pl.DataFrame:
    """First qualifying second of each cluster whose qualifying seconds are < gap apart."""

    qualifying = candidates.filter(pl.col("z") >= threshold).sort("second")
    if qualifying.is_empty():
        return qualifying
    seconds = qualifying["second"].to_numpy()
    new_cluster = np.concatenate([[True], np.diff(seconds) >= gap])
    cluster = np.cumsum(new_cluster)
    summary = (
        qualifying.with_columns(pl.Series("cluster", cluster))
        .group_by("cluster", maintain_order=True)
        .agg(
            pl.col("second").first().alias("trigger_second"),
            pl.col("z").first().alias("trigger_z"),
            pl.col("window_seconds").first().alias("trigger_window_seconds"),
            pl.col("direction").first().alias("direction"),
            pl.col("z").max().alias("peak_z"),
            (pl.col("second").last() - pl.col("second").first()).alias(
                "qualifying_span_seconds"
            ),
            pl.len().alias("qualifying_seconds"),
        )
    )
    return summary.drop("cluster")


def calibrate_threshold(
    candidates: pl.DataFrame, config: dict, target_per_year: float
) -> float:
    start = month_seconds(config["calibration_start_month"])
    end = month_seconds(next_month(config["calibration_end_month"]))
    training = candidates.filter((pl.col("second") >= start) & (pl.col("second") < end))
    years = (end - start) / (365.25 * DAY_SECONDS)
    target = target_per_year * years
    low, high = config["candidate_floor_z"], float(training["z"].max())
    for _ in range(60):
        middle = 0.5 * (low + high)
        count = merge_episodes(training, middle, config["episode_merge_gap_seconds"]).height
        if count > target:
            low = middle
        else:
            high = middle
    return high


def window_sum(chunk: dict, cumulative: np.ndarray, end: np.ndarray, length) -> np.ndarray:
    """Sum over seconds in (end - length, end]; `length` may be scalar or per-row."""

    seconds = chunk["second"]
    upper = np.searchsorted(seconds, end, side="right")
    lower = np.searchsorted(seconds, end - length, side="right")
    return cumulative[upper] - cumulative[lower]


def load_open_interest(root: Path, symbol: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = root / "canonical" / "metrics" / f"symbol={symbol}" / "metrics_5m.parquet"
    if not path.exists():
        print(f"WARNING: {path} missing; open-interest features will be NaN", flush=True)
        empty = np.asarray([], dtype=np.int64)
        return empty, {
            name: np.asarray([], dtype=float)
            for name in [
                "sum_open_interest",
                "count_toptrader_long_short_ratio",
                "sum_taker_long_short_vol_ratio",
            ]
        }
    # Zero open interest marks Binance outages (e.g. 2021-05-22); treat as missing.
    table = pl.read_parquet(path).filter(pl.col("sum_open_interest") > 0).sort("time_ms")
    times = table["time_ms"].to_numpy() // 1_000
    values = {
        name: table[name].to_numpy()
        for name in [
            "sum_open_interest",
            "count_toptrader_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ]
    }
    return times, values


def snapshot(
    times: np.ndarray,
    values: np.ndarray,
    at: np.ndarray,
    *,
    after: bool,
    max_age_seconds: int = 900,
) -> np.ndarray:
    """Nearest snapshot on one side of `at`; NaN if none within `max_age_seconds`."""

    at = np.asarray(at)
    if after:
        index = np.searchsorted(times, at, side="left")
        valid = index < len(times)
    else:
        index = np.searchsorted(times, at, side="right") - 1
        valid = index >= 0
    output = np.full(len(at), np.nan)
    chosen = index[valid]
    fresh = np.abs(times[chosen] - at[valid]) <= max_age_seconds
    output[np.flatnonzero(valid)[fresh]] = values[chosen[fresh]]
    return output


def episode_features(
    root: Path,
    config: dict,
    month: str,
    episodes: pl.DataFrame,
    open_interest: tuple[np.ndarray, dict[str, np.ndarray]],
) -> tuple[list[dict], list[dict], list[dict]]:
    before = (config["volatility_lookback_minutes"] + 5) * 60
    horizon = max(config["holding_horizons_seconds"]) + config["maximum_entry_wait_seconds"]
    if month >= config["last_month"]:
        horizon = 0
    chunk = load_chunk(root, config, month, before=before, after=horizon + 60)
    minutes, sigma = minute_volatility(
        chunk, config["volatility_lookback_minutes"], config["minimum_volatility_minutes"]
    )
    seconds = chunk["second"]
    cumulative = {
        name: np.concatenate([[0.0], np.cumsum(chunk[name].astype(float))])
        for name in [
            "buy_qty", "sell_qty", "buy_events", "sell_events",
            "buy_deep", "sell_deep", "flips", "flip_gap_sum_bps",
        ]
    }
    t0 = episodes["trigger_second"].to_numpy()
    d = episodes["direction"].to_numpy().astype(float)
    window = episodes["trigger_window_seconds"].to_numpy()
    p0 = price_at(chunk, t0)
    sigma_1m = sigma[t0 // 60 - minutes[0]]

    def aligned(name_buy: str, name_sell: str, length) -> tuple[np.ndarray, np.ndarray]:
        buy = window_sum(chunk, cumulative[name_buy], t0, length)
        sell = window_sum(chunk, cumulative[name_sell], t0, length)
        with_move = np.where(d > 0, buy, sell)
        against = np.where(d > 0, sell, buy)
        return with_move, against

    qty_with, qty_against = aligned("buy_qty", "sell_qty", window)
    deep_with, deep_against = aligned("buy_deep", "sell_deep", window)
    deep_with_day, deep_against_day = aligned("buy_deep", "sell_deep", DAY_SECONDS)
    events_with, events_against = aligned("buy_events", "sell_events", window)
    events_with_day, events_against_day = aligned("buy_events", "sell_events", DAY_SECONDS)
    spread_window = int(config["spread_proxy_lookback_seconds"])
    flips_now = window_sum(chunk, cumulative["flips"], t0, spread_window)
    gap_now = window_sum(chunk, cumulative["flip_gap_sum_bps"], t0, spread_window)
    flips_day = window_sum(chunk, cumulative["flips"], t0, DAY_SECONDS)
    gap_day = window_sum(chunk, cumulative["flip_gap_sum_bps"], t0, DAY_SECONDS)

    oi_times, oi_values = open_interest
    lag = int(config["open_interest_realtime_lag_seconds"])
    oi = oi_values["sum_open_interest"]
    oi_now = snapshot(oi_times, oi, t0 - lag, after=False)
    oi_hour = snapshot(oi_times, oi, t0 - lag - 3_600, after=False)
    oi_before = snapshot(oi_times, oi, t0 - window - lag, after=False)
    oi_after = snapshot(
        oi_times, oi, t0 + int(config["open_interest_label_after_seconds"]), after=True
    )

    rows = []
    for position, row in enumerate(episodes.iter_rows(named=True)):
        rows.append(
            {
                **row,
                "month": month,
                "trigger_time_utc": datetime.fromtimestamp(
                    int(t0[position]), tz=timezone.utc
                ).isoformat(),
                "trigger_price": float(p0[position]),
                # US macro releases land on :00/:30; flag triggers shortly after.
                "scheduled_release_window": bool(int(t0[position]) % 1_800 < 120),
                "utc_hour": int((int(t0[position]) % DAY_SECONDS) // 3_600),
                "sigma_1m_bps": float(sigma_1m[position]),
                "move_bps": float(
                    d[position]
                    * 10_000.0
                    * np.log(p0[position] / price_at(chunk, t0[position : position + 1] - window[position])[0])
                ),
                "aggression_share": float(
                    (qty_with[position] - qty_against[position])
                    / max(qty_with[position] + qty_against[position], 1e-12)
                ),
                "gross_qty_window": float(qty_with[position] + qty_against[position]),
                "deep_with": float(deep_with[position]),
                "deep_against": float(deep_against[position]),
                "deep_intensity_ratio": float(
                    deep_with[position] / window[position]
                    / max((deep_with_day[position] + deep_against_day[position]) / DAY_SECONDS, 1e-9)
                ),
                "event_intensity_ratio": float(
                    (events_with[position] + events_against[position]) / window[position]
                    / max((events_with_day[position] + events_against_day[position]) / DAY_SECONDS, 1e-9)
                ),
                "spread_proxy_bps": float(gap_now[position] / max(flips_now[position], 1)),
                "spread_proxy_day_bps": float(gap_day[position] / max(flips_day[position], 1)),
                "oi_change_1h_realtime_pct": float(100.0 * (oi_now[position] / oi_hour[position] - 1.0)),
                "oi_change_episode_label_pct": float(
                    100.0 * (oi_after[position] / oi_before[position] - 1.0)
                ),
                "toptrader_ls_ratio": float(
                    snapshot(oi_times, oi_values["count_toptrader_long_short_ratio"], t0[position : position + 1] - lag, after=False)[0]
                ),
                "taker_ls_vol_ratio": float(
                    snapshot(oi_times, oi_values["sum_taker_long_short_vol_ratio"], t0[position : position + 1] - lag, after=False)[0]
                ),
            }
        )

    offsets = np.asarray(config["path_offsets_seconds"])
    path_rows = []
    trade_rows = []
    for position in range(len(t0)):
        start, direction, price = int(t0[position]), d[position], p0[position]
        path = price_at(chunk, start + offsets)
        reversion = -direction * 10_000.0 * np.log(path / price)
        for offset, value in zip(offsets, reversion):
            path_rows.append(
                {"trigger_second": start, "offset_seconds": int(offset), "reversion_bps": float(value)}
            )
        # Further adverse movement (continuation of the trigger move) after the trigger.
        for horizon_seconds in config["adverse_excursion_horizons_seconds"]:
            upper = np.searchsorted(seconds, start + horizon_seconds, side="right")
            lower = np.searchsorted(seconds, start, side="right")
            if upper <= lower or start + horizon_seconds > seconds[-1]:
                value = np.nan
            elif direction < 0:
                value = 10_000.0 * np.log(price / chunk["low"][lower:upper].min())
            else:
                value = 10_000.0 * np.log(chunk["high"][lower:upper].max() / price)
            rows[position][f"adverse_excursion_{horizon_seconds}s_bps"] = float(max(value, 0.0)) if np.isfinite(value) else float("nan")

        entries = {f"delay_{s}s": start + int(s) for s in config["entry_delays_seconds"]}
        entries["trigger"] = start
        lower = np.searchsorted(seconds, start, side="right")
        limit = np.searchsorted(seconds, start + config["maximum_entry_wait_seconds"], side="right")
        closes = chunk["close"][lower:limit]
        times = seconds[lower:limit]
        for quiet in config["exhaustion_quiet_seconds"]:
            entries[f"exhaustion_{quiet}s"] = None
        if len(times):
            # Exhaustion: the first second at which the trigger-direction extreme
            # has not been extended for `quiet` seconds.
            running = (
                np.minimum.accumulate(closes) if direction < 0 else np.maximum.accumulate(closes)
            )
            new_extreme = np.concatenate([[True], running[1:] != running[:-1]])
            last_extreme_time = np.maximum.accumulate(np.where(new_extreme, times, times[0]))
            for quiet in config["exhaustion_quiet_seconds"]:
                quiet_enough = times - last_extreme_time >= quiet
                if quiet_enough.any():
                    entries[f"exhaustion_{quiet}s"] = int(times[np.argmax(quiet_enough)])
        for rule, entry_time in entries.items():
            if entry_time is None:
                continue
            entry_price = price_at(chunk, np.array([entry_time]))[0]
            spread_flips = window_sum(chunk, cumulative["flips"], np.array([entry_time]), spread_window)[0]
            spread_gap = window_sum(chunk, cumulative["flip_gap_sum_bps"], np.array([entry_time]), spread_window)[0]
            for holding in config["holding_horizons_seconds"]:
                exit_price = price_at(chunk, np.array([entry_time + holding]))[0]
                trade_rows.append(
                    {
                        "trigger_second": start,
                        "rule": rule,
                        "entry_wait_seconds": int(entry_time - start),
                        "holding_seconds": int(holding),
                        "entry_vs_trigger_bps": float(-direction * 10_000.0 * np.log(entry_price / price)),
                        "gross_bps": float(-direction * 10_000.0 * np.log(exit_price / entry_price)),
                        "entry_spread_proxy_bps": float(spread_gap / max(spread_flips, 1)),
                    }
                )
    return rows, path_rows, trade_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "alpha-search" / "configs" / "stress_episodes_v1.json",
    )
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-candidates", action="store_true")
    parser.add_argument("--candidates-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.shared_data_root
    output = args.output_dir or (REPO / "alpha-search" / "reports" / config["experiment_id"])
    output.mkdir(parents=True, exist_ok=True)
    months = list(iter_months(config["first_month"], config["last_month"]))

    candidate_path = output / "candidates.parquet"
    if args.reuse_candidates and candidate_path.exists():
        candidates = pl.read_parquet(candidate_path)
    else:
        parts = []
        for month in months:
            parts.append(candidates_for_month(root, config, month))
            print(f"pass 1 {month}: {parts[-1].height} candidate seconds", flush=True)
        candidates = pl.concat(parts)
        candidates.write_parquet(candidate_path, compression="zstd")

    thresholds = {
        int(target): calibrate_threshold(candidates, config, target)
        for target in config["target_episodes_per_year"]
    }
    print(f"thresholds: {thresholds}", flush=True)
    if args.candidates_only:
        return
    open_interest = load_open_interest(root, config["symbol"])

    episode_rows, path_rows, trade_rows = [], [], []
    for target, threshold in thresholds.items():
        episodes = merge_episodes(candidates, threshold, config["episode_merge_gap_seconds"])
        episodes = episodes.with_columns(
            pl.lit(target).alias("target_per_year"), pl.lit(threshold).alias("threshold_z")
        )
        for month in months:
            start, end = month_seconds(month), month_seconds(next_month(month))
            selected = episodes.filter(
                (pl.col("trigger_second") >= start) & (pl.col("trigger_second") < end)
            )
            if selected.is_empty():
                continue
            rows, paths, trades = episode_features(
                root, config, month, selected, open_interest
            )
            episode_rows.extend(rows)
            path_rows.extend({**row, "target_per_year": target} for row in paths)
            trade_rows.extend({**row, "target_per_year": target} for row in trades)
        print(f"pass 2 target {target}/yr: {episodes.height} episodes", flush=True)

    first_test = month_seconds(config["first_test_month"])
    catalogue = pl.DataFrame(episode_rows, infer_schema_length=None).with_columns(
        pl.when(pl.col("trigger_second") < first_test)
        .then(pl.lit("calibration"))
        .otherwise(pl.lit("test"))
        .alias("period")
    )
    catalogue.write_parquet(output / "episodes.parquet", compression="zstd")
    catalogue.write_csv(output / "episodes.csv")
    pl.DataFrame(path_rows).write_parquet(output / "paths.parquet", compression="zstd")
    pl.DataFrame(trade_rows).write_parquet(output / "trades.parquet", compression="zstd")
    manifest = {
        "experiment_id": config["experiment_id"],
        "config": str(args.config.resolve()),
        "thresholds_z": {str(k): v for k, v in thresholds.items()},
        "holdout_months_excluded": config["holdout_months_excluded"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
