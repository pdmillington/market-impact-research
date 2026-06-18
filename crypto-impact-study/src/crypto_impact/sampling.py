"""Sampling schemes for fixed-window and volume-bar impact observations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_impact.imbalance import filter_valid_windows


def _finalize_observations(windows: pd.DataFrame) -> pd.DataFrame:
    """Add derived impact fields and legacy aliases to sampled observations."""

    windows = windows.dropna(subset=["start_price", "end_price"]).copy()
    windows["abs_signed_imbalance"] = windows["signed_imbalance"].abs()
    windows["participation_ratio"] = windows["abs_signed_imbalance"] / windows["gross_volume"]
    windows["log_return"] = np.log(windows["end_price"] / windows["start_price"])
    windows["signed_impact"] = np.sign(windows["signed_imbalance"]) * windows["log_return"]
    windows["abs_log_return"] = windows["log_return"].abs()
    windows["q_t"] = windows["signed_imbalance"]
    windows["absolute_volume"] = windows["gross_volume"]
    windows["trade_count"] = windows["n_trades"]
    return filter_valid_windows(windows)


def aggregate_time_bars(trades: pd.DataFrame, window_length: str, window_step: str | None = None) -> pd.DataFrame:
    """Aggregate signed trades into time bars, allowing overlapping rolling windows."""

    required = {"price", "quantity", "signed_volume"}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if not isinstance(trades.index, pd.DatetimeIndex):
        raise TypeError("Trades must be indexed by a pandas DatetimeIndex.")

    step = window_step or window_length
    window_offset = pd.tseries.frequencies.to_offset(window_length)
    step_offset = pd.tseries.frequencies.to_offset(step)
    ordered = trades.sort_index()

    if step_offset == window_offset:
        grouped = ordered.resample(window_length)
        windows = grouped.agg(
            signed_imbalance=("signed_volume", "sum"),
            gross_volume=("quantity", "sum"),
            start_price=("price", "first"),
            end_price=("price", "last"),
            n_trades=("price", "count"),
        )
        return _finalize_observations(windows)

    timestamps = ordered.index
    starts = pd.date_range(timestamps.min().floor(step), timestamps.max(), freq=step, tz=timestamps.tz)
    ends = starts + window_offset
    ts_ns = timestamps.view("int64")
    signed_cumsum = np.r_[0.0, ordered["signed_volume"].to_numpy(dtype=float).cumsum()]
    volume_cumsum = np.r_[0.0, ordered["quantity"].to_numpy(dtype=float).cumsum()]
    prices = ordered["price"].to_numpy(dtype=float)
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for start, end in zip(starts, ends, strict=False):
        left = int(np.searchsorted(ts_ns, start.value, side="left"))
        right = int(np.searchsorted(ts_ns, end.value, side="left"))
        if right <= left:
            continue
        rows.append(
            {
                "timestamp_start": start,
                "timestamp_end": end,
                "signed_imbalance": float(signed_cumsum[right] - signed_cumsum[left]),
                "gross_volume": float(volume_cumsum[right] - volume_cumsum[left]),
                "start_price": float(prices[left]),
                "end_price": float(prices[right - 1]),
                "n_trades": int(right - left),
            }
        )
    windows = pd.DataFrame(rows)
    if windows.empty:
        return pd.DataFrame()
    return _finalize_observations(windows.set_index("timestamp_start"))


def aggregate_volume_bars(trades: pd.DataFrame, volume_bar_size: float) -> pd.DataFrame:
    """Aggregate signed trades into bars with at least ``volume_bar_size`` gross volume."""

    if volume_bar_size <= 0:
        raise ValueError("volume_bar_size must be positive.")
    required = {"price", "quantity", "signed_volume"}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    ordered = trades.sort_index()
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    start_time: pd.Timestamp | None = None
    start_price = np.nan
    signed_sum = 0.0
    volume_sum = 0.0
    n_trades = 0

    for timestamp, row in ordered.iterrows():
        if n_trades == 0:
            start_time = timestamp
            start_price = float(row["price"])
        signed_sum += float(row["signed_volume"])
        volume_sum += float(row["quantity"])
        n_trades += 1
        if volume_sum >= volume_bar_size:
            rows.append(
                {
                    "timestamp_start": start_time,
                    "timestamp_end": timestamp,
                    "signed_imbalance": signed_sum,
                    "gross_volume": volume_sum,
                    "start_price": start_price,
                    "end_price": float(row["price"]),
                    "n_trades": n_trades,
                }
            )
            start_time = None
            start_price = np.nan
            signed_sum = 0.0
            volume_sum = 0.0
            n_trades = 0

    windows = pd.DataFrame(rows)
    if windows.empty:
        return pd.DataFrame()
    return _finalize_observations(windows.set_index("timestamp_start"))
