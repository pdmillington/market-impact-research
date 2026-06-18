"""Fixed-window volume imbalance calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_window_imbalance(trades: pd.DataFrame, window: str) -> pd.DataFrame:
    """Aggregate signed trade data into fixed time windows.

    Parameters
    ----------
    trades:
        Trade data indexed by timestamp and containing ``price``,
        ``quantity``, and ``signed_volume``.
    window:
        Pandas offset alias such as ``"1min"`` or ``"5min"``.
    """

    required = {"price", "quantity", "signed_volume"}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if not isinstance(trades.index, pd.DatetimeIndex):
        raise TypeError("Trades must be indexed by a pandas DatetimeIndex.")

    ordered = trades.sort_index()
    grouped = ordered.resample(window)
    windows = grouped.agg(
        q_t=("signed_volume", "sum"),
        absolute_volume=("quantity", "sum"),
        start_price=("price", "first"),
        end_price=("price", "last"),
        trade_count=("price", "count"),
    )
    windows = windows.dropna(subset=["start_price", "end_price"])
    windows["log_return"] = np.log(windows["end_price"] / windows["start_price"])
    windows["signed_impact"] = np.sign(windows["q_t"]) * windows["log_return"]
    return windows


def filter_nonzero_imbalance(windows: pd.DataFrame) -> pd.DataFrame:
    """Keep only windows with nonzero signed imbalance."""

    if "q_t" not in windows.columns:
        raise KeyError("Expected a 'q_t' column.")
    return windows.loc[windows["q_t"] != 0].copy()
