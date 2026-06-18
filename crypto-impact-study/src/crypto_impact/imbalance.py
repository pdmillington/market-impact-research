"""Fixed-window volume imbalance calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


AUDIT_COLUMNS = [
    "timestamp_start",
    "timestamp_end",
    "signed_imbalance",
    "abs_signed_imbalance",
    "gross_volume",
    "participation_ratio",
    "start_price",
    "end_price",
    "log_return",
    "signed_impact",
    "abs_log_return",
    "n_trades",
]


def compute_window_imbalance(trades: pd.DataFrame, window: str) -> pd.DataFrame:
    """Aggregate signed trade data into fixed time windows.

    The returned frame keeps legacy aliases (``q_t``, ``absolute_volume``,
    ``trade_count``) while exposing clearer methodology fields used by the
    current analysis.
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
        signed_imbalance=("signed_volume", "sum"),
        gross_volume=("quantity", "sum"),
        start_price=("price", "first"),
        end_price=("price", "last"),
        n_trades=("price", "count"),
    )
    windows = windows.dropna(subset=["start_price", "end_price"])
    windows["abs_signed_imbalance"] = windows["signed_imbalance"].abs()
    windows["participation_ratio"] = windows["abs_signed_imbalance"] / windows["gross_volume"]
    windows["log_return"] = np.log(windows["end_price"] / windows["start_price"])
    windows["signed_impact"] = np.sign(windows["signed_imbalance"]) * windows["log_return"]
    windows["abs_log_return"] = windows["log_return"].abs()

    # Backward-compatible aliases for earlier notebooks and tests.
    windows["q_t"] = windows["signed_imbalance"]
    windows["absolute_volume"] = windows["gross_volume"]
    windows["trade_count"] = windows["n_trades"]
    return windows


def filter_valid_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Keep windows valid for fixed-window impact analysis."""

    required = {"gross_volume", "signed_imbalance", "participation_ratio"}
    missing = required.difference(windows.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    valid = windows.loc[
        (windows["gross_volume"] > 0)
        & (windows["signed_imbalance"] != 0)
        & windows["participation_ratio"].notna()
        & np.isfinite(windows["participation_ratio"])
    ].copy()
    return valid


def filter_nonzero_imbalance(windows: pd.DataFrame) -> pd.DataFrame:
    """Keep valid windows with nonzero signed imbalance.

    This wrapper preserves the original public function name while applying the
    richer validity filters introduced for participation-ratio analysis.
    """

    if "signed_imbalance" in windows.columns:
        return filter_valid_windows(windows)
    if "q_t" not in windows.columns:
        raise KeyError("Expected a 'signed_imbalance' or 'q_t' column.")
    return windows.loc[windows["q_t"] != 0].copy()


def make_audit_sample(windows: pd.DataFrame, window: str | None = None, n_rows: int = 20) -> pd.DataFrame:
    """Return the first valid windows or bars with calculation fields for manual audit."""

    valid = filter_valid_windows(windows)
    audit = valid.head(n_rows).copy()
    if "timestamp_start" not in audit.columns:
        audit.insert(0, "timestamp_start", audit.index)
    if "timestamp_end" not in audit.columns:
        if window is None:
            audit.insert(1, "timestamp_end", audit.index)
        else:
            audit.insert(1, "timestamp_end", audit.index + pd.tseries.frequencies.to_offset(window))
    return audit[AUDIT_COLUMNS]
