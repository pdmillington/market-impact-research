"""Known unrecoverable gaps in the trade archives (config/btcusdt_data_gaps.csv).

A bar or block whose time span covers a gap contains trades we do not observe:
its volume, imbalance and return are incomplete. Bar builders exclude such bars.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


GAPS_FILE = Path(__file__).resolve().parents[2] / "config" / "btcusdt_data_gaps.csv"


@lru_cache(maxsize=None)
def load_data_gaps(symbol: str = "BTCUSDT") -> tuple[tuple[int, int], ...]:
    """(last observed ms before the gap, first observed ms after it) for each gap."""

    gaps = pd.read_csv(GAPS_FILE, comment="#")
    gaps = gaps.loc[gaps["symbol"] == symbol]
    return tuple(zip(gaps["gap_start_ms"].astype("int64"), gaps["gap_end_ms"].astype("int64")))


def spans_data_gap(start_ms: np.ndarray, end_ms: np.ndarray, symbol: str = "BTCUSDT") -> np.ndarray:
    """True where [start_ms, end_ms] covers a known gap."""

    start_ms, end_ms = np.asarray(start_ms, dtype="int64"), np.asarray(end_ms, dtype="int64")
    flag = np.zeros(start_ms.shape, dtype=bool)
    for gap_start, gap_end in load_data_gaps(symbol):
        flag |= (start_ms <= gap_start) & (end_ms >= gap_end)
    return flag
