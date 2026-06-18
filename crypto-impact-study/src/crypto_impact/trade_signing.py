"""Trade signing for Binance aggregate trade data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_trade_sign(trades: pd.DataFrame) -> pd.Series:
    """Infer aggressor direction from Binance's ``buyer_maker`` flag.

    Binance aggregate trades report whether the buyer was the maker. If the
    buyer was the maker, the aggressor crossed the spread to sell, so the trade
    is seller initiated and receives sign -1. Otherwise it receives sign +1.
    """

    if "buyer_maker" not in trades.columns:
        raise KeyError("Expected a 'buyer_maker' column.")
    return pd.Series(
        np.where(trades["buyer_maker"].astype(bool), -1, 1),
        index=trades.index,
        name="sign",
    )


def add_signed_volume(trades: pd.DataFrame) -> pd.DataFrame:
    """Return trades with ``sign`` and ``signed_volume`` columns."""

    required = {"buyer_maker", "quantity"}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    signed = trades.copy()
    signed["sign"] = infer_trade_sign(signed)
    signed["signed_volume"] = signed["sign"] * signed["quantity"].astype(float)
    return signed
