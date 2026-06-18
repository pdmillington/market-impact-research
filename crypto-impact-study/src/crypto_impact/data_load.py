"""Load Binance aggregate trade CSV data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


AGG_TRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "buyer_maker",
    "best_price_match",
]


def parse_bool(value: object) -> bool:
    """Parse Binance boolean fields from strings or native bools."""

    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_agg_trades(path: Path) -> pd.DataFrame:
    """Load one Binance aggregate trade CSV or zipped CSV file."""

    trades = pd.read_csv(path, names=AGG_TRADE_COLUMNS, header=None, compression="infer")
    trades = trades.loc[trades["agg_trade_id"] != "agg_trade_id"].copy()
    trades["price"] = trades["price"].astype(float)
    trades["quantity"] = trades["quantity"].astype(float)
    trades["transact_time"] = pd.to_datetime(trades["transact_time"], unit="ms", utc=True)
    trades["buyer_maker"] = trades["buyer_maker"].map(parse_bool)
    return trades.set_index("transact_time").sort_index()


def load_agg_trade_files(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate multiple aggregate trade files."""

    if not paths:
        raise ValueError("No aggregate trade files were provided.")
    frames = [load_agg_trades(path) for path in sorted(paths)]
    return pd.concat(frames).sort_index()
