"""Shared construction of fixed-event bars and fixed-length time bars.

Every bar and block in the impact study is built here (notebooks, scripts and the
P&B block files alike), so construction rules live in one place:

- events are `timestamp_direction_v1` events (one per contiguous same-sign run of
  fills at a millisecond timestamp), priced at the run's last fill;
- fixed-event bars are consecutive, non-overlapping blocks of N events, continuous
  across months; a final incomplete block is discarded;
- bar gross and signed volume are rounded to the 0.001 BTC lot, so offsetting fills
  give exact zeros rather than floating-point remainders;
- bars covering a known unrecoverable gap in the trade archives
  (config/btcusdt_data_gaps.csv) carry `spans_data_gap`; by default they are dropped.
  Code that keeps their observed volume in trailing windows passes
  `drop_data_gaps=False` and excludes them after normalising.

Events come from either the monthly paper event files (`paper_events_v2`, 2021-01
onward; `paper_event_sources`) or the shared event lake (`lake_event_sources`, from
2019-09). Both give identical arrays for the same month.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from .data_gaps import load_data_gaps, spans_data_gap


QUANTITY_DECIMALS = 3
EventArrays = dict[str, np.ndarray]


def _epoch_ms(times) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ms").asi8


def _round_bar_volumes(bars: pd.DataFrame) -> pd.DataFrame:
    for column in ("gross_volume", "signed_imbalance"):
        bars[column] = bars[column].round(QUANTITY_DECIMALS)
    return bars


# ---------------------------------------------------------------------------
# Event sources: (label, arrays) pairs in time order.

def _arrays(time_ms: np.ndarray, price: np.ndarray, quantity: np.ndarray, sign: np.ndarray,
            **extra_prices: np.ndarray) -> EventArrays:
    """Event arrays; `extra_prices` (e.g. price_first, price_vwap) become extra start/end bar columns."""

    sign = sign.astype(np.int8)
    arrays = {"time_ms": time_ms.astype(np.int64), "price": price.astype(float),
              "quantity": quantity.astype(float), "signed_volume": sign * quantity.astype(float), "sign": sign}
    arrays.update({key: values.astype(float) for key, values in extra_prices.items()})
    return arrays


def read_paper_events(path: Path) -> EventArrays:
    """One monthly paper event file (price, qty, time_str, is_buyer_maker), in file order."""

    frame = pl.read_parquet(path, columns=["price", "qty", "time_str", "is_buyer_maker"])
    time_ms = frame["time_str"].dt.epoch("ms").to_numpy()
    if np.any(np.diff(time_ms) < 0):
        raise ValueError(f"{Path(path).name} is not sorted by time.")
    sign = np.where(frame["is_buyer_maker"].to_numpy(), -1, 1)
    return _arrays(time_ms, frame["price"].to_numpy(), frame["qty"].to_numpy(), sign)


def read_lake_events(root: Path, symbol: str, month: str, *, extra_prices: bool = False) -> EventArrays:
    """One month of shared `timestamp_direction_v1` events, in exchange order.

    The event price is the last fill. With `extra_prices`, the first fill
    (`price_first`) and the volume-weighted price (`price_vwap`) are carried too,
    for robustness checks of the event-price rule. Needs `crypto_market_data`
    (shared-methodology/src) on the import path.
    """

    from crypto_market_data.layout import DataLayout

    columns = ["timestamp_ms", "first_trade_id", "last_price", "quantity", "aggressor_sign"]
    if extra_prices:
        columns += ["first_price", "vwap"]
    events = (pl.scan_parquet(DataLayout(Path(root)).event_month(symbol, month))
              .select(columns).sort("timestamp_ms", "first_trade_id").collect())
    extra = ({"price_first": events["first_price"].to_numpy(), "price_vwap": events["vwap"].to_numpy()}
             if extra_prices else {})
    return _arrays(events["timestamp_ms"].to_numpy(), events["last_price"].to_numpy(),
                   events["quantity"].to_numpy(), events["aggressor_sign"].to_numpy(), **extra)


def paper_event_sources(files: Sequence[Path]) -> Iterator[tuple[str, EventArrays]]:
    for path in files:
        yield Path(path).name, read_paper_events(Path(path))


def lake_event_sources(root: Path, symbol: str, months: Iterable[str], *,
                       extra_prices: bool = False) -> Iterator[tuple[str, EventArrays]]:
    for month in months:
        yield month, read_lake_events(root, symbol, month, extra_prices=extra_prices)


# ---------------------------------------------------------------------------
# Fixed-event bars.

def _finish_event_bars(bars: pd.DataFrame, drop_data_gaps: bool = True) -> pd.DataFrame:
    """Rounding, gap flag/exclusion and derived columns shared by all event-bar builders."""

    bars = _round_bar_volumes(bars)

    bars["start_time"] = pd.to_datetime(bars["start_time"], utc=True)
    bars["end_time"] = pd.to_datetime(bars["end_time"], utc=True)
    # Bars covering a known gap in the trade archives contain unobserved trades.
    bars["spans_data_gap"] = spans_data_gap(_epoch_ms(bars["start_time"]), _epoch_ms(bars["end_time"]))
    if drop_data_gaps:
        bars = bars.loc[~bars["spans_data_gap"]].copy()

    bars["duration_seconds"] = (
        bars["end_time"] - bars["start_time"]
        ).dt.total_seconds()

    bars["log_return_bps"] = 10_000 * np.log(
        bars["end_price"] / bars["start_price"]
    )
    bars["abs_signed_imbalance"] = bars["signed_imbalance"].abs()
    bars["absolute_imbalance_ratio"] = (
        bars["abs_signed_imbalance"] / bars["gross_volume"]
    )
    bars["high_low_range_bps"] = 10_000 * np.log(
        bars["high_price"] / bars["low_price"]
    )
    return bars.set_index("start_time")


def _event_bars_from_arrays(arrays: EventArrays, n: int) -> pd.DataFrame:
    rows = len(arrays["time_ms"]) // n
    shaped = {key: values[: rows * n].reshape(rows, n) for key, values in arrays.items()}
    to_time = lambda ms: pd.to_datetime(ms, unit="ms", utc=True)
    price = shaped["price"]
    sign_sum = shaped["sign"].sum(axis=1, dtype=np.int64)
    extra = {}
    for key in (k for k in shaped if k.startswith("price_")):
        extra[f"start_{key}"] = shaped[key][:, 0]
        extra[f"end_{key}"] = shaped[key][:, -1]
    return pd.DataFrame({
        "start_time": to_time(shaped["time_ms"][:, 0]),
        "end_time": to_time(shaped["time_ms"][:, -1]),
        "start_price": price[:, 0],
        "end_price": price[:, -1],
        "high_price": price.max(axis=1),
        "low_price": price.min(axis=1),
        "gross_volume": shaped["quantity"].sum(axis=1),
        "signed_imbalance": shaped["signed_volume"].sum(axis=1),
        "trade_count": np.full(rows, n, dtype=np.int64),
        # Order-flow composition and the transaction-price pinning statistics.
        "sign_sum": sign_sum,
        "sign_bias": sign_sum / n,
        "price_change_fraction": (np.diff(price, axis=1) != 0).mean(axis=1) if n > 1 else np.full(rows, np.nan),
        "endpoint_price_changed": price[:, -1] != price[:, 0],
        **extra,
    })


def iter_event_bars(
    sources: Iterable[tuple[str, EventArrays]],
    sizes: Sequence[int],
    *,
    drop_data_gaps: bool = True,
) -> Iterator[tuple[str, dict[int, pd.DataFrame]]]:
    """Yield, for each source in turn, the bars of every size completed within it.

    Events left over at the end of a source are carried into the next one, so bars
    are continuous across months; the last incomplete bar of the sample is never
    yielded. Each source is read once whatever the number of sizes.
    """

    sizes = list(dict.fromkeys(int(n) for n in sizes))
    if not sizes or min(sizes) <= 0:
        raise ValueError("sizes must be positive.")
    carry: dict[int, EventArrays | None] = {n: None for n in sizes}
    last_time = None
    for label, arrays in sources:
        if len(arrays["time_ms"]) and last_time is not None and arrays["time_ms"][0] < last_time:
            raise ValueError(f"Event sources are not in time order at {label}.")
        if len(arrays["time_ms"]):
            last_time = int(arrays["time_ms"][-1])
        completed = {}
        length = len(arrays["time_ms"])
        for n in sizes:
            raw_parts = []
            rest = arrays
            pending = carry[n]
            if pending is not None and len(pending["time_ms"]):
                # Only the bar that straddles the source boundary is assembled by
                # concatenation; the rest is reshaped from views of `arrays`, so no
                # size keeps its own copy of the month.
                lead = n - len(pending["time_ms"])
                if lead > length:
                    carry[n] = {key: np.concatenate([pending[key], values]) for key, values in arrays.items()}
                    continue
                boundary = {key: np.concatenate([pending[key], values[:lead]]) for key, values in arrays.items()}
                raw_parts.append(_event_bars_from_arrays(boundary, n))
                rest = {key: values[lead:] for key, values in arrays.items()}
            complete = (len(rest["time_ms"]) // n) * n
            if complete:
                raw_parts.append(_event_bars_from_arrays({key: values[:complete] for key, values in rest.items()}, n))
            carry[n] = {key: values[complete:].copy() for key, values in rest.items()}
            if raw_parts:
                raw = pd.concat(raw_parts, ignore_index=True) if len(raw_parts) > 1 else raw_parts[0]
                bars = _finish_event_bars(raw, drop_data_gaps)
                bars["source_file"] = label
                completed[n] = bars
        yield label, completed


def build_event_bars(
    sources: Iterable[tuple[str, EventArrays]],
    sizes: Sequence[int],
    *,
    drop_data_gaps: bool = True,
) -> dict[int, pd.DataFrame]:
    """All fixed-event bars for several sizes, one read of each source."""

    sizes = list(dict.fromkeys(int(n) for n in sizes))
    frames: dict[int, list[pd.DataFrame]] = {n: [] for n in sizes}
    seen = False
    for _, completed in iter_event_bars(sources, sizes, drop_data_gaps=drop_data_gaps):
        seen = True
        for n, bars in completed.items():
            frames[n].append(bars)
    if not seen:
        raise ValueError("At least one input file is required.")
    result = {}
    for n in sizes:
        if not frames[n]:
            raise ValueError(f"No complete {n:,}-event bars could be constructed.")
        bars = pd.concat(frames[n], axis=0).sort_index()
        bars["events_per_bar"] = n
        if not bars.index.is_unique:
            raise ValueError("Constructed bar start times are not unique.")
        result[n] = bars
    return result


# ---------------------------------------------------------------------------
# Fixed-length time bars.

def _finish_time_bars(bars: pd.DataFrame, window: str) -> pd.DataFrame:
    """Rounding, gap exclusion and derived columns shared by all time-bar builders."""

    bars = _round_bar_volumes(bars)
    # Drop bars whose interval overlaps a known gap in the trade archives (partial data).
    start_ms = _epoch_ms(bars.index)
    end_ms = start_ms + int(pd.Timedelta(window).total_seconds() * 1000)
    overlaps = np.zeros(len(bars), dtype=bool)
    for gap_start, gap_end in load_data_gaps():
        overlaps |= (start_ms < gap_end) & (end_ms > gap_start)
    bars = bars.loc[~overlaps].copy()
    bars["log_return_bps"] = 10_000 * np.log(
        bars["end_price"] / bars["start_price"]
    )
    bars["abs_signed_imbalance"] = bars["signed_imbalance"].abs()
    bars["absolute_imbalance_ratio"] = (
        bars["abs_signed_imbalance"] / bars["gross_volume"]
    )
    bars["high_low_range_bps"] = 10_000 * np.log(
        bars["high_price"] / bars["low_price"]
    )
    return bars


def build_time_bars_from_sources(
    sources: Iterable[tuple[str, EventArrays]], windows: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Fixed-length time bars for several windows, one read of each source.

    Windows that divide a day align with UTC midnight, so month boundaries never
    split a bar.
    """

    frames: dict[str, list[pd.DataFrame]] = {window: [] for window in windows}
    for _, arrays in sources:
        events = pl.DataFrame({key: arrays[key] for key in ("time_ms", "price", "quantity", "signed_volume")})
        for window in windows:
            width = int(pd.Timedelta(window).total_seconds() * 1000)
            raw = (events.group_by((pl.col("time_ms") // width * width).alias("bucket"), maintain_order=True)
                   .agg(pl.col("price").first().alias("start_price"), pl.col("price").last().alias("end_price"),
                        pl.col("price").max().alias("high_price"), pl.col("price").min().alias("low_price"),
                        pl.col("quantity").sum().alias("gross_volume"),
                        pl.col("signed_volume").sum().alias("signed_imbalance"),
                        pl.len().cast(pl.Int64).alias("trade_count"))
                   .sort("bucket").to_pandas())
            raw.index = pd.DatetimeIndex(pd.to_datetime(raw.pop("bucket"), unit="ms", utc=True), name="timestamp")
            frames[window].append(_finish_time_bars(raw, window))
    return {window: pd.concat(parts).sort_index() for window, parts in frames.items()}
