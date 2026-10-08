"""Exploratory summaries for monthly Binance trade Parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections.abc import Sequence

from .data_gaps import load_data_gaps, spans_data_gap


MONTHLY_COLUMNS = ["price", "qty", "time_str", "is_buyer_maker"]
# BTCUSDT perpetual quantities trade in 0.001 BTC lots. Bar volumes are rounded to
# this precision so that floating-point remainders of offsetting fills (e.g. a net
# imbalance of 1e-14 BTC) become exact zeros instead of spurious tiny values, which
# would otherwise produce extreme logs (ln 1e-14 = -32).
QUANTITY_DECIMALS = 3


def _epoch_ms(times) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(times, utc=True)).as_unit("ms").asi8


def _round_bar_volumes(bars: pd.DataFrame) -> pd.DataFrame:
    for column in ("gross_volume", "signed_imbalance"):
        bars[column] = bars[column].round(QUANTITY_DECIMALS)
    return bars



def load_monthly_trades(path: Path) -> pd.DataFrame:
    """Load the columns required for time-bar exploration."""

    trades = pd.read_parquet(path, columns=MONTHLY_COLUMNS)
    trades = trades.rename(
        columns={"qty": "quantity", "is_buyer_maker": "buyer_maker"}
    )
    trades["timestamp"] = pd.to_datetime(trades.pop("time_str"), utc=True)
    trades["signed_volume"] = np.where(
        trades["buyer_maker"].astype(bool),
        -trades["quantity"],
        trades["quantity"],
    )
    return trades.set_index("timestamp").sort_index()

def make_trade_bars(
        trades: pd.DataFrame, 
        trades_per_bar: int = 4_000,
        drop_incomplete: bool = True,
        ) -> pd.DataFrame:
    """Aggregate trades into non-overlapping fixed trade count bars."""

    required = {"price", "quantity", "signed_volume"}
    missing = required.difference(trades.columns)
    
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
        
    if trades_per_bar <= 0:
        raise ValueError("trades_per_bar must be positive")

    if not isinstance(trades.index, pd.DatetimeIndex):
            raise TypeError(
                "trades must have a DatetimeIndex. "
                f"Received {type(trades.index).__name__}."
            )

    ordered = trades.sort_index().copy()
    ordered['timestamp'] = ordered.index
    
    group_id = np.arange(len(ordered)) // trades_per_bar
    
    bars = ordered.groupby(group_id, sort=False).agg(
        start_time=("timestamp", "first"),
        end_time=("timestamp", "last"),
        start_price=("price", "first"),
        end_price=("price", "last"),
        high_price=("price", "max"),
        low_price=("price", "min"),
        gross_volume=("quantity", "sum"),
        signed_imbalance=("signed_volume", "sum"),
        trade_count=("price", "size"),
    )
    
    if drop_incomplete:
        bars = bars.loc[bars["trade_count"] == trades_per_bar].copy()
    bars = _round_bar_volumes(bars)
    
    bars["start_time"] = pd.to_datetime(bars["start_time"], utc=True)
    bars["end_time"] = pd.to_datetime(bars["end_time"], utc=True)
    # Bars covering a known gap in the trade archives contain unobserved trades.
    bars = bars.loc[~spans_data_gap(_epoch_ms(bars["start_time"]), _epoch_ms(bars["end_time"]))].copy()

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


def make_time_bars(trades: pd.DataFrame, window: str = "5min") -> pd.DataFrame:
    """Aggregate trades into non-overlapping fixed-length time bars."""

    required = {"price", "quantity", "signed_volume"}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    bars = trades.resample(window).agg(
        start_price=("price", "first"),
        end_price=("price", "last"),
        high_price=("price", "max"),
        low_price=("price", "min"),
        gross_volume=("quantity", "sum"),
        signed_imbalance=("signed_volume", "sum"),
        trade_count=("price", "size"),
    )
    bars = bars.loc[bars["trade_count"] > 0].copy()
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




def build_fixed_event_bars(
    files: Sequence[Path],
    *,
    events_per_bar: int,
) -> pd.DataFrame:
    """Build continuous non-overlapping fixed-event bars from monthly files.

    Incomplete events at the end of each file are carried into the following
    file, so bar boundaries do not restart at calendar-month boundaries. Any
    final incomplete bar at the end of the complete sample is discarded.
    """

    if events_per_bar <= 0:
        raise ValueError("events_per_bar must be positive.")

    paths = [Path(path) for path in files]
    if not paths:
        raise ValueError("At least one input file is required.")

    bar_frames: list[pd.DataFrame] = []
    remainder: pd.DataFrame | None = None

    for path in paths:
        events = load_monthly_trades(path)

        if remainder is not None and not remainder.empty:
            events = pd.concat(
                [remainder, events]
            ).sort_index()

        complete_rows = (
            len(events) // events_per_bar
        ) * events_per_bar

        complete_events = events.iloc[:complete_rows]
        remainder = events.iloc[complete_rows:].copy()

        if complete_events.empty:
            continue

        file_bars = make_trade_bars(
            complete_events,
            trades_per_bar=events_per_bar,
            drop_incomplete=True,
        )

        file_bars["source_file"] = path.name
        bar_frames.append(file_bars)

    if not bar_frames:
        raise ValueError(
            f"No complete {events_per_bar:,}-event bars "
            "could be constructed."
        )

    bars = pd.concat(
        bar_frames,
        axis=0,
    ).sort_index()

    bars["events_per_bar"] = events_per_bar

    if not bars.index.is_unique:
        raise ValueError(
            "Constructed bar start times are not unique."
        )

    return bars

def _central_limit(values: pd.Series, quantile: float = 0.995) -> float:
    return float(values.abs().quantile(quantile))


def plot_distributions(bars: pd.DataFrame, output_path: Path) -> Path:
    """Plot six core five-minute bar distributions."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    returns = bars["log_return_bps"]
    return_limit = _central_limit(returns)
    axes[0, 0].hist(returns.clip(-return_limit, return_limit), bins=80)
    axes[0, 0].set_title("Log return (central 99.5%)")
    axes[0, 0].set_xlabel("basis points")

    axes[0, 1].hist(np.log10(bars["gross_volume"]), bins=80)
    axes[0, 1].set_title("Gross traded volume")
    axes[0, 1].set_xlabel("log10(BTC)")

    imbalance = bars["signed_imbalance"]
    imbalance_limit = _central_limit(imbalance)
    axes[0, 2].hist(
        imbalance.clip(-imbalance_limit, imbalance_limit), bins=80
    )
    axes[0, 2].set_title("Net signed volume (central 99.5%)")
    axes[0, 2].set_xlabel("BTC")

    positive_abs = bars.loc[
        bars["abs_signed_imbalance"] > 0, "abs_signed_imbalance"
    ]
    axes[1, 0].hist(np.log10(positive_abs), bins=80)
    axes[1, 0].set_title("Absolute net signed volume")
    axes[1, 0].set_xlabel("log10(BTC)")

    axes[1, 1].hist(bars["absolute_imbalance_ratio"], bins=80)
    axes[1, 1].set_title("Absolute imbalance ratio")
    axes[1, 1].set_xlabel("|net signed volume| / gross volume")

    axes[1, 2].hist(np.log10(bars["trade_count"]), bins=80)
    axes[1, 2].set_title("Trade count")
    axes[1, 2].set_xlabel("log10(count)")

    for ax in axes.flat:
        ax.set_ylabel("5-minute windows")
        ax.grid(alpha=0.2)
    fig.suptitle("BTCUSDT five-minute distributions")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_intraday_profile(bars: pd.DataFrame, output_path: Path) -> Path:
    """Plot median activity and volatility by UTC time of day."""

    minute = bars.index.hour * 60 + bars.index.minute
    profile = bars.assign(minute_of_day=minute).groupby("minute_of_day").agg(
        median_gross_volume=("gross_volume", "median"),
        median_abs_imbalance=("abs_signed_imbalance", "median"),
        median_range_bps=("high_low_range_bps", "median"),
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    columns = [
        ("median_gross_volume", "Median gross volume", "BTC"),
        ("median_abs_imbalance", "Median absolute net signed volume", "BTC"),
        ("median_range_bps", "Median high-low range", "basis points"),
    ]
    for ax, (column, title, ylabel) in zip(axes, columns, strict=True):
        ax.plot(profile.index / 60, profile[column])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("UTC hour")
    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(0, 25, 3))
    fig.suptitle("BTCUSDT intraday profile (five-minute windows)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_imbalance_return(bars: pd.DataFrame, output_path: Path) -> Path:
    """Plot the dense central relationship between imbalance and return."""

    imbalance_limit = _central_limit(bars["signed_imbalance"])
    return_limit = _central_limit(bars["log_return_bps"])
    central = bars.loc[
        bars["signed_imbalance"].between(-imbalance_limit, imbalance_limit)
        & bars["log_return_bps"].between(-return_limit, return_limit)
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    density = ax.hexbin(
        central["signed_imbalance"],
        central["log_return_bps"],
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    fig.colorbar(density, ax=ax, label="window count (log colour scale)")
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_title("Five-minute return versus net signed volume (central 99.5%)")
    ax.set_xlabel("net signed volume (BTC)")
    ax.set_ylabel("log return (basis points)")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_monthly_eda(input_path: Path, output_dir: Path, window: str = "5min") -> dict[str, Path]:
    """Create reusable bars, statistics, and plots for one monthly file."""

    trades = load_monthly_trades(input_path)
    bars = make_time_bars(trades, window)
    stem = input_path.stem.replace("-trades-", "_") + f"_{window}"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "bars": output_dir / f"{stem}_bars.parquet",
        "summary": output_dir / f"{stem}_summary.csv",
        "distributions": output_dir / f"{stem}_distributions.png",
        "intraday": output_dir / f"{stem}_intraday_profile.png",
        "imbalance_return": output_dir / f"{stem}_imbalance_return.png",
    }
    bars.to_parquet(paths["bars"])
    bars.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.to_csv(
        paths["summary"]
    )
    plot_distributions(bars, paths["distributions"])
    plot_intraday_profile(bars, paths["intraday"])
    plot_imbalance_return(bars, paths["imbalance_return"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", default="5min")
    args = parser.parse_args()
    for name, path in run_monthly_eda(
        args.input_path, args.output_dir, args.window
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
