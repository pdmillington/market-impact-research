"""Price and volatility descriptives for the paper's Data section.

Source: the shared `timestamp_direction_v1` events (BTCUSDT perpetual). Daily highs
and lows use each event's min/max fill price, so they are exact trade extremes;
open/close are the first fill of the day's first event and the last fill of its
last event. Days are UTC.

Outputs (reports/redo/data_description/):
- daily.csv: daily OHLC, log return, high/low log range, volume, events, fills,
  5-minute realised variance (sum of squared 5-minute log returns of last prices);
- summary.csv: headline statistics for the primary sample, for 2021 vs 2022 onward,
  and for the paper's original sample (2025-01..2026-05) as a check on the text;
- by_year.csv: the same statistics by calendar year;
- variance_tests.csv: is 2021 more volatile than the rest? Brown-Forsythe test
  (median-centred Levene; robust to fat tails) by year and for 2021 vs rest, and a
  moving-block bootstrap interval for the 2021/rest daily-variance ratio (daily
  returns cluster in volatility, so an F test is not valid);
and figures/00_price_and_volatility.pdf in reports/redo/paper/.

Primary sample 2021-01..2026-08; 2019-10..2020-12 is computed for context.
Months from 2026-09 onward are refused (alpha-search P-006).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crypto-impact-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from crypto_market_data.layout import DataLayout  # noqa: E402
from crypto_market_data.months import iter_months  # noqa: E402

OUT = REPO / "crypto-impact-study" / "reports" / "redo" / "data_description"
FIGURES = REPO / "crypto-impact-study" / "reports" / "redo" / "paper" / "figures"
PRIMARY = ("2021-01-01", "2026-08-31")
PAPER = ("2025-01-01", "2026-05-31")
DAYS_PER_YEAR = 365          # crypto trades every day
BLOCK_DAYS = 20
BOOTSTRAP = 5000
INK, MUTED, BLUE, ORANGE = "#1f1f1f", "#6b6b6b", "#2a78d6", "#eb6834"


def month_frames(layout: DataLayout, month: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    events = pl.scan_parquet(layout.event_month("BTCUSDT", month)).with_columns(
        pl.from_epoch("timestamp_ms", time_unit="ms").alias("t"))
    daily = (events.group_by(pl.col("t").dt.truncate("1d").alias("day"))
             .agg(pl.col("first_price").sort_by("timestamp_ms").first().alias("open"),
                  pl.col("max_price").max().alias("high"), pl.col("min_price").min().alias("low"),
                  pl.col("last_price").sort_by("timestamp_ms").last().alias("close"),
                  pl.col("quantity").sum().alias("volume_btc"), pl.len().alias("events"),
                  pl.col("fill_count").sum().alias("fills"))
             .collect())
    five = (events.group_by(pl.col("t").dt.truncate("5m").alias("bucket"))
            .agg(pl.col("last_price").sort_by("timestamp_ms").last().alias("close"))
            .collect())
    return daily, five


def describe(d: pd.DataFrame) -> dict:
    r = d["log_return"].dropna()
    hi = d.loc[d["high"].idxmax()]
    lo = d.loc[d["low"].idxmin()]
    return {
        "first_day": d["day"].min().date(), "last_day": d["day"].max().date(), "days": len(d),
        "open_price": d["open"].iloc[0], "close_price": d["close"].iloc[-1],
        "high_price": hi["high"], "high_day": hi["day"].date(), "low_price": lo["low"], "low_day": lo["day"].date(),
        "median_daily_range_pct": d["range_pct"].median(), "p95_daily_range_pct": d["range_pct"].quantile(0.95),
        "p99_daily_range_pct": d["range_pct"].quantile(0.99), "max_daily_range_pct": d["range_pct"].max(),
        "max_range_day": d.loc[d["range_pct"].idxmax(), "day"].date(),
        "days_range_ge_10pct": int((d["range_pct"] >= 10).sum()),
        "share_days_range_ge_10pct": float((d["range_pct"] >= 10).mean()),
        "daily_return_mean_pct": 100 * r.mean(), "daily_return_sd_pct": 100 * r.std(),
        "daily_return_variance_pct2": (100 * r).var(),
        "annualised_vol_close_to_close_pct": 100 * r.std() * np.sqrt(DAYS_PER_YEAR),
        "annualised_realised_vol_5min_pct": 100 * np.sqrt(d["rv_5min"].mean() * DAYS_PER_YEAR),
        "median_daily_realised_vol_5min_pct": 100 * np.sqrt(d["rv_5min"]).median(),
        "daily_return_skew": stats.skew(r), "daily_return_excess_kurtosis": stats.kurtosis(r),
        "largest_daily_gain_pct": 100 * r.max(), "largest_daily_loss_pct": 100 * r.min(),
        "days_abs_return_ge_10pct": int((100 * r.abs() >= 10).sum()),
        "median_daily_volume_btc": d["volume_btc"].median(), "median_daily_events": d["events"].median(),
    }


def block_bootstrap_ratio(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    def resample(x):
        starts = rng.integers(0, len(x) - BLOCK_DAYS + 1, size=int(np.ceil(len(x) / BLOCK_DAYS)))
        return np.concatenate([x[s:s + BLOCK_DAYS] for s in starts])[: len(x)]
    return np.array([resample(a).var(ddof=1) / resample(b).var(ddof=1) for _ in range(BOOTSTRAP)])


def figure(d: pd.DataFrame) -> None:
    p = d.loc[(d["day"] >= PRIMARY[0]) & (d["day"] <= PRIMARY[1])].set_index("day")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1]})
    for ax in axes:
        ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"), color=MUTED, alpha=0.10, lw=0)
        ax.axvspan(pd.Timestamp(PAPER[0]), pd.Timestamp(PAPER[1]), color=BLUE, alpha=0.08, lw=0)
        ax.grid(color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].fill_between(p.index, p["low"], p["high"], color=BLUE, alpha=0.35, lw=0, label="daily high–low")
    axes[0].plot(p.index, p["close"], color=INK, lw=1.0, label="daily close")
    axes[0].set_yscale("log")
    axes[0].set_yticks([20000, 30000, 50000, 70000, 100000, 125000])
    axes[0].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    axes[0].get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("BTCUSDT price (log scale)")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    axes[0].text(pd.Timestamp("2021-07-01"), 0.97, "2021", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", color=MUTED, fontsize=9)
    axes[0].text(pd.Timestamp("2025-09-15"), 0.97, "paper's original sample", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="top", color=BLUE, fontsize=9)
    axes[1].bar(p.index, p["range_pct"], width=1.0, color=INK, lw=0)
    axes[1].axhline(10, color=ORANGE, lw=1.0, ls="--")
    axes[1].text(p.index[-1], 10.5, "10%", color=ORANGE, fontsize=8, ha="right", va="bottom")
    axes[1].set_ylabel("Daily high/low range (%)")
    rv = 100 * np.sqrt(p["rv_5min"].rolling(30, min_periods=20).mean() * DAYS_PER_YEAR)
    axes[2].plot(p.index, rv, color=INK, lw=1.2)
    axes[2].set_ylabel("Realised volatility\n(5-min, 30-day, annualised %)")
    axes[2].set_ylim(0, None)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(FIGURES / "00_price_and_volatility.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-10")
    parser.add_argument("--end", default="2026-08")
    args = parser.parse_args()
    if args.end > "2026-08":
        raise SystemExit("Months after 2026-08 are reserved for the alpha-search P-006 test.")
    OUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    layout = DataLayout(REPO / "shared-data")
    dailies, fives = [], []
    for month in iter_months(args.start, args.end):
        d, f = month_frames(layout, month)
        dailies.append(d); fives.append(f)
        print(month, flush=True)
    daily = pl.concat(dailies).sort("day").to_pandas()
    five = pl.concat(fives).sort("bucket").to_pandas()
    five["r2"] = np.log(five["close"]).diff() ** 2
    five["day"] = five["bucket"].dt.floor("1D")
    daily = daily.merge(five.groupby("day")["r2"].sum().rename("rv_5min").reset_index(), on="day", how="left")
    daily["log_return"] = np.log(daily["close"]).diff()
    daily["range_pct"] = 100 * np.log(daily["high"] / daily["low"])
    daily["year"] = daily["day"].dt.year
    # Calendar completeness check.
    full = pd.date_range(daily["day"].min(), daily["day"].max(), freq="D")
    missing = full.difference(daily["day"])
    daily.to_csv(OUT / "daily.csv", index=False)

    def window(start, end):
        return daily.loc[(daily["day"] >= start) & (daily["day"] <= end)]

    primary = window(*PRIMARY)
    summary = {"primary_2021_01_2026_08": describe(primary),
               "year_2021": describe(window("2021-01-01", "2021-12-31")),
               "2022_01_2026_08": describe(window("2022-01-01", PRIMARY[1])),
               "paper_sample_2025_01_2026_05": describe(window(*PAPER)),
               "early_2019_10_2020_12": describe(window("2019-10-01", "2020-12-31"))}
    summary = pd.DataFrame(summary)
    summary.loc["missing_calendar_days"] = len(missing)
    summary.to_csv(OUT / "summary.csv")
    by_year = pd.DataFrame({str(y): describe(g) for y, g in daily.groupby("year")}).T
    by_year.to_csv(OUT / "by_year.csv")

    # Is 2021 more volatile? Daily log returns, primary sample.
    r = primary.dropna(subset=["log_return"])
    a = r.loc[r["year"] == 2021, "log_return"].to_numpy()
    b = r.loc[r["year"] > 2021, "log_return"].to_numpy()
    rng = np.random.default_rng(20261006)
    boot = block_bootstrap_ratio(a, b, rng)
    tests = [
        {"test": "Brown-Forsythe, 2021 vs 2022..2026-08 (daily log returns)",
         "statistic": stats.levene(a, b, center="median").statistic, "p_value": stats.levene(a, b, center="median").pvalue,
         "variance_ratio_2021_over_rest": a.var(ddof=1) / b.var(ddof=1),
         "ratio_block_bootstrap_2_5": np.quantile(boot, 0.025), "ratio_block_bootstrap_97_5": np.quantile(boot, 0.975),
         "block_days": BLOCK_DAYS, "replicates": BOOTSTRAP},
        {"test": "Brown-Forsythe, equal variance across years 2021..2026",
         "statistic": stats.levene(*[g["log_return"].to_numpy() for _, g in r.groupby("year")], center="median").statistic,
         "p_value": stats.levene(*[g["log_return"].to_numpy() for _, g in r.groupby("year")], center="median").pvalue},
    ]
    rv_a = r.loc[r["year"] == 2021, "rv_5min"].to_numpy()
    rv_b = r.loc[r["year"] > 2021, "rv_5min"].to_numpy()
    boot_rv = np.array([
        np.concatenate([rv_a[s:s + BLOCK_DAYS] for s in rng.integers(0, len(rv_a) - BLOCK_DAYS + 1, int(np.ceil(len(rv_a) / BLOCK_DAYS)))])[: len(rv_a)].mean()
        / np.concatenate([rv_b[s:s + BLOCK_DAYS] for s in rng.integers(0, len(rv_b) - BLOCK_DAYS + 1, int(np.ceil(len(rv_b) / BLOCK_DAYS)))])[: len(rv_b)].mean()
        for _ in range(BOOTSTRAP)])
    tests.append({"test": "Mean 5-min realised variance, 2021 / 2022..2026-08",
                  "variance_ratio_2021_over_rest": rv_a.mean() / rv_b.mean(),
                  "ratio_block_bootstrap_2_5": np.quantile(boot_rv, 0.025), "ratio_block_bootstrap_97_5": np.quantile(boot_rv, 0.975),
                  "block_days": BLOCK_DAYS, "replicates": BOOTSTRAP})
    pd.DataFrame(tests).to_csv(OUT / "variance_tests.csv", index=False)

    figure(daily)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    print(summary.to_string())
    print(by_year[["days", "open_price", "close_price", "high_price", "low_price", "median_daily_range_pct",
                   "p99_daily_range_pct", "days_range_ge_10pct", "annualised_vol_close_to_close_pct",
                   "annualised_realised_vol_5min_pct", "daily_return_excess_kurtosis"]].to_string())
    print(pd.DataFrame(tests).to_string())
    print("missing calendar days:", list(missing.date))


if __name__ == "__main__":
    main()
