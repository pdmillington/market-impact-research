"""E-008: rolling-window calibration of the activity surface, as registered.

Question: does a surface fitted on a trailing window describe the contemporaneous
block response (y = r / sigma_24h given q = |Q|/V and a = V/V_24h) better out of sample
than a fit on an expanding window from 2021-01? Not a return-prediction test.

Model: the headline surface of crypto-impact-study (shared alpha, beta, gamma; free
Q_N, R_N; N = 250..16,000), estimator crypto_impact.pb_surface.fit_activity_surface on
20 x 10 equal-count cells per N and side (at least 100 blocks per cell).

Implementation choices (recorded in the ledger run notes before any result):
- training blocks for test month m and window W: start month within W and end day
  before the first day of m; test blocks: start month m (finite y; gap blocks are
  excluded by the loader);
- bin edges recomputed per (window, month, N, side) from the training blocks;
- Amendment 1 (2026-10-29): adaptive grid n_nu = clamp(floor(sqrt(n / 200)), 2, 10),
  n_q = 2 n_nu per (window, month, N, side); an N with fewer than 8 cells in a
  window-month is dropped from that fit; a (month, N) pair is scored only when both
  windows of a comparison include it; warm starts only when the N set is unchanged;
- 8 optimiser starts in a window's first month, then 2 starts plus a warm start from
  that window's previous-month fit;
- prediction y_hat = s R_N F(q a^gamma / Q_N), 0 when q = 0;
- secondary: y clipped at the training window's 0.1% / 99.9% quantiles for that N;
- best-10-days removal via daily contributions c_d (see the ledger);
- month-block bootstrap, 2,000 resamples, seed 20261016.

Data: BTCUSDT pb_blocks_v1, 2021-01 to 2026-08. Nothing from 2026-09 onward (P-006).

Outputs (alpha-search/reports/e008_rolling_surface/): fits.csv, scales.csv,
monthly_scores.csv, daily_contributions.csv, summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "crypto-impact-study" / "src"), str(REPO / "crypto-impact-study" / "scripts"),
                str(REPO / "shared-methodology" / "src")]

import run_activity_surface as run  # noqa: E402
from crypto_impact.pb_collapse import pb_shape  # noqa: E402
from crypto_impact.pb_surface import aggregate_cells, fit_activity_surface, grid_cuts, monthly_cell_stats  # noqa: E402

OUT = REPO / "alpha-search" / "reports" / "e008_rolling_surface"
N_VALUES = (250, 500, 1000, 2000, 4000, 8000, 16000)
LAST_MONTH = "2026-08"                     # P-006: nothing from 2026-09 onward
FIRST_DATA = run.month_index("2021-01")
WINDOWS = {"expanding": None, "trailing_12": 12, "trailing_6": 6, "trailing_24": 24}
FIRST_TEST = {"expanding": "2022-01", "trailing_12": "2022-01", "trailing_6": "2022-01", "trailing_24": "2023-01"}
COMPARISONS = [  # (label, window, base, first test month)
    ("primary: trailing_12 vs expanding", "trailing_12", "expanding", "2022-01"),
    ("robustness: trailing_6 vs expanding", "trailing_6", "expanding", "2022-01"),
    ("robustness: trailing_24 vs expanding", "trailing_24", "expanding", "2023-01"),
    ("common months: trailing_12 vs expanding", "trailing_12", "expanding", "2023-01"),
]
N_BOOT, SEED = 2_000, 20261016
MIN_CELLS_PER_N = 8


def grid_size(n_blocks: int) -> tuple[int, int]:
    """Amendment 1: (n_q, n_nu) for a side with n_blocks training blocks."""

    n_nu = int(min(10, max(2, np.floor(np.sqrt(n_blocks / 200)))))
    return 2 * n_nu, n_nu


def month_start_day(m: int) -> int:
    """Days since 1970-01-01 of the first day of month index m (run.month_index convention)."""

    return int((np.datetime64(f"{m // 12}-{m % 12 + 1:02d}-01", "D") - np.datetime64("1970-01-01", "D")).astype(int))


def label(m: int) -> str:
    return f"{m // 12}-{m % 12 + 1:02d}"


def test_months(window: str) -> list[int]:
    return list(range(run.month_index(FIRST_TEST[window]), run.month_index(LAST_MONTH) + 1))


def training_mask(arr: dict, window: str, m: int) -> np.ndarray:
    first = FIRST_DATA if WINDOWS[window] is None else max(FIRST_DATA, m - WINDOWS[window])
    return (arr["month"] >= first) & (arr["end_day"] < month_start_day(m)) & np.isfinite(arr["y_vol"])


def load(n: int) -> dict:
    arr = run.load_arrays(REPO / "shared-data", n, "2020-12", "2021-01", LAST_MONTH)
    if arr["month"].max() > run.month_index(LAST_MONTH):
        raise RuntimeError("data after 2026-08 loaded (P-006)")
    return arr


def training_stats() -> dict:
    """(window, month) -> cell statistics over all N, from that window's training blocks."""

    stats = {(w, m): [] for w in WINDOWS for m in test_months(w)}
    for n in N_VALUES:
        arr = load(n)
        for (window, m) in stats:
            train = training_mask(arr, window, m)
            cuts = {}
            for s in (-1, 1):
                usable = train & (arr["side"] == s) & (arr["z"] > 0) & np.isfinite(arr["a"]) & (arr["a"] > 0)
                n_q, n_nu = grid_size(int(usable.sum()))
                cuts[s] = grid_cuts(arr["z"][usable], arr["a"][usable], n_q, n_nu)
            frame = monthly_cell_stats(np.zeros(int(train.sum()), dtype=np.int32), arr["side"][train], arr["z"][train],
                                       arr["a"][train], arr["y_vol"][train], cuts)
            frame.insert(0, "n_events", n)
            stats[(window, m)].append(frame)
        print(f"training cells: N = {n:,} done", flush=True)
        del arr
    return stats


def fit_windows(stats: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    fits, rows, scale_rows = {}, [], []
    for window in WINDOWS:
        previous = None
        for m in test_months(window):
            cells = aggregate_cells(pd.concat(stats[(window, m)], ignore_index=True))
            counts = cells.groupby("n_events").size()
            kept = sorted(counts.index[counts >= MIN_CELLS_PER_N])
            cells = cells.loc[cells["n_events"].isin(kept)].reset_index(drop=True)
            same_n = previous is not None and sorted(previous.scales["n_events"]) == kept
            fit = (fit_activity_surface(cells, n_starts=2, initial=previous) if same_n
                   else fit_activity_surface(cells, n_starts=8))
            fits[(window, m)] = previous = fit
            rows.append({"window": window, "test_month": label(m), "gamma": fit.gamma, "alpha": fit.alpha,
                         "beta": fit.beta, "objective": fit.objective, "converged": fit.converged, "cells": len(cells),
                         "n_included": ",".join(str(n) for n in kept), "warm_start": same_n})
            scale_rows.append(fit.scales.assign(window=window, test_month=label(m)))
        print(f"fits: {window} done", flush=True)
    return fits, pd.DataFrame(rows), pd.concat(scale_rows, ignore_index=True)


def score(fits: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly, daily = [], []
    for n in N_VALUES:
        arr = load(n)
        for (window, m), fit in fits.items():
            if n not in set(fit.scales["n_events"]):
                continue                                           # N dropped from this window-month fit
            train = training_mask(arr, window, m)
            lo, hi = np.quantile(arr["y_vol"][train], [0.001, 0.999])
            test = (arr["month"] == m) & np.isfinite(arr["y_vol"]) & np.isfinite(arr["a"]) & (arr["a"] > 0)
            sc = fit.scales.set_index("n_events").loc[n]
            q, a, side, y = arr["z"][test], arr["a"][test], arr["side"][test], arr["y_vol"][test]
            pred = side * sc["r_scale"] * pb_shape(q * a ** fit.gamma / sc["q_scale"], fit.alpha, fit.beta)
            pred = np.where(q > 0, pred, 0.0)
            err2 = (y - pred) ** 2
            err2_clip = (np.clip(y, lo, hi) - pred) ** 2
            frame = pd.DataFrame({"day": arr["day"][test], "err2": err2, "err2_clip": err2_clip, "y2": y ** 2})
            monthly.append({"window": window, "test_month": label(m), "n_events": n, "blocks": int(test.sum()),
                            "sse": float(err2.sum()), "sse_clip": float(err2_clip.sum()), "sum_y2": float((y ** 2).sum())})
            by_day = frame.groupby("day")[["err2", "err2_clip"]].sum().reset_index()
            daily.append(by_day.assign(window=window, test_month=label(m), n_events=n))
        print(f"scored: N = {n:,}", flush=True)
        del arr
    return pd.DataFrame(monthly), pd.concat(daily, ignore_index=True)


def compare(monthly: pd.DataFrame, daily: pd.DataFrame, window: str, base: str, first: str, rng) -> dict:
    out = {}
    for kind, col, dcol in (("raw", "sse", "err2"), ("clipped", "sse_clip", "err2_clip")):
        mw = monthly.loc[(monthly.window == window) & (monthly.test_month >= first)].set_index(["test_month", "n_events"])
        mb = monthly.loc[(monthly.window == base) & (monthly.test_month >= first)].set_index(["test_month", "n_events"])
        common = mw.index.intersection(mb.index)                   # (month, N) scored by both windows
        mw, mb = mw.loc[common], mb.loc[common]
        rel = (mb[col] - mw[col]) / mb[col]                        # blocks are identical, so MSE ratio = SSE ratio
        d_m = rel.groupby(level="test_month").mean()
        boot = [rng.choice(d_m.to_numpy(), len(d_m), replace=True).mean() for _ in range(N_BOOT)]
        dw = daily.loc[(daily.window == window) & (daily.test_month >= first)].set_index(["test_month", "n_events", "day"])
        db = daily.loc[(daily.window == base) & (daily.test_month >= first)].set_index(["test_month", "n_events", "day"])
        keep = db.index.droplevel("day").isin(common)
        db, dw = db.loc[keep], dw.loc[db.index[keep]]
        contrib = (db[dcol] - dw[dcol]) / mb[col].reindex(db.index.droplevel("day")).to_numpy()
        n_per_month = rel.groupby(level="test_month").size()
        sums = contrib.groupby(level=["test_month", "day"]).sum()
        c_d = sums / n_per_month.reindex(sums.index.get_level_values("test_month")).to_numpy()   # mean over N
        top = c_d.sort_values(ascending=False).index[:10]
        d_m_trim = c_d.drop(top).groupby(level="test_month").sum().reindex(d_m.index, fill_value=0.0)
        out[kind] = {"months": int(len(d_m)), "month_n_pairs": int(len(rel)), "mean_D": float(d_m.mean()),
                     "ci_2_5": float(np.quantile(boot, 0.025)), "ci_97_5": float(np.quantile(boot, 0.975)),
                     "share_months_positive": float((d_m > 0).mean()), "mean_D_best10_days_removed": float(d_m_trim.mean()),
                     "best10_days": [str(np.datetime64(int(day), "D")) for _, day in top],
                     "per_n_mean_relative_improvement": {int(n): float(v) for n, v in rel.groupby(level="n_events").mean().items()},
                     "per_year_mean_D": {y: float(v) for y, v in d_m.groupby(d_m.index.str[:4]).mean().items()}}
    raw = out["raw"]
    out["decision_12m_helps"] = bool(raw["mean_D"] > 0 and raw["ci_2_5"] > 0 and raw["share_months_positive"] >= 0.60
                                     and raw["mean_D_best10_days_removed"] > 0)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = training_stats()
    fits, fit_table, scale_table = fit_windows(stats)
    fit_table.to_csv(OUT / "fits.csv", index=False)
    scale_table.to_csv(OUT / "scales.csv", index=False)
    monthly, daily = score(fits)
    monthly.to_csv(OUT / "monthly_scores.csv", index=False)
    daily.to_csv(OUT / "daily_contributions.csv", index=False)

    rng = np.random.default_rng(SEED)
    r2 = (monthly.groupby(["window", "n_events"])[["sse", "sum_y2"]].sum()
          .assign(oos_r2=lambda d: 1 - d["sse"] / d["sum_y2"])["oos_r2"])
    coverage = fit_table.groupby("window")["n_included"].agg(lambda v: {k: int(c) for k, c in v.value_counts().items()}).to_dict()
    summary = {"entry": "E-008", "amendment": 1, "data": f"BTCUSDT pb_blocks_v1 2021-01..{LAST_MONTH}",
               "n_values": list(N_VALUES), "n_coverage_by_window": coverage,
               "comparisons": {name: compare(monthly, daily, w, b, f, rng) for name, w, b, f in COMPARISONS},
               "oos_r2_vs_zero": {f"{w}|{n}": float(v) for (w, n), v in r2.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    primary = summary["comparisons"]["primary: trailing_12 vs expanding"]
    print(json.dumps({k: v for k, v in primary["raw"].items() if k != "best10_days"}, indent=2))
    print("decision (12-month window helps):", primary["decision_12m_helps"])


if __name__ == "__main__":
    main()
