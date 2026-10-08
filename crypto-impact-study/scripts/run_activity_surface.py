"""Activity-adjusted P&B impact surface on the extended BTC sample (paper redo).

Primary sample: 2021-01 to 2026-08 (2020-12 is loaded only to warm up the 24 h
normalisation). Response: block log return divided by trailing 24 h realised
volatility (the paper's main normalisation, I/sigma_24h; sigma from the same
bars' squared returns, current bar excluded, full 24 h of history required).
Robustness: the raw log return (the paper's P&B-section convention), and the
2019-10 to 2020-12 period.

Researcher decisions (2026-10-05): vol-normalised response primary, because on
raw returns relative activity mainly proxies the previous day's volatility
(local gamma < 0); one common gamma as the leading model, with gamma by N
reported from the independent per-N fits.

Outputs (reports/redo/<tag>/):
- fits.csv: full-sample fits for gamma = 1 (P&B), gamma = 0 (imbalance ratio),
  gamma free (activity surface), and per-N independent fits;
- scales.csv, scaling_exponents.csv: per-N Q_N, R_N and their log-log slopes;
- gamma_bootstrap.csv: month-block bootstrap of gamma;
- gamma_by_year.csv: gamma fitted separately for each calendar year;
- oos.csv: chronological out-of-sample comparison (expanding training from 2021);
- displacement.csv: the paper's activity / extreme-imbalance residual table under
  P&B and under the activity surface;
- robustness.csv: volatility-normalised response and 2019-2020.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "crypto-impact-study" / "src"))
sys.path.insert(0, str(REPO / "crypto-impact-study" / "scripts"))
sys.path.insert(0, str(REPO / "shared-methodology" / "src"))

from build_pb_blocks import N_VALUES, output_path  # noqa: E402
from crypto_impact.pb_collapse import pb_shape, scaling_slope  # noqa: E402
from crypto_impact.pb_surface import (  # noqa: E402
    aggregate_cells, fit_activity_surface, grid_cuts, monthly_cell_stats, weighted_rmse,
)
from crypto_market_data.months import iter_months  # noqa: E402


N_Z, N_A = 20, 10
DISPLACEMENT_N = (250, 1000, 4000, 16000)
MODELS = {"pb_gamma_1": 1.0, "imbalance_ratio_gamma_0": 0.0, "activity_surface": None}


def month_index(month: str) -> int:
    year, number = map(int, month.split("-"))
    return year * 12 + number - 1


def load_arrays(root: Path, n: int, warm_start: str, start: str, end: str) -> dict:
    blocks = pl.concat([pl.read_parquet(output_path(root, "BTCUSDT", n, m)) for m in iter_months(warm_start, end)]).to_pandas()
    blocks = blocks.sort_values("end_time_ms", kind="stable")
    end_time = pd.DatetimeIndex(pd.to_datetime(blocks["end_time_ms"], unit="ms", utc=True))
    gross = pd.Series(blocks["gross_volume"].to_numpy(float), index=end_time)
    trailing_volume = gross.rolling("1D", closed="left", min_periods=1).sum().to_numpy()
    r2 = pd.Series(blocks["log_return"].to_numpy(float) ** 2, index=end_time)
    trailing_vol = np.sqrt(r2.rolling("1D", closed="left", min_periods=1).sum().to_numpy())
    full_from = end_time.min() + pd.Timedelta("1D")
    keep = (end_time >= full_from) & (blocks["start_time_ms"].to_numpy() >= pd.Timestamp(start + "-01", tz="UTC").value // 10**6)
    # Blocks covering a known data gap stay in the trailing windows (their volume and
    # returns were observed) but are excluded from the analysis sample.
    keep &= ~blocks["spans_data_gap"].to_numpy()
    b = blocks.loc[keep]
    months = pd.to_datetime(b["start_time_ms"], unit="ms", utc=True).dt.strftime("%Y-%m").map(month_index).to_numpy()
    gross_v = b["gross_volume"].to_numpy(float)
    signed = b["signed_volume"].to_numpy(float)
    tv = trailing_volume[keep]
    return {
        "month": months.astype(np.int32), "side": np.sign(signed).astype(np.int8),
        "z": np.abs(signed) / gross_v, "a": gross_v / tv, "y": b["log_return"].to_numpy(float),
        "y_vol": b["log_return"].to_numpy(float) / trailing_vol[keep],
        "intensity": n / np.where(b["duration_seconds"].to_numpy(float) > 0, b["duration_seconds"].to_numpy(float), np.nan),
        "pb_x": np.abs(signed) / tv,
    }


def stats_for(arrays: dict, n: int, cut_months: set | None, response: str) -> pd.DataFrame:
    chosen = np.isin(arrays["month"], list(cut_months)) if cut_months else np.ones(len(arrays["month"]), bool)
    cuts = {s: grid_cuts(arrays["z"][chosen & (arrays["side"] == s) & (arrays["z"] > 0)],
                         arrays["a"][chosen & (arrays["side"] == s) & np.isfinite(arrays["a"])], N_Z, N_A) for s in (-1, 1)}
    stats = monthly_cell_stats(arrays["month"], arrays["side"], arrays["z"], arrays["a"], arrays[response], cuts)
    stats.insert(0, "n_events", n)
    return stats


def fit_all(cells: pd.DataFrame, n_starts: int = 8) -> dict:
    return {name: fit_activity_surface(cells, gamma=g, n_starts=n_starts) for name, g in MODELS.items()}


def fit_row(name: str, fit, cells: pd.DataFrame, extra: dict) -> dict:
    scales = fit.scales.rename(columns={"q_scale": "fitted_q_scale", "r_scale": "fitted_r_scale"})
    xi = scaling_slope(scales, "fitted_q_scale", "xi") if len(scales) > 2 else {"robust_exponent": np.nan}
    psi = scaling_slope(scales, "fitted_r_scale", "psi") if len(scales) > 2 else {"robust_exponent": np.nan}
    resid = fit.scaled_residuals(cells)
    return {"model": name, **extra, "alpha": fit.alpha, "beta": fit.beta, "gamma": fit.gamma,
            "gamma_fixed": fit.gamma_fixed, "objective": fit.objective, "weighted_rmse": weighted_rmse(fit, cells),
            "scaled_rmse": float(np.sqrt(np.mean(resid ** 2))), "cells": len(cells),
            "xi": xi["robust_exponent"], "psi": psi["robust_exponent"], "converged": fit.converged}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="activity_surface_2021_2026")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--shared-data-root", type=Path, default=REPO / "shared-data")
    args = parser.parse_args()
    out = REPO / "crypto-impact-study" / "reports" / "redo" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    root = args.shared_data_root
    primary_months = {month_index(m) for m in iter_months("2021-01", "2026-08")}

    arrays = {}
    for n in N_VALUES:
        arrays[n] = load_arrays(root, n, "2020-12", "2021-01", "2026-08")
        print(f"N={n}: {len(arrays[n]['y']):,} blocks", flush=True)

    # 1. Full-sample fits (full-sample cuts).
    stats = pd.concat([stats_for(arrays[n], n, None, "y_vol") for n in N_VALUES], ignore_index=True)
    cells = aggregate_cells(stats)
    fits = fit_all(cells)
    rows = [fit_row(name, fit, cells, {"sample": "2021-01..2026-08", "response": "log_return / trailing 24h vol"}) for name, fit in fits.items()]
    for n in N_VALUES:
        sub = cells.loc[cells["n_events"] == n]
        for name, g in MODELS.items():
            rows.append(fit_row(f"independent_N_{name}", fit_activity_surface(sub, gamma=g, n_starts=6), sub,
                                {"sample": "2021-01..2026-08", "response": "log_return / trailing 24h vol", "n_events": n}))
    pd.DataFrame(rows).to_csv(out / "fits.csv", index=False)
    scales = pd.concat([f.scales.assign(model=name) for name, f in fits.items()])
    scales.to_csv(out / "scales.csv", index=False)
    print(pd.DataFrame(rows)[["model", "gamma", "alpha", "beta", "weighted_rmse", "scaled_rmse", "xi", "psi"]].head(3).to_string(), flush=True)

    # Per-N scaled residual RMSE by model.
    per_n = []
    for name, fit in fits.items():
        resid = fit.scaled_residuals(cells)
        for n in N_VALUES:
            mask = (cells["n_events"] == n).to_numpy()
            per_n.append({"model": name, "n_events": n, "scaled_rmse": float(np.sqrt(np.mean(resid[mask] ** 2)))})
    pd.DataFrame(per_n).to_csv(out / "per_n_rmse.csv", index=False)

    # 2. Month-block bootstrap of gamma.
    rng = np.random.default_rng(20261005)
    months = np.array(sorted(primary_months))
    boot = []
    for b in range(args.bootstrap):
        draw = pd.Series(rng.choice(months, len(months), replace=True)).value_counts()
        fit = fit_activity_surface(aggregate_cells(stats, weights=draw), n_starts=1, initial=fits["activity_surface"])
        boot.append({"replicate": b, "gamma": fit.gamma, "alpha": fit.alpha, "beta": fit.beta})
    boot = pd.DataFrame(boot)
    boot.to_csv(out / "gamma_bootstrap.csv", index=False)
    print("gamma bootstrap 2.5/50/97.5%:", np.quantile(boot["gamma"], [0.025, 0.5, 0.975]).round(3), flush=True)

    # 3. Gamma by calendar year (full-sample cuts).
    by_year = []
    for year in range(2021, 2027):
        year_months = [m for m in months if m // 12 == year]
        yc = aggregate_cells(stats, months=year_months)
        for name, g in MODELS.items():
            fit = fit_activity_surface(yc, gamma=g, n_starts=6, initial=fits["activity_surface"] if g is None else None)
            by_year.append(fit_row(name, fit, yc, {"sample": str(year), "response": "log_return / trailing 24h vol"}))
    pd.DataFrame(by_year).to_csv(out / "gamma_by_year.csv", index=False)

    # 4. Chronological out-of-sample: expanding training from 2021, cuts frozen from training.
    oos = []
    for test_year in (2023, 2024, 2025, 2026):
        train_months = {m for m in months if m // 12 < test_year}
        test_months = [m for m in months if m // 12 == test_year]
        fold_stats = pd.concat([stats_for(arrays[n], n, train_months, "y_vol") for n in N_VALUES], ignore_index=True)
        train_cells = aggregate_cells(fold_stats, months=train_months)
        test_cells = aggregate_cells(fold_stats, months=test_months)
        for name, g in MODELS.items():
            fit = fit_activity_surface(train_cells, gamma=g, n_starts=6)
            oos.append({"test_year": test_year, "model": name, "gamma": fit.gamma,
                        "test_weighted_rmse": weighted_rmse(fit, test_cells),
                        "test_scaled_rmse": float(np.sqrt(np.mean(fit.scaled_residuals(test_cells) ** 2)))})
            for n in N_VALUES:
                sub_train = train_cells.loc[train_cells["n_events"] == n]
                sub_test = test_cells.loc[test_cells["n_events"] == n]
                ind = fit_activity_surface(sub_train, gamma=g, n_starts=4)
                oos.append({"test_year": test_year, "model": f"independent_N_{name}", "n_events": n, "gamma": ind.gamma,
                            "test_weighted_rmse": weighted_rmse(ind, sub_test),
                            "test_scaled_rmse": float(np.sqrt(np.mean(ind.scaled_residuals(sub_test) ** 2)))})
        print(f"OOS {test_year} done", flush=True)
    pd.DataFrame(oos).to_csv(out / "oos.csv", index=False)

    # 5. Activity displacement (paper's table) under P&B and the activity surface.
    disp = []
    for name in ("pb_gamma_1", "activity_surface"):
        fit = fits[name]
        lookup = fit.scales.set_index("n_events")
        for n in DISPLACEMENT_N:
            d = arrays[n]
            ok = np.isfinite(d["a"]) & (d["z"] > 0) & np.isfinite(d["intensity"]) & (d["side"] != 0) & np.isfinite(d["y_vol"])
            q, r = lookup.loc[n, "q_scale"], lookup.loc[n, "r_scale"]
            pred = r * pb_shape(d["z"][ok] * d["a"][ok] ** fit.gamma / q, fit.alpha, fit.beta)
            resid = (d["side"][ok] * d["y_vol"][ok] - pred) / r
            act = pd.qcut(d["intensity"][ok], 5, labels=False, duplicates="drop")
            imb = pd.qcut(d["pb_x"][ok], 5, labels=False, duplicates="drop")
            frame = pd.DataFrame({"resid": resid, "high_activity": act == act.max(), "extreme_imbalance": imb == imb.max()})
            g = frame.groupby(["high_activity", "extreme_imbalance"])["resid"].agg(["mean", "std", "size"])
            v = g["mean"]
            disp.append({"model": name, "n_events": n,
                         "ordinary_activity": frame.loc[~frame["high_activity"], "resid"].mean(),
                         "high_activity": frame.loc[frame["high_activity"], "resid"].mean(),
                         "difference_high_minus_ordinary": frame.loc[frame["high_activity"], "resid"].mean() - frame.loc[~frame["high_activity"], "resid"].mean(),
                         "activity_imbalance_interaction": (v.loc[(True, True)] - v.loc[(True, False)]) - (v.loc[(False, True)] - v.loc[(False, False)]),
                         "all_states": frame["resid"].mean()})
    pd.DataFrame(disp).to_csv(out / "displacement.csv", index=False)
    print(pd.DataFrame(disp).round(3).to_string(), flush=True)

    # 6. Robustness: raw log-return response; early period 2019-10..2020-12.
    robust = []
    rstats = pd.concat([stats_for(arrays[n], n, None, "y") for n in N_VALUES], ignore_index=True)
    rcells = aggregate_cells(rstats)
    for name, fit in fit_all(rcells, n_starts=6).items():
        robust.append(fit_row(name, fit, rcells, {"sample": "2021-01..2026-08", "response": "log_return (raw)"}))
    del arrays
    early = {n: load_arrays(root, n, "2019-09", "2019-10", "2020-12") for n in N_VALUES}
    ecells = aggregate_cells(pd.concat([stats_for(early[n], n, None, "y_vol") for n in N_VALUES], ignore_index=True))
    for name, fit in fit_all(ecells, n_starts=6).items():
        robust.append(fit_row(name, fit, ecells, {"sample": "2019-10..2020-12", "response": "log_return / trailing 24h vol"}))
    pd.DataFrame(robust).to_csv(out / "robustness.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({"primary": "2021-01..2026-08", "warm_up": "2020-12", "bins": [N_Z, N_A],
                                                   "n_values": list(N_VALUES), "bootstrap": args.bootstrap, "response": "log_return / trailing 24h realised vol (same bars)",
                                                   "excluded": "2026-09 onward (alpha-search P-006)"}, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
