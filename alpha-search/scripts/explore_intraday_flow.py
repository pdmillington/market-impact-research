"""Exploratory diagnostics for H-012 (intraday flow). Hypothesis generation only.

Reads the H-012 decision panel and writes descriptive tables. Nothing here is a
test: every pattern found must be registered as a new hypothesis before it is
evaluated, and the slices examined are counted in HYPOTHESIS_LEDGER.md.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

from run_intraday_flow import BENCHMARK  # noqa: E402
from run_residual_decomposition import ols, spearman, t_stat  # noqa: E402


RESULTS = REPO / "alpha-search" / "reports" / "intraday_flow_v1"
OUTPUT = RESULTS / "exploration"
PERIODS = {"2021-02..2022-12": ("2021-02", "2022-12"), "2023-01..2026-05": ("2023-01", "2026-05")}


def period_frame(panel: pl.DataFrame, period: str) -> pl.DataFrame:
    first, last = PERIODS[period]
    return panel.filter((pl.col("month") >= first) & (pl.col("month") <= last))


def day_mean_se(values: np.ndarray, days: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(values)
    values, days = values[ok], days[ok]
    if len(values) < 30:
        return float("nan"), float("nan"), int(len(values))
    unique, inverse = np.unique(days, return_inverse=True)
    daily = np.bincount(inverse, weights=values) / np.bincount(inverse)
    return float(values.mean()), float(daily.std(ddof=1) / np.sqrt(len(daily))), int(len(values))


def residualise(frame: pl.DataFrame, feature: str) -> np.ndarray:
    """Feature minus its linear projection on the benchmark within the frame."""

    x = frame.select(BENCHMARK).to_numpy()
    y = frame[feature].to_numpy()
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    output = np.full(len(y), np.nan)
    centre = x[ok].mean(axis=0)
    beta = ols(x[ok] - centre, y[ok] - y[ok].mean())
    output[ok] = y[ok] - y[ok].mean() - (x[ok] - centre) @ beta
    return output


def main() -> None:
    config = json.loads((REPO / "alpha-search" / "configs" / "intraday_flow_v1.json").read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = pl.read_parquet(RESULTS / "decision_panel.parquet").filter(
        pl.all_horizontal([pl.col(f).is_finite() for f in BENCHMARK])
    )
    windows, horizons = config["windows_minutes"], config["horizons_minutes"]
    flow_features = [f"zx_{w}" for w in windows] + [f"zz_{w}" for w in windows] + [f"persist_{w}" for w in windows]

    # 1. Univariate monthly ICs by feature and horizon, per period.
    rows = []
    for period in PERIODS:
        frame = period_frame(panel, period)
        for (month,), part in frame.group_by(["month"], maintain_order=True):
            for h in horizons:
                y = part[f"F{h}m"].to_numpy()
                for feature in flow_features + BENCHMARK:
                    x = part[feature].to_numpy()
                    ok = np.isfinite(x) & np.isfinite(y)
                    rows.append({"period": period, "month": month, "horizon_minutes": h, "feature": feature,
                                 "spearman_ic": spearman(x[ok], y[ok])})
    univariate = pl.DataFrame(rows)
    summary = []
    for (period, h, feature), part in univariate.group_by(["period", "horizon_minutes", "feature"]):
        ic = part["spearman_ic"].to_numpy()
        summary.append({"period": period, "horizon_minutes": h, "feature": feature,
                        "mean_ic": float(np.nanmean(ic)), "ic_t": t_stat(ic),
                        "positive_month_fraction": float(np.mean(ic > 0))})
    pl.DataFrame(summary).sort("period", "feature", "horizon_minutes").write_csv(OUTPUT / "univariate_ic.csv")

    # 2. Correlation of flow with the benchmark returns, and residual (orthogonalised) flow ICs.
    correlation_rows, residual_rows = [], []
    for period in PERIODS:
        frame = period_frame(panel, period)
        for w in windows:
            flow = frame[f"zx_{w}"].to_numpy()
            for r in ("ret_5m", "ret_15m", "ret_60m"):
                other = frame[r].to_numpy()
                ok = np.isfinite(flow) & np.isfinite(other)
                correlation_rows.append({"period": period, "flow": f"zx_{w}", "return": r,
                                         "spearman": spearman(flow[ok], other[ok])})
            residual = residualise(frame, f"zx_{w}")
            frame_r = frame.with_columns(pl.Series(f"resid_{w}", residual))
            for (month,), part in frame_r.group_by(["month"], maintain_order=True):
                for h in horizons:
                    x, y = part[f"resid_{w}"].to_numpy(), part[f"F{h}m"].to_numpy()
                    ok = np.isfinite(x) & np.isfinite(y)
                    residual_rows.append({"period": period, "month": month, "window_minutes": w,
                                          "horizon_minutes": h, "spearman_ic": spearman(x[ok], y[ok])})
    pl.DataFrame(correlation_rows).write_csv(OUTPUT / "flow_return_correlation.csv")
    residual = pl.DataFrame(residual_rows)
    residual_summary = []
    for (period, w, h), part in residual.group_by(["period", "window_minutes", "horizon_minutes"]):
        ic = part["spearman_ic"].to_numpy()
        residual_summary.append({"period": period, "window_minutes": w, "horizon_minutes": h,
                                 "mean_ic": float(np.nanmean(ic)), "ic_t": t_stat(ic),
                                 "positive_month_fraction": float(np.mean(ic > 0))})
    pl.DataFrame(residual_summary).sort("period", "window_minutes", "horizon_minutes").write_csv(
        OUTPUT / "residual_flow_ic.csv")

    # 3. Decile shapes (test period): raw and residual flow, mean future return in bps with day-clustered SE.
    test = period_frame(panel, "2023-01..2026-05")
    for w in windows:
        test = test.with_columns(pl.Series(f"resid_{w}", residualise(test, f"zx_{w}")))
    decile_rows = []
    days = test["day"].to_numpy()
    for w in windows:
        for kind, column in (("raw", f"zx_{w}"), ("residual", f"resid_{w}")):
            x = test[column].to_numpy()
            ok = np.isfinite(x)
            cuts = np.quantile(x[ok], np.linspace(0, 1, 11)[1:-1])
            label = np.digitize(x, cuts)
            for h in horizons:
                y = test[f"F{h}m"].to_numpy()
                for decile in range(10):
                    chosen = ok & (label == decile)
                    mean, se, count = day_mean_se(y[chosen], days[chosen])
                    decile_rows.append({"window_minutes": w, "kind": kind, "horizon_minutes": h,
                                        "decile": decile + 1, "mean_future_bps": mean, "se_bps": se,
                                        "observations": count, "mean_feature": float(np.nanmean(x[chosen]))})
    pl.DataFrame(decile_rows).write_csv(OUTPUT / "decile_shapes.csv")

    # 4. Flow-price divergence grid (test period): 60-minute return tercile x 60-minute flow tercile.
    grid_rows = []
    for w in windows:
        ret_col = "ret_60m" if w >= 60 else "ret_15m"
        r, f = test[ret_col].to_numpy(), test[f"zx_{w}"].to_numpy()
        r_cuts = np.quantile(r[np.isfinite(r)], [1 / 3, 2 / 3])
        f_cuts = np.quantile(f[np.isfinite(f)], [1 / 3, 2 / 3])
        r_label, f_label = np.digitize(r, r_cuts), np.digitize(f, f_cuts)
        for h in horizons:
            y = test[f"F{h}m"].to_numpy()
            for i in range(3):
                for j in range(3):
                    chosen = (r_label == i) & (f_label == j) & np.isfinite(r) & np.isfinite(f)
                    mean, se, count = day_mean_se(y[chosen], days[chosen])
                    grid_rows.append({"window_minutes": w, "return_feature": ret_col, "horizon_minutes": h,
                                      "return_tercile": i + 1, "flow_tercile": j + 1,
                                      "mean_future_bps": mean, "se_bps": se, "observations": count})
    pl.DataFrame(grid_rows).write_csv(OUTPUT / "divergence_grid.csv")

    # 5. Volatility regime (cuts from 2021-22) and year: univariate IC of raw and residual flow.
    calibration = period_frame(panel, "2021-02..2022-12")
    v_cuts = np.quantile(calibration["rv_24h"].drop_nans().to_numpy(), [1 / 3, 2 / 3])
    test = test.with_columns(
        pl.Series("vol_tercile", np.digitize(test["rv_24h"].to_numpy(), v_cuts) + 1),
        pl.col("month").str.slice(0, 4).alias("year"),
    )
    regime_rows = []
    for split in ("vol_tercile", "year"):
        for (value, month), part in test.group_by([split, "month"]):
            for w in windows:
                for kind, column in (("raw", f"zx_{w}"), ("residual", f"resid_{w}")):
                    for h in horizons:
                        x, y = part[column].to_numpy(), part[f"F{h}m"].to_numpy()
                        ok = np.isfinite(x) & np.isfinite(y)
                        if ok.sum() < 200:
                            continue
                        regime_rows.append({"split": split, "value": str(value), "month": month,
                                            "window_minutes": w, "kind": kind, "horizon_minutes": h,
                                            "spearman_ic": spearman(x[ok], y[ok])})
    regime = pl.DataFrame(regime_rows)
    regime_summary = []
    for keys, part in regime.group_by(["split", "value", "window_minutes", "kind", "horizon_minutes"]):
        ic = part["spearman_ic"].to_numpy()
        regime_summary.append({**dict(zip(["split", "value", "window_minutes", "kind", "horizon_minutes"], keys)),
                               "months": len(ic), "mean_ic": float(np.nanmean(ic)), "ic_t": t_stat(ic)})
    pl.DataFrame(regime_summary).sort("split", "value", "window_minutes", "kind", "horizon_minutes").write_csv(
        OUTPUT / "regime_ic.csv")

    # 6. Execution diagnostics (test period, all decisions): future mid move after a passive fill, by fill delay.
    execution_rows = []
    for side, column, sign in (("buy", "fill_buy_s", 1.0), ("sell", "fill_sell_s", -1.0)):
        delay = test[column].to_numpy()
        buckets = [("1s", (delay == 1)), ("2-5s", (delay >= 2) & (delay <= 5)),
                   ("6-15s", (delay >= 6) & (delay <= 15)), ("16-60s", (delay >= 16) & (delay <= 60)),
                   ("unfilled", ~np.isfinite(delay))]
        for h in horizons:
            y = sign * test[f"F{h}m"].to_numpy()
            for name, chosen in buckets:
                mean, se, count = day_mean_se(y[chosen], days[chosen])
                execution_rows.append({"side": side, "fill_delay": name, "horizon_minutes": h,
                                       "share": float(chosen.mean()), "mean_signed_future_bps": mean,
                                       "se_bps": se})
    pl.DataFrame(execution_rows).write_csv(OUTPUT / "fill_delay_adverse_selection.csv")
    print(OUTPUT)


if __name__ == "__main__":
    main()
