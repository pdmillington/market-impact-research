"""E-005 follow-up: does unsigned urgency (ml_share) forecast volatility beyond realised volatility?

Development window and decision grid as E-005 (2021-01 to 2023-12, non-overlapping
8 h / 24 h decisions). Targets: |r_H| and realised volatility over the next H
(square root of summed squared hourly log returns). Conditioning set:
Z = {flow_imbalance_W, own_return_W, log RV_24h, log RV_7d} (HAR-style), all
trailing and known at the decision time.

Two statistics per cell:
- conditional Gaussian-copula MI I(ml_share; target | Z) with the E-005
  circular-shift null (1,000 shifts of at least 30 days);
- HAR regression log RV_next ~ log RV_24h + log RV_7d (+ log RV_W), with and
  without ml_share z: coefficient, Newey-West t and in-sample R^2 gain.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import polars as pl


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "alpha-search" / "scripts"))

import build_signature_features as sig  # noqa: E402
import run_mi_screen_e005 as e005  # noqa: E402


HOUR = 3_600_000
OUT = REPO / "alpha-search" / "reports" / "e005_mi_screen" / "urgency_volatility"


def nw_slope(y: np.ndarray, x: np.ndarray, lag: int = 1) -> tuple[float, float]:
    """Slope of y on [1, x] and its Newey-West t."""

    X = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    xe = X * e[:, None]
    S = xe.T @ xe
    for l in range(1, lag + 1):
        G = xe[l:].T @ xe[:-l]
        S += (1 - l / (lag + 1)) * (G + G.T)
    inv = np.linalg.inv(X.T @ X)
    return float(beta[1]), float(beta[1] / np.sqrt((inv @ S @ inv)[1, 1]))


def hourly_frame(root: Path, symbol: str) -> pl.DataFrame:
    panel = pl.read_parquet(sig.panel_path(root, symbol))
    close = panel["close"].to_numpy()
    r = np.concatenate([[np.nan], np.diff(np.log(close))])
    sq = pl.Series(r**2, nan_to_null=True)
    return panel.with_columns(
        pl.Series("r", r),
        pl.Series("rv_24h", np.sqrt(sq.rolling_sum(24, min_samples=20).fill_null(np.nan).to_numpy())),
        pl.Series("rv_7d", np.sqrt(sq.rolling_sum(168, min_samples=140).fill_null(np.nan).to_numpy() / 7)),
        pl.Series("rv_8h", np.sqrt(sq.rolling_sum(8, min_samples=7).fill_null(np.nan).to_numpy())),
    )


def main() -> None:
    root = REPO / "shared-data"
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(e005.SEED + 7)
    rows = []
    for symbol in e005.INSTRUMENTS:
        frame = hourly_frame(root, symbol)
        index = {int(t): i for i, t in enumerate(frame["decision_ms"].to_numpy())}
        r = frame["r"].to_numpy()
        for label, horizon in e005.HORIZONS.items():
            times = e005.decision_grid(horizon)
            at = np.array([index.get(int(t), -1) for t in times])
            ok = at >= 0
            times, at = times[ok], at[ok]
            col = lambda c: frame[c].to_numpy()[at]  # noqa: E731
            future_sq = np.array([np.nansum(r[i + 1: i + 1 + horizon] ** 2) if i + horizon < len(r) else np.nan for i in at])
            rv_next = np.sqrt(future_sq)
            ret_next = np.array([np.nansum(r[i + 1: i + 1 + horizon]) if i + horizon < len(r) else np.nan for i in at])
            x = col(f"ml_share_{label}_z")
            past_window = col("rv_8h") if horizon == 8 else col("rv_24h")
            rv_columns = [np.log(col("rv_24h")), np.log(col("rv_7d"))] + ([np.log(past_window)] if horizon != 24 else [])
            z = np.column_stack([col(f"flow_imbalance_{label}_z"), col(f"own_return_{label}_z"), *rv_columns])
            for target_name, target in (("abs_return", np.abs(ret_next)), ("realised_vol", rv_next)):
                keep = np.isfinite(x) & np.isfinite(target) & np.all(np.isfinite(z), axis=1) & (target > 0)
                cx, cy, cz = e005.copnorm(x[keep]), e005.copnorm(target[keep]), e005.copnorm(z[keep])
                cz_old = e005.copnorm(z[keep][:, :2])
                min_shift = e005.MIN_SHIFT_DAYS * 24 // horizon
                offsets = rng.integers(min_shift, keep.sum() - min_shift, size=e005.N_SHIFTS)

                def cmi(conditioning):
                    observed = e005.mi_continuous(cx, cy, conditioning)
                    null = np.array([e005.mi_continuous(cx, np.roll(cy, k, axis=0), conditioning) for k in offsets])
                    return observed - null.mean(), (1 + np.sum(null >= observed)) / (len(offsets) + 1)

                excess_old, p_old = cmi(cz_old)
                excess_new, p_new = cmi(cz)
                row = {"symbol": symbol, "horizon": label, "target": target_name, "n": int(keep.sum()),
                       "cmi_excess_flow_return_only": excess_old, "p_flow_return_only": p_old,
                       "cmi_excess_with_rv": excess_new, "p_with_rv": p_new}
                if target_name == "realised_vol":
                    y = np.log(target[keep])
                    har = np.column_stack([np.ones(keep.sum())] + [c[keep] for c in rv_columns])
                    full = np.column_stack([har, x[keep]])
                    b_har, *_ = np.linalg.lstsq(har, y, rcond=None)
                    b_full, *_ = np.linalg.lstsq(full, y, rcond=None)
                    r2 = lambda X, b: 1 - np.var(y - X @ b) / np.var(y)  # noqa: E731
                    resid = y - har @ b_har
                    xr = x[keep] - har @ np.linalg.lstsq(har, x[keep], rcond=None)[0]   # FWL: ml_share net of HAR terms
                    slope, t_slope = nw_slope(resid, xr, lag=1)
                    row |= {"har_r2": r2(har, b_har), "har_plus_ml_share_r2": r2(full, b_full),
                            "ml_share_coef_log_rv_per_sd": float(b_full[-1]), "ml_share_slope_fwl": slope,
                            "ml_share_nw_t": t_slope}
                rows.append(row)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    table = pl.DataFrame(rows)
    table.write_csv(OUT / "urgency_volatility.csv")


if __name__ == "__main__":
    main()
